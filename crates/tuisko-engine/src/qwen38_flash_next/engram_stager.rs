//! Qwen3.8-Flash-Next engram host gather and its stable device plane.
//!
//! The 47.68 GiB FP8 table stays file-backed. Each admitted round hashes one
//! sequence, gathers its represented rows into pinned memory, and enqueues an
//! upload to one stable device plane without decoding the bytes.
//!
//! The plane is consumed on the upload stream. Since the pinned buffer holds
//! one round, an event fences it before the next gather and again on drop.

use crate::common::math::product;
use crate::qwen38_flash_next::engram_stager_layout::{
    Qwen38FlashNextEngramStagerLayout, require_qwen38_flash_next_engram_width,
};
use crate::{EngineError, EngineResult};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaEvent, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_model::{
    QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN, Qwen38FlashNextEngramCarry,
    Qwen38FlashNextEngramRowHasher, Qwen38FlashNextEngramTable,
};

/// Gathers one token window's engram rows into `destination`, advancing `carry`.
///
/// Host-only and allocation-free. `rows` and `destination` are caller-owned
/// scratch, and output is token-major then head-major.
pub fn gather_qwen38_flash_next_engram_window(
    table: Qwen38FlashNextEngramTable<'_>,
    carry: &mut Qwen38FlashNextEngramCarry,
    tokens: &[u32],
    rows: &mut [i64],
    destination: &mut [u8],
) -> EngineResult<()> {
    require_qwen38_flash_next_engram_width(tokens.len())?;

    let expected_bytes = product(
        "Flash-Next engram round bytes",
        tokens.len(),
        table.token_bytes(),
    )?;
    let expected_rows = product(
        "Flash-Next engram round rows",
        tokens.len(),
        QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN,
    )?;

    if rows.len() != expected_rows || destination.len() != expected_bytes {
        return Err(EngineError::layout(format!(
            "Flash-Next engram round of {} tokens needs {expected_rows} rows and {expected_bytes} staged bytes, given {} and {}",
            tokens.len(),
            rows.len(),
            destination.len()
        )));
    }

    let next_carry = require_gather_rows(table, *carry, tokens, rows)?;
    table.gather_rows(rows, destination)?;
    *carry = next_carry;

    Ok(())
}

fn require_gather_rows(
    table: Qwen38FlashNextEngramTable<'_>,
    mut carry: Qwen38FlashNextEngramCarry,
    tokens: &[u32],
    rows: &mut [i64],
) -> EngineResult<Qwen38FlashNextEngramCarry> {
    Qwen38FlashNextEngramRowHasher::new(table.constants()).stream_rows(&mut carry, tokens, rows)?;
    for &row in rows.iter() {
        table.row_codes(row)?;
    }

    Ok(carry)
}

/// Page-locked engram staging over one stable device plane.
pub struct Qwen38FlashNextEngramStager {
    arena: DeviceArena,
    stager: PinnedHostBuffer<u8>,
    rows: Vec<i64>,
    layout: Qwen38FlashNextEngramStagerLayout,
    context: Arc<CudaContext>,
    base_address: u64,
    staged_tokens: usize,
    /// Fence proving the last round's copy has finished reading the stager.
    upload: CudaEvent,
    /// Whether [`Self::upload`] has been recorded and not yet waited on.
    in_flight: bool,
    /// Times the reuse fence made the host wait for a round's copy to land.
    reuse_waits: u64,
}

impl Qwen38FlashNextEngramStager {
    /// Allocates the widest admitted round's stager, device plane, and row scratch.
    pub fn new(context: &Arc<CudaContext>) -> EngineResult<Self> {
        let layout = Qwen38FlashNextEngramStagerLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let stager =
            PinnedHostBuffer::zeroed(context, layout.stager_bytes()).map_err(GpuError::from)?;
        let scratch = layout.row_scratch_bytes() / size_of::<i64>();
        let mut rows = Vec::new();

        rows.try_reserve_exact(scratch)
            .map_err(|_| EngineError::layout("Flash-Next engram row scratch cannot be reserved"))?;
        rows.resize(scratch, 0);

        let base_address = arena.base_address();
        let upload = context.new_event(None).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(Self {
            arena,
            stager,
            rows,
            layout,
            context: context.clone(),
            base_address,
            staged_tokens: 0,
            upload,
            in_flight: false,
            reuse_waits: 0,
        })
    }

    /// Gathers one round of tokens and enqueues its upload to the stable plane.
    ///
    /// The caller owns `carry` and the borrowed table. Admission failures leave
    /// both the carry and the previous staged round unchanged.
    pub fn stage(
        &mut self,
        stream: &CudaStream,
        table: Qwen38FlashNextEngramTable<'_>,
        carry: &mut Qwen38FlashNextEngramCarry,
        tokens: &[u32],
    ) -> EngineResult<()> {
        let staged_bytes = self.layout.round_bytes(tokens.len())?;
        let staged_row_count = self.layout.round_rows(tokens.len())?;

        if table.token_bytes() != self.layout.token_bytes() {
            return Err(EngineError::layout(format!(
                "Flash-Next engram stages {} bytes per token, but this table contributes {}",
                self.layout.token_bytes(),
                table.token_bytes()
            )));
        }
        if stream.context().as_ref() != self.context.as_ref() {
            return Err(EngineError::layout(
                "Qwen3.8 Flash-Next engram staging requires one CUDA context",
            ));
        }
        let next_carry =
            require_gather_rows(table, *carry, tokens, &mut self.rows[..staged_row_count])?;

        // All admission checks precede the fence, so refusal preserves the
        // prior round and does not spend a wait.
        self.wait_for_upload()?;
        self.staged_tokens = 0;
        table.gather_rows(
            &self.rows[..staged_row_count],
            &mut self.stager[..staged_bytes],
        )?;

        // SAFETY: this owner retains the pinned stager and the arena at fixed addresses, and the
        // event recorded below is what proves this copy finished before the stager is written
        // again or freed.
        unsafe {
            self.arena.copy_prefix_from_pinned_host_async(
                stream,
                self.layout.embedding(),
                &self.stager,
                staged_bytes,
            )?;
        }
        if let Err(error) = self.upload.record(stream) {
            self.context.synchronize().map_err(GpuError::from)?;
            return Err(GpuError::from(error).into());
        }
        self.in_flight = true;
        self.staged_tokens = tokens.len();
        *carry = next_carry;

        Ok(())
    }

    /// Times the reuse fence made the host wait for a round's copy to land.
    ///
    /// Diagnostic evidence that the reuse fence ran.
    pub const fn reuse_waits(&self) -> u64 {
        self.reuse_waits
    }

    /// Blocks until the last enqueued round's copy has finished reading the stager.
    fn wait_for_upload(&mut self) -> EngineResult<()> {
        if !self.in_flight {
            return Ok(());
        }
        self.upload.synchronize().map_err(GpuError::from)?;
        self.in_flight = false;
        self.reuse_waits = self.reuse_waits.saturating_add(1);

        Ok(())
    }

    /// Checked staging layout.
    pub const fn layout(&self) -> &Qwen38FlashNextEngramStagerLayout {
        &self.layout
    }

    /// CUDA context both allocations belong to.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of the engram plane's arena.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Tokens the last round staged.
    pub const fn staged_tokens(&self) -> usize {
        self.staged_tokens
    }

    /// Device address of the stable engram plane a consuming owner reads.
    pub fn plane_address(&self) -> GpuResult<*const u8> {
        Ok(self.arena.address(self.layout.embedding())?.cast_const())
    }

    /// The pinned bytes the last round gathered.
    pub fn staged_window(&self) -> EngineResult<&[u8]> {
        let bytes = product(
            "Flash-Next engram staged bytes",
            self.staged_tokens,
            self.layout.token_bytes(),
        )?;

        Ok(&self.stager[..bytes])
    }

    /// Reads one round's staged bytes back from the device plane.
    #[cfg(feature = "qualification")]
    pub fn read_plane(&self, stream: &CudaStream, tokens: usize) -> EngineResult<Vec<u8>> {
        Ok(self.arena.copy_prefix_to_host(
            stream,
            self.layout.embedding(),
            self.layout.round_bytes(tokens)?,
        )?)
    }

    /// Enqueues a qualification consumer of the stable plane.
    ///
    /// # Safety
    ///
    /// Both arenas must outlive the copy or every replay of a graph that
    /// captures it. Their selected prefixes must not overlap.
    #[cfg(feature = "qualification")]
    pub unsafe fn copy_plane_to_arena_async(
        &self,
        stream: &CudaStream,
        destination: &DeviceArena,
        region: tuisko_gpu::ArenaRegion<u8>,
        tokens: usize,
    ) -> EngineResult<()> {
        let bytes = self.layout.round_bytes(tokens)?;
        // SAFETY: the caller owns both lifetimes and the non-overlap contract.
        unsafe {
            destination.copy_prefix_from_arena_async(
                stream,
                region,
                &self.arena,
                self.layout.embedding(),
                bytes,
            )?;
        }
        Ok(())
    }
}

impl Drop for Qwen38FlashNextEngramStager {
    /// Waits before the upload source and destination are freed.
    fn drop(&mut self) {
        if !self.in_flight {
            return;
        }
        let context = self.context.clone();
        context.record_err(context.bind_to_thread());
        context.record_err(self.upload.synchronize());
    }
}
