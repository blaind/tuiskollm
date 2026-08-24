//! Represented-value qualification for provisional GDN state snapshots.

use crate::device_benchmark::{preflight, require_current_process_exclusive};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::GdnStateSnapshotOp;
use tuisko_model::{Arch, Qwen38_27B};

const ALIGNMENT: usize = 256;
const ROWS: usize = 8;
const SELECTED_ROW: u32 = 7;
const HISTORY_VALUES: usize = Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
const STATE_VALUES: usize =
    Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::LINEAR_HEAD_DIM * Qwen38_27B::LINEAR_HEAD_DIM;

/// Failure of the exact provisional-state snapshot gate.
#[derive(Debug, thiserror::Error)]
pub enum GdnStateSnapshotQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device behavior disagreed with the represented-value oracle.
    #[error("GDN state snapshot qualification failed: {0}")]
    Mismatch(String),
}

/// Exact represented-value, replay, address, and byte counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnStateSnapshotQualification {
    /// Source and destination words checked bit-exactly.
    pub represented_values: usize,
    /// Destination words reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Exact payload bytes in the one-allocation fixture.
    pub payload_bytes: usize,
    /// Complete fixture arena bytes.
    pub arena_bytes: usize,
    /// Alignment padding bytes.
    pub padding_bytes: usize,
}

#[derive(Clone, Copy)]
struct Regions {
    source_row: ArenaRegion<u32>,
    history: ArenaRegion<u16>,
    state: ArenaRegion<f32>,
    scratch_history: ArenaRegion<u16>,
    scratch_state: ArenaRegion<f32>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.source_row.byte_len()
            + self.history.byte_len()
            + self.state.byte_len()
            + self.scratch_history.byte_len()
            + self.scratch_state.byte_len()
    }
}

/// Qualifies eager and CUDA Graph snapshot agreement against a bit oracle.
pub fn qualify_gdn_state_snapshot()
-> Result<GdnStateSnapshotQualification, GdnStateSnapshotQualificationError> {
    preflight().map_err(|error| GdnStateSnapshotQualificationError::Mismatch(error.to_string()))?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(GdnStateSnapshotQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let history = (0..ROWS * HISTORY_VALUES)
        .map(|index| (index as u16).rotate_left((index & 15) as u32) ^ 0xa55a)
        .collect::<Vec<_>>();
    let state = (0..ROWS * STATE_VALUES)
        .map(|index| f32::from_bits((index as u32).wrapping_mul(0x9e37_79b9) | 1))
        .collect::<Vec<_>>();
    arena.copy_from_host(&stream, regions.source_row, &[SELECTED_ROW])?;
    arena.copy_from_host(&stream, regions.history, &history)?;
    arena.copy_from_host(&stream, regions.state, &state)?;
    let op = GdnStateSnapshotOp::new(&context)?;
    require_current_process_exclusive()
        .map_err(|error| GdnStateSnapshotQualificationError::Mismatch(error.to_string()))?;
    let stable_addresses = addresses(&arena, regions)?;

    reset(&arena, &stream, regions)?;
    launch(&op, &arena, &stream, regions)?;
    let eager_history = arena.copy_to_host(&stream, regions.scratch_history)?;
    let eager_state = arena.copy_to_host(&stream, regions.scratch_state)?;
    verify(
        &arena,
        &stream,
        regions,
        &history,
        &state,
        &eager_history,
        &eager_state,
    )?;

    reset(&arena, &stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions))?;
    // SAFETY: every allocation this graph captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    unsafe { graph.launch(&stream) }?;
    let replay_history = arena.copy_to_host(&stream, regions.scratch_history)?;
    let replay_state = arena.copy_to_host(&stream, regions.scratch_state)?;
    compare_words("graph history", &replay_history, &eager_history)?;
    compare_f32_bits("graph state", &replay_state, &eager_state)?;

    // SAFETY: every allocation this graph captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    unsafe { graph.launch(&stream) }?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(&context)?;
    for _ in 0..16 {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(&context)?;
    if before != after || addresses(&arena, regions)? != stable_addresses {
        return Err(GdnStateSnapshotQualificationError::Mismatch(
            "snapshot allocation or addresses changed after warmup".to_string(),
        ));
    }

    let payload_bytes = regions.payload_bytes();
    Ok(GdnStateSnapshotQualification {
        represented_values: history.len() + state.len() + eager_history.len() + eager_state.len(),
        graph_replay_values: replay_history.len() + replay_state.len(),
        payload_bytes,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - payload_bytes,
    })
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let regions = Regions {
        source_row: layout.reserve(1, ALIGNMENT)?,
        history: layout.reserve(ROWS * HISTORY_VALUES, ALIGNMENT)?,
        state: layout.reserve(ROWS * STATE_VALUES, ALIGNMENT)?,
        scratch_history: layout.reserve(HISTORY_VALUES, ALIGNMENT)?,
        scratch_state: layout.reserve(STATE_VALUES, ALIGNMENT)?,
    };
    Ok((layout, regions))
}

fn reset(arena: &DeviceArena, stream: &tuisko_gpu::CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.scratch_history, 0xa5)?;
    arena.fill(stream, regions.scratch_state, 0xa5)
}

fn launch(
    op: &GdnStateSnapshotOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    // SAFETY: all fixture regions are 256-byte aligned, disjoint, and cover
    // eight source rows plus one exact destination row.
    unsafe {
        op.launch(
            stream,
            arena.address(regions.source_row)?,
            arena.address(regions.history)?,
            arena.address(regions.state)?,
            arena.address(regions.scratch_history)?,
            arena.address(regions.scratch_state)?,
        )
    }
}

fn verify(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    history: &[u16],
    state: &[f32],
    scratch_history: &[u16],
    scratch_state: &[f32],
) -> Result<(), GdnStateSnapshotQualificationError> {
    let row = SELECTED_ROW as usize;
    compare_words(
        "selected history",
        scratch_history,
        &history[row * HISTORY_VALUES..(row + 1) * HISTORY_VALUES],
    )?;
    compare_f32_bits(
        "selected state",
        scratch_state,
        &state[row * STATE_VALUES..(row + 1) * STATE_VALUES],
    )?;
    compare_words(
        "immutable history",
        &arena.copy_to_host(stream, regions.history)?,
        history,
    )?;
    compare_f32_bits(
        "immutable state",
        &arena.copy_to_host(stream, regions.state)?,
        state,
    )
}

fn compare_words(
    role: &str,
    actual: &[u16],
    expected: &[u16],
) -> Result<(), GdnStateSnapshotQualificationError> {
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        return Err(GdnStateSnapshotQualificationError::Mismatch(format!(
            "{role} differs at word {index}"
        )));
    }
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), GdnStateSnapshotQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(GdnStateSnapshotQualificationError::Mismatch(format!(
            "{role} differs at word {index}"
        )));
    }
    Ok(())
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 5]> {
    Ok([
        arena.address(regions.source_row)?.addr(),
        arena.address(regions.history)?.addr(),
        arena.address(regions.state)?.addr(),
        arena.address(regions.scratch_history)?.addr(),
        arena.address(regions.scratch_state)?.addr(),
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        GdnStateSnapshotQualificationError, HISTORY_VALUES, ROWS, STATE_VALUES, layout,
        qualify_gdn_state_snapshot,
    };

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn selected_row_is_bit_exact_under_eager_and_graph_replay()
    -> Result<(), GdnStateSnapshotQualificationError> {
        let report = qualify_gdn_state_snapshot()?;
        assert_eq!(
            report.represented_values,
            (ROWS + 1) * (HISTORY_VALUES + STATE_VALUES)
        );
        assert_eq!(report.graph_replay_values, HISTORY_VALUES + STATE_VALUES);
        assert_eq!(
            report.arena_bytes,
            report.payload_bytes + report.padding_bytes
        );
        Ok(())
    }

    #[test]
    fn owner_accounting_is_exact() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(regions.payload_bytes(), 28_864_516);
        assert_eq!(layout.byte_len(), regions.payload_bytes() + 252);
    }
}
