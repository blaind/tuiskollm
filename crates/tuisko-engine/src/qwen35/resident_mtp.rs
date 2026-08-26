//! Resident Qwen3.5 target plus one source-BF16 MTP draft program.

use crate::qwen35::mtp_kv::Qwen35MtpKvProgram;
use crate::{
    EngineError, EngineResult, MAX_BATCH, Qwen35MtpLayerProgram, Qwen35ResidentModelProgram,
    Qwen35ResidentMtpLayout, Qwen35ResidentPrefillRoute,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const PROMPT_ROUTES: [usize; 3] = [32, 64, 128];

/// Exact MTP prompt-prime route tied to one stable cache slot and target tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the Qwen3.5 MTP prompt route must be replayed after staging its inputs"]
pub struct Qwen35MtpPromptRoute {
    rows: usize,
    slot: usize,
    first_position: usize,
}

impl Qwen35MtpPromptRoute {
    /// Number of contiguous target-hidden and next-embedding rows.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Stable MTP cache row receiving the primed K/V values.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// Absolute position of the first primed row.
    pub const fn first_position(self) -> usize {
        self.first_position
    }
}

/// Target model and MTP layer sharing one endpoint and mirrored cache lifecycle.
pub struct Qwen35ResidentMtpProgram {
    // Drop cross-owner graphs before every allocation and module they retain.
    draft_graphs: [CudaGraph; MAX_BATCH],
    staged_draft_graphs: [CudaGraph; MAX_BATCH],
    continue_draft_graphs: [CudaGraph; MAX_BATCH],
    prompt_graphs: [CudaGraph; PROMPT_ROUTES.len()],
    mtp: Qwen35MtpLayerProgram,
    target: Qwen35ResidentModelProgram,
    mtp_kv: Qwen35MtpKvProgram,
    embedding_stager: PinnedHostBuffer<u16>,
    context: Arc<CudaContext>,
    layout: Qwen35ResidentMtpLayout,
}

impl Qwen35ResidentMtpProgram {
    /// Loads the target and MTP owners and captures every exact composed route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let layout = Qwen35ResidentMtpLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let target = Qwen35ResidentModelProgram::from_snapshot(context, Arc::clone(&snapshot))?;
        let mtp_kv = Qwen35MtpKvProgram::new(context)?;
        // SAFETY: field order drops `mtp` before `mtp_kv`; the outer graphs drop first.
        let mtp = unsafe {
            Qwen35MtpLayerProgram::from_snapshot_with_kv(
                context,
                snapshot.as_ref(),
                mtp_kv.binding()?,
            )?
        };
        let embedding_stager =
            PinnedHostBuffer::zeroed(context, 128 * Qwen35_9B::HIDDEN).map_err(GpuError::from)?;
        let draft_graphs = capture_draft_routes(&stream, &target, &mtp)?;
        let staged_draft_graphs = capture_staged_draft_routes(&stream, &target, &mtp)?;
        let continue_draft_graphs = capture_continue_draft_routes(&stream, &target, &mtp)?;
        let prompt_graphs = capture_prompt_routes(&stream, &target, &mtp)?;

        Ok(Self {
            draft_graphs,
            staged_draft_graphs,
            continue_draft_graphs,
            prompt_graphs,
            mtp,
            target,
            mtp_kv,
            embedding_stager,
            context: Arc::clone(context),
            layout,
        })
    }

    /// Gathers mmap-backed next-token embeddings into the MTP input plane.
    pub fn stage_mtp_embeddings(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
    ) -> EngineResult<()> {
        require_rows(token_ids.len())?;
        let values = token_ids
            .len()
            .checked_mul(Qwen35_9B::HIDDEN)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP embedding count overflows"))?;
        self.target
            .gather_embedding_rows(token_ids, &mut self.embedding_stager[..values])?;
        self.mtp
            .load_embeddings(stream, token_ids.len(), &self.embedding_stager[..values])
    }

    /// Stages compact target and MTP decode metadata over identical stable slots.
    pub fn load_decode_state(
        &self,
        stream: &CudaStream,
        positions: &[u32],
        slots: &[usize],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_rows(positions.len())?;
        self.require_active_positions(positions, slots)?;
        self.target.load_slot_routes(stream, slots)?;
        self.target
            .load_decode_state(stream, positions.len(), positions, rope_cos, rope_sin)?;
        self.mtp
            .load_compact_draft_state(stream, positions, slots, rope_cos, rope_sin)
    }

    /// Stages compact draft rows from explicit prior target-conditioned hidden values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage_continuation_draft(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        target_hidden: &[u16],
        positions: &[u32],
        slots: &[usize],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_rows(token_ids.len())?;
        self.require_active_positions(positions, slots)?;
        if token_ids.len() != slots.len() {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP continuation has {} tokens and {} slots",
                token_ids.len(),
                slots.len()
            )));
        }
        let values = token_ids
            .len()
            .checked_mul(Qwen35_9B::HIDDEN)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP continuation input overflows"))?;
        if target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP continuation hidden plane has {} values, expected {values}",
                target_hidden.len()
            )));
        }
        self.target
            .gather_embedding_rows(token_ids, &mut self.embedding_stager[..values])?;
        self.mtp.load_inputs(
            stream,
            token_ids.len(),
            &self.embedding_stager[..values],
            target_hidden,
        )?;
        self.mtp
            .load_compact_draft_state(stream, positions, slots, rope_cos, rope_sin)
    }

    /// Stages the exact next-token embeddings and metadata for prompt cache priming.
    pub fn stage_prompt_prime(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen35MtpPromptRoute> {
        require_prompt(token_ids.len())?;
        let values = token_ids
            .len()
            .checked_mul(Qwen35_9B::HIDDEN)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP prompt embedding count overflows"))?;
        self.target
            .gather_embedding_rows(token_ids, &mut self.embedding_stager[..values])?;
        self.mtp
            .load_embeddings(stream, token_ids.len(), &self.embedding_stager[..values])?;
        self.mtp.load_prompt_state(
            stream,
            token_ids.len(),
            slot,
            first_position,
            rope_cos,
            rope_sin,
        )?;
        let last_position = first_position
            .checked_add(token_ids.len())
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| EngineError::route("Qwen3.5 MTP prompt range overflows"))?;
        self.require_slot_route(slot, last_position)?;

        Ok(Qwen35MtpPromptRoute {
            rows: token_ids.len(),
            slot,
            first_position,
        })
    }

    /// Replays one complete MTP draft layer plus the target's shared BF16 endpoint.
    pub fn replay_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_rows(batch)?;
        // SAFETY: this owner retains all target, MTP, cache, and endpoint addresses.
        unsafe { self.draft_graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Replays a compact draft from explicitly uploaded target-hidden rows.
    pub(crate) fn replay_staged_draft(
        &self,
        stream: &CudaStream,
        batch: usize,
    ) -> EngineResult<()> {
        require_rows(batch)?;
        // SAFETY: staging populated the retained MTP target-hidden plane captured here.
        unsafe { self.staged_draft_graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Continues one singleton proposal from the preceding MTP residual boundary.
    pub fn replay_continue_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_rows(batch)?;
        // SAFETY: the graph reads and then replaces compact planes in this retained MTP owner.
        unsafe { self.continue_draft_graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Replays one exact prompt-prime graph after the target prompt graph.
    pub fn replay_prompt_prime(
        &self,
        stream: &CudaStream,
        route: Qwen35MtpPromptRoute,
    ) -> EngineResult<()> {
        let index = prompt_index(route.rows).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.5 MTP prompt rows {} are outside 32,64,128",
                route.rows
            ))
        })?;
        // SAFETY: the owner retains every address captured by this route.
        unsafe { self.prompt_graphs[index].launch(stream) }?;

        Ok(())
    }

    /// Stages target token embeddings through the shared endpoint stager.
    pub fn stage_target_embeddings(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
    ) -> EngineResult<()> {
        self.target.stage_embeddings(stream, token_ids)
    }

    /// Replays one exact target-model decode graph.
    pub fn replay_target(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        self.target.replay(stream, batch)
    }

    /// Stages one contiguous exact `K=1..4` target-verification span.
    pub fn stage_target_verify(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        if !(1..=4).contains(&token_ids.len()) {
            return Err(EngineError::route(format!(
                "Qwen3.5 target verification width {} is outside 1..=4",
                token_ids.len()
            )));
        }
        self.target.stage_embeddings(stream, token_ids)?;
        self.target.load_verify_state(
            stream,
            token_ids.len(),
            slot,
            first_position,
            rope_cos,
            rope_sin,
        )?;
        let last_position = first_position
            .checked_add(token_ids.len())
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| EngineError::route("Qwen3.5 target verification range overflows"))?;
        self.require_slot_route(slot, last_position)
    }

    /// Replays one exact causal target-verification graph.
    pub fn replay_target_verify(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        self.target.replay_verify(stream, rows)
    }

    /// Stages one exact target prompt tile.
    pub fn stage_target_prefill(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen35ResidentPrefillRoute> {
        self.target.stage_prefill_embeddings(stream, token_ids)?;
        self.target.load_prefill_slot_state_at(
            stream,
            token_ids.len(),
            slot,
            first_position,
            rope_cos,
            rope_sin,
        )
    }

    /// Replays one exact target prompt graph.
    pub fn replay_target_prefill(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<()> {
        self.target.replay_prefill(stream, route)
    }

    /// Reads active target or draft BF16 logits from the shared endpoint.
    pub fn read_logits(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        self.target.read_logits(stream, rows)
    }

    /// Reads active target or draft logits into one reusable pinned bank.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        self.target.read_logits_into(stream, rows, destination)
    }

    /// Reads active MTP residual rows for accepted-prefix realignment.
    pub fn read_mtp_residuals(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        self.mtp.read_residual_output(stream, rows)
    }

    pub(crate) fn read_mtp_residuals_into(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        self.mtp
            .read_residual_output_into(stream, rows, destination)
    }

    pub(crate) fn target(&self) -> &Qwen35ResidentModelProgram {
        &self.target
    }

    pub(crate) fn target_mut(&mut self) -> &mut Qwen35ResidentModelProgram {
        &mut self.target
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage_realign(
        &mut self,
        stream: &CudaStream,
        outputs: &[u32],
        target_hidden: &[u16],
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_rows(outputs.len())?;
        let values = outputs
            .len()
            .checked_mul(Qwen35_9B::HIDDEN)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP realignment inputs overflow"))?;
        if target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP realignment has {} target-hidden values, expected {values}",
                target_hidden.len()
            )));
        }
        self.target
            .gather_embedding_rows(outputs, &mut self.embedding_stager[..values])?;
        self.mtp.load_inputs(
            stream,
            outputs.len(),
            &self.embedding_stager[..values],
            target_hidden,
        )?;
        let end = first_position
            .checked_add(outputs.len())
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP realignment range overflows"))?;
        let positions = (first_position..end)
            .map(|position| {
                u32::try_from(position)
                    .map_err(|_| EngineError::generation("Qwen3.5 MTP position exceeds u32"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        self.mtp
            .load_realign_state(stream, outputs.len(), slot, &positions, rope_cos, rope_sin)
    }

    pub(crate) fn replay_realign(
        &self,
        stream: &CudaStream,
        rows: usize,
        prime_only: bool,
    ) -> EngineResult<()> {
        require_rows(rows)?;
        if prime_only {
            self.mtp.replay_prime(stream, rows)?;
        } else {
            self.mtp.replay_realign(stream, rows)?;
            // SAFETY: the MTP graph published the exact final normalized row and both resident
            // owners remain live until the ordered LM-head launch completes.
            unsafe {
                self.target.launch_lm_head_from(
                    stream,
                    1,
                    self.mtp
                        .final_normalized_address()?
                        .add((rows - 1) * Qwen35_9B::HIDDEN),
                )?
            };
        }

        Ok(())
    }

    /// Activates one target/MTP slot pair after verifying equal lifecycle state.
    pub fn activate_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.require_mirror(slot)?;
        self.target.activate_kv_slot(slot)?;
        self.mtp_kv.activate_slot(slot)?;
        self.require_mirror(slot)
    }

    /// Reserves identical target and MTP logical pages and checks every mapping.
    pub fn reserve_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<()> {
        self.require_mirror(slot)?;
        self.target
            .reserve_kv_slot_tokens(stream, slot, token_count)?;
        self.mtp_kv.reserve_slot_tokens(stream, slot, token_count)?;
        self.require_mirror(slot)
    }

    /// Truncates both mirrored page rows to one processed-token boundary.
    pub fn truncate_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<usize> {
        self.require_mirror(slot)?;
        let target = self
            .target
            .truncate_kv_slot_tokens(stream, slot, token_count)?;
        let mtp = self
            .mtp_kv
            .truncate_slot_tokens(stream, slot, token_count)?;
        if target != mtp {
            return Err(EngineError::generation(format!(
                "Qwen3.5 target/MTP truncation released {target}/{mtp} pages"
            )));
        }
        self.require_mirror(slot)?;

        Ok(target)
    }

    /// Retains both mirrored page rows for exact prefix reuse.
    pub fn retain_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.require_mirror(slot)?;
        self.target.retain_kv_slot(slot)?;
        self.mtp_kv.retain_slot(slot)?;
        self.require_mirror(slot)
    }

    /// Recycles both mirrored page rows and returns the released page count.
    pub fn recycle_kv_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        self.require_mirror(slot)?;
        let target = self.target.recycle_kv_slot(stream, slot)?;
        let mtp = self.mtp_kv.recycle_slot(stream, slot)?;
        if target != mtp {
            return Err(EngineError::generation(format!(
                "Qwen3.5 target/MTP recycle released {target}/{mtp} pages"
            )));
        }
        self.require_mirror(slot)?;

        Ok(target)
    }

    /// Clears target state and the complete MTP cache mirror.
    pub fn reset_state(&mut self, stream: &CudaStream) -> EngineResult<()> {
        self.target.reset_state(stream)?;
        self.mtp_kv.reset(stream)?;
        for slot in 0..MAX_BATCH {
            self.require_mirror(slot)?;
        }

        Ok(())
    }

    /// Aggregate target/MTP ownership accounting.
    pub const fn layout(&self) -> &Qwen35ResidentMtpLayout {
        &self.layout
    }

    /// CUDA context shared by every stable owner.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable target, MTP layer, MTP cache, and host-stager addresses.
    pub fn base_addresses(&self) -> Vec<u64> {
        self.target
            .base_addresses()
            .into_iter()
            .chain([self.mtp.base_address(), self.mtp_kv.base_address()])
            .collect()
    }

    /// Fixed page-locked embedding stager bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.target.host_stager_bytes() + self.embedding_stager.num_bytes()
    }

    /// Fixed target plus MTP host page-owner bytes.
    pub const fn kv_host_owner_bytes(&self) -> usize {
        self.target.kv_host_owner_bytes() + self.mtp_kv.host_owner_bytes()
    }

    fn require_active_positions(&self, positions: &[u32], slots: &[usize]) -> EngineResult<()> {
        if positions.len() != slots.len() {
            return Err(EngineError::layout(format!(
                "Qwen3.5 resident MTP decode has {} positions and {} slots",
                positions.len(),
                slots.len()
            )));
        }
        for (&position, &slot) in positions.iter().zip(slots) {
            self.require_slot_route(slot, position as usize)?;
        }

        Ok(())
    }

    fn require_slot_route(&self, slot: usize, position: usize) -> EngineResult<()> {
        self.require_mirror(slot)?;
        let tokens = self.target.kv_slot_token_count(slot)?;
        if position >= tokens {
            return Err(EngineError::route(format!(
                "Qwen3.5 resident MTP position {position} exceeds slot {slot}'s {tokens} reserved tokens"
            )));
        }
        if self.target.kv_route(slot, position)? != self.mtp_kv.route(slot, position)? {
            return Err(EngineError::generation(format!(
                "Qwen3.5 target/MTP slot {slot} maps position {position} differently"
            )));
        }

        Ok(())
    }

    fn require_mirror(&self, slot: usize) -> EngineResult<()> {
        let target_state = self.target.kv_slot_state(slot)?;
        let mtp_state = self.mtp_kv.state(slot)?;
        let target_tokens = self.target.kv_slot_token_count(slot)?;
        let mtp_tokens = self.mtp_kv.token_count(slot)?;
        if target_state != mtp_state || target_tokens != mtp_tokens {
            return Err(EngineError::generation(format!(
                "Qwen3.5 target/MTP slot {slot} lifecycle differs: {target_state:?}/{mtp_state:?}, {target_tokens}/{mtp_tokens} tokens"
            )));
        }
        for position in (0..target_tokens).step_by(64) {
            if self.target.kv_route(slot, position)? != self.mtp_kv.route(slot, position)? {
                return Err(EngineError::generation(format!(
                    "Qwen3.5 target/MTP slot {slot} page mapping differs at position {position}"
                )));
            }
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable composed draft graph.
    pub fn qualification_draft_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_rows(batch)?;
        Ok(&self.draft_graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Launches the same composed draft route without graph replay.
    pub fn qualification_launch_draft(
        &self,
        stream: &CudaStream,
        batch: usize,
    ) -> EngineResult<()> {
        require_rows(batch)?;
        launch_draft(
            stream,
            batch,
            &self.target,
            &self.mtp,
            self.target.final_residual_address()?,
        )?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one target-verification route without graph replay.
    pub fn qualification_launch_target_verify(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<()> {
        self.target.qualification_launch_verify(stream, rows)
    }

    #[cfg(feature = "qualification")]
    /// Launches one prompt-prime route without graph replay.
    pub fn qualification_launch_prompt(
        &self,
        stream: &CudaStream,
        route: Qwen35MtpPromptRoute,
    ) -> EngineResult<()> {
        require_prompt(route.rows)?;
        // SAFETY: the retained target residual covers the staged prompt rows.
        unsafe {
            self.mtp.launch_prompt_prime_from(
                stream,
                route.rows,
                self.target.final_residual_address()?,
            )
        }
    }

    #[cfg(feature = "qualification")]
    /// Fills the MTP and shared endpoint output planes with one sentinel byte.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        self.mtp.qualification_reset_outputs(stream, byte)?;
        self.target.qualification_reset_outputs(stream, byte)
    }

    #[cfg(feature = "qualification")]
    /// Reads all MTP seams plus the active shared-endpoint logits.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Qwen35ResidentMtpObservables> {
        require_rows(rows)?;
        Ok(Qwen35ResidentMtpObservables {
            mtp: self.mtp.qualification_observables(stream)?,
            logits: self.target.read_logits(stream, rows)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Stable target, MTP, cache, and pinned-stager addresses.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = self
            .target
            .base_addresses()
            .into_iter()
            .map(|address| address as usize)
            .collect::<Vec<_>>();
        addresses.extend(self.mtp.qualification_addresses()?);
        addresses.push(self.embedding_stager.as_ptr().addr());
        Ok(addresses)
    }

    #[cfg(feature = "qualification")]
    /// Physical page selected by both mirrored owners.
    pub fn qualification_kv_route(
        &self,
        slot: usize,
        position: usize,
    ) -> EngineResult<crate::PagedKvRoute> {
        self.require_slot_route(slot, position)?;
        self.target.kv_route(slot, position)
    }

    #[cfg(feature = "qualification")]
    /// Reads active target residual rows for serial/K-route comparison.
    pub fn qualification_target_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Vec<u16>> {
        self.target.read_final_residual(stream, rows)
    }

    #[cfg(feature = "qualification")]
    /// Reads one target physical K/V page from every attention layer.
    pub fn qualification_target_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        self.target.qualification_cache_page(stream, physical_page)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated composed draft routes for intrinsic timing.
    pub fn qualification_repeated_draft_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_rows(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.5 resident MTP graph requires at least one operation",
            ));
        }
        let target_hidden = self.target.final_residual_address()?;
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_draft(stream, batch, &self.target, &self.mtp, target_hidden)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Reads every MTP device block-table row.
    pub fn qualification_mtp_tables(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        self.mtp_kv.qualification_block_tables(stream)
    }

    #[cfg(feature = "qualification")]
    /// Reads equal prefixes of the external MTP key and value planes.
    pub fn qualification_mtp_cache_prefix(
        &self,
        stream: &CudaStream,
        values: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        self.mtp_kv.qualification_cache_prefix(stream, values)
    }
}

#[cfg(feature = "qualification")]
/// Complete MTP seams and shared-endpoint logits for composition qualification.
pub struct Qwen35ResidentMtpObservables {
    /// Every mutable MTP transformer-layer boundary.
    pub mtp: crate::Qwen35MtpLayerObservables,
    /// Active BF16 vocabulary rows emitted by the shared endpoint.
    pub logits: Vec<u16>,
}

fn capture_draft_routes(
    stream: &CudaStream,
    target: &Qwen35ResidentModelProgram,
    mtp: &Qwen35MtpLayerProgram,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let target_hidden = target.final_residual_address()?;
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_draft(stream, batch, target, mtp, target_hidden)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 resident MTP draft graph inventory is incomplete")
    })
}

fn capture_continue_draft_routes(
    stream: &CudaStream,
    target: &Qwen35ResidentModelProgram,
    mtp: &Qwen35MtpLayerProgram,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    // Continuing proposals previously used a B=1 graph once per lane. The
    // exact B=1..8 graphs retain the same per-row MTP arithmetic while sharing
    // one launch and one LM-head weight pass across compact active rows.
    let target_hidden = mtp.residual_output_address()?;
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_draft(stream, batch, target, mtp, target_hidden)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 resident MTP continuation inventory is incomplete")
    })
}

fn capture_staged_draft_routes(
    stream: &CudaStream,
    target: &Qwen35ResidentModelProgram,
    mtp: &Qwen35MtpLayerProgram,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let target_hidden = mtp.target_hidden_address()?;
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_draft(stream, batch, target, mtp, target_hidden)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 resident MTP staged-draft inventory is incomplete")
    })
}

fn capture_prompt_routes(
    stream: &CudaStream,
    target: &Qwen35ResidentModelProgram,
    mtp: &Qwen35MtpLayerProgram,
) -> EngineResult<[CudaGraph; PROMPT_ROUTES.len()]> {
    let target_hidden = target.final_residual_address()?;
    let mut graphs = Vec::with_capacity(PROMPT_ROUTES.len());
    for rows in PROMPT_ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            // SAFETY: both stable owners outlive the captured graph and replay.
            unsafe { mtp.launch_prompt_prime_from(stream, rows, target_hidden) }
                .map_err(engine_into_gpu)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 resident MTP prompt graph inventory is incomplete")
    })
}

fn launch_draft(
    stream: &CudaStream,
    batch: usize,
    target: &Qwen35ResidentModelProgram,
    mtp: &Qwen35MtpLayerProgram,
    target_hidden: *const u16,
) -> GpuResult<()> {
    // SAFETY: the resident owner retains both stable source and destination planes.
    unsafe { mtp.launch_draft_from(stream, batch, target_hidden) }.map_err(engine_into_gpu)?;
    // SAFETY: the MTP layer already applies its source final norm; the normalized
    // plane covers the exact batch and both owners outlive replay.
    unsafe { target.launch_lm_head_from(stream, batch, mtp.final_normalized_address()?) }
}

fn engine_into_gpu(error: crate::EngineError) -> GpuError {
    GpuError::invalid_launch(error.to_string())
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&rows) {
        return Err(EngineError::route(format!(
            "Qwen3.5 resident MTP rows {rows} are outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_prompt(rows: usize) -> EngineResult<()> {
    if prompt_index(rows).is_none() {
        return Err(EngineError::route(format!(
            "Qwen3.5 resident MTP prompt rows {rows} are outside 32,64,128"
        )));
    }
    Ok(())
}

fn prompt_index(rows: usize) -> Option<usize> {
    PROMPT_ROUTES.iter().position(|&route| route == rows)
}

#[cfg(test)]
mod tests {
    use super::{PROMPT_ROUTES, prompt_index, require_prompt, require_rows};

    #[test]
    fn qwen35_resident_mtp_route_tables_are_exact() {
        assert_eq!(PROMPT_ROUTES, [32, 64, 128]);
        assert_eq!(PROMPT_ROUTES.map(prompt_index), [Some(0), Some(1), Some(2)]);
        for rows in 1..=8 {
            require_rows(rows).unwrap();
        }
        for rows in PROMPT_ROUTES {
            require_prompt(rows).unwrap();
        }
        for rows in [0, 9, 16, 31, 129, usize::MAX] {
            assert!(require_rows(rows).is_err());
            assert!(require_prompt(rows).is_err());
        }
    }
}
