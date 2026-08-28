//! Qwen3.8 Flash-Next generation routing and request telemetry.
//!
//! Requests stop at the proven 2,051-token dense band even when the paged cache is deeper.
//! Prompts use T=1024/128/64/32 graphs followed by at most 31 scalar rounds.

use crate::common::mtp::next_native_prefill_tile;
use crate::common::slots::device_zero_context;
use crate::common::text_generator::{
    ModelProgram, SingleSlotGenerationSession, SingleSlotTextGenerator,
    sealed::Sealed as ModelProgramSealed,
};
use crate::qwen38_flash_next::resident_model::{
    Qwen38FlashNextResidentModel, Qwen38FlashNextStepTelemetry, Qwen38FlashNextStreamingRoute,
};
use crate::{EngineError, EngineResult};
use std::sync::Arc;
use std::time::Duration;
use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream};
use tuisko_model::{CheckpointSnapshot, Qwen38FlashNext};

/// Slot every single-slot Flash-Next session owns.
const SINGLE_SLOT: usize = 0;

/// Whole-request evidence folded from per-step telemetry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextGenerationTelemetry {
    prime_rounds: usize,
    prime_tiles: usize,
    prime_scalar_rounds: usize,
    prime_rows: usize,
    prime: Duration,
    decode_rounds: usize,
    decode: Duration,
    expert_requests: usize,
    expert_hits: usize,
    expert_misses: usize,
    expert_h2d_bytes: usize,
    decode_expert_requests: usize,
    decode_expert_hits: usize,
    decode_expert_misses: usize,
    decode_expert_h2d_bytes: usize,
    embedding_h2d_bytes: usize,
    engram_h2d_bytes: usize,
    engram_rows: usize,
    kv_append_bytes: usize,
    publication_stalls: usize,
    overlapped_rounds: usize,
    readback_wait: Duration,
    residency_wait: Duration,
}

impl Qwen38FlashNextGenerationTelemetry {
    /// Whole-model rounds the prompt prime issued, tiles and scalar tail together.
    pub const fn prime_rounds(self) -> usize {
        self.prime_rounds
    }

    /// Prefill tiles the prompt prime replayed.
    pub const fn prime_tiles(self) -> usize {
        self.prime_tiles
    }

    /// Decode rounds the prompt prime's scalar tail replayed.
    pub const fn prime_scalar_rounds(self) -> usize {
        self.prime_scalar_rounds
    }

    /// Prompt tokens the prime carried.
    pub const fn prime_rows(self) -> usize {
        self.prime_rows
    }

    /// Host wall time spent inside prompt-prime model forwards.
    pub const fn prime_wall(self) -> Duration {
        self.prime
    }

    /// Decode rounds this request issued, one per generated token after the first.
    pub const fn decode_rounds(self) -> usize {
        self.decode_rounds
    }

    /// Wall time the decode rounds took together.
    pub const fn decode(self) -> Duration {
        self.decode
    }

    /// Mean wall time one decode round took.
    pub fn decode_ms_per_token(self) -> f64 {
        if self.decode_rounds == 0 {
            return 0.0;
        }

        self.decode.as_secs_f64() * 1_000.0 / self.decode_rounds as f64
    }

    /// Tokens per second the decode rounds sustained.
    pub fn decode_tokens_per_second(self) -> f64 {
        if self.decode.is_zero() {
            return 0.0;
        }

        self.decode_rounds as f64 / self.decode.as_secs_f64()
    }

    /// Expert selections the whole request made, prefill and decode together.
    pub const fn expert_requests(self) -> usize {
        self.expert_requests
    }

    /// Host-to-device expert bytes the whole request streamed.
    pub const fn expert_h2d_bytes(self) -> usize {
        self.expert_h2d_bytes
    }

    /// Whole-request expert hit rate over distinct per-round items.
    pub fn expert_hit_rate(self) -> f64 {
        hit_rate(self.expert_hits, self.expert_misses)
    }

    /// Expert hit rate over decode rounds.
    pub fn decode_expert_hit_rate(self) -> f64 {
        hit_rate(self.decode_expert_hits, self.decode_expert_misses)
    }

    /// Host-to-device expert bytes one generated token cost during decode.
    pub fn decode_expert_h2d_bytes_per_token(self) -> f64 {
        if self.decode_rounds == 0 {
            return 0.0;
        }

        self.decode_expert_h2d_bytes as f64 / self.decode_rounds as f64
    }

    /// Expert selections the decode rounds made.
    pub const fn decode_expert_requests(self) -> usize {
        self.decode_expert_requests
    }

    /// Token-embedding bytes this request uploaded.
    pub const fn embedding_h2d_bytes(self) -> usize {
        self.embedding_h2d_bytes
    }

    /// Engram FP8 bytes this request uploaded, and the rows its host hash addressed.
    pub const fn engram_h2d_bytes(self) -> usize {
        self.engram_h2d_bytes
    }

    /// Engram rows the host hash addressed across the whole request.
    pub const fn engram_rows(self) -> usize {
        self.engram_rows
    }

    /// Bytes this request appended to the paged K/V planes.
    pub const fn kv_append_bytes(self) -> usize {
        self.kv_append_bytes
    }

    /// Layer rounds that blocked on the streaming publication fence.
    pub const fn publication_stalls(self) -> usize {
        self.publication_stalls
    }

    /// Rounds whose publication overlapped host work.
    pub const fn overlapped_rounds(self) -> usize {
        self.overlapped_rounds
    }

    /// Host time blocked reading router selections.
    pub const fn readback_wait(self) -> Duration {
        self.readback_wait
    }

    /// Host time spent resolving expert residency.
    pub const fn residency_wait(self) -> Duration {
        self.residency_wait
    }

    /// Adds another request's counts and durations.
    pub fn absorb(&mut self, other: Self) {
        self.prime_rounds += other.prime_rounds;
        self.prime_tiles += other.prime_tiles;
        self.prime_scalar_rounds += other.prime_scalar_rounds;
        self.prime_rows += other.prime_rows;
        self.prime += other.prime;
        self.decode_rounds += other.decode_rounds;
        self.decode += other.decode;
        self.expert_requests += other.expert_requests;
        self.expert_hits += other.expert_hits;
        self.expert_misses += other.expert_misses;
        self.expert_h2d_bytes += other.expert_h2d_bytes;
        self.decode_expert_requests += other.decode_expert_requests;
        self.decode_expert_hits += other.decode_expert_hits;
        self.decode_expert_misses += other.decode_expert_misses;
        self.decode_expert_h2d_bytes += other.decode_expert_h2d_bytes;
        self.embedding_h2d_bytes += other.embedding_h2d_bytes;
        self.engram_h2d_bytes += other.engram_h2d_bytes;
        self.engram_rows += other.engram_rows;
        self.kv_append_bytes += other.kv_append_bytes;
        self.publication_stalls += other.publication_stalls;
        self.overlapped_rounds += other.overlapped_rounds;
        self.readback_wait += other.readback_wait;
        self.residency_wait += other.residency_wait;
    }

    /// Folds one prompt-prime round into the request.
    pub(crate) fn observe_prime(&mut self, step: &Qwen38FlashNextStepTelemetry, tile: bool) {
        self.prime_rounds += 1;
        if tile {
            self.prime_tiles += 1;
        } else {
            self.prime_scalar_rounds += 1;
        }
        self.prime_rows += step.rows();
        self.prime += step.forward();
        self.observe_common(step);
    }

    /// Folds one decode round into the request.
    pub(crate) fn observe_decode(&mut self, step: &Qwen38FlashNextStepTelemetry) {
        self.decode_rounds += 1;
        self.decode += step.forward();
        self.decode_expert_requests += step.expert_requests();
        self.decode_expert_h2d_bytes += step.expert_h2d_bytes();
        for layer in step.layers() {
            self.decode_expert_hits += layer.hits();
            self.decode_expert_misses += layer.misses();
        }
        self.observe_common(step);
    }

    fn observe_common(&mut self, step: &Qwen38FlashNextStepTelemetry) {
        self.expert_requests += step.expert_requests();
        self.expert_h2d_bytes += step.expert_h2d_bytes();
        self.embedding_h2d_bytes += step.embedding_h2d_bytes();
        self.engram_h2d_bytes += step.engram_h2d_bytes();
        self.engram_rows += step.engram_rows();
        self.kv_append_bytes += step.kv_append_bytes();
        self.readback_wait += step.expert_readback_wait();
        self.residency_wait += step.expert_residency_wait();
        for layer in step.layers() {
            self.expert_hits += layer.hits();
            self.expert_misses += layer.misses();
            if layer.stalled() {
                self.publication_stalls += 1;
            }
            if layer.transfer_in_flight() {
                self.overlapped_rounds += 1;
            }
        }
    }
}

fn hit_rate(hits: usize, misses: usize) -> f64 {
    if hits + misses == 0 {
        return 0.0;
    }

    hits as f64 / (hits + misses) as f64
}

impl ModelProgramSealed for Qwen38FlashNextResidentModel {}

impl ModelProgram for Qwen38FlashNextResidentModel {
    type Arch = Qwen38FlashNext;

    #[cfg(feature = "qualification")]
    type QualificationAddresses = [usize; 3];

    fn open_frontend(snapshot: &CheckpointSnapshot<Qwen38FlashNext>) -> EngineResult<TextFrontend> {
        Ok(TextFrontend::open(snapshot)?)
    }

    fn load(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot(context, snapshot)
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.context()
    }

    /// The proven dense band, not the funded cache depth.
    fn context_capacity(&self) -> usize {
        self.generation_capacity()
    }

    fn host_stager_bytes(&self) -> usize {
        self.host_stager_bytes()
    }

    fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        self.read_logits_into(stream, batch, destination)
    }

    fn prime_single_slot(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        required_positions: usize,
    ) -> EngineResult<usize> {
        self.recycle_slot(stream, SINGLE_SLOT)?;
        self.reset_generation_telemetry();
        self.reserve_slot(stream, SINGLE_SLOT, required_positions)?;
        prime_prompt(self, stream, token_ids, SINGLE_SLOT)
    }

    fn replay_token(&mut self, stream: &CudaStream, token: u32, position: u32) -> EngineResult<()> {
        let step = self.decode_step(stream, &[token], &[position], &[SINGLE_SLOT])?;
        self.observe_decode_round(&step);

        Ok(())
    }

    #[cfg(feature = "qualification")]
    fn qualification_addresses(&self, logits: usize) -> [usize; 3] {
        [
            self.base_address() as usize,
            self.kv_base_address() as usize,
            logits,
        ]
    }
}

impl SingleSlotTextGenerator<Qwen38FlashNextResidentModel> {
    /// Opens the Flash-Next text generator on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Longest sequence a request may reach, which is the proven dense band.
    pub fn context_capacity(&self) -> usize {
        self.program().generation_capacity()
    }

    /// Device bytes across the four Flash-Next arenas.
    pub fn device_resident_bytes(&self) -> EngineResult<usize> {
        self.program().layout().total_device_bytes()
    }

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program().context()
    }

    /// Whole-request streaming and timing evidence for the last started request.
    pub const fn telemetry(&self) -> Qwen38FlashNextGenerationTelemetry {
        self.program().generation_telemetry()
    }

    /// Captured executables the resident program retains.
    pub fn executables(&self) -> usize {
        self.program().executables()
    }

    /// Current expert-publication ordering route.
    pub const fn streaming_route(&self) -> Qwen38FlashNextStreamingRoute {
        self.program().streaming_route()
    }

    /// Selects the expert-publication ordering route.
    pub const fn set_streaming_route(&mut self, route: Qwen38FlashNextStreamingRoute) {
        self.program_mut().set_streaming_route(route);
    }
}

impl SingleSlotGenerationSession<'_, Qwen38FlashNextResidentModel> {
    /// Prompt encoding and frontend prefix-cache accounting.
    pub const fn prompt_encoding(&self) -> &PromptEncoding {
        self.control().prompt_encoding()
    }

    /// Observation-only frontend timing and prefix-lookup detail.
    pub const fn prompt_metrics(&self) -> &PromptEncodingMetrics {
        self.control().prompt_metrics()
    }
}

/// Primes with the widest admitted tiles, then scalar decode rounds.
pub(crate) fn prime_prompt(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    let cursor = prime_prompt_tiles(model, stream, token_ids, slot)?;

    for (offset, &token) in token_ids[cursor..].iter().enumerate() {
        let position = prompt_position(cursor + offset)?;
        let step = model.decode_step(stream, &[token], &[position], &[slot])?;
        model.observe_prime_round(&step, false);
    }

    Ok(cursor)
}

/// Replays one slot's tile ladder and returns the native tokens it covered.
pub(crate) fn prime_prompt_tiles(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    if token_ids.is_empty() {
        return Err(EngineError::generation(
            "Flash-Next generation requires a nonempty prompt",
        ));
    }
    let mut cursor = 0usize;

    while let Some(tokens) = next_native_prefill_tile(token_ids.len() - cursor) {
        let first = prompt_position(cursor)?;
        let step = model.prefill_tile(stream, &token_ids[cursor..cursor + tokens], first, slot)?;
        model.observe_prime_round(&step, true);
        cursor += tokens;
    }

    Ok(cursor)
}

pub(crate) fn prompt_position(position: usize) -> EngineResult<u32> {
    u32::try_from(position)
        .map_err(|_| EngineError::generation("Flash-Next prompt position exceeds the route width"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38_flash_next::layer_route::QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING;

    /// Every tile the shared planner emits for a prompt of `tokens`, and its scalar tail.
    fn plan(tokens: usize) -> (Vec<usize>, usize) {
        let mut tiles = Vec::new();
        let mut cursor = 0usize;
        while let Some(tile) = next_native_prefill_tile(tokens - cursor) {
            tiles.push(tile);
            cursor += tile;
        }

        (tiles, tokens - cursor)
    }

    #[test]
    fn the_shared_tile_planner_emits_only_routes_flash_next_captured() {
        for tokens in [1usize, 31, 32, 63, 127, 1_024, 2_046, 2_051] {
            let (tiles, tail) = plan(tokens);
            for tile in &tiles {
                assert!(
                    crate::qwen38_flash_next::layer_route::QWEN38_FLASH_NEXT_PREFILL_ROWS
                        .contains(tile),
                    "prompt {tokens} planned tile {tile}, which Flash-Next captured no graph for"
                );
            }
            assert!(tail < 32, "prompt {tokens} left a {tail}-token tail");
            assert_eq!(tiles.iter().sum::<usize>() + tail, tokens);
        }
    }

    #[test]
    fn the_boundary_sweeps_longest_prompt_costs_ten_tiles_and_thirty_scalar_rounds() {
        let (tiles, tail) = plan(2_046);

        assert_eq!(
            tiles,
            vec![1_024, 128, 128, 128, 128, 128, 128, 128, 64, 32]
        );
        assert_eq!(tail, 30);
        assert_eq!(tiles.len() + tail, 40);
    }

    #[test]
    fn a_prompt_below_the_narrowest_tile_is_all_scalar_tail() {
        let (tiles, tail) = plan(21);

        assert!(tiles.is_empty());
        assert_eq!(tail, 21);
    }

    #[test]
    fn the_generation_ceiling_is_the_dense_band_and_not_the_funded_cache() {
        let plan = crate::Qwen38FlashNextResidentLayout::build().unwrap();

        assert_eq!(plan.context_tokens_per_slot(), 29_376);
        assert_eq!(QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, 2_051);
        assert!(plan.context_tokens_per_slot() > QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING);
    }

    #[test]
    fn the_admitted_new_token_budget_is_the_bands_own_arithmetic() {
        // Admission evaluates `prompt + max_new - 1` against the dense ceiling.
        for (prompt, admitted) in [
            (2_046usize, 6usize),
            (2_047, 5),
            (2_048, 4),
            (2_049, 3),
            (2_050, 2),
            (2_051, 1),
            (2_055, 0),
            (2_099, 0),
        ] {
            let budget = (QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING + 1).saturating_sub(prompt);
            assert_eq!(budget, admitted, "prompt {prompt}");
        }
    }

    #[test]
    fn telemetry_separates_prefill_from_decode() {
        let telemetry = Qwen38FlashNextGenerationTelemetry::default();

        assert_eq!(telemetry.expert_hit_rate(), 0.0);
        assert_eq!(telemetry.decode_expert_hit_rate(), 0.0);
        assert_eq!(telemetry.decode_ms_per_token(), 0.0);
        assert_eq!(telemetry.decode_tokens_per_second(), 0.0);
        assert_eq!(telemetry.decode_expert_h2d_bytes_per_token(), 0.0);
        assert_eq!(telemetry.prime_wall(), Duration::ZERO);
    }

    #[test]
    fn a_hit_rate_over_no_rounds_is_zero_rather_than_a_division() {
        assert_eq!(hit_rate(0, 0), 0.0);
        assert_eq!(hit_rate(9, 1), 0.9);
        assert_eq!(hit_rate(0, 5), 0.0);
        assert_eq!(hit_rate(5, 0), 1.0);
    }
}
