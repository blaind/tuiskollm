//! Qwen3.5 replay glue behind the shared single-slot text generator.

use crate::common::rope::{ROTARY_PAIRS, text_rope};
use crate::common::slots::device_zero_context;
use crate::common::text_generator::{
    ModelProgram, SingleSlotGenerationSession, SingleSlotTextGenerator,
    sealed::Sealed as ModelProgramSealed,
};
use crate::{EngineError, EngineResult, Qwen35ResidentModelProgram};
use std::sync::Arc;
use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream};
use tuisko_model::{CheckpointSnapshot, Qwen35_9B};

const NATIVE_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const MAX_NATIVE_PREFILL_TOKENS: usize = 128;

impl ModelProgramSealed for Qwen35ResidentModelProgram {}

impl ModelProgram for Qwen35ResidentModelProgram {
    type Arch = Qwen35_9B;

    #[cfg(feature = "qualification")]
    type QualificationAddresses = Vec<usize>;

    fn open_frontend(snapshot: &CheckpointSnapshot<Qwen35_9B>) -> EngineResult<TextFrontend> {
        Ok(TextFrontend::open_qwen35(snapshot)?)
    }

    fn load(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot(context, snapshot)
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.context()
    }

    fn context_capacity(&self) -> usize {
        self.context_capacity()
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
        self.reset_state(stream)?;
        self.activate_kv_slot(0)?;
        self.reserve_kv_slot_tokens(stream, 0, required_positions)?;
        prime_prompt(self, stream, token_ids, 0)
    }

    fn replay_token(&mut self, stream: &CudaStream, token: u32, position: u32) -> EngineResult<()> {
        replay_token(self, stream, token, position)
    }

    #[cfg(feature = "qualification")]
    fn qualification_addresses(&self, logits: usize) -> Vec<usize> {
        self.base_addresses()
            .into_iter()
            .map(|address| address as usize)
            .chain(core::iter::once(logits))
            .collect()
    }
}

impl SingleSlotTextGenerator<Qwen35ResidentModelProgram> {
    /// Opens the exact Qwen3.5 text generator on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Exact device bytes across the 34 retained Qwen3.5 arenas.
    pub const fn arena_bytes(&self) -> usize {
        self.program().layout().arena_bytes()
    }

    /// Source-backed decoder and endpoint weights resident on the device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.program().layout().resident_weight_bytes()
    }

    /// Maximum context admitted by the pinned Qwen3.5 config.
    pub const fn context_capacity(&self) -> usize {
        self.program().context_capacity()
    }

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program().context()
    }
}

impl SingleSlotGenerationSession<'_, Qwen35ResidentModelProgram> {
    /// Prompt encoding and frontend prefix-cache accounting.
    pub const fn prompt_encoding(&self) -> &PromptEncoding {
        self.control().prompt_encoding()
    }

    /// Observation-only frontend timing and prefix-lookup detail.
    pub const fn prompt_metrics(&self) -> &PromptEncodingMetrics {
        self.control().prompt_metrics()
    }
}

fn replay_token(
    program: &mut Qwen35ResidentModelProgram,
    stream: &CudaStream,
    token: u32,
    position: u32,
) -> EngineResult<()> {
    let (rope_cos, rope_sin) = text_rope(position);
    program.stage_embeddings(stream, &[token])?;
    program.load_decode_state(stream, 1, &[position], &rope_cos, &rope_sin)?;
    program.replay(stream, 1)
}

pub(crate) fn prime_prompt(
    program: &mut Qwen35ResidentModelProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    program.load_slot_routes(stream, &[slot])?;
    let mut cursor = 0;
    while let Some(tokens) = next_native_prefill_tile(token_ids.len() - cursor) {
        replay_prefill_tile(
            program,
            stream,
            &token_ids[cursor..cursor + tokens],
            slot,
            cursor,
        )?;
        cursor += tokens;
    }

    for (offset, &token) in token_ids[cursor..].iter().enumerate() {
        let position = cursor + offset;
        replay_token(
            program,
            stream,
            token,
            u32::try_from(position).map_err(|_| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?,
        )?;
    }

    Ok(cursor)
}

fn replay_prefill_tile(
    program: &mut Qwen35ResidentModelProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    first_position: usize,
) -> EngineResult<()> {
    let rotary_values = token_ids
        .len()
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| EngineError::generation("Qwen3.5 prompt rotary values overflow"))?;
    let mut rope_cos = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
    let mut rope_sin = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
    for token in 0..token_ids.len() {
        let position = first_position
            .checked_add(token)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?;
        let (cosine, sine) = text_rope(position);
        let begin = token * ROTARY_PAIRS;
        rope_cos[begin..begin + ROTARY_PAIRS].copy_from_slice(&cosine);
        rope_sin[begin..begin + ROTARY_PAIRS].copy_from_slice(&sine);
    }
    program.stage_prefill_embeddings(stream, token_ids)?;
    let route = program.load_prefill_slot_state_at(
        stream,
        token_ids.len(),
        slot,
        first_position,
        &rope_cos[..rotary_values],
        &rope_sin[..rotary_values],
    )?;
    program.replay_prefill(stream, route)
}

const fn next_native_prefill_tile(remaining: usize) -> Option<usize> {
    let mut index = NATIVE_PREFILL_ROUTES.len();
    while index != 0 {
        index -= 1;
        if NATIVE_PREFILL_ROUTES[index] <= remaining {
            return Some(NATIVE_PREFILL_ROUTES[index]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{NATIVE_PREFILL_ROUTES, next_native_prefill_tile};

    #[test]
    fn qwen35_native_prefill_tiles_cover_every_long_context_corner() {
        assert_eq!(NATIVE_PREFILL_ROUTES, [32, 64, 128]);
        for (prompt, expected_native, expected_tail) in [
            (0, 0, 0),
            (1, 0, 1),
            (31, 0, 31),
            (32, 32, 0),
            (33, 32, 1),
            (63, 32, 31),
            (64, 64, 0),
            (65, 64, 1),
            (127, 96, 31),
            (128, 128, 0),
            (129, 128, 1),
            (192, 192, 0),
            (193, 192, 1),
            (4_096, 4_096, 0),
            (262_143, 262_112, 31),
            (262_144, 262_144, 0),
        ] {
            let mut remaining = prompt;
            let mut native = 0;
            while let Some(tile) = next_native_prefill_tile(remaining) {
                assert!(NATIVE_PREFILL_ROUTES.contains(&tile));
                native += tile;
                remaining -= tile;
            }
            assert_eq!((native, remaining), (expected_native, expected_tail));
        }
    }
}
