//! Qwen3.6 replay glue behind the shared single-slot text generator.

use crate::common::rope::{ROTARY_PAIRS, text_rope};
use crate::common::slots::device_zero_context;
use crate::common::text_generator::{
    ModelProgram, SingleSlotGenerationSession, SingleSlotTextGenerator,
    sealed::Sealed as ModelProgramSealed,
};
use crate::{EngineError, EngineResult, Qwen36ResidentModelProgram};
use std::sync::Arc;
use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream};
use tuisko_model::{CheckpointSnapshot, Qwen36Moe35B};

const NATIVE_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const MAX_NATIVE_PREFILL_TOKENS: usize = 128;

impl ModelProgramSealed for Qwen36ResidentModelProgram {}

impl ModelProgram for Qwen36ResidentModelProgram {
    type Arch = Qwen36Moe35B;

    #[cfg(feature = "qualification")]
    type QualificationAddresses = Vec<usize>;

    fn open_frontend(snapshot: &CheckpointSnapshot<Qwen36Moe35B>) -> EngineResult<TextFrontend> {
        Ok(TextFrontend::open_qwen36(snapshot)?)
    }

    fn load(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
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

impl SingleSlotTextGenerator<Qwen36ResidentModelProgram> {
    /// Opens the exact Qwen3.6 text generator on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Exact device bytes across the 42 retained Qwen3.6 arenas.
    pub const fn arena_bytes(&self) -> usize {
        self.program().layout().arena_bytes()
    }

    /// Source-backed decoder and endpoint weights resident on the device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.program().layout().resident_weight_bytes()
    }

    /// Maximum token capacity admitted by the pinned checkpoint.
    pub const fn context_capacity(&self) -> usize {
        self.program().context_capacity()
    }

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program().context()
    }
}

impl SingleSlotGenerationSession<'_, Qwen36ResidentModelProgram> {
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
    program: &mut Qwen36ResidentModelProgram,
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
    program: &mut Qwen36ResidentModelProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    program.load_slot_routes(stream, &[slot])?;
    let native_tokens = native_prefill_tokens(token_ids.len()).unwrap_or(0);
    if native_tokens != 0 {
        let rotary_values = native_tokens
            .checked_mul(ROTARY_PAIRS)
            .ok_or_else(|| EngineError::generation("Qwen3.6 prompt rotary values overflow"))?;
        let mut rope_cos = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
        let mut rope_sin = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
        for position in 0..native_tokens {
            let position_u32 = u32::try_from(position).map_err(|_| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?;
            let (cosine, sine) = text_rope(position_u32);
            let begin = position * ROTARY_PAIRS;
            rope_cos[begin..begin + ROTARY_PAIRS].copy_from_slice(&cosine);
            rope_sin[begin..begin + ROTARY_PAIRS].copy_from_slice(&sine);
        }
        program.stage_prefill_embeddings(stream, &token_ids[..native_tokens])?;
        let route = program.load_prefill_slot_state(
            stream,
            native_tokens,
            slot,
            &rope_cos[..rotary_values],
            &rope_sin[..rotary_values],
        )?;
        program.replay_prefill(stream, route)?;
    }

    for (offset, &token) in token_ids[native_tokens..].iter().enumerate() {
        let position = native_tokens + offset;
        replay_token(
            program,
            stream,
            token,
            u32::try_from(position).map_err(|_| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?,
        )?;
    }

    Ok(native_tokens)
}

const fn native_prefill_tokens(prompt_tokens: usize) -> Option<usize> {
    let mut index = NATIVE_PREFILL_ROUTES.len();
    while index != 0 {
        index -= 1;
        if NATIVE_PREFILL_ROUTES[index] <= prompt_tokens {
            return Some(NATIVE_PREFILL_ROUTES[index]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{NATIVE_PREFILL_ROUTES, native_prefill_tokens};

    #[test]
    fn qwen36_native_prefill_uses_the_largest_exact_prefix() {
        assert_eq!(NATIVE_PREFILL_ROUTES, [32, 64, 128]);
        for (prompt, expected) in [
            (0, None),
            (1, None),
            (31, None),
            (32, Some(32)),
            (33, Some(32)),
            (63, Some(32)),
            (64, Some(64)),
            (65, Some(64)),
            (127, Some(64)),
            (128, Some(128)),
            (129, Some(128)),
            (192, Some(128)),
            (usize::MAX, Some(128)),
        ] {
            assert_eq!(native_prefill_tokens(prompt), expected);
        }
    }
}
