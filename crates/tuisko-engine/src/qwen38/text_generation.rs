//! Qwen3.8 replay glue behind the shared single-slot text generator.

use crate::common::mtp::{MAX_NATIVE_PREFILL_TOKENS, next_native_prefill_tile};
use crate::common::rope::{ROTARY_PAIRS, text_rope};
use crate::common::text_generator::{
    ModelProgram, SingleSlotTextGenerator, sealed::Sealed as ModelProgramSealed,
};
use crate::{EngineError, EngineResult, ResidentModelProgram};
use std::sync::Arc;
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

impl ModelProgramSealed for ResidentModelProgram {}

impl ModelProgram for ResidentModelProgram {
    type Arch = Qwen38_27B;

    #[cfg(feature = "qualification")]
    type QualificationAddresses = [usize; 3];

    fn open_frontend(snapshot: &CheckpointSnapshot<Qwen38_27B>) -> EngineResult<TextFrontend> {
        Ok(TextFrontend::open(snapshot)?)
    }

    fn load(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
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
        self.recycle_kv_slot(stream, 0)?;
        self.reset_slot(stream, 0)?;
        self.activate_kv_slot(0)?;
        self.reserve_kv_slot_tokens(stream, 0, required_positions)?;
        self.load_slot_routes(stream, &[0])?;
        prime_prompt(self, stream, token_ids, 0, 0)
    }

    fn replay_token(&mut self, stream: &CudaStream, token: u32, position: u32) -> EngineResult<()> {
        replay_token(self, stream, token, position)
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

impl SingleSlotTextGenerator<ResidentModelProgram> {
    /// Exact device bytes owned by the resident model arena.
    pub const fn arena_bytes(&self) -> usize {
        self.program().arena_bytes()
    }

    /// Fixed host bytes owning shared page tables and physical-page tags.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program().kv_route_host_bytes()
    }

    /// Current short-context token capacity per single slot.
    pub const fn context_capacity(&self) -> usize {
        self.program().context_capacity()
    }

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program().context()
    }
}

fn replay_token(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    token: u32,
    position: u32,
) -> EngineResult<()> {
    let (rope_cos, rope_sin) = text_rope(position);
    program.stage_embeddings(stream, &[token])?;
    let route = program.load_decode_state(stream, 1, &[position], &rope_cos, &rope_sin)?;
    program.replay(stream, route)
}

pub(crate) fn prime_prompt(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    processed_prefix: usize,
) -> EngineResult<usize> {
    if processed_prefix > token_ids.len() {
        return Err(EngineError::generation(
            "resident processed prefix exceeds its prompt",
        ));
    }

    let mut cursor = processed_prefix;
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
    Ok(cursor - processed_prefix)
}

fn replay_prefill_tile(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    first_position: usize,
) -> EngineResult<()> {
    let rotary_values = token_ids
        .len()
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| EngineError::generation("prompt rotary values overflow"))?;
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
    program.stage_embeddings(stream, token_ids)?;
    let route = program.load_prefill_tile_state(
        stream,
        token_ids.len(),
        slot,
        first_position,
        &rope_cos[..rotary_values],
        &rope_sin[..rotary_values],
    )?;
    program.replay_prefill(stream, route)
}

#[cfg(test)]
mod tests {
    use crate::common::mtp::next_native_prefill_tile;

    #[test]
    fn native_prefill_plan_uses_only_the_exact_largest_first_inventory() {
        for (tokens, expected) in [
            (0, vec![]),
            (31, vec![]),
            (32, vec![32]),
            (63, vec![32]),
            (64, vec![64]),
            (96, vec![64, 32]),
            (127, vec![64, 32]),
            (128, vec![128]),
            (160, vec![128, 32]),
            (1_023, vec![128, 128, 128, 128, 128, 128, 128, 64, 32]),
            (1_024, vec![1_024]),
            (1_055, vec![1_024]),
            (1_056, vec![1_024, 32]),
            (1_152, vec![1_024, 128]),
            (2_208, vec![1_024, 1_024, 128, 32]),
        ] {
            let mut remaining = tokens;
            let mut actual = Vec::new();
            while let Some(tile) = next_native_prefill_tile(remaining) {
                actual.push(tile);
                remaining -= tile;
            }
            assert_eq!(actual, expected, "T={tokens}");
            assert!(remaining < 32);
        }
    }
}
