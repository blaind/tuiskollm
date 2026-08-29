//! Single-slot greedy and unbiased sampled generation over the resident target-plus-MTP owner.

use crate::common::mtp::{
    DRAFT_WINDOW, MAX_NATIVE_PREFILL_TOKENS, MtpRoundRope, ResidentMtpGenerationStats, VERIFY_ROWS,
    decide_greedy_round, decide_sampled_round, next_native_prefill_tile,
    require_generation_capacity,
};
use crate::common::rope::{ROTARY_PAIRS, fill_contiguous_rope, text_rope};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, ResidentMtpProgram, ResidentMtpVerifyRoute,
    SamplingDistribution,
};
use std::sync::Arc;
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const LOGIT_ROWS: usize = VERIFY_ROWS + 1;

/// Concrete single-slot owner for exact draft-three generation.
pub struct ResidentMtpTextGenerator {
    frontend: TextFrontend,
    program: ResidentMtpProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
}

/// One streaming request borrowing the resident target-plus-MTP program.
pub struct ResidentMtpGenerationSession<'a> {
    control: GenerationSession,
    program: &'a mut ResidentMtpProgram,
    stream: &'a CudaStream,
    logits: &'a mut PinnedHostBuffer<u16>,
    maximum_new_tokens: usize,
    stop_ids: [u32; 2],
    next_position: usize,
    started: bool,
    proposal_ready: bool,
    queued: [Option<GenerationStep>; VERIFY_ROWS],
    queue_start: usize,
    queue_len: usize,
    visible_generated: usize,
    native_prefill_tokens: usize,
    greedy: bool,
    stats: ResidentMtpGenerationStats,
}

impl ResidentMtpTextGenerator {
    /// Admits the pinned frontend and adds the exact resident MTP owner to the target program.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let program = ResidentMtpProgram::from_snapshot(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen38_27B::VOCAB
            .checked_mul(LOGIT_ROWS)
            .ok_or_else(|| EngineError::layout("resident MTP generation logits overflow"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
        })
    }

    /// Renders one admitted request and primes both target and MTP prompt state.
    pub fn start<'a>(
        &'a mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentMtpGenerationSession<'a>> {
        let greedy = request.sampling.is_greedy();
        let stop_ids =
            self.frontend.stop_ids().try_into().map_err(|_| {
                EngineError::generation("frontend returned the wrong stop-ID count")
            })?;
        let Self {
            frontend,
            program,
            stream,
            logits,
        } = self;
        let control = GenerationSession::start(frontend, request)?;
        let prompt_tokens = control.prompt_token_ids().len();
        let required_positions = require_generation_capacity(
            prompt_tokens,
            request.max_new_tokens,
            program.target().context_capacity(),
        )?;
        let mut native_prefill_tokens = 0;

        if control.finish_reason().is_none() {
            program.recycle_kv_slot(stream, 0)?;
            program.reset_slot(stream, 0)?;
            program.activate_kv_slot(0)?;
            program.reserve_kv_slot_tokens(stream, 0, required_positions)?;
            program.target().load_slot_routes(stream, &[0])?;
            native_prefill_tokens =
                prime_prompt(program, stream, control.prompt_token_ids(), 0, 0, None)?;
            let target = target_logits_mut(logits, 1);
            program.target().read_logits_into(stream, 1, target)?;
        }

        Ok(ResidentMtpGenerationSession {
            control,
            program,
            stream,
            logits,
            maximum_new_tokens: request.max_new_tokens,
            stop_ids,
            next_position: prompt_tokens,
            started: false,
            proposal_ready: false,
            queued: std::array::from_fn(|_| None),
            queue_start: 0,
            queue_len: 0,
            visible_generated: 0,
            native_prefill_tokens,
            greedy,
            stats: ResidentMtpGenerationStats::default(),
        })
    }

    /// Complete target and incremental MTP device allocation bytes.
    pub const fn device_owner_bytes(&self) -> usize {
        self.program.target().arena_bytes() + self.program.owner_bytes()
    }

    /// Page-locked target, MTP, and generation-logit staging bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.target().host_stager_bytes()
            + self.program.host_stager_bytes()
            + self.logits.num_bytes()
    }

    /// Fixed shared page-routing ownership on the host.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program.target().kv_route_host_bytes()
    }

    /// Exact long-context capacity shared by target and MTP state.
    pub const fn context_capacity(&self) -> usize {
        self.program.target().context_capacity()
    }

    /// CUDA context shared by the complete owner and generation stream.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Stable target, MTP, cache, and pinned-logit addresses.
    pub fn qualification_addresses(&self) -> [usize; 5] {
        [
            self.program.target().base_address() as usize,
            self.program.target().kv_base_address() as usize,
            self.program.base_address() as usize,
            self.program.cache_base_address() as usize,
            self.logits.as_ptr().addr(),
        ]
    }

    #[cfg(feature = "qualification")]
    /// Complete owner exposed only to source-backed qualification and direct benchmarks.
    pub const fn qualification_program(&self) -> &ResidentMtpProgram {
        &self.program
    }
}

impl ResidentMtpGenerationSession<'_> {
    /// Exact prompt token IDs selected by the admitted frontend.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.control.prompt_token_ids()
    }

    /// Tokens already returned to the streaming caller.
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.control.generated_token_ids()[..self.visible_generated]
    }

    /// Prompt tokens processed by native target and matching MTP prompt tiles.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.native_prefill_tokens
    }

    /// Current terminal state after every queued speculative output is visible.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        (self.queue_len == 0)
            .then(|| self.control.finish_reason())
            .flatten()
    }

    /// Exact route and acceptance counters for this request.
    pub const fn stats(&self) -> ResidentMtpGenerationStats {
        self.stats
    }

    /// Returns one streaming token, draining an already-executed round before launching another.
    pub fn step(&mut self) -> EngineResult<GenerationStep> {
        if self.queue_len != 0 {
            return self.take_queued();
        }
        if self.control.finish_reason().is_some() {
            return Err(EngineError::generation(
                "cannot step resident MTP generation after it finished",
            ));
        }
        if !self.started {
            return self.start_anchor();
        }

        let remaining = self
            .maximum_new_tokens
            .checked_sub(self.control.generated_token_ids().len())
            .ok_or_else(|| EngineError::generation("resident MTP generation budget underflows"))?;
        if remaining == 1 {
            return self.run_final_target_step();
        }
        self.run_speculative_round(remaining)?;
        self.take_queued()
    }

    /// Converts a completely drained terminal session into its complete output.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        if self.queue_len != 0 {
            return Err(EngineError::generation(
                "cannot take resident MTP output before queued steps are drained",
            ));
        }
        self.control.into_output()
    }

    fn start_anchor(&mut self) -> EngineResult<GenerationStep> {
        let step = self.control.accept_logits(target_logits(self.logits, 0))?;
        self.started = true;
        self.visible_generated = 1;
        if step.finish_reason.is_none() {
            let position = self.next_position.checked_sub(1).ok_or_else(|| {
                EngineError::generation("resident MTP anchor position underflows")
            })?;
            self.seed_proposal(step.token_id, position)?;
        }
        Ok(step)
    }

    fn seed_proposal(&mut self, token: u32, position: usize) -> EngineResult<()> {
        let position_u32 = u32::try_from(position)
            .map_err(|_| EngineError::generation("resident MTP position exceeds u32"))?;
        let (cosine, sine) = text_rope(position_u32);
        let route = self.program.stage_draft(
            self.stream,
            &[0],
            &[position_u32],
            &[token],
            &cosine,
            &sine,
        )?;
        self.program.replay_draft(self.stream, route)?;
        let draft = draft_logits_mut(self.logits);
        self.program.read_logits_into(self.stream, 1, draft)?;
        self.proposal_ready = true;
        Ok(())
    }

    fn continue_proposal(&mut self, token: u32, position: usize) -> EngineResult<()> {
        let position_u32 = u32::try_from(position)
            .map_err(|_| EngineError::generation("resident MTP position exceeds u32"))?;
        let (cosine, sine) = text_rope(position_u32);
        let route = self.program.stage_draft(
            self.stream,
            &[0],
            &[position_u32],
            &[token],
            &cosine,
            &sine,
        )?;
        self.program.replay_continue_draft(self.stream, route)?;
        let draft = draft_logits_mut(self.logits);
        self.program.read_logits_into(self.stream, 1, draft)?;
        Ok(())
    }

    fn run_final_target_step(&mut self) -> EngineResult<GenerationStep> {
        let anchor = *self
            .control
            .generated_token_ids()
            .last()
            .ok_or_else(|| EngineError::generation("resident MTP final step has no anchor"))?;
        let route = self.verify_target(&[anchor])?;
        let step = self.control.accept_logits(target_logits(self.logits, 0))?;
        self.program
            .target()
            .replay_target_mtp_commit(self.stream, route, 1)?;
        self.realign(&[step.token_id], true)?;
        self.next_position = self
            .next_position
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident MTP position overflows"))?;
        self.stats.verification_routes[0] += 1;
        self.stats.verified_outputs += 1;
        self.proposal_ready = false;
        self.visible_generated += 1;
        Ok(step)
    }

    fn run_speculative_round(&mut self, remaining: usize) -> EngineResult<()> {
        if !self.proposal_ready {
            return Err(EngineError::generation(
                "resident MTP speculative round has no aligned proposal",
            ));
        }
        let extent = DRAFT_WINDOW.min(remaining - 1);
        let mut drafts = [0u32; DRAFT_WINDOW];
        let mut draft_laws: [Option<SamplingDistribution>; DRAFT_WINDOW] =
            std::array::from_fn(|_| None);
        for draft in 0..extent {
            let draft_token = if self.greedy {
                self.control
                    .propose_logits(draft_logits(self.logits), &drafts[..draft])?
                    .token_id
            } else {
                let law = self
                    .control
                    .sampling_distribution(draft_logits(self.logits), &drafts[..draft])?;
                let token = self.control.draw_distribution(&law)?;
                draft_laws[draft] = Some(law);
                token
            };
            drafts[draft] = draft_token;
            self.stats.draft_proposals += 1;
            if draft + 1 < extent {
                let position = self
                    .next_position
                    .checked_add(draft)
                    .ok_or_else(|| EngineError::generation("resident MTP position overflows"))?;
                self.continue_proposal(draft_token, position)?;
            }
        }

        let anchor = *self
            .control
            .generated_token_ids()
            .last()
            .ok_or_else(|| EngineError::generation("resident MTP round has no anchor"))?;
        let mut inputs = [0u32; VERIFY_ROWS];
        inputs[0] = anchor;
        inputs[1..extent + 1].copy_from_slice(&drafts[..extent]);
        let route = self.verify_target(&inputs[..extent + 1])?;

        let (committed, accepted) = if self.greedy {
            self.decide_greedy_round(&drafts[..extent])?
        } else {
            self.decide_sampled_round(&drafts[..extent], &draft_laws[..extent])?
        };
        if committed == 0 {
            return Err(EngineError::generation(
                "resident MTP verification committed no output",
            ));
        }

        self.program
            .target()
            .replay_target_mtp_commit(self.stream, route, committed)?;
        let mut outputs = [0u32; VERIFY_ROWS];
        for (index, step) in self.queued[..committed].iter().enumerate() {
            outputs[index] = step.as_ref().expect("committed MTP step exists").token_id;
        }
        let terminal = self.queued[committed - 1]
            .as_ref()
            .expect("committed MTP step exists")
            .finish_reason
            .is_some();
        self.realign(&outputs[..committed], terminal)?;
        self.next_position = self
            .next_position
            .checked_add(committed)
            .ok_or_else(|| EngineError::generation("resident MTP position overflows"))?;
        self.queue_start = 0;
        self.queue_len = committed;
        self.stats.verification_routes[route.tokens() - 1] += 1;
        self.stats.accepted_drafts += accepted;
        self.stats.verified_outputs += committed;
        self.proposal_ready = !terminal;
        Ok(())
    }

    fn decide_greedy_round(&mut self, drafts: &[u32]) -> EngineResult<(usize, usize)> {
        decide_greedy_round(
            &mut self.control,
            &mut self.queued,
            self.logits,
            Qwen38_27B::VOCAB,
            drafts,
        )
    }

    fn decide_sampled_round(
        &mut self,
        drafts: &[u32],
        draft_laws: &[Option<SamplingDistribution>],
    ) -> EngineResult<(usize, usize)> {
        decide_sampled_round(
            &mut self.control,
            &mut self.queued,
            self.logits,
            Qwen38_27B::VOCAB,
            &self.stop_ids,
            drafts,
            draft_laws,
        )
    }

    fn verify_target(&mut self, inputs: &[u32]) -> EngineResult<ResidentMtpVerifyRoute> {
        self.program
            .target_mut()
            .stage_embeddings(self.stream, inputs)?;
        let rope = MtpRoundRope::contiguous(self.next_position, inputs.len())?;
        let route = self.program.target().load_target_mtp_verify_state(
            self.stream,
            inputs.len(),
            0,
            self.next_position,
            rope.cosine(),
            rope.sine(),
        )?;
        self.program
            .target()
            .replay_target_mtp_verify(self.stream, route)?;
        let target = target_logits_mut(self.logits, inputs.len());
        self.program
            .target()
            .read_logits_into(self.stream, inputs.len(), target)?;
        Ok(route)
    }

    fn realign(&mut self, outputs: &[u32], prime_only: bool) -> EngineResult<()> {
        let rope = MtpRoundRope::contiguous(self.next_position, outputs.len())?;
        let route = self.program.stage_realign(
            self.stream,
            outputs.len(),
            0,
            self.next_position,
            outputs,
            rope.cosine(),
            rope.sine(),
        )?;
        if prime_only {
            self.program.replay_prime(self.stream, route)?;
        } else {
            self.program.replay_realign(self.stream, route)?;
            let draft = draft_logits_mut(self.logits);
            self.program
                .read_logit_row_into(self.stream, outputs.len() - 1, draft)?;
        }
        Ok(())
    }

    fn take_queued(&mut self) -> EngineResult<GenerationStep> {
        let index = self.queue_start;
        let step = self.queued[index]
            .take()
            .ok_or_else(|| EngineError::generation("resident MTP output queue is incomplete"))?;
        self.queue_start += 1;
        self.queue_len -= 1;
        self.visible_generated += 1;
        if self.queue_len == 0 {
            self.queue_start = 0;
        }
        Ok(step)
    }
}

pub(crate) fn prime_prompt(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    processed_prefix: usize,
    boundary_hidden: Option<&[u16]>,
) -> EngineResult<usize> {
    prime_prompt_with_progress(
        program,
        stream,
        token_ids,
        slot,
        processed_prefix,
        boundary_hidden,
        &mut |_| Ok(()),
    )
}

pub(crate) fn prime_prompt_with_progress(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    processed_prefix: usize,
    boundary_hidden: Option<&[u16]>,
    progress: &mut impl FnMut(usize) -> EngineResult<()>,
) -> EngineResult<usize> {
    if token_ids.is_empty() {
        return Err(EngineError::generation(
            "resident MTP generation requires a nonempty prompt",
        ));
    }
    if processed_prefix > token_ids.len() {
        return Err(EngineError::generation(
            "resident MTP processed prefix exceeds its prompt",
        ));
    }
    if processed_prefix == token_ids.len() {
        return Ok(0);
    }
    if processed_prefix != 0 {
        let hidden = boundary_hidden.ok_or_else(|| {
            EngineError::generation("resident MTP prefix reuse has no boundary hidden row")
        })?;
        let position = u32::try_from(processed_prefix - 1)
            .map_err(|_| EngineError::generation("resident MTP prefix boundary exceeds u32"))?;
        let (cosine, sine) = text_rope(position);
        let route = program.stage_continuation_draft(
            stream,
            &[slot],
            &[position],
            &[token_ids[processed_prefix]],
            hidden,
            &cosine,
            &sine,
        )?;
        program.replay_staged_continue_draft(stream, route)?;
    }

    let primed = token_ids.len() - 1;
    let mut cursor = processed_prefix;
    let mut native = 0;
    let mut cosine = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
    let mut sine = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
    while let Some(tokens) = next_native_prefill_tile(primed - cursor) {
        let rotary_values = fill_contiguous_rope(cursor, tokens, &mut cosine, &mut sine)?;
        replay_prefill_tile(
            program,
            stream,
            &token_ids[cursor..cursor + tokens],
            slot,
            cursor,
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        let route = program.stage_prompt(
            stream,
            tokens,
            slot,
            cursor,
            &token_ids[cursor + 1..cursor + tokens + 1],
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        program.replay_prompt(stream, route)?;
        cursor += tokens;
        native += tokens;
        progress(cursor)?;
    }
    while cursor < primed {
        replay_target_token(program, stream, token_ids[cursor], cursor)?;
        let position = u32::try_from(cursor)
            .map_err(|_| EngineError::generation("resident MTP prompt position exceeds u32"))?;
        let (cosine, sine) = text_rope(position);
        let route = program.stage_prompt(
            stream,
            1,
            slot,
            cursor,
            &token_ids[cursor + 1..cursor + 2],
            &cosine,
            &sine,
        )?;
        program.replay_prompt(stream, route)?;
        cursor += 1;
        progress(cursor)?;
    }
    replay_target_token(program, stream, token_ids[primed], primed)?;
    Ok(native)
}

fn replay_target_token(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    token: u32,
    position: usize,
) -> EngineResult<()> {
    let position = u32::try_from(position)
        .map_err(|_| EngineError::generation("resident MTP prompt position exceeds u32"))?;
    let (cosine, sine) = text_rope(position);
    program.target_mut().stage_embeddings(stream, &[token])?;
    let route = program
        .target()
        .load_decode_state(stream, 1, &[position], &cosine, &sine)?;
    program.target().replay(stream, route)
}

fn replay_prefill_tile(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    first_position: usize,
    cosine: &[f32],
    sine: &[f32],
) -> EngineResult<()> {
    program.target_mut().stage_embeddings(stream, token_ids)?;
    let route = program.target().load_prefill_tile_state(
        stream,
        token_ids.len(),
        slot,
        first_position,
        cosine,
        sine,
    )?;
    program.target().replay_prefill(stream, route)
}

fn target_logits(logits: &[u16], row: usize) -> &[u16] {
    let begin = row * Qwen38_27B::VOCAB;
    &logits[begin..begin + Qwen38_27B::VOCAB]
}

fn target_logits_mut(logits: &mut [u16], rows: usize) -> &mut [u16] {
    &mut logits[..rows * Qwen38_27B::VOCAB]
}

fn draft_logits(logits: &[u16]) -> &[u16] {
    let begin = VERIFY_ROWS * Qwen38_27B::VOCAB;
    &logits[begin..begin + Qwen38_27B::VOCAB]
}

fn draft_logits_mut(logits: &mut [u16]) -> &mut [u16] {
    let begin = VERIFY_ROWS * Qwen38_27B::VOCAB;
    &mut logits[begin..begin + Qwen38_27B::VOCAB]
}
