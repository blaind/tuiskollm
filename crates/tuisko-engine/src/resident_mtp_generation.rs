//! Single-slot greedy and unbiased sampled generation over the resident target-plus-MTP owner.

use crate::{
    ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, ResidentMtpProgram, ResidentMtpVerifyRoute,
    SamplingDistribution, speculative_decision,
};
use std::sync::Arc;
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const ROPE_THETA: f64 = 10_000_000.0;
const DRAFT_WINDOW: usize = 3;
const VERIFY_ROWS: usize = DRAFT_WINDOW + 1;
const MAX_NATIVE_PREFILL_TOKENS: usize = 1_024;
const LOGIT_ROWS: usize = VERIFY_ROWS + 1;

/// Exact speculative activity observed by one generation session.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentMtpGenerationStats {
    /// Target verification routes selected for K=1,2,3,4.
    pub verification_routes: [usize; VERIFY_ROWS],
    /// Draft tokens proposed before target verification.
    pub draft_proposals: usize,
    /// Draft tokens licensed by equal target argmax decisions.
    pub accepted_drafts: usize,
    /// Generated tokens committed through target verification routes.
    pub verified_outputs: usize,
}

/// Backward-compatible name for the greedy slice's generation counters.
pub type ResidentMtpGreedyStats = ResidentMtpGenerationStats;

/// Host decision for one exact draft-three sampled MTP transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMtpSampledRound {
    tokens: [u32; VERIFY_ROWS],
    committed: usize,
    accepted: usize,
}

impl ResidentMtpSampledRound {
    /// Target-licensed output prefix, including one correction or bonus when applicable.
    pub fn token_ids(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }

    /// Draft proposals accepted before correction, stop, or full acceptance.
    pub const fn accepted_drafts(&self) -> usize {
        self.accepted
    }
}

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
            native_prefill_tokens = prime_prompt(program, stream, control.prompt_token_ids(), 0)?;
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
        let mut committed = 0;
        let mut accepted = 0;
        for (draft, &draft_token) in drafts.iter().enumerate() {
            let step = self
                .control
                .accept_logits(target_logits(self.logits, draft))?;
            let matches = step.token_id == draft_token;
            self.queued[committed] = Some(step);
            committed += 1;
            if matches {
                accepted += 1;
            }
            let terminal = self.queued[committed - 1]
                .as_ref()
                .expect("committed MTP step exists")
                .finish_reason
                .is_some();
            if terminal || !matches {
                break;
            }
        }
        if accepted == drafts.len() && self.control.finish_reason().is_none() {
            self.queued[committed] = Some(
                self.control
                    .accept_logits(target_logits(self.logits, drafts.len()))?,
            );
            committed += 1;
        }
        Ok((committed, accepted))
    }

    fn decide_sampled_round(
        &mut self,
        drafts: &[u32],
        draft_laws: &[Option<SamplingDistribution>],
    ) -> EngineResult<(usize, usize)> {
        if draft_laws.len() != drafts.len() {
            return Err(EngineError::generation(
                "every sampled MTP proposal requires its draft distribution",
            ));
        }
        let mut target_laws: [Option<SamplingDistribution>; VERIFY_ROWS] =
            std::array::from_fn(|_| None);
        for row in 0..=drafts.len() {
            target_laws[row] = Some(self.control.sampling_distribution(
                target_logits(self.logits, row),
                &drafts[..row.min(drafts.len())],
            )?);
        }
        let mut acceptance_units = [0.0f64; DRAFT_WINDOW];
        let mut residual_units = [0.0f64; DRAFT_WINDOW];
        for row in 0..drafts.len() {
            acceptance_units[row] = self.control.random_unit();
            residual_units[row] = self.control.random_unit();
        }
        let bonus_unit = self.control.random_unit();
        let target_laws = target_laws[..drafts.len() + 1]
            .iter()
            .enumerate()
            .map(|(row, law)| {
                law.as_ref().ok_or_else(|| {
                    EngineError::generation(format!(
                        "sampled MTP target row {row} has no distribution"
                    ))
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let draft_laws = draft_laws
            .iter()
            .enumerate()
            .map(|(row, law)| {
                law.as_ref().ok_or_else(|| {
                    EngineError::generation(format!(
                        "sampled MTP proposal {row} has no draft distribution"
                    ))
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let round = decide_sampled_tokens(
            drafts,
            &target_laws,
            &draft_laws,
            self.stop_ids,
            &acceptance_units[..drafts.len()],
            &residual_units[..drafts.len()],
            bonus_unit,
        )?;
        for (index, &token) in round.token_ids().iter().enumerate() {
            self.queued[index] = Some(self.control.accept_token(token)?);
        }
        Ok((round.token_ids().len(), round.accepted_drafts()))
    }

    fn verify_target(&mut self, inputs: &[u32]) -> EngineResult<ResidentMtpVerifyRoute> {
        self.program
            .target_mut()
            .stage_embeddings(self.stream, inputs)?;
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary_values =
            fill_contiguous_rope(self.next_position, inputs.len(), &mut cosine, &mut sine)?;
        let route = self.program.target().load_target_mtp_verify_state(
            self.stream,
            inputs.len(),
            0,
            self.next_position,
            &cosine[..rotary_values],
            &sine[..rotary_values],
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
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary_values =
            fill_contiguous_rope(self.next_position, outputs.len(), &mut cosine, &mut sine)?;
        let route = self.program.stage_realign(
            self.stream,
            outputs.len(),
            0,
            self.next_position,
            outputs,
            &cosine[..rotary_values],
            &sine[..rotary_values],
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

fn decide_sampled_tokens(
    drafts: &[u32],
    target_laws: &[&SamplingDistribution],
    draft_laws: &[&SamplingDistribution],
    stop_ids: [u32; 2],
    acceptance_units: &[f64],
    residual_units: &[f64],
    bonus_unit: f64,
) -> EngineResult<ResidentMtpSampledRound> {
    let extent = drafts.len();
    if !(1..=DRAFT_WINDOW).contains(&extent)
        || target_laws.len() != extent + 1
        || draft_laws.len() != extent
        || acceptance_units.len() != extent
        || residual_units.len() != extent
    {
        return Err(EngineError::generation(format!(
            "sampled MTP round inventory differs: drafts={extent}, target={}, draft={}, acceptance={}, residual={}",
            target_laws.len(),
            draft_laws.len(),
            acceptance_units.len(),
            residual_units.len()
        )));
    }
    let mut round = ResidentMtpSampledRound {
        tokens: [0; VERIFY_ROWS],
        committed: 0,
        accepted: 0,
    };
    for row in 0..extent {
        let decision = speculative_decision(
            drafts[row],
            target_laws[row],
            draft_laws[row],
            acceptance_units[row],
            residual_units[row],
        )?;
        round.tokens[round.committed] = decision.token_id;
        round.committed += 1;
        if !decision.accepted {
            return Ok(round);
        }
        round.accepted += 1;
        if stop_ids.contains(&decision.token_id) {
            return Ok(round);
        }
    }
    round.tokens[round.committed] = target_laws[extent].draw_at(bonus_unit)?;
    round.committed += 1;
    Ok(round)
}

#[cfg(feature = "qualification")]
/// Runs the exact host commit rule for the independent speculative-sampling oracle.
pub fn qualification_decide_sampled_tokens(
    drafts: &[u32],
    target_laws: &[&SamplingDistribution],
    draft_laws: &[&SamplingDistribution],
    stop_ids: [u32; 2],
    acceptance_units: &[f64],
    residual_units: &[f64],
    bonus_unit: f64,
) -> EngineResult<ResidentMtpSampledRound> {
    decide_sampled_tokens(
        drafts,
        target_laws,
        draft_laws,
        stop_ids,
        acceptance_units,
        residual_units,
        bonus_unit,
    )
}

fn prime_prompt(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    if token_ids.is_empty() {
        return Err(EngineError::generation(
            "resident MTP generation requires a nonempty prompt",
        ));
    }
    let primed = token_ids.len() - 1;
    let mut cursor = 0;
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

fn fill_contiguous_rope(
    first_position: usize,
    rows: usize,
    cosine: &mut [f32],
    sine: &mut [f32],
) -> EngineResult<usize> {
    if rows == 0 || rows > MAX_NATIVE_PREFILL_TOKENS {
        return Err(EngineError::route(format!(
            "resident MTP rotary rows {rows} are outside 1..={MAX_NATIVE_PREFILL_TOKENS}"
        )));
    }
    let values = rows
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| EngineError::generation("resident MTP rotary values overflow"))?;
    if cosine.len() < values || sine.len() < values {
        return Err(EngineError::layout(format!(
            "resident MTP rotary destinations have {}/{} values, expected at least {values}",
            cosine.len(),
            sine.len()
        )));
    }
    for row in 0..rows {
        let position = first_position
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| EngineError::generation("resident MTP position exceeds u32"))?;
        let (row_cosine, row_sine) = text_rope(position);
        let begin = row * ROTARY_PAIRS;
        cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
        sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
    }
    Ok(values)
}

fn text_rope(position: u32) -> ([f32; ROTARY_PAIRS], [f32; ROTARY_PAIRS]) {
    let mut cosine = [0.0f32; ROTARY_PAIRS];
    let mut sine = [0.0f32; ROTARY_PAIRS];
    for pair in 0..ROTARY_PAIRS {
        let frequency = ROPE_THETA.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
        let angle = f64::from(position) * frequency;
        cosine[pair] = angle.cos() as f32;
        sine[pair] = angle.sin() as f32;
    }
    (cosine, sine)
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

const fn next_native_prefill_tile(remaining: usize) -> Option<usize> {
    if remaining >= 1_024 {
        Some(1_024)
    } else if remaining >= 128 {
        Some(128)
    } else if remaining >= 64 {
        Some(64)
    } else if remaining >= 32 {
        Some(32)
    } else {
        None
    }
}

fn require_generation_capacity(
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    context_capacity: usize,
) -> EngineResult<usize> {
    if prompt_tokens == 0 {
        return Err(EngineError::generation(
            "resident MTP generation requires a nonempty prompt",
        ));
    }
    let evaluated = prompt_tokens
        .checked_add(maximum_new_tokens.saturating_sub(1))
        .ok_or_else(|| EngineError::generation("resident MTP token budget overflows"))?;
    if evaluated > context_capacity {
        return Err(EngineError::generation(format!(
            "prompt plus processed MTP generation requires {evaluated} positions, current resident capacity is {context_capacity}"
        )));
    }
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use super::{next_native_prefill_tile, require_generation_capacity};

    #[test]
    fn mtp_prompt_plan_excludes_the_final_target_anchor() {
        for (prompt, expected) in [
            (1, vec![]),
            (32, vec![]),
            (33, vec![32]),
            (65, vec![64]),
            (129, vec![128]),
            (1_025, vec![1_024]),
        ] {
            let mut remaining = prompt - 1;
            let mut plan = Vec::new();
            while let Some(tile) = next_native_prefill_tile(remaining) {
                plan.push(tile);
                remaining -= tile;
            }
            assert_eq!(plan, expected);
            assert!(remaining < 32);
        }
    }

    #[test]
    fn mtp_capacity_counts_only_processed_outputs() {
        require_generation_capacity(220_000, 1, 220_000).unwrap();
        require_generation_capacity(1, 220_000, 220_000).unwrap();
        assert!(require_generation_capacity(220_000, 2, 220_000).is_err());
        assert!(require_generation_capacity(0, 1, 220_000).is_err());
    }
}
