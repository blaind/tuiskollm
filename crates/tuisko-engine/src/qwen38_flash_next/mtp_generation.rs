//! Single-slot Qwen3.8 Flash-Next speculative generation.
//!
//! A rejected suffix restores every recurrent family and replays the accepted prefix. Draft
//! input at position `p` uses the target stream at `p - 1`; accepted rows realign that mirror
//! before the next round.

use crate::common::mtp::{
    DRAFT_WINDOW, VERIFY_ROWS, decide_greedy_round, decide_sampled_round, next_native_prefill_tile,
    require_generation_capacity,
};
use crate::common::progress::ResidentLoadProgress;
use crate::common::slots::device_zero_context;
use crate::qwen38_flash_next::mtp_program::{Qwen38FlashNextMtpProgram, Qwen38FlashNextMtpStream};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, Qwen38FlashNextSlotSnapshot, ResidentMtpGenerationStats,
    SamplingDistribution,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

/// Slot every single-slot session owns.
const SINGLE_SLOT: usize = 0;

/// Per-stage speculative-round cost.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextMtpRoundCost {
    rounds: usize,
    verify: Duration,
    restore: Duration,
    restores: usize,
    draft: Duration,
    draft_launches: usize,
    realign: Duration,
    realign_launches: usize,
    acceptance: Duration,
    snapshot: Duration,
    snapshot_bytes: usize,
    committed: usize,
}

impl Qwen38FlashNextMtpRoundCost {
    /// Speculative rounds this request ran.
    pub const fn rounds(self) -> usize {
        self.rounds
    }

    /// Wall time the verification spans took.
    pub const fn verify(self) -> Duration {
        self.verify
    }

    /// Wall time the rejected-suffix restores and their re-run spans took.
    pub const fn restore(self) -> Duration {
        self.restore
    }

    /// Rounds whose suffix was rejected and therefore paid a second span.
    pub const fn restores(self) -> usize {
        self.restores
    }

    /// Wall time the draft's proposal launches took.
    pub const fn draft(self) -> Duration {
        self.draft
    }

    /// Proposal launches the draft ran.
    pub const fn draft_launches(self) -> usize {
        self.draft_launches
    }

    /// Wall time the draft's realignment launches took.
    pub const fn realign(self) -> Duration {
        self.realign
    }

    /// Realignment launches the draft ran.
    pub const fn realign_launches(self) -> usize {
        self.realign_launches
    }

    /// Wall time the host acceptance law took.
    pub const fn acceptance(self) -> Duration {
        self.acceptance
    }

    /// Wall time taking the restore points took, whether or not they were used.
    pub const fn snapshot(self) -> Duration {
        self.snapshot
    }

    /// Bytes one restore point holds.
    pub const fn snapshot_bytes(self) -> usize {
        self.snapshot_bytes
    }

    /// Outputs the speculative rounds committed, the non-speculative tail excluded.
    pub const fn committed(self) -> usize {
        self.committed
    }

    /// Wall time one round took on average, every term included.
    pub fn round_ms(self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        let total = self.verify
            + self.restore
            + self.draft
            + self.realign
            + self.acceptance
            + self.snapshot;

        total.as_secs_f64() * 1_000.0 / self.rounds as f64
    }

    /// Outputs one speculative round committed on average.
    pub fn accept_length(self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }

        self.committed as f64 / self.rounds as f64
    }

    /// Adds another request's evidence to this one.
    pub fn absorb(&mut self, other: Self) {
        self.rounds += other.rounds;
        self.verify += other.verify;
        self.restore += other.restore;
        self.restores += other.restores;
        self.draft += other.draft;
        self.draft_launches += other.draft_launches;
        self.realign += other.realign;
        self.realign_launches += other.realign_launches;
        self.acceptance += other.acceptance;
        self.snapshot += other.snapshot;
        self.snapshot_bytes = self.snapshot_bytes.max(other.snapshot_bytes);
        self.committed += other.committed;
    }
}

/// Request-wide committed-output distribution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextMtpAcceptance {
    rounds: [usize; VERIFY_ROWS],
}

impl Qwen38FlashNextMtpAcceptance {
    /// Rounds that committed exactly `outputs` tokens, for `outputs` in `1..=VERIFY_ROWS`.
    pub const fn rounds_at(&self, outputs: usize) -> usize {
        if outputs == 0 || outputs > VERIFY_ROWS {
            return 0;
        }

        self.rounds[outputs - 1]
    }

    /// The whole distribution, indexed by committed outputs minus one.
    pub const fn distribution(&self) -> [usize; VERIFY_ROWS] {
        self.rounds
    }

    /// Speculative rounds this request ran.
    pub fn rounds(&self) -> usize {
        self.rounds.iter().sum()
    }

    /// Mean outputs one speculative round committed.
    pub fn mean(&self) -> f64 {
        let rounds = self.rounds();
        if rounds == 0 {
            return 0.0;
        }
        let committed: usize = self
            .rounds
            .iter()
            .enumerate()
            .map(|(index, count)| (index + 1) * count)
            .sum();

        committed as f64 / rounds as f64
    }

    fn observe(&mut self, committed: usize) {
        if (1..=VERIFY_ROWS).contains(&committed) {
            self.rounds[committed - 1] += 1;
        }
    }

    /// Adds another request's distribution to this one.
    pub fn absorb(&mut self, other: Self) {
        for (slot, count) in self.rounds.iter_mut().zip(other.rounds) {
            *slot += count;
        }
    }
}

/// Concrete single-slot owner for draft-three generation.
pub struct Qwen38FlashNextMtpTextGenerator {
    frontend: TextFrontend,
    program: Qwen38FlashNextMtpProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
}

/// Request state kept between serving calls.
pub struct Qwen38FlashNextMtpRoundState {
    control: GenerationSession,
    maximum_new_tokens: usize,
    stop_ids: Vec<u32>,
    next_position: usize,
    reserved_positions: usize,
    started: bool,
    queued: [Option<GenerationStep>; VERIFY_ROWS],
    queue_start: usize,
    queue_len: usize,
    native_prefill_tokens: usize,
    greedy: bool,
    stats: ResidentMtpGenerationStats,
    cost: Qwen38FlashNextMtpRoundCost,
    acceptance: Qwen38FlashNextMtpAcceptance,
}

/// One streaming request borrowing the target-plus-draft owner.
pub struct Qwen38FlashNextMtpGenerationSession<'a> {
    state: Qwen38FlashNextMtpRoundState,
    program: &'a mut Qwen38FlashNextMtpProgram,
    stream: &'a CudaStream,
    logits: &'a mut [u16],
}

impl Qwen38FlashNextMtpTextGenerator {
    /// Opens the speculative owner on CUDA device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Loads the target and the draft block against one joint residency solve.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_progress(context, snapshot, None)
    }

    /// The same construction, reporting the pair's upload phase to a serving thread.
    pub fn from_snapshot_with_progress(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let program =
            Qwen38FlashNextMtpProgram::from_snapshot_with_progress(context, snapshot, progress)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logits = PinnedHostBuffer::zeroed(
            context,
            <A as Arch>::VOCAB.checked_mul(VERIFY_ROWS).ok_or_else(|| {
                EngineError::layout("Qwen3.8 Flash-Next MTP logit bank overflows")
            })?,
        )
        .map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
        })
    }

    /// The loaded pair, for accounting and telemetry.
    pub const fn program(&self) -> &Qwen38FlashNextMtpProgram {
        &self.program
    }

    /// Page-locked staging bytes both programs and this owner hold.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    /// Longest sequence a request may reach, which is the target's own admission.
    pub fn context_capacity(&self) -> usize {
        self.program.target().generation_capacity()
    }

    /// Advances one request state by one committed output.
    pub fn step_state(
        &mut self,
        state: Qwen38FlashNextMtpRoundState,
    ) -> EngineResult<(Qwen38FlashNextMtpRoundState, GenerationStep)> {
        let Self {
            program,
            stream,
            logits,
            ..
        } = self;
        let mut session =
            Qwen38FlashNextMtpGenerationSession::resume(state, program, stream, &mut logits[..]);
        let step = session.step()?;

        Ok((session.into_state(), step))
    }

    /// Returns the single funded slot's pages to the shared pool and clears its carries.
    pub fn release_slot(&mut self) -> EngineResult<()> {
        self.program
            .target_mut()
            .recycle_slot(&self.stream, SINGLE_SLOT)?;

        Ok(())
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> tuisko_frontend::GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// CUDA context shared by every arena, stream, graph, and pinned buffer.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.target().context()
    }

    /// Renders one request and primes matching target and draft cache rows.
    pub fn start<'a>(
        &'a mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<Qwen38FlashNextMtpGenerationSession<'a>> {
        let greedy = request.sampling.is_greedy();
        let stop_ids = self.frontend.stop_ids().to_vec();
        let Self {
            frontend,
            program,
            stream,
            logits,
        } = self;
        let control = GenerationSession::start(frontend, request)?;
        let prompt_tokens = control.prompt_token_ids().len();
        // The mapping covers the draft window beyond the target's current position.
        let required_positions = require_generation_capacity(
            prompt_tokens,
            request.max_new_tokens,
            program.target().generation_capacity(),
        )?
        .saturating_add(DRAFT_WINDOW)
        .min(program.target().generation_capacity());
        let mut native_prefill_tokens = 0;

        if control.finish_reason().is_none() {
            program.target_mut().recycle_slot(stream, SINGLE_SLOT)?;
            program.reset_carry(stream)?;
            program.target_mut().reset_generation_telemetry();
            program
                .target_mut()
                .reserve_slot(stream, SINGLE_SLOT, required_positions)?;
            native_prefill_tokens =
                prime_pair(program, stream, control.prompt_token_ids(), SINGLE_SLOT)?;
            program
                .target()
                .read_logits_into(stream, 1, &mut logits[..<A as Arch>::VOCAB])?;
        }
        let next_position = control.prompt_token_ids().len();

        Ok(Qwen38FlashNextMtpGenerationSession {
            program,
            stream,
            logits,
            state: Qwen38FlashNextMtpRoundState {
                control,
                maximum_new_tokens: request.max_new_tokens,
                stop_ids,
                next_position,
                reserved_positions: required_positions,
                started: false,
                queued: std::array::from_fn(|_| None),
                queue_start: 0,
                queue_len: 0,
                native_prefill_tokens,
                greedy,
                stats: ResidentMtpGenerationStats::default(),
                cost: Qwen38FlashNextMtpRoundCost::default(),
                acceptance: Qwen38FlashNextMtpAcceptance::default(),
            },
        })
    }
}

/// Primes target and draft mirrors over one prompt.
fn prime_pair(
    program: &mut Qwen38FlashNextMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
) -> EngineResult<usize> {
    if token_ids.is_empty() {
        return Err(EngineError::generation(
            "Qwen3.8 Flash-Next MTP generation requires a nonempty prompt",
        ));
    }
    let mut cursor = 0usize;

    while let Some(tokens) = next_native_prefill_tile(token_ids.len() - cursor) {
        let first = u32::try_from(cursor).map_err(|_| {
            EngineError::generation(
                "Qwen3.8 Flash-Next MTP prompt position exceeds the route width",
            )
        })?;
        let tile = &token_ids[cursor..cursor + tokens];
        let step = program
            .target_mut()
            .prefill_tile(stream, tile, first, slot)?;
        program.target_mut().observe_prime_round(&step, true);
        program.prime_tile(stream, tile, first, slot)?;
        program.carry_target_row(stream, tokens - 1)?;
        cursor += tokens;
    }

    for (offset, &token) in token_ids[cursor..].iter().enumerate() {
        let position = u32::try_from(cursor + offset).map_err(|_| {
            EngineError::generation(
                "Qwen3.8 Flash-Next MTP prompt position exceeds the route width",
            )
        })?;
        let step = program
            .target_mut()
            .decode_step(stream, &[token], &[position], &[slot])?;
        program.target_mut().observe_prime_round(&step, false);
        program.draft_extend(
            stream,
            token,
            position,
            slot,
            Qwen38FlashNextMtpStream::Carry,
        )?;
        program.carry_target_row(stream, 0)?;
    }

    Ok(cursor)
}

fn target_logits(logits: &[u16], row: usize) -> &[u16] {
    let vocab = <A as Arch>::VOCAB;

    &logits[row * vocab..(row + 1) * vocab]
}

impl Qwen38FlashNextMtpRoundState {
    /// Terminal state, hidden until all committed outputs are drained.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        if self.queue_len != 0 {
            return None;
        }

        self.control.finish_reason()
    }

    /// Speculative activity this request observed.
    pub const fn stats(&self) -> ResidentMtpGenerationStats {
        self.stats
    }

    /// Where this request's rounds spent their wall time.
    pub const fn cost(&self) -> Qwen38FlashNextMtpRoundCost {
        self.cost
    }

    /// Outputs per speculative round, as a distribution.
    pub const fn acceptance(&self) -> Qwen38FlashNextMtpAcceptance {
        self.acceptance
    }

    /// Prompt tokens processed by exact whole-model prefill graphs.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.native_prefill_tokens
    }

    /// Prompt-cache accounting for request instrumentation.
    pub const fn prompt_encoding(&self) -> &tuisko_frontend::PromptEncoding {
        self.control.prompt_encoding()
    }

    /// Observation-only frontend timing and prefix-lookup detail.
    pub const fn prompt_metrics(&self) -> &tuisko_frontend::PromptEncodingMetrics {
        self.control.prompt_metrics()
    }

    /// Converts a terminal, fully drained request into its complete decoded result.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        self.control.into_output()
    }

    /// Converts an unfinished request into what it produced before it was cancelled.
    pub fn cancel(self) -> EngineResult<crate::CancelledText> {
        self.control.cancel()
    }
}

impl<'a> Qwen38FlashNextMtpGenerationSession<'a> {
    /// Rebuilds a borrowed session over retained state.
    pub fn resume(
        state: Qwen38FlashNextMtpRoundState,
        program: &'a mut Qwen38FlashNextMtpProgram,
        stream: &'a CudaStream,
        logits: &'a mut [u16],
    ) -> Self {
        Self {
            state,
            program,
            stream,
            logits,
        }
    }

    /// Gives the state back so the owner can keep it.
    pub fn into_state(self) -> Qwen38FlashNextMtpRoundState {
        self.state
    }

    /// Borrows the state, for an owner that only wants to read it.
    pub const fn state(&self) -> &Qwen38FlashNextMtpRoundState {
        &self.state
    }
}

impl Qwen38FlashNextMtpGenerationSession<'_> {
    /// Exact prompt token IDs selected by the admitted frontend.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.state.control.prompt_token_ids()
    }

    /// Tokens selected so far, including an unprocessed final token.
    pub fn generated_token_ids(&self) -> &[u32] {
        self.state.control.generated_token_ids()
    }

    /// Prompt tokens processed by exact whole-model prefill graphs.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.state.native_prefill_tokens
    }

    /// Terminal state, hidden until every committed output has been handed out.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        if self.state.queue_len != 0 {
            return None;
        }

        self.state.control.finish_reason()
    }

    /// Speculative activity this request observed.
    pub const fn stats(&self) -> ResidentMtpGenerationStats {
        self.state.stats
    }

    /// Where this request's rounds spent their wall time.
    pub const fn cost(&self) -> Qwen38FlashNextMtpRoundCost {
        self.state.cost
    }

    /// Outputs per speculative round, as a distribution.
    pub const fn acceptance(&self) -> Qwen38FlashNextMtpAcceptance {
        self.state.acceptance
    }

    /// Converts a terminal, fully drained session into its complete decoded result.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        self.state.control.into_output()
    }

    /// Samples one token, running a speculative round whenever the queue is empty.
    pub fn step(&mut self) -> EngineResult<GenerationStep> {
        if self.state.queue_len != 0 {
            return self.take_queued();
        }
        if self.state.control.finish_reason().is_some() {
            return Err(EngineError::generation(
                "a Qwen3.8 Flash-Next MTP session cannot step after it finished",
            ));
        }
        if !self.state.started {
            return self.start_anchor();
        }
        let generated = self.state.control.generated_token_ids().len();
        let remaining = self.state.maximum_new_tokens.saturating_sub(generated);
        if remaining <= 1 {
            return self.run_final_target_step();
        }
        self.run_speculative_round(remaining)?;

        self.take_queued()
    }

    /// The first token, sampled from the logits the prompt prime published.
    fn start_anchor(&mut self) -> EngineResult<GenerationStep> {
        let step = self
            .state
            .control
            .accept_logits(target_logits(self.logits, 0))?;
        self.state.started = true;

        Ok(step)
    }

    /// One non-speculative output, which is the tail the budget leaves.
    ///
    /// A `K = 1` verification span is arithmetically one decode step through the causal route,
    /// and running it here rather than routing to the plain decode entry keeps one code path for
    /// the cache length, the carries and the published stream.
    fn run_final_target_step(&mut self) -> EngineResult<GenerationStep> {
        let anchor = self.anchor_token()?;
        let position = u32::try_from(self.state.next_position).map_err(|_| {
            EngineError::generation("Qwen3.8 Flash-Next MTP position exceeds the route width")
        })?;
        let started = Instant::now();
        self.program
            .target_mut()
            .verify_step(self.stream, &[anchor], position, SINGLE_SLOT)?;
        self.program.target().read_logits_into(
            self.stream,
            1,
            &mut self.logits[..<A as Arch>::VOCAB],
        )?;
        self.state.cost.verify += started.elapsed();
        self.state.cost.rounds += 1;

        let step = self
            .state
            .control
            .accept_logits(target_logits(self.logits, 0))?;
        self.state.next_position += 1;
        self.state.stats.verification_routes[0] += 1;
        self.state.stats.verified_outputs += 1;
        self.state.cost.committed += 1;
        self.state.acceptance.observe(1);

        Ok(step)
    }

    /// One whole speculative round: draft three, verify four, decide, roll back, realign.
    fn run_speculative_round(&mut self, remaining: usize) -> EngineResult<()> {
        let anchor = self.anchor_token()?;
        let extent = DRAFT_WINDOW.min(remaining - 1);
        let position = u32::try_from(self.state.next_position).map_err(|_| {
            EngineError::generation("Qwen3.8 Flash-Next MTP position exceeds the route width")
        })?;

        // Capture before any recurrent state moves.
        let started = Instant::now();
        let snapshot = self
            .program
            .target_mut()
            .snapshot_slot(self.stream, SINGLE_SLOT)?;
        self.state.cost.snapshot += started.elapsed();
        self.state.cost.snapshot_bytes = snapshot.byte_len();

        // Draft the window from the anchor, then from chained proposals.
        let started = Instant::now();
        let mut drafts = [0u32; DRAFT_WINDOW];
        let mut draft_laws: [Option<SamplingDistribution>; DRAFT_WINDOW] =
            std::array::from_fn(|_| None);
        let mut token = anchor;
        for draft in 0..extent {
            let source = if draft == 0 {
                Qwen38FlashNextMtpStream::Carry
            } else {
                Qwen38FlashNextMtpStream::Draft
            };
            let row = position + draft as u32;
            self.program
                .draft_step(self.stream, token, row, SINGLE_SLOT, source)?;
            let proposal = self.program.read_proposal(self.stream)?;
            let proposed = if self.state.greedy {
                self.state
                    .control
                    .propose_logits(proposal, &drafts[..draft])?
                    .token_id
            } else {
                let law = self
                    .state
                    .control
                    .sampling_distribution(proposal, &drafts[..draft])?;
                let token = self.state.control.draw_distribution(&law)?;
                draft_laws[draft] = Some(law);
                token
            };
            drafts[draft] = proposed;
            token = proposed;
            self.state.stats.draft_proposals += 1;
            self.state.cost.draft_launches += 1;
        }
        self.state.cost.draft += started.elapsed();

        // Verify the anchor followed by every proposal.
        let mut inputs = [0u32; VERIFY_ROWS];
        inputs[0] = anchor;
        inputs[1..=extent].copy_from_slice(&drafts[..extent]);
        let rows = extent + 1;
        let started = Instant::now();
        self.program.target_mut().verify_step(
            self.stream,
            &inputs[..rows],
            position,
            SINGLE_SLOT,
        )?;
        self.program.target().read_logits_into(
            self.stream,
            rows,
            &mut self.logits[..rows * <A as Arch>::VOCAB],
        )?;
        self.state.cost.verify += started.elapsed();

        // Apply the target-neutral acceptance law.
        let started = Instant::now();
        let (committed, accepted) = if self.state.greedy {
            decide_greedy_round(
                &mut self.state.control,
                &mut self.state.queued,
                self.logits,
                <A as Arch>::VOCAB,
                &drafts[..extent],
            )?
        } else {
            decide_sampled_round(
                &mut self.state.control,
                &mut self.state.queued,
                self.logits,
                <A as Arch>::VOCAB,
                &self.state.stop_ids,
                &drafts[..extent],
                &draft_laws[..extent],
            )?
        };
        self.state.cost.acceptance += started.elapsed();
        if committed == 0 {
            return Err(EngineError::generation(
                "a Qwen3.8 Flash-Next MTP verification committed no output",
            ));
        }

        // Restore and replay when verification crossed the accepted prefix.
        let verified = accepted + 1;
        if verified < rows {
            let started = Instant::now();
            self.program
                .target_mut()
                .restore_slot(self.stream, &snapshot)?;
            // Restore releases suffix pages; the next draft still needs the full mapping.
            self.program.target_mut().reserve_slot(
                self.stream,
                SINGLE_SLOT,
                self.state.reserved_positions,
            )?;
            self.program.target_mut().verify_step(
                self.stream,
                &inputs[..verified],
                position,
                SINGLE_SLOT,
            )?;
            self.state.cost.restore += started.elapsed();
            self.state.cost.restores += 1;
        }

        // Publish the accepted prefix into the draft mirror.
        let started = Instant::now();
        for row in 0..accepted {
            self.program.draft_extend(
                self.stream,
                inputs[row + 1],
                position + row as u32 + 1,
                SINGLE_SLOT,
                Qwen38FlashNextMtpStream::TargetRow(row),
            )?;
            self.state.cost.realign_launches += 1;
        }
        self.program.carry_target_row(self.stream, accepted)?;
        self.state.cost.realign += started.elapsed();

        self.state.next_position += committed;
        self.state.queue_start = 0;
        self.state.queue_len = committed;
        self.state.stats.verification_routes[rows - 1] += 1;
        self.state.stats.accepted_drafts += accepted;
        self.state.stats.verified_outputs += committed;
        self.state.cost.rounds += 1;
        self.state.cost.committed += committed;
        self.state.acceptance.observe(committed);

        Ok(())
    }

    /// The last generated token, which is the one no target row has processed yet.
    fn anchor_token(&self) -> EngineResult<u32> {
        self.state
            .control
            .generated_token_ids()
            .last()
            .copied()
            .ok_or_else(|| {
                EngineError::generation(
                    "a Qwen3.8 Flash-Next MTP round has no anchor token to verify",
                )
            })
    }

    fn take_queued(&mut self) -> EngineResult<GenerationStep> {
        let step = self.state.queued[self.state.queue_start]
            .take()
            .ok_or_else(|| {
                EngineError::generation(
                    "a Qwen3.8 Flash-Next MTP transaction lost a committed output",
                )
            })?;
        self.state.queue_start += 1;
        self.state.queue_len -= 1;
        if self.state.queue_len == 0 {
            self.state.queue_start = 0;
        }

        Ok(step)
    }
}

/// Restore point taken before each speculative span.
pub type Qwen38FlashNextMtpRestorePoint = Qwen38FlashNextSlotSnapshot;

/// One qualification run through the plain or speculative schedule.
#[cfg(feature = "qualification")]
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpQualificationRun {
    /// Prompt tokens consumed by the run.
    pub prompt_tokens: usize,
    /// Selected tokens in order.
    pub token_ids: Vec<u32>,
    /// Prompt tokens handled by native prefill graphs.
    pub native_prefill_tokens: usize,
    /// Speculative counters, zero for the plain schedule.
    pub stats: ResidentMtpGenerationStats,
    /// Speculative stage timings, zero for the plain schedule.
    pub cost: Qwen38FlashNextMtpRoundCost,
    /// Committed-output distribution, empty for the plain schedule.
    pub acceptance: Qwen38FlashNextMtpAcceptance,
    /// Whole-run wall time including prompt prime.
    pub elapsed: Duration,
    /// Generation wall time after prompt prime.
    pub decode: Duration,
}

#[cfg(feature = "qualification")]
impl Qwen38FlashNextMtpTextGenerator {
    /// Runs raw prompt IDs through the target's plain greedy schedule.
    pub fn qualification_plain_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
    ) -> EngineResult<Qwen38FlashNextMtpQualificationRun> {
        let required_positions = require_generation_capacity(
            token_ids.len(),
            max_new_tokens,
            self.program.target().generation_capacity(),
        )?;
        let mut sampler =
            crate::Sampler::new(crate::SamplingOptions::greedy(), self.frontend.stop_ids())?;
        let started = Instant::now();
        self.program
            .target_mut()
            .recycle_slot(&self.stream, SINGLE_SLOT)?;
        self.program.target_mut().reset_generation_telemetry();
        self.program
            .target_mut()
            .reserve_slot(&self.stream, SINGLE_SLOT, required_positions)?;
        let native_prefill_tokens = crate::qwen38_flash_next::text_generation::prime_prompt(
            self.program.target_mut(),
            &self.stream,
            token_ids,
            SINGLE_SLOT,
        )?;
        let vocab = <A as Arch>::VOCAB;
        self.program
            .target()
            .read_logits_into(&self.stream, 1, &mut self.logits[..vocab])?;
        let primed = Instant::now();

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut position = u32::try_from(token_ids.len()).map_err(|_| {
            EngineError::generation("Qwen3.8 Flash-Next prompt length exceeds the position width")
        })?;
        for _ in 0..max_new_tokens {
            let decision = sampler.sample(&self.logits[..vocab])?;
            generated.push(decision.token_id);
            if decision.stopped || generated.len() == max_new_tokens {
                break;
            }
            self.program.target_mut().decode_step(
                &self.stream,
                &[decision.token_id],
                &[position],
                &[SINGLE_SLOT],
            )?;
            self.program
                .target()
                .read_logits_into(&self.stream, 1, &mut self.logits[..vocab])?;
            position = position.checked_add(1).ok_or_else(|| {
                EngineError::generation("Qwen3.8 Flash-Next generation position overflows")
            })?;
        }

        Ok(Qwen38FlashNextMtpQualificationRun {
            prompt_tokens: token_ids.len(),
            token_ids: generated,
            native_prefill_tokens,
            stats: ResidentMtpGenerationStats::default(),
            cost: Qwen38FlashNextMtpRoundCost::default(),
            acceptance: Qwen38FlashNextMtpAcceptance::default(),
            elapsed: started.elapsed(),
            decode: primed.elapsed(),
        })
    }

    /// Runs raw prompt IDs through the speculative greedy schedule.
    pub fn qualification_mtp_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
    ) -> EngineResult<Qwen38FlashNextMtpQualificationRun> {
        let started = Instant::now();
        let mut session = self.qualification_start_from_tokens(token_ids, max_new_tokens)?;
        let primed = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        while session.finish_reason().is_none() {
            generated.push(session.step()?.token_id);
        }

        Ok(Qwen38FlashNextMtpQualificationRun {
            prompt_tokens: token_ids.len(),
            token_ids: generated,
            native_prefill_tokens: session.native_prefill_tokens(),
            stats: session.stats(),
            cost: session.cost(),
            acceptance: session.acceptance(),
            elapsed: started.elapsed(),
            decode: primed.elapsed(),
        })
    }

    /// Runs one rendered chat request through the production speculative owner.
    pub fn qualification_chat_run(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<Qwen38FlashNextMtpQualificationRun> {
        let started = Instant::now();
        let mut session = self.start(request)?;
        let prompt_tokens = session.prompt_token_ids().len();
        let primed = Instant::now();
        let mut generated = Vec::with_capacity(request.max_new_tokens);
        while session.finish_reason().is_none() {
            generated.push(session.step()?.token_id);
        }

        Ok(Qwen38FlashNextMtpQualificationRun {
            prompt_tokens,
            token_ids: generated,
            native_prefill_tokens: session.native_prefill_tokens(),
            stats: session.stats(),
            cost: session.cost(),
            acceptance: session.acceptance(),
            elapsed: started.elapsed(),
            decode: primed.elapsed(),
        })
    }

    /// Opens a speculative greedy session over raw prompt IDs.
    pub fn qualification_start_from_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
    ) -> EngineResult<Qwen38FlashNextMtpGenerationSession<'_>> {
        let stop_ids = self.frontend.stop_ids().to_vec();
        let Self {
            frontend,
            program,
            stream,
            logits,
        } = self;
        let control = GenerationSession::qualification_from_tokens(
            frontend,
            token_ids,
            max_new_tokens,
            crate::SamplingOptions::greedy(),
        )?;
        let required_positions = require_generation_capacity(
            token_ids.len(),
            max_new_tokens,
            program.target().generation_capacity(),
        )?
        .saturating_add(DRAFT_WINDOW)
        .min(program.target().generation_capacity());
        let mut native_prefill_tokens = 0;

        if control.finish_reason().is_none() {
            program.target_mut().recycle_slot(stream, SINGLE_SLOT)?;
            program.reset_carry(stream)?;
            program.target_mut().reset_generation_telemetry();
            program
                .target_mut()
                .reserve_slot(stream, SINGLE_SLOT, required_positions)?;
            native_prefill_tokens = prime_pair(program, stream, token_ids, SINGLE_SLOT)?;
            program
                .target()
                .read_logits_into(stream, 1, &mut logits[..<A as Arch>::VOCAB])?;
        }

        Ok(Qwen38FlashNextMtpGenerationSession {
            program,
            stream,
            logits,
            state: Qwen38FlashNextMtpRoundState {
                control,
                maximum_new_tokens: max_new_tokens,
                stop_ids,
                next_position: token_ids.len(),
                reserved_positions: required_positions,
                started: false,
                queued: std::array::from_fn(|_| None),
                queue_start: 0,
                queue_len: 0,
                native_prefill_tokens,
                greedy: true,
                stats: ResidentMtpGenerationStats::default(),
                cost: Qwen38FlashNextMtpRoundCost::default(),
                acceptance: Qwen38FlashNextMtpAcceptance::default(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Qwen38FlashNextMtpAcceptance, Qwen38FlashNextMtpRoundCost};
    use crate::common::mtp::VERIFY_ROWS;
    use std::time::Duration;

    #[test]
    fn the_acceptance_distribution_is_the_mean_and_its_shape() {
        let mut acceptance = Qwen38FlashNextMtpAcceptance::default();
        for committed in [4, 4, 1, 3] {
            acceptance.observe(committed);
        }

        assert_eq!(acceptance.distribution(), [1, 0, 1, 2]);
        assert_eq!(acceptance.rounds(), 4);
        assert_eq!(acceptance.rounds_at(4), 2);
        assert_eq!(acceptance.rounds_at(2), 0);
        assert!((acceptance.mean() - 3.0).abs() < 1e-12);

        // A window that is all-or-nothing has exactly the same mean as one that reliably lands
        // three, which is why the shape is reported beside it and not instead of it.
        let mut bimodal = Qwen38FlashNextMtpAcceptance::default();
        for committed in [4, 4, 1, 4, 4, 1] {
            bimodal.observe(committed);
        }

        assert!((bimodal.mean() - acceptance.mean()).abs() < 1e-12);
        assert_eq!(bimodal.distribution(), [2, 0, 0, 4]);
        assert_ne!(bimodal.distribution(), acceptance.distribution());
    }

    #[test]
    fn a_round_out_of_range_is_not_counted_rather_than_clamped() {
        let mut acceptance = Qwen38FlashNextMtpAcceptance::default();
        acceptance.observe(0);
        acceptance.observe(VERIFY_ROWS + 1);

        assert_eq!(acceptance.rounds(), 0);
        assert_eq!(acceptance.mean(), 0.0);
        assert_eq!(acceptance.rounds_at(0), 0);
        assert_eq!(acceptance.rounds_at(VERIFY_ROWS + 1), 0);
    }

    #[test]
    fn the_round_cost_sums_every_term_it_separates() {
        let mut cost = Qwen38FlashNextMtpRoundCost {
            rounds: 2,
            verify: Duration::from_millis(36),
            restore: Duration::from_millis(15),
            restores: 1,
            draft: Duration::from_millis(12),
            draft_launches: 6,
            realign: Duration::from_millis(4),
            realign_launches: 2,
            acceptance: Duration::from_millis(2),
            snapshot: Duration::from_millis(1),
            snapshot_bytes: 115_642_368,
            committed: 6,
        };

        assert!((cost.round_ms() - 35.0).abs() < 1e-9);
        assert!((cost.accept_length() - 3.0).abs() < 1e-12);

        cost.absorb(cost);
        assert_eq!(cost.rounds(), 4);
        assert_eq!(cost.committed(), 12);
        assert_eq!(cost.restores(), 2);
        assert_eq!(cost.snapshot_bytes(), 115_642_368);
        assert!((cost.round_ms() - 35.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_request_reports_zero_rather_than_a_division() {
        let cost = Qwen38FlashNextMtpRoundCost::default();

        assert_eq!(cost.round_ms(), 0.0);
        assert_eq!(cost.accept_length(), 0.0);
        assert_eq!(Qwen38FlashNextMtpAcceptance::default().mean(), 0.0);
    }
}
