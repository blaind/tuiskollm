//! Compact Qwen3.8 Flash-Next generation over eight physical slots.
//!
//! Requests retain their slots while pending rows pack densely. Prime and decode reuse the
//! qualified model entries. Separate compact and per-slot logit banks prevent row/slot aliasing.

use crate::common::banks::{compact, row};
use crate::common::progress::ResidentLoadProgress;
use crate::common::slots::{device_zero_context, require_generation_capacity};
use crate::common::text_generator::ModelProgram;
use crate::qwen38_flash_next::compact_route::{
    Qwen38FlashNextCompactRound, qwen38_flash_next_compact_round,
    qwen38_flash_next_compact_survivors,
};
use crate::qwen38_flash_next::layer_route::QWEN38_FLASH_NEXT_PREFILL_ROWS;
use crate::qwen38_flash_next::resident_model::{
    Qwen38FlashNextDurableSlotSnapshot, Qwen38FlashNextResidentLoadStats,
    Qwen38FlashNextResidentModel, Qwen38FlashNextStepTelemetry,
};
use crate::qwen38_flash_next::slot_lifecycle::QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS;
use crate::qwen38_flash_next::text_generation::{
    Qwen38FlashNextGenerationTelemetry, prime_prompt_tiles_from, prompt_position,
};
use crate::{
    ChatGenerationRequest, EngineError, EngineErrorCode, EngineResult, GenerationSession,
    LayerMemoryLayout, MAX_BATCH, ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents,
    ResidentCancellation, ResidentRequestId,
};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

/// Per-slot rows plus one compact download block.
const LOGIT_BANK_ROWS: usize = 2 * MAX_BATCH;

/// Measured decode cost for one exact batch width.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextBatchWidthTelemetry {
    rounds: usize,
    tokens: usize,
    forward: Duration,
    expert_requests: usize,
    expert_hits: usize,
    expert_misses: usize,
    expert_h2d_bytes: usize,
    embedding_h2d_bytes: usize,
    engram_h2d_bytes: usize,
    engram_rows: usize,
    kv_append_bytes: usize,
}

impl Qwen38FlashNextBatchWidthTelemetry {
    /// Decode rounds run at this width.
    pub const fn rounds(self) -> usize {
        self.rounds
    }

    /// Tokens those rounds committed, which is `rounds * width`.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Wall time those rounds took together.
    pub const fn forward(self) -> Duration {
        self.forward
    }

    /// Mean wall time one round at this width took.
    pub fn round_ms(self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }

        self.forward.as_secs_f64() * 1_000.0 / self.rounds as f64
    }

    /// Tokens per second the rounds at this width sustained.
    pub fn tokens_per_second(self) -> f64 {
        if self.forward.is_zero() {
            return 0.0;
        }

        self.tokens as f64 / self.forward.as_secs_f64()
    }

    /// Expert selections the rounds at this width made.
    pub const fn expert_requests(self) -> usize {
        self.expert_requests
    }

    /// Expert hit rate over distinct per-round items at this width.
    pub fn expert_hit_rate(self) -> f64 {
        let resolved = self.expert_hits + self.expert_misses;
        if resolved == 0 {
            return 0.0;
        }

        self.expert_hits as f64 / resolved as f64
    }

    /// Host-to-device expert bytes one token cost at this width.
    pub fn expert_h2d_bytes_per_token(self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }

        self.expert_h2d_bytes as f64 / self.tokens as f64
    }

    /// Host-to-device bytes one token cost across all four staged families.
    pub fn h2d_bytes_per_token(self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }

        let total = self.expert_h2d_bytes + self.embedding_h2d_bytes + self.engram_h2d_bytes;

        total as f64 / self.tokens as f64
    }

    /// Engram rows the host hash addressed at this width.
    pub const fn engram_rows(self) -> usize {
        self.engram_rows
    }

    /// Engram FP8 bytes uploaded at this width.
    pub const fn engram_h2d_bytes(self) -> usize {
        self.engram_h2d_bytes
    }

    /// Token-embedding bytes uploaded at this width.
    pub const fn embedding_h2d_bytes(self) -> usize {
        self.embedding_h2d_bytes
    }

    /// Bytes appended to the paged K/V planes at this width.
    pub const fn kv_append_bytes(self) -> usize {
        self.kv_append_bytes
    }

    fn observe(&mut self, step: &Qwen38FlashNextStepTelemetry) {
        self.rounds += 1;
        self.tokens += step.rows();
        self.forward += step.forward();
        self.expert_requests += step.expert_requests();
        self.expert_h2d_bytes += step.expert_h2d_bytes();
        self.embedding_h2d_bytes += step.embedding_h2d_bytes();
        self.engram_h2d_bytes += step.engram_h2d_bytes();
        self.engram_rows += step.engram_rows();
        self.kv_append_bytes += step.kv_append_bytes();
        for layer in step.layers() {
            self.expert_hits += layer.hits();
            self.expert_misses += layer.misses();
        }
    }
}

/// Decode evidence split by the width of the round that produced it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextBatchTelemetry {
    widths: [Qwen38FlashNextBatchWidthTelemetry; MAX_BATCH],
    admissions: usize,
    retirements: usize,
    cancellations: usize,
}

impl Qwen38FlashNextBatchTelemetry {
    /// Evidence for the rounds run at exactly `width` rows.
    pub fn at(&self, width: usize) -> EngineResult<Qwen38FlashNextBatchWidthTelemetry> {
        self.widths
            .get(width.wrapping_sub(1))
            .copied()
            .ok_or_else(|| {
                EngineError::route(format!(
                    "Flash-Next batch width {width} is outside 1..={MAX_BATCH}"
                ))
            })
    }

    /// Every width in ascending order, including the ones no round reached.
    pub const fn widths(&self) -> &[Qwen38FlashNextBatchWidthTelemetry; MAX_BATCH] {
        &self.widths
    }

    /// Decode rounds the scheduler ran at every width together.
    pub fn rounds(&self) -> usize {
        self.widths.iter().map(|width| width.rounds).sum()
    }

    /// Tokens every decode round committed.
    pub fn tokens(&self) -> usize {
        self.widths.iter().map(|width| width.tokens).sum()
    }

    /// Mean rows a decode round carried, which is the occupancy the scheduler actually reached.
    pub fn mean_width(&self) -> f64 {
        let rounds = self.rounds();
        if rounds == 0 {
            return 0.0;
        }

        self.tokens() as f64 / rounds as f64
    }

    /// Requests admitted into a physical slot.
    pub const fn admissions(&self) -> usize {
        self.admissions
    }

    /// Requests that reached their own finish reason.
    pub const fn retirements(&self) -> usize {
        self.retirements
    }

    /// Requests whose caller went away before they finished.
    pub const fn cancellations(&self) -> usize {
        self.cancellations
    }

    fn observe(&mut self, step: &Qwen38FlashNextStepTelemetry) -> EngineResult<()> {
        let width = self
            .widths
            .get_mut(step.rows().wrapping_sub(1))
            .ok_or_else(|| {
                EngineError::route(format!(
                    "a Flash-Next decode round reported {} rows, outside 1..={MAX_BATCH}",
                    step.rows()
                ))
            })?;
        width.observe(step);

        Ok(())
    }
}

/// Frontend, resident program, stream, and logit banks behind the served Flash-Next model.
pub struct Qwen38FlashNextResidentBatchGenerator {
    frontend: TextFrontend,
    program: Qwen38FlashNextResidentModel,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    boundary: Qwen38FlashNextBoundaryBank,
    sessions: [Option<Qwen38FlashNextBatchSession>; MAX_BATCH],
    retained: [Option<Qwen38FlashNextRetainedSlot>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
    retention_clock: u64,
    batch: Qwen38FlashNextBatchTelemetry,
}

struct Qwen38FlashNextBatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    pending_token: Option<u32>,
    next_position: u32,
}

struct Qwen38FlashNextBoundaryBank {
    history: PinnedHostBuffer<u16>,
    state: PinnedHostBuffer<f32>,
    ple: PinnedHostBuffer<u16>,
    logits: PinnedHostBuffer<u16>,
    snapshots: [Option<Qwen38FlashNextDurableSlotSnapshot>; MAX_BATCH],
    history_width: usize,
    state_width: usize,
    ple_width: usize,
}

struct Qwen38FlashNextRetainedSlot {
    tokens: Vec<u32>,
    last_used: u64,
}

struct Qwen38FlashNextAdmissionPlan {
    slot: usize,
    reused: usize,
    reset: bool,
    victims: [usize; MAX_BATCH],
    victims_len: usize,
}

struct Qwen38FlashNextPrimingAdmission {
    request_id: ResidentRequestId,
    control: GenerationSession,
    slot: usize,
    device_reused_tokens: usize,
    retained_prefix_tokens: usize,
    native_prefill_tokens: usize,
    primed: usize,
}

#[derive(Clone, Copy)]
enum Qwen38FlashNextPrimePhase {
    StablePrefix,
    FullPrompt,
}

impl Qwen38FlashNextPrimingAdmission {
    fn phase_end(&self, phase: Qwen38FlashNextPrimePhase) -> usize {
        match phase {
            Qwen38FlashNextPrimePhase::StablePrefix => self.retained_prefix_tokens,
            Qwen38FlashNextPrimePhase::FullPrompt => self.control.prompt_token_ids().len(),
        }
    }
}

enum Qwen38FlashNextAdmissionOutcome {
    Settled(EngineResult<ResidentBatchAdmission>),
    Priming(Qwen38FlashNextPrimingAdmission),
}

fn terminal_admission(
    request_id: ResidentRequestId,
    control: GenerationSession,
) -> EngineResult<ResidentBatchAdmission> {
    let prompt_tokens = control.prompt_token_ids().len();
    let prompt_metrics = control.prompt_metrics().clone();

    Ok(ResidentBatchAdmission {
        request_id,
        prompt_tokens,
        device_reused_tokens: 0,
        native_prefill_tokens: 0,
        prompt_metrics,
        completed: Some(control.into_output()?),
    })
}

fn release_slot(
    program: &mut Qwen38FlashNextResidentModel,
    stream: &CudaStream,
    slot: usize,
    error: EngineError,
) -> Qwen38FlashNextAdmissionOutcome {
    Qwen38FlashNextAdmissionOutcome::Settled(match program.recycle_slot(stream, slot) {
        Ok(_) => Err(error),
        Err(release) => Err(release),
    })
}

fn shared_round_failure(error: &EngineError) -> EngineError {
    match error.code() {
        Some(EngineErrorCode::Route) => EngineError::route(error.to_string()),
        Some(EngineErrorCode::Layout) => EngineError::layout(error.to_string()),
        Some(EngineErrorCode::Sampling) => EngineError::sampling(error.to_string()),
        Some(EngineErrorCode::Generation) => EngineError::generation(error.to_string()),
        Some(EngineErrorCode::Capacity) => EngineError::capacity(error.to_string()),
        None => EngineError::generation(error.to_string()),
    }
}

impl Qwen38FlashNextResidentBatchGenerator {
    /// Opens the served Flash-Next program on device zero, reporting load progress.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, Some(progress))
    }

    /// Opens the served Flash-Next program on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, None)
    }

    /// Loads one resident program shared by eight physical request slots.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let program = Qwen38FlashNextResidentModel::from_snapshot_with_progress(
            context,
            Arc::clone(&snapshot),
            progress,
        )?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen38FlashNext::VOCAB
            .checked_mul(LOGIT_BANK_ROWS)
            .ok_or_else(|| EngineError::layout("Flash-Next compact logit banks overflow"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;
        let (history_width, state_width, ple_width) = program.durable_slot_snapshot_values();
        let boundary = Qwen38FlashNextBoundaryBank {
            history: PinnedHostBuffer::zeroed(
                context,
                checked_rows("Flash-Next boundary history", MAX_BATCH, history_width)?,
            )
            .map_err(GpuError::from)?,
            state: PinnedHostBuffer::zeroed(
                context,
                checked_rows("Flash-Next boundary state", MAX_BATCH, state_width)?,
            )
            .map_err(GpuError::from)?,
            ple: PinnedHostBuffer::zeroed(
                context,
                checked_rows("Flash-Next boundary PLE", MAX_BATCH, ple_width)?,
            )
            .map_err(GpuError::from)?,
            logits: PinnedHostBuffer::zeroed(
                context,
                checked_rows(
                    "Flash-Next boundary logits",
                    MAX_BATCH,
                    Qwen38FlashNext::VOCAB,
                )?,
            )
            .map_err(GpuError::from)?,
            snapshots: std::array::from_fn(|_| None),
            history_width,
            state_width,
            ple_width,
        };

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            boundary,
            sessions: std::array::from_fn(|_| None),
            retained: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
            retention_clock: 0,
            batch: Qwen38FlashNextBatchTelemetry::default(),
        })
    }

    /// Admits one request through the grouped prompt-prime path.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        self.admit_batch(std::slice::from_ref(&request))
            .pop()
            .expect("one request produces one admission")
    }

    /// Admits queued requests in order and composes their scalar prompt tails.
    pub fn admit_batch(
        &mut self,
        requests: &[&ChatGenerationRequest],
    ) -> Vec<EngineResult<ResidentBatchAdmission>> {
        let mut outcomes = Vec::with_capacity(requests.len());
        let mut taken = [false; MAX_BATCH];
        for request in requests {
            outcomes.push(self.reserve_admission(request, &mut taken));
        }
        self.prime_group(&mut outcomes, Qwen38FlashNextPrimePhase::StablePrefix);
        self.capture_stable_prefixes(&mut outcomes);
        self.prime_group(&mut outcomes, Qwen38FlashNextPrimePhase::FullPrompt);

        outcomes
            .into_iter()
            .map(|outcome| match outcome {
                Qwen38FlashNextAdmissionOutcome::Settled(result) => result,
                Qwen38FlashNextAdmissionOutcome::Priming(priming) => self.install_session(priming),
            })
            .collect()
    }

    fn reserve_admission(
        &mut self,
        request: &ChatGenerationRequest,
        taken: &mut [bool; MAX_BATCH],
    ) -> Qwen38FlashNextAdmissionOutcome {
        let control = match GenerationSession::start(&self.frontend, request) {
            Ok(control) => control,
            Err(error) => return Qwen38FlashNextAdmissionOutcome::Settled(Err(error)),
        };
        let required_positions = match require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            ModelProgram::context_capacity(&self.program),
        ) {
            Ok(positions) => positions,
            Err(error) => return Qwen38FlashNextAdmissionOutcome::Settled(Err(error)),
        };
        let request_id = match self.next_identity() {
            Ok(request_id) => request_id,
            Err(error) => return Qwen38FlashNextAdmissionOutcome::Settled(Err(error)),
        };
        if control.finish_reason().is_some() {
            return Qwen38FlashNextAdmissionOutcome::Settled(terminal_admission(
                request_id, control,
            ));
        }
        if let Err(error) = prompt_position(control.prompt_token_ids().len()) {
            return Qwen38FlashNextAdmissionOutcome::Settled(Err(error));
        }
        let retained_prefix_tokens = retained_prefix_tokens(
            control.message_boundary_token_ids().len(),
            control.prompt_token_ids().len(),
        );

        let plan = match self.plan_admission(
            &control.prompt_token_ids()[..retained_prefix_tokens],
            required_positions,
            taken,
        ) {
            Ok(plan) => plan,
            Err(error) => return Qwen38FlashNextAdmissionOutcome::Settled(Err(error)),
        };
        let slot = plan.slot;
        if let Err(error) = self.apply_admission_plan(&plan, required_positions) {
            return self.release_priming_slot(slot, error);
        }
        taken[slot] = true;
        self.retained[slot] = None;

        Qwen38FlashNextAdmissionOutcome::Priming(Qwen38FlashNextPrimingAdmission {
            request_id,
            control,
            slot,
            device_reused_tokens: plan.reused,
            retained_prefix_tokens,
            native_prefill_tokens: 0,
            primed: plan.reused,
        })
    }

    fn plan_admission(
        &self,
        prompt: &[u32],
        required_positions: usize,
        taken: &[bool; MAX_BATCH],
    ) -> EngineResult<Qwen38FlashNextAdmissionPlan> {
        let prefix = self
            .retained
            .iter()
            .enumerate()
            .filter(|(slot, _)| !taken[*slot])
            .filter_map(|(slot, retained)| {
                retained.as_ref().and_then(|retained| {
                    prompt.starts_with(&retained.tokens).then_some((
                        slot,
                        retained.tokens.len(),
                        retained.last_used,
                    ))
                })
            })
            .max_by_key(|&(slot, tokens, last_used)| (tokens, last_used, usize::MAX - slot));
        let (slot, reused, reset) = if let Some((slot, tokens, _)) = prefix {
            (slot, tokens, false)
        } else if let Some(slot) = (0..MAX_BATCH).find(|&slot| {
            !taken[slot] && self.sessions[slot].is_none() && self.retained[slot].is_none()
        }) {
            (slot, 0, false)
        } else {
            let slot = self
                .retained
                .iter()
                .enumerate()
                .filter(|(slot, retained)| !taken[*slot] && retained.is_some())
                .min_by_key(|(slot, retained)| {
                    (
                        retained
                            .as_ref()
                            .expect("retained admission victim exists")
                            .last_used,
                        *slot,
                    )
                })
                .map(|(slot, _)| slot)
                .ok_or_else(|| {
                    EngineError::capacity(format!(
                        "all {MAX_BATCH} Flash-Next generation slots are active"
                    ))
                })?;
            (slot, 0, true)
        };

        if reused != 0 && self.boundary.snapshots[slot].is_none() {
            return Err(EngineError::generation(
                "retained Flash-Next prefix has no durable state",
            ));
        }
        let required_pages = required_positions.div_ceil(QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS);
        let existing_pages = self.program.slots().pages(slot)?.len();
        let retained_pages = if reset { 0 } else { existing_pages };
        let additional_pages = required_pages.checked_sub(retained_pages).ok_or_else(|| {
            EngineError::generation(format!(
                "Flash-Next slot {slot} retains {retained_pages} pages but admission requires only {required_pages}"
            ))
        })?;
        let available = self
            .program
            .slots()
            .free_pages()
            .checked_add(if reset { existing_pages } else { 0 })
            .ok_or_else(|| EngineError::generation("available Flash-Next pages overflow"))?;
        let mut candidates = Vec::new();
        for (victim, retained) in self.retained.iter().enumerate() {
            if victim == slot || taken[victim] || retained.is_none() {
                continue;
            }
            candidates.push((
                victim,
                retained
                    .as_ref()
                    .expect("retained admission victim exists")
                    .last_used,
                self.program.slots().pages(victim)?.len(),
            ));
        }
        let (victims, victims_len) =
            plan_reclaim_victims(available, additional_pages, &mut candidates)?;

        Ok(Qwen38FlashNextAdmissionPlan {
            slot,
            reused,
            reset,
            victims,
            victims_len,
        })
    }

    fn apply_admission_plan(
        &mut self,
        plan: &Qwen38FlashNextAdmissionPlan,
        required_positions: usize,
    ) -> EngineResult<()> {
        if plan.reset {
            self.program.recycle_slot(&self.stream, plan.slot)?;
            self.retained[plan.slot] = None;
            self.boundary.snapshots[plan.slot] = None;
        }
        for &victim in &plan.victims[..plan.victims_len] {
            self.program.recycle_slot(&self.stream, victim)?;
            self.retained[victim] = None;
            self.boundary.snapshots[victim] = None;
        }
        self.program
            .reserve_slot(&self.stream, plan.slot, required_positions)?;

        Ok(())
    }

    fn release_priming_slot(
        &mut self,
        slot: usize,
        error: EngineError,
    ) -> Qwen38FlashNextAdmissionOutcome {
        self.retained[slot] = None;
        self.boundary.snapshots[slot] = None;
        release_slot(&mut self.program, &self.stream, slot, error)
    }

    fn fail_priming_group(
        &mut self,
        outcomes: &mut [Qwen38FlashNextAdmissionOutcome],
        error: &EngineError,
    ) {
        for outcome in outcomes {
            let Qwen38FlashNextAdmissionOutcome::Priming(priming) = outcome else {
                continue;
            };
            *outcome = self.release_priming_slot(priming.slot, shared_round_failure(error));
        }
    }

    fn prime_group(
        &mut self,
        outcomes: &mut [Qwen38FlashNextAdmissionOutcome],
        phase: Qwen38FlashNextPrimePhase,
    ) {
        self.prime_group_scalars(outcomes, phase, true);
        self.prime_group_tiles(outcomes, phase);
        self.prime_group_scalars(outcomes, phase, false);
    }

    fn prime_group_tiles(
        &mut self,
        outcomes: &mut [Qwen38FlashNextAdmissionOutcome],
        phase: Qwen38FlashNextPrimePhase,
    ) {
        for outcome in outcomes {
            let Qwen38FlashNextAdmissionOutcome::Priming(priming) = outcome else {
                continue;
            };
            let slot = priming.slot;
            let end = priming.phase_end(phase);
            let begin = priming.primed;
            let tiled = prime_prompt_tiles_from(
                &mut self.program,
                &self.stream,
                priming.control.prompt_token_ids(),
                slot,
                begin,
                end,
            )
            .and_then(|tiled| {
                if tiled == end && begin != end {
                    self.program.read_logits_into(
                        &self.stream,
                        1,
                        &mut self.logits[slot_logits(slot)],
                    )?;
                }
                Ok(tiled)
            });
            match tiled {
                Ok(tiled) => {
                    priming.native_prefill_tokens += tiled - begin;
                    priming.primed = tiled;
                }
                Err(error) => {
                    *outcome = self.release_priming_slot(slot, error);
                }
            }
        }
    }

    fn prime_group_scalars(
        &mut self,
        outcomes: &mut [Qwen38FlashNextAdmissionOutcome],
        phase: Qwen38FlashNextPrimePhase,
        alignment_only: bool,
    ) {
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        loop {
            let mut group = [0usize; MAX_BATCH];
            let mut pending = [false; MAX_BATCH];
            let mut owners = [usize::MAX; MAX_BATCH];
            let mut entries = 0;
            for (index, outcome) in outcomes.iter().enumerate() {
                let Qwen38FlashNextAdmissionOutcome::Priming(priming) = outcome else {
                    continue;
                };
                group[entries] = priming.slot;
                pending[entries] = priming.primed < priming.phase_end(phase)
                    && (!alignment_only
                        || !priming
                            .primed
                            .is_multiple_of(Qwen38FlashNext::INDEXER_COMPRESS_RATIO));
                entries += 1;
                owners[priming.slot] = index;
            }
            let planned =
                match qwen38_flash_next_compact_round(&group[..entries], &pending[..entries]) {
                    Ok(planned) => planned,
                    Err(error) => {
                        self.fail_priming_group(outcomes, &error);
                        return;
                    }
                };
            if planned.is_empty() {
                return;
            }
            let rows = planned.rows();
            let slots = planned.slots();
            for (row, &slot) in slots.iter().enumerate() {
                let Qwen38FlashNextAdmissionOutcome::Priming(priming) = &outcomes[owners[slot]]
                else {
                    unreachable!("a prime row names a priming slot")
                };
                tokens[row] = priming.control.prompt_token_ids()[priming.primed];
                positions[row] = prompt_position(priming.primed)
                    .expect("admission already checked the prompt position range");
            }

            let step = match self.program.decode_step(
                &self.stream,
                &tokens[..rows],
                &positions[..rows],
                slots,
            ) {
                Ok(step) => step,
                Err(error) => {
                    for &slot in slots {
                        outcomes[owners[slot]] =
                            self.release_priming_slot(slot, shared_round_failure(&error));
                    }
                    continue;
                }
            };
            self.program.observe_prime_round(&step, false);

            let mut published = 0;
            for (row, &slot) in slots.iter().enumerate() {
                let Qwen38FlashNextAdmissionOutcome::Priming(priming) = &mut outcomes[owners[slot]]
                else {
                    continue;
                };
                priming.primed += 1;
                if priming.primed == priming.phase_end(phase) {
                    published = row + 1;
                }
            }
            if published == 0 {
                continue;
            }
            let readback = self.program.read_logits_into(
                &self.stream,
                published,
                &mut self.logits[compact_logits(published)],
            );
            for (row, &slot) in slots[..published].iter().enumerate() {
                let Qwen38FlashNextAdmissionOutcome::Priming(priming) = &outcomes[owners[slot]]
                else {
                    continue;
                };
                if priming.primed != priming.phase_end(phase) {
                    continue;
                }
                match &readback {
                    Ok(()) => self
                        .logits
                        .copy_within(compact_row(row), slot * Qwen38FlashNext::VOCAB),
                    Err(error) => {
                        outcomes[owners[slot]] =
                            self.release_priming_slot(slot, shared_round_failure(error));
                    }
                }
            }
        }
    }

    fn capture_stable_prefixes(&mut self, outcomes: &mut [Qwen38FlashNextAdmissionOutcome]) {
        for outcome in outcomes {
            let Qwen38FlashNextAdmissionOutcome::Priming(priming) = outcome else {
                continue;
            };
            let slot = priming.slot;
            if priming.primed != priming.retained_prefix_tokens {
                *outcome = self.release_priming_slot(
                    slot,
                    EngineError::generation("Flash-Next stable prefix was not fully primed"),
                );
                continue;
            }
            if priming.device_reused_tokens == priming.retained_prefix_tokens {
                if self.boundary.snapshots[slot].is_none() {
                    *outcome = self.release_priming_slot(
                        slot,
                        EngineError::generation("reused Flash-Next prefix has no durable state"),
                    );
                }
                continue;
            }

            let history = row(slot, self.boundary.history_width);
            let state = row(slot, self.boundary.state_width);
            let ple = row(slot, self.boundary.ple_width);
            let captured = self.program.capture_durable_slot(
                &self.stream,
                slot,
                &mut self.boundary.history[history],
                &mut self.boundary.state[state],
                &mut self.boundary.ple[ple],
            );
            match captured {
                Ok(snapshot) if snapshot.tokens() == priming.retained_prefix_tokens => {
                    self.boundary.snapshots[slot] = Some(snapshot);
                    self.boundary.logits[slot_logits(slot)]
                        .copy_from_slice(&self.logits[slot_logits(slot)]);
                }
                Ok(snapshot) => {
                    *outcome = self.release_priming_slot(
                        slot,
                        EngineError::generation(format!(
                            "Flash-Next boundary captured {} tokens, expected {}",
                            snapshot.tokens(),
                            priming.retained_prefix_tokens
                        )),
                    );
                }
                Err(error) => *outcome = self.release_priming_slot(slot, error),
            }
        }
    }

    fn install_session(
        &mut self,
        priming: Qwen38FlashNextPrimingAdmission,
    ) -> EngineResult<ResidentBatchAdmission> {
        let Qwen38FlashNextPrimingAdmission {
            request_id,
            control,
            slot,
            device_reused_tokens,
            native_prefill_tokens,
            ..
        } = priming;
        let prompt_tokens = control.prompt_token_ids().len();
        let next_position = prompt_position(prompt_tokens)
            .expect("admission already checked the prompt position range");
        let prompt_metrics = control.prompt_metrics().clone();
        self.sessions[slot] = Some(Qwen38FlashNextBatchSession {
            request_id,
            control,
            pending_token: None,
            next_position,
        });
        self.active_slots[self.active] = slot;
        self.active += 1;
        self.batch.admissions += 1;

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    fn next_identity(&mut self) -> EngineResult<ResidentRequestId> {
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Flash-Next request identity overflows"))?;

        Ok(request_id)
    }

    /// Replays every pending token in one compact round, then samples one event per request.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty Flash-Next generation scheduler",
            ));
        }
        self.replay_pending()?;

        let mut events = std::array::from_fn(|_| None);
        let mut retired = [false; MAX_BATCH];
        let active = self.active;
        for (index, event) in events[..active].iter_mut().enumerate() {
            let slot = self.active_slots[index];
            let logits = slot_logits(slot);
            let step = {
                let session = self.sessions[slot].as_mut().ok_or_else(|| {
                    EngineError::generation("active Flash-Next slot has no generation session")
                })?;
                let step = session.control.accept_logits(&self.logits[logits])?;
                if step.finish_reason.is_none() {
                    session.pending_token = Some(step.token_id);
                }
                step
            };
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("active Flash-Next session survived sampling")
                .request_id;
            let completed = if step.finish_reason.is_some() {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal Flash-Next session exists");
                let retained = retained_tokens(&session.control);
                self.store_retained(slot, retained)?;
                retired[index] = true;
                self.batch.retirements += 1;
                Some(session.control.into_output()?)
            } else {
                None
            };
            *event = Some(ResidentBatchEvent {
                request_id,
                step,
                completed,
            });
        }
        let (survivors, surviving) =
            qwen38_flash_next_compact_survivors(&self.active_slots[..active], &retired[..active])?;
        self.active_slots = survivors;
        self.active = surviving;

        Ok(ResidentBatchEvents::from_events(events, active))
    }

    /// Cancels one request at its retained stable message prefix.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| {
                EngineError::generation("Flash-Next cancellation request is not active")
            })?;
        let slot = self.active_slots[index];
        let session = self.sessions[slot]
            .take()
            .expect("cancelled Flash-Next slot owns a session");
        let mut retired = [false; MAX_BATCH];
        retired[index] = true;
        let (survivors, surviving) = qwen38_flash_next_compact_survivors(
            &self.active_slots[..self.active],
            &retired[..self.active],
        )?;
        self.active_slots = survivors;
        self.active = surviving;
        let retained = retained_tokens(&session.control);
        let device_retained_tokens = retained.len();
        self.store_retained(slot, retained)?;
        self.batch.cancellations += 1;

        Ok(ResidentCancellation {
            request_id,
            output: session.control.cancel()?,
            device_retained_tokens,
        })
    }

    /// Requests currently holding a physical slot.
    pub const fn active_requests(&self) -> usize {
        self.active
    }

    /// Active request identities in stable compact-row order.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active_slots[..self.active].iter().map(|&slot| {
            self.sessions[slot]
                .as_ref()
                .expect("active Flash-Next slot owns a session")
                .request_id
        })
    }

    /// Concurrent requests funded by the slot and carry layouts.
    pub const fn slot_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// The proven dense band a served request may reach, not the funded cache depth.
    pub fn context_capacity(&self) -> usize {
        ModelProgram::context_capacity(&self.program)
    }

    /// Device bytes across the resident, paged-cache, expert, and engram arenas.
    pub fn arena_bytes(&self) -> EngineResult<usize> {
        self.program.layout().total_device_bytes()
    }

    /// Source-backed weights this program uploaded to the device.
    pub fn resident_weight_bytes(&self) -> usize {
        self.program.layout().resident_weight_bytes()
    }

    /// Page-locked staging, engram, and logit-bank bytes this owner holds.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes()
            + self.logits.num_bytes()
            + self.boundary.history.num_bytes()
            + self.boundary.state.num_bytes()
            + self.boundary.ple.num_bytes()
            + self.boundary.logits.num_bytes()
    }

    /// Construction evidence: upload, expert staging, and graph capture.
    pub const fn load_stats(&self) -> Qwen38FlashNextResidentLoadStats {
        self.program.load_stats()
    }

    /// Whether the packed primary extent is borrowed from the checkpoint mapping.
    pub fn mapped_primary(&self) -> bool {
        !self
            .program
            .layout()
            .streaming()
            .primary_source()
            .is_pinned()
    }

    /// Streaming and timing evidence folded over every request the scheduler has run.
    pub const fn telemetry(&self) -> Qwen38FlashNextGenerationTelemetry {
        self.program.generation_telemetry()
    }

    /// Decode evidence split by the width of the round that produced it.
    pub const fn batch_telemetry(&self) -> Qwen38FlashNextBatchTelemetry {
        self.batch
    }

    /// Restarts both telemetry accumulators, so a measurement can exclude its own warm-up.
    pub fn reset_telemetry(&mut self) {
        self.program.reset_generation_telemetry();
        self.batch = Qwen38FlashNextBatchTelemetry::default();
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// CUDA context shared by every arena, stream, graph, and pinned buffer.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Physical slot currently owning one active request.
    pub fn qualification_slot(&self, request_id: ResidentRequestId) -> Option<usize> {
        self.sessions.iter().position(|session| {
            session
                .as_ref()
                .is_some_and(|session| session.request_id == request_id)
        })
    }

    #[cfg(feature = "qualification")]
    /// The round the next [`Self::step`] would replay, without running it.
    pub fn qualification_round(&self) -> EngineResult<Qwen38FlashNextCompactRound> {
        self.pending_round()
    }

    #[cfg(feature = "qualification")]
    /// The two token ids one physical slot's engram carry is holding.
    pub fn qualification_engram_carry(
        &self,
        slot: usize,
    ) -> EngineResult<[u32; tuisko_model::QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN]> {
        self.program.qualification_engram_carry(slot)
    }

    #[cfg(feature = "qualification")]
    /// Tokens one physical slot's paged cache currently covers.
    pub fn qualification_slot_tokens(&self, slot: usize) -> EngineResult<usize> {
        self.program.slot_tokens(slot)
    }

    #[cfg(feature = "qualification")]
    /// Pages no slot currently owns, which is what a refused admission would have drawn from.
    pub fn qualification_free_pages(&self) -> usize {
        self.program.slots().free_pages()
    }

    #[cfg(feature = "qualification")]
    /// Stable retained device and pinned-logit addresses.
    pub fn qualification_addresses(&self) -> [usize; 7] {
        [
            self.program.base_address() as usize,
            self.program.kv_base_address() as usize,
            self.logits.as_ptr().addr(),
            self.boundary.history.as_ptr().addr(),
            self.boundary.state.as_ptr().addr(),
            self.boundary.ple.as_ptr().addr(),
            self.boundary.logits.as_ptr().addr(),
        ]
    }

    #[cfg(feature = "qualification")]
    /// Exact tokens retained at one inactive stable prefix.
    pub fn qualification_retained_tokens(&self, slot: usize) -> Option<usize> {
        self.retained
            .get(slot)
            .and_then(Option::as_ref)
            .map(|retained| retained.tokens.len())
    }

    #[cfg(feature = "qualification")]
    /// Whether one retained slot matches every durable prefix plane.
    pub fn qualification_retained_prefix_matches(&mut self, slot: usize) -> EngineResult<bool> {
        let snapshot = self
            .boundary
            .snapshots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| EngineError::generation("Flash-Next slot has no retained prefix"))?;
        let history = row(slot, self.boundary.history_width);
        let state = row(slot, self.boundary.state_width);
        let ple = row(slot, self.boundary.ple_width);
        let recurrent = self.program.qualification_durable_slot_matches(
            &self.stream,
            snapshot,
            &self.boundary.history[history],
            &self.boundary.state[state],
            &self.boundary.ple[ple],
        )?;

        Ok(recurrent && self.logits[slot_logits(slot)] == self.boundary.logits[slot_logits(slot)])
    }

    #[cfg(feature = "qualification")]
    /// Returns every retained prefix to the shared page pool.
    pub fn qualification_clear_retained(&mut self) -> EngineResult<()> {
        for slot in 0..MAX_BATCH {
            if self.retained[slot].is_some() {
                self.program.recycle_slot(&self.stream, slot)?;
                self.retained[slot] = None;
                self.boundary.snapshots[slot] = None;
            }
        }

        Ok(())
    }

    fn store_retained(&mut self, slot: usize, tokens: Vec<u32>) -> EngineResult<()> {
        let next_clock = self
            .retention_clock
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Flash-Next retention clock overflows"))?;
        let snapshot = self.boundary.snapshots[slot].as_ref().ok_or_else(|| {
            EngineError::generation("Flash-Next request has no captured stable prefix")
        })?;
        if snapshot.tokens() != tokens.len() {
            return Err(EngineError::generation(format!(
                "Flash-Next retained {} tokens against a {}-token boundary",
                tokens.len(),
                snapshot.tokens()
            )));
        }
        let history = row(slot, self.boundary.history_width);
        let state = row(slot, self.boundary.state_width);
        let ple = row(slot, self.boundary.ple_width);
        self.program.restore_durable_slot(
            &self.stream,
            snapshot,
            &self.boundary.history[history],
            &self.boundary.state[state],
            &self.boundary.ple[ple],
        )?;
        self.logits[slot_logits(slot)].copy_from_slice(&self.boundary.logits[slot_logits(slot)]);
        self.program.retain_slot(slot)?;
        self.retention_clock = next_clock;
        self.retained[slot] = Some(Qwen38FlashNextRetainedSlot {
            tokens,
            last_used: next_clock,
        });

        Ok(())
    }

    /// The dense round the pending subset of the active order describes.
    fn pending_round(&self) -> EngineResult<Qwen38FlashNextCompactRound> {
        let active = &self.active_slots[..self.active];
        let mut pending = [false; MAX_BATCH];
        for (flag, &slot) in pending[..self.active].iter_mut().zip(active) {
            *flag = self
                .sessions
                .get(slot)
                .and_then(|session| session.as_ref())
                .ok_or_else(|| {
                    EngineError::generation("active Flash-Next slot has no pending session")
                })?
                .pending_token
                .is_some();
        }

        qwen38_flash_next_compact_round(active, &pending[..self.active])
    }

    /// Runs one compact decode round and scatters its logits into the per-slot bank.
    fn replay_pending(&mut self) -> EngineResult<()> {
        let round = self.pending_round()?;
        if round.is_empty() {
            return Ok(());
        }
        let rows = round.rows();
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        for (row, &slot) in round.slots().iter().enumerate() {
            let session = self.sessions[slot]
                .as_ref()
                .expect("a pending row names a slot the round already resolved");
            tokens[row] = session
                .pending_token
                .expect("a pending row names a slot holding a token");
            positions[row] = session.next_position;
        }

        let step = self.program.decode_step(
            &self.stream,
            &tokens[..rows],
            &positions[..rows],
            round.slots(),
        )?;
        self.program.observe_decode_round(&step);
        self.batch.observe(&step)?;
        self.program.read_logits_into(
            &self.stream,
            rows,
            &mut self.logits[compact_logits(rows)],
        )?;
        for (row, &slot) in round.slots().iter().enumerate() {
            self.logits
                .copy_within(compact_row(row), slot * Qwen38FlashNext::VOCAB);
            let session = self.sessions[slot]
                .as_mut()
                .expect("a pending row names a slot holding a session");
            session.pending_token = None;
            session.next_position = session
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        Ok(())
    }
}

fn retained_prefix_tokens(message_boundary: usize, prompt_tokens: usize) -> usize {
    debug_assert!(message_boundary <= prompt_tokens);
    let alignment = QWEN38_FLASH_NEXT_PREFILL_ROWS[0];
    let remainder = message_boundary % alignment;
    if remainder == 0
        || message_boundary < alignment
        || prompt_tokens - message_boundary < alignment - remainder
    {
        message_boundary
    } else {
        message_boundary - remainder
    }
}

fn retained_tokens(control: &GenerationSession) -> Vec<u32> {
    let tokens = retained_prefix_tokens(
        control.message_boundary_token_ids().len(),
        control.prompt_token_ids().len(),
    );
    control.prompt_token_ids()[..tokens].to_vec()
}

fn plan_reclaim_victims(
    mut available: usize,
    required: usize,
    candidates: &mut [(usize, u64, usize)],
) -> EngineResult<([usize; MAX_BATCH], usize)> {
    candidates.sort_unstable_by_key(|&(slot, last_used, _)| (last_used, slot));
    let mut victims = [usize::MAX; MAX_BATCH];
    let mut len = 0usize;
    for &(slot, _, pages) in candidates.iter() {
        if available >= required {
            break;
        }
        available = available
            .checked_add(pages)
            .ok_or_else(|| EngineError::generation("available Flash-Next pages overflow"))?;
        victims[len] = slot;
        len += 1;
    }
    if available < required {
        return Err(EngineError::capacity(format!(
            "Flash-Next KV admission requires {required} additional pages, but free plus reclaimable pages provide {available}"
        )));
    }

    Ok((victims, len))
}

fn checked_rows(label: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| EngineError::layout(format!("{label} overflows")))
}

fn slot_logits(slot: usize) -> Range<usize> {
    row(slot, Qwen38FlashNext::VOCAB)
}

fn compact_logits(rows: usize) -> Range<usize> {
    compact(rows, Qwen38FlashNext::VOCAB)
}

fn compact_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen38FlashNext::VOCAB;
    begin..begin + Qwen38FlashNext::VOCAB
}

#[cfg(test)]
mod tests {
    use super::{
        LOGIT_BANK_ROWS, MAX_BATCH, Qwen38FlashNextBatchTelemetry, compact_logits, compact_row,
        plan_reclaim_victims, retained_prefix_tokens, shared_round_failure, slot_logits,
    };
    use crate::{EngineError, EngineErrorCode};
    use std::mem::size_of;
    use tuisko_model::{Arch, Qwen38FlashNext};

    #[test]
    fn compact_owner_byte_inventory_is_exact() {
        let layout = crate::Qwen38FlashNextResidentLayout::build().unwrap();
        assert_eq!(layout.persistent_state_bytes() % MAX_BATCH, 0);
        let restore_bank = layout.persistent_state_bytes() / MAX_BATCH;
        let stagers = crate::QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS
            * (Qwen38FlashNext::HIDDEN * size_of::<u16>()
                + Qwen38FlashNext::NGRAM_HEADS * Qwen38FlashNext::NGRAM_HEAD_DIM)
            + 2 * LOGIT_BANK_ROWS * Qwen38FlashNext::VOCAB * size_of::<u16>()
            + restore_bank
            + MAX_BATCH * restore_bank
            + MAX_BATCH * Qwen38FlashNext::VOCAB * size_of::<u16>();

        assert_eq!(layout.total_device_bytes().unwrap(), 30_675_307_776);
        assert_eq!(restore_bank, 115_642_368);
        assert_eq!(stagers, 1_068_511_232);
    }

    #[test]
    fn page_reclaim_is_planned_in_lru_then_slot_order_before_mutation() {
        let mut candidates = [(4, 30, 2), (3, 10, 3), (2, 10, 1)];
        let (victims, len) = plan_reclaim_victims(0, 4, &mut candidates).unwrap();

        assert_eq!(&victims[..len], &[2, 3]);
        let error = plan_reclaim_victims(0, 7, &mut candidates).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Capacity));
        let (_, len) = plan_reclaim_victims(4, 4, &mut candidates).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn retained_prefix_alignment_never_adds_a_scalar_prime_round() {
        assert_eq!(retained_prefix_tokens(6, 13), 6);
        assert_eq!(retained_prefix_tokens(48, 53), 48);
        assert_eq!(retained_prefix_tokens(121, 128), 96);
        assert_eq!(retained_prefix_tokens(128, 128), 128);
    }

    #[test]
    fn the_two_logit_banks_never_overlap() {
        // Compact downloads must not overwrite a slot awaiting sampling.
        for slot in 0..MAX_BATCH {
            let per_slot = slot_logits(slot);
            assert!(per_slot.end <= MAX_BATCH * Qwen38FlashNext::VOCAB);
            for row in 0..MAX_BATCH {
                let compact = compact_row(row);
                assert!(compact.start >= per_slot.end || compact.end <= per_slot.start);
            }
        }
        assert_eq!(
            compact_logits(MAX_BATCH).end,
            LOGIT_BANK_ROWS * Qwen38FlashNext::VOCAB
        );
        assert_eq!(compact_logits(1).start, compact_row(0).start);
    }

    #[test]
    fn every_compact_download_row_is_one_whole_vocabulary_row() {
        for row in 0..MAX_BATCH {
            assert_eq!(compact_row(row).len(), Qwen38FlashNext::VOCAB);
        }
        for rows in 1..=MAX_BATCH {
            assert_eq!(compact_logits(rows).len(), rows * Qwen38FlashNext::VOCAB);
        }
    }

    #[test]
    fn batch_telemetry_addresses_only_admitted_widths() {
        let telemetry = Qwen38FlashNextBatchTelemetry::default();

        for width in 1..=MAX_BATCH {
            assert_eq!(telemetry.at(width).unwrap().rounds(), 0);
        }
        assert!(telemetry.at(0).is_err());
        assert!(telemetry.at(MAX_BATCH + 1).is_err());
        assert_eq!(telemetry.rounds(), 0);
        assert_eq!(telemetry.tokens(), 0);
        assert_eq!(telemetry.mean_width(), 0.0);
    }

    #[test]
    fn an_unreached_width_reports_zero_rather_than_a_division() {
        let width = Qwen38FlashNextBatchTelemetry::default().at(4).unwrap();

        assert_eq!(width.round_ms(), 0.0);
        assert_eq!(width.tokens_per_second(), 0.0);
        assert_eq!(width.expert_hit_rate(), 0.0);
        assert_eq!(width.expert_h2d_bytes_per_token(), 0.0);
        assert_eq!(width.h2d_bytes_per_token(), 0.0);
    }

    #[test]
    fn shared_round_failures_preserve_retryable_capacity() {
        let failure = shared_round_failure(&EngineError::capacity("pages remain active"));

        assert_eq!(failure.code(), Some(EngineErrorCode::Capacity));
        assert!(failure.to_string().contains("pages remain active"));
    }
}
