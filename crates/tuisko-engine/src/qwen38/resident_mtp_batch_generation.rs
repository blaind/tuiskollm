//! Compact greedy and sampled MTP generation over the resident target-plus-draft owner.

use crate::common::banks::{compact, row};
use crate::common::math::{bf16_to_f32, product};
use crate::common::mtp::{
    DRAFT_WINDOW, MAX_NATIVE_PREFILL_TOKENS, MtpEventBuilder, VERIFY_ROWS, decide_greedy_lane,
    decide_sampled_lane, next_native_prefill_tile, require_generation_capacity,
};
use crate::common::rope::{ROTARY_PAIRS, fill_contiguous_rope, text_rope};
use crate::common::slots::device_zero_context;
use crate::qwen38::resident_mtp_generation::{
    prime_prompt_with_progress, replay_prefill_tile, replay_target_token,
};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GeneratedText, GenerationSession,
    GenerationStep, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, PromptLogprobs, PromptTokenLogprob,
    ResidentBatchAdmission, ResidentCancellation, ResidentLoadProgress, ResidentMtpGenerationStats,
    ResidentMtpLoadStats, ResidentMtpProgram, ResidentMtpSegmentedVerifyRoute,
    ResidentMtpVerifyRoute, ResidentRequestId, SamplingDistribution,
};
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const TARGET_DOWNLOAD_ROWS: usize = MAX_BATCH * VERIFY_ROWS;
const TARGET_LOGIT_ROWS: usize = MAX_BATCH + TARGET_DOWNLOAD_ROWS;
const DRAFT_LOGIT_ROWS: usize = 2 * MAX_BATCH;
const DRAFT_HIDDEN_ROWS: usize = 2 * MAX_BATCH;

fn disconnected_prefill() -> EngineError {
    EngineError::generation("client disconnected during prompt prefill")
}

/// Concrete compact MTP scheduler for up to eight resident requests.
pub struct ResidentMtpBatchGenerator {
    frontend: TextFrontend,
    program: ResidentMtpProgram,
    stream: Arc<CudaStream>,
    target_logits: PinnedHostBuffer<u16>,
    draft_logits: PinnedHostBuffer<u16>,
    target_boundary_hidden: PinnedHostBuffer<u16>,
    message_boundary_hidden: PinnedHostBuffer<u16>,
    message_boundary_history: PinnedHostBuffer<u16>,
    message_boundary_state: PinnedHostBuffer<f32>,
    message_boundary_valid: [bool; MAX_BATCH],
    draft_hidden: PinnedHostBuffer<u16>,
    sessions: [Option<ResidentMtpBatchSession>; MAX_BATCH],
    retained: [Option<RetainedMtpSlot>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
    retention_clock: u64,
    stop_ids: [u32; 2],
}

/// Qwen3.8 generator whose target/MTP arena mappings are absent and cannot launch graphs.
pub struct ParkedQwen38Generator {
    generator: ResidentMtpBatchGenerator,
    mirror: Qwen38ParkMirror,
    released_device_bytes: usize,
}

/// Exact ownership transferred from device to pinned host memory by one park.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38ParkStats {
    /// Page-locked mirror bytes plus its fixed typed manifest.
    pub host_bytes: usize,
    /// Physical VMM backing bytes released while retaining virtual addresses.
    pub released_device_bytes: usize,
    /// Retained slots represented by the mirror.
    pub retained_slots: usize,
    /// Shared physical KV pages represented by the mirror.
    pub retained_pages: usize,
}

#[derive(Clone, Copy)]
struct ParkedSlotManifest {
    slot: usize,
    token_count: usize,
    retention_generation: u64,
    first_page: usize,
    page_count: usize,
    checksum: u64,
}

struct Qwen38ParkMirror {
    slots: Box<[ParkedSlotManifest]>,
    physical_pages: Box<[u32]>,
    target_tables: PinnedHostBuffer<u32>,
    target_history: PinnedHostBuffer<u16>,
    target_state: PinnedHostBuffer<f32>,
    target_key: PinnedHostBuffer<u8>,
    target_value: PinnedHostBuffer<u8>,
    mtp_key: PinnedHostBuffer<u16>,
    mtp_value: PinnedHostBuffer<u16>,
    mtp_table_checksum: u64,
}

/// One scheduler event containing every output committed by one MTP transaction.
pub struct ResidentMtpBatchEvent {
    /// Request that produced this event.
    pub request_id: ResidentRequestId,
    steps: [Option<GenerationStep>; VERIFY_ROWS],
    len: usize,
    /// Complete output when the final step terminated the request.
    pub completed: Option<GeneratedText>,
    /// Cumulative exact-route and acceptance counters for this request.
    pub stats: ResidentMtpGenerationStats,
}

/// At most eight request events in the stable active order at round entry.
pub struct ResidentMtpBatchEvents {
    events: [Option<ResidentMtpBatchEvent>; MAX_BATCH],
    len: usize,
}

struct ResidentMtpBatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    message_boundary_tokens: usize,
    next_position: usize,
    maximum_new_tokens: usize,
    started: bool,
    proposal_ready: bool,
    greedy: bool,
    stats: ResidentMtpGenerationStats,
}

struct RetainedMtpSlot {
    tokens: Vec<u32>,
    message_boundary_tokens: usize,
    last_used: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetainedReuse {
    None,
    Complete,
    MessageBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetainedMatch {
    slot: usize,
    tokens: usize,
    last_used: u64,
    reuse: RetainedReuse,
}

struct LaneDrafts {
    tokens: [u32; DRAFT_WINDOW],
    laws: [Option<SamplingDistribution>; DRAFT_WINDOW],
}

#[derive(Clone, Copy)]
enum TargetRoundRoute {
    Single(ResidentMtpVerifyRoute),
    Segmented(ResidentMtpSegmentedVerifyRoute),
}

impl LaneDrafts {
    fn new() -> Self {
        Self {
            tokens: [0; DRAFT_WINDOW],
            laws: std::array::from_fn(|_| None),
        }
    }
}

impl ResidentMtpBatchEvent {
    /// Number of tokens committed for this request in the scheduler transaction.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this request committed no token.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Streaming steps in target-licensed order.
    pub fn steps(&self) -> impl Iterator<Item = &GenerationStep> {
        self.steps[..self.len]
            .iter()
            .map(|step| step.as_ref().expect("MTP event prefix is initialized"))
    }
}

impl ResidentMtpBatchEvents {
    /// Number of requests that produced an event in this scheduler transaction.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the transaction produced no request events.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Request events in the scheduler order that existed at transaction entry.
    pub fn iter(&self) -> impl Iterator<Item = &ResidentMtpBatchEvent> {
        self.events[..self.len].iter().map(|event| {
            event
                .as_ref()
                .expect("active MTP event prefix is initialized")
        })
    }
}

impl ResidentMtpBatchGenerator {
    /// Opens the exact resident MTP scheduler on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Opens device zero while publishing combined target-plus-MTP startup counters.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot_inner(&context, snapshot, Some(progress))
    }

    /// Admits the pinned frontend and complete target-plus-MTP owner.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_inner(context, snapshot, None)
    }

    fn from_snapshot_inner(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let stop_ids = frontend
            .stop_ids()
            .try_into()
            .map_err(|_| EngineError::generation("frontend returned the wrong stop-ID count"))?;
        let program = match progress {
            Some(progress) => {
                ResidentMtpProgram::from_snapshot_with_progress(context, snapshot, progress)?
            }
            None => ResidentMtpProgram::from_snapshot(context, snapshot)?,
        };
        let stream = context.new_stream().map_err(GpuError::from)?;
        let target_logits = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP batch target logits",
                TARGET_LOGIT_ROWS,
                Qwen38_27B::VOCAB,
            )?,
        )
        .map_err(GpuError::from)?;
        let draft_logits = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP batch draft logits",
                DRAFT_LOGIT_ROWS,
                Qwen38_27B::VOCAB,
            )?,
        )
        .map_err(GpuError::from)?;
        let target_boundary_hidden = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP target boundary hidden",
                MAX_BATCH,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let message_boundary_hidden = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP message-boundary hidden",
                MAX_BATCH,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let message_boundary_history = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP message-boundary history",
                MAX_BATCH,
                program.target().gdn_slot_history_values(),
            )?,
        )
        .map_err(GpuError::from)?;
        let message_boundary_state = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP message-boundary state",
                MAX_BATCH,
                program.target().gdn_slot_state_values(),
            )?,
        )
        .map_err(GpuError::from)?;
        let draft_hidden = PinnedHostBuffer::zeroed(
            context,
            checked_rows(
                "resident MTP batch draft hidden",
                DRAFT_HIDDEN_ROWS,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;

        let generator = Self {
            frontend,
            program,
            stream,
            target_logits,
            draft_logits,
            target_boundary_hidden,
            message_boundary_hidden,
            message_boundary_history,
            message_boundary_state,
            message_boundary_valid: [false; MAX_BATCH],
            draft_hidden,
            sessions: std::array::from_fn(|_| None),
            retained: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
            retention_clock: 0,
            stop_ids,
        };
        if let Some(progress) = progress {
            progress.finish();
        }
        Ok(generator)
    }

    /// Admits one request and restores only an exact shared target/MTP prefix.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        self.admit_with_progress(request, |_, _| true)
    }

    /// Scores every causal token in one nonempty token-ID prompt and appends one greedy token.
    ///
    /// This eval-only boundary runs only while the compact scheduler is idle. It deliberately
    /// uses slot zero and applies the normal inactive-prefix eviction policy under page pressure.
    pub fn score_prompt(&mut self, token_ids: &[u32]) -> EngineResult<PromptLogprobs> {
        if self.active != 0 {
            return Err(EngineError::capacity(
                "prompt scoring requires an idle resident scheduler",
            ));
        }
        if token_ids.is_empty() {
            return Err(EngineError::generation(
                "prompt scoring requires at least one token",
            ));
        }
        for (position, &token) in token_ids.iter().enumerate() {
            if usize::try_from(token).map_or(true, |token| token >= Qwen38_27B::VOCAB) {
                return Err(EngineError::generation(format!(
                    "prompt scoring token {token} at position {position} is outside vocabulary 0..{}",
                    Qwen38_27B::VOCAB
                )));
            }
        }
        if token_ids.len() > self.program.target().context_capacity() {
            return Err(EngineError::capacity(format!(
                "prompt scoring requires {} positions, current resident capacity is {}",
                token_ids.len(),
                self.program.target().context_capacity()
            )));
        }

        let slot = 0;
        self.prepare_kv_slot(slot, true, token_ids.len())?;
        self.program.activate_kv_slot(slot)?;
        if let Err(error) = self
            .program
            .reserve_kv_slot_tokens(&self.stream, slot, token_ids.len())
        {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            return Err(error);
        }
        if let Err(error) = self
            .program
            .target()
            .load_slot_routes(&self.stream, &[slot])
        {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            return Err(error);
        }

        let scored = self.score_prompt_in_slot(token_ids, slot);
        let recycled = self.program.recycle_kv_slot(&self.stream, slot);
        self.retained[slot] = None;
        self.message_boundary_valid[slot] = false;
        match (scored, recycled) {
            (Ok(scored), Ok(_)) => Ok(scored),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    fn score_prompt_in_slot(
        &mut self,
        token_ids: &[u32],
        slot: usize,
    ) -> EngineResult<PromptLogprobs> {
        let mut prompt = vec![None; token_ids.len()];
        let primed = token_ids.len() - 1;
        let mut cursor = 0usize;
        let mut cosine = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
        let mut sine = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];

        while let Some(tokens) = next_native_prefill_tile(primed - cursor) {
            let rotary_values = fill_contiguous_rope(cursor, tokens, &mut cosine, &mut sine)?;
            replay_prefill_tile(
                &mut self.program,
                &self.stream,
                &token_ids[cursor..cursor + tokens],
                slot,
                cursor,
                &cosine[..rotary_values],
                &sine[..rotary_values],
            )?;
            for first_row in (0..tokens).step_by(MAX_BATCH) {
                let rows = MAX_BATCH.min(tokens - first_row);
                self.program.target().launch_prefill_lm_head_rows(
                    &self.stream,
                    first_row,
                    rows,
                    tokens,
                )?;
                self.program.target().read_logits_into(
                    &self.stream,
                    rows,
                    &mut self.target_logits[target_download_logits(rows)],
                )?;
                for row_index in 0..rows {
                    let target_position = cursor + first_row + row_index + 1;
                    prompt[target_position] = Some(score_logit_row(
                        &self.target_logits[target_download_row(row_index)],
                        token_ids[target_position],
                    )?);
                }
            }
            cursor += tokens;
        }

        while cursor < primed {
            replay_target_token(&mut self.program, &self.stream, token_ids[cursor], cursor)?;
            self.program.target().read_logits_into(
                &self.stream,
                1,
                &mut self.target_logits[target_download_logits(1)],
            )?;
            prompt[cursor + 1] = Some(score_logit_row(
                &self.target_logits[target_download_row(0)],
                token_ids[cursor + 1],
            )?);
            cursor += 1;
        }

        replay_target_token(&mut self.program, &self.stream, token_ids[primed], primed)?;
        self.program.target().read_logits_into(
            &self.stream,
            1,
            &mut self.target_logits[target_download_logits(1)],
        )?;
        let completion = score_greedy_row(&self.target_logits[target_download_row(0)])?;
        let mut echoed_ids = token_ids.to_vec();
        echoed_ids.push(completion.token_id);
        let echoed_text = self.frontend.decode(&echoed_ids, false)?;
        let token_text = echoed_ids
            .iter()
            .map(|&token| self.frontend.decode(&[token], false))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(PromptLogprobs {
            prompt_token_ids: token_ids.to_vec(),
            prompt,
            completion,
            echoed_text,
            token_text,
        })
    }

    /// Admits one request while reporting processed prompt tokens at existing stream boundaries.
    pub fn admit_with_progress(
        &mut self,
        request: &ChatGenerationRequest,
        mut progress: impl FnMut(usize, usize) -> bool,
    ) -> EngineResult<ResidentBatchAdmission> {
        let control = GenerationSession::start(&self.frontend, request)?;
        let prompt_tokens = control.prompt_token_ids().len();
        let message_boundary_tokens = control.message_boundary_token_ids().len();
        let prompt_metrics = control.prompt_metrics().clone();
        let required_positions = require_generation_capacity(
            prompt_tokens,
            request.max_new_tokens,
            self.program.target().context_capacity(),
        )?;
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident request identity overflows"))?;
        if control.finish_reason().is_some() {
            return Ok(ResidentBatchAdmission {
                request_id,
                prompt_tokens,
                device_reused_tokens: 0,
                native_prefill_tokens: 0,
                prompt_metrics,
                completed: Some(control.into_output()?),
            });
        }

        let (slot, reused, reset, reuse) = self.select_slot(control.prompt_token_ids())?;
        let prefill_tokens = prompt_tokens.saturating_sub(reused);
        if !progress(0, prefill_tokens) {
            return Err(disconnected_prefill());
        }
        if reused > message_boundary_tokens {
            return Err(EngineError::generation(format!(
                "resident MTP reused prefix {reused} exceeds message boundary {message_boundary_tokens}"
            )));
        }
        if reuse == RetainedReuse::MessageBoundary {
            self.restore_retained_message_boundary(slot, reused)?;
        }
        self.prepare_kv_slot(slot, reset, required_positions)?;
        self.program.activate_kv_slot(slot)?;
        if let Err(error) =
            self.program
                .reserve_kv_slot_tokens(&self.stream, slot, required_positions)
        {
            if reset {
                self.program.recycle_kv_slot(&self.stream, slot)?;
            } else {
                self.program
                    .truncate_kv_slot_tokens(&self.stream, slot, reused)?;
                self.program.retain_kv_slot(slot)?;
            }
            return Err(error);
        }
        self.retained[slot] = None;
        self.program
            .target()
            .load_slot_routes(&self.stream, &[slot])?;

        let mut native_prefill_tokens = 0usize;
        if reused < message_boundary_tokens {
            let retained_hidden =
                (reused != 0).then(|| &self.target_boundary_hidden[hidden_slot(slot)]);
            native_prefill_tokens = match prime_prompt_with_progress(
                &mut self.program,
                &self.stream,
                control.message_boundary_token_ids(),
                slot,
                reused,
                retained_hidden,
                &mut |processed| {
                    progress(processed.saturating_sub(reused), prefill_tokens)
                        .then_some(())
                        .ok_or_else(disconnected_prefill)
                },
            ) {
                Ok(tokens) => tokens,
                Err(error) => return self.abort_admission(slot, error),
            };
            self.program.target().read_residual_row_into(
                &self.stream,
                0,
                &mut self.message_boundary_hidden[hidden_slot(slot)],
            )?;
            self.program.target().capture_gdn_slot(
                &self.stream,
                slot,
                &mut self.message_boundary_history,
                &mut self.message_boundary_state,
            )?;
            if !progress(
                message_boundary_tokens.saturating_sub(reused),
                prefill_tokens,
            ) {
                return self.abort_admission(slot, disconnected_prefill());
            }
            self.message_boundary_valid[slot] = true;
        } else if !self.message_boundary_valid[slot] {
            return Err(EngineError::generation(
                "resident MTP reused message boundary has no state snapshot",
            ));
        }
        if message_boundary_tokens < prompt_tokens {
            let suffix_native = match prime_prompt_with_progress(
                &mut self.program,
                &self.stream,
                control.prompt_token_ids(),
                slot,
                message_boundary_tokens,
                Some(&self.message_boundary_hidden[hidden_slot(slot)]),
                &mut |processed| {
                    progress(processed.saturating_sub(reused), prefill_tokens)
                        .then_some(())
                        .ok_or_else(disconnected_prefill)
                },
            ) {
                Ok(tokens) => tokens,
                Err(error) => return self.abort_admission(slot, error),
            };
            native_prefill_tokens = native_prefill_tokens
                .checked_add(suffix_native)
                .ok_or_else(|| {
                    EngineError::generation("resident native prefill count overflows")
                })?;
        }
        if reused < prompt_tokens {
            self.program.target().read_logits_into(
                &self.stream,
                1,
                &mut self.target_logits[target_slot_logits(slot)],
            )?;
            self.program.target().read_residual_row_into(
                &self.stream,
                0,
                &mut self.target_boundary_hidden[hidden_slot(slot)],
            )?;
            if !progress(prefill_tokens, prefill_tokens) {
                return self.abort_admission(slot, disconnected_prefill());
            }
        }
        self.sessions[slot] = Some(ResidentMtpBatchSession {
            request_id,
            control,
            message_boundary_tokens,
            next_position: prompt_tokens,
            maximum_new_tokens: request.max_new_tokens,
            started: false,
            proposal_ready: false,
            greedy: request.sampling.is_greedy(),
            stats: ResidentMtpGenerationStats::default(),
        });
        self.active_slots[self.active] = slot;
        self.active += 1;

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens: reused,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    fn abort_admission(
        &mut self,
        slot: usize,
        error: EngineError,
    ) -> EngineResult<ResidentBatchAdmission> {
        self.retained[slot] = None;
        self.message_boundary_valid[slot] = false;
        self.program.recycle_kv_slot(&self.stream, slot)?;
        Err(error)
    }

    /// Advances every active request by one exact anchor, tail, or speculative transaction.
    pub fn step(&mut self) -> EngineResult<ResidentMtpBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty resident MTP scheduler",
            ));
        }
        let active = self.active;
        let active_slots = self.active_slots;
        let mut builders: [MtpEventBuilder; MAX_BATCH] =
            std::array::from_fn(|_| MtpEventBuilder::new());
        let mut started = [false; MAX_BATCH];
        let mut fresh = [usize::MAX; MAX_BATCH];
        let mut fresh_count = 0;

        for &slot in &active_slots[..active] {
            let session = self.sessions[slot].as_ref().ok_or_else(|| {
                EngineError::generation("active resident MTP slot has no session")
            })?;
            started[slot] = session.started;
            if !session.started {
                fresh[fresh_count] = slot;
                fresh_count += 1;
            }
        }
        if fresh_count != 0 {
            self.start_anchors(&fresh[..fresh_count], &mut builders)?;
        }

        let mut tail = [usize::MAX; MAX_BATCH];
        let mut tail_count = 0;
        let mut speculative = [usize::MAX; MAX_BATCH];
        let mut speculative_count = 0;
        for &slot in &active_slots[..active] {
            if !started[slot] {
                continue;
            }
            let session = self.sessions[slot]
                .as_ref()
                .expect("started MTP session exists");
            if session.control.finish_reason().is_some() {
                return Err(EngineError::generation(
                    "active resident MTP session is already terminal",
                ));
            }
            let remaining = session
                .maximum_new_tokens
                .checked_sub(session.control.generated_token_ids().len())
                .ok_or_else(|| EngineError::generation("resident MTP budget underflows"))?;
            if remaining == 1 {
                tail[tail_count] = slot;
                tail_count += 1;
            } else {
                speculative[speculative_count] = slot;
                speculative_count += 1;
            }
        }
        if tail_count != 0 {
            self.run_tail(&tail[..tail_count], &mut builders)?;
        }
        if speculative_count != 0 {
            self.run_speculative(&speculative[..speculative_count], &mut builders)?;
        }

        self.finish_events(&active_slots[..active], builders)
    }

    /// Cancels one active request at its last complete-message boundary.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| EngineError::generation("resident MTP cancellation is not active"))?;
        let slot = self.active_slots[index];
        if !self.message_boundary_valid[slot] {
            return Err(EngineError::generation(
                "resident MTP cancellation has no message-boundary snapshot",
            ));
        }
        let session = self.sessions[slot]
            .as_ref()
            .expect("cancelled resident MTP session exists");
        if session.control.message_boundary_token_ids().len() != session.message_boundary_tokens {
            return Err(EngineError::generation(
                "resident MTP cancellation message boundary changed after admission",
            ));
        }
        let session = self.sessions[slot]
            .take()
            .expect("validated resident MTP session exists");
        let retained = session.control.message_boundary_token_ids().to_vec();
        self.program.target().restore_gdn_slot(
            &self.stream,
            slot,
            &self.message_boundary_history,
            &self.message_boundary_state,
        )?;
        self.target_boundary_hidden[hidden_slot(slot)]
            .copy_from_slice(&self.message_boundary_hidden[hidden_slot(slot)]);
        let device_retained_tokens = retained.len();
        self.store_retained(slot, retained, device_retained_tokens)?;
        for position in index..self.active - 1 {
            self.active_slots[position] = self.active_slots[position + 1];
        }
        self.active -= 1;
        self.active_slots[self.active] = usize::MAX;
        let output = session.control.cancel()?;
        Ok(ResidentCancellation {
            request_id,
            output,
            device_retained_tokens,
        })
    }

    /// Mirrors every retained durable row/page, then releases its VMM physical backing.
    #[allow(clippy::result_large_err)]
    pub fn park(self) -> Result<(ParkedQwen38Generator, Qwen38ParkStats), (Self, EngineError)> {
        if self.active != 0 {
            return Err((
                self,
                EngineError::generation("cannot park an active resident MTP scheduler"),
            ));
        }
        let mut generator = self;
        let mirror = match generator.capture_park_mirror() {
            Ok(mirror) => mirror,
            Err(error) => return Err((generator, error)),
        };
        let released_device_bytes = match generator.program.park_device_arenas(&generator.stream) {
            Ok(bytes) => bytes,
            Err(error) => return Err((generator, error)),
        };
        let stats = Qwen38ParkStats {
            host_bytes: mirror.host_bytes(),
            released_device_bytes,
            retained_slots: mirror.slots.len(),
            retained_pages: mirror.physical_pages.len(),
        };
        Ok((
            ParkedQwen38Generator {
                generator,
                mirror,
                released_device_bytes,
            },
            stats,
        ))
    }

    fn capture_park_mirror(&self) -> EngineResult<Qwen38ParkMirror> {
        let mtp_table_checksum = self.program.mtp_table_checksum_against_host(&self.stream)?;
        let mut slots = Vec::new();
        let mut physical_pages = Vec::new();
        for (slot, retained) in self.retained.iter().enumerate() {
            let Some(retained) = retained else {
                continue;
            };
            let token_count = self.program.target().mtp_kv_token_count(slot)?;
            if token_count != retained.tokens.len() {
                return Err(EngineError::generation(format!(
                    "retained slot {slot} owns {token_count} device tokens but {} host tokens",
                    retained.tokens.len()
                )));
            }
            let page_count = self.program.kv_slot_pages(slot)?;
            let first_page = physical_pages.len();
            for logical_page in 0..page_count {
                let physical_page = self
                    .program
                    .target()
                    .mtp_kv_physical_page(slot, logical_page)?;
                let physical_page = u32::try_from(physical_page)
                    .map_err(|_| EngineError::layout("resident physical page exceeds u32"))?;
                if physical_pages.contains(&physical_page) {
                    return Err(EngineError::generation(format!(
                        "resident physical page {physical_page} has multiple retained owners"
                    )));
                }
                physical_pages.push(physical_page);
            }
            slots.push(ParkedSlotManifest {
                slot,
                token_count,
                retention_generation: retained.last_used,
                first_page,
                page_count,
                checksum: 0,
            });
        }

        let context = self.context();
        let slot_count = slots.len();
        let page_count = physical_pages.len();
        let table_values = product(
            "resident park target table values",
            slot_count,
            LONG_CONTEXT_PHYSICAL_PAGES,
        )?;
        let history_values = product(
            "resident park target history values",
            slot_count,
            self.program.target().gdn_slot_history_values(),
        )?;
        let state_values = product(
            "resident park target state values",
            slot_count,
            self.program.target().gdn_slot_state_values(),
        )?;
        let target_page_values = product(
            "resident park target cache values",
            page_count,
            self.program.target().park_cache_page_values()?,
        )?;
        let mtp_page_values = product(
            "resident park MTP cache values",
            page_count,
            self.program.park_cache_page_values()?,
        )?;
        let mut mirror = Qwen38ParkMirror {
            slots: slots.into_boxed_slice(),
            physical_pages: physical_pages.into_boxed_slice(),
            target_tables: PinnedHostBuffer::zeroed(context, table_values)
                .map_err(GpuError::from)?,
            target_history: PinnedHostBuffer::zeroed(context, history_values)
                .map_err(GpuError::from)?,
            target_state: PinnedHostBuffer::zeroed(context, state_values)
                .map_err(GpuError::from)?,
            target_key: PinnedHostBuffer::zeroed(context, target_page_values)
                .map_err(GpuError::from)?,
            target_value: PinnedHostBuffer::zeroed(context, target_page_values)
                .map_err(GpuError::from)?,
            mtp_key: PinnedHostBuffer::zeroed(context, mtp_page_values).map_err(GpuError::from)?,
            mtp_value: PinnedHostBuffer::zeroed(context, mtp_page_values)
                .map_err(GpuError::from)?,
            mtp_table_checksum,
        };

        for mirror_row in 0..mirror.slots.len() {
            let manifest = mirror.slots[mirror_row];
            self.program.target().capture_block_table_into(
                &self.stream,
                manifest.slot,
                mirror_row,
                &mut mirror.target_tables,
            )?;
            self.program.target().capture_gdn_slot_into(
                &self.stream,
                manifest.slot,
                mirror_row,
                &mut mirror.target_history,
                &mut mirror.target_state,
            )?;
        }
        for (mirror_page, &physical_page) in mirror.physical_pages.iter().enumerate() {
            let physical_page = physical_page as usize;
            self.program.target().capture_cache_page_into(
                &self.stream,
                physical_page,
                mirror_page,
                &mut mirror.target_key,
                &mut mirror.target_value,
            )?;
            self.program.capture_cache_page_into(
                &self.stream,
                physical_page,
                mirror_page,
                &mut mirror.mtp_key,
                &mut mirror.mtp_value,
            )?;
        }
        for row in 0..mirror.slots.len() {
            mirror.slots[row].checksum = mirror.slot_checksum(row, self)?;
        }
        Ok(mirror)
    }

    /// Current active request count.
    pub const fn active_requests(&self) -> usize {
        self.active
    }

    /// Active request identities in compact scheduler order.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active_slots[..self.active].iter().map(|&slot| {
            self.sessions[slot]
                .as_ref()
                .expect("active resident MTP slot owns a session")
                .request_id
        })
    }

    /// Complete target and incremental MTP device ownership.
    pub fn device_owner_bytes(&self) -> usize {
        self.program.target().mapped_device_bytes() + self.program.mapped_device_bytes()
    }

    /// Complete target-plus-MTP device ownership reported at server startup.
    pub fn arena_bytes(&self) -> usize {
        self.device_owner_bytes()
    }

    /// Page-locked program and scheduler staging ownership.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.target().host_stager_bytes()
            + self.program.host_stager_bytes()
            + self.target_logits.num_bytes()
            + self.draft_logits.num_bytes()
            + self.target_boundary_hidden.num_bytes()
            + self.message_boundary_hidden.num_bytes()
            + self.message_boundary_history.num_bytes()
            + self.message_boundary_state.num_bytes()
            + self.draft_hidden.num_bytes()
    }

    /// Page-locked bytes retaining all eight exact cancellation boundaries.
    pub fn message_boundary_snapshot_bytes(&self) -> usize {
        self.message_boundary_hidden.num_bytes()
            + self.message_boundary_history.num_bytes()
            + self.message_boundary_state.num_bytes()
    }

    /// Shared target/MTP long-context capacity per slot.
    pub const fn context_capacity(&self) -> usize {
        self.program.target().context_capacity()
    }

    /// Sampling defaults admitted from the pinned target snapshot.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// Fixed host bytes owning the shared page routes.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program.target().kv_route_host_bytes()
    }

    /// CUDA context shared by every owner and graph.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    /// Combined target-plus-MTP startup work used by the server report.
    pub const fn load_stats(&self) -> ResidentMtpLoadStats {
        self.program.load_stats()
    }

    #[cfg(feature = "qualification")]
    /// Stable target, MTP, cache, and page-locked scheduler addresses.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = self.program.qualification_addresses()?;
        addresses.extend([
            self.target_logits.as_ptr().addr(),
            self.draft_logits.as_ptr().addr(),
            self.target_boundary_hidden.as_ptr().addr(),
            self.message_boundary_hidden.as_ptr().addr(),
            self.message_boundary_history.as_ptr().addr(),
            self.message_boundary_state.as_ptr().addr(),
            self.draft_hidden.as_ptr().addr(),
        ]);
        Ok(addresses)
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
    /// Exact processed target/MTP prefix retained in an inactive slot.
    pub fn qualification_retained_tokens(&self, slot: usize) -> Option<usize> {
        self.retained
            .get(slot)
            .and_then(Option::as_ref)
            .map(|retained| retained.tokens.len())
    }

    #[cfg(feature = "qualification")]
    /// Complete-message fallback retained beneath one optimistic generated prefix.
    pub fn qualification_retained_message_boundary(&self, slot: usize) -> Option<usize> {
        self.retained
            .get(slot)
            .and_then(Option::as_ref)
            .map(|retained| retained.message_boundary_tokens)
    }

    #[cfg(feature = "qualification")]
    /// Whether one restored cancellation boundary matches every owned host snapshot seam.
    pub fn qualification_message_boundary_matches(&self, slot: usize) -> EngineResult<bool> {
        if !self
            .message_boundary_valid
            .get(slot)
            .copied()
            .unwrap_or(false)
            || self.target_boundary_hidden[hidden_slot(slot)]
                != self.message_boundary_hidden[hidden_slot(slot)]
        {
            return Ok(false);
        }
        self.program
            .target()
            .qualification_gdn_slot_matches_snapshot(
                &self.stream,
                slot,
                &self.message_boundary_history,
                &self.message_boundary_state,
            )
    }

    #[cfg(feature = "qualification")]
    /// Drops all retained slot metadata and returns its pages to the shared pool.
    pub fn qualification_clear_retained(&mut self) -> EngineResult<()> {
        for slot in 0..MAX_BATCH {
            if self.retained[slot].take().is_some() {
                self.program.recycle_kv_slot(&self.stream, slot)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Copies the most recent lane-major target transaction for an independent host oracle.
    pub fn qualification_target_logits(&self, rows: usize) -> EngineResult<Vec<u16>> {
        if rows == 0 || rows > TARGET_DOWNLOAD_ROWS {
            return Err(EngineError::route(format!(
                "resident MTP qualification target rows {rows} are outside 1..={TARGET_DOWNLOAD_ROWS}"
            )));
        }
        Ok(self.target_logits[target_download_logits(rows)].to_vec())
    }

    #[cfg(feature = "qualification")]
    /// Complete owner exposed only to source-backed qualification and direct timing.
    pub const fn qualification_program(&self) -> &ResidentMtpProgram {
        &self.program
    }

    fn start_anchors(
        &mut self,
        slots: &[usize],
        builders: &mut [MtpEventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let mut positions = [0u32; MAX_BATCH];
        let mut anchors = [0u32; MAX_BATCH];
        let mut cosine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut sine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        for (lane, &slot) in slots.iter().enumerate() {
            let session = self.sessions[slot]
                .as_mut()
                .expect("fresh resident MTP session exists");
            let step = session
                .control
                .accept_logits(&self.target_logits[target_slot_logits(slot)])?;
            anchors[lane] = step.token_id;
            positions[lane] = u32::try_from(session.next_position - 1)
                .map_err(|_| EngineError::generation("resident MTP anchor exceeds u32"))?;
            let (row_cosine, row_sine) = text_rope(positions[lane]);
            let begin = lane * ROTARY_PAIRS;
            cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
            sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
            builders[slot].push(step)?;
            session.started = true;
        }
        for (lane, &slot) in slots.iter().enumerate() {
            let source = hidden_slot(slot);
            let destination = compact_hidden_row(lane);
            self.draft_hidden[destination].copy_from_slice(&self.target_boundary_hidden[source]);
        }
        let route = self.program.stage_continuation_draft(
            &self.stream,
            slots,
            &positions[..slots.len()],
            &anchors[..slots.len()],
            &self.draft_hidden[compact_hidden(slots.len())],
            &cosine[..slots.len() * ROTARY_PAIRS],
            &sine[..slots.len() * ROTARY_PAIRS],
        )?;
        self.program
            .replay_staged_continue_draft(&self.stream, route)?;
        self.program.read_logits_into(
            &self.stream,
            slots.len(),
            &mut self.draft_logits[compact_draft_logits(slots.len())],
        )?;
        self.program.read_residuals_into(
            &self.stream,
            slots.len(),
            &mut self.draft_hidden[compact_hidden(slots.len())],
        )?;
        for (lane, &slot) in slots.iter().enumerate() {
            let terminal = self.sessions[slot]
                .as_ref()
                .expect("seeded resident MTP session exists")
                .control
                .finish_reason()
                .is_some();
            if terminal {
                continue;
            }
            self.draft_logits
                .copy_within(compact_draft_row(lane), slot * Qwen38_27B::VOCAB);
            self.draft_hidden
                .copy_within(compact_hidden_row(lane), slot * Qwen38_27B::HIDDEN);
            let session = self.sessions[slot]
                .as_mut()
                .expect("seeded resident MTP session exists");
            session.proposal_ready = true;
        }
        Ok(())
    }

    fn run_tail(
        &mut self,
        slots: &[usize],
        builders: &mut [MtpEventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let mut inputs = Vec::with_capacity(slots.len());
        for &slot in slots {
            inputs.push(self.anchor(slot)?);
        }
        let route = self.verify_target(slots, 1, &inputs)?;
        for (lane, &slot) in slots.iter().enumerate() {
            let row = target_download_row(lane);
            let step = self.sessions[slot]
                .as_mut()
                .expect("tail resident MTP session exists")
                .control
                .accept_logits(&self.target_logits[row])?;
            builders[slot].push(step)?;
        }
        self.commit_and_realign(
            route,
            slots,
            builders,
            &[1; MAX_BATCH][..slots.len()],
            &[0; MAX_BATCH][..slots.len()],
        )
    }

    fn run_speculative(
        &mut self,
        slots: &[usize],
        builders: &mut [MtpEventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let remaining = slots
            .iter()
            .map(|&slot| {
                let session = self.sessions[slot].as_ref().expect("MTP session exists");
                session
                    .maximum_new_tokens
                    .checked_sub(session.control.generated_token_ids().len())
                    .ok_or_else(|| EngineError::generation("resident MTP budget underflows"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let extent = remaining
            .into_iter()
            .map(|remaining| DRAFT_WINDOW.min(remaining - 1))
            .min()
            .ok_or_else(|| EngineError::generation("MTP speculative group is empty"))?;
        let mut drafts = self.propose_drafts(slots, extent)?;
        let tokens = extent + 1;
        let mut inputs = Vec::with_capacity(slots.len() * tokens);
        for (lane, &slot) in slots.iter().enumerate() {
            inputs.push(self.anchor(slot)?);
            inputs.extend_from_slice(&drafts[lane].tokens[..extent]);
        }
        let route = self.verify_target(slots, tokens, &inputs)?;
        let mut committed = [0usize; MAX_BATCH];
        let mut accepted = [0usize; MAX_BATCH];
        for (lane, &slot) in slots.iter().enumerate() {
            let greedy = self.sessions[slot]
                .as_ref()
                .expect("verified resident MTP session exists")
                .greedy;
            if greedy {
                let (lane_committed, lane_accepted) =
                    self.decide_greedy(slot, lane, extent, &drafts[lane], &mut builders[slot])?;
                committed[lane] = lane_committed;
                accepted[lane] = lane_accepted;
            } else {
                let (lane_committed, lane_accepted) = self.decide_sampled(
                    slot,
                    lane,
                    extent,
                    &mut drafts[lane],
                    &mut builders[slot],
                )?;
                committed[lane] = lane_committed;
                accepted[lane] = lane_accepted;
            }
        }
        self.commit_and_realign(
            route,
            slots,
            builders,
            &committed[..slots.len()],
            &accepted[..slots.len()],
        )
    }

    fn propose_drafts(&mut self, slots: &[usize], extent: usize) -> EngineResult<Vec<LaneDrafts>> {
        let mut drafts = (0..slots.len())
            .map(|_| LaneDrafts::new())
            .collect::<Vec<_>>();
        for draft in 0..extent {
            if draft != 0 {
                self.continue_drafts(slots, draft, &drafts)?;
            }
            for (lane, &slot) in slots.iter().enumerate() {
                let logits = if draft == 0 {
                    &self.draft_logits[draft_slot_logits(slot)]
                } else {
                    &self.draft_logits[compact_draft_row(lane)]
                };
                let session = self.sessions[slot]
                    .as_mut()
                    .expect("proposing resident MTP session exists");
                if !session.proposal_ready {
                    return Err(EngineError::generation(
                        "resident MTP speculative lane has no aligned proposal",
                    ));
                }
                let provisional = &drafts[lane].tokens[..draft];
                let token = if session.greedy {
                    session
                        .control
                        .propose_logits(logits, provisional)?
                        .token_id
                } else {
                    let law = session.control.sampling_distribution(logits, provisional)?;
                    let token = session.control.draw_distribution(&law)?;
                    drafts[lane].laws[draft] = Some(law);
                    token
                };
                drafts[lane].tokens[draft] = token;
                session.stats.draft_proposals += 1;
            }
        }
        Ok(drafts)
    }

    fn continue_drafts(
        &mut self,
        slots: &[usize],
        draft: usize,
        drafts: &[LaneDrafts],
    ) -> EngineResult<()> {
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        let mut cosine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut sine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        for (lane, &slot) in slots.iter().enumerate() {
            tokens[lane] = drafts[lane].tokens[draft - 1];
            let position = self.sessions[slot]
                .as_ref()
                .expect("continuing resident MTP session exists")
                .next_position
                .checked_add(draft - 1)
                .and_then(|position| u32::try_from(position).ok())
                .ok_or_else(|| EngineError::generation("resident MTP position exceeds u32"))?;
            positions[lane] = position;
            let (row_cosine, row_sine) = text_rope(position);
            let begin = lane * ROTARY_PAIRS;
            cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
            sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
        }
        let route = if draft == 1 {
            for (lane, &slot) in slots.iter().enumerate() {
                let source = hidden_slot(slot);
                let destination = compact_hidden_row(lane);
                self.draft_hidden.copy_within(source, destination.start);
            }
            self.program.stage_continuation_draft(
                &self.stream,
                slots,
                &positions[..slots.len()],
                &tokens[..slots.len()],
                &self.draft_hidden[compact_hidden(slots.len())],
                &cosine[..slots.len() * ROTARY_PAIRS],
                &sine[..slots.len() * ROTARY_PAIRS],
            )?
        } else {
            self.program.stage_draft(
                &self.stream,
                slots,
                &positions[..slots.len()],
                &tokens[..slots.len()],
                &cosine[..slots.len() * ROTARY_PAIRS],
                &sine[..slots.len() * ROTARY_PAIRS],
            )?
        };
        if draft == 1 {
            self.program
                .replay_staged_continue_draft(&self.stream, route)?;
        } else {
            self.program.replay_continue_draft(&self.stream, route)?;
        }
        self.program.read_logits_into(
            &self.stream,
            slots.len(),
            &mut self.draft_logits[compact_draft_logits(slots.len())],
        )?;
        Ok(())
    }

    fn verify_target(
        &mut self,
        slots: &[usize],
        tokens: usize,
        inputs: &[u32],
    ) -> EngineResult<TargetRoundRoute> {
        let expected = slots
            .len()
            .checked_mul(tokens)
            .ok_or_else(|| EngineError::generation("resident MTP target rows overflow"))?;
        if inputs.len() != expected {
            return Err(EngineError::layout(format!(
                "resident MTP target input has {} rows, expected {expected}",
                inputs.len()
            )));
        }
        let mut first_positions = [0usize; MAX_BATCH];
        let mut cosine = [0.0f32; TARGET_DOWNLOAD_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; TARGET_DOWNLOAD_ROWS * ROTARY_PAIRS];
        for (lane, &slot) in slots.iter().enumerate() {
            first_positions[lane] = self.sessions[slot]
                .as_ref()
                .expect("verified resident MTP session exists")
                .next_position;
            let begin = lane * tokens * ROTARY_PAIRS;
            fill_contiguous_rope(
                first_positions[lane],
                tokens,
                &mut cosine[begin..begin + tokens * ROTARY_PAIRS],
                &mut sine[begin..begin + tokens * ROTARY_PAIRS],
            )?;
        }
        let route = if slots.len() == 1 {
            self.program
                .target_mut()
                .stage_embeddings(&self.stream, inputs)?;
            let route = self.program.target().load_target_mtp_verify_state(
                &self.stream,
                tokens,
                slots[0],
                first_positions[0],
                &cosine[..tokens * ROTARY_PAIRS],
                &sine[..tokens * ROTARY_PAIRS],
            )?;
            self.program
                .target()
                .replay_target_mtp_verify(&self.stream, route)?;
            self.program.target().read_logits_into(
                &self.stream,
                tokens,
                &mut self.target_logits[target_download_logits(tokens)],
            )?;
            TargetRoundRoute::Single(route)
        } else {
            self.program
                .target_mut()
                .stage_target_mtp_segmented_embeddings(&self.stream, inputs)?;
            let route = self
                .program
                .target()
                .load_target_mtp_segmented_verify_state(
                    &self.stream,
                    tokens,
                    slots,
                    &first_positions[..slots.len()],
                    &cosine[..expected * ROTARY_PAIRS],
                    &sine[..expected * ROTARY_PAIRS],
                )?;
            self.program
                .target()
                .replay_target_mtp_segmented_verify(&self.stream, route)?;
            self.program
                .target()
                .read_target_mtp_segmented_logits_into(
                    &self.stream,
                    route,
                    &mut self.target_logits[target_download_logits(expected)],
                )?;
            self.program
                .target()
                .backup_target_mtp_segmented_residuals(&self.stream, route)?;
            TargetRoundRoute::Segmented(route)
        };
        Ok(route)
    }

    fn decide_greedy(
        &mut self,
        slot: usize,
        lane: usize,
        extent: usize,
        drafts: &LaneDrafts,
        builder: &mut MtpEventBuilder,
    ) -> EngineResult<(usize, usize)> {
        let session = self.sessions[slot]
            .as_mut()
            .expect("greedy resident MTP session exists");
        decide_greedy_lane(
            &mut session.control,
            &self.target_logits,
            Qwen38_27B::VOCAB,
            lane,
            &drafts.tokens[..extent],
            builder,
        )
    }

    fn decide_sampled(
        &mut self,
        slot: usize,
        lane: usize,
        extent: usize,
        drafts: &mut LaneDrafts,
        builder: &mut MtpEventBuilder,
    ) -> EngineResult<(usize, usize)> {
        let session = self.sessions[slot]
            .as_mut()
            .expect("sampled resident MTP session exists");
        decide_sampled_lane(
            &mut session.control,
            &self.target_logits,
            Qwen38_27B::VOCAB,
            &self.stop_ids,
            lane,
            &drafts.tokens[..extent],
            &drafts.laws[..extent],
            builder,
        )
    }

    fn commit_and_realign(
        &mut self,
        route: TargetRoundRoute,
        slots: &[usize],
        builders: &[MtpEventBuilder; MAX_BATCH],
        committed: &[usize],
        accepted: &[usize],
    ) -> EngineResult<()> {
        match route {
            TargetRoundRoute::Single(route) => {
                self.program
                    .target()
                    .replay_target_mtp_commit(&self.stream, route, committed[0])?
            }
            TargetRoundRoute::Segmented(route) => self
                .program
                .target()
                .commit_target_mtp_segmented(&self.stream, route, committed)?,
        }
        let tokens = match route {
            TargetRoundRoute::Single(route) => route.tokens(),
            TargetRoundRoute::Segmented(route) => route.tokens(),
        };
        for (lane, &slot) in slots.iter().enumerate() {
            let count = committed[lane];
            if count == 0 || count > tokens {
                return Err(EngineError::generation(format!(
                    "resident MTP lane {lane} commits {count} rows from K={tokens}"
                )));
            }
            if let TargetRoundRoute::Segmented(segmented) = route {
                self.program
                    .target()
                    .select_target_mtp_segmented_residual_lane(
                        &self.stream,
                        segmented,
                        lane,
                        count,
                    )?;
            }
            self.program.target().read_residual_row_into(
                &self.stream,
                count - 1,
                &mut self.target_boundary_hidden[hidden_slot(slot)],
            )?;
            let source = target_download_row(lane * tokens + count - 1);
            self.target_logits
                .copy_within(source, slot * Qwen38_27B::VOCAB);
            let outputs = builders[slot].token_ids().collect::<Vec<_>>();
            if outputs.len() != count {
                return Err(EngineError::generation(format!(
                    "resident MTP lane {lane} has {} outputs for {count} committed rows",
                    outputs.len()
                )));
            }
            let next_position = self.sessions[slot]
                .as_ref()
                .expect("committed resident MTP session exists")
                .next_position;
            let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
            let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
            let rotary = fill_contiguous_rope(next_position, count, &mut cosine, &mut sine)?;
            let realign = self.program.stage_realign(
                &self.stream,
                count,
                slot,
                next_position,
                &outputs,
                &cosine[..rotary],
                &sine[..rotary],
            )?;
            let terminal = self.sessions[slot]
                .as_ref()
                .expect("committed resident MTP session exists")
                .control
                .finish_reason()
                .is_some();
            if terminal {
                self.program.replay_prime(&self.stream, realign)?;
            } else {
                self.program.replay_realign(&self.stream, realign)?;
                self.program.read_logit_row_into(
                    &self.stream,
                    count - 1,
                    &mut self.draft_logits[draft_slot_logits(slot)],
                )?;
                self.program.read_residual_row_into(
                    &self.stream,
                    count - 1,
                    &mut self.draft_hidden[hidden_slot(slot)],
                )?;
            }
            let session = self.sessions[slot]
                .as_mut()
                .expect("committed resident MTP session exists");
            session.next_position = session
                .next_position
                .checked_add(count)
                .ok_or_else(|| EngineError::generation("resident MTP position overflows"))?;
            session.proposal_ready = !terminal;
            session.stats.verification_routes[tokens - 1] += 1;
            session.stats.accepted_drafts += accepted[lane];
            session.stats.verified_outputs += count;
        }
        Ok(())
    }

    fn anchor(&self, slot: usize) -> EngineResult<u32> {
        self.sessions[slot]
            .as_ref()
            .and_then(|session| session.control.generated_token_ids().last().copied())
            .ok_or_else(|| EngineError::generation("resident MTP lane has no anchor"))
    }

    fn finish_events(
        &mut self,
        active_slots: &[usize],
        mut builders: [MtpEventBuilder; MAX_BATCH],
    ) -> EngineResult<ResidentMtpBatchEvents> {
        let mut events = std::array::from_fn(|_| None);
        let mut survivors = [usize::MAX; MAX_BATCH];
        let mut surviving = 0;
        for (index, &slot) in active_slots.iter().enumerate() {
            let mut builder = std::mem::replace(&mut builders[slot], MtpEventBuilder::new());
            if builder.len() == 0 {
                return Err(EngineError::generation(
                    "active resident MTP request produced no event",
                ));
            }
            let terminal = self.sessions[slot]
                .as_ref()
                .expect("active resident MTP session exists")
                .control
                .finish_reason()
                .is_some();
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("active resident MTP session exists")
                .request_id;
            let stats = self.sessions[slot]
                .as_ref()
                .expect("active resident MTP session exists")
                .stats;
            let completed = if terminal {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal resident MTP session exists");
                let retained = processed_tokens(&session)?;
                self.store_retained(slot, retained, session.message_boundary_tokens)?;
                Some(session.control.into_output()?)
            } else {
                survivors[surviving] = slot;
                surviving += 1;
                None
            };
            events[index] = Some(ResidentMtpBatchEvent {
                request_id,
                steps: builder.take_steps(),
                len: builder.len(),
                completed,
                stats,
            });
        }
        self.active_slots = survivors;
        self.active = surviving;
        Ok(ResidentMtpBatchEvents {
            events,
            len: active_slots.len(),
        })
    }

    fn select_slot(&mut self, prompt: &[u32]) -> EngineResult<(usize, usize, bool, RetainedReuse)> {
        if let Some(prefix) = best_retained_prefix(&self.retained, prompt) {
            return Ok((prefix.slot, prefix.tokens, false, prefix.reuse));
        }
        if let Some(slot) = (0..MAX_BATCH)
            .find(|&slot| self.sessions[slot].is_none() && self.retained[slot].is_none())
        {
            return Ok((slot, 0, true, RetainedReuse::None));
        }
        let eviction = self
            .retained
            .iter()
            .enumerate()
            .filter_map(|(slot, retained)| {
                retained.as_ref().map(|retained| (slot, retained.last_used))
            })
            .min_by_key(|&(_, last_used)| last_used)
            .map(|(slot, _)| slot)
            .ok_or_else(|| EngineError::capacity("all eight resident MTP slots are active"))?;
        Ok((eviction, 0, true, RetainedReuse::None))
    }

    fn restore_retained_message_boundary(
        &mut self,
        slot: usize,
        message_boundary_tokens: usize,
    ) -> EngineResult<()> {
        let retained = self.retained[slot].as_ref().ok_or_else(|| {
            EngineError::generation("resident MTP fallback slot has no retained prefix")
        })?;
        if retained.message_boundary_tokens != message_boundary_tokens
            || message_boundary_tokens >= retained.tokens.len()
            || !self.message_boundary_valid[slot]
        {
            return Err(EngineError::generation(format!(
                "resident MTP slot {slot} cannot restore {message_boundary_tokens} fallback tokens from retained {}/{}",
                retained.message_boundary_tokens,
                retained.tokens.len()
            )));
        }

        self.program.target().restore_gdn_slot(
            &self.stream,
            slot,
            &self.message_boundary_history,
            &self.message_boundary_state,
        )?;
        self.target_boundary_hidden[hidden_slot(slot)]
            .copy_from_slice(&self.message_boundary_hidden[hidden_slot(slot)]);
        self.program
            .truncate_kv_slot_tokens(&self.stream, slot, message_boundary_tokens)?;
        self.retained[slot]
            .as_mut()
            .expect("validated retained MTP fallback exists")
            .tokens
            .truncate(message_boundary_tokens);

        Ok(())
    }

    fn prepare_kv_slot(
        &mut self,
        selected: usize,
        reset: bool,
        required_positions: usize,
    ) -> EngineResult<()> {
        let existing_pages = self.program.kv_slot_pages(selected)?;
        let required_pages = required_positions.div_ceil(ATTENTION_PAGE_SIZE);
        let retained_pages = if reset { 0 } else { existing_pages };
        let additional_pages = required_pages.checked_sub(retained_pages).ok_or_else(|| {
            EngineError::generation(format!(
                "resident MTP slot {selected} retains {retained_pages} pages but admission requires only {required_pages}"
            ))
        })?;
        let free_pages = self.program.kv_free_pages();
        let free_after_reset = free_pages
            .checked_add(if reset { existing_pages } else { 0 })
            .ok_or_else(|| EngineError::generation("available KV pages overflow"))?;

        let mut reclaimable_pages = 0usize;
        for (slot, retained) in self.retained.iter().enumerate() {
            if slot != selected && retained.is_some() {
                reclaimable_pages = reclaimable_pages
                    .checked_add(self.program.kv_slot_pages(slot)?)
                    .ok_or_else(|| EngineError::generation("reclaimable KV pages overflow"))?;
            }
        }
        if free_after_reset
            .checked_add(reclaimable_pages)
            .ok_or_else(|| EngineError::generation("available KV pages overflow"))?
            < additional_pages
        {
            return Err(EngineError::capacity(format!(
                "resident MTP KV admission requires {additional_pages} additional pages, {free_after_reset} are immediately available and {reclaimable_pages} belong to other inactive prefixes"
            )));
        }

        if reset {
            self.program.recycle_kv_slot(&self.stream, selected)?;
            self.retained[selected] = None;
            self.program.reset_slot(&self.stream, selected)?;
            self.message_boundary_valid[selected] = false;
        }

        while self.program.kv_free_pages() < additional_pages {
            let victim = self
                .retained
                .iter()
                .enumerate()
                .filter(|(slot, retained)| *slot != selected && retained.is_some())
                .min_by_key(|(_, retained)| {
                    retained
                        .as_ref()
                        .expect("retained page victim exists")
                        .last_used
                })
                .map(|(slot, _)| slot)
                .ok_or_else(|| {
                    EngineError::generation(
                        "resident MTP reclaimable-page accounting has no retained victim",
                    )
                })?;
            self.program.recycle_kv_slot(&self.stream, victim)?;
            self.retained[victim] = None;
            self.message_boundary_valid[victim] = false;
        }
        Ok(())
    }

    fn store_retained(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        message_boundary_tokens: usize,
    ) -> EngineResult<()> {
        if message_boundary_tokens == 0
            || message_boundary_tokens > tokens.len()
            || !self.message_boundary_valid[slot]
        {
            return Err(EngineError::generation(format!(
                "resident MTP slot {slot} cannot retain message boundary {message_boundary_tokens} inside {} processed tokens",
                tokens.len()
            )));
        }
        let next_clock = self
            .retention_clock
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident MTP retention clock overflows"))?;
        self.program
            .truncate_kv_slot_tokens(&self.stream, slot, tokens.len())?;
        self.program.retain_kv_slot(slot)?;
        self.retention_clock = next_clock;
        self.retained[slot] = Some(RetainedMtpSlot {
            tokens,
            message_boundary_tokens,
            last_used: self.retention_clock,
        });
        Ok(())
    }
}

fn best_retained_prefix(
    retained: &[Option<RetainedMtpSlot>; MAX_BATCH],
    prompt: &[u32],
) -> Option<RetainedMatch> {
    retained
        .iter()
        .enumerate()
        .filter_map(|(slot, retained)| {
            let retained = retained.as_ref()?;
            let (tokens, reuse) = if prompt.starts_with(&retained.tokens) {
                (retained.tokens.len(), RetainedReuse::Complete)
            } else {
                let boundary = retained.tokens.get(..retained.message_boundary_tokens)?;
                prompt
                    .starts_with(boundary)
                    .then_some((boundary.len(), RetainedReuse::MessageBoundary))?
            };
            Some(RetainedMatch {
                slot,
                tokens,
                last_used: retained.last_used,
                reuse,
            })
        })
        .max_by_key(|matched| {
            (
                matched.tokens,
                matched.last_used,
                matched.reuse == RetainedReuse::Complete,
            )
        })
}

impl ParkedQwen38Generator {
    /// Exact pinned mirror and typed-manifest ownership retained while parked.
    pub fn host_bytes(&self) -> usize {
        self.mirror.host_bytes()
    }

    /// Exact physical allocation bytes released by the completed park.
    pub const fn released_device_bytes(&self) -> usize {
        self.released_device_bytes
    }

    /// Target/MTP VMM physical arena bytes still mapped while parked.
    pub fn remaining_device_bytes(&self) -> usize {
        self.generator.device_owner_bytes()
    }

    /// Recreates all arena mappings, reloads weights, and restores every durable represented bit.
    #[allow(clippy::result_large_err)]
    pub fn resume(self) -> Result<ResidentMtpBatchGenerator, (Self, EngineError)> {
        let Self {
            mut generator,
            mirror,
            released_device_bytes,
        } = self;
        if let Err(error) = restore_park_mirror(&mut generator, &mirror) {
            let error = match generator.program.park_device_arenas(&generator.stream) {
                Ok(_) => error,
                Err(rollback) => EngineError::generation(format!(
                    "resume failed: {error}; remapping rollback also failed: {rollback}"
                )),
            };
            return Err((
                Self {
                    generator,
                    mirror,
                    released_device_bytes,
                },
                error,
            ));
        }
        Ok(generator)
    }
}

impl Qwen38ParkMirror {
    fn host_bytes(&self) -> usize {
        self.target_tables.num_bytes()
            + self.target_history.num_bytes()
            + self.target_state.num_bytes()
            + self.target_key.num_bytes()
            + self.target_value.num_bytes()
            + self.mtp_key.num_bytes()
            + self.mtp_value.num_bytes()
            + std::mem::size_of_val(self.slots.as_ref())
            + std::mem::size_of_val(self.physical_pages.as_ref())
            + std::mem::size_of::<u64>()
    }

    fn require_checksums(&self, generator: &ResidentMtpBatchGenerator) -> EngineResult<()> {
        for row in 0..self.slots.len() {
            let observed = self.slot_checksum(row, generator)?;
            if observed != self.slots[row].checksum {
                return Err(EngineError::generation(format!(
                    "parked Qwen3.8 mirror checksum changed for slot {}",
                    self.slots[row].slot
                )));
            }
        }
        Ok(())
    }

    fn slot_checksum(
        &self,
        row: usize,
        generator: &ResidentMtpBatchGenerator,
    ) -> EngineResult<u64> {
        let manifest = self
            .slots
            .get(row)
            .ok_or_else(|| EngineError::layout("parked Qwen3.8 mirror row is absent"))?;
        let page_end = manifest
            .first_page
            .checked_add(manifest.page_count)
            .ok_or_else(|| EngineError::layout("parked Qwen3.8 page range overflows"))?;
        if page_end > self.physical_pages.len() {
            return Err(EngineError::layout(
                "parked Qwen3.8 page range exceeds its manifest",
            ));
        }
        let mut checksum = FNV_OFFSET;
        checksum_usize(&mut checksum, manifest.slot);
        checksum_usize(&mut checksum, manifest.token_count);
        checksum_u64(&mut checksum, manifest.retention_generation);
        checksum_u32(
            &mut checksum,
            &self.physical_pages[manifest.first_page..page_end],
        );

        let table = checked_row_range(row, LONG_CONTEXT_PHYSICAL_PAGES)?;
        checksum_u32(&mut checksum, &self.target_tables[table]);
        let history = checked_row_range(row, generator.program.target().gdn_slot_history_values())?;
        checksum_u16(&mut checksum, &self.target_history[history]);
        let state = checked_row_range(row, generator.program.target().gdn_slot_state_values())?;
        checksum_f32(&mut checksum, &self.target_state[state]);

        let target_page_values = generator.program.target().park_cache_page_values()?;
        let target =
            checked_page_range(manifest.first_page, manifest.page_count, target_page_values)?;
        checksum_u8(&mut checksum, &self.target_key[target.clone()]);
        checksum_u8(&mut checksum, &self.target_value[target]);
        let mtp_page_values = generator.program.park_cache_page_values()?;
        let mtp = checked_page_range(manifest.first_page, manifest.page_count, mtp_page_values)?;
        checksum_u16(&mut checksum, &self.mtp_key[mtp.clone()]);
        checksum_u16(&mut checksum, &self.mtp_value[mtp]);
        Ok(checksum)
    }
}

fn restore_park_mirror(
    generator: &mut ResidentMtpBatchGenerator,
    mirror: &Qwen38ParkMirror,
) -> EngineResult<()> {
    mirror.require_checksums(generator)?;
    generator.program.resume_device_arenas(&generator.stream)?;
    generator
        .program
        .reload_released_arenas(&generator.stream)?;
    generator
        .program
        .target()
        .restore_all_block_tables(&generator.stream)?;
    generator.program.restore_block_tables(&generator.stream)?;
    for (row, manifest) in mirror.slots.iter().enumerate() {
        generator.program.target().restore_gdn_slot_from(
            &generator.stream,
            manifest.slot,
            row,
            &mirror.target_history,
            &mirror.target_state,
        )?;
    }
    for (mirror_page, &physical_page) in mirror.physical_pages.iter().enumerate() {
        let physical_page = physical_page as usize;
        generator.program.target().restore_cache_page_from(
            &generator.stream,
            physical_page,
            mirror_page,
            &mirror.target_key,
            &mirror.target_value,
        )?;
        generator.program.restore_cache_page_from(
            &generator.stream,
            physical_page,
            mirror_page,
            &mirror.mtp_key,
            &mirror.mtp_value,
        )?;
    }
    generator.stream.synchronize().map_err(GpuError::from)?;
    generator
        .program
        .target()
        .require_device_tables_match_host(&generator.stream)?;
    let mtp_table_checksum = generator
        .program
        .mtp_table_checksum_against_host(&generator.stream)?;
    if mtp_table_checksum != mirror.mtp_table_checksum {
        return Err(EngineError::generation(
            "resident MTP device block tables changed while parked",
        ));
    }
    mirror.require_checksums(generator)?;
    Ok(())
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn checksum_byte(checksum: &mut u64, byte: u8) {
    *checksum ^= u64::from(byte);
    *checksum = checksum.wrapping_mul(FNV_PRIME);
}

fn checksum_u8(checksum: &mut u64, values: &[u8]) {
    for &value in values {
        checksum_byte(checksum, value);
    }
}

fn checksum_u16(checksum: &mut u64, values: &[u16]) {
    for value in values {
        checksum_u8(checksum, &value.to_ne_bytes());
    }
}

fn checksum_u32(checksum: &mut u64, values: &[u32]) {
    for value in values {
        checksum_u8(checksum, &value.to_ne_bytes());
    }
}

fn checksum_f32(checksum: &mut u64, values: &[f32]) {
    for value in values {
        checksum_u8(checksum, &value.to_bits().to_ne_bytes());
    }
}

fn checksum_u64(checksum: &mut u64, value: u64) {
    checksum_u8(checksum, &value.to_ne_bytes());
}

fn checksum_usize(checksum: &mut u64, value: usize) {
    checksum_u64(checksum, value as u64);
}

fn checked_row_range(row: usize, width: usize) -> EngineResult<std::ops::Range<usize>> {
    checked_page_range(row, 1, width)
}

fn checked_page_range(
    first_page: usize,
    page_count: usize,
    page_values: usize,
) -> EngineResult<std::ops::Range<usize>> {
    let start = product("parked Qwen3.8 mirror range start", first_page, page_values)?;
    let values = product(
        "parked Qwen3.8 mirror range values",
        page_count,
        page_values,
    )?;
    let end = start
        .checked_add(values)
        .ok_or_else(|| EngineError::layout("parked Qwen3.8 mirror range overflows"))?;
    Ok(start..end)
}

fn processed_tokens(session: &ResidentMtpBatchSession) -> EngineResult<Vec<u32>> {
    let prompt = session.control.prompt_token_ids();
    let generated = session.control.generated_token_ids();
    let processed_generated = session
        .next_position
        .checked_sub(prompt.len())
        .ok_or_else(|| {
            EngineError::generation("resident MTP processed position precedes its prompt")
        })?;
    if processed_generated > generated.len() {
        return Err(EngineError::generation(
            "resident MTP processed position exceeds emitted generation",
        ));
    }
    let mut tokens = Vec::with_capacity(session.next_position);
    tokens.extend_from_slice(prompt);
    tokens.extend_from_slice(&generated[..processed_generated]);
    Ok(tokens)
}

fn checked_rows(label: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| EngineError::layout(format!("{label} overflows")))
}

fn score_logit_row(logits: &[u16], token_id: u32) -> EngineResult<PromptTokenLogprob> {
    let token = usize::try_from(token_id)
        .ok()
        .filter(|&token| token < logits.len())
        .ok_or_else(|| EngineError::sampling("scored token is outside its logit row"))?;
    let (top_token, top_logit, log_normalizer) = logit_row_normalizer(logits)?;
    let selected = f64::from(bf16_to_f32(logits[token]));
    Ok(PromptTokenLogprob {
        token_id,
        logprob: (selected - log_normalizer) as f32,
        top_token_id: u32::try_from(top_token)
            .map_err(|_| EngineError::sampling("greedy token exceeds u32"))?,
        top_logprob: (top_logit - log_normalizer) as f32,
    })
}

fn score_greedy_row(logits: &[u16]) -> EngineResult<PromptTokenLogprob> {
    let (top_token, top_logit, log_normalizer) = logit_row_normalizer(logits)?;
    let token_id =
        u32::try_from(top_token).map_err(|_| EngineError::sampling("greedy token exceeds u32"))?;
    let logprob = (top_logit - log_normalizer) as f32;
    Ok(PromptTokenLogprob {
        token_id,
        logprob,
        top_token_id: token_id,
        top_logprob: logprob,
    })
}

fn logit_row_normalizer(logits: &[u16]) -> EngineResult<(usize, f64, f64)> {
    let mut best = None;
    for (token, &bits) in logits.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(EngineError::sampling(format!(
                "prompt scoring logit {token} is not finite"
            )));
        }
        if best.is_none_or(|(_, retained): (usize, f32)| value.total_cmp(&retained).is_gt()) {
            best = Some((token, value));
        }
    }
    let (top_token, top_logit) =
        best.ok_or_else(|| EngineError::sampling("prompt scoring received an empty logit row"))?;
    let top_logit = f64::from(top_logit);
    let denominator = logits
        .iter()
        .map(|&bits| (f64::from(bf16_to_f32(bits)) - top_logit).exp())
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(EngineError::sampling(
            "prompt scoring softmax denominator is not finite and positive",
        ));
    }
    Ok((top_token, top_logit, top_logit + denominator.ln()))
}

fn target_slot_logits(slot: usize) -> std::ops::Range<usize> {
    row(slot, Qwen38_27B::VOCAB)
}

fn target_download_logits(rows: usize) -> std::ops::Range<usize> {
    compact(rows, Qwen38_27B::VOCAB)
}

fn target_download_row(row_index: usize) -> std::ops::Range<usize> {
    row(MAX_BATCH + row_index, Qwen38_27B::VOCAB)
}

fn draft_slot_logits(slot: usize) -> std::ops::Range<usize> {
    row(slot, Qwen38_27B::VOCAB)
}

fn compact_draft_logits(rows: usize) -> std::ops::Range<usize> {
    compact(rows, Qwen38_27B::VOCAB)
}

fn compact_draft_row(row_index: usize) -> std::ops::Range<usize> {
    row(MAX_BATCH + row_index, Qwen38_27B::VOCAB)
}

fn hidden_slot(slot: usize) -> std::ops::Range<usize> {
    row(slot, Qwen38_27B::HIDDEN)
}

fn compact_hidden(rows: usize) -> std::ops::Range<usize> {
    compact(rows, Qwen38_27B::HIDDEN)
}

fn compact_hidden_row(row_index: usize) -> std::ops::Range<usize> {
    row(MAX_BATCH + row_index, Qwen38_27B::HIDDEN)
}

#[cfg(test)]
mod tests {
    use super::{
        DRAFT_HIDDEN_ROWS, DRAFT_LOGIT_ROWS, RetainedMtpSlot, RetainedReuse, TARGET_LOGIT_ROWS,
        best_retained_prefix, score_greedy_row, score_logit_row, target_download_logits,
        target_download_row,
    };
    use crate::MAX_BATCH;
    use crate::common::banks::row;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn host_stager_inventory_has_disjoint_slot_and_compact_banks() {
        assert_eq!(TARGET_LOGIT_ROWS, MAX_BATCH + 4 * MAX_BATCH);
        assert_eq!(DRAFT_LOGIT_ROWS, 2 * MAX_BATCH);
        assert_eq!(DRAFT_HIDDEN_ROWS, 2 * MAX_BATCH);
    }

    #[test]
    fn prompt_logprobs_use_natural_log_softmax_and_first_argmax_ties() {
        let logits = [0.0f32, 1.0, 1.0, -2.0].map(|value| (value.to_bits() >> 16) as u16);
        let selected = score_logit_row(&logits, 0).unwrap();
        let greedy = score_greedy_row(&logits).unwrap();
        let normalizer = (1.0f64.exp() + 1.0f64.exp() + 1.0 + (-2.0f64).exp()).ln();

        assert_eq!(selected.token_id, 0);
        assert_eq!(selected.top_token_id, 1);
        assert!((f64::from(selected.logprob) + normalizer).abs() < 1.0e-6);
        assert_eq!(greedy.token_id, 1);
        assert!((f64::from(greedy.logprob) - (1.0 - normalizer)).abs() < 1.0e-6);
    }

    #[test]
    fn prompt_scoring_download_rows_follow_the_per_slot_logit_bank() {
        let first_download = row(MAX_BATCH, Qwen38_27B::VOCAB);

        assert_eq!(target_download_row(0), first_download);
        assert_eq!(target_download_logits(1), first_download);
    }

    #[test]
    fn retained_lookup_falls_back_to_the_complete_message_boundary() {
        let mut retained = std::array::from_fn(|_| None);
        retained[0] = Some(RetainedMtpSlot {
            tokens: vec![1, 2, 3, 4],
            message_boundary_tokens: 2,
            last_used: 7,
        });

        let exact = best_retained_prefix(&retained, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!((exact.slot, exact.tokens), (0, 4));
        assert_eq!(exact.reuse, RetainedReuse::Complete);

        let fallback = best_retained_prefix(&retained, &[1, 2, 9, 5]).unwrap();
        assert_eq!((fallback.slot, fallback.tokens), (0, 2));
        assert_eq!(fallback.reuse, RetainedReuse::MessageBoundary);
        assert!(best_retained_prefix(&retained, &[1, 9]).is_none());
    }

    #[test]
    fn retained_lookup_prefers_the_longest_safe_prefix() {
        let mut retained = std::array::from_fn(|_| None);
        retained[0] = Some(RetainedMtpSlot {
            tokens: vec![1, 2, 8],
            message_boundary_tokens: 2,
            last_used: 9,
        });
        retained[1] = Some(RetainedMtpSlot {
            tokens: vec![1, 2, 3, 7],
            message_boundary_tokens: 3,
            last_used: 1,
        });

        let matched = best_retained_prefix(&retained, &[1, 2, 3, 6]).unwrap();
        assert_eq!((matched.slot, matched.tokens), (1, 3));
        assert_eq!(matched.reuse, RetainedReuse::MessageBoundary);
    }
}
