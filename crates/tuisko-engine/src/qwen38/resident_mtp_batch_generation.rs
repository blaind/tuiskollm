//! Compact greedy and sampled MTP generation over the resident target-plus-draft owner.

use crate::common::banks::{compact, row};
use crate::common::mtp::{
    DRAFT_WINDOW, MtpEventBuilder, VERIFY_ROWS, decide_greedy_lane, decide_sampled_lane,
    require_generation_capacity,
};
use crate::common::rope::{ROTARY_PAIRS, fill_contiguous_rope, text_rope};
use crate::common::slots::device_zero_context;
use crate::qwen38::resident_mtp_generation::prime_prompt;
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GeneratedText, GenerationSession,
    GenerationStep, MAX_BATCH, ResidentBatchAdmission, ResidentCancellation, ResidentLoadProgress,
    ResidentMtpGenerationStats, ResidentMtpLoadStats, ResidentMtpProgram,
    ResidentMtpSegmentedVerifyRoute, ResidentMtpVerifyRoute, ResidentRequestId,
    SamplingDistribution,
};
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const TARGET_DOWNLOAD_ROWS: usize = MAX_BATCH * VERIFY_ROWS;
const TARGET_LOGIT_ROWS: usize = MAX_BATCH + TARGET_DOWNLOAD_ROWS;
const DRAFT_LOGIT_ROWS: usize = 2 * MAX_BATCH;
const DRAFT_HIDDEN_ROWS: usize = 2 * MAX_BATCH;

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
    last_used: u64,
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

        Ok(Self {
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
        })
    }

    /// Admits one request and restores only an exact shared target/MTP prefix.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
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

        let (slot, reused, reset) = self.select_slot(control.prompt_token_ids())?;
        if reused > message_boundary_tokens {
            return Err(EngineError::generation(format!(
                "resident MTP reused prefix {reused} exceeds message boundary {message_boundary_tokens}"
            )));
        }
        if reset {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            self.program.reset_slot(&self.stream, slot)?;
            self.message_boundary_valid[slot] = false;
        }
        self.program.activate_kv_slot(slot)?;
        self.program
            .reserve_kv_slot_tokens(&self.stream, slot, required_positions)?;
        self.program
            .target()
            .load_slot_routes(&self.stream, &[slot])?;

        let mut native_prefill_tokens = 0usize;
        if reused < message_boundary_tokens {
            let retained_hidden =
                (reused != 0).then(|| &self.target_boundary_hidden[hidden_slot(slot)]);
            native_prefill_tokens = prime_prompt(
                &mut self.program,
                &self.stream,
                control.message_boundary_token_ids(),
                slot,
                reused,
                retained_hidden,
            )?;
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
            self.message_boundary_valid[slot] = true;
        } else if !self.message_boundary_valid[slot] {
            return Err(EngineError::generation(
                "resident MTP reused message boundary has no state snapshot",
            ));
        }
        if message_boundary_tokens < prompt_tokens {
            let suffix_native = prime_prompt(
                &mut self.program,
                &self.stream,
                control.prompt_token_ids(),
                slot,
                message_boundary_tokens,
                Some(&self.message_boundary_hidden[hidden_slot(slot)]),
            )?;
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
        self.store_retained(slot, retained)?;
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
    pub const fn device_owner_bytes(&self) -> usize {
        self.program.target().arena_bytes() + self.program.owner_bytes()
    }

    /// Complete target-plus-MTP device ownership reported at server startup.
    pub const fn arena_bytes(&self) -> usize {
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
                self.store_retained(slot, retained)?;
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

    fn select_slot(&mut self, prompt: &[u32]) -> EngineResult<(usize, usize, bool)> {
        let prefix = self
            .retained
            .iter()
            .enumerate()
            .filter_map(|(slot, retained)| {
                retained.as_ref().and_then(|retained| {
                    prompt.starts_with(&retained.tokens).then_some((
                        slot,
                        retained.tokens.len(),
                        retained.last_used,
                    ))
                })
            })
            .max_by_key(|&(_, tokens, last_used)| (tokens, last_used));
        if let Some((slot, tokens, _)) = prefix {
            self.retained[slot] = None;
            return Ok((slot, tokens, false));
        }
        if let Some(slot) = (0..MAX_BATCH)
            .find(|&slot| self.sessions[slot].is_none() && self.retained[slot].is_none())
        {
            return Ok((slot, 0, true));
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
            .ok_or_else(|| EngineError::route("all eight resident MTP slots are active"))?;
        self.retained[eviction] = None;
        Ok((eviction, 0, true))
    }

    fn store_retained(&mut self, slot: usize, tokens: Vec<u32>) -> EngineResult<()> {
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
            last_used: self.retention_clock,
        });
        Ok(())
    }
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
    use super::{DRAFT_HIDDEN_ROWS, DRAFT_LOGIT_ROWS, TARGET_LOGIT_ROWS};
    use crate::MAX_BATCH;

    #[test]
    fn host_stager_inventory_has_disjoint_slot_and_compact_banks() {
        assert_eq!(TARGET_LOGIT_ROWS, MAX_BATCH + 4 * MAX_BATCH);
        assert_eq!(DRAFT_LOGIT_ROWS, 2 * MAX_BATCH);
        assert_eq!(DRAFT_HIDDEN_ROWS, 2 * MAX_BATCH);
    }
}
