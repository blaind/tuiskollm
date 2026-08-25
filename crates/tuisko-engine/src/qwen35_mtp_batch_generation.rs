//! Compact Qwen3.5 MTP generation over eight mirrored target/draft slots.

use crate::qwen35_mtp_generation::prime_qwen35_mtp_prompt;
use crate::resident_generation::{device_zero_context, require_generation_capacity, text_rope};
use crate::resident_mtp_generation::{
    DRAFT_WINDOW, VERIFY_ROWS, decide_sampled_tokens, fill_contiguous_rope,
};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GeneratedText, GenerationSession,
    GenerationStep, MAX_BATCH, Qwen35ResidentMtpProgram, ResidentBatchAdmission,
    ResidentCancellation, ResidentMtpGenerationStats, ResidentRequestId, SamplingDistribution,
};
use std::ops::Range;
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const ROTARY_PAIRS: usize = 32;
const TARGET_DOWNLOAD_ROWS: usize = MAX_BATCH * VERIFY_ROWS;
const TARGET_LOGIT_ROWS: usize = MAX_BATCH + TARGET_DOWNLOAD_ROWS;
const DRAFT_LOGIT_ROWS: usize = 2 * MAX_BATCH;
const TARGET_HIDDEN_ROWS: usize = MAX_BATCH * VERIFY_ROWS;
const DRAFT_HIDDEN_ROWS: usize = 2 * MAX_BATCH;

/// Concrete Qwen3.5 MTP scheduler for up to eight resident requests.
pub struct Qwen35ResidentMtpBatchGenerator {
    frontend: TextFrontend,
    program: Qwen35ResidentMtpProgram,
    stream: Arc<CudaStream>,
    target_logits: PinnedHostBuffer<u16>,
    draft_logits: PinnedHostBuffer<u16>,
    target_boundary_hidden: PinnedHostBuffer<u16>,
    target_hidden: PinnedHostBuffer<u16>,
    draft_hidden: PinnedHostBuffer<u16>,
    sessions: [Option<Qwen35MtpBatchSession>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
    stop_ids: Vec<u32>,
}

/// One compact scheduler event containing every token committed by one transaction.
pub struct Qwen35ResidentMtpBatchEvent {
    /// Request that produced this event.
    pub request_id: ResidentRequestId,
    steps: [Option<GenerationStep>; VERIFY_ROWS],
    len: usize,
    /// Complete output when the final committed token terminated the request.
    pub completed: Option<GeneratedText>,
    /// Cumulative exact-route and acceptance counters for this request.
    pub stats: ResidentMtpGenerationStats,
}

/// At most eight Qwen3.5 request events in stable active order at round entry.
pub struct Qwen35ResidentMtpBatchEvents {
    events: [Option<Qwen35ResidentMtpBatchEvent>; MAX_BATCH],
    len: usize,
}

struct Qwen35MtpBatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    next_position: usize,
    maximum_new_tokens: usize,
    started: bool,
    proposal_ready: bool,
    greedy: bool,
    stats: ResidentMtpGenerationStats,
}

struct EventBuilder {
    steps: [Option<GenerationStep>; VERIFY_ROWS],
    len: usize,
}

struct LaneDrafts {
    tokens: [u32; DRAFT_WINDOW],
    laws: [Option<SamplingDistribution>; DRAFT_WINDOW],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoundRoute {
    Mtp,
    CompactTarget,
}

impl EventBuilder {
    fn new() -> Self {
        Self {
            steps: std::array::from_fn(|_| None),
            len: 0,
        }
    }

    fn push(&mut self, step: GenerationStep) -> EngineResult<()> {
        let destination = self.steps.get_mut(self.len).ok_or_else(|| {
            EngineError::generation("one Qwen3.5 MTP transaction produced more than four outputs")
        })?;
        *destination = Some(step);
        self.len += 1;
        Ok(())
    }

    fn token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.steps[..self.len].iter().map(|step| {
            step.as_ref()
                .expect("Qwen3.5 MTP event prefix is initialized")
                .token_id
        })
    }
}

impl LaneDrafts {
    fn new() -> Self {
        Self {
            tokens: [0; DRAFT_WINDOW],
            laws: std::array::from_fn(|_| None),
        }
    }
}

impl Qwen35ResidentMtpBatchEvent {
    /// Number of tokens committed for this request in the transaction.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this request committed no token.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Streaming steps in target-licensed order.
    pub fn steps(&self) -> impl Iterator<Item = &GenerationStep> {
        self.steps[..self.len].iter().map(|step| {
            step.as_ref()
                .expect("Qwen3.5 MTP event prefix is initialized")
        })
    }
}

impl Qwen35ResidentMtpBatchEvents {
    /// Number of requests that produced an event in this scheduler transaction.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the transaction produced no request events.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Request events in the stable scheduler order present at round entry.
    pub fn iter(&self) -> impl Iterator<Item = &Qwen35ResidentMtpBatchEvent> {
        self.events[..self.len].iter().map(|event| {
            event
                .as_ref()
                .expect("Qwen3.5 MTP batch event prefix is initialized")
        })
    }
}

impl Qwen35ResidentMtpBatchGenerator {
    /// Opens the exact compact target-plus-MTP scheduler on CUDA device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Loads one target, one MTP layer, and eight persistent request slots.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
        let stop_ids = frontend.stop_ids().to_vec();
        let program = Qwen35ResidentMtpProgram::from_snapshot(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let target_logits = pinned_rows(
            context,
            "Qwen3.5 compact MTP target logits",
            TARGET_LOGIT_ROWS,
            Qwen35_9B::VOCAB,
        )?;
        let draft_logits = pinned_rows(
            context,
            "Qwen3.5 compact MTP draft logits",
            DRAFT_LOGIT_ROWS,
            Qwen35_9B::VOCAB,
        )?;
        let target_boundary_hidden = pinned_rows(
            context,
            "Qwen3.5 compact MTP target boundary",
            MAX_BATCH,
            Qwen35_9B::HIDDEN,
        )?;
        let target_hidden = pinned_rows(
            context,
            "Qwen3.5 compact MTP target transaction",
            TARGET_HIDDEN_ROWS,
            Qwen35_9B::HIDDEN,
        )?;
        let draft_hidden = pinned_rows(
            context,
            "Qwen3.5 compact MTP draft hidden",
            DRAFT_HIDDEN_ROWS,
            Qwen35_9B::HIDDEN,
        )?;

        Ok(Self {
            frontend,
            program,
            stream,
            target_logits,
            draft_logits,
            target_boundary_hidden,
            target_hidden,
            draft_hidden,
            sessions: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
            stop_ids,
        })
    }

    /// Admits one request into the first free mirrored target/MTP slot.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        let control = GenerationSession::start(&self.frontend, request)?;
        let prompt_tokens = control.prompt_token_ids().len();
        let required_positions = require_generation_capacity(
            prompt_tokens,
            request.max_new_tokens,
            self.program.layout().context_capacity(),
        )?;
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP request identity overflows"))?;
        let prompt_metrics = control.prompt_metrics().clone();
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

        let occupied = std::array::from_fn(|slot| self.sessions[slot].is_some());
        let slot = first_free_slot(occupied)
            .ok_or_else(|| EngineError::route("all eight Qwen3.5 MTP slots are active"))?;
        self.program.recycle_kv_slot(&self.stream, slot)?;
        self.program.target_mut().reset_slot(&self.stream, slot)?;
        self.program.activate_kv_slot(slot)?;
        if let Err(error) =
            self.program
                .reserve_kv_slot_tokens(&self.stream, slot, required_positions)
        {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            return Err(error);
        }
        let native_prefill_tokens = prime_qwen35_mtp_prompt(
            &mut self.program,
            &self.stream,
            control.prompt_token_ids(),
            slot,
            &mut self.target_hidden[..Qwen35_9B::HIDDEN],
        )?;
        self.program.target().read_logits_into(
            &self.stream,
            1,
            &mut self.target_logits[target_slot_logits(slot)],
        )?;
        self.program.target().read_final_residual_into(
            &self.stream,
            1,
            &mut self.target_boundary_hidden[hidden_slot(slot)],
        )?;
        self.sessions[slot] = Some(Qwen35MtpBatchSession {
            request_id,
            control,
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
            device_reused_tokens: 0,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    /// Advances every active request by one anchor, tail, or speculative transaction.
    pub fn step(&mut self) -> EngineResult<Qwen35ResidentMtpBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty Qwen3.5 MTP scheduler",
            ));
        }
        let active = self.active;
        let active_slots = self.active_slots;
        let mut builders: [EventBuilder; MAX_BATCH] = std::array::from_fn(|_| EventBuilder::new());
        let mut started = [false; MAX_BATCH];
        let mut fresh = [usize::MAX; MAX_BATCH];
        let mut fresh_count = 0;
        for &slot in &active_slots[..active] {
            let session = self.sessions[slot]
                .as_ref()
                .ok_or_else(|| EngineError::generation("active Qwen3.5 MTP slot has no session"))?;
            started[slot] = session.started;
            if !session.started {
                fresh[fresh_count] = slot;
                fresh_count += 1;
            }
        }
        if fresh_count != 0 {
            self.start_anchors(&fresh[..fresh_count], &mut builders)?;
        }

        let mut continuing = [usize::MAX; MAX_BATCH];
        let mut continuing_count = 0;
        for &slot in &active_slots[..active] {
            if !started[slot] {
                continue;
            }
            continuing[continuing_count] = slot;
            continuing_count += 1;
        }
        match round_route(continuing_count) {
            Some(RoundRoute::Mtp) => {
                let slot = continuing[0];
                let session = self.sessions[slot]
                    .as_ref()
                    .expect("started Qwen3.5 MTP session exists");
                let remaining = session
                    .maximum_new_tokens
                    .checked_sub(session.control.generated_token_ids().len())
                    .ok_or_else(|| EngineError::generation("Qwen3.5 MTP budget underflows"))?;
                if remaining == 1 {
                    self.run_tail(&continuing[..1], &mut builders)?;
                } else {
                    self.run_speculative(&continuing[..1], &mut builders)?;
                }
            }
            Some(RoundRoute::CompactTarget) => {
                self.run_compact_target(&continuing[..continuing_count], &mut builders)?;
            }
            None => {}
        }

        self.finish_events(&active_slots[..active], builders)
    }

    /// Cancels one active request at a complete scheduler-round boundary.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP cancellation is not active"))?;
        let slot = self.active_slots[index];
        let session = self.sessions[slot]
            .take()
            .expect("cancelled Qwen3.5 MTP session exists");
        for position in index..self.active - 1 {
            self.active_slots[position] = self.active_slots[position + 1];
        }
        self.active -= 1;
        self.active_slots[self.active] = usize::MAX;
        self.program.recycle_kv_slot(&self.stream, slot)?;

        Ok(ResidentCancellation {
            request_id,
            output: session.control.cancel()?,
            device_retained_tokens: 0,
        })
    }

    /// Number of currently active physical request slots.
    pub const fn active_requests(&self) -> usize {
        self.active
    }

    /// Active request identities in stable compact-row order.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active_slots[..self.active].iter().map(|&slot| {
            self.sessions[slot]
                .as_ref()
                .expect("active Qwen3.5 MTP slot owns a session")
                .request_id
        })
    }

    /// Complete target, MTP, and mirrored-cache device bytes.
    pub const fn arena_bytes(&self) -> usize {
        self.program.layout().arena_bytes()
    }

    /// Source-backed target and MTP weight bytes resident on the device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.program.layout().resident_weight_bytes()
    }

    /// Fixed page-locked embedding, logit, and hidden staging bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes()
            + self.target_logits.num_bytes()
            + self.draft_logits.num_bytes()
            + self.target_boundary_hidden.num_bytes()
            + self.target_hidden.num_bytes()
            + self.draft_hidden.num_bytes()
    }

    /// Maximum context admitted by the pinned Qwen3.5 snapshot.
    pub const fn context_capacity(&self) -> usize {
        self.program.layout().context_capacity()
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// CUDA context shared by every retained owner and graph.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Stable target, MTP, cache, and pinned scheduler addresses.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = self.program.qualification_addresses()?;
        addresses.extend([
            self.target_logits.as_ptr().addr(),
            self.draft_logits.as_ptr().addr(),
            self.target_boundary_hidden.as_ptr().addr(),
            self.target_hidden.as_ptr().addr(),
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
    /// Complete owner exposed for direct composed timing and accounting.
    pub const fn qualification_program(&self) -> &Qwen35ResidentMtpProgram {
        &self.program
    }

    fn start_anchors(
        &mut self,
        slots: &[usize],
        builders: &mut [EventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let mut seeded_slots = [usize::MAX; MAX_BATCH];
        let mut anchors = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        let mut cosine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut sine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut seeded = 0;
        for &slot in slots {
            let session = self.sessions[slot]
                .as_mut()
                .expect("fresh Qwen3.5 MTP session exists");
            let step = session
                .control
                .accept_logits(&self.target_logits[target_slot_logits(slot)])?;
            session.started = true;
            let terminal = step.finish_reason.is_some();
            builders[slot].push(step)?;
            if terminal {
                continue;
            }
            seeded_slots[seeded] = slot;
            anchors[seeded] = builders[slot].steps[0]
                .as_ref()
                .expect("fresh Qwen3.5 anchor exists")
                .token_id;
            positions[seeded] = u32::try_from(session.next_position - 1)
                .map_err(|_| EngineError::generation("Qwen3.5 MTP anchor exceeds u32"))?;
            let (row_cosine, row_sine) = text_rope(positions[seeded]);
            let rotary = seeded * ROTARY_PAIRS;
            cosine[rotary..rotary + ROTARY_PAIRS].copy_from_slice(&row_cosine);
            sine[rotary..rotary + ROTARY_PAIRS].copy_from_slice(&row_sine);
            let source = hidden_slot(slot);
            let destination = compact_hidden_row(seeded);
            self.draft_hidden[destination].copy_from_slice(&self.target_boundary_hidden[source]);
            seeded += 1;
        }
        if seeded == 0 {
            return Ok(());
        }
        self.program.stage_continuation_draft(
            &self.stream,
            &anchors[..seeded],
            &self.draft_hidden[compact_hidden(seeded)],
            &positions[..seeded],
            &seeded_slots[..seeded],
            &cosine[..seeded * ROTARY_PAIRS],
            &sine[..seeded * ROTARY_PAIRS],
        )?;
        self.program.replay_staged_draft(&self.stream, seeded)?;
        self.program.read_logits_into(
            &self.stream,
            seeded,
            &mut self.draft_logits[compact_draft_logits(seeded)],
        )?;
        self.program.read_mtp_residuals_into(
            &self.stream,
            seeded,
            &mut self.draft_hidden[compact_hidden(seeded)],
        )?;
        for (lane, &slot) in seeded_slots[..seeded].iter().enumerate() {
            self.draft_logits
                .copy_within(compact_draft_row(lane), slot * Qwen35_9B::VOCAB);
            self.draft_hidden
                .copy_within(compact_hidden_row(lane), slot * Qwen35_9B::HIDDEN);
            self.sessions[slot]
                .as_mut()
                .expect("seeded Qwen3.5 MTP session exists")
                .proposal_ready = true;
        }
        Ok(())
    }

    fn run_tail(
        &mut self,
        slots: &[usize],
        builders: &mut [EventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        for (lane, &slot) in slots.iter().enumerate() {
            let input = [self.anchor(slot)?];
            self.verify_target_lane(slot, lane, &input)?;
            let step = self.sessions[slot]
                .as_mut()
                .expect("tail Qwen3.5 MTP session exists")
                .control
                .accept_logits(&self.target_logits[target_download_row(lane)])?;
            builders[slot].push(step)?;
            self.commit_and_realign_lane(slot, lane, &input, &builders[slot], 1, 0)?;
        }
        Ok(())
    }

    fn run_compact_target(
        &mut self,
        slots: &[usize],
        builders: &mut [EventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        let mut cosine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut sine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        for (lane, &slot) in slots.iter().enumerate() {
            tokens[lane] = self.anchor(slot)?;
            positions[lane] = u32::try_from(
                self.sessions[slot]
                    .as_ref()
                    .expect("compact Qwen3.5 MTP session exists")
                    .next_position,
            )
            .map_err(|_| EngineError::generation("Qwen3.5 compact position exceeds u32"))?;
            let (row_cosine, row_sine) = text_rope(positions[lane]);
            let begin = lane * ROTARY_PAIRS;
            cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
            sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
        }

        // Serial K=4 target verification measured 93.54 ms at B=8 for 32
        // committed outputs. One exact B=8 target graph plus one compact MTP
        // alignment pass preserves each row's target and draft arithmetic while
        // sharing all immutable weight passes across the eight active requests.
        self.program
            .stage_target_embeddings(&self.stream, &tokens[..slots.len()])?;
        self.program
            .target()
            .load_slot_routes(&self.stream, slots)?;
        self.program.target().load_decode_state(
            &self.stream,
            slots.len(),
            &positions[..slots.len()],
            &cosine[..slots.len() * ROTARY_PAIRS],
            &sine[..slots.len() * ROTARY_PAIRS],
        )?;
        self.program.replay_target(&self.stream, slots.len())?;
        self.program.target().read_logits_into(
            &self.stream,
            slots.len(),
            &mut self.target_logits[lane_target_logits(0, slots.len())],
        )?;
        self.program.target().read_final_residual_into(
            &self.stream,
            slots.len(),
            &mut self.target_hidden[lane_target_hidden(0, slots.len())],
        )?;

        let mut aligned_slots = [usize::MAX; MAX_BATCH];
        let mut aligned_tokens = [0u32; MAX_BATCH];
        let mut aligned_positions = [0u32; MAX_BATCH];
        let mut aligned_cosine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut aligned_sine = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut aligned = 0;
        for (lane, &slot) in slots.iter().enumerate() {
            let logits = target_download_row(lane);
            let step = self.sessions[slot]
                .as_mut()
                .expect("compact Qwen3.5 MTP session exists")
                .control
                .accept_logits(&self.target_logits[logits])?;
            let terminal = step.finish_reason.is_some();
            builders[slot].push(step)?;
            let hidden = target_hidden_row(0, lane);
            self.target_boundary_hidden[hidden_slot(slot)]
                .copy_from_slice(&self.target_hidden[hidden.clone()]);
            self.target_logits
                .copy_within(target_download_row(lane), slot * Qwen35_9B::VOCAB);
            let session = self.sessions[slot]
                .as_mut()
                .expect("compact Qwen3.5 MTP session exists");
            session.next_position = session
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("Qwen3.5 MTP position overflows"))?;
            session.stats.verification_routes[0] += 1;
            session.stats.verified_outputs += 1;
            session.proposal_ready = !terminal;
            if terminal {
                continue;
            }
            aligned_slots[aligned] = slot;
            aligned_tokens[aligned] = builders[slot].steps[0]
                .as_ref()
                .expect("compact target step exists")
                .token_id;
            aligned_positions[aligned] = positions[lane];
            let source = hidden;
            let destination = compact_hidden_row(aligned);
            self.draft_hidden[destination].copy_from_slice(&self.target_hidden[source]);
            let rotary = aligned * ROTARY_PAIRS;
            aligned_cosine[rotary..rotary + ROTARY_PAIRS]
                .copy_from_slice(&cosine[lane * ROTARY_PAIRS..(lane + 1) * ROTARY_PAIRS]);
            aligned_sine[rotary..rotary + ROTARY_PAIRS]
                .copy_from_slice(&sine[lane * ROTARY_PAIRS..(lane + 1) * ROTARY_PAIRS]);
            aligned += 1;
        }
        if aligned == 0 {
            return Ok(());
        }
        self.program.stage_continuation_draft(
            &self.stream,
            &aligned_tokens[..aligned],
            &self.draft_hidden[compact_hidden(aligned)],
            &aligned_positions[..aligned],
            &aligned_slots[..aligned],
            &aligned_cosine[..aligned * ROTARY_PAIRS],
            &aligned_sine[..aligned * ROTARY_PAIRS],
        )?;
        self.program.replay_staged_draft(&self.stream, aligned)?;
        self.program.read_logits_into(
            &self.stream,
            aligned,
            &mut self.draft_logits[compact_draft_logits(aligned)],
        )?;
        self.program.read_mtp_residuals_into(
            &self.stream,
            aligned,
            &mut self.draft_hidden[compact_hidden(aligned)],
        )?;
        for (lane, &slot) in aligned_slots[..aligned].iter().enumerate() {
            self.draft_logits
                .copy_within(compact_draft_row(lane), slot * Qwen35_9B::VOCAB);
            self.draft_hidden
                .copy_within(compact_hidden_row(lane), slot * Qwen35_9B::HIDDEN);
        }
        Ok(())
    }

    fn run_speculative(
        &mut self,
        slots: &[usize],
        builders: &mut [EventBuilder; MAX_BATCH],
    ) -> EngineResult<()> {
        let extent = slots
            .iter()
            .map(|&slot| {
                let session = self.sessions[slot]
                    .as_ref()
                    .expect("Qwen3.5 MTP session exists");
                session
                    .maximum_new_tokens
                    .checked_sub(session.control.generated_token_ids().len())
                    .map(|remaining| DRAFT_WINDOW.min(remaining - 1))
                    .ok_or_else(|| EngineError::generation("Qwen3.5 MTP budget underflows"))
            })
            .collect::<EngineResult<Vec<_>>>()?
            .into_iter()
            .min()
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP speculative group is empty"))?;
        let mut drafts = self.propose_drafts(slots, extent)?;
        let tokens = extent + 1;
        for (lane, &slot) in slots.iter().enumerate() {
            let mut inputs = [0u32; VERIFY_ROWS];
            inputs[0] = self.anchor(slot)?;
            inputs[1..tokens].copy_from_slice(&drafts[lane].tokens[..extent]);
            self.verify_target_lane(slot, lane, &inputs[..tokens])?;
            let greedy = self.sessions[slot]
                .as_ref()
                .expect("verified Qwen3.5 MTP session exists")
                .greedy;
            let (committed, accepted) = if greedy {
                self.decide_greedy(slot, lane, extent, &drafts[lane], &mut builders[slot])?
            } else {
                self.decide_sampled(slot, lane, extent, &mut drafts[lane], &mut builders[slot])?
            };
            self.commit_and_realign_lane(
                slot,
                lane,
                &inputs[..tokens],
                &builders[slot],
                committed,
                accepted,
            )?;
        }
        Ok(())
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
                    .expect("proposing Qwen3.5 MTP session exists");
                if !session.proposal_ready {
                    return Err(EngineError::generation(
                        "Qwen3.5 MTP speculative lane has no aligned proposal",
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
            positions[lane] = self.sessions[slot]
                .as_ref()
                .expect("continuing Qwen3.5 MTP session exists")
                .next_position
                .checked_add(draft - 1)
                .and_then(|position| u32::try_from(position).ok())
                .ok_or_else(|| EngineError::generation("Qwen3.5 MTP position exceeds u32"))?;
            let (row_cosine, row_sine) = text_rope(positions[lane]);
            let begin = lane * ROTARY_PAIRS;
            cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
            sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
        }
        if draft == 1 {
            for (lane, &slot) in slots.iter().enumerate() {
                self.draft_hidden
                    .copy_within(hidden_slot(slot), compact_hidden_row(lane).start);
            }
            self.program.stage_continuation_draft(
                &self.stream,
                &tokens[..slots.len()],
                &self.draft_hidden[compact_hidden(slots.len())],
                &positions[..slots.len()],
                slots,
                &cosine[..slots.len() * ROTARY_PAIRS],
                &sine[..slots.len() * ROTARY_PAIRS],
            )?;
            self.program
                .replay_staged_draft(&self.stream, slots.len())?;
        } else {
            self.program
                .stage_mtp_embeddings(&self.stream, &tokens[..slots.len()])?;
            self.program.load_decode_state(
                &self.stream,
                &positions[..slots.len()],
                slots,
                &cosine[..slots.len() * ROTARY_PAIRS],
                &sine[..slots.len() * ROTARY_PAIRS],
            )?;
            self.program
                .replay_continue_draft(&self.stream, slots.len())?;
        }
        self.program.read_logits_into(
            &self.stream,
            slots.len(),
            &mut self.draft_logits[compact_draft_logits(slots.len())],
        )?;
        self.program.read_mtp_residuals_into(
            &self.stream,
            slots.len(),
            &mut self.draft_hidden[compact_hidden(slots.len())],
        )?;
        Ok(())
    }

    fn verify_target_lane(&mut self, slot: usize, lane: usize, inputs: &[u32]) -> EngineResult<()> {
        let first_position = self.sessions[slot]
            .as_ref()
            .expect("verified Qwen3.5 MTP session exists")
            .next_position;
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary = fill_contiguous_rope(first_position, inputs.len(), &mut cosine, &mut sine)?;
        self.program.target().capture_gdn_slot(&self.stream, slot)?;
        self.program.stage_target_verify(
            &self.stream,
            inputs,
            slot,
            first_position,
            &cosine[..rotary],
            &sine[..rotary],
        )?;
        self.program
            .replay_target_verify(&self.stream, inputs.len())?;
        let rows = lane_target_logits(lane, inputs.len());
        self.program
            .read_logits_into(&self.stream, inputs.len(), &mut self.target_logits[rows])?;
        let hidden = lane_target_hidden(lane, inputs.len());
        self.program.target().read_final_residual_into(
            &self.stream,
            inputs.len(),
            &mut self.target_hidden[hidden],
        )?;
        Ok(())
    }

    fn decide_greedy(
        &mut self,
        slot: usize,
        lane: usize,
        extent: usize,
        drafts: &LaneDrafts,
        builder: &mut EventBuilder,
    ) -> EngineResult<(usize, usize)> {
        let session = self.sessions[slot]
            .as_mut()
            .expect("greedy Qwen3.5 MTP session exists");
        let mut accepted = 0;
        for draft in 0..extent {
            let row = target_download_row(lane * (extent + 1) + draft);
            let step = session.control.accept_logits(&self.target_logits[row])?;
            let matches = step.token_id == drafts.tokens[draft];
            let terminal = step.finish_reason.is_some();
            builder.push(step)?;
            if matches {
                accepted += 1;
            }
            if terminal || !matches {
                return Ok((builder.len, accepted));
            }
        }
        if session.control.finish_reason().is_none() {
            let row = target_download_row(lane * (extent + 1) + extent);
            builder.push(session.control.accept_logits(&self.target_logits[row])?)?;
        }
        Ok((builder.len, accepted))
    }

    fn decide_sampled(
        &mut self,
        slot: usize,
        lane: usize,
        extent: usize,
        drafts: &mut LaneDrafts,
        builder: &mut EventBuilder,
    ) -> EngineResult<(usize, usize)> {
        let tokens = extent + 1;
        let session = self.sessions[slot]
            .as_mut()
            .expect("sampled Qwen3.5 MTP session exists");
        let mut target_laws = Vec::with_capacity(tokens);
        for row in 0..tokens {
            let logits = target_download_row(lane * tokens + row);
            target_laws.push(session.control.sampling_distribution(
                &self.target_logits[logits],
                &drafts.tokens[..row.min(extent)],
            )?);
        }
        let mut acceptance_units = [0.0f64; DRAFT_WINDOW];
        let mut residual_units = [0.0f64; DRAFT_WINDOW];
        for row in 0..extent {
            acceptance_units[row] = session.control.random_unit();
            residual_units[row] = session.control.random_unit();
        }
        let bonus_unit = session.control.random_unit();
        let target_refs = target_laws.iter().collect::<Vec<_>>();
        let draft_refs = drafts.laws[..extent]
            .iter()
            .enumerate()
            .map(|(row, law)| {
                law.as_ref().ok_or_else(|| {
                    EngineError::generation(format!(
                        "sampled Qwen3.5 MTP proposal {row} has no distribution"
                    ))
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let round = decide_sampled_tokens(
            &drafts.tokens[..extent],
            &target_refs,
            &draft_refs,
            &self.stop_ids,
            &acceptance_units[..extent],
            &residual_units[..extent],
            bonus_unit,
        )?;
        for &token in round.token_ids() {
            builder.push(session.control.accept_token(token)?)?;
        }
        Ok((round.token_ids().len(), round.accepted_drafts()))
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_and_realign_lane(
        &mut self,
        slot: usize,
        lane: usize,
        inputs: &[u32],
        builder: &EventBuilder,
        committed: usize,
        accepted: usize,
    ) -> EngineResult<()> {
        if committed == 0 || committed > inputs.len() {
            return Err(EngineError::generation(format!(
                "Qwen3.5 MTP lane commits {committed} rows from K={}",
                inputs.len()
            )));
        }
        let first_position = self.sessions[slot]
            .as_ref()
            .expect("committed Qwen3.5 MTP session exists")
            .next_position;
        if committed != inputs.len() {
            self.program.target().restore_gdn_slot(&self.stream, slot)?;
            let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
            let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
            let rotary = fill_contiguous_rope(first_position, committed, &mut cosine, &mut sine)?;
            self.program.stage_target_verify(
                &self.stream,
                &inputs[..committed],
                slot,
                first_position,
                &cosine[..rotary],
                &sine[..rotary],
            )?;
            self.program.replay_target_verify(&self.stream, committed)?;
            let hidden = lane_target_hidden(lane, committed);
            self.program.target().read_final_residual_into(
                &self.stream,
                committed,
                &mut self.target_hidden[hidden],
            )?;
        }

        let hidden_source = target_hidden_row(lane, committed - 1);
        self.target_boundary_hidden[hidden_slot(slot)]
            .copy_from_slice(&self.target_hidden[hidden_source]);
        let logit_source = target_download_row(lane * inputs.len() + committed - 1);
        self.target_logits
            .copy_within(logit_source, slot * Qwen35_9B::VOCAB);
        let outputs = builder.token_ids().collect::<Vec<_>>();
        if outputs.len() != committed {
            return Err(EngineError::generation(format!(
                "Qwen3.5 MTP lane has {} outputs for {committed} committed rows",
                outputs.len()
            )));
        }
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary = fill_contiguous_rope(first_position, committed, &mut cosine, &mut sine)?;
        let hidden = lane_target_hidden(lane, committed);
        self.program.stage_realign(
            &self.stream,
            &outputs,
            &self.target_hidden[hidden],
            slot,
            first_position,
            &cosine[..rotary],
            &sine[..rotary],
        )?;
        let terminal = self.sessions[slot]
            .as_ref()
            .expect("committed Qwen3.5 MTP session exists")
            .control
            .finish_reason()
            .is_some();
        self.program
            .replay_realign(&self.stream, committed, terminal)?;
        if !terminal {
            self.program.read_logits_into(
                &self.stream,
                1,
                &mut self.draft_logits[draft_slot_logits(slot)],
            )?;
            self.program.read_mtp_residuals_into(
                &self.stream,
                committed,
                &mut self.draft_hidden[compact_hidden(committed)],
            )?;
            let source = compact_hidden_row(committed - 1);
            self.draft_hidden
                .copy_within(source, slot * Qwen35_9B::HIDDEN);
        }
        let session = self.sessions[slot]
            .as_mut()
            .expect("committed Qwen3.5 MTP session exists");
        session.next_position = session
            .next_position
            .checked_add(committed)
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP position overflows"))?;
        session.proposal_ready = !terminal;
        session.stats.verification_routes[inputs.len() - 1] += 1;
        session.stats.accepted_drafts += accepted;
        session.stats.verified_outputs += committed;
        Ok(())
    }

    fn anchor(&self, slot: usize) -> EngineResult<u32> {
        self.sessions[slot]
            .as_ref()
            .and_then(|session| session.control.generated_token_ids().last().copied())
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP lane has no anchor"))
    }

    fn finish_events(
        &mut self,
        active_slots: &[usize],
        mut builders: [EventBuilder; MAX_BATCH],
    ) -> EngineResult<Qwen35ResidentMtpBatchEvents> {
        let mut events = std::array::from_fn(|_| None);
        let mut survivors = [usize::MAX; MAX_BATCH];
        let mut surviving = 0;
        for (index, &slot) in active_slots.iter().enumerate() {
            let builder = &mut builders[slot];
            if builder.len == 0 {
                return Err(EngineError::generation(format!(
                    "Qwen3.5 MTP slot {slot} produced no event"
                )));
            }
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("completed Qwen3.5 MTP slot owns a session")
                .request_id;
            let terminal = self.sessions[slot]
                .as_ref()
                .expect("completed Qwen3.5 MTP slot owns a session")
                .control
                .finish_reason()
                .is_some();
            let stats = self.sessions[slot]
                .as_ref()
                .expect("completed Qwen3.5 MTP slot owns a session")
                .stats;
            let completed = if terminal {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal Qwen3.5 MTP session exists");
                self.program.recycle_kv_slot(&self.stream, slot)?;
                Some(session.control.into_output()?)
            } else {
                survivors[surviving] = slot;
                surviving += 1;
                None
            };
            events[index] = Some(Qwen35ResidentMtpBatchEvent {
                request_id,
                steps: std::mem::replace(&mut builder.steps, std::array::from_fn(|_| None)),
                len: builder.len,
                completed,
                stats,
            });
        }
        self.active_slots = survivors;
        self.active = surviving;
        Ok(Qwen35ResidentMtpBatchEvents {
            events,
            len: active_slots.len(),
        })
    }
}

fn pinned_rows(
    context: &Arc<CudaContext>,
    label: &str,
    rows: usize,
    columns: usize,
) -> EngineResult<PinnedHostBuffer<u16>> {
    let values = rows
        .checked_mul(columns)
        .ok_or_else(|| EngineError::layout(format!("{label} overflows")))?;
    PinnedHostBuffer::zeroed(context, values)
        .map_err(GpuError::from)
        .map_err(Into::into)
}

fn first_free_slot(occupied: [bool; MAX_BATCH]) -> Option<usize> {
    occupied.iter().position(|&occupied| !occupied)
}

const fn round_route(active: usize) -> Option<RoundRoute> {
    match active {
        0 => None,
        1 => Some(RoundRoute::Mtp),
        2..=MAX_BATCH => Some(RoundRoute::CompactTarget),
        _ => None,
    }
}

fn target_slot_logits(slot: usize) -> Range<usize> {
    let begin = slot * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

#[cfg(test)]
fn target_download_logits(rows: usize) -> Range<usize> {
    let begin = MAX_BATCH * Qwen35_9B::VOCAB;
    begin..begin + rows * Qwen35_9B::VOCAB
}

fn target_download_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

fn lane_target_logits(lane: usize, rows: usize) -> Range<usize> {
    let begin = target_download_row(lane * rows).start;
    begin..begin + rows * Qwen35_9B::VOCAB
}

fn draft_slot_logits(slot: usize) -> Range<usize> {
    let begin = slot * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

fn compact_draft_logits(rows: usize) -> Range<usize> {
    let begin = MAX_BATCH * Qwen35_9B::VOCAB;
    begin..begin + rows * Qwen35_9B::VOCAB
}

fn compact_draft_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

fn hidden_slot(slot: usize) -> Range<usize> {
    let begin = slot * Qwen35_9B::HIDDEN;
    begin..begin + Qwen35_9B::HIDDEN
}

fn compact_hidden(rows: usize) -> Range<usize> {
    MAX_BATCH * Qwen35_9B::HIDDEN..(MAX_BATCH + rows) * Qwen35_9B::HIDDEN
}

fn compact_hidden_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen35_9B::HIDDEN;
    begin..begin + Qwen35_9B::HIDDEN
}

fn lane_target_hidden(lane: usize, rows: usize) -> Range<usize> {
    let begin = lane * VERIFY_ROWS * Qwen35_9B::HIDDEN;
    begin..begin + rows * Qwen35_9B::HIDDEN
}

fn target_hidden_row(lane: usize, row: usize) -> Range<usize> {
    let begin = (lane * VERIFY_ROWS + row) * Qwen35_9B::HIDDEN;
    begin..begin + Qwen35_9B::HIDDEN
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, RoundRoute, first_free_slot, round_route, target_download_logits};
    use tuisko_model::Arch;

    #[test]
    fn physical_slot_selection_uses_the_first_hole() {
        let cases = [
            ([false; MAX_BATCH], Some(0)),
            (
                [true, false, true, false, true, false, true, false],
                Some(1),
            ),
            ([true, true, true, true, true, true, true, false], Some(7)),
            ([true; MAX_BATCH], None),
        ];
        for (occupied, expected) in cases {
            assert_eq!(first_free_slot(occupied), expected, "{occupied:?}");
        }
    }

    #[test]
    fn target_download_bank_covers_all_compact_verification_rows() {
        for rows in 1..=MAX_BATCH * 4 {
            let range = target_download_logits(rows);
            assert_eq!(range.len(), rows * tuisko_model::Qwen35_9B::VOCAB);
        }
    }

    #[test]
    fn round_routing_keeps_mtp_singleton_and_compact_concurrency_exact() {
        assert_eq!(round_route(0), None);
        assert_eq!(round_route(1), Some(RoundRoute::Mtp));
        for active in 2..=MAX_BATCH {
            assert_eq!(round_route(active), Some(RoundRoute::CompactTarget));
        }
        assert_eq!(round_route(MAX_BATCH + 1), None);
    }
}
