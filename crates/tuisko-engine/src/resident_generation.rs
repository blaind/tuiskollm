//! Single-slot text generation over the exact resident model owner.

use crate::{
    CancelledText, ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, MAX_BATCH, ResidentLoadProgress, ResidentModelProgram,
};
#[cfg(feature = "qualification")]
use crate::{Sampler, SamplingOptions};
use std::sync::Arc;
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const ROPE_THETA: f64 = 10_000_000.0;
const LOGIT_BANK_ROWS: usize = 2 * MAX_BATCH;
const NATIVE_PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1024];
const MAX_NATIVE_PREFILL_TOKENS: usize = 1024;

/// Concrete frontend, device program, stream, and host-logit owner for one active request.
pub struct ResidentTextGenerator {
    frontend: TextFrontend,
    program: ResidentModelProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
}

/// Concrete compact-batch owner for up to eight concurrent text requests.
pub struct ResidentBatchGenerator {
    frontend: TextFrontend,
    program: ResidentModelProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    sessions: [Option<ResidentBatchSession>; MAX_BATCH],
    retained: [Option<RetainedSlot>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
    retention_clock: u64,
}

/// Stable request identity assigned by the compact scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResidentRequestId(u64);

/// Admission result, including an immediate zero-token completion when requested.
pub struct ResidentBatchAdmission {
    /// Scheduler-assigned request identity.
    pub request_id: ResidentRequestId,
    /// Exact rendered prompt length in tokens.
    pub prompt_tokens: usize,
    /// Prompt tokens restored from an exact retained device-state prefix.
    pub device_reused_tokens: usize,
    /// Prompt tokens processed by one exact from-empty whole-model prefill graph.
    pub native_prefill_tokens: usize,
    /// Immediate output for a request with `max_new_tokens == 0`.
    pub completed: Option<GeneratedText>,
}

/// One streamed scheduler event and its optional complete terminal output.
pub struct ResidentBatchEvent {
    /// Request that produced this event.
    pub request_id: ResidentRequestId,
    /// Newly selected token, streaming delta, and terminal boundary.
    pub step: GenerationStep,
    /// Complete output when `step` finished the request.
    pub completed: Option<GeneratedText>,
}

/// Observable state returned when an active resident request is cancelled.
pub struct ResidentCancellation {
    /// Cancelled scheduler request.
    pub request_id: ResidentRequestId,
    /// Tokens and text emitted before cancellation.
    pub output: CancelledText,
    /// Processed prefix tokens whose device state remains retained.
    pub device_retained_tokens: usize,
}

/// At most eight events returned in the scheduler's stable active order.
pub struct ResidentBatchEvents {
    events: [Option<ResidentBatchEvent>; MAX_BATCH],
    len: usize,
}

struct ResidentBatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    pending_token: Option<u32>,
    next_position: u32,
}

struct RetainedSlot {
    tokens: Vec<u32>,
    last_used: u64,
}

/// One streaming generation request borrowing the single resident slot.
pub struct ResidentGenerationSession<'a> {
    control: GenerationSession,
    program: &'a mut ResidentModelProgram,
    stream: &'a CudaStream,
    logits: &'a mut [u16],
    pending_token: Option<u32>,
    next_position: u32,
    native_prefill_tokens: usize,
}

impl ResidentRequestId {
    /// Numeric identity suitable for transport correlation and logs.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl ResidentBatchEvents {
    /// Number of requests that produced an event in this scheduler round.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this scheduler round produced no events.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Events in the active order that existed at the start of the round.
    pub fn iter(&self) -> impl Iterator<Item = &ResidentBatchEvent> {
        self.events[..self.len]
            .iter()
            .map(|event| event.as_ref().expect("active event prefix is initialized"))
    }
}

impl ResidentTextGenerator {
    /// Admits the pinned frontend and loads the exact resident model into `context`.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let program = ResidentModelProgram::from_snapshot(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logits =
            PinnedHostBuffer::zeroed(context, Qwen38_27B::VOCAB).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
        })
    }

    /// Renders one request and primes its prompt through an exact resident graph.
    pub fn start<'a>(
        &'a mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentGenerationSession<'a>> {
        let Self {
            frontend,
            program,
            stream,
            logits,
        } = self;
        let control = GenerationSession::start(frontend, request)?;
        let required_positions = require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            program.context_capacity(),
        )?;
        let mut native_prefill_tokens = 0;

        if control.finish_reason().is_none() {
            program.recycle_kv_slot(stream, 0)?;
            program.reset_slot(stream, 0)?;
            program.activate_kv_slot(0)?;
            program.reserve_kv_slot_tokens(stream, 0, required_positions)?;
            program.load_slot_routes(stream, &[0])?;
            native_prefill_tokens =
                prime_prompt(program, stream, control.prompt_token_ids(), 0, 0)?;
            program.read_logits_into(stream, 1, logits)?;
        }
        let next_position = u32::try_from(control.prompt_token_ids().len())
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;

        Ok(ResidentGenerationSession {
            control,
            program,
            stream,
            logits,
            pending_token: None,
            next_position,
            native_prefill_tokens,
        })
    }

    /// Exact device bytes owned by the resident model arena.
    pub const fn arena_bytes(&self) -> usize {
        self.program.arena_bytes()
    }

    /// Page-locked embedding and logit staging bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    /// Fixed host bytes owning shared page tables and physical-page tags.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program.kv_route_host_bytes()
    }

    /// Current short-context token capacity per single slot.
    pub const fn context_capacity(&self) -> usize {
        self.program.context_capacity()
    }

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Runs an independently captured raw-token reference case through the production path.
    pub fn qualification_greedy_after_tokens(&mut self, token_ids: &[u32]) -> EngineResult<u32> {
        self.qualification_greedy_after_tokens_with_route(token_ids)
            .map(|(token, _)| token)
    }

    #[cfg(feature = "qualification")]
    /// Runs raw tokens and reports whether the production path selected native prefill.
    pub fn qualification_greedy_after_tokens_with_route(
        &mut self,
        token_ids: &[u32],
    ) -> EngineResult<(u32, usize)> {
        let required_positions =
            require_generation_capacity(token_ids.len(), 1, self.program.context_capacity())?;
        self.program.recycle_kv_slot(&self.stream, 0)?;
        self.program.reset_slot(&self.stream, 0)?;
        self.program.activate_kv_slot(0)?;
        self.program
            .reserve_kv_slot_tokens(&self.stream, 0, required_positions)?;
        self.program.load_slot_routes(&self.stream, &[0])?;
        let native_prefill_tokens = prime_prompt(&mut self.program, &self.stream, token_ids, 0, 0)?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;
        let stop_ids: [u32; 2] =
            self.frontend.stop_ids().try_into().map_err(|_| {
                EngineError::generation("frontend returned the wrong stop-ID count")
            })?;
        let mut sampler = Sampler::new(SamplingOptions::greedy(), stop_ids)?;
        Ok((
            sampler.sample(&self.logits)?.token_id,
            native_prefill_tokens,
        ))
    }

    #[cfg(feature = "qualification")]
    /// Stable device-arena and pinned-logit addresses owned by this generator.
    pub fn qualification_addresses(&self) -> [usize; 3] {
        [
            self.program.base_address() as usize,
            self.program.kv_base_address() as usize,
            self.logits.as_ptr().addr(),
        ]
    }
}

impl ResidentBatchGenerator {
    /// Opens the exact resident scheduler on device zero and refuses any non-SM120 device.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Opens device zero while publishing resident startup counters.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot_inner(&context, snapshot, Some(progress))
    }

    /// Admits the pinned frontend and complete resident program for compact B=1..8 decoding.
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
        let program = match progress {
            Some(progress) => {
                ResidentModelProgram::from_snapshot_with_progress(context, snapshot, progress)?
            }
            None => ResidentModelProgram::from_snapshot(context, snapshot)?,
        };
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen38_27B::VOCAB
            .checked_mul(LOGIT_BANK_ROWS)
            .ok_or_else(|| EngineError::layout("resident batch logit banks overflow"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            sessions: std::array::from_fn(|_| None),
            retained: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
            retention_clock: 0,
        })
    }

    /// Admits one request, restoring only an exact processed device prefix when available.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        let control = GenerationSession::start(&self.frontend, request)?;
        let required_positions = require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            self.program.context_capacity(),
        )?;
        let request_id = ResidentRequestId(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident request identity overflows"))?;
        let prompt_tokens = control.prompt_token_ids().len();
        if control.finish_reason().is_some() {
            return Ok(ResidentBatchAdmission {
                request_id,
                prompt_tokens,
                device_reused_tokens: 0,
                native_prefill_tokens: 0,
                completed: Some(control.into_output()?),
            });
        }
        let (slot, device_reused_tokens, reset) = self.select_slot(control.prompt_token_ids())?;
        if reset {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            self.program.reset_slot(&self.stream, slot)?;
            self.program.activate_kv_slot(slot)?;
        } else {
            self.program.activate_kv_slot(slot)?;
        }
        self.program
            .reserve_kv_slot_tokens(&self.stream, slot, required_positions)?;
        self.program.load_slot_routes(&self.stream, &[slot])?;
        let native_prefill_tokens = prime_prompt(
            &mut self.program,
            &self.stream,
            control.prompt_token_ids(),
            slot,
            device_reused_tokens,
        )?;
        if device_reused_tokens < prompt_tokens {
            let logits = slot_logits(slot);
            self.program
                .read_logits_into(&self.stream, 1, &mut self.logits[logits])?;
        }
        let next_position = u32::try_from(prompt_tokens)
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        self.sessions[slot] = Some(ResidentBatchSession {
            request_id,
            control,
            pending_token: None,
            next_position,
        });
        self.active_slots[self.active] = slot;
        self.active += 1;

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens,
            native_prefill_tokens,
            completed: None,
        })
    }

    /// Advances pending tokens as one compact batch, then samples one event per active request.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty resident generation scheduler",
            ));
        }
        self.replay_pending()?;

        let mut events = std::array::from_fn(|_| None);
        let mut survivors = [usize::MAX; MAX_BATCH];
        let mut surviving = 0;
        let active = self.active;
        for (index, event) in events[..active].iter_mut().enumerate() {
            let slot = self.active_slots[index];
            let logits = slot_logits(slot);
            let step = {
                let session = self.sessions[slot].as_mut().ok_or_else(|| {
                    EngineError::generation("active resident slot has no generation session")
                })?;
                let step = session.control.accept_logits(&self.logits[logits])?;
                if step.finish_reason.is_none() {
                    session.pending_token = Some(step.token_id);
                }
                step
            };
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("active resident session survived sampling")
                .request_id;
            let completed = if step.finish_reason.is_some() {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal resident session exists");
                let retained = processed_tokens(&session)?;
                self.store_retained(slot, retained)?;
                Some(session.control.into_output()?)
            } else {
                survivors[surviving] = slot;
                surviving += 1;
                None
            };
            *event = Some(ResidentBatchEvent {
                request_id,
                step,
                completed,
            });
        }
        self.active_slots = survivors;
        self.active = surviving;

        Ok(ResidentBatchEvents {
            events,
            len: active,
        })
    }

    /// Cancels one active request while retaining its fully processed device prefix.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| {
                EngineError::generation("resident cancellation request is not active")
            })?;
        let slot = self.active_slots[index];
        let session = self.sessions[slot]
            .take()
            .expect("cancelled resident slot owns a session");
        let retained = processed_tokens(&session)?;
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

    /// Current number of active device-backed requests.
    pub const fn active_requests(&self) -> usize {
        self.active
    }

    /// Active request identities in compact scheduler order.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active_slots[..self.active].iter().map(|&slot| {
            self.sessions[slot]
                .as_ref()
                .expect("active resident slot owns a session")
                .request_id
        })
    }

    /// Exact device bytes owned by the shared resident model arena.
    pub const fn arena_bytes(&self) -> usize {
        self.program.arena_bytes()
    }

    /// Page-locked embedding staging plus slot and compact-download logit banks.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    /// Fixed host bytes owning shared page tables and physical-page tags.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program.kv_route_host_bytes()
    }

    /// Current short-context token capacity per physical slot.
    pub const fn context_capacity(&self) -> usize {
        self.program.context_capacity()
    }

    /// CUDA context shared by all slots, exact graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    /// Exact loading work used to construct the shared resident program.
    pub const fn load_stats(&self) -> crate::ResidentLoadStats {
        self.program.load_stats()
    }

    #[cfg(feature = "qualification")]
    /// Stable device-arena and pinned-logit addresses owned by this scheduler.
    pub fn qualification_addresses(&self) -> [usize; 3] {
        [
            self.program.base_address() as usize,
            self.program.kv_base_address() as usize,
            self.logits.as_ptr().addr(),
        ]
    }

    #[cfg(feature = "qualification")]
    /// Physical slot currently retaining `request_id`, when active.
    pub fn qualification_slot(&self, request_id: ResidentRequestId) -> Option<usize> {
        self.sessions.iter().position(|session| {
            session
                .as_ref()
                .is_some_and(|session| session.request_id == request_id)
        })
    }

    #[cfg(feature = "qualification")]
    /// Number of exact processed tokens retained in one inactive physical slot.
    pub fn qualification_retained_tokens(&self, slot: usize) -> Option<usize> {
        self.retained
            .get(slot)
            .and_then(Option::as_ref)
            .map(|retained| retained.tokens.len())
    }

    #[cfg(feature = "qualification")]
    /// Drops host retention metadata so an independent scheduler fixture starts cold.
    pub fn qualification_clear_retained(&mut self) -> EngineResult<()> {
        for slot in 0..MAX_BATCH {
            if self.retained[slot].take().is_some() {
                self.program.recycle_kv_slot(&self.stream, slot)?;
            }
        }
        Ok(())
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
            .ok_or_else(|| EngineError::route("all eight resident generation slots are active"))?;
        self.retained[eviction] = None;
        Ok((eviction, 0, true))
    }

    fn store_retained(&mut self, slot: usize, tokens: Vec<u32>) -> EngineResult<()> {
        let next_clock = self
            .retention_clock
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident retention clock overflows"))?;
        self.program
            .truncate_kv_slot_tokens(&self.stream, slot, tokens.len())?;
        self.program.retain_kv_slot(slot)?;
        self.retention_clock = next_clock;
        self.retained[slot] = Some(RetainedSlot {
            tokens,
            last_used: self.retention_clock,
        });
        Ok(())
    }

    fn replay_pending(&mut self) -> EngineResult<()> {
        let mut slots = [0usize; MAX_BATCH];
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        let mut rope_cos = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut rope_sin = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut pending = 0;
        // The exact B route contains only previously emitted tokens: fresh
        // admissions already own logits, and cancellation must leave them unadvanced.
        for &slot in &self.active_slots[..self.active] {
            let session = self.sessions[slot].as_ref().ok_or_else(|| {
                EngineError::generation("active resident slot has no pending session")
            })?;
            let Some(token) = session.pending_token else {
                continue;
            };
            slots[pending] = slot;
            tokens[pending] = token;
            positions[pending] = session.next_position;
            let (cosine, sine) = text_rope(session.next_position);
            let begin = pending * ROTARY_PAIRS;
            rope_cos[begin..begin + ROTARY_PAIRS].copy_from_slice(&cosine);
            rope_sin[begin..begin + ROTARY_PAIRS].copy_from_slice(&sine);
            pending += 1;
        }
        if pending == 0 {
            return Ok(());
        }

        self.program
            .stage_embeddings(&self.stream, &tokens[..pending])?;
        self.program
            .load_slot_routes(&self.stream, &slots[..pending])?;
        let route = self.program.load_decode_state(
            &self.stream,
            pending,
            &positions[..pending],
            &rope_cos[..pending * ROTARY_PAIRS],
            &rope_sin[..pending * ROTARY_PAIRS],
        )?;
        self.program.replay(&self.stream, route)?;
        let download = compact_download_logits(pending);
        self.program
            .read_logits_into(&self.stream, pending, &mut self.logits[download])?;
        for (row, &slot) in slots[..pending].iter().enumerate() {
            let source = compact_download_row(row);
            let destination = slot * Qwen38_27B::VOCAB;
            self.logits.copy_within(source, destination);
            let session = self.sessions[slot]
                .as_mut()
                .expect("pending resident slot owns a session");
            session.pending_token = None;
            session.next_position = session
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }
        Ok(())
    }
}

fn device_zero_context() -> EngineResult<Arc<CudaContext>> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(EngineError::route(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    Ok(context)
}

impl ResidentGenerationSession<'_> {
    /// Exact prompt token IDs selected by the admitted frontend.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.control.prompt_token_ids()
    }

    /// Tokens selected so far, including an unprocessed final token.
    pub fn generated_token_ids(&self) -> &[u32] {
        self.control.generated_token_ids()
    }

    /// Prompt tokens processed by one exact from-empty whole-model prefill graph.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.native_prefill_tokens
    }

    /// Current terminal state.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.control.finish_reason()
    }

    /// Samples one token and returns its streaming delta before the next model replay.
    pub fn step(&mut self) -> EngineResult<GenerationStep> {
        if let Some(token) = self.pending_token.take() {
            replay_token(self.program, self.stream, token, self.next_position)?;
            self.program.read_logits_into(self.stream, 1, self.logits)?;
            self.next_position = self
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        let step = self.control.accept_logits(self.logits)?;
        if step.finish_reason.is_none() {
            self.pending_token = Some(step.token_id);
        }
        Ok(step)
    }

    /// Converts a terminal session into its complete decoded result.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        self.control.into_output()
    }
}

fn slot_logits(slot: usize) -> std::ops::Range<usize> {
    let begin = slot * Qwen38_27B::VOCAB;
    begin..begin + Qwen38_27B::VOCAB
}

fn compact_download_logits(rows: usize) -> std::ops::Range<usize> {
    let begin = MAX_BATCH * Qwen38_27B::VOCAB;
    begin..begin + rows * Qwen38_27B::VOCAB
}

fn compact_download_row(row: usize) -> std::ops::Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen38_27B::VOCAB;
    begin..begin + Qwen38_27B::VOCAB
}

fn processed_tokens(session: &ResidentBatchSession) -> EngineResult<Vec<u32>> {
    let processed = usize::try_from(session.next_position)
        .map_err(|_| EngineError::generation("processed position exceeds host width"))?;
    let prompt = session.control.prompt_token_ids();
    let generated = session.control.generated_token_ids();
    let processed_generated = processed.checked_sub(prompt.len()).ok_or_else(|| {
        EngineError::generation("resident processed position precedes its prompt")
    })?;
    if processed_generated > generated.len() {
        return Err(EngineError::generation(
            "resident processed position exceeds emitted generation",
        ));
    }

    let mut tokens = Vec::with_capacity(processed);
    tokens.extend_from_slice(prompt);
    tokens.extend_from_slice(&generated[..processed_generated]);
    Ok(tokens)
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

fn prime_prompt(
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
    if native_prefill_tokens(token_ids.len(), processed_prefix).is_some() {
        let rotary_values = token_ids
            .len()
            .checked_mul(ROTARY_PAIRS)
            .ok_or_else(|| EngineError::generation("prompt rotary values overflow"))?;
        let mut rope_cos = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
        let mut rope_sin = [0.0f32; MAX_NATIVE_PREFILL_TOKENS * ROTARY_PAIRS];
        for position in 0..token_ids.len() {
            let position = u32::try_from(position).map_err(|_| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?;
            let (cosine, sine) = text_rope(position);
            let begin = position as usize * ROTARY_PAIRS;
            rope_cos[begin..begin + ROTARY_PAIRS].copy_from_slice(&cosine);
            rope_sin[begin..begin + ROTARY_PAIRS].copy_from_slice(&sine);
        }
        program.stage_embeddings(stream, token_ids)?;
        let route = program.load_prefill_state(
            stream,
            token_ids.len(),
            slot,
            &rope_cos[..rotary_values],
            &rope_sin[..rotary_values],
        )?;
        program.replay_prefill(stream, route)?;
        return Ok(token_ids.len());
    }

    for (offset, &token) in token_ids[processed_prefix..].iter().enumerate() {
        let position = processed_prefix + offset;
        replay_token(
            program,
            stream,
            token,
            u32::try_from(position).map_err(|_| {
                EngineError::generation("prompt position exceeds the exact route width")
            })?,
        )?;
    }
    Ok(0)
}

const fn native_prefill_tokens(prompt_tokens: usize, processed_prefix: usize) -> Option<usize> {
    if processed_prefix != 0 {
        return None;
    }
    let mut index = 0;
    while index < NATIVE_PREFILL_ROUTES.len() {
        if NATIVE_PREFILL_ROUTES[index] == prompt_tokens {
            return Some(prompt_tokens);
        }
        index += 1;
    }
    None
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

fn require_generation_capacity(
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    context_capacity: usize,
) -> EngineResult<usize> {
    if prompt_tokens == 0 {
        return Err(EngineError::generation(
            "resident generation requires a nonempty prompt",
        ));
    }
    let evaluated = prompt_tokens
        .checked_add(maximum_new_tokens.saturating_sub(1))
        .ok_or_else(|| EngineError::generation("generation token budget overflows"))?;
    if evaluated > context_capacity {
        return Err(EngineError::generation(format!(
            "prompt plus processed generation requires {evaluated} positions, current resident capacity is {context_capacity}"
        )));
    }
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use super::{
        NATIVE_PREFILL_ROUTES, ROTARY_PAIRS, native_prefill_tokens, require_generation_capacity,
        text_rope,
    };
    use crate::EngineErrorCode;

    #[test]
    fn long_context_capacity_counts_only_processed_generated_tokens() {
        require_generation_capacity(220_000, 1, 220_000).unwrap();
        require_generation_capacity(1, 220_000, 220_000).unwrap();
        require_generation_capacity(220_000, 0, 220_000).unwrap();

        for (prompt, generated) in [(0, 1), (220_000, 2), (2, 220_000), (usize::MAX, 2)] {
            assert_eq!(
                require_generation_capacity(prompt, generated, 220_000)
                    .unwrap_err()
                    .code(),
                Some(EngineErrorCode::Generation)
            );
        }
    }

    #[test]
    fn text_rope_uses_the_checkpoint_theta_and_64_wide_pairing() {
        let (zero_cos, zero_sin) = text_rope(0);
        assert_eq!(zero_cos, [1.0; ROTARY_PAIRS]);
        assert_eq!(zero_sin, [0.0; ROTARY_PAIRS]);

        let (cosine, sine) = text_rope(130);
        let frequency = 10_000_000.0f64.powf(-62.0 / 64.0);
        let angle = 130.0 * frequency;
        assert_eq!(cosine[0].to_bits(), (130.0f64.cos() as f32).to_bits());
        assert_eq!(sine[0].to_bits(), (130.0f64.sin() as f32).to_bits());
        assert_eq!(cosine[31].to_bits(), (angle.cos() as f32).to_bits());
        assert_eq!(sine[31].to_bits(), (angle.sin() as f32).to_bits());
    }

    #[test]
    fn native_prefill_inventory_is_exact_and_from_empty_only() {
        for tokens in NATIVE_PREFILL_ROUTES {
            assert_eq!(native_prefill_tokens(tokens, 0), Some(tokens));
            assert_eq!(native_prefill_tokens(tokens, 1), None);
            assert_eq!(native_prefill_tokens(tokens, tokens), None);
            assert_eq!(native_prefill_tokens(tokens - 1, 0), None);
            assert_eq!(native_prefill_tokens(tokens + 1, 0), None);
        }
        for tokens in [0, 1, 31, 33, 63, 65, 127, 129, 1023, 1025, 220_000] {
            assert_eq!(native_prefill_tokens(tokens, 0), None);
        }
    }
}
