//! Compact Qwen3.8 text generation over eight physical persistent-state slots.

use crate::common::banks::{compact, row};
use crate::common::rope::{ROTARY_PAIRS, text_rope};
use crate::common::slots::{device_zero_context, require_generation_capacity};
use crate::qwen38::text_generation::prime_prompt;
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GenerationSession, MAX_BATCH,
    ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents, ResidentCancellation,
    ResidentLoadProgress, ResidentModelProgram, ResidentRequestId,
};
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const LOGIT_BANK_ROWS: usize = 2 * MAX_BATCH;

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
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("resident request identity overflows"))?;
        let prompt_tokens = control.prompt_token_ids().len();
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
        let (slot, device_reused_tokens, reset) = self.select_slot(control.prompt_token_ids())?;
        if reset {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            self.program.reset_slot(&self.stream, slot)?;
        }
        self.program.activate_kv_slot(slot)?;
        if let Err(error) =
            self.program
                .reserve_kv_slot_tokens(&self.stream, slot, required_positions)
        {
            // No replay has happened yet, so the retained prefix is still exact.
            if reset {
                self.program.recycle_kv_slot(&self.stream, slot)?;
            } else {
                self.program
                    .truncate_kv_slot_tokens(&self.stream, slot, device_reused_tokens)?;
                self.program.retain_kv_slot(slot)?;
            }
            return Err(error);
        }
        self.retained[slot] = None;
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
            prompt_metrics,
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

        Ok(ResidentBatchEvents::from_events(events, active))
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

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
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

fn slot_logits(slot: usize) -> std::ops::Range<usize> {
    row(slot, Qwen38_27B::VOCAB)
}

fn compact_download_logits(rows: usize) -> std::ops::Range<usize> {
    compact(rows, Qwen38_27B::VOCAB)
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
