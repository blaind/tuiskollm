//! Compact Qwen3.5 text generation over eight physical persistent-state slots.

use crate::common::rope::text_rope;
use crate::resident_generation::{
    device_zero_context, prime_qwen35_prompt, require_generation_capacity,
};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GenerationSession, MAX_BATCH,
    Qwen35ResidentModelProgram, ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents,
    ResidentCancellation, ResidentRequestId,
};
use std::ops::Range;
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const LOGIT_BANK_ROWS: usize = 2 * MAX_BATCH;
const ROTARY_PAIRS: usize = 32;

/// Concrete Qwen3.5 scheduler for up to eight compact decode rows.
pub struct Qwen35ResidentBatchGenerator {
    frontend: TextFrontend,
    program: Qwen35ResidentModelProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    sessions: [Option<Qwen35BatchSession>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
}

struct Qwen35BatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    pending_token: Option<u32>,
    next_position: u32,
}

impl Qwen35ResidentBatchGenerator {
    /// Opens the exact Qwen3.5 compact scheduler on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Loads one resident program shared by eight persistent request slots.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
        let program = Qwen35ResidentModelProgram::from_snapshot(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen35_9B::VOCAB
            .checked_mul(LOGIT_BANK_ROWS)
            .ok_or_else(|| EngineError::layout("Qwen3.5 compact logit banks overflow"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            sessions: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
        })
    }

    /// Admits one request into the first free physical state/cache slot.
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
            .ok_or_else(|| EngineError::generation("Qwen3.5 request identity overflows"))?;
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

        let occupied = std::array::from_fn(|slot| self.sessions[slot].is_some());
        let slot = first_free_slot(occupied)
            .ok_or_else(|| EngineError::route("all eight Qwen3.5 generation slots are active"))?;
        self.program.reset_slot(&self.stream, slot)?;
        self.program.activate_kv_slot(slot)?;
        if let Err(error) =
            self.program
                .reserve_kv_slot_tokens(&self.stream, slot, required_positions)
        {
            self.program.recycle_kv_slot(&self.stream, slot)?;
            return Err(error);
        }
        let native_prefill_tokens = prime_qwen35_prompt(
            &mut self.program,
            &self.stream,
            control.prompt_token_ids(),
            slot,
        )?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits[slot_logits(slot)])?;
        let next_position = u32::try_from(prompt_tokens)
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        self.sessions[slot] = Some(Qwen35BatchSession {
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
            device_reused_tokens: 0,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    /// Replays pending tokens compactly, then samples one event per active request.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty Qwen3.5 generation scheduler",
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
                    EngineError::generation("active Qwen3.5 slot has no generation session")
                })?;
                let step = session.control.accept_logits(&self.logits[logits])?;
                if step.finish_reason.is_none() {
                    session.pending_token = Some(step.token_id);
                }
                step
            };
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("active Qwen3.5 session survived sampling")
                .request_id;
            let completed = if step.finish_reason.is_some() {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal Qwen3.5 session exists");
                self.program.recycle_kv_slot(&self.stream, slot)?;
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

    /// Cancels an active request at a scheduler-round boundary.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| EngineError::generation("Qwen3.5 cancellation request is not active"))?;
        let slot = self.active_slots[index];
        let session = self.sessions[slot]
            .take()
            .expect("cancelled Qwen3.5 slot owns a session");
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
                .expect("active Qwen3.5 slot owns a session")
                .request_id
        })
    }

    /// Exact bytes across all resident Qwen3.5 arenas.
    pub const fn arena_bytes(&self) -> usize {
        self.program.layout().arena_bytes()
    }

    /// Source-backed Qwen3.5 weights resident on the device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.program.layout().resident_weight_bytes()
    }

    /// Page-locked embedding staging plus physical and compact logit banks.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    /// Maximum context admitted by the pinned Qwen3.5 config.
    pub const fn context_capacity(&self) -> usize {
        self.program.context_capacity()
    }

    /// CUDA context shared by all slots, graphs, and pinned buffers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    #[cfg(feature = "qualification")]
    /// Stable retained device and pinned-logit addresses.
    pub fn qualification_addresses(&self) -> Vec<usize> {
        self.program
            .base_addresses()
            .into_iter()
            .map(|address| address as usize)
            .chain(core::iter::once(self.logits.as_ptr().addr()))
            .collect()
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

    fn replay_pending(&mut self) -> EngineResult<()> {
        let mut slots = [0usize; MAX_BATCH];
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        let mut rope_cos = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut rope_sin = [0.0f32; MAX_BATCH * ROTARY_PAIRS];
        let mut pending = 0;
        for &slot in &self.active_slots[..self.active] {
            let session = self.sessions[slot].as_ref().ok_or_else(|| {
                EngineError::generation("active Qwen3.5 slot has no pending session")
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
        self.program.load_decode_state(
            &self.stream,
            pending,
            &positions[..pending],
            &rope_cos[..pending * ROTARY_PAIRS],
            &rope_sin[..pending * ROTARY_PAIRS],
        )?;
        self.program.replay(&self.stream, pending)?;
        self.program.read_logits_into(
            &self.stream,
            pending,
            &mut self.logits[compact_download_logits(pending)],
        )?;
        for (row, &slot) in slots[..pending].iter().enumerate() {
            self.logits
                .copy_within(compact_download_row(row), slot * Qwen35_9B::VOCAB);
            let session = self.sessions[slot]
                .as_mut()
                .expect("pending Qwen3.5 slot owns a session");
            session.pending_token = None;
            session.next_position = session
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        Ok(())
    }
}

fn first_free_slot(occupied: [bool; MAX_BATCH]) -> Option<usize> {
    occupied.iter().position(|&occupied| !occupied)
}

fn slot_logits(slot: usize) -> Range<usize> {
    let begin = slot * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

fn compact_download_logits(rows: usize) -> Range<usize> {
    let begin = MAX_BATCH * Qwen35_9B::VOCAB;
    begin..begin + rows * Qwen35_9B::VOCAB
}

fn compact_download_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen35_9B::VOCAB;
    begin..begin + Qwen35_9B::VOCAB
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, first_free_slot};

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
}
