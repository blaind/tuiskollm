//! Single-request serving owner for Qwen3.8 Flash-Next.
//!
//! It reuses the qualified prime and decode entries. The explicit slot count keeps transport
//! admission aligned with the one slot this owner schedules.

use crate::common::progress::ResidentLoadProgress;
use crate::common::slots::{device_zero_context, require_generation_capacity};
use crate::common::text_generator::ModelProgram;
use crate::qwen38_flash_next::resident_model::{
    Qwen38FlashNextResidentLoadStats, Qwen38FlashNextResidentModel,
};
use crate::qwen38_flash_next::text_generation::Qwen38FlashNextGenerationTelemetry;
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GenerationSession, LayerMemoryLayout,
    MAX_BATCH, ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents,
    ResidentCancellation, ResidentRequestId,
};
use std::sync::Arc;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

/// Concurrent requests scheduled by this owner.
///
/// The underlying lifecycle has eight slots; compact scheduling is admitted separately.
pub const QWEN38_FLASH_NEXT_SERVING_SLOTS: usize = 1;

/// The one slot every served Flash-Next request owns for its lifetime.
const SERVING_SLOT: usize = 0;

/// Frontend, resident program, stream, and pinned logit row behind the served Flash-Next model.
pub struct Qwen38FlashNextResidentGenerator {
    frontend: TextFrontend,
    program: Qwen38FlashNextResidentModel,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    active: Option<Qwen38FlashNextServingSession>,
    next_request_id: u64,
}

struct Qwen38FlashNextServingSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    pending_token: Option<u32>,
    next_position: u32,
}

impl Qwen38FlashNextResidentGenerator {
    /// Opens the served Flash-Next program on device zero, reporting load progress.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, Some(progress))
    }

    /// Loads the resident program and the pinned frontend into `context`.
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
        let logits =
            PinnedHostBuffer::zeroed(context, Qwen38FlashNext::VOCAB).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            active: None,
            next_request_id: 1,
        })
    }

    /// Admits one request and refuses work beyond the proven dense band before device mutation.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        let control = GenerationSession::start(&self.frontend, request)?;
        // Admission uses the qualified dense band, not the deeper funded KV allocation.
        let required_positions = require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            ModelProgram::context_capacity(&self.program),
        )?;
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Flash-Next request identity overflows"))?;
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
        if self.active.is_some() {
            return Err(EngineError::route(
                "the single funded Flash-Next generation slot is already active",
            ));
        }

        let native_prefill_tokens = match self.program.prime_single_slot(
            &self.stream,
            control.prompt_token_ids(),
            required_positions,
        ) {
            Ok(tokens) => tokens,
            Err(error) => {
                // A failed prime may already own pages.
                self.program.recycle_slot(&self.stream, SERVING_SLOT)?;
                return Err(error);
            }
        };
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;
        let next_position = u32::try_from(prompt_tokens)
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        self.active = Some(Qwen38FlashNextServingSession {
            request_id,
            control,
            pending_token: None,
            next_position,
        });

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens: 0,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    /// Replays the pending token, then samples exactly one event for the active request.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        let Some(session) = self.active.as_mut() else {
            return Err(EngineError::generation(
                "cannot step an idle Flash-Next generation slot",
            ));
        };
        if let Some(token) = session.pending_token.take() {
            let position = session.next_position;
            self.program.replay_token(&self.stream, token, position)?;
            self.program
                .read_logits_into(&self.stream, 1, &mut self.logits)?;
            let session = self
                .active
                .as_mut()
                .expect("the active session survives its own replay");
            session.next_position = position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        let session = self
            .active
            .as_mut()
            .expect("the active session survives its own replay");
        let request_id = session.request_id;
        let step = session.control.accept_logits(&self.logits)?;
        if step.finish_reason.is_none() {
            session.pending_token = Some(step.token_id);
        }
        let completed = if step.finish_reason.is_some() {
            let session = self
                .active
                .take()
                .expect("the terminal session is the active one");
            self.program.recycle_slot(&self.stream, SERVING_SLOT)?;
            Some(session.control.into_output()?)
        } else {
            None
        };
        let mut events = std::array::from_fn(|_| None);
        events[0] = Some(ResidentBatchEvent {
            request_id,
            step,
            completed,
        });

        Ok(ResidentBatchEvents::from_events(events, 1))
    }

    /// Cancels the request, releases its pages, and clears its recurrent carries.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let active = self
            .active
            .take_if(|session| session.request_id == request_id)
            .ok_or_else(|| {
                EngineError::generation("Flash-Next cancellation request is not active")
            })?;
        self.program.recycle_slot(&self.stream, SERVING_SLOT)?;

        Ok(ResidentCancellation {
            request_id,
            output: active.control.cancel()?,
            device_retained_tokens: 0,
        })
    }

    /// Requests currently holding the funded slot.
    pub const fn active_requests(&self) -> usize {
        match self.active {
            Some(_) => 1,
            None => 0,
        }
    }

    /// Active request identities, at most one.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active.iter().map(|session| session.request_id)
    }

    /// Concurrent requests this owner funds on the device.
    pub const fn slot_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_SERVING_SLOTS
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

    /// Page-locked staging, engram, and logit bytes this owner holds.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
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

    /// Whole-request streaming and timing evidence for the request in flight.
    pub const fn telemetry(&self) -> Qwen38FlashNextGenerationTelemetry {
        self.program.generation_telemetry()
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// CUDA context shared by every arena, stream, graph, and pinned buffer.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }
}

// Serving cannot expose more slots than the lifecycle owns.
const _: () =
    assert!(QWEN38_FLASH_NEXT_SERVING_SLOTS >= 1 && QWEN38_FLASH_NEXT_SERVING_SLOTS <= MAX_BATCH);
const _: () = assert!(SERVING_SLOT < QWEN38_FLASH_NEXT_SERVING_SLOTS);
