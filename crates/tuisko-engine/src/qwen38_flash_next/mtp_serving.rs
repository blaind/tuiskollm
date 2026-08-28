//! Single-slot serving owner for speculative Qwen3.8-Flash-Next generation.

use crate::common::progress::ResidentLoadProgress;
use crate::common::slots::device_zero_context;
use crate::qwen38_flash_next::mtp_generation::{
    Qwen38FlashNextMtpRoundState, Qwen38FlashNextMtpTextGenerator,
};
use crate::qwen38_flash_next::mtp_program::Qwen38FlashNextMtpProgram;
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, LayerMemoryLayout, ResidentBatchAdmission,
    ResidentBatchEvent, ResidentBatchEvents, ResidentCancellation, ResidentMtpGenerationStats,
    ResidentRequestId,
};
use std::sync::Arc;
use tuisko_frontend::GenerationDefaults;
use tuisko_gpu::CudaContext;
use tuisko_model::{CheckpointSnapshot, Qwen38FlashNext};

/// Concurrent requests funded by the speculative owner.
pub const QWEN38_FLASH_NEXT_MTP_SERVING_SLOTS: usize = 1;

/// Frontend, target, draft block, and one active request.
pub struct Qwen38FlashNextMtpResidentGenerator {
    generator: Qwen38FlashNextMtpTextGenerator,
    active: Option<ServingSession>,
    next_request_id: u64,
}

struct ServingSession {
    request_id: ResidentRequestId,
    state: Qwen38FlashNextMtpRoundState,
}

impl Qwen38FlashNextMtpResidentGenerator {
    /// Opens the speculative pair on device zero and reports load progress.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, Some(progress))
    }

    /// Loads the pair against one joint residency solve.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let generator = Qwen38FlashNextMtpTextGenerator::from_snapshot_with_progress(
            context, snapshot, progress,
        )?;

        Ok(Self {
            generator,
            active: None,
            next_request_id: 1,
        })
    }

    /// Admits one request and primes the target and draft mirrors.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        if self.active.is_some() {
            return Err(EngineError::route(
                "the Qwen3.8 Flash-Next speculative slot is already active",
            ));
        }
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self.next_request_id.checked_add(1).ok_or_else(|| {
            EngineError::generation("Qwen3.8 Flash-Next request identity overflows")
        })?;

        let state = match self.generator.start(request) {
            Ok(session) => session.into_state(),
            Err(error) => {
                self.generator.release_slot()?;
                return Err(error);
            }
        };
        let prompt_tokens = state.prompt_encoding().token_ids.len();
        let prompt_metrics = state.prompt_metrics().clone();
        let native_prefill_tokens = state.native_prefill_tokens();
        if state.finish_reason().is_some() {
            return Ok(ResidentBatchAdmission {
                request_id,
                prompt_tokens,
                device_reused_tokens: 0,
                native_prefill_tokens,
                prompt_metrics,
                completed: Some(state.into_output()?),
            });
        }
        self.active = Some(ServingSession { request_id, state });

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens: 0,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    /// Returns one committed output, running a speculative round when needed.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        let session = self.active.take().ok_or_else(|| {
            EngineError::generation("cannot step an idle Qwen3.8 Flash-Next speculative slot")
        })?;
        let request_id = session.request_id;
        let (state, step) = self.generator.step_state(session.state)?;
        let completed = if state.finish_reason().is_some() {
            self.generator.release_slot()?;
            Some(state.into_output()?)
        } else {
            self.active = Some(ServingSession { request_id, state });
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

    /// Cancels the active request and releases its pages.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        if self.active.as_ref().map(|active| active.request_id) != Some(request_id) {
            return Err(EngineError::generation(
                "Qwen3.8 Flash-Next speculative cancellation is not active",
            ));
        }
        let active = self.active.take().ok_or_else(|| {
            EngineError::generation("Qwen3.8 Flash-Next speculative cancellation is not active")
        })?;
        self.generator.release_slot()?;

        Ok(ResidentCancellation {
            request_id,
            output: active.state.cancel()?,
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

    /// Concurrent requests funded on the device.
    pub const fn slot_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_MTP_SERVING_SLOTS
    }

    /// Longest sequence the shared page pool admits.
    pub fn context_capacity(&self) -> usize {
        self.generator.context_capacity()
    }

    /// Speculative activity for the request in flight.
    pub fn stats(&self) -> ResidentMtpGenerationStats {
        self.active
            .as_ref()
            .map_or_else(ResidentMtpGenerationStats::default, |session| {
                session.state.stats()
            })
    }

    /// Device bytes occupied by the target and draft pair.
    pub fn arena_bytes(&self) -> EngineResult<usize> {
        let draft = self.program().layout().total_device_bytes()?;
        let target = self.program().target().layout().total_device_bytes()?;
        target
            .checked_add(draft)
            .ok_or_else(|| EngineError::layout("Qwen3.8 Flash-Next pair device bytes overflow"))
    }

    /// Source-backed weights uploaded for both halves.
    pub fn resident_weight_bytes(&self) -> usize {
        self.program().layout().resident_weight_bytes()
            + self.program().target().layout().resident_weight_bytes()
    }

    /// Draft construction measurements.
    pub fn load_stats(&self) -> crate::Qwen38FlashNextMtpLoadStats {
        self.program().load_stats()
    }

    /// Target construction measurements.
    pub const fn target_load_stats(&self) -> crate::Qwen38FlashNextResidentLoadStats {
        self.program().target().load_stats()
    }

    /// Page-locked staging bytes retained by the pair.
    pub fn host_stager_bytes(&self) -> usize {
        self.generator.host_stager_bytes()
    }

    /// Whether the target borrows its primary expert extent from the checkpoint mapping.
    pub fn mapped_primary(&self) -> bool {
        !self
            .program()
            .target()
            .layout()
            .streaming()
            .primary_source()
            .is_pinned()
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.generator.generation_defaults()
    }

    /// CUDA context shared by the pair.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.generator.context()
    }

    /// Loaded target and draft program.
    pub const fn program(&self) -> &Qwen38FlashNextMtpProgram {
        self.generator.program()
    }
}

const _: () = assert!(QWEN38_FLASH_NEXT_MTP_SERVING_SLOTS == 1);
