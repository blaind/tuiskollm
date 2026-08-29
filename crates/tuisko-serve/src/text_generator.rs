//! Serving boundary over the exact resident compact schedulers.
//!
//! This is the only abstraction `tuisko-serve` holds over a resident target.
//! It is crate-private, so it is sealed by construction and can never become an
//! open backend extension point, and the worker monomorphizes over it: no `dyn`,
//! no vtable, and no heap indirection on the scheduler's round path.
//!
//! The traits describe admission, stepping, and cancellation only. Round
//! ordering, acceptance laws, and device commit protocol remain concrete per
//! target inside `tuisko-engine`.

use tuisko_engine::{
    ChatGenerationRequest, EngineError, EngineErrorCode, GeneratedText, GenerationStep, MAX_BATCH,
    PromptLogprobs, Qwen35ResidentMtpBatchEvent, Qwen35ResidentMtpBatchEvents,
    Qwen35ResidentMtpBatchGenerator, Qwen36ResidentBatchGenerator,
    Qwen38FlashNextMtpResidentGenerator, Qwen38FlashNextResidentBatchGenerator,
    ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents, ResidentCancellation,
    ResidentMtpBatchEvent, ResidentMtpBatchEvents, ResidentMtpBatchGenerator,
    ResidentMtpGenerationStats, ResidentRequestId,
};

/// One request's committed steps and terminal output from a scheduler round.
pub(crate) trait GenerationEvent {
    /// Request that produced this event.
    fn request_id(&self) -> ResidentRequestId;

    /// Streaming steps in target-licensed order.
    fn steps(&self) -> impl Iterator<Item = &GenerationStep>;

    /// Complete output when this round terminated the request.
    fn completed(&self) -> Option<&GeneratedText>;

    /// Cumulative exact MTP route, output, and acceptance counters, when exposed by this target.
    fn mtp_stats(&self) -> Option<ResidentMtpGenerationStats> {
        None
    }
}

/// One scheduler round's events in the stable active order at round entry.
pub(crate) trait GenerationEvents {
    /// Per-request event this target's scheduler emits.
    type Event: GenerationEvent;

    /// Events in the active order that existed at the start of the round.
    fn iter(&self) -> impl Iterator<Item = &Self::Event>;
}

/// Compact-scheduler surface the resident worker drives identically per target.
pub(crate) trait TextGenerator {
    /// Round events this target's scheduler returns from one step.
    type Events: GenerationEvents;

    /// Admits one request into an available compact batch slot.
    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError>;

    /// Admits queued requests in ingress order.
    fn admit_batch(
        &mut self,
        requests: &[&ChatGenerationRequest],
    ) -> Vec<Result<ResidentBatchAdmission, EngineError>> {
        requests.iter().map(|request| self.admit(request)).collect()
    }

    /// Admits queued requests while publishing processed and total prefill tokens when supported.
    fn admit_batch_with_progress<P>(
        &mut self,
        requests: &[&ChatGenerationRequest],
        _progress: &mut P,
    ) -> Vec<Result<ResidentBatchAdmission, EngineError>>
    where
        P: FnMut(usize, usize, usize) -> bool,
    {
        self.admit_batch(requests)
    }

    /// Advances every active request by exactly one scheduler round.
    fn step(&mut self) -> Result<Self::Events, EngineError>;

    /// Cancels one active request and releases its physical slot.
    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError>;

    /// Current number of active device-backed requests.
    fn active_requests(&self) -> usize;

    /// Active request identities in compact scheduler order.
    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId>;

    /// Concurrent requests this target schedules on the device.
    fn slot_capacity(&self) -> usize;

    /// Scores token-ID prompts while the scheduler is idle when this exact target admits it.
    fn score_prompts(&mut self, _prompts: &[Vec<u32>]) -> Result<Vec<PromptLogprobs>, EngineError> {
        Err(EngineError::Contract {
            code: EngineErrorCode::Generation,
            message: "prompt scoring is unsupported for this exact target".into(),
        })
    }
}

impl GenerationEvent for ResidentMtpBatchEvent {
    fn request_id(&self) -> ResidentRequestId {
        self.request_id
    }

    fn steps(&self) -> impl Iterator<Item = &GenerationStep> {
        ResidentMtpBatchEvent::steps(self)
    }

    fn completed(&self) -> Option<&GeneratedText> {
        self.completed.as_ref()
    }

    fn mtp_stats(&self) -> Option<ResidentMtpGenerationStats> {
        Some(self.stats)
    }
}

impl GenerationEvents for ResidentMtpBatchEvents {
    type Event = ResidentMtpBatchEvent;

    fn iter(&self) -> impl Iterator<Item = &Self::Event> {
        ResidentMtpBatchEvents::iter(self)
    }
}

impl TextGenerator for ResidentMtpBatchGenerator {
    type Events = ResidentMtpBatchEvents;

    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError> {
        ResidentMtpBatchGenerator::admit(self, request)
    }

    fn admit_batch_with_progress<P>(
        &mut self,
        requests: &[&ChatGenerationRequest],
        progress: &mut P,
    ) -> Vec<Result<ResidentBatchAdmission, EngineError>>
    where
        P: FnMut(usize, usize, usize) -> bool,
    {
        let mut completed_before = 0usize;
        let mut total_before = 0usize;
        requests
            .iter()
            .enumerate()
            .map(|(index, request)| {
                let mut current_total = 0usize;
                let admission = ResidentMtpBatchGenerator::admit_with_progress(
                    self,
                    request,
                    |completed, total| {
                        current_total = total;
                        progress(
                            index,
                            completed_before.saturating_add(completed),
                            total_before.saturating_add(total),
                        )
                    },
                );
                completed_before = completed_before.saturating_add(current_total);
                total_before = total_before.saturating_add(current_total);
                admission
            })
            .collect()
    }

    fn step(&mut self) -> Result<Self::Events, EngineError> {
        ResidentMtpBatchGenerator::step(self)
    }

    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError> {
        ResidentMtpBatchGenerator::cancel(self, request_id)
    }

    fn active_requests(&self) -> usize {
        ResidentMtpBatchGenerator::active_requests(self)
    }

    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> {
        ResidentMtpBatchGenerator::active_request_ids(self)
    }

    fn slot_capacity(&self) -> usize {
        MAX_BATCH
    }

    fn score_prompts(&mut self, prompts: &[Vec<u32>]) -> Result<Vec<PromptLogprobs>, EngineError> {
        prompts
            .iter()
            .map(|prompt| ResidentMtpBatchGenerator::score_prompt(self, prompt))
            .collect()
    }
}

impl GenerationEvent for Qwen35ResidentMtpBatchEvent {
    fn request_id(&self) -> ResidentRequestId {
        self.request_id
    }

    fn steps(&self) -> impl Iterator<Item = &GenerationStep> {
        Qwen35ResidentMtpBatchEvent::steps(self)
    }

    fn completed(&self) -> Option<&GeneratedText> {
        self.completed.as_ref()
    }
}

impl GenerationEvents for Qwen35ResidentMtpBatchEvents {
    type Event = Qwen35ResidentMtpBatchEvent;

    fn iter(&self) -> impl Iterator<Item = &Self::Event> {
        Qwen35ResidentMtpBatchEvents::iter(self)
    }
}

impl TextGenerator for Qwen35ResidentMtpBatchGenerator {
    type Events = Qwen35ResidentMtpBatchEvents;

    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError> {
        Qwen35ResidentMtpBatchGenerator::admit(self, request)
    }

    fn step(&mut self) -> Result<Self::Events, EngineError> {
        Qwen35ResidentMtpBatchGenerator::step(self)
    }

    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError> {
        Qwen35ResidentMtpBatchGenerator::cancel(self, request_id)
    }

    fn active_requests(&self) -> usize {
        Qwen35ResidentMtpBatchGenerator::active_requests(self)
    }

    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> {
        Qwen35ResidentMtpBatchGenerator::active_request_ids(self)
    }

    fn slot_capacity(&self) -> usize {
        MAX_BATCH
    }
}

// The Qwen3.6 compact scheduler commits exactly one token per request per round,
// so its event carries a single step where the MTP schedulers carry a committed run.
impl GenerationEvent for ResidentBatchEvent {
    fn request_id(&self) -> ResidentRequestId {
        self.request_id
    }

    fn steps(&self) -> impl Iterator<Item = &GenerationStep> {
        core::iter::once(&self.step)
    }

    fn completed(&self) -> Option<&GeneratedText> {
        self.completed.as_ref()
    }
}

impl GenerationEvents for ResidentBatchEvents {
    type Event = ResidentBatchEvent;

    fn iter(&self) -> impl Iterator<Item = &Self::Event> {
        ResidentBatchEvents::iter(self)
    }
}

impl TextGenerator for Qwen36ResidentBatchGenerator {
    type Events = ResidentBatchEvents;

    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError> {
        Qwen36ResidentBatchGenerator::admit(self, request)
    }

    fn step(&mut self) -> Result<Self::Events, EngineError> {
        Qwen36ResidentBatchGenerator::step(self)
    }

    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError> {
        Qwen36ResidentBatchGenerator::cancel(self, request_id)
    }

    fn active_requests(&self) -> usize {
        Qwen36ResidentBatchGenerator::active_requests(self)
    }

    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> {
        Qwen36ResidentBatchGenerator::active_request_ids(self)
    }

    fn slot_capacity(&self) -> usize {
        MAX_BATCH
    }
}

impl TextGenerator for Qwen38FlashNextMtpResidentGenerator {
    type Events = ResidentBatchEvents;

    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError> {
        Qwen38FlashNextMtpResidentGenerator::admit(self, request)
    }

    fn step(&mut self) -> Result<Self::Events, EngineError> {
        Qwen38FlashNextMtpResidentGenerator::step(self)
    }

    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError> {
        Qwen38FlashNextMtpResidentGenerator::cancel(self, request_id)
    }

    fn active_requests(&self) -> usize {
        Qwen38FlashNextMtpResidentGenerator::active_requests(self)
    }

    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> {
        Qwen38FlashNextMtpResidentGenerator::active_request_ids(self)
    }

    fn slot_capacity(&self) -> usize {
        Qwen38FlashNextMtpResidentGenerator::slot_capacity(self)
    }
}

impl TextGenerator for Qwen38FlashNextResidentBatchGenerator {
    type Events = ResidentBatchEvents;

    fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<ResidentBatchAdmission, EngineError> {
        Qwen38FlashNextResidentBatchGenerator::admit(self, request)
    }

    fn admit_batch(
        &mut self,
        requests: &[&ChatGenerationRequest],
    ) -> Vec<Result<ResidentBatchAdmission, EngineError>> {
        Qwen38FlashNextResidentBatchGenerator::admit_batch(self, requests)
    }

    fn step(&mut self) -> Result<Self::Events, EngineError> {
        Qwen38FlashNextResidentBatchGenerator::step(self)
    }

    fn cancel(
        &mut self,
        request_id: ResidentRequestId,
    ) -> Result<ResidentCancellation, EngineError> {
        Qwen38FlashNextResidentBatchGenerator::cancel(self, request_id)
    }

    fn active_requests(&self) -> usize {
        Qwen38FlashNextResidentBatchGenerator::active_requests(self)
    }

    fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> {
        Qwen38FlashNextResidentBatchGenerator::active_request_ids(self)
    }

    fn slot_capacity(&self) -> usize {
        Qwen38FlashNextResidentBatchGenerator::slot_capacity(self)
    }
}
