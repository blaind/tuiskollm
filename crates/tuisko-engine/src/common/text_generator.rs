//! Target-neutral single-slot text generation and compact-batch scheduler vocabulary.

use crate::common::slots::require_generation_capacity;
use crate::{
    CancelledText, ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, MAX_BATCH,
};
#[cfg(feature = "qualification")]
use crate::{Sampler, SamplingOptions};
use std::sync::Arc;
#[cfg(feature = "qualification")]
use std::time::{Duration, Instant};
use tuisko_frontend::{GenerationDefaults, PromptEncodingMetrics, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot};

pub(crate) mod sealed {
    /// Restricts `ModelProgram` to the in-crate resident target programs.
    pub trait Sealed {}
}

/// Host-visible replay and readback surface one single-slot generator needs.
///
/// This is deliberately not a model abstraction: it exposes token consumption,
/// logit readback, and slot priming only. Forward-pass composition, graph node
/// sequences, and accumulation order remain concrete per target.
pub trait ModelProgram: sealed::Sealed + Sized {
    /// Pinned checkpoint architecture whose snapshot loads this resident program.
    type Arch: Arch;

    /// Stable-address report this target returns to source-backed qualification.
    #[cfg(feature = "qualification")]
    type QualificationAddresses;

    /// Admits this target's tokenizer, chat template, and sampling defaults.
    fn open_frontend(snapshot: &CheckpointSnapshot<Self::Arch>) -> EngineResult<TextFrontend>;

    /// Loads the complete resident program into `context`.
    fn load(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Self::Arch>>,
    ) -> EngineResult<Self>;

    /// CUDA context shared by the model, stream, graphs, and pinned buffers.
    fn context(&self) -> &Arc<CudaContext>;

    /// Current token capacity of one physical slot.
    fn context_capacity(&self) -> usize;

    /// Page-locked embedding staging bytes owned by the program.
    fn host_stager_bytes(&self) -> usize;

    /// Downloads `batch` exact logit rows into `destination`.
    fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()>;

    /// Runs this target's exact slot reset, reservation, and prompt-priming sequence.
    ///
    /// The implementation reproduces the target's established host call order and
    /// returns the prompt tokens processed by native prefill graphs. The generic
    /// caller adds no device work of its own around it.
    fn prime_single_slot(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        required_positions: usize,
    ) -> EngineResult<usize>;

    /// Replays one exact decode step for `token` at `position`.
    fn replay_token(&mut self, stream: &CudaStream, token: u32, position: u32) -> EngineResult<()>;

    /// Stable device addresses reported beside the pinned logit bank at `logits`.
    #[cfg(feature = "qualification")]
    fn qualification_addresses(&self, logits: usize) -> Self::QualificationAddresses;
}

/// What one raw-token qualification run produced.
#[cfg(feature = "qualification")]
#[derive(Clone, Debug, PartialEq)]
pub struct QualificationGeneration {
    /// Tokens the production sampler selected, in order, including a selected stop token.
    pub token_ids: Vec<u32>,
    /// Prompt tokens the target's native prefill graphs carried.
    pub native_prefill_tokens: usize,
    /// Index of the generated token that was an admitted stop, when one was selected.
    pub stopped_at: Option<usize>,
    /// Host wall time through selection of the first token.
    pub time_to_first_token: Option<Duration>,
    /// Ranked `(token, logit)` candidates per step, empty unless the caller asked for them.
    pub steps: Vec<Vec<(u32, f32)>>,
}

/// The `count` highest logits of one BF16 vocabulary row, descending, ties to the lower token.
///
/// The same order the production sampler's own selection walks, so a gate reporting these is
/// reporting what the sampler saw rather than a second opinion about it.
#[cfg(feature = "qualification")]
fn ranked_candidates(logits: &[u16], count: usize) -> Vec<(u32, f32)> {
    let mut ranked = logits
        .iter()
        .enumerate()
        .map(|(token, &bits)| (token as u32, f32::from_bits(u32::from(bits) << 16)))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(count);

    ranked
}

/// Concrete frontend, device program, stream, and host-logit owner for one active request.
pub struct SingleSlotTextGenerator<M> {
    frontend: TextFrontend,
    program: M,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
}

/// One streaming generation request borrowing the single resident slot.
pub struct SingleSlotGenerationSession<'a, M> {
    control: GenerationSession,
    program: &'a mut M,
    stream: &'a CudaStream,
    logits: &'a mut [u16],
    pending_token: Option<u32>,
    next_position: u32,
    native_prefill_tokens: usize,
}

impl<M> SingleSlotTextGenerator<M> {
    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    pub(crate) const fn program(&self) -> &M {
        &self.program
    }

    pub(crate) const fn program_mut(&mut self) -> &mut M {
        &mut self.program
    }
}

impl<M: ModelProgram> SingleSlotTextGenerator<M> {
    /// Admits the pinned frontend and loads the exact resident model into `context`.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<M::Arch>>,
    ) -> EngineResult<Self> {
        let frontend = M::open_frontend(snapshot.as_ref())?;
        let program = M::load(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logits = PinnedHostBuffer::zeroed(context, M::Arch::VOCAB).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
        })
    }

    /// Renders one request and primes its prompt through exact resident graphs.
    pub fn start<'a>(
        &'a mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<SingleSlotGenerationSession<'a, M>> {
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
            native_prefill_tokens = program.prime_single_slot(
                stream,
                control.prompt_token_ids(),
                required_positions,
            )?;
            program.read_logits_into(stream, 1, logits)?;
        }
        let next_position = u32::try_from(control.prompt_token_ids().len())
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;

        Ok(SingleSlotGenerationSession {
            control,
            program,
            stream,
            logits,
            pending_token: None,
            next_position,
            native_prefill_tokens,
        })
    }

    /// Page-locked embedding and logit staging bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    #[cfg(feature = "qualification")]
    /// Runs an independently captured raw-token reference case through the production path.
    pub fn qualification_greedy_after_tokens(&mut self, token_ids: &[u32]) -> EngineResult<u32> {
        self.qualification_greedy_after_tokens_with_route(token_ids)
            .map(|(token, _)| token)
    }

    #[cfg(feature = "qualification")]
    /// Runs raw tokens and reports how many the production path gave to native prefill.
    pub fn qualification_greedy_after_tokens_with_route(
        &mut self,
        token_ids: &[u32],
    ) -> EngineResult<(u32, usize)> {
        let required_positions =
            require_generation_capacity(token_ids.len(), 1, self.program.context_capacity())?;
        let native_prefill_tokens =
            self.program
                .prime_single_slot(&self.stream, token_ids, required_positions)?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;
        let mut sampler = Sampler::new(SamplingOptions::greedy(), self.frontend.stop_ids())?;
        Ok((
            sampler.sample(&self.logits)?.token_id,
            native_prefill_tokens,
        ))
    }

    #[cfg(feature = "qualification")]
    /// Runs raw-token generation through production admission, replay, and sampling.
    pub fn qualification_generate_from_tokens(
        &mut self,
        token_ids: &[u32],
        max_new_tokens: usize,
        options: SamplingOptions,
        record_top: usize,
    ) -> EngineResult<QualificationGeneration> {
        let required_positions = require_generation_capacity(
            token_ids.len(),
            max_new_tokens,
            self.program.context_capacity(),
        )?;
        let mut sampler = Sampler::new(options, self.frontend.stop_ids())?;
        if max_new_tokens == 0 {
            return Ok(QualificationGeneration {
                token_ids: Vec::new(),
                native_prefill_tokens: 0,
                stopped_at: None,
                time_to_first_token: None,
                steps: Vec::new(),
            });
        }
        let started = Instant::now();
        let native_prefill_tokens =
            self.program
                .prime_single_slot(&self.stream, token_ids, required_positions)?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut position = u32::try_from(token_ids.len())
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        let mut stopped_at = None;
        let mut time_to_first_token = None;
        let mut steps = Vec::with_capacity(if record_top == 0 { 0 } else { max_new_tokens });
        for index in 0..max_new_tokens {
            if record_top != 0 {
                steps.push(ranked_candidates(&self.logits, record_top));
            }
            let decision = sampler.sample(&self.logits)?;
            if index == 0 {
                time_to_first_token = Some(started.elapsed());
            }
            generated.push(decision.token_id);
            if decision.stopped {
                stopped_at = Some(index);
                break;
            }
            if generated.len() == max_new_tokens {
                break;
            }
            self.program
                .replay_token(&self.stream, decision.token_id, position)?;
            self.program
                .read_logits_into(&self.stream, 1, &mut self.logits)?;
            position = position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        Ok(QualificationGeneration {
            token_ids: generated,
            native_prefill_tokens,
            stopped_at,
            time_to_first_token,
            steps,
        })
    }

    #[cfg(feature = "qualification")]
    /// Scores each forced continuation token on the reference context.
    pub fn qualification_score_forced_tokens(
        &mut self,
        token_ids: &[u32],
        continuation: &[u32],
        record_top: usize,
    ) -> EngineResult<Vec<Vec<(u32, f32)>>> {
        let required_positions = require_generation_capacity(
            token_ids.len(),
            continuation.len(),
            self.program.context_capacity(),
        )?;
        if continuation.is_empty() {
            return Ok(Vec::new());
        }
        self.program
            .prime_single_slot(&self.stream, token_ids, required_positions)?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;

        let mut rows = Vec::with_capacity(continuation.len());
        let mut position = u32::try_from(token_ids.len())
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        for (index, &token) in continuation.iter().enumerate() {
            rows.push(ranked_candidates(&self.logits, record_top));
            if index + 1 == continuation.len() {
                break;
            }
            self.program.replay_token(&self.stream, token, position)?;
            self.program
                .read_logits_into(&self.stream, 1, &mut self.logits)?;
            position = position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        Ok(rows)
    }

    #[cfg(feature = "qualification")]
    /// Reads the row for one raw-token prompt without generating anything from it.
    pub fn qualification_prime_logits(&mut self, token_ids: &[u32]) -> EngineResult<&[u16]> {
        let required_positions =
            require_generation_capacity(token_ids.len(), 1, self.program.context_capacity())?;
        self.program
            .prime_single_slot(&self.stream, token_ids, required_positions)?;
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;

        Ok(&self.logits)
    }

    #[cfg(feature = "qualification")]
    /// The loaded resident program, for a gate that needs its target-specific evidence.
    pub const fn qualification_program(&self) -> &M {
        &self.program
    }

    #[cfg(feature = "qualification")]
    /// The loaded resident program, for a gate that drives its lifecycle directly.
    pub const fn qualification_program_mut(&mut self) -> &mut M {
        &mut self.program
    }

    #[cfg(feature = "qualification")]
    /// The loaded program and its execution stream for an in-process qualification sweep.
    pub fn qualification_program_and_stream(&mut self) -> (&mut M, &CudaStream) {
        (&mut self.program, &self.stream)
    }

    #[cfg(feature = "qualification")]
    /// The stream every replay this generator issues runs on.
    pub const fn qualification_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    #[cfg(feature = "qualification")]
    /// Stable device-arena and pinned-logit addresses owned by this generator.
    pub fn qualification_addresses(&self) -> M::QualificationAddresses {
        self.program
            .qualification_addresses(self.logits.as_ptr().addr())
    }
}

impl<M> SingleSlotGenerationSession<'_, M> {
    /// Exact prompt token IDs selected by the admitted frontend.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.control.prompt_token_ids()
    }

    /// Tokens selected so far, including an unprocessed final token.
    pub fn generated_token_ids(&self) -> &[u32] {
        self.control.generated_token_ids()
    }

    /// Prompt tokens processed by exact whole-model prefill graphs.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.native_prefill_tokens
    }

    /// Current terminal state.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.control.finish_reason()
    }

    /// Converts a terminal session into its complete decoded result.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        self.control.into_output()
    }

    pub(crate) const fn control(&self) -> &GenerationSession {
        &self.control
    }
}

impl<M: ModelProgram> SingleSlotGenerationSession<'_, M> {
    /// Samples one token and returns its streaming delta before the next model replay.
    pub fn step(&mut self) -> EngineResult<GenerationStep> {
        if let Some(token) = self.pending_token.take() {
            self.program
                .replay_token(self.stream, token, self.next_position)?;
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
    /// Prompt tokens processed by exact whole-model prefill graphs.
    pub native_prefill_tokens: usize,
    /// Observation-only frontend timing and prefix-lookup detail.
    pub prompt_metrics: PromptEncodingMetrics,
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

impl ResidentRequestId {
    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Numeric identity suitable for transport correlation and logs.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl ResidentBatchEvents {
    pub(crate) const fn from_events(
        events: [Option<ResidentBatchEvent>; MAX_BATCH],
        len: usize,
    ) -> Self {
        Self { events, len }
    }

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
