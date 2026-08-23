//! Single-slot text generation over the exact resident model owner.

use crate::{
    ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, ResidentModelProgram,
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

/// Concrete frontend, device program, stream, and host-logit owner for one active request.
pub struct ResidentTextGenerator {
    frontend: TextFrontend,
    program: ResidentModelProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
}

/// One streaming generation request borrowing the single resident slot.
pub struct ResidentGenerationSession<'a> {
    control: GenerationSession<'a>,
    program: &'a mut ResidentModelProgram,
    stream: &'a CudaStream,
    logits: &'a mut [u16],
    pending_token: Option<u32>,
    next_position: u32,
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

    /// Renders one request and primes its prompt through the admitted B=1 decode graph.
    ///
    /// This is the current short-context correctness route, not an optimized prefill route.
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
        require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            program.context_capacity(),
        )?;

        if control.finish_reason().is_none() {
            program.reset_state(stream)?;
            for (position, &token) in control.prompt_token_ids().iter().enumerate() {
                replay_token(
                    program,
                    stream,
                    token,
                    u32::try_from(position).map_err(|_| {
                        EngineError::generation("prompt position exceeds the exact route width")
                    })?,
                )?;
            }
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
        require_generation_capacity(token_ids.len(), 1, self.program.context_capacity())?;
        self.program.reset_state(&self.stream)?;
        for (position, &token) in token_ids.iter().enumerate() {
            replay_token(&mut self.program, &self.stream, token, position as u32)?;
        }
        self.program
            .read_logits_into(&self.stream, 1, &mut self.logits)?;
        let stop_ids: [u32; 2] =
            self.frontend.stop_ids().try_into().map_err(|_| {
                EngineError::generation("frontend returned the wrong stop-ID count")
            })?;
        let mut sampler = Sampler::new(SamplingOptions::greedy(), stop_ids)?;
        Ok(sampler.sample(&self.logits)?.token_id)
    }

    #[cfg(feature = "qualification")]
    /// Stable device-arena and pinned-logit addresses owned by this generator.
    pub fn qualification_addresses(&self) -> [usize; 2] {
        [
            self.program.base_address() as usize,
            self.logits.as_ptr().addr(),
        ]
    }
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

fn replay_token(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    token: u32,
    position: u32,
) -> EngineResult<()> {
    let (rope_cos, rope_sin) = text_rope(position);
    program.stage_embeddings(stream, &[token])?;
    program.load_decode_state(stream, 1, &[position], &rope_cos, &rope_sin)?;
    program.replay(stream, 1)
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
) -> EngineResult<()> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ROTARY_PAIRS, require_generation_capacity, text_rope};
    use crate::EngineErrorCode;

    #[test]
    fn short_context_capacity_counts_only_processed_generated_tokens() {
        require_generation_capacity(192, 1, 192).unwrap();
        require_generation_capacity(1, 192, 192).unwrap();
        require_generation_capacity(192, 0, 192).unwrap();

        for (prompt, generated) in [(0, 1), (192, 2), (2, 192), (usize::MAX, 2)] {
            assert_eq!(
                require_generation_capacity(prompt, generated, 192)
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
}
