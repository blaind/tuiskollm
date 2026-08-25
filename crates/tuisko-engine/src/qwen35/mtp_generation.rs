//! Single-slot Qwen3.5 generation over the resident target and source-BF16 MTP layer.

use crate::common::mtp::{DRAFT_WINDOW, VERIFY_ROWS, decide_sampled_tokens};
use crate::resident_generation::{device_zero_context, require_generation_capacity, text_rope};
use crate::resident_mtp_generation::fill_contiguous_rope;
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, FinishReason, GeneratedText,
    GenerationSession, GenerationStep, Qwen35ResidentMtpProgram, ResidentMtpGenerationStats,
    SamplingDistribution,
};
use std::sync::Arc;
use tuisko_frontend::TextFrontend;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const ROTARY_PAIRS: usize = 32;
const LOGIT_ROWS: usize = VERIFY_ROWS + 1;

/// Concrete single-slot owner for Qwen3.5 draft-three generation.
pub struct Qwen35ResidentMtpTextGenerator {
    frontend: TextFrontend,
    program: Qwen35ResidentMtpProgram,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    target_hidden: PinnedHostBuffer<u16>,
}

/// One streaming Qwen3.5 request borrowing the target-plus-MTP owner.
pub struct Qwen35ResidentMtpGenerationSession<'a> {
    control: GenerationSession,
    program: &'a mut Qwen35ResidentMtpProgram,
    stream: &'a CudaStream,
    logits: &'a mut PinnedHostBuffer<u16>,
    target_hidden: &'a mut PinnedHostBuffer<u16>,
    maximum_new_tokens: usize,
    stop_ids: Vec<u32>,
    next_position: usize,
    started: bool,
    proposal_ready: bool,
    queued: [Option<GenerationStep>; VERIFY_ROWS],
    queue_start: usize,
    queue_len: usize,
    visible_generated: usize,
    native_prefill_tokens: usize,
    greedy: bool,
    stats: ResidentMtpGenerationStats,
}

impl Qwen35ResidentMtpTextGenerator {
    /// Opens the exact Qwen3.5 owner on CUDA device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot)
    }

    /// Loads one target, one MTP layer, and fixed transaction buffers.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
        let program = Qwen35ResidentMtpProgram::from_snapshot(context, snapshot)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen35_9B::VOCAB
            .checked_mul(LOGIT_ROWS)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP generation logits overflow"))?;
        let hidden_values = Qwen35_9B::HIDDEN
            .checked_mul(VERIFY_ROWS)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP target-hidden bank overflows"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;
        let target_hidden =
            PinnedHostBuffer::zeroed(context, hidden_values).map_err(GpuError::from)?;
        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            target_hidden,
        })
    }

    /// Renders one request and primes matching target and MTP cache rows.
    pub fn start<'a>(
        &'a mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<Qwen35ResidentMtpGenerationSession<'a>> {
        let greedy = request.sampling.is_greedy();
        let stop_ids = self.frontend.stop_ids().to_vec();
        let Self {
            frontend,
            program,
            stream,
            logits,
            target_hidden,
        } = self;
        let control = GenerationSession::start(frontend, request)?;
        let prompt_tokens = control.prompt_token_ids().len();
        let required_positions = require_generation_capacity(
            prompt_tokens,
            request.max_new_tokens,
            program.layout().context_capacity(),
        )?;
        let mut native_prefill_tokens = 0;

        if control.finish_reason().is_none() {
            program.recycle_kv_slot(stream, 0)?;
            program.target_mut().reset_slot(stream, 0)?;
            program.activate_kv_slot(0)?;
            program.reserve_kv_slot_tokens(stream, 0, required_positions)?;
            native_prefill_tokens = prime_qwen35_mtp_prompt(
                program,
                stream,
                control.prompt_token_ids(),
                0,
                target_hidden,
            )?;
            program.read_logits_into(stream, 1, target_logits_mut(logits, 1))?;
        }

        Ok(Qwen35ResidentMtpGenerationSession {
            control,
            program,
            stream,
            logits,
            target_hidden,
            maximum_new_tokens: request.max_new_tokens,
            stop_ids,
            next_position: prompt_tokens,
            started: false,
            proposal_ready: false,
            queued: std::array::from_fn(|_| None),
            queue_start: 0,
            queue_len: 0,
            visible_generated: 0,
            native_prefill_tokens,
            greedy,
            stats: ResidentMtpGenerationStats::default(),
        })
    }

    /// Complete target, MTP layer, and mirrored-cache device bytes.
    pub const fn device_owner_bytes(&self) -> usize {
        self.program.layout().arena_bytes()
    }

    /// Fixed page-locked staging and reversible-state bytes.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes() + self.target_hidden.num_bytes()
    }

    /// Maximum target/MTP context admitted by the pinned snapshot.
    pub const fn context_capacity(&self) -> usize {
        self.program.layout().context_capacity()
    }

    /// Fixed host page-table and physical-owner inventory bytes.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.program.kv_host_owner_bytes()
    }

    /// CUDA context shared by every retained owner.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Complete owner exposed for transaction qualification.
    pub const fn qualification_program(&self) -> &Qwen35ResidentMtpProgram {
        &self.program
    }

    #[cfg(feature = "qualification")]
    /// Stable device and pinned-host addresses retained across requests.
    pub fn qualification_addresses(&self) -> Vec<usize> {
        self.program
            .base_addresses()
            .into_iter()
            .map(|address| address as usize)
            .chain([
                self.logits.as_ptr().addr(),
                self.target_hidden.as_ptr().addr(),
            ])
            .collect()
    }
}

impl Qwen35ResidentMtpGenerationSession<'_> {
    /// Exact prompt token IDs selected by the admitted frontend.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.control.prompt_token_ids()
    }

    /// Tokens already returned to the streaming caller.
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.control.generated_token_ids()[..self.visible_generated]
    }

    /// Prompt tokens processed by native target and matching MTP tiles.
    pub const fn native_prefill_tokens(&self) -> usize {
        self.native_prefill_tokens
    }

    /// Terminal state after every already-executed output becomes visible.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        (self.queue_len == 0)
            .then(|| self.control.finish_reason())
            .flatten()
    }

    /// Exact verification and acceptance counters.
    pub const fn stats(&self) -> ResidentMtpGenerationStats {
        self.stats
    }

    /// Returns one streaming token, draining a completed speculative round first.
    pub fn step(&mut self) -> EngineResult<GenerationStep> {
        if self.queue_len != 0 {
            return self.take_queued();
        }
        if self.control.finish_reason().is_some() {
            return Err(EngineError::generation(
                "cannot step Qwen3.5 MTP generation after it finished",
            ));
        }
        if !self.started {
            return self.start_anchor();
        }
        let remaining = self
            .maximum_new_tokens
            .checked_sub(self.control.generated_token_ids().len())
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP budget underflows"))?;
        if remaining == 1 {
            return self.run_final_target_step();
        }
        self.run_speculative_round(remaining)?;
        self.take_queued()
    }

    /// Converts one completely drained terminal request into its output.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        if self.queue_len != 0 {
            return Err(EngineError::generation(
                "cannot take Qwen3.5 MTP output before queued steps are drained",
            ));
        }
        self.control.into_output()
    }

    fn start_anchor(&mut self) -> EngineResult<GenerationStep> {
        let step = self.control.accept_logits(target_logits(self.logits, 0))?;
        self.started = true;
        self.visible_generated = 1;
        if step.finish_reason.is_none() {
            let position = self
                .next_position
                .checked_sub(1)
                .ok_or_else(|| EngineError::generation("Qwen3.5 MTP anchor underflows"))?;
            self.seed_proposal(step.token_id, position)?;
        }
        Ok(step)
    }

    fn seed_proposal(&mut self, token: u32, position: usize) -> EngineResult<()> {
        self.stage_draft(token, position)?;
        self.program.replay_draft(self.stream, 1)?;
        self.program
            .read_logits_into(self.stream, 1, draft_logits_mut(self.logits))?;
        self.proposal_ready = true;
        Ok(())
    }

    fn continue_proposal(&mut self, token: u32, position: usize) -> EngineResult<()> {
        self.stage_draft(token, position)?;
        self.program.replay_continue_draft(self.stream, 1)?;
        self.program
            .read_logits_into(self.stream, 1, draft_logits_mut(self.logits))
    }

    fn stage_draft(&mut self, token: u32, position: usize) -> EngineResult<()> {
        let position = u32::try_from(position)
            .map_err(|_| EngineError::generation("Qwen3.5 MTP position exceeds u32"))?;
        let (cosine, sine) = text_rope(position);
        self.program.stage_mtp_embeddings(self.stream, &[token])?;
        self.program
            .load_decode_state(self.stream, &[position], &[0], &cosine, &sine)
    }

    fn run_final_target_step(&mut self) -> EngineResult<GenerationStep> {
        let anchor = *self
            .control
            .generated_token_ids()
            .last()
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP final step has no anchor"))?;
        self.verify_target(&[anchor])?;
        let step = self.control.accept_logits(target_logits(self.logits, 0))?;
        self.finish_target_transaction(&[anchor], 1)?;
        self.realign(&[step.token_id], true)?;
        self.next_position = self
            .next_position
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP position overflows"))?;
        self.stats.verification_routes[0] += 1;
        self.stats.verified_outputs += 1;
        self.proposal_ready = false;
        self.visible_generated += 1;
        Ok(step)
    }

    fn run_speculative_round(&mut self, remaining: usize) -> EngineResult<()> {
        if !self.proposal_ready {
            return Err(EngineError::generation(
                "Qwen3.5 MTP speculative round has no aligned proposal",
            ));
        }
        let extent = DRAFT_WINDOW.min(remaining - 1);
        let mut drafts = [0u32; DRAFT_WINDOW];
        let mut draft_laws: [Option<SamplingDistribution>; DRAFT_WINDOW] =
            std::array::from_fn(|_| None);
        for draft in 0..extent {
            let draft_token = if self.greedy {
                self.control
                    .propose_logits(draft_logits(self.logits), &drafts[..draft])?
                    .token_id
            } else {
                let law = self
                    .control
                    .sampling_distribution(draft_logits(self.logits), &drafts[..draft])?;
                let token = self.control.draw_distribution(&law)?;
                draft_laws[draft] = Some(law);
                token
            };
            drafts[draft] = draft_token;
            self.stats.draft_proposals += 1;
            if draft + 1 < extent {
                self.continue_proposal(draft_token, self.next_position + draft)?;
            }
        }

        let anchor = *self
            .control
            .generated_token_ids()
            .last()
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP round has no anchor"))?;
        let mut inputs = [0u32; VERIFY_ROWS];
        inputs[0] = anchor;
        inputs[1..extent + 1].copy_from_slice(&drafts[..extent]);
        self.verify_target(&inputs[..extent + 1])?;
        let (committed, accepted) = if self.greedy {
            self.decide_greedy_round(&drafts[..extent])?
        } else {
            self.decide_sampled_round(&drafts[..extent], &draft_laws[..extent])?
        };
        if committed == 0 {
            return Err(EngineError::generation(
                "Qwen3.5 MTP verification committed no output",
            ));
        }

        self.finish_target_transaction(&inputs[..extent + 1], committed)?;
        let mut outputs = [0u32; VERIFY_ROWS];
        for (index, step) in self.queued[..committed].iter().enumerate() {
            outputs[index] = step.as_ref().expect("committed MTP step exists").token_id;
        }
        let terminal = self.queued[committed - 1]
            .as_ref()
            .expect("committed MTP step exists")
            .finish_reason
            .is_some();
        self.realign(&outputs[..committed], terminal)?;
        self.next_position = self
            .next_position
            .checked_add(committed)
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP position overflows"))?;
        self.queue_start = 0;
        self.queue_len = committed;
        self.stats.verification_routes[extent] += 1;
        self.stats.accepted_drafts += accepted;
        self.stats.verified_outputs += committed;
        self.proposal_ready = !terminal;
        Ok(())
    }

    fn decide_greedy_round(&mut self, drafts: &[u32]) -> EngineResult<(usize, usize)> {
        let mut committed = 0;
        let mut accepted = 0;
        for (row, &draft_token) in drafts.iter().enumerate() {
            let step = self
                .control
                .accept_logits(target_logits(self.logits, row))?;
            let matches = step.token_id == draft_token;
            self.queued[committed] = Some(step);
            committed += 1;
            if matches {
                accepted += 1;
            }
            let terminal = self.queued[committed - 1]
                .as_ref()
                .expect("committed MTP step exists")
                .finish_reason
                .is_some();
            if terminal || !matches {
                break;
            }
        }
        if accepted == drafts.len() && self.control.finish_reason().is_none() {
            self.queued[committed] = Some(
                self.control
                    .accept_logits(target_logits(self.logits, drafts.len()))?,
            );
            committed += 1;
        }
        Ok((committed, accepted))
    }

    fn decide_sampled_round(
        &mut self,
        drafts: &[u32],
        draft_laws: &[Option<SamplingDistribution>],
    ) -> EngineResult<(usize, usize)> {
        let mut target_laws: [Option<SamplingDistribution>; VERIFY_ROWS] =
            std::array::from_fn(|_| None);
        for row in 0..=drafts.len() {
            target_laws[row] = Some(self.control.sampling_distribution(
                target_logits(self.logits, row),
                &drafts[..row.min(drafts.len())],
            )?);
        }
        let mut acceptance_units = [0.0f64; DRAFT_WINDOW];
        let mut residual_units = [0.0f64; DRAFT_WINDOW];
        for row in 0..drafts.len() {
            acceptance_units[row] = self.control.random_unit();
            residual_units[row] = self.control.random_unit();
        }
        let bonus_unit = self.control.random_unit();
        let target_laws = target_laws[..drafts.len() + 1]
            .iter()
            .map(|law| {
                law.as_ref()
                    .ok_or_else(|| EngineError::generation("Qwen3.5 sampled target law is missing"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let draft_laws = draft_laws
            .iter()
            .map(|law| {
                law.as_ref()
                    .ok_or_else(|| EngineError::generation("Qwen3.5 sampled draft law is missing"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let round = decide_sampled_tokens(
            drafts,
            &target_laws,
            &draft_laws,
            &self.stop_ids,
            &acceptance_units[..drafts.len()],
            &residual_units[..drafts.len()],
            bonus_unit,
        )?;
        for (index, &token) in round.token_ids().iter().enumerate() {
            self.queued[index] = Some(self.control.accept_token(token)?);
        }
        Ok((round.token_ids().len(), round.accepted_drafts()))
    }

    fn verify_target(&mut self, inputs: &[u32]) -> EngineResult<()> {
        self.program.target().capture_gdn_slot(self.stream, 0)?;
        self.replay_target_span(inputs)?;
        self.program.read_logits_into(
            self.stream,
            inputs.len(),
            target_logits_mut(self.logits, inputs.len()),
        )?;
        self.program.target().read_final_residual_into(
            self.stream,
            inputs.len(),
            &mut self.target_hidden[..inputs.len() * Qwen35_9B::HIDDEN],
        )?;
        Ok(())
    }

    fn finish_target_transaction(&mut self, inputs: &[u32], committed: usize) -> EngineResult<()> {
        if !requires_target_restore(inputs.len(), committed)? {
            return Ok(());
        }
        self.program.target().restore_gdn_slot(self.stream, 0)?;
        self.replay_target_span(&inputs[..committed])?;
        self.program.target().read_final_residual_into(
            self.stream,
            committed,
            &mut self.target_hidden[..committed * Qwen35_9B::HIDDEN],
        )
    }

    fn replay_target_span(&mut self, inputs: &[u32]) -> EngineResult<()> {
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary_values =
            fill_contiguous_rope(self.next_position, inputs.len(), &mut cosine, &mut sine)?;
        self.program.stage_target_verify(
            self.stream,
            inputs,
            0,
            self.next_position,
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        self.program.replay_target_verify(self.stream, inputs.len())
    }

    fn realign(&mut self, outputs: &[u32], prime_only: bool) -> EngineResult<()> {
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let rotary_values =
            fill_contiguous_rope(self.next_position, outputs.len(), &mut cosine, &mut sine)?;
        let hidden_values = outputs.len() * Qwen35_9B::HIDDEN;
        self.program.stage_realign(
            self.stream,
            outputs,
            &self.target_hidden[..hidden_values],
            0,
            self.next_position,
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        self.program
            .replay_realign(self.stream, outputs.len(), prime_only)?;
        if !prime_only {
            self.program
                .read_logits_into(self.stream, 1, draft_logits_mut(self.logits))?;
        }
        Ok(())
    }

    fn take_queued(&mut self) -> EngineResult<GenerationStep> {
        let step = self.queued[self.queue_start]
            .take()
            .ok_or_else(|| EngineError::generation("Qwen3.5 MTP output queue is incomplete"))?;
        self.queue_start += 1;
        self.queue_len -= 1;
        self.visible_generated += 1;
        if self.queue_len == 0 {
            self.queue_start = 0;
        }
        Ok(step)
    }
}

pub(crate) fn prime_qwen35_mtp_prompt(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
    token_ids: &[u32],
    slot: usize,
    target_hidden: &mut [u16],
) -> EngineResult<usize> {
    if token_ids.is_empty() {
        return Err(EngineError::generation(
            "Qwen3.5 MTP generation requires a nonempty prompt",
        ));
    }
    let primed = token_ids.len() - 1;
    let mut cursor = 0;
    let mut native = 0;
    let mut cosine = [0.0f32; 128 * ROTARY_PAIRS];
    let mut sine = [0.0f32; 128 * ROTARY_PAIRS];
    while let Some(rows) = next_prefill_tile(primed - cursor) {
        let rotary_values = fill_contiguous_rope(cursor, rows, &mut cosine, &mut sine)?;
        let target_route = program.stage_target_prefill(
            stream,
            &token_ids[cursor..cursor + rows],
            slot,
            cursor,
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        program.replay_target_prefill(stream, target_route)?;
        let mtp_route = program.stage_prompt_prime(
            stream,
            &token_ids[cursor + 1..cursor + rows + 1],
            slot,
            cursor,
            &cosine[..rotary_values],
            &sine[..rotary_values],
        )?;
        program.replay_prompt_prime(stream, mtp_route)?;
        cursor += rows;
        native += rows;
    }
    while cursor < primed {
        replay_prompt_target_token(program, stream, token_ids[cursor], slot, cursor)?;
        program.target().read_final_residual_into(
            stream,
            1,
            &mut target_hidden[..Qwen35_9B::HIDDEN],
        )?;
        let position = u32::try_from(cursor)
            .map_err(|_| EngineError::generation("Qwen3.5 prompt position exceeds u32"))?;
        let (row_cosine, row_sine) = text_rope(position);
        program.stage_realign(
            stream,
            &token_ids[cursor + 1..cursor + 2],
            &target_hidden[..Qwen35_9B::HIDDEN],
            slot,
            cursor,
            &row_cosine,
            &row_sine,
        )?;
        program.replay_realign(stream, 1, true)?;
        cursor += 1;
    }
    replay_prompt_target_token(program, stream, token_ids[primed], slot, primed)?;

    Ok(native)
}

fn replay_prompt_target_token(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
    token: u32,
    slot: usize,
    position: usize,
) -> EngineResult<()> {
    let position = u32::try_from(position)
        .map_err(|_| EngineError::generation("Qwen3.5 prompt position exceeds u32"))?;
    let (cosine, sine) = text_rope(position);
    program.stage_target_embeddings(stream, &[token])?;
    program.target().load_slot_routes(stream, &[slot])?;
    program
        .target()
        .load_decode_state(stream, 1, &[position], &cosine, &sine)?;
    program.replay_target(stream, 1)
}

const fn next_prefill_tile(remaining: usize) -> Option<usize> {
    if remaining >= 128 {
        Some(128)
    } else if remaining >= 64 {
        Some(64)
    } else if remaining >= 32 {
        Some(32)
    } else {
        None
    }
}

fn requires_target_restore(verified: usize, committed: usize) -> EngineResult<bool> {
    if !(1..=VERIFY_ROWS).contains(&verified) || committed == 0 || committed > verified {
        return Err(EngineError::generation(format!(
            "Qwen3.5 MTP verified/committed widths {verified}/{committed} are invalid"
        )));
    }
    Ok(committed != verified)
}

fn target_logits(logits: &[u16], row: usize) -> &[u16] {
    let begin = row * Qwen35_9B::VOCAB;
    &logits[begin..begin + Qwen35_9B::VOCAB]
}

fn target_logits_mut(logits: &mut [u16], rows: usize) -> &mut [u16] {
    &mut logits[..rows * Qwen35_9B::VOCAB]
}

fn draft_logits(logits: &[u16]) -> &[u16] {
    let begin = VERIFY_ROWS * Qwen35_9B::VOCAB;
    &logits[begin..begin + Qwen35_9B::VOCAB]
}

fn draft_logits_mut(logits: &mut [u16]) -> &mut [u16] {
    let begin = VERIFY_ROWS * Qwen35_9B::VOCAB;
    &mut logits[begin..begin + Qwen35_9B::VOCAB]
}

#[cfg(test)]
mod tests {
    use super::{next_prefill_tile, requires_target_restore};

    #[test]
    fn prompt_tiling_leaves_only_an_exact_scalar_tail() {
        let cases = [
            (0, None),
            (31, None),
            (32, Some(32)),
            (63, Some(32)),
            (64, Some(64)),
            (127, Some(64)),
            (128, Some(128)),
            (255, Some(128)),
            (262_143, Some(128)),
        ];
        for (remaining, expected) in cases {
            assert_eq!(
                next_prefill_tile(remaining),
                expected,
                "remaining={remaining}"
            );
        }
    }

    #[test]
    fn provisional_target_state_is_kept_only_for_the_complete_verified_prefix() {
        for verified in 1..=4 {
            for committed in 1..=verified {
                assert_eq!(
                    requires_target_restore(verified, committed).unwrap(),
                    committed != verified,
                    "K={verified}, committed={committed}"
                );
            }
        }
        for (verified, committed) in [(0, 0), (1, 0), (4, 5), (5, 1)] {
            assert!(requires_target_restore(verified, committed).is_err());
        }
    }
}
