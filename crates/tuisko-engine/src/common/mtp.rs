//! Target-neutral MTP host coordination shared by every speculative target.

use crate::common::rope::{ROTARY_PAIRS, fill_contiguous_rope};
use crate::{
    EngineError, EngineResult, GenerationSession, GenerationStep, SamplingDistribution,
    speculative_decision,
};

pub(crate) const DRAFT_WINDOW: usize = 3;
pub(crate) const VERIFY_ROWS: usize = DRAFT_WINDOW + 1;
pub(crate) const MAX_NATIVE_PREFILL_TOKENS: usize = 1_024;

/// Exact speculative activity observed by one generation session.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentMtpGenerationStats {
    /// Target verification routes selected for K=1,2,3,4.
    pub verification_routes: [usize; VERIFY_ROWS],
    /// Draft tokens proposed before target verification.
    pub draft_proposals: usize,
    /// Draft tokens licensed by equal target argmax decisions.
    pub accepted_drafts: usize,
    /// Generated tokens committed through target verification routes.
    pub verified_outputs: usize,
}

/// Backward-compatible name for the greedy slice's generation counters.
pub type ResidentMtpGreedyStats = ResidentMtpGenerationStats;

/// Host decision for one exact draft-three sampled MTP transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMtpSampledRound {
    tokens: [u32; VERIFY_ROWS],
    committed: usize,
    accepted: usize,
}

impl ResidentMtpSampledRound {
    /// Target-licensed output prefix, including one correction or bonus when applicable.
    pub fn token_ids(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }

    /// Draft proposals accepted before correction, stop, or full acceptance.
    pub const fn accepted_drafts(&self) -> usize {
        self.accepted
    }
}

pub(crate) fn decide_sampled_tokens(
    drafts: &[u32],
    target_laws: &[&SamplingDistribution],
    draft_laws: &[&SamplingDistribution],
    stop_ids: &[u32],
    acceptance_units: &[f64],
    residual_units: &[f64],
    bonus_unit: f64,
) -> EngineResult<ResidentMtpSampledRound> {
    let extent = drafts.len();
    if !(1..=DRAFT_WINDOW).contains(&extent)
        || target_laws.len() != extent + 1
        || draft_laws.len() != extent
        || acceptance_units.len() != extent
        || residual_units.len() != extent
    {
        return Err(EngineError::generation(format!(
            "sampled MTP round inventory differs: drafts={extent}, target={}, draft={}, acceptance={}, residual={}",
            target_laws.len(),
            draft_laws.len(),
            acceptance_units.len(),
            residual_units.len()
        )));
    }
    let mut round = ResidentMtpSampledRound {
        tokens: [0; VERIFY_ROWS],
        committed: 0,
        accepted: 0,
    };
    for row in 0..extent {
        let decision = speculative_decision(
            drafts[row],
            target_laws[row],
            draft_laws[row],
            acceptance_units[row],
            residual_units[row],
        )?;
        round.tokens[round.committed] = decision.token_id;
        round.committed += 1;
        if !decision.accepted {
            return Ok(round);
        }
        round.accepted += 1;
        if stop_ids.contains(&decision.token_id) {
            return Ok(round);
        }
    }
    round.tokens[round.committed] = target_laws[extent].draw_at(bonus_unit)?;
    round.committed += 1;
    Ok(round)
}

#[cfg(feature = "qualification")]
/// Runs the exact host commit rule for the independent speculative-sampling oracle.
pub fn qualification_decide_sampled_tokens(
    drafts: &[u32],
    target_laws: &[&SamplingDistribution],
    draft_laws: &[&SamplingDistribution],
    stop_ids: [u32; 2],
    acceptance_units: &[f64],
    residual_units: &[f64],
    bonus_unit: f64,
) -> EngineResult<ResidentMtpSampledRound> {
    decide_sampled_tokens(
        drafts,
        target_laws,
        draft_laws,
        &stop_ids,
        acceptance_units,
        residual_units,
        bonus_unit,
    )
}

/// Fixed rotary scratch for one MTP verification or realignment span.
///
/// Both spans are bounded by the exact four-row transaction, so the scratch is
/// stack-resident and never reallocates inside a speculative round.
pub(crate) struct MtpRoundRope {
    cosine: [f32; VERIFY_ROWS * ROTARY_PAIRS],
    sine: [f32; VERIFY_ROWS * ROTARY_PAIRS],
    values: usize,
}

impl MtpRoundRope {
    pub(crate) fn contiguous(first_position: usize, rows: usize) -> EngineResult<Self> {
        let mut cosine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let mut sine = [0.0f32; VERIFY_ROWS * ROTARY_PAIRS];
        let values = fill_contiguous_rope(first_position, rows, &mut cosine, &mut sine)?;
        Ok(Self {
            cosine,
            sine,
            values,
        })
    }

    pub(crate) fn cosine(&self) -> &[f32] {
        &self.cosine[..self.values]
    }

    pub(crate) fn sine(&self) -> &[f32] {
        &self.sine[..self.values]
    }
}

fn target_row(logits: &[u16], row: usize, vocab: usize) -> &[u16] {
    let begin = row * vocab;
    &logits[begin..begin + vocab]
}

/// Runs the greedy acceptance law over one single-slot verification transaction.
///
/// Every draft is licensed only by an equal target argmax decision, so the loop
/// stops at the first mismatch or terminal step and adds the bonus row only when
/// the complete draft window survived.
pub(crate) fn decide_greedy_round(
    control: &mut GenerationSession,
    queued: &mut [Option<GenerationStep>; VERIFY_ROWS],
    logits: &[u16],
    vocab: usize,
    drafts: &[u32],
) -> EngineResult<(usize, usize)> {
    let mut committed = 0;
    let mut accepted = 0;
    for (draft, &draft_token) in drafts.iter().enumerate() {
        let step = control.accept_logits(target_row(logits, draft, vocab))?;
        let matches = step.token_id == draft_token;
        queued[committed] = Some(step);
        committed += 1;
        if matches {
            accepted += 1;
        }
        let terminal = queued[committed - 1]
            .as_ref()
            .expect("committed MTP step exists")
            .finish_reason
            .is_some();
        if terminal || !matches {
            break;
        }
    }
    if accepted == drafts.len() && control.finish_reason().is_none() {
        queued[committed] = Some(control.accept_logits(target_row(logits, drafts.len(), vocab))?);
        committed += 1;
    }
    Ok((committed, accepted))
}

/// Runs the unbiased sampled acceptance law over one single-slot transaction.
///
/// Target laws are built for every verified row first, then the acceptance,
/// residual, and bonus units are drawn in that fixed order; the commit rule
/// itself stays in `decide_sampled_tokens`.
pub(crate) fn decide_sampled_round(
    control: &mut GenerationSession,
    queued: &mut [Option<GenerationStep>; VERIFY_ROWS],
    logits: &[u16],
    vocab: usize,
    stop_ids: &[u32],
    drafts: &[u32],
    draft_laws: &[Option<SamplingDistribution>],
) -> EngineResult<(usize, usize)> {
    if draft_laws.len() != drafts.len() {
        return Err(EngineError::generation(
            "every sampled MTP proposal requires its draft distribution",
        ));
    }
    let mut target_laws: [Option<SamplingDistribution>; VERIFY_ROWS] =
        std::array::from_fn(|_| None);
    for row in 0..=drafts.len() {
        target_laws[row] = Some(control.sampling_distribution(
            target_row(logits, row, vocab),
            &drafts[..row.min(drafts.len())],
        )?);
    }
    let mut acceptance_units = [0.0f64; DRAFT_WINDOW];
    let mut residual_units = [0.0f64; DRAFT_WINDOW];
    for row in 0..drafts.len() {
        acceptance_units[row] = control.random_unit();
        residual_units[row] = control.random_unit();
    }
    let bonus_unit = control.random_unit();
    let target_laws = target_laws[..drafts.len() + 1]
        .iter()
        .enumerate()
        .map(|(row, law)| {
            law.as_ref().ok_or_else(|| {
                EngineError::generation(format!("sampled MTP target row {row} has no distribution"))
            })
        })
        .collect::<EngineResult<Vec<_>>>()?;
    let draft_laws = draft_laws
        .iter()
        .enumerate()
        .map(|(row, law)| {
            law.as_ref().ok_or_else(|| {
                EngineError::generation(format!(
                    "sampled MTP proposal {row} has no draft distribution"
                ))
            })
        })
        .collect::<EngineResult<Vec<_>>>()?;
    let round = decide_sampled_tokens(
        drafts,
        &target_laws,
        &draft_laws,
        stop_ids,
        &acceptance_units[..drafts.len()],
        &residual_units[..drafts.len()],
        bonus_unit,
    )?;
    for (index, &token) in round.token_ids().iter().enumerate() {
        queued[index] = Some(control.accept_token(token)?);
    }
    Ok((round.token_ids().len(), round.accepted_drafts()))
}

pub(crate) const fn next_native_prefill_tile(remaining: usize) -> Option<usize> {
    if remaining >= 1_024 {
        Some(1_024)
    } else if remaining >= 128 {
        Some(128)
    } else if remaining >= 64 {
        Some(64)
    } else if remaining >= 32 {
        Some(32)
    } else {
        None
    }
}

pub(crate) fn require_generation_capacity(
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    context_capacity: usize,
) -> EngineResult<usize> {
    if prompt_tokens == 0 {
        return Err(EngineError::generation(
            "resident MTP generation requires a nonempty prompt",
        ));
    }
    let evaluated = prompt_tokens
        .checked_add(maximum_new_tokens.saturating_sub(1))
        .ok_or_else(|| EngineError::generation("resident MTP token budget overflows"))?;
    if evaluated > context_capacity {
        return Err(EngineError::generation(format!(
            "prompt plus processed MTP generation requires {evaluated} positions, current resident capacity is {context_capacity}"
        )));
    }
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use super::{next_native_prefill_tile, require_generation_capacity};

    #[test]
    fn mtp_prompt_plan_excludes_the_final_target_anchor() {
        for (prompt, expected) in [
            (1, vec![]),
            (32, vec![]),
            (33, vec![32]),
            (65, vec![64]),
            (129, vec![128]),
            (1_025, vec![1_024]),
        ] {
            let mut remaining = prompt - 1;
            let mut plan = Vec::new();
            while let Some(tile) = next_native_prefill_tile(remaining) {
                plan.push(tile);
                remaining -= tile;
            }
            assert_eq!(plan, expected);
            assert!(remaining < 32);
        }
    }

    #[test]
    fn mtp_capacity_counts_only_processed_outputs() {
        require_generation_capacity(220_000, 1, 220_000).unwrap();
        require_generation_capacity(1, 220_000, 220_000).unwrap();
        assert!(require_generation_capacity(220_000, 2, 220_000).is_err());
        assert!(require_generation_capacity(0, 1, 220_000).is_err());
    }
}
