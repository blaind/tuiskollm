//! Independent exact-law oracle for target-plus-MTP speculative sampling.

use std::error::Error;
use tuisko_engine::{
    SamplingDistribution, qualification_decide_sampled_tokens, speculative_accept_probability,
    speculative_residual,
};

const UNIT_LATTICE: f64 = 1.0 / 9_007_199_254_740_992.0;
const SCAN: usize = 4_096;
type SequenceLaw = Vec<(Vec<usize>, f64)>;

/// Exact host evidence for the production speculative-sampling rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculativeSamplingQualification {
    /// Support-overlap cases accepted by the induced-law oracle.
    pub induced_law_cases: usize,
    /// Deliberately biased implementations rejected on their emitted law.
    pub rejected_step_mutants: usize,
    /// Exact draft width covered by the sequence-level conditional oracle.
    pub sequence_window: usize,
    /// Deliberately broken sequence composition rules rejected.
    pub rejected_sequence_mutants: usize,
}

trait ObservedStep {
    fn propose(&self, unit: f64) -> (usize, Vec<f64>);
    fn accept_probability(&self, token: usize, target: &[f64], draft: &[f64]) -> f64;
    fn residual(&self, token: usize, target: &[f64], draft: &[f64]) -> Vec<f64>;
}

struct ProductionStep {
    target: SamplingDistribution,
    draft: SamplingDistribution,
    vocabulary: usize,
}

impl ObservedStep for ProductionStep {
    fn propose(&self, unit: f64) -> (usize, Vec<f64>) {
        (
            self.draft.draw_at(unit).expect("valid oracle draw") as usize,
            densify(&self.draft, self.vocabulary),
        )
    }

    fn accept_probability(&self, token: usize, _target: &[f64], _draft: &[f64]) -> f64 {
        speculative_accept_probability(token as u32, &self.target, &self.draft)
    }

    fn residual(&self, _token: usize, _target: &[f64], _draft: &[f64]) -> Vec<f64> {
        densify(
            &speculative_residual(&self.target, &self.draft).expect("reachable residual"),
            self.vocabulary,
        )
    }
}

#[derive(Clone, Copy)]
enum StepMutant {
    InvertedRatio,
    TargetResidual,
    MismatchedDraft,
}

impl ObservedStep for StepMutant {
    fn propose(&self, unit: f64) -> (usize, Vec<f64>) {
        const DRAFT: [f64; 6] = [0.10, 0.35, 0.25, 0.20, 0.06, 0.04];
        let actual = match self {
            Self::MismatchedDraft => normalize(&[0.10, 0.35, 0.25, 0.0, 0.0, 0.0]),
            _ => DRAFT.to_vec(),
        };
        (inverse_cdf(&actual, unit), DRAFT.to_vec())
    }

    fn accept_probability(&self, token: usize, target: &[f64], draft: &[f64]) -> f64 {
        match self {
            Self::InvertedRatio => (draft[token] / target[token]).min(1.0),
            _ => (target[token] / draft[token]).min(1.0),
        }
    }

    fn residual(&self, _token: usize, target: &[f64], draft: &[f64]) -> Vec<f64> {
        match self {
            Self::TargetResidual => normalize(target),
            _ => residual_dense(target, draft),
        }
    }
}

/// Runs the exact induced-law and sequence-composition oracles.
pub fn qualify_speculative_sampling() -> Result<SpeculativeSamplingQualification, Box<dyn Error>> {
    let cases: [(Vec<f64>, Vec<f64>); 4] = [
        (
            vec![0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
            vec![0.10, 0.35, 0.25, 0.20, 0.10, 0.0],
        ),
        (
            vec![0.60, 0.40, 0.0, 0.0, 0.0, 0.0],
            vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
        ),
        (
            vec![0.20, 0.20, 0.30, 0.30, 0.0, 0.0],
            vec![0.50, 0.50, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            vec![0.10, 0.0, 0.0, 0.40, 0.50, 0.0],
            vec![0.70, 0.30, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    for (target, draft) in &cases {
        let production = ProductionStep {
            target: distribution(target)?,
            draft: distribution(draft)?,
            vocabulary: target.len(),
        };
        assert_induced_law(&production, target, SCAN)?;
    }

    const TARGET: [f64; 6] = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03];
    let mut rejected_step_mutants = 0;
    for mutant in [
        StepMutant::InvertedRatio,
        StepMutant::TargetResidual,
        StepMutant::MismatchedDraft,
    ] {
        let error = assert_induced_law(&mutant, &TARGET, SCAN)
            .expect_err("the independent oracle must reject a biased step")
            .to_string();
        if !error.contains("induced law differs") {
            return Err(format!("step mutant was rejected for the wrong reason: {error}").into());
        }
        rejected_step_mutants += 1;
    }

    let production = ProductionSequence {
        append_after_rejection: false,
    };
    assert_sequence_conditionals(&production, 3, 3)?;
    let broken = ProductionSequence {
        append_after_rejection: true,
    };
    let error = assert_sequence_conditionals(&broken, 3, 3)
        .expect_err("the sequence oracle must reject output after a correction")
        .to_string();
    if !error.contains("sequence law sums") && !error.contains("next emitted token") {
        return Err(format!("sequence mutant was rejected for the wrong reason: {error}").into());
    }

    Ok(SpeculativeSamplingQualification {
        induced_law_cases: cases.len(),
        rejected_step_mutants,
        sequence_window: 3,
        rejected_sequence_mutants: 1,
    })
}

fn distribution(probabilities: &[f64]) -> Result<SamplingDistribution, Box<dyn Error>> {
    let entries = probabilities
        .iter()
        .enumerate()
        .filter_map(|(token, &probability)| {
            (probability > 0.0).then_some((token as u32, probability))
        })
        .collect::<Vec<_>>();
    Ok(SamplingDistribution::qualification_from_probabilities(
        &entries,
    )?)
}

fn densify(distribution: &SamplingDistribution, vocabulary: usize) -> Vec<f64> {
    let mut dense = vec![0.0; vocabulary];
    for (token, probability) in distribution.probabilities() {
        dense[token as usize] = probability;
    }
    dense
}

fn recover_draft_law(
    step: &dyn ObservedStep,
    vocabulary: usize,
    scan: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let at = |unit: f64| step.propose(unit).0;
    let mut boundaries = Vec::new();
    let mut previous = at(0.0);
    for index in 1..=scan {
        let unit = (index as f64 / scan as f64).min(1.0 - UNIT_LATTICE);
        let current = at(unit);
        if current != previous {
            boundaries.push(((index - 1) as f64 / scan as f64, unit));
            previous = current;
        }
    }
    let mut cuts = Vec::with_capacity(boundaries.len());
    for (mut low, mut high) in boundaries {
        let low_token = at(low);
        while high - low > UNIT_LATTICE {
            let middle = low + (high - low) * 0.5;
            if at(middle) == low_token {
                low = middle;
            } else {
                high = middle;
            }
        }
        cuts.push(high);
    }
    let mut mass = vec![0.0; vocabulary];
    let mut cursor = 0.0;
    for cut in cuts {
        let token = at(cursor);
        if token >= vocabulary || mass[token] != 0.0 {
            return Err("proposal is not one contiguous inverse CDF".into());
        }
        mass[token] = cut - cursor;
        cursor = cut;
    }
    let token = at((1.0 - UNIT_LATTICE).max(cursor));
    if token >= vocabulary || mass[token] != 0.0 {
        return Err("proposal final interval is invalid".into());
    }
    mass[token] = 1.0 - cursor;
    Ok(mass)
}

fn assert_induced_law(
    step: &dyn ObservedStep,
    target: &[f64],
    scan: usize,
) -> Result<(), Box<dyn Error>> {
    let actual_draft = recover_draft_law(step, target.len(), scan)?;
    let mut induced = vec![0.0; target.len()];
    for token in 0..target.len() {
        let draft_mass = actual_draft[token];
        if draft_mass == 0.0 {
            continue;
        }
        let probe = actual_draft[..token].iter().sum::<f64>() + draft_mass * 0.5;
        let (drawn, reported_draft) = step.propose(probe);
        if drawn != token {
            return Err("proposal changed inside its recovered interval".into());
        }
        let accept = step.accept_probability(token, target, &reported_draft);
        induced[token] += draft_mass * accept;
        let rejected = draft_mass * (1.0 - accept);
        if rejected > 0.0 {
            let residual = step.residual(token, target, &reported_draft);
            for (emitted, probability) in residual.into_iter().enumerate() {
                induced[emitted] += rejected * probability;
            }
        }
    }
    let budget = 64.0 * f64::EPSILON;
    for (token, (&actual, &expected)) in induced.iter().zip(target).enumerate() {
        if (actual - expected).abs() > budget {
            return Err(format!(
                "induced law differs at token {token}: actual={actual}, expected={expected}"
            )
            .into());
        }
    }
    Ok(())
}

trait SequenceScheme {
    fn target(&self, committed: &[usize]) -> Vec<f64>;
    fn draft(&self, drafted: &[usize]) -> Vec<f64>;
    fn emit(
        &self,
        drafted: &[usize],
        accepted: usize,
        correction: usize,
    ) -> Result<Vec<usize>, Box<dyn Error>>;
}

struct ProductionSequence {
    append_after_rejection: bool,
}

impl SequenceScheme for ProductionSequence {
    fn target(&self, committed: &[usize]) -> Vec<f64> {
        rotate(&[0.50, 0.30, 0.20], committed.iter().sum::<usize>() % 3)
    }

    fn draft(&self, drafted: &[usize]) -> Vec<f64> {
        rotate(&[0.20, 0.50, 0.30], drafted.iter().sum::<usize>() % 3)
    }

    fn emit(
        &self,
        drafted: &[usize],
        accepted: usize,
        correction: usize,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        let targets = (0..=drafted.len())
            .map(|row| distribution(&self.target(&drafted[..row])))
            .collect::<Result<Vec<_>, _>>()?;
        let drafts = (0..drafted.len())
            .map(|row| distribution(&self.draft(&drafted[..row])))
            .collect::<Result<Vec<_>, _>>()?;
        let target_refs = targets.iter().collect::<Vec<_>>();
        let draft_refs = drafts.iter().collect::<Vec<_>>();
        let mut acceptance = vec![0.0; drafted.len()];
        let mut residual = vec![0.0; drafted.len()];
        let mut bonus = 0.0;
        if accepted < drafted.len() {
            acceptance[accepted] = speculative_accept_probability(
                drafted[accepted] as u32,
                target_refs[accepted],
                draft_refs[accepted],
            );
            let law = speculative_residual(target_refs[accepted], draft_refs[accepted])?;
            residual[accepted] = unit_for_token(&law, correction as u32)?;
        } else {
            bonus = unit_for_token(target_refs[drafted.len()], correction as u32)?;
        }
        let drafted = drafted
            .iter()
            .map(|&token| token as u32)
            .collect::<Vec<_>>();
        let round = qualification_decide_sampled_tokens(
            &drafted,
            &target_refs,
            &draft_refs,
            [248_046, 248_044],
            &acceptance,
            &residual,
            bonus,
        )?;
        let mut emitted = round
            .token_ids()
            .iter()
            .map(|&token| token as usize)
            .collect::<Vec<_>>();
        if self.append_after_rejection && accepted < drafted.len() && accepted + 1 < drafted.len() {
            emitted.push(drafted[accepted + 1] as usize);
        }
        Ok(emitted)
    }
}

fn assert_sequence_conditionals(
    scheme: &dyn SequenceScheme,
    vocabulary: usize,
    window: usize,
) -> Result<(), Box<dyn Error>> {
    let law = sequence_law(scheme, vocabulary, window)?;
    let total = law.iter().map(|(_, mass)| mass).sum::<f64>();
    if (total - 1.0).abs() > 1e-9 {
        return Err(format!("sequence law sums to {total}, not 1").into());
    }
    let mut prefixes = Vec::<Vec<usize>>::new();
    for (sequence, _) in &law {
        for length in 0..sequence.len() {
            let prefix = sequence[..length].to_vec();
            if !prefixes.contains(&prefix) {
                prefixes.push(prefix);
            }
        }
    }
    for prefix in prefixes {
        let mut next = vec![0.0; vocabulary];
        for (sequence, mass) in &law {
            if sequence.len() > prefix.len() && sequence[..prefix.len()] == prefix {
                next[sequence[prefix.len()]] += mass;
            }
        }
        let observed = next.iter().sum::<f64>();
        let target = scheme.target(&prefix);
        for token in 0..vocabulary {
            if ((next[token] / observed) - target[token]).abs() > 1e-9 {
                return Err(format!(
                    "next emitted token after {prefix:?} differs at token {token}"
                )
                .into());
            }
        }
    }
    Ok(())
}

fn sequence_law(
    scheme: &dyn SequenceScheme,
    vocabulary: usize,
    window: usize,
) -> Result<SequenceLaw, Box<dyn Error>> {
    let mut law = Vec::<(Vec<usize>, f64)>::new();
    for encoded in 0..vocabulary.pow(window as u32) {
        let mut drafted = Vec::with_capacity(window);
        let mut rest = encoded;
        for _ in 0..window {
            drafted.push(rest % vocabulary);
            rest /= vocabulary;
        }
        let mut surviving = 1.0;
        for row in 0..window {
            surviving *= scheme.draft(&drafted[..row])[drafted[row]];
        }
        if surviving == 0.0 {
            continue;
        }
        for accepted in 0..window {
            let target = scheme.target(&drafted[..accepted]);
            let draft = scheme.draft(&drafted[..accepted]);
            let ratio = (target[drafted[accepted]] / draft[drafted[accepted]]).min(1.0);
            let rejected = surviving * (1.0 - ratio);
            if rejected > 0.0 {
                let residual = residual_dense(&target, &draft);
                for (correction, probability) in residual.into_iter().enumerate() {
                    if probability == 0.0 {
                        continue;
                    }
                    record_sequence(
                        &mut law,
                        scheme.emit(&drafted, accepted, correction)?,
                        rejected * probability,
                    );
                }
            }
            surviving *= ratio;
        }
        let bonus = scheme.target(&drafted);
        for (token, probability) in bonus.into_iter().enumerate() {
            if probability == 0.0 {
                continue;
            }
            record_sequence(
                &mut law,
                scheme.emit(&drafted, window, token)?,
                surviving * probability,
            );
        }
    }
    Ok(law)
}

fn record_sequence(law: &mut Vec<(Vec<usize>, f64)>, sequence: Vec<usize>, mass: f64) {
    if mass == 0.0 {
        return;
    }
    if let Some((_, existing)) = law.iter_mut().find(|(candidate, _)| *candidate == sequence) {
        *existing += mass;
    } else {
        law.push((sequence, mass));
    }
}

fn unit_for_token(distribution: &SamplingDistribution, token: u32) -> Result<f64, Box<dyn Error>> {
    let mut before = 0.0;
    for (candidate, probability) in distribution.probabilities() {
        if candidate == token {
            return Ok(before + probability * 0.5);
        }
        before += probability;
    }
    Err(format!("token {token} is outside the requested residual support").into())
}

fn inverse_cdf(law: &[f64], unit: f64) -> usize {
    let mut cumulative = 0.0;
    for (token, &mass) in law.iter().enumerate() {
        cumulative += mass;
        if cumulative >= unit {
            return token;
        }
    }
    law.len() - 1
}

fn normalize(values: &[f64]) -> Vec<f64> {
    let total = values.iter().sum::<f64>();
    values.iter().map(|value| value / total).collect()
}

fn residual_dense(target: &[f64], draft: &[f64]) -> Vec<f64> {
    normalize(
        &target
            .iter()
            .zip(draft)
            .map(|(target, draft)| (target - draft).max(0.0))
            .collect::<Vec<_>>(),
    )
}

fn rotate(values: &[f64], shift: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| values[(index + shift) % values.len()])
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn speculative_sampling_oracle_accepts_production_and_rejects_mutants() {
        let report = super::qualify_speculative_sampling().unwrap();
        assert_eq!(report.induced_law_cases, 4);
        assert_eq!(report.rejected_step_mutants, 3);
        assert_eq!(report.sequence_window, 3);
        assert_eq!(report.rejected_sequence_mutants, 1);
    }
}
