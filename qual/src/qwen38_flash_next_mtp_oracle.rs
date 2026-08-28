//! Independent f64 oracle for one Qwen3.8 Flash-Next MTP step.
//!
//! It reuses the target's pre-mixer stream and compares grouped with flat hidden normalization.
//! Length one makes attention exact while preserving checkpoint source words.

use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::qwen38_flash_next_gdn_moe_layer::bf16_words;
use crate::qwen38_flash_next_hyper_connection::{
    grouped_rms_norm_oracle, low_rank_oracle, mixed_oracle, widen,
};
use crate::qwen38_flash_next_layer_oracle::{
    HIDDEN, bracket_oracle, projection_oracle, router_oracle, write_back,
};
use crate::qwen38_flash_next_moe_experts::{bf16_dot, sigmoid, silu};
use crate::qwen38_flash_next_qsa_moe_layer::qsa_attention_oracle;
use crate::qwen38_flash_next_resident_model_oracle::{
    Qwen38FlashNextModelOracleError, embedding_row,
};
use std::path::Path;
use tuisko_model::{
    Arch, CheckpointSnapshot, MaterializedQwen38FlashNextMtpLayer, Qwen38FlashNext,
    Qwen38FlashNextMtpBindings, Qwen38FlashNextTextEndpointBindings,
};

type A = Qwen38FlashNext;

const WIDTH: usize = A::HC_WIDTH;
const BRANCHES: usize = A::HC_COUNT;
const VOCAB: usize = <A as Arch>::VOCAB;
const INTERMEDIATE: usize = <A as Arch>::INTERMEDIATE;
const QKV_ROWS: usize = A::ATTENTION_QKV_ROWS;
const ATTENTION_COLUMNS: usize = <A as Arch>::NUM_ATTENTION_HEADS * <A as Arch>::HEAD_DIM;
const EPSILON: f64 = <A as Arch>::RMS_NORM_EPSILON as f64;

type OracleResult<T> = Result<T, Qwen38FlashNextModelOracleError>;

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextModelOracleError {
    Qwen38FlashNextModelOracleError::Mismatch(message.into())
}

/// Reduction used by `mtp.pre_fc_norm_hidden` over the four-branch stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen38FlashNextMtpHiddenNorm {
    /// Normalize each 2,560-wide branch independently.
    Grouped,
    /// Normalize all 10,240 values with one RMS.
    Flat,
}

impl Qwen38FlashNextMtpHiddenNorm {
    /// Stable diagnostic label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Grouped => "grouped (per 2,560)",
            Self::Flat => "flat (over 10,240)",
        }
    }
}

/// One MTP step and its normalization diagnostics.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpOracle {
    /// Hidden normalization law used for this run.
    pub hidden_norm: Qwen38FlashNextMtpHiddenNorm,
    /// Token embedded by input fusion.
    pub next_token: u32,
    /// BF16 logits from the shared target head.
    pub logits: Vec<u16>,
    /// Number of routed experts.
    pub expert_selections: usize,
    /// Largest absolute output logit.
    pub peak_absolute_logit: f32,
    /// Per-branch RMS before normalization.
    pub input_branch_rms: [f64; BRANCHES],
    /// Per-branch RMS after normalization.
    pub normalized_branch_rms: [f64; BRANCHES],
    /// Per-branch RMS after `fc_hidden`.
    pub projected_branch_rms: [f64; BRANCHES],
    /// RMS of the shared `fc_embedding` term.
    pub embedding_term_rms: f64,
}

impl Qwen38FlashNextMtpOracle {
    /// Highest-logit token, breaking ties by token id.
    pub fn argmax(&self) -> u32 {
        let mut best = (0u32, f32::NEG_INFINITY);
        for (token, &bits) in self.logits.iter().enumerate() {
            let value = f32::from_bits(u32::from(bits) << 16);
            if value.total_cmp(&best.1).is_gt() {
                best = (token as u32, value);
            }
        }

        best.0
    }

    /// The `count` strongest tokens in descending order.
    pub fn ranked(&self, count: usize) -> Vec<(u32, f32)> {
        let mut ranked = self
            .logits
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

    /// Softmax entropy in nats.
    pub fn entropy(&self) -> f64 {
        let peak = self
            .logits
            .iter()
            .map(|&bits| f64::from(f32::from_bits(u32::from(bits) << 16)))
            .fold(f64::NEG_INFINITY, f64::max);
        let mass = self
            .logits
            .iter()
            .map(|&bits| (f64::from(f32::from_bits(u32::from(bits) << 16)) - peak).exp())
            .sum::<f64>();

        self.logits
            .iter()
            .map(|&bits| {
                let probability =
                    (f64::from(f32::from_bits(u32::from(bits) << 16)) - peak).exp() / mass;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Composes one MTP step from the target's pre-mixer stream and committed token.
pub fn qwen38_flash_next_mtp_oracle(
    root: &Path,
    stream: &[u16],
    next_token: u32,
    hidden_norm: Qwen38FlashNextMtpHiddenNorm,
) -> OracleResult<Qwen38FlashNextMtpOracle> {
    let snapshot = CheckpointSnapshot::<Qwen38FlashNext>::open(root)?;
    if next_token as usize >= VOCAB {
        return Err(mismatch(format!(
            "token {next_token} is outside the vocabulary 0..{VOCAB}"
        )));
    }
    if stream.len() != WIDTH {
        return Err(mismatch(format!(
            "the draft block reads the target's {WIDTH}-wide pre-mixer stream, not {} values",
            stream.len()
        )));
    }

    let mtp = Qwen38FlashNextMtpBindings::bind(&snapshot)?.materialize()?;
    let [layer] = mtp.layers.as_slice() else {
        return Err(mismatch(format!(
            "the Flash-Next draft block is one layer; this checkpoint binds {}",
            mtp.layers.len()
        )));
    };

    // The embedding term is shared by all four independently projected branches.
    let normalized_hidden = match hidden_norm {
        Qwen38FlashNextMtpHiddenNorm::Grouped => grouped_rms_norm_oracle(
            &widen(stream),
            &widen(&mtp.pre_fc_norm_hidden.words().collect::<Vec<_>>()),
        ),
        Qwen38FlashNextMtpHiddenNorm::Flat => flat_rms_norm_oracle(
            &widen(stream),
            &widen(&mtp.pre_fc_norm_hidden.words().collect::<Vec<_>>()),
        ),
    };

    let embedding = Qwen38FlashNextTextEndpointBindings::bind_embedding(&snapshot)?;
    let normalized_embedding = flat_rms_norm_oracle(
        &widen(&embedding_row(embedding.bytes(), next_token)?),
        &widen(&mtp.pre_fc_norm_embedding.words().collect::<Vec<_>>()),
    );
    let embedding_term = projection_oracle(
        &normalized_embedding,
        &mtp.fc_embedding.words().collect::<Vec<_>>(),
        HIDDEN,
        HIDDEN,
    );

    let fc_hidden = mtp.fc_hidden.words().collect::<Vec<_>>();
    let mut fused = vec![0u16; WIDTH];
    let mut projected_branch_rms = [0.0f64; BRANCHES];
    for (branch, branch_rms) in projected_branch_rms.iter_mut().enumerate() {
        let span = branch * HIDDEN..(branch + 1) * HIDDEN;
        let projected =
            projection_oracle(&normalized_hidden[span.clone()], &fc_hidden, HIDDEN, HIDDEN);

        *branch_rms = rms(&projected);
        for (column, slot) in fused[span].iter_mut().enumerate() {
            *slot =
                f32_to_bf16(bf16_to_f32(projected[column]) + bf16_to_f32(embedding_term[column]));
        }
    }

    let attention_inject = layer
        .attention_hyper_connection
        .block_inject
        .ok_or_else(|| mismatch("the draft's attention bracket cannot write back"))?;
    let bracket = bracket_oracle(
        &fused,
        &layer
            .attention_hyper_connection
            .hc_norm
            .words()
            .collect::<Vec<_>>(),
        &layer
            .attention_hyper_connection
            .input_mix_down
            .words()
            .collect::<Vec<_>>(),
        &layer
            .attention_hyper_connection
            .input_mix_up
            .words()
            .collect::<Vec<_>>(),
        &attention_inject.words().collect::<Vec<_>>(),
    );

    // With one visible key, sparse and dense attention both select that key.
    let qkv = projection_oracle(
        &bracket.mixed,
        &bf16_words(&layer.attention.qkv_weight_bf16)
            .map_err(|error| mismatch(error.to_string()))?,
        HIDDEN,
        QKV_ROWS,
    );
    let gated = qsa_attention_oracle(&qkv)
        .into_iter()
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    let block_output = projection_oracle(
        &gated,
        &layer.attention.output_weight.words().collect::<Vec<_>>(),
        ATTENTION_COLUMNS,
        HIDDEN,
    );
    let residual = write_back(&fused, &block_output, &bracket.write_gate);

    let mlp_inject = layer
        .mlp_hyper_connection
        .block_inject
        .ok_or_else(|| mismatch("the draft's MLP bracket cannot write back"))?;
    let mlp = bracket_oracle(
        &residual,
        &layer
            .mlp_hyper_connection
            .hc_norm
            .words()
            .collect::<Vec<_>>(),
        &layer
            .mlp_hyper_connection
            .input_mix_down
            .words()
            .collect::<Vec<_>>(),
        &layer
            .mlp_hyper_connection
            .input_mix_up
            .words()
            .collect::<Vec<_>>(),
        &mlp_inject.words().collect::<Vec<_>>(),
    );
    let router = router_oracle(
        &mlp.mixed,
        &layer.mlp.router_weight.words().collect::<Vec<_>>(),
    );
    let moe_output = fused_moe_oracle(&mlp.mixed, &router, layer)?;
    let stream_out = write_back(&residual, &moe_output, &mlp.write_gate);

    // MTP has its own collapsing mixer and borrows the target LM head.
    if mtp.mixer.block_inject.is_some() {
        return Err(mismatch(
            "the draft's mixer collapses the stream and must not write back",
        ));
    }
    let normalized = grouped_rms_norm_oracle(
        &widen(&stream_out),
        &widen(&mtp.mixer.hc_norm.words().collect::<Vec<_>>()),
    );
    let widened = widen(&normalized);
    let low_rank = low_rank_oracle(
        &widened,
        &widen(&mtp.mixer.input_mix_down.words().collect::<Vec<_>>()),
    );
    let mixed = mixed_oracle(
        &widened,
        &widen(&mtp.mixer.input_mix_up.words().collect::<Vec<_>>()),
        &low_rank,
    );
    let endpoint = Qwen38FlashNextTextEndpointBindings::bind(&snapshot)?.materialize()?;
    let logits = projection_oracle(
        &mixed,
        &endpoint.lm_head.words().collect::<Vec<_>>(),
        HIDDEN,
        VOCAB,
    );

    let peak_absolute_logit = logits
        .iter()
        .map(|&bits| f32::from_bits(u32::from(bits) << 16).abs())
        .fold(0.0f32, f32::max);
    let mut input_branch_rms = [0.0f64; BRANCHES];
    let mut normalized_branch_rms = [0.0f64; BRANCHES];
    for (branch, (input_rms, normalized_rms)) in input_branch_rms
        .iter_mut()
        .zip(&mut normalized_branch_rms)
        .enumerate()
    {
        let span = branch * HIDDEN..(branch + 1) * HIDDEN;
        *input_rms = rms(&stream[span.clone()]);
        *normalized_rms = rms(&normalized_hidden[span]);
    }

    Ok(Qwen38FlashNextMtpOracle {
        hidden_norm,
        next_token,
        logits,
        expert_selections: router.experts.len(),
        peak_absolute_logit,
        input_branch_rms,
        normalized_branch_rms,
        projected_branch_rms,
        embedding_term_rms: rms(&embedding_term),
    })
}

/// RMSNorm over the entire supplied row, with `(1 + w)` gain.
fn flat_rms_norm_oracle(row: &[f64], weight: &[f64]) -> Vec<u16> {
    let mean = row.iter().map(|value| value * value).sum::<f64>() / row.len() as f64;
    let scale = 1.0 / (mean + EPSILON).sqrt();

    row.iter()
        .zip(weight)
        .map(|(value, gain)| f32_to_bf16((value * scale * (1.0 + gain)) as f32))
        .collect()
}

/// BF16 routed experts plus the shared expert, accumulated by expert id.
fn fused_moe_oracle(
    mixed: &[u16],
    router: &crate::qwen38_flash_next_layer_oracle::RouterOracle,
    layer: &MaterializedQwen38FlashNextMtpLayer<'_>,
) -> OracleResult<Vec<u16>> {
    let pool = &layer.mlp.experts;
    let mut total = vec![0.0f32; HIDDEN];

    for (rank, &expert) in router.experts.iter().enumerate() {
        let (gate_up, down) = pool.expert(expert as usize).ok_or_else(|| {
            mismatch(format!(
                "the draft's router named expert {expert}, outside its {}-expert pool",
                pool.expert_count
            ))
        })?;
        let gate_up = bf16_words(gate_up).map_err(|error| mismatch(error.to_string()))?;
        let down = bf16_words(down).map_err(|error| mismatch(error.to_string()))?;

        // Gate rows precede up rows in `[2 * intermediate, hidden]`.
        let intermediate = (0..INTERMEDIATE)
            .map(|row| {
                let (gate, _) = bf16_dot(mixed, &gate_up, row, HIDDEN);
                let (up, _) = bf16_dot(mixed, &gate_up, INTERMEDIATE + row, HIDDEN);
                f32_to_bf16(silu(gate) * up)
            })
            .collect::<Vec<_>>();
        let weight = bf16_to_f32(router.weights[rank]);

        for (row, accumulator) in total.iter_mut().enumerate() {
            let (value, _) = bf16_dot(&intermediate, &down, row, INTERMEDIATE);
            *accumulator += value * weight;
        }
    }

    let shared = &layer.mlp.shared_expert;
    let gate_weight = shared.gate_proj_weight.words().collect::<Vec<_>>();
    let up_weight = shared.up_proj_weight.words().collect::<Vec<_>>();
    let down_weight = shared.down_proj_weight.words().collect::<Vec<_>>();
    let gate_logit_weight = shared.gate_weight.words().collect::<Vec<_>>();
    let shared_intermediate = (0..A::SHARED_EXPERT_INTERMEDIATE)
        .map(|row| {
            let (gate, _) = bf16_dot(mixed, &gate_weight, row, HIDDEN);
            let (up, _) = bf16_dot(mixed, &up_weight, row, HIDDEN);
            f32_to_bf16(silu(gate) * up)
        })
        .collect::<Vec<_>>();
    let (logit, _) = bf16_dot(mixed, &gate_logit_weight, 0, HIDDEN);
    let shared_gate = sigmoid(logit);

    Ok(total
        .iter()
        .enumerate()
        .map(|(row, routed)| {
            let (value, _) = bf16_dot(
                &shared_intermediate,
                &down_weight,
                row,
                A::SHARED_EXPERT_INTERMEDIATE,
            );
            f32_to_bf16(routed + value * shared_gate)
        })
        .collect())
}

fn rms(values: &[u16]) -> f64 {
    let mean = values
        .iter()
        .map(|&bits| {
            let value = f64::from(bf16_to_f32(bits));
            value * value
        })
        .sum::<f64>()
        / values.len() as f64;

    mean.sqrt()
}

/// Prints one MTP composition.
pub fn print_qwen38_flash_next_mtp_oracle(oracle: &Qwen38FlashNextMtpOracle) {
    println!(
        "Qwen3.8 Flash-Next MTP oracle - pre_fc_norm_hidden read {}",
        oracle.hidden_norm.label()
    );
    println!("  fusion token          {}", oracle.next_token);
    println!("  routed experts        {}", oracle.expert_selections);
    println!("  peak |logit|          {:.4}", oracle.peak_absolute_logit);
    println!("  softmax entropy       {:.4} nats", oracle.entropy());
    println!("  argmax                {}", oracle.argmax());
    println!(
        "  input branch RMS      {:?}",
        oracle
            .input_branch_rms
            .map(|value| (value * 1e4).round() / 1e4)
    );
    println!(
        "  normalized branch RMS {:?}",
        oracle
            .normalized_branch_rms
            .map(|value| (value * 1e4).round() / 1e4)
    );
    println!(
        "  fc_hidden branch RMS  {:?}",
        oracle
            .projected_branch_rms
            .map(|value| (value * 1e4).round() / 1e4)
    );
    println!("  fc_embedding term RMS {:.6}", oracle.embedding_term_rms);
    println!("  strongest tokens");
    for (token, logit) in oracle.ranked(8) {
        println!("    {token:>7}  {logit:>10.4}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38_flash_next_resident_model_oracle::qwen38_flash_next_model_oracle;

    /// Distinguishes grouped and flat normalization using source-backed anchor streams.
    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete Flash-Next \
                checkpoint, and several minutes of scalar f64 for the target composition"]
    fn the_draft_blocks_input_fusion_pins_its_hidden_norm() {
        let root = std::env::var("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").unwrap();
        let root = std::path::Path::new(&root);

        const ANCHORS: [u32; 6] = [11, 1_200, 5_003, 15_017, 40_009, 128_003];

        let mut proposals_agreed = 0usize;
        let mut sharper_flat = 0usize;
        let mut grouped_seen: Vec<[f64; BRANCHES]> = Vec::new();
        let mut flat_seen: Vec<[f64; BRANCHES]> = Vec::new();

        for anchor in ANCHORS {
            let target = qwen38_flash_next_model_oracle(root, anchor).unwrap();
            let next_token = target.argmax();

            assert_eq!(target.pre_mixer_stream.len(), WIDTH);

            let mut branch_rms = [0.0f64; BRANCHES];
            for branch in 0..BRANCHES {
                branch_rms[branch] =
                    rms(&target.pre_mixer_stream[branch * HIDDEN..(branch + 1) * HIDDEN]);
            }
            let spread = branch_rms.iter().copied().fold(f64::MIN, f64::max)
                / branch_rms.iter().copied().fold(f64::MAX, f64::min);

            println!();
            println!("=== anchor {anchor} ===");
            println!("target argmax         {next_token}");
            println!("target peak |logit|   {:.4}", target.peak_absolute_logit);
            println!("input branch spread   {spread:.4}x  {branch_rms:?}");

            let grouped = qwen38_flash_next_mtp_oracle(
                root,
                &target.pre_mixer_stream,
                next_token,
                Qwen38FlashNextMtpHiddenNorm::Grouped,
            )
            .unwrap();
            let flat = qwen38_flash_next_mtp_oracle(
                root,
                &target.pre_mixer_stream,
                next_token,
                Qwen38FlashNextMtpHiddenNorm::Flat,
            )
            .unwrap();

            print_qwen38_flash_next_mtp_oracle(&grouped);
            print_qwen38_flash_next_mtp_oracle(&flat);

            assert_eq!(grouped.logits.len(), VOCAB);
            assert_eq!(flat.logits.len(), VOCAB);

            // The input must separate the two normalization laws.
            assert!(
                spread > 1.05,
                "anchor {anchor}: the four branches carry equal scale, so the two readings \
                 cannot be distinguished"
            );
            assert_ne!(
                grouped.normalized_branch_rms, flat.normalized_branch_rms,
                "anchor {anchor}: the two readings must disagree about what `fc_hidden` sees, \
                 or this fixture cannot separate them"
            );

            if grouped.argmax() == flat.argmax() {
                proposals_agreed += 1;
            }
            if flat.entropy() < grouped.entropy() {
                sharper_flat += 1;
            }
            grouped_seen.push(grouped.normalized_branch_rms);
            flat_seen.push(flat.normalized_branch_rms);
        }

        println!();
        println!(
            "proposals agreeing across the two readings: {proposals_agreed}/{}",
            ANCHORS.len()
        );
        println!(
            "anchors where flat produced the sharper row: {sharper_flat}/{}",
            ANCHORS.len()
        );

        let grouped_swing = branch_swing(&grouped_seen);
        let flat_swing = branch_swing(&flat_seen);

        println!("grouped branch-scale swing across anchors: {grouped_swing:?}");
        println!("flat    branch-scale swing across anchors: {flat_swing:?}");

        let grouped_worst = grouped_swing.iter().copied().fold(f64::MIN, f64::max);
        let flat_worst = flat_swing.iter().copied().fold(f64::MIN, f64::max);

        // Grouped normalization should keep the shared projection's input scale steadier.
        assert!(
            grouped_worst < 1.30,
            "grouped no longer holds the projection's input scale steady ({grouped_worst:.3}x); \
             the reading this pins would not be the one that normalizes what `fc_hidden` reads"
        );
        assert!(
            flat_worst > 1.60,
            "flat no longer swings the projection's input scale ({flat_worst:.3}x), so this \
             fixture set can no longer separate the two readings"
        );
        assert!(
            flat_worst > grouped_worst * 1.5,
            "the two readings must differ in how steady they hold `fc_hidden`'s input, or the \
             source-backed anchors cannot decide between them"
        );
    }

    fn branch_swing(seen: &[[f64; BRANCHES]]) -> [f64; BRANCHES] {
        let mut swing = [0.0f64; BRANCHES];
        for (branch, slot) in swing.iter_mut().enumerate() {
            let highest = seen.iter().map(|row| row[branch]).fold(f64::MIN, f64::max);
            let lowest = seen.iter().map(|row| row[branch]).fold(f64::MAX, f64::min);
            *slot = highest / lowest;
        }

        swing
    }
}
