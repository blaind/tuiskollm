//! Independent source-word oracles for both Qwen3.8-Flash-Next layer shapes.
//!
//! These reuse the qualified hyper-connection and MoE arithmetic while independently composing
//! stage order, plane flow, and represented-value rounding.

use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::qwen38_flash_next_hyper_connection::{
    grouped_rms_norm_oracle, low_rank_oracle, mixed_oracle, through_bf16, widen, write_back_oracle,
    write_gate_oracle,
};
use crate::qwen38_flash_next_moe_experts::{bf16_dot, nvfp4_dot, sigmoid, silu};
use tuisko_model::{Arch, Qwen38FlashNext};

type A = Qwen38FlashNext;

pub(crate) const HIDDEN: usize = <A as Arch>::HIDDEN;
pub(crate) const INTERMEDIATE: usize = <A as Arch>::INTERMEDIATE;
pub(crate) const EXPERTS: usize = A::NUM_EXPERTS;
pub(crate) const TOP_K: usize = A::NUM_EXPERTS_PER_TOKEN;

/// One expert's sealed slot image geometry, mirroring the kernels' own offsets.
pub(crate) const SLOT_BYTES: usize = 2_764_800;
const DOWN_CODE_OFFSET: usize = 0;
const GATE_UP_CODE_OFFSET: usize = HIDDEN * INTERMEDIATE / 2;
const GATE_UP_SCALE_OFFSET: usize = GATE_UP_CODE_OFFSET + 2 * INTERMEDIATE * HIDDEN / 2;
const DOWN_SCALE_OFFSET: usize = GATE_UP_SCALE_OFFSET + 2 * INTERMEDIATE * (HIDDEN / 16);

const _: () = assert!(DOWN_SCALE_OFFSET + HIDDEN * (INTERMEDIATE / 16) == SLOT_BYTES);

/// Everything one gated-residual bracket publishes for one token.
pub(crate) struct BracketOracle {
    /// Four-way folded block input.
    pub mixed: Vec<u16>,
    /// Per-branch scalar write gates, each in `(0, 2)`.
    pub write_gate: Vec<u16>,
}

/// One gated-residual Mix arm for a 10,240-wide token.
pub(crate) fn bracket_oracle(
    stream: &[u16],
    norm_weight: &[u16],
    down: &[u16],
    up: &[u16],
    inject: &[u16],
) -> BracketOracle {
    let row = widen(stream);
    let normalized = grouped_rms_norm_oracle(&row, &widen(norm_weight));
    let normalized_wide = widen(&normalized);
    let low_rank = low_rank_oracle(&normalized_wide, &widen(down));
    let mixed = mixed_oracle(&normalized_wide, &widen(up), &low_rank);
    let write_gate = write_gate_oracle(&normalized_wide, &widen(inject));

    BracketOracle { mixed, write_gate }
}

/// One gated residual's write-back into the raw stream.
pub(crate) fn write_back(stream: &[u16], block_output: &[u16], write_gate: &[u16]) -> Vec<u16> {
    write_back_oracle(stream, block_output, write_gate)
}

/// What the router publishes for one token.
pub(crate) struct RouterOracle {
    /// Selected expert ids, **ascending by expert index**, as the combine walks them.
    pub experts: Vec<u16>,
    /// Renormalized BF16 weights, paired with `experts`.
    pub weights: Vec<u16>,
}

/// Full 512-way FP32 softmax, top-ten selection, and selected-mass renormalization.
/// Ties prefer the lower expert index; final weights round to BF16.
pub(crate) fn router_oracle(mixed: &[u16], router_weight: &[u16]) -> RouterOracle {
    let logits = (0..EXPERTS)
        .map(|expert| {
            let (value, _) = bf16_dot(mixed, router_weight, expert, HIDDEN);
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();

    let widened = logits
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let peak = widened.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = widened
        .iter()
        .map(|&value| (value - peak).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f32>();
    let probabilities = exponentials
        .iter()
        .map(|value| value / total)
        .collect::<Vec<_>>();

    let mut ranked = (0..EXPERTS).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then(left.cmp(&right))
    });
    let mut selected = ranked[..TOP_K].to_vec();
    selected.sort_unstable();

    let mass = selected
        .iter()
        .map(|&expert| probabilities[expert])
        .sum::<f32>();

    RouterOracle {
        experts: selected.iter().map(|&expert| expert as u16).collect(),
        weights: selected
            .iter()
            .map(|&expert| f32_to_bf16(probabilities[expert] / mass))
            .collect(),
    }
}

/// One routed expert's SwiGLU and down projection from its sealed slot image.
pub(crate) fn routed_expert_oracle(
    mixed: &[u16],
    slot_pool: &[u8],
    slot: usize,
    weight_scales_2: &[f32],
    expert: usize,
) -> Vec<u16> {
    // `nvfp4_dot`'s `scale_rows` argument is a swizzle row *offset* added to `row`, not a row
    // count: the gate/up split is expressed by passing `INTERMEDIATE + row` instead.
    let image = &slot_pool[slot * SLOT_BYTES..(slot + 1) * SLOT_BYTES];
    let gate_scale = weight_scales_2[expert * 3];
    let up_scale = weight_scales_2[expert * 3 + 1];
    let down_scale = weight_scales_2[expert * 3 + 2];

    let intermediate = (0..INTERMEDIATE)
        .map(|row| {
            let (gate, _) = nvfp4_dot(
                mixed,
                image,
                GATE_UP_CODE_OFFSET,
                GATE_UP_SCALE_OFFSET,
                row,
                HIDDEN,
                0,
                gate_scale,
            );
            let (up, _) = nvfp4_dot(
                mixed,
                image,
                GATE_UP_CODE_OFFSET,
                GATE_UP_SCALE_OFFSET,
                INTERMEDIATE + row,
                HIDDEN,
                0,
                up_scale,
            );
            f32_to_bf16(silu(gate) * up)
        })
        .collect::<Vec<_>>();

    (0..HIDDEN)
        .map(|row| {
            let (value, _) = nvfp4_dot(
                &intermediate,
                image,
                DOWN_CODE_OFFSET,
                DOWN_SCALE_OFFSET,
                row,
                INTERMEDIATE,
                0,
                down_scale,
            );
            f32_to_bf16(value)
        })
        .collect()
}

/// The always-active shared expert and its scalar sigmoid gate.
pub(crate) fn shared_expert_oracle(
    mixed: &[u16],
    gate_weight: &[u16],
    up_weight: &[u16],
    down_weight: &[u16],
    gate_logit_weight: &[u16],
) -> (Vec<u16>, f32) {
    let intermediate = (0..INTERMEDIATE)
        .map(|row| {
            let (gate, _) = bf16_dot(mixed, gate_weight, row, HIDDEN);
            let (up, _) = bf16_dot(mixed, up_weight, row, HIDDEN);
            f32_to_bf16(silu(gate) * up)
        })
        .collect::<Vec<_>>();
    let output = (0..HIDDEN)
        .map(|row| {
            let (value, _) = bf16_dot(&intermediate, down_weight, row, INTERMEDIATE);
            f32_to_bf16(value)
        })
        .collect();
    let (logit, _) = bf16_dot(mixed, gate_logit_weight, 0, HIDDEN);

    (output, sigmoid(logit))
}

/// Combines routed experts in ascending expert order, then adds the gated shared expert.
pub(crate) fn moe_oracle(
    mixed: &[u16],
    router: &RouterOracle,
    slot_table: &[u32],
    slot_pool: &[u8],
    weight_scales_2: &[f32],
    shared: (&[u16], &[u16], &[u16], &[u16]),
) -> Vec<u16> {
    let mut total = vec![0.0f32; HIDDEN];
    for (rank, &expert) in router.experts.iter().enumerate() {
        let expert = expert as usize;
        let slot = slot_table[expert] as usize;
        let routed = routed_expert_oracle(mixed, slot_pool, slot, weight_scales_2, expert);
        let weight = bf16_to_f32(router.weights[rank]);
        for (accumulator, value) in total.iter_mut().zip(&routed) {
            *accumulator += bf16_to_f32(*value) * weight;
        }
    }

    let (shared_output, shared_gate) =
        shared_expert_oracle(mixed, shared.0, shared.1, shared.2, shared.3);
    total
        .iter()
        .zip(&shared_output)
        .map(|(routed, shared)| f32_to_bf16(routed + bf16_to_f32(*shared) * shared_gate))
        .collect()
}

/// A BF16 projection with FP32 accumulation and one BF16 store rounding.
pub(crate) fn projection_oracle(
    input: &[u16],
    weight: &[u16],
    columns: usize,
    output_rows: usize,
) -> Vec<u16> {
    (0..output_rows)
        .map(|row| {
            let total = weight[row * columns..(row + 1) * columns]
                .iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(bf16_to_f32(w)) * f64::from(bf16_to_f32(x)))
                .sum::<f64>();
            f32_to_bf16(total as f32)
        })
        .collect()
}

/// Rounds an `f64` through the BF16 grid the reference's intermediates carry.
pub(crate) fn round_bf16(value: f64) -> f64 {
    through_bf16(value)
}

#[cfg(test)]
mod tests {
    use super::{
        DOWN_SCALE_OFFSET, EXPERTS, GATE_UP_CODE_OFFSET, GATE_UP_SCALE_OFFSET, HIDDEN,
        INTERMEDIATE, SLOT_BYTES, TOP_K, router_oracle,
    };
    use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
    use crate::qwen38_flash_next_hyper_connection::{BRANCH, BRANCHES, RANK, WIDTH};

    #[test]
    fn the_slot_image_offsets_match_the_kernels_own_extents() {
        assert_eq!(GATE_UP_CODE_OFFSET, 819_200);
        assert_eq!(GATE_UP_SCALE_OFFSET, 2_457_600);
        assert_eq!(DOWN_SCALE_OFFSET, 2_662_400);
        assert_eq!(SLOT_BYTES, 2_764_800);
        assert_eq!(
            SLOT_BYTES,
            tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES
        );
    }

    #[test]
    fn the_geometry_this_oracle_composes_is_the_targets_own() {
        assert_eq!((WIDTH, BRANCH, BRANCHES, RANK), (10_240, 2_560, 4, 320));
        assert_eq!((HIDDEN, INTERMEDIATE), (2_560, 640));
        assert_eq!((EXPERTS, TOP_K), (512, 10));
    }

    #[test]
    fn the_router_selects_ten_ascending_experts_whose_weights_sum_to_one() {
        // A router weight plane whose expert `e` sees a strictly increasing logit, so the
        // selection is the top ten indices and the ordering is unambiguous.
        let mixed = vec![f32_to_bf16(1.0); HIDDEN];
        let mut weight = vec![0u16; EXPERTS * HIDDEN];
        for expert in 0..EXPERTS {
            weight[expert * HIDDEN] = f32_to_bf16(expert as f32 / 512.0);
        }
        let router = router_oracle(&mixed, &weight);

        assert_eq!(router.experts.len(), TOP_K);
        assert_eq!(router.weights.len(), TOP_K);
        assert_eq!(
            router.experts,
            (502..512).map(|expert| expert as u16).collect::<Vec<_>>()
        );
        assert!(router.experts.windows(2).all(|pair| pair[0] < pair[1]));

        let mass = router
            .weights
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .sum::<f32>();
        assert!((mass - 1.0).abs() < 0.02, "renormalized mass was {mass}");
    }

    #[test]
    fn an_exactly_tied_router_breaks_to_the_lower_expert_index() {
        // Every logit identical: the selection must be the first ten indices, deterministically.
        let mixed = vec![f32_to_bf16(1.0); HIDDEN];
        let weight = vec![0u16; EXPERTS * HIDDEN];
        let router = router_oracle(&mixed, &weight);

        assert_eq!(
            router.experts,
            (0..TOP_K).map(|expert| expert as u16).collect::<Vec<_>>()
        );
    }
}
