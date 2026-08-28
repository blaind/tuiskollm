//! Independent whole-model oracle for Qwen3.8 Flash-Next.
//!
//! It composes all 48 layers from checkpoint words without reading a device plane. A cold
//! position-zero token closes every carry to a known state; sequence behavior remains covered by
//! the per-family carry gates and the later external generation authority.

use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::qwen38_flash_next_gdn_moe_layer::{bf16_words, gdn_block_oracle};
use crate::qwen38_flash_next_hyper_connection::{
    grouped_rms_norm_oracle, low_rank_oracle, mixed_oracle, widen,
};
use crate::qwen38_flash_next_layer_oracle::{
    HIDDEN, bracket_oracle, moe_oracle, projection_oracle, router_oracle, write_back,
};
use crate::qwen38_flash_next_qsa_moe_layer::qsa_attention_oracle;
use std::path::Path;
use tuisko_engine::gather_qwen38_flash_next_engram_window;
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextEngramBindings,
    Qwen38FlashNextEngramCarry, Qwen38FlashNextGdnBindings, Qwen38FlashNextLayerHyperConnections,
    Qwen38FlashNextMoeBindings, Qwen38FlashNextSparseAttentionBindings,
    Qwen38FlashNextTextEndpointBindings,
};

type A = Qwen38FlashNext;

const WIDTH: usize = A::HC_WIDTH;
const BRANCHES: usize = A::HC_COUNT;
const VOCAB: usize = <A as Arch>::VOCAB;
const GDN_INPUT_ROWS: usize = A::GDN_INPUT_ROWS;
const VALUE_WIDTH: usize = <A as Arch>::LINEAR_VALUE_HEADS * <A as Arch>::LINEAR_HEAD_DIM;
const QKV_ROWS: usize = A::ATTENTION_QKV_ROWS;
const ATTENTION_COLUMNS: usize = <A as Arch>::NUM_ATTENTION_HEADS * <A as Arch>::HEAD_DIM;
const SLOT_BYTES: usize = 2_764_800;

/// Failure of the whole-model oracle.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextModelOracleError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// A borrowed plane was not the shape the composition reads.
    #[error("Qwen3.8 Flash-Next whole-model oracle failed: {0}")]
    Mismatch(String),
}

type OracleResult<T> = Result<T, Qwen38FlashNextModelOracleError>;

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextModelOracleError {
    Qwen38FlashNextModelOracleError::Mismatch(message.into())
}

/// Everything the composition produced for one token.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextModelOracle {
    /// Token the composition ran.
    pub token: u32,
    /// Complete BF16 vocabulary row, exactly as the head would publish it.
    pub logits: Vec<u16>,
    /// Layers composed.
    pub layers: usize,
    /// Expert selections the routers made across the stack.
    pub expert_selections: usize,
    /// Engram rows the production host hash addressed for this token.
    pub engram_rows: usize,
    /// Largest absolute logit the composition produced.
    pub peak_absolute_logit: f32,
    /// Pre-mixer four-branch stream consumed by MTP.
    pub pre_mixer_stream: Vec<u16>,
}

impl Qwen38FlashNextModelOracle {
    /// Highest-logit token, with ties going to the lower id.
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

    /// The `count` strongest tokens, descending.
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
}

/// Composes the whole model for one token at position zero, in the spec's own algebra.
pub fn qwen38_flash_next_model_oracle(
    root: &Path,
    token: u32,
) -> OracleResult<Qwen38FlashNextModelOracle> {
    let snapshot = CheckpointSnapshot::<Qwen38FlashNext>::open(root)?;
    if token as usize >= VOCAB {
        return Err(mismatch(format!(
            "token {token} is outside the vocabulary 0..{VOCAB}"
        )));
    }

    // The device widens one embedding row across all four residual branches.
    let embedding = Qwen38FlashNextTextEndpointBindings::bind_embedding(&snapshot)?;
    let row = embedding_row(embedding.bytes(), token)?;
    let mut stream = vec![0u16; WIDTH];
    for branch in 0..BRANCHES {
        stream[branch * HIDDEN..(branch + 1) * HIDDEN].copy_from_slice(&row);
    }

    let mut expert_selections = 0usize;
    let mut engram_rows = 0usize;
    for layer in 0..<A as Arch>::LAYERS {
        let hc = Qwen38FlashNextLayerHyperConnections::bind(&snapshot, layer)?.materialize()?;
        let moe = Qwen38FlashNextMoeBindings::bind(&snapshot, layer)?.materialize()?;

        // The sole engram module injects before its attention bracket.
        let stream_in = if layer == A::PLE_LAYER {
            let (injected, rows) = engram_oracle(&snapshot, layer, token, &stream)?;
            engram_rows += rows;
            injected
        } else {
            stream.clone()
        };

        let attention_inject = hc.attention.block_inject.ok_or_else(|| {
            mismatch(format!(
                "layer {layer}'s attention bracket cannot write back"
            ))
        })?;
        let bracket = bracket_oracle(
            &stream_in,
            &hc.attention.hc_norm.words().collect::<Vec<_>>(),
            &hc.attention.input_mix_down.words().collect::<Vec<_>>(),
            &hc.attention.input_mix_up.words().collect::<Vec<_>>(),
            &attention_inject.words().collect::<Vec<_>>(),
        );

        let block_output = if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            let qsa =
                Qwen38FlashNextSparseAttentionBindings::bind(&snapshot, layer)?.materialize()?;
            let qkv = projection_oracle(
                &bracket.mixed,
                &bf16_words(&qsa.qkv_weight_bf16).map_err(|error| mismatch(error.to_string()))?,
                HIDDEN,
                QKV_ROWS,
            );
            let gated = qsa_attention_oracle(&qkv)
                .into_iter()
                .map(f32_to_bf16)
                .collect::<Vec<_>>();
            projection_oracle(
                &gated,
                &qsa.output_weight.words().collect::<Vec<_>>(),
                ATTENTION_COLUMNS,
                HIDDEN,
            )
        } else {
            let gdn = Qwen38FlashNextGdnBindings::bind(&snapshot, layer)?.materialize()?;
            let projected = projection_oracle(
                &bracket.mixed,
                &bf16_words(&gdn.input_weight_bf16).map_err(|error| mismatch(error.to_string()))?,
                HIDDEN,
                GDN_INPUT_ROWS,
            );
            let recurrent = gdn_block_oracle(&bracket.mixed, &projected, &gdn)
                .map_err(|error| mismatch(error.to_string()))?;
            projection_oracle(
                &recurrent,
                &gdn.output_weight.words().collect::<Vec<_>>(),
                VALUE_WIDTH,
                HIDDEN,
            )
        };
        let residual = write_back(&stream_in, &block_output, &bracket.write_gate);

        let mlp_inject = hc
            .mlp
            .block_inject
            .ok_or_else(|| mismatch(format!("layer {layer}'s MLP bracket cannot write back")))?;
        let mlp = bracket_oracle(
            &residual,
            &hc.mlp.hc_norm.words().collect::<Vec<_>>(),
            &hc.mlp.input_mix_down.words().collect::<Vec<_>>(),
            &hc.mlp.input_mix_up.words().collect::<Vec<_>>(),
            &mlp_inject.words().collect::<Vec<_>>(),
        );
        let router = router_oracle(&mlp.mixed, &moe.router_weight.words().collect::<Vec<_>>());
        expert_selections += router.experts.len();

        // Assemble only the ten selected experts, not the 1.4 GiB layer pool.
        let (pool, table, scales) = routed_pool(&moe, &router.experts)?;
        let shared = &moe.shared_expert;
        let moe_output = moe_oracle(
            &mlp.mixed,
            &router,
            &table,
            &pool,
            &scales,
            (
                // `gate_weight` is the scalar sigmoid gate, after the three SwiGLU projections.
                &shared.gate_proj_weight.words().collect::<Vec<_>>(),
                &shared.up_proj_weight.words().collect::<Vec<_>>(),
                &shared.down_proj_weight.words().collect::<Vec<_>>(),
                &shared.gate_weight.words().collect::<Vec<_>>(),
            ),
        );
        stream = write_back(&residual, &moe_output, &mlp.write_gate);
    }

    // The endpoint is the collapsing input mixer followed directly by the LM head.
    let endpoint = Qwen38FlashNextTextEndpointBindings::bind(&snapshot)?.materialize()?;
    let normalized = grouped_rms_norm_oracle(
        &widen(&stream),
        &widen(&endpoint.mixer.hc_norm.words().collect::<Vec<_>>()),
    );
    let widened = widen(&normalized);
    let low_rank = low_rank_oracle(
        &widened,
        &widen(&endpoint.mixer.input_mix_down.words().collect::<Vec<_>>()),
    );
    let mixed = mixed_oracle(
        &widened,
        &widen(&endpoint.mixer.input_mix_up.words().collect::<Vec<_>>()),
        &low_rank,
    );
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

    Ok(Qwen38FlashNextModelOracle {
        token,
        logits,
        layers: <A as Arch>::LAYERS,
        expert_selections,
        engram_rows,
        peak_absolute_logit,
        pre_mixer_stream: stream,
    })
}

/// Composes the engram path after its separately qualified production hash and gather.
fn engram_oracle(
    snapshot: &CheckpointSnapshot<Qwen38FlashNext>,
    layer: usize,
    token: u32,
    stream: &[u16],
) -> OracleResult<(Vec<u16>, usize)> {
    use crate::qwen38_flash_next_ple::{
        conv_oracle, dequant_oracle, gate_activation, gated_oracle, grouped_rms_norm_oracle,
        projection_oracle as ple_projection_oracle,
    };

    let bindings = Qwen38FlashNextEngramBindings::bind(snapshot, layer)?.materialize()?;
    let table = bindings.table()?;
    let mut carry = Qwen38FlashNextEngramCarry::start();
    let mut rows = vec![0i64; A::NGRAM_HEADS];
    let mut codes = vec![0u8; table.token_bytes()];
    gather_qwen38_flash_next_engram_window(table, &mut carry, &[token], &mut rows, &mut codes)
        .map_err(|error| mismatch(error.to_string()))?;

    let scale = f64::from(f32::from_bits(
        u32::from(crate::qwen38_flash_next_ple::TABLE_SCALE_BITS) << 16,
    ));
    let embedding = dequant_oracle(&codes, scale);
    let widened = widen(&embedding);
    let key = ple_projection_oracle(
        &widened,
        &widen(&bindings.key_proj_weight.words().collect::<Vec<_>>()),
        WIDTH,
    );
    let value = ple_projection_oracle(
        &widened,
        &widen(&bindings.value_proj_weight.words().collect::<Vec<_>>()),
        A::PLE_EMBED_DIM,
    );
    let key_normed = grouped_rms_norm_oracle(
        &widen(&key),
        &widen(&bindings.norm_key.words().collect::<Vec<_>>()),
    );
    let query_normed = grouped_rms_norm_oracle(
        &widen(stream),
        &widen(&bindings.norm_query.words().collect::<Vec<_>>()),
    );
    let normed_key = widen(&key_normed);
    let normed_query = widen(&query_normed);
    let activation = (0..BRANCHES)
        .map(|branch| {
            let begin = branch * HIDDEN;
            gate_activation(
                &normed_key[begin..begin + HIDDEN],
                &normed_query[begin..begin + HIDDEN],
            )
        })
        .collect::<Vec<_>>();
    let gated = gated_oracle(&activation, &widen(&value));
    let gated_normed = grouped_rms_norm_oracle(
        &widen(&gated),
        &widen(&bindings.norm_conv.words().collect::<Vec<_>>()),
    );

    // A cold slot leaves only the current dilated-convolution tap.
    let weights = widen(&bindings.convolution_weight.words().collect::<Vec<_>>());
    let gated_wide = widen(&gated);
    let normed_wide = widen(&gated_normed);
    let taps = crate::qwen38_flash_next_ple::CONV_TAPS;
    let injected = (0..WIDTH)
        .map(|channel| {
            let delta = conv_oracle(
                &weights[channel * taps..(channel + 1) * taps],
                [0.0, 0.0, 0.0, normed_wide[channel]],
                gated_wide[channel],
            );

            f32_to_bf16(bf16_to_f32(stream[channel]) + bf16_to_f32(delta))
        })
        .collect();

    Ok((injected, A::NGRAM_HEADS))
}

/// Assembles selected expert slots, their table, and their scalar triples.
fn routed_pool(
    moe: &tuisko_model::MaterializedQwen38FlashNextMoe<'_>,
    selected: &[u16],
) -> OracleResult<(Vec<u8>, Vec<u32>, Vec<f32>)> {
    let mut pool = Vec::with_capacity(selected.len() * SLOT_BYTES);
    let mut table = vec![0u32; A::NUM_EXPERTS];
    for (slot, &expert) in selected.iter().enumerate() {
        let expert = expert as usize;
        let source = moe.experts.experts.get(expert).ok_or_else(|| {
            mismatch(format!(
                "router selected expert {expert}, which has no plane"
            ))
        })?;
        pool.extend_from_slice(source.down_weight_e2m1);
        pool.extend_from_slice(source.gate_weight_e2m1);
        pool.extend_from_slice(source.up_weight_e2m1);
        for extent in [source.gate_up_scale, source.down_scale] {
            pool.extend_from_slice(
                moe.experts
                    .scale_e4m3_swizzled
                    .get(extent.offset..extent.offset + extent.bytes)
                    .ok_or_else(|| {
                        mismatch(format!(
                            "expert {expert} names a scale extent outside its pool"
                        ))
                    })?,
            );
        }
        if pool.len() != (slot + 1) * SLOT_BYTES {
            return Err(mismatch(format!(
                "expert {expert}'s slot image is {} bytes, expected {SLOT_BYTES}",
                pool.len() - slot * SLOT_BYTES
            )));
        }
        table[expert] = slot as u32;
    }

    // Gate and up share the fused source plane's scalar.
    let mut scales = Vec::with_capacity(A::NUM_EXPERTS * 3);
    for expert in &moe.experts.experts {
        scales.extend([
            expert.gate_up_weight_scale_2,
            expert.gate_up_weight_scale_2,
            expert.down_weight_scale_2,
        ]);
    }

    Ok((pool, table, scales))
}

pub(crate) fn embedding_row(source: &[u8], token: u32) -> OracleResult<Vec<u16>> {
    let begin = token as usize * HIDDEN * size_of::<u16>();
    let bytes = source
        .get(begin..begin + HIDDEN * size_of::<u16>())
        .ok_or_else(|| {
            mismatch(format!(
                "token {token}'s embedding row is outside the mapping"
            ))
        })?;

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

/// Prints one composition in the house's diagnostic shape.
pub fn print_qwen38_flash_next_model_oracle(oracle: &Qwen38FlashNextModelOracle) {
    println!("Qwen3.8 Flash-Next whole-model oracle - no device plane read");
    println!("  token                    {}", oracle.token);
    println!("  layers composed          {}", oracle.layers);
    println!("  expert selections        {}", oracle.expert_selections);
    println!("  engram rows hashed       {}", oracle.engram_rows);
    println!(
        "  peak |logit|             {:.4}",
        oracle.peak_absolute_logit
    );
    println!("  argmax                   {}", oracle.argmax());
    for (rank, (token, logit)) in oracle.ranked(8).into_iter().enumerate() {
        println!("    {rank}. {token:>7}  {logit:+.6}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_composition_covers_the_whole_stack_and_both_layer_kinds() {
        let mut gdn = 0usize;
        let mut qsa = 0usize;
        for layer in 0..<A as Arch>::LAYERS {
            if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
                qsa += 1;
            } else {
                gdn += 1;
            }
        }

        assert_eq!((gdn, qsa), (36, 12));
        assert_eq!(gdn + qsa, 48);
        assert_eq!(A::PLE_LAYER, 1);
        // Engram injects ahead of a GDN block.
        assert!(!(A::PLE_LAYER + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL));
    }

    #[test]
    fn the_geometry_this_composition_walks_is_the_targets_own() {
        assert_eq!((WIDTH, HIDDEN, BRANCHES), (10_240, 2_560, 4));
        assert_eq!(VOCAB, 248_320);
        assert_eq!(GDN_INPUT_ROWS, 16_384);
        assert_eq!(VALUE_WIDTH, 6_144);
        // Query/gate rows plus the two 512-row K/V planes form 13,312 rows.
        assert_eq!(
            (
                <A as Arch>::ATTENTION_QUERY_ROWS,
                <A as Arch>::ATTENTION_KV_ROWS
            ),
            (12_288, 512)
        );
        assert_eq!(QKV_ROWS, 13_312);
        assert_eq!(ATTENTION_COLUMNS, 6_144);
        assert_eq!(<A as Arch>::NUM_ATTENTION_HEADS, 24);
        assert_eq!(
            SLOT_BYTES,
            tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES
        );
    }

    #[test]
    fn one_routed_pool_holds_only_the_ten_experts_a_router_named() {
        // Per-round assembly is 27.6 MB instead of the model inventory's 67.9 GB.
        assert_eq!(A::NUM_EXPERTS_PER_TOKEN, 10);
        assert_eq!(A::NUM_EXPERTS_PER_TOKEN * SLOT_BYTES, 27_648_000);
        assert_eq!(
            <A as Arch>::LAYERS * A::NUM_EXPERTS * SLOT_BYTES,
            67_947_724_800
        );
    }
}

#[cfg(test)]
mod device {
    use super::*;
    use std::sync::Arc;
    use tuisko_engine::Qwen38FlashNextResidentModel;
    use tuisko_gpu::CudaContext;

    /// Compares the scalar composition with the resident device route at selection scope.
    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT, an exclusive compute-capability 12.0 device, and several minutes of scalar f64"]
    fn qwen38_flash_next_resident_model_oracle_matches_device_selection()
    -> Result<(), Box<dyn std::error::Error>> {
        let _preflight = crate::device_benchmark::preflight()?;
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT")
            .ok_or("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required for the whole-model oracle")?;
        let root = std::path::Path::new(&root);

        let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let mut model = Qwen38FlashNextResidentModel::from_snapshot(&context, snapshot)?;

        for token in [11u32, 5_003] {
            let started = std::time::Instant::now();
            let oracle = qwen38_flash_next_model_oracle(root, token)?;
            let composed = started.elapsed();
            print_qwen38_flash_next_model_oracle(&oracle);
            println!("  composition wall time    {composed:?}");

            model.reset_state(&stream)?;
            model.reserve_slot(&stream, 0, 1)?;
            model.decode_step(&stream, &[token], &[0], &[0])?;
            let device = model.read_logits(&stream, 1)?.to_vec();
            if device.len() != oracle.logits.len() {
                return Err("the device and the oracle published different logit widths".into());
            }

            let mut peak = 0.0f32;
            let mut worst = 0.0f32;
            let mut worst_token = 0usize;
            for (index, (&left, &right)) in device.iter().zip(&oracle.logits).enumerate() {
                let left = f32::from_bits(u32::from(left) << 16);
                let right = f32::from_bits(u32::from(right) << 16);
                peak = peak.max(left.abs()).max(right.abs());
                if (left - right).abs() > worst {
                    worst = (left - right).abs();
                    worst_token = index;
                }
            }
            let device_argmax = ranked(&device, 8);
            println!("  device ranked candidates");
            for (rank, &(token, logit)) in device_argmax.iter().enumerate() {
                println!("    {rank}. {token:>7}  {logit:+.6}");
            }
            println!(
                "  worst |device - oracle|  {worst:.6} at token {worst_token}, peak |logit| {peak:.4}"
            );

            assert_eq!(
                device_argmax[0].0,
                oracle.argmax(),
                "the device and the whole-model composition selected different tokens for {token}"
            );
            assert!(oracle.peak_absolute_logit.is_finite());
            assert_eq!(oracle.layers, 48);
            assert_eq!(oracle.expert_selections, 48 * 10);
            assert_eq!(oracle.engram_rows, 16);
        }

        Ok(())
    }

    fn ranked(row: &[u16], count: usize) -> Vec<(u32, f32)> {
        let mut ranked = row
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
}
