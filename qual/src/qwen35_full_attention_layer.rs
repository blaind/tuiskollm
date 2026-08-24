//! Source-backed qualification for one Qwen3.5 full-attention layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16,
};
use crate::nvfp4_down::{decode_e2m1, decode_e4m3fn};
use crate::nvfp4_swiglu::encode_e4m3fn;
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen35FullAttentionLayerImmutable, Qwen35FullAttentionLayerObservables,
    Qwen35FullAttentionLayerProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MaterializedModelOptNvfp4Attention,
    MaterializedModelOptNvfp4Mlp, ModelOptNvfp4AttentionBindings, ModelOptNvfp4LinearBindings,
    ModelOptNvfp4MlpBindings, Qwen35_9B,
};

const SOURCE_LAYER: usize = 31;
const GROUP: usize = 16;
const TABLE_STRIDE: usize = 3;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const PHYSICAL_PAGES: usize = MAX_BATCH * TABLE_STRIDE;
const CACHE_POSITIONS: [u32; MAX_BATCH] = [0, 1, 63, 64, 65, 97, 128, 130];
const W4A4_BATCHES: [bool; MAX_BATCH] = [true, false, true, true, true, true, true, true];

/// Failure of the complete source-backed Qwen3.5 attention-layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35FullAttentionLayerQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with an independent represented-value formula.
    #[error("Qwen3.5 full-attention qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35FullAttentionLayerQualification {
    /// Residual and normalization values checked at every exact batch.
    pub boundary_values: usize,
    /// Prepared query and appended BF16 cache values checked independently.
    pub qk_values: usize,
    /// Gate/up activation codes and scales checked on W4A4 routes.
    pub activation_values: usize,
    /// Complete real-source projection and MLP values checked through B=1.
    pub source_values: usize,
    /// Mutable owner values reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and cache values verified unchanged.
    pub inactive_values: usize,
    /// Immutable source/materialized device values proved unchanged.
    pub immutable_values: usize,
    /// Complete one-allocation owner bytes.
    pub arena_bytes: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact represented BF16 cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the owner arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from an accepted represented-value formula.
    pub maximum_absolute_error: f32,
}

struct Fixture {
    key_pages: Vec<u16>,
    value_pages: Vec<u16>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

#[derive(Clone, Copy)]
struct SourceBindings<'a> {
    attention: ModelOptNvfp4AttentionBindings<'a>,
    mlp: ModelOptNvfp4MlpBindings<'a>,
}

struct SourceMaterialized<'a> {
    attention: MaterializedModelOptNvfp4Attention<'a>,
    mlp: MaterializedModelOptNvfp4Mlp<'a>,
}

/// Qualifies source-backed Qwen3.5 layer 31 at every exact decode batch.
pub fn qualify_qwen35_full_attention_layer(
    root: &Path,
) -> Result<Qwen35FullAttentionLayerQualification, Qwen35FullAttentionLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let bindings = SourceBindings {
        attention: ModelOptNvfp4AttentionBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?,
        mlp: ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?,
    };
    let materialized = SourceMaterialized {
        attention: bindings.attention.materialize()?,
        mlp: bindings.mlp.materialize()?,
    };
    let fixture = fixture();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        Qwen35FullAttentionLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 37 {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!(
                "Qwen3.5 owner exposes {} addresses, expected 37",
                stable_addresses.len()
            ),
        ));
    }
    let mut report = Qwen35FullAttentionLayerQualification {
        boundary_values: 0,
        qk_values: 0,
        activation_values: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: program.arena_bytes(),
        weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        padding_bytes: program.arena_bytes()
            - program.resident_weight_bytes()
            - program.cache_bytes()
            - program.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };

    verify_scales(&program, &materialized)?;
    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &materialized,
        &mut report,
    )?;

    for batch in 1..=MAX_BATCH {
        let first_input = make_input(batch, 0);
        prepare_run(&program, &stream, batch, &first_input, &fixture)?;
        program.launch_eager(&stream, batch)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(batch, 1);
        prepare_run(&program, &stream, batch, &input, &fixture)?;
        program.replay(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;

        prepare_run(&program, &stream, batch, &input, &fixture)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;

        verify_boundaries(batch, &input, bindings, &replay, &mut report)?;
        verify_activation_quantization(batch, &materialized, &replay, &mut report)?;
        verify_qk_prepare(batch, bindings, &fixture, &replay, &mut report)?;
        if batch == 1 {
            verify_source_formula(bindings, &materialized, &replay, &mut report)?;
        }
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement_input(batch, &first, &replay)?;
        verify_inactive(batch, &fixture, &replay, &mut report)?;
        verify_inactive(batch, &fixture, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                format!("Qwen3.5 owner addresses changed while qualifying B={batch}"),
            ));
        }
    }

    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &materialized,
        &mut report,
    )?;
    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn fixture() -> Fixture {
    let plane_values =
        PHYSICAL_PAGES * Qwen35_9B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen35_9B::HEAD_DIM;
    let key_pages = (0..plane_values)
        .map(|index| f32_to_bf16([0.0, 0.5, -0.5, 0.25, -0.25][(index + index / 256) % 5]))
        .collect();
    let value_pages = (0..plane_values)
        .map(|index| f32_to_bf16([0.125, -0.125, 0.75, -0.75][(3 * index + 1) & 3]))
        .collect();
    let mut rope_cos = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut rope_sin = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for (token, &position) in CACHE_POSITIONS.iter().enumerate() {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
            let angle = f64::from(position) * frequency;
            let (sin, cos) = angle.sin_cos();
            rope_cos[token * ROTARY_PAIRS + pair] = cos as f32;
            rope_sin[token * ROTARY_PAIRS + pair] = sin as f32;
        }
    }

    Fixture {
        key_pages,
        value_pages,
        rope_cos,
        rope_sin,
    }
}

fn make_input(batch: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..batch * Qwen35_9B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / Qwen35_9B::HIDDEN) & 15]))
        .collect()
}

fn prepare_run(
    program: &Qwen35FullAttentionLayerProgram,
    stream: &CudaStream,
    batch: usize,
    input: &[u16],
    fixture: &Fixture,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    program.load_residual(stream, batch, input)?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    program.load_decode_state(
        stream,
        batch,
        &CACHE_POSITIONS[..batch],
        &fixture.rope_cos[..batch * ROTARY_PAIRS],
        &fixture.rope_sin[..batch * ROTARY_PAIRS],
    )?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

fn verify_scales(
    program: &Qwen35FullAttentionLayerProgram,
    source: &SourceMaterialized<'_>,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let expected_source = [
        source.attention.qkv_input_scale,
        source.attention.qkv_weight_scales_2[0],
        source.attention.qkv_weight_scales_2[1],
        source.attention.qkv_weight_scales_2[2],
        source.attention.output.input_scale,
        source.attention.output.weight_scale_2,
        source.mlp.gate_up_input_scale,
        source.mlp.gate_up_weight_scale_2,
        source.mlp.down_input_scale,
        source.mlp.down_weight_scale_2,
    ]
    .map(f32::to_bits);
    let expected_divisors = [
        source.attention.qkv_input_scale_divisor,
        source.attention.qkv_weight_scale_divisors[0],
        source.attention.qkv_weight_scale_divisors[1],
        source.attention.qkv_weight_scale_divisors[2],
        source.attention.output.input_scale_divisor,
        source.attention.output.weight_scale_divisor,
        source.mlp.gate_up.input_scale_divisor,
        source.mlp.gate_up.weight_scale_divisor,
        source.mlp.down.input_scale_divisor,
        source.mlp.down.weight_scale_divisor,
    ]
    .map(f32::to_bits);

    if program.qualification_source_scales().map(f32::to_bits) != expected_source
        || program.qualification_divisors().map(f32::to_bits) != expected_divisors
    {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            "resident ModelOpt scales differ from the materialized source contract".to_string(),
        ));
    }

    Ok(())
}

fn verify_immutable(
    actual: &Qwen35FullAttentionLayerImmutable,
    source: &SourceMaterialized<'_>,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let input_norm = source.attention.input_norm.words().collect::<Vec<_>>();
    let query_norm = source.attention.query_norm.words().collect::<Vec<_>>();
    let key_norm = source.attention.key_norm.words().collect::<Vec<_>>();
    let post_attention_norm = source
        .attention
        .post_attention_norm
        .words()
        .collect::<Vec<_>>();
    let next_norm = source.mlp.next_norm.words().collect::<Vec<_>>();

    macro_rules! same {
        ($field:ident, $expected:expr) => {{
            let expected = $expected;
            if let Some(index) = actual
                .$field
                .iter()
                .zip(expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "immutable plane `{}` differs at value {index}",
                        stringify!($field)
                    ),
                ));
            }
            report.immutable_values += actual.$field.len();
        }};
    }

    same!(input_norm, &input_norm);
    same!(qkv_weight_codes, &source.attention.qkv_weight_e2m1);
    same!(qkv_weight_scales, &source.attention.qkv_scale_e4m3_swizzled);
    same!(query_norm, &query_norm);
    same!(key_norm, &key_norm);
    same!(output_weight_codes, source.attention.output.weight_e2m1);
    same!(
        output_weight_scales,
        &source.attention.output.scale_e4m3_swizzled
    );
    same!(post_attention_norm, &post_attention_norm);
    same!(gate_weight_codes, source.mlp.gate_up.gate_weight_e2m1);
    same!(up_weight_codes, source.mlp.gate_up.up_weight_e2m1);
    same!(
        gate_up_weight_scales,
        &source.mlp.gate_up.scale_e4m3_swizzled
    );
    same!(down_weight_codes, source.mlp.down.weight_e2m1);
    same!(down_weight_scales, &source.mlp.down.scale_e4m3_swizzled);
    same!(next_norm, &next_norm);

    Ok(())
}

fn verify_boundaries(
    batch: usize,
    input: &[u16],
    sources: SourceBindings<'_>,
    observed: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let input_norm = sources.attention.input_norm.words().collect::<Vec<_>>();
    let post_norm = sources
        .attention
        .post_attention_norm
        .words()
        .collect::<Vec<_>>();
    let next_norm = sources.mlp.next_norm.words().collect::<Vec<_>>();

    for token in 0..batch {
        let begin = token * Qwen35_9B::HIDDEN;
        let end = begin + Qwen35_9B::HIDDEN;
        let mixer_normalized = rms_norm_oracle::<Qwen35_9B>(&input[begin..end], &input_norm);
        compare_bf16(
            "input RMSNorm",
            &observed.mixer_normalized[begin..end],
            &mixer_normalized,
            &mut report.maximum_absolute_error,
        )?;

        let mixer_residual =
            residual_oracle(&input[begin..end], &observed.mixer_branch[begin..end]);
        compare_exact(
            "attention residual",
            &observed.mixer_residual[begin..end],
            &mixer_residual,
        )?;
        let mlp_normalized = rms_norm_oracle::<Qwen35_9B>(&mixer_residual, &post_norm);
        compare_bf16(
            "post-attention RMSNorm",
            &observed.mlp_normalized[begin..end],
            &mlp_normalized,
            &mut report.maximum_absolute_error,
        )?;

        let residual = residual_oracle(
            &observed.mixer_residual[begin..end],
            &observed.mlp_branch[begin..end],
        );
        compare_exact(
            "layer residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        let next = rms_norm_oracle::<Qwen35_9B>(&residual, &next_norm);
        compare_bf16(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.boundary_values += batch * Qwen35_9B::HIDDEN * 5;

    Ok(())
}

fn verify_activation_quantization(
    batch: usize,
    source: &SourceMaterialized<'_>,
    observed: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let code_width = Qwen35_9B::HIDDEN / 2;
    let scale_width = Qwen35_9B::HIDDEN / GROUP;
    if W4A4_BATCHES[batch - 1] {
        for token in 0..batch {
            let begin = token * Qwen35_9B::HIDDEN;
            let (codes, scales) = quantize_oracle(
                &observed.mlp_normalized[begin..begin + Qwen35_9B::HIDDEN],
                source.mlp.gate_up.input_scale_divisor,
            )?;
            compare_exact(
                "gate/up activation codes",
                &observed.gate_up_activation_codes[token * code_width..(token + 1) * code_width],
                &codes,
            )?;
            compare_exact(
                "gate/up activation scales",
                &observed.gate_up_activation_scales[token * scale_width..(token + 1) * scale_width],
                &scales,
            )?;
        }
        report.activation_values += batch * (code_width + scale_width);
    } else if observed
        .gate_up_activation_codes
        .iter()
        .chain(&observed.gate_up_activation_scales)
        .any(|&value| value != BYTE_SENTINEL)
    {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("B={batch} A16 SwiGLU modified W4A4 scratch"),
        ));
    }

    Ok(())
}

fn verify_qk_prepare(
    batch: usize,
    sources: SourceBindings<'_>,
    fixture: &Fixture,
    observed: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let query_norm = sources.attention.query_norm.words().collect::<Vec<_>>();
    let key_norm = sources.attention.key_norm.words().collect::<Vec<_>>();
    let mut expected_key = fixture.key_pages.clone();
    let mut expected_value = fixture.value_pages.clone();

    for (token, &position) in CACHE_POSITIONS.iter().enumerate().take(batch) {
        let qkv_base = token * Qwen35_9B::ATTENTION_QKV_ROWS;
        let cosine = &fixture.rope_cos[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        let sine = &fixture.rope_sin[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        for head in 0..Qwen35_9B::NUM_ATTENTION_HEADS {
            let source = qkv_base + head * 2 * Qwen35_9B::HEAD_DIM;
            let destination = (token * Qwen35_9B::NUM_ATTENTION_HEADS + head) * Qwen35_9B::HEAD_DIM;
            let mut expected = vec![0.0; Qwen35_9B::HEAD_DIM];
            normalize_rotate(
                &observed.qkv[source..source + Qwen35_9B::HEAD_DIM],
                &query_norm,
                cosine,
                sine,
                &mut expected,
            );
            compare_f32(
                "prepared query",
                &observed.query[destination..destination + Qwen35_9B::HEAD_DIM],
                &expected,
                0.002,
                0.003,
                &mut report.maximum_absolute_error,
            )?;
        }

        let position = position as usize;
        let physical_page = token * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
        let key_source = qkv_base + Qwen35_9B::ATTENTION_QUERY_ROWS;
        let value_source = key_source + Qwen35_9B::ATTENTION_KV_ROWS;
        for head in 0..Qwen35_9B::NUM_KV_HEADS {
            let source = key_source + head * Qwen35_9B::HEAD_DIM;
            let mut normalized = vec![0.0; Qwen35_9B::HEAD_DIM];
            normalize_rotate(
                &observed.qkv[source..source + Qwen35_9B::HEAD_DIM],
                &key_norm,
                cosine,
                sine,
                &mut normalized,
            );
            for (dimension, &key_value) in normalized.iter().enumerate() {
                let destination = cache_offset(physical_page, head, position, dimension);
                expected_key[destination] = f32_to_bf16(key_value);
                expected_value[destination] =
                    observed.qkv[value_source + head * Qwen35_9B::HEAD_DIM + dimension];
            }
        }
    }

    compare_exact("BF16 key cache", &observed.key_pages, &expected_key)?;
    compare_exact("BF16 value cache", &observed.value_pages, &expected_value)?;
    report.qk_values +=
        batch * (Qwen35_9B::ATTENTION_OUTPUT_COLUMNS + 2 * Qwen35_9B::ATTENTION_KV_ROWS);

    Ok(())
}

fn normalize_rotate(
    source: &[u16],
    norm: &[u16],
    cosine: &[f32],
    sine: &[f32],
    output: &mut [f32],
) {
    let sum = source
        .iter()
        .map(|&bits| {
            let value = f64::from(bf16_to_f32(bits));
            value * value
        })
        .sum::<f64>();
    let inverse =
        1.0 / (sum / Qwen35_9B::HEAD_DIM as f64 + f64::from(Qwen35_9B::RMS_NORM_EPSILON)).sqrt();
    let normalized = source
        .iter()
        .zip(norm)
        .map(|(&value, &weight)| {
            f64::from(bf16_to_f32(value)) * inverse * (1.0 + f64::from(bf16_to_f32(weight)))
        })
        .collect::<Vec<_>>();
    for dimension in 0..Qwen35_9B::HEAD_DIM {
        output[dimension] = if dimension < ROTARY_PAIRS {
            (normalized[dimension] * f64::from(cosine[dimension])
                - normalized[dimension + ROTARY_PAIRS] * f64::from(sine[dimension]))
                as f32
        } else if dimension < ROTARY_DIM {
            let pair = dimension - ROTARY_PAIRS;
            (normalized[pair] * f64::from(sine[pair])
                + normalized[dimension] * f64::from(cosine[pair])) as f32
        } else {
            normalized[dimension] as f32
        };
    }
}

fn verify_source_formula(
    bindings: SourceBindings<'_>,
    materialized: &SourceMaterialized<'_>,
    observed: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let mut qkv_offset = 0;
    for (role, binding, divisor) in [
        (
            "query/gate projection",
            bindings.attention.query_gate,
            materialized.attention.qkv_weight_scale_divisors[0],
        ),
        (
            "key projection",
            bindings.attention.key,
            materialized.attention.qkv_weight_scale_divisors[1],
        ),
        (
            "value projection",
            bindings.attention.value,
            materialized.attention.qkv_weight_scale_divisors[2],
        ),
    ] {
        let rows = binding.weight.shape()[0] as usize;
        verify_a16_projection(
            role,
            &observed.mixer_normalized[..Qwen35_9B::HIDDEN],
            binding,
            divisor,
            &observed.qkv[qkv_offset..qkv_offset + rows],
            &mut report.maximum_absolute_error,
        )?;
        qkv_offset += rows;
    }
    if qkv_offset != Qwen35_9B::ATTENTION_QKV_ROWS {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("source QKV rows total {qkv_offset}, expected 10240"),
        ));
    }

    let mut gated = vec![0.0f32; Qwen35_9B::ATTENTION_OUTPUT_COLUMNS];
    for query_head in 0..Qwen35_9B::NUM_ATTENTION_HEADS {
        let kv_head = query_head / (Qwen35_9B::NUM_ATTENTION_HEADS / Qwen35_9B::NUM_KV_HEADS);
        for dimension in 0..Qwen35_9B::HEAD_DIM {
            let column = query_head * Qwen35_9B::HEAD_DIM + dimension;
            let cache = cache_offset(0, kv_head, 0, dimension);
            let value = f64::from(bf16_to_f32(observed.value_pages[cache]));
            let gate = f64::from(bf16_to_f32(
                observed.qkv
                    [query_head * 2 * Qwen35_9B::HEAD_DIM + Qwen35_9B::HEAD_DIM + dimension],
            ));
            let expected = value / (1.0 + (-gate).exp());
            gated[column] = expected as f32;
            require_close(
                "paged attention and gate",
                column,
                observed.attention[column],
                expected,
                0.000_05,
                0.000_25,
                &mut report.maximum_absolute_error,
            )?;
            if observed.output_activation[column] != f32_to_bf16(expected as f32) {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!("attention BF16 activation differs at column {column}"),
                ));
            }
        }
    }
    verify_a16_projection(
        "attention output projection",
        &observed.output_activation[..Qwen35_9B::ATTENTION_OUTPUT_COLUMNS],
        bindings.attention.output,
        materialized.attention.output.weight_scale_divisor,
        &observed.mixer_branch[..Qwen35_9B::HIDDEN],
        &mut report.maximum_absolute_error,
    )?;

    let (activation_codes, activation_scales) = quantize_oracle(
        &observed.mlp_normalized[..Qwen35_9B::HIDDEN],
        materialized.mlp.gate_up.input_scale_divisor,
    )?;
    let activation = QuantizedActivation {
        codes: &activation_codes,
        scales: &activation_scales,
        scale_divisor: materialized.mlp.gate_up.input_scale_divisor,
    };
    for row in 0..Qwen35_9B::INTERMEDIATE {
        let gate = nvfp4_dot_w4a4(
            activation,
            bindings.mlp.gate.weight.bytes(),
            bindings.mlp.gate.block_scale.codes(),
            row,
            Qwen35_9B::HIDDEN,
            materialized.mlp.gate_up.weight_scale_divisor,
        )?;
        let up = nvfp4_dot_w4a4(
            activation,
            bindings.mlp.up.weight.bytes(),
            bindings.mlp.up.block_scale.codes(),
            row,
            Qwen35_9B::HIDDEN,
            materialized.mlp.gate_up.weight_scale_divisor,
        )?;
        let gate = f64::from(bf16_to_f32(f32_to_bf16(gate as f32)));
        let up = f64::from(bf16_to_f32(f32_to_bf16(up as f32)));
        let expected = gate / (1.0 + (-gate).exp()) * up;
        require_close(
            "source SwiGLU",
            row,
            bf16_to_f32(observed.swiglu[row]),
            expected,
            0.25,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }
    verify_a16_projection(
        "source down projection",
        &observed.swiglu[..Qwen35_9B::INTERMEDIATE],
        bindings.mlp.down,
        materialized.mlp.down.weight_scale_divisor,
        &observed.mlp_branch[..Qwen35_9B::HIDDEN],
        &mut report.maximum_absolute_error,
    )?;

    report.source_values += Qwen35_9B::ATTENTION_QKV_ROWS
        + Qwen35_9B::ATTENTION_OUTPUT_COLUMNS
        + Qwen35_9B::HIDDEN
        + Qwen35_9B::INTERMEDIATE
        + Qwen35_9B::HIDDEN;

    Ok(())
}

pub(crate) fn verify_a16_projection(
    role: &str,
    activation: &[u16],
    binding: ModelOptNvfp4LinearBindings<'_>,
    weight_scale_divisor: f32,
    actual: &[u16],
    maximum: &mut f32,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let rows = binding.weight.shape()[0] as usize;
    let columns = binding.weight.shape()[1] as usize * 2;
    if rows != actual.len() || columns != activation.len() {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("{role} source geometry differs from the observable plane"),
        ));
    }

    for (row, &actual) in actual.iter().enumerate() {
        let expected = nvfp4_dot_a16(
            activation,
            binding.weight.bytes(),
            binding.block_scale.codes(),
            row,
            columns,
            weight_scale_divisor,
        )?;
        require_close(
            role,
            row,
            bf16_to_f32(actual),
            expected,
            0.25,
            0.025,
            maximum,
        )?;
    }

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct QuantizedActivation<'a> {
    pub(crate) codes: &'a [u8],
    pub(crate) scales: &'a [u8],
    pub(crate) scale_divisor: f32,
}

pub(crate) fn nvfp4_dot_w4a4(
    activation: QuantizedActivation<'_>,
    weights: &[u8],
    scales: &[u8],
    row: usize,
    columns: usize,
    weight_scale_divisor: f32,
) -> Result<f64, Qwen35FullAttentionLayerQualificationError> {
    let groups = columns / GROUP;
    let code_bytes = columns / 2;
    let weight_begin = row * code_bytes;
    let mut sum = 0.0f64;

    for group in 0..groups {
        let activation_scale = decode_scale(activation.scales[group])?;
        let weight_scale = decode_scale(scales[row * groups + group])?;
        let mut group_sum = 0.0f64;
        for pair in 0..GROUP / 2 {
            let activation_pair = activation.codes[group * (GROUP / 2) + pair];
            let weight_pair = weights[weight_begin + group * (GROUP / 2) + pair];
            for nibble in 0..2 {
                let shift = nibble * 4;
                group_sum += f64::from(decode_e2m1((activation_pair >> shift) & 15))
                    * f64::from(decode_e2m1((weight_pair >> shift) & 15));
            }
        }
        sum += group_sum
            * f64::from(activation_scale / activation.scale_divisor)
            * f64::from(weight_scale / weight_scale_divisor);
    }

    Ok(sum)
}

fn nvfp4_dot_a16(
    activation: &[u16],
    weights: &[u8],
    scales: &[u8],
    row: usize,
    columns: usize,
    weight_scale_divisor: f32,
) -> Result<f64, Qwen35FullAttentionLayerQualificationError> {
    let groups = columns / GROUP;
    let code_bytes = columns / 2;
    let weight_begin = row * code_bytes;
    let mut sum = 0.0f64;

    for group in 0..groups {
        let scale = decode_scale(scales[row * groups + group])?;
        let mut group_sum = 0.0f64;
        for column in 0..GROUP {
            let packed = weights[weight_begin + group * (GROUP / 2) + column / 2];
            let code = if column & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            group_sum += f64::from(bf16_to_f32(activation[group * GROUP + column]))
                * f64::from(decode_e2m1(code));
        }
        sum += group_sum * f64::from(scale / weight_scale_divisor);
    }

    Ok(sum)
}

pub(crate) fn quantize_oracle(
    input: &[u16],
    input_scale_divisor: f32,
) -> Result<(Vec<u8>, Vec<u8>), Qwen35FullAttentionLayerQualificationError> {
    if !input.len().is_multiple_of(GROUP) {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!(
                "oracle input width {} is not divisible by {GROUP}",
                input.len()
            ),
        ));
    }
    let groups = input.len() / GROUP;
    let mut codes = vec![0u8; input.len() / 2];
    let mut scales = vec![0u8; groups];

    for group in 0..groups {
        let begin = group * GROUP;
        let maximum = input[begin..begin + GROUP]
            .iter()
            .map(|&value| bf16_to_f32(value).abs())
            .fold(0.0f32, f32::max);
        let scale = encode_e4m3fn(input_scale_divisor * maximum / 6.0).map_err(|error| {
            Qwen35FullAttentionLayerQualificationError::Mismatch(error.to_string())
        })?;
        scales[group] = scale;
        if scale == 0 {
            continue;
        }

        let decoded_scale = decode_scale(scale)?;
        for pair in 0..GROUP / 2 {
            let low = encode_e2m1(
                bf16_to_f32(input[begin + 2 * pair]) * input_scale_divisor / decoded_scale,
            );
            let high = encode_e2m1(
                bf16_to_f32(input[begin + 2 * pair + 1]) * input_scale_divisor / decoded_scale,
            );
            codes[group * (GROUP / 2) + pair] = low | (high << 4);
        }
    }

    Ok((codes, scales))
}

fn encode_e2m1(value: f32) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    let candidates = if value.is_sign_negative() {
        8u8..16
    } else {
        0u8..8
    };

    for code in candidates {
        let distance = (value - decode_e2m1(code)).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    best
}

fn decode_scale(code: u8) -> Result<f32, Qwen35FullAttentionLayerQualificationError> {
    decode_e4m3fn(code)
        .map_err(|error| Qwen35FullAttentionLayerQualificationError::Mismatch(error.to_string()))
}

fn verify_replay(
    batch: usize,
    eager: &Qwen35FullAttentionLayerObservables,
    replay: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "B={batch} graph plane `{}` differs at value {index}",
                        stringify!($field)
                    ),
                ));
            }
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .map(|value| value.to_bits())
                .zip(eager.$field.iter().map(|value| value.to_bits()))
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "B={batch} graph plane `{}` differs at value {index}",
                        stringify!($field)
                    ),
                ));
            }
        };
    }

    same!(mixer_normalized);
    same!(qkv);
    same_f32!(query);
    same!(key_pages);
    same!(value_pages);
    same_f32!(attention);
    same!(output_activation);
    same!(mixer_branch);
    same!(mixer_residual);
    same!(mlp_normalized);
    same!(gate_up_activation_codes);
    same!(gate_up_activation_scales);
    same!(swiglu);
    same!(mlp_branch);
    same!(residual_output);
    same!(next_normalized);
    report.graph_replay_values += observable_values(replay);

    Ok(())
}

fn observable_values(values: &Qwen35FullAttentionLayerObservables) -> usize {
    values.mixer_normalized.len()
        + values.qkv.len()
        + values.query.len()
        + values.key_pages.len()
        + values.value_pages.len()
        + values.attention.len()
        + values.output_activation.len()
        + values.mixer_branch.len()
        + values.mixer_residual.len()
        + values.mlp_normalized.len()
        + values.gate_up_activation_codes.len()
        + values.gate_up_activation_scales.len()
        + values.swiglu.len()
        + values.mlp_branch.len()
        + values.residual_output.len()
        + values.next_normalized.len()
}

fn verify_replacement_input(
    batch: usize,
    first: &Qwen35FullAttentionLayerObservables,
    replay: &Qwen35FullAttentionLayerObservables,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let active = batch * Qwen35_9B::HIDDEN;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("B={batch} graph ignored replacement residual rows"),
        ));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    fixture: &Fixture,
    observed: &Qwen35FullAttentionLayerObservables,
    report: &mut Qwen35FullAttentionLayerQualification,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = batch * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!("B={batch} modified inactive `{}` value", stringify!($field)),
                ));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_f32 {
        ($field:ident, $width:expr) => {{
            let begin = batch * $width;
            if observed.$field[begin..]
                .iter()
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                    format!("B={batch} modified inactive `{}` value", stringify!($field)),
                ));
            }
            observed.$field.len() - begin
        }};
    }

    let mut inactive = 0;
    inactive += sentinel_u16!(mixer_normalized, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(qkv, Qwen35_9B::ATTENTION_QKV_ROWS);
    inactive += sentinel_f32!(query, Qwen35_9B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_f32!(attention, Qwen35_9B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_u16!(output_activation, Qwen35_9B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_u16!(mixer_branch, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(mixer_residual, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(mlp_normalized, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(swiglu, Qwen35_9B::INTERMEDIATE);
    inactive += sentinel_u16!(mlp_branch, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(residual_output, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(next_normalized, Qwen35_9B::HIDDEN);

    let code_width = Qwen35_9B::HIDDEN / 2;
    let scale_width = Qwen35_9B::HIDDEN / GROUP;
    let code_begin = if W4A4_BATCHES[batch - 1] {
        batch * code_width
    } else {
        0
    };
    let scale_begin = if W4A4_BATCHES[batch - 1] {
        batch * scale_width
    } else {
        0
    };
    for (role, values) in [
        (
            "gate/up activation codes",
            &observed.gate_up_activation_codes[code_begin..],
        ),
        (
            "gate/up activation scales",
            &observed.gate_up_activation_scales[scale_begin..],
        ),
    ] {
        if values.iter().any(|&value| value != BYTE_SENTINEL) {
            return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
                format!("B={batch} modified inactive {role}"),
            ));
        }
        inactive += values.len();
    }

    let first_inactive_cache =
        batch * TABLE_STRIDE * Qwen35_9B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen35_9B::HEAD_DIM;
    if observed.key_pages[first_inactive_cache..] != fixture.key_pages[first_inactive_cache..]
        || observed.value_pages[first_inactive_cache..]
            != fixture.value_pages[first_inactive_cache..]
    {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("B={batch} modified an inactive slot cache page"),
        ));
    }
    inactive += 2 * (observed.key_pages.len() - first_inactive_cache);
    report.inactive_values += inactive;

    Ok(())
}

fn verify_no_device_allocation(
    program: &Qwen35FullAttentionLayerProgram,
    stream: &CudaStream,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    program.replay(stream, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            program.replay(stream, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

fn cache_offset(physical_page: usize, head: usize, position: usize, dimension: usize) -> usize {
    Qwen35_9B::HEAD_DIM
        * ((position & (ATTENTION_PAGE_SIZE - 1))
            + ATTENTION_PAGE_SIZE * (head + Qwen35_9B::NUM_KV_HEADS * physical_page))
        + dimension
}

fn residual_oracle(input: &[u16], branch: &[u16]) -> Vec<u16> {
    input
        .iter()
        .zip(branch)
        .map(|(&input, &branch)| f32_to_bf16(bf16_to_f32(input) + bf16_to_f32(branch)))
        .collect()
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!("{role} differs at value {index}"),
        ));
    }

    Ok(())
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            0.25,
            0.025,
            maximum,
        )?;
    }

    Ok(())
}

fn compare_f32(
    role: &str,
    actual: &[f32],
    expected: &[f32],
    absolute: f32,
    relative: f32,
    maximum: &mut f32,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            actual,
            f64::from(expected),
            absolute,
            relative,
            maximum,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    absolute: f32,
    relative: f32,
    maximum: &mut f32,
) -> Result<(), Qwen35FullAttentionLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * relative);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen35FullAttentionLayerQualificationError::Mismatch(
            format!(
                "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            ),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen35FullAttentionLayerQualificationError, SOURCE_LAYER, W4A4_BATCHES,
        qualify_qwen35_full_attention_layer,
    };

    #[test]
    fn source_layer_and_swiglu_routes_are_exact() {
        assert_eq!(SOURCE_LAYER, 31);
        assert_eq!(
            W4A4_BATCHES,
            [true, false, true, true, true, true, true, true]
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer31_matches_complete_oracles_and_graph_replay()
    -> Result<(), Qwen35FullAttentionLayerQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            Qwen35FullAttentionLayerQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_qwen35_full_attention_layer(std::path::Path::new(&root))?;

        assert_eq!(report.boundary_values, 737_280);
        assert_eq!(report.qk_values, 221_184);
        assert_eq!(report.activation_values, 78_336);
        assert_eq!(report.source_values, 34_816);
        assert_eq!(report.weight_bytes, 117_990_400);
        assert_eq!(report.cache_bytes, 6_291_456);
        assert_eq!(report.workspace_bytes, 1_233_088);
        assert_eq!(report.arena_bytes, 125_515_776);
        assert_eq!(report.padding_bytes, 832);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.maximum_absolute_error <= 0.5);

        Ok(())
    }
}
