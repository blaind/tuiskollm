//! Source-backed qualification for one resident full-attention layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, TokenOracle, bf16_to_f32, decode_e4m3fn,
    encode_e4m3fn, f32_to_bf16, quantize_oracle,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, FullAttentionLayerObservables, FullAttentionLayerProgram, MAX_BATCH,
};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, DenseFp8MlpBindings, FullAttentionPostBindings,
    FullAttentionQkvBindings, Qwen38_27B,
};

const SOURCE_LAYER: usize = 63;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];
const TABLE_STRIDE: usize = 3;
const MACRO_PARTITIONS: usize = 4;
const PARTIAL_VALUES: usize = Qwen38_27B::HEAD_DIM + 2;
const MACRO_PARTIAL_FLOATS: usize = PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES / size_of::<f32>();
const ACTIVE_MACRO_PARTIAL_FLOATS: usize =
    MAX_ROWS * Qwen38_27B::NUM_ATTENTION_HEADS * MACRO_PARTITIONS * PARTIAL_VALUES;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const PHYSICAL_PAGES: usize = MAX_BATCH * TABLE_STRIDE;
const CACHE_POSITIONS: [u32; MAX_BATCH] = [0, 1, 63, 64, 65, 97, 128, 130];
const CACHE_CODES: [u8; 9] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20, 0xa0];

/// Failure of the complete source-backed full-attention layer gate.
#[derive(Debug, thiserror::Error)]
pub enum FullAttentionLayerQualificationError {
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
    #[error("full-attention layer qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from one complete source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FullAttentionLayerQualification {
    /// Residual and normalization values checked at every exact route.
    pub boundary_values: usize,
    /// Q/K query and appended cache values checked by a mathematical oracle.
    pub qk_values: usize,
    /// Dynamic E4M3 codes checked bit-exactly at all four quantization seams.
    pub activation_codes: usize,
    /// Dynamic FP32 scales checked bit-exactly at all four quantization seams.
    pub activation_scales: usize,
    /// Real-source projection, attention, and MLP values checked through B=1.
    pub source_values: usize,
    /// Active P4 macro-partial values proved finite and reproduced by replay.
    pub macro_partial_values: usize,
    /// Loaded residual, rotary, page-table, position, and length values checked.
    pub metadata_values: usize,
    /// Complete mutable owner state reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and cache values verified unchanged.
    pub inactive_values: usize,
    /// Immutable tensor-map words checked after every route.
    pub immutable_descriptor_words: usize,
    /// Exact source-backed device weight bytes.
    pub resident_weight_bytes: usize,
    /// Exact represented KV-cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Exact weights, cache, and workspace bytes without padding.
    pub owner_bytes: usize,
    /// Complete arena allocation bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Four address-bound tensor-map descriptor bytes.
    pub descriptor_bytes: usize,
    /// Largest absolute difference from a represented-value or FP64 oracle.
    pub maximum_absolute_error: f32,
}

struct SourcePlanes {
    qkv_weight_codes: Vec<u8>,
    qkv_scales: Vec<u16>,
    output_weight_codes: Vec<u8>,
    output_scales: Vec<u16>,
    input_norm: Vec<u16>,
    query_norm: Vec<u16>,
    key_norm: Vec<u16>,
    post_norm: Vec<u16>,
    gate_up_scales: Vec<u16>,
    down_scales: Vec<u16>,
    next_norm: Vec<u16>,
    key_cache_scale: f32,
    value_cache_scale: f32,
}

struct Fixture {
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

/// Qualifies source-backed late layer 63 at every exact decode and prefill route.
pub fn qualify_full_attention_layer(
    root: &Path,
) -> Result<FullAttentionLayerQualification, FullAttentionLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let qkv = FullAttentionQkvBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let post = FullAttentionPostBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let mlp = DenseFp8MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let sources = source_planes(qkv, post, mlp)?;
    let fixture = fixture();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        FullAttentionLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let stable_descriptors = program.qualification_descriptors(&stream)?;
    if stable_addresses.len() != 51 {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "owner exposes {} addresses, expected 51",
            stable_addresses.len()
        )));
    }
    if program.resident_weight_bytes() != 372_395_008
        || program.cache_bytes() != 3_145_728
        || program.workspace_bytes() != 639_924_416
        || program.owner_bytes() != 1_015_465_152
        || program.arena_bytes() != 1_015_465_984
        || program.descriptor_bytes() != 512
    {
        return Err(FullAttentionLayerQualificationError::Mismatch(
            "owner byte accounting differs from the admitted layout".to_string(),
        ));
    }
    let mut report = FullAttentionLayerQualification {
        boundary_values: 0,
        qk_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        source_values: 0,
        macro_partial_values: 0,
        metadata_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_descriptor_words: 0,
        resident_weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        owner_bytes: program.owner_bytes(),
        arena_bytes: program.arena_bytes(),
        padding_bytes: program.arena_bytes() - program.owner_bytes(),
        descriptor_bytes: program.descriptor_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        let first_input = make_input(rows, 0);
        prepare_run(&program, &stream, rows, &first_input, &fixture)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(rows, 1);
        prepare_run(&program, &stream, rows, &input, &fixture)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;

        prepare_run(&program, &stream, rows, &input, &fixture)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;

        verify_metadata(rows, &input, &fixture, &replay, &mut report)?;
        verify_boundaries(rows, &input, &sources, &replay, &mut report)?;
        verify_quantization(rows, &replay, &mut report)?;
        verify_qk_prepare(rows, &sources, &fixture, &replay, &mut report)?;
        verify_macro_partials(rows, &replay, &mut report)?;
        if rows == 1 {
            verify_source_formula(mlp, &sources, &fixture, &replay, &mut report)?;
        }
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &fixture, &replay, &mut report)?;
        verify_inactive(rows, &fixture, &eager, &mut report)?;
        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying rows={rows}"
            )));
        }
        let descriptors = program.qualification_descriptors(&stream)?;
        if descriptors != stable_descriptors {
            return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                "tensor-map descriptors changed while qualifying rows={rows}"
            )));
        }
        report.immutable_descriptor_words += descriptors.iter().map(Vec::len).sum::<usize>();
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn source_planes(
    qkv: FullAttentionQkvBindings<'_>,
    post: FullAttentionPostBindings<'_>,
    mlp: DenseFp8MlpBindings<'_>,
) -> Result<SourcePlanes, FullAttentionLayerQualificationError> {
    let qkv = qkv.materialize()?;
    Ok(SourcePlanes {
        qkv_weight_codes: qkv.weight_e4m3,
        qkv_scales: little_endian_words(&qkv.scale_bf16)?,
        output_weight_codes: post.output_weight.codes().to_vec(),
        output_scales: post.output_scale.words().collect(),
        input_norm: post.input_norm.words().collect(),
        query_norm: post.query_norm.words().collect(),
        key_norm: post.key_norm.words().collect(),
        post_norm: post.post_attention_norm.words().collect(),
        gate_up_scales: little_endian_words(mlp.gate_up.scale_bf16)?,
        down_scales: mlp.down.scale.words().collect(),
        next_norm: mlp.next_norm.words().collect(),
        key_cache_scale: bf16_to_f32(post.key_cache_scale_bf16),
        value_cache_scale: bf16_to_f32(post.value_cache_scale_bf16),
    })
}

fn fixture() -> Fixture {
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = (0..plane_bytes)
        .map(|index| CACHE_CODES[(index + index / Qwen38_27B::HEAD_DIM) % CACHE_CODES.len()])
        .collect();
    let value_pages = (0..plane_bytes)
        .map(|index| CACHE_CODES[(3 * index + 5) % CACHE_CODES.len()])
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

fn make_input(rows: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..rows * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / Qwen38_27B::HIDDEN) & 15]))
        .collect()
}

fn prepare_run(
    program: &FullAttentionLayerProgram,
    stream: &tuisko_gpu::CudaStream,
    rows: usize,
    input: &[u16],
    fixture: &Fixture,
) -> Result<(), FullAttentionLayerQualificationError> {
    program.load_residual(stream, rows, input)?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    if rows <= MAX_BATCH {
        program.load_decode_state(
            stream,
            rows,
            &CACHE_POSITIONS[..rows],
            &fixture.rope_cos[..rows * ROTARY_PAIRS],
            &fixture.rope_sin[..rows * ROTARY_PAIRS],
        )?;
    } else {
        let (rope_cos, rope_sin) = prefill_rope(rows);
        program.load_prefill_state(stream, rows, &rope_cos, &rope_sin)?;
    }
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    Ok(())
}

fn prefill_rope(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; tokens * ROTARY_PAIRS];
    let mut sine = vec![0.0; tokens * ROTARY_PAIRS];
    for token in 0..tokens {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
            let angle = token as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[token * ROTARY_PAIRS + pair] = cos as f32;
            sine[token * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn route_state(rows: usize, fixture: &Fixture) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    if rows <= MAX_BATCH {
        return (
            CACHE_POSITIONS[..rows].to_vec(),
            fixture.rope_cos[..rows * ROTARY_PAIRS].to_vec(),
            fixture.rope_sin[..rows * ROTARY_PAIRS].to_vec(),
        );
    }
    let positions = (0..rows as u32).collect::<Vec<_>>();
    let (cosine, sine) = prefill_rope(rows);

    (positions, cosine, sine)
}

fn verify_metadata(
    rows: usize,
    input: &[u16],
    fixture: &Fixture,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    compare_exact(
        "residual input",
        &observed.residual_input[..input.len()],
        input,
    )?;
    let (positions, cosine, sine) = route_state(rows, fixture);
    let (observed_cosine, observed_sine, observed_rows, observed_positions, observed_lengths) =
        if rows <= MAX_BATCH {
            (
                &observed.rope_cos,
                &observed.rope_sin,
                &observed.table_rows,
                &observed.cache_positions,
                &observed.lengths,
            )
        } else {
            (
                &observed.prefill_rope_cos,
                &observed.prefill_rope_sin,
                &observed.prefill_table_rows,
                &observed.prefill_cache_positions,
                &observed.prefill_lengths,
            )
        };
    compare_exact("rotary cosine", &observed_cosine[..cosine.len()], &cosine)?;
    compare_exact("rotary sine", &observed_sine[..sine.len()], &sine)?;
    compare_exact("cache positions", &observed_positions[..rows], &positions)?;
    let lengths = positions
        .iter()
        .map(|position| position + 1)
        .collect::<Vec<_>>();
    compare_exact("causal lengths", &observed_lengths[..rows], &lengths)?;
    let table_rows = if rows <= MAX_BATCH {
        (0..rows as u32).collect::<Vec<_>>()
    } else {
        vec![0u32; rows]
    };
    compare_exact("table rows", &observed_rows[..rows], &table_rows)?;
    let block_tables = (0..PHYSICAL_PAGES as u32).collect::<Vec<_>>();
    compare_exact("block tables", &observed.block_tables, &block_tables)?;
    report.metadata_values += input.len() + 2 * cosine.len() + 3 * rows + block_tables.len();

    Ok(())
}

fn verify_boundaries(
    batch: usize,
    input: &[u16],
    sources: &SourcePlanes,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for token in 0..batch {
        let begin = token * hidden;
        let end = begin + hidden;
        let mixer_normalized =
            rms_norm_oracle::<Qwen38_27B>(&input[begin..end], &sources.input_norm);
        compare_bf16_slice(
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
        let mlp_normalized = rms_norm_oracle::<Qwen38_27B>(&mixer_residual, &sources.post_norm);
        compare_bf16_slice(
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
        let next = rms_norm_oracle::<Qwen38_27B>(&residual, &sources.next_norm);
        compare_bf16_slice(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }
    report.boundary_values += batch * hidden * 5;
    Ok(())
}

fn verify_quantization(
    batch: usize,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    for token in 0..batch {
        check_quantized_bf16(
            "QKV",
            token,
            Qwen38_27B::HIDDEN,
            &observed.mixer_normalized,
            &observed.qkv_activation_codes,
            &observed.qkv_activation_scales,
        )?;
        check_quantized_f32(
            "attention output",
            token,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
            &observed.attention,
            &observed.output_activation_codes,
            &observed.output_activation_scales,
        )?;
        check_quantized_bf16(
            "gate/up",
            token,
            Qwen38_27B::HIDDEN,
            &observed.mlp_normalized,
            &observed.gate_up_activation_codes,
            &observed.gate_up_activation_scales,
        )?;
        check_quantized_bf16(
            "down",
            token,
            Qwen38_27B::INTERMEDIATE,
            &observed.swiglu,
            &observed.down_activation_codes,
            &observed.down_activation_scales,
        )?;
    }
    report.activation_codes += batch
        * (2 * Qwen38_27B::HIDDEN
            + Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + Qwen38_27B::INTERMEDIATE);
    report.activation_scales += batch * 4;
    Ok(())
}

fn check_quantized_bf16(
    role: &str,
    token: usize,
    width: usize,
    input: &[u16],
    codes: &[u8],
    scales: &[f32],
) -> Result<(), FullAttentionLayerQualificationError> {
    let begin = token * width;
    let oracle = quantize_oracle(&input[begin..begin + width])
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    check_quantized(role, token, begin, &oracle, codes, scales)
}

fn check_quantized_f32(
    role: &str,
    token: usize,
    width: usize,
    input: &[f32],
    codes: &[u8],
    scales: &[f32],
) -> Result<(), FullAttentionLayerQualificationError> {
    let begin = token * width;
    let oracle = quantize_f32(&input[begin..begin + width])?;
    check_quantized(role, token, begin, &oracle, codes, scales)
}

fn check_quantized(
    role: &str,
    token: usize,
    begin: usize,
    oracle: &TokenOracle,
    codes: &[u8],
    scales: &[f32],
) -> Result<(), FullAttentionLayerQualificationError> {
    if let Some(column) = codes[begin..begin + oracle.codes.len()]
        .iter()
        .zip(&oracle.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "{role} code at token={token}, column={column} differs"
        )));
    }
    if scales[token].to_bits() != oracle.scale.to_bits() {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "{role} scale at token={token} differs"
        )));
    }
    Ok(())
}

fn quantize_f32(input: &[f32]) -> Result<TokenOracle, FullAttentionLayerQualificationError> {
    let maximum = input
        .iter()
        .fold(0.0f32, |current, value| current.max(value.abs()));
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 448.0 };
    // The represented seam publishes `scale` and encodes x*(1/scale). Keeping
    // those two FP32 roundings avoids algebraic reassociation at E4M3 ties.
    let inverse_scale = 1.0 / scale;
    let codes = input
        .iter()
        .map(|&value| encode_e4m3fn(value * inverse_scale))
        .collect::<Result<Vec<_>, _>>()
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    let represented_sum = codes
        .iter()
        .map(|&code| {
            decode_e4m3fn(code)
                .map(f64::from)
                .map_err(FullAttentionLayerQualificationError::Mismatch)
        })
        .sum::<Result<f64, _>>()?;
    Ok(TokenOracle {
        codes,
        scale,
        represented_sum,
    })
}

fn verify_qk_prepare(
    rows: usize,
    sources: &SourcePlanes,
    fixture: &Fixture,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    let mut expected_key = fixture.key_pages.clone();
    let mut expected_value = fixture.value_pages.clone();
    let (positions, rope_cos, rope_sin) = route_state(rows, fixture);
    for (token, &cache_position) in positions.iter().enumerate() {
        let qkv_base = token * Qwen38_27B::ATTENTION_QKV_ROWS;
        let cosine = &rope_cos[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        let sine = &rope_sin[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        for head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let source = qkv_base + head * 2 * Qwen38_27B::HEAD_DIM;
            let destination =
                (token * Qwen38_27B::NUM_ATTENTION_HEADS + head) * Qwen38_27B::HEAD_DIM;
            let mut expected = vec![0.0; Qwen38_27B::HEAD_DIM];
            normalize_rotate(
                &observed.qkv[source..source + Qwen38_27B::HEAD_DIM],
                &sources.query_norm,
                cosine,
                sine,
                &mut expected,
            );
            compare_f32_slice(
                "prepared query",
                &observed.query[destination..destination + Qwen38_27B::HEAD_DIM],
                &expected,
                0.002,
                0.003,
                &mut report.maximum_absolute_error,
            )?;
        }
        let position = cache_position as usize;
        let physical_page = if rows <= MAX_BATCH {
            token * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE
        } else {
            position / ATTENTION_PAGE_SIZE
        };
        let key_source = qkv_base + Qwen38_27B::ATTENTION_QUERY_ROWS;
        let value_source = key_source + Qwen38_27B::ATTENTION_KV_ROWS;
        for head in 0..Qwen38_27B::NUM_KV_HEADS {
            let source = key_source + head * Qwen38_27B::HEAD_DIM;
            let mut normalized = vec![0.0; Qwen38_27B::HEAD_DIM];
            normalize_rotate(
                &observed.qkv[source..source + Qwen38_27B::HEAD_DIM],
                &sources.key_norm,
                cosine,
                sine,
                &mut normalized,
            );
            for (dimension, &normalized_value) in normalized.iter().enumerate() {
                let destination = cache_offset(physical_page, head, position, dimension);
                expected_key[destination] =
                    encode_e4m3fn(normalized_value / sources.key_cache_scale)
                        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
                expected_value[destination] = encode_e4m3fn(
                    bf16_to_f32(
                        observed.qkv[value_source + head * Qwen38_27B::HEAD_DIM + dimension],
                    ) / sources.value_cache_scale,
                )
                .map_err(FullAttentionLayerQualificationError::Mismatch)?;
            }
        }
    }
    compare_exact("key cache", &observed.key_pages, &expected_key)?;
    compare_exact("value cache", &observed.value_pages, &expected_value)?;
    report.qk_values +=
        rows * (Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + 2 * Qwen38_27B::ATTENTION_KV_ROWS);
    Ok(())
}

fn verify_macro_partials(
    rows: usize,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    let active = if rows == MAX_ROWS {
        ACTIVE_MACRO_PARTIAL_FLOATS
    } else {
        0
    };
    if let Some(index) = observed.macro_partials[..active]
        .iter()
        .position(|value| !value.is_finite() || value.to_bits() == F32_SENTINEL_BITS)
    {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "T=1024 macro partial {index} was not produced as a finite value"
        )));
    }
    if let Some(relative) = observed.macro_partials[active..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "rows={rows} modified inactive macro partial {relative}"
        )));
    }
    report.macro_partial_values += active;
    report.inactive_values += MACRO_PARTIAL_FLOATS - active;

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
        .map(|&bits| f64::from(bf16_to_f32(bits)).powi(2))
        .sum::<f64>();
    let inverse =
        1.0 / (sum / Qwen38_27B::HEAD_DIM as f64 + f64::from(Qwen38_27B::RMS_NORM_EPSILON)).sqrt();
    let normalized = source
        .iter()
        .zip(norm)
        .map(|(&value, &weight)| {
            f64::from(bf16_to_f32(value)) * inverse * (1.0 + f64::from(bf16_to_f32(weight)))
        })
        .collect::<Vec<_>>();
    for dimension in 0..Qwen38_27B::HEAD_DIM {
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
    mlp: DenseFp8MlpBindings<'_>,
    sources: &SourcePlanes,
    _fixture: &Fixture,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    let qkv_activation = quantize_oracle(&observed.mixer_normalized[..Qwen38_27B::HIDDEN])
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    verify_fp8_projection(
        "QKV projection",
        &qkv_activation,
        &sources.qkv_weight_codes,
        &sources.qkv_scales,
        &observed.qkv[..Qwen38_27B::ATTENTION_QKV_ROWS],
        Qwen38_27B::HIDDEN,
        &mut report.maximum_absolute_error,
    )?;

    let position = CACHE_POSITIONS[0] as usize;
    debug_assert_eq!(position, 0);
    let mut gated = vec![0.0f32; Qwen38_27B::ATTENTION_OUTPUT_COLUMNS];
    for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
        let kv_head = query_head / (Qwen38_27B::NUM_ATTENTION_HEADS / Qwen38_27B::NUM_KV_HEADS);
        for dimension in 0..Qwen38_27B::HEAD_DIM {
            let column = query_head * Qwen38_27B::HEAD_DIM + dimension;
            let cache = cache_offset(0, kv_head, position, dimension);
            let attention = f64::from(
                decode_e4m3fn(observed.value_pages[cache])
                    .map_err(FullAttentionLayerQualificationError::Mismatch)?,
            ) * f64::from(sources.value_cache_scale);
            let gate = f64::from(bf16_to_f32(
                observed.qkv
                    [query_head * 2 * Qwen38_27B::HEAD_DIM + Qwen38_27B::HEAD_DIM + dimension],
            ));
            let expected = attention / (1.0 + (-gate).exp());
            gated[column] = expected as f32;
            require_f32_close(
                "paged attention and gate",
                column,
                observed.attention[column],
                expected,
                0.000_05,
                0.000_25,
                report,
            )?;
        }
    }
    let output_activation = quantize_f32(&gated)?;
    verify_fp8_projection(
        "attention output projection",
        &output_activation,
        &sources.output_weight_codes,
        &sources.output_scales,
        &observed.mixer_branch[..Qwen38_27B::HIDDEN],
        Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        &mut report.maximum_absolute_error,
    )?;

    let gate_up_activation = quantize_oracle(&observed.mlp_normalized[..Qwen38_27B::HIDDEN])
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    for row in 0..Qwen38_27B::INTERMEDIATE {
        let gate_begin = row * Qwen38_27B::HIDDEN;
        let up_begin = (Qwen38_27B::INTERMEDIATE + row) * Qwen38_27B::HIDDEN;
        let gate = fp8_dot(
            &gate_up_activation,
            &mlp.gate_up.weight_e4m3[gate_begin..gate_begin + Qwen38_27B::HIDDEN],
            sources.gate_up_scales[row],
        )?;
        let up = fp8_dot(
            &gate_up_activation,
            &mlp.gate_up.weight_e4m3[up_begin..up_begin + Qwen38_27B::HIDDEN],
            sources.gate_up_scales[Qwen38_27B::INTERMEDIATE + row],
        )?;
        require_close(
            "source SwiGLU",
            row,
            bf16_to_f32(observed.swiglu[row]),
            gate / (1.0 + (-gate).exp()) * up,
            &mut report.maximum_absolute_error,
        )?;
    }
    let down_activation = quantize_oracle(&observed.swiglu[..Qwen38_27B::INTERMEDIATE])
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    verify_fp8_projection(
        "dense-FP8 down projection",
        &down_activation,
        mlp.down.weight.codes(),
        &sources.down_scales,
        &observed.mlp_branch[..Qwen38_27B::HIDDEN],
        Qwen38_27B::INTERMEDIATE,
        &mut report.maximum_absolute_error,
    )?;

    report.source_values += Qwen38_27B::ATTENTION_QKV_ROWS
        + Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
        + Qwen38_27B::HIDDEN
        + Qwen38_27B::INTERMEDIATE
        + Qwen38_27B::HIDDEN;
    Ok(())
}

fn verify_fp8_projection(
    role: &str,
    activation: &TokenOracle,
    weights: &[u8],
    scales: &[u16],
    actual: &[u16],
    columns: usize,
    maximum: &mut f32,
) -> Result<(), FullAttentionLayerQualificationError> {
    for (row, (&scale, &actual)) in scales.iter().zip(actual).enumerate() {
        let begin = row * columns;
        let expected = fp8_dot(activation, &weights[begin..begin + columns], scale)?;
        require_close(role, row, bf16_to_f32(actual), expected, maximum)?;
    }
    Ok(())
}

fn fp8_dot(
    activation: &TokenOracle,
    weights: &[u8],
    weight_scale: u16,
) -> Result<f64, FullAttentionLayerQualificationError> {
    let sum = activation
        .codes
        .iter()
        .zip(weights)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            Ok::<_, String>(
                sum + f64::from(decode_e4m3fn(activation)?) * f64::from(decode_e4m3fn(weight)?),
            )
        })
        .map_err(FullAttentionLayerQualificationError::Mismatch)?;
    Ok(sum * f64::from(activation.scale) * f64::from(bf16_to_f32(weight_scale)))
}

fn verify_replay(
    rows: usize,
    eager: &FullAttentionLayerObservables,
    replay: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                    "rows={rows} graph plane `{}` differs at value {index}",
                    stringify!($field)
                )));
            }
        };
    }
    same!(residual_input);
    same!(mixer_normalized);
    same!(qkv_activation_codes);
    same!(qkv_activation_scales);
    same!(qkv);
    same!(rope_cos);
    same!(rope_sin);
    same!(block_tables);
    same!(table_rows);
    same!(cache_positions);
    same!(lengths);
    same!(prefill_rope_cos);
    same!(prefill_rope_sin);
    same!(prefill_table_rows);
    same!(prefill_cache_positions);
    same!(prefill_lengths);
    same!(query);
    same!(key_pages);
    same!(value_pages);
    same!(attention);
    same!(macro_partials);
    same!(output_activation_codes);
    same!(output_activation_scales);
    same!(mixer_branch);
    same!(mixer_residual);
    same!(mlp_normalized);
    same!(gate_up_activation_codes);
    same!(gate_up_activation_scales);
    same!(swiglu);
    same!(down_activation_codes);
    same!(down_activation_scales);
    same!(mlp_branch);
    same!(residual_output);
    same!(next_normalized);
    report.graph_replay_values += observable_values();
    Ok(())
}

fn observable_values() -> usize {
    MAX_ROWS
        * (10 * Qwen38_27B::HIDDEN
            + Qwen38_27B::ATTENTION_QKV_ROWS
            + 3 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + 2 * Qwen38_27B::INTERMEDIATE
            + 2 * ROTARY_PAIRS
            + 7)
        + 2 * PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM
        + PHYSICAL_PAGES
        + MACRO_PARTIAL_FLOATS
        + MAX_BATCH * (2 * ROTARY_PAIRS + 3)
}

fn verify_replacement_input(
    rows: usize,
    first: &FullAttentionLayerObservables,
    replay: &FullAttentionLayerObservables,
) -> Result<(), FullAttentionLayerQualificationError> {
    let active = rows * Qwen38_27B::HIDDEN;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "rows={rows} graph ignored replacement input"
        )));
    }
    Ok(())
}

fn verify_inactive(
    rows: usize,
    fixture: &Fixture,
    observed: &FullAttentionLayerObservables,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_u8 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BYTE_SENTINEL)
            {
                return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_f32 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    let mut inactive = 0;
    inactive += sentinel_u16!(mixer_normalized, Qwen38_27B::HIDDEN);
    inactive += sentinel_u8!(qkv_activation_codes, Qwen38_27B::HIDDEN);
    inactive += sentinel_f32!(qkv_activation_scales, 1);
    inactive += sentinel_u16!(qkv, Qwen38_27B::ATTENTION_QKV_ROWS);
    inactive += sentinel_f32!(query, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_f32!(attention, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_u8!(
        output_activation_codes,
        Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
    );
    inactive += sentinel_f32!(output_activation_scales, 1);
    inactive += sentinel_u16!(mixer_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mixer_residual, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mlp_normalized, Qwen38_27B::HIDDEN);
    inactive += sentinel_u8!(gate_up_activation_codes, Qwen38_27B::HIDDEN);
    inactive += sentinel_f32!(gate_up_activation_scales, 1);
    inactive += sentinel_u16!(swiglu, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_u8!(down_activation_codes, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_f32!(down_activation_scales, 1);
    inactive += sentinel_u16!(mlp_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(residual_output, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(next_normalized, Qwen38_27B::HIDDEN);

    let first_inactive_page = if rows <= MAX_BATCH {
        rows * TABLE_STRIDE
    } else {
        rows.div_ceil(ATTENTION_PAGE_SIZE)
    };
    let first_inactive_cache =
        first_inactive_page * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    if observed.key_pages[first_inactive_cache..] != fixture.key_pages[first_inactive_cache..]
        || observed.value_pages[first_inactive_cache..]
            != fixture.value_pages[first_inactive_cache..]
    {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "rows={rows} modified an inactive cache page"
        )));
    }
    inactive += 2 * (observed.key_pages.len() - first_inactive_cache);
    report.inactive_values += inactive;
    Ok(())
}

fn cache_offset(physical_page: usize, head: usize, position: usize, dimension: usize) -> usize {
    Qwen38_27B::HEAD_DIM
        * ((position & (ATTENTION_PAGE_SIZE - 1))
            + ATTENTION_PAGE_SIZE * (head + Qwen38_27B::NUM_KV_HEADS * physical_page))
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
) -> Result<(), FullAttentionLayerQualificationError> {
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_bf16_slice(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            maximum,
        )?;
    }
    Ok(())
}

fn compare_f32_slice(
    role: &str,
    actual: &[f32],
    expected: &[f32],
    absolute: f32,
    relative: f32,
    maximum: &mut f32,
) -> Result<(), FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        *maximum = maximum.max(error);
        let tolerance = absolute.max(expected.abs() * relative);
        if !actual.is_finite() || error > tolerance {
            return Err(FullAttentionLayerQualificationError::Mismatch(format!(
                "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    Ok(())
}

fn require_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    maximum: &mut f32,
) -> Result<(), FullAttentionLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.5f32.max(expected.abs() as f32 * 0.03);
    if !actual.is_finite() || error > tolerance {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_f32_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    absolute: f32,
    relative: f32,
    report: &mut FullAttentionLayerQualification,
) -> Result<(), FullAttentionLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * relative);
    if !actual.is_finite() || error > tolerance {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn verify_no_device_allocation(
    program: &FullAttentionLayerProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), FullAttentionLayerQualificationError> {
    program.replay(stream, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in [1, 32, 8, 64, 3, 128, 6, MAX_ROWS, 2, 7, 4, 5] {
            program.replay(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(FullAttentionLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

fn little_endian_words(bytes: &[u8]) -> Result<Vec<u16>, FullAttentionLayerQualificationError> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(FullAttentionLayerQualificationError::Mismatch(
            "source BF16 plane has an odd byte length".to_string(),
        ));
    }
    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, MAX_ROWS, SOURCE_LAYER, observable_values, qualify_full_attention_layer,
    };
    use std::path::PathBuf;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn full_attention_layer_suite_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
    }

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn full_attention_layer_suite_source_layer63_matches_complete_seam_oracles_and_graph_replay()
    -> Result<(), super::FullAttentionLayerQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            super::FullAttentionLayerQualificationError::Mismatch(
                "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_full_attention_layer(&PathBuf::from(root))?;
        let active = EXACT_ROUTES.into_iter().sum::<usize>();
        assert_eq!(SOURCE_LAYER, 63);
        assert_eq!(report.boundary_values, active * 5 * Qwen38_27B::HIDDEN);
        assert_eq!(
            report.qk_values,
            active * (Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + 2 * Qwen38_27B::ATTENTION_KV_ROWS)
        );
        assert_eq!(
            report.activation_codes,
            active
                * (2 * Qwen38_27B::HIDDEN
                    + Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
                    + Qwen38_27B::INTERMEDIATE)
        );
        assert_eq!(report.activation_scales, active * 4);
        assert_eq!(
            report.source_values,
            Qwen38_27B::ATTENTION_QKV_ROWS
                + Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
                + 2 * Qwen38_27B::HIDDEN
                + Qwen38_27B::INTERMEDIATE
        );
        assert_eq!(report.macro_partial_values, 25_362_432);
        assert_eq!(report.metadata_values, 6_660_396);
        assert_eq!(report.graph_replay_values, 12 * observable_values());
        assert!(report.inactive_values > 0);
        assert_eq!(report.immutable_descriptor_words, 768);
        assert_eq!(report.resident_weight_bytes, 372_395_008);
        assert_eq!(report.cache_bytes, 3_145_728);
        assert_eq!(report.workspace_bytes, 639_924_416);
        assert_eq!(report.owner_bytes, 1_015_465_152);
        assert_eq!(report.arena_bytes, 1_015_465_984);
        assert_eq!(report.padding_bytes, 832);
        assert_eq!(report.descriptor_bytes, 512);
        assert!(report.maximum_absolute_error.is_finite());
        Ok(())
    }
}
