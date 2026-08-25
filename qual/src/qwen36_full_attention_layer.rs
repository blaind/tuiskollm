//! Source-backed qualification for one Qwen3.6 attention plus MoE layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, encode_e4m3fn,
    f32_to_bf16,
};
use crate::qwen36_moe_experts::nvfp4_dot;
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen36FullAttentionLayerImmutable, Qwen36FullAttentionLayerInputs,
    Qwen36FullAttentionLayerObservables, Qwen36FullAttentionLayerProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MaterializedQwen36FullAttention,
    MaterializedQwen36MoeLayer, Qwen36FullAttentionBindings, Qwen36Moe35B, Qwen36MoeLayerBindings,
};

const SOURCE_LAYER: usize = 3;
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const QKV_ROWS: usize = Qwen36Moe35B::ATTENTION_QKV_ROWS;
const QUERY_ROWS: usize = Qwen36Moe35B::ATTENTION_QUERY_ROWS;
const KV_ROWS: usize = Qwen36Moe35B::ATTENTION_KV_ROWS;
const ATTENTION_COLUMNS: usize = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
const HEAD_DIM: usize = Qwen36Moe35B::HEAD_DIM;
const QUERY_HEADS: usize = Qwen36Moe35B::NUM_ATTENTION_HEADS;
const KV_HEADS: usize = Qwen36Moe35B::NUM_KV_HEADS;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const SLOTS: usize = TOP_K + 1;
const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
const MAX_ROWS: usize = 128;
const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, MAX_ROWS];

/// Failure of the complete source-backed Qwen3.6 attention/MoE gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36FullAttentionLayerQualificationError {
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
    #[error("Qwen3.6 full-attention layer qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36FullAttentionLayerQualification {
    /// Residual and normalization values checked at every exact batch.
    pub boundary_values: usize,
    /// Real-source attention and MoE values checked through B=1.
    pub source_values: usize,
    /// Mutable owner values reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Inactive mutable values verified unchanged.
    pub inactive_values: usize,
    /// Immutable source/materialized device values proved unchanged.
    pub immutable_values: usize,
    /// Runtime-owned graph-input values proved unchanged.
    pub runtime_input_values: usize,
    /// Complete one-allocation owner bytes.
    pub arena_bytes: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact represented E4M3 cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the owner arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from an accepted represented-value formula.
    pub maximum_absolute_error: f32,
}

struct SourceMaterialized<'a> {
    attention: MaterializedQwen36FullAttention<'a>,
    moe: MaterializedQwen36MoeLayer<'a>,
}

/// Qualifies source-backed Qwen3.6 layer 3 at every exact decode and prefill route.
pub fn qualify_qwen36_full_attention_layer(
    root: &Path,
) -> Result<Qwen36FullAttentionLayerQualification, Qwen36FullAttentionLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen36Moe35B>::open(root)?);
    let attention_binding = Qwen36FullAttentionBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let moe_binding = Qwen36MoeLayerBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let source = SourceMaterialized {
        attention: attention_binding.materialize()?,
        moe: moe_binding.materialize()?,
    };
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        Qwen36FullAttentionLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 52 {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!(
                "Qwen3.6 attention owner exposes {} addresses, expected 52",
                stable_addresses.len()
            ),
        ));
    }
    let mut report = Qwen36FullAttentionLayerQualification {
        boundary_values: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        runtime_input_values: 0,
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
    verify_scales(&program, &source)?;

    for rows in EXACT_ROUTES {
        let first_input = make_input(rows, 0);
        prepare_run(&program, &stream, rows, &first_input)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(rows, 1);
        prepare_run(&program, &stream, rows, &input)?;
        let replay_inputs = program.qualification_runtime_inputs(&stream)?;
        verify_runtime_input_contract(rows, &input, &replay_inputs)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;
        let replay_inputs_after = program.qualification_runtime_inputs(&stream)?;
        report.runtime_input_values +=
            verify_runtime_inputs_unchanged(rows, &replay_inputs, &replay_inputs_after)?;

        prepare_run(&program, &stream, rows, &input)?;
        let eager_inputs = program.qualification_runtime_inputs(&stream)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;
        let eager_inputs_after = program.qualification_runtime_inputs(&stream)?;
        report.runtime_input_values +=
            verify_runtime_inputs_unchanged(rows, &eager_inputs, &eager_inputs_after)?;

        verify_boundaries(rows, &input, &source, &replay, &mut report)?;
        if rows == 1 {
            verify_source_formula(&source, &replay, &mut report)?;
        }
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        report.inactive_values += verify_inactive(rows, &replay)?;
        report.inactive_values += verify_inactive(rows, &eager)?;
        report.inactive_values += verify_inactive_cache(rows, &replay)?;
        report.inactive_values += verify_inactive_cache(rows, &eager)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                format!(
                    "Qwen3.6 attention owner addresses changed at {}",
                    route_label(rows)
                ),
            ));
        }
    }

    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &source,
        &mut report,
    )?;
    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn make_input(rows: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..rows * HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / HIDDEN) & 15]))
        .collect()
}

fn prepare_run(
    program: &Qwen36FullAttentionLayerProgram,
    stream: &CudaStream,
    rows: usize,
    input: &[u16],
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    program.reset_cache(stream)?;
    program.load_residual(stream, rows, input)?;
    if rows <= MAX_BATCH {
        program.load_decode_state(
            stream,
            rows,
            &vec![0; rows],
            &vec![1.0; rows * 32],
            &vec![0.0; rows * 32],
        )?;
    } else {
        program.load_prefill_state(stream, rows, &vec![1.0; rows * 32], &vec![0.0; rows * 32])?;
    }
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

fn verify_runtime_input_contract(
    rows: usize,
    input: &[u16],
    actual: &Qwen36FullAttentionLayerInputs,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    compare_exact(
        "runtime residual input",
        &actual.residual_input[..rows * HIDDEN],
        input,
    )?;
    compare_exact(
        "runtime physical page inventory",
        &actual.block_tables,
        &(0..24u32).collect::<Vec<_>>(),
    )?;
    if rows <= MAX_BATCH {
        compare_exact(
            "decode table rows",
            &actual.table_rows[..rows],
            &(0..rows as u32).collect::<Vec<_>>(),
        )?;
        compare_exact(
            "decode cache positions",
            &actual.cache_positions[..rows],
            &vec![0; rows],
        )?;
        compare_exact(
            "decode causal lengths",
            &actual.lengths[..rows],
            &vec![1; rows],
        )?;
        compare_f32_bits(
            "decode rotary cosine",
            &actual.rope_cos[..rows * 32],
            &vec![1.0; rows * 32],
        )?;
        compare_f32_bits(
            "decode rotary sine",
            &actual.rope_sin[..rows * 32],
            &vec![0.0; rows * 32],
        )?;
    } else {
        compare_exact(
            "prefill table rows",
            &actual.prefill_table_rows[..rows],
            &vec![0; rows],
        )?;
        compare_exact(
            "prefill cache positions",
            &actual.prefill_cache_positions[..rows],
            &(0..rows as u32).collect::<Vec<_>>(),
        )?;
        compare_exact(
            "prefill causal lengths",
            &actual.prefill_lengths[..rows],
            &(1..=rows as u32).collect::<Vec<_>>(),
        )?;
        compare_f32_bits(
            "prefill rotary cosine",
            &actual.prefill_rope_cos[..rows * 32],
            &vec![1.0; rows * 32],
        )?;
        compare_f32_bits(
            "prefill rotary sine",
            &actual.prefill_rope_sin[..rows * 32],
            &vec![0.0; rows * 32],
        )?;
    }

    Ok(())
}

fn verify_runtime_inputs_unchanged(
    rows: usize,
    before: &Qwen36FullAttentionLayerInputs,
    after: &Qwen36FullAttentionLayerInputs,
) -> Result<usize, Qwen36FullAttentionLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            compare_exact(
                &format!(
                    "{} immutable runtime `{}`",
                    route_label(rows),
                    stringify!($field)
                ),
                &after.$field,
                &before.$field,
            )?;
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            compare_f32_bits(
                &format!(
                    "{} immutable runtime `{}`",
                    route_label(rows),
                    stringify!($field)
                ),
                &after.$field,
                &before.$field,
            )?;
        };
    }
    same!(residual_input);
    same!(block_tables);
    same_f32!(rope_cos);
    same_f32!(rope_sin);
    same!(table_rows);
    same!(cache_positions);
    same!(lengths);
    same_f32!(prefill_rope_cos);
    same_f32!(prefill_rope_sin);
    same!(prefill_table_rows);
    same!(prefill_cache_positions);
    same!(prefill_lengths);

    Ok(runtime_input_values(after))
}

fn runtime_input_values(values: &Qwen36FullAttentionLayerInputs) -> usize {
    values.residual_input.len()
        + values.block_tables.len()
        + values.rope_cos.len()
        + values.rope_sin.len()
        + values.table_rows.len()
        + values.cache_positions.len()
        + values.lengths.len()
        + values.prefill_rope_cos.len()
        + values.prefill_rope_sin.len()
        + values.prefill_table_rows.len()
        + values.prefill_cache_positions.len()
        + values.prefill_lengths.len()
}

fn verify_scales(
    program: &Qwen36FullAttentionLayerProgram,
    source: &SourceMaterialized<'_>,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let expected = [
        source.attention.qkv_input_scale,
        source.attention.qkv_weight_scales[0],
        source.attention.qkv_weight_scales[1],
        source.attention.qkv_weight_scales[2],
        source.attention.output.input_scale,
        source.attention.output.weight_scale,
        Qwen36Moe35B::FP8_CACHE_SCALE,
        source.moe.shared_expert.gate_up_weight_scales_2[0],
        source.moe.shared_expert.down_weight_scales_2[0],
        Qwen36Moe35B::RMS_NORM_EPSILON,
    ]
    .map(f32::to_bits);
    if program.qualification_source_scales().map(f32::to_bits) != expected {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            "resident scales differ from the materialized source contract".to_string(),
        ));
    }
    Ok(())
}

fn verify_boundaries(
    rows: usize,
    input: &[u16],
    source: &SourceMaterialized<'_>,
    observed: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let input_norm = source.attention.input_norm.words().collect::<Vec<_>>();
    let post_norm = source
        .attention
        .post_attention_norm
        .words()
        .collect::<Vec<_>>();
    let next_norm = source.moe.next_norm.words().collect::<Vec<_>>();
    for token in 0..rows {
        let begin = token * HIDDEN;
        let end = begin + HIDDEN;
        compare_bf16(
            "input RMSNorm",
            &observed.mixer_normalized[begin..end],
            &rms_norm_oracle::<Qwen36Moe35B>(&input[begin..end], &input_norm),
            report,
        )?;
        let mixer_residual =
            residual_oracle(&input[begin..end], &observed.mixer_branch[begin..end]);
        compare_exact(
            "attention residual",
            &observed.mixer_residual[begin..end],
            &mixer_residual,
        )?;
        compare_bf16(
            "post-attention RMSNorm",
            &observed.moe_normalized[begin..end],
            &rms_norm_oracle::<Qwen36Moe35B>(&mixer_residual, &post_norm),
            report,
        )?;
        let residual = residual_oracle(
            &observed.mixer_residual[begin..end],
            &observed.moe_branch[begin..end],
        );
        compare_exact(
            "layer residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        compare_bf16(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &rms_norm_oracle::<Qwen36Moe35B>(&residual, &next_norm),
            report,
        )?;
    }
    report.boundary_values += rows * HIDDEN * 5;
    Ok(())
}

fn verify_source_formula(
    source: &SourceMaterialized<'_>,
    observed: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let input = &observed.mixer_normalized[..HIDDEN];
    let input_codes = quantize_static(input, source.attention.qkv_input_scale)?;
    compare_exact(
        "QKV activation codes",
        &observed.qkv_activation_codes[..HIDDEN],
        &input_codes,
    )?;
    for row in 0..QKV_ROWS {
        let scale = if row < QUERY_ROWS {
            source.attention.qkv_weight_scales[0]
        } else if row < QUERY_ROWS + KV_ROWS {
            source.attention.qkv_weight_scales[1]
        } else {
            source.attention.qkv_weight_scales[2]
        };
        let expected = fp8_dot(
            &input_codes,
            &source.attention.qkv_weight_e4m3[row * HIDDEN..(row + 1) * HIDDEN],
            source.attention.qkv_input_scale,
            scale,
        )?;
        require_close(
            "source QKV projection",
            row,
            bf16_to_f32(observed.qkv[row]),
            expected,
            0.25,
            0.025,
            report,
        )?;
    }
    verify_attention(source, observed, report)?;
    verify_router_and_experts(source, observed, report)?;
    report.source_values += HIDDEN + QKV_ROWS + ATTENTION_COLUMNS * 4 + HIDDEN;
    Ok(())
}

fn verify_attention(
    source: &SourceMaterialized<'_>,
    observed: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let query_norm = source.attention.query_norm.words().collect::<Vec<_>>();
    let key_norm = source.attention.key_norm.words().collect::<Vec<_>>();
    let key_begin = QUERY_ROWS;
    let value_begin = key_begin + KV_ROWS;
    for head in 0..QUERY_HEADS {
        let source_begin = head * 2 * HEAD_DIM;
        let normalized = head_norm(
            &observed.qkv[source_begin..source_begin + HEAD_DIM],
            &query_norm,
        );
        let output_begin = head * HEAD_DIM;
        compare_f32(
            "prepared query",
            &observed.query[output_begin..output_begin + HEAD_DIM],
            &normalized,
            report,
        )?;
    }
    for head in 0..KV_HEADS {
        let normalized = head_norm(
            &observed.qkv[key_begin + head * HEAD_DIM..key_begin + (head + 1) * HEAD_DIM],
            &key_norm,
        );
        let page = head * ATTENTION_PAGE_SIZE * HEAD_DIM;
        let expected = normalized
            .into_iter()
            .map(|value| {
                encode_e4m3fn(value / Qwen36Moe35B::FP8_CACHE_SCALE)
                    .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        compare_exact(
            "represented key cache",
            &observed.key_pages[page..page + HEAD_DIM],
            &expected,
        )?;
        let expected = observed.qkv
            [value_begin + head * HEAD_DIM..value_begin + (head + 1) * HEAD_DIM]
            .iter()
            .map(|&bits| {
                encode_e4m3fn(bf16_to_f32(bits) / Qwen36Moe35B::FP8_CACHE_SCALE)
                    .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        compare_exact(
            "represented value cache",
            &observed.value_pages[page..page + HEAD_DIM],
            &expected,
        )?;
    }

    let mut gated = vec![0.0f32; ATTENTION_COLUMNS];
    for head in 0..QUERY_HEADS {
        let kv_head = head / (QUERY_HEADS / KV_HEADS);
        let gate_begin = head * 2 * HEAD_DIM + HEAD_DIM;
        for dimension in 0..HEAD_DIM {
            let cache = cache_offset(0, kv_head, 0, dimension);
            let value = decode_e4m3fn(observed.value_pages[cache])
                .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)?
                * Qwen36Moe35B::FP8_CACHE_SCALE;
            let gate = bf16_to_f32(observed.qkv[gate_begin + dimension]);
            gated[head * HEAD_DIM + dimension] = value / (1.0 + (-gate).exp());
        }
    }
    compare_f32(
        "one-token gated attention",
        &observed.attention[..ATTENTION_COLUMNS],
        &gated,
        report,
    )?;
    let gated_bf16 = gated.iter().copied().map(f32_to_bf16).collect::<Vec<_>>();
    compare_bf16_tolerance(
        "gated attention BF16",
        &observed.output_activation[..ATTENTION_COLUMNS],
        &gated_bf16,
        0.002,
        report,
    )?;
    let codes = quantize_static(
        &observed.output_activation[..ATTENTION_COLUMNS],
        source.attention.output.input_scale,
    )?;
    compare_exact(
        "attention output activation codes",
        &observed.output_activation_codes[..ATTENTION_COLUMNS],
        &codes,
    )?;
    for row in 0..HIDDEN {
        let expected = fp8_dot(
            &codes,
            &source.attention.output.weight_e4m3
                [row * ATTENTION_COLUMNS..(row + 1) * ATTENTION_COLUMNS],
            source.attention.output.input_scale,
            source.attention.output.weight_scale,
        )?;
        require_close(
            "attention output projection",
            row,
            bf16_to_f32(observed.mixer_branch[row]),
            expected,
            0.25,
            0.025,
            report,
        )?;
    }
    Ok(())
}

fn verify_router_and_experts(
    source: &SourceMaterialized<'_>,
    observed: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let input = &observed.moe_normalized[..HIDDEN];
    let router = source.moe.router_weight.words().collect::<Vec<_>>();
    let mut logits = Vec::with_capacity(EXPERTS);
    for expert in 0..EXPERTS {
        let expected = bf16_dot(input, &router[expert * HIDDEN..(expert + 1) * HIDDEN]);
        require_close(
            "router logit",
            expert,
            bf16_to_f32(observed.router_logits[expert]),
            expected,
            0.25,
            0.025,
            report,
        )?;
        logits.push(bf16_to_f32(observed.router_logits[expert]));
    }
    let mut order = (0..EXPERTS).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    let selected = &order[..TOP_K];
    for (position, &expert) in selected.iter().enumerate() {
        if observed.expert_indices[position] as usize != expert {
            return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                format!(
                    "router selected {} at {position}, expected {expert}",
                    observed.expert_indices[position]
                ),
            ));
        }
    }
    let maximum = selected
        .iter()
        .map(|&index| logits[index])
        .fold(f32::NEG_INFINITY, f32::max);
    let exponentials = selected
        .iter()
        .map(|&index| (logits[index] - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    for (position, exponential) in exponentials.into_iter().enumerate() {
        let expected = f32_to_bf16(exponential / denominator);
        compare_exact(
            "routing weight",
            &observed.routing_weights[position..position + 1],
            &[expected],
        )?;
    }
    verify_experts(source, observed, report)
}

fn verify_experts(
    source: &SourceMaterialized<'_>,
    observed: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let input = &observed.moe_normalized[..HIDDEN];
    let mut intermediate = vec![0u16; SLOTS * INTERMEDIATE];
    let mut outputs = vec![0u16; SLOTS * HIDDEN];
    for position in 0..SLOTS {
        let routed = position < TOP_K;
        let expert = if routed {
            observed.expert_indices[position] as usize
        } else {
            0
        };
        let (gate_codes, gate_scales, gate_scale) = if routed {
            (
                source.moe.experts.gate_up_weight_e2m1.as_slice(),
                source.moe.experts.gate_up_scale_e4m3_swizzled.as_slice(),
                source.moe.experts.gate_up_weight_scales_2[expert],
            )
        } else {
            (
                source.moe.shared_expert.gate_up_weight_e2m1.as_slice(),
                source
                    .moe
                    .shared_expert
                    .gate_up_scale_e4m3_swizzled
                    .as_slice(),
                source.moe.shared_expert.gate_up_weight_scales_2[0],
            )
        };
        for row in 0..INTERMEDIATE {
            let gate = nvfp4_dot(
                input,
                gate_codes,
                gate_scales,
                expert,
                row,
                2 * INTERMEDIATE,
                HIDDEN,
                gate_scale,
            );
            let up = nvfp4_dot(
                input,
                gate_codes,
                gate_scales,
                expert,
                row + INTERMEDIATE,
                2 * INTERMEDIATE,
                HIDDEN,
                gate_scale,
            );
            intermediate[position * INTERMEDIATE + row] =
                f32_to_bf16(gate / (1.0 + (-gate).exp()) * up);
        }
        let values = &intermediate[position * INTERMEDIATE..(position + 1) * INTERMEDIATE];
        let (down_codes, down_scales, down_scale) = if routed {
            (
                source.moe.experts.down_weight_e2m1.as_slice(),
                source.moe.experts.down_scale_e4m3_swizzled.as_slice(),
                source.moe.experts.down_weight_scales_2[expert],
            )
        } else {
            (
                source.moe.shared_expert.down_weight_e2m1.as_slice(),
                source.moe.shared_expert.down_scale_e4m3_swizzled.as_slice(),
                source.moe.shared_expert.down_weight_scales_2[0],
            )
        };
        for row in 0..HIDDEN {
            outputs[position * HIDDEN + row] = f32_to_bf16(nvfp4_dot(
                values,
                down_codes,
                down_scales,
                expert,
                row,
                HIDDEN,
                INTERMEDIATE,
                down_scale,
            ));
        }
    }
    compare_bf16_tolerance(
        "expert intermediate",
        &observed.expert_intermediate[..SLOTS * INTERMEDIATE],
        &intermediate,
        0.02,
        report,
    )?;
    compare_bf16_tolerance(
        "expert output",
        &observed.expert_output[..SLOTS * HIDDEN],
        &outputs,
        0.04,
        report,
    )?;
    let gate_weights = source
        .moe
        .shared_expert_gate_weight
        .words()
        .collect::<Vec<_>>();
    let shared_gate = bf16_dot(input, &gate_weights);
    require_close(
        "shared expert gate",
        0,
        bf16_to_f32(observed.shared_gate[0]),
        shared_gate,
        0.25,
        0.025,
        report,
    )?;
    let multiplier = 1.0 / (1.0 + (-(shared_gate as f32)).exp());
    for column in 0..HIDDEN {
        let mut sum = 0.0f32;
        for position in 0..TOP_K {
            sum = bf16_to_f32(outputs[position * HIDDEN + column])
                .mul_add(bf16_to_f32(observed.routing_weights[position]), sum);
        }
        let shared = bf16_to_f32(outputs[TOP_K * HIDDEN + column]);
        let expected = f32_to_bf16(shared.mul_add(multiplier, sum));
        require_close(
            "combined MoE output",
            column,
            bf16_to_f32(observed.moe_branch[column]),
            f64::from(bf16_to_f32(expected)),
            0.08,
            0.025,
            report,
        )?;
    }
    report.source_values += EXPERTS + 2 * TOP_K + SLOTS * (INTERMEDIATE + HIDDEN) + 1 + HIDDEN;
    Ok(())
}

fn head_norm(source: &[u16], norm: &[u16]) -> Vec<f32> {
    let sum = source
        .iter()
        .map(|&bits| f64::from(bf16_to_f32(bits)).powi(2))
        .sum::<f64>();
    let inverse = 1.0 / (sum / HEAD_DIM as f64 + f64::from(Qwen36Moe35B::RMS_NORM_EPSILON)).sqrt();
    source
        .iter()
        .zip(norm)
        .map(|(&value, &weight)| {
            (f64::from(bf16_to_f32(value)) * inverse * (1.0 + f64::from(bf16_to_f32(weight))))
                as f32
        })
        .collect()
}

fn quantize_static(
    values: &[u16],
    scale: f32,
) -> Result<Vec<u8>, Qwen36FullAttentionLayerQualificationError> {
    values
        .iter()
        .map(|&bits| {
            encode_e4m3fn(bf16_to_f32(bits) / scale)
                .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)
        })
        .collect()
}

fn fp8_dot(
    activation: &[u8],
    weight: &[u8],
    activation_scale: f32,
    weight_scale: f32,
) -> Result<f64, Qwen36FullAttentionLayerQualificationError> {
    activation
        .iter()
        .zip(weight)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            let activation = decode_e4m3fn(activation)
                .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)?;
            let weight = decode_e4m3fn(weight)
                .map_err(Qwen36FullAttentionLayerQualificationError::Mismatch)?;
            Ok(sum + f64::from(activation) * f64::from(weight))
        })
        .map(|sum| sum * f64::from(activation_scale * weight_scale))
}

fn bf16_dot(left: &[u16], right: &[u16]) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (&left, &right)| {
        sum + f64::from(bf16_to_f32(left)) * f64::from(bf16_to_f32(right))
    })
}

fn verify_replay(
    rows: usize,
    eager: &Qwen36FullAttentionLayerObservables,
    replay: &Qwen36FullAttentionLayerObservables,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            compare_exact(
                &format!("{} graph plane `{}`", route_label(rows), stringify!($field)),
                &replay.$field,
                &eager.$field,
            )?;
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            compare_exact(
                &format!("{} graph plane `{}`", route_label(rows), stringify!($field)),
                &replay
                    .$field
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                &eager
                    .$field
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            )?;
        };
    }
    same!(mixer_normalized);
    same!(qkv_activation_codes);
    same!(qkv);
    same_f32!(query);
    same!(key_pages);
    same!(value_pages);
    same_f32!(attention);
    same!(output_activation);
    same!(output_activation_codes);
    same!(mixer_branch);
    same!(mixer_residual);
    same!(moe_normalized);
    same!(router_logits);
    same!(expert_indices);
    same!(routing_weights);
    same!(expert_intermediate);
    same!(expert_output);
    same!(shared_gate);
    same!(moe_branch);
    same!(residual_output);
    same!(next_normalized);
    report.graph_replay_values += observable_values(replay);
    Ok(())
}

fn observable_values(values: &Qwen36FullAttentionLayerObservables) -> usize {
    values.mixer_normalized.len()
        + values.qkv_activation_codes.len()
        + values.qkv.len()
        + values.query.len()
        + values.key_pages.len()
        + values.value_pages.len()
        + values.attention.len()
        + values.output_activation.len()
        + values.output_activation_codes.len()
        + values.mixer_branch.len()
        + values.mixer_residual.len()
        + values.moe_normalized.len()
        + values.router_logits.len()
        + values.expert_indices.len()
        + values.routing_weights.len()
        + values.expert_intermediate.len()
        + values.expert_output.len()
        + values.shared_gate.len()
        + values.moe_branch.len()
        + values.residual_output.len()
        + values.next_normalized.len()
}

fn verify_replacement_input(
    rows: usize,
    first: &Qwen36FullAttentionLayerObservables,
    replay: &Qwen36FullAttentionLayerObservables,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    if first.residual_output[..rows * HIDDEN] == replay.residual_output[..rows * HIDDEN] {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!(
                "{} graph ignored replacement residual rows",
                route_label(rows)
            ),
        ));
    }
    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Qwen36FullAttentionLayerObservables,
) -> Result<usize, Qwen36FullAttentionLayerQualificationError> {
    macro_rules! u16_tail {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "{} modified inactive `{}`",
                        route_label(rows),
                        stringify!($field)
                    ),
                ));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! u8_tail {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BYTE_SENTINEL)
            {
                return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "{} modified inactive `{}`",
                        route_label(rows),
                        stringify!($field)
                    ),
                ));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! f32_tail {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                    format!(
                        "{} modified inactive `{}`",
                        route_label(rows),
                        stringify!($field)
                    ),
                ));
            }
            observed.$field.len() - begin
        }};
    }
    let mut count = 0;
    count += u16_tail!(mixer_normalized, HIDDEN);
    count += u8_tail!(qkv_activation_codes, HIDDEN);
    count += u16_tail!(qkv, QKV_ROWS);
    count += f32_tail!(query, ATTENTION_COLUMNS);
    count += f32_tail!(attention, ATTENTION_COLUMNS);
    count += u16_tail!(output_activation, ATTENTION_COLUMNS);
    count += u8_tail!(output_activation_codes, ATTENTION_COLUMNS);
    count += u16_tail!(mixer_branch, HIDDEN);
    count += u16_tail!(mixer_residual, HIDDEN);
    count += u16_tail!(moe_normalized, HIDDEN);
    count += u16_tail!(router_logits, EXPERTS);
    count += u16_tail!(expert_indices, TOP_K);
    count += u16_tail!(routing_weights, TOP_K);
    count += u16_tail!(expert_intermediate, SLOTS * INTERMEDIATE);
    count += u16_tail!(expert_output, SLOTS * HIDDEN);
    count += u16_tail!(shared_gate, 1);
    count += u16_tail!(moe_branch, HIDDEN);
    count += u16_tail!(residual_output, HIDDEN);
    count += u16_tail!(next_normalized, HIDDEN);
    Ok(count)
}

fn verify_inactive_cache(
    rows: usize,
    observed: &Qwen36FullAttentionLayerObservables,
) -> Result<usize, Qwen36FullAttentionLayerQualificationError> {
    let mut active = vec![false; observed.key_pages.len()];
    for token in 0..rows {
        let (physical_page, page_position) = if rows <= MAX_BATCH {
            (token * 3, 0)
        } else {
            (token / ATTENTION_PAGE_SIZE, token % ATTENTION_PAGE_SIZE)
        };
        for head in 0..KV_HEADS {
            for dimension in 0..HEAD_DIM {
                active[cache_offset(physical_page, head, page_position, dimension)] = true;
            }
        }
    }
    for (index, ((&key, &value), &is_active)) in observed
        .key_pages
        .iter()
        .zip(&observed.value_pages)
        .zip(&active)
        .enumerate()
    {
        if !is_active && (key != 0 || value != 0) {
            return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
                format!(
                    "{} modified inactive cache value {index}",
                    route_label(rows)
                ),
            ));
        }
    }

    Ok(2 * active.iter().filter(|&&is_active| !is_active).count())
}

fn cache_offset(
    physical_page: usize,
    head: usize,
    page_position: usize,
    dimension: usize,
) -> usize {
    dimension + HEAD_DIM * (page_position + ATTENTION_PAGE_SIZE * (head + KV_HEADS * physical_page))
}

fn verify_immutable(
    actual: &Qwen36FullAttentionLayerImmutable,
    source: &SourceMaterialized<'_>,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    macro_rules! same {
        ($field:ident, $expected:expr) => {{
            let expected = $expected;
            compare_exact(stringify!($field), &actual.$field, expected)?;
            report.immutable_values += actual.$field.len();
        }};
    }
    same!(
        input_norm,
        &source.attention.input_norm.words().collect::<Vec<_>>()
    );
    same!(qkv_weight_codes, &source.attention.qkv_weight_e4m3);
    same!(
        query_norm,
        &source.attention.query_norm.words().collect::<Vec<_>>()
    );
    same!(
        key_norm,
        &source.attention.key_norm.words().collect::<Vec<_>>()
    );
    same!(output_weight_codes, source.attention.output.weight_e4m3);
    same!(
        post_attention_norm,
        &source
            .attention
            .post_attention_norm
            .words()
            .collect::<Vec<_>>()
    );
    same!(
        router_weight,
        &source.moe.router_weight.words().collect::<Vec<_>>()
    );
    same!(
        routed_gate_up_codes,
        &source.moe.experts.gate_up_weight_e2m1
    );
    same!(
        routed_gate_up_scales,
        &source.moe.experts.gate_up_scale_e4m3_swizzled
    );
    same!(
        routed_gate_up_weight_scales_2,
        &source.moe.experts.gate_up_weight_scales_2
    );
    same!(routed_down_codes, &source.moe.experts.down_weight_e2m1);
    same!(
        routed_down_scales,
        &source.moe.experts.down_scale_e4m3_swizzled
    );
    same!(
        routed_down_weight_scales_2,
        &source.moe.experts.down_weight_scales_2
    );
    same!(
        shared_gate_up_codes,
        &source.moe.shared_expert.gate_up_weight_e2m1
    );
    same!(
        shared_gate_up_scales,
        &source.moe.shared_expert.gate_up_scale_e4m3_swizzled
    );
    same!(
        shared_down_codes,
        &source.moe.shared_expert.down_weight_e2m1
    );
    same!(
        shared_down_scales,
        &source.moe.shared_expert.down_scale_e4m3_swizzled
    );
    same!(
        shared_gate_weight,
        &source
            .moe
            .shared_expert_gate_weight
            .words()
            .collect::<Vec<_>>()
    );
    same!(next_norm, &source.moe.next_norm.words().collect::<Vec<_>>());
    Ok(())
}

fn verify_no_device_allocation(
    program: &Qwen36FullAttentionLayerProgram,
    stream: &CudaStream,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    program.replay(stream, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in [128, 1, 64, 8, 32, 3, 6, 2, 7, 4, 5] {
            program.replay(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }
    Ok(())
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
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!(
                "{role} has {} values, expected {}",
                actual.len(),
                expected.len()
            ),
        ));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!("{role} differs at value {index}"),
        ));
    }
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    compare_exact(
        role,
        &actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        &expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    )
}

fn route_label(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!("{role} length differs"),
        ));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            0.25,
            0.025,
            report,
        )?;
    }
    Ok(())
}

fn compare_bf16_tolerance(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    tolerance: f32,
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            tolerance,
            0.025,
            report,
        )?;
    }
    Ok(())
}

fn compare_f32(
    role: &str,
    actual: &[f32],
    expected: &[f32],
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            actual,
            f64::from(expected),
            0.002,
            0.003,
            report,
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
    report: &mut Qwen36FullAttentionLayerQualification,
) -> Result<(), Qwen36FullAttentionLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * relative);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen36FullAttentionLayerQualificationError::Mismatch(
            format!("{role} at {index}: device={actual}, oracle={expected}, tolerance={tolerance}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_layer_and_owner_geometry_are_exact() {
        assert_eq!(SOURCE_LAYER, 3);
        assert_eq!(ATTENTION_PAGE_SIZE, 64);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(
            2 * 24 * KV_HEADS * ATTENTION_PAGE_SIZE * HEAD_DIM,
            1_572_864
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer3_matches_complete_oracles_and_graph_replay()
    -> Result<(), Qwen36FullAttentionLayerQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT").ok_or_else(|| {
            Qwen36FullAttentionLayerQualificationError::Mismatch(
                "TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_qwen36_full_attention_layer(Path::new(&root))?;

        assert_eq!(report.boundary_values, 2_662_400);
        assert_eq!(report.weight_bytes, 483_085_312);
        assert_eq!(report.cache_bytes, 1_572_864);
        assert_eq!(report.workspace_bytes, 18_587_584);
        assert_eq!(report.arena_bytes, 503_246_592);
        assert_eq!(report.padding_bytes, 832);
        assert!(report.source_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.runtime_input_values > 0);
        assert!(report.maximum_absolute_error.is_finite());
        Ok(())
    }
}
