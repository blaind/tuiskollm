//! Source-backed qualification for one composed Qwen3.8-Flash-Next QSA plus MoE decoder layer.
//!
//! The one-key oracle makes softmax exactly 1.0 and checks the composition from checkpoint words.
//! Boundary cases prove dense/selected bit identity at 2,051 keys and the first dropped block at
//! 2,052 keys. Longer requests must remain within owner and checkpoint capacity.

use crate::device_benchmark::{self, DeviceBenchmarkError};
use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::harness::graph_replay::first_moved_address;
use crate::qwen38_flash_next_layer_oracle::{
    EXPERTS, HIDDEN, TOP_K, bracket_oracle, moe_oracle, projection_oracle, router_oracle,
    write_back,
};
use crate::qwen38_flash_next_moe_experts::decode_e4m3;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen38FlashNextQsaMoeLayerObservables,
    Qwen38FlashNextQsaMoeLayerProgram, Qwen38FlashNextQsaRound, Qwen38FlashNextQsaRoute,
    qwen38_flash_next_qsa_block_rotary_rows,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext,
    Qwen38FlashNextLayerHyperConnections, Qwen38FlashNextMoeBindings,
    Qwen38FlashNextSparseAttentionBindings,
};

type A = Qwen38FlashNext;

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const WIDTH: usize = A::HC_WIDTH;
pub(crate) const QKV_ROWS: usize = A::ATTENTION_QKV_ROWS;
pub(crate) const OUTPUT_COLUMNS: usize = A::ATTENTION_OUTPUT_COLUMNS;
pub(crate) const HEAD_DIM: usize = <A as Arch>::HEAD_DIM;
pub(crate) const QUERY_HEADS: usize = <A as Arch>::NUM_ATTENTION_HEADS;
pub(crate) const KV_HEADS: usize = <A as Arch>::NUM_KV_HEADS;
pub(crate) const QUERY_ROWS: usize = A::ATTENTION_QUERY_ROWS;
const ROTARY_ELEMENTS: usize = 32;
pub(crate) const VALUE_CACHE_SCALE: f32 = 0.062_5;

/// Failure of the complete source-backed Qwen3.8-Flash-Next QSA/MoE layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextQsaMoeLayerQualificationError {
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
    #[error("Qwen3.8-Flash-Next QSA/MoE layer qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextQsaMoeLayerQualificationError>;

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextQsaMoeLayerQualification {
    /// Values checked against the layer-level oracle on real weights.
    pub oracle_values: usize,
    /// Mutable owner values reproduced by captured-graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace values verified unchanged.
    pub inactive_values: usize,
    /// Runtime-owned graph-input values proved unchanged.
    pub runtime_input_values: usize,
    /// Rounds refused because they left this owner's mapping or the position ceiling.
    pub refused_rounds: usize,
    /// Attention values proved bit-identical across the two routes at the sharp boundary.
    pub route_switch_values: usize,
    /// Selected positions compared entry for entry at and past the boundary.
    pub selected_values: usize,
    /// Complete layer allocation bytes.
    pub arena_bytes: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact represented cache bytes, indexer plane included.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Largest absolute difference from the layer oracle.
    pub maximum_absolute_error: f32,
}

/// Qualifies one source-backed QSA/MoE layer at every exact decode and prefill route.
pub fn qualify_qwen38_flash_next_qsa_moe_layer(
    root: &Path,
    layer: usize,
) -> QualResult<Qwen38FlashNextQsaMoeLayerQualification> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        Qwen38FlashNextQsaMoeLayerProgram::from_snapshot(&context, snapshot.clone(), layer)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;

    let mut report = Qwen38FlashNextQsaMoeLayerQualification {
        oracle_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        runtime_input_values: 0,
        refused_rounds: 0,
        route_switch_values: 0,
        selected_values: 0,
        arena_bytes: program.arena_bytes(),
        weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };

    verify_route_classification(&program, &stream, &mut report)?;
    verify_route_switch_continuity(&program, &stream, &mut report)?;
    verify_first_dropped_block(&program, &stream, &mut report)?;
    verify_selective_replay_agreement(&program, &stream, &mut report)?;

    for rows in EXACT_ROUTES {
        let first_input = make_stream(rows, 0);
        let route = prepare_run(&program, &stream, rows, &first_input)?;
        program.launch_eager(&stream, rows, route)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_stream(rows, 1);
        prepare_run(&program, &stream, rows, &input)?;
        let before = program.qualification_runtime_inputs(&stream)?;
        program.replay(&stream, rows, route)?;
        let replay = program.qualification_observables(&stream)?;
        report.runtime_input_values += verify_runtime_inputs_unchanged(
            &before,
            &program.qualification_runtime_inputs(&stream)?,
        )?;

        prepare_run(&program, &stream, rows, &input)?;
        program.launch_eager(&stream, rows, route)?;
        let eager = program.qualification_observables(&stream)?;

        verify_replay(&eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &replay, &mut report)?;

        if rows == 1 {
            verify_layer_oracle(
                &program,
                &stream,
                snapshot.as_ref(),
                layer,
                &replay,
                &mut report,
            )?;
        }

        if program.base_address() != stable_base
            || first_moved_address(&stable_addresses, &program.qualification_addresses()?).is_some()
        {
            return Err(mismatch(format!(
                "Qwen3.8-Flash-Next QSA/MoE layer addresses moved while qualifying {}",
                route_label(rows)
            )));
        }
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn mismatch(message: String) -> Qwen38FlashNextQsaMoeLayerQualificationError {
    Qwen38FlashNextQsaMoeLayerQualificationError::Mismatch(message)
}

fn route_label(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn make_stream(rows: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.1875, -0.125, 0.09375, -0.0625, 1.25, -1.5,
        0.5625, -0.8125, 0.3125, -0.4375,
    ];
    (0..rows * WIDTH)
        .map(|index| {
            let branch = index % WIDTH / HIDDEN;
            let seed = index
                .wrapping_mul(2_654_435_761)
                .wrapping_add(salt.wrapping_mul(97))
                .wrapping_add(branch.wrapping_mul(13));
            f32_to_bf16(PATTERN[(seed >> 5) % PATTERN.len()])
        })
        .collect()
}

/// Production metadata for an exact decode batch or causal prompt tile.
fn round_for(rows: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let (table_rows, cache_positions) = if rows <= MAX_BATCH {
        ((0..rows as u32).collect(), vec![0u32; rows])
    } else {
        (vec![0; rows], (0..rows as u32).collect())
    };
    let lengths = cache_positions
        .iter()
        .map(|&position| position + 1)
        .collect();
    // Position zero: cos is one and sin is zero, so the rotation is the identity and the
    // oracle does not have to reproduce MRoPE to check the composition around it.
    let rope_cos = vec![1.0f32; rows * ROTARY_ELEMENTS];
    let rope_sin = vec![0.0f32; rows * ROTARY_ELEMENTS];

    (table_rows, cache_positions, lengths, rope_cos, rope_sin)
}

/// The identity rotary rows one round's block compression indexes.
fn block_rotary_for(rows: usize) -> (Vec<f32>, Vec<f32>) {
    let values = qwen38_flash_next_qsa_block_rotary_rows(rows) * ROTARY_ELEMENTS;

    (vec![1.0f32; values], vec![0.0f32; values])
}

fn prepare_run(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    rows: usize,
    input: &[u16],
) -> QualResult<Qwen38FlashNextQsaRoute> {
    let (table_rows, cache_positions, lengths, rope_cos, rope_sin) = round_for(rows);
    let (block_rope_cos, block_rope_sin) = block_rotary_for(rows);
    program.reset_cache(stream)?;
    program.load_residual(stream, rows, input)?;
    let route = program.load_round(
        stream,
        rows,
        Qwen38FlashNextQsaRound {
            table_rows: &table_rows,
            cache_positions: &cache_positions,
            lengths: &lengths,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
            block_rope_cos: &block_rope_cos,
            block_rope_sin: &block_rope_sin,
        },
    )?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(route)
}

/// One decode row of one sequence at an exact visible length.
fn boundary_round(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    visible: u32,
) -> QualResult<Qwen38FlashNextQsaRoute> {
    let (block_rope_cos, block_rope_sin) = block_rotary_for(1);

    Ok(program.load_round(
        stream,
        1,
        Qwen38FlashNextQsaRound {
            table_rows: &[0],
            cache_positions: &[visible - 1],
            lengths: &[visible],
            rope_cos: &[1.0; ROTARY_ELEMENTS],
            rope_sin: &[0.0; ROTARY_ELEMENTS],
            block_rope_cos: &block_rope_cos,
            block_rope_sin: &block_rope_sin,
        },
    )?)
}

/// Checks eager/graph agreement for selected decode and a prompt straddling the route boundary.
fn verify_selective_replay_agreement(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    for (rows, base) in [(1usize, 2_599u32), (32, 2_048)] {
        let input = make_stream(rows, 4);
        let table_rows = vec![0u32; rows];
        let cache_positions = (0..rows as u32).map(|row| base + row).collect::<Vec<_>>();
        let lengths = cache_positions
            .iter()
            .map(|&position| position + 1)
            .collect::<Vec<_>>();
        let (block_rope_cos, block_rope_sin) = block_rotary_for(rows);
        let round = Qwen38FlashNextQsaRound {
            table_rows: &table_rows,
            cache_positions: &cache_positions,
            lengths: &lengths,
            rope_cos: &vec![1.0; rows * ROTARY_ELEMENTS],
            rope_sin: &vec![0.0; rows * ROTARY_ELEMENTS],
            block_rope_cos: &block_rope_cos,
            block_rope_sin: &block_rope_sin,
        };

        let mut observed = Vec::with_capacity(2);
        for captured in [false, true] {
            program.reset_cache(stream)?;
            program.load_residual(stream, rows, &input)?;
            let route = program.load_round(stream, rows, round)?;
            if route != Qwen38FlashNextQsaRoute::Selected {
                return Err(mismatch(format!(
                    "a {rows}-row round from position {base} classified as {route:?}; this check \
                     exists to compare the selection route and would compare nothing"
                )));
            }
            program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
            if captured {
                program.replay(stream, rows, route)?;
            } else {
                program.launch_eager(stream, rows, route)?;
            }
            observed.push(program.qualification_observables(stream)?);
        }

        verify_replay(&observed[0], &observed[1], report)?;
    }

    Ok(())
}

/// The route boundary classifies rather than refuses, and what is still refused is checked.
fn verify_route_classification(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    // At and below 2,051 the dense route is the reference's own function; above it only the
    // selection is, and this owner's pool reaches 4,096 so both bands are drivable.
    for (visible, expected) in [
        (1u32, Qwen38FlashNextQsaRoute::Dense),
        (2_051, Qwen38FlashNextQsaRoute::Dense),
        (2_052, Qwen38FlashNextQsaRoute::Selected),
        (4_096, Qwen38FlashNextQsaRoute::Selected),
    ] {
        let route = boundary_round(program, stream, visible)?;
        if route != expected {
            return Err(mismatch(format!(
                "a round of {visible} visible keys classified as {route:?}, expected {expected:?}"
            )));
        }
    }

    // Past this owner's own mapping, and past the checkpoint's position ceiling: both refuse,
    // and neither truncates.
    for (visible, needle) in [
        (4_097u32, "page capacity"),
        (262_145, "refused rather than truncated"),
    ] {
        let Err(error) = boundary_round(program, stream, visible) else {
            return Err(mismatch(format!(
                "a round asking for {visible} visible keys was admitted"
            )));
        };
        let message = error.to_string();
        if !message.contains(needle) {
            return Err(mismatch(format!(
                "the refusal for {visible} does not say why: {message}"
            )));
        }
        report.refused_rounds += 1;
    }

    Ok(())
}

/// Proves dense and selected routes publish identical bits at 2,051 visible keys.
fn verify_route_switch_continuity(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    let input = make_stream(1, 2);
    let mut published: Option<Vec<f32>> = None;
    for route in [
        Qwen38FlashNextQsaRoute::Dense,
        Qwen38FlashNextQsaRoute::Selected,
    ] {
        program.reset_cache(stream)?;
        program.load_residual(stream, 1, &input)?;
        boundary_round(program, stream, 2_051)?;
        program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
        program.replay(stream, 1, route)?;
        let observed = program.qualification_observables(stream)?;

        if route == Qwen38FlashNextQsaRoute::Selected {
            // The selection at the boundary is the whole visible list, ascending.
            if observed.selected_counts[0] != 2_051 {
                return Err(mismatch(format!(
                    "the selection at 2,051 visible keys named {} positions, expected all of them",
                    observed.selected_counts[0]
                )));
            }
            for position in 0..2_051usize {
                if observed.selected[position] != position as u32 {
                    return Err(mismatch(format!(
                        "the selection at 2,051 visible keys is not the identity at entry {position}"
                    )));
                }
                report.selected_values += 1;
            }
        }
        // The active row only: the reset fills the whole plane with one sentinel, so comparing
        // the inactive tail would count 6,285,312 values neither route wrote.
        let active = observed.attention[..OUTPUT_COLUMNS].to_vec();
        match &published {
            None => published = Some(active),
            Some(dense) => {
                for (index, (&ours, &theirs)) in dense.iter().zip(active.iter()).enumerate() {
                    if ours.to_bits() != theirs.to_bits() {
                        return Err(mismatch(format!(
                            "the two routes disagree at 2,051 visible keys, attention value \
                             {index}: dense {ours} against selected {theirs}"
                        )));
                    }
                    report.route_switch_values += 1;
                }
            }
        }
    }

    Ok(())
}

/// One key past the boundary the indexer drops exactly one four-token block.
///
/// The sharpness of 2,051, observed at the composed layer: at 2,052 the candidate count reaches
/// 513 against a 512-block budget, so the selection publishes 2,048 positions and not 2,052.
fn verify_first_dropped_block(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    let input = make_stream(1, 3);
    program.reset_cache(stream)?;
    program.load_residual(stream, 1, &input)?;
    let route = boundary_round(program, stream, 2_052)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    program.replay(stream, 1, route)?;
    let observed = program.qualification_observables(stream)?;

    if observed.selected_counts[0] != 2_048 {
        return Err(mismatch(format!(
            "at 2,052 visible keys the selection named {} positions, expected 2,048: the budget \
             times the compression ratio, with no tail because 2,052 is block-aligned",
            observed.selected_counts[0]
        )));
    }
    report.selected_values += observed.selected_counts[0] as usize;

    // The compression published this round's closing block, so the block-key plane is no longer
    // the zero plane a reserved-and-unwritten cache would still be.
    if observed.block_keys.iter().all(|&value| value == 0) {
        return Err(mismatch(
            "the block-key plane is entirely zero after a round that closed a micro-block; the \
             compression published nothing"
                .to_string(),
        ));
    }

    Ok(())
}

fn verify_runtime_inputs_unchanged(
    before: &tuisko_engine::Qwen38FlashNextQsaMoeLayerInputs,
    after: &tuisko_engine::Qwen38FlashNextQsaMoeLayerInputs,
) -> QualResult<usize> {
    let mut values = 0;
    if before.residual_input != after.residual_input {
        return Err(mismatch("the layer wrote its own input stream".to_string()));
    }
    values += before.residual_input.len();
    for (left, right, name) in [
        (&before.rope_cos, &after.rope_cos, "rotary cosines"),
        (&before.rope_sin, &after.rope_sin, "rotary sines"),
    ] {
        if left != right {
            return Err(mismatch(format!("the layer wrote its own {name}")));
        }
        values += left.len();
    }
    for (left, right, name) in [
        (&before.table_rows, &after.table_rows, "page-table rows"),
        (&before.lengths, &after.lengths, "visible lengths"),
        (&before.block_tables, &after.block_tables, "block table"),
        (&before.slot_table, &after.slot_table, "slot table"),
    ] {
        if left != right {
            return Err(mismatch(format!("the layer wrote its own {name}")));
        }
        values += left.len();
    }

    Ok(values)
}

fn verify_replay(
    eager: &Qwen38FlashNextQsaMoeLayerObservables,
    replay: &Qwen38FlashNextQsaMoeLayerObservables,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    macro_rules! same {
        ($field:ident, $name:literal) => {
            if eager.$field != replay.$field {
                return Err(mismatch(format!(
                    "{} differs between eager and replay",
                    $name
                )));
            }
            report.graph_replay_values += eager.$field.len();
        };
    }
    same!(hc_normalized, "hc_normalized");
    same!(hc_low_rank, "hc_low_rank");
    same!(hc_mixed, "hc_mixed");
    same!(hc_write_gate, "hc_write_gate");
    same!(qkv, "qkv");
    same!(query, "query");
    same!(attention, "attention");
    same!(attention_gated, "attention_gated");
    same!(key_pages, "key_pages");
    same!(value_pages, "value_pages");
    same!(block_keys, "block_keys");
    same!(indexer_ring, "indexer_ring");
    same!(indexer_qk, "indexer_qk");
    same!(indexer_query, "indexer_query");
    same!(selected, "selected");
    same!(selected_counts, "selected_counts");
    same!(attention_residual, "attention_residual");
    same!(router_logits, "router_logits");
    same!(expert_indices, "expert_indices");
    same!(routing_weights, "routing_weights");
    same!(routed_intermediate, "routed_intermediate");
    same!(routed_output, "routed_output");
    same!(shared_intermediate, "shared_intermediate");
    same!(shared_output, "shared_output");
    same!(shared_gate_logit, "shared_gate_logit");
    same!(block_output, "block_output");
    same!(residual_output, "residual_output");

    Ok(())
}

fn verify_replacement_input(
    rows: usize,
    first: &Qwen38FlashNextQsaMoeLayerObservables,
    replay: &Qwen38FlashNextQsaMoeLayerObservables,
) -> QualResult<()> {
    let active = rows * WIDTH;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(mismatch(format!(
            "{} reproduced its previous output after the input stream was replaced",
            route_label(rows)
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Qwen38FlashNextQsaMoeLayerObservables,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    let mut checked = 0usize;
    for (values, width, name) in [
        (&observed.hc_normalized, WIDTH, "hc_normalized"),
        (&observed.hc_mixed, HIDDEN, "hc_mixed"),
        (&observed.qkv, QKV_ROWS, "qkv"),
        (&observed.attention_gated, OUTPUT_COLUMNS, "attention_gated"),
        (&observed.router_logits, EXPERTS, "router_logits"),
        (&observed.block_output, HIDDEN, "block_output"),
        (&observed.residual_output, WIDTH, "residual_output"),
    ] {
        for (index, &value) in values.iter().enumerate().skip(rows * width) {
            if value != BF16_SENTINEL {
                return Err(mismatch(format!(
                    "{name} wrote inactive value {index} at {}",
                    route_label(rows)
                )));
            }
            checked += 1;
        }
    }
    report.inactive_values += checked;

    Ok(())
}

/// The full layer at one visible key, composed in `f64` from the checkpoint's own words.
#[allow(clippy::too_many_arguments)]
fn verify_layer_oracle(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
    snapshot: &CheckpointSnapshot<Qwen38FlashNext>,
    layer: usize,
    observed: &Qwen38FlashNextQsaMoeLayerObservables,
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    let hc = Qwen38FlashNextLayerHyperConnections::bind(snapshot, layer)?.materialize()?;
    let qsa = Qwen38FlashNextSparseAttentionBindings::bind(snapshot, layer)?.materialize()?;
    let moe = Qwen38FlashNextMoeBindings::bind(snapshot, layer)?.materialize()?;
    let immutable = program.qualification_immutable(stream)?;
    let stream_in = &observed_input(program, stream)?[..WIDTH];

    let attention_inject = hc
        .attention
        .block_inject
        .ok_or_else(|| mismatch("the attention bracket cannot write back".to_string()))?;
    let bracket = bracket_oracle(
        stream_in,
        &hc.attention.hc_norm.words().collect::<Vec<_>>(),
        &hc.attention.input_mix_down.words().collect::<Vec<_>>(),
        &hc.attention.input_mix_up.words().collect::<Vec<_>>(),
        &attention_inject.words().collect::<Vec<_>>(),
    );
    // The staging planes are reserved once and written by both brackets, so the attention
    // bracket is checked through `qkv` and `attention_residual` rather than through a plane
    // the MLP bracket has since overwritten.

    let qkv = projection_oracle(
        &bracket.mixed,
        &bf16_words(&qsa.qkv_weight_bf16)?,
        HIDDEN,
        QKV_ROWS,
    );
    compare_bf16("qkv", &observed.qkv[..QKV_ROWS], &qkv, report)?;

    // One visible key: the softmax is exactly 1.0, so the attention output is the value the
    // cache round-tripped through E4M3.
    //
    // The gate entry takes `attention` as `*mut f32` and gates it **in place** as well as
    // publishing the BF16 `activation`, so both planes hold the gated result after a full
    // launch. Checking them against one oracle at their two precisions is what proves the
    // in-place arm and the published arm agree.
    let gated_f32 = qsa_attention_oracle(&qkv);
    compare_f32(
        "attention (gated in place)",
        &observed.attention[..OUTPUT_COLUMNS],
        &gated_f32,
        report,
    )?;

    let gated = gated_f32
        .iter()
        .copied()
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    compare_bf16(
        "attention_gated",
        &observed.attention_gated[..OUTPUT_COLUMNS],
        &gated,
        report,
    )?;

    let block_output = projection_oracle(
        &gated,
        &qsa.output_weight.words().collect::<Vec<_>>(),
        OUTPUT_COLUMNS,
        HIDDEN,
    );
    let attention_residual = write_back(stream_in, &block_output, &bracket.write_gate);
    compare_bf16(
        "attention_residual",
        &observed.attention_residual[..WIDTH],
        &attention_residual,
        report,
    )?;

    let mlp_inject = hc
        .mlp
        .block_inject
        .ok_or_else(|| mismatch("the MLP bracket cannot write back".to_string()))?;
    let mlp = bracket_oracle(
        &attention_residual,
        &hc.mlp.hc_norm.words().collect::<Vec<_>>(),
        &hc.mlp.input_mix_down.words().collect::<Vec<_>>(),
        &hc.mlp.input_mix_up.words().collect::<Vec<_>>(),
        &mlp_inject.words().collect::<Vec<_>>(),
    );
    compare_bf16(
        "hc_mixed (MLP bracket, the surviving writer)",
        &observed.hc_mixed[..HIDDEN],
        &mlp.mixed,
        report,
    )?;
    let router = router_oracle(&mlp.mixed, &moe.router_weight.words().collect::<Vec<_>>());
    if observed.expert_indices[..TOP_K] != router.experts[..] {
        return Err(mismatch(
            "the device and the oracle selected different experts".to_string(),
        ));
    }
    report.oracle_values += TOP_K;

    let moe_output = moe_oracle(
        &mlp.mixed,
        &router,
        &(0..EXPERTS as u32).collect::<Vec<_>>(),
        &immutable.slot_pool,
        &immutable.expert_weight_scales_2,
        (
            &immutable.shared_gate_weight,
            &immutable.shared_up_weight,
            &immutable.shared_down_weight,
            &immutable.shared_gate_logit_weight,
        ),
    );
    let residual_output = write_back(&attention_residual, &moe_output, &mlp.write_gate);
    compare_bf16(
        "residual_output",
        &observed.residual_output[..WIDTH],
        &residual_output,
        report,
    )?;

    Ok(())
}

/// One-key gated attention in represented E4M3 cache precision.
///
/// Returns the in-place FP32 gate output; the published plane is its BF16 rounding.
pub(crate) fn qsa_attention_oracle(qkv: &[u16]) -> Vec<f32> {
    let attended = (0..OUTPUT_COLUMNS)
        .map(|index| {
            let head = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let kv_head = head / (QUERY_HEADS / KV_HEADS);
            let value =
                bf16_to_f32(qkv[QUERY_ROWS + KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dimension]);

            represent_e4m3(value, VALUE_CACHE_SCALE)
        })
        .collect::<Vec<_>>();

    (0..OUTPUT_COLUMNS)
        .map(|index| {
            let head = index / HEAD_DIM;
            let dimension = index % HEAD_DIM;
            let gate = bf16_to_f32(qkv[head * 2 * HEAD_DIM + HEAD_DIM + dimension]);

            attended[index] * logistic(gate)
        })
        .collect()
}

fn observed_input(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
) -> QualResult<Vec<u16>> {
    Ok(program.qualification_runtime_inputs(stream)?.residual_input)
}

/// Decodes signed cache E4M3 with the unsigned scale decoder.
pub(crate) fn decode_signed_e4m3(code: u8) -> f32 {
    let magnitude = decode_e4m3(code & 0x7f);

    if code & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// One value's round trip through the represented E4M3 cache plane.
pub(crate) fn represent_e4m3(value: f32, scale: f32) -> f32 {
    let scaled = value / scale;
    let mut best = 0u8;
    let mut best_error = f32::INFINITY;
    for code in 0..=255u8 {
        // 0x7f and 0xff are this format's NaN encodings and are never stored.
        if code & 0x7f == 0x7f {
            continue;
        }
        let error = (decode_signed_e4m3(code) - scaled).abs();
        if error < best_error {
            best_error = error;
            best = code;
        }
    }

    decode_signed_e4m3(best) * scale
}

pub(crate) fn logistic(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn bf16_words(bytes: &[u8]) -> QualResult<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(mismatch(
            "a BF16 source plane has an odd byte length".to_string(),
        ));
    }

    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    if actual.len() != expected.len() {
        return Err(mismatch(format!(
            "{role}: device published {} values, oracle {}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (&device, &oracle)) in actual.iter().zip(expected).enumerate() {
        let device = bf16_to_f32(device);
        let oracle = bf16_to_f32(oracle);
        let error = (device - oracle).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = (oracle.abs() * 0.05).max(0.02);
        if !device.is_finite() || error > tolerance {
            return Err(mismatch(format!(
                "{role} at value {index}: device={device}, oracle={oracle}, tolerance={tolerance}"
            )));
        }
        report.oracle_values += 1;
    }

    Ok(())
}

/// The FP32 attention plane, compared with the same relative band the BF16 planes use.
fn compare_f32(
    role: &str,
    actual: &[f32],
    expected: &[f32],
    report: &mut Qwen38FlashNextQsaMoeLayerQualification,
) -> QualResult<()> {
    for (index, (&device, &oracle)) in actual.iter().zip(expected).enumerate() {
        let error = (device - oracle).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = (oracle.abs() * 0.05).max(0.02);
        if !device.is_finite() || error > tolerance {
            return Err(mismatch(format!(
                "{role} at value {index}: device={device}, oracle={oracle}, tolerance={tolerance}"
            )));
        }
        report.oracle_values += 1;
    }

    Ok(())
}

fn verify_no_device_allocation(
    program: &Qwen38FlashNextQsaMoeLayerProgram,
    stream: &CudaStream,
) -> QualResult<()> {
    // Three warm passes before the snapshot, not one: the driver releases module-load
    // scratch lazily, and a counter taken too early reads that release as drift.
    for _ in 0..3 {
        for rows in EXACT_ROUTES {
            program.replay(stream, rows, Qwen38FlashNextQsaRoute::Dense)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in EXACT_ROUTES.iter().rev() {
            program.replay(stream, *rows, Qwen38FlashNextQsaRoute::Selected)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, MAX_ROWS, Qwen38FlashNextQsaMoeLayerQualificationError, ROTARY_ELEMENTS,
        qualify_qwen38_flash_next_qsa_moe_layer, round_for, route_label,
    };
    use tuisko_engine::MAX_BATCH;

    #[test]
    fn the_route_table_is_the_admitted_twelve() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(route_label(1_024), "T=1024");
    }

    #[test]
    fn the_oracle_round_keeps_one_visible_key_per_row() {
        let (table_rows, positions, lengths, cos, sin) = round_for(4);

        assert_eq!(table_rows, vec![0, 1, 2, 3]);
        assert_eq!(positions, vec![0; 4]);
        assert_eq!(lengths, vec![1; 4]);
        assert_eq!(cos.len(), 4 * ROTARY_ELEMENTS);
        assert_eq!(sin.len(), 4 * ROTARY_ELEMENTS);
        // At position zero the rotation is the identity, so the composition is checked without
        // this suite reimplementing MRoPE.
        assert!(cos.iter().all(|&value| value == 1.0));
        assert!(sin.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn a_wide_round_reuses_the_eight_carry_slots() {
        let (table_rows, ..) = round_for(32);

        assert_eq!(table_rows.len(), 32);
        assert!(table_rows.iter().all(|&row| (row as usize) < MAX_BATCH));
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer3_matches_the_layer_oracle_and_graph_replay()
    -> Result<(), Qwen38FlashNextQsaMoeLayerQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").ok_or_else(|| {
            Qwen38FlashNextQsaMoeLayerQualificationError::Mismatch(
                "TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required for the source-backed gate"
                    .to_string(),
            )
        })?;
        let report = qualify_qwen38_flash_next_qsa_moe_layer(std::path::Path::new(&root), 3)?;

        assert_eq!(report.weight_bytes, 141_775_360);
        assert_eq!(report.cache_bytes, 35_659_776);
        // Activations and metadata plus one caller-funded selection scratch plane.
        assert_eq!(report.workspace_bytes, 272_963_904);
        assert_eq!(report.refused_rounds, 2);
        assert_eq!(report.route_switch_values, 6_144);
        assert_eq!(report.selected_values, 4_099);
        assert!(report.oracle_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.runtime_input_values > 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
