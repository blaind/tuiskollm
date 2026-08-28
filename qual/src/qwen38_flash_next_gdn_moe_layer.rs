//! Source-backed qualification for the Qwen3.8-Flash-Next GDN/MoE layer.
//!
//! The independent `f64` oracle covers `B=1`; every admitted route checks eager/replay agreement,
//! inactive tails, immutable inputs, address stability, slot-table invariance, and device heap use.

use crate::device_benchmark::{self, DeviceBenchmarkError};
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16,
};
use crate::harness::graph_replay::first_moved_address;
use crate::qwen38_flash_next_layer_oracle::{
    EXPERTS, HIDDEN, TOP_K, bracket_oracle, moe_oracle, projection_oracle, round_bf16,
    router_oracle, write_back,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen38FlashNextGdnMoeLayerImmutable,
    Qwen38FlashNextGdnMoeLayerObservables, Qwen38FlashNextGdnMoeLayerProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextGdnBindings,
    Qwen38FlashNextLayerHyperConnections, Qwen38FlashNextMoeBindings,
};

type A = Qwen38FlashNext;

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const WIDTH: usize = A::HC_WIDTH;
const HEAD_DIM: usize = <A as Arch>::LINEAR_HEAD_DIM;
const KEY_HEADS: usize = <A as Arch>::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = <A as Arch>::LINEAR_VALUE_HEADS;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const GDN_INPUT_ROWS: usize = A::GDN_INPUT_ROWS;
const GDN_QKV_ROWS: usize = A::GDN_QKV_ROWS;
const RMS_EPSILON: f64 = 1.0e-6;
const DELTA_SCALE: f64 = 0.088_388_35;

/// Failure of the complete source-backed Qwen3.8-Flash-Next GDN/MoE layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextGdnMoeLayerQualificationError {
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
    #[error("Qwen3.8-Flash-Next GDN/MoE layer qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextGdnMoeLayerQualificationError>;

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextGdnMoeLayerQualification {
    /// Values checked against the layer-level oracle on real weights.
    pub oracle_values: usize,
    /// Mutable owner values reproduced by captured-graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and carry values verified unchanged.
    pub inactive_values: usize,
    /// Immutable source values proved unchanged.
    pub immutable_values: usize,
    /// Runtime-owned graph-input values proved unchanged.
    pub runtime_input_values: usize,
    /// Values proved byte-identical across a permuted slot assignment.
    pub permuted_identity_values: usize,
    /// Complete layer allocation bytes.
    pub arena_bytes: usize,
    /// Routed-expert pool allocation bytes.
    pub pool_arena_bytes: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable workspace and carry bytes.
    pub workspace_bytes: usize,
    /// Largest absolute difference from the layer oracle.
    pub maximum_absolute_error: f32,
}

/// Qualifies one source-backed GDN/MoE layer at every exact decode and prefill route.
pub fn qualify_qwen38_flash_next_gdn_moe_layer(
    root: &Path,
    layer: usize,
) -> QualResult<Qwen38FlashNextGdnMoeLayerQualification> {
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
        Qwen38FlashNextGdnMoeLayerProgram::from_snapshot(&context, snapshot.clone(), layer)?;
    let stable_base = program.base_address();
    let stable_pool_base = program.pool_base_address();
    let stable_addresses = program.qualification_addresses()?;

    let mut report = Qwen38FlashNextGdnMoeLayerQualification {
        oracle_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        runtime_input_values: 0,
        permuted_identity_values: 0,
        arena_bytes: program.arena_bytes(),
        pool_arena_bytes: program.pool_arena_bytes(),
        weight_bytes: program.resident_weight_bytes(),
        workspace_bytes: program.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        let first_input = make_stream(rows, 0);
        prepare_run(&program, &stream, rows, &first_input)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_stream(rows, 1);
        prepare_run(&program, &stream, rows, &input)?;
        let before = program.qualification_runtime_inputs(&stream)?;
        verify_runtime_input_contract(rows, &input, &before)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;
        report.runtime_input_values += verify_runtime_inputs_unchanged(
            &before,
            &program.qualification_runtime_inputs(&stream)?,
        )?;

        prepare_run(&program, &stream, rows, &input)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;

        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &replay, &mut report)?;
        verify_inactive(rows, &eager, &mut report)?;

        if rows == 1 {
            verify_layer_oracle(
                &program,
                &stream,
                snapshot.as_ref(),
                layer,
                &input,
                &replay,
                &mut report,
            )?;
        }

        if program.base_address() != stable_base
            || program.pool_base_address() != stable_pool_base
            || first_moved_address(&stable_addresses, &program.qualification_addresses()?).is_some()
        {
            return Err(mismatch(format!(
                "Qwen3.8-Flash-Next GDN/MoE layer addresses moved while qualifying {}",
                route_label(rows)
            )));
        }
    }

    verify_permuted_slot_assignment(&program, &stream, &mut report)?;
    verify_immutable(&program, &stream, snapshot.as_ref(), layer, &mut report)?;
    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn mismatch(message: String) -> Qwen38FlashNextGdnMoeLayerQualificationError {
    Qwen38FlashNextGdnMoeLayerQualificationError::Mismatch(message)
}

fn route_label(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

/// A deterministic BF16 stream whose branches are **not** copies of each other.
///
/// The four-branch fold and the per-branch write gates both collapse under branch symmetry, so a
/// fixture that repeats one 2,560-wide block four times would pass while the branch indexing was
/// wrong. Every branch gets its own offset into the pattern.
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

fn prepare_run(
    program: &Qwen38FlashNextGdnMoeLayerProgram,
    stream: &CudaStream,
    rows: usize,
    input: &[u16],
) -> QualResult<()> {
    program.reset_state(stream)?;
    program.load_residual(stream, rows, input)?;
    if program.layout().carries_ple_state() {
        program.load_engram_codes(stream, rows, &make_engram_codes(rows))?;
    }
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

/// Deterministic FP8 engram codes; the gather itself is the staging seam's own qualification.
fn make_engram_codes(rows: usize) -> Vec<u8> {
    let token_bytes = A::NGRAM_HEADS * A::NGRAM_HEAD_DIM;
    (0..rows * token_bytes)
        .map(|index| {
            // E4M3 codes in a benign exponent band, avoiding the NaN encoding 0x7f/0xff.
            let code = (index.wrapping_mul(37) % 120) as u8;
            code.wrapping_add(8)
        })
        .collect()
}

fn verify_runtime_input_contract(
    rows: usize,
    input: &[u16],
    actual: &tuisko_engine::Qwen38FlashNextGdnMoeLayerInputs,
) -> QualResult<()> {
    let active = rows * WIDTH;
    if actual.residual_input[..active] != *input {
        return Err(mismatch(
            "the layer's stable input storage does not hold the round's stream".to_string(),
        ));
    }
    if actual.state_rows != (0..MAX_BATCH as u32).collect::<Vec<_>>() {
        return Err(mismatch(
            "the carry routing table is not the identity this round expects".to_string(),
        ));
    }

    Ok(())
}

fn verify_runtime_inputs_unchanged(
    before: &tuisko_engine::Qwen38FlashNextGdnMoeLayerInputs,
    after: &tuisko_engine::Qwen38FlashNextGdnMoeLayerInputs,
) -> QualResult<usize> {
    let mut values = 0;
    if before.residual_input != after.residual_input {
        return Err(mismatch("the layer wrote its own input stream".to_string()));
    }
    values += before.residual_input.len();
    if before.state_rows != after.state_rows {
        return Err(mismatch(
            "the layer wrote its carry routing table".to_string(),
        ));
    }
    values += before.state_rows.len();
    if before.slot_table != after.slot_table {
        return Err(mismatch(
            "the layer wrote its expert indirection table".to_string(),
        ));
    }
    values += before.slot_table.len();
    if before.engram_codes != after.engram_codes {
        return Err(mismatch(
            "the layer wrote its staged engram codes".to_string(),
        ));
    }
    values += before.engram_codes.as_ref().map_or(0, Vec::len);

    Ok(values)
}

macro_rules! same {
    ($report:expr, $left:expr, $right:expr, $name:literal) => {
        if $left != $right {
            return Err(mismatch(format!(
                "{} differs between eager and replay",
                $name
            )));
        }
        $report.graph_replay_values += $left.len();
    };
}

fn verify_replay(
    rows: usize,
    eager: &Qwen38FlashNextGdnMoeLayerObservables,
    replay: &Qwen38FlashNextGdnMoeLayerObservables,
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
) -> QualResult<()> {
    let _ = rows;
    same!(
        report,
        eager.hc_normalized,
        replay.hc_normalized,
        "hc_normalized"
    );
    same!(report, eager.hc_low_rank, replay.hc_low_rank, "hc_low_rank");
    same!(report, eager.hc_mixed, replay.hc_mixed, "hc_mixed");
    same!(
        report,
        eager.hc_write_gate,
        replay.hc_write_gate,
        "hc_write_gate"
    );
    same!(
        report,
        eager.gdn_projected,
        replay.gdn_projected,
        "gdn_projected"
    );
    same!(
        report,
        eager.gdn_convolved,
        replay.gdn_convolved,
        "gdn_convolved"
    );
    same!(
        report,
        eager.gdn_log_decay,
        replay.gdn_log_decay,
        "gdn_log_decay"
    );
    same!(report, eager.gdn_beta, replay.gdn_beta, "gdn_beta");
    same!(
        report,
        eager.gdn_recurrent_output,
        replay.gdn_recurrent_output,
        "gdn_recurrent_output"
    );
    same!(report, eager.history, replay.history, "history");
    same!(report, eager.state, replay.state, "state");
    same!(
        report,
        eager.attention_residual,
        replay.attention_residual,
        "attention_residual"
    );
    same!(
        report,
        eager.router_logits,
        replay.router_logits,
        "router_logits"
    );
    same!(
        report,
        eager.expert_indices,
        replay.expert_indices,
        "expert_indices"
    );
    same!(
        report,
        eager.routing_weights,
        replay.routing_weights,
        "routing_weights"
    );
    same!(
        report,
        eager.routed_intermediate,
        replay.routed_intermediate,
        "routed_intermediate"
    );
    same!(
        report,
        eager.routed_output,
        replay.routed_output,
        "routed_output"
    );
    same!(
        report,
        eager.shared_intermediate,
        replay.shared_intermediate,
        "shared_intermediate"
    );
    same!(
        report,
        eager.shared_output,
        replay.shared_output,
        "shared_output"
    );
    same!(
        report,
        eager.shared_gate_logit,
        replay.shared_gate_logit,
        "shared_gate_logit"
    );
    same!(
        report,
        eager.block_output,
        replay.block_output,
        "block_output"
    );
    same!(
        report,
        eager.residual_output,
        replay.residual_output,
        "residual_output"
    );
    // The engram planes exist only on the layer that runs the module; absent on both sides is
    // agreement, present on one side is a composition bug.
    for (left, right, name) in [
        (&eager.ple_injected, &replay.ple_injected, "ple_injected"),
        (&eager.ple_gated, &replay.ple_gated, "ple_gated"),
        (&eager.ple_delta, &replay.ple_delta, "ple_delta"),
        (
            &eager.ple_conv_state,
            &replay.ple_conv_state,
            "ple_conv_state",
        ),
    ] {
        if left != right {
            return Err(mismatch(format!("{name} differs between eager and replay")));
        }
        report.graph_replay_values += left.as_ref().map_or(0, Vec::len);
    }

    Ok(())
}

/// A replay that ignored its replaced input would reproduce the first round's output.
fn verify_replacement_input(
    rows: usize,
    first: &Qwen38FlashNextGdnMoeLayerObservables,
    replay: &Qwen38FlashNextGdnMoeLayerObservables,
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
    observed: &Qwen38FlashNextGdnMoeLayerObservables,
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
) -> QualResult<()> {
    let mut checked = 0usize;
    let mut scan_u16 = |values: &[u16], width: usize, name: &str| -> QualResult<()> {
        for (index, &value) in values.iter().enumerate().skip(rows * width) {
            if value != BF16_SENTINEL {
                return Err(mismatch(format!(
                    "{name} wrote inactive value {index} at {}",
                    route_label(rows)
                )));
            }
            checked += 1;
        }
        Ok(())
    };
    scan_u16(&observed.hc_normalized, WIDTH, "hc_normalized")?;
    scan_u16(&observed.hc_mixed, HIDDEN, "hc_mixed")?;
    scan_u16(&observed.gdn_projected, GDN_INPUT_ROWS, "gdn_projected")?;
    scan_u16(&observed.gdn_convolved, GDN_QKV_ROWS, "gdn_convolved")?;
    scan_u16(
        &observed.gdn_recurrent_output,
        VALUE_WIDTH,
        "gdn_recurrent_output",
    )?;
    scan_u16(&observed.router_logits, EXPERTS, "router_logits")?;
    scan_u16(&observed.block_output, HIDDEN, "block_output")?;
    scan_u16(&observed.residual_output, WIDTH, "residual_output")?;

    for (index, &value) in observed
        .gdn_log_decay
        .iter()
        .enumerate()
        .skip(rows * VALUE_HEADS)
    {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(mismatch(format!(
                "gdn_log_decay wrote inactive value {index} at {}",
                route_label(rows)
            )));
        }
        checked += 1;
    }
    report.inactive_values += checked;

    Ok(())
}

/// The full layer, composed in `f64` from the checkpoint's own words.
#[allow(clippy::too_many_arguments)]
fn verify_layer_oracle(
    program: &Qwen38FlashNextGdnMoeLayerProgram,
    stream: &CudaStream,
    snapshot: &CheckpointSnapshot<Qwen38FlashNext>,
    layer: usize,
    input: &[u16],
    observed: &Qwen38FlashNextGdnMoeLayerObservables,
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
) -> QualResult<()> {
    let hc = Qwen38FlashNextLayerHyperConnections::bind(snapshot, layer)?.materialize()?;
    let gdn = Qwen38FlashNextGdnBindings::bind(snapshot, layer)?.materialize()?;
    let moe = Qwen38FlashNextMoeBindings::bind(snapshot, layer)?.materialize()?;
    let immutable = program.qualification_immutable(stream)?;

    // The engram module injects before the attention bracket, so the stream the bracket reads
    // is the device's injected stream on layer 1 and the raw input elsewhere. The engram
    // pipeline has its own qualified suite; this layer proves the composition around it.
    let stream_in = match &observed.ple_injected {
        Some(injected) => injected[..WIDTH].to_vec(),
        None => input[..WIDTH].to_vec(),
    };

    let attention_hc = hc
        .attention
        .block_inject
        .ok_or_else(|| mismatch("the attention bracket cannot write back".to_string()))?;
    let bracket = bracket_oracle(
        &stream_in,
        &hc.attention.hc_norm.words().collect::<Vec<_>>(),
        &hc.attention.input_mix_down.words().collect::<Vec<_>>(),
        &hc.attention.input_mix_up.words().collect::<Vec<_>>(),
        &attention_hc.words().collect::<Vec<_>>(),
    );
    // The four hyper-connection staging planes are reserved once and used by both brackets,
    // so after a full launch they hold the *MLP* bracket's values. The attention bracket is
    // therefore checked through what it produced -- `gdn_projected`, which is a function of
    // its `mixed`, and `attention_residual`, which is a function of its `write_gate`.

    // The fused input projection, which the prepare and recurrence entries both read.
    let input_weight = bf16_words(&gdn.input_weight_bf16)?;
    let projected = projection_oracle(&bracket.mixed, &input_weight, HIDDEN, GDN_INPUT_ROWS);
    compare_bf16(
        "gdn_projected",
        &observed.gdn_projected[..GDN_INPUT_ROWS],
        &projected,
        report,
    )?;

    // One decode token against a zeroed carry: the convolution sees three zero taps, and the
    // recurrence starts from a zero state, so the whole GDN block closes in one step.
    let recurrent = gdn_block_oracle(&bracket.mixed, &projected, &gdn)?;
    compare_bf16(
        "gdn_recurrent_output",
        &observed.gdn_recurrent_output[..VALUE_WIDTH],
        &recurrent,
        report,
    )?;

    let block_output = projection_oracle(
        &recurrent,
        &gdn.output_weight.words().collect::<Vec<_>>(),
        VALUE_WIDTH,
        HIDDEN,
    );
    // `block_output` is written twice per layer and by observation time holds the MoE
    // combine, so the attention arm is proved through the residual it produced: the
    // write-back is exact in FP32, so `attention_residual` matching pins `block_output`.
    let attention_residual = write_back(&stream_in, &block_output, &bracket.write_gate);
    compare_bf16(
        "attention_residual",
        &observed.attention_residual[..WIDTH],
        &attention_residual,
        report,
    )?;

    // --- MLP bracket ---
    let mlp_hc = hc
        .mlp
        .block_inject
        .ok_or_else(|| mismatch("the MLP bracket cannot write back".to_string()))?;
    let mlp = bracket_oracle(
        &attention_residual,
        &hc.mlp.hc_norm.words().collect::<Vec<_>>(),
        &hc.mlp.input_mix_down.words().collect::<Vec<_>>(),
        &hc.mlp.input_mix_up.words().collect::<Vec<_>>(),
        &mlp_hc.words().collect::<Vec<_>>(),
    );
    compare_bf16(
        "hc_mixed (MLP bracket, the surviving writer)",
        &observed.hc_mixed[..HIDDEN],
        &mlp.mixed,
        report,
    )?;
    compare_bf16(
        "hc_write_gate (MLP bracket, the surviving writer)",
        &observed.hc_write_gate[..A::HC_COUNT],
        &mlp.write_gate,
        report,
    )?;
    let router = router_oracle(&mlp.mixed, &moe.router_weight.words().collect::<Vec<_>>());
    compare_exact(
        "expert_indices",
        &observed.expert_indices[..TOP_K],
        &router.experts,
    )?;
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

/// The gated DeltaNet block for one decode token against a zeroed carry.
pub(crate) fn gdn_block_oracle(
    mixed: &[u16],
    projected: &[u16],
    gdn: &tuisko_model::MaterializedQwen38FlashNextGdn<'_>,
) -> QualResult<Vec<u16>> {
    let control_weight = bf16_words(&gdn.control_weight_bf16)?;
    let a_log = gdn.a_log.words().collect::<Vec<_>>();
    let dt_bias = gdn.dt_bias.words().collect::<Vec<_>>();
    let conv = gdn.convolution_weight.words().collect::<Vec<_>>();
    let norm = gdn.norm.words().collect::<Vec<_>>();

    // A/B control projections: the reference computes `beta = sigmoid(b)` in BF16 before the
    // fp32 cast, so beta is BF16-quantized and the oracle must quantize it too.
    let controls = projection_oracle(mixed, &control_weight, HIDDEN, 2 * VALUE_HEADS);
    let mut log_decay = vec![0.0f64; VALUE_HEADS];
    let mut beta = vec![0.0f64; VALUE_HEADS];
    for head in 0..VALUE_HEADS {
        let a = f64::from(bf16_to_f32(controls[head]));
        let b = f64::from(bf16_to_f32(controls[VALUE_HEADS + head]));
        let bias = f64::from(bf16_to_f32(dt_bias[head]));
        let decay = f64::from(bf16_to_f32(a_log[head])).exp();
        log_decay[head] = -decay * softplus(a + bias);
        beta[head] = f64::from(bf16_to_f32(f32_to_bf16(logistic(b) as f32)));
    }

    // Width-four causal convolution against a zeroed history: only the current tap survives,
    // and the activation is SiLU.
    let convolved = (0..GDN_QKV_ROWS)
        .map(|channel| {
            let tap = f64::from(bf16_to_f32(conv[channel * 4 + 3]));
            let value = tap * f64::from(bf16_to_f32(projected[channel]));
            value * logistic(value)
        })
        .collect::<Vec<_>>();

    // Per-head l2-normalized q/k, the delta rule from a zero state, then the gated norm.
    let mut output = vec![0u16; VALUE_WIDTH];
    for value_head in 0..VALUE_HEADS {
        let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
        let query = l2_normalize(&convolved[key_head * HEAD_DIM..(key_head + 1) * HEAD_DIM]);
        let key = l2_normalize(
            &convolved[QK_WIDTH + key_head * HEAD_DIM..QK_WIDTH + (key_head + 1) * HEAD_DIM],
        );
        let value_base = 2 * QK_WIDTH + value_head * HEAD_DIM;
        let value = &convolved[value_base..value_base + HEAD_DIM];

        // S starts at zero, so `S * exp(g)` is zero and `delta = v * beta`, `S = k (x) delta`,
        // and `o = S^T q = k.q * delta`.
        let alignment = query
            .iter()
            .zip(&key)
            .map(|(q, k)| q * k * DELTA_SCALE)
            .sum::<f64>();
        for dimension in 0..HEAD_DIM {
            let recurrent = alignment * value[dimension] * beta[value_head];
            output[value_head * HEAD_DIM + dimension] = f32_to_bf16(recurrent as f32);
        }
    }

    // Gated RMSNorm over each 128-wide value head, weight applied plain (not `1 + w`), then the
    // **sigmoid** gate on `z` -- never SiLU, which is this target's one divergence from Qwen3.5.
    let mut normed = vec![0u16; VALUE_WIDTH];
    for head in 0..VALUE_HEADS {
        let base = head * HEAD_DIM;
        let row = &output[base..base + HEAD_DIM];
        let squares = row
            .iter()
            .map(|&bits| {
                let value = f64::from(bf16_to_f32(bits));
                value * value
            })
            .sum::<f64>();
        let inverse = 1.0 / (squares / HEAD_DIM as f64 + RMS_EPSILON).sqrt();
        for dimension in 0..HEAD_DIM {
            let value = f64::from(bf16_to_f32(row[dimension])) * inverse;
            let scaled = f64::from(bf16_to_f32(norm[dimension])) * round_bf16(value);
            let z = f64::from(bf16_to_f32(projected[GDN_QKV_ROWS + base + dimension]));
            normed[base + dimension] = f32_to_bf16((scaled * logistic(z)) as f32);
        }
    }

    Ok(normed)
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn l2_normalize(values: &[f64]) -> Vec<f64> {
    let widened = values.to_vec();
    let squares = widened.iter().map(|value| value * value).sum::<f64>();
    let inverse = 1.0 / (squares + RMS_EPSILON).sqrt();
    widened.iter().map(|value| value * inverse).collect()
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

fn compare_exact<T: PartialEq + std::fmt::Debug>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> QualResult<()> {
    if actual != expected {
        return Err(mismatch(format!(
            "{role}: device and oracle selections differ"
        )));
    }

    Ok(())
}

/// One BF16 comparison with a relative band, tracking the worst absolute error seen.
fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
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
        // A composed layer folds many BF16 roundings, so the band is a relative one with an
        // absolute floor rather than a bitwise claim.
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

/// Moving every expert to a different slot must not move a bit of the layer's output.
fn verify_permuted_slot_assignment(
    program: &Qwen38FlashNextGdnMoeLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
) -> QualResult<()> {
    let rows = 1;
    let input = make_stream(rows, 3);
    prepare_run(program, stream, rows, &input)?;
    program.replay(stream, rows)?;
    let identity = program.qualification_observables(stream)?;

    // A rotation by one: every expert changes slot, and no two share one.
    let permuted = (0..EXPERTS as u32)
        .map(|expert| (expert + 1) % EXPERTS as u32)
        .collect::<Vec<_>>();
    let pool = program.qualification_immutable(stream)?.slot_pool;
    let mut rotated = vec![0u8; pool.len()];
    for expert in 0..EXPERTS {
        let slot = permuted[expert] as usize;
        let width = crate::qwen38_flash_next_layer_oracle::SLOT_BYTES;
        rotated[slot * width..(slot + 1) * width]
            .copy_from_slice(&pool[expert * width..(expert + 1) * width]);
    }
    program.qualification_load_slot_pool(stream, &rotated)?;
    program.load_slot_table(stream, &permuted)?;

    prepare_run(program, stream, rows, &input)?;
    program.replay(stream, rows)?;
    let moved = program.qualification_observables(stream)?;

    if identity.residual_output != moved.residual_output
        || identity.routed_output != moved.routed_output
    {
        return Err(mismatch(
            "a permuted slot assignment changed the layer's output".to_string(),
        ));
    }
    report.permuted_identity_values +=
        identity.residual_output.len() + identity.routed_output.len();

    Ok(())
}

fn verify_immutable(
    program: &Qwen38FlashNextGdnMoeLayerProgram,
    stream: &CudaStream,
    snapshot: &CheckpointSnapshot<Qwen38FlashNext>,
    layer: usize,
    report: &mut Qwen38FlashNextGdnMoeLayerQualification,
) -> QualResult<()> {
    let actual: Qwen38FlashNextGdnMoeLayerImmutable = program.qualification_immutable(stream)?;
    let hc = Qwen38FlashNextLayerHyperConnections::bind(snapshot, layer)?.materialize()?;
    let gdn = Qwen38FlashNextGdnBindings::bind(snapshot, layer)?.materialize()?;
    let moe = Qwen38FlashNextMoeBindings::bind(snapshot, layer)?.materialize()?;

    let mut check = |role: &str, actual: &[u16], expected: Vec<u16>| -> QualResult<()> {
        if actual != expected {
            return Err(mismatch(format!("{role} moved during qualification")));
        }
        report.immutable_values += actual.len();
        Ok(())
    };
    check(
        "attention hc_norm",
        &actual.attention_hc_norm,
        hc.attention.hc_norm.words().collect(),
    )?;
    check(
        "mlp hc_norm",
        &actual.mlp_hc_norm,
        hc.mlp.hc_norm.words().collect(),
    )?;
    check(
        "gdn convolution",
        &actual.gdn_convolution_weight,
        gdn.convolution_weight.words().collect(),
    )?;
    check("gdn norm", &actual.gdn_norm, gdn.norm.words().collect())?;
    check(
        "gdn out_proj",
        &actual.gdn_output_weight,
        gdn.output_weight.words().collect(),
    )?;
    check(
        "router",
        &actual.router_weight,
        moe.router_weight.words().collect(),
    )?;
    check(
        "gdn input projection",
        &actual.gdn_input_weight,
        bf16_words(&gdn.input_weight_bf16)?,
    )?;

    Ok(())
}

/// Replaying every admitted route after warmup must not move the driver's memory counters.
fn verify_no_device_allocation(
    program: &Qwen38FlashNextGdnMoeLayerProgram,
    stream: &CudaStream,
) -> QualResult<()> {
    // Three warm passes before the snapshot, not one: the driver releases module-load
    // scratch lazily, and a counter taken too early reads that release as drift.
    for _ in 0..3 {
        for rows in EXACT_ROUTES {
            program.replay(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in EXACT_ROUTES.iter().rev() {
            program.replay(stream, *rows)?;
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
        EXACT_ROUTES, MAX_ROWS, Qwen38FlashNextGdnMoeLayerQualificationError, WIDTH, make_stream,
        qualify_qwen38_flash_next_gdn_moe_layer, route_label,
    };
    use crate::qwen38_flash_next_layer_oracle::HIDDEN;
    use tuisko_engine::MAX_BATCH;
    use tuisko_model::Qwen38FlashNext;

    #[test]
    fn the_route_table_is_the_admitted_twelve() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(route_label(1), "B=1");
        assert_eq!(route_label(MAX_BATCH), "B=8");
        assert_eq!(route_label(1_024), "T=1024");
    }

    #[test]
    fn the_fixture_breaks_branch_symmetry() {
        // If the four branches were copies, the four-way fold and the per-branch write gates
        // would both pass while the branch indexing was wrong.
        let stream = make_stream(1, 0);
        let branches = (0..4)
            .map(|branch| &stream[branch * HIDDEN..(branch + 1) * HIDDEN])
            .collect::<Vec<_>>();
        for left in 0..4 {
            for right in (left + 1)..4 {
                assert_ne!(
                    branches[left], branches[right],
                    "branches {left} and {right} are copies"
                );
            }
        }
        assert_eq!(stream.len(), WIDTH);
    }

    #[test]
    fn a_salted_fixture_moves_every_row() {
        let first = make_stream(2, 0);
        let second = make_stream(2, 1);

        assert_eq!(first.len(), second.len());
        assert_ne!(first, second);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer0_matches_the_layer_oracle_and_graph_replay()
    -> Result<(), Qwen38FlashNextGdnMoeLayerQualificationError> {
        let report = run_snapshot_gate(0)?;

        assert_eq!(report.weight_bytes, 154_799_552);
        assert_eq!(report.workspace_bytes, 286_541_856);
        assert_eq!(report.pool_arena_bytes, 1_415_579_648);
        assert!(report.oracle_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.runtime_input_values > 0);
        assert!(report.permuted_identity_values > 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer1_composes_the_engram_module_it_alone_carries()
    -> Result<(), Qwen38FlashNextGdnMoeLayerQualificationError> {
        let report = run_snapshot_gate(Qwen38FlashNext::PLE_LAYER)?;

        // Layer 1 pays for the whole engram module and no other GDN layer does.
        assert_eq!(report.weight_bytes, 220_478_912);
        assert_eq!(report.workspace_bytes, 447_924_256);
        assert!(report.oracle_values > 0);
        assert!(report.graph_replay_values > 0);

        Ok(())
    }

    fn run_snapshot_gate(
        layer: usize,
    ) -> Result<
        super::Qwen38FlashNextGdnMoeLayerQualification,
        Qwen38FlashNextGdnMoeLayerQualificationError,
    > {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").ok_or_else(|| {
            Qwen38FlashNextGdnMoeLayerQualificationError::Mismatch(
                "TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required for the source-backed gate"
                    .to_string(),
            )
        })?;

        qualify_qwen38_flash_next_gdn_moe_layer(std::path::Path::new(&root), layer)
    }
}
