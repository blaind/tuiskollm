//! Numerical and graph qualification for exact Qwen3.8-Flash-Next GDN recurrence routes.
//!
//! The independent FP64 oracle covers the reused recurrence law and the
//! target-specific sigmoid output gate.

use crate::device_benchmark::{preflight, require_current_process_exclusive};
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16,
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::Qwen38FlashNextGdnRecurrenceOp;
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const CAUSAL_ROUTES: [usize; 4] = [1, 2, 3, 4];
const ALIGNMENT: usize = 256;
const HEAD_DIM: usize = Qwen38FlashNext::LINEAR_HEAD_DIM;
const KEY_HEADS: usize = Qwen38FlashNext::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = Qwen38FlashNext::LINEAR_VALUE_HEADS;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const RMS_EPSILON: f64 = 1.0e-6;
const DELTA_SCALE: f64 = 0.088_388_35;

/// Gate values that separate sigmoid from SiLU, including the decisive zero case.
const GATE: [f32; 8] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0];

/// Failure of the exact Qwen3.8-Flash-Next GDN recurrence qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextGdnRecurrenceQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next GDN recurrence qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact recurrence route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextGdnRecurrenceQualification {
    /// FP32 state values compared with the FP64 transition formula.
    pub state_values: usize,
    /// BF16 sigmoid-gated outputs compared with the FP64 formula.
    pub output_values: usize,
    /// FP32 prefill intermediates compared with the FP64 formula.
    pub recurrent_values: usize,
    /// State and outputs reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Unmapped state and inactive outputs verified unchanged.
    pub inactive_values: usize,
    /// Caller-owned input, controls, mapping, and norm values verified unchanged.
    pub immutable_values: usize,
    /// Outputs whose SiLU counterpart lies outside the acceptance contract.
    pub silu_separated_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact payload bytes, excluding alignment padding.
    pub payload_bytes: usize,
    /// Alignment padding bytes in the qualification arena.
    pub padding_bytes: usize,
    /// Largest absolute state error.
    pub maximum_state_error: f32,
    /// Largest absolute output error.
    pub maximum_output_error: f32,
    /// Largest absolute prefill-intermediate error.
    pub maximum_recurrent_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    qkv: ArenaRegion<u16>,
    projected: ArenaRegion<u16>,
    log_decay: ArenaRegion<f32>,
    beta: ArenaRegion<f32>,
    norm_weight: ArenaRegion<u16>,
    state_rows: ArenaRegion<u32>,
    state: ArenaRegion<f32>,
    recurrent_plane: ArenaRegion<f32>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.qkv.byte_len()
            + self.projected.byte_len()
            + self.log_decay.byte_len()
            + self.beta.byte_len()
            + self.norm_weight.byte_len()
            + self.state_rows.byte_len()
            + self.state.byte_len()
            + self.recurrent_plane.byte_len()
            + self.output.byte_len()
    }
}

struct Fixture {
    qkv: Vec<u16>,
    projected: Vec<u16>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    norm_weight: Vec<u16>,
    state: Vec<f32>,
}

struct Observed {
    qkv: Vec<u16>,
    projected: Vec<u16>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    norm_weight: Vec<u16>,
    state_rows: Vec<u32>,
    state: Vec<f32>,
    recurrent_plane: Vec<f32>,
    output: Vec<u16>,
}

/// Sigmoid oracle plus the rejected SiLU comparison.
struct Oracle {
    state: Vec<f64>,
    recurrent: Vec<f64>,
    output: Vec<f64>,
    silu_output: Vec<f64>,
}

/// Qualifies eager and captured Qwen3.8-Flash-Next recurrence routes at exact rows.
pub fn qualify_qwen38_flash_next_gdn_recurrence()
-> Result<Qwen38FlashNextGdnRecurrenceQualification, Qwen38FlashNextGdnRecurrenceQualificationError>
{
    preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = Qwen38FlashNextGdnRecurrenceOp::new(&context)?;
    require_current_process_exclusive()?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen38FlashNextGdnRecurrenceQualification {
        state_values: 0,
        output_values: 0,
        recurrent_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        silu_separated_values: 0,
        arena_bytes: layout.byte_len(),
        payload_bytes: regions.payload_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_state_error: 0.0,
        maximum_output_error: 0.0,
        maximum_recurrent_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(rows, false, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &replay, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!("device addresses changed while qualifying row count {rows}"),
            ));
        }
    }

    for tokens in CAUSAL_ROUTES {
        reset(&arena, &stream, regions, &fixture)?;
        launch_causal(&op, &arena, &stream, regions, tokens)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(tokens, true, &fixture, &eager, &mut report)?;
        if tokens == 1 {
            reset(&arena, &stream, regions, &fixture)?;
            launch(&op, &arena, &stream, regions, 1)?;
            let decode = observe(&arena, &stream, regions)?;
            verify_causal_k1_decode_agreement(&eager, &decode)?;
        }

        reset(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch_causal(&op, &arena, &stream, regions, tokens)
        })?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &replay, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!("device addresses changed while qualifying causal K={tokens}"),
            ));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;

    Ok(report)
}

fn verify_causal_k1_decode_agreement(
    causal: &Observed,
    decode: &Observed,
) -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
    if let Some(index) = causal
        .state
        .iter()
        .zip(&decode.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("causal K=1 state differs bit-exactly from decode B=1 at {index}"),
        ));
    }
    if let Some(index) = causal
        .output
        .iter()
        .zip(&decode.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("causal K=1 output differs bit-exactly from decode B=1 at {index}"),
        ));
    }
    Ok(())
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_INPUT_ROWS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let norm_weight = layout.reserve(HEAD_DIM, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let state = layout.reserve(MAX_BATCH * STATE_PER_ROW, ALIGNMENT)?;
    let recurrent_plane = layout.reserve(MAX_ROWS * VALUE_WIDTH, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * VALUE_WIDTH, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            qkv,
            projected,
            log_decay,
            beta,
            norm_weight,
            state_rows,
            state,
            recurrent_plane,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    const QK: [f32; 8] = [0.75, -0.625, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625];
    const VALUE: [f32; 8] = [0.5, -0.375, 0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.25];
    const NORM: [f32; 8] = [0.75, 0.875, 1.0, 1.125, 0.625, 1.25, 0.5, 1.5];
    let mut qkv = vec![0; MAX_ROWS * Qwen38FlashNext::GDN_QKV_ROWS];
    let mut projected = vec![0; MAX_ROWS * Qwen38FlashNext::GDN_INPUT_ROWS];
    for token in 0..MAX_ROWS {
        let qkv_base = token * Qwen38FlashNext::GDN_QKV_ROWS;
        for index in 0..QK_WIDTH {
            qkv[qkv_base + index] = f32_to_bf16(QK[(index + token) & 7]);
            qkv[qkv_base + QK_WIDTH + index] = f32_to_bf16(QK[(3 * index + token + 1) & 7]);
        }
        for index in 0..VALUE_WIDTH {
            qkv[qkv_base + 2 * QK_WIDTH + index] = f32_to_bf16(VALUE[(5 * index + token) & 7]);
            projected
                [token * Qwen38FlashNext::GDN_INPUT_ROWS + Qwen38FlashNext::GDN_QKV_ROWS + index] =
                f32_to_bf16(GATE[(index + index / HEAD_DIM + token) & 7]);
        }
    }
    let log_decay = (0..MAX_ROWS * VALUE_HEADS)
        .map(|index| -0.125 - (index & 7) as f32 * 0.03125)
        .collect();
    let beta = (0..MAX_ROWS * VALUE_HEADS)
        .map(|index| 0.25 + (index & 3) as f32 * 0.125)
        .collect();
    let norm_weight = (0..HEAD_DIM)
        .map(|index| f32_to_bf16(NORM[index & 7]))
        .collect();
    let state = (0..MAX_BATCH * STATE_PER_ROW)
        .map(|index| ((index.wrapping_mul(13) & 31) as f32 - 15.5) / 2048.0)
        .collect();

    Fixture {
        qkv,
        projected,
        log_decay,
        beta,
        norm_weight,
        state,
    }
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.projected, &fixture.projected)?;
    arena.copy_from_host(stream, regions.log_decay, &fixture.log_decay)?;
    arena.copy_from_host(stream, regions.beta, &fixture.beta)?;
    arena.copy_from_host(stream, regions.norm_weight, &fixture.norm_weight)?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)
}

fn reset(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.state, &fixture.state)?;
    arena.fill(stream, regions.recurrent_plane, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 9]> {
    Ok([
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.projected)?.addr(),
        arena.address(regions.log_decay)?.addr(),
        arena.address(regions.beta)?.addr(),
        arena.address(regions.norm_weight)?.addr(),
        arena.address(regions.state_rows)?.addr(),
        arena.address(regions.state)?.addr(),
        arena.address(regions.recurrent_plane)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &Qwen38FlashNextGdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: all regions cover their maximum extents and every mapped row is
    // below the eight-row state capacity.
    unsafe {
        op.launch(
            stream,
            rows,
            arena.address(regions.qkv)?,
            arena.address(regions.projected)?,
            arena.address(regions.log_decay)?,
            arena.address(regions.beta)?,
            arena.address(regions.norm_weight)?,
            arena.address(regions.state_rows)?,
            arena.address(regions.state)?,
            arena.address(regions.recurrent_plane)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_causal(
    op: &Qwen38FlashNextGdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: all exact causal rows share STATE_ROWS[0]; every plane covers
    // K=4 and the selected state row is below eight.
    unsafe {
        op.launch_causal(
            stream,
            tokens,
            arena.address(regions.qkv)?,
            arena.address(regions.projected)?,
            arena.address(regions.log_decay)?,
            arena.address(regions.beta)?,
            arena.address(regions.norm_weight)?,
            arena.address(regions.state_rows)?,
            arena.address(regions.state)?,
            arena.address(regions.recurrent_plane)?,
            arena.address(regions.output)?,
        )
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        qkv: arena.copy_to_host(stream, regions.qkv)?,
        projected: arena.copy_to_host(stream, regions.projected)?,
        log_decay: arena.copy_to_host(stream, regions.log_decay)?,
        beta: arena.copy_to_host(stream, regions.beta)?,
        norm_weight: arena.copy_to_host(stream, regions.norm_weight)?,
        state_rows: arena.copy_to_host(stream, regions.state_rows)?,
        state: arena.copy_to_host(stream, regions.state)?,
        recurrent_plane: arena.copy_to_host(stream, regions.recurrent_plane)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

/// `sigmoid(z)`, written from its own definition rather than from the kernel's
/// `1 / (1 + exp2(-z log2 e))` formulation.
fn sigmoid(gate: f64) -> f64 {
    1.0 / (1.0 + (-gate).exp())
}

/// `silu(z) = z * sigmoid(z)`, the activation this target must NOT apply.
fn silu(gate: f64) -> f64 {
    gate * sigmoid(gate)
}

fn oracle(rows: usize, causal: bool, fixture: &Fixture) -> Oracle {
    let mut state = fixture
        .state
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let mut recurrent_plane = vec![0.0; rows * VALUE_WIDTH];
    let mut output = vec![0.0; rows * VALUE_WIDTH];
    let mut silu_output = vec![0.0; rows * VALUE_WIDTH];
    let decode_state_rows = if !causal && rows <= MAX_BATCH {
        &STATE_ROWS[..rows]
    } else {
        &[]
    };
    for token in 0..rows {
        let state_row = decode_state_rows
            .get(token)
            .copied()
            .unwrap_or(STATE_ROWS[0]);
        let qkv_base = token * Qwen38FlashNext::GDN_QKV_ROWS;
        let mut query = vec![[0.0f64; HEAD_DIM]; KEY_HEADS];
        let mut key = vec![[0.0f64; HEAD_DIM]; KEY_HEADS];
        for head in 0..KEY_HEADS {
            for (plane, destination) in [(0, &mut query), (1, &mut key)] {
                let base = qkv_base + plane * QK_WIDTH + head * HEAD_DIM;
                let sum = fixture.qkv[base..base + HEAD_DIM]
                    .iter()
                    .map(|&bits| f64::from(bf16_to_f32(bits)).powi(2))
                    .sum::<f64>();
                let inverse = 1.0 / (sum + RMS_EPSILON).sqrt();
                for (column, destination) in destination[head].iter_mut().enumerate() {
                    *destination = f64::from(bf16_to_f32(fixture.qkv[base + column])) * inverse;
                }
            }
        }
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
            let control = token * VALUE_HEADS + value_head;
            let decay = f64::from(fixture.log_decay[control]).exp();
            let beta = f64::from(fixture.beta[control]);
            let state_base = (state_row as usize * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM;
            let value_base = qkv_base + 2 * QK_WIDTH + value_head * HEAD_DIM;
            let mut recurrent = [0.0f64; HEAD_DIM];
            for (row, recurrent) in recurrent.iter_mut().enumerate() {
                let row_base = state_base + row * HEAD_DIM;
                let state_key = (0..HEAD_DIM)
                    .map(|column| state[row_base + column] * key[key_head][column])
                    .sum::<f64>();
                let update = beta
                    * (f64::from(bf16_to_f32(fixture.qkv[value_base + row])) - decay * state_key);
                for column in 0..HEAD_DIM {
                    state[row_base + column] =
                        decay * state[row_base + column] + update * key[key_head][column];
                    *recurrent += state[row_base + column] * query[key_head][column];
                }
                *recurrent *= DELTA_SCALE;
            }
            let output_base = token * VALUE_WIDTH + value_head * HEAD_DIM;
            recurrent_plane[output_base..output_base + HEAD_DIM].copy_from_slice(&recurrent);
            let rms = (recurrent.iter().map(|value| value * value).sum::<f64>() / HEAD_DIM as f64
                + RMS_EPSILON)
                .sqrt();
            let gate_base = token * Qwen38FlashNext::GDN_INPUT_ROWS
                + Qwen38FlashNext::GDN_QKV_ROWS
                + value_head * HEAD_DIM;
            for row in 0..HEAD_DIM {
                let gate = f64::from(bf16_to_f32(fixture.projected[gate_base + row]));
                let normalized =
                    recurrent[row] / rms * f64::from(bf16_to_f32(fixture.norm_weight[row]));
                output[output_base + row] = normalized * sigmoid(gate);
                silu_output[output_base + row] = normalized * silu(gate);
            }
        }
    }

    Oracle {
        state,
        recurrent: recurrent_plane,
        output,
        silu_output,
    }
}

/// Per-value BF16 output acceptance contract.
fn output_tolerance(expected: f64) -> f32 {
    0.015_625f32.max(expected.abs() as f32 * 0.01)
}

fn verify_oracle(
    rows: usize,
    causal: bool,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnRecurrenceQualification,
) -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
    verify_immutable(rows, fixture, observed)?;
    let oracle = oracle(rows, causal, fixture);
    for (index, (&actual, &expected)) in observed.state.iter().zip(&oracle.state).enumerate() {
        let error = (f64::from(actual) - expected).abs() as f32;
        report.maximum_state_error = report.maximum_state_error.max(error);
        if error > 2.0e-4f32.max(expected.abs() as f32 * 0.002) {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!(
                    "state at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
                ),
            ));
        }
        let state_row = (index / STATE_PER_ROW) as u32;
        let active = if !causal && rows <= MAX_BATCH {
            STATE_ROWS[..rows].contains(&state_row)
        } else {
            state_row == STATE_ROWS[0]
        };
        if active {
            report.state_values += 1;
        } else {
            report.inactive_values += 1;
        }
    }
    for (index, &expected) in oracle.output.iter().enumerate() {
        let actual = f64::from(bf16_to_f32(observed.output[index]));
        let error = (actual - expected).abs() as f32;
        report.maximum_output_error = report.maximum_output_error.max(error);
        if error > output_tolerance(expected) {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!(
                    "output at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
                ),
            ));
        }
        report.output_values += 1;
        // Count values for which SiLU lies outside the sigmoid contract.
        let silu = oracle.silu_output[index];
        if (actual - silu).abs() as f32 > output_tolerance(silu) {
            report.silu_separated_values += 1;
        }
    }
    let prefill = causal || rows > MAX_BATCH;
    if prefill {
        for (index, &expected) in oracle.recurrent.iter().enumerate() {
            let actual = observed.recurrent_plane[index];
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_recurrent_error = report.maximum_recurrent_error.max(error);
            if error > 2.0e-4f32.max(expected.abs() as f32 * 0.002) {
                return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                    format!(
                        "recurrent plane at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
                    ),
                ));
            }
            report.recurrent_values += 1;
        }
    }
    let recurrent_start = usize::from(prefill) * rows * VALUE_WIDTH;
    if let Some(relative) = observed.recurrent_plane[recurrent_start..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!(
                "rows={rows} modified inactive recurrent value {}",
                recurrent_start + relative
            ),
        ));
    }
    report.inactive_values += observed.recurrent_plane.len() - recurrent_start;
    if let Some(relative) = observed.output[rows * VALUE_WIDTH..]
        .iter()
        .position(|&bits| bits != BF16_SENTINEL)
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!(
                "rows={rows} modified inactive output {}",
                rows * VALUE_WIDTH + relative
            ),
        ));
    }
    report.inactive_values += (MAX_ROWS - rows) * VALUE_WIDTH;
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_immutable(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
    for (name, actual, expected) in [
        ("qkv", observed.qkv.as_slice(), fixture.qkv.as_slice()),
        (
            "projected",
            observed.projected.as_slice(),
            fixture.projected.as_slice(),
        ),
        (
            "norm_weight",
            observed.norm_weight.as_slice(),
            fixture.norm_weight.as_slice(),
        ),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!("rows={rows} modified immutable {name} value {index}"),
            ));
        }
    }
    for (name, actual, expected) in [
        (
            "log_decay",
            observed.log_decay.as_slice(),
            fixture.log_decay.as_slice(),
        ),
        ("beta", observed.beta.as_slice(), fixture.beta.as_slice()),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
                format!("rows={rows} modified immutable {name} value {index}"),
            ));
        }
    }
    if observed.state_rows != STATE_ROWS {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("rows={rows} modified immutable state-row mapping"),
        ));
    }

    Ok(())
}

fn immutable_values(fixture: &Fixture) -> usize {
    fixture.qkv.len()
        + fixture.projected.len()
        + fixture.log_decay.len()
        + fixture.beta.len()
        + fixture.norm_weight.len()
        + STATE_ROWS.len()
}

fn verify_replay(
    rows: usize,
    fixture: &Fixture,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen38FlashNextGdnRecurrenceQualification,
) -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
    verify_immutable(rows, fixture, replay)?;
    if let Some(index) = replay
        .state
        .iter()
        .zip(&eager.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("rows={rows} graph state value {index} differs from eager"),
        ));
    }
    if let Some(index) = replay
        .recurrent_plane
        .iter()
        .zip(&eager.recurrent_plane)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("rows={rows} graph recurrent value {index} differs from eager"),
        ));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("rows={rows} graph output value {index} differs from eager"),
        ));
    }
    report.graph_replay_values +=
        replay.state.len() + replay.recurrent_plane.len() + rows * VALUE_WIDTH;
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen38FlashNextGdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
    let mut graphs = Vec::with_capacity(EXACT_ROUTES.len() + CAUSAL_ROUTES.len());
    for rows in EXACT_ROUTES {
        reset(arena, stream, regions, fixture)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch(op, arena, stream, regions, rows)
        })?);
    }
    for tokens in CAUSAL_ROUTES {
        reset(arena, stream, regions, fixture)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch_causal(op, arena, stream, regions, tokens)
        })?);
    }
    for graph in &graphs {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen38FlashNextGdnRecurrenceQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CAUSAL_ROUTES, EXACT_ROUTES, GATE, MAX_BATCH, MAX_ROWS,
        Qwen38FlashNextGdnRecurrenceQualificationError, STATE_PER_ROW, VALUE_WIDTH, fixture,
        immutable_values, layout, output_tolerance, qualify_qwen38_flash_next_gdn_recurrence,
        sigmoid, silu,
    };

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen38FlashNextGdnRecurrenceQualificationError> {
        let report = qualify_qwen38_flash_next_gdn_recurrence()?;
        let active_rows = EXACT_ROUTES.iter().chain(&CAUSAL_ROUTES).sum::<usize>();
        let route_count = EXACT_ROUTES.len() + CAUSAL_ROUTES.len();
        let active_state_rows = (1..=MAX_BATCH).sum::<usize>() + 4 + CAUSAL_ROUTES.len();
        let inactive_state_rows = route_count * MAX_BATCH - active_state_rows;
        let inactive_output_rows = EXACT_ROUTES
            .iter()
            .chain(&CAUSAL_ROUTES)
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();
        let prefill_rows = EXACT_ROUTES
            .iter()
            .copied()
            .filter(|&rows| rows > MAX_BATCH)
            .chain(CAUSAL_ROUTES)
            .sum::<usize>();
        let inactive_recurrent_rows = route_count * MAX_ROWS - prefill_rows;

        assert_eq!(report.state_values, active_state_rows * STATE_PER_ROW);
        assert_eq!(report.output_values, active_rows * VALUE_WIDTH);
        assert_eq!(report.recurrent_values, prefill_rows * VALUE_WIDTH);
        assert_eq!(
            report.graph_replay_values,
            2 * (route_count * MAX_BATCH * STATE_PER_ROW
                + route_count * MAX_ROWS * VALUE_WIDTH
                + active_rows * VALUE_WIDTH)
        );
        assert_eq!(
            report.inactive_values,
            inactive_state_rows * STATE_PER_ROW
                + (inactive_output_rows + inactive_recurrent_rows) * VALUE_WIDTH
        );
        assert_eq!(
            report.immutable_values,
            3 * route_count * immutable_values(&fixture())
        );
        assert_eq!(
            report.arena_bytes,
            report.payload_bytes + report.padding_bytes
        );
        assert!(report.maximum_state_error <= 0.002);
        assert!(report.maximum_output_error <= 0.03125);
        assert!(report.maximum_recurrent_error <= 0.002);
        // A SiLU epilogue would be rejected outright on most emitted values;
        // requiring a majority keeps this from passing on rounding alone.
        assert!(report.silu_separated_values * 2 > report.output_values);

        Ok(())
    }

    #[test]
    fn sigmoid_gate_separates_from_silu() {
        assert_eq!(GATE, [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0]);
        // `silu(0) = 0` exactly while `sigmoid(0) = 1/2`: the decisive case.
        assert_eq!(silu(0.0), 0.0);
        assert_eq!(sigmoid(0.0), 0.5);
        // Negative gates flip sign between the two activations.
        for gate in [-4.0, -1.0, -0.25] {
            assert!(silu(gate) < 0.0, "silu({gate})");
            assert!(sigmoid(gate) > 0.0, "sigmoid({gate})");
        }
        // `z = 1` is the one coincidence point; every other fixture value
        // separates by more than a unit-magnitude row's tolerance.
        assert_eq!(silu(1.0), sigmoid(1.0));
        let coincident = GATE
            .iter()
            .filter(|&&gate| {
                let gate = f64::from(gate);
                (silu(gate) - sigmoid(gate)).abs() as f32 <= output_tolerance(sigmoid(gate))
            })
            .count();
        assert_eq!(coincident, 1);
    }

    #[test]
    fn route_inventory_and_arena_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(CAUSAL_ROUTES, [1, 2, 3, 4]);
        let (layout, regions) = layout().unwrap();
        assert_eq!(layout.byte_len(), regions.payload_bytes() + 224);
    }
}
