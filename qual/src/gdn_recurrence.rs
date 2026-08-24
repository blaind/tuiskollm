//! Numerical and graph qualification for exact GDN recurrence routes.

use crate::device_benchmark::{preflight, require_current_process_exclusive};
use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::GdnRecurrenceOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const CAUSAL_ROUTES: [usize; 4] = [1, 2, 3, 4];
const ALIGNMENT: usize = 256;
const HEAD_DIM: usize = 128;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const RMS_EPSILON: f64 = 1.0e-6;
const DELTA_SCALE: f64 = 0.088_388_35;

/// Failure of the exact GDN recurrence qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum GdnRecurrenceQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("GDN recurrence qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact recurrence route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnRecurrenceQualification {
    /// FP32 state values compared with the FP64 transition formula.
    pub state_values: usize,
    /// BF16 gated-normalized outputs compared with the FP64 formula.
    pub output_values: usize,
    /// State and outputs reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Unmapped state and inactive outputs verified unchanged.
    pub inactive_values: usize,
    /// Caller-owned input, controls, mapping, and norm values verified unchanged.
    pub immutable_values: usize,
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
    output: Vec<u16>,
}

struct Oracle {
    state: Vec<f64>,
    output: Vec<f64>,
}

/// Qualifies eager and captured recurrence routes at exact decode and prefill rows.
pub fn qualify_gdn_recurrence()
-> Result<GdnRecurrenceQualification, GdnRecurrenceQualificationError> {
    preflight().map_err(|error| GdnRecurrenceQualificationError::Mismatch(error.to_string()))?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = GdnRecurrenceOp::new(&context)?;
    require_current_process_exclusive()
        .map_err(|error| GdnRecurrenceQualificationError::Mismatch(error.to_string()))?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = GdnRecurrenceQualification {
        state_values: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        payload_bytes: regions.payload_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_state_error: 0.0,
        maximum_output_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(rows, false, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &replay, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        graph.launch(&stream)?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "device addresses changed while qualifying row count {rows}"
            )));
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
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &replay, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        graph.launch(&stream)?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "device addresses changed while qualifying causal K={tokens}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;

    Ok(report)
}

fn verify_causal_k1_decode_agreement(
    causal: &Observed,
    decode: &Observed,
) -> Result<(), GdnRecurrenceQualificationError> {
    if let Some(index) = causal
        .state
        .iter()
        .zip(&decode.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "causal K=1 state differs bit-exactly from decode B=1 at {index}"
        )));
    }
    if let Some(index) = causal
        .output
        .iter()
        .zip(&decode.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "causal K=1 output differs bit-exactly from decode B=1 at {index}"
        )));
    }
    Ok(())
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let norm_weight = layout.reserve(HEAD_DIM, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let state = layout.reserve(MAX_BATCH * STATE_PER_ROW, ALIGNMENT)?;
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
            output,
        },
    ))
}

fn fixture() -> Fixture {
    const QK: [f32; 8] = [0.75, -0.625, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625];
    const VALUE: [f32; 8] = [0.5, -0.375, 0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.25];
    const GATE: [f32; 8] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    const NORM: [f32; 8] = [0.75, 0.875, 1.0, 1.125, 0.625, 1.25, 0.5, 1.5];
    let mut qkv = vec![0; MAX_ROWS * Qwen38_27B::GDN_QKV_ROWS];
    let mut projected = vec![0; MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS];
    for token in 0..MAX_ROWS {
        let qkv_base = token * Qwen38_27B::GDN_QKV_ROWS;
        for index in 0..QK_WIDTH {
            qkv[qkv_base + index] = f32_to_bf16(QK[(index + token) & 7]);
            qkv[qkv_base + QK_WIDTH + index] = f32_to_bf16(QK[(3 * index + token + 1) & 7]);
        }
        for index in 0..VALUE_WIDTH {
            qkv[qkv_base + 2 * QK_WIDTH + index] = f32_to_bf16(VALUE[(5 * index + token) & 7]);
            projected[token * Qwen38_27B::GDN_INPUT_ROWS + Qwen38_27B::GDN_QKV_ROWS + index] =
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
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 8]> {
    Ok([
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.projected)?.addr(),
        arena.address(regions.log_decay)?.addr(),
        arena.address(regions.beta)?.addr(),
        arena.address(regions.norm_weight)?.addr(),
        arena.address(regions.state_rows)?.addr(),
        arena.address(regions.state)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &GdnRecurrenceOp,
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
            arena.address(regions.output)?,
        )
    }
}

fn launch_causal(
    op: &GdnRecurrenceOp,
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
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn oracle(rows: usize, causal: bool, fixture: &Fixture) -> Oracle {
    let mut state = fixture
        .state
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; rows * VALUE_WIDTH];
    for token in 0..rows {
        let state_row = if !causal && rows <= MAX_BATCH {
            STATE_ROWS[token]
        } else {
            STATE_ROWS[0]
        };
        let qkv_base = token * Qwen38_27B::GDN_QKV_ROWS;
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
            let rms = (recurrent.iter().map(|value| value * value).sum::<f64>() / HEAD_DIM as f64
                + RMS_EPSILON)
                .sqrt();
            let gate_base = token * Qwen38_27B::GDN_INPUT_ROWS
                + Qwen38_27B::GDN_QKV_ROWS
                + value_head * HEAD_DIM;
            let output_base = token * VALUE_WIDTH + value_head * HEAD_DIM;
            for row in 0..HEAD_DIM {
                let gate = f64::from(bf16_to_f32(fixture.projected[gate_base + row]));
                let silu = gate / (1.0 + (-gate).exp());
                output[output_base + row] =
                    recurrent[row] / rms * f64::from(bf16_to_f32(fixture.norm_weight[row])) * silu;
            }
        }
    }

    Oracle { state, output }
}

fn verify_oracle(
    rows: usize,
    causal: bool,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnRecurrenceQualification,
) -> Result<(), GdnRecurrenceQualificationError> {
    verify_immutable(rows, fixture, observed)?;
    let expected = oracle(rows, causal, fixture);
    let active_state_rows = if !causal && rows <= MAX_BATCH {
        rows
    } else {
        1
    };
    for (index, (&actual, &expected)) in observed.state.iter().zip(&expected.state).enumerate() {
        let error = (f64::from(actual) - expected).abs() as f32;
        report.maximum_state_error = report.maximum_state_error.max(error);
        if error > 2.0e-4f32.max(expected.abs() as f32 * 0.002) {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "state at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
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
    for (index, &expected) in expected.output.iter().enumerate() {
        let actual = f64::from(bf16_to_f32(observed.output[index]));
        let error = (actual - expected).abs() as f32;
        report.maximum_output_error = report.maximum_output_error.max(error);
        if error > 0.015625f32.max(expected.abs() as f32 * 0.01) {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "output at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
        report.output_values += 1;
    }
    if let Some(relative) = observed.output[rows * VALUE_WIDTH..]
        .iter()
        .position(|&bits| bits != BF16_SENTINEL)
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} modified inactive output {}",
            rows * VALUE_WIDTH + relative
        )));
    }
    debug_assert!(active_state_rows <= MAX_BATCH);
    report.inactive_values += (MAX_ROWS - rows) * VALUE_WIDTH;
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_immutable(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), GdnRecurrenceQualificationError> {
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
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "rows={rows} modified immutable {name} value {index}"
            )));
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
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "rows={rows} modified immutable {name} value {index}"
            )));
        }
    }
    if observed.state_rows != STATE_ROWS {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} modified immutable state-row mapping"
        )));
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
    report: &mut GdnRecurrenceQualification,
) -> Result<(), GdnRecurrenceQualificationError> {
    verify_immutable(rows, fixture, replay)?;
    if let Some(index) = replay
        .state
        .iter()
        .zip(&eager.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} graph state value {index} differs from eager"
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} graph output value {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.state.len() + rows * VALUE_WIDTH;
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), GdnRecurrenceQualificationError> {
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
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            graph.launch(stream)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CAUSAL_ROUTES, EXACT_ROUTES, GdnRecurrenceQualificationError, MAX_BATCH, MAX_ROWS,
        STATE_PER_ROW, VALUE_WIDTH, fixture, immutable_values, layout, qualify_gdn_recurrence,
    };

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), GdnRecurrenceQualificationError> {
        let report = qualify_gdn_recurrence()?;
        let active_rows = EXACT_ROUTES.iter().chain(&CAUSAL_ROUTES).sum::<usize>();
        let route_count = EXACT_ROUTES.len() + CAUSAL_ROUTES.len();
        let active_state_rows = (1..=MAX_BATCH).sum::<usize>() + 4 + CAUSAL_ROUTES.len();
        let inactive_state_rows = route_count * MAX_BATCH - active_state_rows;
        let inactive_output_rows = EXACT_ROUTES
            .iter()
            .chain(&CAUSAL_ROUTES)
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();

        assert_eq!(report.state_values, active_state_rows * STATE_PER_ROW);
        assert_eq!(report.output_values, active_rows * VALUE_WIDTH);
        assert_eq!(
            report.graph_replay_values,
            2 * (route_count * MAX_BATCH * STATE_PER_ROW + active_rows * VALUE_WIDTH)
        );
        assert_eq!(
            report.inactive_values,
            inactive_state_rows * STATE_PER_ROW + inactive_output_rows * VALUE_WIDTH
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

        Ok(())
    }

    #[test]
    fn route_inventory_and_arena_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(CAUSAL_ROUTES, [1, 2, 3, 4]);
        let (layout, regions) = layout().unwrap();
        assert_eq!(layout.byte_len(), regions.payload_bytes() + 224);
    }
}
