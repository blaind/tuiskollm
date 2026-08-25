//! Numerical and graph qualification for Qwen3.5 GDN recurrence.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::target::Qwen35GdnRecurrenceOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 128;
pub(crate) const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, MAX_ROWS];
pub(crate) const CAUSAL_ROUTES: [usize; 3] = [2, 3, 4];
const ALIGNMENT: usize = 256;
pub(crate) const HEAD_DIM: usize = Qwen35_9B::LINEAR_HEAD_DIM;
pub(crate) const KEY_HEADS: usize = Qwen35_9B::LINEAR_KEY_HEADS;
pub(crate) const VALUE_HEADS: usize = Qwen35_9B::LINEAR_VALUE_HEADS;
pub(crate) const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
pub(crate) const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
pub(crate) const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
pub(crate) const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const RMS_EPSILON: f64 = 1.0e-6;
const QUERY_SCALE: f64 = 0.088_388_35;

/// Failure of the exact Qwen3.5 recurrence qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35GdnRecurrenceQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 GDN recurrence qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact recurrence route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35GdnRecurrenceQualification {
    /// FP32 state values compared with the FP64 transition formula.
    pub state_values: usize,
    /// BF16 gated-normalized outputs compared with the FP64 formula.
    pub output_values: usize,
    /// State and outputs reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Unmapped state and inactive outputs verified unchanged.
    pub inactive_values: usize,
    /// Read-only source values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact immutable parameter bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input, state, and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute state error.
    pub maximum_state_error: f32,
    /// Largest absolute output error.
    pub maximum_output_error: f32,
}

/// Qwen3.6 qualification report for the shared exact-geometry recurrence.
pub type Qwen36GdnRecurrenceQualification = Qwen35GdnRecurrenceQualification;

/// Qwen3.6 qualification failure for the shared exact-geometry recurrence.
pub type Qwen36GdnRecurrenceQualificationError = Qwen35GdnRecurrenceQualificationError;

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) projected: ArenaRegion<u16>,
    pub(crate) log_decay: ArenaRegion<f32>,
    pub(crate) beta: ArenaRegion<f32>,
    pub(crate) norm_weight: ArenaRegion<u16>,
    pub(crate) state_rows: ArenaRegion<u32>,
    pub(crate) state: ArenaRegion<f32>,
    pub(crate) recurrent_plane: ArenaRegion<f32>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.norm_weight.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
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

pub(crate) struct Fixture {
    pub(crate) qkv: Vec<u16>,
    pub(crate) projected: Vec<u16>,
    pub(crate) log_decay: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) norm_weight: Vec<u16>,
    pub(crate) state: Vec<f32>,
}

struct Observed {
    state: Vec<f32>,
    output: Vec<u16>,
}

struct Oracle {
    state: Vec<f64>,
    output: Vec<f64>,
}

/// Qualifies eager and captured Qwen3.5 recurrence at every exact route.
pub fn qualify_qwen35_gdn_recurrence()
-> Result<Qwen35GdnRecurrenceQualification, Qwen35GdnRecurrenceQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = make_fixture();
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let op = Qwen35GdnRecurrenceOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35GdnRecurrenceQualification {
        state_values: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_state_error: 0.0,
        maximum_output_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset_state(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(rows, false, &fixture, &eager, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
                "device addresses changed while qualifying row count {rows}"
            )));
        }
    }
    for rows in CAUSAL_ROUTES {
        reset_state(&arena, &stream, regions, &fixture)?;
        launch_causal(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(rows, true, &fixture, &eager, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch_causal(&op, &arena, &stream, regions, rows)
        })?;
        // SAFETY: every captured allocation remains live through synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

/// Qualifies Qwen3.6 with the same binary and independent FP64 state oracle.
///
/// Kernel compile-time assertions bind both profiles to the exact 16-QK-head,
/// 32-value-head, width-128 state and gate mapping.
pub fn qualify_qwen36_gdn_recurrence()
-> Result<Qwen36GdnRecurrenceQualification, Qwen36GdnRecurrenceQualificationError> {
    qualify_qwen35_gdn_recurrence()
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_ROWS * Qwen35_9B::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen35_9B::GDN_INPUT_ROWS, ALIGNMENT)?;
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

pub(crate) fn make_fixture() -> Fixture {
    const QK: [f32; 8] = [0.75, -0.625, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625];
    const VALUE: [f32; 8] = [0.5, -0.375, 0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.25];
    const GATE: [f32; 8] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    const NORM: [f32; 8] = [0.75, 0.875, 1.0, 1.125, 0.625, 1.25, 0.5, 1.5];
    let mut qkv = vec![0; MAX_ROWS * Qwen35_9B::GDN_QKV_ROWS];
    let mut projected = vec![0; MAX_ROWS * Qwen35_9B::GDN_INPUT_ROWS];
    for token in 0..MAX_ROWS {
        let qkv_base = token * Qwen35_9B::GDN_QKV_ROWS;
        for index in 0..QK_WIDTH {
            qkv[qkv_base + index] = f32_to_bf16(QK[(index + token) & 7]);
            qkv[qkv_base + QK_WIDTH + index] = f32_to_bf16(QK[(3 * index + token + 1) & 7]);
        }
        for index in 0..VALUE_WIDTH {
            qkv[qkv_base + 2 * QK_WIDTH + index] = f32_to_bf16(VALUE[(5 * index + token) & 7]);
            projected[token * Qwen35_9B::GDN_INPUT_ROWS + Qwen35_9B::GDN_QKV_ROWS + index] =
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

pub(crate) fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.projected, &fixture.projected)?;
    arena.copy_from_host(stream, regions.log_decay, &fixture.log_decay)?;
    arena.copy_from_host(stream, regions.beta, &fixture.beta)?;
    arena.copy_from_host(stream, regions.norm_weight, &fixture.norm_weight)?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)?;
    arena.copy_from_host(stream, regions.state, &fixture.state)
}

fn reset_state(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.state, &fixture.state)?;
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

pub(crate) fn launch(
    op: &Qwen35GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
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
    op: &Qwen35GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch_causal(
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

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
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
        let state_row = if causal || rows > MAX_BATCH {
            STATE_ROWS[0]
        } else {
            STATE_ROWS[token]
        };
        let qkv_base = token * Qwen35_9B::GDN_QKV_ROWS;
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
                *recurrent *= QUERY_SCALE;
            }
            let rms = (recurrent.iter().map(|value| value * value).sum::<f64>() / HEAD_DIM as f64
                + RMS_EPSILON)
                .sqrt();
            let gate_base =
                token * Qwen35_9B::GDN_INPUT_ROWS + Qwen35_9B::GDN_QKV_ROWS + value_head * HEAD_DIM;
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
    report: &mut Qwen35GdnRecurrenceQualification,
) -> Result<(), Qwen35GdnRecurrenceQualificationError> {
    let expected = oracle(rows, causal, fixture);
    for (index, (&actual, &expected)) in observed.state.iter().zip(&expected.state).enumerate() {
        let error = (f64::from(actual) - expected).abs() as f32;
        report.maximum_state_error = report.maximum_state_error.max(error);
        if error > 2.0e-4f32.max(expected.abs() as f32 * 0.002) {
            return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
                "state at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
        let state_row = (index / STATE_PER_ROW) as u32;
        let active = if causal || rows > MAX_BATCH {
            state_row == STATE_ROWS[0]
        } else {
            STATE_ROWS[..rows].contains(&state_row)
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
            return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
                "output at rows={rows}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
        report.output_values += 1;
    }
    if let Some(relative) = observed.output[rows * VALUE_WIDTH..]
        .iter()
        .position(|&bits| bits != BF16_SENTINEL)
    {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} modified inactive output {}",
            rows * VALUE_WIDTH + relative
        )));
    }
    report.inactive_values += (MAX_ROWS - rows) * VALUE_WIDTH;

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen35GdnRecurrenceQualification,
) -> Result<(), Qwen35GdnRecurrenceQualificationError> {
    if let Some(index) = replay
        .state
        .iter()
        .zip(&eager.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} graph state value {index} differs from eager"
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
            "rows={rows} graph output value {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.state.len() + rows * VALUE_WIDTH;

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen35GdnRecurrenceQualification,
) -> Result<(), Qwen35GdnRecurrenceQualificationError> {
    let qkv = arena.copy_to_host(stream, regions.qkv)?;
    let projected = arena.copy_to_host(stream, regions.projected)?;
    let log_decay = arena.copy_to_host(stream, regions.log_decay)?;
    let beta = arena.copy_to_host(stream, regions.beta)?;
    let norm_weight = arena.copy_to_host(stream, regions.norm_weight)?;
    let state_rows = arena.copy_to_host(stream, regions.state_rows)?;
    if qkv != fixture.qkv
        || projected != fixture.projected
        || log_decay != fixture.log_decay
        || beta != fixture.beta
        || norm_weight != fixture.norm_weight
        || state_rows != STATE_ROWS
    {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(
            "read-only input or parameter plane changed".to_string(),
        ));
    }
    report.immutable_input_values = qkv.len()
        + projected.len()
        + log_decay.len()
        + beta.len()
        + norm_weight.len()
        + state_rows.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen35GdnRecurrenceQualificationError> {
    let graphs = EXACT_ROUTES
        .iter()
        .map(|&rows| CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows)))
        .chain(CAUSAL_ROUTES.iter().map(|&rows| {
            CudaGraph::capture(stream, || launch_causal(op, arena, stream, regions, rows))
        }))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen35GdnRecurrenceQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_and_accounting_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 256);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 25_198_624);
        assert_eq!(regions.payload_bytes(), 25_198_880);
        assert_eq!(layout.byte_len(), 25_199_104);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 224);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35GdnRecurrenceQualificationError> {
        let report = qualify_qwen35_gdn_recurrence()?;
        let active_output_rows = EXACT_ROUTES.iter().chain(&CAUSAL_ROUTES).sum::<usize>();
        let active_state_rows = (1..=MAX_BATCH).sum::<usize>() + 3 + CAUSAL_ROUTES.len();
        let inactive_state_rows =
            (0..MAX_BATCH).sum::<usize>() + (3 + CAUSAL_ROUTES.len()) * (MAX_BATCH - 1);
        let inactive_output_rows = EXACT_ROUTES
            .iter()
            .chain(&CAUSAL_ROUTES)
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();
        let routes = EXACT_ROUTES.len() + CAUSAL_ROUTES.len();

        assert_eq!(report.state_values, active_state_rows * STATE_PER_ROW);
        assert_eq!(report.output_values, active_output_rows * VALUE_WIDTH);
        assert_eq!(
            report.graph_replay_values,
            routes * MAX_BATCH * STATE_PER_ROW + active_output_rows * VALUE_WIDTH
        );
        assert_eq!(
            report.inactive_values,
            inactive_state_rows * STATE_PER_ROW + inactive_output_rows * VALUE_WIDTH
        );
        assert_eq!(report.immutable_input_values, 2_629_768);
        assert_eq!(report.arena_bytes, 25_199_104);
        assert_eq!(report.weight_bytes, 256);
        assert_eq!(report.workspace_bytes, 25_198_624);
        assert_eq!(report.padding_bytes, 224);
        assert!(report.maximum_state_error <= 0.002);
        assert!(report.maximum_output_error <= 0.03125);

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen36_exact_routes_match_shared_independent_oracle()
    -> Result<(), Qwen36GdnRecurrenceQualificationError> {
        let report = qualify_qwen36_gdn_recurrence()?;

        assert_eq!(report.state_values, 22_020_096);
        assert_eq!(report.output_values, 1_101_824);
        assert_eq!(report.arena_bytes, 25_199_104);
        assert_eq!(report.weight_bytes, 256);
        assert!(report.maximum_state_error <= 0.002);
        assert!(report.maximum_output_error <= 0.03125);

        Ok(())
    }
}
