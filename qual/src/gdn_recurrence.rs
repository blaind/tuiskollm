//! Numerical and graph qualification for exact GDN recurrence routes.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::GdnRecurrenceOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
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

struct Fixture {
    qkv: Vec<u16>,
    projected: Vec<u16>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    norm_weight: Vec<u16>,
    state: Vec<f32>,
}

struct Observed {
    state: Vec<f32>,
    output: Vec<u16>,
}

struct Oracle {
    state: Vec<f64>,
    output: Vec<f64>,
}

/// Qualifies eager and captured recurrence routes at exact `B=1..=8`.
pub fn qualify_gdn_recurrence()
-> Result<GdnRecurrenceQualification, GdnRecurrenceQualificationError> {
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
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = GdnRecurrenceQualification {
        state_values: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_state_error: 0.0,
        maximum_output_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(batch, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_BATCH * VALUE_HEADS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_BATCH * VALUE_HEADS, ALIGNMENT)?;
    let norm_weight = layout.reserve(HEAD_DIM, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let state = layout.reserve(MAX_BATCH * STATE_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * VALUE_WIDTH, ALIGNMENT)?;

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
    let mut qkv = vec![0; MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS];
    let mut projected = vec![0; MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS];
    for token in 0..MAX_BATCH {
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
    let log_decay = (0..MAX_BATCH * VALUE_HEADS)
        .map(|index| -0.125 - (index & 7) as f32 * 0.03125)
        .collect();
    let beta = (0..MAX_BATCH * VALUE_HEADS)
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
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: all regions cover their maximum extents and every mapped row is
    // below the eight-row state capacity.
    unsafe {
        op.launch(
            stream,
            batch,
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
        state: arena.copy_to_host(stream, regions.state)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn oracle(batch: usize, fixture: &Fixture) -> Oracle {
    let mut state = fixture
        .state
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; batch * VALUE_WIDTH];
    for (token, &state_row) in STATE_ROWS[..batch].iter().enumerate() {
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
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnRecurrenceQualification,
) -> Result<(), GdnRecurrenceQualificationError> {
    let expected = oracle(batch, fixture);
    let active_rows = &STATE_ROWS[..batch];
    for (index, (&actual, &expected)) in observed.state.iter().zip(&expected.state).enumerate() {
        let error = (f64::from(actual) - expected).abs() as f32;
        report.maximum_state_error = report.maximum_state_error.max(error);
        if error > 2.0e-4f32.max(expected.abs() as f32 * 0.002) {
            return Err(GdnRecurrenceQualificationError::Mismatch(format!(
                "state at B={batch}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
        if active_rows.contains(&((index / STATE_PER_ROW) as u32)) {
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
                "output at B={batch}, index={index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
        report.output_values += 1;
    }
    if let Some(relative) = observed.output[batch * VALUE_WIDTH..]
        .iter()
        .position(|&bits| bits != BF16_SENTINEL)
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "B={batch} modified inactive output {}",
            batch * VALUE_WIDTH + relative
        )));
    }
    report.inactive_values += (MAX_BATCH - batch) * VALUE_WIDTH;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut GdnRecurrenceQualification,
) -> Result<(), GdnRecurrenceQualificationError> {
    if let Some(index) = replay
        .state
        .iter()
        .zip(&eager.state)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "B={batch} graph state value {index} differs from eager"
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnRecurrenceQualificationError::Mismatch(format!(
            "B={batch} graph output value {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.state.len() + batch * VALUE_WIDTH;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GdnRecurrenceQualificationError, MAX_BATCH, STATE_PER_ROW, VALUE_WIDTH,
        qualify_gdn_recurrence,
    };

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), GdnRecurrenceQualificationError> {
        let report = qualify_gdn_recurrence()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();

        assert_eq!(report.state_values, active_rows * STATE_PER_ROW);
        assert_eq!(report.output_values, active_rows * VALUE_WIDTH);
        assert_eq!(
            report.graph_replay_values,
            MAX_BATCH * MAX_BATCH * STATE_PER_ROW + active_rows * VALUE_WIDTH
        );
        assert_eq!(
            report.inactive_values,
            (0..MAX_BATCH).sum::<usize>() * (STATE_PER_ROW + VALUE_WIDTH)
        );
        assert!(report.maximum_state_error <= 0.002);
        assert!(report.maximum_output_error <= 0.03125);

        Ok(())
    }
}
