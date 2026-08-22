//! Numerical and graph qualification for GDN control and causal convolution.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::GdnPrepareOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const HISTORY_TAPS: usize = 3;
const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const INPUT_PATTERN: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.09375, 0.046875, 0.0,
];
const WEIGHT_PATTERN: [f32; 8] = [
    0.00390625,
    -0.001953125,
    0.0009765625,
    0.001953125,
    -0.00390625,
    0.00048828125,
    0.0014648438,
    -0.0009765625,
];
const CONV_PATTERN: [f32; 4] = [0.5, -0.25, 0.125, 0.25];

/// Failure of the exact GDN prepare qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum GdnPrepareQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("GDN prepare qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact GDN prepare route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnPrepareQualification {
    /// FP32 decay and beta control values compared with the FP64 formula.
    pub control_values: usize,
    /// BF16 convolution values compared with the FP64 formula.
    pub convolution_values: usize,
    /// BF16 history words compared bit-exactly after each mapped update.
    pub history_values: usize,
    /// Active outputs and complete history reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel output values verified beyond every active batch.
    pub inactive_values: usize,
    /// Largest absolute control error.
    pub maximum_control_error: f32,
    /// Largest absolute convolution error.
    pub maximum_convolution_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    control_weights: ArenaRegion<u16>,
    a_log: ArenaRegion<u16>,
    dt_bias: ArenaRegion<u16>,
    projected: ArenaRegion<u16>,
    convolution_weights: ArenaRegion<u16>,
    state_rows: ArenaRegion<u32>,
    history: ArenaRegion<u16>,
    log_decay: ArenaRegion<f32>,
    beta: ArenaRegion<f32>,
    convolved: ArenaRegion<u16>,
}

struct Fixture {
    input: Vec<u16>,
    control_weights: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    projected: Vec<u16>,
    convolution_weights: Vec<u16>,
    history: Vec<u16>,
}

struct Observed {
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    convolved: Vec<u16>,
    history: Vec<u16>,
}

/// Qualifies eager and captured control/convolution routes at exact `B=1..=8`.
pub fn qualify_gdn_prepare() -> Result<GdnPrepareQualification, GdnPrepareQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = GdnPrepareOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = GdnPrepareQualification {
        control_values: 0,
        convolution_values: 0,
        history_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_control_error: 0.0,
        maximum_convolution_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_state(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(batch, &fixture, &eager, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(GdnPrepareQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * Qwen38_27B::HIDDEN, ALIGNMENT)?;
    let control_weights = layout.reserve(
        2 * Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::HIDDEN,
        ALIGNMENT,
    )?;
    let a_log = layout.reserve(Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let dt_bias = layout.reserve(Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let convolution_weights = layout.reserve(
        Qwen38_27B::GDN_QKV_ROWS * Qwen38_27B::LINEAR_CONV_KERNEL_DIM,
        ALIGNMENT,
    )?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let history = layout.reserve(
        MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS,
        ALIGNMENT,
    )?;
    let log_decay = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let convolved = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            control_weights,
            a_log,
            dt_bias,
            projected,
            convolution_weights,
            state_rows,
            history,
            log_decay,
            beta,
            convolved,
        },
    ))
}

fn fixture() -> Fixture {
    let input = (0..MAX_BATCH * Qwen38_27B::HIDDEN)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[(index + token) & 7] * (1.0 + token as f32 / 8.0))
        })
        .collect();
    let control_weights = (0..2 * Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| {
            let row = index / Qwen38_27B::HIDDEN;
            let column = index - row * Qwen38_27B::HIDDEN;
            f32_to_bf16(WEIGHT_PATTERN[(column + 3 * row) & 7])
        })
        .collect();
    let a_log = (0..Qwen38_27B::GDN_CONTROL_ROWS)
        .map(|row| f32_to_bf16(-2.0 + (row & 3) as f32 * 0.125))
        .collect();
    let dt_bias = (0..Qwen38_27B::GDN_CONTROL_ROWS)
        .map(|row| f32_to_bf16((row as f32 - 23.5) / 256.0))
        .collect();
    let projected = (0..MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::GDN_INPUT_ROWS;
            f32_to_bf16(INPUT_PATTERN[(3 * index + token) & 7] * (1.0 + token as f32 / 16.0))
        })
        .collect();
    let convolution_weights = (0..Qwen38_27B::GDN_QKV_ROWS * 4)
        .map(|index| f32_to_bf16(CONV_PATTERN[index & 3]))
        .collect();
    let history = (0..MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS)
        .map(|index| f32_to_bf16(INPUT_PATTERN[(5 * index + index / HISTORY_TAPS) & 7] * 0.5))
        .collect();

    Fixture {
        input,
        control_weights,
        a_log,
        dt_bias,
        projected,
        convolution_weights,
        history,
    }
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input)?;
    arena.copy_from_host(stream, regions.control_weights, &fixture.control_weights)?;
    arena.copy_from_host(stream, regions.a_log, &fixture.a_log)?;
    arena.copy_from_host(stream, regions.dt_bias, &fixture.dt_bias)?;
    arena.copy_from_host(stream, regions.projected, &fixture.projected)?;
    arena.copy_from_host(
        stream,
        regions.convolution_weights,
        &fixture.convolution_weights,
    )?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)
}

fn reset_state(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.history, &fixture.history)?;
    arena.fill(stream, regions.log_decay, BYTE_SENTINEL)?;
    arena.fill(stream, regions.beta, BYTE_SENTINEL)?;
    arena.fill(stream, regions.convolved, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 11]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.control_weights)?.addr(),
        arena.address(regions.a_log)?.addr(),
        arena.address(regions.dt_bias)?.addr(),
        arena.address(regions.projected)?.addr(),
        arena.address(regions.convolution_weights)?.addr(),
        arena.address(regions.state_rows)?.addr(),
        arena.address(regions.history)?.addr(),
        arena.address(regions.log_decay)?.addr(),
        arena.address(regions.beta)?.addr(),
        arena.address(regions.convolved)?.addr(),
    ])
}

fn launch(
    op: &GdnPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: all regions are aligned, non-overlapping, context-local, and
    // cover the maximum exact batch. Every mapped state row is below eight.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.control_weights)?,
            arena.address(regions.a_log)?,
            arena.address(regions.dt_bias)?,
            arena.address(regions.projected)?,
            arena.address(regions.convolution_weights)?,
            arena.address(regions.state_rows)?,
            arena.address(regions.history)?,
            arena.address(regions.log_decay)?,
            arena.address(regions.beta)?,
            arena.address(regions.convolved)?,
        )
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        log_decay: arena.copy_to_host(stream, regions.log_decay)?,
        beta: arena.copy_to_host(stream, regions.beta)?,
        convolved: arena.copy_to_host(stream, regions.convolved)?,
        history: arena.copy_to_host(stream, regions.history)?,
    })
}

fn verify_oracle(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnPrepareQualification,
) -> Result<(), GdnPrepareQualificationError> {
    verify_controls(batch, fixture, observed, report)?;
    verify_convolution(batch, fixture, observed, report)?;
    verify_inactive(batch, observed)?;

    report.control_values += batch * 2 * Qwen38_27B::GDN_CONTROL_ROWS;
    report.convolution_values += batch * Qwen38_27B::GDN_QKV_ROWS;
    report.history_values += observed.history.len();
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_controls(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnPrepareQualification,
) -> Result<(), GdnPrepareQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let controls = Qwen38_27B::GDN_CONTROL_ROWS;
    for token in 0..batch {
        for row in 0..2 * controls {
            let sum = (0..hidden)
                .map(|column| {
                    f64::from(bf16_to_f32(fixture.input[token * hidden + column]))
                        * f64::from(bf16_to_f32(fixture.control_weights[row * hidden + column]))
                })
                .sum::<f64>();
            let expected = if row < controls {
                let control = sum + f64::from(bf16_to_f32(fixture.dt_bias[row]));
                -f64::from(bf16_to_f32(fixture.a_log[row])).exp() * (1.0 + control.exp()).ln()
            } else {
                1.0 / (1.0 + (-sum).exp())
            };
            let actual = if row < controls {
                observed.log_decay[token * controls + row]
            } else {
                observed.beta[token * controls + row - controls]
            };
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_control_error = report.maximum_control_error.max(error);
            if error > 0.002 {
                return Err(GdnPrepareQualificationError::Mismatch(format!(
                    "control at B={batch}, token={token}, row={row}: device={actual}, oracle={expected}, error={error}"
                )));
            }
        }
    }

    Ok(())
}

fn verify_convolution(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnPrepareQualification,
) -> Result<(), GdnPrepareQualificationError> {
    let qkv = Qwen38_27B::GDN_QKV_ROWS;
    let input_rows = Qwen38_27B::GDN_INPUT_ROWS;
    let mut expected_history = fixture.history.clone();

    for (token, &state_row) in STATE_ROWS[..batch].iter().enumerate() {
        for channel in 0..qkv {
            let history_base = (state_row as usize * qkv + channel) * HISTORY_TAPS;
            let current = fixture.projected[token * input_rows + channel];
            let values = [
                expected_history[history_base],
                expected_history[history_base + 1],
                expected_history[history_base + 2],
                current,
            ];
            let sum = (0..4)
                .map(|tap| {
                    f64::from(bf16_to_f32(fixture.convolution_weights[channel * 4 + tap]))
                        * f64::from(bf16_to_f32(values[tap]))
                })
                .sum::<f64>();
            let expected = sum / (1.0 + (-sum).exp());
            let actual = bf16_to_f32(observed.convolved[token * qkv + channel]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_convolution_error = report.maximum_convolution_error.max(error);
            if error > 0.002 {
                return Err(GdnPrepareQualificationError::Mismatch(format!(
                    "convolution at B={batch}, token={token}, channel={channel}: device={actual}, oracle={expected}, error={error}"
                )));
            }

            expected_history[history_base] = values[1];
            expected_history[history_base + 1] = values[2];
            expected_history[history_base + 2] = current;
        }
    }

    if let Some(index) = observed
        .history
        .iter()
        .zip(&expected_history)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "history at B={batch}, index={index}: device={:#06x}, oracle={:#06x}",
            observed.history[index], expected_history[index]
        )));
    }

    Ok(())
}

fn verify_inactive(batch: usize, observed: &Observed) -> Result<(), GdnPrepareQualificationError> {
    let controls = Qwen38_27B::GDN_CONTROL_ROWS;
    let qkv = Qwen38_27B::GDN_QKV_ROWS;
    for (name, values) in [("log_decay", &observed.log_decay), ("beta", &observed.beta)] {
        let begin = batch * controls;
        if let Some(relative) = values[begin..]
            .iter()
            .position(|value| value.to_bits() != 0xa5a5_a5a5)
        {
            return Err(GdnPrepareQualificationError::Mismatch(format!(
                "B={batch} modified inactive {name} value {}",
                begin + relative
            )));
        }
    }
    let begin = batch * qkv;
    if let Some(relative) = observed.convolved[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "B={batch} modified inactive convolution value {}",
            begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (2 * Qwen38_27B::GDN_CONTROL_ROWS + Qwen38_27B::GDN_QKV_ROWS)
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut GdnPrepareQualification,
) -> Result<(), GdnPrepareQualificationError> {
    let eager_control = eager
        .log_decay
        .iter()
        .map(|value| value.to_bits())
        .chain(eager.beta.iter().map(|value| value.to_bits()));
    let replay_control = replay
        .log_decay
        .iter()
        .map(|value| value.to_bits())
        .chain(replay.beta.iter().map(|value| value.to_bits()));
    if let Some(index) = replay_control
        .zip(eager_control)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "B={batch} graph control value {index} differs from eager"
        )));
    }
    if let Some(index) = replay
        .convolved
        .iter()
        .zip(&eager.convolved)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "B={batch} graph convolution value {index} differs from eager"
        )));
    }
    if let Some(index) = replay
        .history
        .iter()
        .zip(&eager.history)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(GdnPrepareQualificationError::Mismatch(format!(
            "B={batch} graph history value {index} differs from eager"
        )));
    }
    verify_inactive(batch, replay)?;

    report.graph_replay_values += batch
        * (2 * Qwen38_27B::GDN_CONTROL_ROWS + Qwen38_27B::GDN_QKV_ROWS)
        + replay.history.len();
    report.inactive_values += inactive_values(batch);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GdnPrepareQualificationError, HISTORY_TAPS, MAX_BATCH, Qwen38_27B, qualify_gdn_prepare,
    };
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), GdnPrepareQualificationError> {
        let report = qualify_gdn_prepare()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let complete_history = MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS;
        let active_outputs =
            active_rows * (2 * Qwen38_27B::GDN_CONTROL_ROWS + Qwen38_27B::GDN_QKV_ROWS);
        let inactive_outputs = (0..MAX_BATCH).sum::<usize>()
            * (2 * Qwen38_27B::GDN_CONTROL_ROWS + Qwen38_27B::GDN_QKV_ROWS);

        assert_eq!(
            report.control_values,
            active_rows * 2 * Qwen38_27B::GDN_CONTROL_ROWS
        );
        assert_eq!(
            report.convolution_values,
            active_rows * Qwen38_27B::GDN_QKV_ROWS
        );
        assert_eq!(report.history_values, MAX_BATCH * complete_history);
        assert_eq!(
            report.graph_replay_values,
            active_outputs + MAX_BATCH * complete_history
        );
        assert_eq!(report.inactive_values, 2 * inactive_outputs);
        assert!(report.maximum_control_error <= 0.002);
        assert!(report.maximum_convolution_error <= 0.002);

        Ok(())
    }
}
