//! Numerical and graph qualification for Qwen3.5 GDN preparation.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::target::Qwen35GdnPrepareOp;
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
pub(crate) const CONTROL_ROWS: usize = Qwen35_9B::GDN_CONTROL_ROWS;
pub(crate) const CONTROL_STRIDE: usize = 128;
pub(crate) const PROJECTED_ROWS: usize = Qwen35_9B::GDN_INPUT_ROWS;
pub(crate) const QKV_ROWS: usize = Qwen35_9B::GDN_QKV_ROWS;
pub(crate) const HISTORY_TAPS: usize = Qwen35_9B::LINEAR_CONV_KERNEL_DIM - 1;
pub(crate) const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const CONTROL_PATTERN: [f32; 8] = [0.5, -0.25, 0.125, -0.0625, 0.375, -0.1875, 0.09375, 0.0];
const PROJECTED_PATTERN: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.09375, 0.046875, 0.0,
];
const CONV_PATTERN: [f32; 4] = [0.5, -0.25, 0.125, 0.25];

/// Failure of the exact Qwen3.5 GDN prepare qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35GdnPrepareQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 GDN prepare qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact decode and prompt route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35GdnPrepareQualification {
    /// FP32 decay and beta values compared with the independent formula.
    pub control_values: usize,
    /// BF16 convolution values compared with the independent formula.
    pub convolution_values: usize,
    /// BF16 history words compared exactly after every mapped update.
    pub history_values: usize,
    /// Active outputs and complete history reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside every active batch extent.
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
    /// Largest absolute control error.
    pub maximum_control_error: f32,
    /// Largest absolute convolution error.
    pub maximum_convolution_error: f32,
}

/// Qwen3.6 qualification report for the shared exact-geometry route.
pub type Qwen36GdnPrepareQualification = Qwen35GdnPrepareQualification;

/// Qwen3.6 qualification failure for the shared exact-geometry route.
pub type Qwen36GdnPrepareQualificationError = Qwen35GdnPrepareQualificationError;

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) projected_controls: ArenaRegion<u16>,
    pub(crate) a_log: ArenaRegion<u16>,
    pub(crate) dt_bias: ArenaRegion<u16>,
    pub(crate) projected: ArenaRegion<u16>,
    pub(crate) convolution_weights: ArenaRegion<u16>,
    pub(crate) state_rows: ArenaRegion<u32>,
    pub(crate) history: ArenaRegion<u16>,
    pub(crate) log_decay: ArenaRegion<f32>,
    pub(crate) beta: ArenaRegion<f32>,
    pub(crate) convolved: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.a_log.byte_len() + self.dt_bias.byte_len() + self.convolution_weights.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.projected_controls.byte_len()
            + self.a_log.byte_len()
            + self.dt_bias.byte_len()
            + self.projected.byte_len()
            + self.convolution_weights.byte_len()
            + self.state_rows.byte_len()
            + self.history.byte_len()
            + self.log_decay.byte_len()
            + self.beta.byte_len()
            + self.convolved.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) projected_controls: Vec<u16>,
    pub(crate) a_log: Vec<u16>,
    pub(crate) dt_bias: Vec<u16>,
    pub(crate) projected: Vec<u16>,
    pub(crate) convolution_weights: Vec<u16>,
    pub(crate) history: Vec<u16>,
}

struct Observed {
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    convolved: Vec<u16>,
    history: Vec<u16>,
}

/// Qualifies eager and captured Qwen3.5 control/convolution at every exact route.
pub fn qualify_qwen35_gdn_prepare()
-> Result<Qwen35GdnPrepareQualification, Qwen35GdnPrepareQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = make_fixture();
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let op = Qwen35GdnPrepareOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35GdnPrepareQualification {
        control_values: 0,
        convolution_values: 0,
        history_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_control_error: 0.0,
        maximum_convolution_error: 0.0,
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
            return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
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

/// Qualifies Qwen3.6 with the same binary and independent state-transition oracle.
///
/// Kernel compile-time assertions bind both architecture profiles to the exact
/// 32-control, 8,192-QKV, width-four-history contract.
pub fn qualify_qwen36_gdn_prepare()
-> Result<Qwen36GdnPrepareQualification, Qwen36GdnPrepareQualificationError> {
    qualify_qwen35_gdn_prepare()
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let projected_controls = layout.reserve(MAX_ROWS * CONTROL_STRIDE, ALIGNMENT)?;
    let a_log = layout.reserve(CONTROL_ROWS, ALIGNMENT)?;
    let dt_bias = layout.reserve(CONTROL_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * PROJECTED_ROWS, ALIGNMENT)?;
    let convolution_weights = layout.reserve(QKV_ROWS * (HISTORY_TAPS + 1), ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let history = layout.reserve(MAX_BATCH * QKV_ROWS * HISTORY_TAPS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_ROWS * CONTROL_ROWS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * CONTROL_ROWS, ALIGNMENT)?;
    let convolved = layout.reserve(MAX_ROWS * QKV_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            projected_controls,
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

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 10]> {
    Ok([
        arena.address(regions.projected_controls)?.addr(),
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

pub(crate) fn make_fixture() -> Fixture {
    let mut projected_controls = vec![0u16; MAX_ROWS * CONTROL_STRIDE];
    for token in 0..MAX_ROWS {
        let token_factor = 1.0 + (token & 7) as f32 / 16.0;
        for row in 0..2 * CONTROL_ROWS {
            projected_controls[token * CONTROL_STRIDE + row] =
                f32_to_bf16(CONTROL_PATTERN[(row + 3 * token) & 7] * token_factor);
        }
    }
    let a_log = (0..CONTROL_ROWS)
        .map(|row| f32_to_bf16(-2.0 + (row & 3) as f32 * 0.125))
        .collect();
    let dt_bias = (0..CONTROL_ROWS)
        .map(|row| f32_to_bf16((row as f32 - 15.5) / 128.0))
        .collect();
    let projected = (0..MAX_ROWS * PROJECTED_ROWS)
        .map(|index| {
            let token = index / PROJECTED_ROWS;
            let token_factor = 1.0 + (token & 7) as f32 / 16.0;
            f32_to_bf16(PROJECTED_PATTERN[(3 * index + token) & 7] * token_factor)
        })
        .collect();
    let convolution_weights = (0..QKV_ROWS * (HISTORY_TAPS + 1))
        .map(|index| f32_to_bf16(CONV_PATTERN[index & 3]))
        .collect();
    let history = (0..MAX_BATCH * QKV_ROWS * HISTORY_TAPS)
        .map(|index| f32_to_bf16(PROJECTED_PATTERN[(5 * index + index / HISTORY_TAPS) & 7] * 0.5))
        .collect();

    Fixture {
        projected_controls,
        a_log,
        dt_bias,
        projected,
        convolution_weights,
        history,
    }
}

pub(crate) fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(
        stream,
        regions.projected_controls,
        &fixture.projected_controls,
    )?;
    arena.copy_from_host(stream, regions.a_log, &fixture.a_log)?;
    arena.copy_from_host(stream, regions.dt_bias, &fixture.dt_bias)?;
    arena.copy_from_host(stream, regions.projected, &fixture.projected)?;
    arena.copy_from_host(
        stream,
        regions.convolution_weights,
        &fixture.convolution_weights,
    )?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)?;
    arena.copy_from_host(stream, regions.history, &fixture.history)
}

fn reset_state(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.history, &fixture.history)?;
    arena.fill(stream, regions.log_decay, BYTE_SENTINEL)?;
    arena.fill(stream, regions.beta, BYTE_SENTINEL)?;
    arena.fill(stream, regions.convolved, BYTE_SENTINEL)
}

pub(crate) fn launch(
    op: &Qwen35GdnPrepareOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            rows,
            arena.address(regions.projected_controls)?,
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

fn launch_causal(
    op: &Qwen35GdnPrepareOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch_causal(
            stream,
            rows,
            arena.address(regions.projected_controls)?,
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

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        log_decay: arena.copy_to_host(stream, regions.log_decay)?,
        beta: arena.copy_to_host(stream, regions.beta)?,
        convolved: arena.copy_to_host(stream, regions.convolved)?,
        history: arena.copy_to_host(stream, regions.history)?,
    })
}

fn verify_oracle(
    rows: usize,
    causal: bool,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35GdnPrepareQualification,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    verify_controls(rows, fixture, observed, report)?;
    verify_convolution(rows, causal, fixture, observed, report)?;
    verify_inactive(rows, observed)?;

    report.control_values += rows * 2 * CONTROL_ROWS;
    report.convolution_values += rows * QKV_ROWS;
    report.history_values += observed.history.len();
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_controls(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35GdnPrepareQualification,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    for token in 0..rows {
        for row in 0..2 * CONTROL_ROWS {
            let raw = f64::from(bf16_to_f32(
                fixture.projected_controls[token * CONTROL_STRIDE + row],
            ));
            let expected = if row < CONTROL_ROWS {
                let control = raw + f64::from(bf16_to_f32(fixture.dt_bias[row]));
                -f64::from(bf16_to_f32(fixture.a_log[row])).exp() * (1.0 + control.exp()).ln()
            } else {
                1.0 / (1.0 + (-raw).exp())
            };
            let actual = if row < CONTROL_ROWS {
                observed.log_decay[token * CONTROL_ROWS + row]
            } else {
                observed.beta[token * CONTROL_ROWS + row - CONTROL_ROWS]
            };
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_control_error = report.maximum_control_error.max(error);
            if error > 0.002 {
                return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
                    "control at rows={rows}, token={token}, row={row}: device={actual}, oracle={expected}, error={error}"
                )));
            }
        }
    }

    Ok(())
}

fn verify_convolution(
    rows: usize,
    causal: bool,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35GdnPrepareQualification,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    let mut expected_history = fixture.history.clone();
    for token in 0..rows {
        let state_row = if causal || rows > MAX_BATCH {
            STATE_ROWS[0]
        } else {
            STATE_ROWS[token]
        };
        for channel in 0..QKV_ROWS {
            let history_base = (state_row as usize * QKV_ROWS + channel) * HISTORY_TAPS;
            let current = fixture.projected[token * PROJECTED_ROWS + channel];
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
            let actual = bf16_to_f32(observed.convolved[token * QKV_ROWS + channel]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_convolution_error = report.maximum_convolution_error.max(error);
            if error > 0.002 {
                return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
                    "convolution at rows={rows}, token={token}, channel={channel}: device={actual}, oracle={expected}, error={error}"
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
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
            "history at rows={rows}, index={index}: device={:#06x}, oracle={:#06x}",
            observed.history[index], expected_history[index]
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Observed,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    for (name, values) in [("log_decay", &observed.log_decay), ("beta", &observed.beta)] {
        let begin = rows * CONTROL_ROWS;
        if let Some(relative) = values[begin..]
            .iter()
            .position(|value| value.to_bits() != 0xa5a5_a5a5)
        {
            return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
                "rows={rows} modified inactive {name} value {}",
                begin + relative
            )));
        }
    }
    let begin = rows * QKV_ROWS;
    if let Some(relative) = observed.convolved[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
            "rows={rows} modified inactive convolution value {}",
            begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (2 * CONTROL_ROWS + QKV_ROWS)
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen35GdnPrepareQualification,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    let eager_controls = eager
        .log_decay
        .iter()
        .map(|value| value.to_bits())
        .chain(eager.beta.iter().map(|value| value.to_bits()));
    let replay_controls = replay
        .log_decay
        .iter()
        .map(|value| value.to_bits())
        .chain(replay.beta.iter().map(|value| value.to_bits()));
    if replay_controls.ne(eager_controls)
        || replay.convolved != eager.convolved
        || replay.history != eager.history
    {
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
            "rows={rows} graph replay differs from eager execution"
        )));
    }
    verify_inactive(rows, replay)?;
    report.graph_replay_values += rows * (2 * CONTROL_ROWS + QKV_ROWS) + replay.history.len();
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen35GdnPrepareQualification,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
    let controls = arena.copy_to_host(stream, regions.projected_controls)?;
    let a_log = arena.copy_to_host(stream, regions.a_log)?;
    let dt_bias = arena.copy_to_host(stream, regions.dt_bias)?;
    let projected = arena.copy_to_host(stream, regions.projected)?;
    let convolution_weights = arena.copy_to_host(stream, regions.convolution_weights)?;
    let state_rows = arena.copy_to_host(stream, regions.state_rows)?;
    if controls != fixture.projected_controls
        || a_log != fixture.a_log
        || dt_bias != fixture.dt_bias
        || projected != fixture.projected
        || convolution_weights != fixture.convolution_weights
        || state_rows != STATE_ROWS
    {
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(
            "read-only input or parameter plane changed".to_string(),
        ));
    }
    report.immutable_input_values = controls.len()
        + a_log.len()
        + dt_bias.len()
        + projected.len()
        + convolution_weights.len()
        + state_rows.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35GdnPrepareOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen35GdnPrepareQualificationError> {
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
        return Err(Qwen35GdnPrepareQualificationError::Mismatch(format!(
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

        assert_eq!(regions.weight_bytes(), 65_664);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 5_701_664);
        assert_eq!(regions.payload_bytes(), 5_767_328);
        assert_eq!(layout.byte_len(), 5_767_936);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35GdnPrepareQualificationError> {
        let report = qualify_qwen35_gdn_prepare()?;
        let active_rows = EXACT_ROUTES.iter().chain(&CAUSAL_ROUTES).sum::<usize>();
        let inactive_rows = EXACT_ROUTES
            .iter()
            .chain(&CAUSAL_ROUTES)
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();
        let complete_history = MAX_BATCH * QKV_ROWS * HISTORY_TAPS;
        let routes = EXACT_ROUTES.len() + CAUSAL_ROUTES.len();

        assert_eq!(report.control_values, active_rows * 2 * CONTROL_ROWS);
        assert_eq!(report.convolution_values, active_rows * QKV_ROWS);
        assert_eq!(report.history_values, routes * complete_history);
        assert_eq!(
            report.graph_replay_values,
            active_rows * (2 * CONTROL_ROWS + QKV_ROWS) + routes * complete_history
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive_rows * (2 * CONTROL_ROWS + QKV_ROWS)
        );
        assert_eq!(report.immutable_input_values, 1_622_088);
        assert_eq!(report.arena_bytes, 5_767_936);
        assert_eq!(report.weight_bytes, 65_664);
        assert_eq!(report.workspace_bytes, 5_701_664);
        assert_eq!(report.padding_bytes, 608);
        assert!(report.maximum_control_error <= 0.002);
        assert!(report.maximum_convolution_error <= 0.002);

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen36_exact_routes_match_shared_independent_oracle()
    -> Result<(), Qwen36GdnPrepareQualificationError> {
        let report = qualify_qwen36_gdn_prepare()?;

        assert_eq!(report.control_values, 17_216);
        assert_eq!(report.convolution_values, 2_203_648);
        assert_eq!(report.arena_bytes, 5_767_936);
        assert_eq!(report.weight_bytes, 65_664);
        assert!(report.maximum_control_error <= 0.002);
        assert!(report.maximum_convolution_error <= 0.002);

        Ok(())
    }
}
