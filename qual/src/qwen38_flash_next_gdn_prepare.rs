//! Numerical and graph qualification for Qwen3.8-Flash-Next GDN control and convolution.
//!
//! The independent FP64 oracle covers the target-specific 2,560-wide controls
//! and the exact Qwen3.8-27B convolution/history routes reused by this target.

use crate::device_benchmark::{preflight, require_current_process_exclusive};
use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::Qwen38FlashNextGdnPrepareOp;
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const CAUSAL_ROUTES: [usize; 4] = [1, 2, 3, 4];
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

/// Failure of the exact Qwen3.8-Flash-Next GDN prepare qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextGdnPrepareQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next GDN prepare qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact prepare route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextGdnPrepareQualification {
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
    /// Caller-owned input and weight values verified unchanged.
    pub immutable_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact payload bytes, excluding alignment padding.
    pub payload_bytes: usize,
    /// Alignment padding bytes in the qualification arena.
    pub padding_bytes: usize,
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

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.control_weights.byte_len()
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
    input: Vec<u16>,
    control_weights: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    projected: Vec<u16>,
    convolution_weights: Vec<u16>,
    state_rows: Vec<u32>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    convolved: Vec<u16>,
    history: Vec<u16>,
}

/// Qualifies eager and captured control/convolution decode and prefill routes.
pub fn qualify_qwen38_flash_next_gdn_prepare()
-> Result<Qwen38FlashNextGdnPrepareQualification, Qwen38FlashNextGdnPrepareQualificationError> {
    preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
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
    let op = Qwen38FlashNextGdnPrepareOp::new(&context)?;
    require_current_process_exclusive()?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen38FlashNextGdnPrepareQualification {
        control_values: 0,
        convolution_values: 0,
        history_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        payload_bytes: regions.payload_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_control_error: 0.0,
        maximum_convolution_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset_state(&arena, &stream, regions, &fixture)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(rows, &fixture, &eager, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &replay, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                format!("device addresses changed while qualifying row count {rows}"),
            ));
        }
    }

    for tokens in CAUSAL_ROUTES {
        reset_state(&arena, &stream, regions, &fixture)?;
        launch_causal(&op, &arena, &stream, regions, tokens)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_causal_oracle(tokens, &fixture, &eager, &mut report)?;
        if tokens == 1 {
            reset_state(&arena, &stream, regions, &fixture)?;
            launch(&op, &arena, &stream, regions, 1)?;
            let decode = observe(&arena, &stream, regions)?;
            verify_causal_k1_decode_agreement(&eager, &decode)?;
        }

        reset_state(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch_causal(&op, &arena, &stream, regions, tokens)
        })?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &replay, &mut report)?;

        reset_state(&arena, &stream, regions, &fixture)?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let second_replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &fixture, &eager, &second_replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
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
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let controls = Qwen38FlashNext::GDN_CONTROL_ROWS;
    let qkv = Qwen38FlashNext::GDN_QKV_ROWS;
    for (name, actual, expected) in [
        (
            "log_decay",
            &causal.log_decay[..controls],
            &decode.log_decay[..controls],
        ),
        ("beta", &causal.beta[..controls], &decode.beta[..controls]),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                format!("causal K=1 {name} differs bit-exactly from decode B=1 at {index}"),
            ));
        }
    }
    for (name, actual, expected) in [
        (
            "convolution",
            &causal.convolved[..qkv],
            &decode.convolved[..qkv],
        ),
        (
            "history",
            causal.history.as_slice(),
            decode.history.as_slice(),
        ),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                format!("causal K=1 {name} differs bit-exactly from decode B=1 at {index}"),
            ));
        }
    }
    Ok(())
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * Qwen38FlashNext::HIDDEN, ALIGNMENT)?;
    let control_weights = layout.reserve(
        2 * Qwen38FlashNext::GDN_CONTROL_ROWS * Qwen38FlashNext::HIDDEN,
        ALIGNMENT,
    )?;
    let a_log = layout.reserve(Qwen38FlashNext::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let dt_bias = layout.reserve(Qwen38FlashNext::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_INPUT_ROWS, ALIGNMENT)?;
    let convolution_weights = layout.reserve(
        Qwen38FlashNext::GDN_QKV_ROWS * Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM,
        ALIGNMENT,
    )?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let history = layout.reserve(
        MAX_BATCH * Qwen38FlashNext::GDN_QKV_ROWS * HISTORY_TAPS,
        ALIGNMENT,
    )?;
    let log_decay = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let convolved = layout.reserve(MAX_ROWS * Qwen38FlashNext::GDN_QKV_ROWS, ALIGNMENT)?;

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
    let input = (0..MAX_ROWS * Qwen38FlashNext::HIDDEN)
        .map(|index| {
            let token = index / Qwen38FlashNext::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[(index + token) & 7] * (1.0 + (token & 7) as f32 / 8.0))
        })
        .collect();
    let control_weights = (0..2 * Qwen38FlashNext::GDN_CONTROL_ROWS * Qwen38FlashNext::HIDDEN)
        .map(|index| {
            let row = index / Qwen38FlashNext::HIDDEN;
            let column = index - row * Qwen38FlashNext::HIDDEN;
            f32_to_bf16(WEIGHT_PATTERN[(column + 3 * row) & 7])
        })
        .collect();
    let a_log = (0..Qwen38FlashNext::GDN_CONTROL_ROWS)
        .map(|row| f32_to_bf16(-2.0 + (row & 3) as f32 * 0.125))
        .collect();
    let dt_bias = (0..Qwen38FlashNext::GDN_CONTROL_ROWS)
        .map(|row| f32_to_bf16((row as f32 - 23.5) / 256.0))
        .collect();
    let projected = (0..MAX_ROWS * Qwen38FlashNext::GDN_INPUT_ROWS)
        .map(|index| {
            let token = index / Qwen38FlashNext::GDN_INPUT_ROWS;
            f32_to_bf16(INPUT_PATTERN[(3 * index + token) & 7] * (1.0 + (token & 15) as f32 / 16.0))
        })
        .collect();
    let convolution_weights = (0..Qwen38FlashNext::GDN_QKV_ROWS * 4)
        .map(|index| f32_to_bf16(CONV_PATTERN[index & 3]))
        .collect();
    let history = (0..MAX_BATCH * Qwen38FlashNext::GDN_QKV_ROWS * HISTORY_TAPS)
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
    op: &Qwen38FlashNextGdnPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: all regions are aligned, non-overlapping, context-local, and
    // cover the maximum exact batch. Every mapped state row is below eight.
    unsafe {
        op.launch(
            stream,
            rows,
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

fn launch_causal(
    op: &Qwen38FlashNextGdnPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: every exact causal row shares STATE_ROWS[0]; all other planes
    // cover K=4 and the mapped history row is below eight.
    unsafe {
        op.launch_causal(
            stream,
            tokens,
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
        input: arena.copy_to_host(stream, regions.input)?,
        control_weights: arena.copy_to_host(stream, regions.control_weights)?,
        a_log: arena.copy_to_host(stream, regions.a_log)?,
        dt_bias: arena.copy_to_host(stream, regions.dt_bias)?,
        projected: arena.copy_to_host(stream, regions.projected)?,
        convolution_weights: arena.copy_to_host(stream, regions.convolution_weights)?,
        state_rows: arena.copy_to_host(stream, regions.state_rows)?,
        log_decay: arena.copy_to_host(stream, regions.log_decay)?,
        beta: arena.copy_to_host(stream, regions.beta)?,
        convolved: arena.copy_to_host(stream, regions.convolved)?,
        history: arena.copy_to_host(stream, regions.history)?,
    })
}

fn verify_oracle(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    verify_immutable(rows, fixture, observed)?;
    verify_controls(rows, fixture, observed, report)?;
    verify_convolution(rows, fixture, observed, report)?;
    verify_inactive(rows, observed)?;

    report.control_values += rows * 2 * Qwen38FlashNext::GDN_CONTROL_ROWS;
    report.convolution_values += rows * Qwen38FlashNext::GDN_QKV_ROWS;
    report.history_values += observed.history.len();
    report.inactive_values += inactive_values(rows);
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_causal_oracle(
    tokens: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    verify_immutable(tokens, fixture, observed)?;
    verify_controls(tokens, fixture, observed, report)?;
    verify_causal_convolution(tokens, fixture, observed, report)?;
    verify_inactive(tokens, observed)?;

    report.control_values += tokens * 2 * Qwen38FlashNext::GDN_CONTROL_ROWS;
    report.convolution_values += tokens * Qwen38FlashNext::GDN_QKV_ROWS;
    report.history_values += observed.history.len();
    report.inactive_values += inactive_values(tokens);
    report.immutable_values += immutable_values(fixture);
    Ok(())
}

fn verify_immutable(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    for (name, actual, expected) in [
        ("input", observed.input.as_slice(), fixture.input.as_slice()),
        (
            "control_weights",
            observed.control_weights.as_slice(),
            fixture.control_weights.as_slice(),
        ),
        ("a_log", observed.a_log.as_slice(), fixture.a_log.as_slice()),
        (
            "dt_bias",
            observed.dt_bias.as_slice(),
            fixture.dt_bias.as_slice(),
        ),
        (
            "projected",
            observed.projected.as_slice(),
            fixture.projected.as_slice(),
        ),
        (
            "convolution_weights",
            observed.convolution_weights.as_slice(),
            fixture.convolution_weights.as_slice(),
        ),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                format!("rows={rows} modified immutable {name} value {index}"),
            ));
        }
    }
    if observed.state_rows != STATE_ROWS {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!("rows={rows} modified immutable state-row mapping"),
        ));
    }

    Ok(())
}

fn immutable_values(fixture: &Fixture) -> usize {
    fixture.input.len()
        + fixture.control_weights.len()
        + fixture.a_log.len()
        + fixture.dt_bias.len()
        + fixture.projected.len()
        + fixture.convolution_weights.len()
        + STATE_ROWS.len()
}

fn verify_controls(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let hidden = Qwen38FlashNext::HIDDEN;
    let controls = Qwen38FlashNext::GDN_CONTROL_ROWS;
    for token in 0..rows {
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
                return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                    format!(
                        "control at rows={rows}, token={token}, row={row}: device={actual}, oracle={expected}, error={error}"
                    ),
                ));
            }
        }
    }

    Ok(())
}

/// History row each decode token advances; every prefill row shares the first.
fn mapped_state_row(rows: usize, token: usize) -> usize {
    let state_row = if rows <= MAX_BATCH {
        STATE_ROWS[token]
    } else {
        STATE_ROWS[0]
    };

    state_row as usize
}

fn verify_convolution(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let qkv = Qwen38FlashNext::GDN_QKV_ROWS;
    let input_rows = Qwen38FlashNext::GDN_INPUT_ROWS;
    let mut expected_history = fixture.history.clone();

    for token in 0..rows {
        let state_row = mapped_state_row(rows, token);
        for channel in 0..qkv {
            let history_base = (state_row * qkv + channel) * HISTORY_TAPS;
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
                return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                    format!(
                        "convolution at rows={rows}, token={token}, channel={channel}: device={actual}, oracle={expected}, error={error}"
                    ),
                ));
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
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!(
                "history at rows={rows}, index={index}: device={:#06x}, oracle={:#06x}",
                observed.history[index], expected_history[index]
            ),
        ));
    }

    Ok(())
}

fn verify_causal_convolution(
    tokens: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let qkv = Qwen38FlashNext::GDN_QKV_ROWS;
    let input_rows = Qwen38FlashNext::GDN_INPUT_ROWS;
    let state_row = STATE_ROWS[0] as usize;
    let mut expected_history = fixture.history.clone();

    for token in 0..tokens {
        for channel in 0..qkv {
            let history_base = (state_row * qkv + channel) * HISTORY_TAPS;
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
                return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                    format!(
                        "causal convolution at K={tokens}, token={token}, channel={channel}: device={actual}, oracle={expected}, error={error}"
                    ),
                ));
            }
            expected_history[history_base..history_base + HISTORY_TAPS]
                .copy_from_slice(&[values[1], values[2], current]);
        }
    }

    if let Some(index) = observed
        .history
        .iter()
        .zip(&expected_history)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!(
                "causal history at K={tokens}, index={index}: device={:#06x}, oracle={:#06x}",
                observed.history[index], expected_history[index]
            ),
        ));
    }
    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Observed,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let controls = Qwen38FlashNext::GDN_CONTROL_ROWS;
    let qkv = Qwen38FlashNext::GDN_QKV_ROWS;
    for (name, values) in [("log_decay", &observed.log_decay), ("beta", &observed.beta)] {
        let begin = rows * controls;
        if let Some(relative) = values[begin..]
            .iter()
            .position(|value| value.to_bits() != 0xa5a5_a5a5)
        {
            return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
                format!(
                    "rows={rows} modified inactive {name} value {}",
                    begin + relative
                ),
            ));
        }
    }
    let begin = rows * qkv;
    if let Some(relative) = observed.convolved[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!(
                "rows={rows} modified inactive convolution value {}",
                begin + relative
            ),
        ));
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (2 * Qwen38FlashNext::GDN_CONTROL_ROWS + Qwen38FlashNext::GDN_QKV_ROWS)
}

fn verify_replay(
    rows: usize,
    fixture: &Fixture,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen38FlashNextGdnPrepareQualification,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    verify_immutable(rows, fixture, replay)?;
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
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!("rows={rows} graph control value {index} differs from eager"),
        ));
    }
    if let Some(index) = replay
        .convolved
        .iter()
        .zip(&eager.convolved)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!("rows={rows} graph convolution value {index} differs from eager"),
        ));
    }
    if let Some(index) = replay
        .history
        .iter()
        .zip(&eager.history)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!("rows={rows} graph history value {index} differs from eager"),
        ));
    }
    verify_inactive(rows, replay)?;

    report.graph_replay_values += rows
        * (2 * Qwen38FlashNext::GDN_CONTROL_ROWS + Qwen38FlashNext::GDN_QKV_ROWS)
        + replay.history.len();
    report.inactive_values += inactive_values(rows);
    report.immutable_values += immutable_values(fixture);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen38FlashNextGdnPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
    let mut graphs = Vec::with_capacity(EXACT_ROUTES.len() + CAUSAL_ROUTES.len());
    for rows in EXACT_ROUTES {
        reset_state(arena, stream, regions, fixture)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch(op, arena, stream, regions, rows)
        })?);
    }
    for tokens in CAUSAL_ROUTES {
        reset_state(arena, stream, regions, fixture)?;
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
        return Err(Qwen38FlashNextGdnPrepareQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CAUSAL_ROUTES, EXACT_ROUTES, HISTORY_TAPS, MAX_BATCH, MAX_ROWS, Qwen38FlashNext,
        Qwen38FlashNextGdnPrepareQualificationError, fixture, immutable_values, layout,
        qualify_qwen38_flash_next_gdn_prepare,
    };
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen38FlashNextGdnPrepareQualificationError> {
        let report = qualify_qwen38_flash_next_gdn_prepare()?;
        let active_rows = EXACT_ROUTES.iter().chain(&CAUSAL_ROUTES).sum::<usize>();
        let route_count = EXACT_ROUTES.len() + CAUSAL_ROUTES.len();
        let complete_history = MAX_BATCH * Qwen38FlashNext::GDN_QKV_ROWS * HISTORY_TAPS;
        let active_outputs =
            active_rows * (2 * Qwen38FlashNext::GDN_CONTROL_ROWS + Qwen38FlashNext::GDN_QKV_ROWS);
        let inactive_outputs = EXACT_ROUTES
            .iter()
            .chain(&CAUSAL_ROUTES)
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>()
            * (2 * Qwen38FlashNext::GDN_CONTROL_ROWS + Qwen38FlashNext::GDN_QKV_ROWS);

        assert_eq!(
            report.control_values,
            active_rows * 2 * Qwen38FlashNext::GDN_CONTROL_ROWS
        );
        assert_eq!(
            report.convolution_values,
            active_rows * Qwen38FlashNext::GDN_QKV_ROWS
        );
        assert_eq!(report.history_values, route_count * complete_history);
        assert_eq!(
            report.graph_replay_values,
            2 * (active_outputs + route_count * complete_history)
        );
        assert_eq!(report.inactive_values, 3 * inactive_outputs);
        assert_eq!(
            report.immutable_values,
            3 * route_count * immutable_values(&fixture())
        );
        assert_eq!(
            report.arena_bytes,
            report.payload_bytes + report.padding_bytes
        );
        assert!(report.maximum_control_error <= 0.002);
        assert!(report.maximum_convolution_error <= 0.002);

        Ok(())
    }

    #[test]
    fn route_inventory_and_arena_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(CAUSAL_ROUTES, [1, 2, 3, 4]);
        let (layout, regions) = layout().unwrap();

        assert_eq!(layout.byte_len(), 61_227_776);
        assert_eq!(regions.payload_bytes(), 61_227_232);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 544);
    }

    #[test]
    fn qwen38_flash_next_control_width_is_the_only_divergence() {
        assert_eq!(Qwen38FlashNext::HIDDEN, 2_560);
        assert_ne!(Qwen38FlashNext::HIDDEN, Qwen38_27B::HIDDEN);
        assert_eq!(Qwen38FlashNext::GDN_QKV_ROWS, Qwen38_27B::GDN_QKV_ROWS);
        assert_eq!(Qwen38FlashNext::GDN_INPUT_ROWS, Qwen38_27B::GDN_INPUT_ROWS);
        assert_eq!(
            Qwen38FlashNext::GDN_CONTROL_ROWS,
            Qwen38_27B::GDN_CONTROL_ROWS
        );
        assert_eq!(
            Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM,
            Qwen38_27B::LINEAR_CONV_KERNEL_DIM
        );
    }
}
