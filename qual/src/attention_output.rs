//! Numerical, seam, and graph qualification for gated FP8 attention output.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, SCALE_VALUES, WEIGHT_CODES, bf16_to_f32,
    decode_e4m3fn, encode_e4m3fn, f32_to_bf16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::AttentionOutputOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const FP8_MAX: f32 = 448.0;
const GATE_PATTERN: [f32; 8] = [0.0, -1.0, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125];

/// Failure of the exact attention-output qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum AttentionOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("attention-output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, storage ownership, and worst error across exact `B=1..8`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOutputQualification {
    /// Gated FP32 values compared with mathematical sigmoid.
    pub gated_values: usize,
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// BF16 projection values compared with the represented-value oracle.
    pub output_values: usize,
    /// Complete mutable arena seams reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values proved untouched outside each exact batch.
    pub inactive_values: usize,
    /// Read-only fused QKV and projection-weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute gated-FP32 error.
    pub maximum_gated_error: f32,
    /// Largest absolute BF16 projection error.
    pub maximum_projection_error: f32,
    /// Largest relative BF16 projection error, with a nonzero denominator floor.
    pub maximum_projection_relative_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    attention: ArenaRegion<f32>,
    qkv: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_codes.byte_len()
            + self.weight_scales.byte_len()
            + self.output.byte_len()
    }
}

struct Fixture {
    attention: Vec<f32>,
    qkv: Vec<u16>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u16>,
}

struct TokenOracle {
    gated: Vec<f32>,
    codes: Vec<u8>,
    scale: f32,
    correlations: [f64; 4],
}

struct Observed {
    attention: Vec<f32>,
    activation_codes: Vec<u8>,
    activation_scales: Vec<f32>,
    output: Vec<u16>,
}

/// Qualifies eager and captured attention-output routes at exact `B=1..=8`.
pub fn qualify_attention_output()
-> Result<AttentionOutputQualification, AttentionOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(AttentionOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    let oracles = make_oracles(&fixture)?;
    load_immutable(&arena, &stream, regions, &fixture)?;
    let op = AttentionOutputOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = AttentionOutputQualification {
        gated_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_gated_error: 0.0,
        maximum_projection_error: 0.0,
        maximum_projection_relative_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions, &fixture, batch)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(batch, &oracles, &eager, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        reset(&arena, &stream, regions, &fixture, batch)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(AttentionOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let activation_codes =
        layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let weight_codes = layout.reserve(
        Qwen38_27B::HIDDEN * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ALIGNMENT,
    )?;
    let weight_scales = layout.reserve(Qwen38_27B::HIDDEN, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * Qwen38_27B::HIDDEN, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    let attention = (0..MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let token = index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let column = index - token * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let head = column / Qwen38_27B::HEAD_DIM;
            let dimension = column - head * Qwen38_27B::HEAD_DIM;
            let gate = GATE_PATTERN[(dimension + head + token) & 7];
            let magnitude = if gate == 0.0 { 1.0 } else { 0.25 };
            let sign = if (dimension + head * 3 + token) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            sign * magnitude * TOKEN_FACTORS[token]
        })
        .collect::<Vec<_>>();
    let mut qkv = vec![BF16_SENTINEL; MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS];
    for token in 0..MAX_BATCH {
        for head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let gate = token * Qwen38_27B::ATTENTION_QKV_ROWS
                    + head * 2 * Qwen38_27B::HEAD_DIM
                    + Qwen38_27B::HEAD_DIM
                    + dimension;
                qkv[gate] = f32_to_bf16(GATE_PATTERN[(dimension + head + token) & 7]);
            }
        }
    }
    let mut weight_codes = vec![0u8; Qwen38_27B::HIDDEN * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS];
    for row in 0..Qwen38_27B::HIDDEN {
        for column in 0..Qwen38_27B::ATTENTION_OUTPUT_COLUMNS {
            weight_codes[row * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + column] =
                WEIGHT_CODES[(row + column) & 3];
        }
    }
    let weight_scales = (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect();

    Fixture {
        attention,
        qkv,
        weight_codes,
        weight_scales,
    }
}

fn make_oracles(fixture: &Fixture) -> Result<Vec<TokenOracle>, AttentionOutputQualificationError> {
    (0..MAX_BATCH)
        .map(|token| {
            let gated = (0..Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
                .map(|column| {
                    let head = column / Qwen38_27B::HEAD_DIM;
                    let dimension = column - head * Qwen38_27B::HEAD_DIM;
                    let gate_offset = token * Qwen38_27B::ATTENTION_QKV_ROWS
                        + head * 2 * Qwen38_27B::HEAD_DIM
                        + Qwen38_27B::HEAD_DIM
                        + dimension;
                    let gate = bf16_to_f32(fixture.qkv[gate_offset]);
                    fixture.attention[token * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + column]
                        / (1.0 + (-gate).exp())
                })
                .collect::<Vec<_>>();
            let maximum = gated
                .iter()
                .fold(0.0f32, |current, value| current.max(value.abs()));
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / FP8_MAX
            };
            let codes = gated
                .iter()
                .map(|&value| encode_e4m3fn(value / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(AttentionOutputQualificationError::Mismatch)?;
            let mut correlations = [0.0f64; 4];
            for (column, &code) in codes.iter().enumerate() {
                let activation = f64::from(
                    decode_e4m3fn(code).map_err(AttentionOutputQualificationError::Mismatch)?,
                );
                for phase in 0..4 {
                    let weight = decode_e4m3fn(WEIGHT_CODES[(phase + column) & 3])
                        .map_err(AttentionOutputQualificationError::Mismatch)?;
                    correlations[phase] += activation * f64::from(weight);
                }
            }

            Ok(TokenOracle {
                gated,
                codes,
                scale,
                correlations,
            })
        })
        .collect()
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(stream, regions.weight_scales, &fixture.weight_scales)
}

fn reset(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    batch: usize,
) -> GpuResult<()> {
    arena.fill(stream, regions.attention, BYTE_SENTINEL)?;
    arena.copy_prefix_from_host(
        stream,
        regions.attention,
        &fixture.attention[..batch * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS],
    )?;
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 7]> {
    Ok([
        arena.address(regions.attention)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.activation_scales)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &AttentionOutputOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: every region is aligned, disjoint, context-local, and covers B=8.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.attention)?,
            arena.address(regions.qkv)?,
            arena.address(regions.activation_codes)?,
            arena.address(regions.activation_scales)?,
            arena.address(regions.weight_codes)?,
            arena.address(regions.weight_scales)?,
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
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_eager(
    batch: usize,
    oracles: &[TokenOracle],
    observed: &Observed,
    report: &mut AttentionOutputQualification,
) -> Result<(), AttentionOutputQualificationError> {
    let columns = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    for (token, oracle) in oracles[..batch].iter().enumerate() {
        let begin = token * columns;
        for (column, (&actual, &expected)) in observed.attention[begin..begin + columns]
            .iter()
            .zip(&oracle.gated)
            .enumerate()
        {
            let error = (actual - expected).abs();
            report.maximum_gated_error = report.maximum_gated_error.max(error);
            let tolerance = 0.000_05f32.max(expected.abs() * 0.000_25);
            if !actual.is_finite() || error > tolerance {
                return Err(AttentionOutputQualificationError::Mismatch(format!(
                    "gated value at B={batch}, token={token}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        if let Some(column) = observed.activation_codes[begin..begin + columns]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(AttentionOutputQualificationError::Mismatch(format!(
                "activation code at B={batch}, token={token}, column={column} differs"
            )));
        }
        if observed.activation_scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(AttentionOutputQualificationError::Mismatch(format!(
                "activation scale at B={batch}, token={token}: device={}, oracle={} differs",
                observed.activation_scales[token], oracle.scale
            )));
        }
        for row in 0..Qwen38_27B::HIDDEN {
            let expected = oracle.correlations[row & 3]
                * f64::from(oracle.scale)
                * f64::from(SCALE_VALUES[row & 3]);
            let actual = bf16_to_f32(observed.output[token * Qwen38_27B::HIDDEN + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_projection_error = report.maximum_projection_error.max(error);
            let relative_error = error / (expected.abs() as f32).max(1.0e-12);
            report.maximum_projection_relative_error =
                report.maximum_projection_relative_error.max(relative_error);
            let tolerance = 0.125f32.max(expected.abs() as f32 * 0.015);
            if !actual.is_finite() || error > tolerance {
                return Err(AttentionOutputQualificationError::Mismatch(format!(
                    "projection at B={batch}, token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }
    verify_inactive(batch, observed)?;
    report.gated_values += batch * columns;
    report.activation_codes += batch * columns;
    report.activation_scales += batch;
    report.output_values += batch * Qwen38_27B::HIDDEN;
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
) -> Result<(), AttentionOutputQualificationError> {
    let attention_begin = batch * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    let output_begin = batch * Qwen38_27B::HIDDEN;
    if observed.attention[attention_begin..]
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.activation_codes[attention_begin..]
            .iter()
            .any(|&value| value != BYTE_SENTINEL)
        || observed.activation_scales[batch..]
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} modified an inactive value"
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (2 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + 1 + Qwen38_27B::HIDDEN)
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut AttentionOutputQualification,
) -> Result<(), AttentionOutputQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(AttentionOutputQualificationError::Mismatch(format!(
                    "read-only {} changed at index {index}",
                    $name
                )));
            }
            report.immutable_input_values += actual.len();
        }};
    }

    check!(regions.qkv, &fixture.qkv, "fused QKV");
    check!(regions.weight_codes, &fixture.weight_codes, "weight codes");
    check!(
        regions.weight_scales,
        &fixture.weight_scales,
        "weight scales"
    );

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut AttentionOutputQualification,
) -> Result<(), AttentionOutputQualificationError> {
    let same = replay
        .attention
        .iter()
        .map(|value| value.to_bits())
        .eq(eager.attention.iter().map(|value| value.to_bits()))
        && replay.activation_codes == eager.activation_codes
        && replay
            .activation_scales
            .iter()
            .map(|value| value.to_bits())
            .eq(eager.activation_scales.iter().map(|value| value.to_bits()))
        && replay.output == eager.output;
    if !same {
        return Err(AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += replay.attention.len()
        + replay.activation_codes.len()
        + replay.activation_scales.len()
        + replay.output.len();
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &AttentionOutputOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), AttentionOutputQualificationError> {
    reset(arena, stream, regions, fixture, MAX_BATCH)?;
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            graphs[batch - 1].launch(stream)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(AttentionOutputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, Qwen38_27B, layout, qualify_attention_output};
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::AttentionOutputQualificationError> {
        let report = qualify_attention_output()?;
        let active = (1..=MAX_BATCH).sum::<usize>();
        let mutable = 2 * MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + MAX_BATCH
            + MAX_BATCH * Qwen38_27B::HIDDEN;
        let immutable = MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS
            + Qwen38_27B::HIDDEN * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + Qwen38_27B::HIDDEN;

        assert_eq!(
            report.gated_values,
            active * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
        );
        assert_eq!(report.activation_codes, report.gated_values);
        assert_eq!(report.activation_scales, active);
        assert_eq!(report.output_values, active * Qwen38_27B::HIDDEN);
        assert_eq!(report.graph_replay_values, MAX_BATCH * mutable);
        assert_eq!(
            report.inactive_values,
            2 * (0..MAX_BATCH).sum::<usize>()
                * (2 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + 1 + Qwen38_27B::HIDDEN)
        );
        assert_eq!(report.immutable_input_values, 2 * MAX_BATCH * immutable);
        assert!(report.maximum_gated_error <= 0.000_05);
        assert!(report.maximum_projection_relative_error <= 0.015);
        let (arena, regions) = layout()?;
        assert_eq!(regions.payload_bytes(), 32_024_608);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 224);
        assert_eq!(arena.byte_len(), 32_024_832);
        assert_eq!(report.padding_bytes, 224);
        assert_eq!(report.arena_bytes, 32_024_832);

        Ok(())
    }
}
