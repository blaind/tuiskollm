//! Represented-value qualification for Qwen3.6 gated static-FP8 attention output.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, WEIGHT_CODES, WEIGHT_VALUES, bf16_to_f32,
    decode_e4m3fn, encode_e4m3fn, f32_to_bf16,
};
use crate::target::Qwen36AttentionOutputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const COLUMNS: usize = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
pub(crate) const QKV_ROWS: usize = Qwen36Moe35B::ATTENTION_QKV_ROWS;
pub(crate) const OUTPUT_ROWS: usize = Qwen36Moe35B::HIDDEN;
pub(crate) const INPUT_SCALE: f32 = 0.125;
pub(crate) const WEIGHT_SCALE: f32 = 0.25;
const GROUP: usize = 16;
const ATTENTION_PATTERN: [f32; GROUP] = [
    1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.75, -0.75, 0.375, -0.375, 0.1875, -0.1875,
    0.0625, -0.0625,
];
const GATE_PATTERN: [f32; GROUP] = [
    0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, -0.75, 0.75, 0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact Qwen3.6 attention-output qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36AttentionOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.6 attention-output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36AttentionOutputQualification {
    /// Gated FP32 seam values compared with mathematical sigmoid.
    pub gated_values: usize,
    /// BF16 projection inputs compared bit-exactly.
    pub activation_values: usize,
    /// Static E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// BF16 projection outputs compared with the represented-value oracle.
    pub output_values: usize,
    /// Complete mutable seams reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside every active route extent.
    pub inactive_values: usize,
    /// Read-only QKV and represented weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact source E4M3 weight bytes.
    pub weight_bytes: usize,
    /// Exact mutable and read-only QKV workspace bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute gated-FP32 difference.
    pub maximum_gated_error: f32,
    /// Largest absolute BF16 projection difference.
    pub maximum_projection_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) activation: ArenaRegion<u16>,
    pub(crate) activation_codes: ArenaRegion<u8>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation.byte_len()
            + self.activation_codes.byte_len()
            + self.weight_codes.byte_len()
            + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) attention: Vec<f32>,
    pub(crate) qkv: Vec<u16>,
    gated: Vec<f32>,
    pub(crate) activation_bf16: Vec<u16>,
    activation_codes: Vec<u8>,
    pub(crate) weight_codes: Vec<u8>,
    row_sums: Vec<f64>,
}

struct Observed {
    attention: Vec<f32>,
    activation: Vec<u16>,
    activation_codes: Vec<u8>,
    output: Vec<u16>,
}

/// Qualifies eager and captured Qwen3.6 attention output at exact `B=1..=8`.
pub fn qualify_qwen36_attention_output()
-> Result<Qwen36AttentionOutputQualification, Qwen36AttentionOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen36AttentionOutputOp::new(&context)?;
    let fixture = make_fixture()?;
    load_immutable(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen36AttentionOutputQualification {
        gated_values: 0,
        activation_values: 0,
        activation_codes: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_gated_error: 0.0,
        maximum_projection_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions, &fixture, batch)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions, &fixture, batch)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_BATCH * QKV_ROWS, ALIGNMENT)?;
    let activation = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * COLUMNS, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation,
            activation_codes,
            weight_codes,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.attention)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.activation)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)
}

fn reset(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    batch: usize,
) -> GpuResult<()> {
    arena.fill(stream, regions.attention, BYTE_SENTINEL)?;
    arena.copy_prefix_from_host(
        stream,
        regions.attention,
        &fixture.attention[..batch * COLUMNS],
    )?;
    arena.fill(stream, regions.activation, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    op: &Qwen36AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.attention)?,
            arena.address(regions.qkv)?,
            arena.address(regions.activation)?,
            arena.address(regions.activation_codes)?,
            INPUT_SCALE,
            arena.address(regions.weight_codes)?,
            WEIGHT_SCALE,
            arena.address(regions.output)?,
        )
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation: arena.copy_to_host(stream, regions.activation)?,
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

pub(crate) fn make_fixture() -> Result<Fixture, Qwen36AttentionOutputQualificationError> {
    let attention = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let token = index / COLUMNS;
            ATTENTION_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token]
        })
        .collect::<Vec<_>>();
    let mut qkv = vec![BF16_SENTINEL; MAX_BATCH * QKV_ROWS];
    for token in 0..MAX_BATCH {
        for head in 0..Qwen36Moe35B::NUM_ATTENTION_HEADS {
            for dimension in 0..Qwen36Moe35B::HEAD_DIM {
                let gate = token * QKV_ROWS
                    + head * 2 * Qwen36Moe35B::HEAD_DIM
                    + Qwen36Moe35B::HEAD_DIM
                    + dimension;
                qkv[gate] = f32_to_bf16(GATE_PATTERN[dimension & (GROUP - 1)]);
            }
        }
    }
    let gated = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let gate = f64::from(GATE_PATTERN[index & (GROUP - 1)]);
            (f64::from(attention[index]) / (1.0 + (-gate).exp())) as f32
        })
        .collect::<Vec<_>>();
    let activation_bf16 = gated.iter().copied().map(f32_to_bf16).collect::<Vec<_>>();
    let activation_codes = activation_bf16
        .iter()
        .map(|&bits| {
            encode_e4m3fn(bf16_to_f32(bits) / INPUT_SCALE)
                .map_err(Qwen36AttentionOutputQualificationError::Mismatch)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weight_codes = (0..OUTPUT_ROWS * COLUMNS)
        .map(|index| WEIGHT_CODES[(index / COLUMNS) & 3])
        .collect::<Vec<_>>();
    let row_sums = (0..MAX_BATCH)
        .map(|token| {
            activation_codes[token * COLUMNS..(token + 1) * COLUMNS]
                .iter()
                .map(|&code| {
                    decode_e4m3fn(code)
                        .map(f64::from)
                        .map_err(Qwen36AttentionOutputQualificationError::Mismatch)
                })
                .sum::<Result<f64, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Fixture {
        attention,
        qkv,
        gated,
        activation_bf16,
        activation_codes,
        weight_codes,
        row_sums,
    })
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen36AttentionOutputQualification,
) -> Result<(), Qwen36AttentionOutputQualificationError> {
    let active_columns = batch * COLUMNS;
    for index in 0..active_columns {
        let actual = observed.attention[index];
        let expected = fixture.gated[index];
        let error = (actual - expected).abs();
        let tolerance = 0.000_05f32.max(expected.abs() * 0.000_25);
        report.maximum_gated_error = report.maximum_gated_error.max(error);
        if !actual.is_finite() || error > tolerance {
            return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
                "B={batch} gated index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    report.gated_values += active_columns;

    if observed.activation[..active_columns] != fixture.activation_bf16[..active_columns] {
        let index = observed.activation[..active_columns]
            .iter()
            .zip(&fixture.activation_bf16[..active_columns])
            .position(|(actual, expected)| actual != expected)
            .expect("unequal slices contain one differing value");
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} BF16 activation {index} differs"
        )));
    }
    report.activation_values += active_columns;

    if observed.activation_codes[..active_columns] != fixture.activation_codes[..active_columns] {
        let index = observed.activation_codes[..active_columns]
            .iter()
            .zip(&fixture.activation_codes[..active_columns])
            .position(|(actual, expected)| actual != expected)
            .expect("unequal slices contain one differing code");
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} activation code {index} is {:#04x}, expected {:#04x}",
            observed.activation_codes[index], fixture.activation_codes[index]
        )));
    }
    report.activation_codes += active_columns;

    for token in 0..batch {
        for row in 0..OUTPUT_ROWS {
            let expected = fixture.row_sums[token]
                * f64::from(WEIGHT_VALUES[row & 3])
                * f64::from(INPUT_SCALE * WEIGHT_SCALE);
            let actual = f64::from(bf16_to_f32(observed.output[token * OUTPUT_ROWS + row]));
            let error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error as f32);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
                    "B={batch} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }
    report.output_values += batch * OUTPUT_ROWS;

    verify_inactive(batch, observed)?;
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen36AttentionOutputQualification,
) -> Result<(), Qwen36AttentionOutputQualificationError> {
    let same = replay
        .attention
        .iter()
        .map(|value| value.to_bits())
        .eq(eager.attention.iter().map(|value| value.to_bits()))
        && replay.activation == eager.activation
        && replay.activation_codes == eager.activation_codes
        && replay.output == eager.output;
    if !same {
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (3 * COLUMNS + OUTPUT_ROWS);
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
) -> Result<(), Qwen36AttentionOutputQualificationError> {
    let columns_begin = batch * COLUMNS;
    let output_begin = batch * OUTPUT_ROWS;
    if observed.attention[columns_begin..]
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.activation[columns_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
        || observed.activation_codes[columns_begin..]
            .iter()
            .any(|&value| value != BYTE_SENTINEL)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} modified an inactive value"
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (3 * COLUMNS + OUTPUT_ROWS)
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen36AttentionOutputQualification,
) -> Result<(), Qwen36AttentionOutputQualificationError> {
    let qkv = arena.copy_to_host(stream, regions.qkv)?;
    let weights = arena.copy_to_host(stream, regions.weight_codes)?;
    if qkv != fixture.qkv || weights != fixture.weight_codes {
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(
            "read-only QKV or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = qkv.len() + weights.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen36AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), Qwen36AttentionOutputQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| {
            reset(arena, stream, regions, fixture, batch)?;
            stream.synchronize().map_err(GpuError::from)?;
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch))
        })
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graphs[batch - 1].launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen36AttentionOutputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_accounting_matches_exact_geometry() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 8_388_608);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 409_600);
        assert_eq!(layout.byte_len(), 8_798_208);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen36AttentionOutputQualificationError> {
        let report = qualify_qwen36_attention_output()?;
        let active = (1..=MAX_BATCH).sum::<usize>();
        let inactive = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.gated_values, active * COLUMNS);
        assert_eq!(report.activation_values, active * COLUMNS);
        assert_eq!(report.activation_codes, active * COLUMNS);
        assert_eq!(report.output_values, active * OUTPUT_ROWS);
        assert_eq!(
            report.graph_replay_values,
            active * (3 * COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive * (3 * COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(report.immutable_input_values, 8_462_336);
        assert_eq!(report.arena_bytes, 8_798_208);
        assert_eq!(report.weight_bytes, 8_388_608);
        assert_eq!(report.workspace_bytes, 409_600);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_gated_error.is_finite());
        assert!(report.maximum_projection_error.is_finite());

        Ok(())
    }
}
