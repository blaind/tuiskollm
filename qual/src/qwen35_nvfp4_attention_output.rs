//! Qwen3.5 gated NVFP4 attention-output qualification.

use crate::device_benchmark;
use crate::nvfp4_down::{bf16_to_f32, decode_e2m1, decode_e4m3fn, f32_to_bf16};
use crate::target::Qwen35Nvfp4AttentionOutputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const COLUMNS: usize = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
const QKV_ROWS: usize = Qwen35_9B::ATTENTION_QKV_ROWS;
const OUTPUT_ROWS: usize = Qwen35_9B::HIDDEN;
const GROUP: usize = 16;
const GROUPS_PER_ROW: usize = COLUMNS / GROUP;
const CODE_BYTES_PER_ROW: usize = COLUMNS / 2;
const WEIGHT_SCALE_DIVISOR: f32 = 16.0;
const BYTE_SENTINEL: u8 = 0xa5;
const BF16_SENTINEL: u16 = 0xa5a5;
const F32_SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const ATTENTION_PATTERN: [f32; GROUP] = [
    1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.75, -0.75, 0.375, -0.375, 0.1875, -0.1875,
    0.0625, -0.0625,
];
const GATE_PATTERN: [f32; GROUP] = [
    0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, -0.75, 0.75, 0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of Qwen3.5 gated NVFP4 attention-output qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35Nvfp4AttentionOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 NVFP4 attention-output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35Nvfp4AttentionOutputQualification {
    /// Gated FP32 seam values compared with mathematical sigmoid.
    pub gated_values: usize,
    /// BF16 projection inputs compared bit-exactly.
    pub activation_values: usize,
    /// BF16 outputs compared with the represented-value oracle.
    pub output_values: usize,
    /// Complete mutable seams reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside every active route extent.
    pub inactive_values: usize,
    /// Read-only QKV and represented weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact packed-weight and block-scale bytes.
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
struct Regions {
    attention: ArenaRegion<f32>,
    qkv: ArenaRegion<u16>,
    activation: ArenaRegion<u16>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u8>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    fn payload_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation.byte_len()
            + self.weight_bytes()
            + self.output.byte_len()
    }
}

struct Fixture {
    attention: Vec<f32>,
    qkv: Vec<u16>,
    gated: Vec<f32>,
    activation_bf16: Vec<u16>,
    activation_f32: Vec<f32>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u8>,
}

struct Observed {
    attention: Vec<f32>,
    activation: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies eager and captured Qwen3.5 attention output at exact `B=1..=8`.
pub fn qualify_qwen35_nvfp4_attention_output()
-> Result<Qwen35Nvfp4AttentionOutputQualification, Qwen35Nvfp4AttentionOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen35Nvfp4AttentionOutputOp::new(&context)?;
    let fixture = make_fixture();
    load_immutable(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4AttentionOutputQualification {
        gated_values: 0,
        activation_values: 0,
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
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                format!("device addresses changed while qualifying B={batch}"),
            ));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_BATCH * QKV_ROWS, ALIGNMENT)?;
    let activation = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.attention)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.activation)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
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
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(stream, regions.weight_scales, &fixture.weight_scales)
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
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch(
    op: &Qwen35Nvfp4AttentionOutputOp,
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
            arena.address(regions.weight_codes)?,
            arena.address(regions.weight_scales)?,
            WEIGHT_SCALE_DIVISOR,
            arena.address(regions.output)?,
        )
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation: arena.copy_to_host(stream, regions.activation)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn make_fixture() -> Fixture {
    let attention = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let token = index / COLUMNS;
            ATTENTION_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token]
        })
        .collect::<Vec<_>>();
    let mut qkv = vec![BF16_SENTINEL; MAX_BATCH * QKV_ROWS];
    for token in 0..MAX_BATCH {
        for head in 0..Qwen35_9B::NUM_ATTENTION_HEADS {
            for dimension in 0..Qwen35_9B::HEAD_DIM {
                let gate = token * QKV_ROWS
                    + head * 2 * Qwen35_9B::HEAD_DIM
                    + Qwen35_9B::HEAD_DIM
                    + dimension;
                qkv[gate] = f32_to_bf16(GATE_PATTERN[dimension & (GROUP - 1)]);
            }
        }
    }
    let gated = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let token = index / COLUMNS;
            let column = index - token * COLUMNS;
            let gate = f64::from(GATE_PATTERN[column & (GROUP - 1)]);
            (f64::from(attention[index]) / (1.0 + (-gate).exp())) as f32
        })
        .collect::<Vec<_>>();
    let activation_bf16 = gated.iter().copied().map(f32_to_bf16).collect::<Vec<_>>();
    let activation_f32 = activation_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let (weight_codes, weight_scales) = make_weights();

    Fixture {
        attention,
        qkv,
        gated,
        activation_bf16,
        activation_f32,
        weight_codes,
        weight_scales,
    }
}

fn make_weights() -> (Vec<u8>, Vec<u8>) {
    const BASE: [u8; 8] = [0xf7, 0xd5, 0xb3, 0x70, 0x5f, 0x3d, 0x0b, 0xf7];
    const SPARSE: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
    const SCALE_CODES: [u8; 4] = [0x38, 0x30, 0x40, 0x28];
    let negative = BASE.map(|byte| byte ^ 0x88);
    let mut codes = vec![0u8; OUTPUT_ROWS * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; OUTPUT_ROWS * GROUPS_PER_ROW];

    for row in 0..OUTPUT_ROWS {
        let base_is_base = row & 1 == 0;
        let base = if base_is_base { &BASE } else { &negative };
        let exceptional = if base_is_base { &SPARSE } else { &BASE };
        let exceptional_group = exceptional_group(row);
        for group in 0..GROUPS_PER_ROW {
            let begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
            let pattern = if group == exceptional_group {
                exceptional
            } else {
                base
            };
            codes[begin..begin + GROUP / 2].copy_from_slice(pattern);
            let scale_index = if group == exceptional_group {
                (row + 1) & 3
            } else {
                row & 3
            };
            scales[scale_offset(row, group)] = SCALE_CODES[scale_index];
        }
    }

    (codes, scales)
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35Nvfp4AttentionOutputQualification,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    for token in 0..batch {
        let begin = token * COLUMNS;
        for column in 0..COLUMNS {
            let actual = observed.attention[begin + column];
            let expected = fixture.gated[begin + column];
            let error = (actual - expected).abs();
            let tolerance = 0.000_05f32.max(expected.abs() * 0.000_25);
            report.maximum_gated_error = report.maximum_gated_error.max(error);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                    format!(
                        "B={batch} gated token={token}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
                    ),
                ));
            }
        }
        if let Some(column) = observed.activation[begin..begin + COLUMNS]
            .iter()
            .zip(&fixture.activation_bf16[begin..begin + COLUMNS])
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                format!("B={batch} BF16 activation token={token}, column={column} differs"),
            ));
        }
        for row in 0..OUTPUT_ROWS {
            let expected = dot_oracle(token, row, fixture)?;
            let actual = f64::from(bf16_to_f32(observed.output[token * OUTPUT_ROWS + row]));
            let error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error as f32);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                    format!(
                        "B={batch} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                    ),
                ));
            }
        }
    }

    verify_inactive(batch, observed)?;
    report.gated_values += batch * COLUMNS;
    report.activation_values += batch * COLUMNS;
    report.output_values += batch * OUTPUT_ROWS;
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen35Nvfp4AttentionOutputQualification,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let same = replay
        .attention
        .iter()
        .map(|value| value.to_bits())
        .eq(eager.attention.iter().map(|value| value.to_bits()))
        && replay.activation == eager.activation
        && replay.output == eager.output;
    if !same {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!("B={batch} graph replay differs from eager execution"),
        ));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (2 * COLUMNS + OUTPUT_ROWS);
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let columns_begin = batch * COLUMNS;
    let output_begin = batch * OUTPUT_ROWS;
    if observed.attention[columns_begin..]
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.activation[columns_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!("B={batch} modified an inactive value"),
        ));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (2 * COLUMNS + OUTPUT_ROWS)
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen35Nvfp4AttentionOutputQualification,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let qkv = arena.copy_to_host(stream, regions.qkv)?;
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if qkv != fixture.qkv
        || weight_codes != fixture.weight_codes
        || weight_scales != fixture.weight_scales
    {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            "read-only QKV or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = qkv.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Nvfp4AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| {
            reset(arena, stream, regions, fixture, batch)?;
            stream.synchronize().map_err(GpuError::from)?;
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch))
        })
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
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

fn dot_oracle(
    token: usize,
    row: usize,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4AttentionOutputQualificationError> {
    let exceptional = exceptional_group(row);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, fixture);
    let exceptional_dot = group_dot(token, row, exceptional, fixture);
    let ordinary_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, ordinary)])
        .map_err(|error| {
            Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(error.to_string())
        })?;
    let exceptional_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, exceptional)])
        .map_err(|error| {
            Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(error.to_string())
        })?;

    Ok(
        ((GROUPS_PER_ROW - 1) as f64 * ordinary_dot * f64::from(ordinary_scale)
            + exceptional_dot * f64::from(exceptional_scale))
            / f64::from(WEIGHT_SCALE_DIVISOR),
    )
}

fn group_dot(token: usize, row: usize, group: usize, fixture: &Fixture) -> f64 {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * COLUMNS + group * GROUP;
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = fixture.weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        sum +=
            f64::from(fixture.activation_f32[input_begin + column]) * f64::from(decode_e2m1(code));
    }

    sum
}

fn exceptional_group(row: usize) -> usize {
    (row * 17 + row / 128 * 13) % GROUPS_PER_ROW
}

fn scale_offset(row: usize, group: usize) -> usize {
    let tile = row / 128;
    let row_in_tile = row & 127;
    let scale_tile = group / 4;
    let scale_lane = group & 3;
    let row_mod32 = row_in_tile & 31;
    let row_quartile = row_in_tile >> 5;

    (tile * (GROUPS_PER_ROW / 4) + scale_tile) * 512
        + row_mod32 * 16
        + row_quartile * 4
        + scale_lane
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_swizzle_and_fixture_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();
        let fixture = make_fixture();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(127, 255), 32_767);
        assert_eq!(scale_offset(128, 0), 32_768);
        assert_eq!(fixture.activation_bf16.len(), MAX_BATCH * COLUMNS);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 425_984);
        assert_eq!(layout.byte_len(), 9_863_168);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
        let report = qualify_qwen35_nvfp4_attention_output()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.gated_values, active_rows * COLUMNS);
        assert_eq!(report.activation_values, active_rows * COLUMNS);
        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(
            report.graph_replay_values,
            active_rows * (2 * COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive_rows * (2 * COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(report.immutable_input_values, 9_519_104);
        assert_eq!(report.arena_bytes, 9_863_168);
        assert_eq!(report.weight_bytes, 9_437_184);
        assert_eq!(report.workspace_bytes, 425_984);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_gated_error.is_finite());
        assert!(report.maximum_projection_error.is_finite());

        Ok(())
    }
}
