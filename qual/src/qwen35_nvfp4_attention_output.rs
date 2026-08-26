//! Qwen3.5 gated NVFP4 attention-output qualification.

use crate::device_benchmark;
use crate::nvfp4_down_sm120::{bf16_to_f32, decode_e2m1, decode_e4m3fn, f32_to_bf16};
use crate::oracles::codecs;
use crate::oracles::codecs::encode_e2m1;
use crate::target::Qwen35Nvfp4AttentionOutputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 128;
pub(crate) const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
pub(crate) const COLUMNS: usize = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
pub(crate) const QKV_ROWS: usize = Qwen35_9B::ATTENTION_QKV_ROWS;
pub(crate) const OUTPUT_ROWS: usize = Qwen35_9B::HIDDEN;
const GROUP: usize = 16;
pub(crate) const GROUPS_PER_ROW: usize = COLUMNS / GROUP;
pub(crate) const CODE_BYTES_PER_ROW: usize = COLUMNS / 2;
pub(crate) const INPUT_SCALE_DIVISOR: f32 = 3.0;
pub(crate) const WEIGHT_SCALE_DIVISOR: f32 = 16.0;
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
    /// Exact represented activation codes produced by prompt quantization.
    pub activation_codes: usize,
    /// Exact E4M3 activation scales produced by prompt quantization.
    pub activation_scales: usize,
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
pub(crate) struct Regions {
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) activation: ArenaRegion<u16>,
    pub(crate) activation_codes: ArenaRegion<u8>,
    pub(crate) activation_scales: ArenaRegion<u8>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) weight_scales: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_bytes()
            + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) attention: Vec<f32>,
    pub(crate) qkv: Vec<u16>,
    gated: Vec<f32>,
    pub(crate) activation_bf16: Vec<u16>,
    pub(crate) activation_f32: Vec<f32>,
    pub(crate) activation_codes: Vec<u8>,
    pub(crate) activation_scales: Vec<u8>,
    pub(crate) weight_codes: Vec<u8>,
    pub(crate) weight_scales: Vec<u8>,
}

struct Observed {
    attention: Vec<f32>,
    activation: Vec<u16>,
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    output: Vec<u16>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Schedule {
    A16,
    W4a4,
}

/// Qualifies eager and captured Qwen3.5 attention output at every exact route.
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
    let fixture = make_fixture()?;
    load_immutable(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4AttentionOutputQualification {
        gated_values: 0,
        activation_values: 0,
        activation_codes: 0,
        activation_scales: 0,
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

    for rows in EXACT_ROUTES {
        let schedule = schedule(rows);
        reset(&arena, &stream, regions, &fixture, rows)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, schedule, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions, &fixture, rows)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, schedule, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                format!(
                    "device addresses changed while qualifying {}",
                    route_name(rows)
                ),
            ));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_ROWS * COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_ROWS * QKV_ROWS, ALIGNMENT)?;
    let activation = layout.reserve(MAX_ROWS * COLUMNS, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 8]> {
    Ok([
        arena.address(regions.attention)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.activation)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.activation_scales)?.addr(),
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
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch(
    op: &Qwen35Nvfp4AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        if rows <= MAX_BATCH {
            op.launch(
                stream,
                rows,
                arena.address(regions.attention)?,
                arena.address(regions.qkv)?,
                arena.address(regions.activation)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                WEIGHT_SCALE_DIVISOR,
                arena.address(regions.output)?,
            )
        } else {
            op.launch_prefill(
                stream,
                rows,
                arena.address(regions.attention)?,
                arena.address(regions.qkv)?,
                arena.address(regions.activation)?,
                arena.address(regions.activation_codes)?,
                arena.address(regions.activation_scales)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                arena.address(regions.output)?,
            )
        }
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation: arena.copy_to_host(stream, regions.activation)?,
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

pub(crate) fn make_fixture() -> Result<Fixture, Qwen35Nvfp4AttentionOutputQualificationError> {
    let attention = (0..MAX_ROWS * COLUMNS)
        .map(|index| {
            let token = index / COLUMNS;
            ATTENTION_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token & 7]
        })
        .collect::<Vec<_>>();
    let mut qkv = vec![BF16_SENTINEL; MAX_ROWS * QKV_ROWS];
    for token in 0..MAX_ROWS {
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
    let gated = (0..MAX_ROWS * COLUMNS)
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
    let (activation_codes, activation_scales) = quantize_oracle(&activation_f32)?;
    let (weight_codes, weight_scales) = make_weights();

    Ok(Fixture {
        attention,
        qkv,
        gated,
        activation_bf16,
        activation_f32,
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
    })
}

fn quantize_oracle(
    input: &[f32],
) -> Result<(Vec<u8>, Vec<u8>), Qwen35Nvfp4AttentionOutputQualificationError> {
    let tokens = input.len() / COLUMNS;
    let mut codes = vec![0u8; tokens * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; tokens * GROUPS_PER_ROW];

    for token in 0..tokens {
        for group in 0..GROUPS_PER_ROW {
            let input_begin = token * COLUMNS + group * GROUP;
            let values = &input[input_begin..input_begin + GROUP];
            let maximum = values
                .iter()
                .fold(0.0f32, |current, value| current.max(value.abs()));
            let scale = encode_e4m3fn(INPUT_SCALE_DIVISOR * maximum / 6.0)?;
            scales[token * GROUPS_PER_ROW + group] = scale;
            if scale == 0 {
                continue;
            }

            let decoded_scale = decode_e4m3fn(scale).map_err(|error| {
                Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(error.to_string())
            })?;
            for pair in 0..GROUP / 2 {
                let low = encode_e2m1(values[2 * pair] * INPUT_SCALE_DIVISOR / decoded_scale);
                let high = encode_e2m1(values[2 * pair + 1] * INPUT_SCALE_DIVISOR / decoded_scale);
                let destination = token * CODE_BYTES_PER_ROW + group * (GROUP / 2) + pair;
                codes[destination] = low | (high << 4);
            }
        }
    }

    Ok((codes, scales))
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
    rows: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35Nvfp4AttentionOutputQualification,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    verify_scratch(rows, schedule, fixture, observed)?;
    for token in 0..rows {
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
                        "{} gated token={token}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}",
                        route_name(rows)
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
                format!(
                    "{} BF16 activation token={token}, column={column} differs",
                    route_name(rows)
                ),
            ));
        }
        for row in 0..OUTPUT_ROWS {
            let expected = dot_oracle_for_schedule(token, row, schedule, fixture)?;
            let actual = f64::from(bf16_to_f32(observed.output[token * OUTPUT_ROWS + row]));
            let error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error as f32);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
                    format!(
                        "{} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}",
                        route_name(rows)
                    ),
                ));
            }
        }
    }

    verify_inactive(rows, schedule, observed)?;
    report.gated_values += rows * COLUMNS;
    report.activation_values += rows * COLUMNS;
    if schedule == Schedule::W4a4 {
        report.activation_codes += rows * CODE_BYTES_PER_ROW;
        report.activation_scales += rows * GROUPS_PER_ROW;
    }
    report.output_values += rows * OUTPUT_ROWS;
    report.inactive_values += inactive_values(rows, schedule);

    Ok(())
}

fn verify_replay(
    rows: usize,
    schedule: Schedule,
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
        && replay.activation_codes == eager.activation_codes
        && replay.activation_scales == eager.activation_scales
        && replay.output == eager.output;
    if !same {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!(
                "{} graph replay differs from eager execution",
                route_name(rows)
            ),
        ));
    }
    verify_inactive(rows, schedule, replay)?;
    report.graph_replay_values += rows * (2 * COLUMNS + OUTPUT_ROWS);
    if schedule == Schedule::W4a4 {
        report.graph_replay_values += rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    }
    report.inactive_values += inactive_values(rows, schedule);

    Ok(())
}

fn verify_scratch(
    rows: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let active_codes = if schedule == Schedule::W4a4 {
        rows * CODE_BYTES_PER_ROW
    } else {
        0
    };
    let active_scales = if schedule == Schedule::W4a4 {
        rows * GROUPS_PER_ROW
    } else {
        0
    };
    if let Some(index) = observed.activation_codes[..active_codes]
        .iter()
        .zip(&fixture.activation_codes[..active_codes])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!(
                "{} activation code {index}: device={:#04x}, oracle={:#04x}",
                route_name(rows),
                observed.activation_codes[index],
                fixture.activation_codes[index]
            ),
        ));
    }
    if let Some(index) = observed.activation_scales[..active_scales]
        .iter()
        .zip(&fixture.activation_scales[..active_scales])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!(
                "{} activation scale {index}: device={:#04x}, oracle={:#04x}",
                route_name(rows),
                observed.activation_scales[index],
                fixture.activation_scales[index]
            ),
        ));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    schedule: Schedule,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
    let columns_begin = rows * COLUMNS;
    let code_begin = if schedule == Schedule::W4a4 {
        rows * CODE_BYTES_PER_ROW
    } else {
        0
    };
    let scale_begin = if schedule == Schedule::W4a4 {
        rows * GROUPS_PER_ROW
    } else {
        0
    };
    let output_begin = rows * OUTPUT_ROWS;
    if observed.attention[columns_begin..]
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.activation[columns_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
        || observed.activation_codes[code_begin..]
            .iter()
            .any(|&value| value != BYTE_SENTINEL)
        || observed.activation_scales[scale_begin..]
            .iter()
            .any(|&value| value != BYTE_SENTINEL)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!("{} modified an inactive value", route_name(rows)),
        ));
    }

    Ok(())
}

fn inactive_values(rows: usize, schedule: Schedule) -> usize {
    let scratch = if schedule == Schedule::W4a4 {
        (MAX_ROWS - rows) * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    } else {
        MAX_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    };

    (MAX_ROWS - rows) * (2 * COLUMNS + OUTPUT_ROWS) + scratch
}

fn schedule(rows: usize) -> Schedule {
    if rows <= MAX_BATCH {
        Schedule::A16
    } else {
        Schedule::W4a4
    }
}

fn route_name(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
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
    let graphs = EXACT_ROUTES
        .into_iter()
        .map(|rows| {
            reset(arena, stream, regions, fixture, rows)?;
            stream.synchronize().map_err(GpuError::from)?;
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows))
        })
        .collect::<GpuResult<Vec<_>>>()?;
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
        return Err(Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

pub(crate) fn dot_oracle_for_rows(
    token: usize,
    row: usize,
    rows: usize,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4AttentionOutputQualificationError> {
    dot_oracle_for_schedule(token, row, schedule(rows), fixture)
}

fn dot_oracle_for_schedule(
    token: usize,
    row: usize,
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4AttentionOutputQualificationError> {
    let exceptional = exceptional_group(row);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, schedule, fixture)?;
    let exceptional_dot = group_dot(token, row, exceptional, schedule, fixture)?;
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

fn group_dot(
    token: usize,
    row: usize,
    group: usize,
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4AttentionOutputQualificationError> {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * COLUMNS + group * GROUP;
    let activation_begin = token * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let activation_scale = if schedule == Schedule::W4a4 {
        decode_e4m3fn(fixture.activation_scales[token * GROUPS_PER_ROW + group]).map_err(
            |error| Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(error.to_string()),
        )?
    } else {
        1.0
    };
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = fixture.weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        let activation = match schedule {
            Schedule::A16 => f64::from(fixture.activation_f32[input_begin + column]),
            Schedule::W4a4 => {
                let packed = fixture.activation_codes[activation_begin + column / 2];
                let code = if column & 1 == 0 {
                    packed & 15
                } else {
                    packed >> 4
                };
                f64::from(decode_e2m1(code) * activation_scale / INPUT_SCALE_DIVISOR)
            }
        };
        sum += activation * f64::from(decode_e2m1(code));
    }

    Ok(sum)
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

fn encode_e4m3fn(value: f32) -> Result<u8, Qwen35Nvfp4AttentionOutputQualificationError> {
    codecs::encode_e4m3fn_scale(value).ok_or_else(|| {
        Qwen35Nvfp4AttentionOutputQualificationError::Mismatch(
            "Qwen3.5 oracle E4M3 scale is not finite and non-negative".to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_swizzle_and_fixture_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();
        let fixture = make_fixture().unwrap();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(127, 255), 32_767);
        assert_eq!(scale_offset(128, 0), 32_768);
        assert_eq!(fixture.activation_bf16.len(), MAX_ROWS * COLUMNS);
        assert_eq!(
            fixture.activation_codes.len(),
            MAX_ROWS * CODE_BYTES_PER_ROW
        );
        assert_eq!(fixture.activation_scales.len(), MAX_ROWS * GROUPS_PER_ROW);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 7_110_656);
        assert_eq!(layout.byte_len(), 16_547_840);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4AttentionOutputQualificationError> {
        let report = qualify_qwen35_nvfp4_attention_output()?;
        let active_rows = EXACT_ROUTES.into_iter().sum::<usize>();
        let prefill_rows = EXACT_ROUTES
            .into_iter()
            .filter(|&rows| schedule(rows) == Schedule::W4a4)
            .sum::<usize>();
        let inactive = EXACT_ROUTES
            .into_iter()
            .map(|rows| inactive_values(rows, schedule(rows)))
            .sum::<usize>();

        assert_eq!(report.gated_values, active_rows * COLUMNS);
        assert_eq!(report.activation_values, active_rows * COLUMNS);
        assert_eq!(report.activation_codes, prefill_rows * CODE_BYTES_PER_ROW);
        assert_eq!(report.activation_scales, prefill_rows * GROUPS_PER_ROW);
        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, 3_710_976);
        assert_eq!(report.inactive_values, 2 * inactive);
        assert_eq!(report.immutable_input_values, 10_747_904);
        assert_eq!(report.arena_bytes, 16_547_840);
        assert_eq!(report.weight_bytes, 9_437_184);
        assert_eq!(report.workspace_bytes, 7_110_656);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_gated_error.is_finite());
        assert!(report.maximum_projection_error.is_finite());

        Ok(())
    }
}
