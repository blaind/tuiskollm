//! Qwen3.5 represented-value qualification for NVFP4 GDN input projections.

use crate::device_benchmark;
use crate::nvfp4_down_sm120::{bf16_to_f32, decode_e2m1, decode_e4m3fn, f32_to_bf16};
use crate::oracles::codecs;
use crate::oracles::codecs::encode_e2m1;
use crate::target::Qwen35Nvfp4GdnInputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 128;
pub(crate) const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
pub(crate) const INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
pub(crate) const PROJECTED_ROWS: usize = Qwen35_9B::GDN_INPUT_ROWS;
pub(crate) const CONTROL_ROWS: usize = 2 * Qwen35_9B::GDN_CONTROL_ROWS;
pub(crate) const PADDED_CONTROL_ROWS: usize = 128;
const GROUP: usize = 16;
pub(crate) const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
pub(crate) const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
pub(crate) const INPUT_SCALE_DIVISOR: f32 = 3.0;
pub(crate) const PROJECTED_WEIGHT_SCALE_DIVISOR: f32 = 0.125;
pub(crate) const CONTROL_WEIGHT_SCALE_DIVISOR: f32 = 0.5;
const PROJECTED_SEED: usize = 0;
const CONTROL_SEED: usize = 7;
const BYTE_SENTINEL: u8 = 0xa5;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of the exact Qwen3.5 NVFP4 GDN input qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35Nvfp4GdnInputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 NVFP4 GDN input qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35Nvfp4GdnInputQualification {
    /// Exact represented activation codes produced by prompt quantization.
    pub activation_codes: usize,
    /// Exact E4M3 activation scales produced by prompt quantization.
    pub activation_scales: usize,
    /// Represented Q/K/V/Z and A/B outputs compared with the independent oracle.
    pub output_values: usize,
    /// Padded control outputs proved to remain exact zero.
    pub padded_output_values: usize,
    /// Active BF16 outputs reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active route extent.
    pub inactive_values: usize,
    /// Read-only input and represented weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact packed-weight and block-scale bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from the independent oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) activation_codes: ArenaRegion<u8>,
    pub(crate) activation_scales: ArenaRegion<u8>,
    pub(crate) projected_weight_codes: ArenaRegion<u8>,
    pub(crate) projected_weight_scales: ArenaRegion<u8>,
    pub(crate) control_weight_codes: ArenaRegion<u8>,
    pub(crate) control_weight_scales: ArenaRegion<u8>,
    pub(crate) projected_output: ArenaRegion<u16>,
    pub(crate) control_output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.projected_weight_codes.byte_len()
            + self.projected_weight_scales.byte_len()
            + self.control_weight_codes.byte_len()
            + self.control_weight_scales.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_bytes()
            + self.projected_output.byte_len()
            + self.control_output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    pub(crate) projected_weight_codes: Vec<u8>,
    pub(crate) projected_weight_scales: Vec<u8>,
    pub(crate) control_weight_codes: Vec<u8>,
    pub(crate) control_weight_scales: Vec<u8>,
}

struct Observed {
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    projected_output: Vec<u16>,
    control_output: Vec<u16>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Schedule {
    A16,
    W4a4,
}

/// Qualifies eager and captured Qwen3.5 GDN input decode and prefill routes.
pub fn qualify_qwen35_nvfp4_gdn_input()
-> Result<Qwen35Nvfp4GdnInputQualification, Qwen35Nvfp4GdnInputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen35Nvfp4GdnInputOp::new(&context)?;
    let fixture = make_fixture()?;
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4GdnInputQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        padded_output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        let schedule = schedule(rows);
        reset_observed(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify_eager(rows, schedule, &fixture, &eager, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        reset_observed(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(rows, schedule, &eager, &replay, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying {}",
                route_name(rows)
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * INPUT_COLUMNS, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let projected_weight_codes = layout.reserve(PROJECTED_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let projected_weight_scales = layout.reserve(PROJECTED_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let control_weight_codes =
        layout.reserve(PADDED_CONTROL_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let control_weight_scales = layout.reserve(PADDED_CONTROL_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let projected_output = layout.reserve(MAX_ROWS * PROJECTED_ROWS, ALIGNMENT)?;
    let control_output = layout.reserve(MAX_ROWS * PADDED_CONTROL_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            activation_codes,
            activation_scales,
            projected_weight_codes,
            projected_weight_scales,
            control_weight_codes,
            control_weight_scales,
            projected_output,
            control_output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 9]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.activation_scales)?.addr(),
        arena.address(regions.projected_weight_codes)?.addr(),
        arena.address(regions.projected_weight_scales)?.addr(),
        arena.address(regions.control_weight_codes)?.addr(),
        arena.address(regions.control_weight_scales)?.addr(),
        arena.address(regions.projected_output)?.addr(),
        arena.address(regions.control_output)?.addr(),
    ])
}

fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(
        stream,
        regions.projected_weight_codes,
        &fixture.projected_weight_codes,
    )?;
    arena.copy_from_host(
        stream,
        regions.projected_weight_scales,
        &fixture.projected_weight_scales,
    )?;
    arena.copy_from_host(
        stream,
        regions.control_weight_codes,
        &fixture.control_weight_codes,
    )?;
    arena.copy_from_host(
        stream,
        regions.control_weight_scales,
        &fixture.control_weight_scales,
    )
}

fn reset_observed(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.projected_output, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.control_output, BF16_SENTINEL as u8)
}

fn read_observed(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        projected_output: arena.copy_to_host(stream, regions.projected_output)?,
        control_output: arena.copy_to_host(stream, regions.control_output)?,
    })
}

fn launch(
    op: &Qwen35Nvfp4GdnInputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let activation_codes = arena.address(regions.activation_codes)?;
    let activation_scales = arena.address(regions.activation_scales)?;
    let projected_weight_codes = arena.address(regions.projected_weight_codes)?;
    let projected_weight_scales = arena.address(regions.projected_weight_scales)?;
    let control_weight_codes = arena.address(regions.control_weight_codes)?;
    let control_weight_scales = arena.address(regions.control_weight_scales)?;
    let projected_output = arena.address(regions.projected_output)?;
    let control_output = arena.address(regions.control_output)?;

    unsafe {
        if rows <= MAX_BATCH {
            op.launch(
                stream,
                rows,
                input,
                projected_weight_codes,
                projected_weight_scales,
                PROJECTED_WEIGHT_SCALE_DIVISOR,
                control_weight_codes,
                control_weight_scales,
                CONTROL_WEIGHT_SCALE_DIVISOR,
                projected_output,
                control_output,
            )
        } else {
            op.launch_prefill(
                stream,
                rows,
                input,
                activation_codes,
                activation_scales,
                projected_weight_codes,
                projected_weight_scales,
                PROJECTED_WEIGHT_SCALE_DIVISOR,
                control_weight_codes,
                control_weight_scales,
                CONTROL_WEIGHT_SCALE_DIVISOR,
                INPUT_SCALE_DIVISOR,
                projected_output,
                control_output,
            )
        }
    }
}

pub(crate) fn make_fixture() -> Result<Fixture, Qwen35Nvfp4GdnInputQualificationError> {
    let input_bf16 = (0..MAX_ROWS * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token & 7])
        })
        .collect::<Vec<_>>();
    let input_f32 = input_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let (activation_codes, activation_scales) = quantize_oracle(&input_f32)?;
    let (projected_weight_codes, projected_weight_scales) =
        make_weights(PROJECTED_ROWS, PROJECTED_ROWS, PROJECTED_SEED);
    let (control_weight_codes, control_weight_scales) =
        make_weights(PADDED_CONTROL_ROWS, CONTROL_ROWS, CONTROL_SEED);

    Ok(Fixture {
        input_bf16,
        input_f32,
        activation_codes,
        activation_scales,
        projected_weight_codes,
        projected_weight_scales,
        control_weight_codes,
        control_weight_scales,
    })
}

fn quantize_oracle(
    input: &[f32],
) -> Result<(Vec<u8>, Vec<u8>), Qwen35Nvfp4GdnInputQualificationError> {
    let tokens = input.len() / INPUT_COLUMNS;
    let mut codes = vec![0u8; tokens * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; tokens * GROUPS_PER_ROW];

    for token in 0..tokens {
        for group in 0..GROUPS_PER_ROW {
            let input_begin = token * INPUT_COLUMNS + group * GROUP;
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
                Qwen35Nvfp4GdnInputQualificationError::Mismatch(error.to_string())
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

fn make_weights(rows: usize, represented_rows: usize, seed: usize) -> (Vec<u8>, Vec<u8>) {
    const BASE: [u8; 8] = [0xf7, 0xd5, 0xb3, 0x70, 0x5f, 0x3d, 0x0b, 0xf7];
    const SPARSE: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
    const SCALE_CODES: [u8; 4] = [0x38, 0x01, 0x40, 0x01];
    let negative = BASE.map(|byte| byte ^ 0x88);
    let mut codes = vec![0u8; rows * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; rows * GROUPS_PER_ROW];

    for row in 0..represented_rows {
        let base_is_base = (row + seed) & 1 == 0;
        let base = if base_is_base { &BASE } else { &negative };
        let exceptional = if base_is_base { &SPARSE } else { &BASE };
        let exceptional_group = exceptional_group(row, seed);
        for group in 0..GROUPS_PER_ROW {
            let begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
            let pattern = if group == exceptional_group {
                exceptional
            } else {
                base
            };
            codes[begin..begin + GROUP / 2].copy_from_slice(pattern);
            let scale_index = if group == exceptional_group {
                (row + seed + 1) & 3
            } else {
                (row + seed) & 3
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
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    verify_scratch(rows, schedule, fixture, observed)?;
    verify_plane(
        "QKV/Z",
        rows,
        PROJECTED_ROWS,
        PROJECTED_ROWS,
        &fixture.projected_weight_codes,
        &fixture.projected_weight_scales,
        PROJECTED_WEIGHT_SCALE_DIVISOR,
        PROJECTED_SEED,
        schedule,
        fixture,
        &observed.projected_output,
        report,
    )?;
    verify_plane(
        "A/B control",
        rows,
        CONTROL_ROWS,
        PADDED_CONTROL_ROWS,
        &fixture.control_weight_codes,
        &fixture.control_weight_scales,
        CONTROL_WEIGHT_SCALE_DIVISOR,
        CONTROL_SEED,
        schedule,
        fixture,
        &observed.control_output,
        report,
    )?;
    verify_control_padding(rows, &observed.control_output, report)?;
    verify_inactive(rows, schedule, observed)?;
    if schedule == Schedule::W4a4 {
        report.activation_codes += rows * CODE_BYTES_PER_ROW;
        report.activation_scales += rows * GROUPS_PER_ROW;
    }
    report.inactive_values += inactive_values(rows, schedule);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_plane(
    role: &str,
    tokens: usize,
    rows: usize,
    stride: usize,
    weight_codes: &[u8],
    weight_scales: &[u8],
    divisor: f32,
    seed: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &[u16],
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for token in 0..tokens {
        for row in 0..rows {
            let expected = dot_oracle(
                token,
                row,
                weight_codes,
                weight_scales,
                divisor,
                seed,
                schedule,
                fixture,
            )?;
            let index = token * stride + row;
            let actual = f64::from(bf16_to_f32(observed[index]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                    "{} {role} token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}",
                    route_name(tokens)
                )));
            }
        }
    }
    report.output_values += tokens * rows;

    Ok(())
}

fn verify_control_padding(
    rows: usize,
    observed: &[u16],
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for token in 0..rows {
        let begin = token * PADDED_CONTROL_ROWS + CONTROL_ROWS;
        let end = (token + 1) * PADDED_CONTROL_ROWS;
        if let Some(relative) = observed[begin..end].iter().position(|&value| value != 0) {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "{} padded control output {} is {:#06x}, expected zero",
                route_name(rows),
                begin + relative,
                observed[begin + relative]
            )));
        }
    }
    report.padded_output_values += rows * (PADDED_CONTROL_ROWS - CONTROL_ROWS);

    Ok(())
}

fn verify_replay(
    rows: usize,
    schedule: Schedule,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for (name, actual, expected) in [
        (
            "activation code",
            replay.activation_codes.as_slice(),
            eager.activation_codes.as_slice(),
        ),
        (
            "activation scale",
            replay.activation_scales.as_slice(),
            eager.activation_scales.as_slice(),
        ),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "{} graph {name} {index} differs: replay={:#04x}, eager={:#04x}",
                route_name(rows),
                actual[index],
                expected[index]
            )));
        }
    }
    for (name, actual, expected) in [
        (
            "projected output",
            replay.projected_output.as_slice(),
            eager.projected_output.as_slice(),
        ),
        (
            "control output",
            replay.control_output.as_slice(),
            eager.control_output.as_slice(),
        ),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "{} graph {name} {index} differs: replay={:#06x}, eager={:#06x}",
                route_name(rows),
                actual[index],
                expected[index]
            )));
        }
    }

    verify_inactive(rows, schedule, replay)?;
    report.graph_replay_values += rows * (PROJECTED_ROWS + PADDED_CONTROL_ROWS);
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
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
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
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
            "{} activation code {index}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_codes[index],
            fixture.activation_codes[index]
        )));
    }
    if let Some(index) = observed.activation_scales[..active_scales]
        .iter()
        .zip(&fixture.activation_scales[..active_scales])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
            "{} activation scale {index}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_scales[index],
            fixture.activation_scales[index]
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    schedule: Schedule,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
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
    for (name, begin, relative) in [
        (
            "activation code",
            code_begin,
            observed.activation_codes[code_begin..]
                .iter()
                .position(|&value| value != BYTE_SENTINEL),
        ),
        (
            "activation scale",
            scale_begin,
            observed.activation_scales[scale_begin..]
                .iter()
                .position(|&value| value != BYTE_SENTINEL),
        ),
    ] {
        if let Some(relative) = relative {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "{} modified inactive {name} {}",
                route_name(rows),
                begin + relative
            )));
        }
    }

    for (role, begin, plane) in [
        (
            "QKV/Z",
            rows * PROJECTED_ROWS,
            observed.projected_output.as_slice(),
        ),
        (
            "A/B control",
            rows * PADDED_CONTROL_ROWS,
            observed.control_output.as_slice(),
        ),
    ] {
        if let Some(relative) = plane[begin..]
            .iter()
            .position(|&value| value != BF16_SENTINEL)
        {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "{} modified inactive {role} output {}",
                route_name(rows),
                begin + relative
            )));
        }
    }

    Ok(())
}

fn inactive_values(rows: usize, schedule: Schedule) -> usize {
    let inactive_codes = if schedule == Schedule::W4a4 {
        (MAX_ROWS - rows) * CODE_BYTES_PER_ROW
    } else {
        MAX_ROWS * CODE_BYTES_PER_ROW
    };
    let inactive_scales = if schedule == Schedule::W4a4 {
        (MAX_ROWS - rows) * GROUPS_PER_ROW
    } else {
        MAX_ROWS * GROUPS_PER_ROW
    };

    inactive_codes + inactive_scales + (MAX_ROWS - rows) * (PROJECTED_ROWS + PADDED_CONTROL_ROWS)
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
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let projected_codes = arena.copy_to_host(stream, regions.projected_weight_codes)?;
    let projected_scales = arena.copy_to_host(stream, regions.projected_weight_scales)?;
    let control_codes = arena.copy_to_host(stream, regions.control_weight_codes)?;
    let control_scales = arena.copy_to_host(stream, regions.control_weight_scales)?;
    if input != fixture.input_bf16
        || projected_codes != fixture.projected_weight_codes
        || projected_scales != fixture.projected_weight_scales
        || control_codes != fixture.control_weight_codes
        || control_scales != fixture.control_weight_scales
    {
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(
            "read-only input or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values += input.len()
        + projected_codes.len()
        + projected_scales.len()
        + control_codes.len()
        + control_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Nvfp4GdnInputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    let graphs = EXACT_ROUTES
        .into_iter()
        .map(|rows| CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows)))
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
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dot_oracle(
    token: usize,
    row: usize,
    weight_codes: &[u8],
    weight_scales: &[u8],
    divisor: f32,
    seed: usize,
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4GdnInputQualificationError> {
    let exceptional = exceptional_group(row, seed);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, weight_codes, schedule, fixture)?;
    let exceptional_dot = group_dot(token, row, exceptional, weight_codes, schedule, fixture)?;
    let ordinary_scale = decode_e4m3fn(weight_scales[scale_offset(row, ordinary)])
        .map_err(|error| Qwen35Nvfp4GdnInputQualificationError::Mismatch(error.to_string()))?;
    let exceptional_scale = decode_e4m3fn(weight_scales[scale_offset(row, exceptional)])
        .map_err(|error| Qwen35Nvfp4GdnInputQualificationError::Mismatch(error.to_string()))?;

    Ok(
        ((GROUPS_PER_ROW - 1) as f64 * ordinary_dot * f64::from(ordinary_scale)
            + exceptional_dot * f64::from(exceptional_scale))
            / f64::from(divisor),
    )
}

fn group_dot(
    token: usize,
    row: usize,
    group: usize,
    weight_codes: &[u8],
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4GdnInputQualificationError> {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * INPUT_COLUMNS + group * GROUP;
    let activation_begin = token * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let activation_scale = if schedule == Schedule::W4a4 {
        decode_e4m3fn(fixture.activation_scales[token * GROUPS_PER_ROW + group])
            .map_err(|error| Qwen35Nvfp4GdnInputQualificationError::Mismatch(error.to_string()))?
    } else {
        1.0
    };
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        let activation = match schedule {
            Schedule::A16 => f64::from(fixture.input_f32[input_begin + column]),
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

fn exceptional_group(row: usize, seed: usize) -> usize {
    (row * 17 + row / 128 * 13 + seed) % GROUPS_PER_ROW
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

fn encode_e4m3fn(value: f32) -> Result<u8, Qwen35Nvfp4GdnInputQualificationError> {
    codecs::encode_e4m3fn_scale(value).ok_or_else(|| {
        Qwen35Nvfp4GdnInputQualificationError::Mismatch(
            "Qwen3.5 GDN oracle E4M3 scale is not finite and non-negative".to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_swizzle_and_padding_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();
        let fixture = make_fixture().unwrap();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(127, 255), 32_767);
        assert_eq!(scale_offset(128, 0), 32_768);
        assert_eq!(
            fixture.activation_codes.len(),
            MAX_ROWS * CODE_BYTES_PER_ROW
        );
        assert_eq!(fixture.activation_scales.len(), MAX_ROWS * GROUPS_PER_ROW);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(regions.weight_bytes(), 28_606_464);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 4_521_984);
        assert_eq!(layout.byte_len(), 33_128_448);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
        let report = qualify_qwen35_nvfp4_gdn_input()?;
        let active_rows = EXACT_ROUTES.into_iter().sum::<usize>();
        let prefill_rows = EXACT_ROUTES
            .into_iter()
            .filter(|&rows| schedule(rows) == Schedule::W4a4)
            .sum::<usize>();
        let inactive = EXACT_ROUTES
            .into_iter()
            .map(|rows| inactive_values(rows, schedule(rows)))
            .sum::<usize>();

        assert_eq!(report.activation_codes, prefill_rows * CODE_BYTES_PER_ROW);
        assert_eq!(report.activation_scales, prefill_rows * GROUPS_PER_ROW);
        assert_eq!(
            report.output_values,
            active_rows * (PROJECTED_ROWS + CONTROL_ROWS)
        );
        assert_eq!(
            report.padded_output_values,
            active_rows * (PADDED_CONTROL_ROWS - CONTROL_ROWS)
        );
        assert_eq!(
            report.graph_replay_values,
            active_rows * (PROJECTED_ROWS + PADDED_CONTROL_ROWS)
                + prefill_rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
        assert_eq!(report.inactive_values, 2 * inactive);
        assert_eq!(report.immutable_input_values, 640_876_544);
        assert_eq!(report.arena_bytes, 33_128_448);
        assert_eq!(report.weight_bytes, 28_606_464);
        assert_eq!(report.workspace_bytes, 4_521_984);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
