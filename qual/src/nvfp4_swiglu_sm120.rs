//! Independent represented-value qualification for NVFP4 gate/up SwiGLU.

use crate::oracles::codecs;
pub(crate) use crate::oracles::codecs::{bf16_to_f32, decode_e2m1, encode_e2m1, f32_to_bf16};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::Nvfp4SwiGluOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * OUTPUT_ROWS;
const GROUP: usize = 16;
const GROUPS_PER_ROW: usize = HIDDEN / GROUP;
const CODE_BYTES_PER_ROW: usize = HIDDEN / 2;
const INPUT_SCALE_DIVISOR: f32 = 3.0;
const WEIGHT_SCALE_DIVISOR: f32 = 0.125;
const BYTE_SENTINEL: u8 = 0xa5;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; 8] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact NVFP4 SwiGLU qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Nvfp4SwiGluQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("NVFP4 SwiGLU qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact decode and prefill route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nvfp4SwiGluQualification {
    /// Dynamically generated E2M1 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamically generated E4M3 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// Production-route BF16 outputs compared with the FP64 oracle.
    pub output_values: usize,
    /// Retained A16 B=1 comparison outputs checked independently.
    pub a16_comparison_values: usize,
    /// Other schedule-candidate outputs checked before route selection.
    pub candidate_comparison_values: usize,
    /// Active codes, scales, and outputs reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside active route extents.
    pub inactive_values: usize,
    /// Read-only represented input and weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact packed-weight and block-scale bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input, scratch, and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from a represented-value oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<u8>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u8>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_bytes()
            + self.output.byte_len()
    }
}

struct Fixture {
    input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u8>,
}

struct Observed {
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    output: Vec<u16>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Schedule {
    A16,
    W4a4,
}

/// Qualifies eager and captured NVFP4 SwiGLU at every admitted row count.
pub fn qualify_nvfp4_swiglu() -> Result<Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Nvfp4SwiGluOp::new(&context)?;
    let fixture = make_fixture()?;

    arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Nvfp4SwiGluQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        a16_comparison_values: 0,
        candidate_comparison_values: 0,
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
        let schedule = production_schedule(rows);
        reset_observed(&arena, &stream, regions)?;
        launch_production(&op, &arena, &stream, regions, rows)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify_eager(rows, schedule, &fixture, &eager, &mut report, false)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        reset_observed(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch_production(&op, &arena, &stream, regions, rows)
        })?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(rows, schedule, &eager, &replay, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        require_stable_addresses(&arena, regions, stable_addresses, &route_name(rows))?;
    }

    qualify_a16_b1(
        &op,
        &arena,
        &stream,
        regions,
        &fixture,
        stable_addresses,
        &mut report,
    )?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * HIDDEN, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let weight_codes = layout.reserve(GATE_UP_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(GATE_UP_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.activation_scales)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn require_stable_addresses(
    arena: &DeviceArena,
    regions: Regions,
    expected: [usize; 6],
    route: &str,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    if addresses(arena, regions)? != expected {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "device addresses changed while qualifying {route}"
        )));
    }

    Ok(())
}

fn make_fixture() -> Result<Fixture, Nvfp4SwiGluQualificationError> {
    let input_bf16 = (0..MAX_ROWS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token & 7])
        })
        .collect::<Vec<_>>();
    let input_f32 = input_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let (activation_codes, activation_scales) = quantize_oracle(&input_f32)?;
    let (weight_codes, weight_scales) = make_weights();

    Ok(Fixture {
        input_bf16,
        input_f32,
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
    })
}

fn make_weights() -> (Vec<u8>, Vec<u8>) {
    const BASE: [u8; 8] = [0xf7, 0xd5, 0xb3, 0x70, 0x5f, 0x3d, 0x0b, 0xf7];
    const SPARSE: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
    const SCALE_CODES: [u8; 4] = [0x38, 0x01, 0x40, 0x01];
    let negative = BASE.map(|byte| byte ^ 0x88);
    let mut codes = vec![0u8; GATE_UP_ROWS * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; GATE_UP_ROWS * GROUPS_PER_ROW];

    for row in 0..GATE_UP_ROWS {
        let pattern = if row < OUTPUT_ROWS && row & 1 != 0 {
            &SPARSE
        } else if row >= OUTPUT_ROWS && row & 1 != 0 {
            &negative
        } else {
            &BASE
        };

        for group in 0..GROUPS_PER_ROW {
            let begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
            codes[begin..begin + GROUP / 2].copy_from_slice(pattern);
            scales[scale_offset(row, group)] = SCALE_CODES[row & 3];
        }
    }

    (codes, scales)
}

fn quantize_oracle(input: &[f32]) -> Result<(Vec<u8>, Vec<u8>), Nvfp4SwiGluQualificationError> {
    let tokens = input.len() / HIDDEN;
    let mut codes = vec![0u8; tokens * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; tokens * GROUPS_PER_ROW];

    for token in 0..tokens {
        for group in 0..GROUPS_PER_ROW {
            let input_begin = token * HIDDEN + group * GROUP;
            let values = &input[input_begin..input_begin + GROUP];
            let maximum = values
                .iter()
                .fold(0.0f32, |current, value| current.max(value.abs()));
            let scale = encode_e4m3fn(INPUT_SCALE_DIVISOR * maximum / 6.0)?;
            scales[token * GROUPS_PER_ROW + group] = scale;
            if scale == 0 {
                continue;
            }

            let decoded_scale = decode_e4m3fn(scale)?;
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

fn reset_observed(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch_production(
    op: &Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let activation_codes = arena.address(regions.activation_codes)?;
    let activation_scales = arena.address(regions.activation_scales)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the arena regions are aligned, disjoint, context-local, and own
    // every maximum-row extent documented by the production operation.
    unsafe {
        op.launch(
            stream,
            rows,
            input,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            INPUT_SCALE_DIVISOR,
            WEIGHT_SCALE_DIVISOR,
            output,
        )
    }
}

fn launch_a16_b1(
    op: &Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the same arena contract as the production launch applies to the
    // retained A16 subset, with one active row.
    unsafe {
        op.launch_a16(
            stream,
            1,
            input,
            weight_codes,
            weight_scales,
            WEIGHT_SCALE_DIVISOR,
            output,
        )
    }
}

fn read_observed(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Nvfp4SwiGluQualification,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    if let Some(index) = input
        .iter()
        .zip(&fixture.input_bf16)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "read-only input {index} changed: device={:#06x}, source={:#06x}",
            input[index], fixture.input_bf16[index]
        )));
    }
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    if let Some(index) = weight_codes
        .iter()
        .zip(&fixture.weight_codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "read-only weight code {index} changed: device={:#04x}, source={:#04x}",
            weight_codes[index], fixture.weight_codes[index]
        )));
    }
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if let Some(index) = weight_scales
        .iter()
        .zip(&fixture.weight_scales)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "read-only weight scale {index} changed: device={:#04x}, source={:#04x}",
            weight_scales[index], fixture.weight_scales[index]
        )));
    }
    report.immutable_input_values += input.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn production_schedule(rows: usize) -> Schedule {
    if rows == 1 || rows >= 5 {
        Schedule::W4a4
    } else {
        Schedule::A16
    }
}

fn verify_eager(
    rows: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Nvfp4SwiGluQualification,
    comparison: bool,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    verify_scratch(rows, schedule, fixture, observed)?;

    for token in 0..rows {
        for row in 0..OUTPUT_ROWS {
            let expected = swiglu_oracle(token, row, schedule, fixture)?;
            let index = token * OUTPUT_ROWS + row;
            let actual = f64::from(bf16_to_f32(observed.output[index]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);

            if absolute_error > tolerance {
                let name = if comparison {
                    "A16 comparison"
                } else {
                    "production"
                };
                return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
                    "{name} {} output at token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}",
                    route_name(rows)
                )));
            }
        }
    }

    verify_inactive(rows, schedule, observed)?;
    let outputs = rows * OUTPUT_ROWS;
    if comparison {
        report.a16_comparison_values += outputs;
    } else {
        report.output_values += outputs;
        if schedule == Schedule::W4a4 {
            report.activation_codes += rows * CODE_BYTES_PER_ROW;
            report.activation_scales += rows * GROUPS_PER_ROW;
        }
    }
    report.inactive_values += inactive_values(rows, schedule);

    Ok(())
}

fn verify_scratch(
    rows: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), Nvfp4SwiGluQualificationError> {
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

    if let Some(relative) = observed.activation_codes[..active_codes]
        .iter()
        .zip(&fixture.activation_codes[..active_codes])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "{} activation code {relative}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_codes[relative],
            fixture.activation_codes[relative]
        )));
    }
    if let Some(relative) = observed.activation_scales[..active_scales]
        .iter()
        .zip(&fixture.activation_scales[..active_scales])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "{} activation scale {relative}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_scales[relative],
            fixture.activation_scales[relative]
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    schedule: Schedule,
    observed: &Observed,
) -> Result<(), Nvfp4SwiGluQualificationError> {
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

    for (name, relative) in [
        (
            "activation code",
            observed.activation_codes[code_begin..]
                .iter()
                .position(|&value| value != BYTE_SENTINEL),
        ),
        (
            "activation scale",
            observed.activation_scales[scale_begin..]
                .iter()
                .position(|&value| value != BYTE_SENTINEL),
        ),
    ] {
        if let Some(relative) = relative {
            let begin = if name == "activation code" {
                code_begin
            } else {
                scale_begin
            };
            return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
                "{} modified inactive {name} {}",
                route_name(rows),
                begin + relative
            )));
        }
    }

    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "{} modified inactive output {}",
            route_name(rows),
            output_begin + relative
        )));
    }

    Ok(())
}

fn verify_replay(
    rows: usize,
    schedule: Schedule,
    eager: &Observed,
    replay: &Observed,
    report: &mut Nvfp4SwiGluQualification,
) -> Result<(), Nvfp4SwiGluQualificationError> {
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
            return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
                "{} graph {name} {index} differs: replay={:#04x}, eager={:#04x}",
                route_name(rows),
                actual[index],
                expected[index]
            )));
        }
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "{} graph output {index} differs: replay={:#06x}, eager={:#06x}",
            route_name(rows),
            replay.output[index],
            eager.output[index]
        )));
    }

    verify_inactive(rows, schedule, replay)?;
    report.graph_replay_values += rows * OUTPUT_ROWS;
    if schedule == Schedule::W4a4 {
        report.graph_replay_values += rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    }
    report.inactive_values += inactive_values(rows, schedule);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qualify_a16_b1(
    op: &Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    stable_addresses: [usize; 6],
    report: &mut Nvfp4SwiGluQualification,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    reset_observed(arena, stream, regions)?;
    launch_a16_b1(op, arena, stream, regions)?;
    let eager = read_observed(arena, stream, regions)?;
    verify_eager(1, Schedule::A16, fixture, &eager, report, true)?;
    verify_immutable(arena, stream, regions, fixture, report)?;

    reset_observed(arena, stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    let graph = CudaGraph::capture(stream, || launch_a16_b1(op, arena, stream, regions))?;
    // SAFETY: every allocation this graph captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    unsafe { graph.launch(stream) }?;
    // SAFETY: every allocation this graph captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    unsafe { graph.launch(stream) }?;
    let replay = read_observed(arena, stream, regions)?;
    verify_replay(1, Schedule::A16, &eager, &replay, report)?;
    verify_immutable(arena, stream, regions, fixture, report)?;
    require_stable_addresses(arena, regions, stable_addresses, "A16 B=1")
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let graphs = EXACT_ROUTES
        .into_iter()
        .map(|rows| {
            CudaGraph::capture(stream, || {
                launch_production(op, arena, stream, regions, rows)
            })
        })
        .collect::<GpuResult<Vec<_>>>()?;
    let a16 = CudaGraph::capture(stream, || launch_a16_b1(op, arena, stream, regions))?;
    for graph in &graphs {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(stream) }?;
    }
    // SAFETY: every allocation this graph captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    unsafe { a16.launch(stream) }?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graph.launch(stream) }?;
        }
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { a16.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
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

    inactive_codes + inactive_scales + (MAX_ROWS - rows) * OUTPUT_ROWS
}

fn route_name(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn swiglu_oracle(
    token: usize,
    row: usize,
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Nvfp4SwiGluQualificationError> {
    let mut gate = dot_oracle(token, row, schedule, fixture)?;
    let mut up = dot_oracle(token, row + OUTPUT_ROWS, schedule, fixture)?;
    if schedule == Schedule::W4a4 {
        gate = f64::from(bf16_to_f32(f32_to_bf16(gate as f32)));
        up = f64::from(bf16_to_f32(f32_to_bf16(up as f32)));
    }

    Ok(gate / (1.0 + (-gate).exp()) * up)
}

fn dot_oracle(
    token: usize,
    row: usize,
    schedule: Schedule,
    fixture: &Fixture,
) -> Result<f64, Nvfp4SwiGluQualificationError> {
    let mut group_sum = 0.0f64;
    let weight_begin = row * CODE_BYTES_PER_ROW;
    let activation_begin = token * CODE_BYTES_PER_ROW;

    for column in 0..GROUP {
        let weight_packed = fixture.weight_codes[weight_begin + column / 2];
        let weight_code = if column & 1 == 0 {
            weight_packed & 0x0f
        } else {
            weight_packed >> 4
        };
        let activation = match schedule {
            Schedule::A16 => f64::from(fixture.input_f32[token * HIDDEN + column]),
            Schedule::W4a4 => {
                let packed = fixture.activation_codes[activation_begin + column / 2];
                let code = if column & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale = decode_e4m3fn(fixture.activation_scales[token * GROUPS_PER_ROW])?;

                f64::from(decode_e2m1(code) * scale / INPUT_SCALE_DIVISOR)
            }
        };

        group_sum += activation * f64::from(decode_e2m1(weight_code));
    }

    let scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, 0)])?;
    Ok(group_sum * GROUPS_PER_ROW as f64 * f64::from(scale / WEIGHT_SCALE_DIVISOR))
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

pub(crate) fn encode_e4m3fn(value: f32) -> Result<u8, Nvfp4SwiGluQualificationError> {
    codecs::encode_e4m3fn_scale(value).ok_or_else(|| {
        Nvfp4SwiGluQualificationError::Mismatch(
            "oracle E4M3 scale is not finite and non-negative".to_string(),
        )
    })
}

pub(crate) fn decode_e4m3fn(word: u8) -> Result<f32, Nvfp4SwiGluQualificationError> {
    codecs::decode_e4m3fn(word).ok_or_else(|| {
        Nvfp4SwiGluQualificationError::Mismatch("oracle encountered an E4M3FN NaN".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_BYTES_PER_ROW, EXACT_ROUTES, GATE_UP_ROWS, GROUPS_PER_ROW, MAX_ROWS,
        Nvfp4SwiGluQualificationError, OUTPUT_ROWS, decode_e2m1, decode_e4m3fn, encode_e2m1,
        encode_e4m3fn, layout, qualify_nvfp4_swiglu, scale_offset,
    };

    #[test]
    fn independent_codecs_and_swizzle_are_pinned() {
        assert_eq!(decode_e2m1(0x07), 6.0);
        assert_eq!(decode_e2m1(0x0f), -6.0);
        assert_eq!(encode_e2m1(1.25), 0x02);
        assert_eq!(encode_e4m3fn(1.0).unwrap(), 0x38);
        assert_eq!(decode_e4m3fn(0x40).unwrap(), 2.0);
        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(32, 0), 4);
        assert_eq!(scale_offset(127, 319), 40_959);
        assert_eq!(scale_offset(128, 0), 40_960);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Nvfp4SwiGluQualificationError> {
        let report = qualify_nvfp4_swiglu()?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        let w4a4_rows = active_rows - (2 + 3 + 4);

        assert_eq!(report.activation_codes, w4a4_rows * CODE_BYTES_PER_ROW);
        assert_eq!(report.activation_scales, w4a4_rows * GROUPS_PER_ROW);
        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.a16_comparison_values, OUTPUT_ROWS);
        assert_eq!(report.candidate_comparison_values, 0);
        assert_eq!(
            report.graph_replay_values,
            (active_rows + 1) * OUTPUT_ROWS + w4a4_rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
        assert!(report.inactive_values > 0);
        assert_eq!(
            report.immutable_input_values,
            2 * (EXACT_ROUTES.len() + 1)
                * (MAX_ROWS * super::HIDDEN + GATE_UP_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW))
        );
        assert_eq!(report.arena_bytes, 149_356_544);
        assert_eq!(report.weight_bytes, 100_270_080);
        assert_eq!(report.workspace_bytes, 49_086_464);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }

    #[test]
    fn arena_accounting_exposes_every_owned_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(layout.byte_len(), 149_356_544);
        assert_eq!(regions.weight_bytes(), 100_270_080);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 49_086_464);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }
}
