//! Qwen3.5 represented-value qualification for NVFP4 gate/up SwiGLU.

use crate::device_benchmark;
use crate::nvfp4_swiglu::{
    Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError, bf16_to_f32, decode_e2m1,
    decode_e4m3fn, encode_e2m1, encode_e4m3fn, f32_to_bf16,
};
use crate::target::Qwen35Nvfp4SwiGluOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const HIDDEN: usize = Qwen35_9B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen35_9B::INTERMEDIATE;
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
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Schedule {
    A16,
    W4a4,
}

/// Qualifies all exact Qwen3.5 production routes and crossover candidates.
pub fn qualify_qwen35_nvfp4_swiglu()
-> Result<Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError> {
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
    let op = Qwen35Nvfp4SwiGluOp::new(&context)?;
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

    for batch in 1..=MAX_BATCH {
        let schedule = production_schedule(batch);
        reset_observed(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch, schedule)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify(batch, schedule, &fixture, &eager, &mut report, false)?;

        reset_observed(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &arena, &stream, regions, batch, schedule)
        })?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(batch, schedule, &eager, &replay, &mut report)?;
        require_stable_addresses(&arena, regions, stable_addresses, &format!("B={batch}"))?;

        if batch <= 4 {
            let alternate = if schedule == Schedule::A16 {
                Schedule::W4a4
            } else {
                Schedule::A16
            };
            reset_observed(&arena, &stream, regions)?;
            launch(&op, &arena, &stream, regions, batch, alternate)?;
            let observed = read_observed(&arena, &stream, regions)?;
            verify(batch, alternate, &fixture, &observed, &mut report, true)?;
            require_stable_addresses(
                &arena,
                regions,
                stable_addresses,
                &format!("candidate B={batch}"),
            )?;
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_BATCH * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_BATCH * GROUPS_PER_ROW, ALIGNMENT)?;
    let weight_codes = layout.reserve(GATE_UP_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(GATE_UP_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

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
            "device addresses changed while qualifying Qwen3.5 {route}"
        )));
    }

    Ok(())
}

fn make_fixture() -> Result<Fixture, Nvfp4SwiGluQualificationError> {
    let input_bf16 = (0..MAX_BATCH * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token])
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
    let mut codes = vec![0u8; MAX_BATCH * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; MAX_BATCH * GROUPS_PER_ROW];

    for token in 0..MAX_BATCH {
        for group in 0..GROUPS_PER_ROW {
            let begin = token * HIDDEN + group * GROUP;
            let values = &input[begin..begin + GROUP];
            let maximum = values
                .iter()
                .fold(0.0f32, |current, value| current.max(value.abs()));
            let scale = encode_e4m3fn(INPUT_SCALE_DIVISOR * maximum / 6.0)?;
            scales[token * GROUPS_PER_ROW + group] = scale;
            if scale == 0 {
                continue;
            }

            let represented_scale = decode_e4m3fn(scale)?;
            for pair in 0..GROUP / 2 {
                let low = encode_e2m1(values[2 * pair] * INPUT_SCALE_DIVISOR / represented_scale);
                let high =
                    encode_e2m1(values[2 * pair + 1] * INPUT_SCALE_DIVISOR / represented_scale);
                codes[token * CODE_BYTES_PER_ROW + group * (GROUP / 2) + pair] = low | (high << 4);
            }
        }
    }

    Ok((codes, scales))
}

fn production_schedule(batch: usize) -> Schedule {
    if batch == 2 {
        Schedule::A16
    } else {
        Schedule::W4a4
    }
}

fn reset_observed(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch(
    op: &Qwen35Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
    schedule: Schedule,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let activation_codes = arena.address(regions.activation_codes)?;
    let activation_scales = arena.address(regions.activation_scales)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: all regions are aligned, disjoint, context-local, and sized for
    // the documented maximum-batch Qwen3.5 operation.
    unsafe {
        match schedule {
            Schedule::A16 => op.launch_a16(
                stream,
                batch,
                input,
                weight_codes,
                weight_scales,
                WEIGHT_SCALE_DIVISOR,
                output,
            ),
            Schedule::W4a4 => op.launch_w4a4(
                stream,
                batch,
                input,
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                output,
            ),
        }
    }
}

fn read_observed(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify(
    batch: usize,
    schedule: Schedule,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Nvfp4SwiGluQualification,
    comparison: bool,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let active_codes = if schedule == Schedule::W4a4 {
        batch * CODE_BYTES_PER_ROW
    } else {
        0
    };
    let active_scales = if schedule == Schedule::W4a4 {
        batch * GROUPS_PER_ROW
    } else {
        0
    };
    if observed.activation_codes[..active_codes] != fixture.activation_codes[..active_codes] {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 {schedule:?} B={batch} activation codes differ from the oracle"
        )));
    }
    if observed.activation_scales[..active_scales] != fixture.activation_scales[..active_scales] {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 {schedule:?} B={batch} activation scales differ from the oracle"
        )));
    }

    for token in 0..batch {
        for row in 0..OUTPUT_ROWS {
            let expected = swiglu_oracle(token, row, schedule, fixture)?;
            let index = token * OUTPUT_ROWS + row;
            let actual = f64::from(bf16_to_f32(observed.output[index]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
                    "Qwen3.5 {schedule:?} B={batch} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(batch, schedule, observed)?;
    let outputs = batch * OUTPUT_ROWS;
    if comparison {
        if schedule == Schedule::A16 {
            report.a16_comparison_values += outputs;
        } else {
            report.candidate_comparison_values += outputs;
        }
    } else {
        report.output_values += outputs;
        if schedule == Schedule::W4a4 {
            report.activation_codes += active_codes;
            report.activation_scales += active_scales;
        }
    }
    report.inactive_values += inactive_values(batch, schedule);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    schedule: Schedule,
    observed: &Observed,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let code_begin = if schedule == Schedule::W4a4 {
        batch * CODE_BYTES_PER_ROW
    } else {
        0
    };
    let scale_begin = if schedule == Schedule::W4a4 {
        batch * GROUPS_PER_ROW
    } else {
        0
    };
    if let Some(index) = observed.activation_codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 B={batch} modified inactive activation code {}",
            code_begin + index
        )));
    }
    if let Some(index) = observed.activation_scales[scale_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 B={batch} modified inactive activation scale {}",
            scale_begin + index
        )));
    }
    let output_begin = batch * OUTPUT_ROWS;
    if let Some(index) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 B={batch} modified inactive output {}",
            output_begin + index
        )));
    }

    Ok(())
}

fn verify_replay(
    batch: usize,
    schedule: Schedule,
    eager: &Observed,
    replay: &Observed,
    report: &mut Nvfp4SwiGluQualification,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    if eager.activation_codes != replay.activation_codes
        || eager.activation_scales != replay.activation_scales
        || eager.output != replay.output
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 {schedule:?} B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, schedule, replay)?;
    report.graph_replay_values += batch * OUTPUT_ROWS;
    if schedule == Schedule::W4a4 {
        report.graph_replay_values += batch * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    }
    report.inactive_values += inactive_values(batch, schedule);

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Nvfp4SwiGluQualification,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if input != fixture.input_bf16
        || weight_codes != fixture.weight_codes
        || weight_scales != fixture.weight_scales
    {
        return Err(Nvfp4SwiGluQualificationError::Mismatch(
            "Qwen3.5 read-only input or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = input.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Nvfp4SwiGluOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Nvfp4SwiGluQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| {
            CudaGraph::capture(stream, || {
                launch(
                    op,
                    arena,
                    stream,
                    regions,
                    batch,
                    production_schedule(batch),
                )
            })
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
        return Err(Nvfp4SwiGluQualificationError::Mismatch(format!(
            "Qwen3.5 device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize, schedule: Schedule) -> usize {
    let inactive_codes = if schedule == Schedule::W4a4 {
        (MAX_BATCH - batch) * CODE_BYTES_PER_ROW
    } else {
        MAX_BATCH * CODE_BYTES_PER_ROW
    };
    let inactive_scales = if schedule == Schedule::W4a4 {
        (MAX_BATCH - batch) * GROUPS_PER_ROW
    } else {
        MAX_BATCH * GROUPS_PER_ROW
    };

    inactive_codes + inactive_scales + (MAX_BATCH - batch) * OUTPUT_ROWS
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
    let weight_begin = row * CODE_BYTES_PER_ROW;
    let activation_begin = token * CODE_BYTES_PER_ROW;
    let mut group_sum = 0.0f64;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_and_swizzle_match_exact_qwen35_geometry() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(32, 0), 4);
        assert_eq!(scale_offset(127, 255), 32_767);
        assert_eq!(scale_offset(128, 0), 32_768);
        assert_eq!(layout.byte_len(), 56_903_680);
        assert_eq!(regions.weight_bytes(), 56_623_104);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 280_576);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_and_candidates_match_independent_oracles_and_graph_replay()
    -> Result<(), Nvfp4SwiGluQualificationError> {
        let report = qualify_qwen35_nvfp4_swiglu()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let w4a4_rows = 1 + 3 + 4 + 5 + 6 + 7 + 8;

        assert_eq!(report.activation_codes, w4a4_rows * CODE_BYTES_PER_ROW);
        assert_eq!(report.activation_scales, w4a4_rows * GROUPS_PER_ROW);
        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.a16_comparison_values, (1 + 3 + 4) * OUTPUT_ROWS);
        assert_eq!(report.candidate_comparison_values, 2 * OUTPUT_ROWS);
        assert_eq!(
            report.graph_replay_values,
            active_rows * OUTPUT_ROWS + w4a4_rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
        assert!(report.inactive_values > 0);
        assert_eq!(report.arena_bytes, 56_903_680);
        assert_eq!(report.weight_bytes, 56_623_104);
        assert_eq!(report.workspace_bytes, 280_576);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
