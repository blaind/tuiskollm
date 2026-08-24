//! Qwen3.5 represented-value qualification for NVFP4 down projection.

use crate::device_benchmark;
use crate::nvfp4_down::{
    Nvfp4DownQualification, Nvfp4DownQualificationError, bf16_to_f32, decode_e2m1, decode_e4m3fn,
    f32_to_bf16,
};
use crate::target::Qwen35Nvfp4DownOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen35_9B::HIDDEN;
pub(crate) const INPUT_COLUMNS: usize = Qwen35_9B::INTERMEDIATE;
pub(crate) const OUTPUT_ROWS: usize = HIDDEN;
const GROUP: usize = 16;
pub(crate) const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
pub(crate) const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
pub(crate) const WEIGHT_SCALE_DIVISOR: f32 = 0.125;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) weight_scales: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len() + self.weight_bytes() + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    pub(crate) weight_codes: Vec<u8>,
    pub(crate) weight_scales: Vec<u8>,
}

/// Qualifies eager and captured Qwen3.5 A16 down at exact `B=1..=8`.
pub fn qualify_qwen35_nvfp4_down() -> Result<Nvfp4DownQualification, Nvfp4DownQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen35Nvfp4DownOp::new(&context)?;
    let fixture = make_fixture();
    arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Nvfp4DownQualification {
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
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Nvfp4DownQualificationError::Mismatch(format!(
                "Qwen3.5 device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * INPUT_COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 4]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &Qwen35Nvfp4DownOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the disjoint arena regions are aligned, context-local, and own
    // every maximum-batch extent documented by the Qwen3.5 operation.
    unsafe {
        op.launch(
            stream,
            batch,
            input,
            weight_codes,
            weight_scales,
            WEIGHT_SCALE_DIVISOR,
            output,
        )
    }
}

pub(crate) fn make_fixture() -> Fixture {
    let input_bf16 = (0..MAX_BATCH * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token])
        })
        .collect::<Vec<_>>();
    let input_f32 = input_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let (weight_codes, weight_scales) = make_weights();

    Fixture {
        input_bf16,
        input_f32,
        weight_codes,
        weight_scales,
    }
}

fn make_weights() -> (Vec<u8>, Vec<u8>) {
    const BASE: [u8; 8] = [0xf7, 0xd5, 0xb3, 0x70, 0x5f, 0x3d, 0x0b, 0xf7];
    const SPARSE: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
    const SCALE_CODES: [u8; 4] = [0x38, 0x01, 0x40, 0x01];
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
    observed: &[u16],
    report: &mut Nvfp4DownQualification,
) -> Result<(), Nvfp4DownQualificationError> {
    for token in 0..batch {
        for row in 0..OUTPUT_ROWS {
            let expected = dot_oracle(token, row, fixture)?;
            let index = token * OUTPUT_ROWS + row;
            let actual = f64::from(bf16_to_f32(observed[index]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Nvfp4DownQualificationError::Mismatch(format!(
                    "Qwen3.5 B={batch} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(batch, observed)?;
    report.output_values += batch * OUTPUT_ROWS;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Nvfp4DownQualification,
) -> Result<(), Nvfp4DownQualificationError> {
    if eager != replay {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "Qwen3.5 B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * OUTPUT_ROWS;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_inactive(batch: usize, observed: &[u16]) -> Result<(), Nvfp4DownQualificationError> {
    let begin = batch * OUTPUT_ROWS;
    if let Some(relative) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "Qwen3.5 B={batch} modified inactive output {}",
            begin + relative
        )));
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Nvfp4DownQualification,
) -> Result<(), Nvfp4DownQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if input != fixture.input_bf16
        || weight_codes != fixture.weight_codes
        || weight_scales != fixture.weight_scales
    {
        return Err(Nvfp4DownQualificationError::Mismatch(
            "Qwen3.5 read-only input or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = input.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Nvfp4DownOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Nvfp4DownQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graphs[batch - 1].launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "Qwen3.5 device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn dot_oracle(
    token: usize,
    row: usize,
    fixture: &Fixture,
) -> Result<f64, Nvfp4DownQualificationError> {
    let exceptional = exceptional_group(row);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, fixture);
    let exceptional_dot = group_dot(token, row, exceptional, fixture);
    let ordinary_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, ordinary)])?;
    let exceptional_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, exceptional)])?;

    Ok(
        ((GROUPS_PER_ROW - 1) as f64 * ordinary_dot * f64::from(ordinary_scale)
            + exceptional_dot * f64::from(exceptional_scale))
            / f64::from(WEIGHT_SCALE_DIVISOR),
    )
}

fn group_dot(token: usize, row: usize, group: usize, fixture: &Fixture) -> f64 {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * INPUT_COLUMNS + group * GROUP;
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = fixture.weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        sum += f64::from(fixture.input_f32[input_begin + column]) * f64::from(decode_e2m1(code));
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
    fn arena_swizzle_and_fixture_match_exact_qwen35_geometry() {
        let (layout, regions) = layout().unwrap();
        let fixture = make_fixture();
        let exceptional = exceptional_group(127);
        let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
        let exceptional_begin = 127 * CODE_BYTES_PER_ROW + exceptional * (GROUP / 2);
        let ordinary_begin = 127 * CODE_BYTES_PER_ROW + ordinary * (GROUP / 2);

        assert_ne!(
            &fixture.weight_codes[exceptional_begin..exceptional_begin + GROUP / 2],
            &fixture.weight_codes[ordinary_begin..ordinary_begin + GROUP / 2]
        );
        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(32, 0), 4);
        assert_eq!(scale_offset(127, 767), 98_303);
        assert_eq!(scale_offset(128, 0), 98_304);
        assert_eq!(layout.byte_len(), 28_573_696);
        assert_eq!(regions.weight_bytes(), 28_311_552);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 262_144);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Nvfp4DownQualificationError> {
        let report = qualify_qwen35_nvfp4_down()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * inactive_rows * OUTPUT_ROWS);
        assert_eq!(report.immutable_input_values, 28_409_856);
        assert_eq!(report.arena_bytes, 28_573_696);
        assert_eq!(report.weight_bytes, 28_311_552);
        assert_eq!(report.workspace_bytes, 262_144);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
