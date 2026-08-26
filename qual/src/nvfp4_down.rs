//! Independent represented-value qualification for portable NVFP4 A16 down projection.

use crate::oracles::codecs::{self, bf16_to_f32, decode_e2m1, f32_to_bf16};
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, Nvfp4DownOp};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const INPUT_COLUMNS: usize = Qwen38_27B::INTERMEDIATE;
const OUTPUT_ROWS: usize = HIDDEN;
const GROUP: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const WEIGHT_SCALE_DIVISOR: f32 = 0.125;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of the exact-target NVFP4 down projection qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Nvfp4DownQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("NVFP4 down projection qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nvfp4DownQualification {
    /// BF16 outputs compared with the independent FP64 represented-value oracle.
    pub output_values: usize,
    /// Active BF16 outputs reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active route extent.
    pub inactive_values: usize,
    /// Largest absolute difference from the independent oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u8>,
    output: ArenaRegion<u16>,
}

struct Fixture {
    input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u8>,
}

/// Qualifies eager and captured NVFP4 A16 down projection at every exact `B=1..=8`.
pub fn qualify_nvfp4_down() -> Result<Nvfp4DownQualification, Nvfp4DownQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != EXPECTED_COMPUTE_CAPABILITY {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected {}.{}",
            capability.0,
            capability.1,
            EXPECTED_COMPUTE_CAPABILITY.0,
            EXPECTED_COMPUTE_CAPABILITY.1,
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Nvfp4DownOp::new(&context)?;
    let fixture = make_fixture();

    arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Nvfp4DownQualification {
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
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
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
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
    op: &Nvfp4DownOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the disjoint arena regions are aligned, context-local, and own
    // every maximum-batch extent documented by the production operation.
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

fn make_fixture() -> Fixture {
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
            let expected = down_oracle(token, row, fixture)?;
            let index = token * OUTPUT_ROWS + row;
            let actual = f64::from(bf16_to_f32(observed[index]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);

            if absolute_error > tolerance {
                return Err(Nvfp4DownQualificationError::Mismatch(format!(
                    "B={batch} output at token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}",
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
    if let Some(index) = replay
        .iter()
        .zip(eager)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4DownQualificationError::Mismatch(format!(
            "B={batch} graph output {index} differs: replay={:#06x}, eager={:#06x}",
            replay[index], eager[index]
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
            "B={batch} modified inactive output {}",
            begin + relative
        )));
    }

    Ok(())
}

fn down_oracle(
    token: usize,
    row: usize,
    fixture: &Fixture,
) -> Result<f64, Nvfp4DownQualificationError> {
    dot_oracle(token, row, fixture)
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
        let activation = f64::from(fixture.input_f32[input_begin + column]);
        sum += activation * f64::from(decode_e2m1(code));
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

fn decode_e4m3fn(word: u8) -> Result<f32, Nvfp4DownQualificationError> {
    codecs::decode_e4m3fn(word).ok_or_else(|| {
        Nvfp4DownQualificationError::Mismatch("oracle encountered an E4M3FN NaN".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_BYTES_PER_ROW, GROUP, MAX_BATCH, Nvfp4DownQualificationError, OUTPUT_ROWS,
        decode_e2m1, decode_e4m3fn, down_oracle, exceptional_group, make_fixture,
        qualify_nvfp4_down, scale_offset,
    };

    #[test]
    fn independent_codecs_and_swizzle_are_pinned() {
        assert_eq!(decode_e2m1(0x07), 6.0);
        assert_eq!(decode_e2m1(0x0f), -6.0);
        assert_eq!(decode_e4m3fn(0x01).unwrap(), 2.0f32.powi(-9));
        assert_eq!(decode_e4m3fn(0x38).unwrap(), 1.0);
        assert_eq!(decode_e4m3fn(0x40).unwrap(), 2.0);
        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(32, 0), 4);
        assert_eq!(scale_offset(127, 1_087), 139_263);
        assert_eq!(scale_offset(128, 0), 139_264);
    }

    #[test]
    fn structured_fixture_exercises_one_exceptional_group_per_row() {
        let fixture = make_fixture();

        for row in [0, 1, 127, 128, OUTPUT_ROWS - 1] {
            let exceptional = exceptional_group(row);
            let ordinary = (exceptional + 1) % super::GROUPS_PER_ROW;
            let exceptional_begin = row * CODE_BYTES_PER_ROW + exceptional * (GROUP / 2);
            let ordinary_begin = row * CODE_BYTES_PER_ROW + ordinary * (GROUP / 2);

            assert_ne!(
                &fixture.weight_codes[exceptional_begin..exceptional_begin + GROUP / 2],
                &fixture.weight_codes[ordinary_begin..ordinary_begin + GROUP / 2],
            );
            assert_ne!(
                fixture.weight_scales[scale_offset(row, exceptional)],
                fixture.weight_scales[scale_offset(row, ordinary)],
            );
            assert!(
                down_oracle(0, row % OUTPUT_ROWS, &fixture)
                    .unwrap()
                    .is_finite()
            );
        }
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 8.9 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Nvfp4DownQualificationError> {
        let report = qualify_nvfp4_down()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * inactive_rows * OUTPUT_ROWS);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
