//! Qwen3.5 represented-value qualification for NVFP4 GDN input projections.

use crate::device_benchmark;
use crate::nvfp4_down::{bf16_to_f32, decode_e2m1, decode_e4m3fn, f32_to_bf16};
use crate::target::Qwen35Nvfp4GdnInputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen35_9B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
pub(crate) const PROJECTED_ROWS: usize = Qwen35_9B::GDN_INPUT_ROWS;
pub(crate) const CONTROL_ROWS: usize = 2 * Qwen35_9B::GDN_CONTROL_ROWS;
pub(crate) const PADDED_CONTROL_ROWS: usize = 128;
const GROUP: usize = 16;
pub(crate) const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
pub(crate) const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
pub(crate) const PROJECTED_WEIGHT_SCALE_DIVISOR: f32 = 0.125;
pub(crate) const CONTROL_WEIGHT_SCALE_DIVISOR: f32 = 0.5;
const PROJECTED_SEED: usize = 0;
const CONTROL_SEED: usize = 7;
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
            + self.weight_bytes()
            + self.projected_output.byte_len()
            + self.control_output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    pub(crate) projected_weight_codes: Vec<u8>,
    pub(crate) projected_weight_scales: Vec<u8>,
    pub(crate) control_weight_codes: Vec<u8>,
    pub(crate) control_weight_scales: Vec<u8>,
}

/// Qualifies eager and captured Qwen3.5 A16 GDN inputs at exact `B=1..=8`.
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
    let fixture = make_fixture();
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4GdnInputQualification {
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

    for batch in 1..=MAX_BATCH {
        fill_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        fill_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
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
    let projected_weight_codes = layout.reserve(PROJECTED_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let projected_weight_scales = layout.reserve(PROJECTED_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let control_weight_codes =
        layout.reserve(PADDED_CONTROL_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let control_weight_scales = layout.reserve(PADDED_CONTROL_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let projected_output = layout.reserve(MAX_BATCH * PROJECTED_ROWS, ALIGNMENT)?;
    let control_output = layout.reserve(MAX_BATCH * PADDED_CONTROL_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            projected_weight_codes,
            projected_weight_scales,
            control_weight_codes,
            control_weight_scales,
            projected_output,
            control_output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 7]> {
    Ok([
        arena.address(regions.input)?.addr(),
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

fn fill_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.projected_output, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.control_output, BF16_SENTINEL as u8)
}

fn read_outputs(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<(Vec<u16>, Vec<u16>)> {
    Ok((
        arena.copy_to_host(stream, regions.projected_output)?,
        arena.copy_to_host(stream, regions.control_output)?,
    ))
}

fn launch(
    op: &Qwen35Nvfp4GdnInputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.projected_weight_codes)?,
            arena.address(regions.projected_weight_scales)?,
            PROJECTED_WEIGHT_SCALE_DIVISOR,
            arena.address(regions.control_weight_codes)?,
            arena.address(regions.control_weight_scales)?,
            CONTROL_WEIGHT_SCALE_DIVISOR,
            arena.address(regions.projected_output)?,
            arena.address(regions.control_output)?,
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
    let (projected_weight_codes, projected_weight_scales) =
        make_weights(PROJECTED_ROWS, PROJECTED_ROWS, PROJECTED_SEED);
    let (control_weight_codes, control_weight_scales) =
        make_weights(PADDED_CONTROL_ROWS, CONTROL_ROWS, CONTROL_SEED);

    Fixture {
        input_bf16,
        input_f32,
        projected_weight_codes,
        projected_weight_scales,
        control_weight_codes,
        control_weight_scales,
    }
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
    batch: usize,
    fixture: &Fixture,
    observed: &(Vec<u16>, Vec<u16>),
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    verify_plane(
        "QKV/Z",
        batch,
        PROJECTED_ROWS,
        PROJECTED_ROWS,
        &fixture.projected_weight_codes,
        &fixture.projected_weight_scales,
        PROJECTED_WEIGHT_SCALE_DIVISOR,
        PROJECTED_SEED,
        fixture,
        &observed.0,
        report,
    )?;
    verify_plane(
        "A/B control",
        batch,
        CONTROL_ROWS,
        PADDED_CONTROL_ROWS,
        &fixture.control_weight_codes,
        &fixture.control_weight_scales,
        CONTROL_WEIGHT_SCALE_DIVISOR,
        CONTROL_SEED,
        fixture,
        &observed.1,
        report,
    )?;
    verify_control_padding(batch, &observed.1, report)?;
    verify_inactive(batch, observed)?;
    report.inactive_values += (MAX_BATCH - batch) * (PROJECTED_ROWS + PADDED_CONTROL_ROWS);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_plane(
    role: &str,
    batch: usize,
    rows: usize,
    stride: usize,
    weight_codes: &[u8],
    weight_scales: &[u8],
    divisor: f32,
    seed: usize,
    fixture: &Fixture,
    observed: &[u16],
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for token in 0..batch {
        for row in 0..rows {
            let expected = dot_oracle(
                token,
                row,
                weight_codes,
                weight_scales,
                divisor,
                seed,
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
                    "B={batch} {role} token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }
    report.output_values += batch * rows;

    Ok(())
}

fn verify_control_padding(
    batch: usize,
    observed: &[u16],
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for token in 0..batch {
        let begin = token * PADDED_CONTROL_ROWS + CONTROL_ROWS;
        let end = (token + 1) * PADDED_CONTROL_ROWS;
        if let Some(relative) = observed[begin..end].iter().position(|&value| value != 0) {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "B={batch} padded control output {} is {:#06x}, expected zero",
                begin + relative,
                observed[begin + relative]
            )));
        }
    }
    report.padded_output_values += batch * (PADDED_CONTROL_ROWS - CONTROL_ROWS);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &(Vec<u16>, Vec<u16>),
    replay: &(Vec<u16>, Vec<u16>),
    report: &mut Qwen35Nvfp4GdnInputQualification,
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    if eager != replay {
        return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (PROJECTED_ROWS + PADDED_CONTROL_ROWS);
    report.inactive_values += (MAX_BATCH - batch) * (PROJECTED_ROWS + PADDED_CONTROL_ROWS);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &(Vec<u16>, Vec<u16>),
) -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
    for (role, begin, plane) in [
        ("QKV/Z", batch * PROJECTED_ROWS, observed.0.as_slice()),
        (
            "A/B control",
            batch * PADDED_CONTROL_ROWS,
            observed.1.as_slice(),
        ),
    ] {
        if let Some(relative) = plane[begin..]
            .iter()
            .position(|&value| value != BF16_SENTINEL)
        {
            return Err(Qwen35Nvfp4GdnInputQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} output {}",
                begin + relative
            )));
        }
    }

    Ok(())
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
    report.immutable_input_values = input.len()
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
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
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
    fixture: &Fixture,
) -> Result<f64, Qwen35Nvfp4GdnInputQualificationError> {
    let exceptional = exceptional_group(row, seed);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, weight_codes, fixture);
    let exceptional_dot = group_dot(token, row, exceptional, weight_codes, fixture);
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
    fixture: &Fixture,
) -> f64 {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * INPUT_COLUMNS + group * GROUP;
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        sum += f64::from(fixture.input_f32[input_begin + column]) * f64::from(decode_e2m1(code));
    }

    sum
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_swizzle_and_padding_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(127, 255), 32_767);
        assert_eq!(scale_offset(128, 0), 32_768);
        assert_eq!(regions.weight_bytes(), 28_606_464);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 264_192);
        assert_eq!(layout.byte_len(), 28_870_656);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4GdnInputQualificationError> {
        let report = qualify_qwen35_nvfp4_gdn_input()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

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
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive_rows * (PROJECTED_ROWS + PADDED_CONTROL_ROWS)
        );
        assert_eq!(report.immutable_input_values, 28_639_232);
        assert_eq!(report.arena_bytes, 28_870_656);
        assert_eq!(report.weight_bytes, 28_606_464);
        assert_eq!(report.workspace_bytes, 264_192);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
