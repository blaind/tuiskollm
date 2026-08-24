//! Exact prefill residual/RMSNorm seam and graph qualification.

use crate::device_benchmark;
use crate::residual_norm::{bf16_to_f32, f32_to_bf16, rms_norm_oracle};
use crate::{DeviceBenchmarkError, target::ResidualNormOp};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38_27B};

const ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const MAX_ROWS: usize = 1_024;
const ALIGNMENT: usize = 256;
const INACTIVE_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];
const BRANCH_PATTERN: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, -0.25, 0.125, -0.0625, 0.03125,
];
const WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];

/// Failure of exact SM120 residual-norm prefill qualification.
#[derive(Debug, thiserror::Error)]
pub enum ResidualNormPrefillQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively under the checked clock policy.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("residual-norm prefill qualification failed: {0}")]
    Mismatch(String),
}

/// Complete observable accounting across T=32/64/128/1024.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualNormPrefillQualification {
    /// Plain RMSNorm BF16 values checked against an FP64 oracle.
    pub plain_values: usize,
    /// Published residual BF16 values checked bit-exactly.
    pub residual_values: usize,
    /// Post-residual RMSNorm BF16 values checked against an FP64 oracle.
    pub normalized_values: usize,
    /// Mutable arena values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Inactive sentinel values proved untouched.
    pub inactive_values: usize,
    /// Read-only input, branch, and weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from either normalization oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    branch: ArenaRegion<u16>,
    weight: ArenaRegion<u16>,
    plain: ArenaRegion<u16>,
    residual: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.branch.byte_len()
            + self.weight.byte_len()
            + self.plain.byte_len()
            + self.residual.byte_len()
            + self.normalized.byte_len()
    }
}

struct Fixture {
    input: Vec<u16>,
    branch: Vec<u16>,
    weight: Vec<u16>,
}

type OutputPlanes = (Vec<u16>, Vec<u16>, Vec<u16>);

/// Qualifies every exact SM120 residual-norm prefill route and public seam.
pub fn qualify_residual_norm_prefill()
-> Result<ResidualNormPrefillQualification, ResidualNormPrefillQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_immutable(&arena, &stream, regions, &fixture)?;
    let op = ResidualNormOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = ResidualNormPrefillQualification {
        plain_values: 0,
        residual_values: 0,
        normalized_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, &fixture, &eager, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
                "device addresses changed while qualifying T={rows}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let rows = MAX_ROWS * Qwen38_27B::HIDDEN;
    let input = layout.reserve(rows, ALIGNMENT)?;
    let branch = layout.reserve(rows, ALIGNMENT)?;
    let weight = layout.reserve(Qwen38_27B::HIDDEN, ALIGNMENT)?;
    let plain = layout.reserve(rows, ALIGNMENT)?;
    let residual = layout.reserve(rows, ALIGNMENT)?;
    let normalized = layout.reserve(rows, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            branch,
            weight,
            plain,
            residual,
            normalized,
        },
    ))
}

fn fixture() -> Fixture {
    let elements = MAX_ROWS * Qwen38_27B::HIDDEN;
    let input = (0..elements)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[(index + token) & 15] * (1.0 - (token & 7) as f32 / 32.0))
        })
        .collect();
    let branch = (0..elements)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(BRANCH_PATTERN[(index * 3 + token) & 7])
        })
        .collect();
    let weight = (0..Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
        .collect();

    Fixture {
        input,
        branch,
        weight,
    }
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input)?;
    arena.copy_from_host(stream, regions.branch, &fixture.branch)?;
    arena.copy_from_host(stream, regions.weight, &fixture.weight)
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.plain, 0xa5)?;
    arena.fill(stream, regions.residual, 0xa5)?;
    arena.fill(stream, regions.normalized, 0xa5)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.branch)?.addr(),
        arena.address(regions.weight)?.addr(),
        arena.address(regions.plain)?.addr(),
        arena.address(regions.residual)?.addr(),
        arena.address(regions.normalized)?.addr(),
    ])
}

fn launch(
    op: &ResidualNormOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every region is aligned, disjoint, context-local, and covers T=1024.
    unsafe {
        op.launch_plain(
            stream,
            rows,
            arena.address(regions.input)?,
            arena.address(regions.weight)?,
            arena.address(regions.plain)?,
        )?;
        op.launch_residual(
            stream,
            rows,
            arena.address(regions.input)?,
            arena.address(regions.branch)?,
            arena.address(regions.weight)?,
            arena.address(regions.residual)?,
            arena.address(regions.normalized)?,
        )
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<OutputPlanes> {
    Ok((
        arena.copy_to_host(stream, regions.plain)?,
        arena.copy_to_host(stream, regions.residual)?,
        arena.copy_to_host(stream, regions.normalized)?,
    ))
}

fn verify_eager(
    rows: usize,
    fixture: &Fixture,
    observed: &OutputPlanes,
    report: &mut ResidualNormPrefillQualification,
) -> Result<(), ResidualNormPrefillQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for token in 0..rows {
        let begin = token * hidden;
        let end = begin + hidden;
        let plain_oracle =
            rms_norm_oracle::<Qwen38_27B>(&fixture.input[begin..end], &fixture.weight);
        let residual_oracle = fixture.input[begin..end]
            .iter()
            .zip(&fixture.branch[begin..end])
            .map(|(&value, &branch)| f32_to_bf16(bf16_to_f32(value) + bf16_to_f32(branch)))
            .collect::<Vec<_>>();
        let normalized_oracle = rms_norm_oracle::<Qwen38_27B>(&residual_oracle, &fixture.weight);

        for column in 0..hidden {
            let index = begin + column;
            check_close(
                "plain RMSNorm",
                rows,
                token,
                column,
                observed.0[index],
                plain_oracle[column],
                &mut report.maximum_absolute_error,
            )?;
            if observed.1[index] != residual_oracle[column] {
                return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
                    "residual publication at T={rows}, row={token}, column={column}: device={:#06x}, oracle={:#06x}",
                    observed.1[index], residual_oracle[column]
                )));
            }
            check_close(
                "residual RMSNorm",
                rows,
                token,
                column,
                observed.2[index],
                normalized_oracle[column],
                &mut report.maximum_absolute_error,
            )?;
        }
    }

    verify_inactive(rows, observed)?;
    let active = rows * hidden;
    report.plain_values += active;
    report.residual_values += active;
    report.normalized_values += active;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &OutputPlanes,
) -> Result<(), ResidualNormPrefillQualificationError> {
    let active = rows * Qwen38_27B::HIDDEN;
    for (name, plane) in [
        ("plain", &observed.0),
        ("residual", &observed.1),
        ("normalized", &observed.2),
    ] {
        if let Some(relative) = plane[active..]
            .iter()
            .position(|&value| value != INACTIVE_SENTINEL)
        {
            let index = active + relative;
            return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
                "T={rows} {name} route modified inactive value {index}: device={:#06x}",
                plane[index]
            )));
        }
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * Qwen38_27B::HIDDEN * 3
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut ResidualNormPrefillQualification,
) -> Result<(), ResidualNormPrefillQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
                    "read-only {} changed at index {index}",
                    $name
                )));
            }
            report.immutable_input_values += actual.len();
        }};
    }

    check!(regions.input, &fixture.input, "input");
    check!(regions.branch, &fixture.branch, "branch");
    check!(regions.weight, &fixture.weight, "weight");

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &OutputPlanes,
    replay: &OutputPlanes,
    report: &mut ResidualNormPrefillQualification,
) -> Result<(), ResidualNormPrefillQualificationError> {
    for (name, expected, actual) in [
        ("plain", &eager.0, &replay.0),
        ("residual", &eager.1, &replay.1),
        ("normalized", &eager.2, &replay.2),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
                "T={rows} {name} graph replay differs from eager at value {index}: replay={:#06x}, eager={:#06x}",
                actual[index], expected[index]
            )));
        }
    }

    verify_inactive(rows, replay)?;
    report.graph_replay_values += 3 * MAX_ROWS * Qwen38_27B::HIDDEN;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &ResidualNormOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), ResidualNormPrefillQualificationError> {
    let mut graphs = Vec::with_capacity(ROUTES.len());
    for rows in ROUTES {
        reset_outputs(arena, stream, regions)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch(op, arena, stream, regions, rows)
        })?);
    }
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
        return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_close(
    operation: &str,
    rows: usize,
    token: usize,
    column: usize,
    actual_bits: u16,
    oracle_bits: u16,
    maximum_absolute_error: &mut f32,
) -> Result<(), ResidualNormPrefillQualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_absolute_error = maximum_absolute_error.max(error);
    let tolerance = 0.015625f32.max(oracle.abs() * 0.005);
    if error > tolerance {
        return Err(ResidualNormPrefillQualificationError::Mismatch(format!(
            "{operation} at T={rows}, row={token}, column={column}: device={actual}, oracle={oracle}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_ROWS, ROUTES, layout, qualify_residual_norm_prefill};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn residual_norm_suite_prefill_routes_match_every_oracle_seam_and_graph_replay()
    -> Result<(), super::ResidualNormPrefillQualificationError> {
        let report = qualify_residual_norm_prefill()?;
        let active = ROUTES.into_iter().sum::<usize>() * Qwen38_27B::HIDDEN;
        let immutable = 2 * MAX_ROWS * Qwen38_27B::HIDDEN + Qwen38_27B::HIDDEN;

        assert_eq!(report.plain_values, active);
        assert_eq!(report.residual_values, active);
        assert_eq!(report.normalized_values, active);
        assert_eq!(
            report.graph_replay_values,
            ROUTES.len() * 3 * MAX_ROWS * Qwen38_27B::HIDDEN
        );
        assert_eq!(
            report.inactive_values,
            2 * ROUTES
                .into_iter()
                .map(|rows| (MAX_ROWS - rows) * Qwen38_27B::HIDDEN * 3)
                .sum::<usize>()
        );
        assert_eq!(report.immutable_input_values, 2 * ROUTES.len() * immutable);
        assert!(report.maximum_absolute_error <= 0.015625);
        let (arena, regions) = layout()?;
        assert_eq!(report.padding_bytes, 0);
        assert_eq!(report.arena_bytes, 52_439_040);
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(regions.payload_bytes(), arena.byte_len());

        Ok(())
    }
}
