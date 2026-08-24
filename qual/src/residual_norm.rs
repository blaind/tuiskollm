//! Numerical and graph qualification for the exact residual-norm routes.

#[cfg(feature = "device")]
use crate::target::Qwen35ResidualNormOp;
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, ResidualNormOp};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
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

/// Failure of an exact residual-norm qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidualNormQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("residual-norm qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from all exact residual-norm routes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualNormQualification {
    /// Plain RMSNorm BF16 values compared with the FP64 oracle.
    pub plain_values: usize,
    /// Published residual BF16 values compared bit-exactly.
    pub residual_values: usize,
    /// Post-residual RMSNorm BF16 values compared with the FP64 oracle.
    pub normalized_values: usize,
    /// Active values reproduced bit-exactly by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each exact batch.
    pub inactive_values: usize,
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

pub(crate) trait ResidualNormLauncher {
    unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()>;
}

impl ResidualNormLauncher for ResidualNormOp {
    unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: this adapter preserves the operator's pointer contract.
        unsafe { ResidualNormOp::launch_plain(self, stream, batch, input, weight, output) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: this adapter preserves the operator's pointer contract.
        unsafe {
            ResidualNormOp::launch_residual(
                self,
                stream,
                batch,
                residual_input,
                branch,
                weight,
                residual_output,
                normalized_output,
            )
        }
    }
}

#[cfg(feature = "device")]
impl ResidualNormLauncher for Qwen35ResidualNormOp {
    unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: this adapter preserves the operator's pointer contract.
        unsafe { Qwen35ResidualNormOp::launch_plain(self, stream, batch, input, weight, output) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: this adapter preserves the operator's pointer contract.
        unsafe {
            Qwen35ResidualNormOp::launch_residual(
                self,
                stream,
                batch,
                residual_input,
                branch,
                weight,
                residual_output,
                normalized_output,
            )
        }
    }
}

/// Qualifies eager and captured exact `B=1..8` execution on device zero.
pub fn qualify_residual_norm() -> Result<ResidualNormQualification, ResidualNormQualificationError>
{
    qualify_target::<Qwen38_27B, ResidualNormOp>(ResidualNormOp::new)
}

/// Qualifies the exact Qwen3.5 4,096-wide routes on SM120 device zero.
#[cfg(feature = "device")]
pub fn qualify_qwen35_residual_norm()
-> Result<ResidualNormQualification, ResidualNormQualificationError> {
    qualify_target::<Qwen35_9B, Qwen35ResidualNormOp>(Qwen35ResidualNormOp::new)
}

fn qualify_target<A: Arch, O: ResidualNormLauncher>(
    prepare: fn(&Arc<CudaContext>) -> GpuResult<O>,
) -> Result<ResidualNormQualification, ResidualNormQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != EXPECTED_COMPUTE_CAPABILITY {
        return Err(ResidualNormQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected {}.{}",
            capability.0,
            capability.1,
            EXPECTED_COMPUTE_CAPABILITY.0,
            EXPECTED_COMPUTE_CAPABILITY.1,
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout::<A>()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = prepare(&context)?;
    let elements = MAX_BATCH * A::HIDDEN;
    let input = (0..elements)
        .map(|index| {
            f32_to_bf16(
                INPUT_PATTERN[(index + index / A::HIDDEN) & 15]
                    * (1.0 - (index / A::HIDDEN) as f32 * 0.03125),
            )
        })
        .collect::<Vec<_>>();
    let branch = (0..elements)
        .map(|index| f32_to_bf16(BRANCH_PATTERN[(index * 3 + index / A::HIDDEN) & 7]))
        .collect::<Vec<_>>();
    let weight = (0..A::HIDDEN)
        .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
        .collect::<Vec<_>>();

    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.branch, &branch)?;
    arena.copy_from_host(&stream, regions.weight, &weight)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = ResidualNormQualification {
        plain_values: 0,
        residual_values: 0,
        normalized_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_outputs(&arena, &stream, regions)?;
        launch_all(&op, &arena, &stream, regions, batch)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_eager::<A>(batch, &input, &branch, &weight, &eager, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph =
            CudaGraph::capture(&stream, || launch_all(&op, &arena, &stream, regions, batch))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay::<A>(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(ResidualNormQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout<A: Arch>() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let rows = MAX_BATCH * A::HIDDEN;
    let input = layout.reserve(rows, ALIGNMENT)?;
    let branch = layout.reserve(rows, ALIGNMENT)?;
    let weight = layout.reserve(A::HIDDEN, ALIGNMENT)?;
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

fn reset_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.plain, 0xa5)?;
    arena.fill(stream, regions.residual, 0xa5)?;
    arena.fill(stream, regions.normalized, 0xa5)
}

fn launch_all<O: ResidualNormLauncher>(
    op: &O,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let branch = arena.address(regions.branch)?;
    let weight = arena.address(regions.weight)?;
    let plain = arena.address(regions.plain)?;
    let residual = arena.address(regions.residual)?;
    let normalized = arena.address(regions.normalized)?;

    // SAFETY: the checked arena regions are 256-byte aligned, non-overlapping,
    // context-local, and cover the maximum batch accepted by both launches.
    unsafe {
        op.launch_plain(stream, batch, input, weight, plain)?;
        op.launch_residual(stream, batch, input, branch, weight, residual, normalized)
    }
}

type OutputPlanes = (Vec<u16>, Vec<u16>, Vec<u16>);

fn read_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<OutputPlanes> {
    Ok((
        arena.copy_to_host(stream, regions.plain)?,
        arena.copy_to_host(stream, regions.residual)?,
        arena.copy_to_host(stream, regions.normalized)?,
    ))
}

fn verify_eager<A: Arch>(
    batch: usize,
    input: &[u16],
    branch: &[u16],
    weight: &[u16],
    observed: &OutputPlanes,
    report: &mut ResidualNormQualification,
) -> Result<(), ResidualNormQualificationError> {
    let hidden = A::HIDDEN;
    let active = batch * hidden;

    for token in 0..batch {
        let begin = token * hidden;
        let end = begin + hidden;
        let plain_oracle = rms_norm_oracle::<A>(&input[begin..end], weight);
        let residual_oracle = input[begin..end]
            .iter()
            .zip(&branch[begin..end])
            .map(|(&value, &branch)| f32_to_bf16(bf16_to_f32(value) + bf16_to_f32(branch)))
            .collect::<Vec<_>>();
        let normalized_oracle = rms_norm_oracle::<A>(&residual_oracle, weight);

        for column in 0..hidden {
            let index = begin + column;
            check_close(
                "plain RMSNorm",
                batch,
                token,
                column,
                observed.0[index],
                plain_oracle[column],
                &mut report.maximum_absolute_error,
            )?;
            if observed.1[index] != residual_oracle[column] {
                return Err(ResidualNormQualificationError::Mismatch(format!(
                    "residual publication at B={batch}, row={token}, column={column}: device={:#06x}, oracle={:#06x}",
                    observed.1[index], residual_oracle[column]
                )));
            }
            check_close(
                "residual RMSNorm",
                batch,
                token,
                column,
                observed.2[index],
                normalized_oracle[column],
                &mut report.maximum_absolute_error,
            )?;
        }
    }

    verify_inactive::<A>(batch, observed)?;
    report.plain_values += active;
    report.residual_values += active;
    report.normalized_values += active;
    report.inactive_values += (MAX_BATCH - batch) * hidden * 3;

    Ok(())
}

fn verify_replay<A: Arch>(
    batch: usize,
    eager: &OutputPlanes,
    replay: &OutputPlanes,
    report: &mut ResidualNormQualification,
) -> Result<(), ResidualNormQualificationError> {
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
            return Err(ResidualNormQualificationError::Mismatch(format!(
                "B={batch} {name} graph replay differs from eager execution at value {index}: replay={:#06x}, eager={:#06x}",
                actual[index], expected[index]
            )));
        }
    }

    verify_inactive::<A>(batch, replay)?;
    report.graph_replay_values += batch * A::HIDDEN * 3;
    report.inactive_values += (MAX_BATCH - batch) * A::HIDDEN * 3;

    Ok(())
}

fn verify_inactive<A: Arch>(
    batch: usize,
    observed: &OutputPlanes,
) -> Result<(), ResidualNormQualificationError> {
    let active = batch * A::HIDDEN;
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
            return Err(ResidualNormQualificationError::Mismatch(format!(
                "B={batch} {name} route modified inactive value {index}: device={:#06x}, sentinel={INACTIVE_SENTINEL:#06x}",
                plane[index]
            )));
        }
    }

    Ok(())
}

pub(crate) fn rms_norm_oracle<A: Arch>(input: &[u16], weight: &[u16]) -> Vec<u16> {
    let squared_sum = input
        .iter()
        .map(|&bits| {
            let value = f64::from(bf16_to_f32(bits));
            value * value
        })
        .sum::<f64>();
    let mean = squared_sum / A::HIDDEN as f64;
    let inverse_rms = 1.0 / (mean + f64::from(A::RMS_NORM_EPSILON)).sqrt();

    input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| {
            let value = f64::from(bf16_to_f32(value));
            let weight = f64::from(bf16_to_f32(weight));
            f32_to_bf16((value * inverse_rms * (1.0 + weight)) as f32)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn check_close(
    operation: &str,
    batch: usize,
    token: usize,
    column: usize,
    actual_bits: u16,
    oracle_bits: u16,
    maximum_absolute_error: &mut f32,
) -> Result<(), ResidualNormQualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_absolute_error = maximum_absolute_error.max(error);
    let tolerance = 0.015625f32.max(oracle.abs() * 0.005);
    if error > tolerance {
        return Err(ResidualNormQualificationError::Mismatch(format!(
            "{operation} at B={batch}, row={token}, column={column}: device={actual}, oracle={oracle}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "device")]
    use super::qualify_qwen35_residual_norm;
    use super::{
        MAX_BATCH, Qwen35_9B, Qwen38_27B, ResidualNormQualificationError, bf16_to_f32, f32_to_bf16,
        qualify_residual_norm,
    };
    use tuisko_model::Arch;

    #[test]
    fn residual_norm_suite_bf16_conversion_uses_round_to_nearest_even() {
        let even_halfway = 1.0 + 0.00390625;
        let odd_halfway = bf16_to_f32(0x3f81) + 0.00390625;

        assert_eq!(f32_to_bf16(even_halfway), 0x3f80);
        assert_eq!(f32_to_bf16(odd_halfway), 0x3f82);
    }

    #[test]
    #[ignore = "requires the GPU selected by the qualification feature"]
    fn residual_norm_suite_decode_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), ResidualNormQualificationError> {
        let report = qualify_residual_norm()?;
        let active_per_plane = (1..=MAX_BATCH).sum::<usize>() * Qwen38_27B::HIDDEN;
        let inactive_per_run = (0..MAX_BATCH).sum::<usize>() * Qwen38_27B::HIDDEN * 3;

        assert_eq!(report.plain_values, active_per_plane);
        assert_eq!(report.residual_values, active_per_plane);
        assert_eq!(report.normalized_values, active_per_plane);
        assert_eq!(report.graph_replay_values, active_per_plane * 3);
        assert_eq!(report.inactive_values, inactive_per_run * 2);
        assert!(report.maximum_absolute_error <= 0.015625);

        Ok(())
    }

    #[cfg(feature = "device")]
    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn qwen35_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), ResidualNormQualificationError> {
        let report = qualify_qwen35_residual_norm()?;
        let active_per_plane = (1..=MAX_BATCH).sum::<usize>() * Qwen35_9B::HIDDEN;
        let inactive_per_run = (0..MAX_BATCH).sum::<usize>() * Qwen35_9B::HIDDEN * 3;

        assert_eq!(report.plain_values, active_per_plane);
        assert_eq!(report.residual_values, active_per_plane);
        assert_eq!(report.normalized_values, active_per_plane);
        assert_eq!(report.graph_replay_values, active_per_plane * 3);
        assert_eq!(report.inactive_values, inactive_per_run * 2);
        assert!(report.maximum_absolute_error <= 0.015625);

        Ok(())
    }
}
