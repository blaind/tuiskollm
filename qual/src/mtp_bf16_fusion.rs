//! Source-backed qualification for the Qwen3.8 BF16 MTP input fusion.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, target::MtpBf16FusionOp};
use std::path::Path;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];

/// Failure of the complete source-BF16 MTP fusion qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpBf16FusionQualificationError {
    /// Snapshot admission or MTP source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact device was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent source formula.
    #[error("MTP BF16 fusion qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst source-formula error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpBf16FusionQualification {
    /// Normalized embedding and hidden BF16 values checked mathematically.
    pub normalized_values: usize,
    /// Projection values checked against exact-B route references.
    pub projection_values: usize,
    /// B=1 projection values checked against the complete source-BF16 formula.
    pub source_projection_values: usize,
    /// Active seam values reproduced bit-exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact batch extent.
    pub inactive_values: usize,
    /// Read-only input, norm, and projection values proved unchanged.
    pub immutable_values: usize,
    /// Exact source-BF16 norm and projection bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input, seam, and output bytes.
    pub workspace_bytes: usize,
    /// Exact one-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute difference from a mathematical oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) embedding: ArenaRegion<u16>,
    pub(crate) hidden: ArenaRegion<u16>,
    pub(crate) embedding_norm_weight: ArenaRegion<u16>,
    pub(crate) hidden_norm_weight: ArenaRegion<u16>,
    pub(crate) normalized_embedding: ArenaRegion<u16>,
    pub(crate) normalized_hidden: ArenaRegion<u16>,
    pub(crate) projection_weight: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.embedding_norm_weight.byte_len()
            + self.hidden_norm_weight.byte_len()
            + self.projection_weight.byte_len()
    }

    pub(crate) fn workspace_bytes(self) -> usize {
        self.embedding.byte_len()
            + self.hidden.byte_len()
            + self.normalized_embedding.byte_len()
            + self.normalized_hidden.byte_len()
            + self.output.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes() + self.workspace_bytes()
    }
}

struct Fixture {
    embedding: Vec<u16>,
    hidden: Vec<u16>,
}

struct Source {
    embedding_norm: Vec<u16>,
    hidden_norm: Vec<u16>,
    projection: Vec<u16>,
}

#[derive(Clone)]
struct Observed {
    normalized_embedding: Vec<u16>,
    normalized_hidden: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies source-backed MTP input fusion at every exact `B=1..=8` route.
pub fn qualify_mtp_bf16_fusion(
    root: &Path,
) -> Result<MtpBf16FusionQualification, MtpBf16FusionQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let source = Source {
        embedding_norm: bindings.embedding_norm.words().collect(),
        hidden_norm: bindings.hidden_norm.words().collect(),
        projection: bindings.input_projection.words().collect(),
    };
    let first_fixture = make_fixture(0);
    let replacement_fixture = make_fixture(1);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(MtpBf16FusionQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = MtpBf16FusionOp::new(&context)?;
    upload_source(&arena, &stream, regions, &source)?;
    upload_fixture(&arena, &stream, regions, &replacement_fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let route_reference = b1_route_references(&op, &arena, &stream, regions)?;
    let source_expected = source_projection_oracle(&replacement_fixture, &source);
    let mut report = MtpBf16FusionQualification {
        normalized_values: 0,
        projection_values: 0,
        source_projection_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        upload_fixture(&arena, &stream, regions, &first_fixture)?;
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let first = read_observed(&arena, &stream, regions)?;

        upload_fixture(&arena, &stream, regions, &replacement_fixture)?;
        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = read_observed(&arena, &stream, regions)?;

        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = read_observed(&arena, &stream, regions)?;

        verify_seams(
            batch,
            &replacement_fixture,
            &source,
            &route_reference,
            &source_expected,
            &replay,
            &mut report,
        )?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement(batch, &first, &replay)?;
        verify_inactive(batch, &eager, &mut report)?;
        verify_inactive(batch, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(MtpBf16FusionQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(
        &arena,
        &stream,
        regions,
        &replacement_fixture,
        &source,
        &mut report,
    )?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let mut layout = ArenaLayout::new();
    let embedding = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let hidden_input = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let embedding_norm_weight = layout.reserve(hidden, ALIGNMENT)?;
    let hidden_norm_weight = layout.reserve(hidden, ALIGNMENT)?;
    let normalized_embedding = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let normalized_hidden = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let projection_weight = layout.reserve(hidden * 2 * hidden, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            embedding,
            hidden: hidden_input,
            embedding_norm_weight,
            hidden_norm_weight,
            normalized_embedding,
            normalized_hidden,
            projection_weight,
            output,
        },
    ))
}

fn make_fixture(salt: usize) -> Fixture {
    let elements = MAX_BATCH * Qwen38_27B::HIDDEN;
    let embedding = (0..elements)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(
                INPUT_PATTERN[(index + salt * 3 + token) & 15] * (1.0 - token as f32 * 0.03125),
            )
        })
        .collect();
    let hidden = (0..elements)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(
                INPUT_PATTERN[(index * 5 + salt * 7 + token * 3) & 15]
                    * (0.75 + token as f32 * 0.015625),
            )
        })
        .collect();

    Fixture { embedding, hidden }
}

fn upload_source(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    source: &Source,
) -> GpuResult<()> {
    arena.copy_from_host(
        stream,
        regions.embedding_norm_weight,
        &source.embedding_norm,
    )?;
    arena.copy_from_host(stream, regions.hidden_norm_weight, &source.hidden_norm)?;
    arena.copy_from_host(stream, regions.projection_weight, &source.projection)
}

fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.embedding, &fixture.embedding)?;
    arena.copy_from_host(stream, regions.hidden, &fixture.hidden)
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.normalized_embedding, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.normalized_hidden, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)
}

fn launch(
    op: &MtpBf16FusionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: every checked arena region is 256-byte aligned, non-overlapping,
    // context-local, and covers the maximum extent admitted by B=1..8.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.embedding)?,
            arena.address(regions.hidden)?,
            arena.address(regions.embedding_norm_weight)?,
            arena.address(regions.hidden_norm_weight)?,
            arena.address(regions.normalized_embedding)?,
            arena.address(regions.normalized_hidden)?,
            arena.address(regions.projection_weight)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_b1_row(
    op: &MtpBf16FusionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    let offset = row * Qwen38_27B::HIDDEN;
    // SAFETY: `row < MAX_BATCH`; offset pointers retain four-byte alignment and
    // each selected suffix covers one complete B=1 row.
    unsafe {
        op.launch(
            stream,
            1,
            arena.address(regions.embedding)?.add(offset),
            arena.address(regions.hidden)?.add(offset),
            arena.address(regions.embedding_norm_weight)?,
            arena.address(regions.hidden_norm_weight)?,
            arena.address(regions.normalized_embedding)?.add(offset),
            arena.address(regions.normalized_hidden)?.add(offset),
            arena.address(regions.projection_weight)?,
            arena.address(regions.output)?.add(offset),
        )
    }
}

fn b1_route_references(
    op: &MtpBf16FusionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Vec<u16>> {
    reset_outputs(arena, stream, regions)?;
    for row in 0..MAX_BATCH {
        launch_b1_row(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.output)
}

fn read_observed(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        normalized_embedding: arena.copy_to_host(stream, regions.normalized_embedding)?,
        normalized_hidden: arena.copy_to_host(stream, regions.normalized_hidden)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 8]> {
    Ok([
        arena.address(regions.embedding)?.addr(),
        arena.address(regions.hidden)?.addr(),
        arena.address(regions.embedding_norm_weight)?.addr(),
        arena.address(regions.hidden_norm_weight)?.addr(),
        arena.address(regions.normalized_embedding)?.addr(),
        arena.address(regions.normalized_hidden)?.addr(),
        arena.address(regions.projection_weight)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn source_projection_oracle(fixture: &Fixture, source: &Source) -> Vec<f64> {
    let hidden = Qwen38_27B::HIDDEN;
    let embedding =
        rms_norm_oracle::<Qwen38_27B>(&fixture.embedding[..hidden], &source.embedding_norm);
    let hidden_input =
        rms_norm_oracle::<Qwen38_27B>(&fixture.hidden[..hidden], &source.hidden_norm);
    source
        .projection
        .chunks_exact(2 * hidden)
        .map(|weight| {
            embedding
                .iter()
                .zip(&weight[..hidden])
                .chain(hidden_input.iter().zip(&weight[hidden..]))
                .fold(0.0f64, |sum, (&activation, &weight)| {
                    sum + f64::from(bf16_to_f32(activation)) * f64::from(bf16_to_f32(weight))
                })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn verify_seams(
    batch: usize,
    fixture: &Fixture,
    source: &Source,
    route_reference: &[u16],
    source_expected: &[f64],
    observed: &Observed,
    report: &mut MtpBf16FusionQualification,
) -> Result<(), MtpBf16FusionQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for row in 0..batch {
        let begin = row * hidden;
        let end = begin + hidden;
        let embedding =
            rms_norm_oracle::<Qwen38_27B>(&fixture.embedding[begin..end], &source.embedding_norm);
        let hidden_input =
            rms_norm_oracle::<Qwen38_27B>(&fixture.hidden[begin..end], &source.hidden_norm);
        compare_close_slice(
            &format!("normalized embedding at B={batch}, row={row}"),
            &observed.normalized_embedding[begin..end],
            &embedding,
            &mut report.maximum_absolute_error,
        )?;
        compare_close_slice(
            &format!("normalized hidden at B={batch}, row={row}"),
            &observed.normalized_hidden[begin..end],
            &hidden_input,
            &mut report.maximum_absolute_error,
        )?;
        if observed.output[begin..end] != route_reference[begin..end] {
            let column = observed.output[begin..end]
                .iter()
                .zip(&route_reference[begin..end])
                .position(|(actual, expected)| actual != expected)
                .expect("slices differ");
            return Err(MtpBf16FusionQualificationError::Mismatch(format!(
                "B={batch}, row={row} projection differs from its exact B=1 route at column {column}"
            )));
        }
    }

    if batch == 1 {
        for (column, (&actual, &expected)) in observed.output[..hidden]
            .iter()
            .zip(source_expected)
            .enumerate()
        {
            require_close(
                "complete source-BF16 projection",
                column,
                bf16_to_f32(actual),
                expected,
                &mut report.maximum_absolute_error,
            )?;
        }
        report.source_projection_values += hidden;
    }
    report.normalized_values += 2 * batch * hidden;
    report.projection_values += batch * hidden;

    Ok(())
}

fn compare_close_slice(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), MtpBf16FusionQualificationError> {
    for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            column,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            maximum,
        )?;
    }
    Ok(())
}

fn require_close(
    role: &str,
    column: usize,
    actual: f32,
    expected: f64,
    maximum: &mut f32,
) -> Result<(), MtpBf16FusionQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.25f32.max(expected.abs() as f32 * 0.015);
    if !actual.is_finite() || error > tolerance {
        return Err(MtpBf16FusionQualificationError::Mismatch(format!(
            "{role}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut MtpBf16FusionQualification,
) -> Result<(), MtpBf16FusionQualificationError> {
    let active = batch * Qwen38_27B::HIDDEN;
    for (role, eager, replay) in [
        (
            "normalized embedding",
            &eager.normalized_embedding[..active],
            &replay.normalized_embedding[..active],
        ),
        (
            "normalized hidden",
            &eager.normalized_hidden[..active],
            &replay.normalized_hidden[..active],
        ),
        (
            "projection output",
            &eager.output[..active],
            &replay.output[..active],
        ),
    ] {
        if let Some(index) = eager
            .iter()
            .zip(replay)
            .position(|(left, right)| left != right)
        {
            return Err(MtpBf16FusionQualificationError::Mismatch(format!(
                "B={batch} eager and graph {role} differ at value {index}"
            )));
        }
    }
    report.graph_replay_values += 3 * active;
    Ok(())
}

fn verify_replacement(
    batch: usize,
    first: &Observed,
    replacement: &Observed,
) -> Result<(), MtpBf16FusionQualificationError> {
    let active = batch * Qwen38_27B::HIDDEN;
    if first.normalized_embedding[..active] == replacement.normalized_embedding[..active]
        || first.normalized_hidden[..active] == replacement.normalized_hidden[..active]
        || first.output[..active] == replacement.output[..active]
    {
        return Err(MtpBf16FusionQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement inputs at every seam"
        )));
    }
    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
    report: &mut MtpBf16FusionQualification,
) -> Result<(), MtpBf16FusionQualificationError> {
    let begin = batch * Qwen38_27B::HIDDEN;
    for (role, values) in [
        (
            "normalized embedding",
            &observed.normalized_embedding[begin..],
        ),
        ("normalized hidden", &observed.normalized_hidden[begin..]),
        ("projection output", &observed.output[begin..]),
    ] {
        if let Some(index) = values.iter().position(|&value| value != BF16_SENTINEL) {
            return Err(MtpBf16FusionQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} value {index}"
            )));
        }
    }
    report.inactive_values += 3 * (MAX_BATCH - batch) * Qwen38_27B::HIDDEN;
    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    source: &Source,
    report: &mut MtpBf16FusionQualification,
) -> Result<(), MtpBf16FusionQualificationError> {
    for (role, actual, expected) in [
        (
            "embedding input",
            arena.copy_to_host(stream, regions.embedding)?,
            &fixture.embedding,
        ),
        (
            "hidden input",
            arena.copy_to_host(stream, regions.hidden)?,
            &fixture.hidden,
        ),
        (
            "embedding norm",
            arena.copy_to_host(stream, regions.embedding_norm_weight)?,
            &source.embedding_norm,
        ),
        (
            "hidden norm",
            arena.copy_to_host(stream, regions.hidden_norm_weight)?,
            &source.hidden_norm,
        ),
        (
            "fusion projection",
            arena.copy_to_host(stream, regions.projection_weight)?,
            &source.projection,
        ),
    ] {
        if actual != *expected {
            return Err(MtpBf16FusionQualificationError::Mismatch(format!(
                "read-only {role} values changed"
            )));
        }
        report.immutable_values += actual.len();
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &std::sync::Arc<CudaContext>,
    op: &MtpBf16FusionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), MtpBf16FusionQualificationError> {
    launch(op, arena, stream, regions, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            launch(op, arena, stream, regions, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(MtpBf16FusionQualificationError::Mismatch(format!(
            "post-warmup launches changed device memory from {before:?} to {after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, qualify_mtp_bf16_fusion};
    use std::path::PathBuf;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn mtp_bf16_fusion_suite_route_inventory_is_exact() {
        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(Qwen38_27B::HIDDEN, 5_120);
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_bf16_fusion_suite_source_values_match_every_seam_and_graph_replay() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_bf16_fusion(&root).expect("MTP BF16 fusion qualification");

        assert_eq!(report.normalized_values, 2 * 36 * Qwen38_27B::HIDDEN);
        assert_eq!(report.projection_values, 36 * Qwen38_27B::HIDDEN);
        assert_eq!(report.source_projection_values, Qwen38_27B::HIDDEN);
        assert_eq!(report.graph_replay_values, 3 * 36 * Qwen38_27B::HIDDEN);
        assert_eq!(report.inactive_values, 2 * 3 * 28 * Qwen38_27B::HIDDEN);
        assert_eq!(report.weight_bytes, 104_878_080);
        assert_eq!(report.workspace_bytes, 409_600);
        assert_eq!(report.arena_bytes, 105_287_680);
        assert_eq!(report.padding_bytes, 0);
    }
}
