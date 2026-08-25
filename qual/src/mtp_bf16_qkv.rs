//! Source-backed qualification for the Qwen3.8 BF16 MTP Q/gate/K/V projection.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::{
    DeviceBenchmarkError,
    target::{MtpBf16QkvOp, Qwen35MtpBf16QkvOp, Qwen36MtpBf16QkvOp},
};
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{
    Arch, CheckpointError, CheckpointResult, CheckpointSnapshot, MtpBindings, Qwen35_9B,
    Qwen36Moe35B, Qwen36MtpBindings, Qwen38_27B,
};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_TOKENS: usize = 1_024;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const QWEN35_MAX_TOKENS: usize = 128;
const QWEN35_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
pub(crate) const OUTPUT_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
const BF16_SENTINEL: u16 = 0xa5a5;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];

/// Failure of the complete source-BF16 MTP QKV qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpBf16QkvQualificationError {
    /// Snapshot admission, binding, or lossless gathering failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent source formula.
    #[error("MTP BF16 QKV qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst source-formula error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpBf16QkvQualification {
    /// Active BF16 outputs checked against exact-B route references.
    pub output_values: usize,
    /// B=1 outputs checked against the complete source-BF16 formula.
    pub source_output_values: usize,
    /// Active BF16 outputs reproduced bit-exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact batch extent.
    pub inactive_values: usize,
    /// Read-only input and gathered source words proved unchanged.
    pub immutable_values: usize,
    /// Exact gathered source-BF16 Q/gate/K/V bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input and output bytes.
    pub workspace_bytes: usize,
    /// Exact one-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute difference from the mathematical oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) weight: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight.byte_len()
    }

    pub(crate) fn workspace_bytes(self) -> usize {
        self.input.byte_len() + self.output.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes() + self.workspace_bytes()
    }
}

pub(crate) struct Fixture {
    pub(crate) first_input: Vec<u16>,
    pub(crate) replacement_input: Vec<u16>,
    pub(crate) weight: Vec<u16>,
}

trait QualifiedQkvOp: Sized {
    fn new(context: &Arc<CudaContext>) -> GpuResult<Self>;

    unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

macro_rules! impl_qualified_qkv_op {
    ($op:ty) => {
        impl QualifiedQkvOp for $op {
            fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
                <$op>::new(context)
            }

            unsafe fn launch(
                &self,
                stream: &CudaStream,
                rows: usize,
                input: *const u16,
                weight: *const u16,
                output: *mut u16,
            ) -> GpuResult<()> {
                unsafe { <$op>::launch(self, stream, rows, input, weight, output) }
            }
        }
    };
}

impl_qualified_qkv_op!(MtpBf16QkvOp);
impl_qualified_qkv_op!(Qwen35MtpBf16QkvOp);
impl_qualified_qkv_op!(Qwen36MtpBf16QkvOp);

trait QkvSource: Arch {
    fn materialize(snapshot: &CheckpointSnapshot<Self>) -> CheckpointResult<Vec<u8>>;
}

macro_rules! impl_dense_qkv_source {
    ($arch:ty) => {
        impl QkvSource for $arch {
            fn materialize(snapshot: &CheckpointSnapshot<Self>) -> CheckpointResult<Vec<u8>> {
                Ok(MtpBindings::bind(snapshot)?.materialize_qkv()?.weight_bf16)
            }
        }
    };
}

impl_dense_qkv_source!(Qwen38_27B);
impl_dense_qkv_source!(Qwen35_9B);

impl QkvSource for Qwen36Moe35B {
    fn materialize(snapshot: &CheckpointSnapshot<Self>) -> CheckpointResult<Vec<u8>> {
        Ok(Qwen36MtpBindings::bind(snapshot)?
            .materialize_qkv()?
            .weight_bf16)
    }
}

/// Qualifies gathered source-BF16 MTP Q/gate/K/V projection at decode and prompt widths.
pub fn qualify_mtp_bf16_qkv(
    root: &Path,
) -> Result<MtpBf16QkvQualification, MtpBf16QkvQualificationError> {
    qualify_qkv::<Qwen38_27B, MtpBf16QkvOp>(root, &ROUTES, MAX_TOKENS, 0.015625)
}

/// Qualifies gathered Qwen3.5 source-BF16 MTP Q/gate/K/V projection.
pub fn qualify_qwen35_mtp_bf16_qkv(
    root: &Path,
) -> Result<MtpBf16QkvQualification, MtpBf16QkvQualificationError> {
    qualify_qkv::<Qwen35_9B, Qwen35MtpBf16QkvOp>(
        root,
        &QWEN35_ROUTES,
        QWEN35_MAX_TOKENS,
        0.001953125,
    )
}

/// Qualifies gathered Qwen3.6 source-BF16 MTP Q/gate/K/V projection.
pub fn qualify_qwen36_mtp_bf16_qkv(
    root: &Path,
) -> Result<MtpBf16QkvQualification, MtpBf16QkvQualificationError> {
    qualify_qkv::<Qwen36Moe35B, Qwen36MtpBf16QkvOp>(
        root,
        &QWEN35_ROUTES,
        QWEN35_MAX_TOKENS,
        0.00390625,
    )
}

fn qualify_qkv<A: QkvSource, O: QualifiedQkvOp>(
    root: &Path,
    routes: &[usize],
    max_tokens: usize,
    token_step: f32,
) -> Result<MtpBf16QkvQualification, MtpBf16QkvQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<A>::open(root)?;
    let gathered = A::materialize(&snapshot)?;
    let expected_bytes = A::ATTENTION_QKV_ROWS
        .checked_mul(A::HIDDEN)
        .and_then(|values| values.checked_mul(size_of::<u16>()))
        .ok_or_else(|| MtpBf16QkvQualificationError::Mismatch("QKV bytes overflow".into()))?;
    if gathered.len() != expected_bytes {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "gathered source has {} bytes, expected {expected_bytes}",
            gathered.len()
        )));
    }
    let fixture = Fixture {
        first_input: make_input::<A>(max_tokens, 0, token_step),
        replacement_input: make_input::<A>(max_tokens, 1, token_step),
        weight: gathered
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect(),
    };
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout_for::<A>(max_tokens)?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = O::new(&context)?;
    arena.copy_from_host(&stream, regions.weight, &fixture.weight)?;
    arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
    let stable_addresses = addresses(&arena, regions)?;
    let route_reference = b1_route_references::<A, O>(&op, &arena, &stream, regions, max_tokens)?;
    let source_expected =
        source_oracle::<A>(&fixture.replacement_input[..A::HIDDEN], &fixture.weight);
    let mut report = MtpBf16QkvQualification {
        output_values: 0,
        source_output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for &batch in routes {
        arena.copy_from_host(&stream, regions.input, &fixture.first_input)?;
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let first = arena.copy_to_host(&stream, regions.output)?;

        arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
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

        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;

        verify_outputs::<A>(
            batch,
            &route_reference,
            &source_expected,
            &replay,
            &mut report,
        )?;
        verify_replay::<A>(batch, &eager, &replay, &mut report)?;
        verify_replacement::<A>(batch, &first, &replay)?;
        verify_inactive::<A>(batch, max_tokens, &eager, &mut report)?;
        verify_inactive::<A>(batch, max_tokens, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(MtpBf16QkvQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, routes, max_tokens)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    layout_for::<Qwen38_27B>(MAX_TOKENS)
}

fn layout_for<A: Arch>(max_tokens: usize) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(max_tokens * A::HIDDEN, ALIGNMENT)?;
    let weight = layout.reserve(A::ATTENTION_QKV_ROWS * A::HIDDEN, ALIGNMENT)?;
    let output = layout.reserve(max_tokens * A::ATTENTION_QKV_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight,
            output,
        },
    ))
}

fn make_input<A: Arch>(max_tokens: usize, salt: usize, token_step: f32) -> Vec<u16> {
    (0..max_tokens * A::HIDDEN)
        .map(|index| {
            let token = index / A::HIDDEN;
            f32_to_bf16(
                INPUT_PATTERN[(index * 5 + token * 3 + salt * 7) & 15]
                    * (0.75 + token as f32 * token_step),
            )
        })
        .collect()
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 3]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weight)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch<O: QualifiedQkvOp>(
    op: &O,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: aligned, non-overlapping regions cover every maximum exact-B extent.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.weight)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_b1_row<A: Arch, O: QualifiedQkvOp>(
    op: &O,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    // SAFETY: the caller keeps `row` inside the allocated token capacity;
    // each offset selects one complete aligned row.
    unsafe {
        op.launch(
            stream,
            1,
            arena.address(regions.input)?.add(row * A::HIDDEN),
            arena.address(regions.weight)?,
            arena
                .address(regions.output)?
                .add(row * A::ATTENTION_QKV_ROWS),
        )
    }
}

fn b1_route_references<A: Arch, O: QualifiedQkvOp>(
    op: &O,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    max_tokens: usize,
) -> GpuResult<Vec<u16>> {
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
    for row in 0..max_tokens {
        launch_b1_row::<A, O>(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.output)
}

fn source_oracle<A: Arch>(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(A::HIDDEN)
        .map(|row| {
            input.iter().zip(row).fold(0.0f64, |sum, (&x, &w)| {
                sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
            })
        })
        .collect()
}

fn verify_outputs<A: Arch>(
    batch: usize,
    route_reference: &[u16],
    source_expected: &[f64],
    observed: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    for row in 0..batch {
        let begin = row * A::ATTENTION_QKV_ROWS;
        let end = begin + A::ATTENTION_QKV_ROWS;
        if observed[begin..end] != route_reference[begin..end] {
            let output = observed[begin..end]
                .iter()
                .zip(&route_reference[begin..end])
                .position(|(actual, expected)| actual != expected)
                .expect("slices differ");
            return Err(MtpBf16QkvQualificationError::Mismatch(format!(
                "B={batch}, row={row} differs from its exact B=1 route at fused output {output}"
            )));
        }
    }
    if batch == 1 {
        for (output, (&actual, &expected)) in observed[..A::ATTENTION_QKV_ROWS]
            .iter()
            .zip(source_expected)
            .enumerate()
        {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = 0.25f32.max(expected.abs() as f32 * 0.015);
            if !actual.is_finite() || error > tolerance {
                return Err(MtpBf16QkvQualificationError::Mismatch(format!(
                    "complete source projection output={output}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        report.source_output_values += A::ATTENTION_QKV_ROWS;
    }
    report.output_values += batch * A::ATTENTION_QKV_ROWS;
    Ok(())
}

fn verify_replay<A: Arch>(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    let active = batch * A::ATTENTION_QKV_ROWS;
    if let Some(index) = eager[..active]
        .iter()
        .zip(&replay[..active])
        .position(|(left, right)| left != right)
    {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "B={batch} eager and graph outputs differ at value {index}"
        )));
    }
    report.graph_replay_values += active;
    Ok(())
}

fn verify_replacement<A: Arch>(
    batch: usize,
    first: &[u16],
    replacement: &[u16],
) -> Result<(), MtpBf16QkvQualificationError> {
    let active = batch * A::ATTENTION_QKV_ROWS;
    if first[..active] == replacement[..active] {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement input"
        )));
    }
    Ok(())
}

fn verify_inactive<A: Arch>(
    batch: usize,
    max_tokens: usize,
    observed: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    let begin = batch * A::ATTENTION_QKV_ROWS;
    if let Some(index) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "B={batch} modified inactive output value {index}"
        )));
    }
    report.inactive_values += (max_tokens - batch) * A::ATTENTION_QKV_ROWS;
    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    for (role, actual, expected) in [
        (
            "input",
            arena.copy_to_host(stream, regions.input)?,
            &fixture.replacement_input,
        ),
        (
            "gathered Q/gate/K/V weights",
            arena.copy_to_host(stream, regions.weight)?,
            &fixture.weight,
        ),
    ] {
        if actual != *expected {
            return Err(MtpBf16QkvQualificationError::Mismatch(format!(
                "read-only {role} values changed"
            )));
        }
        report.immutable_values += actual.len();
    }
    Ok(())
}

fn verify_no_post_warmup_allocation<O: QualifiedQkvOp>(
    context: &Arc<CudaContext>,
    op: &O,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    routes: &[usize],
    max_tokens: usize,
) -> Result<(), MtpBf16QkvQualificationError> {
    launch(op, arena, stream, regions, max_tokens)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in routes {
            launch(op, arena, stream, regions, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "post-warmup launches changed device memory from {before:?} to {after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, MAX_TOKENS, OUTPUT_ROWS, QWEN35_MAX_TOKENS, QWEN35_ROUTES, ROUTES, layout,
        layout_for, qualify_mtp_bf16_qkv, qualify_qwen35_mtp_bf16_qkv, qualify_qwen36_mtp_bf16_qkv,
    };
    use std::path::PathBuf;
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B};

    #[test]
    fn mtp_bf16_qkv_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();
        let active_tokens = ROUTES.iter().sum::<usize>();
        let inactive_tokens = ROUTES
            .iter()
            .map(|&batch| MAX_TOKENS - batch)
            .sum::<usize>();

        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(OUTPUT_ROWS, 14_336);
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(active_tokens, 1_284);
        assert_eq!(inactive_tokens, 11_004);
        assert_eq!(MAX_TOKENS, 1_024);
        assert_eq!(regions.weight_bytes(), 146_800_640);
        assert_eq!(regions.workspace_bytes(), 39_845_888);
        assert_eq!(regions.payload_bytes(), 186_646_528);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_bf16_qkv_suite_source_values_match_every_route_and_graph_replay() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_bf16_qkv(&root).expect("MTP BF16 QKV qualification");
        let active_tokens = ROUTES.iter().sum::<usize>();
        let inactive_tokens = ROUTES
            .iter()
            .map(|&batch| MAX_TOKENS - batch)
            .sum::<usize>();

        assert_eq!(report.output_values, active_tokens * OUTPUT_ROWS);
        assert_eq!(report.source_output_values, OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, active_tokens * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * inactive_tokens * OUTPUT_ROWS);
        assert_eq!(report.weight_bytes, 146_800_640);
        assert_eq!(report.workspace_bytes, 39_845_888);
        assert_eq!(report.arena_bytes, 186_646_528);
        assert_eq!(report.padding_bytes, 0);
    }

    #[test]
    fn qwen35_mtp_qkv_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout_for::<Qwen35_9B>(QWEN35_MAX_TOKENS).unwrap();
        let active_tokens = QWEN35_ROUTES.iter().sum::<usize>();
        let inactive_tokens = QWEN35_ROUTES
            .iter()
            .map(|&batch| QWEN35_MAX_TOKENS - batch)
            .sum::<usize>();

        assert_eq!(QWEN35_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(active_tokens, 260);
        assert_eq!(inactive_tokens, 1_148);
        assert_eq!(Qwen35_9B::ATTENTION_QKV_ROWS, 10_240);
        assert_eq!(regions.weight_bytes(), 83_886_080);
        assert_eq!(regions.workspace_bytes(), 3_670_016);
        assert_eq!(regions.payload_bytes(), 87_556_096);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.5 snapshot"]
    fn qwen35_mtp_qkv_suite_source_values_match_every_route_and_graph_replay() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_QWEN35_SNAPSHOT")
                .expect("TUISKO_QWEN35_SNAPSHOT must name the snapshot"),
        );
        let report =
            qualify_qwen35_mtp_bf16_qkv(&root).expect("Qwen3.5 MTP BF16 QKV qualification");
        let active_tokens = QWEN35_ROUTES.iter().sum::<usize>();
        let inactive_tokens = QWEN35_ROUTES
            .iter()
            .map(|&batch| QWEN35_MAX_TOKENS - batch)
            .sum::<usize>();
        let output_rows = Qwen35_9B::ATTENTION_QKV_ROWS;

        assert_eq!(report.output_values, active_tokens * output_rows);
        assert_eq!(report.source_output_values, output_rows);
        assert_eq!(report.graph_replay_values, active_tokens * output_rows);
        assert_eq!(report.inactive_values, 2 * inactive_tokens * output_rows);
        assert_eq!(report.weight_bytes, 83_886_080);
        assert_eq!(report.workspace_bytes, 3_670_016);
        assert_eq!(report.arena_bytes, 87_556_096);
        assert_eq!(report.padding_bytes, 0);
    }

    #[test]
    fn qwen36_mtp_qkv_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout_for::<Qwen36Moe35B>(QWEN35_MAX_TOKENS).unwrap();

        assert_eq!(Qwen36Moe35B::ATTENTION_QKV_ROWS, 9_216);
        assert_eq!(regions.weight_bytes(), 37_748_736);
        assert_eq!(regions.workspace_bytes(), 2_883_584);
        assert_eq!(regions.payload_bytes(), 40_632_320);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.6 snapshot"]
    fn qwen36_mtp_qkv_suite_source_values_match_every_route_and_graph_replay() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
                .expect("TUISKO_QWEN36_SNAPSHOT must name the snapshot"),
        );
        let report =
            qualify_qwen36_mtp_bf16_qkv(&root).expect("Qwen3.6 MTP BF16 QKV qualification");
        let active_tokens = QWEN35_ROUTES.iter().sum::<usize>();
        let inactive_tokens = QWEN35_ROUTES
            .iter()
            .map(|&batch| QWEN35_MAX_TOKENS - batch)
            .sum::<usize>();
        let output_rows = Qwen36Moe35B::ATTENTION_QKV_ROWS;

        assert_eq!(report.output_values, active_tokens * output_rows);
        assert_eq!(report.source_output_values, output_rows);
        assert_eq!(report.graph_replay_values, active_tokens * output_rows);
        assert_eq!(report.inactive_values, 2 * inactive_tokens * output_rows);
        assert_eq!(report.weight_bytes, 37_748_736);
        assert_eq!(report.workspace_bytes, 2_883_584);
        assert_eq!(report.arena_bytes, 40_632_320);
        assert_eq!(report.padding_bytes, 0);
    }
}
