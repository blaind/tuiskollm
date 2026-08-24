//! Source-backed qualification for the Qwen3.8 BF16 MTP Q/gate/K/V projection.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::{DeviceBenchmarkError, target::MtpBf16QkvOp};
use std::path::Path;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B};

pub(crate) const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
pub(crate) const INPUT_COLUMNS: usize = Qwen38_27B::HIDDEN;
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

/// Qualifies gathered source-BF16 MTP Q/gate/K/V projection at decode and prompt widths.
pub fn qualify_mtp_bf16_qkv(
    root: &Path,
) -> Result<MtpBf16QkvQualification, MtpBf16QkvQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let gathered = bindings.materialize_qkv()?;
    if gathered.rows != OUTPUT_ROWS || gathered.columns != INPUT_COLUMNS {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "gathered source shape is [{},{}], expected [{OUTPUT_ROWS},{INPUT_COLUMNS}]",
            gathered.rows, gathered.columns
        )));
    }
    let fixture = Fixture {
        first_input: make_input(0),
        replacement_input: make_input(1),
        weight: gathered
            .weight_bf16
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
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = MtpBf16QkvOp::new(&context)?;
    arena.copy_from_host(&stream, regions.weight, &fixture.weight)?;
    arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
    let stable_addresses = addresses(&arena, regions)?;
    let route_reference = b1_route_references(&op, &arena, &stream, regions)?;
    let source_expected =
        source_oracle(&fixture.replacement_input[..INPUT_COLUMNS], &fixture.weight);
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

    for batch in ROUTES {
        arena.copy_from_host(&stream, regions.input, &fixture.first_input)?;
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let first = arena.copy_to_host(&stream, regions.output)?;

        arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = arena.copy_to_host(&stream, regions.output)?;

        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;

        verify_outputs(
            batch,
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
            return Err(MtpBf16QkvQualificationError::Mismatch(format!(
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
    let input = layout.reserve(MAX_TOKENS * INPUT_COLUMNS, ALIGNMENT)?;
    let weight = layout.reserve(OUTPUT_ROWS * INPUT_COLUMNS, ALIGNMENT)?;
    let output = layout.reserve(MAX_TOKENS * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight,
            output,
        },
    ))
}

fn make_input(salt: usize) -> Vec<u16> {
    (0..MAX_TOKENS * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(
                INPUT_PATTERN[(index * 5 + token * 3 + salt * 7) & 15]
                    * (0.75 + token as f32 * 0.015625),
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

fn launch(
    op: &MtpBf16QkvOp,
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

fn launch_b1_row(
    op: &MtpBf16QkvOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    // SAFETY: `row < MAX_TOKENS`; each offset selects one complete aligned row.
    unsafe {
        op.launch(
            stream,
            1,
            arena.address(regions.input)?.add(row * INPUT_COLUMNS),
            arena.address(regions.weight)?,
            arena.address(regions.output)?.add(row * OUTPUT_ROWS),
        )
    }
}

fn b1_route_references(
    op: &MtpBf16QkvOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Vec<u16>> {
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
    for row in 0..MAX_TOKENS {
        launch_b1_row(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.output)
}

fn source_oracle(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(INPUT_COLUMNS)
        .map(|row| {
            input.iter().zip(row).fold(0.0f64, |sum, (&x, &w)| {
                sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
            })
        })
        .collect()
}

fn verify_outputs(
    batch: usize,
    route_reference: &[u16],
    source_expected: &[f64],
    observed: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    for row in 0..batch {
        let begin = row * OUTPUT_ROWS;
        let end = begin + OUTPUT_ROWS;
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
        for (output, (&actual, &expected)) in observed[..OUTPUT_ROWS]
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
        report.source_output_values += OUTPUT_ROWS;
    }
    report.output_values += batch * OUTPUT_ROWS;
    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    let active = batch * OUTPUT_ROWS;
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

fn verify_replacement(
    batch: usize,
    first: &[u16],
    replacement: &[u16],
) -> Result<(), MtpBf16QkvQualificationError> {
    let active = batch * OUTPUT_ROWS;
    if first[..active] == replacement[..active] {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement input"
        )));
    }
    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &[u16],
    report: &mut MtpBf16QkvQualification,
) -> Result<(), MtpBf16QkvQualificationError> {
    let begin = batch * OUTPUT_ROWS;
    if let Some(index) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(MtpBf16QkvQualificationError::Mismatch(format!(
            "B={batch} modified inactive output value {index}"
        )));
    }
    report.inactive_values += (MAX_TOKENS - batch) * OUTPUT_ROWS;
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

fn verify_no_post_warmup_allocation(
    context: &std::sync::Arc<CudaContext>,
    op: &MtpBf16QkvOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), MtpBf16QkvQualificationError> {
    launch(op, arena, stream, regions, MAX_TOKENS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for batch in ROUTES {
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
    use super::{MAX_BATCH, MAX_TOKENS, OUTPUT_ROWS, ROUTES, layout, qualify_mtp_bf16_qkv};
    use std::path::PathBuf;

    #[test]
    fn mtp_bf16_qkv_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(OUTPUT_ROWS, 14_336);
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
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

        assert_eq!(report.output_values, 1_320 * OUTPUT_ROWS);
        assert_eq!(report.source_output_values, OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, 1_320 * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * 10_968 * OUTPUT_ROWS);
        assert_eq!(report.weight_bytes, 146_800_640);
        assert_eq!(report.workspace_bytes, 39_845_888);
        assert_eq!(report.arena_bytes, 186_646_528);
        assert_eq!(report.padding_bytes, 0);
    }
}
