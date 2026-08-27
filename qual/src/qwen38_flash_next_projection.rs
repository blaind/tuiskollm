//! Qualification for the Qwen3.8-Flash-Next BF16 backbone projections.
//!
//! Three shapes, twelve routes each. Every one is a plain `nn.Linear` over the
//! plane the model lane materializes: FP32 accumulation across the whole
//! contraction and one round to BF16 at the store. Nothing rounds between a
//! projection and its consumer; the ops that read these planes take BF16 words,
//! so the store is the only rounding site the oracle has to reproduce, and the
//! probe below pins it as round-to-nearest rather than truncation.
//!
//! The fixture is built entirely from exact BF16 values, so every product is
//! exact in FP32 and the `f64` oracle is the exact mathematical value of the
//! represented inputs rather than a second approximation of it.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::target::{
    Qwen38FlashNextBlockOutputProjectionOp, Qwen38FlashNextGdnInputProjectionOp,
    Qwen38FlashNextQsaQkvProjectionOp,
};
use crate::{DeviceBenchmarkError, harness::immutable_sentinel::first_difference};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// The captured Qwen3.8-Flash-Next row schedule every backbone shape admits.
pub(crate) const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
/// Widest admitted prompt tile, and the row capacity every plane is sized for.
pub(crate) const MAX_ROWS: usize = 1_024;
/// Largest admitted decode batch.
pub(crate) const MAX_BATCH: usize = 8;
/// Residual-stream width the two widening shapes contract over.
pub(crate) const HIDDEN: usize = <Qwen38FlashNext as Arch>::HIDDEN;
/// Fused QKV-then-Z rows the gated DeltaNet mixer reads.
pub(crate) const GDN_INPUT_ROWS: usize = <Qwen38FlashNext as Arch>::GDN_INPUT_ROWS;
/// Fused query/gate, key, and value rows the sparse-attention prepare reads.
pub(crate) const QSA_QKV_ROWS: usize = <Qwen38FlashNext as Arch>::ATTENTION_QKV_ROWS;
/// Block width both output projection call sites contract over.
pub(crate) const BLOCK_COLUMNS: usize = <Qwen38FlashNext as Arch>::ATTENTION_OUTPUT_COLUMNS;

const ALIGNMENT: usize = 256;
/// Prompt tile used to reach the prefill body with the rounding probe.
const PROBE_TILE: usize = 32;

/// Exactly representable BF16 values, so every fixture product is exact in FP32.
const PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];

/// BF16 word `1.0078125`: the nearest-rounding of the probe's exact dot product.
const PROBE_NEAREST: u16 = 0x3f81;
/// BF16 word `1.0`: what a truncating store would publish for the same value.
const PROBE_TRUNCATED: u16 = 0x3f80;

/// Failure of the Qwen3.8-Flash-Next backbone projection qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextProjectionQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behaviour disagreed with the independent projection law.
    #[error("Qwen3.8-Flash-Next backbone projection qualification failed: {0}")]
    Mismatch(String),
}

type QualificationResult<T> = Result<T, Qwen38FlashNextProjectionQualificationError>;

/// Observable counts, ownership, and worst oracle error across the three shapes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Qwen38FlashNextProjectionQualification {
    /// Active BF16 outputs checked against their exact single-row reference.
    pub output_values: usize,
    /// Active BF16 outputs checked against the independent `f64` projection law.
    pub oracle_values: usize,
    /// Active BF16 outputs reproduced bit-exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact route's extent.
    pub inactive_values: usize,
    /// Read-only input and weight words proved unchanged.
    pub immutable_values: usize,
    /// Probe outputs whose word separates nearest rounding from truncation.
    pub rne_separated_values: usize,
    /// Materialized source-BF16 weight bytes across the three shapes.
    pub weight_bytes: usize,
    /// Address-stable activation bytes across the three shapes.
    pub workspace_bytes: usize,
    /// One-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute difference from the `f64` projection law.
    pub maximum_absolute_error: f32,
}

/// One shape's planes, in launch order.
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

/// One admitted backbone shape: its widths and its prepared owner.
pub(crate) trait ProjectionShape {
    /// Prepared owner of this shape's twelve routes.
    type Op;

    /// Name this shape carries in a rejection message.
    const LABEL: &'static str;
    /// Route identity this shape carries in a benchmark case.
    const OPERATION: &'static str;
    /// Contraction width.
    const COLUMNS: usize;
    /// Output rows published per represented row.
    const OUTPUT_ROWS: usize;

    /// Prepares every route of this shape.
    fn new(context: &Arc<CudaContext>) -> GpuResult<Self::Op>;

    /// Applies this shape's projection at one admitted row count.
    ///
    /// # Safety
    ///
    /// Carries the owner's pointer contract unchanged.
    unsafe fn launch(
        op: &Self::Op,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// The fused GDN QKV/Z input projection.
pub(crate) struct GdnInputShape;
/// The fused sparse-attention query/key/value projection.
pub(crate) struct QsaQkvShape;
/// The block output projection shared by both layer kinds.
pub(crate) struct BlockOutputShape;

macro_rules! impl_projection_shape {
    ($shape:ty, $op:ty, $label:literal, $operation:literal, $columns:expr, $rows:expr) => {
        impl ProjectionShape for $shape {
            type Op = $op;

            const LABEL: &'static str = $label;
            const OPERATION: &'static str = $operation;
            const COLUMNS: usize = $columns;
            const OUTPUT_ROWS: usize = $rows;

            fn new(context: &Arc<CudaContext>) -> GpuResult<Self::Op> {
                <$op>::new(context)
            }

            unsafe fn launch(
                op: &Self::Op,
                stream: &CudaStream,
                rows: usize,
                input: *const u16,
                weight: *const u16,
                output: *mut u16,
            ) -> GpuResult<()> {
                // SAFETY: the caller carries the owner's pointer contract.
                unsafe { <$op>::launch(op, stream, rows, input, weight, output) }
            }
        }
    };
}

impl_projection_shape!(
    GdnInputShape,
    Qwen38FlashNextGdnInputProjectionOp,
    "gdn_input",
    "qwen38_flash_next/projection/gdn_input_bf16",
    HIDDEN,
    GDN_INPUT_ROWS
);
impl_projection_shape!(
    QsaQkvShape,
    Qwen38FlashNextQsaQkvProjectionOp,
    "qsa_qkv",
    "qwen38_flash_next/projection/qsa_qkv_bf16",
    HIDDEN,
    QSA_QKV_ROWS
);
impl_projection_shape!(
    BlockOutputShape,
    Qwen38FlashNextBlockOutputProjectionOp,
    "block_output",
    "qwen38_flash_next/projection/block_output_bf16",
    BLOCK_COLUMNS,
    HIDDEN
);

/// One shape's host-side activation and weight planes.
pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) replacement_input: Vec<u16>,
    pub(crate) weight: Vec<u16>,
}

/// A non-periodic mix of two indices.
///
/// A periodic pattern would make output rows congruent modulo the period share
/// a dot product byte for byte, and a per-row reference comparison would then
/// hold vacuously for most of the plane.
fn mix(first: usize, second: usize) -> usize {
    let seed = first
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(second.wrapping_mul(0x85EB_CA77));

    (seed ^ (seed >> 15)).wrapping_mul(0xC2B2_AE35) >> 16
}

/// Builds one shape's fixture from exactly representable BF16 values.
pub(crate) fn make_fixture<S: ProjectionShape>() -> Fixture {
    Fixture {
        input: make_input::<S>(0),
        replacement_input: make_input::<S>(1),
        weight: (0..S::OUTPUT_ROWS * S::COLUMNS)
            .map(|index| {
                let row = index / S::COLUMNS;
                let column = index - row * S::COLUMNS;
                f32_to_bf16(PATTERN[mix(row + 0x51ED, column) & 15])
            })
            .collect(),
    }
}

pub(crate) fn make_input<S: ProjectionShape>(salt: usize) -> Vec<u16> {
    (0..MAX_ROWS * S::COLUMNS)
        .map(|index| {
            let row = index / S::COLUMNS;
            let column = index - row * S::COLUMNS;
            f32_to_bf16(PATTERN[mix(row + salt * 0x2545, column + salt) & 15])
        })
        .collect()
}

/// One row of the rounding probe: `[1, 2^-4, 2^-8, 0, ...]`.
///
/// Against a second row carrying the same three values the exact dot product is
/// `1 + 2^-8 + 2^-16`, which lies strictly above the BF16 midpoint between
/// `1.0` and `1.0078125`. Nearest rounding therefore publishes `1.0078125` and
/// truncation publishes `1.0`, so the emitted word names the store's rule with
/// no tolerance and no appeal to the accumulation order.
pub(crate) fn probe_row(columns: usize) -> Vec<u16> {
    (0..columns)
        .map(|column| match column {
            0 => f32_to_bf16(1.0),
            1 => f32_to_bf16(0.0625),
            2 => f32_to_bf16(0.003_906_25),
            _ => 0,
        })
        .collect()
}

/// The rounding probe's activation plane: `rows` copies of [`probe_row`].
fn probe_plane<S: ProjectionShape>(rows: usize) -> Vec<u16> {
    let row = probe_row(S::COLUMNS);

    (0..rows).flat_map(|_| row.clone()).collect()
}

/// The probe's weight plane: row zero carries the probe, every other row is zero.
fn probe_weight<S: ProjectionShape>() -> Vec<u16> {
    let mut weight = vec![0u16; S::OUTPUT_ROWS * S::COLUMNS];
    weight[..S::COLUMNS].copy_from_slice(&probe_row(S::COLUMNS));

    weight
}

/// The exact dot product the probe fixture produces, in `f64`.
pub(crate) fn probe_value() -> f64 {
    1.0 + f64::from(0.0625f32) * f64::from(0.0625f32)
        + f64::from(0.003_906_25f32) * f64::from(0.003_906_25f32)
}

pub(crate) fn layout<S: ProjectionShape>() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let regions = reserve::<S>(&mut layout)?;

    Ok((layout, regions))
}

fn reserve<S: ProjectionShape>(layout: &mut ArenaLayout) -> GpuResult<Regions> {
    let input = layout.reserve(MAX_ROWS * S::COLUMNS, ALIGNMENT)?;
    let weight = layout.reserve(S::OUTPUT_ROWS * S::COLUMNS, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * S::OUTPUT_ROWS, ALIGNMENT)?;

    Ok(Regions {
        input,
        weight,
        output,
    })
}

pub(crate) fn launch<S: ProjectionShape>(
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: aligned, non-overlapping regions cover every admitted extent.
    unsafe {
        S::launch(
            op,
            stream,
            rows,
            arena.address(regions.input)?,
            arena.address(regions.weight)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_single_row<S: ProjectionShape>(
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    // SAFETY: `row` stays inside the reserved row capacity, and each offset
    // selects one complete row of its own plane.
    unsafe {
        S::launch(
            op,
            stream,
            1,
            arena.address(regions.input)?.add(row * S::COLUMNS),
            arena.address(regions.weight)?,
            arena.address(regions.output)?.add(row * S::OUTPUT_ROWS),
        )
    }
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 3]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weight)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

/// The independent projection law: one `f64` dot product per output row.
///
/// It reads the same represented BF16 words the device reads and applies the
/// plain GEMM definition. No intermediate rounding appears anywhere, because
/// the reference has none between the contraction and the store.
pub(crate) fn oracle<S: ProjectionShape>(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(S::COLUMNS)
        .map(|row| oracle_row(input, row))
        .collect()
}

/// One output row of the projection law, over the represented source words.
pub(crate) fn oracle_row(input: &[u16], weight_row: &[u16]) -> f64 {
    input
        .iter()
        .zip(weight_row)
        .fold(0.0f64, |sum, (&activation, &coefficient)| {
            sum + f64::from(bf16_to_f32(activation)) * f64::from(bf16_to_f32(coefficient))
        })
}

/// Per-value acceptance contract: one BF16 ulp, with a floor near zero.
///
/// The device rounds once, so the exact result can move by half an ulp; the
/// FP32 accumulation over at most 6,144 terms contributes some four orders of
/// magnitude less than that.
pub(crate) fn output_tolerance(expected: f64) -> f32 {
    (expected.abs() as f32 * 0.003_906_25).max(0.007_812_5)
}

/// Qualifies every admitted route of all three backbone shapes.
pub fn qualify_qwen38_flash_next_projections()
-> QualificationResult<Qwen38FlashNextProjectionQualification> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut report = Qwen38FlashNextProjectionQualification::default();
    qualify_shape::<GdnInputShape>(&context, &stream, &mut report)?;
    qualify_shape::<QsaQkvShape>(&context, &stream, &mut report)?;
    qualify_shape::<BlockOutputShape>(&context, &stream, &mut report)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn qualify_shape<S: ProjectionShape>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    let (layout, regions) = layout::<S>()?;
    let arena = DeviceArena::zeroed(stream, &layout)?;
    let op = S::new(context)?;
    report.weight_bytes += regions.weight_bytes();
    report.workspace_bytes += regions.workspace_bytes();
    report.arena_bytes += layout.byte_len();
    report.padding_bytes += layout.byte_len() - regions.payload_bytes();

    verify_rounding_site::<S>(&op, &arena, stream, regions, report)?;

    let fixture = make_fixture::<S>();
    arena.copy_from_host(stream, regions.weight, &fixture.weight)?;
    arena.copy_from_host(stream, regions.input, &fixture.replacement_input)?;
    let stable_addresses = addresses(&arena, regions)?;
    let single_row_reference = single_row_references::<S>(&op, &arena, stream, regions)?;
    let expected = oracle::<S>(&fixture.replacement_input[..S::COLUMNS], &fixture.weight);

    for &rows in &EXACT_ROUTES {
        arena.copy_from_host(stream, regions.input, &fixture.input)?;
        arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
        launch::<S>(&op, &arena, stream, regions, rows)?;
        let first = arena.copy_to_host(stream, regions.output)?;

        arena.copy_from_host(stream, regions.input, &fixture.replacement_input)?;
        arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(stream, || launch::<S>(&op, &arena, stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope
        // and outlives both replays and the copy that follows them.
        unsafe { graph.launch(stream) }?;
        // SAFETY: as above; a second replay proves the graph is re-runnable.
        unsafe { graph.launch(stream) }?;
        let replay = arena.copy_to_host(stream, regions.output)?;

        arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
        launch::<S>(&op, &arena, stream, regions, rows)?;
        let eager = arena.copy_to_host(stream, regions.output)?;

        verify_rows::<S>(rows, &single_row_reference, &expected, &replay, report)?;
        verify_replay::<S>(rows, &eager, &replay, report)?;
        verify_replacement::<S>(rows, &first, &replay)?;
        verify_inactive::<S>(rows, &eager, report)?;
        verify_inactive::<S>(rows, &replay, report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
                format!(
                    "{} device addresses changed while qualifying rows={rows}",
                    S::LABEL
                ),
            ));
        }
    }

    verify_immutable::<S>(&arena, stream, regions, &fixture, report)?;
    verify_no_post_warmup_allocation::<S>(context, &op, &arena, stream, regions)?;

    Ok(())
}

/// Proves the store rounds to nearest rather than truncating, in both bodies.
fn verify_rounding_site<S: ProjectionShape>(
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    arena.copy_from_host(stream, regions.weight, &probe_weight::<S>())?;
    let mut input = vec![0u16; MAX_ROWS * S::COLUMNS];
    let probe = probe_plane::<S>(PROBE_TILE);
    input[..probe.len()].copy_from_slice(&probe);
    arena.copy_from_host(stream, regions.input, &input)?;

    // The decode body publishes one accumulator half; the prompt body publishes
    // both, so the probe has to reach a whole prompt tile to name the rule for
    // every emitted store.
    for rows in [1, PROBE_TILE] {
        arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
        launch::<S>(op, arena, stream, regions, rows)?;
        let observed = arena.copy_to_host(stream, regions.output)?;
        for row in 0..rows {
            let word = observed[row * S::OUTPUT_ROWS];
            if word != PROBE_NEAREST {
                let rule = if word == PROBE_TRUNCATED {
                    "truncates"
                } else {
                    "publishes neither the nearest nor the truncated word"
                };
                return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
                    format!(
                        "{} rows={rows} row={row} store {rule}: {word:#06x}, expected \
                     {PROBE_NEAREST:#06x} for the exact value {}",
                        S::LABEL,
                        probe_value()
                    ),
                ));
            }
            report.rne_separated_values += 1;
        }
    }

    Ok(())
}

/// Projects every reserved row on its own, so each route has a per-row reference.
fn single_row_references<S: ProjectionShape>(
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Vec<u16>> {
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)?;
    for row in 0..MAX_ROWS {
        launch_single_row::<S>(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.output)
}

fn verify_rows<S: ProjectionShape>(
    rows: usize,
    single_row_reference: &[u16],
    expected: &[f64],
    observed: &[u16],
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    for row in 0..rows {
        let begin = row * S::OUTPUT_ROWS;
        let end = begin + S::OUTPUT_ROWS;
        if let Some(index) =
            first_difference(&observed[begin..end], &single_row_reference[begin..end])
        {
            return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
                format!(
                    "{} rows={rows}, row={row} differs from its single-row projection at output {index}",
                    S::LABEL
                ),
            ));
        }
    }

    if rows == 1 {
        for (output, (&actual, &expected)) in
            observed[..S::OUTPUT_ROWS].iter().zip(expected).enumerate()
        {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = output_tolerance(expected);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
                    format!(
                        "{} output={output}: device={actual}, oracle={expected}, tolerance={tolerance}",
                        S::LABEL
                    ),
                ));
            }
        }
        report.oracle_values += S::OUTPUT_ROWS;
    }
    report.output_values += rows * S::OUTPUT_ROWS;

    Ok(())
}

fn verify_replay<S: ProjectionShape>(
    rows: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    let active = rows * S::OUTPUT_ROWS;
    if let Some(index) = first_difference(&eager[..active], &replay[..active]) {
        return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
            format!(
                "{} rows={rows} eager and graph outputs differ at value {index}",
                S::LABEL
            ),
        ));
    }
    report.graph_replay_values += active;

    Ok(())
}

/// A replay that ignored its input plane would reproduce the first output.
fn verify_replacement<S: ProjectionShape>(
    rows: usize,
    first: &[u16],
    replacement: &[u16],
) -> QualificationResult<()> {
    let active = rows * S::OUTPUT_ROWS;
    if first[..active] == replacement[..active] {
        return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
            format!(
                "{} rows={rows} graph replay did not observe the replacement input",
                S::LABEL
            ),
        ));
    }

    Ok(())
}

fn verify_inactive<S: ProjectionShape>(
    rows: usize,
    observed: &[u16],
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    let begin = rows * S::OUTPUT_ROWS;
    if let Some(index) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
            format!(
                "{} rows={rows} modified inactive output value {index}",
                S::LABEL
            ),
        ));
    }
    report.inactive_values += (MAX_ROWS - rows) * S::OUTPUT_ROWS;

    Ok(())
}

fn verify_immutable<S: ProjectionShape>(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextProjectionQualification,
) -> QualificationResult<()> {
    for (role, actual, expected) in [
        (
            "input",
            arena.copy_to_host(stream, regions.input)?,
            &fixture.replacement_input,
        ),
        (
            "weight",
            arena.copy_to_host(stream, regions.weight)?,
            &fixture.weight,
        ),
    ] {
        if let Some(index) = first_difference(&actual, expected) {
            return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
                format!("{} read-only {role} value {index} changed", S::LABEL),
            ));
        }
        report.immutable_values += actual.len();
    }

    Ok(())
}

fn verify_no_post_warmup_allocation<S: ProjectionShape>(
    context: &Arc<CudaContext>,
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> QualificationResult<()> {
    launch::<S>(op, arena, stream, regions, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &rows in &EXACT_ROUTES {
            launch::<S>(op, arena, stream, regions, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen38FlashNextProjectionQualificationError::Mismatch(
            format!(
                "{} post-warmup launches changed device memory from {before:?} to {after:?}",
                S::LABEL
            ),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_COLUMNS, BlockOutputShape, EXACT_ROUTES, GDN_INPUT_ROWS, GdnInputShape, HIDDEN,
        MAX_BATCH, MAX_ROWS, PROBE_NEAREST, PROBE_TRUNCATED, ProjectionShape, QSA_QKV_ROWS,
        QsaQkvShape, layout, make_fixture, make_input, oracle, output_tolerance, probe_plane,
        probe_value, probe_weight, qualify_qwen38_flash_next_projections,
    };
    use crate::fp8_projection_oracle::f32_to_bf16;
    use std::collections::BTreeSet;

    fn shape_bytes<S: ProjectionShape>() -> (usize, usize, usize) {
        let (layout, regions) = layout::<S>().unwrap();
        (
            regions.weight_bytes(),
            regions.workspace_bytes(),
            layout.byte_len(),
        )
    }

    /// The probe is decisive only if the two candidate stores actually differ,
    /// and only if nearest is the one this fixture selects.
    #[test]
    fn qwen38_flash_next_projection_suite_the_probe_separates_nearest_from_truncation() {
        let value = probe_value();
        let nearest = f32_to_bf16(value as f32);
        let truncated = (((value as f32).to_bits()) >> 16) as u16;

        assert_ne!(PROBE_NEAREST, PROBE_TRUNCATED);
        assert_eq!(nearest, PROBE_NEAREST);
        assert_eq!(truncated, PROBE_TRUNCATED);
        // Strictly above the midpoint, so no rounding mode can tie.
        assert!(value > 1.003_906_25);
        assert!(value < 1.007_812_5);
    }

    /// The probe planes carry exactly the three values the law above names, and
    /// the probe weight touches no output row but the first.
    #[test]
    fn qwen38_flash_next_projection_suite_the_probe_planes_are_what_the_law_describes() {
        let plane = probe_plane::<GdnInputShape>(1);
        assert_eq!(plane.len(), HIDDEN);
        assert_eq!(
            &plane[..4],
            &[
                f32_to_bf16(1.0),
                f32_to_bf16(0.0625),
                f32_to_bf16(0.003_906_25),
                0
            ]
        );

        let weight = probe_weight::<GdnInputShape>();
        assert_eq!(weight.len(), GDN_INPUT_ROWS * HIDDEN);
        assert!(weight[HIDDEN..].iter().all(|&word| word == 0));
        // Row zero carries the probe and row one carries nothing, so the probe
        // names one output word and leaves the rest of the plane at zero.
        assert_eq!(
            oracle::<GdnInputShape>(&plane, &weight[..2 * HIDDEN]),
            vec![probe_value(), 0.0]
        );
    }

    /// The fixture must separate the weight rows and spread the values it
    /// produces, or an entry that read the wrong row would still agree.
    ///
    /// Value collisions are *expected* and are not degeneracy: the fixture is
    /// drawn from sixteen exactly representable values, so 6,144-term sums land
    /// on a coarse lattice and a 2,560-row plane meets the birthday bound long
    /// before it is periodic. What has to hold is that the rows themselves
    /// differ, which is the property a misread row would violate.
    #[test]
    fn qwen38_flash_next_projection_suite_the_fixture_separates_its_weight_rows() {
        let fixture = make_fixture::<BlockOutputShape>();
        assert_eq!(fixture.weight.len(), HIDDEN * BLOCK_COLUMNS);
        for row in 1..HIDDEN {
            let previous = (row - 1) * BLOCK_COLUMNS;
            let current = row * BLOCK_COLUMNS;
            assert_ne!(
                fixture.weight[previous..previous + BLOCK_COLUMNS],
                fixture.weight[current..current + BLOCK_COLUMNS],
                "weight rows {} and {row} are identical",
                row - 1
            );
        }

        let expected = oracle::<BlockOutputShape>(&fixture.input[..BLOCK_COLUMNS], &fixture.weight);
        assert_eq!(expected.len(), HIDDEN);
        let distinct = expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<BTreeSet<_>>();
        // Pinned exactly, so any drift in the fixture is visible in review.
        assert_eq!(distinct.len(), 2_408);

        // Most values must be large enough that the acceptance contract is its
        // relative term rather than its floor, or the comparison is slack.
        assert!(
            expected
                .iter()
                .filter(|value| output_tolerance(**value) > 0.007_812_5)
                .count()
                * 4
                > expected.len() * 3
        );
    }

    /// Replacing the activation plane must change what every route publishes.
    #[test]
    fn qwen38_flash_next_projection_suite_the_replacement_input_moves_every_row() {
        let input = make_input::<QsaQkvShape>(0);
        let replacement = make_input::<QsaQkvShape>(1);

        assert_eq!(input.len(), MAX_ROWS * HIDDEN);
        for row in [0, 1, 7, 31, MAX_ROWS - 1] {
            let begin = row * HIDDEN;
            assert_ne!(
                input[begin..begin + HIDDEN],
                replacement[begin..begin + HIDDEN]
            );
        }
    }

    #[test]
    fn qwen38_flash_next_projection_suite_route_and_byte_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(HIDDEN, 2_560);
        assert_eq!(GDN_INPUT_ROWS, 16_384);
        assert_eq!(QSA_QKV_ROWS, 13_312);
        assert_eq!(BLOCK_COLUMNS, 6_144);

        assert_eq!(
            shape_bytes::<GdnInputShape>(),
            (83_886_080, 38_797_312, 122_683_392)
        );
        assert_eq!(
            shape_bytes::<QsaQkvShape>(),
            (68_157_440, 32_505_856, 100_663_296)
        );
        assert_eq!(
            shape_bytes::<BlockOutputShape>(),
            (31_457_280, 17_825_792, 49_283_072)
        );

        // Every reserved byte is a payload byte, at every shape.
        for (weight, workspace, arena) in [
            shape_bytes::<GdnInputShape>(),
            shape_bytes::<QsaQkvShape>(),
            shape_bytes::<BlockOutputShape>(),
        ] {
            assert_eq!(arena, weight + workspace);
        }
    }

    /// The reduction is one entry family because its two call sites have the
    /// same geometry; this holds the premise the sharing rests on.
    #[test]
    fn qwen38_flash_next_projection_suite_the_reduction_serves_both_call_sites() {
        use tuisko_model::{Arch, Qwen38FlashNext};

        assert_eq!(
            <Qwen38FlashNext as Arch>::GDN_VALUE_ROWS,
            <Qwen38FlashNext as Arch>::ATTENTION_OUTPUT_COLUMNS
        );
        assert_eq!(BlockOutputShape::COLUMNS, BLOCK_COLUMNS);
        assert_eq!(BlockOutputShape::OUTPUT_ROWS, HIDDEN);
        // The reduction's output is the two widening shapes' contraction, which
        // is what makes the three a closed backbone.
        assert_eq!(BlockOutputShape::OUTPUT_ROWS, GdnInputShape::COLUMNS);
        assert_eq!(BlockOutputShape::OUTPUT_ROWS, QsaQkvShape::COLUMNS);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_projection_suite_routes_match_independent_oracles_and_graph_replay() {
        let report =
            qualify_qwen38_flash_next_projections().expect("backbone projection qualification");
        let active = EXACT_ROUTES.iter().sum::<usize>();
        let inactive = EXACT_ROUTES
            .iter()
            .map(|&rows| MAX_ROWS - rows)
            .sum::<usize>();
        let output_rows = GDN_INPUT_ROWS + QSA_QKV_ROWS + HIDDEN;

        assert_eq!(report.output_values, active * output_rows);
        assert_eq!(report.oracle_values, output_rows);
        assert_eq!(report.graph_replay_values, active * output_rows);
        assert_eq!(report.inactive_values, 2 * inactive * output_rows);
        // Every shape names the store's rule at one decode row and a whole
        // prompt tile.
        assert_eq!(report.rne_separated_values, 3 * (1 + 32));
        assert_eq!(report.weight_bytes, 183_500_800);
        assert_eq!(report.workspace_bytes, 89_128_960);
        assert_eq!(report.arena_bytes, 272_629_760);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.immutable_values > 0);
        assert!(report.maximum_absolute_error.is_finite());
    }
}
