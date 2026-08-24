//! Source-backed qualification for Qwen3.8 MTP gated BF16 attention output.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::{DeviceBenchmarkError, target::MtpBf16AttentionOutputOp};
use std::path::Path;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const COLUMNS: usize = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
pub(crate) const QKV_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
pub(crate) const OUTPUT_ROWS: usize = Qwen38_27B::HIDDEN;
const BF16_SENTINEL: u16 = 0xa5a5;
const BYTE_SENTINEL: u8 = 0xa5;
const F32_SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const ATTENTION_PATTERN: [f32; 16] = [
    1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.75, -0.75, 0.375, -0.375, 0.1875, -0.1875,
    0.0625, -0.0625,
];
const GATE_PATTERN: [f32; 16] = [
    0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, -0.75, 0.75, 0.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of the source-backed MTP BF16 attention-output gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpBf16AttentionOutputQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent source formula.
    #[error("MTP BF16 attention-output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable seam counts, ownership, and worst formula errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpBf16AttentionOutputQualification {
    /// Active gated FP32 values compared with mathematical sigmoid.
    pub gated_values: usize,
    /// Active represented BF16 projection inputs checked bit-exactly.
    pub activation_values: usize,
    /// Active BF16 projection outputs checked against exact-B references.
    pub output_values: usize,
    /// B=1 outputs checked against the complete source matrix formula.
    pub source_output_values: usize,
    /// Complete active mutable state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact route extent.
    pub inactive_values: usize,
    /// Read-only QKV and source weight values proved unchanged.
    pub immutable_values: usize,
    /// Exact unchanged source-BF16 output-matrix bytes.
    pub weight_bytes: usize,
    /// Exact address-stable attention, QKV, activation, and output bytes.
    pub workspace_bytes: usize,
    /// Exact one-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute gated-FP32 formula error.
    pub maximum_gated_error: f32,
    /// Largest absolute complete source-projection error.
    pub maximum_projection_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) activation: ArenaRegion<u16>,
    pub(crate) weight: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight.byte_len()
    }

    pub(crate) fn workspace_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation.byte_len()
            + self.output.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes() + self.workspace_bytes()
    }
}

pub(crate) struct Fixture {
    pub(crate) attention: Vec<f32>,
    pub(crate) qkv: Vec<u16>,
    gated: Vec<f32>,
    activation: Vec<u16>,
    pub(crate) weight: Vec<u16>,
}

struct Observed {
    attention: Vec<f32>,
    activation: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies source-BF16 MTP gated attention output at exact `B=1..=8`.
pub fn qualify_mtp_bf16_attention_output(
    root: &Path,
) -> Result<MtpBf16AttentionOutputQualification, MtpBf16AttentionOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let fixture = make_fixture(bindings.attention_output_weight.words().collect());
    let source_expected = source_oracle(&fixture.activation[..COLUMNS], &fixture.weight);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = MtpBf16AttentionOutputOp::new(&context)?;
    load_immutable(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let route_reference = b1_route_references(&op, &arena, &stream, regions, &fixture)?;
    let mut report = MtpBf16AttentionOutputQualification {
        gated_values: 0,
        activation_values: 0,
        output_values: 0,
        source_output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_gated_error: 0.0,
        maximum_projection_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions, &fixture, batch)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_outputs(
            batch,
            &fixture,
            &route_reference,
            &source_expected,
            &eager,
            &mut report,
        )?;

        reset(&arena, &stream, regions, &fixture, batch)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_BATCH * QKV_ROWS, ALIGNMENT)?;
    let activation = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
    let weight = layout.reserve(OUTPUT_ROWS * COLUMNS, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation,
            weight,
            output,
        },
    ))
}

pub(crate) fn make_fixture(weight: Vec<u16>) -> Fixture {
    let attention = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let token = index / COLUMNS;
            ATTENTION_PATTERN[index & 15] * TOKEN_FACTORS[token]
        })
        .collect::<Vec<_>>();
    let mut qkv = vec![BF16_SENTINEL; MAX_BATCH * QKV_ROWS];
    for token in 0..MAX_BATCH {
        for head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let gate = token * QKV_ROWS
                    + head * 2 * Qwen38_27B::HEAD_DIM
                    + Qwen38_27B::HEAD_DIM
                    + dimension;
                qkv[gate] = f32_to_bf16(GATE_PATTERN[dimension & 15]);
            }
        }
    }
    let gated = (0..MAX_BATCH * COLUMNS)
        .map(|index| {
            let gate = f64::from(GATE_PATTERN[index & 15]);
            (f64::from(attention[index]) / (1.0 + (-gate).exp())) as f32
        })
        .collect::<Vec<_>>();
    let activation = gated.iter().copied().map(f32_to_bf16).collect();

    Fixture {
        attention,
        qkv,
        gated,
        activation,
        weight,
    }
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 5]> {
    Ok([
        arena.address(regions.attention)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.activation)?.addr(),
        arena.address(regions.weight)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.weight, &fixture.weight)
}

fn reset(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    batch: usize,
) -> GpuResult<()> {
    arena.fill(stream, regions.attention, BYTE_SENTINEL)?;
    arena.copy_prefix_from_host(
        stream,
        regions.attention,
        &fixture.attention[..batch * COLUMNS],
    )?;
    arena.fill(stream, regions.activation, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch(
    op: &MtpBf16AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: one aligned arena owns every complete maximum-B extent.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.attention)?,
            arena.address(regions.qkv)?,
            arena.address(regions.activation)?,
            arena.address(regions.weight)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_b1_row(
    op: &MtpBf16AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    // SAFETY: `row < MAX_BATCH`; each offset selects one complete aligned row.
    unsafe {
        op.launch(
            stream,
            1,
            arena.address(regions.attention)?.add(row * COLUMNS),
            arena.address(regions.qkv)?.add(row * QKV_ROWS),
            arena.address(regions.activation)?.add(row * COLUMNS),
            arena.address(regions.weight)?,
            arena.address(regions.output)?.add(row * OUTPUT_ROWS),
        )
    }
}

fn b1_route_references(
    op: &MtpBf16AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<Vec<u16>> {
    arena.copy_from_host(stream, regions.attention, &fixture.attention)?;
    arena.fill(stream, regions.activation, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)?;
    for row in 0..MAX_BATCH {
        launch_b1_row(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.output)
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation: arena.copy_to_host(stream, regions.activation)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn source_oracle(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(COLUMNS)
        .map(|row| {
            input.iter().zip(row).fold(0.0f64, |sum, (&x, &w)| {
                sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
            })
        })
        .collect()
}

fn verify_outputs(
    batch: usize,
    fixture: &Fixture,
    route_reference: &[u16],
    source_expected: &[f64],
    observed: &Observed,
    report: &mut MtpBf16AttentionOutputQualification,
) -> Result<(), MtpBf16AttentionOutputQualificationError> {
    for token in 0..batch {
        let begin = token * COLUMNS;
        for column in 0..COLUMNS {
            let actual = observed.attention[begin + column];
            let expected = fixture.gated[begin + column];
            let error = (actual - expected).abs();
            let tolerance = 0.000_05f32.max(expected.abs() * 0.000_25);
            report.maximum_gated_error = report.maximum_gated_error.max(error);
            if !actual.is_finite() || error > tolerance {
                return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
                    "B={batch} gated token={token}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        if let Some(column) = observed.activation[begin..begin + COLUMNS]
            .iter()
            .zip(&fixture.activation[begin..begin + COLUMNS])
            .position(|(actual, expected)| actual != expected)
        {
            return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
                "B={batch} BF16 activation token={token}, column={column} differs"
            )));
        }
        let output_begin = token * OUTPUT_ROWS;
        if let Some(output) = observed.output[output_begin..output_begin + OUTPUT_ROWS]
            .iter()
            .zip(&route_reference[output_begin..output_begin + OUTPUT_ROWS])
            .position(|(actual, expected)| actual != expected)
        {
            return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
                "B={batch}, token={token} differs from exact B=1 at output {output}"
            )));
        }
    }
    if batch == 1 {
        for (output, (&actual, &expected)) in observed.output[..OUTPUT_ROWS]
            .iter()
            .zip(source_expected)
            .enumerate()
        {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            let tolerance = 0.25f32.max(expected.abs() as f32 * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error);
            if !actual.is_finite() || error > tolerance {
                return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
                    "source projection output={output}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        report.source_output_values += OUTPUT_ROWS;
    }

    verify_inactive(batch, observed)?;
    report.gated_values += batch * COLUMNS;
    report.activation_values += batch * COLUMNS;
    report.output_values += batch * OUTPUT_ROWS;
    report.inactive_values += inactive_values(batch);
    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut MtpBf16AttentionOutputQualification,
) -> Result<(), MtpBf16AttentionOutputQualificationError> {
    let active_columns = batch * COLUMNS;
    let active_outputs = batch * OUTPUT_ROWS;
    let same = replay.attention[..active_columns]
        .iter()
        .map(|value| value.to_bits())
        .eq(eager.attention[..active_columns]
            .iter()
            .map(|value| value.to_bits()))
        && replay.activation[..active_columns] == eager.activation[..active_columns]
        && replay.output[..active_outputs] == eager.output[..active_outputs];
    if !same {
        return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (2 * COLUMNS + OUTPUT_ROWS);
    report.inactive_values += inactive_values(batch);
    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
) -> Result<(), MtpBf16AttentionOutputQualificationError> {
    let columns_begin = batch * COLUMNS;
    let output_begin = batch * OUTPUT_ROWS;
    if observed.attention[columns_begin..]
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.activation[columns_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
            "B={batch} modified an inactive value"
        )));
    }
    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (2 * COLUMNS + OUTPUT_ROWS)
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut MtpBf16AttentionOutputQualification,
) -> Result<(), MtpBf16AttentionOutputQualificationError> {
    let qkv = arena.copy_to_host(stream, regions.qkv)?;
    let weight = arena.copy_to_host(stream, regions.weight)?;
    if qkv != fixture.qkv || weight != fixture.weight {
        return Err(MtpBf16AttentionOutputQualificationError::Mismatch(
            "read-only QKV or source weight plane changed".to_string(),
        ));
    }
    report.immutable_values = qkv.len() + weight.len();
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &MtpBf16AttentionOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), MtpBf16AttentionOutputQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| {
            reset(arena, stream, regions, fixture, batch)?;
            stream.synchronize().map_err(GpuError::from)?;
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch))
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
        return Err(MtpBf16AttentionOutputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{COLUMNS, MAX_BATCH, OUTPUT_ROWS, layout, qualify_mtp_bf16_attention_output};
    use std::path::PathBuf;

    #[test]
    fn mtp_bf16_attention_output_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(COLUMNS, 6_144);
        assert_eq!(OUTPUT_ROWS, 5_120);
        assert_eq!(regions.weight_bytes(), 62_914_560);
        assert_eq!(regions.workspace_bytes(), 606_208);
        assert_eq!(regions.payload_bytes(), 63_520_768);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_bf16_attention_output_suite_source_values_match_every_seam_route_and_graph() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_bf16_attention_output(&root)
            .expect("MTP BF16 attention-output qualification");

        assert_eq!(report.gated_values, 36 * COLUMNS);
        assert_eq!(report.activation_values, 36 * COLUMNS);
        assert_eq!(report.output_values, 36 * OUTPUT_ROWS);
        assert_eq!(report.source_output_values, OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, 36 * (2 * COLUMNS + OUTPUT_ROWS));
        assert_eq!(report.inactive_values, 2 * 28 * (2 * COLUMNS + OUTPUT_ROWS));
        assert_eq!(report.immutable_values, 31_571_968);
        assert_eq!(report.weight_bytes, 62_914_560);
        assert_eq!(report.workspace_bytes, 606_208);
        assert_eq!(report.arena_bytes, 63_520_768);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_gated_error.is_finite());
        assert!(report.maximum_projection_error.is_finite());
    }
}
