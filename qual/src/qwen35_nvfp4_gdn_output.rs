//! Qwen3.5 recurrent-output NVFP4 projection qualification.

use crate::device_benchmark;
use crate::nvfp4_down::bf16_to_f32;
use crate::qwen35_nvfp4_attention_output::{
    CODE_BYTES_PER_ROW, COLUMNS, Fixture, GROUPS_PER_ROW, MAX_BATCH, OUTPUT_ROWS,
    WEIGHT_SCALE_DIVISOR, dot_oracle, make_fixture,
};
use crate::target::Qwen35Nvfp4GdnOutputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};

const ALIGNMENT: usize = 256;
const BF16_SENTINEL: u16 = 0xa5a5;

/// Failure of Qwen3.5 recurrent-output projection qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35Nvfp4GdnOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 NVFP4 GDN-output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error across every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35Nvfp4GdnOutputQualification {
    /// BF16 outputs compared with the represented-value oracle.
    pub output_values: usize,
    /// Complete mutable output seams reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside every active route extent.
    pub inactive_values: usize,
    /// Read-only activation and represented weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact packed-weight and block-scale bytes.
    pub weight_bytes: usize,
    /// Exact activation and output workspace bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute BF16 projection difference.
    pub maximum_projection_error: f32,
}

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

/// Qualifies eager and captured Qwen3.5 GDN output at exact `B=1..=8`.
pub fn qualify_qwen35_nvfp4_gdn_output()
-> Result<Qwen35Nvfp4GdnOutputQualification, Qwen35Nvfp4GdnOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen35Nvfp4GdnOutputOp::new(&context)?;
    let fixture = make_fixture();
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4GdnOutputQualification {
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_projection_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
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
    let input = layout.reserve(MAX_BATCH * COLUMNS, ALIGNMENT)?;
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

pub(crate) fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.activation_bf16)?;
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(stream, regions.weight_scales, &fixture.weight_scales)
}

fn reset(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.output, 0xa5)
}

pub(crate) fn launch(
    op: &Qwen35Nvfp4GdnOutputOp,
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
            arena.address(regions.weight_codes)?,
            arena.address(regions.weight_scales)?,
            WEIGHT_SCALE_DIVISOR,
            arena.address(regions.output)?,
        )
    }
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    output: &[u16],
    report: &mut Qwen35Nvfp4GdnOutputQualification,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    for token in 0..batch {
        for row in 0..OUTPUT_ROWS {
            let expected = dot_oracle(token, row, fixture).map_err(|error| {
                Qwen35Nvfp4GdnOutputQualificationError::Mismatch(error.to_string())
            })?;
            let actual = f64::from(bf16_to_f32(output[token * OUTPUT_ROWS + row]));
            let error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error as f32);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
                    "B={batch} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(batch, output)?;
    report.output_values += batch * OUTPUT_ROWS;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Qwen35Nvfp4GdnOutputQualification,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    if replay != eager {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * OUTPUT_ROWS;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_inactive(
    batch: usize,
    output: &[u16],
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    if output[batch * OUTPUT_ROWS..]
        .iter()
        .any(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "B={batch} modified an inactive output value"
        )));
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen35Nvfp4GdnOutputQualification,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if input != fixture.activation_bf16
        || weight_codes != fixture.weight_codes
        || weight_scales != fixture.weight_scales
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(
            "read-only activation or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = input.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Nvfp4GdnOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
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
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_and_fixture_match_exact_geometry() {
        let (layout, regions) = layout().unwrap();
        let fixture = make_fixture();

        assert_eq!(fixture.activation_bf16.len(), MAX_BATCH * COLUMNS);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 131_072);
        assert_eq!(layout.byte_len(), 9_568_256);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
        let report = qualify_qwen35_nvfp4_gdn_output()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * inactive_rows * OUTPUT_ROWS);
        assert_eq!(report.immutable_input_values, 9_469_952);
        assert_eq!(report.arena_bytes, 9_568_256);
        assert_eq!(report.weight_bytes, 9_437_184);
        assert_eq!(report.workspace_bytes, 131_072);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_projection_error.is_finite());

        Ok(())
    }
}
