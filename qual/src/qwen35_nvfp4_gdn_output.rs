//! Qwen3.5 recurrent-output NVFP4 projection qualification.

use crate::device_benchmark;
use crate::nvfp4_down_sm120::bf16_to_f32;
use crate::qwen35_nvfp4_attention_output::{
    CODE_BYTES_PER_ROW, COLUMNS, EXACT_ROUTES, Fixture, GROUPS_PER_ROW, INPUT_SCALE_DIVISOR,
    MAX_BATCH, MAX_ROWS, OUTPUT_ROWS, WEIGHT_SCALE_DIVISOR, dot_oracle_for_rows, make_fixture,
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
    /// Exact represented activation codes produced by prompt quantization.
    pub activation_codes: usize,
    /// Exact E4M3 activation scales produced by prompt quantization.
    pub activation_scales: usize,
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
    pub(crate) activation_codes: ArenaRegion<u8>,
    pub(crate) activation_scales: ArenaRegion<u8>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) weight_scales: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_bytes()
            + self.output.byte_len()
    }
}

struct Observed {
    activation_codes: Vec<u8>,
    activation_scales: Vec<u8>,
    output: Vec<u16>,
}

/// Qualifies eager and captured Qwen3.5 GDN output at every exact route.
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
    let fixture = make_fixture().map_err(|error| {
        Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "Qwen3.5 GDN-output fixture construction failed: {error}"
        ))
    })?;
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Nvfp4GdnOutputQualification {
        activation_codes: 0,
        activation_scales: 0,
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

    for rows in EXACT_ROUTES {
        reset(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying {}",
                route_name(rows)
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
    let input = layout.reserve(MAX_ROWS * COLUMNS, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.activation_scales)?.addr(),
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
    arena.fill(stream, regions.activation_codes, 0xa5)?;
    arena.fill(stream, regions.activation_scales, 0xa5)?;
    arena.fill(stream, regions.output, 0xa5)
}

pub(crate) fn launch(
    op: &Qwen35Nvfp4GdnOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        if rows <= MAX_BATCH {
            op.launch(
                stream,
                rows,
                arena.address(regions.input)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                WEIGHT_SCALE_DIVISOR,
                arena.address(regions.output)?,
            )
        } else {
            op.launch_prefill(
                stream,
                rows,
                arena.address(regions.input)?,
                arena.address(regions.activation_codes)?,
                arena.address(regions.activation_scales)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                arena.address(regions.output)?,
            )
        }
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        activation_scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_eager(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen35Nvfp4GdnOutputQualification,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    verify_scratch(rows, fixture, observed)?;
    for token in 0..rows {
        for row in 0..OUTPUT_ROWS {
            let expected = dot_oracle_for_rows(token, row, rows, fixture).map_err(|error| {
                Qwen35Nvfp4GdnOutputQualificationError::Mismatch(error.to_string())
            })?;
            let actual = f64::from(bf16_to_f32(observed.output[token * OUTPUT_ROWS + row]));
            let error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_projection_error = report.maximum_projection_error.max(error as f32);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
                    "{} output token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}",
                    route_name(rows)
                )));
            }
        }
    }

    verify_inactive(rows, observed)?;
    if rows > MAX_BATCH {
        report.activation_codes += rows * CODE_BYTES_PER_ROW;
        report.activation_scales += rows * GROUPS_PER_ROW;
    }
    report.output_values += rows * OUTPUT_ROWS;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen35Nvfp4GdnOutputQualification,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    if replay.activation_codes != eager.activation_codes
        || replay.activation_scales != eager.activation_scales
        || replay.output != eager.output
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "{} graph replay differs from eager execution",
            route_name(rows)
        )));
    }
    verify_inactive(rows, replay)?;
    report.graph_replay_values += rows * OUTPUT_ROWS;
    if rows > MAX_BATCH {
        report.graph_replay_values += rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    }
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_scratch(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    if rows <= MAX_BATCH {
        return Ok(());
    }
    let active_codes = rows * CODE_BYTES_PER_ROW;
    let active_scales = rows * GROUPS_PER_ROW;
    if let Some(index) = observed.activation_codes[..active_codes]
        .iter()
        .zip(&fixture.activation_codes[..active_codes])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "{} activation code {index}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_codes[index],
            fixture.activation_codes[index]
        )));
    }
    if let Some(index) = observed.activation_scales[..active_scales]
        .iter()
        .zip(&fixture.activation_scales[..active_scales])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "{} activation scale {index}: device={:#04x}, oracle={:#04x}",
            route_name(rows),
            observed.activation_scales[index],
            fixture.activation_scales[index]
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Observed,
) -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
    let code_begin = if rows > MAX_BATCH {
        rows * CODE_BYTES_PER_ROW
    } else {
        0
    };
    let scale_begin = if rows > MAX_BATCH {
        rows * GROUPS_PER_ROW
    } else {
        0
    };
    if observed.activation_codes[code_begin..]
        .iter()
        .any(|&value| value != 0xa5)
        || observed.activation_scales[scale_begin..]
            .iter()
            .any(|&value| value != 0xa5)
        || observed.output[rows * OUTPUT_ROWS..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35Nvfp4GdnOutputQualificationError::Mismatch(format!(
            "{} modified an inactive value",
            route_name(rows)
        )));
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    let scratch = if rows > MAX_BATCH {
        (MAX_ROWS - rows) * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    } else {
        MAX_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    };

    (MAX_ROWS - rows) * OUTPUT_ROWS + scratch
}

fn route_name(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
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
    let graphs = EXACT_ROUTES
        .into_iter()
        .map(|rows| CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
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
        let fixture = make_fixture().unwrap();

        assert_eq!(fixture.activation_bf16.len(), MAX_ROWS * COLUMNS);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 2_392_064);
        assert_eq!(layout.byte_len(), 11_829_248);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen35Nvfp4GdnOutputQualificationError> {
        let report = qualify_qwen35_nvfp4_gdn_output()?;
        let active_rows = EXACT_ROUTES.into_iter().sum::<usize>();
        let prefill_rows = EXACT_ROUTES
            .into_iter()
            .filter(|&rows| rows > MAX_BATCH)
            .sum::<usize>();
        let inactive = EXACT_ROUTES.into_iter().map(inactive_values).sum::<usize>();

        assert_eq!(report.activation_codes, prefill_rows * CODE_BYTES_PER_ROW);
        assert_eq!(report.activation_scales, prefill_rows * GROUPS_PER_ROW);
        assert_eq!(report.output_values, active_rows * OUTPUT_ROWS);
        assert_eq!(
            report.graph_replay_values,
            active_rows * OUTPUT_ROWS + prefill_rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
        assert_eq!(report.inactive_values, 2 * inactive);
        assert_eq!(report.immutable_input_values, 9_961_472);
        assert_eq!(report.arena_bytes, 11_829_248);
        assert_eq!(report.weight_bytes, 9_437_184);
        assert_eq!(report.workspace_bytes, 2_392_064);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_projection_error.is_finite());

        Ok(())
    }
}
