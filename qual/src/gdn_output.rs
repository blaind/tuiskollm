//! Numerical and graph qualification for the source-native FP8 GDN output.

use crate::fp8_projection_oracle::{
    SCALE_VALUES, TokenOracle, WEIGHT_CODES, WEIGHT_VALUES, bf16_to_f32, f32_to_bf16,
    quantize_oracle,
};
use crate::harness::graph_replay::post_warmup_allocation_drift;
use crate::harness::immutable_sentinel::{
    SentinelPattern, first_bit_difference_f32, first_difference, first_non_sentinel,
    first_non_sentinel_f32,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::{DenseFp8GdnOutputTmaMaps, GdnOutputProjectionOp};
use tuisko_model::{Arch, Qwen38_27B};

const SENTINEL: SentinelPattern = SentinelPattern::new(0xa5);
const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const TOKEN_FACTORS: [f32; 16] = [
    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, -1.0, -0.875, -0.75, -0.625, -0.5, -0.375,
    -0.25, -0.125,
];

/// Failure of the exact FP8 GDN output qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum GdnOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("FP8 GDN output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, storage ownership, and worst error from every exact route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnOutputQualification {
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// BF16 projection values compared with the represented-value oracle.
    pub output_values: usize,
    /// Active values reproduced by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each exact batch.
    pub inactive_values: usize,
    /// Read-only input and projection values proved unchanged.
    pub immutable_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Sum of owned region payload bytes.
    pub payload_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute projection difference from the FP64 oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    codes: ArenaRegion<u8>,
    scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.codes.byte_len()
            + self.scales.byte_len()
            + self.weight_codes.byte_len()
            + self.weight_scales.byte_len()
            + self.output.byte_len()
    }
}

struct Fixture {
    input: Vec<u16>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u16>,
}

struct Observed {
    input: Vec<u16>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    weight_codes: Vec<u8>,
    weight_scales: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies eager and captured GDN output projection at every exact route.
pub fn qualify_gdn_output() -> Result<GdnOutputQualification, GdnOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    let oracles = fixture
        .input
        .as_slice()
        .as_chunks::<{ Qwen38_27B::GDN_VALUE_ROWS }>()
        .0
        .iter()
        .map(|row| quantize_oracle(row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(GdnOutputQualificationError::Mismatch)?;
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = GdnOutputProjectionOp::new(&context)?;
    // SAFETY: the arena owns exact stable T=1024 activation and source weight planes.
    let maps = unsafe {
        DenseFp8GdnOutputTmaMaps::new(
            &stream,
            arena.address(regions.codes)?,
            arena.address(regions.weight_codes)?,
        )?
    };
    let stable = addresses(&arena, regions)?;
    let mut report = GdnOutputQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        payload_bytes: regions.payload_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset(&arena, &stream, regions)?;
        launch(&op, &maps, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, &oracles, &fixture, &eager, &mut report)?;

        reset(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &maps, &arena, &stream, regions, rows)
        })?;
        for replay_index in 1..=2 {
            reset(&arena, &stream, regions)?;
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graph.launch(&stream) }?;
            let replay = observe(&arena, &stream, regions)?;
            verify_replay(rows, replay_index, &fixture, &eager, &replay, &mut report)?;
        }
        if addresses(&arena, regions)? != stable {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying rows={rows}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &maps, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let rows = Qwen38_27B::HIDDEN;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * columns, ALIGNMENT)?;
    let codes = layout.reserve(MAX_ROWS * columns, ALIGNMENT)?;
    let scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * columns, ALIGNMENT)?;
    let weight_scales = layout.reserve(rows, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * rows, ALIGNMENT)?;
    Ok((
        layout,
        Regions {
            input,
            codes,
            scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    let input = (0..MAX_ROWS * Qwen38_27B::GDN_VALUE_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::GDN_VALUE_ROWS;
            f32_to_bf16(INPUT_PATTERN[index & 7] * TOKEN_FACTORS[token & 15])
        })
        .collect();
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let mut weight_codes = vec![0; Qwen38_27B::HIDDEN * columns];
    for (row, values) in weight_codes
        .as_mut_slice()
        .as_chunks_mut::<6_144>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[row & 3]);
    }
    let weight_scales = (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect();

    Fixture {
        input,
        weight_codes,
        weight_scales,
    }
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input)?;
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(stream, regions.weight_scales, &fixture.weight_scales)
}

fn reset(arena: &DeviceArena, stream: &tuisko_gpu::CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.codes, SENTINEL.byte())?;
    arena.fill(stream, regions.scales, SENTINEL.byte())?;
    arena.fill(stream, regions.output, SENTINEL.byte())
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 6]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.codes)?.addr(),
        arena.address(regions.scales)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &GdnOutputProjectionOp,
    maps: &DenseFp8GdnOutputTmaMaps,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: regions are aligned, non-overlapping, context-local, and cover T=1024.
    unsafe {
        if rows == MAX_ROWS {
            op.launch_macro_prefill(
                stream,
                arena.address(regions.input)?,
                arena.address(regions.codes)?,
                arena.address(regions.scales)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                arena.address(regions.output)?,
                maps,
            )
        } else {
            op.launch(
                stream,
                rows,
                arena.address(regions.input)?,
                arena.address(regions.codes)?,
                arena.address(regions.scales)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                arena.address(regions.output)?,
            )
        }
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        input: arena.copy_to_host(stream, regions.input)?,
        codes: arena.copy_to_host(stream, regions.codes)?,
        scales: arena.copy_to_host(stream, regions.scales)?,
        weight_codes: arena.copy_to_host(stream, regions.weight_codes)?,
        weight_scales: arena.copy_to_host(stream, regions.weight_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_eager(
    rows: usize,
    oracles: &[TokenOracle],
    fixture: &Fixture,
    observed: &Observed,
    report: &mut GdnOutputQualification,
) -> Result<(), GdnOutputQualificationError> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let output_rows = Qwen38_27B::HIDDEN;
    for (token, oracle) in oracles[..rows].iter().enumerate() {
        let begin = token * columns;
        if let Some(column) =
            first_difference(&observed.codes[begin..begin + columns], &oracle.codes)
        {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "activation code at rows={rows}, token={token}, column={column} differs"
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "activation scale at rows={rows}, token={token} differs"
            )));
        }
        for row in 0..output_rows {
            let expected = oracle.represented_sum
                * f64::from(WEIGHT_VALUES[row & 3])
                * f64::from(oracle.scale)
                * f64::from(SCALE_VALUES[row & 3]);
            let actual = bf16_to_f32(observed.output[token * output_rows + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            if error > 0.0625f32.max(expected.abs() as f32 * 0.01) {
                return Err(GdnOutputQualificationError::Mismatch(format!(
                    "projection at rows={rows}, token={token}, row={row}: device={actual}, oracle={expected}"
                )));
            }
        }
    }
    verify_immutable(rows, fixture, observed)?;
    verify_inactive(rows, observed)?;
    report.activation_codes += rows * columns;
    report.activation_scales += rows;
    report.output_values += rows * output_rows;
    report.inactive_values += inactive_values(rows);
    report.immutable_values += immutable_values(fixture);
    Ok(())
}

fn verify_inactive(rows: usize, observed: &Observed) -> Result<(), GdnOutputQualificationError> {
    let code_begin = rows * Qwen38_27B::GDN_VALUE_ROWS;
    let output_begin = rows * Qwen38_27B::HIDDEN;
    if first_non_sentinel(&observed.codes[code_begin..], SENTINEL.byte()).is_some()
        || first_non_sentinel_f32(&observed.scales[rows..], SENTINEL.word_bits()).is_some()
        || first_non_sentinel(&observed.output[output_begin..], SENTINEL.half()).is_some()
    {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "rows={rows} modified an inactive value"
        )));
    }
    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (Qwen38_27B::GDN_VALUE_ROWS + 1 + Qwen38_27B::HIDDEN)
}

fn verify_immutable(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), GdnOutputQualificationError> {
    if let Some(index) = first_difference(&observed.input, &fixture.input) {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "rows={rows} modified immutable input value {index}"
        )));
    }
    if let Some(index) = first_difference(&observed.weight_codes, &fixture.weight_codes) {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "rows={rows} modified immutable weight code {index}"
        )));
    }
    if let Some(index) = first_difference(&observed.weight_scales, &fixture.weight_scales) {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "rows={rows} modified immutable weight scale {index}"
        )));
    }

    Ok(())
}

fn immutable_values(fixture: &Fixture) -> usize {
    fixture.input.len() + fixture.weight_codes.len() + fixture.weight_scales.len()
}

fn verify_replay(
    rows: usize,
    replay_index: usize,
    fixture: &Fixture,
    eager: &Observed,
    replay: &Observed,
    report: &mut GdnOutputQualification,
) -> Result<(), GdnOutputQualificationError> {
    let same = replay.codes == eager.codes
        && replay.output == eager.output
        && first_bit_difference_f32(&replay.scales, &eager.scales).is_none();
    if !same {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "rows={rows} graph replay {replay_index} differs from eager"
        )));
    }
    verify_immutable(rows, fixture, replay)?;
    verify_inactive(rows, replay)?;
    report.graph_replay_values += replay.codes.len() + replay.scales.len() + replay.output.len();
    report.inactive_values += inactive_values(rows);
    report.immutable_values += immutable_values(fixture);
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &GdnOutputProjectionOp,
    maps: &DenseFp8GdnOutputTmaMaps,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), GdnOutputQualificationError> {
    let graphs = EXACT_ROUTES
        .iter()
        .map(|&rows| CudaGraph::capture(stream, || launch(op, maps, arena, stream, regions, rows)))
        .collect::<GpuResult<Vec<_>>>()?;
    // SAFETY: every allocation these graphs captured is owned by this scope or
    // its caller and outlives the replays and the synchronize that follows.
    if let Some(drift) = unsafe { post_warmup_allocation_drift(context, stream, &graphs, 4) }? {
        return Err(GdnOutputQualificationError::Mismatch(drift));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, GdnOutputQualificationError, MAX_ROWS, Qwen38_27B, fixture, immutable_values,
        layout, qualify_gdn_output,
    };
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), GdnOutputQualificationError> {
        let report = qualify_gdn_output()?;
        let active = EXACT_ROUTES.iter().sum::<usize>();
        let values = Qwen38_27B::GDN_VALUE_ROWS + 1 + Qwen38_27B::HIDDEN;
        assert_eq!(report.activation_codes, active * Qwen38_27B::GDN_VALUE_ROWS);
        assert_eq!(report.activation_scales, active);
        assert_eq!(report.output_values, active * Qwen38_27B::HIDDEN);
        assert_eq!(
            report.graph_replay_values,
            2 * EXACT_ROUTES.len() * MAX_ROWS * values
        );
        assert_eq!(
            report.inactive_values,
            3 * EXACT_ROUTES
                .iter()
                .map(|rows| MAX_ROWS - rows)
                .sum::<usize>()
                * values
        );
        assert_eq!(
            report.immutable_values,
            3 * EXACT_ROUTES.len() * immutable_values(&fixture())
        );
        assert_eq!(
            report.arena_bytes,
            report.payload_bytes + report.padding_bytes
        );
        assert!(report.maximum_absolute_error <= 0.0625);
        Ok(())
    }

    #[test]
    fn route_inventory_and_arena_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        let (layout, regions) = layout().unwrap();
        assert_eq!(regions.payload_bytes(), 60_831_744);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }
}
