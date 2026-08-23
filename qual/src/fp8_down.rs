//! Numerical and graph qualification for the source-native dense-FP8 down projection.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, SCALE_VALUES, TokenOracle, WEIGHT_CODES,
    WEIGHT_VALUES, bf16_to_f32, decode_e4m3fn, f32_to_bf16, quantize_oracle,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{DenseFp8DownOp, DenseFp8DownTmaMaps};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 1_024];
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.0,
    0.5, -0.25, 0.125,
];
const TOKEN_FACTORS: [f32; 8] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact dense-FP8 down qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Fp8DownQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact device was not available exclusively under checked clocks.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent represented-value contract.
    #[error("dense-FP8 down qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact dense-FP8 down route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8DownQualification {
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// BF16 projection values compared with the represented-value oracle.
    pub output_values: usize,
    /// Active values reproduced by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active batch extent.
    pub inactive_values: usize,
    /// Read-only input and weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Opaque tensor-map words proved unchanged.
    pub immutable_descriptor_words: usize,
    /// Exact bytes in the one-allocation tensor arena.
    pub arena_bytes: usize,
    /// Exact alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Exact bytes in the two address-bound tensor-map descriptors.
    pub descriptor_bytes: usize,
    /// Largest absolute output difference from the FP64 oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_codes.byte_len()
            + self.weight_scales.byte_len()
            + self.output.byte_len()
    }
}

struct Observed {
    codes: Vec<u8>,
    scales: Vec<f32>,
    output: Vec<u16>,
}

/// Qualifies eager and captured dense-FP8 down projection at every admitted row count.
pub fn qualify_fp8_down() -> Result<Fp8DownQualification, Fp8DownQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = DenseFp8DownOp::new(&context)?;
    let input = make_input();
    let token_oracles = input
        .as_slice()
        .as_chunks::<{ Qwen38_27B::INTERMEDIATE }>()
        .0
        .iter()
        .map(|row| quantize_oracle(row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Fp8DownQualificationError::Mismatch)?;
    let correlations = token_oracles
        .iter()
        .map(weight_correlations)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Fp8DownQualificationError::Mismatch)?;
    let (weight_codes, weight_scales) = make_weights();

    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &weight_scales)?;
    // SAFETY: the arena owns exact stable T=1024 activation and source weight planes.
    let maps = unsafe {
        DenseFp8DownTmaMaps::new(
            &stream,
            arena.address(regions.activation_codes)?,
            arena.address(regions.weight_codes)?,
        )?
    };
    let descriptor_words = maps.copy_to_host(&stream)?;
    let stable_addresses = addresses(&arena, regions)?;
    let stable_descriptor_addresses = maps.device_addresses();
    let mut report = Fp8DownQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        immutable_descriptor_words: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        descriptor_bytes: maps.byte_len(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &maps, &arena, &stream, regions, rows)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify_eager(rows, &token_oracles, &correlations, &eager, &mut report)?;
        verify_immutable(
            &arena,
            &stream,
            regions,
            &input,
            &weight_codes,
            &weight_scales,
            &maps,
            &descriptor_words,
            &mut report,
        )?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &maps, &arena, &stream, regions, rows)
        })?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_immutable(
            &arena,
            &stream,
            regions,
            &input,
            &weight_codes,
            &weight_scales,
            &maps,
            &descriptor_words,
            &mut report,
        )?;

        if addresses(&arena, regions)? != stable_addresses
            || maps.device_addresses() != stable_descriptor_addresses
        {
            return Err(Fp8DownQualificationError::Mismatch(format!(
                "device addresses changed while qualifying row count {rows}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &maps, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * intermediate, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * intermediate, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(hidden * intermediate, ALIGNMENT)?;
    let weight_scales = layout.reserve(hidden, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;

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

fn make_input() -> Vec<u16> {
    (0..MAX_ROWS * Qwen38_27B::INTERMEDIATE)
        .map(|index| {
            let token = index / Qwen38_27B::INTERMEDIATE;
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token & 7])
        })
        .collect()
}

fn make_weights() -> (Vec<u8>, Vec<u16>) {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let codes = (0..hidden * intermediate)
        .map(|index| {
            let row = index / intermediate;
            let column = index - row * intermediate;
            WEIGHT_CODES[(row + column) & 3]
        })
        .collect();
    let scales = (0..hidden)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect();

    (codes, scales)
}

fn weight_correlations(oracle: &TokenOracle) -> Result<[f64; 4], String> {
    let mut correlations = [0.0f64; 4];
    for (column, &code) in oracle.codes.iter().enumerate() {
        let activation = f64::from(decode_e4m3fn(code)?);
        for phase in 0..4 {
            correlations[phase] += activation * f64::from(WEIGHT_VALUES[(phase + column) & 3]);
        }
    }

    Ok(correlations)
}

fn reset_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation_scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn launch(
    op: &DenseFp8DownOp,
    maps: &DenseFp8DownTmaMaps,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every arena region is aligned, non-overlapping, context-local,
    // and covers the maximum extent admitted by all exact routes.
    unsafe {
        if rows == MAX_ROWS {
            op.launch_macro_prefill(
                stream,
                arena.address(regions.input)?,
                arena.address(regions.activation_codes)?,
                arena.address(regions.activation_scales)?,
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
                arena.address(regions.activation_codes)?,
                arena.address(regions.activation_scales)?,
                arena.address(regions.weight_codes)?,
                arena.address(regions.weight_scales)?,
                arena.address(regions.output)?,
            )
        }
    }
}

fn read_observed(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        codes: arena.copy_to_host(stream, regions.activation_codes)?,
        scales: arena.copy_to_host(stream, regions.activation_scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_eager(
    active_batch: usize,
    oracles: &[TokenOracle],
    correlations: &[[f64; 4]],
    observed: &Observed,
    report: &mut Fp8DownQualification,
) -> Result<(), Fp8DownQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;

    for (token, oracle) in oracles[..active_batch].iter().enumerate() {
        let code_begin = token * intermediate;
        if let Some(relative) = observed.codes[code_begin..code_begin + intermediate]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Fp8DownQualificationError::Mismatch(format!(
                "activation code at batch={active_batch}, row={token}, column={relative}: device={:#04x}, oracle={:#04x}",
                observed.codes[code_begin + relative],
                oracle.codes[relative]
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(Fp8DownQualificationError::Mismatch(format!(
                "activation scale at batch={active_batch}, row={token}: device={:#010x}, oracle={:#010x}",
                observed.scales[token].to_bits(),
                oracle.scale.to_bits()
            )));
        }

        for row in 0..hidden {
            let expected = correlations[token][row & 3]
                * f64::from(oracle.scale)
                * f64::from(SCALE_VALUES[row & 3]);
            let actual = bf16_to_f32(observed.output[token * hidden + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = 0.125f32.max(expected.abs() as f32 * 0.015);
            if error > tolerance {
                return Err(Fp8DownQualificationError::Mismatch(format!(
                    "projection at batch={active_batch}, row={token}, output={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(active_batch, observed)?;
    report.activation_codes += active_batch * intermediate;
    report.activation_scales += active_batch;
    report.output_values += active_batch * hidden;
    report.inactive_values += inactive_values(active_batch);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Fp8DownQualification,
) -> Result<(), Fp8DownQualificationError> {
    if let Some(index) = replay
        .codes
        .iter()
        .zip(&eager.codes)
        .position(|(a, b)| a != b)
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} graph activation code {index} differs"
        )));
    }
    if let Some(index) = replay
        .scales
        .iter()
        .zip(&eager.scales)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} graph activation scale {index} differs"
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(a, b)| a != b)
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} graph output {index} differs"
        )));
    }

    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (Qwen38_27B::INTERMEDIATE + 1 + Qwen38_27B::HIDDEN);
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_inactive(batch: usize, observed: &Observed) -> Result<(), Fp8DownQualificationError> {
    let code_begin = batch * Qwen38_27B::INTERMEDIATE;
    if let Some(relative) = observed.codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} modified inactive activation code {}",
            code_begin + relative
        )));
    }
    if let Some(relative) = observed.scales[batch..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} modified inactive activation scale {}",
            batch + relative
        )));
    }
    let output_begin = batch * Qwen38_27B::HIDDEN;
    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "batch={batch} modified inactive output {}",
            output_begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_ROWS - batch) * (Qwen38_27B::INTERMEDIATE + 1 + Qwen38_27B::HIDDEN)
}

#[allow(clippy::too_many_arguments)]
fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    input: &[u16],
    weight_codes: &[u8],
    weight_scales: &[u16],
    maps: &DenseFp8DownTmaMaps,
    descriptor_words: &[Vec<u64>; 2],
    report: &mut Fp8DownQualification,
) -> Result<(), Fp8DownQualificationError> {
    let observed_input = arena.copy_to_host(stream, regions.input)?;
    let observed_weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let observed_weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if observed_input != input {
        let index = observed_input
            .iter()
            .zip(input)
            .position(|(actual, expected)| actual != expected)
            .unwrap();
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "read-only input changed at {index}"
        )));
    }
    if observed_weight_codes != weight_codes {
        let index = observed_weight_codes
            .iter()
            .zip(weight_codes)
            .position(|(actual, expected)| actual != expected)
            .unwrap();
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "read-only weight code changed at {index}"
        )));
    }
    if observed_weight_scales != weight_scales {
        let index = observed_weight_scales
            .iter()
            .zip(weight_scales)
            .position(|(actual, expected)| actual != expected)
            .unwrap();
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "read-only weight scale changed at {index}"
        )));
    }

    let observed_descriptors = maps.copy_to_host(stream)?;
    for (map, (actual, expected)) in observed_descriptors
        .iter()
        .zip(descriptor_words)
        .enumerate()
    {
        if actual != expected {
            let word = actual
                .iter()
                .zip(expected)
                .position(|(actual, expected)| actual != expected)
                .unwrap();
            return Err(Fp8DownQualificationError::Mismatch(format!(
                "read-only tensor map {map} changed at word {word}"
            )));
        }
    }

    report.immutable_input_values += input.len() + weight_codes.len() + weight_scales.len();
    report.immutable_descriptor_words += descriptor_words.iter().map(Vec::len).sum::<usize>();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &DenseFp8DownOp,
    maps: &DenseFp8DownTmaMaps,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Fp8DownQualificationError> {
    let mut graphs = Vec::with_capacity(EXACT_ROUTES.len());
    for rows in EXACT_ROUTES {
        reset_outputs(arena, stream, regions)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch(op, maps, arena, stream, regions, rows)
        })?);
    }
    for graph in &graphs {
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            graph.launch(stream)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Fp8DownQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, Fp8DownQualificationError, MAX_ROWS, Qwen38_27B, layout, qualify_fp8_down,
    };
    use crate::fp8_projection_oracle::verify_host_codecs;
    use tuisko_model::Arch;

    #[test]
    fn fp8_down_suite_host_codecs_pin_bf16_and_e4m3_rounding() {
        verify_host_codecs().unwrap();
    }

    #[test]
    #[ignore = "requires an idle NVIDIA compute-capability 12.0 device"]
    fn fp8_down_suite_exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Fp8DownQualificationError> {
        let report = qualify_fp8_down()?;
        let active_batch = EXACT_ROUTES.iter().sum::<usize>();
        let active_per_run = Qwen38_27B::INTERMEDIATE + 1 + Qwen38_27B::HIDDEN;
        let inactive_per_pass = EXACT_ROUTES
            .iter()
            .map(|batch| MAX_ROWS - batch)
            .sum::<usize>()
            * active_per_run;

        assert_eq!(
            report.activation_codes,
            active_batch * Qwen38_27B::INTERMEDIATE
        );
        assert_eq!(report.activation_scales, active_batch);
        assert_eq!(report.output_values, active_batch * Qwen38_27B::HIDDEN);
        assert_eq!(report.graph_replay_values, active_batch * active_per_run);
        assert_eq!(report.inactive_values, inactive_per_pass * 2);
        let immutable = MAX_ROWS * Qwen38_27B::INTERMEDIATE
            + Qwen38_27B::HIDDEN * Qwen38_27B::INTERMEDIATE
            + Qwen38_27B::HIDDEN;
        assert_eq!(
            report.immutable_input_values,
            2 * EXACT_ROUTES.len() * immutable
        );
        assert_eq!(
            report.immutable_descriptor_words,
            2 * EXACT_ROUTES.len() * 32
        );
        assert_eq!(report.descriptor_bytes, 256);
        let (arena, regions) = layout()?;
        assert_eq!(report.padding_bytes, 0);
        assert_eq!(report.arena_bytes, 153_106_432);
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(regions.payload_bytes(), arena.byte_len());
        // BF16 ULP grows with magnitude; `verify_eager` already checks every
        // value against the represented FP64 oracle's absolute/relative bound.
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
