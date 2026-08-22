//! Numerical and graph qualification for dense-FP8 gate/up SwiGLU.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, SCALE_VALUES, TokenOracle, WEIGHT_CODES,
    WEIGHT_VALUES, bf16_to_f32, f32_to_bf16, quantize_oracle,
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::DenseFp8SwiGluOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_ROWS: usize = 128;
const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.0,
    0.5, -0.25, 0.125,
];
const TOKEN_FACTORS: [f32; 8] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact dense-FP8 SwiGLU qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Fp8SwiGluQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent represented-value contract.
    #[error("dense-FP8 SwiGLU qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from all exact dense-FP8 SwiGLU routes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8SwiGluQualification {
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// BF16 fused SwiGLU values compared with the represented-value oracle.
    pub output_values: usize,
    /// Active values reproduced by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active row extent.
    pub inactive_values: usize,
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

struct Observed {
    codes: Vec<u8>,
    scales: Vec<f32>,
    output: Vec<u16>,
}

/// Qualifies eager and captured dense-FP8 SwiGLU at every admitted row count.
pub fn qualify_fp8_swiglu() -> Result<Fp8SwiGluQualification, Fp8SwiGluQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = DenseFp8SwiGluOp::new(&context)?;
    let input = make_input();
    let token_oracles = input
        .as_slice()
        .as_chunks::<{ Qwen38_27B::HIDDEN }>()
        .0
        .iter()
        .map(|row| quantize_oracle(row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Fp8SwiGluQualificationError::Mismatch)?;
    let (weight_codes, weight_scales) = make_weights();

    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Fp8SwiGluQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify_eager(rows, &token_oracles, &eager, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Fp8SwiGluQualificationError::Mismatch(format!(
                "device addresses changed while qualifying row count {rows}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(2 * intermediate * hidden, ALIGNMENT)?;
    let weight_scales = layout.reserve(2 * intermediate, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * intermediate, ALIGNMENT)?;

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
    (0..MAX_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token & 7])
        })
        .collect()
}

fn make_weights() -> (Vec<u8>, Vec<u16>) {
    let rows = 2 * Qwen38_27B::INTERMEDIATE;
    let mut codes = vec![0; rows * Qwen38_27B::HIDDEN];
    for (row, values) in codes
        .as_mut_slice()
        .as_chunks_mut::<{ Qwen38_27B::HIDDEN }>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[(row + usize::from(row >= Qwen38_27B::INTERMEDIATE)) & 3]);
    }
    let scales = (0..rows)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect();

    (codes, scales)
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
    op: &DenseFp8SwiGluOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every arena region is aligned, non-overlapping, context-local,
    // and covers the maximum extent admitted by all exact routes.
    unsafe {
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
    active_rows: usize,
    oracles: &[TokenOracle],
    observed: &Observed,
    report: &mut Fp8SwiGluQualification,
) -> Result<(), Fp8SwiGluQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;

    for (token, oracle) in oracles[..active_rows].iter().enumerate() {
        let code_begin = token * hidden;
        if let Some(relative) = observed.codes[code_begin..code_begin + hidden]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Fp8SwiGluQualificationError::Mismatch(format!(
                "activation code at rows={active_rows}, row={token}, column={relative}: device={:#04x}, oracle={:#04x}",
                observed.codes[code_begin + relative],
                oracle.codes[relative]
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(Fp8SwiGluQualificationError::Mismatch(format!(
                "activation scale at rows={active_rows}, row={token}: device={:#010x}, oracle={:#010x}",
                observed.scales[token].to_bits(),
                oracle.scale.to_bits()
            )));
        }

        for row in 0..intermediate {
            let gate_pattern = row & 3;
            let up_source_row = row + intermediate;
            let up_pattern = (up_source_row + 1) & 3;
            let gate = oracle.represented_sum
                * f64::from(oracle.scale)
                * f64::from(WEIGHT_VALUES[gate_pattern])
                * f64::from(SCALE_VALUES[gate_pattern]);
            let up = oracle.represented_sum
                * f64::from(oracle.scale)
                * f64::from(WEIGHT_VALUES[up_pattern])
                * f64::from(SCALE_VALUES[up_source_row & 3]);
            let expected = gate / (1.0 + (-gate).exp()) * up;
            let actual = bf16_to_f32(observed.output[token * intermediate + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = 0.125f32.max(expected.abs() as f32 * 0.015);
            if error > tolerance {
                return Err(Fp8SwiGluQualificationError::Mismatch(format!(
                    "SwiGLU at rows={active_rows}, row={token}, output={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(active_rows, observed)?;
    report.activation_codes += active_rows * hidden;
    report.activation_scales += active_rows;
    report.output_values += active_rows * intermediate;
    report.inactive_values += inactive_values(active_rows);

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Fp8SwiGluQualification,
) -> Result<(), Fp8SwiGluQualificationError> {
    if let Some(index) = replay
        .codes
        .iter()
        .zip(&eager.codes)
        .position(|(a, b)| a != b)
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} graph activation code {index} differs"
        )));
    }
    if let Some(index) = replay
        .scales
        .iter()
        .zip(&eager.scales)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} graph activation scale {index} differs"
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(a, b)| a != b)
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} graph output {index} differs"
        )));
    }

    verify_inactive(rows, replay)?;
    report.graph_replay_values += rows * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::INTERMEDIATE);
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_inactive(rows: usize, observed: &Observed) -> Result<(), Fp8SwiGluQualificationError> {
    let code_begin = rows * Qwen38_27B::HIDDEN;
    if let Some(relative) = observed.codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} modified inactive activation code {}",
            code_begin + relative
        )));
    }
    if let Some(relative) = observed.scales[rows..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} modified inactive activation scale {}",
            rows + relative
        )));
    }
    let output_begin = rows * Qwen38_27B::INTERMEDIATE;
    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Fp8SwiGluQualificationError::Mismatch(format!(
            "rows={rows} modified inactive output {}",
            output_begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::INTERMEDIATE)
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, Fp8SwiGluQualificationError, MAX_ROWS, Qwen38_27B, qualify_fp8_swiglu,
    };
    use crate::fp8_projection_oracle::verify_host_codecs;
    use tuisko_model::Arch;

    #[test]
    fn host_codecs_pin_bf16_and_e4m3_rounding() {
        verify_host_codecs().unwrap();
    }

    #[test]
    #[ignore = "requires an idle NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Fp8SwiGluQualificationError> {
        let report = qualify_fp8_swiglu()?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        let active_per_run = Qwen38_27B::HIDDEN + 1 + Qwen38_27B::INTERMEDIATE;
        let inactive_per_pass = EXACT_ROUTES
            .iter()
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>()
            * active_per_run;

        assert_eq!(report.activation_codes, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.activation_scales, active_rows);
        assert_eq!(report.output_values, active_rows * Qwen38_27B::INTERMEDIATE);
        assert_eq!(report.graph_replay_values, active_rows * active_per_run);
        assert_eq!(report.inactive_values, inactive_per_pass * 2);
        assert!(report.maximum_absolute_error <= 0.125);

        Ok(())
    }
}
