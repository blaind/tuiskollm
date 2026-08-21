//! Numerical and graph qualification for the admitted full-attention FP8 QKV routes.

use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::FullAttentionQkvOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 16;
const EXACT_ROUTES: [usize; MAX_BATCH + 1] = [1, 2, 3, 4, 5, 6, 7, 8, 16];
const ALIGNMENT: usize = 256;
const BYTE_SENTINEL: u8 = 0xa5;
const BF16_SENTINEL: u16 = 0xa5a5;
const F32_SENTINEL_BITS: u32 = 0xa5a5a5a5;
const FP8_MAX: f32 = 448.0;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.0,
    0.5, -0.25, 0.125,
];
const TOKEN_FACTORS: [f32; MAX_ROWS] = [
    1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125, 0.75, -0.75, 0.375, -0.375, 0.1875, -0.1875,
    0.09375, -0.09375,
];
const WEIGHT_CODES: [u8; 4] = [0x38, 0xb0, 0x30, 0x28];
const WEIGHT_VALUES: [f32; 4] = [1.0, -0.5, 0.5, 0.25];
const SCALE_VALUES: [f32; 4] = [1.0, 0.5, 0.25, 2.0];

/// Failure of the exact full-attention FP8 QKV qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Fp8QkvQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("FP8 QKV qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from all exact FP8 QKV routes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8QkvQualification {
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// QKV BF16 values compared with the represented-value oracle.
    pub output_values: usize,
    /// Active codes, scales, and outputs reproduced by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each exact active-row extent.
    pub inactive_values: usize,
    /// Largest absolute QKV difference from the FP64 oracle.
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

struct TokenOracle {
    codes: Vec<u8>,
    scale: f32,
    represented_sum: f64,
}

struct Observed {
    codes: Vec<u8>,
    scales: Vec<f32>,
    output: Vec<u16>,
}

/// Qualifies eager and captured full-attention QKV at exact `B=1..=8` and `T=16`.
pub fn qualify_fp8_qkv() -> Result<Fp8QkvQualification, Fp8QkvQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = FullAttentionQkvOp::new(&context)?;
    let input = make_input();
    let token_oracles = input
        .chunks_exact(Qwen38_27B::HIDDEN)
        .map(quantize_oracle)
        .collect::<Result<Vec<_>, _>>()?;
    let (weight_codes, weight_scales) = make_weights();

    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Fp8QkvQualification {
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
            return Err(Fp8QkvQualificationError::Mismatch(format!(
                "device addresses changed while qualifying row count {rows}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let rows = Qwen38_27B::ATTENTION_QKV_ROWS;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * hidden, ALIGNMENT)?;
    let weight_scales = layout.reserve(rows, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * rows, ALIGNMENT)?;

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
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn make_weights() -> (Vec<u8>, Vec<u16>) {
    let rows = Qwen38_27B::ATTENTION_QKV_ROWS;
    let mut codes = vec![0; rows * Qwen38_27B::HIDDEN];
    for (row, values) in codes
        .as_mut_slice()
        .as_chunks_mut::<{ Qwen38_27B::HIDDEN }>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[row & 3]);
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
    op: &FullAttentionQkvOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let activation_codes = arena.address(regions.activation_codes)?;
    let activation_scales = arena.address(regions.activation_scales)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the arena regions are 256-byte aligned, non-overlapping, context-local,
    // and cover the maximum extents admitted by every exact route.
    unsafe {
        op.launch(
            stream,
            rows,
            input,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
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
    token_oracles: &[TokenOracle],
    observed: &Observed,
    report: &mut Fp8QkvQualification,
) -> Result<(), Fp8QkvQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let output_rows = Qwen38_27B::ATTENTION_QKV_ROWS;

    for (token, oracle) in token_oracles[..active_rows].iter().enumerate() {
        let code_begin = token * hidden;
        let code_end = code_begin + hidden;
        if let Some(relative) = observed.codes[code_begin..code_end]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            let column = relative;
            return Err(Fp8QkvQualificationError::Mismatch(format!(
                "activation code at rows={active_rows}, row={token}, column={column}: device={:#04x}, oracle={:#04x}",
                observed.codes[code_begin + relative],
                oracle.codes[relative]
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(Fp8QkvQualificationError::Mismatch(format!(
                "activation scale at rows={active_rows}, row={token}: device={:#010x}, oracle={:#010x}",
                observed.scales[token].to_bits(),
                oracle.scale.to_bits()
            )));
        }

        for row in 0..output_rows {
            let expected = oracle.represented_sum
                * f64::from(WEIGHT_VALUES[row & 3])
                * f64::from(oracle.scale)
                * f64::from(SCALE_VALUES[row & 3]);
            let actual = bf16_to_f32(observed.output[token * output_rows + row]);
            let error = (actual as f64 - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = 0.0625f32.max(expected.abs() as f32 * 0.01);
            if error > tolerance {
                return Err(Fp8QkvQualificationError::Mismatch(format!(
                    "projection at rows={active_rows}, row={token}, output={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(active_rows, observed)?;
    report.activation_codes += active_rows * hidden;
    report.activation_scales += active_rows;
    report.output_values += active_rows * output_rows;
    report.inactive_values += inactive_values(active_rows);

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Fp8QkvQualification,
) -> Result<(), Fp8QkvQualificationError> {
    if let Some(index) = replay
        .codes
        .iter()
        .zip(&eager.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} graph activation code {index} differs: replay={:#04x}, eager={:#04x}",
            replay.codes[index], eager.codes[index]
        )));
    }
    if let Some(index) = replay
        .scales
        .iter()
        .zip(&eager.scales)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} graph activation scale {index} differs: replay={:#010x}, eager={:#010x}",
            replay.scales[index].to_bits(),
            eager.scales[index].to_bits()
        )));
    }
    if let Some(index) = replay
        .output
        .iter()
        .zip(&eager.output)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} graph output {index} differs: replay={:#06x}, eager={:#06x}",
            replay.output[index], eager.output[index]
        )));
    }

    verify_inactive(rows, replay)?;
    report.graph_replay_values += rows * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::ATTENTION_QKV_ROWS);
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_inactive(rows: usize, observed: &Observed) -> Result<(), Fp8QkvQualificationError> {
    let code_begin = rows * Qwen38_27B::HIDDEN;
    if let Some(relative) = observed.codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} modified inactive activation code {}",
            code_begin + relative
        )));
    }
    if let Some(relative) = observed.scales[rows..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} modified inactive activation scale {}",
            rows + relative
        )));
    }
    let output_begin = rows * Qwen38_27B::ATTENTION_QKV_ROWS;
    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Fp8QkvQualificationError::Mismatch(format!(
            "rows={rows} modified inactive output {}",
            output_begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::ATTENTION_QKV_ROWS)
}

fn quantize_oracle(input: &[u16]) -> Result<TokenOracle, Fp8QkvQualificationError> {
    let represented = input
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let maximum = represented
        .iter()
        .fold(0.0f32, |current, value| current.max(value.abs()));
    let scale = if maximum == 0.0 {
        1.0
    } else {
        maximum / FP8_MAX
    };
    let codes = represented
        .iter()
        .map(|&value| encode_e4m3fn(value / scale))
        .collect::<Result<Vec<_>, _>>()?;
    let represented_sum = codes
        .iter()
        .map(|&code| f64::from(decode_e4m3fn(code).expect("encoder emitted finite E4M3")))
        .sum();

    Ok(TokenOracle {
        codes,
        scale,
        represented_sum,
    })
}

fn encode_e4m3fn(value: f32) -> Result<u8, Fp8QkvQualificationError> {
    if !value.is_finite() {
        return Err(Fp8QkvQualificationError::Mismatch(
            "oracle E4M3 input is not finite".to_string(),
        ));
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() { 0x80 } else { 0x00 });
    }

    let value = value.clamp(-FP8_MAX, FP8_MAX);
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=u8::MAX {
        if matches!(code, 0x7f | 0xff) {
            continue;
        }
        let represented = decode_e4m3fn(code).expect("NaN codes were excluded");
        if represented == 0.0 && represented.is_sign_negative() != value.is_sign_negative() {
            continue;
        }
        let distance = (value - represented).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    Ok(best)
}

fn decode_e4m3fn(code: u8) -> Result<f32, Fp8QkvQualificationError> {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let fraction = code & 0x07;
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0.0,
        (0, fraction) => f32::from(fraction) * 2.0f32.powi(-9),
        (15, 7) => {
            return Err(Fp8QkvQualificationError::Mismatch(
                "oracle encountered an E4M3FN NaN code".to_string(),
            ));
        }
        (exponent, fraction) => {
            (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
        }
    };

    Ok(sign * magnitude)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, Fp8QkvQualificationError, MAX_ROWS, Qwen38_27B, bf16_to_f32, decode_e4m3fn,
        encode_e4m3fn, f32_to_bf16, qualify_fp8_qkv,
    };
    use tuisko_model::Arch;

    #[test]
    fn host_codecs_pin_bf16_and_e4m3_rounding() {
        assert_eq!(f32_to_bf16(1.0 + 0.00390625), 0x3f80);
        assert_eq!(f32_to_bf16(bf16_to_f32(0x3f81) + 0.00390625), 0x3f82);
        assert_eq!(encode_e4m3fn(1.0).unwrap(), 0x38);
        assert_eq!(encode_e4m3fn(-0.5).unwrap(), 0xb0);
        assert_eq!(decode_e4m3fn(0x7e).unwrap(), 448.0);
        assert!(decode_e4m3fn(0x7f).is_err());
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Fp8QkvQualificationError> {
        let report = qualify_fp8_qkv()?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        let active_per_run = Qwen38_27B::HIDDEN + 1 + Qwen38_27B::ATTENTION_QKV_ROWS;
        let inactive_per_pass = EXACT_ROUTES
            .iter()
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>()
            * active_per_run;

        assert_eq!(report.activation_codes, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.activation_scales, active_rows);
        assert_eq!(
            report.output_values,
            active_rows * Qwen38_27B::ATTENTION_QKV_ROWS
        );
        assert_eq!(report.graph_replay_values, active_rows * active_per_run);
        assert_eq!(report.inactive_values, inactive_per_pass * 2);
        assert!(report.maximum_absolute_error <= 0.0625);

        Ok(())
    }
}
