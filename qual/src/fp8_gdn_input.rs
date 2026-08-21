//! Numerical and graph qualification for exact FP8 GDN input projection batches.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, Observed, SCALE_VALUES, TokenOracle,
    WEIGHT_CODES, WEIGHT_VALUES, bf16_to_f32, f32_to_bf16, quantize_oracle,
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::GdnInputProjectionOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 16] = [
    0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.0,
    0.5, -0.25, 0.125,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact FP8 GDN input qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Fp8GdnInputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("FP8 GDN input qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact GDN input batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8GdnInputQualification {
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// GDN Q/K/V/Z BF16 values compared with the represented-value oracle.
    pub output_values: usize,
    /// Active codes, scales, and outputs reproduced by captured graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each exact batch.
    pub inactive_values: usize,
    /// Largest absolute projection difference from the FP64 oracle.
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

/// Qualifies eager and captured GDN Q/K/V/Z projection at exact `B=1..=8`.
pub fn qualify_fp8_gdn_input() -> Result<Fp8GdnInputQualification, Fp8GdnInputQualificationError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = GdnInputProjectionOp::new(&context)?;
    let input = make_input();
    let token_oracles = input
        .chunks_exact(Qwen38_27B::HIDDEN)
        .map(quantize_oracle)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Fp8GdnInputQualificationError::Mismatch)?;
    let (weight_codes, weight_scales) = make_weights();

    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Fp8GdnInputQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = read_observed(&arena, &stream, regions)?;
        verify_eager(batch, &token_oracles, &eager, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = read_observed(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Fp8GdnInputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let rows = Qwen38_27B::GDN_INPUT_ROWS;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_BATCH * hidden, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * hidden, ALIGNMENT)?;
    let weight_scales = layout.reserve(rows, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * rows, ALIGNMENT)?;

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
    (0..MAX_BATCH * Qwen38_27B::HIDDEN)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn make_weights() -> (Vec<u8>, Vec<u16>) {
    let rows = Qwen38_27B::GDN_INPUT_ROWS;
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
    op: &GdnInputProjectionOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    let input = arena.address(regions.input)?;
    let activation_codes = arena.address(regions.activation_codes)?;
    let activation_scales = arena.address(regions.activation_scales)?;
    let weight_codes = arena.address(regions.weight_codes)?;
    let weight_scales = arena.address(regions.weight_scales)?;
    let output = arena.address(regions.output)?;

    // SAFETY: the arena regions are aligned, non-overlapping, context-local, and
    // cover the maximum extents admitted by every exact-B route.
    unsafe {
        op.launch(
            stream,
            batch,
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
    batch: usize,
    token_oracles: &[TokenOracle],
    observed: &Observed,
    report: &mut Fp8GdnInputQualification,
) -> Result<(), Fp8GdnInputQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let output_rows = Qwen38_27B::GDN_INPUT_ROWS;

    for (token, oracle) in token_oracles[..batch].iter().enumerate() {
        let code_begin = token * hidden;
        let code_end = code_begin + hidden;
        if let Some(relative) = observed.codes[code_begin..code_end]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Fp8GdnInputQualificationError::Mismatch(format!(
                "activation code at B={batch}, row={token}, column={relative}: device={:#04x}, oracle={:#04x}",
                observed.codes[code_begin + relative],
                oracle.codes[relative]
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(Fp8GdnInputQualificationError::Mismatch(format!(
                "activation scale at B={batch}, row={token}: device={:#010x}, oracle={:#010x}",
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
                return Err(Fp8GdnInputQualificationError::Mismatch(format!(
                    "projection at B={batch}, row={token}, output={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    verify_inactive(batch, observed)?;
    report.activation_codes += batch * hidden;
    report.activation_scales += batch;
    report.output_values += batch * output_rows;
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Fp8GdnInputQualification,
) -> Result<(), Fp8GdnInputQualificationError> {
    if let Some(index) = replay
        .codes
        .iter()
        .zip(&eager.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} graph activation code {index} differs: replay={:#04x}, eager={:#04x}",
            replay.codes[index], eager.codes[index]
        )));
    }
    if let Some(index) = replay
        .scales
        .iter()
        .zip(&eager.scales)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} graph activation scale {index} differs: replay={:#010x}, eager={:#010x}",
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
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} graph output {index} differs: replay={:#06x}, eager={:#06x}",
            replay.output[index], eager.output[index]
        )));
    }

    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::GDN_INPUT_ROWS);
    report.inactive_values += inactive_values(batch);

    Ok(())
}

fn verify_inactive(batch: usize, observed: &Observed) -> Result<(), Fp8GdnInputQualificationError> {
    let code_begin = batch * Qwen38_27B::HIDDEN;
    if let Some(relative) = observed.codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} modified inactive activation code {}",
            code_begin + relative
        )));
    }
    if let Some(relative) = observed.scales[batch..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} modified inactive activation scale {}",
            batch + relative
        )));
    }
    let output_begin = batch * Qwen38_27B::GDN_INPUT_ROWS;
    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Fp8GdnInputQualificationError::Mismatch(format!(
            "B={batch} modified inactive output {}",
            output_begin + relative
        )));
    }

    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (Qwen38_27B::HIDDEN + 1 + Qwen38_27B::GDN_INPUT_ROWS)
}

#[cfg(test)]
mod tests {
    use super::{Fp8GdnInputQualificationError, MAX_BATCH, Qwen38_27B, qualify_fp8_gdn_input};
    use crate::fp8_projection_oracle::verify_host_codecs;
    use tuisko_model::Arch;

    #[test]
    fn host_codecs_pin_bf16_and_e4m3_rounding() {
        verify_host_codecs().unwrap();
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Fp8GdnInputQualificationError> {
        let report = qualify_fp8_gdn_input()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let active_per_run = Qwen38_27B::HIDDEN + 1 + Qwen38_27B::GDN_INPUT_ROWS;
        let inactive_per_pass = (0..MAX_BATCH).sum::<usize>() * active_per_run;

        assert_eq!(report.activation_codes, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.activation_scales, active_rows);
        assert_eq!(
            report.output_values,
            active_rows * Qwen38_27B::GDN_INPUT_ROWS
        );
        assert_eq!(report.graph_replay_values, active_rows * active_per_run);
        assert_eq!(report.inactive_values, inactive_per_pass * 2);
        assert!(report.maximum_absolute_error <= 0.0625);

        Ok(())
    }
}
