//! Numerical and graph qualification for the source-native FP8 GDN output.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, Observed, SCALE_VALUES, TokenOracle,
    WEIGHT_CODES, WEIGHT_VALUES, bf16_to_f32, f32_to_bf16, quantize_oracle,
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::GdnOutputProjectionOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact FP8 GDN output qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum GdnOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Device behavior disagreed with the independent contract.
    #[error("FP8 GDN output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact GDN output batch.
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

/// Qualifies eager and captured GDN output projection at exact `B=1..=8`.
pub fn qualify_gdn_output() -> Result<GdnOutputQualification, GdnOutputQualificationError> {
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
    let input = make_input();
    let oracles = input
        .as_slice()
        .as_chunks::<{ Qwen38_27B::GDN_VALUE_ROWS }>()
        .0
        .iter()
        .map(|row| quantize_oracle(row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(GdnOutputQualificationError::Mismatch)?;
    let (weight_codes, weight_scales) = make_weights();
    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_from_host(&stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &weight_scales)?;
    let stable = addresses(&arena, regions)?;
    let mut report = GdnOutputQualification {
        activation_codes: 0,
        activation_scales: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_absolute_error: 0.0,
    };

    let op = GdnOutputProjectionOp::new(&context)?;
    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(batch, &oracles, &eager, &mut report)?;

        reset(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let rows = Qwen38_27B::HIDDEN;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * columns, ALIGNMENT)?;
    let codes = layout.reserve(MAX_BATCH * columns, ALIGNMENT)?;
    let scales = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * columns, ALIGNMENT)?;
    let weight_scales = layout.reserve(rows, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * rows, ALIGNMENT)?;
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

fn make_input() -> Vec<u16> {
    (0..MAX_BATCH * Qwen38_27B::GDN_VALUE_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::GDN_VALUE_ROWS;
            f32_to_bf16(INPUT_PATTERN[index & 7] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn make_weights() -> (Vec<u8>, Vec<u16>) {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let mut codes = vec![0; Qwen38_27B::HIDDEN * columns];
    for (row, values) in codes
        .as_mut_slice()
        .as_chunks_mut::<6_144>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[row & 3]);
    }
    let scales = (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect();
    (codes, scales)
}

fn reset(arena: &DeviceArena, stream: &tuisko_gpu::CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.scales, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
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
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: regions are aligned, non-overlapping, context-local, and cover B=8.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.codes)?,
            arena.address(regions.scales)?,
            arena.address(regions.weight_codes)?,
            arena.address(regions.weight_scales)?,
            arena.address(regions.output)?,
        )
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        codes: arena.copy_to_host(stream, regions.codes)?,
        scales: arena.copy_to_host(stream, regions.scales)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn verify_eager(
    batch: usize,
    oracles: &[TokenOracle],
    observed: &Observed,
    report: &mut GdnOutputQualification,
) -> Result<(), GdnOutputQualificationError> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let rows = Qwen38_27B::HIDDEN;
    for (token, oracle) in oracles[..batch].iter().enumerate() {
        let begin = token * columns;
        if let Some(column) = observed.codes[begin..begin + columns]
            .iter()
            .zip(&oracle.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "activation code at B={batch}, token={token}, column={column} differs"
            )));
        }
        if observed.scales[token].to_bits() != oracle.scale.to_bits() {
            return Err(GdnOutputQualificationError::Mismatch(format!(
                "activation scale at B={batch}, token={token} differs"
            )));
        }
        for row in 0..rows {
            let expected = oracle.represented_sum
                * f64::from(WEIGHT_VALUES[row & 3])
                * f64::from(oracle.scale)
                * f64::from(SCALE_VALUES[row & 3]);
            let actual = bf16_to_f32(observed.output[token * rows + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            if error > 0.0625f32.max(expected.abs() as f32 * 0.01) {
                return Err(GdnOutputQualificationError::Mismatch(format!(
                    "projection at B={batch}, token={token}, row={row}: device={actual}, oracle={expected}"
                )));
            }
        }
    }
    verify_inactive(batch, observed)?;
    report.activation_codes += batch * columns;
    report.activation_scales += batch;
    report.output_values += batch * rows;
    report.inactive_values += inactive_values(batch);
    Ok(())
}

fn verify_inactive(batch: usize, observed: &Observed) -> Result<(), GdnOutputQualificationError> {
    let code_begin = batch * Qwen38_27B::GDN_VALUE_ROWS;
    let output_begin = batch * Qwen38_27B::HIDDEN;
    if observed.codes[code_begin..]
        .iter()
        .any(|&value| value != BYTE_SENTINEL)
        || observed.scales[batch..]
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed.output[output_begin..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "B={batch} modified an inactive value"
        )));
    }
    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (Qwen38_27B::GDN_VALUE_ROWS + 1 + Qwen38_27B::HIDDEN)
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut GdnOutputQualification,
) -> Result<(), GdnOutputQualificationError> {
    let same = replay.codes == eager.codes
        && replay.output == eager.output
        && replay
            .scales
            .iter()
            .zip(&eager.scales)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits());
    if !same {
        return Err(GdnOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (Qwen38_27B::GDN_VALUE_ROWS + 1 + Qwen38_27B::HIDDEN);
    report.inactive_values += inactive_values(batch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{GdnOutputQualificationError, MAX_BATCH, Qwen38_27B, qualify_gdn_output};
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), GdnOutputQualificationError> {
        let report = qualify_gdn_output()?;
        let active = (1..=MAX_BATCH).sum::<usize>();
        let values = Qwen38_27B::GDN_VALUE_ROWS + 1 + Qwen38_27B::HIDDEN;
        assert_eq!(report.activation_codes, active * Qwen38_27B::GDN_VALUE_ROWS);
        assert_eq!(report.activation_scales, active);
        assert_eq!(report.output_values, active * Qwen38_27B::HIDDEN);
        assert_eq!(report.graph_replay_values, active * values);
        assert_eq!(
            report.inactive_values,
            2 * (0..MAX_BATCH).sum::<usize>() * values
        );
        assert!(report.maximum_absolute_error <= 0.0625);
        Ok(())
    }
}
