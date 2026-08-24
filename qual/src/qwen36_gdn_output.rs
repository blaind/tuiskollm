//! Represented-value qualification for the Qwen3.6 GDN output projection.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, WEIGHT_CODES, WEIGHT_VALUES, bf16_to_f32, decode_e4m3fn,
    encode_e4m3fn, f32_to_bf16,
};
use crate::target::Qwen36GdnOutputOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const INPUT_COLUMNS: usize = Qwen36Moe35B::GDN_VALUE_ROWS;
pub(crate) const OUTPUT_ROWS: usize = Qwen36Moe35B::HIDDEN;
pub(crate) const INPUT_SCALE: f32 = 0.125;
pub(crate) const WEIGHT_SCALE: f32 = 0.25;
const INPUT_PATTERN: [f32; 16] = [
    0.5, -0.25, 0.125, 0.0, 0.25, -0.125, 0.0625, 0.0, 0.5, -0.25, 0.125, 0.0, 0.25, -0.125,
    0.0625, 0.0,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact Qwen3.6 GDN output qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36GdnOutputQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.6 GDN output qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36GdnOutputQualification {
    /// Static E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// BF16 output values compared with the independent formula.
    pub output_values: usize,
    /// Active codes and outputs reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each exact active extent.
    pub inactive_values: usize,
    /// Read-only input and weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact source E4M3 weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input, code, and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from the independent oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) activation_codes: ArenaRegion<u8>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.activation_codes.byte_len()
            + self.weight_codes.byte_len()
            + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input_bf16: Vec<u16>,
    pub(crate) weight_codes: Vec<u8>,
    activation_codes: Vec<u8>,
    row_sums: Vec<f64>,
}

/// Qualifies eager and captured Qwen3.6 GDN outputs at exact `B=1..=8`.
pub fn qualify_qwen36_gdn_output()
-> Result<Qwen36GdnOutputQualification, Qwen36GdnOutputQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen36GdnOutputOp::new(&context)?;
    let fixture = make_fixture()?;
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen36GdnOutputQualification {
        activation_codes: 0,
        output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        fill_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        fill_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
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
    let input = layout.reserve(MAX_BATCH * INPUT_COLUMNS, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_BATCH * INPUT_COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * INPUT_COLUMNS, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            activation_codes,
            weight_codes,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 4]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.activation_codes)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(stream, regions.weight_codes, &fixture.weight_codes)
}

fn fill_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.activation_codes, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)
}

struct Observed {
    activation_codes: Vec<u8>,
    output: Vec<u16>,
}

fn read_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        activation_codes: arena.copy_to_host(stream, regions.activation_codes)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

pub(crate) fn launch(
    op: &Qwen36GdnOutputOp,
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
            arena.address(regions.activation_codes)?,
            INPUT_SCALE,
            arena.address(regions.weight_codes)?,
            WEIGHT_SCALE,
            arena.address(regions.output)?,
        )
    }
}

pub(crate) fn make_fixture() -> Result<Fixture, Qwen36GdnOutputQualificationError> {
    let input_bf16 = (0..MAX_BATCH * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token])
        })
        .collect::<Vec<_>>();
    let activation_codes = input_bf16
        .iter()
        .map(|&bits| {
            encode_e4m3fn(bf16_to_f32(bits) / INPUT_SCALE)
                .map_err(Qwen36GdnOutputQualificationError::Mismatch)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weight_codes = (0..OUTPUT_ROWS * INPUT_COLUMNS)
        .map(|index| WEIGHT_CODES[(index / INPUT_COLUMNS) & 3])
        .collect::<Vec<_>>();
    let row_sums = (0..MAX_BATCH)
        .map(|token| {
            activation_codes[token * INPUT_COLUMNS..(token + 1) * INPUT_COLUMNS]
                .iter()
                .map(|&code| {
                    decode_e4m3fn(code)
                        .map(f64::from)
                        .map_err(Qwen36GdnOutputQualificationError::Mismatch)
                })
                .sum::<Result<f64, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Fixture {
        input_bf16,
        weight_codes,
        activation_codes,
        row_sums,
    })
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen36GdnOutputQualification,
) -> Result<(), Qwen36GdnOutputQualificationError> {
    let active_codes = batch * INPUT_COLUMNS;
    if observed.activation_codes[..active_codes] != fixture.activation_codes[..active_codes] {
        let index = observed.activation_codes[..active_codes]
            .iter()
            .zip(&fixture.activation_codes[..active_codes])
            .position(|(actual, expected)| actual != expected)
            .expect("unequal slices contain one differing code");
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "B={batch} activation code {index} is {:#04x}, expected {:#04x}",
            observed.activation_codes[index], fixture.activation_codes[index]
        )));
    }
    report.activation_codes += active_codes;

    for token in 0..batch {
        for row in 0..OUTPUT_ROWS {
            let expected = fixture.row_sums[token]
                * f64::from(WEIGHT_VALUES[row & 3])
                * f64::from(INPUT_SCALE * WEIGHT_SCALE);
            let actual = f64::from(bf16_to_f32(observed.output[token * OUTPUT_ROWS + row]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
                    "B={batch} token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }
    report.output_values += batch * OUTPUT_ROWS;
    verify_inactive(batch, observed)?;
    report.inactive_values += (MAX_BATCH - batch) * (INPUT_COLUMNS + OUTPUT_ROWS);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen36GdnOutputQualification,
) -> Result<(), Qwen36GdnOutputQualificationError> {
    if eager.activation_codes != replay.activation_codes || eager.output != replay.output {
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (INPUT_COLUMNS + OUTPUT_ROWS);
    report.inactive_values += (MAX_BATCH - batch) * (INPUT_COLUMNS + OUTPUT_ROWS);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Observed,
) -> Result<(), Qwen36GdnOutputQualificationError> {
    let code_begin = batch * INPUT_COLUMNS;
    if let Some(relative) = observed.activation_codes[code_begin..]
        .iter()
        .position(|&value| value != BYTE_SENTINEL)
    {
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "B={batch} modified inactive activation code {}",
            code_begin + relative
        )));
    }
    let output_begin = batch * OUTPUT_ROWS;
    if let Some(relative) = observed.output[output_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "B={batch} modified inactive output {}",
            output_begin + relative
        )));
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen36GdnOutputQualification,
) -> Result<(), Qwen36GdnOutputQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weights = arena.copy_to_host(stream, regions.weight_codes)?;
    if input != fixture.input_bf16 || weights != fixture.weight_codes {
        return Err(Qwen36GdnOutputQualificationError::Mismatch(
            "read-only input or weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = input.len() + weights.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen36GdnOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen36GdnOutputQualificationError> {
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
        return Err(Qwen36GdnOutputQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_accounting_matches_exact_geometry() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 8_388_608);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 131_072);
        assert_eq!(layout.byte_len(), 8_519_680);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen36GdnOutputQualificationError> {
        let report = qualify_qwen36_gdn_output()?;
        let active = (1..=MAX_BATCH).sum::<usize>();
        let inactive = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.activation_codes, active * INPUT_COLUMNS);
        assert_eq!(report.output_values, active * OUTPUT_ROWS);
        assert_eq!(
            report.graph_replay_values,
            active * (INPUT_COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive * (INPUT_COLUMNS + OUTPUT_ROWS)
        );
        assert_eq!(report.immutable_input_values, 8_421_376);
        assert_eq!(report.arena_bytes, 8_519_680);
        assert_eq!(report.weight_bytes, 8_388_608);
        assert_eq!(report.workspace_bytes, 131_072);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
