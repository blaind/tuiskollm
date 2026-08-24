//! Source-backed qualification for the Qwen3.8 MTP BF16 MLP.

use crate::device_benchmark;
use crate::fp8_projection_oracle::bf16_to_f32;
use crate::{DeviceBenchmarkError, target::MtpBf16MlpOp};
use std::path::Path;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen38_27B::HIDDEN;
pub(crate) const INTERMEDIATE: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const BF16_SENTINEL: u16 = 0xa5a5;
const BYTE_SENTINEL: u8 = 0xa5;
const INPUT_PATTERN: [f32; 16] = [
    0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.1875, -0.1875, 0.09375,
    -0.09375, 0.046875, -0.046875, 0.015625, -0.015625,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of the source-backed MTP BF16 MLP gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpBf16MlpQualificationError {
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
    #[error("MTP BF16 MLP qualification failed: {0}")]
    Mismatch(String),
}

/// Observable seam counts, ownership, and worst formula errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpBf16MlpQualification {
    /// Active represented BF16 SwiGLU values checked across every route.
    pub activation_values: usize,
    /// Active represented BF16 down outputs checked across every route.
    pub output_values: usize,
    /// Complete B=1 SwiGLU seam checked against source weights and f64 math.
    pub source_activation_values: usize,
    /// Complete B=1 down output checked against source weights and f64 math.
    pub source_output_values: usize,
    /// Complete mutable active state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact route extent.
    pub inactive_values: usize,
    /// Read-only input and source-weight values proved unchanged.
    pub immutable_values: usize,
    /// Exact unchanged gate/up and down source-BF16 bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input, activation, and output bytes.
    pub workspace_bytes: usize,
    /// Exact one-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute complete SwiGLU formula error.
    pub maximum_activation_error: f32,
    /// Largest absolute complete down-projection formula error.
    pub maximum_output_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) gate_up_weight: ArenaRegion<u16>,
    pub(crate) activation: ArenaRegion<u16>,
    pub(crate) down_weight: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.gate_up_weight.byte_len() + self.down_weight.byte_len()
    }

    pub(crate) fn workspace_bytes(self) -> usize {
        self.input.byte_len() + self.activation.byte_len() + self.output.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes() + self.workspace_bytes()
    }
}

pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) gate_up_weight: Vec<u16>,
    pub(crate) down_weight: Vec<u16>,
}

struct Observed {
    activation: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies the source-BF16 MTP MLP at exact `B=1..=8`.
pub fn qualify_mtp_bf16_mlp(
    root: &Path,
) -> Result<MtpBf16MlpQualification, MtpBf16MlpQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let fixture = make_fixture(
        bf16_words(bindings.gate_up_weight_bf16)?,
        bindings.down_weight.words().collect(),
    )?;
    let source_activation = source_swiglu_oracle(&fixture.input[..HIDDEN], &fixture.gate_up_weight);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = MtpBf16MlpOp::new(&context)?;
    load_immutable(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let route_reference = b1_route_references(&op, &arena, &stream, regions, &fixture)?;
    let source_output = source_projection_oracle(
        &route_reference.activation[..INTERMEDIATE],
        &fixture.down_weight,
    );
    let mut report = MtpBf16MlpQualification {
        activation_values: 0,
        output_values: 0,
        source_activation_values: 0,
        source_output_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_activation_error: 0.0,
        maximum_output_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_outputs(
            batch,
            &route_reference,
            &source_activation,
            &source_output,
            &eager,
            &mut report,
        )?;

        reset(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(MtpBf16MlpQualificationError::Mismatch(format!(
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
    let input = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;
    let gate_up_weight = layout.reserve(GATE_UP_ROWS * HIDDEN, ALIGNMENT)?;
    let activation = layout.reserve(MAX_BATCH * INTERMEDIATE, ALIGNMENT)?;
    let down_weight = layout.reserve(HIDDEN * INTERMEDIATE, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            gate_up_weight,
            activation,
            down_weight,
            output,
        },
    ))
}

fn bf16_words(bytes: &[u8]) -> Result<Vec<u16>, MtpBf16MlpQualificationError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(MtpBf16MlpQualificationError::Mismatch(
            "MTP gate/up source span has an odd byte length".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn make_fixture(
    gate_up_weight: Vec<u16>,
    down_weight: Vec<u16>,
) -> Result<Fixture, MtpBf16MlpQualificationError> {
    if gate_up_weight.len() != GATE_UP_ROWS * HIDDEN || down_weight.len() != HIDDEN * INTERMEDIATE {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "source MLP geometry differs: gate/up {}, down {}",
            gate_up_weight.len(),
            down_weight.len()
        )));
    }
    let input = (0..MAX_BATCH * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let value = INPUT_PATTERN[(index + 3 * token) & 15] * TOKEN_FACTORS[token];
            f32_to_bf16(value)
        })
        .collect();

    Ok(Fixture {
        input,
        gate_up_weight,
        down_weight,
    })
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input)?;
    arena.copy_from_host(stream, regions.gate_up_weight, &fixture.gate_up_weight)?;
    arena.copy_from_host(stream, regions.down_weight, &fixture.down_weight)
}

fn reset(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.activation, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 5]> {
    Ok([
        arena.address(regions.input)? as usize,
        arena.address(regions.gate_up_weight)? as usize,
        arena.address(regions.activation)? as usize,
        arena.address(regions.down_weight)? as usize,
        arena.address(regions.output)? as usize,
    ])
}

fn launch(
    op: &MtpBf16MlpOp,
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
            arena.address(regions.input)?,
            arena.address(regions.gate_up_weight)?,
            arena.address(regions.activation)?,
            arena.address(regions.down_weight)?,
            arena.address(regions.output)?,
        )
    }
}

fn launch_b1_row(
    op: &MtpBf16MlpOp,
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
            arena.address(regions.input)?.add(row * HIDDEN),
            arena.address(regions.gate_up_weight)?,
            arena.address(regions.activation)?.add(row * INTERMEDIATE),
            arena.address(regions.down_weight)?,
            arena.address(regions.output)?.add(row * HIDDEN),
        )
    }
}

fn b1_route_references(
    op: &MtpBf16MlpOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    _fixture: &Fixture,
) -> GpuResult<Observed> {
    reset(arena, stream, regions)?;
    for row in 0..MAX_BATCH {
        launch_b1_row(op, arena, stream, regions, row)?;
    }
    observe(arena, stream, regions)
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        activation: arena.copy_to_host(stream, regions.activation)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn source_swiglu_oracle(input: &[u16], gate_up_weight: &[u16]) -> Vec<f64> {
    let (gate, up) = gate_up_weight.split_at(INTERMEDIATE * HIDDEN);
    gate.chunks_exact(HIDDEN)
        .zip(up.chunks_exact(HIDDEN))
        .map(|(gate_row, up_row)| {
            let gate = dot(input, gate_row);
            let up = dot(input, up_row);
            gate / (1.0 + (-gate).exp()) * up
        })
        .collect()
}

fn source_projection_oracle(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(INTERMEDIATE)
        .map(|row| dot(input, row))
        .collect()
}

fn dot(left: &[u16], right: &[u16]) -> f64 {
    left.iter().zip(right).fold(0.0f64, |sum, (&x, &w)| {
        sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
    })
}

fn verify_outputs(
    batch: usize,
    route_reference: &Observed,
    source_activation: &[f64],
    source_output: &[f64],
    observed: &Observed,
    report: &mut MtpBf16MlpQualification,
) -> Result<(), MtpBf16MlpQualificationError> {
    let active_activation = batch * INTERMEDIATE;
    let active_output = batch * HIDDEN;
    if let Some(index) = observed.activation[..active_activation]
        .iter()
        .zip(&route_reference.activation[..active_activation])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "B={batch} SwiGLU differs from exact B=1 at active index {index}"
        )));
    }
    if let Some(index) = observed.output[..active_output]
        .iter()
        .zip(&route_reference.output[..active_output])
        .position(|(actual, expected)| actual != expected)
    {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "B={batch} down output differs from exact B=1 at active index {index}"
        )));
    }
    if batch == 1 {
        for (row, (&actual, &expected)) in observed.activation[..INTERMEDIATE]
            .iter()
            .zip(source_activation)
            .enumerate()
        {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            let tolerance = 0.03125f32.max(expected.abs() as f32 * 0.03);
            report.maximum_activation_error = report.maximum_activation_error.max(error);
            if !actual.is_finite() || error > tolerance {
                return Err(MtpBf16MlpQualificationError::Mismatch(format!(
                    "source SwiGLU row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        for (row, (&actual, &expected)) in observed.output[..HIDDEN]
            .iter()
            .zip(source_output)
            .enumerate()
        {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            let tolerance = 0.25f32.max(expected.abs() as f32 * 0.025);
            report.maximum_output_error = report.maximum_output_error.max(error);
            if !actual.is_finite() || error > tolerance {
                return Err(MtpBf16MlpQualificationError::Mismatch(format!(
                    "source down row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        report.source_activation_values += INTERMEDIATE;
        report.source_output_values += HIDDEN;
    }

    verify_inactive(batch, observed)?;
    report.activation_values += active_activation;
    report.output_values += active_output;
    report.inactive_values += inactive_values(batch);
    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut MtpBf16MlpQualification,
) -> Result<(), MtpBf16MlpQualificationError> {
    let active_activation = batch * INTERMEDIATE;
    let active_output = batch * HIDDEN;
    if replay.activation[..active_activation] != eager.activation[..active_activation]
        || replay.output[..active_output] != eager.output[..active_output]
    {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += active_activation + active_output;
    report.inactive_values += inactive_values(batch);
    Ok(())
}

fn verify_inactive(batch: usize, observed: &Observed) -> Result<(), MtpBf16MlpQualificationError> {
    if observed.activation[batch * INTERMEDIATE..]
        .iter()
        .any(|&value| value != BF16_SENTINEL)
        || observed.output[batch * HIDDEN..]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
    {
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "B={batch} modified an inactive MLP value"
        )));
    }
    Ok(())
}

fn inactive_values(batch: usize) -> usize {
    (MAX_BATCH - batch) * (INTERMEDIATE + HIDDEN)
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut MtpBf16MlpQualification,
) -> Result<(), MtpBf16MlpQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let gate_up_weight = arena.copy_to_host(stream, regions.gate_up_weight)?;
    let down_weight = arena.copy_to_host(stream, regions.down_weight)?;
    if input != fixture.input
        || gate_up_weight != fixture.gate_up_weight
        || down_weight != fixture.down_weight
    {
        return Err(MtpBf16MlpQualificationError::Mismatch(
            "read-only MLP input or source weight changed".to_string(),
        ));
    }
    report.immutable_values = input.len() + gate_up_weight.len() + down_weight.len();
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &MtpBf16MlpOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), MtpBf16MlpQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
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
        return Err(MtpBf16MlpQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HIDDEN, INTERMEDIATE, MAX_BATCH, layout, qualify_mtp_bf16_mlp};
    use std::path::PathBuf;

    #[test]
    fn mtp_bf16_mlp_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(HIDDEN, 5_120);
        assert_eq!(INTERMEDIATE, 17_408);
        assert_eq!(regions.weight_bytes(), 534_773_760);
        assert_eq!(regions.workspace_bytes(), 442_368);
        assert_eq!(regions.payload_bytes(), 535_216_128);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_bf16_mlp_suite_source_values_match_every_seam_route_and_graph() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_bf16_mlp(&root).expect("MTP BF16 MLP qualification");

        assert_eq!(report.activation_values, 36 * INTERMEDIATE);
        assert_eq!(report.output_values, 36 * HIDDEN);
        assert_eq!(report.source_activation_values, INTERMEDIATE);
        assert_eq!(report.source_output_values, HIDDEN);
        assert_eq!(report.graph_replay_values, 36 * (INTERMEDIATE + HIDDEN));
        assert_eq!(report.inactive_values, 2 * 28 * (INTERMEDIATE + HIDDEN));
        assert_eq!(report.immutable_values, 267_427_840);
        assert_eq!(report.weight_bytes, 534_773_760);
        assert_eq!(report.workspace_bytes, 442_368);
        assert_eq!(report.arena_bytes, 535_216_128);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_activation_error.is_finite());
        assert!(report.maximum_output_error.is_finite());
    }
}
