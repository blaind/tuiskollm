//! Direct timing for one source-backed Qwen3.6 GDN plus MoE layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen36GdnMoeLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

const SOURCE_LAYER: usize = 0;
const HISTORY_TAPS: usize = Qwen36Moe35B::LINEAR_CONV_KERNEL_DIM - 1;
const STATE_PER_ROW: usize =
    Qwen36Moe35B::GDN_CONTROL_ROWS * Qwen36Moe35B::LINEAR_HEAD_DIM * Qwen36Moe35B::LINEAR_HEAD_DIM;
const EXPERT_SLOTS: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1;
const NVFP4_GROUP: usize = 16;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen36GdnMoeLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen36Moe35B>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = Qwen36GdnMoeLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_BATCH, &benchmark_input())?;
        program.reset_state(&stream)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                Ok(RouteGraph {
                    batch,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        batch,
                        repeated_operations,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;
        let timer = GpuTimer::new(&context)?;

        Ok(Self {
            routes,
            timer,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for batch in 1..=MAX_BATCH {
                // SAFETY: the program retains every captured layer allocation through this replay.
                unsafe {
                    self.program
                        .qualification_graph(batch)?
                        .launch(&self.stream)
                }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    fn cases(
        &self,
        repeated_operations: u64,
    ) -> Result<Vec<ExactDeviceCase<'_>>, DeviceBenchmarkError> {
        self.routes
            .iter()
            .map(|route| {
                Ok(ExactDeviceCase::new(
                    "qwen36_35b_a3b/gdn_moe/layer0",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_layer_decode(route.batch as u32),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    self.program.qualification_graph(route.batch)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
            })
            .collect()
    }
}

fn benchmark_input() -> Vec<u16> {
    const PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
    (0..MAX_BATCH * Qwen36Moe35B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen36Moe35B::HIDDEN) & 7]))
        .collect()
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let input_rows = Qwen36Moe35B::GDN_INPUT_ROWS;
    let qkv_rows = Qwen36Moe35B::GDN_QKV_ROWS;
    let value_rows = Qwen36Moe35B::GDN_VALUE_ROWS;
    let controls = Qwen36Moe35B::GDN_CONTROL_ROWS;
    let intermediate = Qwen36Moe35B::INTERMEDIATE;
    let experts = Qwen36Moe35B::NUM_EXPERTS;
    let top_k = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;

    let plain_norm = 3 * batch * hidden * size_of::<u16>();
    let input_projection_weights = input_rows * hidden + 2 * controls * hidden * size_of::<u16>();
    let input_projection = input_projection_weights
        + batch
            * (hidden * (size_of::<u16>() + size_of::<u8>())
                + (input_rows + 2 * controls) * size_of::<u16>());
    let prepare = batch
        * (4 * controls * size_of::<u16>()
            + 2 * controls * size_of::<f32>()
            + qkv_rows * size_of::<u16>()
            + qkv_rows * (HISTORY_TAPS + 1) * size_of::<u16>()
            + size_of::<u32>()
            + 2 * qkv_rows * HISTORY_TAPS * size_of::<u16>()
            + qkv_rows * size_of::<u16>());
    let recurrence = Qwen36Moe35B::LINEAR_HEAD_DIM * size_of::<u16>()
        + batch
            * (qkv_rows * size_of::<u16>()
                + value_rows * size_of::<u16>()
                + 2 * controls * size_of::<f32>()
                + size_of::<u32>()
                + 2 * STATE_PER_ROW * size_of::<f32>()
                + value_rows * size_of::<u16>());
    let output_projection = hidden * value_rows
        + batch * (value_rows * (size_of::<u16>() + size_of::<u8>()) + hidden * size_of::<u16>());
    let residual_boundaries = 2 * 5 * batch * hidden * size_of::<u16>();
    let router = experts * hidden * size_of::<u16>()
        + batch
            * (hidden * size_of::<u16>()
                + experts * size_of::<u16>()
                + 2 * top_k * size_of::<u16>());
    let gate_up_codes = 2 * intermediate * hidden / 2;
    let gate_up_scales = 2 * intermediate * hidden / NVFP4_GROUP;
    let down_codes = hidden * intermediate / 2;
    let down_scales = hidden * intermediate / NVFP4_GROUP;
    let selected_expert_weights =
        batch * EXPERT_SLOTS * (gate_up_codes + gate_up_scales + down_codes + down_scales);
    let experts_path = selected_expert_weights
        + batch
            * (hidden * size_of::<u16>()
                + 2 * top_k * size_of::<u16>()
                + hidden * size_of::<u16>()
                + 2 * EXPERT_SLOTS * intermediate * size_of::<u16>()
                + 2 * EXPERT_SLOTS * hidden * size_of::<u16>()
                + hidden * size_of::<u16>());

    plain_norm
        + input_projection
        + prepare
        + recurrence
        + output_projection
        + residual_boundaries
        + router
        + experts_path
}

/// Measures every exact graph of one source-backed Qwen3.6 GDN/MoE layer.
pub fn benchmark_qwen36_gdn_moe_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_moe/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=0 source-native parameters and losslessly materialized NVFP4 planes",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_moe/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 including causal history, FP32 state, routing, and nine expert slots",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_moe/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.workspace_bytes(),
        "single 256-byte-aligned arena",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen36-gdn-moe-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust submission/completion, repeated production graph, and repeated-operation graph",
        },
        preflight,
        baseline_sha256,
        options,
        metrics,
        energy_metrics,
        telemetry,
        memory,
    )
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_covers_every_composed_leaf_and_selected_expert() {
        assert_eq!(logical_bytes(1), 55_424_712);
        assert_eq!(logical_bytes(MAX_BATCH), 199_339_840);
    }
}
