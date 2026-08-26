//! Direct timing for one source-backed dense-FP8 GDN layer owner.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{DenseFp8GdnLayerProgram, MAX_BATCH};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const SOURCE_LAYER: usize = 60;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];

struct RouteGraph {
    rows: usize,
    preparation: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: DenseFp8GdnLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = DenseFp8GdnLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
        program.reset_state(&stream)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| {
                Ok(RouteGraph {
                    rows,
                    preparation: program.qualification_state_reset_graph(&stream)?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;
        Ok(Self {
            routes,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: this Session owns both the preparation graphs and the
                // program whose arena they captured, dropping the graphs first.
                unsafe { route.preparation.launch(&self.stream) }?;
                let graph = self.program.qualification_graph(route.rows)?;
                // SAFETY: this Session's program owns the graph and every
                // allocation it captured, outliving the replay and synchronize.
                unsafe { graph.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    fn cases(&self) -> Result<Vec<ExactDeviceCase<'_>>, DeviceBenchmarkError> {
        self.routes
            .iter()
            .map(|route| {
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_layer_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_layer_prefill(route.rows as u64),
                    )
                };
                Ok(ExactDeviceCase::new(
                    "dense_fp8_gdn_layer/layer60",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    self.program.qualification_graph(route.rows)?,
                    None,
                )
                .with_preparation(&route.preparation))
            })
            .collect()
    }
}

fn benchmark_input() -> Vec<u16> {
    const PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
    (0..MAX_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen38_27B::HIDDEN) & 7]))
        .collect()
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let input_rows = Qwen38_27B::GDN_INPUT_ROWS;
    let qkv_rows = Qwen38_27B::GDN_QKV_ROWS;
    let value_rows = Qwen38_27B::GDN_VALUE_ROWS;
    let controls = Qwen38_27B::GDN_CONTROL_ROWS;
    let state = controls * Qwen38_27B::LINEAR_HEAD_DIM * Qwen38_27B::LINEAR_HEAD_DIM;
    let weights = 383_949_248;
    // Logical traffic follows every production kernel plane once. The control
    // matrices reread the normalized row for each of 96 outputs, and recurrence
    // accounts for both reads and writes of the complete FP32 state.
    let per_token = 38 * hidden
        + 2 * input_rows
        + 4 * controls * hidden
        + 16 * controls
        + 16 * qkv_rows
        + 8 * state
        + 16 * value_rows
        + 6 * intermediate
        + 2 * size_of::<f32>() * 4;
    weights + rows * per_token
}

/// Measures every exact graph of one source-backed dense-FP8 GDN layer owner.
pub fn benchmark_dense_fp8_gdn_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "dense_fp8_gdn_layer/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=60 source-native mixer and dense-FP8 MLP",
    )?;
    memory.register_owned(
        "dense_fp8_gdn_layer/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 plus eight causal-history and FP32-state rows",
    )?;
    memory.register_owned(
        "dense_fp8_gdn_layer/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.workspace_bytes(),
        "single 256-byte-aligned arena",
    )?;
    memory.register_owned(
        "dense_fp8_gdn_layer/address_bound_tensor_maps",
        BenchmarkMemoryKind::Other,
        session.program.descriptor_bytes(),
        "four 128-byte gate/up and down tensor maps",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases()?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-dense-fp8-gdn-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust production-graph submission/completion after an untimed exact-state reset",
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

#[cfg(test)]
mod tests {
    use super::{EXACT_ROUTES, MAX_BATCH, MAX_ROWS, logical_bytes};

    #[test]
    fn byte_accounting_scales_only_per_token_traffic() {
        let one = logical_bytes(1);
        let per_token = logical_bytes(2) - one;
        assert_eq!(per_token, 7_869_216);
        assert_eq!(one, 391_818_464);
        assert_eq!(logical_bytes(MAX_BATCH), one + (MAX_BATCH - 1) * per_token);
        assert_eq!(logical_bytes(MAX_ROWS), one + (MAX_ROWS - 1) * per_token);
    }

    #[test]
    fn benchmark_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
    }
}
