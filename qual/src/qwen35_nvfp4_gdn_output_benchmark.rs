//! Paired timings for exact Qwen3.5 recurrent-output projection routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_nvfp4_attention_output::{
    CODE_BYTES_PER_ROW, COLUMNS, EXACT_ROUTES, GROUPS_PER_ROW, MAX_BATCH, OUTPUT_ROWS, make_fixture,
};
use crate::qwen35_nvfp4_gdn_output::{Regions, launch, layout, upload_fixture};
use crate::target::Qwen35Nvfp4GdnOutputOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen35Nvfp4GdnOutputOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }

        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let fixture = make_fixture().map_err(|error| {
            DeviceBenchmarkError::Precondition(format!(
                "Qwen3.5 GDN-output fixture construction failed: {error}"
            ))
        })?;
        upload_fixture(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = Qwen35Nvfp4GdnOutputOp::new(&context)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| capture_route(&op, &arena, &stream, regions, rows, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _op: op,
            arena,
            regions,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_operator_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_operator_prefill(route.rows as u64),
                    )
                };
                ExactDeviceCase::new(
                    "qwen35_9b/gdn_output/nvfp4_projection",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen35Nvfp4GdnOutputOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, arena, stream, regions, rows)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        leaf,
        repeated,
    })
}

fn logical_bytes(rows: usize) -> usize {
    let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let per_token = (COLUMNS + OUTPUT_ROWS) * size_of::<u16>();
    let scratch = if rows > MAX_BATCH {
        2 * rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    } else {
        0
    };

    weights + rows * per_token + scratch
}

/// Measures every exact Qwen3.5 recurrent-output NVFP4 projection.
pub fn benchmark_qwen35_nvfp4_gdn_output(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen35_9b/gdn_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "packed output weights plus swizzled block scales",
    )?;
    memory.register_owned(
        "qwen35_9b/gdn_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=128 activation, quantization scratch, and output rows",
    )?;
    memory.register_owned(
        "qwen35_9b/gdn_output/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-nvfp4-gdn-output",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph",
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
    use super::*;

    #[test]
    fn accounting_covers_the_complete_projection() {
        let weights = 9_437_184;

        assert_eq!(logical_bytes(1), weights + 16_384);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * 16_384);
        assert_eq!(logical_bytes(32), weights + 32 * 20_992);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(
            crate::qwen35_nvfp4_attention_output::WEIGHT_SCALE_DIVISOR,
            16.0
        );
    }
}
