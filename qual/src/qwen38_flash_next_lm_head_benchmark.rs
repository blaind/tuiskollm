//! Paired timings for every exact Qwen3.8-Flash-Next BF16 LM-head decode route.
//!
//! Every route reads the whole untied vocabulary plane once, so the accounting
//! is dominated by 1,271,398,400 weight bytes and the row count only moves the
//! stream in and the logits out.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen38_flash_next_lm_head::{
    EXACT_ROUTES, HIDDEN, Regions, VOCAB, launch, layout, make_fixture,
};
use crate::target::Qwen38FlashNextBf16LmHeadOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen38FlashNextBf16LmHeadOp,
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
        let fixture = make_fixture();
        arena.copy_from_host(&stream, regions.input, &fixture.input)?;
        arena.copy_from_host(&stream, regions.weight, &fixture.weight)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen38FlashNextBf16LmHeadOp::new(&context)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| capture_route(&op, &arena, &stream, regions, rows, repeated_operations))
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
                // SAFETY: this session retains every captured allocation
                // through the synchronized replay below.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    "qwen38_flash_next/lm_head/bf16",
                    format!("B={}", route.rows),
                    BenchmarkWorkload::warm_operator_decode(route.rows as u32),
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen38FlashNextBf16LmHeadOp,
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
    let weight = VOCAB * HIDDEN * size_of::<u16>();
    let input = rows * HIDDEN * size_of::<u16>();
    let logits = rows * VOCAB * size_of::<u16>();

    weight + input + logits
}

/// Measures every exact Qwen3.8-Flash-Next BF16 LM-head decode route.
pub fn benchmark_qwen38_flash_next_lm_head(
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
        "qwen38_flash_next/lm_head/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "untied source BF16 [248320,2560] vocabulary matrix",
    )?;
    memory.register_owned(
        "qwen38_flash_next/lm_head/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "B=8 collapsed stream rows and their BF16 logits",
    )?;
    memory.register_owned(
        "qwen38_flash_next/lm_head/alignment_padding",
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
            suite: "bench-qwen38-flash-next-lm-head",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, repeated leaf graph, and repeated-operation graph",
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
    use super::{EXACT_ROUTES, HIDDEN, VOCAB, layout, logical_bytes};
    use crate::qwen38_flash_next_lm_head::MAX_BATCH;

    /// The head is weight-bound: even at the widest decode batch the stream and
    /// the logits together are under a third of a percent of the weight plane.
    #[test]
    fn qwen38_flash_next_lm_head_benchmark_byte_accounting_is_dominated_by_the_vocabulary_plane() {
        let weight = VOCAB * HIDDEN * size_of::<u16>();

        assert_eq!(weight, 1_271_398_400);
        assert_eq!(logical_bytes(0), weight);
        assert_eq!(
            logical_bytes(MAX_BATCH),
            weight + MAX_BATCH * (HIDDEN + VOCAB) * size_of::<u16>()
        );
        assert!(logical_bytes(MAX_BATCH) - weight < weight / 300);
    }

    #[test]
    fn qwen38_flash_next_lm_head_benchmark_inventory_and_accounting_are_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(regions.weight_bytes(), 1_271_398_400);
        assert_eq!(regions.workspace_bytes(), 4_014_080);
        // Every reserved byte is a payload byte the accounting names.
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(layout.byte_len(), 1_275_412_480);
    }
}
