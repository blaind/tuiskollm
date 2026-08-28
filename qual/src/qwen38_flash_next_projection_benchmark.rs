//! Paired timings for every exact Qwen3.8-Flash-Next backbone projection route.
//!
//! These four shapes are the whole per-token weight stream of a QSA decoder layer
//! outside the expert pool, so the accounting names the weight plane every route
//! reads in full and the activation planes it moves across it.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen38_flash_next_projection::{
    BlockOutputShape, EXACT_ROUTES, GdnInputShape, IndexerQkShape, MAX_BATCH, ProjectionShape,
    QsaQkvShape, Regions, launch, layout, make_fixture,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

/// One shape's prepared owner, arena, and captured routes.
struct ShapeSession<S: ProjectionShape> {
    routes: Vec<RouteGraphs>,
    _op: S::Op,
    arena: DeviceArena,
    regions: Regions,
}

impl<S: ProjectionShape> ShapeSession<S> {
    fn new(
        context: &Arc<CudaContext>,
        stream: &CudaStream,
        repeated_operations: u64,
    ) -> Result<Self, DeviceBenchmarkError> {
        let (layout, regions) = layout::<S>()?;
        let arena = DeviceArena::zeroed(stream, &layout)?;
        let fixture = make_fixture::<S>();
        arena.copy_from_host(stream, regions.input, &fixture.input)?;
        arena.copy_from_host(stream, regions.weight, &fixture.weight)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = S::new(context)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| {
                capture_route::<S>(&op, &arena, stream, regions, rows, repeated_operations)
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _op: op,
            arena,
            regions,
        })
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
                    S::OPERATION,
                    shape,
                    workload,
                    OperationAccounting::new(
                        logical_bytes::<S>(route.rows),
                        route.rows as u64,
                        "token",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

/// The four shape sessions, each owning its own arena.
struct Session {
    gdn_input: ShapeSession<GdnInputShape>,
    qsa_qkv: ShapeSession<QsaQkvShape>,
    indexer_qk: ShapeSession<IndexerQkShape>,
    block_output: ShapeSession<BlockOutputShape>,
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
        let gdn_input = ShapeSession::new(&context, &stream, repeated_operations)?;
        let qsa_qkv = ShapeSession::new(&context, &stream, repeated_operations)?;
        let indexer_qk = ShapeSession::new(&context, &stream, repeated_operations)?;
        let block_output = ShapeSession::new(&context, &stream, repeated_operations)?;

        Ok(Self {
            gdn_input,
            qsa_qkv,
            indexer_qk,
            block_output,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in self
                .gdn_input
                .routes
                .iter()
                .chain(&self.qsa_qkv.routes)
                .chain(&self.indexer_qk.routes)
                .chain(&self.block_output.routes)
            {
                // SAFETY: every captured allocation is retained by its shape
                // session through this synchronized replay.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        let mut cases = self.gdn_input.cases(repeated_operations);
        cases.extend(self.qsa_qkv.cases(repeated_operations));
        cases.extend(self.indexer_qk.cases(repeated_operations));
        cases.extend(self.block_output.cases(repeated_operations));

        cases
    }

    fn weight_bytes(&self) -> usize {
        self.gdn_input.regions.weight_bytes()
            + self.qsa_qkv.regions.weight_bytes()
            + self.indexer_qk.regions.weight_bytes()
            + self.block_output.regions.weight_bytes()
    }

    fn workspace_bytes(&self) -> usize {
        self.gdn_input.regions.workspace_bytes()
            + self.qsa_qkv.regions.workspace_bytes()
            + self.indexer_qk.regions.workspace_bytes()
            + self.block_output.regions.workspace_bytes()
    }

    fn padding_bytes(&self) -> usize {
        self.gdn_input.arena.byte_len() - self.gdn_input.regions.payload_bytes()
            + self.qsa_qkv.arena.byte_len()
            - self.qsa_qkv.regions.payload_bytes()
            + self.indexer_qk.arena.byte_len()
            - self.indexer_qk.regions.payload_bytes()
            + self.block_output.arena.byte_len()
            - self.block_output.regions.payload_bytes()
    }
}

fn capture_route<S: ProjectionShape>(
    op: &S::Op,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch::<S>(op, arena, stream, regions, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch::<S>(op, arena, stream, regions, rows)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        leaf,
        repeated,
    })
}

/// Bytes one route moves: the whole weight plane once, plus its two activations.
fn logical_bytes<S: ProjectionShape>(rows: usize) -> usize {
    let weight = S::OUTPUT_ROWS * S::COLUMNS * size_of::<u16>();
    let input = rows * S::COLUMNS * size_of::<u16>();
    let output = rows * S::OUTPUT_ROWS * size_of::<u16>();

    weight + input + output
}

/// Measures every exact Qwen3.8-Flash-Next backbone projection route.
pub fn benchmark_qwen38_flash_next_projections(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "qwen38_flash_next/projection/weights",
        BenchmarkMemoryKind::Weights,
        session.weight_bytes(),
        "materialized source BF16 [16384,2560], [13312,2560], [640,2560], and [2560,6144] planes",
    )?;
    memory.register_owned(
        "qwen38_flash_next/projection/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.workspace_bytes(),
        "max_rows=1024 activation inputs and projected outputs for all four shapes",
    )?;
    memory.register_owned(
        "qwen38_flash_next/projection/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.padding_bytes(),
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
            suite: "bench-qwen38-flash-next-projections",
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
    use super::logical_bytes;
    use crate::qwen38_flash_next_projection::{
        BLOCK_COLUMNS, BlockOutputShape, EXACT_ROUTES, GDN_INPUT_ROWS, GdnInputShape, HIDDEN,
        INDEXER_QK_ROWS, IndexerQkShape, MAX_BATCH, MAX_ROWS, ProjectionShape, QSA_QKV_ROWS,
        QsaQkvShape, layout,
    };

    /// A route reads its whole weight plane once; the activations are what the
    /// row count moves.
    #[test]
    fn qwen38_flash_next_projection_benchmark_byte_accounting_covers_weights_and_both_activations()
    {
        for (weight, columns, rows) in [
            (
                logical_bytes::<GdnInputShape>(0),
                GdnInputShape::COLUMNS,
                GdnInputShape::OUTPUT_ROWS,
            ),
            (
                logical_bytes::<QsaQkvShape>(0),
                QsaQkvShape::COLUMNS,
                QsaQkvShape::OUTPUT_ROWS,
            ),
            (
                logical_bytes::<IndexerQkShape>(0),
                IndexerQkShape::COLUMNS,
                IndexerQkShape::OUTPUT_ROWS,
            ),
            (
                logical_bytes::<BlockOutputShape>(0),
                BlockOutputShape::COLUMNS,
                BlockOutputShape::OUTPUT_ROWS,
            ),
        ] {
            assert_eq!(weight, rows * columns * size_of::<u16>());
        }

        assert_eq!(
            logical_bytes::<GdnInputShape>(MAX_BATCH),
            83_886_080 + MAX_BATCH * (HIDDEN + GDN_INPUT_ROWS) * size_of::<u16>()
        );
        assert_eq!(
            logical_bytes::<QsaQkvShape>(MAX_ROWS),
            68_157_440 + MAX_ROWS * (HIDDEN + QSA_QKV_ROWS) * size_of::<u16>()
        );
        assert_eq!(
            logical_bytes::<IndexerQkShape>(MAX_ROWS),
            3_276_800 + MAX_ROWS * (HIDDEN + INDEXER_QK_ROWS) * size_of::<u16>()
        );
        assert_eq!(
            logical_bytes::<BlockOutputShape>(MAX_ROWS),
            31_457_280 + MAX_ROWS * (BLOCK_COLUMNS + HIDDEN) * size_of::<u16>()
        );
    }

    /// The benchmark measures the same twelve routes the oracle qualifies, over
    /// the same arenas, and every reserved byte is a payload byte.
    #[test]
    fn qwen38_flash_next_projection_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(
            GdnInputShape::OPERATION,
            "qwen38_flash_next/projection/gdn_input_bf16"
        );
        assert_eq!(
            QsaQkvShape::OPERATION,
            "qwen38_flash_next/projection/qsa_qkv_bf16"
        );
        assert_eq!(
            IndexerQkShape::OPERATION,
            "qwen38_flash_next/projection/indexer_qk_bf16"
        );
        assert_eq!(
            BlockOutputShape::OPERATION,
            "qwen38_flash_next/projection/block_output_bf16"
        );

        let (gdn, gdn_regions) = layout::<GdnInputShape>().unwrap();
        let (qsa, qsa_regions) = layout::<QsaQkvShape>().unwrap();
        let (indexer, indexer_regions) = layout::<IndexerQkShape>().unwrap();
        let (block, block_regions) = layout::<BlockOutputShape>().unwrap();

        assert_eq!(gdn.byte_len(), gdn_regions.payload_bytes());
        assert_eq!(qsa.byte_len(), qsa_regions.payload_bytes());
        assert_eq!(indexer.byte_len(), indexer_regions.payload_bytes());
        assert_eq!(block.byte_len(), block_regions.payload_bytes());
        assert_eq!(
            gdn.byte_len() + qsa.byte_len() + indexer.byte_len() + block.byte_len(),
            282_460_160
        );
        assert_eq!(4 * EXACT_ROUTES.len(), 48);
    }
}
