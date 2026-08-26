//! Direct timing for one source-backed full-attention layer owner.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::attention::prefill_rope_tables;
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{FullAttentionLayerProgram, MAX_BATCH};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const SOURCE_LAYER: usize = 63;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];
const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;

struct RouteGraph {
    rows: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: FullAttentionLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
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
        let program = FullAttentionLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
        program.reset_cache(&stream)?;
        let (rope_cos, rope_sin) = benchmark_rope();
        program.load_decode_state(
            &stream,
            MAX_BATCH,
            &[CACHE_POSITION; MAX_BATCH],
            &rope_cos,
            &rope_sin,
        )?;
        let (prefill_cos, prefill_sin) = prefill_rope(MAX_ROWS);
        program.load_prefill_state(&stream, MAX_ROWS, &prefill_cos, &prefill_sin)?;
        // Position 130 crosses both 64-token page seams, so direct timing
        // exercises the initial owner's full three-page route without padding it.
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| {
                Ok(RouteGraph {
                    rows,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        rows,
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
            for rows in EXACT_ROUTES {
                let graph = self.program.qualification_graph(rows)?;
                // SAFETY: this Session's program owns the graph and every
                // allocation it captured, outliving the replay and synchronize.
                unsafe { graph.launch(&self.stream) }?;
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
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_attention_layer_decode(
                            route.rows as u32,
                            CONTEXT_TOKENS as u64,
                        ),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_attention_layer_prefill(route.rows as u64),
                    )
                };
                Ok(ExactDeviceCase::new(
                    "full_attention_layer/layer63",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    self.program.qualification_graph(route.rows)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
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

fn benchmark_rope() -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for token in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let angle = f64::from(CACHE_POSITION) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[token * ROTARY_PAIRS + pair] = cos as f32;
            sine[token * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn prefill_rope(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    prefill_rope_tables(0, tokens, ROTARY_PAIRS, 2 * ROTARY_PAIRS, 10_000_000.0)
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let qkv = Qwen38_27B::ATTENTION_QKV_ROWS;
    let attention = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let weights = 372_395_008;
    // Paged GQA rereads one represented key and value byte per causal context
    // element and query head. Other terms count every published
    // production seam once at its represented element width.
    let cache_reads = if rows <= MAX_BATCH {
        2 * rows * CONTEXT_TOKENS * attention
    } else {
        2 * attention * (rows * (rows + 1) / 2)
    };
    let macro_partial_traffic = if rows == MAX_ROWS {
        let active_partials =
            rows * Qwen38_27B::NUM_ATTENTION_HEADS * 4 * (Qwen38_27B::HEAD_DIM + 2);
        2 * active_partials * size_of::<f32>()
    } else {
        0
    };
    let route_bytes = cache_reads
        + rows * (18 * hidden + 5 * qkv + 18 * attention + 6 * intermediate + 4 * size_of::<f32>())
        + macro_partial_traffic;
    weights + route_bytes
}

/// Measures every exact graph of one source-backed full-attention layer owner.
pub fn benchmark_full_attention_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "full_attention_layer/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=63 source-native attention and dense-FP8 MLP",
    )?;
    memory.register_owned(
        "full_attention_layer/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "8 slots * 3 pages * 4 KV heads * 64 tokens * 256 E4M3 values * K/V",
    )?;
    memory.register_owned(
        "full_attention_layer/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 including maximum macro partials; production graph uses P4",
    )?;
    memory.register_owned(
        "full_attention_layer/address_bound_tensor_maps",
        BenchmarkMemoryKind::Other,
        session.program.descriptor_bytes(),
        "four 128-byte dense-FP8 MLP tensor maps",
    )?;
    memory.register_owned(
        "full_attention_layer/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.cache_bytes()
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
            suite: "bench-full-attention-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust production-graph submission/completion and repeated full-attention-layer route",
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
    use super::{CONTEXT_TOKENS, EXACT_ROUTES, MAX_BATCH, MAX_ROWS, logical_bytes};

    #[test]
    fn full_attention_layer_suite_byte_accounting_covers_decode_and_causal_prefill() {
        let one = logical_bytes(1);
        let per_token = logical_bytes(2) - one;
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(per_token, 1_988_624);
        assert_eq!(one, 374_383_632);
        assert_eq!(logical_bytes(MAX_BATCH), one + (MAX_BATCH - 1) * per_token);
        assert!(logical_bytes(MAX_ROWS) > logical_bytes(128));
    }

    #[test]
    fn full_attention_layer_suite_benchmark_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
    }
}
