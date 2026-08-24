//! Direct timing for one source-backed Qwen3.5 full-attention layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35FullAttentionLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const SOURCE_LAYER: usize = 31;
const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const NVFP4_GROUP: usize = 16;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen35FullAttentionLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program =
            Qwen35FullAttentionLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_BATCH, &benchmark_input())?;
        program.reset_cache(&stream)?;
        let (rope_cos, rope_sin) = benchmark_rope();
        program.load_decode_state(
            &stream,
            MAX_BATCH,
            &[CACHE_POSITION; MAX_BATCH],
            &rope_cos,
            &rope_sin,
        )?;
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
                let graph = self.program.qualification_graph(batch)?;
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
                Ok(ExactDeviceCase::new(
                    "qwen35_9b/full_attention/layer31",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_attention_layer_decode(
                        route.batch as u32,
                        CONTEXT_TOKENS as u64,
                    ),
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

    (0..MAX_BATCH * Qwen35_9B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen35_9B::HIDDEN) & 7]))
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

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let qkv = Qwen35_9B::ATTENTION_QKV_ROWS;
    let attention = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
    let kv = Qwen35_9B::ATTENTION_KV_ROWS;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let packed_row = hidden / 2 + hidden / NVFP4_GROUP;
    let output_packed_row = attention / 2 + attention / NVFP4_GROUP;
    let down_packed_row = intermediate / 2 + intermediate / NVFP4_GROUP;
    let projection_weights = qkv * packed_row
        + hidden * output_packed_row
        + 2 * intermediate * packed_row
        + hidden * down_packed_row;

    let plain_norm = 3 * hidden * size_of::<u16>();
    let qkv_projection = (hidden + qkv) * size_of::<u16>();
    let qk_prepare = {
        let heads = Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS;
        let source = attention + 2 * kv;
        let norms = heads * Qwen35_9B::HEAD_DIM;
        let rotary = heads * ROTARY_PAIRS * 2;
        (source + norms) * size_of::<u16>()
            + rotary * size_of::<f32>()
            + 3 * size_of::<u32>()
            + attention * size_of::<f32>()
            + 2 * kv * size_of::<u16>()
    };
    let paged_gqa = {
        let cache = 2
            * Qwen35_9B::NUM_ATTENTION_HEADS
            * CONTEXT_TOKENS
            * Qwen35_9B::HEAD_DIM
            * size_of::<u16>();
        let metadata = 2 * size_of::<u32>()
            + Qwen35_9B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
        2 * attention * size_of::<f32>() + cache + metadata
    };
    let attention_output = 14 * attention + 2 * hidden;
    let fused_residual = 5 * hidden * size_of::<u16>();
    let swiglu =
        (hidden + intermediate) * size_of::<u16>() + usize::from(uses_w4a4(batch)) * 2 * packed_row;
    let down = (intermediate + hidden) * size_of::<u16>();
    let per_token = plain_norm
        + qkv_projection
        + qk_prepare
        + paged_gqa
        + attention_output
        + 2 * fused_residual
        + swiglu
        + down;

    projection_weights + batch * per_token
}

fn uses_w4a4(batch: usize) -> bool {
    batch == 1 || batch >= 3
}

/// Measures every exact graph of one source-backed Qwen3.5 attention layer.
pub fn benchmark_qwen35_full_attention_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen35_9b/full_attention/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=31 source-native norms and losslessly materialized projections",
    )?;
    memory.register_owned(
        "qwen35_9b/full_attention/bf16_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "8 slots * 3 pages * 4 KV heads * 64 tokens * 256 BF16 values * K/V",
    )?;
    memory.register_owned(
        "qwen35_9b/full_attention/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 including metadata and every published seam",
    )?;
    memory.register_owned(
        "qwen35_9b/full_attention/alignment_padding",
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
            suite: "bench-qwen35-full-attention-layer",
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
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_TOKENS, MAX_BATCH, logical_bytes, uses_w4a4};

    #[test]
    fn accounting_covers_bf16_cache_and_both_mlp_routes() {
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 120_471_252);
        assert_eq!(logical_bytes(2), 122_968_488);
        assert_eq!(logical_bytes(MAX_BATCH), 138_016_416);
        assert_eq!(
            (1..=MAX_BATCH).map(uses_w4a4).collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true]
        );
    }
}
