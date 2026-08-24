//! Direct timing for one source-backed Qwen3.6 attention plus MoE layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen36FullAttentionLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

const SOURCE_LAYER: usize = 3;
const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const EXPERT_SLOTS: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1;
const NVFP4_GROUP: usize = 16;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen36FullAttentionLayerProgram,
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
        let program =
            Qwen36FullAttentionLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
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
                    "qwen36_35b_a3b/full_attention/layer3",
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
    (0..MAX_BATCH * Qwen36Moe35B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen36Moe35B::HIDDEN) & 7]))
        .collect()
}

fn benchmark_rope() -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for token in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let angle = (token * ROTARY_PAIRS + pair + 1) as f32 * 0.007_812_5;
            cosine[token * ROTARY_PAIRS + pair] = angle.cos();
            sine[token * ROTARY_PAIRS + pair] = angle.sin();
        }
    }
    (cosine, sine)
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let qkv = Qwen36Moe35B::ATTENTION_QKV_ROWS;
    let attention = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
    let kv = Qwen36Moe35B::ATTENTION_KV_ROWS;
    let intermediate = Qwen36Moe35B::INTERMEDIATE;
    let experts = Qwen36Moe35B::NUM_EXPERTS;
    let top_k = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;

    let projection_weights = qkv * hidden + hidden * attention;
    let plain_norm = 3 * hidden * size_of::<u16>();
    let qkv_projection = hidden * (2 * size_of::<u16>() + size_of::<u8>()) + qkv * size_of::<u16>();
    let qk_prepare = {
        let heads = Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS;
        let source = attention + 2 * kv;
        let norms = heads * Qwen36Moe35B::HEAD_DIM;
        let rotary = heads * ROTARY_PAIRS * 2;
        (source + norms) * size_of::<u16>()
            + rotary * size_of::<f32>()
            + 3 * size_of::<u32>()
            + attention * size_of::<f32>()
            + 2 * kv * size_of::<u16>()
    };
    let paged_gqa = {
        let cache = 2
            * Qwen36Moe35B::NUM_ATTENTION_HEADS
            * CONTEXT_TOKENS
            * Qwen36Moe35B::HEAD_DIM
            * size_of::<u16>();
        let metadata = 2 * size_of::<u32>()
            + Qwen36Moe35B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
        2 * attention * size_of::<f32>() + cache + metadata
    };
    let attention_output = 18 * attention + 2 * hidden;
    let residual_boundaries = 2 * 5 * hidden * size_of::<u16>();
    let router_weights = experts * hidden * size_of::<u16>();
    let router =
        hidden * size_of::<u16>() + experts * size_of::<u16>() + 2 * top_k * size_of::<u16>();
    let gate_up_codes = 2 * intermediate * hidden / 2;
    let gate_up_scales = 2 * intermediate * hidden / NVFP4_GROUP;
    let down_codes = hidden * intermediate / 2;
    let down_scales = hidden * intermediate / NVFP4_GROUP;
    let selected_expert_weights =
        EXPERT_SLOTS * (gate_up_codes + gate_up_scales + down_codes + down_scales);
    let experts_path = selected_expert_weights
        + hidden * size_of::<u16>()
        + 2 * top_k * size_of::<u16>()
        + hidden * size_of::<u16>()
        + 2 * EXPERT_SLOTS * intermediate * size_of::<u16>()
        + 2 * EXPERT_SLOTS * hidden * size_of::<u16>()
        + hidden * size_of::<u16>();
    let per_token = plain_norm
        + qkv_projection
        + qk_prepare
        + paged_gqa
        + attention_output
        + residual_boundaries
        + router
        + experts_path;

    projection_weights + router_weights + batch * per_token
}

/// Measures every exact graph of one source-backed Qwen3.6 attention/MoE layer.
pub fn benchmark_qwen36_full_attention_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen36_35b_a3b/full_attention/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=3 source-native FP8 attention and losslessly materialized MoE weights",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/full_attention/bf16_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "8 slots * 3 pages * 2 KV heads * 64 tokens * 256 BF16 values * K/V",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/full_attention/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 including metadata, routing, expert slots, and all published seams",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/full_attention/alignment_padding",
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
            suite: "bench-qwen36-full-attention-layer",
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
    fn accounting_covers_attention_cache_and_selected_experts() {
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 46_735_636);
        assert_eq!(logical_bytes(MAX_BATCH), 175_704_224);
    }
}
