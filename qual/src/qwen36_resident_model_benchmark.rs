//! Direct complete-graph timing for the resident Qwen3.6 text model.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen36ResidentModelProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
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
    program: Qwen36ResidentModelProgram,
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
        let mut program = Qwen36ResidentModelProgram::from_snapshot(&context, snapshot)?;
        program.stage_embeddings(&stream, &benchmark_token_ids())?;
        program.reset_state(&stream)?;
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
                // SAFETY: the program retains every captured model allocation through this replay.
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
                    "qwen36_35b_a3b/resident_model/decode",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_model_decode(route.batch as u32, CONTEXT_TOKENS as u64),
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

fn benchmark_token_ids() -> [u32; MAX_BATCH] {
    core::array::from_fn(|slot| (100 + slot * 17) as u32)
}

fn benchmark_rope() -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for slot in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let angle = f64::from(CACHE_POSITION) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[slot * ROTARY_PAIRS + pair] = cos as f32;
            sine[slot * ROTARY_PAIRS + pair] = sin as f32;
        }
    }

    (cosine, sine)
}

fn logical_bytes(batch: usize) -> usize {
    const GDN_LAYERS: usize = 30;
    const ATTENTION_LAYERS: usize = 10;

    GDN_LAYERS * gdn_logical_bytes(batch)
        + ATTENTION_LAYERS * attention_logical_bytes(batch)
        + endpoint_logical_bytes(batch)
}

fn gdn_logical_bytes(batch: usize) -> usize {
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
    let selected_expert_weights = batch * EXPERT_SLOTS * selected_expert_weight_bytes();
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

fn attention_logical_bytes(batch: usize) -> usize {
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
    let heads = Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS;
    let source = attention + 2 * kv;
    let norms = heads * Qwen36Moe35B::HEAD_DIM;
    let rotary = heads * ROTARY_PAIRS * 2;
    let qk_prepare = (source + norms) * size_of::<u16>()
        + rotary * size_of::<f32>()
        + 3 * size_of::<u32>()
        + attention * size_of::<f32>()
        + 2 * kv * size_of::<u16>();
    let cache = 2
        * Qwen36Moe35B::NUM_ATTENTION_HEADS
        * CONTEXT_TOKENS
        * Qwen36Moe35B::HEAD_DIM
        * size_of::<u16>();
    let metadata = 2 * size_of::<u32>()
        + Qwen36Moe35B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
    let paged_gqa = 2 * attention * size_of::<f32>() + cache + metadata;
    let attention_output = 18 * attention + 2 * hidden;
    let residual_boundaries = 2 * 5 * hidden * size_of::<u16>();
    let router_weights = experts * hidden * size_of::<u16>();
    let router =
        hidden * size_of::<u16>() + experts * size_of::<u16>() + 2 * top_k * size_of::<u16>();
    let experts_path = EXPERT_SLOTS * selected_expert_weight_bytes()
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

fn selected_expert_weight_bytes() -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let intermediate = Qwen36Moe35B::INTERMEDIATE;
    let gate_up_codes = 2 * intermediate * hidden / 2;
    let gate_up_scales = 2 * intermediate * hidden / NVFP4_GROUP;
    let down_codes = hidden * intermediate / 2;
    let down_scales = hidden * intermediate / NVFP4_GROUP;

    gate_up_codes + gate_up_scales + down_codes + down_scales
}

fn endpoint_logical_bytes(batch: usize) -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let vocab = Qwen36Moe35B::VOCAB;
    let weights = 2 * hidden + vocab * (hidden / 2 + hidden / NVFP4_GROUP);
    let per_token = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;

    weights + batch * per_token
}

/// Measures every exact complete Qwen3.6 text-model graph.
pub fn benchmark_qwen36_resident_model(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    let layout = session.program.layout();
    memory.register_owned(
        "qwen36_35b_a3b/resident_model/weights",
        BenchmarkMemoryKind::Weights,
        layout.resident_weight_bytes(),
        "40 decoder layers, final norm, and represented NVFP4 LM head",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/resident_model/bf16_kv_cache",
        BenchmarkMemoryKind::KvCache,
        layout.cache_bytes(),
        "10 attention layers * 8 slots * 192 positions",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/resident_model/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        layout.workspace_bytes(),
        "40 retained layer arenas plus endpoint workspace",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/resident_model/alignment_padding",
        BenchmarkMemoryKind::Other,
        layout.padding_bytes(),
        "aggregate 256-byte alignment across 41 arenas",
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
            suite: "bench-qwen36-resident-model",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust submission/completion, production graph, and repeated complete 40-layer plus endpoint path",
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
    fn accounting_covers_every_layer_endpoint_and_selected_expert() {
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(gdn_logical_bytes(1), 55_424_712);
        assert_eq!(gdn_logical_bytes(MAX_BATCH), 199_339_840);
        assert_eq!(attention_logical_bytes(1), 46_735_636);
        assert_eq!(attention_logical_bytes(MAX_BATCH), 175_704_224);
        assert_eq!(endpoint_logical_bytes(1), 286_581_768);
        assert_eq!(endpoint_logical_bytes(MAX_BATCH), 290_172_992);
        assert_eq!(logical_bytes(1), 2_416_679_488);
        assert_eq!(logical_bytes(MAX_BATCH), 8_027_410_432);
    }
}
