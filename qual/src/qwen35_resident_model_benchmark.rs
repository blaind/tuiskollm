//! Direct complete-graph timing for the resident Qwen3.5 text model.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35ResidentModelProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const CONTROL_STRIDE: usize = 128;
const NVFP4_GROUP: usize = 16;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen35ResidentModelProgram,
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
        let mut program = Qwen35ResidentModelProgram::from_snapshot(&context, snapshot)?;
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
                    "qwen35_9b/resident_model/decode",
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
    const GDN_LAYERS: usize = 24;
    const ATTENTION_LAYERS: usize = 8;
    const RESIDENT_WEIGHTS: usize = 5_931_820_032;
    let endpoint = 8 * Qwen35_9B::HIDDEN + 2 * size_of::<f32>() + 2 * Qwen35_9B::VOCAB;
    let per_token = GDN_LAYERS * gdn_per_token(batch)
        + ATTENTION_LAYERS * attention_per_token(batch)
        + endpoint;

    RESIDENT_WEIGHTS + batch * per_token
}

fn gdn_per_token(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let input_rows = Qwen35_9B::GDN_INPUT_ROWS;
    let qkv_rows = Qwen35_9B::GDN_QKV_ROWS;
    let value_rows = Qwen35_9B::GDN_VALUE_ROWS;
    let controls = Qwen35_9B::GDN_CONTROL_ROWS;
    let state = controls * Qwen35_9B::LINEAR_HEAD_DIM * Qwen35_9B::LINEAR_HEAD_DIM;

    4 * hidden
        + 4 * hidden
        + 2 * (input_rows + CONTROL_STRIDE)
        + 4 * controls
        + 16 * qkv_rows
        + 8 * controls
        + 2 * qkv_rows
        + 2 * value_rows
        + 8 * controls
        + 8 * state
        + 2 * value_rows
        + 2 * (value_rows + hidden)
        + 2 * 8 * hidden
        + 2 * (hidden + intermediate)
        + usize::from(uses_w4a4(batch)) * (hidden / 2 + hidden / NVFP4_GROUP)
        + 2 * (intermediate + hidden)
}

fn attention_per_token(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let qkv = Qwen35_9B::ATTENTION_QKV_ROWS;
    let attention = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
    let kv = Qwen35_9B::ATTENTION_KV_ROWS;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let packed_row = hidden / 2 + hidden / NVFP4_GROUP;
    let heads = Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS;
    let source = attention + 2 * kv;
    let norms = heads * Qwen35_9B::HEAD_DIM;
    let rotary = heads * ROTARY_PAIRS * 2;
    let qk_prepare = (source + norms) * size_of::<u16>()
        + rotary * size_of::<f32>()
        + 3 * size_of::<u32>()
        + attention * size_of::<f32>()
        + 2 * kv * size_of::<u16>();
    let cache = 2
        * Qwen35_9B::NUM_ATTENTION_HEADS
        * CONTEXT_TOKENS
        * Qwen35_9B::HEAD_DIM
        * size_of::<u16>();
    let metadata =
        2 * size_of::<u32>() + Qwen35_9B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
    let paged_gqa = 2 * attention * size_of::<f32>() + cache + metadata;

    3 * hidden * size_of::<u16>()
        + (hidden + qkv) * size_of::<u16>()
        + qk_prepare
        + paged_gqa
        + 14 * attention
        + 2 * hidden
        + 2 * 5 * hidden * size_of::<u16>()
        + (hidden + intermediate) * size_of::<u16>()
        + usize::from(uses_w4a4(batch)) * 2 * packed_row
        + (intermediate + hidden) * size_of::<u16>()
}

fn uses_w4a4(batch: usize) -> bool {
    batch == 1 || batch >= 3
}

/// Measures every exact complete Qwen3.5 text-model graph.
pub fn benchmark_qwen35_resident_model(
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
        "qwen35_9b/resident_model/weights",
        BenchmarkMemoryKind::Weights,
        layout.resident_weight_bytes(),
        "32 decoder layers, final norm, and BF16 LM head",
    )?;
    memory.register_owned(
        "qwen35_9b/resident_model/bf16_kv_cache",
        BenchmarkMemoryKind::KvCache,
        layout.cache_bytes(),
        "8 attention layers * 8 slots * 192 positions",
    )?;
    memory.register_owned(
        "qwen35_9b/resident_model/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        layout.workspace_bytes(),
        "32 retained layer arenas plus endpoint workspace",
    )?;
    memory.register_owned(
        "qwen35_9b/resident_model/alignment_padding",
        BenchmarkMemoryKind::Other,
        layout.padding_bytes(),
        "aggregate 256-byte alignment across 33 arenas",
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
            suite: "bench-qwen35-resident-model",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust submission/completion, production graph, and repeated complete 32-layer plus endpoint path",
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
    use super::{MAX_BATCH, logical_bytes, uses_w4a4};

    #[test]
    fn accounting_covers_every_layer_endpoint_and_route() {
        assert_eq!(logical_bytes(1), 6_061_988_520);
        assert_eq!(logical_bytes(2), 6_191_972_688);
        assert_eq!(logical_bytes(MAX_BATCH), 6_973_167_936);
        assert_eq!(
            (1..=MAX_BATCH).map(uses_w4a4).collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true]
        );
    }
}
