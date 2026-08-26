//! Direct complete-graph timing for the resident Qwen3.5 text model.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::attention::prefill_rope_tables;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35ResidentModelProgram, Qwen35ResidentPrefillRoute};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const CONTROL_STRIDE: usize = 128;
const NVFP4_GROUP: usize = 16;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];

#[derive(Clone, Copy)]
enum ExactRoute {
    Decode(usize),
    Prefill(Qwen35ResidentPrefillRoute),
}

struct RouteGraph {
    route: ExactRoute,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
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
        let slots = (0..MAX_BATCH).collect::<Vec<_>>();
        for &slot in &slots {
            program.activate_kv_slot(slot)?;
            program.reserve_kv_slot_tokens(&stream, slot, CONTEXT_TOKENS)?;
        }
        program.load_slot_routes(&stream, &slots)?;
        let (rope_cos, rope_sin) = benchmark_rope();
        program.load_decode_state(
            &stream,
            MAX_BATCH,
            &[CACHE_POSITION; MAX_BATCH],
            &rope_cos,
            &rope_sin,
        )?;
        program.stage_prefill_embeddings(&stream, &prefill_token_ids(128))?;
        let (prefill_cos, prefill_sin) = prefill_rope(128);
        let prefill_routes = PREFILL_ROUTES
            .into_iter()
            .map(|tokens| {
                program.load_prefill_state(
                    &stream,
                    tokens,
                    &prefill_cos[..tokens * ROTARY_PAIRS],
                    &prefill_sin[..tokens * ROTARY_PAIRS],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut routes = Vec::with_capacity(MAX_BATCH + PREFILL_ROUTES.len());
        for batch in 1..=MAX_BATCH {
            routes.push(RouteGraph {
                route: ExactRoute::Decode(batch),
                repeated: program.qualification_repeated_graph(
                    &stream,
                    batch,
                    repeated_operations,
                )?,
            });
        }
        for route in prefill_routes {
            routes.push(RouteGraph {
                route: ExactRoute::Prefill(route),
                repeated: program.qualification_repeated_prefill_graph(
                    &stream,
                    route,
                    repeated_operations,
                )?,
            });
        }

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
                let graph = match route.route {
                    ExactRoute::Decode(batch) => self.program.qualification_graph(batch)?,
                    ExactRoute::Prefill(route) => {
                        self.program.qualification_prefill_graph(route)?
                    }
                };
                // SAFETY: the program retains every captured model allocation through this replay.
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
                let (operation, shape, workload, accounting, graph) = match route.route {
                    ExactRoute::Decode(batch) => (
                        "qwen35_9b/resident_model/decode",
                        format!("B={batch}"),
                        BenchmarkWorkload::warm_model_decode(batch as u32, CONTEXT_TOKENS as u64),
                        OperationAccounting::new(logical_bytes(batch), batch as u64, "token"),
                        self.program.qualification_graph(batch)?,
                    ),
                    ExactRoute::Prefill(prefill) => {
                        let tokens = prefill.tokens();
                        (
                            "qwen35_9b/resident_model/prefill",
                            format!("T={tokens}"),
                            BenchmarkWorkload::warm_model_prefill(tokens as u64),
                            OperationAccounting::new(
                                prefill_logical_bytes(tokens),
                                tokens as u64,
                                "token",
                            ),
                            self.program.qualification_prefill_graph(prefill)?,
                        )
                    }
                };
                Ok(ExactDeviceCase::new(
                    operation,
                    shape,
                    workload,
                    accounting,
                    graph,
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

fn prefill_token_ids(tokens: usize) -> Vec<u32> {
    (0..tokens)
        .map(|token| 100u32 + (token % 251) as u32)
        .collect()
}

fn prefill_rope(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    prefill_rope_tables(0, tokens, ROTARY_PAIRS, 2 * ROTARY_PAIRS, 10_000_000.0)
}

fn logical_bytes(batch: usize) -> usize {
    const GDN_LAYERS: usize = 24;
    const ATTENTION_LAYERS: usize = 8;

    GDN_LAYERS * gdn_logical_bytes(batch)
        + ATTENTION_LAYERS * attention_logical_bytes(batch)
        + endpoint_logical_bytes(batch)
}

fn gdn_logical_bytes(rows: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let input_rows = Qwen35_9B::GDN_INPUT_ROWS;
    let qkv_rows = Qwen35_9B::GDN_QKV_ROWS;
    let value_rows = Qwen35_9B::GDN_VALUE_ROWS;
    let controls = Qwen35_9B::GDN_CONTROL_ROWS;
    let state = controls * Qwen35_9B::LINEAR_HEAD_DIM * Qwen35_9B::LINEAR_HEAD_DIM;
    let weights = 123_068_800;
    let hidden_scratch = hidden / 2 + hidden / NVFP4_GROUP;
    let intermediate_scratch = intermediate / 2 + intermediate / NVFP4_GROUP;
    let plain_norm = 4 * hidden;
    let input_projection = if rows <= MAX_BATCH {
        4 * hidden + 2 * (input_rows + CONTROL_STRIDE)
    } else {
        2 * hidden + 3 * hidden_scratch + 2 * (input_rows + CONTROL_STRIDE)
    };
    let prepare = 4 * controls + 16 * qkv_rows + 8 * controls;
    let recurrence = 2 * qkv_rows + 2 * value_rows + 8 * controls + 8 * state + 2 * value_rows;
    let output_projection = if rows <= MAX_BATCH {
        2 * (value_rows + hidden)
    } else {
        2 * value_rows + 2 * hidden_scratch + 2 * hidden
    };
    let fused_residual = 8 * hidden;
    let swiglu = 2 * (hidden + intermediate) + usize::from(uses_w4a4(rows)) * 2 * hidden_scratch;
    let down = if rows <= MAX_BATCH {
        2 * (intermediate + hidden)
    } else {
        2 * intermediate + 2 * intermediate_scratch + 2 * hidden
    };
    let per_token = plain_norm
        + input_projection
        + prepare
        + recurrence
        + output_projection
        + 2 * fused_residual
        + swiglu
        + down;

    weights + rows * per_token
}

fn attention_logical_bytes(rows: usize) -> usize {
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
    let prompt = rows > MAX_BATCH;
    let plain_norm = 3 * hidden * size_of::<u16>();
    let qkv_projection = (hidden + qkv) * size_of::<u16>() + usize::from(prompt) * 2 * packed_row;
    let heads = Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS;
    let source = attention + 2 * kv;
    let norms = heads * Qwen35_9B::HEAD_DIM;
    let rotary = heads * ROTARY_PAIRS * 2;
    let qk_prepare = (source + norms) * size_of::<u16>()
        + rotary * size_of::<f32>()
        + 3 * size_of::<u32>()
        + attention * size_of::<f32>()
        + 2 * kv * size_of::<u16>();
    let context_values = if rows <= MAX_BATCH {
        rows * CONTEXT_TOKENS
    } else {
        rows * (rows + 1) / 2
    };
    let cache = 2
        * Qwen35_9B::NUM_ATTENTION_HEADS
        * context_values
        * Qwen35_9B::HEAD_DIM
        * size_of::<u16>();
    let metadata = rows * 2 * size_of::<u32>()
        + Qwen35_9B::NUM_ATTENTION_HEADS * context_values * size_of::<u32>();
    let paged_gqa = rows * 2 * attention * size_of::<f32>() + cache + metadata;
    let attention_output =
        14 * attention + 2 * hidden + usize::from(prompt) * 2 * output_packed_row;
    let fused_residual = 5 * hidden * size_of::<u16>();
    let swiglu =
        (hidden + intermediate) * size_of::<u16>() + usize::from(uses_w4a4(rows)) * 2 * packed_row;
    let down =
        (intermediate + hidden) * size_of::<u16>() + usize::from(prompt) * 2 * down_packed_row;
    let per_token_without_gqa = plain_norm
        + qkv_projection
        + qk_prepare
        + attention_output
        + 2 * fused_residual
        + swiglu
        + down;

    projection_weights + rows * per_token_without_gqa + paged_gqa
}

fn endpoint_logical_bytes(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let vocab = Qwen35_9B::VOCAB;
    let weights = 2 * hidden + 2 * vocab * hidden;
    let per_token = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;

    weights + batch * per_token
}

fn prefill_logical_bytes(tokens: usize) -> usize {
    const GDN_LAYERS: usize = 24;
    const ATTENTION_LAYERS: usize = 8;

    GDN_LAYERS * gdn_logical_bytes(tokens)
        + ATTENTION_LAYERS * attention_logical_bytes(tokens)
        + endpoint_logical_bytes(1)
}

fn uses_w4a4(rows: usize) -> bool {
    rows > MAX_BATCH || rows == 1 || rows >= 3
}

/// Measures every exact complete Qwen3.5 decode and native-prompt graph.
pub fn benchmark_qwen35_resident_model(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
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
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-resident-model",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust submission/completion, production graph, and repeated complete 32-layer plus endpoint decode/prompt paths",
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
    use super::{MAX_BATCH, PREFILL_ROUTES, logical_bytes, prefill_logical_bytes, uses_w4a4};

    #[test]
    fn qwen35_resident_model_suite_benchmark_accounting_covers_every_layer_endpoint_and_route() {
        assert_eq!(logical_bytes(1), 6_061_839_016);
        assert_eq!(logical_bytes(2), 6_191_767_888);
        assert_eq!(logical_bytes(MAX_BATCH), 6_973_405_504);
        assert_eq!(prefill_logical_bytes(32), 9_619_311_624);
        assert_eq!(prefill_logical_bytes(64), 13_441_220_616);
        assert_eq!(prefill_logical_bytes(128), 21_489_264_648);
        assert!(
            PREFILL_ROUTES
                .map(prefill_logical_bytes)
                .windows(2)
                .all(|pair| pair[1] > pair[0])
        );
        assert_eq!(
            (1..=MAX_BATCH).map(uses_w4a4).collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true]
        );
    }
}
