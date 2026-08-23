//! Direct complete-graph timing for the resident text model.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    MAX_BATCH, ResidentDecodeRoute, ResidentEmbeddingStageGraph, ResidentLayerKind,
    ResidentModelLayout, ResidentModelProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuTimer, profiler_start, profiler_stop};
use tuisko_kernels_sm120::{LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const LONG_CONTEXT_TOKENS: usize = 131_073;
const ROTARY_PAIRS: usize = 32;
const LONG_ROUTE_SLOTS: [usize; MAX_BATCH] = [7, 0, 6, 1, 5, 2, 4, 3];

#[derive(Clone, Copy)]
enum BenchmarkProfile {
    Short,
    Long,
}

/// One semantic production-owner range in the resident CUDA Graph.
#[derive(Debug, Serialize)]
pub struct ResidentProfileStage {
    /// Stable stage order within one complete graph replay.
    pub ordinal: usize,
    /// One-based first CUDA Graph node expected for this stage.
    pub first_graph_node_ordinal: usize,
    /// Number of kernel nodes launched by the production owner.
    pub kernel_nodes: usize,
    /// Decoder layer, or `None` for input/endpoint stages.
    pub layer: Option<usize>,
    /// Semantic boundary within the layer or endpoint.
    pub component: &'static str,
    /// Exact source route selected by the resident layout.
    pub source_route: &'static str,
    /// Kernel-name families expected in graph order.
    pub kernel_families: Vec<&'static str>,
}

/// Semantic sidecar for joining profiler nodes to exact resident owners.
#[derive(Debug, Serialize)]
pub struct ResidentModelProfileManifest {
    /// Manifest schema revision.
    pub schema_version: u32,
    /// Exact profiling boundary.
    pub suite: &'static str,
    /// Compiled decode batch.
    pub batch_size: usize,
    /// Fixed warmed attention context.
    pub context_tokens: usize,
    /// Warmups completed before the captured replays.
    pub warmup_launches: u64,
    /// Complete production graph replays requested from the profiler process.
    pub captured_replays: u64,
    /// CUDA-generated structural graph inventory.
    pub graph_dot: String,
    /// Complete expected kernel-node inventory for one replay.
    pub graph_kernel_nodes: usize,
    /// Ordered semantic production-owner ranges.
    pub stages: Vec<ResidentProfileStage>,
}

struct Session {
    timer: GpuTimer,
    program: ResidentModelProgram,
    routes: [ResidentDecodeRoute; MAX_BATCH],
    context_lengths: [usize; MAX_BATCH],
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, profile: BenchmarkProfile) -> Result<Self, DeviceBenchmarkError> {
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
        let mut program = ResidentModelProgram::from_snapshot(&context, snapshot)?;
        let (slots, positions, context_lengths) = match profile {
            BenchmarkProfile::Short => {
                for slot in 0..MAX_BATCH {
                    program.activate_kv_slot(slot)?;
                    program.reserve_kv_slot_tokens(&stream, slot, 192)?;
                }
                (
                    core::array::from_fn(|slot| slot),
                    [CACHE_POSITION; MAX_BATCH],
                    [CONTEXT_TOKENS; MAX_BATCH],
                )
            }
            BenchmarkProfile::Long => {
                for slot in 0..MAX_BATCH {
                    program.activate_kv_slot(slot)?;
                }
                program.reserve_kv_slot_tokens(&stream, 7, LONG_CONTEXT_TOKENS)?;
                for slot in 0..MAX_BATCH - 1 {
                    program.reserve_kv_slot_tokens(&stream, slot, 1)?;
                }
                let mut positions = [0; MAX_BATCH];
                positions[0] = (LONG_CONTEXT_TOKENS - 1) as u32;
                let mut context_lengths = [1; MAX_BATCH];
                context_lengths[0] = LONG_CONTEXT_TOKENS;
                (LONG_ROUTE_SLOTS, positions, context_lengths)
            }
        };
        program.load_slot_routes(&stream, &slots)?;
        program.stage_embeddings(&stream, &benchmark_token_ids())?;
        program.reset_state(&stream)?;
        let (rope_cos, rope_sin) = benchmark_rope(&positions);
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                program.load_decode_state(
                    &stream,
                    batch,
                    &positions[..batch],
                    &rope_cos[..batch * ROTARY_PAIRS],
                    &rope_sin[..batch * ROTARY_PAIRS],
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                DeviceBenchmarkError::Precondition(
                    "resident decode route inventory has wrong cardinality".to_string(),
                )
            })?;
        let timer = GpuTimer::new(&context)?;
        Ok(Self {
            timer,
            program,
            routes,
            context_lengths,
            stream,
            _context: context,
        })
    }

    fn embedding_graphs(
        &self,
    ) -> Result<[ResidentEmbeddingStageGraph<'_>; MAX_BATCH], DeviceBenchmarkError> {
        (1..=MAX_BATCH)
            .map(|batch| {
                self.program
                    .qualification_embedding_stage_graph(&self.stream, batch)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                DeviceBenchmarkError::Precondition(
                    "resident embedding graph inventory has wrong cardinality".to_string(),
                )
            })
    }

    fn warm(
        &self,
        embedding_graphs: &[ResidentEmbeddingStageGraph<'_>; MAX_BATCH],
        launches: u64,
        selected_batch_size: Option<u32>,
    ) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for batch in 1..=MAX_BATCH {
                if selected_batch_size.is_some_and(|selected| selected as usize != batch) {
                    continue;
                }
                embedding_graphs[batch - 1].graph().launch(&self.stream)?;
                self.program
                    .qualification_graph(self.routes[batch - 1])
                    .launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    fn cases<'a>(
        &'a self,
        embedding_graphs: &'a [ResidentEmbeddingStageGraph<'a>; MAX_BATCH],
        operation: &'static str,
    ) -> Result<Vec<ExactDeviceCase<'a>>, DeviceBenchmarkError> {
        (1..=MAX_BATCH)
            .map(|batch| {
                Ok(ExactDeviceCase::new(
                    operation,
                    format!("B={batch}"),
                    BenchmarkWorkload::warm_model_decode(
                        batch as u32,
                        *self.context_lengths[..batch]
                            .iter()
                            .max()
                            .expect("exact batch is nonempty") as u64,
                    ),
                    OperationAccounting::new(
                        logical_bytes(batch, &self.context_lengths[..batch]),
                        batch as u64,
                        "token",
                    ),
                    self.program.qualification_graph(self.routes[batch - 1]),
                    None,
                )
                .with_preparation(embedding_graphs[batch - 1].graph()))
            })
            .collect()
    }
}

fn benchmark_token_ids() -> [u32; MAX_BATCH] {
    core::array::from_fn(|slot| (100 + slot * 17) as u32)
}

fn benchmark_rope(positions: &[u32; MAX_BATCH]) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for slot in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let angle = f64::from(positions[slot]) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[slot * ROTARY_PAIRS + pair] = cos as f32;
            sine[slot * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn logical_bytes(batch: usize, context_lengths: &[usize]) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let vocab = Qwen38_27B::VOCAB;
    let resident_weights = 19_103_682_560usize;
    let dense_gdn_layer = 7_869_216usize;
    let dense_attention_layer = 1_988_624usize;
    let dense_mlp = 22 * hidden + 6 * intermediate + 4 * size_of::<f32>();
    let mut nvfp4_mlp = 20 * hidden + 4 * intermediate;
    if batch == 1 || batch >= 5 {
        nvfp4_mlp += hidden + hidden / 8;
    }
    let endpoint = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;
    // Per-token terms reuse the admitted direct leaf traffic formulas. Replacing
    // the common dense MLP term with NVFP4 is exact for the first 56 layers;
    // 64 duplicate plain boundary norms are absent from the fused model graph.
    let per_token = 48 * dense_gdn_layer + 16 * dense_attention_layer + endpoint
        - 56 * (dense_mlp - nvfp4_mlp)
        - 64 * 6 * hidden;
    let short_total = resident_weights + batch * per_token;
    if context_lengths
        .iter()
        .all(|&length| length == CONTEXT_TOKENS)
    {
        return short_total;
    }
    short_total - 16 * short_gqa_bytes(batch) + 16 * long_gqa_bytes(context_lengths)
}

fn short_gqa_bytes(batch: usize) -> usize {
    let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let cache = 2 * Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * Qwen38_27B::HEAD_DIM;
    let metadata =
        2 * size_of::<u32>() + Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    batch * (query + cache + metadata + output)
}

fn long_gqa_bytes(context_lengths: &[usize]) -> usize {
    let maximum_length = context_lengths
        .iter()
        .copied()
        .max()
        .expect("exact batch is nonempty");
    let required = maximum_length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
    let launched_partitions = LONG_CONTEXT_GQA_PARTITION_BUCKETS
        .iter()
        .copied()
        .find(|&partitions| partitions >= required)
        .expect("resident context has a graph partition bucket");
    context_lengths
        .iter()
        .copied()
        .map(|context_tokens| {
            let partitions = context_tokens.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
            let partials = Qwen38_27B::NUM_ATTENTION_HEADS * partitions;
            let partial_query = partials * Qwen38_27B::HEAD_DIM * size_of::<f32>();
            let cache = 2 * Qwen38_27B::NUM_ATTENTION_HEADS * context_tokens * Qwen38_27B::HEAD_DIM;
            let block_table = Qwen38_27B::NUM_ATTENTION_HEADS * context_tokens * size_of::<u32>();
            let metadata = Qwen38_27B::NUM_ATTENTION_HEADS
                * (launched_partitions + partitions + 1)
                * size_of::<u32>();
            let partial = partials
                * (2 * size_of::<f32>()
                    + Qwen38_27B::HEAD_DIM * size_of::<f32>()
                    + 3 * size_of::<f32>()
                    + Qwen38_27B::HEAD_DIM * size_of::<f32>());
            let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
            partial_query + cache + block_table + metadata + partial + output
        })
        .sum()
}

/// Replays the exact resident graph under an external profiler and emits its structural inventory.
pub fn profile_resident_model(
    root: &Path,
    batch: usize,
    warmup_launches: u64,
    captured_replays: u64,
    graph_dot: &Path,
) -> Result<ResidentModelProfileManifest, DeviceBenchmarkError> {
    if !(1..=MAX_BATCH).contains(&batch) || warmup_launches == 0 || captured_replays == 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "resident profile requires B=1..8 and nonzero warmup/captured replay counts"
                .to_string(),
        ));
    }
    let _preflight = preflight()?;
    let session = Session::new(root, BenchmarkProfile::Short)?;
    let embedding_graphs = session.embedding_graphs()?;
    session.warm(&embedding_graphs, warmup_launches, Some(batch as u32))?;
    require_current_process_exclusive()?;
    session
        .program
        .qualification_graph(session.routes[batch - 1])
        .debug_dot(graph_dot)?;
    profiler_start(&session._context)?;
    for _ in 0..captured_replays {
        embedding_graphs[batch - 1]
            .graph()
            .launch(&session.stream)?;
        session.stream.synchronize().map_err(GpuError::from)?;
        session
            .program
            .qualification_graph(session.routes[batch - 1])
            .launch(&session.stream)?;
        session.stream.synchronize().map_err(GpuError::from)?;
    }
    profiler_stop(&session._context)?;
    require_current_process_exclusive()?;

    Ok(resident_profile_manifest(
        session.program.layout(),
        batch,
        warmup_launches,
        captured_replays,
        graph_dot,
    ))
}

fn resident_profile_manifest(
    layout: &ResidentModelLayout,
    batch: usize,
    warmup_launches: u64,
    captured_replays: u64,
    graph_dot: &Path,
) -> ResidentModelProfileManifest {
    let mut stages = Vec::new();
    let mut next_node = 1usize;
    let mut push = |layer, component, source_route, kernel_families: Vec<&'static str>| {
        let kernel_nodes = kernel_families.len();
        stages.push(ResidentProfileStage {
            ordinal: stages.len() + 1,
            first_graph_node_ordinal: next_node,
            kernel_nodes,
            layer,
            component,
            source_route,
            kernel_families,
        });
        next_node += kernel_nodes;
    };
    push(None, "input_norm", "shared", vec!["rms_norm"]);
    for layer in 0..layout.layer_count() {
        let kind = layout
            .layer_kind(layer)
            .expect("resident layout layer count and kind inventory agree");
        let route = match kind {
            ResidentLayerKind::Nvfp4Gdn => "nvfp4_gdn",
            ResidentLayerKind::Nvfp4Attention => "nvfp4_attention",
            ResidentLayerKind::DenseFp8Gdn => "dense_fp8_gdn",
            ResidentLayerKind::DenseFp8Attention => "dense_fp8_attention",
        };
        match kind {
            ResidentLayerKind::Nvfp4Gdn | ResidentLayerKind::DenseFp8Gdn => {
                push(
                    Some(layer),
                    "gdn_input",
                    route,
                    vec!["quantize_activation_e4m3", "fp8_gdn_input"],
                );
                push(
                    Some(layer),
                    "gdn_prepare",
                    route,
                    vec!["gdn_control_exact", "gdn_convolution_exact"],
                );
                push(
                    Some(layer),
                    "gdn_recurrence",
                    route,
                    vec!["gdn_recurrence_exact"],
                );
                push(
                    Some(layer),
                    "gdn_output",
                    route,
                    vec!["gdn_output_quantize", "gdn_output_projection"],
                );
            }
            ResidentLayerKind::Nvfp4Attention | ResidentLayerKind::DenseFp8Attention => {
                push(
                    Some(layer),
                    "attention_qkv",
                    route,
                    vec!["quantize_activation_e4m3", "fp8_qkv"],
                );
                push(
                    Some(layer),
                    "attention_qk_prepare",
                    route,
                    vec!["attention_qk_prepare_exact"],
                );
                push(Some(layer), "paged_gqa", route, vec!["paged_gqa_exact"]);
                push(
                    Some(layer),
                    "attention_output",
                    route,
                    vec![
                        "attention_gate_quantize_exact",
                        "attention_output_projection",
                    ],
                );
            }
        }
        push(
            Some(layer),
            "post_mixer_residual_norm",
            route,
            vec!["residual_rms_norm"],
        );
        match kind {
            ResidentLayerKind::Nvfp4Gdn | ResidentLayerKind::Nvfp4Attention => {
                let swiglu = if batch == 1 || batch >= 5 {
                    vec!["nvfp4_quantize", "nvfp4_swiglu_w4a4"]
                } else {
                    vec!["nvfp4_swiglu_a16"]
                };
                push(Some(layer), "mlp_swiglu", route, swiglu);
                push(Some(layer), "mlp_down", route, vec!["nvfp4_down_a16"]);
            }
            ResidentLayerKind::DenseFp8Gdn | ResidentLayerKind::DenseFp8Attention => {
                push(
                    Some(layer),
                    "mlp_swiglu",
                    route,
                    vec!["fp8_swiglu_quantize", "fp8_swiglu_decode"],
                );
                push(
                    Some(layer),
                    "mlp_down",
                    route,
                    vec!["fp8_down_quantize", "fp8_down"],
                );
            }
        }
        push(
            Some(layer),
            "post_mlp_residual_norm",
            route,
            vec!["residual_rms_norm"],
        );
    }
    push(
        None,
        "lm_head",
        "text_endpoint",
        vec!["quantize_activation_e4m3", "fp8_lm_head"],
    );

    ResidentModelProfileManifest {
        schema_version: 1,
        suite: "resident_model/text_decode",
        batch_size: batch,
        context_tokens: CONTEXT_TOKENS,
        warmup_launches,
        captured_replays,
        graph_dot: graph_dot.display().to_string(),
        graph_kernel_nodes: next_node - 1,
        stages,
    }
}

/// Measures every exact complete-model graph directly without summing leaf medians.
pub fn benchmark_resident_model(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_resident_profile(
        root,
        options,
        BenchmarkProfile::Short,
        "bench-resident-model",
        "resident_model/text_decode",
    )
}

/// Measures every exact long-context complete-model graph directly.
pub fn benchmark_resident_long_context_model(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_resident_profile(
        root,
        options,
        BenchmarkProfile::Long,
        "bench-resident-long-context-model",
        "resident_long_context_model/text_decode",
    )
}

fn benchmark_resident_profile(
    root: &Path,
    options: DeviceBenchmarkOptions,
    profile: BenchmarkProfile,
    suite: &'static str,
    operation: &'static str,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let names = match profile {
        BenchmarkProfile::Short => [
            "resident_model/resident_weights",
            "resident_model/gdn_history",
            "resident_model/gdn_state",
            "resident_model/represented_kv_cache",
            "resident_model/kv_block_tables",
            "resident_model/shared_workspace",
            "resident_model/alignment_padding",
        ],
        BenchmarkProfile::Long => [
            "resident_long_context_model/resident_weights",
            "resident_long_context_model/gdn_history",
            "resident_long_context_model/gdn_state",
            "resident_long_context_model/represented_kv_cache",
            "resident_long_context_model/kv_block_tables",
            "resident_long_context_model/shared_workspace",
            "resident_long_context_model/alignment_padding",
        ],
    };
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, profile)?;
    let embedding_graphs = session.embedding_graphs()?;
    memory.register_owned(
        names[0],
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "64 exact source-routed layers plus final norm and LM head",
    )?;
    memory.register_owned(
        names[1],
        BenchmarkMemoryKind::Other,
        session.program.history_bytes(),
        "48 layers * 8 slots * 10,240 rows * 3 BF16 values",
    )?;
    memory.register_owned(
        names[2],
        BenchmarkMemoryKind::Other,
        session.program.state_bytes(),
        "48 layers * 8 slots * 48 FP32 128x128 matrices",
    )?;
    memory.register_owned(
        names[3],
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "16 layers * one shared 3,438-page pool * 4 heads * 64 * 256 E4M3 K/V values",
    )?;
    memory.register_owned(
        names[4],
        BenchmarkMemoryKind::Other,
        session.program.kv_table_bytes(),
        "8 stable slot rows * 3,438 u32 page-table entries",
    )?;
    memory.register_owned(
        names[5],
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "one max_batch=8 workspace shared sequentially by all layers and endpoint",
    )?;
    memory.register_owned(
        names[6],
        BenchmarkMemoryKind::Other,
        session.program.padding_bytes(),
        "256-byte alignment across the resident and shared-KV arenas",
    )?;
    memory.capture("after_setup")?;
    session.warm(&embedding_graphs, warmup_launches, options.batch_size)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(&embedding_graphs, operation)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite,
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust submission/completion and direct complete production graph replay",
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
    use super::{
        CONTEXT_TOKENS, LONG_CONTEXT_TOKENS, MAX_BATCH, logical_bytes, resident_profile_manifest,
    };
    use std::path::Path;
    use tuisko_engine::ResidentModelLayout;

    #[test]
    fn byte_accounting_tracks_the_exact_nvfp4_batch_routes() {
        let contexts = [CONTEXT_TOKENS; MAX_BATCH];
        let one = logical_bytes(1, &contexts[..1]);
        let two_per_token = (logical_bytes(2, &contexts[..2]) - 19_103_682_560) / 2;
        let five_per_token = (logical_bytes(5, &contexts[..5]) - 19_103_682_560) / 5;
        assert_eq!(one - 19_103_682_560, five_per_token);
        assert_eq!(five_per_token - two_per_token, 56 * (5_120 + 5_120 / 8));
        assert!(logical_bytes(MAX_BATCH, &contexts) > logical_bytes(1, &contexts[..1]));
    }

    #[test]
    fn long_context_accounting_tracks_one_deep_row_and_compact_survivors() {
        let mut contexts = [1; MAX_BATCH];
        contexts[0] = LONG_CONTEXT_TOKENS;
        for batch in 1..=MAX_BATCH {
            assert!(
                logical_bytes(batch, &contexts[..batch])
                    > logical_bytes(batch, &[CONTEXT_TOKENS; MAX_BATCH][..batch])
            );
        }
    }

    #[test]
    fn semantic_manifest_covers_every_resident_graph_kernel_node() {
        let layout = ResidentModelLayout::build().unwrap();
        let b1 = resident_profile_manifest(&layout, 1, 16, 3, Path::new("graph.dot"));
        let b2 = resident_profile_manifest(&layout, 2, 16, 3, Path::new("graph.dot"));

        assert_eq!(b1.stages.len(), 514);
        assert_eq!(b1.graph_kernel_nodes, 763);
        assert_eq!(b2.graph_kernel_nodes, 707);
        assert_eq!(b1.stages.first().unwrap().first_graph_node_ordinal, 1);
        let last = b1.stages.last().unwrap();
        assert_eq!(last.first_graph_node_ordinal + last.kernel_nodes - 1, 763);
    }
}
