//! Direct complete-graph timing for the resident text model.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, ResidentEmbeddingStageGraph, ResidentModelProgram};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;

struct Session {
    timer: GpuTimer,
    program: ResidentModelProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path) -> Result<Self, DeviceBenchmarkError> {
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
        let timer = GpuTimer::new(&context)?;
        Ok(Self {
            timer,
            program,
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
    ) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for batch in 1..=MAX_BATCH {
                embedding_graphs[batch - 1].graph().launch(&self.stream)?;
                self.program
                    .qualification_graph(batch)?
                    .launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    fn cases<'a>(
        &'a self,
        embedding_graphs: &'a [ResidentEmbeddingStageGraph<'a>; MAX_BATCH],
    ) -> Result<Vec<ExactDeviceCase<'a>>, DeviceBenchmarkError> {
        (1..=MAX_BATCH)
            .map(|batch| {
                Ok(ExactDeviceCase::new(
                    "resident_model/text_decode",
                    format!("B={batch}"),
                    BenchmarkWorkload::warm_model_decode(batch as u32, CONTEXT_TOKENS as u64),
                    OperationAccounting::new(logical_bytes(batch), batch as u64, "token"),
                    self.program.qualification_graph(batch)?,
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
    resident_weights + batch * per_token
}

/// Measures every exact complete-model graph directly without summing leaf medians.
pub fn benchmark_resident_model(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root)?;
    let embedding_graphs = session.embedding_graphs()?;
    memory.register_owned(
        "resident_model/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "64 exact source-routed layers plus final norm and LM head",
    )?;
    memory.register_owned(
        "resident_model/gdn_history",
        BenchmarkMemoryKind::Other,
        session.program.history_bytes(),
        "48 layers * 8 slots * 10,240 rows * 3 BF16 values",
    )?;
    memory.register_owned(
        "resident_model/gdn_state",
        BenchmarkMemoryKind::Other,
        session.program.state_bytes(),
        "48 layers * 8 slots * 48 FP32 128x128 matrices",
    )?;
    memory.register_owned(
        "resident_model/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "16 layers * one shared 3,438-page pool * 4 heads * 64 * 256 E4M3 K/V values",
    )?;
    memory.register_owned(
        "resident_model/kv_block_tables",
        BenchmarkMemoryKind::Other,
        session.program.kv_table_bytes(),
        "8 stable slot rows * 3,438 u32 page-table entries",
    )?;
    memory.register_owned(
        "resident_model/shared_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "one max_batch=8 workspace shared sequentially by all layers and endpoint",
    )?;
    memory.register_owned(
        "resident_model/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.padding_bytes(),
        "256-byte alignment across the resident and shared-KV arenas",
    )?;
    memory.capture("after_setup")?;
    session.warm(&embedding_graphs, warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(&embedding_graphs)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-resident-model",
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
    use super::{MAX_BATCH, logical_bytes};

    #[test]
    fn byte_accounting_tracks_the_exact_nvfp4_batch_routes() {
        let one = logical_bytes(1);
        let two_per_token = (logical_bytes(2) - 19_103_682_560) / 2;
        let five_per_token = (logical_bytes(5) - 19_103_682_560) / 5;
        assert_eq!(one - 19_103_682_560, five_per_token);
        assert_eq!(five_per_token - two_per_token, 56 * (5_120 + 5_120 / 8));
        assert!(logical_bytes(MAX_BATCH) > logical_bytes(1));
    }
}
