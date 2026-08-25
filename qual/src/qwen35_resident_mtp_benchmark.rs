//! Direct timing for resident Qwen3.5 MTP draft graphs.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35ResidentMtpProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const MTP_WEIGHT_BYTES: usize = 486_581_248;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen35ResidentMtpProgram,
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
        let mut program = Qwen35ResidentMtpProgram::from_snapshot(&context, snapshot)?;
        let slots = (0..MAX_BATCH).collect::<Vec<_>>();
        for &slot in &slots {
            program.activate_kv_slot(slot)?;
            program.reserve_kv_slot_tokens(&stream, slot, CONTEXT_TOKENS)?;
        }
        let positions = [CACHE_POSITION; MAX_BATCH];
        let (cosine, sine) = benchmark_rope();
        program.stage_target_embeddings(&stream, &target_token_ids())?;
        program.load_decode_state(&stream, &positions, &slots, &cosine, &sine)?;
        program.replay_target(&stream, MAX_BATCH)?;
        program.stage_mtp_embeddings(&stream, &draft_token_ids())?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                Ok(RouteGraph {
                    batch,
                    repeated: program.qualification_repeated_draft_graph(
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
                self.program.replay_draft(&self.stream, batch)?;
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
                    "qwen35_9b/mtp/resident_draft",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_mtp(route.batch as u64),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "draft",
                    ),
                    self.program.qualification_draft_graph(route.batch)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
            })
            .collect()
    }
}

/// Measures every exact MTP layer plus shared-LM-head `B=1..8` graph.
pub fn benchmark_qwen35_resident_mtp(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    let layout = session.program.layout();
    for (name, kind, bytes, description) in [
        (
            "qwen35_9b/mtp/resident_weights",
            BenchmarkMemoryKind::Weights,
            layout.resident_weight_bytes(),
            "target weights plus one source-BF16 MTP layer with a shared endpoint",
        ),
        (
            "qwen35_9b/mtp/resident_kv_cache",
            BenchmarkMemoryKind::KvCache,
            layout.cache_bytes(),
            "target long-context cache plus the separate BF16 MTP mirror",
        ),
        (
            "qwen35_9b/mtp/resident_workspace",
            BenchmarkMemoryKind::Workspace,
            layout.workspace_bytes(),
            "target and MTP stable workspaces and page tables",
        ),
        (
            "qwen35_9b/mtp/resident_padding",
            BenchmarkMemoryKind::Other,
            layout.padding_bytes(),
            "alignment across the target, MTP layer, and MTP cache arenas",
        ),
    ] {
        memory.register_owned(name, kind, bytes, description)?;
    }
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
            suite: "bench-qwen35-resident-mtp",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust production-graph submission/completion and repeated device timing for one complete source-BF16 MTP draft layer plus the shared Qwen3.5 BF16 LM head",
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

fn target_token_ids() -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| ((101 + row * 7_919) % Qwen35_9B::VOCAB) as u32)
}

fn draft_token_ids() -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| ((211 + row * 65_537) % Qwen35_9B::VOCAB) as u32)
}

fn benchmark_rope() -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for row in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let (sin, cos) = (f64::from(CACHE_POSITION) * frequency).sin_cos();
            cosine[row * ROTARY_PAIRS + pair] = cos as f32;
            sine[row * ROTARY_PAIRS + pair] = sin as f32;
        }
    }

    (cosine, sine)
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let qkv = Qwen35_9B::ATTENTION_QKV_ROWS;
    let attention = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let vocab = Qwen35_9B::VOCAB;
    let weights = MTP_WEIGHT_BYTES + 2 * vocab * hidden;
    let cache_reads = 2 * size_of::<u16>() * batch * CONTEXT_TOKENS * attention;
    let layer_per_row = 2 * (13 * hidden + qkv + 3 * attention + intermediate)
        + 2 * ROTARY_PAIRS * size_of::<f32>();
    let lm_head_per_row = 2 * hidden + 2 * vocab;

    weights + cache_reads + batch * (layer_per_row + lm_head_per_row)
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_TOKENS, MAX_BATCH, logical_bytes};

    #[test]
    fn qwen35_resident_mtp_suite_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 2_523_646_208);
        assert_eq!(logical_bytes(8), 2_543_438_848);
    }
}
