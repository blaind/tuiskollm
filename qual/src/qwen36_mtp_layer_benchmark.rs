//! Direct timing for the complete source-backed Qwen3.6 MTP draft layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen36MtpLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const NON_ROUTED_WEIGHT_BYTES: usize = 78_668_800;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen36MtpLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(root)?;
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = Qwen36MtpLayerProgram::from_snapshot(&context, &snapshot)?;
        let (embedding, hidden) = benchmark_inputs();
        program.load_inputs(&stream, MAX_BATCH, &embedding, &hidden)?;
        program.reset_cache(&stream)?;
        let (cosine, sine) = benchmark_rope();
        program.load_draft_state(
            &stream,
            MAX_BATCH,
            &[CACHE_POSITION; MAX_BATCH],
            &cosine,
            &sine,
        )?;
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
                let graph = self.program.qualification_draft_graph(batch)?;
                // SAFETY: the program retains every captured allocation and module.
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
                    "qwen36_35b_a3b/mtp/layer",
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

fn benchmark_inputs() -> (Vec<u16>, Vec<u16>) {
    const PATTERN: [f32; 8] = [
        0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
    ];
    let values = MAX_BATCH * Qwen36Moe35B::HIDDEN;
    let embedding = (0..values)
        .map(|index| f32_to_bf16(PATTERN[index & 7]))
        .collect();
    let hidden = (0..values)
        .map(|index| f32_to_bf16(PATTERN[(3 * index + 1) & 7]))
        .collect();

    (embedding, hidden)
}

fn benchmark_rope() -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0; MAX_BATCH * ROTARY_PAIRS];
    for row in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let angle = f64::from(CACHE_POSITION) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[row * ROTARY_PAIRS + pair] = cos as f32;
            sine[row * ROTARY_PAIRS + pair] = sin as f32;
        }
    }

    (cosine, sine)
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let qkv = Qwen36Moe35B::ATTENTION_QKV_ROWS;
    let attention = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
    let intermediate = Qwen36Moe35B::INTERMEDIATE;
    let experts = Qwen36Moe35B::NUM_EXPERTS;
    let top_k = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
    let slots = top_k + 1;
    let routed_expert_weight =
        (2 * intermediate * hidden + hidden * intermediate) * size_of::<u16>();
    // The owner retains all 256 experts, but each represented row dispatches
    // exactly eight 6,291,456-byte gate/up plus down source families.
    let selected_weight_reads = batch * top_k * routed_expert_weight;
    let cache_reads = 2 * batch * CONTEXT_TOKENS * attention;
    let per_row = 2
        * (13 * hidden
            + qkv
            + 3 * attention
            + experts
            + 2 * top_k
            + slots * (intermediate + hidden)
            + 1)
        + 2 * ROTARY_PAIRS * size_of::<f32>();

    NON_ROUTED_WEIGHT_BYTES + selected_weight_reads + cache_reads + batch * per_row
}

/// Measures every exact complete Qwen3.6 MTP draft-layer graph.
pub fn benchmark_qwen36_mtp_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen36_35b_a3b/mtp/layer/represented_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "unchanged source-BF16 MTP matrices and norms",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/mtp/layer/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "8 slots * 3 pages * 2 KV heads * 64 tokens * 256 E4M3 values * K/V",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/mtp/layer/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 complete draft seams without the shared text endpoint",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/mtp/layer/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes() - session.program.owner_bytes(),
        "single 256-byte-aligned owner arena",
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
            suite: "bench-qwen36-mtp-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust submission/completion and repeated complete source-BF16 Qwen3.6 MTP draft-layer route",
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
    use super::{CONTEXT_TOKENS, MAX_BATCH, logical_bytes};

    #[test]
    fn qwen36_mtp_layer_suite_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 130_216_738);
        assert_eq!(logical_bytes(8), 491_052_304);
    }
}
