//! Direct timing for the complete source-backed Qwen3.5 MTP draft layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35MtpLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;
const RESIDENT_WEIGHT_BYTES: usize = 486_581_248;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen35MtpLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = CheckpointSnapshot::<Qwen35_9B>::open(root)?;
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = Qwen35MtpLayerProgram::from_snapshot(&context, &snapshot)?;
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
                    "qwen35_9b/mtp/layer",
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
    let values = MAX_BATCH * Qwen35_9B::HIDDEN;
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
    let hidden = Qwen35_9B::HIDDEN;
    let qkv = Qwen35_9B::ATTENTION_QKV_ROWS;
    let attention = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let cache_reads = 2 * size_of::<u16>() * batch * CONTEXT_TOKENS * attention;
    let per_row = 2 * (13 * hidden + qkv + 3 * attention + intermediate)
        + 2 * ROTARY_PAIRS * size_of::<f32>();

    RESIDENT_WEIGHT_BYTES + cache_reads + batch * per_row
}

/// Measures every exact complete Qwen3.5 MTP draft-layer graph.
pub fn benchmark_qwen35_mtp_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen35_9b/mtp/layer/represented_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "unchanged source-BF16 MTP matrices and norms",
    )?;
    memory.register_owned(
        "qwen35_9b/mtp/layer/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "8 slots * 3 pages * 4 KV heads * 64 tokens * 256 BF16 values * K/V",
    )?;
    memory.register_owned(
        "qwen35_9b/mtp/layer/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 complete draft seams without the shared text endpoint",
    )?;
    memory.register_owned(
        "qwen35_9b/mtp/layer/alignment_padding",
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
            suite: "bench-qwen35-mtp-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust submission/completion and repeated complete source-BF16 Qwen3.5 MTP draft-layer route",
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
    use super::{CONTEXT_TOKENS, MAX_BATCH, logical_bytes};

    #[test]
    fn qwen35_mtp_layer_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 488_903_936);
        assert_eq!(logical_bytes(8), 505_162_752);
    }
}
