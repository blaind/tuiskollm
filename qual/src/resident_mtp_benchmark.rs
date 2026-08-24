//! Direct timing for resident long-context Qwen3.8 MTP draft graphs.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, ResidentModelProgram, ResidentMtpDraftRoute, ResidentMtpProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const CACHE_POSITION: u32 = 130;
const CONTEXT_TOKENS: usize = CACHE_POSITION as usize + 1;
const ROTARY_PAIRS: usize = 32;

struct RouteGraph {
    route: ResidentMtpDraftRoute,
    repeated_draft: CudaGraph,
    repeated_continuation: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: ResidentMtpProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
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
        let mut target = ResidentModelProgram::from_snapshot(&context, snapshot)?;
        for slot in 0..MAX_BATCH {
            target.activate_kv_slot(slot)?;
            target.reserve_kv_slot_tokens(&stream, slot, 1_024)?;
        }
        let mut program = ResidentMtpProgram::from_target(target)?;
        let slots = (0..MAX_BATCH).collect::<Vec<_>>();
        let positions = [CACHE_POSITION; MAX_BATCH];
        let token_ids = token_ids(MAX_BATCH);
        let hidden = hidden_fixture();
        let (cosine, sine) = benchmark_rope();
        program
            .target()
            .load_residual(&stream, MAX_BATCH, &hidden)?;
        let staged =
            program.stage_draft(&stream, &slots, &positions, &token_ids, &cosine, &sine)?;
        if staged.batch() != MAX_BATCH {
            return Err(DeviceBenchmarkError::Precondition(
                "resident MTP benchmark did not stage the B=8 prefix superset".to_string(),
            ));
        }
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                let route = ResidentMtpDraftRoute::qualified(batch)?;
                Ok(RouteGraph {
                    route,
                    repeated_draft: program.qualification_repeated_draft_graph(
                        &stream,
                        batch,
                        repeated_operations,
                    )?,
                    repeated_continuation: program.qualification_repeated_continue_draft_graph(
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
            for route in &self.routes {
                self.program.replay_draft(&self.stream, route.route)?;
                self.program
                    .replay_continue_draft(&self.stream, route.route)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    fn cases(
        &self,
        repeated_operations: u64,
    ) -> Result<Vec<ExactDeviceCase<'_>>, DeviceBenchmarkError> {
        let mut cases = Vec::with_capacity(2 * MAX_BATCH);
        for route in &self.routes {
            let batch = route.route.batch();
            cases.push(ExactDeviceCase::new(
                "qwen3_8/mtp/resident_draft",
                format!("B={batch}"),
                BenchmarkWorkload::warm_operator_mtp(batch as u64),
                OperationAccounting::new(logical_bytes(batch), batch as u64, "draft"),
                self.program.qualification_draft_graph(route.route)?,
                Some(RepeatedGraph::new(
                    &route.repeated_draft,
                    repeated_operations,
                )),
            ));
            cases.push(ExactDeviceCase::new(
                "qwen3_8/mtp/resident_continuation",
                format!("B={batch}"),
                BenchmarkWorkload::warm_operator_mtp(batch as u64),
                OperationAccounting::new(logical_bytes(batch), batch as u64, "continuation"),
                self.program
                    .qualification_continue_draft_graph(route.route)?,
                Some(RepeatedGraph::new(
                    &route.repeated_continuation,
                    repeated_operations,
                )),
            ));
        }
        Ok(cases)
    }
}

/// Measures every exact resident MTP seeded-draft and continuation `B=1..8` graph directly.
pub fn benchmark_resident_mtp(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    let target = session.program.target();
    for (name, kind, bytes, description) in [
        (
            "resident_mtp/target_weights",
            BenchmarkMemoryKind::Weights,
            target.resident_weight_bytes(),
            "64 exact source-routed target layers plus shared final norm and LM head",
        ),
        (
            "resident_mtp/target_gdn_history",
            BenchmarkMemoryKind::Other,
            target.history_bytes(),
            "48 layers * 8 persistent causal-history slots",
        ),
        (
            "resident_mtp/target_gdn_state",
            BenchmarkMemoryKind::Other,
            target.state_bytes(),
            "48 layers * 8 persistent recurrent-state slots",
        ),
        (
            "resident_mtp/target_kv_cache",
            BenchmarkMemoryKind::KvCache,
            target.cache_bytes(),
            "16 target layers sharing the exact 3,438-page pool",
        ),
        (
            "resident_mtp/target_kv_tables",
            BenchmarkMemoryKind::Other,
            target.kv_table_bytes(),
            "8 target slot rows * 3,438 physical-page entries",
        ),
        (
            "resident_mtp/target_workspace",
            BenchmarkMemoryKind::Workspace,
            target.workspace_bytes(),
            "target address-stable route workspace supplying raw residual handoff rows",
        ),
        (
            "resident_mtp/target_tensor_maps",
            BenchmarkMemoryKind::Other,
            target.descriptor_bytes(),
            "eight dense target layers * four address-bound tensor maps",
        ),
        (
            "resident_mtp/target_padding",
            BenchmarkMemoryKind::Other,
            target.padding_bytes(),
            "target resident and KV arena alignment",
        ),
        (
            "resident_mtp/mtp_weights",
            BenchmarkMemoryKind::Weights,
            session.program.resident_weight_bytes(),
            "one unchanged source-BF16 MTP weight set without duplicated endpoint weights",
        ),
        (
            "resident_mtp/mtp_kv_cache",
            BenchmarkMemoryKind::KvCache,
            session.program.cache_bytes(),
            "one BF16 K/V mirror using the target physical-page lifecycle",
        ),
        (
            "resident_mtp/mtp_workspace",
            BenchmarkMemoryKind::Workspace,
            session.program.workspace_bytes(),
            "maximum prompt plus exact B=8 full-draft seams",
        ),
        (
            "resident_mtp/mtp_padding",
            BenchmarkMemoryKind::Other,
            session.program.padding_bytes(),
            "two 256-byte-aligned MTP arenas",
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
            suite: "bench-resident-mtp",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust production-graph submission/completion and repeated resident seeded target-handoff or prior-residual continuation through the long-context MTP route",
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

fn token_ids(rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|row| ((row * 7_919 + 101) % Qwen38_27B::VOCAB) as u32)
        .collect()
}

fn hidden_fixture() -> Vec<u16> {
    const PATTERN: [f32; 8] = [
        0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
    ];
    (0..MAX_BATCH * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(3 * index + 1) & 7]))
        .collect()
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
    let hidden = Qwen38_27B::HIDDEN;
    let qkv = Qwen38_27B::ATTENTION_QKV_ROWS;
    let attention = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let vocab = Qwen38_27B::VOCAB;
    let weights = 2_121_293_824;
    let table_handoff = 2 * MAX_BATCH * 3_438 * size_of::<u32>();
    let cache_reads = 2 * size_of::<u16>() * batch * CONTEXT_TOKENS * attention;
    let route_uploads = 2 * hidden + 3 * size_of::<u32>() + 2 * ROTARY_PAIRS * size_of::<f32>();
    let residual_handoff = 2 * 2 * hidden;
    let internal = 2 * (13 * hidden + qkv + 3 * attention + intermediate + vocab)
        + 2 * ROTARY_PAIRS * size_of::<f32>();
    weights + table_handoff + cache_reads + batch * (route_uploads + residual_handoff + internal)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_TOKENS, MAX_BATCH, logical_bytes};

    #[test]
    fn resident_mtp_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(CONTEXT_TOKENS, 131);
        assert_eq!(logical_bytes(1), 2_125_301_900);
        assert_eq!(logical_bytes(8), 2_151_818_208);
    }
}
