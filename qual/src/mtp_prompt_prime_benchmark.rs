//! Direct timing for target-residual handoff and source-backed MTP prompt priming.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::attention::rope_tables;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MtpPromptPrimeProgram, MtpPromptPrimeRoute, ResidentModelProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const ROUTES: [usize; 5] = [1, 32, 64, 128, 1_024];
const ROTARY_PAIRS: usize = 32;
const SLOT: usize = 0;

struct RouteGraph {
    route: MtpPromptPrimeRoute,
    repeated: CudaGraph,
}

struct Session<'a> {
    routes: Vec<RouteGraph>,
    program: MtpPromptPrimeProgram<'a>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl<'a> Session<'a> {
    fn new(
        target: &'a ResidentModelProgram,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        repeated_operations: u64,
    ) -> Result<Self, DeviceBenchmarkError> {
        let mut program = MtpPromptPrimeProgram::from_target(target)?;
        let tokens = token_ids(1, ROUTES[ROUTES.len() - 1]);
        let positions = positions(ROUTES[ROUTES.len() - 1]);
        let (cosine, sine) = rope(&positions);
        let _staged_route = program.stage(
            stream,
            ROUTES[ROUTES.len() - 1],
            SLOT,
            0,
            &tokens,
            &cosine,
            &sine,
        )?;

        // The staged T=1024 metadata and next-token embeddings are exact prefix
        // supersets for the four smaller graphs, so timing never includes host restaging.
        let routes = ROUTES
            .into_iter()
            .map(|rows| {
                let route = MtpPromptPrimeRoute::qualified(rows, SLOT, 0)?;
                Ok(RouteGraph {
                    route,
                    repeated: program.qualification_repeated_graph(
                        stream,
                        route,
                        repeated_operations,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;
        Ok(Self {
            routes,
            program,
            stream: stream.clone(),
            _context: context.clone(),
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for route in &self.routes {
                self.program.replay(&self.stream, route.route)?;
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
                let rows = route.route.rows();
                Ok(ExactDeviceCase::new(
                    "qwen3_8/mtp/prompt_prime",
                    if rows == 1 {
                        "K=1".to_string()
                    } else {
                        format!("T={rows}")
                    },
                    BenchmarkWorkload::warm_operator_prefill(rows as u64),
                    OperationAccounting::new(logical_bytes(rows), rows as u64, "prompt_token"),
                    self.program.qualification_graph(route.route)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
            })
            .collect()
    }
}

fn prepare_target(
    root: &Path,
) -> Result<(Arc<CudaContext>, Arc<CudaStream>, ResidentModelProgram), DeviceBenchmarkError> {
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
    target.activate_kv_slot(SLOT)?;
    target.reserve_kv_slot_tokens(&stream, SLOT, ROUTES[ROUTES.len() - 1])?;
    target.reset_slot(&stream, SLOT)?;
    target.stage_embeddings(&stream, &token_ids(0, ROUTES[ROUTES.len() - 1]))?;
    let positions = positions(ROUTES[ROUTES.len() - 1]);
    let (cosine, sine) = rope(&positions);
    let route =
        target.load_prefill_state(&stream, ROUTES[ROUTES.len() - 1], SLOT, &cosine, &sine)?;
    target.replay_prefill(&stream, route)?;
    stream.synchronize().map_err(GpuError::from)?;
    Ok((context, stream, target))
}

fn token_ids(first: usize, rows: usize) -> Vec<u32> {
    (first..first + rows)
        .map(|position| {
            ((position.wrapping_mul(7_919).wrapping_add(101)) % Qwen38_27B::VOCAB) as u32
        })
        .collect()
}

fn positions(rows: usize) -> Vec<u32> {
    (0..rows).map(|position| position as u32).collect()
}

fn rope(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    rope_tables(positions, ROTARY_PAIRS, 2 * ROTARY_PAIRS, 10_000_000.0)
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let qkv = Qwen38_27B::ATTENTION_QKV_ROWS;
    let attention = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    let kv = Qwen38_27B::ATTENTION_KV_ROWS;
    let fixed = 251_689_984 + 2 * 8 * 3_438 * size_of::<u32>();
    let uploads = 2 * hidden + 2 * size_of::<u32>() + 2 * ROTARY_PAIRS * size_of::<f32>();
    let target_handoff = 2 * 2 * hidden;
    let three_norm_seams = 3 * 2 * 2 * hidden;
    let fusion_seams = 2 * (2 * hidden) + 2 * hidden;
    let qkv_seams = 2 * hidden + 2 * qkv + 2 * qkv;
    let qk_outputs = attention * size_of::<f32>() + 2 * kv * size_of::<u16>();
    fixed
        + rows
            * (uploads + target_handoff + three_norm_seams + fusion_seams + qkv_seams + qk_outputs)
}

/// Measures every exact target-to-MTP prompt-prime production graph directly.
pub fn benchmark_mtp_prompt_prime(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let (context, stream, target) = prepare_target(root)?;
    let session = Session::new(&target, &context, &stream, options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    for (name, kind, bytes, description) in [
        (
            "mtp_prompt_prime/target_weights",
            BenchmarkMemoryKind::Weights,
            target.resident_weight_bytes(),
            "64 exact source-routed target layers plus final norm and LM head",
        ),
        (
            "mtp_prompt_prime/target_gdn_history",
            BenchmarkMemoryKind::Other,
            target.history_bytes(),
            "48 layers * 8 persistent causal-history slots",
        ),
        (
            "mtp_prompt_prime/target_gdn_state",
            BenchmarkMemoryKind::Other,
            target.state_bytes(),
            "48 layers * 8 persistent FP32 recurrent-state slots",
        ),
        (
            "mtp_prompt_prime/target_kv_cache",
            BenchmarkMemoryKind::KvCache,
            target.cache_bytes(),
            "16 target layers sharing the exact 220,000-token physical page pool",
        ),
        (
            "mtp_prompt_prime/target_kv_tables",
            BenchmarkMemoryKind::Other,
            target.kv_table_bytes(),
            "8 target slot rows * 3,438 physical-page entries",
        ),
        (
            "mtp_prompt_prime/target_workspace",
            BenchmarkMemoryKind::Workspace,
            target.workspace_bytes(),
            "target maximum-route address-stable workspace supplying the handoff",
        ),
        (
            "mtp_prompt_prime/target_tensor_maps",
            BenchmarkMemoryKind::Other,
            target.descriptor_bytes(),
            "eight dense target layers * four address-bound tensor maps",
        ),
        (
            "mtp_prompt_prime/target_padding",
            BenchmarkMemoryKind::Other,
            target.padding_bytes(),
            "256-byte target arena alignment",
        ),
        (
            "mtp_prompt_prime/represented_weights",
            BenchmarkMemoryKind::Weights,
            session.program.resident_weight_bytes(),
            "unchanged source-BF16 fusion, normalization, and QKV weights",
        ),
        (
            "mtp_prompt_prime/represented_kv_cache",
            BenchmarkMemoryKind::KvCache,
            session.program.cache_bytes(),
            "complete MTP K/V mirror using the target physical-page inventory",
        ),
        (
            "mtp_prompt_prime/address_stable_workspace",
            BenchmarkMemoryKind::Workspace,
            session.program.workspace_bytes(),
            "maximum T=1024 prompt seams and copied target page tables",
        ),
        (
            "mtp_prompt_prime/alignment_padding",
            BenchmarkMemoryKind::Other,
            session.program.padding_bytes(),
            "single 256-byte-aligned MTP prompt arena",
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
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-mtp-prompt-prime",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust production-graph submission/completion and repeated exact target-residual handoff plus MTP prompt-prime route",
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
    use super::{ROUTES, logical_bytes};

    #[test]
    fn prompt_prime_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(ROUTES, [1, 32, 64, 128, 1_024]);
        assert!(logical_bytes(1) > 251_689_984);
        for routes in ROUTES.windows(2) {
            assert!(logical_bytes(routes[1]) > logical_bytes(routes[0]));
        }
    }
}
