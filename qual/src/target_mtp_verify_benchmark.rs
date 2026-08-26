//! Direct complete-model timing for target MTP verification and accepted-prefix commit.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::oracles::attention::rope_tables;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    MAX_BATCH, ResidentModelProgram, ResidentMtpSegmentedStageGraph,
    ResidentMtpSegmentedVerifyRoute,
};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const VERIFY_ROUTES: usize = 4;
const FIRST_POSITION: usize = 128;
const ROTARY_PAIRS: usize = 32;
const TOKEN_IDS: [u32; VERIFY_ROUTES] = [101, 7_919, 48_127, 199_933];

struct RouteGraphs {
    route: ResidentMtpSegmentedVerifyRoute,
    _accepted: Vec<usize>,
    verify_commit: CudaGraph,
}

struct Session {
    program: ResidentModelProgram,
    routes: Vec<RouteGraphs>,
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
        for slot in 0..MAX_BATCH {
            program.activate_kv_slot(slot)?;
            program.reserve_kv_slot_tokens(&stream, slot, 192)?;
        }
        let staged_tokens = (0..MAX_BATCH)
            .flat_map(|lane| {
                TOKEN_IDS
                    .iter()
                    .map(move |&token| (token + lane as u32 * 8_191) % Qwen38_27B::VOCAB as u32)
            })
            .collect::<Vec<_>>();
        program.stage_target_mtp_segmented_embeddings(&stream, &staged_tokens)?;
        program.reset_state(&stream)?;

        let mut exact_routes = Vec::with_capacity(MAX_BATCH * VERIFY_ROUTES);
        for batch in 1..=MAX_BATCH {
            for tokens in 1..=VERIFY_ROUTES {
                let positions = (0..batch)
                    .flat_map(|_| FIRST_POSITION..FIRST_POSITION + tokens)
                    .map(|position| position as u32)
                    .collect::<Vec<_>>();
                let slots = (0..batch).collect::<Vec<_>>();
                let first_positions = vec![FIRST_POSITION; batch];
                let (cosine, sine) = rope(&positions);
                exact_routes.push(program.load_target_mtp_segmented_verify_state(
                    &stream,
                    tokens,
                    &slots,
                    &first_positions,
                    &cosine,
                    &sine,
                )?);
            }
        }
        let routes = exact_routes
            .into_iter()
            .map(|route| {
                let accepted = vec![route.tokens(); route.batch()];
                Ok(RouteGraphs {
                    route,
                    verify_commit: program.qualification_repeated_target_mtp_segmented_graph(
                        &stream,
                        route,
                        Some(&accepted),
                        1,
                    )?,
                    _accepted: accepted,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;
        Ok(Self {
            program,
            routes,
            stream,
            _context: context,
        })
    }

    fn stage_graphs(
        &self,
    ) -> Result<Vec<ResidentMtpSegmentedStageGraph<'_>>, DeviceBenchmarkError> {
        self.routes
            .iter()
            .map(|entry| {
                let route = entry.route;
                let positions = (0..route.batch())
                    .flat_map(|_| FIRST_POSITION..FIRST_POSITION + route.tokens())
                    .map(|position| position as u32)
                    .collect::<Vec<_>>();
                let (cosine, sine) = rope(&positions);
                let slots = (0..route.batch()).collect::<Vec<_>>();
                let first_positions = vec![FIRST_POSITION; route.batch()];
                Ok(self
                    .program
                    .qualification_target_mtp_segmented_stage_graph(
                        &self.stream,
                        route,
                        &slots,
                        &first_positions,
                        &cosine,
                        &sine,
                    )?)
            })
            .collect()
    }

    fn warm(
        &self,
        stage_graphs: &[ResidentMtpSegmentedStageGraph<'_>],
        launches: u64,
    ) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for (route, stage) in self.routes.iter().zip(stage_graphs) {
                // SAFETY: the stage graph borrows this Session's program, which
                // owns every captured device allocation, and itself retains its
                // pinned sources; all outlive the replays and the synchronize.
                unsafe { stage.graph().launch(&self.stream) }?;
                let verify = self
                    .program
                    .qualification_target_mtp_segmented_verify_graph(route.route)?;
                // SAFETY: this Session's program owns the graph and every
                // allocation it captured, outliving the replays and synchronize.
                unsafe { verify.launch(&self.stream) }?;
                // SAFETY: as for the stage replay above.
                unsafe { stage.graph().launch(&self.stream) }?;
                // SAFETY: this Session owns both the route graphs and the
                // program whose allocations they captured, all alive across
                // these replays and the synchronize below.
                unsafe { route.verify_commit.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    fn cases<'a>(
        &'a self,
        stage_graphs: &'a [ResidentMtpSegmentedStageGraph<'a>],
    ) -> Result<Vec<ExactDeviceCase<'a>>, DeviceBenchmarkError> {
        let mut cases = Vec::with_capacity(2 * MAX_BATCH * VERIFY_ROUTES);
        for (route, stage) in self.routes.iter().zip(stage_graphs) {
            let tokens = route.route.tokens();
            let batch = route.route.batch();
            let context = route.route.maximum_length();
            cases.push(
                ExactDeviceCase::new(
                    "resident_model/target_mtp_segmented_verify",
                    format!("B={batch},K={tokens},context={context},verify"),
                    BenchmarkWorkload::warm_model_mtp((batch * tokens) as u64, context as u64),
                    OperationAccounting::new(
                        batch * verify_logical_bytes(tokens),
                        (batch * tokens) as u64,
                        "target_token",
                    ),
                    self.program
                        .qualification_target_mtp_segmented_verify_graph(route.route)?,
                    None,
                )
                .with_preparation(stage.graph()),
            );
            cases.push(
                ExactDeviceCase::new(
                    "resident_model/target_mtp_segmented_verify_commit",
                    format!("B={batch},K={tokens},context={context},verify+commit={tokens}"),
                    BenchmarkWorkload::warm_model_mtp((batch * tokens) as u64, context as u64),
                    OperationAccounting::new(
                        batch * (verify_logical_bytes(tokens) + commit_logical_bytes(tokens)),
                        (batch * tokens) as u64,
                        "accepted_token",
                    ),
                    &route.verify_commit,
                    None,
                )
                .with_preparation(stage.graph()),
            );
        }
        Ok(cases)
    }
}

fn rope(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    rope_tables(positions, ROTARY_PAIRS, 2 * ROTARY_PAIRS, 10_000_000.0)
}

fn short_gqa_bytes(lengths: impl IntoIterator<Item = usize>) -> usize {
    lengths
        .into_iter()
        .map(|length| {
            let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
            let cache = 2 * Qwen38_27B::NUM_ATTENTION_HEADS * length * Qwen38_27B::HEAD_DIM;
            let metadata =
                2 * size_of::<u32>() + Qwen38_27B::NUM_ATTENTION_HEADS * length * size_of::<u32>();
            let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
            query + cache + metadata + output
        })
        .sum()
}

fn verify_logical_bytes(tokens: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let vocab = Qwen38_27B::VOCAB;
    let resident_weights = 19_103_682_560usize;
    let dense_gdn_layer = 7_869_216usize;
    let dense_attention_layer = 1_988_624usize;
    let dense_mlp = 22 * hidden + 6 * intermediate + 4 * size_of::<f32>();
    let mut nvfp4_mlp = 20 * hidden + 4 * intermediate;
    if tokens == 1 {
        nvfp4_mlp += hidden + hidden / 8;
    }
    let endpoint = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;
    let per_token = 48 * dense_gdn_layer + 16 * dense_attention_layer + endpoint
        - 56 * (dense_mlp - nvfp4_mlp)
        - 64 * 6 * hidden;
    let ordinary_gqa = 16 * short_gqa_bytes(std::iter::repeat_n(131, tokens));
    let target_gqa =
        16 * short_gqa_bytes((FIRST_POSITION + 1..=FIRST_POSITION + tokens).collect::<Vec<_>>());

    let qkv = Qwen38_27B::GDN_QKV_ROWS;
    let history = qkv * 3 * size_of::<u16>();
    let state = Qwen38_27B::GDN_CONTROL_ROWS
        * Qwen38_27B::LINEAR_HEAD_DIM
        * Qwen38_27B::LINEAR_HEAD_DIM
        * size_of::<f32>();
    let snapshot = size_of::<u32>() + 2 * (history + state);
    let ordinary_convolution = tokens * 8 * qkv * size_of::<u16>();
    let causal_convolution =
        tokens * 5 * qkv * size_of::<u16>() + 6 * qkv * size_of::<u16>() + size_of::<u32>();
    resident_weights + tokens * per_token - ordinary_gqa
        + target_gqa
        + 48 * (snapshot + causal_convolution - ordinary_convolution)
}

fn commit_logical_bytes(tokens: usize) -> usize {
    let qkv = Qwen38_27B::GDN_QKV_ROWS;
    let values = Qwen38_27B::GDN_VALUE_ROWS;
    let controls = Qwen38_27B::GDN_CONTROL_ROWS;
    let state = controls * Qwen38_27B::LINEAR_HEAD_DIM * Qwen38_27B::LINEAR_HEAD_DIM;
    let convolution_weights = qkv * 4 * size_of::<u16>();
    let convolution =
        tokens * 5 * qkv * size_of::<u16>() + 6 * qkv * size_of::<u16>() + size_of::<u32>();
    let recurrence_per_token = 2 * qkv
        + 2 * values
        + 2 * controls * size_of::<f32>()
        + size_of::<u32>()
        + 2 * state * size_of::<f32>()
        + values * size_of::<u16>();
    48 * (convolution_weights
        + convolution
        + Qwen38_27B::LINEAR_HEAD_DIM * size_of::<u16>()
        + tokens * recurrence_per_token)
}

/// Measures every exact K=1..4 verify and full-prefix commit graph directly.
pub fn benchmark_target_mtp_verify(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let stage_graphs = session.stage_graphs()?;
    for (name, kind, bytes, description) in [
        (
            "target_mtp_verify/resident_weights",
            BenchmarkMemoryKind::Weights,
            session.program.resident_weight_bytes(),
            "64 exact source-routed layers plus final norm and LM head",
        ),
        (
            "target_mtp_verify/gdn_history",
            BenchmarkMemoryKind::Other,
            session.program.history_bytes(),
            "48 layers * 8 persistent causal-history slots",
        ),
        (
            "target_mtp_verify/gdn_state",
            BenchmarkMemoryKind::Other,
            session.program.state_bytes(),
            "48 layers * 8 persistent FP32 recurrent-state slots",
        ),
        (
            "target_mtp_verify/represented_kv_cache",
            BenchmarkMemoryKind::KvCache,
            session.program.cache_bytes(),
            "16 layers sharing the exact 220,000-token physical page pool",
        ),
        (
            "target_mtp_verify/kv_block_tables",
            BenchmarkMemoryKind::Other,
            session.program.kv_table_bytes(),
            "8 stable slot rows * 3,438 u32 page-table entries",
        ),
        (
            "target_mtp_verify/shared_workspace",
            BenchmarkMemoryKind::Workspace,
            session.program.workspace_bytes(),
            "one address-stable workspace including provisional and recorded GDN planes",
        ),
        (
            "target_mtp_verify/address_bound_tensor_maps",
            BenchmarkMemoryKind::Other,
            session.program.descriptor_bytes(),
            "eight dense layers * four address-bound 128-byte tensor maps",
        ),
        (
            "target_mtp_verify/alignment_padding",
            BenchmarkMemoryKind::Other,
            session.program.padding_bytes(),
            "256-byte alignment across the resident and shared-KV arenas",
        ),
    ] {
        memory.register_owned(name, kind, bytes, description)?;
    }
    memory.capture("after_setup")?;
    session.warm(&stage_graphs, warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(&stage_graphs)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-target-mtp-verify",
            classification: "performance_sensitive_model",
            timing_scope: "paired Rust production-graph submission/completion for each complete target verify or verify-plus-commit route; matching metadata restoration is outside timing",
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
    use super::{VERIFY_ROUTES, commit_logical_bytes, verify_logical_bytes};

    #[test]
    fn target_mtp_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(VERIFY_ROUTES, 4);
        for tokens in 1..=VERIFY_ROUTES {
            assert!(verify_logical_bytes(tokens) > 19_000_000_000);
            assert!(commit_logical_bytes(tokens) > 300_000_000);
            if tokens > 1 {
                assert!(verify_logical_bytes(tokens) > verify_logical_bytes(tokens - 1));
                assert!(commit_logical_bytes(tokens) > commit_logical_bytes(tokens - 1));
            }
        }
    }
}
