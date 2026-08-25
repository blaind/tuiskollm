//! Remote execution of prebuilt device qualification and benchmark artifacts.

use std::error::Error;
#[cfg(feature = "remote")]
use std::ffi::OsString;
#[cfg(feature = "remote")]
use std::path::Path;

#[cfg(any(feature = "remote", test))]
use crate::PerformanceSuite;
#[cfg(any(feature = "remote", test))]
use crate::gpu_target::GpuTarget;
#[cfg(feature = "remote")]
use crate::gpu_target::{BuildTargetProfile, has_full_kernel_inventory};

/// Static resource gate a route proves locally before any billable pod exists.
/// The row names it, so no substring of a subcommand name can select a sibling
/// suite's baseline.
#[cfg(any(feature = "remote", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceGate {
    ResidualNorm,
    Nvfp4Swiglu,
    Nvfp4Down,
    /// Only SM89 has its own FP8 QKV baseline; every other target proves
    /// residual norm instead.
    Fp8Qkv,
    ResidentMtp,
    MtpPromptPrime,
    ResidentModel,
    MtpLayer,
    MtpBf16Fusion,
    MtpBf16Qkv,
    MtpBf16QkPrepare,
    MtpBf16PagedGqa,
    MtpBf16AttentionOutput,
    MtpBf16Mlp,
}

#[cfg(feature = "remote")]
impl ResourceGate {
    fn verify(self, root: &Path, gpu: GpuTarget) -> Result<(), Box<dyn Error>> {
        match self {
            Self::ResidualNorm => crate::gate_residual_norm_target(root, gpu),
            Self::Nvfp4Swiglu => crate::gate_nvfp4_swiglu_target(root, gpu),
            Self::Nvfp4Down => crate::gate_nvfp4_down_target(root, gpu),
            Self::Fp8Qkv if gpu == GpuTarget::Sm89 => crate::gate_fp8_qkv_sm89(root),
            Self::Fp8Qkv => crate::gate_residual_norm_target(root, gpu),
            Self::ResidentMtp => crate::gate_resident_mtp(root),
            Self::MtpPromptPrime => crate::gate_mtp_prompt_prime(root),
            Self::ResidentModel => crate::gate_resident_model_resources(root),
            Self::MtpLayer => crate::gate_mtp_layer(root),
            Self::MtpBf16Fusion => crate::gate_mtp_bf16_fusion(root),
            Self::MtpBf16Qkv => crate::gate_mtp_bf16_qkv(root),
            Self::MtpBf16QkPrepare => crate::gate_mtp_bf16_qk_prepare(root),
            Self::MtpBf16PagedGqa => crate::gate_mtp_bf16_paged_gqa(root),
            Self::MtpBf16AttentionOutput => crate::gate_mtp_bf16_attention_output(root),
            Self::MtpBf16Mlp => crate::gate_mtp_bf16_mlp(root),
        }
    }
}

/// Which resources a benchmark's prepared artifact binds: one leaf suite's
/// baseline, or every leaf resource a composed route launches.
#[cfg(any(feature = "remote", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkRoute {
    Leaf(PerformanceSuite),
    Nvfp4Mlp,
    FullAttentionLayer,
    MtpLayer,
    MtpPromptPrime,
    ResidentMtp,
    ResidentModel,
}

#[cfg(feature = "remote")]
impl BenchmarkRoute {
    fn prepare(
        self,
        root: &Path,
        gpu: GpuTarget,
    ) -> Result<crate::RemoteBenchmark, Box<dyn Error>> {
        match self {
            Self::Leaf(suite) => crate::prepare_remote_benchmark(root, gpu, suite),
            Self::Nvfp4Mlp => crate::prepare_remote_nvfp4_mlp_benchmark(root, gpu),
            Self::FullAttentionLayer => {
                crate::prepare_remote_full_attention_layer_benchmark(root, gpu)
            }
            Self::MtpLayer => crate::prepare_remote_mtp_layer_benchmark(root, gpu),
            Self::MtpPromptPrime => crate::prepare_remote_mtp_prompt_prime_benchmark(root, gpu),
            Self::ResidentMtp => crate::prepare_remote_resident_mtp_benchmark(root, gpu),
            Self::ResidentModel => crate::prepare_remote_resident_model_benchmark(root, gpu),
        }
    }
}

/// What one `remote <name>` route runs on the pod.
#[cfg(any(feature = "remote", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RemoteRun {
    /// A qualification suite and the libtest filter the pod's `qual` runs.
    Qualification(&'static str),
    Benchmark(BenchmarkRoute),
}

/// One `remote <name>` route: every decision the runner makes before it may
/// create a billable pod — which targets admit the suite, which static resource
/// gate proves the local artifact, and whether the pinned source snapshot has
/// to be staged.
#[cfg(any(feature = "remote", test))]
#[derive(Clone, Copy)]
struct RemoteSubcommand {
    name: &'static str,
    run: RemoteRun,
    source_snapshot: bool,
    gate: ResourceGate,
    targets: &'static [GpuTarget],
}

/// Residual norm and the represented-weight NVFP4 gate/up route.
#[cfg(any(feature = "remote", test))]
const EVERY_TARGET: &[GpuTarget] = &GpuTarget::ALL;
/// The complete inventory is Blackwell-only, and is a route's default.
#[cfg(any(feature = "remote", test))]
const SM120_ONLY: &[GpuTarget] = &[GpuTarget::Sm120];
/// Routes the partial Ada inventory also implements.
#[cfg(any(feature = "remote", test))]
const THROUGH_SM89: &[GpuTarget] = &[GpuTarget::Sm120, GpuTarget::Sm89];

#[cfg(any(feature = "remote", test))]
const fn route(name: &'static str, run: RemoteRun) -> RemoteSubcommand {
    RemoteSubcommand {
        name,
        run,
        source_snapshot: false,
        gate: ResourceGate::ResidualNorm,
        targets: SM120_ONLY,
    }
}

#[cfg(any(feature = "remote", test))]
const fn qualify(name: &'static str, filter: &'static str) -> RemoteSubcommand {
    route(name, RemoteRun::Qualification(filter))
}

#[cfg(any(feature = "remote", test))]
const fn leaf(name: &'static str, suite: PerformanceSuite) -> RemoteSubcommand {
    route(name, RemoteRun::Benchmark(BenchmarkRoute::Leaf(suite)))
}

#[cfg(any(feature = "remote", test))]
const fn bench(name: &'static str, benchmark: BenchmarkRoute) -> RemoteSubcommand {
    route(name, RemoteRun::Benchmark(benchmark))
}

#[cfg(any(feature = "remote", test))]
impl RemoteSubcommand {
    /// The pod stages the pinned source snapshot for this route.
    const fn snapshot(self) -> Self {
        Self {
            source_snapshot: true,
            ..self
        }
    }

    /// This route proves a gate other than residual norm.
    const fn gated(self, gate: ResourceGate) -> Self {
        Self { gate, ..self }
    }

    /// Partial-inventory targets that also admit this route.
    const fn on(self, targets: &'static [GpuTarget]) -> Self {
        Self { targets, ..self }
    }

    /// Suite name in the remote report: the subcommand without its verb.
    fn suite(self) -> &'static str {
        self.name
            .strip_prefix("qualify-")
            .or_else(|| self.name.strip_prefix("bench-"))
            .unwrap_or(self.name)
    }

    fn is_benchmark(self) -> bool {
        matches!(self.run, RemoteRun::Benchmark(_))
    }

    fn admits(self, gpu: GpuTarget) -> bool {
        self.targets.contains(&gpu)
    }
}

/// The complete `remote` route set, in usage order. Names, filters, gates,
/// snapshot staging and target admission live here, so the usage line, the
/// command/target decision table and the spawner cannot drift apart.
#[cfg(any(feature = "remote", test))]
const REMOTE_SUBCOMMANDS: &[RemoteSubcommand] = &[
    qualify("qualify-residual-norm", "residual_norm_suite_").on(EVERY_TARGET),
    qualify("qualify-nvfp4-swiglu", "nvfp4_swiglu::tests")
        .gated(ResourceGate::Nvfp4Swiglu)
        .on(EVERY_TARGET),
    qualify("qualify-nvfp4-down", "nvfp4_down::tests")
        .gated(ResourceGate::Nvfp4Down)
        .on(THROUGH_SM89),
    qualify("qualify-fp8-qkv", "fp8_qkv")
        .gated(ResourceGate::Fp8Qkv)
        .on(THROUGH_SM89),
    qualify("qualify-fp8-gdn-input", "fp8_gdn_input"),
    qualify("qualify-fp8-lm-head", "fp8_lm_head"),
    qualify(
        "qualify-nvfp4-mlp",
        "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay",
    )
    .snapshot(),
    qualify("qualify-attention-qk-prepare", "attention_qk_prepare"),
    qualify("qualify-paged-gqa", "paged_gqa_suite_"),
    qualify("qualify-long-context-paged-gqa", "long_context_paged_gqa"),
    qualify("qualify-attention-output", "attention_output::tests"),
    qualify("qualify-mtp-bf16-fusion", "mtp_bf16_fusion_suite_")
        .gated(ResourceGate::MtpBf16Fusion)
        .snapshot(),
    qualify("qualify-mtp-bf16-qkv", "mtp_bf16_qkv_suite_")
        .gated(ResourceGate::MtpBf16Qkv)
        .snapshot(),
    qualify("qualify-mtp-bf16-qk-prepare", "mtp_bf16_qk_prepare_suite_")
        .gated(ResourceGate::MtpBf16QkPrepare)
        .snapshot(),
    qualify("qualify-mtp-bf16-paged-gqa", "mtp_bf16_paged_gqa_suite_")
        .gated(ResourceGate::MtpBf16PagedGqa),
    qualify(
        "qualify-mtp-bf16-attention-output",
        "mtp_bf16_attention_output_suite_",
    )
    .gated(ResourceGate::MtpBf16AttentionOutput)
    .snapshot(),
    qualify("qualify-mtp-bf16-mlp", "mtp_bf16_mlp_suite_")
        .gated(ResourceGate::MtpBf16Mlp)
        .snapshot(),
    qualify(
        "qualify-full-attention-layer",
        "full_attention_layer::tests::source_layer63_matches_complete_seam_oracles_and_graph_replay",
    )
    .snapshot(),
    qualify("qualify-mtp-layer", "mtp_layer_suite_")
        .gated(ResourceGate::MtpLayer)
        .snapshot(),
    qualify(
        "qualify-target-mtp-verify",
        "target_mtp_verify::tests::exact_target_verify_and_commit_match_source_oracles",
    )
    .gated(ResourceGate::ResidentModel)
    .snapshot(),
    qualify("qualify-mtp-prompt-prime", "mtp_prompt_prime_suite_")
        .gated(ResourceGate::MtpPromptPrime)
        .snapshot(),
    qualify("qualify-resident-mtp", "resident_mtp_suite_")
        .gated(ResourceGate::ResidentMtp)
        .snapshot(),
    qualify(
        "qualify-resident-model",
        "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay",
    )
    .snapshot(),
    qualify(
        "qualify-resident-generation",
        "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming",
    )
    .snapshot(),
    qualify(
        "qualify-generation-mtp-greedy",
        "resident_mtp_generation_suite_",
    )
    .snapshot(),
    qualify(
        "qualify-generation-mtp-sampling",
        "resident_mtp_sampling_suite_",
    )
    .snapshot(),
    qualify("qualify-generation-mtp-batch", "resident_mtp_batch_suite_").snapshot(),
    qualify(
        "qualify-resident-batch-generation",
        "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes",
    )
    .snapshot(),
    leaf("bench-residual-norm", PerformanceSuite::ResidualNorm).on(EVERY_TARGET),
    leaf("bench-nvfp4-swiglu", PerformanceSuite::Nvfp4SwiGlu)
        .gated(ResourceGate::Nvfp4Swiglu)
        .on(EVERY_TARGET),
    leaf("bench-nvfp4-down", PerformanceSuite::Nvfp4Down)
        .gated(ResourceGate::Nvfp4Down)
        .on(THROUGH_SM89),
    bench("bench-nvfp4-mlp", BenchmarkRoute::Nvfp4Mlp).snapshot(),
    leaf("bench-fp8-qkv", PerformanceSuite::Fp8Qkv)
        .gated(ResourceGate::Fp8Qkv)
        .on(THROUGH_SM89),
    leaf("bench-fp8-gdn-input", PerformanceSuite::Fp8GdnInput),
    leaf("bench-fp8-lm-head", PerformanceSuite::Fp8LmHead),
    leaf(
        "bench-attention-qk-prepare",
        PerformanceSuite::AttentionQkPrepare,
    ),
    leaf("bench-paged-gqa", PerformanceSuite::PagedGqa),
    leaf(
        "bench-long-context-paged-gqa",
        PerformanceSuite::LongContextPagedGqa,
    ),
    leaf("bench-attention-output", PerformanceSuite::AttentionOutput),
    leaf("bench-mtp-bf16-fusion", PerformanceSuite::MtpBf16Fusion)
        .gated(ResourceGate::MtpBf16Fusion),
    leaf("bench-mtp-bf16-qkv", PerformanceSuite::MtpBf16Qkv).gated(ResourceGate::MtpBf16Qkv),
    leaf(
        "bench-mtp-bf16-qk-prepare",
        PerformanceSuite::MtpBf16QkPrepare,
    )
    .gated(ResourceGate::MtpBf16QkPrepare),
    leaf("bench-mtp-bf16-paged-gqa", PerformanceSuite::MtpBf16PagedGqa)
        .gated(ResourceGate::MtpBf16PagedGqa),
    leaf(
        "bench-mtp-bf16-attention-output",
        PerformanceSuite::MtpBf16AttentionOutput,
    )
    .gated(ResourceGate::MtpBf16AttentionOutput),
    leaf("bench-mtp-bf16-mlp", PerformanceSuite::MtpBf16Mlp).gated(ResourceGate::MtpBf16Mlp),
    bench(
        "bench-full-attention-layer",
        BenchmarkRoute::FullAttentionLayer,
    )
    .snapshot(),
    bench("bench-mtp-layer", BenchmarkRoute::MtpLayer)
        .gated(ResourceGate::MtpLayer)
        .snapshot(),
    bench("bench-target-mtp-verify", BenchmarkRoute::ResidentModel)
        .gated(ResourceGate::ResidentModel)
        .snapshot(),
    bench("bench-mtp-prompt-prime", BenchmarkRoute::MtpPromptPrime)
        .gated(ResourceGate::MtpPromptPrime)
        .snapshot(),
    bench("bench-resident-mtp", BenchmarkRoute::ResidentMtp)
        .gated(ResourceGate::ResidentMtp)
        .snapshot(),
    bench("bench-generation-mtp-greedy", BenchmarkRoute::ResidentMtp).snapshot(),
    bench("bench-generation-mtp-sampling", BenchmarkRoute::ResidentMtp).snapshot(),
    bench("bench-generation-mtp-batch", BenchmarkRoute::ResidentMtp).snapshot(),
    bench("bench-resident-model", BenchmarkRoute::ResidentModel).snapshot(),
    bench("bench-resident-prefill", BenchmarkRoute::ResidentModel).snapshot(),
    bench(
        "bench-resident-long-context-model",
        BenchmarkRoute::ResidentModel,
    )
    .snapshot(),
];

/// The usage line, derived from the route table so no row can go unadvertised.
/// `usage_lists_every_route` binds it to the legacy text.
#[cfg(any(feature = "remote", test))]
fn usage() -> String {
    let mut names = REMOTE_SUBCOMMANDS
        .iter()
        .map(|route| route.name)
        .collect::<Vec<_>>();
    names.extend(["probe", "check", "sweep"]);

    format!(
        "usage: cargo run -p xtask --features remote -- remote <{}> \
        [--gpu 5090|4090|3090] [--max-minutes N] [--image NAME] [--keep-on-fail] \
        [--samples N] [--launches-per-sample N] [--energy-seconds N]",
        names.join("|")
    )
}

#[cfg(any(feature = "remote", test))]
struct RemoteOptions {
    gpu: GpuTarget,
    max_minutes: u32,
    image: String,
    keep_on_fail: bool,
    benchmark_args: Vec<String>,
}

/// Entry point for `xtask remote`.
#[cfg(feature = "remote")]
pub fn run(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let arguments = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    run_impl(root, &arguments)
}

#[cfg(feature = "remote")]
fn run_impl(root: &Path, arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let Some(command) = arguments.first().map(String::as_str) else {
        return Err(usage().into());
    };
    match command {
        "check" if arguments.len() == 1 => {
            tuisko_remote::check().map_err(|error| format!("{error}"))?;
            return Ok(());
        }
        "sweep" if arguments.len() == 1 => {
            tuisko_remote::sweep_stale().map_err(|error| format!("{error}"))?;
            return Ok(());
        }
        "sentry" => return run_sentry(arguments),
        _ => {}
    }

    if command == "probe" {
        let options = parse_options(&arguments[1..], false)?;
        tuisko_remote::check_credentials().map_err(|error| format!("{error}"))?;
        tuisko_remote::run_probe(&tuisko_remote::ProbeOptions {
            gpu: options.gpu.remote_gpu(),
            image: options.image,
            max_minutes: options.max_minutes,
            keep_on_fail: options.keep_on_fail,
        })
        .map_err(|error| format!("{error}"))?;
        return Ok(());
    }

    let Some(route) = REMOTE_SUBCOMMANDS.iter().find(|row| row.name == command) else {
        return Err(usage().into());
    };
    let options = parse_options(&arguments[1..], route.is_benchmark())?;

    if !route.admits(options.gpu) {
        return Err(format!(
            "GPU {} does not admit `{command}`; the remaining {} inventory has not been implemented",
            options.gpu.key(),
            options.gpu.kernel_crate(),
        )
        .into());
    }

    tuisko_remote::check_credentials().map_err(|error| format!("{error}"))?;
    if has_full_kernel_inventory(options.gpu) {
        crate::build_sm120(root)?;
    } else if route.is_benchmark() {
        crate::build_residual_benchmark_target(root, options.gpu)?;
    }

    match route.run {
        // A qualification artifact is compiled here, so its gate runs after
        // that build; a benchmark's artifact is prebuilt and gates first.
        RemoteRun::Qualification(filter) => {
            let prepared = crate::prepare_remote_qualify(root, options.gpu, filter)?;
            route.gate.verify(root, options.gpu)?;
            tuisko_remote::run_qualification(
                root,
                &tuisko_remote::QualificationOptions {
                    suite: route.suite().to_owned(),
                    executable: prepared.executable,
                    test_args: prepared.test_args,
                    gpu: options.gpu.remote_gpu(),
                    source_snapshot: route.source_snapshot,
                    image: options.image,
                    max_minutes: options.max_minutes,
                    keep_on_fail: options.keep_on_fail,
                },
            )
            .map_err(|error| format!("{error}"))?;
        }
        RemoteRun::Benchmark(benchmark) => {
            route.gate.verify(root, options.gpu)?;
            let prepared = benchmark.prepare(root, options.gpu)?;
            tuisko_remote::run_benchmark(
                root,
                &tuisko_remote::BenchmarkOptions {
                    suite: route.suite().to_owned(),
                    executable: prepared.executable,
                    benchmark_args: options.benchmark_args,
                    generator_baseline_sha256: prepared.generator_baseline_sha256,
                    gpu: options.gpu.remote_gpu(),
                    source_snapshot: route.source_snapshot,
                    image: options.image,
                    max_minutes: options.max_minutes,
                    keep_on_fail: options.keep_on_fail,
                },
            )
            .map_err(|error| format!("{error}"))?;
        }
    }

    Ok(())
}

#[cfg(any(feature = "remote", test))]
fn parse_options(arguments: &[String], benchmark: bool) -> Result<RemoteOptions, Box<dyn Error>> {
    let mut options = RemoteOptions {
        gpu: GpuTarget::Sm120,
        max_minutes: 30,
        image: tuisko_remote_default_image(),
        keep_on_fail: false,
        benchmark_args: Vec::new(),
    };
    let mut cursor = 0usize;
    while cursor < arguments.len() {
        let argument = arguments[cursor].as_str();
        match argument {
            "--gpu" => {
                cursor += 1;
                options.gpu =
                    GpuTarget::parse(arguments.get(cursor).ok_or("--gpu requires a value")?)?;
            }
            "--max-minutes" => {
                cursor += 1;
                options.max_minutes = arguments
                    .get(cursor)
                    .ok_or("--max-minutes requires a value")?
                    .parse()
                    .map_err(|_| "--max-minutes must be an integer")?;
            }
            "--image" => {
                cursor += 1;
                options.image = arguments
                    .get(cursor)
                    .ok_or("--image requires a value")?
                    .clone();
            }
            "--keep-on-fail" => options.keep_on_fail = true,
            "--samples" | "--launches-per-sample" | "--energy-seconds" if benchmark => {
                cursor += 1;
                let value = arguments
                    .get(cursor)
                    .ok_or_else(|| format!("{argument} requires a value"))?;
                validate_benchmark_control(argument, value)?;
                options.benchmark_args.push(argument.to_owned());
                options.benchmark_args.push(value.clone());
            }
            other => return Err(format!("unknown remote option `{other}`").into()),
        }
        cursor += 1;
    }

    Ok(options)
}

#[cfg(any(feature = "remote", test))]
fn validate_benchmark_control(name: &str, value: &str) -> Result<(), Box<dyn Error>> {
    match name {
        "--samples" if value.parse::<usize>().is_ok_and(|samples| samples >= 3) => Ok(()),
        "--launches-per-sample" if value.parse::<u64>().is_ok_and(|launches| launches != 0) => {
            Ok(())
        }
        "--energy-seconds"
            if value
                .parse::<f64>()
                .is_ok_and(|seconds| seconds.is_finite() && seconds >= 2.0) =>
        {
            Ok(())
        }
        _ => Err(format!("invalid value `{value}` for {name}").into()),
    }
}

#[cfg(feature = "remote")]
fn tuisko_remote_default_image() -> String {
    tuisko_remote::DEFAULT_IMAGE.to_owned()
}

#[cfg(all(test, not(feature = "remote")))]
fn tuisko_remote_default_image() -> String {
    "default-image".to_owned()
}

#[cfg(feature = "remote")]
fn run_sentry(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let [_, pod_id, parent_flag, parent, deadline_flag, deadline] = arguments else {
        return Err("invalid remote-runner watchdog invocation".into());
    };
    if parent_flag != "--parent" || deadline_flag != "--deadline-secs" {
        return Err("invalid remote-runner watchdog arguments".into());
    }

    tuisko_remote::run_sentry(pod_id, parent.parse()?, deadline.parse()?)
        .map_err(|error| format!("{error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BenchmarkRoute, REMOTE_SUBCOMMANDS, RemoteRun, ResourceGate, parse_options, usage,
    };
    use crate::PerformanceSuite;
    use crate::gpu_target::GpuTarget;
    use std::collections::BTreeSet;

    /// The route table must name exactly the suites the legacy
    /// `Qualification::parse` and `Benchmark::parse` ladders named, and give
    /// each the same report name, prepared artifact and snapshot staging.
    /// Transcribed from those ladders; this test, not a hand-counted number,
    /// is the authority on the route set.
    #[test]
    fn remote_suite_inventory_is_exact() {
        // (subcommand, report suite, libtest filter, stages the snapshot)
        const LEGACY_QUALIFICATIONS: &[(&str, &str, &str, bool)] = &[
            (
                "qualify-residual-norm",
                "residual-norm",
                "residual_norm_suite_",
                false,
            ),
            (
                "qualify-nvfp4-swiglu",
                "nvfp4-swiglu",
                "nvfp4_swiglu::tests",
                false,
            ),
            (
                "qualify-nvfp4-down",
                "nvfp4-down",
                "nvfp4_down::tests",
                false,
            ),
            ("qualify-fp8-qkv", "fp8-qkv", "fp8_qkv", false),
            (
                "qualify-fp8-gdn-input",
                "fp8-gdn-input",
                "fp8_gdn_input",
                false,
            ),
            ("qualify-fp8-lm-head", "fp8-lm-head", "fp8_lm_head", false),
            (
                "qualify-nvfp4-mlp",
                "nvfp4-mlp",
                "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay",
                true,
            ),
            (
                "qualify-attention-qk-prepare",
                "attention-qk-prepare",
                "attention_qk_prepare",
                false,
            ),
            ("qualify-paged-gqa", "paged-gqa", "paged_gqa_suite_", false),
            (
                "qualify-long-context-paged-gqa",
                "long-context-paged-gqa",
                "long_context_paged_gqa",
                false,
            ),
            (
                "qualify-attention-output",
                "attention-output",
                "attention_output::tests",
                false,
            ),
            (
                "qualify-mtp-bf16-fusion",
                "mtp-bf16-fusion",
                "mtp_bf16_fusion_suite_",
                true,
            ),
            (
                "qualify-mtp-bf16-qkv",
                "mtp-bf16-qkv",
                "mtp_bf16_qkv_suite_",
                true,
            ),
            (
                "qualify-mtp-bf16-qk-prepare",
                "mtp-bf16-qk-prepare",
                "mtp_bf16_qk_prepare_suite_",
                true,
            ),
            (
                "qualify-mtp-bf16-paged-gqa",
                "mtp-bf16-paged-gqa",
                "mtp_bf16_paged_gqa_suite_",
                false,
            ),
            (
                "qualify-mtp-bf16-attention-output",
                "mtp-bf16-attention-output",
                "mtp_bf16_attention_output_suite_",
                true,
            ),
            (
                "qualify-mtp-bf16-mlp",
                "mtp-bf16-mlp",
                "mtp_bf16_mlp_suite_",
                true,
            ),
            (
                "qualify-full-attention-layer",
                "full-attention-layer",
                "full_attention_layer::tests::source_layer63_matches_complete_seam_oracles_and_graph_replay",
                true,
            ),
            ("qualify-mtp-layer", "mtp-layer", "mtp_layer_suite_", true),
            (
                "qualify-target-mtp-verify",
                "target-mtp-verify",
                "target_mtp_verify::tests::exact_target_verify_and_commit_match_source_oracles",
                true,
            ),
            (
                "qualify-mtp-prompt-prime",
                "mtp-prompt-prime",
                "mtp_prompt_prime_suite_",
                true,
            ),
            (
                "qualify-resident-mtp",
                "resident-mtp",
                "resident_mtp_suite_",
                true,
            ),
            (
                "qualify-resident-model",
                "resident-model",
                "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay",
                true,
            ),
            (
                "qualify-resident-generation",
                "resident-generation",
                "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming",
                true,
            ),
            (
                "qualify-generation-mtp-greedy",
                "generation-mtp-greedy",
                "resident_mtp_generation_suite_",
                true,
            ),
            (
                "qualify-generation-mtp-sampling",
                "generation-mtp-sampling",
                "resident_mtp_sampling_suite_",
                true,
            ),
            (
                "qualify-generation-mtp-batch",
                "generation-mtp-batch",
                "resident_mtp_batch_suite_",
                true,
            ),
            (
                "qualify-resident-batch-generation",
                "resident-batch-generation",
                "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes",
                true,
            ),
        ];

        // (subcommand, report suite, the artifact its `Benchmark` arm prepared,
        // stages the snapshot)
        const LEGACY_BENCHMARKS: &[(&str, &str, BenchmarkRoute, bool)] = &[
            (
                "bench-residual-norm",
                "residual-norm",
                BenchmarkRoute::Leaf(PerformanceSuite::ResidualNorm),
                false,
            ),
            (
                "bench-nvfp4-swiglu",
                "nvfp4-swiglu",
                BenchmarkRoute::Leaf(PerformanceSuite::Nvfp4SwiGlu),
                false,
            ),
            (
                "bench-nvfp4-down",
                "nvfp4-down",
                BenchmarkRoute::Leaf(PerformanceSuite::Nvfp4Down),
                false,
            ),
            (
                "bench-nvfp4-mlp",
                "nvfp4-mlp",
                BenchmarkRoute::Nvfp4Mlp,
                true,
            ),
            (
                "bench-fp8-qkv",
                "fp8-qkv",
                BenchmarkRoute::Leaf(PerformanceSuite::Fp8Qkv),
                false,
            ),
            (
                "bench-fp8-gdn-input",
                "fp8-gdn-input",
                BenchmarkRoute::Leaf(PerformanceSuite::Fp8GdnInput),
                false,
            ),
            (
                "bench-fp8-lm-head",
                "fp8-lm-head",
                BenchmarkRoute::Leaf(PerformanceSuite::Fp8LmHead),
                false,
            ),
            (
                "bench-attention-qk-prepare",
                "attention-qk-prepare",
                BenchmarkRoute::Leaf(PerformanceSuite::AttentionQkPrepare),
                false,
            ),
            (
                "bench-paged-gqa",
                "paged-gqa",
                BenchmarkRoute::Leaf(PerformanceSuite::PagedGqa),
                false,
            ),
            (
                "bench-long-context-paged-gqa",
                "long-context-paged-gqa",
                BenchmarkRoute::Leaf(PerformanceSuite::LongContextPagedGqa),
                false,
            ),
            (
                "bench-attention-output",
                "attention-output",
                BenchmarkRoute::Leaf(PerformanceSuite::AttentionOutput),
                false,
            ),
            (
                "bench-mtp-bf16-fusion",
                "mtp-bf16-fusion",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16Fusion),
                false,
            ),
            (
                "bench-mtp-bf16-qkv",
                "mtp-bf16-qkv",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16Qkv),
                false,
            ),
            (
                "bench-mtp-bf16-qk-prepare",
                "mtp-bf16-qk-prepare",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16QkPrepare),
                false,
            ),
            (
                "bench-mtp-bf16-paged-gqa",
                "mtp-bf16-paged-gqa",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16PagedGqa),
                false,
            ),
            (
                "bench-mtp-bf16-attention-output",
                "mtp-bf16-attention-output",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16AttentionOutput),
                false,
            ),
            (
                "bench-mtp-bf16-mlp",
                "mtp-bf16-mlp",
                BenchmarkRoute::Leaf(PerformanceSuite::MtpBf16Mlp),
                false,
            ),
            (
                "bench-full-attention-layer",
                "full-attention-layer",
                BenchmarkRoute::FullAttentionLayer,
                true,
            ),
            (
                "bench-mtp-layer",
                "mtp-layer",
                BenchmarkRoute::MtpLayer,
                true,
            ),
            (
                "bench-target-mtp-verify",
                "target-mtp-verify",
                BenchmarkRoute::ResidentModel,
                true,
            ),
            (
                "bench-mtp-prompt-prime",
                "mtp-prompt-prime",
                BenchmarkRoute::MtpPromptPrime,
                true,
            ),
            (
                "bench-resident-mtp",
                "resident-mtp",
                BenchmarkRoute::ResidentMtp,
                true,
            ),
            (
                "bench-generation-mtp-greedy",
                "generation-mtp-greedy",
                BenchmarkRoute::ResidentMtp,
                true,
            ),
            (
                "bench-generation-mtp-sampling",
                "generation-mtp-sampling",
                BenchmarkRoute::ResidentMtp,
                true,
            ),
            (
                "bench-generation-mtp-batch",
                "generation-mtp-batch",
                BenchmarkRoute::ResidentMtp,
                true,
            ),
            (
                "bench-resident-model",
                "resident-model",
                BenchmarkRoute::ResidentModel,
                true,
            ),
            (
                "bench-resident-prefill",
                "resident-prefill",
                BenchmarkRoute::ResidentModel,
                true,
            ),
            (
                "bench-resident-long-context-model",
                "resident-long-context-model",
                BenchmarkRoute::ResidentModel,
                true,
            ),
        ];

        assert_eq!(LEGACY_QUALIFICATIONS.len(), 28);
        assert_eq!(LEGACY_BENCHMARKS.len(), 28);
        assert_eq!(
            REMOTE_SUBCOMMANDS.len(),
            LEGACY_QUALIFICATIONS.len() + LEGACY_BENCHMARKS.len()
        );

        let names = REMOTE_SUBCOMMANDS
            .iter()
            .map(|route| route.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names.len(),
            REMOTE_SUBCOMMANDS.len(),
            "the table repeats a name"
        );
        assert_eq!(
            names,
            LEGACY_QUALIFICATIONS
                .iter()
                .map(|(name, ..)| *name)
                .chain(LEGACY_BENCHMARKS.iter().map(|(name, ..)| *name))
                .collect::<BTreeSet<_>>()
        );

        let find = |name: &str| {
            *REMOTE_SUBCOMMANDS
                .iter()
                .find(|route| route.name == name)
                .unwrap()
        };

        for &(name, suite, filter, snapshot) in LEGACY_QUALIFICATIONS {
            let route = find(name);
            assert_eq!(route.suite(), suite, "`{name}` report suite");
            assert_eq!(
                route.run,
                RemoteRun::Qualification(filter),
                "`{name}` route"
            );
            assert_eq!(route.source_snapshot, snapshot, "`{name}` snapshot");
            assert!(!route.is_benchmark(), "`{name}` is a qualification");
        }

        for &(name, suite, benchmark, snapshot) in LEGACY_BENCHMARKS {
            let route = find(name);
            assert_eq!(route.suite(), suite, "`{name}` report suite");
            assert_eq!(route.run, RemoteRun::Benchmark(benchmark), "`{name}` route");
            assert_eq!(route.source_snapshot, snapshot, "`{name}` snapshot");
            // A leaf benchmark reports under its suite's own name, exactly as
            // `Benchmark::name()` returned it.
            if let BenchmarkRoute::Leaf(leaf) = benchmark {
                assert_eq!(leaf.name(), suite, "`{name}` leaf suite name");
            }
        }

        for absent in ["perf", "probe", "check", "sweep"] {
            assert!(!names.contains(absent));
        }
    }

    /// Every route names the static resource gate the legacy substring chain
    /// in `gate_static_resources` selected for it.
    #[test]
    fn resource_gates_reproduce_the_legacy_chain() {
        // Transcribed from `gate_static_resources`, in its `if`/`else if`
        // order. `Fp8Qkv` stands for the arm that fires only on SM89 and
        // otherwise falls through to the residual-norm gate.
        fn legacy_gate(command: &str) -> ResourceGate {
            if command.contains("nvfp4-swiglu") {
                ResourceGate::Nvfp4Swiglu
            } else if command.contains("nvfp4-down") {
                ResourceGate::Nvfp4Down
            } else if command.contains("fp8-qkv") {
                ResourceGate::Fp8Qkv
            } else if command.contains("resident-mtp") {
                ResourceGate::ResidentMtp
            } else if command.contains("mtp-prompt-prime") {
                ResourceGate::MtpPromptPrime
            } else if command.contains("target-mtp-verify") {
                ResourceGate::ResidentModel
            } else if command.contains("mtp-layer") {
                ResourceGate::MtpLayer
            } else if command.contains("mtp-bf16-fusion") {
                ResourceGate::MtpBf16Fusion
            } else if command.contains("mtp-bf16-qkv") {
                ResourceGate::MtpBf16Qkv
            } else if command.contains("mtp-bf16-qk-prepare") {
                ResourceGate::MtpBf16QkPrepare
            } else if command.contains("mtp-bf16-paged-gqa") {
                ResourceGate::MtpBf16PagedGqa
            } else if command.contains("mtp-bf16-attention-output") {
                ResourceGate::MtpBf16AttentionOutput
            } else if command.contains("mtp-bf16-mlp") {
                ResourceGate::MtpBf16Mlp
            } else {
                ResourceGate::ResidualNorm
            }
        }

        for route in REMOTE_SUBCOMMANDS {
            assert_eq!(route.gate, legacy_gate(route.name), "`{}` gate", route.name);
        }
    }

    /// The advertised usage is the route table, so no row can exist
    /// unadvertised and no advertised name can be missing a row.
    #[test]
    fn usage_lists_every_route() {
        // The pre-table `USAGE` literal, transcribed verbatim.
        const LEGACY_USAGE: &str = "usage: cargo run -p xtask --features remote -- remote \
            <qualify-residual-norm|qualify-nvfp4-swiglu|qualify-nvfp4-down|qualify-fp8-qkv|qualify-fp8-gdn-input|qualify-fp8-lm-head|\
            qualify-nvfp4-mlp|qualify-attention-qk-prepare|qualify-paged-gqa|qualify-long-context-paged-gqa|qualify-attention-output|qualify-mtp-bf16-fusion|qualify-mtp-bf16-qkv|qualify-mtp-bf16-qk-prepare|qualify-mtp-bf16-paged-gqa|qualify-mtp-bf16-attention-output|qualify-mtp-bf16-mlp|qualify-full-attention-layer|qualify-mtp-layer|qualify-target-mtp-verify|qualify-mtp-prompt-prime|qualify-resident-mtp|\
            qualify-resident-model|qualify-resident-generation|qualify-generation-mtp-greedy|qualify-generation-mtp-sampling|qualify-generation-mtp-batch|qualify-resident-batch-generation|\
            bench-residual-norm|bench-nvfp4-swiglu|bench-nvfp4-down|bench-nvfp4-mlp|bench-fp8-qkv|bench-fp8-gdn-input|\
            bench-fp8-lm-head|bench-attention-qk-prepare|bench-paged-gqa|bench-long-context-paged-gqa|bench-attention-output|bench-mtp-bf16-fusion|bench-mtp-bf16-qkv|bench-mtp-bf16-qk-prepare|bench-mtp-bf16-paged-gqa|bench-mtp-bf16-attention-output|bench-mtp-bf16-mlp|bench-full-attention-layer|bench-mtp-layer|bench-target-mtp-verify|bench-mtp-prompt-prime|bench-resident-mtp|bench-generation-mtp-greedy|bench-generation-mtp-sampling|bench-generation-mtp-batch|\
            bench-resident-model|bench-resident-prefill|bench-resident-long-context-model|\
            probe|check|sweep> \
            [--gpu 5090|4090|3090] [--max-minutes N] [--image NAME] [--keep-on-fail] \
            [--samples N] [--launches-per-sample N] [--energy-seconds N]";

        assert_eq!(usage(), LEGACY_USAGE);
    }

    #[test]
    fn benchmark_controls_are_validated_before_renting_a_pod() {
        assert!(parse_options(&["--samples".to_owned(), "3".to_owned()], true).is_ok());
        assert!(parse_options(&["--samples".to_owned(), "2".to_owned()], true).is_err());
        assert!(parse_options(&["--samples".to_owned(), "3".to_owned()], false).is_err());
    }

    #[test]
    fn gpu_target_is_explicit_and_defaults_to_the_product_target() {
        assert_eq!(parse_options(&[], false).unwrap().gpu, GpuTarget::Sm120);
        assert_eq!(
            parse_options(&["--gpu".to_owned(), "4090".to_owned()], false)
                .unwrap()
                .gpu,
            GpuTarget::Sm89
        );
        assert!(parse_options(&["--gpu".to_owned(), "A100".to_owned()], false).is_err());
    }

    #[test]
    fn target_support_is_a_complete_decision_table() {
        let admits = |command: &str, gpu: GpuTarget| {
            REMOTE_SUBCOMMANDS
                .iter()
                .find(|route| route.name == command)
                .expect("known route")
                .admits(gpu)
        };

        for command in [
            "qualify-residual-norm",
            "bench-residual-norm",
            "qualify-fp8-qkv",
            "bench-fp8-qkv",
        ] {
            assert!(admits(command, GpuTarget::Sm120));
        }
        for gpu in [GpuTarget::Sm89, GpuTarget::Sm86] {
            assert!(admits("qualify-residual-norm", gpu));
            assert!(admits("bench-residual-norm", gpu));
        }
        assert!(admits("qualify-nvfp4-swiglu", GpuTarget::Sm89));
        assert!(admits("bench-nvfp4-swiglu", GpuTarget::Sm89));
        assert!(admits("qualify-nvfp4-down", GpuTarget::Sm89));
        assert!(admits("bench-nvfp4-down", GpuTarget::Sm89));
        assert!(admits("qualify-fp8-qkv", GpuTarget::Sm89));
        assert!(admits("bench-fp8-qkv", GpuTarget::Sm89));
        assert!(admits("qualify-nvfp4-swiglu", GpuTarget::Sm86));
        assert!(admits("bench-nvfp4-swiglu", GpuTarget::Sm86));
        assert!(!admits("qualify-nvfp4-down", GpuTarget::Sm86));
        assert!(!admits("bench-nvfp4-down", GpuTarget::Sm86));
        assert!(!admits("qualify-fp8-qkv", GpuTarget::Sm86));
        assert!(!admits("bench-fp8-qkv", GpuTarget::Sm86));

        // SM120 carries the complete inventory and admits every route; the
        // partial targets admit exactly the four suites they implement.
        const PARTIAL: &[&str] = &[
            "qualify-residual-norm",
            "bench-residual-norm",
            "qualify-nvfp4-swiglu",
            "bench-nvfp4-swiglu",
            "qualify-nvfp4-down",
            "bench-nvfp4-down",
            "qualify-fp8-qkv",
            "bench-fp8-qkv",
        ];
        for route in REMOTE_SUBCOMMANDS {
            assert!(route.admits(GpuTarget::Sm120), "`{}` on SM120", route.name);
            if !PARTIAL.contains(&route.name) {
                assert!(!route.admits(GpuTarget::Sm89), "`{}` on SM89", route.name);
                assert!(!route.admits(GpuTarget::Sm86), "`{}` on SM86", route.name);
            }
        }
    }
}
