//! Remote execution of prebuilt device qualification and benchmark artifacts.

use std::error::Error;
use std::ffi::OsString;
use std::path::Path;

#[cfg(feature = "remote")]
use crate::gpu_target::BuildTargetProfile;
#[cfg(any(feature = "remote", test))]
use crate::gpu_target::{GpuTarget, has_full_kernel_inventory};

#[cfg(feature = "remote")]
const USAGE: &str = "usage: cargo run -p xtask --features remote -- remote \
    <qualify-residual-norm|qualify-nvfp4-swiglu|qualify-nvfp4-down|qualify-fp8-qkv|qualify-fp8-gdn-input|qualify-fp8-lm-head|\
    qualify-nvfp4-mlp|qualify-attention-qk-prepare|qualify-paged-gqa|qualify-long-context-paged-gqa|qualify-attention-output|qualify-mtp-bf16-fusion|qualify-mtp-bf16-qkv|qualify-mtp-bf16-qk-prepare|qualify-mtp-bf16-paged-gqa|qualify-mtp-bf16-attention-output|qualify-mtp-bf16-mlp|qualify-full-attention-layer|qualify-mtp-layer|qualify-target-mtp-verify|qualify-mtp-prompt-prime|qualify-resident-mtp|\
    qualify-resident-model|qualify-resident-generation|qualify-generation-mtp-greedy|qualify-resident-batch-generation|\
    bench-residual-norm|bench-nvfp4-swiglu|bench-nvfp4-down|bench-nvfp4-mlp|bench-fp8-qkv|bench-fp8-gdn-input|\
    bench-fp8-lm-head|bench-attention-qk-prepare|bench-paged-gqa|bench-long-context-paged-gqa|bench-attention-output|bench-mtp-bf16-fusion|bench-mtp-bf16-qkv|bench-mtp-bf16-qk-prepare|bench-mtp-bf16-paged-gqa|bench-mtp-bf16-attention-output|bench-mtp-bf16-mlp|bench-full-attention-layer|bench-mtp-layer|bench-target-mtp-verify|bench-mtp-prompt-prime|bench-resident-mtp|bench-generation-mtp-greedy|\
    bench-resident-model|bench-resident-prefill|bench-resident-long-context-model|\
    probe|check|sweep> \
    [--gpu 5090|4090|3090] [--max-minutes N] [--image NAME] [--keep-on-fail] \
    [--samples N] [--launches-per-sample N] [--energy-seconds N]";

#[cfg(any(feature = "remote", test))]
struct Qualification {
    name: &'static str,
    filter: &'static str,
    source_snapshot: bool,
}

#[cfg(any(feature = "remote", test))]
impl Qualification {
    fn parse(name: &str) -> Option<Self> {
        let filter = match name {
            "qualify-residual-norm" => "residual_norm_suite_",
            "qualify-nvfp4-swiglu" => "nvfp4_swiglu::tests",
            "qualify-nvfp4-down" => "nvfp4_down::tests",
            "qualify-fp8-qkv" => "fp8_qkv",
            "qualify-fp8-gdn-input" => "fp8_gdn_input",
            "qualify-fp8-lm-head" => "fp8_lm_head",
            "qualify-nvfp4-mlp" => {
                "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay"
            }
            "qualify-attention-qk-prepare" => "attention_qk_prepare",
            "qualify-paged-gqa" => "paged_gqa_suite_",
            "qualify-long-context-paged-gqa" => "long_context_paged_gqa",
            "qualify-attention-output" => "attention_output::tests",
            "qualify-mtp-bf16-fusion" => "mtp_bf16_fusion_suite_",
            "qualify-mtp-bf16-qkv" => "mtp_bf16_qkv_suite_",
            "qualify-mtp-bf16-qk-prepare" => "mtp_bf16_qk_prepare_suite_",
            "qualify-mtp-bf16-paged-gqa" => "mtp_bf16_paged_gqa_suite_",
            "qualify-mtp-bf16-attention-output" => "mtp_bf16_attention_output_suite_",
            "qualify-mtp-bf16-mlp" => "mtp_bf16_mlp_suite_",
            "qualify-full-attention-layer" => {
                "full_attention_layer::tests::source_layer63_matches_complete_seam_oracles_and_graph_replay"
            }
            "qualify-mtp-layer" => "mtp_layer_suite_",
            "qualify-target-mtp-verify" => {
                "target_mtp_verify::tests::exact_target_verify_and_commit_match_source_oracles"
            }
            "qualify-mtp-prompt-prime" => "mtp_prompt_prime_suite_",
            "qualify-resident-mtp" => "resident_mtp_suite_",
            "qualify-generation-mtp-greedy" => "resident_mtp_generation_suite_",
            "qualify-resident-model" => {
                "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay"
            }
            "qualify-resident-generation" => {
                "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming"
            }
            "qualify-resident-batch-generation" => {
                "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes"
            }
            _ => return None,
        };

        Some(Self {
            name: match name {
                "qualify-residual-norm" => "residual-norm",
                "qualify-nvfp4-swiglu" => "nvfp4-swiglu",
                "qualify-nvfp4-down" => "nvfp4-down",
                "qualify-fp8-qkv" => "fp8-qkv",
                "qualify-fp8-gdn-input" => "fp8-gdn-input",
                "qualify-fp8-lm-head" => "fp8-lm-head",
                "qualify-nvfp4-mlp" => "nvfp4-mlp",
                "qualify-attention-qk-prepare" => "attention-qk-prepare",
                "qualify-paged-gqa" => "paged-gqa",
                "qualify-long-context-paged-gqa" => "long-context-paged-gqa",
                "qualify-attention-output" => "attention-output",
                "qualify-mtp-bf16-fusion" => "mtp-bf16-fusion",
                "qualify-mtp-bf16-qkv" => "mtp-bf16-qkv",
                "qualify-mtp-bf16-qk-prepare" => "mtp-bf16-qk-prepare",
                "qualify-mtp-bf16-paged-gqa" => "mtp-bf16-paged-gqa",
                "qualify-mtp-bf16-attention-output" => "mtp-bf16-attention-output",
                "qualify-mtp-bf16-mlp" => "mtp-bf16-mlp",
                "qualify-full-attention-layer" => "full-attention-layer",
                "qualify-mtp-layer" => "mtp-layer",
                "qualify-target-mtp-verify" => "target-mtp-verify",
                "qualify-mtp-prompt-prime" => "mtp-prompt-prime",
                "qualify-resident-mtp" => "resident-mtp",
                "qualify-generation-mtp-greedy" => "generation-mtp-greedy",
                "qualify-resident-model" => "resident-model",
                "qualify-resident-generation" => "resident-generation",
                "qualify-resident-batch-generation" => "resident-batch-generation",
                _ => unreachable!(),
            },
            filter,
            source_snapshot: matches!(
                name,
                "qualify-nvfp4-mlp"
                    | "qualify-mtp-bf16-fusion"
                    | "qualify-mtp-bf16-qkv"
                    | "qualify-mtp-bf16-qk-prepare"
                    | "qualify-mtp-bf16-attention-output"
                    | "qualify-mtp-bf16-mlp"
                    | "qualify-full-attention-layer"
                    | "qualify-mtp-layer"
                    | "qualify-target-mtp-verify"
                    | "qualify-mtp-prompt-prime"
                    | "qualify-resident-mtp"
                    | "qualify-generation-mtp-greedy"
                    | "qualify-resident-model"
                    | "qualify-resident-generation"
                    | "qualify-resident-batch-generation"
            ),
        })
    }
}

#[cfg(any(feature = "remote", test))]
#[derive(Clone, Copy)]
enum Benchmark {
    Leaf(crate::PerformanceSuite),
    Nvfp4Mlp,
    FullAttentionLayer,
    MtpLayer,
    TargetMtpVerify,
    MtpPromptPrime,
    ResidentMtp,
    GenerationMtpGreedy,
    ResidentModel,
    ResidentPrefill,
    ResidentLongContextModel,
}

#[cfg(any(feature = "remote", test))]
impl Benchmark {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "bench-residual-norm" => Self::Leaf(crate::PerformanceSuite::ResidualNorm),
            "bench-fp8-qkv" => Self::Leaf(crate::PerformanceSuite::Fp8Qkv),
            "bench-fp8-gdn-input" => Self::Leaf(crate::PerformanceSuite::Fp8GdnInput),
            "bench-fp8-lm-head" => Self::Leaf(crate::PerformanceSuite::Fp8LmHead),
            "bench-nvfp4-swiglu" => Self::Leaf(crate::PerformanceSuite::Nvfp4SwiGlu),
            "bench-nvfp4-down" => Self::Leaf(crate::PerformanceSuite::Nvfp4Down),
            "bench-nvfp4-mlp" => Self::Nvfp4Mlp,
            "bench-attention-qk-prepare" => Self::Leaf(crate::PerformanceSuite::AttentionQkPrepare),
            "bench-paged-gqa" => Self::Leaf(crate::PerformanceSuite::PagedGqa),
            "bench-long-context-paged-gqa" => {
                Self::Leaf(crate::PerformanceSuite::LongContextPagedGqa)
            }
            "bench-attention-output" => Self::Leaf(crate::PerformanceSuite::AttentionOutput),
            "bench-mtp-bf16-fusion" => Self::Leaf(crate::PerformanceSuite::MtpBf16Fusion),
            "bench-mtp-bf16-qkv" => Self::Leaf(crate::PerformanceSuite::MtpBf16Qkv),
            "bench-mtp-bf16-qk-prepare" => Self::Leaf(crate::PerformanceSuite::MtpBf16QkPrepare),
            "bench-mtp-bf16-paged-gqa" => Self::Leaf(crate::PerformanceSuite::MtpBf16PagedGqa),
            "bench-mtp-bf16-attention-output" => {
                Self::Leaf(crate::PerformanceSuite::MtpBf16AttentionOutput)
            }
            "bench-mtp-bf16-mlp" => Self::Leaf(crate::PerformanceSuite::MtpBf16Mlp),
            "bench-full-attention-layer" => Self::FullAttentionLayer,
            "bench-mtp-layer" => Self::MtpLayer,
            "bench-target-mtp-verify" => Self::TargetMtpVerify,
            "bench-mtp-prompt-prime" => Self::MtpPromptPrime,
            "bench-resident-mtp" => Self::ResidentMtp,
            "bench-generation-mtp-greedy" => Self::GenerationMtpGreedy,
            "bench-resident-model" => Self::ResidentModel,
            "bench-resident-prefill" => Self::ResidentPrefill,
            "bench-resident-long-context-model" => Self::ResidentLongContextModel,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Leaf(suite) => suite.name(),
            Self::Nvfp4Mlp => "nvfp4-mlp",
            Self::FullAttentionLayer => "full-attention-layer",
            Self::MtpLayer => "mtp-layer",
            Self::TargetMtpVerify => "target-mtp-verify",
            Self::MtpPromptPrime => "mtp-prompt-prime",
            Self::ResidentMtp => "resident-mtp",
            Self::GenerationMtpGreedy => "generation-mtp-greedy",
            Self::ResidentModel => "resident-model",
            Self::ResidentPrefill => "resident-prefill",
            Self::ResidentLongContextModel => "resident-long-context-model",
        }
    }

    fn source_snapshot(self) -> bool {
        matches!(
            self,
            Self::Nvfp4Mlp
                | Self::FullAttentionLayer
                | Self::MtpLayer
                | Self::TargetMtpVerify
                | Self::MtpPromptPrime
                | Self::ResidentMtp
                | Self::GenerationMtpGreedy
                | Self::ResidentModel
                | Self::ResidentPrefill
                | Self::ResidentLongContextModel
        )
    }
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
        return Err(USAGE.into());
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

    let qualification = Qualification::parse(command);
    let benchmark = Benchmark::parse(command);
    if qualification.is_none() && benchmark.is_none() {
        return Err(USAGE.into());
    }
    let options = parse_options(&arguments[1..], benchmark.is_some())?;

    if !target_supports(options.gpu, command) {
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
    } else if benchmark.is_some() {
        crate::build_residual_benchmark_target(root, options.gpu)?;
    }
    if let Some(qualification) = qualification {
        let prepared = crate::prepare_remote_qualify(root, options.gpu, qualification.filter)?;
        gate_static_resources(root, options.gpu, command)?;
        tuisko_remote::run_qualification(
            root,
            &tuisko_remote::QualificationOptions {
                suite: qualification.name.to_owned(),
                executable: prepared.executable,
                test_args: prepared.test_args,
                gpu: options.gpu.remote_gpu(),
                source_snapshot: qualification.source_snapshot,
                image: options.image,
                max_minutes: options.max_minutes,
                keep_on_fail: options.keep_on_fail,
            },
        )
        .map_err(|error| format!("{error}"))?;
    } else if let Some(benchmark) = benchmark {
        gate_static_resources(root, options.gpu, command)?;
        let prepared = match benchmark {
            Benchmark::Leaf(suite) => crate::prepare_remote_benchmark(root, options.gpu, suite)?,
            Benchmark::Nvfp4Mlp => crate::prepare_remote_nvfp4_mlp_benchmark(root, options.gpu)?,
            Benchmark::FullAttentionLayer => {
                crate::prepare_remote_full_attention_layer_benchmark(root, options.gpu)?
            }
            Benchmark::MtpLayer => crate::prepare_remote_mtp_layer_benchmark(root, options.gpu)?,
            Benchmark::TargetMtpVerify => {
                crate::prepare_remote_resident_model_benchmark(root, options.gpu)?
            }
            Benchmark::MtpPromptPrime => {
                crate::prepare_remote_mtp_prompt_prime_benchmark(root, options.gpu)?
            }
            Benchmark::ResidentMtp => {
                crate::prepare_remote_resident_mtp_benchmark(root, options.gpu)?
            }
            Benchmark::GenerationMtpGreedy => {
                crate::prepare_remote_resident_mtp_benchmark(root, options.gpu)?
            }
            Benchmark::ResidentModel => {
                crate::prepare_remote_resident_model_benchmark(root, options.gpu)?
            }
            Benchmark::ResidentPrefill => {
                crate::prepare_remote_resident_model_benchmark(root, options.gpu)?
            }
            Benchmark::ResidentLongContextModel => {
                crate::prepare_remote_resident_model_benchmark(root, options.gpu)?
            }
        };
        tuisko_remote::run_benchmark(
            root,
            &tuisko_remote::BenchmarkOptions {
                suite: benchmark.name().to_owned(),
                executable: prepared.executable,
                benchmark_args: options.benchmark_args,
                generator_baseline_sha256: prepared.generator_baseline_sha256,
                gpu: options.gpu.remote_gpu(),
                source_snapshot: benchmark.source_snapshot(),
                image: options.image,
                max_minutes: options.max_minutes,
                keep_on_fail: options.keep_on_fail,
            },
        )
        .map_err(|error| format!("{error}"))?;
    }

    Ok(())
}

#[cfg(any(feature = "remote", test))]
fn target_supports(gpu: GpuTarget, command: &str) -> bool {
    has_full_kernel_inventory(gpu)
        || matches!(command, "qualify-residual-norm" | "bench-residual-norm")
        || matches!(gpu, GpuTarget::Sm89)
            && matches!(
                command,
                "qualify-nvfp4-down" | "bench-nvfp4-down" | "qualify-fp8-qkv" | "bench-fp8-qkv"
            )
        || matches!(gpu, GpuTarget::Sm89 | GpuTarget::Sm86)
            && matches!(command, "qualify-nvfp4-swiglu" | "bench-nvfp4-swiglu")
}

#[cfg(feature = "remote")]
fn gate_static_resources(root: &Path, gpu: GpuTarget, command: &str) -> Result<(), Box<dyn Error>> {
    if command.contains("nvfp4-swiglu") {
        crate::gate_nvfp4_swiglu_target(root, gpu)
    } else if command.contains("nvfp4-down") {
        crate::gate_nvfp4_down_target(root, gpu)
    } else if command.contains("fp8-qkv") && gpu == GpuTarget::Sm89 {
        crate::gate_fp8_qkv_sm89(root)
    } else if command.contains("resident-mtp") {
        crate::gate_resident_mtp(root)
    } else if command.contains("mtp-prompt-prime") {
        crate::gate_mtp_prompt_prime(root)
    } else if command.contains("target-mtp-verify") {
        crate::gate_resident_model_resources(root)
    } else if command.contains("mtp-layer") {
        crate::gate_mtp_layer(root)
    } else if command.contains("mtp-bf16-fusion") {
        crate::gate_mtp_bf16_fusion(root)
    } else if command.contains("mtp-bf16-qkv") {
        crate::gate_mtp_bf16_qkv(root)
    } else if command.contains("mtp-bf16-qk-prepare") {
        crate::gate_mtp_bf16_qk_prepare(root)
    } else if command.contains("mtp-bf16-paged-gqa") {
        crate::gate_mtp_bf16_paged_gqa(root)
    } else if command.contains("mtp-bf16-attention-output") {
        crate::gate_mtp_bf16_attention_output(root)
    } else if command.contains("mtp-bf16-mlp") {
        crate::gate_mtp_bf16_mlp(root)
    } else {
        crate::gate_residual_norm_target(root, gpu)
    }
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

#[cfg(not(feature = "remote"))]
fn run_impl(_root: &Path, _arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("remote execution requires `cargo run -p xtask --features remote -- remote ...`".into())
}

#[cfg(test)]
mod tests {
    use super::{Benchmark, Qualification, parse_options, target_supports};
    use crate::gpu_target::GpuTarget;

    #[test]
    fn remote_suite_inventory_is_exact() {
        let qualification = Qualification::parse("qualify-residual-norm").expect("known suite");
        assert_eq!(qualification.name, "residual-norm");
        assert_eq!(qualification.filter, "residual_norm_suite_");
        let qualification = Qualification::parse("qualify-nvfp4-swiglu").expect("known suite");
        assert_eq!(qualification.name, "nvfp4-swiglu");
        assert_eq!(qualification.filter, "nvfp4_swiglu::tests");
        let qualification = Qualification::parse("qualify-nvfp4-down").expect("known suite");
        assert_eq!(qualification.name, "nvfp4-down");
        assert_eq!(qualification.filter, "nvfp4_down::tests");
        let attention = Qualification::parse("qualify-attention-qk-prepare").expect("known suite");
        assert_eq!(attention.name, "attention-qk-prepare");
        assert_eq!(attention.filter, "attention_qk_prepare");
        let paged = Qualification::parse("qualify-paged-gqa").expect("known suite");
        assert_eq!(paged.name, "paged-gqa");
        assert_eq!(paged.filter, "paged_gqa_suite_");
        let long = Qualification::parse("qualify-long-context-paged-gqa").expect("known suite");
        assert_eq!(long.name, "long-context-paged-gqa");
        assert_eq!(long.filter, "long_context_paged_gqa");
        let output = Qualification::parse("qualify-attention-output").expect("known suite");
        assert_eq!(output.name, "attention-output");
        assert_eq!(output.filter, "attention_output::tests");
        let mtp = Qualification::parse("qualify-mtp-bf16-fusion").expect("known suite");
        assert_eq!(mtp.name, "mtp-bf16-fusion");
        assert_eq!(mtp.filter, "mtp_bf16_fusion_suite_");
        assert!(mtp.source_snapshot);
        let mtp_qkv = Qualification::parse("qualify-mtp-bf16-qkv").expect("known suite");
        assert_eq!(mtp_qkv.name, "mtp-bf16-qkv");
        assert_eq!(mtp_qkv.filter, "mtp_bf16_qkv_suite_");
        assert!(mtp_qkv.source_snapshot);
        let mtp_qk = Qualification::parse("qualify-mtp-bf16-qk-prepare").expect("known suite");
        assert_eq!(mtp_qk.name, "mtp-bf16-qk-prepare");
        assert_eq!(mtp_qk.filter, "mtp_bf16_qk_prepare_suite_");
        assert!(mtp_qk.source_snapshot);
        let mtp_attention =
            Qualification::parse("qualify-mtp-bf16-paged-gqa").expect("known suite");
        assert_eq!(mtp_attention.name, "mtp-bf16-paged-gqa");
        assert_eq!(mtp_attention.filter, "mtp_bf16_paged_gqa_suite_");
        assert!(!mtp_attention.source_snapshot);
        let mtp_output =
            Qualification::parse("qualify-mtp-bf16-attention-output").expect("known suite");
        assert_eq!(mtp_output.name, "mtp-bf16-attention-output");
        assert_eq!(mtp_output.filter, "mtp_bf16_attention_output_suite_");
        assert!(mtp_output.source_snapshot);
        let mtp_mlp = Qualification::parse("qualify-mtp-bf16-mlp").expect("known suite");
        assert_eq!(mtp_mlp.name, "mtp-bf16-mlp");
        assert_eq!(mtp_mlp.filter, "mtp_bf16_mlp_suite_");
        assert!(mtp_mlp.source_snapshot);
        let nvfp4_mlp = Qualification::parse("qualify-nvfp4-mlp").expect("known suite");
        assert_eq!(nvfp4_mlp.name, "nvfp4-mlp");
        assert!(nvfp4_mlp.source_snapshot);
        let full = Qualification::parse("qualify-full-attention-layer").expect("known suite");
        assert_eq!(full.name, "full-attention-layer");
        assert!(full.source_snapshot);
        let mtp_layer = Qualification::parse("qualify-mtp-layer").expect("known suite");
        assert_eq!(mtp_layer.name, "mtp-layer");
        assert_eq!(mtp_layer.filter, "mtp_layer_suite_");
        assert!(mtp_layer.source_snapshot);
        let target = Qualification::parse("qualify-target-mtp-verify").expect("known suite");
        assert_eq!(target.name, "target-mtp-verify");
        assert!(target.filter.contains("exact_target_verify_and_commit"));
        assert!(target.source_snapshot);
        let prompt = Qualification::parse("qualify-mtp-prompt-prime").expect("known suite");
        assert_eq!(prompt.name, "mtp-prompt-prime");
        assert_eq!(prompt.filter, "mtp_prompt_prime_suite_");
        assert!(prompt.source_snapshot);
        let resident_mtp = Qualification::parse("qualify-resident-mtp").expect("known suite");
        assert_eq!(resident_mtp.name, "resident-mtp");
        assert_eq!(resident_mtp.filter, "resident_mtp_suite_");
        assert!(resident_mtp.source_snapshot);
        let mtp_generation =
            Qualification::parse("qualify-generation-mtp-greedy").expect("known suite");
        assert_eq!(mtp_generation.name, "generation-mtp-greedy");
        assert_eq!(mtp_generation.filter, "resident_mtp_generation_suite_");
        assert!(mtp_generation.source_snapshot);
        let generation = Qualification::parse("qualify-resident-generation").expect("known suite");
        assert_eq!(generation.name, "resident-generation");
        assert!(generation.source_snapshot);
        let batch = Qualification::parse("qualify-resident-batch-generation").expect("known suite");
        assert_eq!(batch.name, "resident-batch-generation");
        assert!(batch.source_snapshot);
        assert_eq!(Benchmark::parse("bench-fp8-qkv").unwrap().name(), "fp8-qkv");
        assert_eq!(
            Benchmark::parse("bench-nvfp4-swiglu").unwrap().name(),
            "nvfp4-swiglu"
        );
        assert_eq!(
            Benchmark::parse("bench-nvfp4-down").unwrap().name(),
            "nvfp4-down"
        );
        let mlp = Benchmark::parse("bench-nvfp4-mlp").expect("known benchmark");
        assert_eq!(mlp.name(), "nvfp4-mlp");
        assert!(mlp.source_snapshot());
        let full = Benchmark::parse("bench-full-attention-layer").expect("known benchmark");
        assert_eq!(full.name(), "full-attention-layer");
        assert!(full.source_snapshot());
        let mtp_layer = Benchmark::parse("bench-mtp-layer").expect("known benchmark");
        assert_eq!(mtp_layer.name(), "mtp-layer");
        assert!(mtp_layer.source_snapshot());
        let target = Benchmark::parse("bench-target-mtp-verify").expect("known benchmark");
        assert_eq!(target.name(), "target-mtp-verify");
        assert!(target.source_snapshot());
        let prompt = Benchmark::parse("bench-mtp-prompt-prime").expect("known benchmark");
        assert_eq!(prompt.name(), "mtp-prompt-prime");
        assert!(prompt.source_snapshot());
        let resident_mtp = Benchmark::parse("bench-resident-mtp").expect("known benchmark");
        assert_eq!(resident_mtp.name(), "resident-mtp");
        assert!(resident_mtp.source_snapshot());
        let generation_mtp =
            Benchmark::parse("bench-generation-mtp-greedy").expect("known benchmark");
        assert_eq!(generation_mtp.name(), "generation-mtp-greedy");
        assert!(generation_mtp.source_snapshot());
        assert_eq!(
            Benchmark::parse("bench-attention-qk-prepare")
                .expect("known benchmark")
                .name(),
            "attention-qk-prepare"
        );
        assert_eq!(
            Benchmark::parse("bench-paged-gqa")
                .expect("known benchmark")
                .name(),
            "paged-gqa"
        );
        assert_eq!(
            Benchmark::parse("bench-long-context-paged-gqa")
                .expect("known benchmark")
                .name(),
            "long-context-paged-gqa"
        );
        assert_eq!(
            Benchmark::parse("bench-attention-output")
                .expect("known benchmark")
                .name(),
            "attention-output"
        );
        let mtp = Benchmark::parse("bench-mtp-bf16-fusion").expect("known benchmark");
        assert_eq!(mtp.name(), "mtp-bf16-fusion");
        assert!(!mtp.source_snapshot());
        let mtp_qkv = Benchmark::parse("bench-mtp-bf16-qkv").expect("known benchmark");
        assert_eq!(mtp_qkv.name(), "mtp-bf16-qkv");
        assert!(!mtp_qkv.source_snapshot());
        let mtp_qk = Benchmark::parse("bench-mtp-bf16-qk-prepare").expect("known benchmark");
        assert_eq!(mtp_qk.name(), "mtp-bf16-qk-prepare");
        assert!(!mtp_qk.source_snapshot());
        let mtp_attention = Benchmark::parse("bench-mtp-bf16-paged-gqa").expect("known benchmark");
        assert_eq!(mtp_attention.name(), "mtp-bf16-paged-gqa");
        assert!(!mtp_attention.source_snapshot());
        let mtp_output =
            Benchmark::parse("bench-mtp-bf16-attention-output").expect("known benchmark");
        assert_eq!(mtp_output.name(), "mtp-bf16-attention-output");
        assert!(!mtp_output.source_snapshot());
        let mtp_mlp = Benchmark::parse("bench-mtp-bf16-mlp").expect("known benchmark");
        assert_eq!(mtp_mlp.name(), "mtp-bf16-mlp");
        assert!(!mtp_mlp.source_snapshot());
        assert!(Qualification::parse("bench-residual-norm").is_none());
        assert!(Benchmark::parse("perf").is_none());
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
        for command in [
            "qualify-residual-norm",
            "bench-residual-norm",
            "qualify-fp8-qkv",
            "bench-fp8-qkv",
        ] {
            assert!(target_supports(GpuTarget::Sm120, command));
        }
        for gpu in [GpuTarget::Sm89, GpuTarget::Sm86] {
            assert!(target_supports(gpu, "qualify-residual-norm"));
            assert!(target_supports(gpu, "bench-residual-norm"));
        }
        assert!(target_supports(GpuTarget::Sm89, "qualify-nvfp4-swiglu"));
        assert!(target_supports(GpuTarget::Sm89, "bench-nvfp4-swiglu"));
        assert!(target_supports(GpuTarget::Sm89, "qualify-nvfp4-down"));
        assert!(target_supports(GpuTarget::Sm89, "bench-nvfp4-down"));
        assert!(target_supports(GpuTarget::Sm89, "qualify-fp8-qkv"));
        assert!(target_supports(GpuTarget::Sm89, "bench-fp8-qkv"));
        assert!(target_supports(GpuTarget::Sm86, "qualify-nvfp4-swiglu"));
        assert!(target_supports(GpuTarget::Sm86, "bench-nvfp4-swiglu"));
        assert!(!target_supports(GpuTarget::Sm86, "qualify-nvfp4-down"));
        assert!(!target_supports(GpuTarget::Sm86, "bench-nvfp4-down"));
        assert!(!target_supports(GpuTarget::Sm86, "qualify-fp8-qkv"));
        assert!(!target_supports(GpuTarget::Sm86, "bench-fp8-qkv"));
    }
}
