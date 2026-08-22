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
    bench-residual-norm|bench-nvfp4-swiglu|bench-nvfp4-down|bench-fp8-qkv|bench-fp8-gdn-input|bench-fp8-lm-head|probe|check|sweep> \
    [--gpu 5090|4090|3090] [--max-minutes N] [--image NAME] [--keep-on-fail] \
    [--samples N] [--launches-per-sample N] [--energy-seconds N]";

#[cfg(any(feature = "remote", test))]
struct Qualification {
    name: &'static str,
    filter: &'static str,
}

#[cfg(any(feature = "remote", test))]
impl Qualification {
    fn parse(name: &str) -> Option<Self> {
        let filter = match name {
            "qualify-residual-norm" => "residual_norm::tests",
            "qualify-nvfp4-swiglu" => "nvfp4_swiglu::tests",
            "qualify-nvfp4-down" => "nvfp4_down::tests",
            "qualify-fp8-qkv" => "fp8_qkv",
            "qualify-fp8-gdn-input" => "fp8_gdn_input",
            "qualify-fp8-lm-head" => "fp8_lm_head",
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
                _ => unreachable!(),
            },
            filter,
        })
    }
}

#[cfg(any(feature = "remote", test))]
struct Benchmark {
    suite: crate::PerformanceSuite,
}

#[cfg(any(feature = "remote", test))]
impl Benchmark {
    fn parse(name: &str) -> Option<Self> {
        let suite = match name {
            "bench-residual-norm" => crate::PerformanceSuite::ResidualNorm,
            "bench-nvfp4-swiglu" => crate::PerformanceSuite::Nvfp4SwiGlu,
            "bench-nvfp4-down" => crate::PerformanceSuite::Nvfp4Down,
            "bench-fp8-qkv" => crate::PerformanceSuite::Fp8Qkv,
            "bench-fp8-gdn-input" => crate::PerformanceSuite::Fp8GdnInput,
            "bench-fp8-lm-head" => crate::PerformanceSuite::Fp8LmHead,
            _ => return None,
        };
        Some(Self { suite })
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
                image: options.image,
                max_minutes: options.max_minutes,
                keep_on_fail: options.keep_on_fail,
            },
        )
        .map_err(|error| format!("{error}"))?;
    } else if let Some(benchmark) = benchmark {
        gate_static_resources(root, options.gpu, command)?;
        let prepared = crate::prepare_remote_benchmark(root, options.gpu, benchmark.suite)?;
        tuisko_remote::run_benchmark(
            root,
            &tuisko_remote::BenchmarkOptions {
                suite: benchmark.suite.name().to_owned(),
                executable: prepared.executable,
                benchmark_args: options.benchmark_args,
                generator_baseline_sha256: prepared.generator_baseline_sha256,
                gpu: options.gpu.remote_gpu(),
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
        assert_eq!(qualification.filter, "residual_norm::tests");
        let qualification = Qualification::parse("qualify-nvfp4-swiglu").expect("known suite");
        assert_eq!(qualification.name, "nvfp4-swiglu");
        assert_eq!(qualification.filter, "nvfp4_swiglu::tests");
        let qualification = Qualification::parse("qualify-nvfp4-down").expect("known suite");
        assert_eq!(qualification.name, "nvfp4-down");
        assert_eq!(qualification.filter, "nvfp4_down::tests");
        assert_eq!(
            Benchmark::parse("bench-fp8-qkv")
                .expect("known benchmark")
                .suite
                .name(),
            "fp8-qkv"
        );
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
