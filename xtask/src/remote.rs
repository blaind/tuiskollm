//! Remote execution of prebuilt device qualification and benchmark artifacts.

use std::error::Error;
use std::ffi::OsString;
use std::path::Path;

#[cfg(any(feature = "remote", test))]
use crate::gpu_target::GpuTarget;

#[cfg(feature = "remote")]
const USAGE: &str = "usage: cargo run -p xtask --features remote -- remote \
    <qualify-residual-norm|qualify-fp8-qkv|qualify-fp8-gdn-input|qualify-fp8-lm-head|\
    bench-residual-norm|bench-fp8-qkv|bench-fp8-gdn-input|bench-fp8-lm-head|probe|check|sweep> \
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
            "qualify-fp8-qkv" => "fp8_qkv",
            "qualify-fp8-gdn-input" => "fp8_gdn_input",
            "qualify-fp8-lm-head" => "fp8_lm_head",
            _ => return None,
        };

        Some(Self {
            name: match name {
                "qualify-residual-norm" => "residual-norm",
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

    let kernel_crate = options.gpu.kernel_crate().ok_or_else(|| {
        format!(
            "GPU {} ({}, compute capability {}, cuda-oxide {}) has no qualified kernel crate yet; use `remote probe --gpu {}` to test provisioning only",
            options.gpu.key(),
            options.gpu.device_name(),
            options.gpu.compute_capability(),
            options.gpu.oxide_arch(),
            options.gpu.key(),
        )
    })?;
    if kernel_crate != "tuisko-kernels-sm120" {
        return Err(format!("unsupported kernel crate `{kernel_crate}`").into());
    }

    tuisko_remote::check_credentials().map_err(|error| format!("{error}"))?;
    crate::build_sm120(root)?;
    if let Some(qualification) = qualification {
        let prepared = crate::prepare_remote_qualify(root, qualification.filter)?;
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
        let prepared = crate::prepare_remote_benchmark(root, benchmark.suite)?;
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
fn parse_options(arguments: &[String], benchmark: bool) -> Result<RemoteOptions, Box<dyn Error>> {
    let mut options = RemoteOptions {
        gpu: GpuTarget::Rtx5090,
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
    use super::{Benchmark, Qualification, parse_options};
    use crate::gpu_target::GpuTarget;

    #[test]
    fn remote_suite_inventory_is_exact() {
        let qualification = Qualification::parse("qualify-residual-norm").expect("known suite");
        assert_eq!(qualification.name, "residual-norm");
        assert_eq!(qualification.filter, "residual_norm::tests");
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
        assert_eq!(parse_options(&[], false).unwrap().gpu, GpuTarget::Rtx5090);
        assert_eq!(
            parse_options(&["--gpu".to_owned(), "4090".to_owned()], false)
                .unwrap()
                .gpu,
            GpuTarget::Rtx4090
        );
        assert!(parse_options(&["--gpu".to_owned(), "A100".to_owned()], false).is_err());
    }
}
