//! Repository build and qualification gates.

mod performance;

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const RESIDUAL_NORM_RESOURCE_BASELINE: &str = "qual/baselines/residual-norm-sm120.txt";
const FP8_QKV_RESOURCE_BASELINE: &str = "qual/baselines/fp8-qkv-sm120.txt";
const FP8_GDN_INPUT_RESOURCE_BASELINE: &str = "qual/baselines/fp8-gdn-input-sm120.txt";
const FP8_LM_HEAD_RESOURCE_BASELINE: &str = "qual/baselines/fp8-lm-head-sm120.txt";
const FP8_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/fp8-swiglu-sm120.txt";
const FP8_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/fp8-down-sm120.txt";
const PTX: &str = "target/cuda/tuisko_kernels_sm120.ptx";
const CUDA_OXIDE_BUILD_TARGET: &str = "target/cuda-oxide-build";
const CUDA_OXIDE_TEST_TARGET: &str = "target/cuda-oxide-test";
const CUDA_OXIDE_REPOSITORY: &str = "https://github.com/NVlabs/cuda-oxide.git";
const CUDA_OXIDE_REVISION: &str = "1f4d813719012d384f2db12b88efc9314c8bf50c";

#[derive(Clone, Copy)]
enum PerformanceSuite {
    ResidualNorm,
    Fp8Qkv,
    Fp8GdnInput,
    Fp8LmHead,
    Fp8SwiGlu,
    Fp8Down,
}

const PERFORMANCE_SUITES: [PerformanceSuite; 4] = [
    PerformanceSuite::ResidualNorm,
    PerformanceSuite::Fp8Qkv,
    PerformanceSuite::Fp8GdnInput,
    PerformanceSuite::Fp8LmHead,
];

impl PerformanceSuite {
    const fn name(self) -> &'static str {
        match self {
            Self::ResidualNorm => "residual-norm",
            Self::Fp8Qkv => "fp8-qkv",
            Self::Fp8GdnInput => "fp8-gdn-input",
            Self::Fp8LmHead => "fp8-lm-head",
            Self::Fp8SwiGlu => "fp8-swiglu",
            Self::Fp8Down => "fp8-down",
        }
    }

    const fn resource_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => RESIDUAL_NORM_RESOURCE_BASELINE,
            Self::Fp8Qkv => FP8_QKV_RESOURCE_BASELINE,
            Self::Fp8GdnInput => FP8_GDN_INPUT_RESOURCE_BASELINE,
            Self::Fp8LmHead => FP8_LM_HEAD_RESOURCE_BASELINE,
            Self::Fp8SwiGlu => FP8_SWIGLU_RESOURCE_BASELINE,
            Self::Fp8Down => FP8_DOWN_RESOURCE_BASELINE,
        }
    }

    const fn performance_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => "qual/baselines/residual-norm-sm120.json",
            Self::Fp8Qkv => "qual/baselines/fp8-qkv-sm120.json",
            Self::Fp8GdnInput => "qual/baselines/fp8-gdn-input-sm120.json",
            Self::Fp8LmHead => "qual/baselines/fp8-lm-head-sm120.json",
            Self::Fp8SwiGlu => "qual/baselines/fp8-swiglu-sm120.json",
            Self::Fp8Down => "qual/baselines/fp8-down-sm120.json",
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "residual-norm" => Ok(Self::ResidualNorm),
            "fp8-qkv" => Ok(Self::Fp8Qkv),
            "fp8-gdn-input" => Ok(Self::Fp8GdnInput),
            "fp8-lm-head" => Ok(Self::Fp8LmHead),
            "fp8-swiglu" => Ok(Self::Fp8SwiGlu),
            "fp8-down" => Ok(Self::Fp8Down),
            _ => Err(format!("unknown performance suite `{value}`").into()),
        }
    }

    fn qualify(self, root: &Path) -> Result<(), Box<dyn Error>> {
        match self {
            Self::ResidualNorm => qualify_residual_norm(root),
            Self::Fp8Qkv => qualify_fp8_qkv(root),
            Self::Fp8GdnInput => qualify_fp8_gdn_input(root),
            Self::Fp8LmHead => qualify_fp8_lm_head(root),
            Self::Fp8SwiGlu => qualify_fp8_swiglu(root),
            Self::Fp8Down => qualify_fp8_down(root),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(command) = arguments.next() else {
        return Err("usage: cargo run -p xtask -- <bootstrap-cuda-oxide|build-sm120|qualify-frontend|qualify-generation|qualify-residual-norm|qualify-fp8-qkv|qualify-fp8-gdn-input|qualify-fp8-lm-head|qualify-fp8-swiglu|qualify-fp8-down|qualify-dense-fp8-mlp|qualify-text-endpoint|bench-residual-norm|bench-fp8-qkv|bench-fp8-gdn-input|bench-fp8-lm-head|bench-fp8-swiglu|bench-fp8-down|bench-dense-fp8-mlp|bench-text-endpoint|gate-residual-norm|gate-fp8-qkv|gate-fp8-gdn-input|gate-fp8-lm-head|gate-fp8-swiglu|gate-fp8-down|perf>".into());
    };
    let remaining = arguments.collect::<Vec<_>>();
    let root = workspace_root()?;

    match command.to_str() {
        Some("bootstrap-cuda-oxide") if remaining.is_empty() => bootstrap_cuda_oxide(root),
        Some("build-sm120") if remaining.is_empty() => build_sm120(root),
        Some("qualify-frontend") => qualify_frontend(root, &remaining),
        Some("qualify-generation") => qualify_generation(root, &remaining),
        Some("qualify-residual-norm") if remaining.is_empty() => qualify_residual_norm(root),
        Some("qualify-fp8-qkv") if remaining.is_empty() => qualify_fp8_qkv(root),
        Some("qualify-fp8-gdn-input") if remaining.is_empty() => qualify_fp8_gdn_input(root),
        Some("qualify-fp8-lm-head") if remaining.is_empty() => qualify_fp8_lm_head(root),
        Some("qualify-fp8-swiglu") if remaining.is_empty() => qualify_fp8_swiglu(root),
        Some("qualify-fp8-down") if remaining.is_empty() => qualify_fp8_down(root),
        Some("qualify-dense-fp8-mlp") => qualify_dense_fp8_mlp(root, &remaining),
        Some("qualify-text-endpoint") => qualify_text_endpoint(root, &remaining),
        Some("bench-residual-norm") => bench_residual_norm(root, &remaining),
        Some("bench-fp8-qkv") => bench_fp8_qkv(root, &remaining),
        Some("bench-fp8-gdn-input") => bench_fp8_gdn_input(root, &remaining),
        Some("bench-fp8-lm-head") => bench_fp8_lm_head(root, &remaining),
        Some("bench-fp8-swiglu") => bench_fp8_swiglu(root, &remaining),
        Some("bench-fp8-down") => bench_fp8_down(root, &remaining),
        Some("bench-dense-fp8-mlp") => bench_dense_fp8_mlp(root, &remaining),
        Some("bench-text-endpoint") => bench_text_endpoint(root, &remaining),
        Some("gate-residual-norm") if remaining.is_empty() => gate_residual_norm(root),
        Some("gate-fp8-qkv") if remaining.is_empty() => gate_fp8_qkv(root),
        Some("gate-fp8-gdn-input") if remaining.is_empty() => gate_fp8_gdn_input(root),
        Some("gate-fp8-lm-head") if remaining.is_empty() => gate_fp8_lm_head(root),
        Some("gate-fp8-swiglu") if remaining.is_empty() => gate_fp8_swiglu(root),
        Some("gate-fp8-down") if remaining.is_empty() => gate_fp8_down(root),
        Some("perf") => perf(root, &remaining),
        Some(known)
            if matches!(
                known,
                "bootstrap-cuda-oxide"
                    | "build-sm120"
                    | "qualify-residual-norm"
                    | "qualify-fp8-qkv"
                    | "qualify-fp8-gdn-input"
                    | "qualify-fp8-lm-head"
                    | "qualify-fp8-swiglu"
                    | "qualify-fp8-down"
                    | "gate-residual-norm"
                    | "gate-fp8-qkv"
                    | "gate-fp8-gdn-input"
                    | "gate-fp8-lm-head"
                    | "gate-fp8-swiglu"
                    | "gate-fp8-down"
            ) =>
        {
            Err(format!("`{known}` takes no arguments").into())
        }
        _ => Err(format!("unknown xtask command `{}`", command.to_string_lossy()).into()),
    }
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest has no workspace parent".into())
}

fn bootstrap_cuda_oxide(root: &Path) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    if !source.join(".git").is_dir() {
        fs::create_dir_all(
            source
                .parent()
                .ok_or("cuda-oxide source path has no parent")?,
        )?;
        run_visible(
            Command::new("git")
                .args(["clone", "--no-checkout", CUDA_OXIDE_REPOSITORY])
                .arg(&source),
        )?;
        run_visible(Command::new("git").arg("-C").arg(&source).args([
            "checkout",
            "--detach",
            CUDA_OXIDE_REVISION,
        ]))?;
    }
    require_cuda_oxide_revision(&source)?;

    let driver_target = root.join("target/cuda-oxide-driver");
    run_visible(
        Command::new("cargo")
            .arg("+nightly-2026-04-03")
            .arg("build")
            .arg("--manifest-path")
            .arg(source.join("Cargo.toml"))
            .args(["--package", "cargo-oxide", "--target-dir"])
            .arg(&driver_target)
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let wrapper = driver_target.join("debug/cargo-oxide");
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    run_visible(
        Command::new(&wrapper)
            .arg("setup")
            .current_dir(&source)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS"),
    )?;

    println!(
        "cuda-oxide ready: {} at {}",
        CUDA_OXIDE_REVISION,
        wrapper.display()
    );
    Ok(())
}

fn build_sm120(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-device",
            "--release",
        ],
    )?;
    gate_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_fp8_gdn_input(root)?;
    gate_fp8_lm_head(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_frontend(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-frontend SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--bin",
                "qualify-frontend",
                "--",
            ])
            .arg(snapshot),
    )
}

fn qualify_generation(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--features",
                "engine",
                "--bin",
                "qualify-generation",
                "--",
            ])
            .arg(snapshot),
    )
}

fn qualify_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "residual_norm::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_residual_norm(root)
}

fn qualify_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_qkv",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_qkv(root)
}

fn qualify_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_swiglu",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_swiglu(root)
}

fn qualify_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_down",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_down(root)
}

fn qualify_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-dense-fp8-mlp SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "dense_fp8_mlp::tests::source_layer60_matches_complete_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_gdn_input",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_gdn_input(root)
}

fn qualify_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_lm_head",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_lm_head(root)
}

fn qualify_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-text-endpoint SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "text_endpoint::tests::source_endpoint_matches_independent_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_lm_head(root)
}

fn bench_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::ResidualNorm, arguments)
}

fn bench_fp8_qkv(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Qkv, arguments)
}

fn bench_fp8_gdn_input(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8GdnInput, arguments)
}

fn bench_fp8_lm_head(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8LmHead, arguments)
}

fn bench_fp8_swiglu(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8SwiGlu, arguments)
}

fn bench_fp8_down(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Down, arguments)
}

fn bench_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-dense-fp8-mlp SNAPSHOT [options]".into());
    };
    build_sm120(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(FP8_SWIGLU_RESOURCE_BASELINE))?);
    baselines.extend_from_slice(&fs::read(root.join(FP8_DOWN_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("dense-fp8-mlp")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-text-endpoint SNAPSHOT [options]".into());
    };
    build_sm120(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(FP8_LM_HEAD_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("text-endpoint")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120(root)?;
    run_benchmark_suite(root, suite, arguments)
}

fn run_benchmark_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut command = Command::new(&executable);
    command.arg(suite.name()).args(arguments).env(
        "TUISKO_GENERATOR_BASELINE_SHA256",
        sha256(&fs::read(root.join(suite.resource_baseline()))?),
    );
    run_visible(&mut command)?;

    Ok(())
}

fn perf(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some(mode) = arguments.first() else {
        return Err(
            "usage: cargo run -p xtask -- perf <smoke|leaf|energy|gate|bless SUITE>".into(),
        );
    };
    let mode = mode.to_str().ok_or("perf mode is not UTF-8")?;
    if mode == "bless" {
        let [_, suite] = arguments else {
            return Err(
                "usage: cargo run -p xtask -- perf bless <residual-norm|fp8-qkv|fp8-gdn-input|fp8-lm-head>"
                    .into(),
            );
        };
        let suite = PerformanceSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
        return bless_suite(root, suite);
    }
    if arguments.len() != 1 {
        return Err(format!("`perf {mode}` takes no additional arguments").into());
    }

    let options = match mode {
        "smoke" => vec![
            "--samples".into(),
            "3".into(),
            "--launches-per-sample".into(),
            "1024".into(),
        ],
        "leaf" | "gate" => Vec::new(),
        "energy" => vec!["--energy-seconds".into(), "2".into()],
        _ => return Err(format!("unknown perf mode `{mode}`").into()),
    };
    if mode == "gate" {
        for suite in PERFORMANCE_SUITES {
            suite.qualify(root)?;
        }
    }
    build_sm120(root)?;
    run_performance_suites(root, mode, &options, mode == "gate")
}

fn run_performance_suites(
    root: &Path,
    mode: &str,
    options: &[std::ffi::OsString],
    compare: bool,
) -> Result<(), Box<dyn Error>> {
    for (index, suite) in PERFORMANCE_SUITES.into_iter().enumerate() {
        if index != 0 {
            wait_for_device_idle()?;
        }
        let report = performance_report_path(root, mode, suite);
        let mut arguments = options.to_vec();
        arguments.push("--json".into());
        arguments.push(path_text(&report)?.into());
        run_benchmark_suite(root, suite, &arguments)?;
    }
    if compare {
        for suite in PERFORMANCE_SUITES {
            let report = performance_report_path(root, mode, suite);
            performance::compare(&report, &root.join(suite.performance_baseline()))?;
        }
    }

    Ok(())
}

fn wait_for_device_idle() -> Result<(), Box<dyn Error>> {
    let timeout = Duration::from_secs(10);
    let deadline = Instant::now() + timeout;
    loop {
        let row = command_text(
            "nvidia-smi",
            &[
                "-i",
                "0",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
        )?;
        let (utilization, memory_mib) = parse_idle_sample(&row)?;
        if utilization == 0 && memory_mib <= 1_024 {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "device zero remained busy for {} seconds between performance suites: utilization={utilization}%, memory={memory_mib} MiB",
                timeout.as_secs()
            )
            .into());
        }

        std::thread::sleep(Duration::from_millis(100));
    }
}

fn parse_idle_sample(row: &str) -> Result<(u32, u64), Box<dyn Error>> {
    let fields = row.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [utilization, memory_mib] = fields.as_slice() else {
        return Err(format!("unexpected nvidia-smi idle row `{}`", row.trim()).into());
    };

    Ok((utilization.parse()?, memory_mib.parse()?))
}

fn sass_function_body<'a>(sass: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("Function : {name}");
    let body = &sass[sass.find(&marker)? + marker.len()..];

    Some(body.split("\n\t\tFunction :").next().unwrap_or(body))
}

fn bless_suite(root: &Path, suite: PerformanceSuite) -> Result<(), Box<dyn Error>> {
    suite.qualify(root)?;
    build_sm120(root)?;
    let report = performance_report_path(root, "bless", suite);
    run_benchmark_suite(root, suite, &["--json".into(), path_text(&report)?.into()])?;
    performance::bless(&report, &root.join(suite.performance_baseline()))
}

fn performance_report_path(root: &Path, mode: &str, suite: PerformanceSuite) -> PathBuf {
    root.join(format!(
        "target/benchmarks/perf-{mode}/{}.json",
        suite.name()
    ))
}

fn run_oxide(root: &Path, arguments: &[&str]) -> Result<(), Box<dyn Error>> {
    run_oxide_with_env(root, arguments, None)
}

fn run_oxide_with_env(
    root: &Path,
    arguments: &[&str],
    environment: Option<(&str, &OsStr)>,
) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    require_cuda_oxide_revision(&source)?;
    let wrapper = root.join("target/cuda-oxide-driver/debug/cargo-oxide");
    if !wrapper.is_file() {
        return Err(
            "cuda-oxide is not bootstrapped; run `cargo run -p xtask -- bootstrap-cuda-oxide`"
                .into(),
        );
    }
    let backend = local_backend(root)?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    fs::create_dir_all(root.join("target/tmp"))?;
    let mut command = Command::new(wrapper);
    command
        .args(arguments)
        .current_dir(root)
        .env("CARGO_HOME", task_cargo_home(root))
        .env("CUDA_OXIDE_BACKEND", backend)
        .env("CUDA_OXIDE_SOURCE", source)
        .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
        .env_remove("RUSTFLAGS")
        .env("TMPDIR", root.join("target/tmp"));
    if let Some((name, value)) = environment {
        command.env(name, value);
    }

    run_visible(&mut command)
}

fn run_visible(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(format!("{program} failed with {status}").into());
    }

    Ok(())
}

fn require_cuda_oxide_revision(source: &Path) -> Result<(), Box<dyn Error>> {
    let commit = command_text("git", &["-C", path_text(source)?, "rev-parse", "HEAD"])?;
    if commit.trim() != CUDA_OXIDE_REVISION {
        return Err(format!(
            "cuda-oxide source is at {}, expected {}",
            commit.trim(),
            CUDA_OXIDE_REVISION
        )
        .into());
    }

    Ok(())
}

fn gate_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(RESIDUAL_NORM_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let plain = entries
        .iter()
        .filter(|entry| entry.name == "rms_norm_b1" || entry.name.starts_with("rms_norm_TID_"))
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| entry.name.starts_with("residual_rms_norm_TID_"))
        .collect::<Vec<_>>();
    require_count("plain RMSNorm", plain.len(), 8)?;
    require_count("residual RMSNorm", residual.len(), 8)?;

    for entry in plain.iter().chain(&residual) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("residual-norm-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let mut plain_registers = Vec::new();
    let mut residual_registers = Vec::new();
    let mut shared = Vec::new();

    for entry in plain {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted plain entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        plain_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    for entry in residual {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted residual entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        residual_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "residual-norm gate passed: 8 plain + 8 residual entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        plain_registers, residual_registers, shared
    );
    Ok(())
}

fn gate_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_QKV_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name == "quantize_activation_e4m3")
        .collect::<Vec<_>>();
    let qkv = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    let qkv_t16 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_qkv_mma_t16")
        .collect::<Vec<_>>();
    require_count("FP8 activation quantization", quantize.len(), 1)?;
    require_count("FP8 QKV", qkv.len(), 8)?;
    require_count("FP8 QKV T=16", qkv_t16.len(), 1)?;

    for entry in quantize.iter().chain(&qkv) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !qkv_t16[0].body.contains(".reqntid 64, 1, 1")
        || !qkv_t16[0].body.contains(".minnctapersm 4")
    {
        return Err("FP8 QKV T=16 lost its 64-thread/four-CTA launch bounds".into());
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-qkv-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let sass = require_success(&cuobjdump, &[OsStr::new("--dump-sass"), cubin.as_os_str()])?;
    let sass = String::from_utf8(sass.stdout)?;
    let t16_sass =
        sass_function_body(&sass, qkv_t16[0].name).ok_or("cuobjdump omitted FP8 QKV T=16 SASS")?;
    if !t16_sass.contains("QMMA.16832.F32.E4M3.E4M3") {
        return Err("FP8 QKV T=16 lost its native E4M3 tensor-core instruction".into());
    }
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted FP8 activation quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut qkv_registers = Vec::new();
    let mut qkv_shared = Vec::new();
    for entry in qkv {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_registers.push(resource.registers);
        qkv_shared.push(resource.shared);
    }
    qkv_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &qkv_registers)?;
    let qkv_t16_resource = resources
        .get(qkv_t16[0].name)
        .ok_or("cuobjdump omitted FP8 QKV T=16")?;
    require_spill_free(qkv_t16[0].name, qkv_t16_resource)?;
    require_registers(
        &baseline,
        "qkv_t16_registers",
        &[qkv_t16_resource.registers],
    )?;

    println!(
        "FP8 QKV gate passed: 1 quantize + 8 decode + 1 T=16 projection entries, REG {} / {:?} / {}, STACK:0 LOCAL:0, SHARED {} / {:?} / {}",
        quantize_resource.registers,
        qkv_registers,
        qkv_t16_resource.registers,
        quantize_resource.shared,
        qkv_shared,
        qkv_t16_resource.shared,
    );
    Ok(())
}

fn gate_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_GDN_INPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let gdn_input = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_gdn_input_TID_"))
        .collect::<Vec<_>>();
    require_count("FP8 GDN input", gdn_input.len(), 8)?;

    for entry in &gdn_input {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-gdn-input-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in gdn_input {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 GDN input entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "gdn_input_registers", &registers)?;

    println!(
        "FP8 GDN input gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let lm_head = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_lm_head_TID_"))
        .collect::<Vec<_>>();
    require_count("FP8 LM head", lm_head.len(), 8)?;

    for entry in &lm_head {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-lm-head-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in lm_head {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 LM-head entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;

    println!(
        "FP8 LM-head gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_quantize_TID_"))
        .collect::<Vec<_>>();
    let decode = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_decode_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.name,
                "fp8_swiglu_mma_t32" | "fp8_swiglu_mma_t64" | "fp8_swiglu_mma_t128"
            )
        })
        .collect::<Vec<_>>();
    require_count("dense-FP8 SwiGLU quantization", quantize.len(), 1)?;
    require_count("dense-FP8 SwiGLU decode", decode.len(), 8)?;
    require_count("dense-FP8 SwiGLU prefill", prefill.len(), 3)?;

    for entry in quantize.iter().chain(&decode).chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-swiglu-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let sass = require_success(&cuobjdump, &[OsStr::new("--dump-sass"), cubin.as_os_str()])?;
    let sass = String::from_utf8(sass.stdout)?;
    for entry in &prefill {
        let body = sass_function_body(&sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "dense-FP8 SwiGLU entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }

    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 SwiGLU quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut decode_registers = Vec::new();
    for entry in &decode {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        decode_registers.push(resource.registers);
    }
    decode_registers.sort_unstable();
    require_registers(&baseline, "decode_registers", &decode_registers)?;

    let mut prefill_registers = Vec::new();
    for entry in &prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
    }
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prefill_registers", &prefill_registers)?;

    println!(
        "dense-FP8 SwiGLU gate passed: 1 quantize + 8 decode + 3 prefill entries, REG {} / {:?} / {:?}, STACK:0 LOCAL:0",
        quantize_resource.registers, decode_registers, prefill_registers
    );
    Ok(())
}

fn gate_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_DOWN_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_quantize_TID_"))
        .collect::<Vec<_>>();
    let down = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_TID_"))
        .collect::<Vec<_>>();
    require_count("dense-FP8 down quantization", quantize.len(), 1)?;
    require_count("dense-FP8 down projection", down.len(), 8)?;

    for entry in quantize.iter().chain(&down) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-down-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 down quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut down_registers = Vec::new();
    let mut shared = Vec::new();
    for entry in down {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        down_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    down_registers.sort_unstable();
    require_registers(&baseline, "down_registers", &down_registers)?;

    println!(
        "dense-FP8 down gate passed: 1 quantize + 8 projection entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}",
        quantize_resource.registers, down_registers, quantize_resource.shared, shared,
    );
    Ok(())
}

fn verify_generator_stamp(
    root: &Path,
    baseline: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let backend = backend_path(root)?;
    let source = cuda_oxide_source(root, &backend)?;
    let commit = command_text("git", &["-C", path_text(&source)?, "rev-parse", "HEAD"])?;
    let changes = command_text(
        "git",
        &[
            "-C",
            path_text(&source)?,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )?;
    if !changes.trim().is_empty() {
        return Err("cuda-oxide source has tracked changes; restore the pinned checkout".into());
    }
    require_stamp(baseline, "cuda_oxide_commit", commit.trim())?;
    let rustc = require_success(Path::new("rustc"), &[OsStr::new("-vV")])?;
    let (rustc_release, rustc_commit) = parse_rustc_identity(&String::from_utf8(rustc.stdout)?)?;
    require_stamp(baseline, "rustc_release", &rustc_release)?;
    require_stamp(baseline, "rustc_commit", &rustc_commit)?;

    let ptxas = cuda_tool("ptxas");
    let cuobjdump = cuda_tool("cuobjdump");
    let ptxas_identity = cuda_toolkit_identity(&ptxas)?;
    let cuobjdump_identity = cuda_toolkit_identity(&cuobjdump)?;
    if ptxas_identity != cuobjdump_identity {
        return Err(format!(
            "CUDA tools come from different Toolkit versions: {} reports release {} / V{}, while {} reports release {} / V{}",
            ptxas.display(),
            ptxas_identity.release,
            ptxas_identity.version,
            cuobjdump.display(),
            cuobjdump_identity.release,
            cuobjdump_identity.version,
        )
        .into());
    }
    require_stamp(baseline, "cuda_toolkit_release", &ptxas_identity.release)?;
    require_stamp(baseline, "cuda_toolkit_version", &ptxas_identity.version)?;

    let lock = fs::read_to_string(root.join("Cargo.lock"))?;
    let expected_commit = baseline
        .get("cuda_oxide_commit")
        .ok_or("baseline is missing `cuda_oxide_commit`")?;
    if !lock.contains(&format!("rev={expected_commit}")) {
        return Err("Cargo.lock does not contain the stamped cuda-oxide revision".into());
    }

    Ok(())
}

fn backend_path(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_BACKEND") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = local_backend(root) {
        return Ok(path);
    }
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .ok_or("set CUDA_OXIDE_BACKEND or CARGO_HOME")?;

    Ok(cargo_home
        .join("cuda-oxide")
        .join("librustc_codegen_cuda.so"))
}

fn cuda_oxide_source(root: &Path, backend: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_SOURCE") {
        return Ok(PathBuf::from(path));
    }
    for ancestor in backend.ancestors() {
        if ancestor.join(".git").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }
    if let Some(parent) = backend.parent() {
        let cached = parent.join("src");
        if cached.join(".git").exists() {
            return Ok(cached);
        }
    }
    let local = local_cuda_oxide_source(root);
    if local.join(".git").exists() {
        return Ok(local);
    }

    Err("could not locate cuda-oxide source; set CUDA_OXIDE_SOURCE".into())
}

fn local_cuda_oxide_source(root: &Path) -> PathBuf {
    root.join("target/cuda-oxide-source")
}

fn task_cargo_home(root: &Path) -> PathBuf {
    root.join("target/cargo-home")
}

fn encoded_backend_rustflags(root: &Path, source: &Path) -> Result<String, Box<dyn Error>> {
    let sysroot = command_text("rustc", &["--print", "sysroot"])?;
    let cargo_home = task_cargo_home(root);
    let prefixes = [
        (source, "/cuda-oxide"),
        (cargo_home.as_path(), "/cargo-home"),
        (Path::new(sysroot.trim()), "/rust-toolchain"),
    ];
    let flags = prefixes
        .into_iter()
        .map(|(path, replacement)| {
            Ok(format!(
                "--remap-path-prefix={}={replacement}",
                path_text(path)?
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(flags.join("\u{1f}"))
}

fn local_backend(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let rustc = command_text("rustc", &["-vV"])?;
    let host = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or("rustc -vV omitted its host triple")?;
    let path = local_cuda_oxide_source(root)
        .join("crates/rustc-codegen-cuda/target")
        .join(host)
        .join("debug/librustc_codegen_cuda.so");
    if !path.is_file() {
        return Err(format!("cuda-oxide backend does not exist at {}", path.display()).into());
    }

    Ok(path)
}

fn cuda_tool(name: &str) -> PathBuf {
    let home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let candidate = home.join("bin").join(name);
    if candidate.is_file() {
        candidate
    } else {
        PathBuf::from(name)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CudaToolkitIdentity {
    release: String,
    version: String,
}

fn cuda_toolkit_identity(tool: &Path) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let output = require_success(tool, &[OsStr::new("--version")])?;

    parse_cuda_toolkit_identity(&String::from_utf8(output.stdout)?)
}

fn parse_cuda_toolkit_identity(text: &str) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let identity = text
        .lines()
        .find_map(|line| line.strip_prefix("Cuda compilation tools, release "))
        .ok_or("CUDA tool omitted its release identity")?;
    let (release, version) = identity
        .split_once(", V")
        .ok_or("CUDA tool emitted an invalid release identity")?;

    Ok(CudaToolkitIdentity {
        release: release.to_string(),
        version: version.to_string(),
    })
}

fn parse_rustc_identity(text: &str) -> Result<(String, String), Box<dyn Error>> {
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name}: ")))
            .map(str::to_string)
            .ok_or_else(|| format!("rustc -vV omitted `{name}`"))
    };

    Ok((field("release")?, field("commit-hash")?))
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let arguments = arguments.iter().map(OsStr::new).collect::<Vec<_>>();
    let output = require_success(Path::new(program), &arguments)?;

    Ok(String::from_utf8(output.stdout)?)
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

fn require_success(program: &Path, arguments: &[&OsStr]) -> Result<Output, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            program.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(output)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_baseline(text: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut fields = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid baseline line `{line}`"))?;
        fields.insert(key.to_string(), value.to_string());
    }

    Ok(fields)
}

fn require_stamp(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &str,
) -> Result<(), Box<dyn Error>> {
    let expected = baseline
        .get(key)
        .ok_or_else(|| format!("baseline is missing `{key}`"))?;
    if expected != actual {
        return Err(format!(
            "generator stamp `{key}` is `{actual}`, expected `{expected}`; re-baseline separately"
        )
        .into());
    }

    Ok(())
}

struct Entry<'a> {
    name: &'a str,
    body: &'a str,
}

fn parse_entries(ptx: &str) -> Vec<Entry<'_>> {
    let marker = ".visible .entry ";
    let offsets = ptx.match_indices(marker).collect::<Vec<_>>();
    offsets
        .iter()
        .enumerate()
        .filter_map(|(index, (offset, _))| {
            let begin = offset + marker.len();
            let end = offsets
                .get(index + 1)
                .map(|(offset, _)| *offset)
                .unwrap_or(ptx.len());
            let body = &ptx[begin..end];
            let name_end = body.find('(')?;

            Some(Entry {
                name: body[..name_end].trim(),
                body,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Resource {
    registers: u32,
    stack: u32,
    shared: u32,
    local: u32,
}

fn parse_resources(text: &str) -> Result<BTreeMap<String, Resource>, Box<dyn Error>> {
    let mut resources = BTreeMap::new();
    let mut function = None;
    for line in text.lines() {
        if let Some(name) = line
            .trim()
            .strip_prefix("Function ")
            .and_then(|name| name.strip_suffix(':'))
        {
            function = Some(name.to_string());
            continue;
        }
        let Some(name) = function.take() else {
            continue;
        };
        let fields = line
            .split_whitespace()
            .filter_map(|field| field.split_once(':'))
            .collect::<BTreeMap<_, _>>();
        let field = |key: &str| -> Result<u32, Box<dyn Error>> {
            Ok(fields
                .get(key)
                .ok_or_else(|| format!("resource line for `{name}` is missing `{key}`"))?
                .parse()?)
        };
        let resource = Resource {
            registers: field("REG")?,
            stack: field("STACK")?,
            shared: field("SHARED")?,
            local: field("LOCAL")?,
        };
        resources.insert(name, resource);
    }

    Ok(resources)
}

fn require_count(family: &str, actual: usize, expected: usize) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(format!(
            "{family} emitted {actual} entries, expected {expected}; zero entries is a silent generic-instantiation failure"
        )
        .into());
    }

    Ok(())
}

fn require_spill_free(name: &str, resource: &Resource) -> Result<(), Box<dyn Error>> {
    if resource.stack != 0 || resource.local != 0 {
        return Err(format!(
            "entry `{name}` uses STACK:{} LOCAL:{}",
            resource.stack, resource.local
        )
        .into());
    }

    Ok(())
}

fn require_registers(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &[u32],
) -> Result<(), Box<dyn Error>> {
    let actual = actual
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    require_stamp(baseline, key, &actual)
}

fn require_uniform_value(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &[u32],
) -> Result<(), Box<dyn Error>> {
    let Some((&first, remaining)) = actual.split_first() else {
        return Err(format!("resource inventory `{key}` is empty").into());
    };
    if remaining.iter().any(|value| *value != first) {
        return Err(format!("resource inventory `{key}` is not uniform: {actual:?}").into());
    }

    require_stamp(baseline, key, &first.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        PERFORMANCE_SUITES, PerformanceSuite, parse_cuda_toolkit_identity, parse_entries,
        parse_idle_sample, parse_resources, parse_rustc_identity, require_count,
        require_uniform_value, sass_function_body,
    };
    use std::collections::BTreeMap;

    #[test]
    fn parses_hashed_and_concrete_entries() {
        let ptx = ".visible .entry rms_norm_b1()\n.reqntid 512, 1, 1\n\
                   .visible .entry residual_rms_norm_TID_abc()\n.reqntid 512, 1, 1\n";
        let entries = parse_entries(ptx);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "rms_norm_b1");
        assert_eq!(entries[1].name, "residual_rms_norm_TID_abc");
    }

    #[test]
    fn parses_cuobjdump_resource_lines() {
        let resources = parse_resources(
            " Function rms_norm_b1:\n  REG:20 STACK:0 SHARED:1088 LOCAL:0 CONSTANT[0]:920\n",
        )
        .unwrap();
        let resource = resources["rms_norm_b1"];

        assert_eq!(resource.registers, 20);
        assert_eq!(resource.stack, 0);
        assert_eq!(resource.shared, 1_088);
        assert_eq!(resource.local, 0);
    }

    #[test]
    fn zero_generic_entries_fail_loudly() {
        let error = require_count("plain RMSNorm", 0, 8).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("silent generic-instantiation failure")
        );
    }

    #[test]
    fn parses_readable_compiler_identities() {
        let rustc = parse_rustc_identity(
            "rustc 1.96.0-nightly (55e86c996 2026-04-02)\n\
             commit-hash: 55e86c996809902e8bbad512cfb4d2c18be446d9\n\
             release: 1.96.0-nightly\n",
        )
        .unwrap();
        let cuda = parse_cuda_toolkit_identity(
            "Cuda compilation tools, release 13.3, V13.3.73\n\
             Build cuda_13.3.r13.3/compiler.38244171_0\n",
        )
        .unwrap();

        assert_eq!(rustc.0, "1.96.0-nightly");
        assert_eq!(rustc.1, "55e86c996809902e8bbad512cfb4d2c18be446d9");
        assert_eq!(cuda.release, "13.3");
        assert_eq!(cuda.version, "13.3.73");
    }

    #[test]
    fn shared_memory_contract_requires_one_value() {
        let baseline = BTreeMap::from([("shared_bytes".to_string(), "1088".to_string())]);

        require_uniform_value(&baseline, "shared_bytes", &[1_088; 16]).unwrap();
        assert!(require_uniform_value(&baseline, "shared_bytes", &[1_088, 1_024]).is_err());
    }

    #[test]
    fn performance_suite_names_select_the_complete_inventory() {
        let names = PERFORMANCE_SUITES
            .iter()
            .map(|suite| suite.name())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            ["residual-norm", "fp8-qkv", "fp8-gdn-input", "fp8-lm-head"]
        );
        for suite in PERFORMANCE_SUITES {
            assert_eq!(
                PerformanceSuite::parse(suite.name()).unwrap().name(),
                suite.name()
            );
        }
        assert!(PerformanceSuite::parse("unknown").is_err());
    }

    #[test]
    fn parses_idle_device_samples() {
        assert_eq!(parse_idle_sample("0, 234\n").unwrap(), (0, 234));
        assert_eq!(parse_idle_sample("69, 1024").unwrap(), (69, 1_024));
        assert!(parse_idle_sample("0, 234, 1").is_err());
    }

    #[test]
    fn isolates_one_sass_function() {
        let sass = "Function : first\nQMMA.16832.F32.E4M3.E4M3 R0, R1, R2, R3;\n\
                    \t\tFunction : second\nNOP;\n";

        assert!(
            sass_function_body(sass, "first")
                .unwrap()
                .contains("QMMA.16832.F32.E4M3.E4M3")
        );
        assert!(!sass_function_body(sass, "second").unwrap().contains("QMMA"));
        assert!(sass_function_body(sass, "missing").is_none());
    }
}
