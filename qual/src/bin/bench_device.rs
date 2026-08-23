//! Device benchmark entry point for the selected architecture artifact.

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(any(feature = "device", feature = "sm89"))]
use tuisko_qual::benchmark_fp8_qkv;
#[cfg(any(feature = "device", feature = "sm89"))]
use tuisko_qual::benchmark_nvfp4_down;
#[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
use tuisko_qual::benchmark_nvfp4_swiglu;
use tuisko_qual::{DeviceBenchmarkOptions, DeviceBenchmarkReport, benchmark_residual_norm};
#[cfg(feature = "device")]
use tuisko_qual::{
    benchmark_attention_output, benchmark_attention_qk_prepare, benchmark_dense_fp8_gdn_layer,
    benchmark_dense_fp8_mlp, benchmark_fp8_down, benchmark_fp8_gdn_input, benchmark_fp8_lm_head,
    benchmark_fp8_swiglu, benchmark_full_attention_layer, benchmark_gdn_output,
    benchmark_gdn_prepare, benchmark_gdn_recurrence, benchmark_long_context_paged_gqa,
    benchmark_nvfp4_mlp, benchmark_paged_gqa, benchmark_qwen35_nvfp4_down,
    benchmark_qwen35_nvfp4_mlp, benchmark_qwen35_nvfp4_swiglu, benchmark_qwen35_residual_norm,
    benchmark_resident_long_context_model, benchmark_resident_model, benchmark_text_endpoint,
    profile_resident_model,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!();
            eprintln!("============================================================");
            eprintln!("TUISKO BENCHMARK REFUSED");
            eprintln!("============================================================");
            eprintln!("{error}");
            eprintln!("============================================================");

            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let suite = arguments
        .next()
        .ok_or("usage: bench-device <attention-qk-prepare|paged-gqa|long-context-paged-gqa|attention-output|residual-norm|qwen35-residual-norm|qwen35-nvfp4-swiglu|qwen35-nvfp4-down|qwen35-nvfp4-mlp|fp8-qkv|fp8-gdn-input|fp8-lm-head|fp8-swiglu|fp8-down|nvfp4-swiglu|nvfp4-down|nvfp4-mlp|gdn-prepare|gdn-recurrence|gdn-output|dense-fp8-mlp|dense-fp8-gdn-layer|full-attention-layer|resident-model|resident-long-context-model|text-endpoint|profile-resident-model> [SNAPSHOT] [options]")?;
    #[cfg(feature = "device")]
    if suite == "profile-resident-model" {
        return run_resident_profile(arguments);
    }
    let report = match suite.as_str() {
        #[cfg(feature = "device")]
        "attention-qk-prepare" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_attention_qk_prepare(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "paged-gqa" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (benchmark_paged_gqa(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "long-context-paged-gqa" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (benchmark_long_context_paged_gqa(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "attention-output" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_attention_output(options)?, json_path)
        }
        "residual-norm" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_residual_norm(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "qwen35-residual-norm" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_qwen35_residual_norm(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "qwen35-nvfp4-swiglu" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_qwen35_nvfp4_swiglu(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "qwen35-nvfp4-down" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_qwen35_nvfp4_down(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "qwen35-nvfp4-mlp" => {
            let snapshot = arguments
                .next()
                .ok_or("qwen35-nvfp4-mlp requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (
                benchmark_qwen35_nvfp4_mlp(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
        "nvfp4-swiglu" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_nvfp4_swiglu(options)?, json_path)
        }
        #[cfg(any(feature = "device", feature = "sm89"))]
        "nvfp4-down" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_nvfp4_down(options)?, json_path)
        }
        #[cfg(any(feature = "device", feature = "sm89"))]
        "fp8-qkv" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_fp8_qkv(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "fp8-gdn-input" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_fp8_gdn_input(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "fp8-lm-head" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (benchmark_fp8_lm_head(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "fp8-swiglu" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_fp8_swiglu(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "fp8-down" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_fp8_down(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "nvfp4-mlp" => {
            let snapshot = arguments
                .next()
                .ok_or("nvfp4-mlp requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (
                benchmark_nvfp4_mlp(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "gdn-prepare" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_gdn_prepare(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "gdn-recurrence" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_gdn_recurrence(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "gdn-output" => {
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::short_graph())?;
            (benchmark_gdn_output(options)?, json_path)
        }
        #[cfg(feature = "device")]
        "dense-fp8-mlp" => {
            let snapshot = arguments
                .next()
                .ok_or("dense-fp8-mlp requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (
                benchmark_dense_fp8_mlp(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "dense-fp8-gdn-layer" => {
            let snapshot = arguments
                .next()
                .ok_or("dense-fp8-gdn-layer requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (
                benchmark_dense_fp8_gdn_layer(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "full-attention-layer" => {
            let snapshot = arguments
                .next()
                .ok_or("full-attention-layer requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (
                benchmark_full_attention_layer(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "resident-model" => {
            let snapshot = arguments
                .next()
                .ok_or("resident-model requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::resident_model())?;
            (
                benchmark_resident_model(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "resident-long-context-model" => {
            let snapshot = arguments
                .next()
                .ok_or("resident-long-context-model requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::resident_model())?;
            (
                benchmark_resident_long_context_model(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        #[cfg(feature = "device")]
        "text-endpoint" => {
            let snapshot = arguments
                .next()
                .ok_or("text-endpoint requires the admitted snapshot path")?;
            let (options, json_path) =
                parse_options(arguments, DeviceBenchmarkOptions::long_graph())?;
            (
                benchmark_text_endpoint(&PathBuf::from(snapshot), options)?,
                json_path,
            )
        }
        _ => return Err(format!("unknown benchmark suite `{suite}`").into()),
    };
    print_report(&report.0);
    write_report(&report.0, report.1)?;
    if report.0.clock_policy == "diagnostic_uncontrolled"
        && std::env::var("TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT").as_deref() != Ok("1")
    {
        return Err(format!(
            "timings were preserved as diagnostic_uncontrolled because clocks drifted during the full measurement: SM {}..{} MHz, memory {}..{} MHz; the report cannot be blessed",
            report.0.sm_clock_min_mhz,
            report.0.sm_clock_max_mhz,
            report.0.memory_clock_min_mhz,
            report.0.memory_clock_max_mhz,
        )
        .into());
    }

    Ok(())
}

#[cfg(feature = "device")]
fn run_resident_profile(mut arguments: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let snapshot = PathBuf::from(
        arguments
            .next()
            .ok_or("profile-resident-model requires the admitted snapshot path")?,
    );
    let mut batch = 1usize;
    let mut warmup_launches = 16u64;
    let mut captured_replays = 3u64;
    let mut graph_dot = None;
    let mut manifest_path = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("`{argument}` requires a value"))?;
        match argument.as_str() {
            "--batch" => batch = value.parse()?,
            "--warmup-launches" => warmup_launches = value.parse()?,
            "--captured-replays" => captured_replays = value.parse()?,
            "--graph-dot" => graph_dot = Some(PathBuf::from(value)),
            "--manifest" => manifest_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown resident profile argument `{argument}`").into()),
        }
    }
    let graph_dot = graph_dot.ok_or("profile-resident-model requires --graph-dot PATH")?;
    let manifest_path = manifest_path.ok_or("profile-resident-model requires --manifest PATH")?;
    for path in [&graph_dot, &manifest_path] {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
    }
    let manifest = profile_resident_model(
        &snapshot,
        batch,
        warmup_launches,
        captured_replays,
        &graph_dot,
    )?;
    let mut json = serde_json::to_vec_pretty(&manifest)?;
    json.push(b'\n');
    std::fs::write(&manifest_path, json)?;
    eprintln!("CUDA Graph DOT: {}", graph_dot.display());
    eprintln!("semantic manifest: {}", manifest_path.display());

    Ok(())
}

fn print_report(report: &DeviceBenchmarkReport) {
    eprintln!(
        "route                            shape metric              median us    p10 us    p90 us    GiB/s"
    );
    for metric in &report.metrics {
        let throughput = metric
            .logical_gib_per_second
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        eprintln!(
            "{:<32} {:<5} {:<19} {:>9.3} {:>9.3} {:>9.3} {:>8}",
            metric.route,
            metric.shape,
            metric.measurement.as_str(),
            metric.median_microseconds,
            metric.p10_microseconds,
            metric.p90_microseconds,
            throughput,
        );
    }
    if !report.energy_metrics.is_empty() {
        eprintln!();
        eprintln!(
            "route                            shape board W idle W  dyn W   J/unit  dyn J/unit units/J unit"
        );
        for metric in &report.energy_metrics {
            eprintln!(
                "{:<32} {:<5} {:>7.1} {:>6.1} {:>6.1} {:>8.6} {:>11.6} {:>7.0} {}",
                metric.route,
                metric.shape,
                metric.average_board_watts,
                metric.idle_board_watts,
                metric.dynamic_board_watts,
                metric.estimated_board_joules_per_unit,
                metric.estimated_dynamic_joules_per_unit,
                metric.estimated_units_per_board_joule,
                metric.unit,
            );
        }
    }
    eprintln!();
    eprintln!("memory                                      measurement                    MiB");
    for metric in &report.memory.metrics {
        eprintln!(
            "{:<43} {:<27} {:>9.2}",
            metric.name,
            metric.measurement.as_str(),
            metric.bytes as f64 / (1024.0 * 1024.0),
        );
    }
}

fn write_report(
    report: &DeviceBenchmarkReport,
    json_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = json_path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, json)?;
        eprintln!("JSON: {}", path.display());
    } else {
        std::io::stdout().write_all(&json)?;
    }

    Ok(())
}

fn parse_options(
    mut arguments: impl Iterator<Item = String>,
    mut options: DeviceBenchmarkOptions,
) -> Result<(DeviceBenchmarkOptions, Option<PathBuf>), Box<dyn Error>> {
    let mut json_path = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("`{argument}` requires a value"))?;
        match argument.as_str() {
            "--samples" => options.samples = value.parse()?,
            "--launches-per-sample" => options.launches_per_sample = value.parse()?,
            "--warmup-launches" => options.warmup_launches = value.parse()?,
            "--batch" => options.batch_size = Some(value.parse()?),
            "--energy-seconds" => options.energy_seconds = Some(value.parse()?),
            "--json" => json_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument `{argument}`").into()),
        }
    }
    if options.samples < 3 {
        return Err("`--samples` must be at least 3".into());
    }
    if options.launches_per_sample == 0 {
        return Err("`--launches-per-sample` must be nonzero".into());
    }
    if options.warmup_launches == 0 {
        return Err("`--warmup-launches` must be nonzero".into());
    }
    if options
        .batch_size
        .is_some_and(|batch| !(1..=8).contains(&batch))
    {
        return Err("`--batch` must be in 1..=8".into());
    }
    if options
        .energy_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds < 2.0)
    {
        return Err("`--energy-seconds` must be finite and at least 2".into());
    }

    Ok((options, json_path))
}
