//! SM120 device benchmark entry point.

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use tuisko_qual::{
    DeviceBenchmarkOptions, DeviceBenchmarkReport, benchmark_fp8_qkv, benchmark_residual_norm,
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
        .ok_or("usage: bench-device <residual-norm|fp8-qkv> [options]")?;
    let (options, json_path) = parse_options(arguments)?;
    let report = match suite.as_str() {
        "residual-norm" => benchmark_residual_norm(options)?,
        "fp8-qkv" => benchmark_fp8_qkv(options)?,
        _ => return Err(format!("unknown benchmark suite `{suite}`").into()),
    };
    print_report(&report);
    write_report(&report, json_path)
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
) -> Result<(DeviceBenchmarkOptions, Option<PathBuf>), Box<dyn Error>> {
    let mut options = DeviceBenchmarkOptions::default();
    let mut json_path = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("`{argument}` requires a value"))?;
        match argument.as_str() {
            "--samples" => options.samples = value.parse()?,
            "--launches-per-sample" => options.launches_per_sample = value.parse()?,
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
    if options
        .energy_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds < 2.0)
    {
        return Err("`--energy-seconds` must be finite and at least 2".into());
    }

    Ok((options, json_path))
}
