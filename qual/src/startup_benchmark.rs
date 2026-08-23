use crate::DeviceBenchmarkError;
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, EXPECTED_DEVICE_NAME};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tuisko_engine::ResidentModelProgram;
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_SAMPLES: usize = 3;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_REPORT: &str = "target/benchmarks/startup/legacy-sm120.json";

#[derive(Debug)]
struct StartupOptions {
    snapshot: PathBuf,
    samples: usize,
    warmups: usize,
    json_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StartupSample {
    schema_version: u32,
    loader: String,
    filesystem_cache: String,
    checkpoint_revision: String,
    device_name: String,
    compute_capability: String,
    checkpoint_admission_ms: f64,
    cuda_initialization_ms: f64,
    resident_program_ms: f64,
    checkpoint_to_ready_ms: f64,
    tensor_count: usize,
    resident_weight_bytes: usize,
    resident_arena_bytes: usize,
    kv_arena_bytes: usize,
    total_device_arena_bytes: usize,
    peak_rss_bytes: u64,
}

#[derive(Debug, Serialize)]
struct StartupTimingSummary {
    phase: &'static str,
    minimum_ms: f64,
    median_ms: f64,
    maximum_ms: f64,
}

#[derive(Debug, Serialize)]
struct StartupBenchmarkReport {
    schema_version: u32,
    loader: &'static str,
    filesystem_cache: &'static str,
    warmups: usize,
    samples: Vec<StartupSample>,
    timings: Vec<StartupTimingSummary>,
}

/// Runs the release startup benchmark CLI or one hidden fresh-process child sample.
pub fn run_startup_benchmark_cli() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if matches!(
        arguments.first().and_then(|value| value.to_str()),
        Some("--child")
    ) {
        return run_child(&arguments[1..]);
    }

    let options = parse_options(&arguments)?;
    let executable = std::env::current_exe()?;
    for index in 0..options.warmups {
        eprintln!(
            "warming checkpoint pages in fresh process {}/{}",
            index + 1,
            options.warmups
        );
        let _ = launch_child(&executable, &options.snapshot)?;
    }

    let mut samples = Vec::with_capacity(options.samples);
    for index in 0..options.samples {
        eprintln!(
            "measuring legacy startup in fresh process {}/{}",
            index + 1,
            options.samples
        );
        samples.push(launch_child(&executable, &options.snapshot)?);
    }

    let report = StartupBenchmarkReport::new(options.warmups, samples)?;
    print_report(&report);
    write_report(&options.json_path, &report)?;
    eprintln!("startup report: {}", options.json_path.display());
    Ok(())
}

impl StartupBenchmarkReport {
    fn new(warmups: usize, samples: Vec<StartupSample>) -> Result<Self, DeviceBenchmarkError> {
        if samples.is_empty() {
            return Err(DeviceBenchmarkError::Precondition(
                "startup benchmark requires at least one sample".into(),
            ));
        }
        validate_samples(&samples)?;
        let timings = vec![
            summarize(
                "checkpoint_admission",
                samples.iter().map(|sample| sample.checkpoint_admission_ms),
            ),
            summarize(
                "cuda_initialization",
                samples.iter().map(|sample| sample.cuda_initialization_ms),
            ),
            summarize(
                "resident_program",
                samples.iter().map(|sample| sample.resident_program_ms),
            ),
            summarize(
                "checkpoint_to_ready",
                samples.iter().map(|sample| sample.checkpoint_to_ready_ms),
            ),
        ];
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            loader: "legacy",
            filesystem_cache: "warm",
            warmups,
            samples,
            timings,
        })
    }
}

fn parse_options(arguments: &[OsString]) -> Result<StartupOptions, Box<dyn Error>> {
    let Some(snapshot) = arguments.first() else {
        return Err("usage: bench-startup SNAPSHOT [--samples N] [--warmups N] [--filesystem-cache warm] [--loaders legacy] [--json PATH]".into());
    };
    if snapshot
        .to_str()
        .is_some_and(|value| value.starts_with('-'))
    {
        return Err("bench-startup requires the admitted snapshot path first".into());
    }

    let mut samples = DEFAULT_SAMPLES;
    let mut warmups = DEFAULT_WARMUPS;
    let mut json_path = PathBuf::from(DEFAULT_REPORT);
    let mut index = 1;
    while index < arguments.len() {
        let option = arguments[index]
            .to_str()
            .ok_or("startup benchmark arguments must be valid UTF-8")?;
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("startup benchmark option `{option}` requires a value"))?;
        match option {
            "--samples" => samples = parse_count(option, value)?,
            "--warmups" => warmups = parse_count(option, value)?,
            "--json" => json_path = PathBuf::from(value),
            "--filesystem-cache" if value == "warm" => {}
            "--filesystem-cache" => {
                return Err(
                    "only `--filesystem-cache warm` is reproducible in the initial benchmark"
                        .into(),
                );
            }
            "--loaders" if value == "legacy" => {}
            "--loaders" => {
                return Err("only the unchanged `legacy` loader exists in this slice".into());
            }
            _ => return Err(format!("unknown startup benchmark option `{option}`").into()),
        }
        index += 2;
    }

    Ok(StartupOptions {
        snapshot: PathBuf::from(snapshot),
        samples,
        warmups,
        json_path,
    })
}

fn parse_count(option: &str, value: &OsStr) -> Result<usize, Box<dyn Error>> {
    let parsed = value
        .to_str()
        .ok_or_else(|| format!("{option} must be valid UTF-8"))?
        .parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{option} must be greater than zero").into());
    }
    Ok(parsed)
}

fn run_child(arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("internal startup child requires exactly one snapshot path".into());
    };
    let sample = measure_legacy_startup(Path::new(snapshot))?;
    serde_json::to_writer(io::stdout().lock(), &sample)?;
    println!();
    Ok(())
}

fn launch_child(executable: &Path, snapshot: &Path) -> Result<StartupSample, Box<dyn Error>> {
    let output = Command::new(executable)
        .arg("--child")
        .arg(snapshot)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        io::stderr().write_all(&output.stderr)?;
        return Err(format!("startup child failed with {}", output.status).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn measure_legacy_startup(snapshot_path: &Path) -> Result<StartupSample, DeviceBenchmarkError> {
    let ready_start = Instant::now();

    let checkpoint_start = Instant::now();
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(snapshot_path)?;
    let checkpoint_admission = checkpoint_start.elapsed();
    let tensor_count = snapshot.tensor_count();
    let checkpoint_revision = snapshot
        .root()
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition(
                "admitted snapshot root omitted its UTF-8 revision directory".into(),
            )
        })?
        .to_owned();

    let cuda_start = Instant::now();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let device_name = context.device_name().map_err(GpuError::from)?;
    let compute_capability = context.compute_capability().map_err(GpuError::from)?;
    if device_name != EXPECTED_DEVICE_NAME || compute_capability != EXPECTED_COMPUTE_CAPABILITY {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "device zero is {device_name} with compute capability {}.{}, expected {EXPECTED_DEVICE_NAME} with compute capability {}.{}",
            compute_capability.0,
            compute_capability.1,
            EXPECTED_COMPUTE_CAPABILITY.0,
            EXPECTED_COMPUTE_CAPABILITY.1,
        )));
    }
    let cuda_initialization = cuda_start.elapsed();

    let program_start = Instant::now();
    let program = ResidentModelProgram::from_snapshot(&context, snapshot.into())?;
    context.synchronize().map_err(GpuError::from)?;
    let resident_program = program_start.elapsed();
    let checkpoint_to_ready = ready_start.elapsed();
    let peak_rss_bytes = process_peak_rss_bytes()?;

    Ok(StartupSample {
        schema_version: SCHEMA_VERSION,
        loader: "legacy".into(),
        filesystem_cache: "warm".into(),
        checkpoint_revision,
        device_name,
        compute_capability: format!("{}.{}", compute_capability.0, compute_capability.1),
        checkpoint_admission_ms: milliseconds(checkpoint_admission),
        cuda_initialization_ms: milliseconds(cuda_initialization),
        resident_program_ms: milliseconds(resident_program),
        checkpoint_to_ready_ms: milliseconds(checkpoint_to_ready),
        tensor_count,
        resident_weight_bytes: program.resident_weight_bytes(),
        resident_arena_bytes: program.resident_arena_bytes(),
        kv_arena_bytes: program.kv_arena_bytes(),
        total_device_arena_bytes: program.arena_bytes(),
        peak_rss_bytes,
    })
}

fn validate_samples(samples: &[StartupSample]) -> Result<(), DeviceBenchmarkError> {
    let first = &samples[0];
    for sample in samples {
        let timings = [
            sample.checkpoint_admission_ms,
            sample.cuda_initialization_ms,
            sample.resident_program_ms,
            sample.checkpoint_to_ready_ms,
        ];
        if timings
            .into_iter()
            .any(|milliseconds| !milliseconds.is_finite() || milliseconds <= 0.0)
        {
            return Err(DeviceBenchmarkError::Precondition(
                "startup timings must all be finite and greater than zero".into(),
            ));
        }
        if sample.schema_version != SCHEMA_VERSION
            || sample.loader != first.loader
            || sample.filesystem_cache != first.filesystem_cache
            || sample.checkpoint_revision != first.checkpoint_revision
            || sample.device_name != first.device_name
            || sample.compute_capability != first.compute_capability
            || sample.tensor_count != first.tensor_count
            || sample.resident_weight_bytes != first.resident_weight_bytes
            || sample.resident_arena_bytes != first.resident_arena_bytes
            || sample.kv_arena_bytes != first.kv_arena_bytes
            || sample.total_device_arena_bytes != first.total_device_arena_bytes
        {
            return Err(DeviceBenchmarkError::Precondition(
                "fresh-process startup samples disagree on their exact product identity or byte accounting"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn summarize(phase: &'static str, values: impl Iterator<Item = f64>) -> StartupTimingSummary {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    StartupTimingSummary {
        phase,
        minimum_ms: values[0],
        median_ms: percentile(&values, 0.5),
        maximum_ms: values[values.len() - 1],
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn process_peak_rss_bytes() -> io::Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    parse_peak_rss_bytes(&status)
}

fn parse_peak_rss_bytes(status: &str) -> io::Result<u64> {
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or_else(|| io::Error::other("/proc/self/status omitted VmHWM"))?;
    let mut fields = line.split_whitespace();
    let _name = fields.next();
    let kibibytes = fields
        .next()
        .ok_or_else(|| io::Error::other("VmHWM omitted its value"))?
        .parse::<u64>()
        .map_err(io::Error::other)?;
    if fields.next() != Some("kB") || fields.next().is_some() {
        return Err(io::Error::other("VmHWM did not use Linux kB units"));
    }
    kibibytes
        .checked_mul(1_024)
        .ok_or_else(|| io::Error::other("VmHWM byte count overflowed u64"))
}

fn print_report(report: &StartupBenchmarkReport) {
    eprintln!();
    eprintln!("legacy startup, warm filesystem cache");
    eprintln!("phase                         minimum ms   median ms   maximum ms");
    for timing in &report.timings {
        eprintln!(
            "{:<29} {:>10.3} {:>11.3} {:>12.3}",
            timing.phase, timing.minimum_ms, timing.median_ms, timing.maximum_ms,
        );
    }
    let sample = &report.samples[0];
    eprintln!();
    eprintln!(
        "{} ({}) · {} tensors · {:.2} GiB weights · {:.2} GiB device arenas",
        sample.device_name,
        sample.compute_capability,
        sample.tensor_count,
        sample.resident_weight_bytes as f64 / (1_u64 << 30) as f64,
        sample.total_device_arena_bytes as f64 / (1_u64 << 30) as f64,
    );
    let peak_rss = report
        .samples
        .iter()
        .map(|sample| sample.peak_rss_bytes)
        .max()
        .unwrap_or(0);
    eprintln!(
        "peak child RSS: {:.2} GiB",
        peak_rss as f64 / (1_u64 << 30) as f64
    );
}

fn write_report(path: &Path, report: &StartupBenchmarkReport) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut output, report)?;
    writeln!(output)
}

#[cfg(test)]
mod tests {
    use super::{parse_options, parse_peak_rss_bytes};
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn parses_linux_peak_rss_in_bytes() {
        assert_eq!(
            parse_peak_rss_bytes("Name:\ttest\nVmHWM:\t12345 kB\n").unwrap(),
            12_641_280
        );
    }

    #[test]
    fn startup_options_pin_the_reproducible_cache_regime() {
        let arguments = [
            OsString::from("/snapshot"),
            OsString::from("--samples"),
            OsString::from("5"),
            OsString::from("--warmups"),
            OsString::from("2"),
            OsString::from("--filesystem-cache"),
            OsString::from("warm"),
            OsString::from("--loaders"),
            OsString::from("legacy"),
            OsString::from("--json"),
            OsString::from("report.json"),
        ];
        let options = parse_options(&arguments).unwrap();
        assert_eq!(options.snapshot, PathBuf::from("/snapshot"));
        assert_eq!(options.samples, 5);
        assert_eq!(options.warmups, 2);
        assert_eq!(options.json_path, PathBuf::from("report.json"));
    }
}
