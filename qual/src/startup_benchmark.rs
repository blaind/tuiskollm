use crate::DeviceBenchmarkError;
use crate::startup_h2d::{
    H2dCalibrationReport, measure_h2d_calibration, pinned_h2d_gib_s, print_h2d_calibration,
};
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, EXPECTED_DEVICE_NAME};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tuisko_engine::{ResidentLoadMode, ResidentModelProgram};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const SCHEMA_VERSION: u32 = 5;
const DEFAULT_SAMPLES: usize = 3;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_REPORT: &str = "target/benchmarks/startup/loader-comparison-sm120.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StartupLoader {
    Legacy,
    Selective,
}

impl StartupLoader {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Selective => "selective",
        }
    }

    const fn resident_mode(self) -> ResidentLoadMode {
        match self {
            Self::Legacy => ResidentLoadMode::Legacy,
            Self::Selective => ResidentLoadMode::Selective,
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "selective" => Ok(Self::Selective),
            _ => Err(format!("unknown startup loader `{value}`").into()),
        }
    }
}

#[derive(Debug)]
struct StartupOptions {
    snapshot: PathBuf,
    samples: usize,
    warmups: usize,
    loaders: Vec<StartupLoader>,
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
    layout_plan_ms: f64,
    arena_allocation_ms: f64,
    operator_setup_ms: f64,
    weight_prepare_ms: f64,
    source_binding_ms: f64,
    qkv_gather_ms: f64,
    nvfp4_materialize_ms: f64,
    preparation_other_ms: f64,
    weight_copy_ms: f64,
    weight_load_ms: f64,
    nonweight_init_ms: f64,
    graph_capture_ms: f64,
    resident_other_ms: f64,
    resident_program_ms: f64,
    checkpoint_to_ready_ms: f64,
    tensor_count: usize,
    resident_weight_bytes: usize,
    resident_arena_bytes: usize,
    kv_arena_bytes: usize,
    total_device_arena_bytes: usize,
    peak_rss_bytes: u64,
    upload_bytes: usize,
    upload_submissions: usize,
    zeroed_bytes: usize,
    pinned_stager_bytes: usize,
    borrowed_source_bytes: usize,
    gathered_source_bytes: usize,
    swizzled_source_bytes: usize,
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
    filesystem_cache: &'static str,
    warmups: usize,
    h2d_calibration: H2dCalibrationReport,
    loaders: Vec<StartupLoaderReport>,
}

#[derive(Debug, Serialize)]
struct StartupLoaderReport {
    loader: &'static str,
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
    if matches!(
        arguments.first().and_then(|value| value.to_str()),
        Some("--h2d-child")
    ) {
        let report = measure_h2d_calibration()?;
        serde_json::to_writer(io::stdout().lock(), &report)?;
        println!();
        return Ok(());
    }

    let options = parse_options(&arguments)?;
    let executable = std::env::current_exe()?;
    eprintln!("calibrating contiguous pageable and pinned host-to-device transfers");
    let h2d_calibration = launch_h2d_child(&executable)?;
    for loader in &options.loaders {
        for index in 0..options.warmups {
            eprintln!(
                "warming {} startup in fresh process {}/{}",
                loader.as_str(),
                index + 1,
                options.warmups
            );
            let _ = launch_child(&executable, &options.snapshot, *loader)?;
        }
    }

    let mut samples = options
        .loaders
        .iter()
        .map(|loader| (*loader, Vec::with_capacity(options.samples)))
        .collect::<Vec<_>>();
    for index in 0..options.samples {
        let order = if index.is_multiple_of(2) {
            options.loaders.clone()
        } else {
            options.loaders.iter().rev().copied().collect()
        };
        for loader in order {
            eprintln!(
                "measuring {} startup in fresh process {}/{}",
                loader.as_str(),
                index + 1,
                options.samples
            );
            let sample = launch_child(&executable, &options.snapshot, loader)?;
            samples
                .iter_mut()
                .find(|(candidate, _)| *candidate == loader)
                .expect("startup loader sample inventory is complete")
                .1
                .push(sample);
        }
    }

    let loaders = samples
        .into_iter()
        .map(|(loader, samples)| StartupLoaderReport::new(loader, samples))
        .collect::<Result<Vec<_>, _>>()?;
    let report = StartupBenchmarkReport {
        schema_version: SCHEMA_VERSION,
        filesystem_cache: "warm",
        warmups: options.warmups,
        h2d_calibration,
        loaders,
    };
    print_report(&report);
    write_report(&options.json_path, &report)?;
    eprintln!("startup report: {}", options.json_path.display());
    Ok(())
}

impl StartupLoaderReport {
    fn new(
        loader: StartupLoader,
        samples: Vec<StartupSample>,
    ) -> Result<Self, DeviceBenchmarkError> {
        if samples.is_empty() {
            return Err(DeviceBenchmarkError::Precondition(
                "startup benchmark requires at least one sample".into(),
            ));
        }
        if samples
            .iter()
            .any(|sample| sample.loader != loader.as_str())
        {
            return Err(DeviceBenchmarkError::Precondition(
                "startup sample was assigned to the wrong loader report".into(),
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
                "layout_and_upload_plan",
                samples.iter().map(|sample| sample.layout_plan_ms),
            ),
            summarize(
                "arena_allocation",
                samples.iter().map(|sample| sample.arena_allocation_ms),
            ),
            summarize(
                "operator_setup",
                samples.iter().map(|sample| sample.operator_setup_ms),
            ),
            summarize(
                "weight_host_preparation",
                samples.iter().map(|sample| sample.weight_prepare_ms),
            ),
            summarize(
                "source_binding_validation",
                samples.iter().map(|sample| sample.source_binding_ms),
            ),
            summarize(
                "qkv_gather",
                samples.iter().map(|sample| sample.qkv_gather_ms),
            ),
            summarize(
                "nvfp4_scale_swizzle",
                samples.iter().map(|sample| sample.nvfp4_materialize_ms),
            ),
            summarize(
                "preparation_other",
                samples.iter().map(|sample| sample.preparation_other_ms),
            ),
            summarize(
                "weight_cuda_copy_calls",
                samples.iter().map(|sample| sample.weight_copy_ms),
            ),
            summarize(
                "weight_materialize_upload",
                samples.iter().map(|sample| sample.weight_load_ms),
            ),
            summarize(
                "nonweight_initialization",
                samples.iter().map(|sample| sample.nonweight_init_ms),
            ),
            summarize(
                "pointer_bind_graph_capture",
                samples.iter().map(|sample| sample.graph_capture_ms),
            ),
            summarize(
                "resident_unattributed",
                samples.iter().map(|sample| sample.resident_other_ms),
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
            loader: loader.as_str(),
            samples,
            timings,
        })
    }
}

fn parse_options(arguments: &[OsString]) -> Result<StartupOptions, Box<dyn Error>> {
    let Some(snapshot) = arguments.first() else {
        return Err("usage: bench-startup SNAPSHOT [--samples N] [--warmups N] [--filesystem-cache warm] [--loaders legacy,selective] [--json PATH]".into());
    };
    if snapshot
        .to_str()
        .is_some_and(|value| value.starts_with('-'))
    {
        return Err("bench-startup requires the admitted snapshot path first".into());
    }

    let mut samples = DEFAULT_SAMPLES;
    let mut warmups = DEFAULT_WARMUPS;
    let mut loaders = vec![StartupLoader::Legacy, StartupLoader::Selective];
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
            "--loaders" => {
                let value = value.to_str().ok_or("--loaders must be valid UTF-8")?;
                loaders.clear();
                for value in value.split(',') {
                    let loader = StartupLoader::parse(value)?;
                    if loaders.contains(&loader) {
                        return Err(format!("startup loader `{value}` is duplicated").into());
                    }
                    loaders.push(loader);
                }
                if loaders.is_empty() {
                    return Err("--loaders requires at least one loader".into());
                }
            }
            _ => return Err(format!("unknown startup benchmark option `{option}`").into()),
        }
        index += 2;
    }

    Ok(StartupOptions {
        snapshot: PathBuf::from(snapshot),
        samples,
        warmups,
        loaders,
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
    let [loader, snapshot] = arguments else {
        return Err("internal startup child requires one loader and one snapshot path".into());
    };
    let loader = StartupLoader::parse(
        loader
            .to_str()
            .ok_or("internal startup loader must be valid UTF-8")?,
    )?;
    let sample = measure_startup(Path::new(snapshot), loader)?;
    serde_json::to_writer(io::stdout().lock(), &sample)?;
    println!();
    Ok(())
}

fn launch_child(
    executable: &Path,
    snapshot: &Path,
    loader: StartupLoader,
) -> Result<StartupSample, Box<dyn Error>> {
    let output = Command::new(executable)
        .arg("--child")
        .arg(loader.as_str())
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

fn launch_h2d_child(executable: &Path) -> Result<H2dCalibrationReport, Box<dyn Error>> {
    let output = Command::new(executable)
        .arg("--h2d-child")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        io::stderr().write_all(&output.stderr)?;
        return Err(format!("H2D calibration child failed with {}", output.status).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn measure_startup(
    snapshot_path: &Path,
    loader: StartupLoader,
) -> Result<StartupSample, DeviceBenchmarkError> {
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
    let snapshot = snapshot.into();
    let mode = loader.resident_mode();
    let program = match mode {
        ResidentLoadMode::Legacy => ResidentModelProgram::from_snapshot(&context, snapshot)?,
        ResidentLoadMode::Selective => {
            ResidentModelProgram::from_snapshot_selective(&context, snapshot)?
        }
    };
    context.synchronize().map_err(GpuError::from)?;
    let resident_program = program_start.elapsed();
    let checkpoint_to_ready = ready_start.elapsed();
    let peak_rss_bytes = process_peak_rss_bytes()?;
    if program.load_stats().mode() != mode {
        return Err(DeviceBenchmarkError::Precondition(
            "resident program reported the wrong startup loader".into(),
        ));
    }
    let load_stats = program.load_stats();
    let resident_program_ms = milliseconds(resident_program);
    let layout_plan_ms = nanoseconds_to_milliseconds(load_stats.layout_plan_ns());
    let arena_allocation_ms = nanoseconds_to_milliseconds(load_stats.arena_allocation_ns());
    let operator_setup_ms = nanoseconds_to_milliseconds(load_stats.operator_setup_ns());
    let weight_prepare_ms = nanoseconds_to_milliseconds(load_stats.weight_prepare_ns());
    let source_binding_ms = nanoseconds_to_milliseconds(load_stats.source_binding_ns());
    let qkv_gather_ms = nanoseconds_to_milliseconds(load_stats.qkv_gather_ns());
    let nvfp4_materialize_ms = nanoseconds_to_milliseconds(load_stats.nvfp4_materialize_ns());
    let preparation_other_ms = nanoseconds_to_milliseconds(load_stats.preparation_other_ns());
    let weight_copy_ms = nanoseconds_to_milliseconds(load_stats.weight_copy_ns());
    let weight_load_ms = nanoseconds_to_milliseconds(load_stats.weight_load_ns());
    let nonweight_init_ms = nanoseconds_to_milliseconds(load_stats.nonweight_init_ns());
    let graph_capture_ms = nanoseconds_to_milliseconds(load_stats.graph_capture_ns());
    let detailed_ms = layout_plan_ms
        + arena_allocation_ms
        + operator_setup_ms
        + weight_load_ms
        + nonweight_init_ms
        + graph_capture_ms;
    let resident_other_ms = resident_program_ms - detailed_ms;
    if !resident_other_ms.is_finite() || resident_other_ms < 0.0 {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "resident startup phases total {detailed_ms:.3} ms but the enclosing constructor measured {resident_program_ms:.3} ms"
        )));
    }

    Ok(StartupSample {
        schema_version: SCHEMA_VERSION,
        loader: loader.as_str().into(),
        filesystem_cache: "warm".into(),
        checkpoint_revision,
        device_name,
        compute_capability: format!("{}.{}", compute_capability.0, compute_capability.1),
        checkpoint_admission_ms: milliseconds(checkpoint_admission),
        cuda_initialization_ms: milliseconds(cuda_initialization),
        layout_plan_ms,
        arena_allocation_ms,
        operator_setup_ms,
        weight_prepare_ms,
        source_binding_ms,
        qkv_gather_ms,
        nvfp4_materialize_ms,
        preparation_other_ms,
        weight_copy_ms,
        weight_load_ms,
        nonweight_init_ms,
        graph_capture_ms,
        resident_other_ms,
        resident_program_ms,
        checkpoint_to_ready_ms: milliseconds(checkpoint_to_ready),
        tensor_count,
        resident_weight_bytes: program.resident_weight_bytes(),
        resident_arena_bytes: program.resident_arena_bytes(),
        kv_arena_bytes: program.kv_arena_bytes(),
        total_device_arena_bytes: program.arena_bytes(),
        peak_rss_bytes,
        upload_bytes: program.load_stats().upload_bytes(),
        upload_submissions: program.load_stats().upload_submissions(),
        zeroed_bytes: program.load_stats().zeroed_bytes(),
        pinned_stager_bytes: program.load_stats().pinned_stager_bytes(),
        borrowed_source_bytes: load_stats.borrowed_source_bytes(),
        gathered_source_bytes: load_stats.gathered_source_bytes(),
        swizzled_source_bytes: load_stats.swizzled_source_bytes(),
    })
}

fn validate_samples(samples: &[StartupSample]) -> Result<(), DeviceBenchmarkError> {
    let first = &samples[0];
    for sample in samples {
        let timings = [
            sample.checkpoint_admission_ms,
            sample.cuda_initialization_ms,
            sample.layout_plan_ms,
            sample.arena_allocation_ms,
            sample.operator_setup_ms,
            sample.weight_prepare_ms,
            sample.source_binding_ms,
            sample.qkv_gather_ms,
            sample.nvfp4_materialize_ms,
            sample.preparation_other_ms,
            sample.weight_copy_ms,
            sample.weight_load_ms,
            sample.nonweight_init_ms,
            sample.graph_capture_ms,
            sample.resident_other_ms,
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
            || sample.upload_bytes != first.upload_bytes
            || sample.upload_submissions != first.upload_submissions
            || sample.zeroed_bytes != first.zeroed_bytes
            || sample.pinned_stager_bytes != first.pinned_stager_bytes
            || sample.borrowed_source_bytes != first.borrowed_source_bytes
            || sample.gathered_source_bytes != first.gathered_source_bytes
            || sample.swizzled_source_bytes != first.swizzled_source_bytes
        {
            return Err(DeviceBenchmarkError::Precondition(
                "fresh-process startup samples disagree on their exact product identity or byte accounting"
                    .into(),
            ));
        }
        let classified_weight_bytes = sample
            .borrowed_source_bytes
            .checked_add(sample.gathered_source_bytes)
            .and_then(|bytes| bytes.checked_add(sample.swizzled_source_bytes));
        if classified_weight_bytes != Some(sample.resident_weight_bytes) {
            return Err(DeviceBenchmarkError::Precondition(
                "startup source-preparation bytes do not cover the resident weights exactly".into(),
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

fn nanoseconds_to_milliseconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000.0
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
    print_h2d_calibration(&report.h2d_calibration);
    let pinned_h2d_gib_s = pinned_h2d_gib_s(&report.h2d_calibration);
    for loader in &report.loaders {
        eprintln!();
        eprintln!("{} startup, warm filesystem cache", loader.loader);
        eprintln!("phase                         minimum ms   median ms   maximum ms");
        for timing in &loader.timings {
            eprintln!(
                "{:<29} {:>10.3} {:>11.3} {:>12.3}",
                timing.phase, timing.minimum_ms, timing.median_ms, timing.maximum_ms,
            );
        }
        let sample = &loader.samples[0];
        eprintln!(
            "uploads: {:.2} GiB in {} submissions · zeroed: {:.2} GiB · pinned: {:.0} MiB",
            sample.upload_bytes as f64 / (1_u64 << 30) as f64,
            sample.upload_submissions,
            sample.zeroed_bytes as f64 / (1_u64 << 30) as f64,
            sample.pinned_stager_bytes as f64 / (1_u64 << 20) as f64,
        );
        eprintln!(
            "source bytes: {:.2} GiB borrowed · {:.2} GiB gathered · {:.2} GiB swizzled",
            sample.borrowed_source_bytes as f64 / (1_u64 << 30) as f64,
            sample.gathered_source_bytes as f64 / (1_u64 << 30) as f64,
            sample.swizzled_source_bytes as f64 / (1_u64 << 30) as f64,
        );
        let weight_copy_gib_s =
            sample.upload_bytes as f64 / (1_u64 << 30) as f64 / (sample.weight_copy_ms / 1_000.0);
        eprintln!(
            "copy-call aggregate: {:.2} GiB/s · {:.1}% of contiguous pinned H2D",
            weight_copy_gib_s,
            weight_copy_gib_s / pinned_h2d_gib_s * 100.0,
        );
    }
    let sample = &report.loaders[0].samples[0];
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
        .loaders
        .iter()
        .flat_map(|loader| loader.samples.iter())
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
    use super::{StartupLoader, parse_options, parse_peak_rss_bytes};
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
        assert_eq!(options.loaders, vec![StartupLoader::Legacy]);
        assert_eq!(options.json_path, PathBuf::from("report.json"));
    }

    #[test]
    fn startup_options_compare_both_loaders_by_default() {
        let options = parse_options(&[OsString::from("/snapshot")]).unwrap();
        assert_eq!(
            options.loaders,
            vec![StartupLoader::Legacy, StartupLoader::Selective]
        );
    }
}
