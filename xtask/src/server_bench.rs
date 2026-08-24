//! Clock-authoritative production server timing and energy envelope.

use super::{parse_compute_pids, run_visible, server_performance, server_qual};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const TELEMETRY_INTERVAL: Duration = Duration::from_millis(10);
const IDLE_SECONDS: f64 = 2.0;
const LOADED_PROBE_SECONDS: f64 = 3.0;
// Ten 7.5 MHz boost steps are 3.4% at 2,197 MHz and recur within one exact HTTP suite.
const MAX_SM_CLOCK_SPREAD_MHZ: u32 = 75;
const MAX_MEMORY_CLOCK_SPREAD_MHZ: u32 = 250;
const DEFAULT_SAMPLES: usize = 5;
const DIAGNOSTIC_CLOCK_ENV: &str = "TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT";

pub(super) fn run(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let options = Options::parse(arguments)?;
    let baseline = root.join(server_performance::baseline_path(options.long_context));
    if options.baseline_action == Some(BaselineAction::Check) {
        server_performance::preflight(&baseline)?;
    }
    let (tools, mut server) =
        server_qual::start(root, &options.snapshot, "server performance setup")?;
    let result = run_authority(root, &options, &tools, &server);
    let stop = server.stop_and_wait();
    result?;
    stop?;
    let report = root.join(&options.output);
    match options.baseline_action {
        Some(BaselineAction::Check) => server_performance::compare(&report, &baseline),
        Some(BaselineAction::Bless) => server_performance::bless(&report, &baseline),
        None => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineAction {
    Check,
    Bless,
}

struct Options {
    snapshot: PathBuf,
    output: PathBuf,
    samples: usize,
    long_context: bool,
    baseline_action: Option<BaselineAction>,
}

impl Options {
    fn parse(arguments: &[OsString]) -> Result<Self, Box<dyn Error>> {
        let Some((snapshot, remaining)) = arguments.split_first() else {
            return Err("usage: cargo run -p xtask -- bench-server SNAPSHOT [--samples N] [--long-context] [--json target/PATH] [--check|--bless]".into());
        };
        let snapshot = PathBuf::from(snapshot);
        if !snapshot.is_dir() {
            return Err(format!("snapshot `{}` is not a directory", snapshot.display()).into());
        }
        let mut output = PathBuf::from("target/benchmarks/server-authority/server.json");
        let mut samples = DEFAULT_SAMPLES;
        let mut long_context = false;
        let mut baseline_action = None;
        let mut index = 0;
        while index < remaining.len() {
            match remaining[index].to_str() {
                Some("--samples") => {
                    let value = remaining
                        .get(index + 1)
                        .ok_or("--samples requires a count")?;
                    samples = value
                        .to_str()
                        .ok_or("--samples count is not UTF-8")?
                        .parse()?;
                    index += 2;
                }
                Some("--long-context") => {
                    long_context = true;
                    index += 1;
                }
                Some("--check") if baseline_action.is_none() => {
                    baseline_action = Some(BaselineAction::Check);
                    index += 1;
                }
                Some("--bless") if baseline_action.is_none() => {
                    baseline_action = Some(BaselineAction::Bless);
                    index += 1;
                }
                Some("--json") => {
                    output =
                        PathBuf::from(remaining.get(index + 1).ok_or("--json requires a path")?);
                    index += 2;
                }
                Some(option) => {
                    return Err(format!("unknown bench-server option `{option}`").into());
                }
                None => return Err("bench-server option is not UTF-8".into()),
            }
        }
        if !(3..=40).contains(&samples) {
            return Err("--samples must be in 3..=40".into());
        }
        if baseline_action == Some(BaselineAction::Bless)
            && samples < server_performance::MINIMUM_BASELINE_SAMPLES
        {
            return Err(format!(
                "--bless requires at least {} samples per case",
                server_performance::MINIMUM_BASELINE_SAMPLES
            )
            .into());
        }
        validate_target_path(&output)?;
        Ok(Self {
            snapshot,
            output,
            samples,
            long_context,
            baseline_action,
        })
    }
}

#[derive(Serialize)]
struct AuthorityReport {
    schema_version: u32,
    suite: &'static str,
    status: &'static str,
    snapshot: String,
    server_log: String,
    raw_benchmark_report: String,
    server_binary_sha256: String,
    device: DeviceIdentity,
    server_pid: u32,
    clock_policy: ClockPolicy,
    idle: TelemetryEvidence,
    loaded_probe: TelemetryEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    measurement: Option<TelemetryEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    energy: Option<EnergyEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    benchmark: Option<Value>,
}

#[derive(Serialize)]
struct DeviceIdentity {
    name: String,
    uuid: String,
    driver: String,
    compute_capability: String,
    total_memory_mib: u64,
}

#[derive(Clone, Copy, Serialize)]
struct ClockPolicy {
    minimum_samples: usize,
    maximum_sm_clock_spread_mhz: u32,
    maximum_memory_clock_spread_mhz: u32,
}

#[derive(Serialize)]
struct TelemetryEvidence {
    samples: Vec<TelemetrySample>,
    summary: TelemetrySummary,
}

#[derive(Clone, Debug, Serialize)]
struct TelemetrySample {
    elapsed_milliseconds: f64,
    sm_clock_mhz: u32,
    memory_clock_mhz: u32,
    temperature_celsius: u32,
    board_power_watts: f64,
    memory_used_mib: u64,
    memory_free_mib: u64,
}

#[derive(Serialize)]
struct TelemetrySummary {
    sample_count: usize,
    duration_seconds: f64,
    sm_clock_min_mhz: u32,
    sm_clock_max_mhz: u32,
    sm_clock_spread_mhz: u32,
    memory_clock_min_mhz: u32,
    memory_clock_max_mhz: u32,
    memory_clock_spread_mhz: u32,
    temperature_min_celsius: u32,
    temperature_max_celsius: u32,
    power_min_watts: f64,
    power_average_watts: f64,
    power_max_watts: f64,
    energy_joules: f64,
    memory_used_min_mib: u64,
    memory_used_max_mib: u64,
}

#[derive(Serialize)]
struct EnergyEvidence {
    scope: &'static str,
    denominator: &'static str,
    idle_average_watts: f64,
    whole_board_joules: f64,
    above_idle_joules: f64,
    completion_tokens: u64,
    whole_board_joules_per_completion_token: f64,
    above_idle_joules_per_completion_token: f64,
}

fn run_authority(
    root: &Path,
    options: &Options,
    tools: &server_qual::HostTools,
    server: &server_qual::ProductionServer,
) -> Result<(), Box<dyn Error>> {
    let server_pid = server.pid()?;
    let server_binary_sha256 = file_sha256(server.executable())?;
    require_only_server_process(server_pid)?;
    run_visible(
        Command::new(tools.qualifier())
            .arg(server.base_url())
            .current_dir(root),
    )?;
    require_only_server_process(server_pid)?;

    let idle = sample_for(Duration::from_secs_f64(IDLE_SECONDS))?;
    let raw_path = raw_report_path(&options.output)?;
    let probe_path = probe_report_path(&options.output)?;
    let probe_seconds = LOADED_PROBE_SECONDS.to_string();
    let probe_sampler = TelemetrySampler::start();
    let probe = run_visible(
        Command::new(tools.benchmark())
            .arg(server.base_url())
            .args(["--json", path_text(&probe_path)?])
            .args(["--load-probe-seconds", probe_seconds.as_str()])
            .current_dir(root),
    );
    let loaded_probe = probe_sampler.finish()?;
    probe?;
    require_only_server_process(server_pid)?;

    let device = device_identity()?;
    let loaded_clock_refusal = require_comparable_clocks(&loaded_probe.summary).err();
    if let Some(refusal) = loaded_clock_refusal.as_ref()
        && !preserve_refused_measurements()
    {
        let report = AuthorityReport {
            schema_version: 1,
            suite: "server",
            status: "refused_loaded_clocks",
            snapshot: options.snapshot.display().to_string(),
            server_log: server.log_path().display().to_string(),
            raw_benchmark_report: raw_path.display().to_string(),
            server_binary_sha256,
            device,
            server_pid,
            clock_policy: clock_policy(),
            idle,
            loaded_probe,
            measurement: None,
            energy: None,
            refusal: Some(refusal.clone()),
            benchmark: None,
        };
        write_authority(&options.output, &report)?;
        return Err(refusal.clone().into());
    }
    if let Some(refusal) = loaded_clock_refusal.as_ref() {
        eprintln!(
            "{refusal}; {DIAGNOSTIC_CLOCK_ENV}=1 preserves the directly measured suite as refused evidence"
        );
    }

    if raw_path.exists() {
        fs::remove_file(&raw_path)?;
    }
    let sampler = TelemetrySampler::start();
    let mut command = Command::new(tools.benchmark());
    command
        .arg(server.base_url())
        .args(["--json", path_text(&raw_path)?])
        .args(["--samples", &options.samples.to_string()])
        .current_dir(root);
    if options.long_context {
        command.arg("--long-context");
    }
    let benchmark_result = run_visible(&mut command);
    let measurement = sampler.finish()?;
    require_only_server_process(server_pid)?;
    let benchmark = read_json_if_present(&raw_path)?;
    let measurement_clock_refusal = require_comparable_clocks(&measurement.summary).err();
    let (status, refusal) = authority_outcome(
        benchmark_result.is_err(),
        loaded_clock_refusal,
        measurement_clock_refusal,
    );
    let energy = benchmark
        .as_ref()
        .and_then(completion_tokens)
        .filter(|tokens| *tokens != 0)
        .map(|tokens| energy_evidence(&idle.summary, &measurement.summary, tokens));
    let report = AuthorityReport {
        schema_version: 1,
        suite: "server",
        status,
        snapshot: options.snapshot.display().to_string(),
        server_log: server.log_path().display().to_string(),
        raw_benchmark_report: raw_path.display().to_string(),
        server_binary_sha256,
        device,
        server_pid,
        clock_policy: clock_policy(),
        idle,
        loaded_probe,
        measurement: Some(measurement),
        energy,
        refusal: refusal.clone(),
        benchmark,
    };
    write_authority(&options.output, &report)?;
    benchmark_result?;
    if let Some(refusal) = refusal {
        return Err(refusal.into());
    }

    println!(
        "authoritative server report: {} (raw timing: {})",
        options.output.display(),
        raw_path.display()
    );
    Ok(())
}

fn preserve_refused_measurements() -> bool {
    std::env::var(DIAGNOSTIC_CLOCK_ENV).as_deref() == Ok("1")
}

fn authority_outcome(
    benchmark_failed: bool,
    loaded_clock_refusal: Option<String>,
    measurement_clock_refusal: Option<String>,
) -> (&'static str, Option<String>) {
    let status = if benchmark_failed {
        "failed"
    } else if loaded_clock_refusal.is_some() {
        "refused_loaded_clocks"
    } else if measurement_clock_refusal.is_some() {
        "refused_measurement_clocks"
    } else {
        "complete"
    };
    let refusal = match (loaded_clock_refusal, measurement_clock_refusal) {
        (Some(loaded), Some(measurement)) => Some(format!(
            "loaded probe refused: {loaded}; complete measurement refused: {measurement}"
        )),
        (Some(loaded), None) => Some(loaded),
        (None, Some(measurement)) => Some(measurement),
        (None, None) => None,
    };
    (status, refusal)
}

struct TelemetrySampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<TelemetryEvidence, String>>>,
}

impl TelemetrySampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || collect_telemetry(&thread_stop));
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> Result<TelemetryEvidence, Box<dyn Error>> {
        self.stop.store(true, Ordering::Release);
        let handle = self
            .handle
            .take()
            .ok_or("telemetry sampler was already finished")?;
        handle
            .join()
            .map_err(|_| "telemetry sampler panicked")?
            .map_err(Into::into)
    }
}

impl Drop for TelemetrySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn sample_for(duration: Duration) -> Result<TelemetryEvidence, Box<dyn Error>> {
    let sampler = TelemetrySampler::start();
    thread::sleep(duration);
    sampler.finish()
}

fn collect_telemetry(stop: &AtomicBool) -> Result<TelemetryEvidence, String> {
    let started = Instant::now();
    let mut samples = Vec::new();
    while !stop.load(Ordering::Acquire) || samples.len() < 3 {
        let iteration = Instant::now();
        samples.push(query_telemetry(started.elapsed().as_secs_f64() * 1_000.0)?);
        let elapsed = iteration.elapsed();
        if elapsed < TELEMETRY_INTERVAL {
            thread::sleep(TELEMETRY_INTERVAL - elapsed);
        }
    }
    let summary = summarize_telemetry(&samples)?;
    Ok(TelemetryEvidence { samples, summary })
}

fn query_telemetry(elapsed_milliseconds: f64) -> Result<TelemetrySample, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw.instant,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("nvidia-smi telemetry query failed: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi telemetry query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_telemetry_row(
        elapsed_milliseconds,
        &String::from_utf8_lossy(&output.stdout),
    )
}

fn parse_telemetry_row(elapsed_milliseconds: f64, row: &str) -> Result<TelemetrySample, String> {
    let fields = row.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [sm, memory, temperature, power, used, free] = fields.as_slice() else {
        return Err(format!(
            "unexpected nvidia-smi telemetry row `{}`",
            row.trim()
        ));
    };
    Ok(TelemetrySample {
        elapsed_milliseconds,
        sm_clock_mhz: parse_u32(sm, "SM clock")?,
        memory_clock_mhz: parse_u32(memory, "memory clock")?,
        temperature_celsius: parse_u32(temperature, "temperature")?,
        board_power_watts: power
            .parse()
            .map_err(|error| format!("invalid board power `{power}`: {error}"))?,
        memory_used_mib: parse_u64(used, "used memory")?,
        memory_free_mib: parse_u64(free, "free memory")?,
    })
}

fn summarize_telemetry(samples: &[TelemetrySample]) -> Result<TelemetrySummary, String> {
    if samples.len() < 3 {
        return Err(format!(
            "telemetry captured {} samples, expected at least three",
            samples.len()
        ));
    }
    let sm_min = samples
        .iter()
        .map(|sample| sample.sm_clock_mhz)
        .min()
        .unwrap();
    let sm_max = samples
        .iter()
        .map(|sample| sample.sm_clock_mhz)
        .max()
        .unwrap();
    let memory_min = samples
        .iter()
        .map(|sample| sample.memory_clock_mhz)
        .min()
        .unwrap();
    let memory_max = samples
        .iter()
        .map(|sample| sample.memory_clock_mhz)
        .max()
        .unwrap();
    let temperature_min = samples
        .iter()
        .map(|sample| sample.temperature_celsius)
        .min()
        .unwrap();
    let temperature_max = samples
        .iter()
        .map(|sample| sample.temperature_celsius)
        .max()
        .unwrap();
    let power_min = samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .reduce(f64::min)
        .unwrap();
    let power_max = samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .reduce(f64::max)
        .unwrap();
    let power_average = samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .sum::<f64>()
        / samples.len() as f64;
    let energy_joules: f64 = samples
        .windows(2)
        .map(|pair| {
            let seconds = (pair[1].elapsed_milliseconds - pair[0].elapsed_milliseconds) / 1_000.0;
            seconds * (pair[0].board_power_watts + pair[1].board_power_watts) / 2.0
        })
        .sum();
    let duration_seconds = (samples.last().unwrap().elapsed_milliseconds
        - samples.first().unwrap().elapsed_milliseconds)
        / 1_000.0;
    if !power_average.is_finite()
        || !energy_joules.is_finite()
        || energy_joules < 0.0
        || duration_seconds <= 0.0
    {
        return Err("telemetry produced invalid power or duration evidence".into());
    }
    Ok(TelemetrySummary {
        sample_count: samples.len(),
        duration_seconds,
        sm_clock_min_mhz: sm_min,
        sm_clock_max_mhz: sm_max,
        sm_clock_spread_mhz: sm_max - sm_min,
        memory_clock_min_mhz: memory_min,
        memory_clock_max_mhz: memory_max,
        memory_clock_spread_mhz: memory_max - memory_min,
        temperature_min_celsius: temperature_min,
        temperature_max_celsius: temperature_max,
        power_min_watts: power_min,
        power_average_watts: power_average,
        power_max_watts: power_max,
        energy_joules,
        memory_used_min_mib: samples
            .iter()
            .map(|sample| sample.memory_used_mib)
            .min()
            .unwrap(),
        memory_used_max_mib: samples
            .iter()
            .map(|sample| sample.memory_used_mib)
            .max()
            .unwrap(),
    })
}

fn require_comparable_clocks(summary: &TelemetrySummary) -> Result<(), String> {
    if summary.sm_clock_spread_mhz > MAX_SM_CLOCK_SPREAD_MHZ {
        return Err(format!(
            "server SM clock spread is {} MHz ({}..{}), above the allowed {} MHz",
            summary.sm_clock_spread_mhz,
            summary.sm_clock_min_mhz,
            summary.sm_clock_max_mhz,
            MAX_SM_CLOCK_SPREAD_MHZ
        ));
    }
    if summary.memory_clock_spread_mhz > MAX_MEMORY_CLOCK_SPREAD_MHZ {
        return Err(format!(
            "server memory clock spread is {} MHz ({}..{}), above the allowed {} MHz",
            summary.memory_clock_spread_mhz,
            summary.memory_clock_min_mhz,
            summary.memory_clock_max_mhz,
            MAX_MEMORY_CLOCK_SPREAD_MHZ
        ));
    }
    Ok(())
}

fn energy_evidence(
    idle: &TelemetrySummary,
    measurement: &TelemetrySummary,
    completion_tokens: u64,
) -> EnergyEvidence {
    let idle_joules = idle.power_average_watts * measurement.duration_seconds;
    let above_idle_joules = (measurement.energy_joules - idle_joules).max(0.0);
    EnergyEvidence {
        scope: "whole board over the complete directly timed HTTP suite",
        denominator: "all completion tokens including reported benchmark setup requests",
        idle_average_watts: idle.power_average_watts,
        whole_board_joules: measurement.energy_joules,
        above_idle_joules,
        completion_tokens,
        whole_board_joules_per_completion_token: measurement.energy_joules
            / completion_tokens as f64,
        above_idle_joules_per_completion_token: above_idle_joules / completion_tokens as f64,
    }
}

const fn clock_policy() -> ClockPolicy {
    ClockPolicy {
        minimum_samples: 3,
        maximum_sm_clock_spread_mhz: MAX_SM_CLOCK_SPREAD_MHZ,
        maximum_memory_clock_spread_mhz: MAX_MEMORY_CLOCK_SPREAD_MHZ,
    }
}

fn completion_tokens(report: &Value) -> Option<u64> {
    let setup = report["setup_completion_tokens"].as_u64().unwrap_or(0);
    report["cases"]
        .as_array()?
        .iter()
        .flat_map(|case| case["observations"].as_array().into_iter().flatten())
        .try_fold(setup, |total, observation| {
            total.checked_add(observation["completion_tokens"].as_u64()?)
        })
}

fn require_only_server_process(server_pid: u32) -> Result<(), Box<dyn Error>> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi compute-process query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let pids = parse_compute_pids(&String::from_utf8(output.stdout)?)?;
    if pids != [server_pid] {
        return Err(format!(
            "server performance requires only child PID {server_pid} on device zero, found {pids:?}"
        )
        .into());
    }
    Ok(())
}

fn device_identity() -> Result<DeviceIdentity, Box<dyn Error>> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err("nvidia-smi device identity query failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let fields = text.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [name, uuid, driver, compute_capability, total_memory] = fields.as_slice() else {
        return Err(format!("unexpected device identity row `{}`", text.trim()).into());
    };
    Ok(DeviceIdentity {
        name: (*name).into(),
        uuid: (*uuid).into(),
        driver: (*driver).into(),
        compute_capability: (*compute_capability).into(),
        total_memory_mib: parse_u64(total_memory, "total memory")?,
    })
}

fn file_sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn raw_report_path(authority: &Path) -> Result<PathBuf, Box<dyn Error>> {
    Ok(authority
        .parent()
        .ok_or("authority report has no parent")?
        .join("server-raw.json"))
}

fn probe_report_path(authority: &Path) -> Result<PathBuf, Box<dyn Error>> {
    Ok(authority
        .parent()
        .ok_or("authority report has no parent")?
        .join("server-probe-unused.json"))
}

fn read_json_if_present(path: &Path) -> Result<Option<Value>, Box<dyn Error>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn write_authority(path: &Path, report: &AuthorityReport) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("authority report has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn validate_target_path(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.is_absolute()
        || !matches!(path.components().next(), Some(Component::Normal(first)) if first == "target")
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("server authority JSON must be a relative path below target/".into());
    }
    Ok(())
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

fn parse_u32(value: &str, label: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {label} `{value}`: {error}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {label} `{value}`: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{
        BaselineAction, MAX_MEMORY_CLOCK_SPREAD_MHZ, MAX_SM_CLOCK_SPREAD_MHZ, Options,
        authority_outcome, completion_tokens, parse_telemetry_row, require_comparable_clocks,
        summarize_telemetry,
    };
    use serde_json::json;
    use std::ffi::OsString;

    #[test]
    fn telemetry_parser_and_summary_preserve_clock_power_and_memory_evidence() {
        let samples = [
            parse_telemetry_row(0.0, "2182, 13801, 51, 300.0, 30554, 2000").unwrap(),
            parse_telemetry_row(10.0, "2190, 13801, 52, 320.0, 30560, 1994").unwrap(),
            parse_telemetry_row(20.0, "2197, 14001, 53, 340.0, 30558, 1996").unwrap(),
        ];
        let summary = summarize_telemetry(&samples).unwrap();
        assert_eq!(summary.sm_clock_spread_mhz, 15);
        assert_eq!(summary.memory_clock_spread_mhz, 200);
        assert_eq!(summary.memory_used_max_mib, 30_560);
        assert_eq!(summary.power_average_watts, 320.0);
        assert!((summary.energy_joules - 6.4).abs() < 1e-9);
        require_comparable_clocks(&summary).unwrap();
    }

    #[test]
    fn clock_policy_refuses_only_outside_the_checked_spreads() {
        let samples = [
            parse_telemetry_row(0.0, "2100, 13801, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(10.0, "2175, 13801, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(20.0, "2175, 13801, 51, 300.0, 1, 1").unwrap(),
        ];
        let summary = summarize_telemetry(&samples).unwrap();
        assert_eq!(summary.sm_clock_spread_mhz, MAX_SM_CLOCK_SPREAD_MHZ);
        require_comparable_clocks(&summary).unwrap();

        let samples = [
            parse_telemetry_row(0.0, "2100, 13801, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(10.0, "2176, 13801, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(20.0, "2176, 13801, 51, 300.0, 1, 1").unwrap(),
        ];
        let summary = summarize_telemetry(&samples).unwrap();
        assert_eq!(summary.sm_clock_spread_mhz, MAX_SM_CLOCK_SPREAD_MHZ + 1);
        assert!(require_comparable_clocks(&summary).is_err());

        let samples = [
            parse_telemetry_row(0.0, "2190, 13750, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(10.0, "2190, 14001, 51, 300.0, 1, 1").unwrap(),
            parse_telemetry_row(20.0, "2190, 14001, 51, 300.0, 1, 1").unwrap(),
        ];
        let summary = summarize_telemetry(&samples).unwrap();
        assert_eq!(
            summary.memory_clock_spread_mhz,
            MAX_MEMORY_CLOCK_SPREAD_MHZ + 1
        );
        assert!(require_comparable_clocks(&summary).is_err());
    }

    #[test]
    fn completion_token_denominator_counts_aggregate_concurrency_rows() {
        let report = json!({
            "setup_completion_tokens": 4,
            "cases": [
                {"observations": [{"completion_tokens": 8}, {"completion_tokens": 8}]},
                {"observations": [{"completion_tokens": 64}]}
            ]
        });
        assert_eq!(completion_tokens(&report), Some(84));
    }

    #[test]
    fn loaded_clock_refusal_remains_terminal_after_preserved_measurement() {
        let (status, refusal) = authority_outcome(
            false,
            Some("loaded drift".into()),
            Some("measurement drift".into()),
        );
        assert_eq!(status, "refused_loaded_clocks");
        let refusal = refusal.unwrap();
        assert!(refusal.contains("loaded probe refused: loaded drift"));
        assert!(refusal.contains("complete measurement refused: measurement drift"));

        let (status, refusal) = authority_outcome(false, None, None);
        assert_eq!(status, "complete");
        assert!(refusal.is_none());

        let (status, refusal) = authority_outcome(true, Some("loaded drift".into()), None);
        assert_eq!(status, "failed");
        assert_eq!(refusal.as_deref(), Some("loaded drift"));
    }

    #[test]
    fn baseline_modes_are_mutually_exclusive_and_blessing_requires_five_samples() {
        let arguments = [OsString::from("."), OsString::from("--check")];
        assert_eq!(
            Options::parse(&arguments).unwrap().baseline_action,
            Some(BaselineAction::Check)
        );
        let arguments = [
            OsString::from("."),
            OsString::from("--bless"),
            OsString::from("--samples"),
            OsString::from("3"),
        ];
        assert!(Options::parse(&arguments).is_err());
        let arguments = [
            OsString::from("."),
            OsString::from("--check"),
            OsString::from("--bless"),
        ];
        assert!(Options::parse(&arguments).is_err());
    }
}
