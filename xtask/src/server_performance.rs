//! Checked performance authority for the complete production HTTP boundary.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

const AUTHORITY_SCHEMA: u32 = 1;
const BENCHMARK_SCHEMA: u32 = 1;
const BASELINE_SCHEMA: u32 = 1;
const MODEL: &str = "unsloth/Qwen3.8-27B-NVFP4";
const REVISION: &str = "16b6615af3548b88e2d8e382457bc705b00479cf";
const DEVICE: &str = "NVIDIA GeForce RTX 5090";
const COMPUTE_CAPABILITY: &str = "12.0";
const RAW_AUTHORITY: &str = "diagnostic_external_no_clock_evidence";
const SM_CLOCK_PADDING_MHZ: u32 = 15;
const MEMORY_CLOCK_PADDING_MHZ: u32 = 50;
const DEFAULT_RELATIVE_TOLERANCE_PERCENT: f64 = 15.0;
const DEFAULT_LATENCY_TOLERANCE_MS: f64 = 0.5;
const DEFAULT_THROUGHPUT_TOLERANCE: f64 = 1.0;
const DEFAULT_ENERGY_TOLERANCE: f64 = 0.01;
const DEFAULT_MEMORY_TOLERANCE_MIB: u64 = 16;
pub(super) const MINIMUM_BASELINE_SAMPLES: usize = 5;

pub(super) const fn baseline_path(long_context: bool) -> &'static str {
    if long_context {
        "qual/baselines/server-http-long-context-sm120.json"
    } else {
        "qual/baselines/server-http-sm120.json"
    }
}

pub(super) fn preflight(path: &Path) -> Result<(), Box<dyn Error>> {
    let baseline: ServerBaseline = serde_json::from_slice(&fs::read(path).map_err(|error| {
        format!(
            "could not read server performance baseline {}: {error}; run `bench-server ... --bless` only after reviewing a complete authoritative report",
            path.display()
        )
    })?)?;
    validate_baseline(&baseline)
}

pub(super) fn compare(report_path: &Path, baseline_path: &Path) -> Result<(), Box<dyn Error>> {
    let candidate = read_candidate(report_path)?;
    let baseline: ServerBaseline = serde_json::from_slice(&fs::read(baseline_path)?)?;
    validate_baseline(&baseline)?;
    validate_environment(&candidate, &baseline)?;

    let candidates = candidate_metrics(&candidate);
    let authorities = baseline_metrics(&baseline)?;
    if candidates.keys().ne(authorities.keys()) {
        return Err("server performance report and baseline metric inventories differ".into());
    }

    println!(
        "status case                             metric                         reference    candidate"
    );
    let mut failures = 0usize;
    for (key, value) in candidates {
        let authority = authorities
            .get(&key)
            .expect("matching server metric inventories were checked");
        let passed = !authority.enforced || metric_passes(value, authority);
        let status = if !authority.enforced {
            "INFO"
        } else if passed {
            "PASS"
        } else {
            failures += 1;
            "FAIL"
        };
        println!(
            "{status:<6} {:<32} {:<30} {:>10.3} {:>12.3}",
            key.case, key.metric, authority.reference, value
        );
    }

    let memory_limit = baseline
        .resident_memory_reference_mib
        .checked_add(baseline.resident_memory_tolerance_mib)
        .ok_or("server baseline memory limit overflows")?;
    let memory = candidate.measurement.summary.memory_used_max_mib;
    let memory_passed = !baseline.resident_memory_enforced || memory <= memory_limit;
    if !memory_passed {
        failures += 1;
    }
    println!(
        "{:<6} {:<32} {:<30} {:>10} {:>12}",
        if !baseline.resident_memory_enforced {
            "INFO"
        } else if memory_passed {
            "PASS"
        } else {
            "FAIL"
        },
        "suite",
        "resident_memory_mib",
        baseline.resident_memory_reference_mib,
        memory
    );

    if failures != 0 {
        return Err(format!("{failures} enforced server performance metrics regressed").into());
    }
    println!(
        "server performance gate passed: {} timing/energy metrics and one memory metric",
        authorities.len()
    );
    Ok(())
}

pub(super) fn bless(report_path: &Path, baseline_path: &Path) -> Result<(), Box<dyn Error>> {
    let candidate = read_candidate(report_path)?;
    if candidate.benchmark.samples_per_case < MINIMUM_BASELINE_SAMPLES {
        return Err(format!(
            "server baseline blessing requires at least {MINIMUM_BASELINE_SAMPLES} samples per case"
        )
        .into());
    }
    let previous = if baseline_path.is_file() {
        let baseline: ServerBaseline = serde_json::from_slice(&fs::read(baseline_path)?)?;
        validate_baseline(&baseline)?;
        Some(baseline)
    } else {
        None
    };
    if previous.as_ref().is_some_and(|baseline| {
        baseline.long_context_enabled != candidate.benchmark.long_context_enabled
    }) {
        return Err(
            "cannot replace a server baseline with a different long-context inventory".into(),
        );
    }
    let retained = previous.as_ref().map(baseline_metrics).transpose()?;
    let metrics = candidate_metrics(&candidate)
        .into_iter()
        .map(|(key, reference)| {
            let previous = retained.as_ref().and_then(|metrics| metrics.get(&key));
            let (relative, absolute, enforced) = previous.map_or_else(
                || default_controls(&key),
                |metric| {
                    (
                        metric.relative_tolerance_percent,
                        metric.absolute_tolerance,
                        metric.enforced,
                    )
                },
            );
            BaselineMetric {
                case: key.case,
                metric: key.metric,
                unit: key.unit,
                direction: key.direction,
                reference,
                relative_tolerance_percent: relative,
                absolute_tolerance: absolute,
                enforced,
            }
        })
        .collect();
    let measurement = &candidate.measurement.summary;
    let baseline = ServerBaseline {
        schema_version: BASELINE_SCHEMA,
        suite: "server-http".into(),
        model: MODEL.into(),
        checkpoint_revision: REVISION.into(),
        device: candidate.device.name.clone(),
        driver: candidate.device.driver.clone(),
        compute_capability: candidate.device.compute_capability.clone(),
        blessed_server_binary_sha256: candidate.server_binary_sha256.clone(),
        long_context_enabled: candidate.benchmark.long_context_enabled,
        minimum_samples_per_case: candidate.benchmark.samples_per_case,
        loaded_sm_clock_band_mhz: padded_band(
            candidate.loaded_probe.summary.sm_clock_min_mhz,
            candidate.loaded_probe.summary.sm_clock_max_mhz,
            SM_CLOCK_PADDING_MHZ,
        ),
        loaded_memory_clock_band_mhz: padded_band(
            candidate.loaded_probe.summary.memory_clock_min_mhz,
            candidate.loaded_probe.summary.memory_clock_max_mhz,
            MEMORY_CLOCK_PADDING_MHZ,
        ),
        measurement_sm_clock_band_mhz: padded_band(
            measurement.sm_clock_min_mhz,
            measurement.sm_clock_max_mhz,
            SM_CLOCK_PADDING_MHZ,
        ),
        measurement_memory_clock_band_mhz: padded_band(
            measurement.memory_clock_min_mhz,
            measurement.memory_clock_max_mhz,
            MEMORY_CLOCK_PADDING_MHZ,
        ),
        device_total_memory_mib: candidate.device.total_memory_mib,
        resident_memory_reference_mib: measurement.memory_used_max_mib,
        resident_memory_tolerance_mib: previous
            .as_ref()
            .map_or(DEFAULT_MEMORY_TOLERANCE_MIB, |baseline| {
                baseline.resident_memory_tolerance_mib
            }),
        resident_memory_enforced: previous
            .as_ref()
            .is_none_or(|baseline| baseline.resident_memory_enforced),
        metrics,
    };
    validate_baseline(&baseline)?;
    if let Some(parent) = baseline_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(&baseline)?;
    bytes.push(b'\n');
    fs::write(baseline_path, bytes)?;
    println!(
        "server performance baseline blessed explicitly: {}",
        baseline_path.display()
    );
    Ok(())
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct AuthorityReport {
    schema_version: u32,
    suite: String,
    status: String,
    snapshot: String,
    server_log: String,
    raw_benchmark_report: String,
    server_binary_sha256: String,
    device: DeviceIdentity,
    server_pid: u32,
    clock_policy: ClockPolicy,
    idle: TelemetryEvidence,
    loaded_probe: TelemetryEvidence,
    measurement: Option<TelemetryEvidence>,
    energy: Option<EnergyEvidence>,
    refusal: Option<String>,
    benchmark: Option<BenchmarkReport>,
}

struct Candidate {
    server_binary_sha256: String,
    device: DeviceIdentity,
    loaded_probe: TelemetryEvidence,
    measurement: TelemetryEvidence,
    energy: EnergyEvidence,
    benchmark: BenchmarkReport,
}

#[derive(Clone, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct DeviceIdentity {
    name: String,
    uuid: String,
    driver: String,
    compute_capability: String,
    total_memory_mib: u64,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct ClockPolicy {
    minimum_samples: usize,
    maximum_sm_clock_spread_mhz: u32,
    maximum_memory_clock_spread_mhz: u32,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct TelemetryEvidence {
    samples: Vec<TelemetrySample>,
    summary: TelemetrySummary,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct TelemetrySample {
    elapsed_milliseconds: f64,
    sm_clock_mhz: u32,
    memory_clock_mhz: u32,
    temperature_celsius: u32,
    board_power_watts: f64,
    memory_used_mib: u64,
    memory_free_mib: u64,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
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

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct EnergyEvidence {
    scope: String,
    denominator: String,
    idle_average_watts: f64,
    whole_board_joules: f64,
    above_idle_joules: f64,
    completion_tokens: u64,
    whole_board_joules_per_completion_token: f64,
    above_idle_joules_per_completion_token: f64,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct BenchmarkReport {
    schema_version: u32,
    suite: String,
    model: String,
    server_url: String,
    authority: String,
    status: String,
    samples_per_case: usize,
    long_context_enabled: bool,
    setup_completion_tokens: usize,
    cases: Vec<CaseReport>,
    in_progress_case: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct CaseReport {
    name: String,
    timing_boundary: String,
    cache_regime: String,
    external_concurrency: usize,
    observations: Vec<Observation>,
    summary: CaseSummary,
}

#[derive(Clone, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct Observation {
    request_count: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    completion_tokens: usize,
    visible_chunks: usize,
    ttft_ms: Option<f64>,
    mean_intertoken_ms: Option<f64>,
    e2e_ms: f64,
    completion_tokens_per_second: f64,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct CaseSummary {
    e2e_ms: MetricSummary,
    completion_tokens_per_second: MetricSummary,
    cached_prompt_fraction: MetricSummary,
    ttft_ms: Option<MetricSummary>,
    mean_intertoken_ms: Option<MetricSummary>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(deny_unknown_fields)]
struct MetricSummary {
    samples: usize,
    minimum: f64,
    median: f64,
    p95: f64,
    maximum: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ServerBaseline {
    schema_version: u32,
    suite: String,
    model: String,
    checkpoint_revision: String,
    device: String,
    driver: String,
    compute_capability: String,
    blessed_server_binary_sha256: String,
    long_context_enabled: bool,
    minimum_samples_per_case: usize,
    loaded_sm_clock_band_mhz: ClockBand,
    loaded_memory_clock_band_mhz: ClockBand,
    measurement_sm_clock_band_mhz: ClockBand,
    measurement_memory_clock_band_mhz: ClockBand,
    device_total_memory_mib: u64,
    resident_memory_reference_mib: u64,
    resident_memory_tolerance_mib: u64,
    resident_memory_enforced: bool,
    metrics: Vec<BaselineMetric>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClockBand {
    minimum: u32,
    maximum: u32,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineMetric {
    case: String,
    metric: String,
    unit: String,
    direction: Direction,
    reference: f64,
    relative_tolerance_percent: f64,
    absolute_tolerance: f64,
    enforced: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MetricKey {
    case: String,
    metric: String,
    unit: String,
    direction: Direction,
}

struct ExpectedCase {
    name: String,
    timing_boundary: &'static str,
    cache_regime: &'static str,
    concurrency: usize,
    streaming: bool,
}

fn read_candidate(path: &Path) -> Result<Candidate, Box<dyn Error>> {
    let report: AuthorityReport = serde_json::from_slice(&fs::read(path)?)?;
    validate_authority(report)
}

fn validate_authority(report: AuthorityReport) -> Result<Candidate, Box<dyn Error>> {
    if report.schema_version != AUTHORITY_SCHEMA || report.suite != "server" {
        return Err("server authority report has an unsupported schema or suite".into());
    }
    if report.status != "complete" || report.refusal.is_some() {
        return Err(format!(
            "server report status is `{}`; only complete, non-refused evidence is comparable or blessable",
            report.status
        )
        .into());
    }
    if Path::new(&report.snapshot)
        .file_name()
        .and_then(|value| value.to_str())
        != Some(REVISION)
    {
        return Err("server authority report does not use the pinned checkpoint revision".into());
    }
    validate_sha256(&report.server_binary_sha256, "server binary")?;
    if report.server_pid == 0
        || report.server_log.is_empty()
        || report.raw_benchmark_report.is_empty()
    {
        return Err("server authority report omitted lifecycle identity".into());
    }
    validate_device(&report.device)?;
    if report.clock_policy.minimum_samples != 3
        || report.clock_policy.maximum_sm_clock_spread_mhz != 50
        || report.clock_policy.maximum_memory_clock_spread_mhz != 250
    {
        return Err("server authority report used an unsupported clock policy".into());
    }
    validate_telemetry(&report.idle)?;
    validate_telemetry(&report.loaded_probe)?;
    let measurement = report
        .measurement
        .ok_or("complete server authority omitted measurement telemetry")?;
    validate_telemetry(&measurement)?;
    for (name, evidence) in [
        ("loaded probe", &report.loaded_probe),
        ("measurement", &measurement),
    ] {
        if evidence.summary.sm_clock_spread_mhz > report.clock_policy.maximum_sm_clock_spread_mhz
            || evidence.summary.memory_clock_spread_mhz
                > report.clock_policy.maximum_memory_clock_spread_mhz
        {
            return Err(format!("complete server {name} exceeds its declared clock policy").into());
        }
    }
    let energy = report
        .energy
        .ok_or("complete server authority omitted energy evidence")?;
    validate_energy(&energy)?;
    let benchmark = report
        .benchmark
        .ok_or("complete server authority omitted its raw benchmark")?;
    validate_benchmark(&benchmark)?;
    if completion_tokens(&benchmark) != Some(energy.completion_tokens) {
        return Err("server energy denominator differs from benchmark completion tokens".into());
    }
    Ok(Candidate {
        server_binary_sha256: report.server_binary_sha256,
        device: report.device,
        loaded_probe: report.loaded_probe,
        measurement,
        energy,
        benchmark,
    })
}

fn validate_device(device: &DeviceIdentity) -> Result<(), Box<dyn Error>> {
    if device.name != DEVICE || device.compute_capability != COMPUTE_CAPABILITY {
        return Err(format!(
            "server evidence targets {} cc {}, expected {DEVICE} cc {COMPUTE_CAPABILITY}",
            device.name, device.compute_capability
        )
        .into());
    }
    if device.uuid.is_empty() || device.driver.is_empty() || device.total_memory_mib == 0 {
        return Err("server device identity is incomplete".into());
    }
    Ok(())
}

fn validate_telemetry(evidence: &TelemetryEvidence) -> Result<(), Box<dyn Error>> {
    let summary = &evidence.summary;
    if evidence.samples.len() != summary.sample_count || summary.sample_count < 3 {
        return Err("server telemetry sample inventory differs from its summary".into());
    }
    if summary.sm_clock_min_mhz > summary.sm_clock_max_mhz
        || summary.sm_clock_spread_mhz != summary.sm_clock_max_mhz - summary.sm_clock_min_mhz
        || summary.memory_clock_min_mhz > summary.memory_clock_max_mhz
        || summary.memory_clock_spread_mhz
            != summary.memory_clock_max_mhz - summary.memory_clock_min_mhz
        || summary.temperature_min_celsius > summary.temperature_max_celsius
        || summary.power_min_watts > summary.power_max_watts
        || summary.memory_used_min_mib > summary.memory_used_max_mib
    {
        return Err("server telemetry summary ranges are inconsistent".into());
    }
    for (name, value) in [
        ("duration", summary.duration_seconds),
        ("minimum power", summary.power_min_watts),
        ("average power", summary.power_average_watts),
        ("maximum power", summary.power_max_watts),
        ("energy", summary.energy_joules),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("server telemetry {name} is not finite and positive").into());
        }
    }
    if summary.power_average_watts < summary.power_min_watts
        || summary.power_average_watts > summary.power_max_watts
    {
        return Err("server telemetry average power is outside its observed range".into());
    }
    for (index, sample) in evidence.samples.iter().enumerate() {
        if !sample.elapsed_milliseconds.is_finite()
            || sample.elapsed_milliseconds < 0.0
            || !positive(sample.board_power_watts)
            || sample.sm_clock_mhz == 0
            || sample.memory_clock_mhz == 0
            || sample.temperature_celsius == 0
            || sample.memory_used_mib == 0
            || sample.memory_free_mib == 0
        {
            return Err(format!("server telemetry sample {index} is invalid").into());
        }
        if index != 0
            && sample.elapsed_milliseconds <= evidence.samples[index - 1].elapsed_milliseconds
        {
            return Err("server telemetry timestamps are not strictly increasing".into());
        }
    }
    let first = evidence.samples.first().expect("at least three samples");
    let last = evidence.samples.last().expect("at least three samples");
    require_close(
        summary.duration_seconds,
        (last.elapsed_milliseconds - first.elapsed_milliseconds) / 1_000.0,
        "telemetry duration",
    )?;
    let average_power = evidence
        .samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .sum::<f64>()
        / evidence.samples.len() as f64;
    require_close(
        summary.power_average_watts,
        average_power,
        "telemetry average power",
    )?;
    let energy = evidence
        .samples
        .windows(2)
        .map(|pair| {
            let seconds = (pair[1].elapsed_milliseconds - pair[0].elapsed_milliseconds) / 1_000.0;
            seconds * (pair[0].board_power_watts + pair[1].board_power_watts) / 2.0
        })
        .sum::<f64>();
    require_close(summary.energy_joules, energy, "telemetry energy")?;
    let sm_min = evidence
        .samples
        .iter()
        .map(|sample| sample.sm_clock_mhz)
        .min()
        .expect("at least three samples");
    let sm_max = evidence
        .samples
        .iter()
        .map(|sample| sample.sm_clock_mhz)
        .max()
        .expect("at least three samples");
    let memory_min = evidence
        .samples
        .iter()
        .map(|sample| sample.memory_clock_mhz)
        .min()
        .expect("at least three samples");
    let memory_max = evidence
        .samples
        .iter()
        .map(|sample| sample.memory_clock_mhz)
        .max()
        .expect("at least three samples");
    let temperature_min = evidence
        .samples
        .iter()
        .map(|sample| sample.temperature_celsius)
        .min()
        .expect("at least three samples");
    let temperature_max = evidence
        .samples
        .iter()
        .map(|sample| sample.temperature_celsius)
        .max()
        .expect("at least three samples");
    let power_min = evidence
        .samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .reduce(f64::min)
        .expect("at least three samples");
    let power_max = evidence
        .samples
        .iter()
        .map(|sample| sample.board_power_watts)
        .reduce(f64::max)
        .expect("at least three samples");
    let used_min = evidence
        .samples
        .iter()
        .map(|sample| sample.memory_used_mib)
        .min()
        .expect("at least three samples");
    let used_max = evidence
        .samples
        .iter()
        .map(|sample| sample.memory_used_mib)
        .max()
        .expect("at least three samples");
    if (
        sm_min,
        sm_max,
        memory_min,
        memory_max,
        temperature_min,
        temperature_max,
        used_min,
        used_max,
    ) != (
        summary.sm_clock_min_mhz,
        summary.sm_clock_max_mhz,
        summary.memory_clock_min_mhz,
        summary.memory_clock_max_mhz,
        summary.temperature_min_celsius,
        summary.temperature_max_celsius,
        summary.memory_used_min_mib,
        summary.memory_used_max_mib,
    ) {
        return Err("server telemetry summary differs from its raw samples".into());
    }
    require_close(
        summary.power_min_watts,
        power_min,
        "telemetry minimum power",
    )?;
    require_close(
        summary.power_max_watts,
        power_max,
        "telemetry maximum power",
    )?;
    Ok(())
}

fn validate_energy(energy: &EnergyEvidence) -> Result<(), Box<dyn Error>> {
    if energy.scope != "whole board over the complete directly timed HTTP suite"
        || energy.denominator != "all completion tokens including reported benchmark setup requests"
        || energy.completion_tokens == 0
    {
        return Err("server energy scope or denominator changed".into());
    }
    for (name, value) in [
        ("idle power", energy.idle_average_watts),
        ("whole-board joules", energy.whole_board_joules),
        ("above-idle joules", energy.above_idle_joules),
        (
            "whole-board joules per completion token",
            energy.whole_board_joules_per_completion_token,
        ),
        (
            "above-idle joules per completion token",
            energy.above_idle_joules_per_completion_token,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("server {name} is invalid").into());
        }
    }
    if energy.above_idle_joules > energy.whole_board_joules {
        return Err("server above-idle energy exceeds whole-board energy".into());
    }
    require_close(
        energy.whole_board_joules_per_completion_token,
        energy.whole_board_joules / energy.completion_tokens as f64,
        "whole-board energy denominator",
    )?;
    require_close(
        energy.above_idle_joules_per_completion_token,
        energy.above_idle_joules / energy.completion_tokens as f64,
        "above-idle energy denominator",
    )?;
    Ok(())
}

fn validate_benchmark(report: &BenchmarkReport) -> Result<(), Box<dyn Error>> {
    if report.schema_version != BENCHMARK_SCHEMA
        || report.suite != "server-http"
        || report.model != MODEL
        || report.authority != RAW_AUTHORITY
        || report.status != "complete"
        || report.in_progress_case.is_some()
        || report.error.is_some()
        || report.server_url.is_empty()
        || report.setup_completion_tokens == 0
        || !(3..=40).contains(&report.samples_per_case)
    {
        return Err("server raw benchmark identity or terminal state is invalid".into());
    }
    let expected = expected_cases(report.long_context_enabled);
    if report.cases.len() != expected.len() {
        return Err(format!(
            "server benchmark contains {} cases, expected {}",
            report.cases.len(),
            expected.len()
        )
        .into());
    }
    for (case, expected) in report.cases.iter().zip(expected) {
        validate_case(case, &expected, report.samples_per_case)?;
    }
    Ok(())
}

fn validate_case(
    case: &CaseReport,
    expected: &ExpectedCase,
    samples: usize,
) -> Result<(), Box<dyn Error>> {
    if case.name != expected.name
        || case.timing_boundary != expected.timing_boundary
        || case.cache_regime != expected.cache_regime
        || case.external_concurrency != expected.concurrency
        || case.observations.len() != samples
    {
        return Err(format!(
            "server benchmark case `{}` changed its exact contract",
            case.name
        )
        .into());
    }
    for observation in &case.observations {
        if observation.request_count != expected.concurrency
            || observation.prompt_tokens == 0
            || observation.completion_tokens == 0
            || observation.cached_prompt_tokens > observation.prompt_tokens
            || !positive(observation.e2e_ms)
            || !positive(observation.completion_tokens_per_second)
        {
            return Err(format!(
                "server case `{}` contains an invalid observation",
                case.name
            )
            .into());
        }
        let expected_throughput =
            observation.completion_tokens as f64 * 1_000.0 / observation.e2e_ms;
        require_close(
            observation.completion_tokens_per_second,
            expected_throughput,
            "completion throughput",
        )?;
        if expected.cache_regime == "reported full prompt reuse" {
            if observation.cached_prompt_tokens != observation.prompt_tokens {
                return Err("full-prefix server case did not report complete reuse".into());
            }
        } else if observation.cached_prompt_tokens.saturating_mul(4) > observation.prompt_tokens {
            return Err("low-reuse server case exceeded its 25% cached-token ceiling".into());
        }
        if expected.streaming {
            if observation.visible_chunks == 0
                || !observation.ttft_ms.is_some_and(positive)
                || !observation.mean_intertoken_ms.is_some_and(positive)
            {
                return Err("streaming server observation omitted its visible timing seams".into());
            }
        } else if observation.visible_chunks != 0
            || observation.ttft_ms.is_some()
            || observation.mean_intertoken_ms.is_some()
        {
            return Err("blocking server observation exposed streaming-only seams".into());
        }
    }
    validate_summary(
        &case.summary.e2e_ms,
        case.observations.iter().map(|value| value.e2e_ms),
        samples,
    )?;
    validate_summary(
        &case.summary.completion_tokens_per_second,
        case.observations
            .iter()
            .map(|value| value.completion_tokens_per_second),
        samples,
    )?;
    validate_summary(
        &case.summary.cached_prompt_fraction,
        case.observations
            .iter()
            .map(|value| value.cached_prompt_tokens as f64 / value.prompt_tokens as f64),
        samples,
    )?;
    validate_optional_summary(
        case.summary.ttft_ms.as_ref(),
        case.observations.iter().map(|value| value.ttft_ms),
        samples,
    )?;
    validate_optional_summary(
        case.summary.mean_intertoken_ms.as_ref(),
        case.observations
            .iter()
            .map(|value| value.mean_intertoken_ms),
        samples,
    )
}

fn validate_summary(
    summary: &MetricSummary,
    values: impl IntoIterator<Item = f64>,
    samples: usize,
) -> Result<(), Box<dyn Error>> {
    let expected = summarize(values)?;
    if summary.samples != samples {
        return Err("server metric summary sample count changed".into());
    }
    for (candidate, expected) in [
        (summary.minimum, expected.minimum),
        (summary.median, expected.median),
        (summary.p95, expected.p95),
        (summary.maximum, expected.maximum),
    ] {
        require_close(candidate, expected, "metric summary")?;
    }
    Ok(())
}

fn validate_optional_summary(
    summary: Option<&MetricSummary>,
    values: impl IntoIterator<Item = Option<f64>>,
    samples: usize,
) -> Result<(), Box<dyn Error>> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.iter().all(Option::is_none) {
        return if summary.is_none() {
            Ok(())
        } else {
            Err("server summary contains a metric absent from every observation".into())
        };
    }
    if values.iter().any(Option::is_none) {
        return Err("server optional metric inventory differs across observations".into());
    }
    match summary {
        Some(summary) => validate_summary(summary, values.into_iter().flatten(), samples),
        None => Err("server summary omitted an observed optional metric".into()),
    }
}

fn summarize(values: impl IntoIterator<Item = f64>) -> Result<MetricSummary, Box<dyn Error>> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("server metric samples must be nonempty, finite, and nonnegative".into());
    }
    values.sort_by(f64::total_cmp);
    let samples = values.len();
    let median = if samples % 2 == 0 {
        (values[samples / 2 - 1] + values[samples / 2]) / 2.0
    } else {
        values[samples / 2]
    };
    let p95 = ((samples as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(samples - 1);
    Ok(MetricSummary {
        samples,
        minimum: values[0],
        median,
        p95: values[p95],
        maximum: values[samples - 1],
    })
}

fn expected_cases(long_context: bool) -> Vec<ExpectedCase> {
    let mut cases = vec![
        ExpectedCase {
            name: "stream/full-prefix".into(),
            timing_boundary: "SSE request start through DONE",
            cache_regime: "reported full prompt reuse",
            concurrency: 1,
            streaming: true,
        },
        ExpectedCase {
            name: "stream/low-reuse-256".into(),
            timing_boundary: "SSE request start through DONE",
            cache_regime: "reported cached prompt tokens at most 25%",
            concurrency: 1,
            streaming: true,
        },
    ];
    cases.extend((1..=8).map(|concurrency| ExpectedCase {
        name: format!("blocking/external-concurrency-{concurrency}"),
        timing_boundary: "barrier release through all blocking responses",
        cache_regime: "reported cached prompt tokens at most 25% per request",
        concurrency,
        streaming: false,
    }));
    if long_context {
        cases.extend([4_096, 16_384, 65_536, 178_000].map(|tokens| ExpectedCase {
            name: format!("stream/long-context-{tokens}"),
            timing_boundary: "SSE request start through DONE",
            cache_regime: "reported cached prompt tokens at most 25%",
            concurrency: 1,
            streaming: true,
        }));
    }
    cases
}

fn completion_tokens(report: &BenchmarkReport) -> Option<u64> {
    report.cases.iter().try_fold(
        u64::try_from(report.setup_completion_tokens).ok()?,
        |total, case| {
            case.observations
                .iter()
                .try_fold(total, |total, observation| {
                    total.checked_add(u64::try_from(observation.completion_tokens).ok()?)
                })
        },
    )
}

fn candidate_metrics(candidate: &Candidate) -> BTreeMap<MetricKey, f64> {
    let mut metrics = BTreeMap::new();
    for case in &candidate.benchmark.cases {
        insert_metric(
            &mut metrics,
            &case.name,
            "e2e_median",
            "ms",
            Direction::LowerIsBetter,
            case.summary.e2e_ms.median,
        );
        insert_metric(
            &mut metrics,
            &case.name,
            "completion_throughput_median",
            "tokens_per_second",
            Direction::HigherIsBetter,
            case.summary.completion_tokens_per_second.median,
        );
        if let Some(summary) = &case.summary.ttft_ms {
            insert_metric(
                &mut metrics,
                &case.name,
                "ttft_median",
                "ms",
                Direction::LowerIsBetter,
                summary.median,
            );
        }
        if let Some(summary) = &case.summary.mean_intertoken_ms {
            insert_metric(
                &mut metrics,
                &case.name,
                "mean_intertoken_median",
                "ms",
                Direction::LowerIsBetter,
                summary.median,
            );
        }
    }
    insert_metric(
        &mut metrics,
        "suite",
        "whole_board_energy_per_completion_token",
        "joules_per_token",
        Direction::LowerIsBetter,
        candidate.energy.whole_board_joules_per_completion_token,
    );
    insert_metric(
        &mut metrics,
        "suite",
        "above_idle_energy_per_completion_token",
        "joules_per_token",
        Direction::LowerIsBetter,
        candidate.energy.above_idle_joules_per_completion_token,
    );
    metrics
}

fn insert_metric(
    metrics: &mut BTreeMap<MetricKey, f64>,
    case: &str,
    metric: &str,
    unit: &str,
    direction: Direction,
    value: f64,
) {
    let replaced = metrics.insert(
        MetricKey {
            case: case.into(),
            metric: metric.into(),
            unit: unit.into(),
            direction,
        },
        value,
    );
    debug_assert!(replaced.is_none());
}

fn baseline_metrics(
    baseline: &ServerBaseline,
) -> Result<BTreeMap<MetricKey, &BaselineMetric>, Box<dyn Error>> {
    let mut metrics = BTreeMap::new();
    for metric in &baseline.metrics {
        if !nonnegative(metric.reference)
            || !nonnegative(metric.relative_tolerance_percent)
            || !nonnegative(metric.absolute_tolerance)
        {
            return Err("server baseline contains an invalid metric value or tolerance".into());
        }
        let key = MetricKey {
            case: metric.case.clone(),
            metric: metric.metric.clone(),
            unit: metric.unit.clone(),
            direction: metric.direction,
        };
        if metrics.insert(key, metric).is_some() {
            return Err("server baseline contains a duplicate metric".into());
        }
    }
    if metrics.is_empty() {
        return Err("server baseline contains no performance metrics".into());
    }
    Ok(metrics)
}

fn validate_baseline(baseline: &ServerBaseline) -> Result<(), Box<dyn Error>> {
    if baseline.schema_version != BASELINE_SCHEMA
        || baseline.suite != "server-http"
        || baseline.model != MODEL
        || baseline.checkpoint_revision != REVISION
        || baseline.device != DEVICE
        || baseline.compute_capability != COMPUTE_CAPABILITY
        || baseline.driver.is_empty()
        || baseline.device_total_memory_mib == 0
        || !(MINIMUM_BASELINE_SAMPLES..=40).contains(&baseline.minimum_samples_per_case)
    {
        return Err("server performance baseline identity is invalid".into());
    }
    validate_sha256(
        &baseline.blessed_server_binary_sha256,
        "blessed server binary",
    )?;
    for (name, band) in [
        ("loaded SM", baseline.loaded_sm_clock_band_mhz),
        ("loaded memory", baseline.loaded_memory_clock_band_mhz),
        ("measurement SM", baseline.measurement_sm_clock_band_mhz),
        (
            "measurement memory",
            baseline.measurement_memory_clock_band_mhz,
        ),
    ] {
        if band.minimum > band.maximum {
            return Err(format!("server baseline {name} clock band is inverted").into());
        }
    }
    let _ = baseline_metrics(baseline)?;
    Ok(())
}

fn validate_environment(
    candidate: &Candidate,
    baseline: &ServerBaseline,
) -> Result<(), Box<dyn Error>> {
    for (name, candidate, authority) in [
        ("device", &candidate.device.name, &baseline.device),
        ("driver", &candidate.device.driver, &baseline.driver),
        (
            "compute capability",
            &candidate.device.compute_capability,
            &baseline.compute_capability,
        ),
    ] {
        if candidate != authority {
            return Err(format!(
                "server performance {name} is `{candidate}`, baseline requires `{authority}`"
            )
            .into());
        }
    }
    if candidate.device.total_memory_mib != baseline.device_total_memory_mib {
        return Err("server performance device-memory capacity differs from baseline".into());
    }
    if candidate.benchmark.long_context_enabled != baseline.long_context_enabled {
        return Err("server performance long-context inventory differs from baseline".into());
    }
    if candidate.benchmark.samples_per_case < baseline.minimum_samples_per_case {
        return Err(format!(
            "server report has {} samples per case, baseline requires at least {}",
            candidate.benchmark.samples_per_case, baseline.minimum_samples_per_case
        )
        .into());
    }
    for (name, minimum, maximum, band) in [
        (
            "loaded SM",
            candidate.loaded_probe.summary.sm_clock_min_mhz,
            candidate.loaded_probe.summary.sm_clock_max_mhz,
            baseline.loaded_sm_clock_band_mhz,
        ),
        (
            "loaded memory",
            candidate.loaded_probe.summary.memory_clock_min_mhz,
            candidate.loaded_probe.summary.memory_clock_max_mhz,
            baseline.loaded_memory_clock_band_mhz,
        ),
        (
            "measurement SM",
            candidate.measurement.summary.sm_clock_min_mhz,
            candidate.measurement.summary.sm_clock_max_mhz,
            baseline.measurement_sm_clock_band_mhz,
        ),
        (
            "measurement memory",
            candidate.measurement.summary.memory_clock_min_mhz,
            candidate.measurement.summary.memory_clock_max_mhz,
            baseline.measurement_memory_clock_band_mhz,
        ),
    ] {
        if minimum < band.minimum || maximum > band.maximum {
            return Err(format!(
                "server {name} clock range {minimum}..={maximum} MHz is outside baseline band {}..={} MHz",
                band.minimum, band.maximum
            )
            .into());
        }
    }
    Ok(())
}

fn metric_passes(candidate: f64, baseline: &BaselineMetric) -> bool {
    let tolerance = (baseline.reference * baseline.relative_tolerance_percent / 100.0)
        .max(baseline.absolute_tolerance);
    match baseline.direction {
        Direction::LowerIsBetter => candidate <= baseline.reference + tolerance,
        Direction::HigherIsBetter => candidate >= (baseline.reference - tolerance).max(0.0),
    }
}

fn default_controls(key: &MetricKey) -> (f64, f64, bool) {
    let absolute = match key.unit.as_str() {
        "ms" => DEFAULT_LATENCY_TOLERANCE_MS,
        "tokens_per_second" => DEFAULT_THROUGHPUT_TOLERANCE,
        "joules_per_token" => DEFAULT_ENERGY_TOLERANCE,
        _ => 0.0,
    };
    (DEFAULT_RELATIVE_TOLERANCE_PERCENT, absolute, true)
}

const fn padded_band(minimum: u32, maximum: u32, padding: u32) -> ClockBand {
    ClockBand {
        minimum: minimum.saturating_sub(padding),
        maximum: maximum.saturating_add(padding),
    }
}

fn validate_sha256(value: &str, label: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} SHA-256 is not lowercase hexadecimal").into());
    }
    Ok(())
}

fn require_close(candidate: f64, expected: f64, label: &str) -> Result<(), Box<dyn Error>> {
    let tolerance = expected.abs().max(1.0) * 1e-9;
    if !candidate.is_finite() || (candidate - expected).abs() > tolerance {
        return Err(format!("server {label} is inconsistent with raw observations").into());
    }
    Ok(())
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::{
        AUTHORITY_SCHEMA, AuthorityReport, BENCHMARK_SCHEMA, BaselineMetric, BenchmarkReport,
        COMPUTE_CAPABILITY, CaseReport, CaseSummary, ClockPolicy, DEVICE, DeviceIdentity,
        Direction, EnergyEvidence, MODEL, MetricKey, Observation, RAW_AUTHORITY, REVISION,
        TelemetryEvidence, TelemetrySample, TelemetrySummary, baseline_path, bless,
        candidate_metrics, compare, default_controls, expected_cases, metric_passes, preflight,
        summarize, validate_authority,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn exact_case_inventory_separates_short_and_long_context() {
        let short = expected_cases(false);
        let long = expected_cases(true);
        assert_eq!(short.len(), 10);
        assert_eq!(long.len(), 14);
        assert_eq!(short[0].name, "stream/full-prefix");
        assert_eq!(short[9].name, "blocking/external-concurrency-8");
        assert_eq!(long[13].name, "stream/long-context-178000");
        assert_ne!(baseline_path(false), baseline_path(true));
    }

    #[test]
    fn summary_matches_even_median_and_nearest_rank_p95() {
        let summary = summarize([4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.median, 2.5);
        assert_eq!(summary.p95, 4.0);
        assert_eq!(summary.maximum, 4.0);
    }

    #[test]
    fn regression_directions_and_default_controls_are_explicit() {
        let latency = BaselineMetric {
            case: "case".into(),
            metric: "ttft".into(),
            unit: "ms".into(),
            direction: Direction::LowerIsBetter,
            reference: 100.0,
            relative_tolerance_percent: 10.0,
            absolute_tolerance: 0.5,
            enforced: true,
        };
        assert!(metric_passes(110.0, &latency));
        assert!(!metric_passes(110.01, &latency));
        let throughput = BaselineMetric {
            direction: Direction::HigherIsBetter,
            ..latency
        };
        assert!(metric_passes(90.0, &throughput));
        assert!(!metric_passes(89.99, &throughput));

        let key = MetricKey {
            case: "case".into(),
            metric: "energy".into(),
            unit: "joules_per_token".into(),
            direction: Direction::LowerIsBetter,
        };
        assert_eq!(default_controls(&key), (15.0, 0.01, true));
    }

    #[test]
    fn complete_authority_validates_every_external_inventory() {
        let candidate = validate_authority(authority("complete")).unwrap();
        assert_eq!(candidate_metrics(&candidate).len(), 26);

        let error = validate_authority(authority("refused_loaded_clocks"))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("only complete, non-refused"));
    }

    #[test]
    fn recomputed_summaries_refuse_tampered_medians() {
        let mut report = authority("complete");
        report.benchmark.as_mut().unwrap().cases[0]
            .summary
            .e2e_ms
            .median += 1.0;
        assert!(validate_authority(report).is_err());
    }

    #[test]
    fn raw_telemetry_refuses_tampered_clock_summaries() {
        let mut report = authority("complete");
        report.loaded_probe.summary.sm_clock_min_mhz += 1;
        report.loaded_probe.summary.sm_clock_spread_mhz -= 1;
        assert!(validate_authority(report).is_err());
    }

    #[test]
    fn explicit_blessing_round_trips_into_a_checked_comparison() {
        let directory = std::env::temp_dir().join(format!(
            "tuisko-server-performance-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        let report = directory.join("report.json");
        let baseline = directory.join("baseline.json");
        fs::write(&report, serde_json::to_vec(&authority("complete")).unwrap()).unwrap();

        bless(&report, &baseline).unwrap();
        preflight(&baseline).unwrap();
        compare(&report, &baseline).unwrap();

        fs::remove_dir_all(directory).unwrap();
    }

    fn authority(status: &str) -> AuthorityReport {
        let benchmark = benchmark();
        let completion_tokens = benchmark
            .cases
            .iter()
            .flat_map(|case| &case.observations)
            .map(|observation| observation.completion_tokens as u64)
            .sum::<u64>()
            + benchmark.setup_completion_tokens as u64;
        AuthorityReport {
            schema_version: AUTHORITY_SCHEMA,
            suite: "server".into(),
            status: status.into(),
            snapshot: format!("/snapshot/{REVISION}"),
            server_log: "target/server.log".into(),
            raw_benchmark_report: "target/server-raw.json".into(),
            server_binary_sha256: "a".repeat(64),
            device: DeviceIdentity {
                name: DEVICE.into(),
                uuid: "GPU-test".into(),
                driver: "580.173.02".into(),
                compute_capability: COMPUTE_CAPABILITY.into(),
                total_memory_mib: 32_607,
            },
            server_pid: 42,
            clock_policy: ClockPolicy {
                minimum_samples: 3,
                maximum_sm_clock_spread_mhz: 50,
                maximum_memory_clock_spread_mhz: 250,
            },
            idle: telemetry(),
            loaded_probe: telemetry(),
            measurement: Some(telemetry()),
            energy: Some(EnergyEvidence {
                scope: "whole board over the complete directly timed HTTP suite".into(),
                denominator: "all completion tokens including reported benchmark setup requests"
                    .into(),
                idle_average_watts: 70.0,
                whole_board_joules: completion_tokens as f64 * 2.0,
                above_idle_joules: completion_tokens as f64,
                completion_tokens,
                whole_board_joules_per_completion_token: 2.0,
                above_idle_joules_per_completion_token: 1.0,
            }),
            refusal: (status != "complete").then(|| "clock drift".into()),
            benchmark: Some(benchmark),
        }
    }

    fn telemetry() -> TelemetryEvidence {
        TelemetryEvidence {
            samples: vec![
                TelemetrySample {
                    elapsed_milliseconds: 0.0,
                    sm_clock_mhz: 2_182,
                    memory_clock_mhz: 13_801,
                    temperature_celsius: 40,
                    board_power_watts: 60.0,
                    memory_used_mib: 30_554,
                    memory_free_mib: 2_053,
                },
                TelemetrySample {
                    elapsed_milliseconds: 500.0,
                    sm_clock_mhz: 2_190,
                    memory_clock_mhz: 13_801,
                    temperature_celsius: 45,
                    board_power_watts: 180.0,
                    memory_used_mib: 30_554,
                    memory_free_mib: 2_053,
                },
                TelemetrySample {
                    elapsed_milliseconds: 1_000.0,
                    sm_clock_mhz: 2_197,
                    memory_clock_mhz: 13_801,
                    temperature_celsius: 50,
                    board_power_watts: 300.0,
                    memory_used_mib: 30_554,
                    memory_free_mib: 2_053,
                },
            ],
            summary: TelemetrySummary {
                sample_count: 3,
                duration_seconds: 1.0,
                sm_clock_min_mhz: 2_182,
                sm_clock_max_mhz: 2_197,
                sm_clock_spread_mhz: 15,
                memory_clock_min_mhz: 13_801,
                memory_clock_max_mhz: 13_801,
                memory_clock_spread_mhz: 0,
                temperature_min_celsius: 40,
                temperature_max_celsius: 50,
                power_min_watts: 60.0,
                power_average_watts: 180.0,
                power_max_watts: 300.0,
                energy_joules: 180.0,
                memory_used_min_mib: 30_554,
                memory_used_max_mib: 30_554,
            },
        }
    }

    fn benchmark() -> BenchmarkReport {
        BenchmarkReport {
            schema_version: BENCHMARK_SCHEMA,
            suite: "server-http".into(),
            model: MODEL.into(),
            server_url: "http://127.0.0.1:12345".into(),
            authority: RAW_AUTHORITY.into(),
            status: "complete".into(),
            samples_per_case: 5,
            long_context_enabled: false,
            setup_completion_tokens: 64,
            cases: expected_cases(false)
                .into_iter()
                .map(|expected| {
                    let completion_tokens = if expected.streaming { 32 } else { 8 };
                    let prompt_tokens = 300 * expected.concurrency;
                    let cached_prompt_tokens = if expected.name == "stream/full-prefix" {
                        prompt_tokens
                    } else {
                        0
                    };
                    let e2e_ms = 10.0 + expected.concurrency as f64;
                    let throughput = completion_tokens as f64 * 1_000.0 / e2e_ms;
                    let observations = (0..5)
                        .map(|_| Observation {
                            request_count: expected.concurrency,
                            prompt_tokens,
                            cached_prompt_tokens,
                            completion_tokens,
                            visible_chunks: usize::from(expected.streaming),
                            ttft_ms: expected.streaming.then_some(5.0),
                            mean_intertoken_ms: expected.streaming.then_some(0.5),
                            e2e_ms,
                            completion_tokens_per_second: throughput,
                        })
                        .collect::<Vec<_>>();
                    CaseReport {
                        name: expected.name,
                        timing_boundary: expected.timing_boundary.into(),
                        cache_regime: expected.cache_regime.into(),
                        external_concurrency: expected.concurrency,
                        summary: CaseSummary {
                            e2e_ms: summarize(observations.iter().map(|value| value.e2e_ms))
                                .unwrap(),
                            completion_tokens_per_second: summarize(
                                observations
                                    .iter()
                                    .map(|value| value.completion_tokens_per_second),
                            )
                            .unwrap(),
                            cached_prompt_fraction: summarize(observations.iter().map(|value| {
                                value.cached_prompt_tokens as f64 / value.prompt_tokens as f64
                            }))
                            .unwrap(),
                            ttft_ms: expected.streaming.then(|| summarize([5.0; 5]).unwrap()),
                            mean_intertoken_ms: expected
                                .streaming
                                .then(|| summarize([0.5; 5]).unwrap()),
                        },
                        observations,
                    }
                })
                .collect(),
            in_progress_case: None,
            error: None,
        }
    }
}
