//! Machine-readable exact-target performance baselines.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

const BASELINE_SCHEMA: u32 = 5;
const REPORT_SCHEMA: u32 = 7;
const CLOCK_PADDING_MHZ: u32 = 15;
const MEMORY_CLOCK_PADDING_MHZ: u32 = 50;
const DEVICE_RELATIVE_TOLERANCE_PERCENT: f64 = 5.0;
const DEVICE_ABSOLUTE_TOLERANCE_MICROSECONDS: f64 = 0.05;
const HOST_RELATIVE_TOLERANCE_PERCENT: f64 = 15.0;
const HOST_ABSOLUTE_TOLERANCE_MICROSECONDS: f64 = 0.10;
const OBSERVED_MEMORY_TOLERANCE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Deserialize, Serialize)]
struct PerformanceReport {
    schema_version: u32,
    suite: String,
    device: String,
    driver_version: String,
    compute_capability: String,
    clock_policy: String,
    binary_sha256: String,
    generator_baseline_sha256: String,
    sm_clock_min_mhz: u32,
    sm_clock_max_mhz: u32,
    memory_clock_min_mhz: u32,
    memory_clock_max_mhz: u32,
    samples: usize,
    warmup_launches: u64,
    case_policy: String,
    #[serde(default)]
    selected_batch_size: Option<u32>,
    timing_scope: String,
    power_scope: String,
    metrics: Vec<ReportMetric>,
    memory: ReportMemory,
}

#[derive(Deserialize, Serialize)]
struct ReportMetric {
    route: String,
    shape: String,
    workload: Workload,
    measurement: String,
    median_microseconds: f64,
    operations_per_interval: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct Workload {
    scope: String,
    phase: String,
    batch_size: Option<u32>,
    concurrency: Option<u32>,
    active_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    context_tokens: Option<u64>,
    output_tokens: Option<u64>,
    device_cache: String,
    prefix_cache: Option<String>,
    execution: String,
}

#[derive(Deserialize, Serialize)]
struct ReportMemory {
    device_total_bytes: u64,
    metrics: Vec<ReportMemoryMetric>,
}

#[derive(Deserialize, Serialize)]
struct ReportMemoryMetric {
    name: String,
    measurement: String,
    kind: Option<String>,
    scaling: Option<String>,
    bytes: u64,
    comparison: String,
}

#[derive(Deserialize, Serialize)]
struct PerformanceBaseline {
    schema_version: u32,
    suite: String,
    device: String,
    driver_version: String,
    compute_capability: String,
    clock_policy: String,
    blessed_binary_sha256: String,
    generator_baseline_sha256: String,
    sm_clock_band_mhz: ClockBand,
    memory_clock_band_mhz: ClockBand,
    minimum_samples: usize,
    warmup_launches: u64,
    case_policy: String,
    timing_scope: String,
    power_scope: String,
    metrics: Vec<BaselineMetric>,
    memory: MemoryBaseline,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct ClockBand {
    minimum: u32,
    maximum: u32,
}

#[derive(Clone, Deserialize, Serialize)]
struct BaselineMetric {
    route: String,
    shape: String,
    workload: Workload,
    measurement: String,
    reference_microseconds: f64,
    relative_tolerance_percent: f64,
    absolute_tolerance_microseconds: f64,
    operations_per_interval: u64,
    enforced: bool,
}

#[derive(Deserialize, Serialize)]
struct MemoryBaseline {
    device_total_bytes: u64,
    metrics: Vec<BaselineMemoryMetric>,
}

#[derive(Clone, Deserialize, Serialize)]
struct BaselineMemoryMetric {
    name: String,
    measurement: String,
    kind: Option<String>,
    scaling: Option<String>,
    comparison: String,
    reference_bytes: u64,
    absolute_tolerance_bytes: u64,
    enforced: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiagnosticComparison {
    pub authoritative: bool,
    pub suite: String,
    pub case_policy: String,
    pub selected_batch_size: Option<u32>,
    pub generator_provenance_changed: bool,
    pub candidate_generator_baseline_sha256: String,
    pub authority_generator_baseline_sha256: String,
    pub timing_regressions: usize,
    pub memory_regressions: usize,
    pub timing_metrics: Vec<DiagnosticTimingMetric>,
    pub memory_metrics: Vec<DiagnosticMemoryMetric>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiagnosticTimingMetric {
    route: String,
    shape: String,
    measurement: String,
    reference_microseconds: f64,
    candidate_microseconds: f64,
    delta_percent: f64,
    maximum_microseconds: f64,
    enforced: bool,
    regressed: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct DiagnosticMemoryMetric {
    name: String,
    measurement: String,
    reference_bytes: u64,
    candidate_bytes: u64,
    delta_bytes: i128,
    enforced: bool,
    regressed: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MetricKey {
    route: String,
    shape: String,
    workload: Workload,
    measurement: String,
}

impl MetricKey {
    fn from_report(metric: &ReportMetric) -> Self {
        Self {
            route: metric.route.clone(),
            shape: metric.shape.clone(),
            workload: metric.workload.clone(),
            measurement: metric.measurement.clone(),
        }
    }

    fn from_baseline(metric: &BaselineMetric) -> Self {
        Self {
            route: metric.route.clone(),
            shape: metric.shape.clone(),
            workload: metric.workload.clone(),
            measurement: metric.measurement.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MemoryMetricKey {
    name: String,
    measurement: String,
}

impl MemoryMetricKey {
    fn from_report(metric: &ReportMemoryMetric) -> Self {
        Self {
            name: metric.name.clone(),
            measurement: metric.measurement.clone(),
        }
    }

    fn from_baseline(metric: &BaselineMemoryMetric) -> Self {
        Self {
            name: metric.name.clone(),
            measurement: metric.measurement.clone(),
        }
    }
}

pub(crate) fn compare(report_path: &Path, baseline_path: &Path) -> Result<(), Box<dyn Error>> {
    let report = read_report(report_path)?;
    let baseline: PerformanceBaseline = serde_json::from_slice(&fs::read(baseline_path).map_err(
        |error| {
            format!(
                "could not read performance baseline {}: {error}; run the matching `cargo run -p xtask -- perf bless <suite>` explicitly",
                baseline_path.display()
            )
        },
    )?)?;
    validate_environment(&report, &baseline)?;

    let candidates = report_metrics(&report)?;
    let authorities = baseline_metrics(&baseline)?;
    if candidates.keys().ne(authorities.keys()) {
        return Err("performance report and baseline metric inventories differ".into());
    }

    println!(
        "status route                            shape metric               reference  candidate    delta"
    );
    let mut failures = 0usize;
    for (key, candidate) in &candidates {
        let authority = authorities
            .get(key)
            .expect("matching metric inventories were checked");
        if candidate.operations_per_interval != authority.operations_per_interval {
            return Err(format!(
                "{} {} {} uses {} operations per interval, baseline requires {}",
                key.route,
                key.shape,
                key.measurement,
                candidate.operations_per_interval,
                authority.operations_per_interval
            )
            .into());
        }
        let delta_percent =
            (candidate.median_microseconds / authority.reference_microseconds - 1.0) * 100.0;
        let maximum = maximum_allowed(authority);
        let passed = !authority.enforced || candidate.median_microseconds <= maximum;
        let status = if !authority.enforced {
            "INFO"
        } else if passed {
            "PASS"
        } else {
            failures += 1;
            "FAIL"
        };
        println!(
            "{status:<6} {:<32} {:<5} {:<19} {:>8.3} us {:>8.3} us {delta_percent:>+7.2}%",
            key.route,
            key.shape,
            key.measurement,
            authority.reference_microseconds,
            candidate.median_microseconds,
        );
    }
    let memory_candidates = report_memory_metrics(&report.memory)?;
    let memory_authorities = baseline_memory_metrics(&baseline.memory)?;
    if memory_candidates.keys().ne(memory_authorities.keys()) {
        return Err("performance report and baseline memory metric inventories differ".into());
    }
    println!();
    println!("status memory                                      reference    candidate");
    for (key, candidate) in &memory_candidates {
        let authority = memory_authorities
            .get(key)
            .expect("matching memory metric inventories were checked");
        require_memory_contract(candidate, authority)?;
        let passed = !authority.enforced || memory_passes(candidate.bytes, authority)?;
        let status = if !authority.enforced {
            "INFO"
        } else if passed {
            "PASS"
        } else {
            failures += 1;
            "FAIL"
        };
        println!(
            "{status:<6} {:<43} {:>9.2} MiB {:>9.2} MiB",
            key.name,
            authority.reference_bytes as f64 / (1024.0 * 1024.0),
            candidate.bytes as f64 / (1024.0 * 1024.0),
        );
    }
    if failures != 0 {
        return Err(format!("{failures} enforced timing or memory metrics regressed").into());
    }

    println!(
        "performance gate passed: {} timing and {} memory metrics",
        candidates.len(),
        memory_candidates.len()
    );
    Ok(())
}

pub(crate) fn diagnose(
    report_path: &Path,
    baseline_path: &Path,
) -> Result<DiagnosticComparison, Box<dyn Error>> {
    let report = read_report(report_path)?;
    let baseline: PerformanceBaseline =
        serde_json::from_slice(&fs::read(baseline_path).map_err(|error| {
            format!(
                "could not read performance baseline {}: {error}",
                baseline_path.display()
            )
        })?)?;
    validate_diagnostic_environment(&report, &baseline)?;

    let candidates = report_metrics(&report)?;
    let authorities = baseline_metrics(&baseline)?;
    match report.case_policy.as_str() {
        "complete_inventory" => {
            if candidates.keys().ne(authorities.keys()) {
                return Err("performance report and baseline metric inventories differ".into());
            }
        }
        "diagnostic_subset" => {
            let selected = report
                .selected_batch_size
                .ok_or("diagnostic subset report omitted its selected batch size")?;
            if !(1..=8).contains(&selected) {
                return Err(
                    format!("diagnostic subset selected invalid batch size {selected}").into(),
                );
            }
            if candidates.is_empty() {
                return Err("diagnostic subset report contains no timing metrics".into());
            }
            for key in candidates.keys() {
                if key.workload.batch_size != Some(selected) {
                    return Err(format!(
                        "diagnostic subset metric {} {} {} is not B={selected}",
                        key.route, key.shape, key.measurement
                    )
                    .into());
                }
                if !authorities.contains_key(key) {
                    return Err(format!(
                        "diagnostic subset metric {} {} {} has no matching authority",
                        key.route, key.shape, key.measurement
                    )
                    .into());
                }
            }
        }
        policy => {
            return Err(format!("unsupported diagnostic case policy `{policy}`").into());
        }
    }

    println!(
        "status route                            shape metric               reference  candidate    delta"
    );
    let mut timing_regressions = 0usize;
    let mut timing_metrics = Vec::with_capacity(candidates.len());
    for (key, candidate) in &candidates {
        let authority = authorities
            .get(key)
            .expect("diagnostic metric authority was checked");
        if candidate.operations_per_interval != authority.operations_per_interval {
            return Err(format!(
                "{} {} {} uses {} operations per interval, baseline requires {}",
                key.route,
                key.shape,
                key.measurement,
                candidate.operations_per_interval,
                authority.operations_per_interval
            )
            .into());
        }
        let delta_percent =
            (candidate.median_microseconds / authority.reference_microseconds - 1.0) * 100.0;
        let maximum = maximum_allowed(authority);
        let regressed = authority.enforced && candidate.median_microseconds > maximum;
        if regressed {
            timing_regressions += 1;
        }
        let status = if !authority.enforced {
            "INFO"
        } else if regressed {
            "REGRESS"
        } else {
            "WITHIN"
        };
        println!(
            "{status:<7} {:<32} {:<5} {:<19} {:>8.3} us {:>8.3} us {delta_percent:>+7.2}%",
            key.route,
            key.shape,
            key.measurement,
            authority.reference_microseconds,
            candidate.median_microseconds,
        );
        timing_metrics.push(DiagnosticTimingMetric {
            route: key.route.clone(),
            shape: key.shape.clone(),
            measurement: key.measurement.clone(),
            reference_microseconds: authority.reference_microseconds,
            candidate_microseconds: candidate.median_microseconds,
            delta_percent,
            maximum_microseconds: maximum,
            enforced: authority.enforced,
            regressed,
        });
    }

    let memory_candidates = report_memory_metrics(&report.memory)?;
    let memory_authorities = baseline_memory_metrics(&baseline.memory)?;
    if memory_candidates.keys().ne(memory_authorities.keys()) {
        return Err("performance report and baseline memory metric inventories differ".into());
    }
    println!();
    println!("status memory                                      reference    candidate");
    let mut memory_regressions = 0usize;
    let mut memory_metrics = Vec::with_capacity(memory_candidates.len());
    for (key, candidate) in &memory_candidates {
        let authority = memory_authorities
            .get(key)
            .expect("matching memory metric inventories were checked");
        require_memory_contract(candidate, authority)?;
        let regressed = authority.enforced && !memory_passes(candidate.bytes, authority)?;
        if regressed {
            memory_regressions += 1;
        }
        let status = if !authority.enforced {
            "INFO"
        } else if regressed {
            "REGRESS"
        } else {
            "WITHIN"
        };
        println!(
            "{status:<7} {:<43} {:>9.2} MiB {:>9.2} MiB",
            key.name,
            authority.reference_bytes as f64 / (1024.0 * 1024.0),
            candidate.bytes as f64 / (1024.0 * 1024.0),
        );
        memory_metrics.push(DiagnosticMemoryMetric {
            name: key.name.clone(),
            measurement: key.measurement.clone(),
            reference_bytes: authority.reference_bytes,
            candidate_bytes: candidate.bytes,
            delta_bytes: candidate.bytes as i128 - authority.reference_bytes as i128,
            enforced: authority.enforced,
            regressed,
        });
    }

    let generator_provenance_changed =
        report.generator_baseline_sha256 != baseline.generator_baseline_sha256;
    println!();
    println!(
        "diagnostic only: {} timing and {} memory metrics, {} enforced regressions, generator provenance {}",
        timing_metrics.len(),
        memory_metrics.len(),
        timing_regressions + memory_regressions,
        if generator_provenance_changed {
            "changed"
        } else {
            "matched"
        }
    );

    Ok(DiagnosticComparison {
        authoritative: false,
        suite: report.suite,
        case_policy: report.case_policy,
        selected_batch_size: report.selected_batch_size,
        generator_provenance_changed,
        candidate_generator_baseline_sha256: report.generator_baseline_sha256,
        authority_generator_baseline_sha256: baseline.generator_baseline_sha256,
        timing_regressions,
        memory_regressions,
        timing_metrics,
        memory_metrics,
    })
}

pub(crate) fn bless(report_path: &Path, baseline_path: &Path) -> Result<(), Box<dyn Error>> {
    let report = read_report(report_path)?;
    if report.clock_policy != "controlled" {
        return Err(format!(
            "cannot bless a performance report with clock policy `{}`",
            report.clock_policy
        )
        .into());
    }
    if report.case_policy != "complete_inventory" {
        return Err(format!(
            "cannot bless a performance report with case policy `{}`",
            report.case_policy
        )
        .into());
    }
    let _ = report_metrics(&report)?;
    let _ = report_memory_metrics(&report.memory)?;
    let previous = if baseline_path.is_file() {
        Some(serde_json::from_slice::<PerformanceBaseline>(&fs::read(
            baseline_path,
        )?)?)
    } else {
        None
    };
    if previous
        .as_ref()
        .is_some_and(|baseline| baseline.schema_version != BASELINE_SCHEMA)
    {
        return Err("existing performance baseline has an unsupported schema".into());
    }
    let previous_metrics = previous
        .as_ref()
        .map(baseline_metrics)
        .transpose()?
        .unwrap_or_default();
    let previous_memory_metrics = previous
        .as_ref()
        .map(|baseline| baseline_memory_metrics(&baseline.memory))
        .transpose()?
        .unwrap_or_default();
    let mut metrics = Vec::with_capacity(report.metrics.len());
    for candidate in &report.metrics {
        validate_time(candidate.median_microseconds, "candidate median")?;
        let key = MetricKey::from_report(candidate);
        let retained = previous_metrics.get(&key);
        let device = matches!(
            candidate.measurement.as_str(),
            "device_graph" | "device_path"
        );
        metrics.push(BaselineMetric {
            route: candidate.route.clone(),
            shape: candidate.shape.clone(),
            workload: candidate.workload.clone(),
            measurement: candidate.measurement.clone(),
            reference_microseconds: candidate.median_microseconds,
            relative_tolerance_percent: retained.map_or_else(
                || {
                    if device {
                        DEVICE_RELATIVE_TOLERANCE_PERCENT
                    } else {
                        HOST_RELATIVE_TOLERANCE_PERCENT
                    }
                },
                |metric| metric.relative_tolerance_percent,
            ),
            absolute_tolerance_microseconds: retained.map_or_else(
                || {
                    if device {
                        DEVICE_ABSOLUTE_TOLERANCE_MICROSECONDS
                    } else {
                        HOST_ABSOLUTE_TOLERANCE_MICROSECONDS
                    }
                },
                |metric| metric.absolute_tolerance_microseconds,
            ),
            operations_per_interval: candidate.operations_per_interval,
            enforced: retained.map_or(device, |metric| metric.enforced),
        });
    }
    metrics.sort_by_key(MetricKey::from_baseline);

    let mut memory_metrics = Vec::with_capacity(report.memory.metrics.len());
    for candidate in &report.memory.metrics {
        let key = MemoryMetricKey::from_report(candidate);
        let retained = previous_memory_metrics.get(&key);
        if let Some(authority) = retained {
            require_memory_contract(candidate, authority)?;
        }
        memory_metrics.push(BaselineMemoryMetric {
            name: candidate.name.clone(),
            measurement: candidate.measurement.clone(),
            kind: candidate.kind.clone(),
            scaling: candidate.scaling.clone(),
            comparison: candidate.comparison.clone(),
            reference_bytes: candidate.bytes,
            absolute_tolerance_bytes: retained.map_or_else(
                || default_memory_tolerance(&candidate.measurement),
                |metric| metric.absolute_tolerance_bytes,
            ),
            enforced: retained.map_or_else(
                || default_memory_enforced(&candidate.measurement),
                |metric| metric.enforced,
            ),
        });
    }
    memory_metrics.sort_by_key(MemoryMetricKey::from_baseline);

    let baseline = PerformanceBaseline {
        schema_version: BASELINE_SCHEMA,
        suite: report.suite,
        device: report.device,
        driver_version: report.driver_version,
        compute_capability: report.compute_capability,
        clock_policy: report.clock_policy,
        blessed_binary_sha256: report.binary_sha256,
        generator_baseline_sha256: report.generator_baseline_sha256,
        sm_clock_band_mhz: padded_band(
            report.sm_clock_min_mhz,
            report.sm_clock_max_mhz,
            CLOCK_PADDING_MHZ,
        ),
        memory_clock_band_mhz: padded_band(
            report.memory_clock_min_mhz,
            report.memory_clock_max_mhz,
            MEMORY_CLOCK_PADDING_MHZ,
        ),
        minimum_samples: report.samples,
        warmup_launches: report.warmup_launches,
        case_policy: report.case_policy,
        timing_scope: report.timing_scope,
        power_scope: report.power_scope,
        metrics,
        memory: MemoryBaseline {
            device_total_bytes: report.memory.device_total_bytes,
            metrics: memory_metrics,
        },
    };
    if let Some(parent) = baseline_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(&baseline)?;
    json.push(b'\n');
    fs::write(baseline_path, json)?;
    println!(
        "performance baseline blessed explicitly: {}",
        baseline_path.display()
    );

    Ok(())
}

pub(crate) fn preflight_baseline(baseline_path: &Path) -> Result<(), Box<dyn Error>> {
    serde_json::from_slice::<PerformanceBaseline>(&fs::read(baseline_path)?)?;
    Ok(())
}

fn read_report(path: &Path) -> Result<PerformanceReport, Box<dyn Error>> {
    let report: PerformanceReport = serde_json::from_slice(&fs::read(path)?)?;
    if report.schema_version != REPORT_SCHEMA {
        return Err(format!(
            "performance report schema is {}, expected {REPORT_SCHEMA}",
            report.schema_version
        )
        .into());
    }

    Ok(report)
}

fn validate_environment(
    report: &PerformanceReport,
    baseline: &PerformanceBaseline,
) -> Result<(), Box<dyn Error>> {
    validate_common_environment(report, baseline, true)?;
    for (name, candidate, authority) in [
        ("case policy", &report.case_policy, &baseline.case_policy),
        (
            "generator baseline",
            &report.generator_baseline_sha256,
            &baseline.generator_baseline_sha256,
        ),
    ] {
        if candidate != authority {
            return Err(format!(
                "performance {name} is `{candidate}`, baseline requires `{authority}`"
            )
            .into());
        }
    }

    Ok(())
}

fn validate_diagnostic_environment(
    report: &PerformanceReport,
    baseline: &PerformanceBaseline,
) -> Result<(), Box<dyn Error>> {
    validate_common_environment(report, baseline, false)?;
    if baseline.case_policy != "complete_inventory" {
        return Err(format!(
            "diagnostic comparison requires a complete-inventory authority, found `{}`",
            baseline.case_policy
        )
        .into());
    }
    if !matches!(
        report.case_policy.as_str(),
        "complete_inventory" | "diagnostic_subset"
    ) {
        return Err(format!(
            "diagnostic comparison does not admit case policy `{}`",
            report.case_policy
        )
        .into());
    }
    if report.samples < 3 {
        return Err("diagnostic performance reports require at least three samples".into());
    }

    Ok(())
}

fn validate_common_environment(
    report: &PerformanceReport,
    baseline: &PerformanceBaseline,
    enforce_sample_authority: bool,
) -> Result<(), Box<dyn Error>> {
    if baseline.schema_version != BASELINE_SCHEMA {
        return Err(format!(
            "performance baseline schema is {}, expected {BASELINE_SCHEMA}",
            baseline.schema_version
        )
        .into());
    }
    for (name, candidate, authority) in [
        ("suite", &report.suite, &baseline.suite),
        ("device", &report.device, &baseline.device),
        ("driver", &report.driver_version, &baseline.driver_version),
        (
            "compute capability",
            &report.compute_capability,
            &baseline.compute_capability,
        ),
        ("clock policy", &report.clock_policy, &baseline.clock_policy),
    ] {
        if candidate != authority {
            return Err(format!(
                "performance {name} is `{candidate}`, baseline requires `{authority}`"
            )
            .into());
        }
    }
    if enforce_sample_authority && report.samples < baseline.minimum_samples {
        return Err(format!(
            "performance report has {} samples, baseline requires at least {}",
            report.samples, baseline.minimum_samples
        )
        .into());
    }
    if report.warmup_launches != baseline.warmup_launches {
        return Err(format!(
            "performance report uses {} warmup launches, baseline requires {}",
            report.warmup_launches, baseline.warmup_launches
        )
        .into());
    }
    if report.timing_scope != baseline.timing_scope {
        return Err(format!(
            "performance timing scope is `{}`, baseline requires `{}`",
            report.timing_scope, baseline.timing_scope
        )
        .into());
    }
    if report.power_scope != baseline.power_scope {
        return Err(format!(
            "performance power scope is `{}`, baseline requires `{}`",
            report.power_scope, baseline.power_scope
        )
        .into());
    }
    if report.memory.device_total_bytes != baseline.memory.device_total_bytes {
        return Err(format!(
            "device memory capacity is {} bytes, baseline requires {} bytes",
            report.memory.device_total_bytes, baseline.memory.device_total_bytes
        )
        .into());
    }
    require_clock_band(
        "SM",
        report.sm_clock_min_mhz,
        report.sm_clock_max_mhz,
        baseline.sm_clock_band_mhz,
    )?;
    require_clock_band(
        "memory",
        report.memory_clock_min_mhz,
        report.memory_clock_max_mhz,
        baseline.memory_clock_band_mhz,
    )?;

    Ok(())
}

fn require_clock_band(
    name: &str,
    minimum: u32,
    maximum: u32,
    band: ClockBand,
) -> Result<(), Box<dyn Error>> {
    if band.minimum > band.maximum {
        return Err(format!(
            "{name} baseline clock band {}..={} MHz is inverted",
            band.minimum, band.maximum
        )
        .into());
    }
    if minimum < band.minimum || maximum > band.maximum {
        return Err(format!(
            "{name} clock range {minimum}..={maximum} MHz is outside baseline band {}..={} MHz",
            band.minimum, band.maximum
        )
        .into());
    }

    Ok(())
}

fn report_metrics(
    report: &PerformanceReport,
) -> Result<BTreeMap<MetricKey, &ReportMetric>, Box<dyn Error>> {
    let mut metrics = BTreeMap::new();
    for metric in &report.metrics {
        validate_time(metric.median_microseconds, "candidate median")?;
        validate_workload(&metric.workload)?;
        if metrics
            .insert(MetricKey::from_report(metric), metric)
            .is_some()
        {
            return Err("performance report contains a duplicate metric key".into());
        }
    }

    Ok(metrics)
}

fn baseline_metrics(
    baseline: &PerformanceBaseline,
) -> Result<BTreeMap<MetricKey, &BaselineMetric>, Box<dyn Error>> {
    let mut metrics = BTreeMap::new();
    for metric in &baseline.metrics {
        validate_time(metric.reference_microseconds, "baseline reference")?;
        validate_workload(&metric.workload)?;
        validate_nonnegative(
            metric.relative_tolerance_percent,
            "relative performance tolerance",
        )?;
        validate_nonnegative(
            metric.absolute_tolerance_microseconds,
            "absolute performance tolerance",
        )?;
        if metric.operations_per_interval == 0 {
            return Err("baseline operations per interval must be nonzero".into());
        }
        if metrics
            .insert(MetricKey::from_baseline(metric), metric)
            .is_some()
        {
            return Err("performance baseline contains a duplicate metric key".into());
        }
    }

    Ok(metrics)
}

fn validate_workload(workload: &Workload) -> Result<(), Box<dyn Error>> {
    for (name, value, admitted) in [
        (
            "scope",
            workload.scope.as_str(),
            &["operator", "layer", "model", "server"][..],
        ),
        (
            "phase",
            workload.phase.as_str(),
            &["startup", "prefill", "decode", "mtp", "request"][..],
        ),
        (
            "device cache",
            workload.device_cache.as_str(),
            &["cold", "warm", "not_applicable"][..],
        ),
        (
            "execution",
            workload.execution.as_str(),
            &["eager", "cuda_graph", "server"][..],
        ),
    ] {
        if !admitted.contains(&value) {
            return Err(format!("unknown workload {name} `{value}`").into());
        }
    }
    if workload
        .prefix_cache
        .as_deref()
        .is_some_and(|value| !matches!(value, "miss" | "partial_hit" | "full_hit"))
    {
        return Err("unknown workload prefix-cache regime".into());
    }
    for (name, value) in [
        ("batch size", workload.batch_size.map(u64::from)),
        ("concurrency", workload.concurrency.map(u64::from)),
        ("active tokens", workload.active_tokens),
        ("prompt tokens", workload.prompt_tokens),
        ("output tokens", workload.output_tokens),
    ] {
        if value == Some(0) {
            return Err(format!("workload {name} must be nonzero when present").into());
        }
    }

    Ok(())
}

fn report_memory_metrics(
    memory: &ReportMemory,
) -> Result<BTreeMap<MemoryMetricKey, &ReportMemoryMetric>, Box<dyn Error>> {
    if memory.device_total_bytes == 0 {
        return Err("performance report device memory capacity must be nonzero".into());
    }
    let mut metrics = BTreeMap::new();
    for metric in &memory.metrics {
        validate_memory_report_metric(metric)?;
        if metrics
            .insert(MemoryMetricKey::from_report(metric), metric)
            .is_some()
        {
            return Err("performance report contains a duplicate memory metric key".into());
        }
    }
    if metrics.is_empty() {
        return Err("performance report contains no memory metrics".into());
    }

    Ok(metrics)
}

fn baseline_memory_metrics(
    memory: &MemoryBaseline,
) -> Result<BTreeMap<MemoryMetricKey, &BaselineMemoryMetric>, Box<dyn Error>> {
    if memory.device_total_bytes == 0 {
        return Err("performance baseline device memory capacity must be nonzero".into());
    }
    let mut metrics = BTreeMap::new();
    for metric in &memory.metrics {
        validate_memory_comparison(&metric.comparison)?;
        if metrics
            .insert(MemoryMetricKey::from_baseline(metric), metric)
            .is_some()
        {
            return Err("performance baseline contains a duplicate memory metric key".into());
        }
    }
    if metrics.is_empty() {
        return Err("performance baseline contains no memory metrics".into());
    }

    Ok(metrics)
}

fn validate_memory_report_metric(metric: &ReportMemoryMetric) -> Result<(), Box<dyn Error>> {
    validate_memory_comparison(&metric.comparison)?;
    match metric.measurement.as_str() {
        "owned" => {
            if metric.kind.is_none() && metric.name != "summary/accounted_resident" {
                return Err(
                    format!("owned memory metric `{}` has no memory kind", metric.name).into(),
                );
            }
        }
        "setup_device_delta"
        | "timed_peak_device_delta"
        | "timed_growth_after_warmup"
        | "minimum_device_headroom"
        | "process_peak_rss"
        | "device_reserved"
        | "unattributed_setup_delta" => {
            if metric.kind.is_some() || metric.scaling.is_some() {
                return Err(format!(
                    "summary memory metric `{}` unexpectedly names an owner kind or scaling rule",
                    metric.name
                )
                .into());
            }
        }
        measurement => {
            return Err(format!("unknown memory measurement `{measurement}`").into());
        }
    }

    Ok(())
}

fn validate_memory_comparison(comparison: &str) -> Result<(), Box<dyn Error>> {
    if matches!(comparison, "at_most" | "at_least") {
        Ok(())
    } else {
        Err(format!("unknown memory comparison `{comparison}`").into())
    }
}

fn require_memory_contract(
    candidate_metric: &ReportMemoryMetric,
    authority_metric: &BaselineMemoryMetric,
) -> Result<(), Box<dyn Error>> {
    for (name, candidate, authority) in [
        (
            "kind",
            candidate_metric.kind.as_deref(),
            authority_metric.kind.as_deref(),
        ),
        (
            "scaling",
            candidate_metric.scaling.as_deref(),
            authority_metric.scaling.as_deref(),
        ),
        (
            "comparison",
            Some(candidate_metric.comparison.as_str()),
            Some(authority_metric.comparison.as_str()),
        ),
    ] {
        if candidate != authority {
            return Err(format!(
                "memory metric `{}` {name} is {:?}, baseline requires {:?}",
                candidate_metric.name, candidate, authority
            )
            .into());
        }
    }

    Ok(())
}

fn memory_passes(
    candidate_bytes: u64,
    authority: &BaselineMemoryMetric,
) -> Result<bool, Box<dyn Error>> {
    match authority.comparison.as_str() {
        "at_most" => Ok(candidate_bytes
            <= authority
                .reference_bytes
                .saturating_add(authority.absolute_tolerance_bytes)),
        "at_least" => Ok(candidate_bytes
            >= authority
                .reference_bytes
                .saturating_sub(authority.absolute_tolerance_bytes)),
        comparison => Err(format!("unknown memory comparison `{comparison}`").into()),
    }
}

fn validate_time(value: f64, name: &str) -> Result<(), Box<dyn Error>> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be finite and positive, got {value}").into());
    }

    Ok(())
}

fn validate_nonnegative(value: f64, name: &str) -> Result<(), Box<dyn Error>> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and nonnegative, got {value}").into());
    }

    Ok(())
}

fn padded_band(minimum: u32, maximum: u32, padding: u32) -> ClockBand {
    ClockBand {
        minimum: minimum.saturating_sub(padding),
        maximum: maximum.saturating_add(padding),
    }
}

fn maximum_allowed(metric: &BaselineMetric) -> f64 {
    let relative_slack = metric.reference_microseconds * metric.relative_tolerance_percent / 100.0;
    metric.reference_microseconds + relative_slack.max(metric.absolute_tolerance_microseconds)
}

fn default_memory_tolerance(measurement: &str) -> u64 {
    if matches!(measurement, "owned" | "timed_growth_after_warmup") {
        0
    } else {
        OBSERVED_MEMORY_TOLERANCE_BYTES
    }
}

fn default_memory_enforced(measurement: &str) -> bool {
    matches!(measurement, "owned" | "timed_growth_after_warmup")
}

#[cfg(test)]
mod tests {
    use super::{
        BaselineMemoryMetric, BaselineMetric, ClockBand, MemoryBaseline, PerformanceBaseline,
        PerformanceReport, ReportMemory, ReportMemoryMetric, ReportMetric, Workload,
        default_memory_enforced, default_memory_tolerance, diagnose, maximum_allowed,
        memory_passes, padded_band, require_clock_band, validate_environment,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn workload() -> Workload {
        Workload {
            scope: "operator".to_string(),
            phase: "decode".to_string(),
            batch_size: Some(1),
            concurrency: None,
            active_tokens: Some(1),
            prompt_tokens: None,
            context_tokens: None,
            output_tokens: None,
            device_cache: "warm".to_string(),
            prefix_cache: None,
            execution: "cuda_graph".to_string(),
        }
    }

    #[test]
    fn clock_bands_are_padded_and_enforced() {
        assert_eq!(padded_band(2_197, 2_197, 15).minimum, 2_182);
        assert_eq!(padded_band(2_197, 2_197, 15).maximum, 2_212);
        let band = ClockBand {
            minimum: 2_182,
            maximum: 2_212,
        };
        assert!(require_clock_band("SM", 2_197, 2_197, band).is_ok());
        assert!(require_clock_band("SM", 2_100, 2_197, band).is_err());
    }

    #[test]
    fn checked_baseline_contains_provenance_but_no_runner_identity() {
        let baseline = PerformanceBaseline {
            schema_version: super::BASELINE_SCHEMA,
            suite: "suite".to_string(),
            device: "NVIDIA GeForce RTX 5090".to_string(),
            driver_version: "driver".to_string(),
            compute_capability: "12.0".to_string(),
            clock_policy: "controlled".to_string(),
            blessed_binary_sha256: "a".repeat(64),
            generator_baseline_sha256: "b".repeat(64),
            sm_clock_band_mhz: ClockBand {
                minimum: 2_182,
                maximum: 2_212,
            },
            memory_clock_band_mhz: ClockBand {
                minimum: 13_951,
                maximum: 14_051,
            },
            minimum_samples: 40,
            warmup_launches: 1_024,
            case_policy: "complete_inventory".to_string(),
            timing_scope: "scope".to_string(),
            power_scope: "scope".to_string(),
            metrics: Vec::new(),
            memory: MemoryBaseline {
                device_total_bytes: 1,
                metrics: Vec::new(),
            },
        };
        let json = serde_json::to_value(baseline).unwrap();

        assert_eq!(json["blessed_binary_sha256"], "a".repeat(64));
        assert_eq!(json["clock_policy"], "controlled");
        assert!(json.get("device_uuid").is_none());
        assert!(json.get("runner").is_none());
    }

    #[test]
    fn metric_limit_uses_the_larger_declared_slack() {
        let metric = BaselineMetric {
            route: "route".to_string(),
            shape: "B=1".to_string(),
            workload: workload(),
            measurement: "device_graph".to_string(),
            reference_microseconds: 4.0,
            relative_tolerance_percent: 5.0,
            absolute_tolerance_microseconds: 0.05,
            operations_per_interval: 256,
            enforced: true,
        };

        assert_eq!(maximum_allowed(&metric), 4.2);

        let small = BaselineMetric {
            reference_microseconds: 0.5,
            ..metric
        };
        assert_eq!(maximum_allowed(&small), 0.55);
    }

    #[test]
    fn only_owned_and_post_warmup_growth_memory_are_enforced_by_default() {
        assert!(default_memory_enforced("owned"));
        assert!(default_memory_enforced("timed_growth_after_warmup"));
        assert!(!default_memory_enforced("setup_device_delta"));
        assert!(!default_memory_enforced("minimum_device_headroom"));
        assert_eq!(default_memory_tolerance("owned"), 0);
        assert_eq!(default_memory_tolerance("timed_growth_after_warmup"), 0);
        assert_eq!(
            default_memory_tolerance("setup_device_delta"),
            16 * 1024 * 1024
        );
    }

    #[test]
    fn memory_limits_obey_the_declared_direction() {
        let mut metric = BaselineMemoryMetric {
            name: "memory".to_string(),
            measurement: "owned".to_string(),
            kind: Some("workspace".to_string()),
            scaling: Some("max_batch=8".to_string()),
            comparison: "at_most".to_string(),
            reference_bytes: 100,
            absolute_tolerance_bytes: 10,
            enforced: true,
        };
        assert!(memory_passes(110, &metric).unwrap());
        assert!(!memory_passes(111, &metric).unwrap());

        metric.comparison = "at_least".to_string();
        assert!(memory_passes(90, &metric).unwrap());
        assert!(!memory_passes(89, &metric).unwrap());
    }

    #[test]
    fn diagnostic_subset_allows_provenance_lag_but_is_never_authoritative() {
        let report = PerformanceReport {
            schema_version: super::REPORT_SCHEMA,
            suite: "suite".to_string(),
            device: "NVIDIA GeForce RTX 5090".to_string(),
            driver_version: "driver".to_string(),
            compute_capability: "12.0".to_string(),
            clock_policy: "controlled".to_string(),
            binary_sha256: "c".repeat(64),
            generator_baseline_sha256: "new".to_string(),
            sm_clock_min_mhz: 2_197,
            sm_clock_max_mhz: 2_197,
            memory_clock_min_mhz: 14_001,
            memory_clock_max_mhz: 14_001,
            samples: 9,
            warmup_launches: 1_024,
            case_policy: "diagnostic_subset".to_string(),
            selected_batch_size: Some(1),
            timing_scope: "scope".to_string(),
            power_scope: "scope".to_string(),
            metrics: vec![ReportMetric {
                route: "route-b1".to_string(),
                shape: "B=1".to_string(),
                workload: workload(),
                measurement: "device_graph".to_string(),
                median_microseconds: 3.9,
                operations_per_interval: 256,
            }],
            memory: ReportMemory {
                device_total_bytes: 1,
                metrics: vec![ReportMemoryMetric {
                    name: "summary/post_warmup_growth".to_string(),
                    measurement: "timed_growth_after_warmup".to_string(),
                    kind: None,
                    scaling: None,
                    bytes: 0,
                    comparison: "at_most".to_string(),
                }],
            },
        };
        let mut batch_two = workload();
        batch_two.batch_size = Some(2);
        batch_two.active_tokens = Some(2);
        let baseline = PerformanceBaseline {
            schema_version: super::BASELINE_SCHEMA,
            suite: "suite".to_string(),
            device: "NVIDIA GeForce RTX 5090".to_string(),
            driver_version: "driver".to_string(),
            compute_capability: "12.0".to_string(),
            clock_policy: "controlled".to_string(),
            blessed_binary_sha256: "a".repeat(64),
            generator_baseline_sha256: "old".to_string(),
            sm_clock_band_mhz: ClockBand {
                minimum: 2_182,
                maximum: 2_212,
            },
            memory_clock_band_mhz: ClockBand {
                minimum: 13_951,
                maximum: 14_051,
            },
            minimum_samples: 40,
            warmup_launches: 1_024,
            case_policy: "complete_inventory".to_string(),
            timing_scope: "scope".to_string(),
            power_scope: "scope".to_string(),
            metrics: vec![
                BaselineMetric {
                    route: "route-b1".to_string(),
                    shape: "B=1".to_string(),
                    workload: workload(),
                    measurement: "device_graph".to_string(),
                    reference_microseconds: 4.0,
                    relative_tolerance_percent: 5.0,
                    absolute_tolerance_microseconds: 0.05,
                    operations_per_interval: 256,
                    enforced: true,
                },
                BaselineMetric {
                    route: "route-b2".to_string(),
                    shape: "B=2".to_string(),
                    workload: batch_two,
                    measurement: "device_graph".to_string(),
                    reference_microseconds: 5.0,
                    relative_tolerance_percent: 5.0,
                    absolute_tolerance_microseconds: 0.05,
                    operations_per_interval: 256,
                    enforced: true,
                },
            ],
            memory: MemoryBaseline {
                device_total_bytes: 1,
                metrics: vec![BaselineMemoryMetric {
                    name: "summary/post_warmup_growth".to_string(),
                    measurement: "timed_growth_after_warmup".to_string(),
                    kind: None,
                    scaling: None,
                    comparison: "at_most".to_string(),
                    reference_bytes: 0,
                    absolute_tolerance_bytes: 0,
                    enforced: true,
                }],
            },
        };
        assert!(validate_environment(&report, &baseline).is_err());

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "tuiskollm-performance-diagnostic-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let report_path = directory.join("report.json");
        let baseline_path = directory.join("baseline.json");
        fs::write(&report_path, serde_json::to_vec(&report).unwrap()).unwrap();
        fs::write(&baseline_path, serde_json::to_vec(&baseline).unwrap()).unwrap();

        let diagnostic = diagnose(&report_path, &baseline_path).unwrap();

        assert!(!diagnostic.authoritative);
        assert!(diagnostic.generator_provenance_changed);
        assert_eq!(diagnostic.selected_batch_size, Some(1));
        assert_eq!(diagnostic.timing_metrics.len(), 1);
        fs::remove_dir_all(directory).unwrap();
    }
}
