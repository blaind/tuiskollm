//! Shared controls for multi-sweep benchmark processes.

use crate::device_benchmark::DeviceBenchmarkError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const CACHE_SCHEMA: u32 = 1;

/// Refuses a non-idle device before a combined benchmark opens its checkpoint.
pub fn benchmark_device_preflight() -> Result<(), DeviceBenchmarkError> {
    let _preflight = crate::device_benchmark::preflight()?;
    Ok(())
}

/// Wall attribution for one sweep in a construct-once process.
#[derive(Clone, Debug)]
pub struct BenchmarkSweepTiming {
    /// Stable sweep name.
    pub name: String,
    /// Host wall time spent inside the sweep.
    pub wall: Duration,
}

/// Runs named sweeps against one already-constructed session.
pub fn run_benchmark_sweeps<S, E, I, N>(
    session: &mut S,
    sweeps: I,
    mut run: impl FnMut(&str, &mut S) -> Result<(), E>,
) -> Result<Vec<BenchmarkSweepTiming>, E>
where
    I: IntoIterator<Item = N>,
    N: AsRef<str>,
{
    let mut timings = Vec::new();
    for sweep in sweeps {
        let name = sweep.as_ref();
        let started = Instant::now();
        run(name, session)?;
        let wall = started.elapsed();
        eprintln!("benchmark sweep `{name}` completed in {wall:.2?}");
        timings.push(BenchmarkSweepTiming {
            name: name.to_string(),
            wall,
        });
    }

    Ok(timings)
}

/// One exact route and shape retained by a diagnostic cell filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkCellSelector {
    /// Original `ROUTE:SHAPE` spelling.
    pub original: String,
    /// Full route or its final slash-separated component.
    pub route: String,
    /// Exact shape such as `B=1` or `T=1024`.
    pub shape: String,
}

impl BenchmarkCellSelector {
    /// Returns whether this selector names `route` and `shape`.
    pub fn matches(&self, route: &str, shape: &str) -> bool {
        (route == self.route || route.rsplit('/').next() == Some(self.route.as_str()))
            && shape == self.shape
    }
}

/// Parses comma-separated `ROUTE:SHAPE` selectors.
pub fn parse_benchmark_cells(
    cells: Option<&str>,
) -> Result<Vec<BenchmarkCellSelector>, DeviceBenchmarkError> {
    let Some(cells) = cells else {
        return Ok(Vec::new());
    };
    let mut selectors = Vec::new();
    for value in cells.split(',') {
        let value = value.trim();
        let Some((route, shape)) = value.rsplit_once(':') else {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "cell `{value}` must be ROUTE:SHAPE, for example decode:B=1"
            )));
        };
        if route.is_empty() || shape.is_empty() {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "cell `{value}` must name both a route and a shape"
            )));
        }
        if selectors.iter().any(|selector: &BenchmarkCellSelector| {
            selector.route == route && selector.shape == shape
        }) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "cell `{value}` is repeated"
            )));
        }
        selectors.push(BenchmarkCellSelector {
            original: value.to_string(),
            route: route.to_string(),
            shape: shape.to_string(),
        });
    }
    if selectors.is_empty() {
        return Err(DeviceBenchmarkError::Precondition(
            "the cell filter is empty".to_string(),
        ));
    }

    Ok(selectors)
}

/// Exact environment identity for a reusable live baseline arm.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchmarkFingerprint {
    /// NVIDIA driver version observed before the run.
    pub driver_version: String,
    /// SM clock observed before the run.
    pub sm_clock_mhz: u32,
    /// Admitted source snapshot revision.
    pub snapshot_revision: String,
    /// Device-code generator revision.
    pub cuda_oxide_commit: String,
    /// Base revision whose baseline arm was measured.
    pub base_sha: String,
}

impl BenchmarkFingerprint {
    /// Records the current device state and caller-supplied source identities.
    pub fn record(
        snapshot_revision: impl Into<String>,
        cuda_oxide_commit: impl Into<String>,
        base_sha: impl Into<String>,
    ) -> Result<Self, DeviceBenchmarkError> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=driver_version,clocks.current.sm",
                "--format=csv,noheader,nounits",
                "--id=0",
            ])
            .output()?;
        if !output.status.success() {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "nvidia-smi fingerprint query failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            DeviceBenchmarkError::Precondition(format!(
                "nvidia-smi fingerprint output was not UTF-8: {error}"
            ))
        })?;
        let rows = stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>();
        if rows.len() != 1 {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "nvidia-smi fingerprint query returned {} rows instead of one",
                rows.len()
            )));
        }
        let Some((driver_version, sm_clock)) = rows[0].split_once(',') else {
            return Err(DeviceBenchmarkError::Precondition(
                "nvidia-smi fingerprint row omitted a field".to_string(),
            ));
        };
        let fingerprint = Self {
            driver_version: driver_version.trim().to_string(),
            sm_clock_mhz: sm_clock.trim().parse().map_err(|error| {
                DeviceBenchmarkError::Precondition(format!(
                    "nvidia-smi returned an invalid SM clock: {error}"
                ))
            })?,
            snapshot_revision: snapshot_revision.into(),
            cuda_oxide_commit: cuda_oxide_commit.into(),
            base_sha: base_sha.into(),
        };
        for (name, value) in [
            ("driver version", fingerprint.driver_version.as_str()),
            ("snapshot revision", fingerprint.snapshot_revision.as_str()),
            ("cuda-oxide commit", fingerprint.cuda_oxide_commit.as_str()),
            ("base SHA", fingerprint.base_sha.as_str()),
        ] {
            if value.is_empty() {
                return Err(DeviceBenchmarkError::Precondition(format!(
                    "benchmark fingerprint has an empty {name}"
                )));
            }
        }
        for (name, value) in [
            ("snapshot revision", fingerprint.snapshot_revision.as_str()),
            ("cuda-oxide commit", fingerprint.cuda_oxide_commit.as_str()),
            ("base SHA", fingerprint.base_sha.as_str()),
        ] {
            if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(DeviceBenchmarkError::Precondition(format!(
                    "benchmark fingerprint {name} must be a 40-character hexadecimal commit"
                )));
            }
        }

        Ok(fingerprint)
    }
}

#[derive(Deserialize, Serialize)]
struct CachedBaseline<T> {
    schema_version: u32,
    fingerprint: BenchmarkFingerprint,
    rows: T,
}

/// Result of checking an optional live-baseline cache.
pub enum BaselineCacheLookup<T> {
    /// The exact fingerprint matched and the rows may be reused.
    Hit(T),
    /// The cache was absent, malformed, or recorded under another fingerprint.
    Miss(String),
}

/// Reads reusable baseline rows only when every fingerprint field matches.
pub fn read_baseline_cache<T: DeserializeOwned>(
    path: &Path,
    expected: &BenchmarkFingerprint,
) -> Result<BaselineCacheLookup<T>, DeviceBenchmarkError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BaselineCacheLookup::Miss("cache absent".to_string()));
        }
        Err(error) => return Err(error.into()),
    };
    let cached = match serde_json::from_slice::<CachedBaseline<T>>(&bytes) {
        Ok(cached) => cached,
        Err(error) => {
            return Ok(BaselineCacheLookup::Miss(format!(
                "cache is unreadable: {error}"
            )));
        }
    };
    if cached.schema_version != CACHE_SCHEMA {
        return Ok(BaselineCacheLookup::Miss(format!(
            "cache schema {} != {CACHE_SCHEMA}",
            cached.schema_version
        )));
    }
    if cached.fingerprint != *expected {
        return Ok(BaselineCacheLookup::Miss(format!(
            "fingerprint mismatch: cached={:?}, current={expected:?}",
            cached.fingerprint
        )));
    }

    Ok(BaselineCacheLookup::Hit(cached.rows))
}

/// Atomically replaces reusable baseline rows under an exact fingerprint.
pub fn write_baseline_cache<T: Serialize>(
    path: &Path,
    fingerprint: &BenchmarkFingerprint,
    rows: &T,
) -> Result<(), DeviceBenchmarkError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(&CachedBaseline {
        schema_version: CACHE_SCHEMA,
        fingerprint: fingerprint.clone(),
        rows,
    })
    .map_err(|error| {
        DeviceBenchmarkError::Precondition(format!("baseline cache serialization failed: {error}"))
    })?;
    let temporary = temporary_path(path);
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;

    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    PathBuf::from(temporary)
}

/// Optional fail-closed wall budget for one diagnostic cell.
pub struct CellWallBudget {
    label: String,
    limit: Option<Duration>,
    started: Instant,
}

impl CellWallBudget {
    /// Starts an uncapped or explicitly capped diagnostic cell.
    pub fn new(label: impl Into<String>, limit: Option<Duration>) -> Self {
        Self {
            label: label.into(),
            limit,
            started: Instant::now(),
        }
    }

    /// Refuses the next indivisible step when it would exceed the cell budget.
    pub fn admit_next(&self, estimated: Option<Duration>) -> Result<(), DeviceBenchmarkError> {
        let (Some(limit), Some(estimated)) = (self.limit, estimated) else {
            return Ok(());
        };
        if self.started.elapsed().saturating_add(estimated) > limit {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "diagnostic wall budget refused `{}` after {:.1}s: the next step is estimated at {:.1}s and the cell limit is {:.1}s; this cell is incomplete and is not evidence",
                self.label,
                self.started.elapsed().as_secs_f64(),
                estimated.as_secs_f64(),
                limit.as_secs_f64()
            )));
        }

        Ok(())
    }

    /// Elapsed wall time inside the cell.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_requires_an_exact_fingerprint() {
        let directory = std::env::temp_dir().join(format!(
            "tuisko-benchmark-cache-test-{}",
            std::process::id()
        ));
        let path = directory.join("baseline.json");
        let fingerprint = BenchmarkFingerprint {
            driver_version: "test-driver".to_string(),
            sm_clock_mhz: 2_200,
            snapshot_revision: "snapshot".to_string(),
            cuda_oxide_commit: "oxide".to_string(),
            base_sha: "base".to_string(),
        };
        write_baseline_cache(&path, &fingerprint, &vec![1_u32, 2, 3]).unwrap();
        assert!(matches!(
            read_baseline_cache::<Vec<u32>>(&path, &fingerprint).unwrap(),
            BaselineCacheLookup::Hit(rows) if rows == vec![1, 2, 3]
        ));
        let mut mismatch = fingerprint.clone();
        mismatch.base_sha = "other".to_string();
        assert!(matches!(
            read_baseline_cache::<Vec<u32>>(&path, &mismatch).unwrap(),
            BaselineCacheLookup::Miss(reason) if reason.contains("fingerprint mismatch")
        ));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn wall_budget_refuses_only_a_projected_overrun() {
        let budget = CellWallBudget::new("cell", Some(Duration::from_secs(4)));
        assert!(budget.admit_next(Some(Duration::from_secs(3))).is_ok());
        assert!(budget.admit_next(Some(Duration::from_secs(5))).is_err());
        assert!(CellWallBudget::new("cell", None).admit_next(None).is_ok());
    }

    #[test]
    fn cell_selectors_require_route_shape_pairs() {
        let selectors = parse_benchmark_cells(Some("decode:B=1,prefill:T=1024")).unwrap();
        assert_eq!(selectors.len(), 2);
        assert!(selectors[0].matches("family/model/decode", "B=1"));
        assert!(parse_benchmark_cells(Some("decode")).is_err());
        assert!(parse_benchmark_cells(Some("decode:B=1,decode:B=1")).is_err());
        assert!(parse_benchmark_cells(Some("")).is_err());
    }

    #[test]
    fn sweep_runner_reuses_one_session_in_requested_order() {
        let mut session = Vec::new();
        let timings = run_benchmark_sweeps(&mut session, ["a", "b"], |name, session| {
            session.push(name.to_string());
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(session, ["a", "b"]);
        assert_eq!(
            timings
                .iter()
                .map(|timing| timing.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}
