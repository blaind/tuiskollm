//! Persistent diagnostic optimization-iteration records.

use crate::performance::DiagnosticComparison;
use rusqlite::{Connection, params};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MANIFEST_SCHEMA: u32 = 1;

#[derive(Debug, Default, Serialize)]
pub(crate) struct StageTiming {
    status: String,
    wall_milliseconds: u64,
}

#[derive(Debug, Serialize)]
struct IterationManifest {
    schema_version: u32,
    authoritative: bool,
    suite: String,
    batch_size: u32,
    hypothesis: String,
    command: Vec<String>,
    git_head: String,
    git_dirty: bool,
    device_input_sha256: String,
    agent_started_unix_milliseconds: Option<u64>,
    started_unix_milliseconds: u64,
    finished_unix_milliseconds: Option<u64>,
    agent_loop_wall_milliseconds: Option<u64>,
    command_wall_milliseconds: Option<u64>,
    status: String,
    preflight: StageTiming,
    qualification: StageTiming,
    build: StageTiming,
    benchmark: StageTiming,
    comparison: StageTiming,
    report_path: String,
    baseline_path: String,
    diagnostic: Option<DiagnosticComparison>,
    error: Option<String>,
}

pub(crate) struct IterationRecorder {
    root: PathBuf,
    output_dir: PathBuf,
    manifest: IterationManifest,
    started: Instant,
}

impl IterationRecorder {
    pub(crate) fn start(
        root: &Path,
        suite: &str,
        batch_size: u32,
        hypothesis: String,
        device_input_sha256: String,
        baseline_path: &Path,
    ) -> Result<Self, Box<dyn Error>> {
        let started_unix_milliseconds = now_milliseconds()?;
        let agent_started_unix_milliseconds =
            env::var("TUISKO_AGENT_ITERATION_STARTED_UNIX_MILLISECONDS")
                .ok()
                .map(|value| value.parse())
                .transpose()?;
        if agent_started_unix_milliseconds
            .is_some_and(|started| started > started_unix_milliseconds)
        {
            return Err("agent iteration start is later than the perf iterate command".into());
        }
        let output_dir = root.join(format!(
            "target/optimization/{suite}/iteration-{started_unix_milliseconds}-b{batch_size}"
        ));
        fs::create_dir_all(&output_dir)?;
        let report = output_dir.join("report.json");
        let git_head = git_text(root, &["rev-parse", "HEAD"])?;
        let git_dirty =
            !git_text(root, &["status", "--porcelain", "--untracked-files=normal"])?.is_empty();
        let manifest = IterationManifest {
            schema_version: MANIFEST_SCHEMA,
            authoritative: false,
            suite: suite.to_string(),
            batch_size,
            command: vec![
                "cargo".to_string(),
                "run".to_string(),
                "-p".to_string(),
                "xtask".to_string(),
                "--".to_string(),
                "perf".to_string(),
                "iterate".to_string(),
                suite.to_string(),
                "--batch".to_string(),
                batch_size.to_string(),
                "--hypothesis".to_string(),
                hypothesis.clone(),
            ],
            hypothesis,
            git_head,
            git_dirty,
            device_input_sha256,
            agent_started_unix_milliseconds,
            started_unix_milliseconds,
            finished_unix_milliseconds: None,
            agent_loop_wall_milliseconds: None,
            command_wall_milliseconds: None,
            status: "running".to_string(),
            preflight: StageTiming::default(),
            qualification: StageTiming::default(),
            build: StageTiming::default(),
            benchmark: StageTiming::default(),
            comparison: StageTiming::default(),
            report_path: relative_text(root, &report)?,
            baseline_path: relative_text(root, baseline_path)?,
            diagnostic: None,
            error: None,
        };
        let recorder = Self {
            root: root.to_path_buf(),
            output_dir,
            manifest,
            started: Instant::now(),
        };
        recorder.write_manifest()?;
        Ok(recorder)
    }

    pub(crate) fn report_path(&self) -> PathBuf {
        self.root.join(&self.manifest.report_path)
    }

    pub(crate) fn record_stage(&mut self, name: &str, status: &str, elapsed: Duration) {
        let stage = match name {
            "preflight" => &mut self.manifest.preflight,
            "qualification" => &mut self.manifest.qualification,
            "build" => &mut self.manifest.build,
            "benchmark" => &mut self.manifest.benchmark,
            "comparison" => &mut self.manifest.comparison,
            _ => panic!("unknown performance iteration stage `{name}`"),
        };
        stage.status = status.to_string();
        stage.wall_milliseconds = elapsed.as_millis().try_into().unwrap_or(u64::MAX);
    }

    pub(crate) fn succeed(
        mut self,
        diagnostic: DiagnosticComparison,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let regressions = diagnostic.timing_regressions + diagnostic.memory_regressions;
        self.manifest.status = if regressions == 0 {
            "diagnostic_within_authority"
        } else {
            "diagnostic_regression"
        }
        .to_string();
        self.manifest.diagnostic = Some(diagnostic);
        self.finish()?;
        Ok(self.output_dir)
    }

    pub(crate) fn fail(mut self, error: &dyn std::fmt::Display) -> Result<PathBuf, Box<dyn Error>> {
        self.manifest.status = "refused_or_failed".to_string();
        self.manifest.error = Some(error.to_string());
        self.finish()?;
        Ok(self.output_dir)
    }

    fn finish(&mut self) -> Result<(), Box<dyn Error>> {
        let finished = now_milliseconds()?;
        self.manifest.finished_unix_milliseconds = Some(finished);
        self.manifest.agent_loop_wall_milliseconds = self
            .manifest
            .agent_started_unix_milliseconds
            .map(|started| finished - started);
        self.manifest.command_wall_milliseconds = Some(
            self.started
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        self.write_manifest()?;
        self.insert_database()
    }

    fn write_manifest(&self) -> Result<(), Box<dyn Error>> {
        let path = self.output_dir.join("manifest.json");
        let mut json = serde_json::to_vec_pretty(&self.manifest)?;
        json.push(b'\n');
        fs::write(path, json)?;
        Ok(())
    }

    fn insert_database(&self) -> Result<(), Box<dyn Error>> {
        let database = self.root.join("target/optimization/iterations.sqlite3");
        let parent = database
            .parent()
            .ok_or("optimization database path has no parent")?;
        fs::create_dir_all(parent)?;
        let connection = Connection::open(database)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS iterations (
               id INTEGER PRIMARY KEY,
               schema_version INTEGER NOT NULL,
               suite TEXT NOT NULL,
               batch_size INTEGER NOT NULL,
               hypothesis TEXT NOT NULL,
               command_json TEXT NOT NULL,
               git_head TEXT NOT NULL,
               git_dirty INTEGER NOT NULL,
               device_input_sha256 TEXT NOT NULL,
               agent_started_unix_milliseconds INTEGER,
               started_unix_milliseconds INTEGER NOT NULL,
               finished_unix_milliseconds INTEGER NOT NULL,
               agent_loop_wall_milliseconds INTEGER,
               command_wall_milliseconds INTEGER NOT NULL,
               status TEXT NOT NULL,
               preflight_wall_milliseconds INTEGER NOT NULL,
               qualification_wall_milliseconds INTEGER NOT NULL,
               build_wall_milliseconds INTEGER NOT NULL,
               benchmark_wall_milliseconds INTEGER NOT NULL,
               comparison_wall_milliseconds INTEGER NOT NULL,
               report_path TEXT NOT NULL,
               baseline_path TEXT NOT NULL,
               generator_provenance_changed INTEGER,
               timing_regressions INTEGER,
               memory_regressions INTEGER,
               error TEXT
             );",
        )?;
        let diagnostic = self.manifest.diagnostic.as_ref();
        connection.execute(
            "INSERT INTO iterations (
               schema_version, suite, batch_size, hypothesis, command_json, git_head, git_dirty,
               device_input_sha256, agent_started_unix_milliseconds,
               started_unix_milliseconds, finished_unix_milliseconds,
               agent_loop_wall_milliseconds, command_wall_milliseconds, status,
               preflight_wall_milliseconds,
               qualification_wall_milliseconds, build_wall_milliseconds,
               benchmark_wall_milliseconds, comparison_wall_milliseconds, report_path,
               baseline_path, generator_provenance_changed, timing_regressions,
               memory_regressions, error
             ) VALUES (
               ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
               ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25
             )",
            params![
                self.manifest.schema_version,
                self.manifest.suite,
                self.manifest.batch_size,
                self.manifest.hypothesis,
                serde_json::to_string(&self.manifest.command)?,
                self.manifest.git_head,
                self.manifest.git_dirty,
                self.manifest.device_input_sha256,
                self.manifest.agent_started_unix_milliseconds,
                self.manifest.started_unix_milliseconds,
                self.manifest.finished_unix_milliseconds,
                self.manifest.agent_loop_wall_milliseconds,
                self.manifest.command_wall_milliseconds,
                self.manifest.status,
                self.manifest.preflight.wall_milliseconds,
                self.manifest.qualification.wall_milliseconds,
                self.manifest.build.wall_milliseconds,
                self.manifest.benchmark.wall_milliseconds,
                self.manifest.comparison.wall_milliseconds,
                self.manifest.report_path,
                self.manifest.baseline_path,
                diagnostic.map(|value| value.generator_provenance_changed),
                diagnostic.map(|value| value.timing_regressions),
                diagnostic.map(|value| value.memory_regressions),
                self.manifest.error,
            ],
        )?;
        Ok(())
    }
}

fn now_milliseconds() -> Result<u64, Box<dyn Error>> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

fn git_text(root: &Path, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn relative_text(root: &Path, path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| {
            format!(
                "optimization artifact {} is outside repository root {}",
                path.display(),
                root.display()
            )
        })?
        .to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()))?
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::IterationRecorder;
    use rusqlite::Connection;
    use std::fs;
    use std::process::Command;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn refused_iteration_is_preserved_in_json_and_sqlite() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tuiskollm-performance-iteration-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .arg(&root)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&root)
                .args([
                    "-c",
                    "user.name=TuiskoLLM test",
                    "-c",
                    "user.email=test@invalid",
                    "commit",
                    "--quiet",
                    "--allow-empty",
                    "-m",
                    "initial",
                ])
                .status()
                .unwrap()
                .success()
        );
        let baseline = root.join("qual/baselines/suite-sm120.json");
        let mut recorder = IterationRecorder::start(
            &root,
            "suite",
            1,
            "one hypothesis".to_string(),
            "input".to_string(),
            &baseline,
        )
        .unwrap();
        recorder.record_stage("preflight", "refused", Duration::from_millis(7));
        let output = recorder.fail(&"expected refusal").unwrap();

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["authoritative"], false);
        assert_eq!(manifest["status"], "refused_or_failed");
        assert_eq!(manifest["preflight"]["wall_milliseconds"], 7);
        let database =
            Connection::open(root.join("target/optimization/iterations.sqlite3")).unwrap();
        let status: String = database
            .query_row("SELECT status FROM iterations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(status, "refused_or_failed");
        drop(database);
        fs::remove_dir_all(root).unwrap();
    }
}
