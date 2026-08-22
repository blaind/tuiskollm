//! Lifecycle for one remote executable.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::key::{resolve_api_key, resolve_env};
use crate::sentry::{delete_pod, mark_keep, spawn_sentry};
use crate::ssh::Ssh;
use crate::v2::{POD_NAME_PREFIX, REMOTE_WORKDIR, V2, is_missing, wait_until_ssh};
use crate::{GpuTarget, RemoteError, RemoteResult};

const PROVISIONING_DEADLINE: Duration = Duration::from_secs(240);
const SENTRY_GRACE: Duration = Duration::from_secs(300);
const OWNED_POD_SUFFIX: &str = ".pod";
const SNAPSHOT_REVISION: &str = "16b6615af3548b88e2d8e382457bc705b00479cf";
const SNAPSHOT_ROOT: &str = "/tmp/tuiskollm/snapshots/16b6615af3548b88e2d8e382457bc705b00479cf";

/// Inputs for one prebuilt qualification run.
pub struct QualificationOptions {
    /// Human-readable suite name used in logs.
    pub suite: String,
    /// Locally built qualification test executable.
    pub executable: PathBuf,
    /// Fixed libtest arguments selected by `xtask`.
    pub test_args: Vec<String>,
    /// Exact GPU to provision and admit.
    pub gpu: GpuTarget,
    /// Download and expose the one pinned admitted snapshot before execution.
    pub source_snapshot: bool,
    /// Container image for the pod.
    pub image: String,
    /// Whole-run budget, including provisioning and report retrieval.
    pub max_minutes: u32,
    /// Retain a failed pod for inspection.
    pub keep_on_fail: bool,
}

/// Inputs for one diagnostic device benchmark run.
pub struct BenchmarkOptions {
    /// Benchmark suite name accepted by `bench-device`.
    pub suite: String,
    /// Locally built device benchmark executable.
    pub executable: PathBuf,
    /// Benchmark controls selected by `xtask`.
    pub benchmark_args: Vec<String>,
    /// Hash of the matching checked static-resource baseline.
    pub generator_baseline_sha256: String,
    /// Exact GPU to provision and admit.
    pub gpu: GpuTarget,
    /// Download and pass the one pinned admitted snapshot before execution.
    pub source_snapshot: bool,
    /// Container image for the pod.
    pub image: String,
    /// Whole-run budget, including provisioning and report retrieval.
    pub max_minutes: u32,
    /// Retain a failed pod for inspection.
    pub keep_on_fail: bool,
}

/// Inputs for one transport and device-identity probe.
pub struct ProbeOptions {
    /// Exact GPU to provision and admit.
    pub gpu: GpuTarget,
    /// Container image for the pod.
    pub image: String,
    /// Whole-run budget, including provisioning.
    pub max_minutes: u32,
    /// Retain a failed pod for inspection.
    pub keep_on_fail: bool,
}

struct PodGuard {
    pod_id: String,
    ownership_path: PathBuf,
    keep_on_fail: bool,
    failed: bool,
}

impl PodGuard {
    fn new(pod_id: String, ownership_path: PathBuf, keep_on_fail: bool) -> Self {
        Self {
            pod_id,
            ownership_path,
            keep_on_fail,
            failed: false,
        }
    }
}

impl Drop for PodGuard {
    fn drop(&mut self) {
        if self.keep_on_fail && self.failed {
            if let Err(error) = mark_keep(&self.pod_id) {
                eprintln!("warning: failed to retain pod {}: {error}", self.pod_id);
            } else {
                eprintln!("failed remote pod {} retained", self.pod_id);
            }
            return;
        }
        match delete_pod(&self.pod_id) {
            Ok(()) => forget_owned_pod(&self.ownership_path),
            Err(error) => {
                eprintln!(
                    "warning: pod {} was not deleted ({error}); the watchdog will retry",
                    self.pod_id
                );
            }
        }
    }
}

/// Validates the local API and SSH credentials before an expensive build.
pub fn check_credentials() -> RemoteResult<()> {
    let _ = resolve_api_key()?;
    let _ = checked_ssh_key()?;

    Ok(())
}

fn checked_ssh_key() -> RemoteResult<PathBuf> {
    let key_file = resolve_env("RUNPOD_SSH_KEY_FILE").ok_or_else(|| RemoteError::Precheck {
        detail: "RUNPOD_SSH_KEY_FILE must name a registered account SSH private key".to_owned(),
    })?;
    std::fs::File::open(&key_file).map_err(|source| RemoteError::Read {
        what: format!("SSH key file {key_file}"),
        source,
    })?;
    Ssh::validate_key(Path::new(&key_file))?;

    Ok(PathBuf::from(key_file))
}

/// Provisions and validates a GPU without uploading a device artifact.
pub fn run_probe(options: &ProbeOptions) -> RemoteResult<()> {
    if options.max_minutes < 2 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: options.max_minutes,
        });
    }
    let key_file = checked_ssh_key()?;
    let api = V2::new(resolve_api_key()?);
    let started = Instant::now();
    let budget = Duration::from_secs(u64::from(options.max_minutes) * 60);
    let provisioning_deadline = started + PROVISIONING_DEADLINE.min(budget);
    let (guard, ssh, pod_id, cost_per_hour) = open_pod(
        &api,
        options.gpu,
        &options.image,
        options.keep_on_fail,
        key_file,
        budget,
        provisioning_deadline,
    )?;
    wait_for_precheck(&ssh, options.gpu, provisioning_deadline)?;

    finish_run(&api, guard, &pod_id, 0, started, cost_per_hour)
}

/// Runs a prebuilt qualification executable on a fresh target GPU pod.
pub fn run_qualification(
    workspace_root: &Path,
    options: &QualificationOptions,
) -> RemoteResult<()> {
    if options.max_minutes < 2 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: options.max_minutes,
        });
    }
    let arguments = sanitize_arguments(&options.test_args)?;
    if !options.executable.is_file() {
        return Err(RemoteError::Read {
            what: format!("qualification executable {}", options.executable.display()),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file does not exist"),
        });
    }
    let key_file = checked_ssh_key()?;

    let digest = sha256_hex(&options.executable)?;
    println!(
        "qualification binary: {} ({digest})",
        options.executable.display()
    );
    let api = V2::new(resolve_api_key()?);
    let started = Instant::now();
    let budget = Duration::from_secs(u64::from(options.max_minutes) * 60);
    let provisioning_deadline = started + PROVISIONING_DEADLINE.min(budget);
    let (mut guard, ssh, pod_id, cost_per_hour) = open_pod(
        &api,
        options.gpu,
        &options.image,
        options.keep_on_fail,
        key_file,
        budget,
        provisioning_deadline,
    )?;

    wait_for_precheck(&ssh, options.gpu, provisioning_deadline)?;

    let elapsed = started.elapsed();
    let run_seconds = budget
        .as_secs()
        .saturating_sub(elapsed.as_secs())
        .saturating_sub(60);
    if run_seconds < 30 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: options.max_minutes,
        });
    }
    let (status, output) = ssh.run(&format!("mkdir -p {REMOTE_WORKDIR}"), 30)?;
    if status != 0 {
        return Err(RemoteError::Precheck {
            detail: format!("creating {REMOTE_WORKDIR} exited {status}:\n{output}"),
        });
    }
    ssh.put_file(&options.executable, &format!("{REMOTE_WORKDIR}/qual"))?;

    if options.source_snapshot {
        prepare_source_snapshot(&ssh, run_seconds)?;
    }
    let run_seconds = remaining_run_seconds(started, budget, options.max_minutes)?;
    let command = qualification_command(&arguments, options.source_snapshot);
    println!("running {} (budget {run_seconds}s)", options.suite);
    let (status, output) = ssh.run(
        &format!("timeout -k 30 {run_seconds} sh -c '{command}'"),
        run_seconds + 60,
    )?;
    guard.failed = status != 0;
    if !output.is_empty() {
        println!("{output}");
    }
    if guard.failed {
        eprintln!("qualification exited {status}");
    }

    let report = report_path(workspace_root)?;
    ssh.get_file(&format!("{REMOTE_WORKDIR}/gate.out"), &report)?;
    println!("report: {}", report.display());
    finish_run(&api, guard, &pod_id, status, started, cost_per_hour)
}

/// Runs one prebuilt device benchmark on a fresh target GPU pod.
///
/// The result is diagnostic evidence only. It is never compared with or written to a checked
/// performance baseline.
pub fn run_benchmark(workspace_root: &Path, options: &BenchmarkOptions) -> RemoteResult<()> {
    if options.max_minutes < 2 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: options.max_minutes,
        });
    }
    let mut arguments = vec![options.suite.clone()];
    arguments.extend(options.benchmark_args.iter().cloned());
    arguments = sanitize_arguments(&arguments)?;
    if options.source_snapshot {
        arguments.insert(1, SNAPSHOT_ROOT.to_owned());
    }
    require_sha256(&options.generator_baseline_sha256)?;
    if !options.executable.is_file() {
        return Err(RemoteError::Read {
            what: format!("benchmark executable {}", options.executable.display()),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file does not exist"),
        });
    }
    let key_file = checked_ssh_key()?;

    let digest = sha256_hex(&options.executable)?;
    println!(
        "benchmark binary: {} ({digest})",
        options.executable.display()
    );
    println!("remote timings are diagnostic and cannot bless or satisfy `perf gate`");
    let api = V2::new(resolve_api_key()?);
    let started = Instant::now();
    let budget = Duration::from_secs(u64::from(options.max_minutes) * 60);
    let provisioning_deadline = started + PROVISIONING_DEADLINE.min(budget);
    let (mut guard, ssh, pod_id, cost_per_hour) = open_pod(
        &api,
        options.gpu,
        &options.image,
        options.keep_on_fail,
        key_file,
        budget,
        provisioning_deadline,
    )?;

    wait_for_precheck(&ssh, options.gpu, provisioning_deadline)?;

    let elapsed = started.elapsed();
    let run_seconds = budget
        .as_secs()
        .saturating_sub(elapsed.as_secs())
        .saturating_sub(60);
    if run_seconds < 30 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: options.max_minutes,
        });
    }
    let (status, output) = ssh.run(&format!("mkdir -p {REMOTE_WORKDIR}"), 30)?;
    if status != 0 {
        return Err(RemoteError::Precheck {
            detail: format!("creating {REMOTE_WORKDIR} exited {status}:\n{output}"),
        });
    }
    ssh.put_file(
        &options.executable,
        &format!("{REMOTE_WORKDIR}/bench-device"),
    )?;

    if options.source_snapshot {
        prepare_source_snapshot(&ssh, run_seconds)?;
    }
    let run_seconds = remaining_run_seconds(started, budget, options.max_minutes)?;
    let command = benchmark_command(&arguments, &options.generator_baseline_sha256);
    println!(
        "running {} benchmark (budget {run_seconds}s)",
        options.suite
    );
    let (status, output) = ssh.run(
        &format!("timeout -k 30 {run_seconds} sh -c '{command}'"),
        run_seconds + 60,
    )?;
    guard.failed = status != 0;
    if !output.is_empty() {
        println!("{output}");
    }
    if guard.failed {
        eprintln!("benchmark exited {status}");
    }

    let report_directory = report_directory(workspace_root)?;
    let log = report_directory.join("benchmark.out");
    ssh.get_file(&format!("{REMOTE_WORKDIR}/benchmark.out"), &log)?;
    println!("log: {}", log.display());
    if status == 0 {
        let json = report_directory.join("benchmark.json");
        ssh.get_file(&format!("{REMOTE_WORKDIR}/benchmark.json"), &json)?;
        println!("JSON: {}", json.display());
    }

    finish_run(&api, guard, &pod_id, status, started, cost_per_hour)
}

/// Verifies API authentication and reports pods owned by this worktree.
pub fn check() -> RemoteResult<()> {
    let pods = V2::new(resolve_api_key()?).list_pods()?;
    let owned_ids = owned_pod_ids()?;
    let now = unix_seconds();
    let owned = pods
        .iter()
        .filter(|pod| {
            pod.id
                .as_deref()
                .is_some_and(|pod_id| owned_ids.contains(pod_id))
        })
        .collect::<Vec<_>>();
    let expired = pods
        .iter()
        .filter(|pod| {
            pod.name
                .as_deref()
                .is_some_and(|name| lease_expired(name, now))
        })
        .collect::<Vec<_>>();
    for pod in &owned {
        println!(
            "current-worktree remote pod: {} ({:?})",
            pod.id.as_deref().unwrap_or("missing-id"),
            pod.status
        );
    }
    for pod in &expired {
        println!(
            "expired remote lease: {} ({:?})",
            pod.id.as_deref().unwrap_or("missing-id"),
            pod.status
        );
    }
    println!(
        "RunPod API reachable; auth ok: {} account pod(s), {} owned by this worktree, {} expired lease(s)",
        pods.len(),
        owned.len(),
        expired.len()
    );

    Ok(())
}

/// Deletes pods recorded as owned by this worktree.
pub fn sweep_stale() -> RemoteResult<u32> {
    let api = V2::new(resolve_api_key()?);
    let now = unix_seconds();
    let mut targets = owned_pod_ids()?
        .into_iter()
        .map(|pod_id| (pod_id, "current-worktree ownership"))
        .collect::<BTreeMap<_, _>>();
    for pod in api.list_pods()? {
        if !pod
            .name
            .as_deref()
            .is_some_and(|name| lease_expired(name, now))
        {
            continue;
        }
        let pod_id = pod
            .id
            .filter(|id| !id.is_empty())
            .ok_or_else(|| RemoteError::Api {
                operation: "list pods",
                status: 0,
                body: "expired remote lease is missing its pod id".to_owned(),
            })?;
        targets.entry(pod_id).or_insert("expired lease");
    }
    let mut swept = 0u32;
    for (pod_id, reason) in targets {
        delete_pod(&pod_id)?;
        forget_owned_pod(&owned_pod_path(&pod_id)?);
        std::fs::remove_file(crate::sentry::keep_file_path(&pod_id)).ok();
        println!("deleted remote pod {pod_id} ({reason})");
        swept += 1;
    }
    println!("swept {swept} owned or expired remote pod(s)");

    Ok(swept)
}

fn open_pod(
    api: &V2,
    gpu: GpuTarget,
    image: &str,
    keep_on_fail: bool,
    key_file: PathBuf,
    budget: Duration,
    deadline: Instant,
) -> RemoteResult<(PodGuard, Ssh, String, Option<f64>)> {
    let ownership_directory = ownership_directory()?;
    let name = leased_pod_name(unix_seconds(), budget, keep_on_fail);
    println!("creating {name} (1x secure {})", gpu.device_name());
    let pod = api.create_gate_pod(&name, image, gpu)?;
    let pod_id = pod
        .id
        .filter(|id| !id.is_empty())
        .ok_or_else(|| RemoteError::Api {
            operation: "create pod",
            status: 0,
            body: "response is missing a pod id".to_owned(),
        })?;
    let cost = pod.extras.get("cost").and_then(serde_json::Value::as_f64);
    let ownership_path = ownership_directory.join(format!("{pod_id}{OWNED_POD_SUFFIX}"));
    let guard = PodGuard::new(pod_id.clone(), ownership_path.clone(), keep_on_fail);
    record_owned_pod(&ownership_path)?;
    match spawn_sentry(&pod_id, (budget + SENTRY_GRACE).as_secs()) {
        Ok(pid) => println!("watchdog {pid} guards pod {pod_id}"),
        Err(error) => eprintln!("warning: watchdog did not start ({error})"),
    }

    let routes = wait_until_ssh(api, &pod_id, deadline)?;
    let direct = routes
        .direct
        .expect("wait_until_ssh requires the direct route");
    let host =
        direct
            .host
            .filter(|host| !host.is_empty())
            .ok_or_else(|| RemoteError::Precheck {
                detail: "RunPod direct SSH route is missing its host".to_owned(),
            })?;
    let port = direct.port.ok_or_else(|| RemoteError::Precheck {
        detail: "RunPod direct SSH route is missing its port".to_owned(),
    })?;
    let user = direct
        .username
        .filter(|user| !user.is_empty())
        .unwrap_or_else(|| "root".to_owned());
    println!("direct ssh/sftp: {user}@{host}:{port}");
    let ssh = loop {
        match Ssh::connect(&key_file, &host, port, &user) {
            Ok(ssh) => break ssh,
            Err(error) if Instant::now() < deadline => {
                eprintln!("SSH not ready ({error}); retrying in 5s");
                std::thread::sleep(Duration::from_secs(5));
            }
            Err(error) => return Err(error),
        }
    };

    Ok((guard, ssh, pod_id, cost))
}

fn wait_for_precheck(ssh: &Ssh, gpu: GpuTarget, deadline: Instant) -> RemoteResult<()> {
    loop {
        match precheck(ssh, gpu) {
            Ok(()) => return Ok(()),
            Err(error @ RemoteError::WrongGpu { .. }) => return Err(error),
            Err(error) if Instant::now() < deadline => {
                eprintln!("pod not ready ({error}); retrying in 5s");
                std::thread::sleep(Duration::from_secs(5));
            }
            Err(error) => return Err(error),
        }
    }
}

fn precheck(ssh: &Ssh, gpu: GpuTarget) -> RemoteResult<()> {
    let (status, _) = ssh.run("whoami", 45)?;
    if status != 0 {
        return Err(RemoteError::Precheck {
            detail: format!("whoami exited {status}"),
        });
    }
    let command = "timeout 30 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; \
        echo ---; ldd --version 2>&1 | head -n 1; echo ---; nproc";
    let (status, output) = ssh.run(command, 45)?;
    let expected = gpu.expected_identity();
    let observed_gpu = output.lines().find(|line| line.starts_with("NVIDIA "));
    match observed_gpu {
        Some(observed) if observed == expected => {
            println!("precheck ok:\n{output}");
            Ok(())
        }
        Some(observed) => Err(RemoteError::WrongGpu {
            expected,
            observed: observed.to_owned(),
        }),
        None => Err(RemoteError::Precheck {
            detail: format!("GPU not ready (status {status}):\n{output}"),
        }),
    }
}

fn qualification_command(arguments: &[String], source_snapshot: bool) -> String {
    let environment = if source_snapshot {
        format!("TUISKO_SNAPSHOT={SNAPSHOT_ROOT} ")
    } else {
        String::new()
    };
    let mut command = format!("cd {REMOTE_WORKDIR} && {environment}./qual");
    for argument in arguments {
        command.push(' ');
        command.push_str(argument);
    }
    command.push_str(" > gate.out 2>&1; gate_status=$?; cat gate.out; test $gate_status -eq 0");

    command
}

fn prepare_source_snapshot(ssh: &Ssh, timeout_secs: u64) -> RemoteResult<()> {
    let command = format!(
        "set -eu; python3 -m pip install --quiet --disable-pip-version-check --no-input \
           --break-system-packages \
           'huggingface_hub[hf_xet]==0.34.4'; \
         mkdir -p {SNAPSHOT_ROOT}; \
         HF_HOME=/tmp/tuiskollm/hf-home HF_HUB_DISABLE_PROGRESS_BARS=1 \
         HF_XET_CHUNK_CACHE_SIZE_BYTES=0 hf download unsloth/Qwen3.8-27B-NVFP4 \
           --revision {SNAPSHOT_REVISION} --local-dir {SNAPSHOT_ROOT} \
           --include config.json model.safetensors.index.json model.safetensors model_mtp.safetensors; \
         cd {SNAPSHOT_ROOT}; \
         test \"$(stat -c %s model.safetensors)\" = 22568192096; \
         test \"$(stat -c %s model_mtp.safetensors)\" = 849400392; \
         test \"$(stat -c %s model.safetensors.index.json)\" = 164371"
    );
    println!("downloading pinned source snapshot {SNAPSHOT_REVISION}");
    let (status, output) = ssh.run(&command, timeout_secs)?;
    if status != 0 {
        return Err(RemoteError::Precheck {
            detail: format!("pinned snapshot download exited {status}:\n{output}"),
        });
    }
    println!("pinned source snapshot admitted by exact shard lengths");

    Ok(())
}

fn remaining_run_seconds(
    started: Instant,
    budget: Duration,
    max_minutes: u32,
) -> RemoteResult<u64> {
    let seconds = budget
        .as_secs()
        .saturating_sub(started.elapsed().as_secs())
        .saturating_sub(60);
    if seconds < 30 {
        return Err(RemoteError::DeadlineExceeded {
            minutes: max_minutes,
        });
    }

    Ok(seconds)
}

fn benchmark_command(arguments: &[String], generator_baseline_sha256: &str) -> String {
    let mut command = format!(
        "cd {REMOTE_WORKDIR} && TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT=1 \
         TUISKO_GENERATOR_BASELINE_SHA256={generator_baseline_sha256} ./bench-device"
    );
    for argument in arguments {
        command.push(' ');
        command.push_str(argument);
    }
    command.push_str(
        " --json benchmark.json > benchmark.out 2>&1; benchmark_status=$?; \
         cat benchmark.out; test $benchmark_status -eq 0",
    );

    command
}

fn sanitize_arguments(arguments: &[String]) -> RemoteResult<Vec<String>> {
    arguments
        .iter()
        .map(|argument| {
            if !argument.is_empty()
                && argument.chars().all(|character| {
                    character.is_ascii_alphanumeric() || "._-=:".contains(character)
                })
            {
                Ok(argument.clone())
            } else {
                Err(RemoteError::Precheck {
                    detail: format!("unsafe remote argument {argument:?}"),
                })
            }
        })
        .collect()
}

fn require_sha256(value: &str) -> RemoteResult<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(());
    }

    Err(RemoteError::Precheck {
        detail: "generator baseline hash is not a SHA-256 digest".to_owned(),
    })
}

fn report_path(workspace_root: &Path) -> RemoteResult<PathBuf> {
    Ok(report_directory(workspace_root)?.join("gate.out"))
}

fn report_directory(workspace_root: &Path) -> RemoteResult<PathBuf> {
    let directory = workspace_root
        .join("target/remote-reports")
        .join(unix_seconds().to_string());
    std::fs::create_dir_all(&directory).map_err(|source| RemoteError::Write {
        path: directory.clone(),
        source,
    })?;

    Ok(directory)
}

fn finish_run(
    api: &V2,
    guard: PodGuard,
    pod_id: &str,
    status: u32,
    started: Instant,
    cost_per_hour: Option<f64>,
) -> RemoteResult<()> {
    println!("elapsed: {}s", started.elapsed().as_secs());
    if let Some(cost) = cost_per_hour {
        println!(
            "estimated cost: {:.4} credits",
            cost * started.elapsed().as_secs_f64() / 3600.0
        );
    }

    let retained = guard.keep_on_fail && guard.failed;
    drop(guard);
    if retained {
        return Err(RemoteError::RemoteExit { status });
    }
    match api.get_pod(pod_id) {
        Err(error) if is_missing(&error) => println!("pod {pod_id} gone (API verified)"),
        Err(error) => {
            return Err(RemoteError::Precheck {
                detail: format!("could not verify deletion of pod {pod_id}: {error}"),
            });
        }
        Ok(pod) => {
            return Err(RemoteError::Precheck {
                detail: format!("pod {pod_id} remains present at status {:?}", pod.status),
            });
        }
    }
    if status != 0 {
        return Err(RemoteError::RemoteExit { status });
    }

    Ok(())
}

fn sha256_hex(path: &Path) -> RemoteResult<String> {
    let bytes = std::fs::read(path).map_err(|source| RemoteError::Read {
        what: format!("remote executable {}", path.display()),
        source,
    })?;

    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn ownership_directory() -> RemoteResult<PathBuf> {
    let current = std::env::current_dir().map_err(|source| RemoteError::Read {
        what: "current worktree directory".to_owned(),
        source,
    })?;
    Ok(current.join("target/remote-state/pods"))
}

fn owned_pod_path(pod_id: &str) -> RemoteResult<PathBuf> {
    Ok(ownership_directory()?.join(format!("{pod_id}{OWNED_POD_SUFFIX}")))
}

fn record_owned_pod(path: &Path) -> RemoteResult<()> {
    let directory = path
        .parent()
        .expect("owned pod records always have a parent directory");
    std::fs::create_dir_all(directory).map_err(|source| RemoteError::Write {
        path: directory.to_path_buf(),
        source,
    })?;
    std::fs::write(path, b"owned by this worktree\n").map_err(|source| RemoteError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn forget_owned_pod(path: &Path) {
    if let Err(error) = std::fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        eprintln!(
            "warning: could not remove remote ownership record {}: {error}",
            path.display()
        );
    }
}

fn owned_pod_ids() -> RemoteResult<BTreeSet<String>> {
    let directory = ownership_directory()?;
    let entries = match std::fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(source) => {
            return Err(RemoteError::Read {
                what: format!("remote ownership directory {}", directory.display()),
                source,
            });
        }
    };
    let mut pod_ids = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|source| RemoteError::Read {
            what: format!("remote ownership directory {}", directory.display()),
            source,
        })?;
        if let Some(pod_id) = owned_pod_id(&entry.file_name()) {
            pod_ids.insert(pod_id);
        }
    }
    Ok(pod_ids)
}

fn owned_pod_id(file_name: &std::ffi::OsStr) -> Option<String> {
    let name = file_name.to_str()?;
    let pod_id = name.strip_suffix(OWNED_POD_SUFFIX)?;
    (!pod_id.is_empty()).then(|| pod_id.to_owned())
}

fn leased_pod_name(now: u64, budget: Duration, keep_on_fail: bool) -> String {
    if keep_on_fail {
        return format!("{POD_NAME_PREFIX}-retained-{now}");
    }
    let expires = now
        .saturating_add(budget.as_secs())
        .saturating_add(SENTRY_GRACE.as_secs());
    format!("{POD_NAME_PREFIX}-lease-{expires}-{now}")
}

fn lease_expired(name: &str, now: u64) -> bool {
    let Some(lease) = name.strip_prefix(&format!("{POD_NAME_PREFIX}-lease-")) else {
        return false;
    };
    let expires = lease.split_once('-').map_or(lease, |(expires, _)| expires);
    expires.parse::<u64>().is_ok_and(|expires| expires <= now)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{
        benchmark_command, lease_expired, leased_pod_name, owned_pod_id, qualification_command,
        sanitize_arguments,
    };

    #[test]
    fn qualification_command_preserves_a_failure_after_printing_the_log() {
        let command = qualification_command(
            &["suite::tests".to_owned(), "--nocapture".to_owned()],
            false,
        );

        assert!(command.contains("gate_status=$?"));
        assert!(command.ends_with("test $gate_status -eq 0"));
        assert!(!command.contains("| tee"));
    }

    #[test]
    fn source_qualification_exports_only_the_pinned_snapshot() {
        let command = qualification_command(&["suite::tests".to_owned()], true);

        assert!(command.contains("TUISKO_SNAPSHOT=/tmp/tuiskollm/snapshots/16b6615a"));
    }

    #[test]
    fn benchmark_command_preserves_a_failure_and_writes_json_separately() {
        let command = benchmark_command(
            &[
                "residual-norm".to_owned(),
                "--samples".to_owned(),
                "3".to_owned(),
            ],
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );

        assert!(command.contains("--json benchmark.json"));
        assert!(command.contains("benchmark_status=$?"));
        assert!(command.ends_with("test $benchmark_status -eq 0"));
    }

    #[test]
    fn qualification_arguments_reject_shell_syntax() {
        assert!(sanitize_arguments(&["--nocapture".to_owned()]).is_ok());
        assert!(sanitize_arguments(&["ok; false".to_owned()]).is_err());
    }

    #[test]
    fn sweep_inventory_accepts_only_explicit_ownership_records() {
        assert_eq!(
            owned_pod_id(OsStr::new("pod-from-this-worktree.pod")),
            Some("pod-from-this-worktree".to_owned())
        );
        assert_eq!(owned_pod_id(OsStr::new("tuiskollm-gate-foreign")), None);
        assert_eq!(owned_pod_id(OsStr::new(".pod")), None);
    }

    #[test]
    fn account_wide_sweep_obeys_declared_leases() {
        let leased = leased_pod_name(1_000, std::time::Duration::from_secs(600), false);
        assert!(!lease_expired(&leased, 1_899));
        assert!(lease_expired(&leased, 1_900));

        let retained = leased_pod_name(1_000, std::time::Duration::from_secs(600), true);
        assert!(!lease_expired(&retained, u64::MAX));
        assert!(!lease_expired("tuiskollm-gate-1000", u64::MAX));
        assert!(!lease_expired("tuiskollm-gateway-lease-1", u64::MAX));
    }
}
