//! Detached watchdog for remote-runner pods.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::key::resolve_api_key;
use crate::v2::{V2, is_missing};
use crate::{RemoteError, RemoteResult};

const POLL_INTERVAL: Duration = Duration::from_secs(20);
const DELETE_RETRY_INTERVAL: Duration = Duration::from_secs(10);
const DELETE_RETRY_BUDGET: Duration = Duration::from_secs(10 * 60);

pub(crate) fn keep_file_path(pod_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tuiskollm-keep-{pod_id}"))
}

/// Prevents the watchdog from deleting a failed pod retained for inspection.
pub fn mark_keep(pod_id: &str) -> RemoteResult<()> {
    let path = keep_file_path(pod_id);
    fs::write(&path, "retained by qualification runner\n")
        .map_err(|source| RemoteError::Write { path, source })
}

/// Starts a detached watchdog for a newly created pod.
pub fn spawn_sentry(pod_id: &str, deadline_seconds: u64) -> RemoteResult<u32> {
    let executable = std::env::current_exe().map_err(|source| RemoteError::Spawn {
        operation: "locating xtask",
        source,
    })?;
    let parent = std::process::id().to_string();
    let deadline = deadline_seconds.to_string();
    let arguments = [
        "remote",
        "sentry",
        pod_id,
        "--parent",
        &parent,
        "--deadline-secs",
        &deadline,
    ];
    let log = sentry_log(pod_id);
    let child = Command::new("setsid")
        .arg(executable)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .map_err(|source| RemoteError::Spawn {
            operation: "starting remote-runner watchdog",
            source,
        })?;

    Ok(child.id())
}

/// Runs the watchdog loop in its detached process.
pub fn run_sentry(pod_id: &str, parent: u32, deadline_seconds: u64) -> RemoteResult<()> {
    let started = Instant::now();
    let deadline = Duration::from_secs(deadline_seconds);
    loop {
        let expired = started.elapsed() >= deadline;
        if (parent_process_alive(parent) && !expired) || keep_file_path(pod_id).exists() {
            std::thread::sleep(POLL_INTERVAL);
            continue;
        }

        delete_with_retries(pod_id)?;
        return Ok(());
    }
}

pub(crate) fn delete_pod(pod_id: &str) -> RemoteResult<()> {
    let api = V2::new(resolve_api_key()?);
    match api.delete_pod(pod_id) {
        Ok(()) => Ok(()),
        Err(error) if is_missing(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

fn delete_with_retries(pod_id: &str) -> RemoteResult<()> {
    let started = Instant::now();
    loop {
        match delete_pod(pod_id) {
            Ok(()) => return Ok(()),
            Err(error) if started.elapsed() >= DELETE_RETRY_BUDGET => {
                return Err(RemoteError::Precheck {
                    detail: format!("watchdog could not delete pod {pod_id}: {error}"),
                });
            }
            Err(_) => std::thread::sleep(DELETE_RETRY_INTERVAL),
        }
    }
}

fn sentry_log(pod_id: &str) -> Stdio {
    let path = std::env::temp_dir().join(format!("tuiskollm-sentry-{pod_id}.log"));
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null())
}

fn parent_process_alive(pid: u32) -> bool {
    PathBuf::from(format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use super::parent_process_alive;

    #[test]
    fn current_process_is_alive() {
        assert!(parent_process_alive(std::process::id()));
    }

    #[test]
    fn impossible_pid_is_not_alive() {
        assert!(!parent_process_alive(u32::MAX));
    }
}
