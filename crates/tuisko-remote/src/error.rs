//! Error types for remote gate execution.

use thiserror::Error;

/// Failures while driving a RunPod pod from the local machine.
#[derive(Debug, Error)]
pub enum RemoteError {
    /// No API key in env, credentials file, or `.env`.
    #[error(
        "no RunPod API key found; set RUNPOD_API_KEY, write ~/.runpod/credentials.json, or add a .env file"
    )]
    MissingKey,

    /// Reading a local file failed.
    #[error("reading {what}: {source}")]
    Read {
        /// Human name of the input being read.
        what: String,
        /// Underlying I/O failure.
        source: std::io::Error,
    },

    /// Writing a local file failed.
    #[error("writing {path}: {source}")]
    Write {
        /// Destination path.
        path: std::path::PathBuf,
        /// Underlying I/O failure.
        source: std::io::Error,
    },

    /// A RunPod JSON document failed to parse.
    #[error("parsing RunPod JSON: {source}")]
    Json {
        /// Underlying parse failure.
        source: serde_json::Error,
    },

    /// RunPod answered a request with a non-success status.
    #[error("RunPod API {operation} failed with HTTP {status}: {body}")]
    Api {
        /// Short name of the operation.
        operation: &'static str,
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },

    /// The connection to RunPod failed.
    #[error("RunPod API {operation} network error: {source}")]
    Network {
        /// Short name of the operation.
        operation: &'static str,
        /// Underlying transport failure.
        source: ureq::Error,
    },

    /// The pod never reached RUNNING within its startup budget.
    #[error("pod did not reach RUNNING within {seconds}s")]
    PodNotRunning {
        /// Seconds waited.
        seconds: u64,
    },

    /// Spawning a local process failed.
    #[error("spawning {operation}: {source}")]
    Spawn {
        /// Short name of the process being spawned.
        operation: &'static str,
        /// Underlying spawn failure.
        source: std::io::Error,
    },

    /// The pod's GPU or OS failed the prechecks.
    #[error("pod precheck failed:\n{detail}")]
    Precheck {
        /// Observed precheck output.
        detail: String,
    },

    /// An I/O operation around the SSH session failed.
    #[error("ssh {operation} failed: {source}")]
    SshIo {
        /// Short name of the operation.
        operation: &'static str,
        /// Underlying I/O failure.
        source: std::io::Error,
    },

    /// The SSH protocol or session failed.
    #[error("ssh {operation} failed: {source}")]
    Ssh {
        /// Short name of the operation.
        operation: &'static str,
        /// Underlying SSH failure.
        source: russh::Error,
    },

    /// The configured SSH private key could not be loaded.
    #[error("loading SSH private key: {source}")]
    SshKey {
        /// Underlying key-decoding failure.
        source: russh::keys::Error,
    },

    /// An SFTP operation failed.
    #[error("SFTP {operation} failed: {source}")]
    Sftp {
        /// Short name of the operation.
        operation: &'static str,
        /// Underlying SFTP failure.
        source: russh_sftp::client::error::Error,
    },

    /// The remote command exited non-zero.
    #[error("remote execution exited with status {status}")]
    RemoteExit {
        /// Remote exit status.
        status: u32,
    },

    /// The overall run budget expired.
    #[error("remote execution deadline exceeded ({minutes} min); pod deleted")]
    DeadlineExceeded {
        /// Configured budget in minutes.
        minutes: u32,
    },
}
