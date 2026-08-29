//! Production server lifecycle for the external HTTP qualification suite.

use super::{
    CUDA_OXIDE_BUILD_TARGET, build_server, require_device_idle, run_visible, task_cargo_home,
    wait_for_device_idle,
};
use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
const PROBE_INTERVAL: Duration = Duration::from_millis(100);
/// The Qwen3.8 target's generation route, which is the one `qualify-server` drives.
pub(super) const ROUTE: &str = "mtp-draft-3";
pub(super) const MODEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

pub(super) fn run(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    run_mode(root, arguments, false)
}

pub(super) fn run_long_context(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    run_mode(root, arguments, true)
}

fn run_mode(root: &Path, arguments: &[OsString], long_context: bool) -> Result<(), Box<dyn Error>> {
    let command = if long_context {
        "qualify-server-long-context"
    } else {
        "qualify-server"
    };
    let snapshot = parse_snapshot(arguments, command)?;
    let (tools, mut server) = start(root, snapshot, MODEL, ROUTE, "server qualification setup")?;
    let mut qualification = Command::new(tools.qualifier());
    qualification.arg(server.base_url()).current_dir(root);
    if long_context {
        qualification.arg("--long-context");
    }
    let qualification = run_visible(&mut qualification);
    let stop = server.stop_and_wait();
    qualification?;
    stop?;
    validate_request_log(&fs::read_to_string(server.log_path())?)?;

    println!(
        "production {command} passed; lifecycle log: {}",
        server.log_path().display()
    );
    Ok(())
}

pub(super) fn parse_snapshot<'a>(
    arguments: &'a [OsString],
    command: &str,
) -> Result<&'a Path, Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(format!("usage: cargo run -p xtask -- {command} SNAPSHOT").into());
    };
    let snapshot = Path::new(snapshot);
    if !snapshot.is_dir() {
        return Err(format!("snapshot `{}` is not a directory", snapshot.display()).into());
    }
    Ok(snapshot)
}

pub(super) struct HostTools {
    qualifier: PathBuf,
    benchmark: PathBuf,
    qwen38_flash_next_qualifier: PathBuf,
}

impl HostTools {
    pub(super) fn qualifier(&self) -> &Path {
        &self.qualifier
    }

    pub(super) fn benchmark(&self) -> &Path {
        &self.benchmark
    }

    pub(super) fn qwen38_flash_next_qualifier(&self) -> &Path {
        &self.qwen38_flash_next_qualifier
    }
}

fn build_host_tools(root: &Path) -> Result<HostTools, Box<dyn Error>> {
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args(["build", "--package", "tuisko-server-qual", "--release"])
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let qualifier = root.join("target/release/tuisko-server-qual");
    let benchmark = root.join("target/release/bench-server");
    let qwen38_flash_next_qualifier = root.join("target/release/qwen38-flash-next-server-qual");
    if let Some(missing) = [&qualifier, &benchmark, &qwen38_flash_next_qualifier]
        .into_iter()
        .find(|tool| !tool.is_file())
    {
        return Err(format!("host build omitted server tool `{}`", missing.display()).into());
    }
    Ok(HostTools {
        qualifier,
        benchmark,
        qwen38_flash_next_qualifier,
    })
}

pub(super) fn start(
    root: &Path,
    snapshot: &Path,
    model: &'static str,
    route: &'static str,
    activity: &str,
) -> Result<(HostTools, ProductionServer), Box<dyn Error>> {
    require_device_idle(activity)?;
    let tools = build_host_tools(root)?;
    build_server(root)?;
    require_device_idle(activity)?;

    let address = private_address()?;
    let log_path = root.join("target/server-qualification/server.log");
    let mut server = ProductionServer::spawn(
        root,
        &root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm"),
        snapshot,
        model,
        address,
        log_path,
        route,
    )?;
    if let Err(error) = server.wait_ready() {
        server.stop_and_wait()?;
        return Err(error);
    }
    Ok((tools, server))
}

fn private_address() -> Result<SocketAddr, Box<dyn Error>> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    drop(listener);
    Ok(address)
}

pub(super) struct ProductionServer {
    child: Option<Child>,
    executable: PathBuf,
    address: SocketAddr,
    log_path: PathBuf,
    route: &'static str,
}

impl ProductionServer {
    fn spawn(
        root: &Path,
        executable: &Path,
        snapshot: &Path,
        model: &'static str,
        address: SocketAddr,
        log_path: PathBuf,
        route: &'static str,
    ) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(
            log_path
                .parent()
                .ok_or("server qualification log has no parent")?,
        )?;
        let stdout = File::create(&log_path)?;
        let stderr = stdout.try_clone()?;
        let child = Command::new(executable)
            .current_dir(root)
            .arg("serve")
            .arg(model)
            .arg("--snapshot")
            .arg(snapshot)
            .arg("--address")
            .arg(address.to_string())
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()?;
        Ok(Self {
            child: Some(child),
            executable: executable.to_path_buf(),
            address,
            log_path,
            route,
        })
    }

    fn wait_ready(&mut self) -> Result<(), Box<dyn Error>> {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Some(status) = self.status()? {
                return Err(format!(
                    "production server exited with {status} before becoming ready; inspect {}",
                    self.log_path.display()
                )
                .into());
            }
            if probe_health(self.address, self.route)? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "production server did not become ready within {} seconds; inspect {}",
                    STARTUP_TIMEOUT.as_secs(),
                    self.log_path.display()
                )
                .into());
            }
            std::thread::sleep(PROBE_INTERVAL);
        }
    }

    fn status(&mut self) -> Result<Option<ExitStatus>, Box<dyn Error>> {
        match self.child.as_mut() {
            Some(child) => Ok(child.try_wait()?),
            None => Ok(None),
        }
    }

    fn stop(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        if child.try_wait()?.is_none() {
            child.kill()?;
        }
        let _status = child.wait()?;
        Ok(())
    }

    pub(super) fn base_url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub(super) fn pid(&self) -> Result<u32, Box<dyn Error>> {
        self.child
            .as_ref()
            .map(Child::id)
            .ok_or_else(|| "production server child is no longer running".into())
    }

    pub(super) fn log_path(&self) -> &Path {
        &self.log_path
    }

    pub(super) fn executable(&self) -> &Path {
        &self.executable
    }

    pub(super) fn stop_and_wait(&mut self) -> Result<(), Box<dyn Error>> {
        let stop = self.stop();
        let idle = wait_for_device_idle();
        stop?;
        idle
    }
}

impl Drop for ProductionServer {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn probe_health(address: SocketAddr, route: &str) -> Result<bool, Box<dyn Error>> {
    let mut stream = match TcpStream::connect_timeout(&address, PROBE_TIMEOUT) {
        Ok(stream) => stream,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::WouldBlock
            ) =>
        {
            return Ok(false);
        }
        Err(error) => return Err(error.into()),
    };
    stream.set_read_timeout(Some(PROBE_TIMEOUT))?;
    stream.set_write_timeout(Some(PROBE_TIMEOUT))?;
    write!(
        stream,
        "GET /health HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    validate_health_response(&response, route)?;
    Ok(true)
}

fn validate_health_response(response: &[u8], route: &str) -> Result<(), Box<dyn Error>> {
    let boundary = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("health response omitted the HTTP header boundary")?;
    let headers = std::str::from_utf8(&response[..boundary])?;
    let status = headers.lines().next().unwrap_or_default();
    if status != "HTTP/1.1 200 OK" {
        return Err(format!("health probe returned `{status}`").into());
    }
    let body: Value = serde_json::from_slice(&response[boundary + 4..])?;
    let expected = serde_json::json!({"status": "ok", "generation_route": route});
    if body != expected {
        return Err(format!("health probe returned {body}, expected {expected}").into());
    }
    Ok(())
}

pub(super) fn validate_request_log(log: &str) -> Result<(), Box<dyn Error>> {
    const REQUIRED: [&str; 17] = [
        " ms (+",
        "), prompt ",
        " tok, cached ",
        "%), input ",
        " tok, queue ",
        " ms, prefill B=",
        " tok/s), gen ",
        " tok, ttft ",
        " ms, decode ",
        " tok/s, verify ",
        " (K=",
        " wall ms/v, ",
        " tok/v, mtp accept ",
        "%), route ",
        ", render ",
        ", encode ",
        ", bpe-tail ",
    ];
    let complete = log.lines().find(|line| {
        line.starts_with("TuiskoLLM request ")
            && (line.contains(", finish stop") || line.contains(", finish length"))
    });
    let Some(complete) = complete else {
        return Err("server log omitted a completed request timing line".into());
    };
    if let Some(missing) = REQUIRED.iter().find(|marker| !complete.contains(**marker)) {
        return Err(format!("server request timing line omitted `{missing}`: {complete}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_health_response, validate_request_log};

    #[test]
    fn health_probe_requires_the_exact_ready_route() {
        validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"mtp-draft-3\",\"status\":\"ok\"}",
            "mtp-draft-3",
        )
        .unwrap();
        let error = validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"target-only\",\"status\":\"ok\"}",
            "mtp-draft-3",
        )
        .unwrap_err();
        assert!(error.to_string().contains("mtp-draft-3"));

        validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"mtp-draft-3-b1-1\",\"status\":\"ok\"}",
            "mtp-draft-3-b1-1",
        )
        .unwrap();
        let crossed = validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"mtp-draft-3\",\"status\":\"ok\"}",
            "mtp-draft-3-b1-1",
        )
        .unwrap_err();
        assert!(crossed.to_string().contains("mtp-draft-3-b1-1"));
    }

    #[test]
    fn health_probe_rejects_non_success_status() {
        let error = validate_health_response(
            b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n",
            "mtp-draft-3",
        )
        .unwrap_err();
        assert!(error.to_string().contains("503 Service Unavailable"));
    }

    #[test]
    fn production_log_requires_one_complete_request_timing_line() {
        validate_request_log(
            "startup\nTuiskoLLM request 4: 12 ms (+120.0 ms), prompt 10 tok, cached 5 tok (50.0%), input 5 tok, queue 1.0 ms, prefill B=1 5 tok/2.0 ms (2500.0 tok/s), gen 3 tok, ttft 80.0 ms, decode 50.0 tok/s, verify 1 (K=0/0/0/1), 40.0 wall ms/v, 3.00 tok/v, mtp accept 2/3 (66.7%), route mtp-draft-3, render 1.0 ms, encode 2.0 ms, bpe-tail 4 tok, finish length\n",
        )
        .unwrap();
        let error = validate_request_log("startup only\n").unwrap_err();
        assert!(error.to_string().contains("completed request timing line"));
    }
}
