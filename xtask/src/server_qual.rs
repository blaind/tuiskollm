//! Production server lifecycle for the external HTTP qualification suite.

use super::{
    CUDA_OXIDE_BUILD_TARGET, build_server, require_device_idle, run_visible, task_cargo_home,
    wait_for_device_idle,
};
use serde_json::Value;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
const PROBE_INTERVAL: Duration = Duration::from_millis(100);
const ROUTE: &str = "mtp-draft-3";

pub(super) fn run(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-server SNAPSHOT".into());
    };
    let snapshot = Path::new(snapshot);
    if !snapshot.is_dir() {
        return Err(format!("snapshot `{}` is not a directory", snapshot.display()).into());
    }

    require_device_idle("server qualification setup")?;
    let qualifier = build_qualifier(root)?;
    build_server(root)?;
    require_device_idle("server qualification setup")?;

    let address = private_address()?;
    let log_path = root.join("target/server-qualification/server.log");
    let mut server = ServerChild::spawn(
        root,
        &root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm"),
        snapshot,
        address,
        &log_path,
    )?;
    if let Err(error) = server.wait_ready(address, &log_path) {
        let stop = server.stop();
        let idle = wait_for_device_idle();
        stop?;
        idle?;
        return Err(error);
    }

    let base_url = format!("http://{address}");
    let qualification = run_visible(Command::new(qualifier).arg(&base_url).current_dir(root));
    let stop = server.stop();
    let idle = wait_for_device_idle();
    qualification?;
    stop?;
    idle?;

    println!(
        "production server qualification passed; lifecycle log: {}",
        log_path.display()
    );
    Ok(())
}

fn build_qualifier(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args(["build", "--package", "tuisko-server-qual", "--release"])
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let executable = root.join("target/release/tuisko-server-qual");
    if !executable.is_file() {
        return Err(format!(
            "host build omitted server qualifier `{}`",
            executable.display()
        )
        .into());
    }
    Ok(executable)
}

fn private_address() -> Result<SocketAddr, Box<dyn Error>> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    drop(listener);
    Ok(address)
}

struct ServerChild {
    child: Option<Child>,
}

impl ServerChild {
    fn spawn(
        root: &Path,
        executable: &Path,
        snapshot: &Path,
        address: SocketAddr,
        log_path: &Path,
    ) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(
            log_path
                .parent()
                .ok_or("server qualification log has no parent")?,
        )?;
        let stdout = File::create(log_path)?;
        let stderr = stdout.try_clone()?;
        let child = Command::new(executable)
            .current_dir(root)
            .args([OsStr::new("serve"), snapshot.as_os_str()])
            .arg(address.to_string())
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()?;
        Ok(Self { child: Some(child) })
    }

    fn wait_ready(&mut self, address: SocketAddr, log_path: &Path) -> Result<(), Box<dyn Error>> {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Some(status) = self.status()? {
                return Err(format!(
                    "production server exited with {status} before becoming ready; inspect {}",
                    log_path.display()
                )
                .into());
            }
            if probe_health(address)? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "production server did not become ready within {} seconds; inspect {}",
                    STARTUP_TIMEOUT.as_secs(),
                    log_path.display()
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
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn probe_health(address: SocketAddr) -> Result<bool, Box<dyn Error>> {
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
    validate_health_response(&response)?;
    Ok(true)
}

fn validate_health_response(response: &[u8]) -> Result<(), Box<dyn Error>> {
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
    let expected = serde_json::json!({"status": "ok", "generation_route": ROUTE});
    if body != expected {
        return Err(format!("health probe returned {body}, expected {expected}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_health_response;

    #[test]
    fn health_probe_requires_the_exact_ready_route() {
        validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"mtp-draft-3\",\"status\":\"ok\"}",
        )
        .unwrap();
        let error = validate_health_response(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"generation_route\":\"target-only\",\"status\":\"ok\"}",
        )
        .unwrap_err();
        assert!(error.to_string().contains("mtp-draft-3"));
    }

    #[test]
    fn health_probe_rejects_non_success_status() {
        let error = validate_health_response(
            b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("503 Service Unavailable"));
    }
}
