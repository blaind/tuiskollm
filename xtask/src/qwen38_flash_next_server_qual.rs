//! Production lifecycle for the Qwen3.8 Flash-Next HTTP gates.
//!
//! Cross-model qualification is sequential because two resident models do not fit concurrently.

use super::server_qual::validate_request_log;
use super::wait_for_device_idle;
use super::{require_performance_device_idle, run_visible, server_qual, server_qualification};
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Health route the Flash-Next server publishes.
const ROUTE: &str = "single-slot-b1-1";
const MODEL: &str = "RadixArk/Qwen3.8-Flash-Next-NVFP4";

pub(super) fn run(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let (snapshot, sibling) = parse(arguments, "qualify-qwen38-flash-next-server")?;
    let base = drive(root, snapshot, &[], "Flash-Next server qualification setup")?;
    println!("production qualify-qwen38-flash-next-server passed; lifecycle log: {base}");
    let Some(sibling) = sibling else {
        println!(
            "no sibling snapshot given; the cross-target half of the multi-model check was skipped"
        );
        return Ok(());
    };

    require_performance_device_idle()?;
    let executable = root
        .join(super::CUDA_OXIDE_BUILD_TARGET)
        .join("release/tuiskollm");
    server_qualification::qualify_sibling(&executable, sibling)?;
    wait_for_device_idle()?;
    println!(
        "the sibling target at {} still serves from the same build",
        sibling.display()
    );
    Ok(())
}

pub(super) fn bench(root: &Path, arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    let (snapshot, sibling) = parse(arguments, "bench-qwen38-flash-next-server")?;
    if sibling.is_some() {
        return Err("usage: cargo run -p xtask -- bench-qwen38-flash-next-server SNAPSHOT".into());
    }
    let base = drive(
        root,
        snapshot,
        &["--measure"],
        "Flash-Next server measurement setup",
    )?;
    println!("bench-qwen38-flash-next-server completed; lifecycle log: {base}");
    Ok(())
}

/// Starts the server, runs the host suite against it, and stops it whatever the suite said.
fn drive(
    root: &Path,
    snapshot: &Path,
    extra: &[&str],
    activity: &str,
) -> Result<String, Box<dyn Error>> {
    let (tools, mut server) = server_qual::start(root, snapshot, MODEL, ROUTE, activity)?;
    let mut suite = Command::new(tools.qwen38_flash_next_qualifier());
    suite.arg(server.base_url()).current_dir(root);
    suite.args(extra);
    let suite = run_visible(&mut suite);
    let stop = server.stop_and_wait();
    suite?;
    stop?;
    // The request log is the server's own account of what it just served, and it is checked here
    // rather than over HTTP because it is the only place a served request's timing, its cached
    // prefix, and its finish reason are all written down together.
    validate_request_log(&fs::read_to_string(server.log_path())?)?;

    Ok(server.log_path().display().to_string())
}

fn parse<'a>(
    arguments: &'a [OsString],
    command: &str,
) -> Result<(&'a Path, Option<&'a Path>), Box<dyn Error>> {
    let (snapshot, sibling) = match arguments {
        [snapshot] => (snapshot, None),
        [snapshot, sibling] => (snapshot, Some(sibling)),
        _ => {
            return Err(format!(
                "usage: cargo run -p xtask -- {command} SNAPSHOT [SIBLING_SNAPSHOT]"
            )
            .into());
        }
    };
    let snapshot = require_directory(snapshot)?;
    let sibling = sibling
        .map(|sibling| require_directory(sibling))
        .transpose()?;
    Ok((snapshot, sibling))
}

fn require_directory(path: &OsString) -> Result<&Path, Box<dyn Error>> {
    let path = Path::new(path);
    if !path.is_dir() {
        return Err(format!("snapshot `{}` is not a directory", path.display()).into());
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::{ROUTE, parse};
    use std::ffi::OsString;

    #[test]
    fn the_driver_probes_the_flash_next_route_rather_than_the_qwen38_one() {
        // Both suites share one server lifecycle, so the route each one waits for has to be its
        // own: a Flash-Next server that came up healthy on `mtp-draft-3` would mean the readiness
        // probe stopped checking which target it reached.
        assert_eq!(ROUTE, "single-slot-b1-1");
        assert_ne!(ROUTE, crate::server_qual::ROUTE);
    }

    #[test]
    fn the_sibling_snapshot_is_optional_and_a_third_argument_is_not_admitted() {
        let root = OsString::from(".");
        let (snapshot, sibling) = parse(
            std::slice::from_ref(&root),
            "qualify-qwen38-flash-next-server",
        )
        .expect("one existing directory is admitted");
        assert_eq!(snapshot, std::path::Path::new("."));
        assert!(sibling.is_none());

        let pair = [root.clone(), root.clone()];
        let (_, sibling) =
            parse(&pair, "qualify-qwen38-flash-next-server").expect("two directories are admitted");
        assert!(sibling.is_some());

        let triple = [root.clone(), root.clone(), root];
        assert!(parse(&triple, "qualify-qwen38-flash-next-server").is_err());
        assert!(parse(&[], "qualify-qwen38-flash-next-server").is_err());
        assert!(
            parse(
                std::slice::from_ref(&OsString::from("no-such-directory")),
                "qualify-qwen38-flash-next-server"
            )
            .is_err()
        );
    }
}
