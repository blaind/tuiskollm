//! Repository build tasks.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const CUDA_OXIDE_REPOSITORY: &str = "https://github.com/NVlabs/cuda-oxide.git";
const CUDA_OXIDE_REVISION: &str = "1f4d813719012d384f2db12b88efc9314c8bf50c";

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args();
    let _program = arguments.next();
    match (arguments.next().as_deref(), arguments.next()) {
        (Some("bootstrap-cuda-oxide"), None) => bootstrap_cuda_oxide(workspace_root()?),
        _ => Err("usage: cargo run -p xtask -- bootstrap-cuda-oxide".into()),
    }
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest has no workspace parent".into())
}

fn bootstrap_cuda_oxide(root: &Path) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    if !source.join(".git").is_dir() {
        fs::create_dir_all(
            source
                .parent()
                .ok_or("cuda-oxide source path has no parent")?,
        )?;
        run_visible(
            Command::new("git")
                .args(["clone", "--no-checkout", CUDA_OXIDE_REPOSITORY])
                .arg(&source),
        )?;
        run_visible(Command::new("git").arg("-C").arg(&source).args([
            "checkout",
            "--detach",
            CUDA_OXIDE_REVISION,
        ]))?;
    }
    require_cuda_oxide_revision(&source)?;

    let driver_target = root.join("target/cuda-oxide-driver");
    run_visible(
        Command::new("cargo")
            .arg("+nightly-2026-04-03")
            .arg("build")
            .arg("--manifest-path")
            .arg(source.join("Cargo.toml"))
            .args(["--package", "cargo-oxide", "--target-dir"])
            .arg(&driver_target)
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    run_visible(
        Command::new(driver_target.join("debug/cargo-oxide"))
            .arg("setup")
            .current_dir(&source)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS"),
    )?;

    println!(
        "cuda-oxide ready: {} at {}",
        CUDA_OXIDE_REVISION,
        driver_target.join("debug/cargo-oxide").display()
    );
    Ok(())
}

fn run_visible(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(format!("{program} failed with {status}").into());
    }

    Ok(())
}

fn require_cuda_oxide_revision(source: &Path) -> Result<(), Box<dyn Error>> {
    let commit = command_text("git", &["-C", path_text(source)?, "rev-parse", "HEAD"])?;
    if commit.trim() != CUDA_OXIDE_REVISION {
        return Err(format!(
            "cuda-oxide source is at {}, expected {}",
            commit.trim(),
            CUDA_OXIDE_REVISION
        )
        .into());
    }

    Ok(())
}

fn local_cuda_oxide_source(root: &Path) -> PathBuf {
    root.join("target/cuda-oxide-source")
}

fn task_cargo_home(root: &Path) -> PathBuf {
    root.join("target/cargo-home")
}

fn encoded_backend_rustflags(root: &Path, source: &Path) -> Result<String, Box<dyn Error>> {
    let sysroot = command_text("rustc", &["--print", "sysroot"])?;
    let cargo_home = task_cargo_home(root);
    let prefixes = [
        (source, "/cuda-oxide"),
        (cargo_home.as_path(), "/cargo-home"),
        (Path::new(sysroot.trim()), "/rust-toolchain"),
    ];
    let flags = prefixes
        .into_iter()
        .map(|(path, replacement)| {
            Ok(format!(
                "--remap-path-prefix={}={replacement}",
                path_text(path)?
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(flags.join("\u{1f}"))
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let arguments = arguments
        .iter()
        .map(std::ffi::OsStr::new)
        .collect::<Vec<_>>();
    let output = require_success(Path::new(program), &arguments)?;

    Ok(String::from_utf8(output.stdout)?)
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

fn require_success(
    program: &Path,
    arguments: &[&std::ffi::OsStr],
) -> Result<Output, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            program.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(output)
}
