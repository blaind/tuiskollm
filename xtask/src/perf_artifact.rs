//! Verified worktree-local performance artifact receipts and reuse.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const RECEIPT_SCHEMA: u32 = 1;
const BUILD_RECEIPT: &str = "target/perf-state/build-sm120.json";
const BENCHMARK_EXECUTABLE: &str = "target/cuda-oxide-build-sm120/release/bench-device";
const PTX: &str = "target/cuda/tuisko_kernels_sm120.ptx";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuildReceipt {
    schema_version: u32,
    device_input_sha256: String,
    executable_sha256: String,
    ptx_sha256: String,
    resource_baselines_sha256: String,
    cuda_oxide_revision: String,
    created_unix_milliseconds: u128,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct QualificationReceipt {
    schema_version: u32,
    suite: String,
    device_input_sha256: String,
    device_identity_sha256: String,
    created_unix_milliseconds: u128,
}

pub(crate) fn device_input_sha256(root: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .args([
            "-C",
            path_text(root)?,
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed while fingerprinting device inputs: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let mut paths = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8(path.to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| is_device_input(path));
    paths.sort();
    paths.dedup();

    let mut digest = Sha256::new();
    digest.update(b"tuiskollm-device-input-v1\0");
    for relative in paths {
        let path = root.join(&relative);
        let metadata = fs::symlink_metadata(&path)?;
        digest.update(relative.as_bytes());
        digest.update([0]);
        if metadata.file_type().is_symlink() {
            digest.update(b"symlink\0");
            digest.update(
                fs::read_link(&path)?
                    .to_str()
                    .ok_or_else(|| format!("symlink target for `{relative}` is not UTF-8"))?
                    .as_bytes(),
            );
        } else if metadata.is_file() {
            digest.update(b"file\0");
            digest.update(fs::read(&path)?);
        } else {
            return Err(format!("device input `{relative}` is not a file or symlink").into());
        }
        digest.update([0]);
    }

    Ok(hex_digest(digest.finalize().as_slice()))
}

pub(crate) fn resource_baselines_sha256(
    root: &Path,
    baselines: &[&str],
) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    digest.update(b"tuiskollm-resource-baselines-v1\0");
    for baseline in baselines {
        digest.update(baseline.as_bytes());
        digest.update([0]);
        digest.update(fs::read(root.join(baseline))?);
        digest.update([0]);
    }
    Ok(hex_digest(digest.finalize().as_slice()))
}

pub(crate) fn local_build_is_current(
    root: &Path,
    device_input_sha256: &str,
    resource_baselines_sha256: &str,
    cuda_oxide_revision: &str,
) -> Result<bool, Box<dyn Error>> {
    let Some(receipt) = read_build_receipt(&root.join(BUILD_RECEIPT))? else {
        return Ok(false);
    };
    if !receipt_matches(
        &receipt,
        device_input_sha256,
        resource_baselines_sha256,
        cuda_oxide_revision,
    ) {
        return Ok(false);
    }
    if let Err(error) = verify_artifacts(root, &receipt) {
        eprintln!("ignoring stale local SM120 build receipt: {error}");
        return Ok(false);
    }
    Ok(true)
}

pub(crate) fn restore_build_from_worktrees(
    root: &Path,
    device_input_sha256: &str,
    resource_baselines_sha256: &str,
    cuda_oxide_revision: &str,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    for worktree in worktree_paths(root)? {
        if worktree == root {
            continue;
        }
        let receipt_path = worktree.join(BUILD_RECEIPT);
        let Some(receipt) = read_build_receipt(&receipt_path)? else {
            continue;
        };
        if !receipt_matches(
            &receipt,
            device_input_sha256,
            resource_baselines_sha256,
            cuda_oxide_revision,
        ) {
            continue;
        }
        if let Err(error) = verify_artifacts(&worktree, &receipt) {
            eprintln!(
                "ignoring stale SM120 build receipt from {}: {error}",
                worktree.display()
            );
            continue;
        }
        copy_artifact(
            &worktree.join(BENCHMARK_EXECUTABLE),
            &root.join(BENCHMARK_EXECUTABLE),
        )?;
        copy_artifact(&worktree.join(PTX), &root.join(PTX))?;
        // the sibling can rebuild between hash and copy; only locally verified bytes count
        if let Err(error) = verify_artifacts(root, &receipt) {
            eprintln!(
                "discarding SM120 build restored from {}: {error}",
                worktree.display()
            );
            continue;
        }
        write_json(&root.join(BUILD_RECEIPT), &receipt)?;
        return Ok(Some(worktree));
    }

    Ok(None)
}

pub(crate) fn record_build(
    root: &Path,
    device_input_sha256: String,
    resource_baselines_sha256: String,
    cuda_oxide_revision: &str,
) -> Result<(), Box<dyn Error>> {
    let executable = root.join(BENCHMARK_EXECUTABLE);
    let ptx = root.join(PTX);
    if !executable.is_file() || !ptx.is_file() {
        return Err(
            "cannot record SM120 build receipt without bench-device and PTX artifacts".into(),
        );
    }
    let receipt = BuildReceipt {
        schema_version: RECEIPT_SCHEMA,
        device_input_sha256,
        executable_sha256: file_sha256(&executable)?,
        ptx_sha256: file_sha256(&ptx)?,
        resource_baselines_sha256,
        cuda_oxide_revision: cuda_oxide_revision.to_string(),
        created_unix_milliseconds: now_milliseconds()?,
    };
    write_json(&root.join(BUILD_RECEIPT), &receipt)
}

pub(crate) fn qualification_is_current(
    root: &Path,
    suite: &str,
    device_input_sha256: &str,
    device_identity_sha256: &str,
) -> Result<bool, Box<dyn Error>> {
    let local = qualification_receipt_path(root, suite);
    if qualification_receipt_matches(&local, suite, device_input_sha256, device_identity_sha256)? {
        return Ok(true);
    }
    for worktree in worktree_paths(root)? {
        if worktree == root {
            continue;
        }
        let candidate = qualification_receipt_path(&worktree, suite);
        if qualification_receipt_matches(
            &candidate,
            suite,
            device_input_sha256,
            device_identity_sha256,
        )? {
            let receipt: QualificationReceipt = serde_json::from_slice(&fs::read(candidate)?)?;
            write_json(&local, &receipt)?;
            return Ok(true);
        }
    }

    Ok(false)
}

pub(crate) fn record_qualification(
    root: &Path,
    suite: &str,
    device_input_sha256: String,
    device_identity_sha256: String,
) -> Result<(), Box<dyn Error>> {
    let receipt = QualificationReceipt {
        schema_version: RECEIPT_SCHEMA,
        suite: suite.to_string(),
        device_input_sha256,
        device_identity_sha256,
        created_unix_milliseconds: now_milliseconds()?,
    };
    write_json(&qualification_receipt_path(root, suite), &receipt)
}

fn is_device_input(path: &str) -> bool {
    let admitted = matches!(
        path,
        "Cargo.toml" | "Cargo.lock" | "rust-toolchain" | "rust-toolchain.toml"
    ) || path.starts_with(".cargo/")
        || path.starts_with("crates/")
        || path.starts_with("qual/")
        || path.starts_with("xtask/");
    admitted && !(path.starts_with("qual/baselines/") && path.ends_with(".json"))
}

fn receipt_matches(
    receipt: &BuildReceipt,
    device_input_sha256: &str,
    resource_baselines_sha256: &str,
    cuda_oxide_revision: &str,
) -> bool {
    receipt.schema_version == RECEIPT_SCHEMA
        && receipt.device_input_sha256 == device_input_sha256
        && receipt.resource_baselines_sha256 == resource_baselines_sha256
        && receipt.cuda_oxide_revision == cuda_oxide_revision
}

fn verify_artifacts(root: &Path, receipt: &BuildReceipt) -> Result<(), Box<dyn Error>> {
    for (label, path, expected) in [
        (
            "benchmark executable",
            root.join(BENCHMARK_EXECUTABLE),
            receipt.executable_sha256.as_str(),
        ),
        ("PTX", root.join(PTX), receipt.ptx_sha256.as_str()),
    ] {
        if !path.is_file() {
            return Err(format!(
                "verified build receipt exists but its {label} is missing at {}",
                path.display()
            )
            .into());
        }
        let actual = file_sha256(&path)?;
        if actual != expected {
            return Err(format!(
                "verified build receipt requires {label} SHA-256 {expected}, found {actual}"
            )
            .into());
        }
    }

    Ok(())
}

fn read_build_receipt(path: &Path) -> Result<Option<BuildReceipt>, Box<dyn Error>> {
    if !path.is_file() {
        return Ok(None);
    }
    let receipt: BuildReceipt = match serde_json::from_slice(&fs::read(path)?) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!(
                "ignoring unreadable SM120 build receipt {}: {error}",
                path.display()
            );
            return Ok(None);
        }
    };
    if receipt.schema_version != RECEIPT_SCHEMA {
        eprintln!(
            "ignoring SM120 build receipt {} with schema {}, expected {RECEIPT_SCHEMA}",
            path.display(),
            receipt.schema_version
        );
        return Ok(None);
    }
    Ok(Some(receipt))
}

fn qualification_receipt_path(root: &Path, suite: &str) -> PathBuf {
    root.join(format!("target/perf-state/qualify-{suite}.json"))
}

fn qualification_receipt_matches(
    path: &Path,
    suite: &str,
    device_input_sha256: &str,
    device_identity_sha256: &str,
) -> Result<bool, Box<dyn Error>> {
    if !path.is_file() {
        return Ok(false);
    }
    let receipt: QualificationReceipt = match serde_json::from_slice(&fs::read(path)?) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!(
                "ignoring unreadable qualification receipt {}: {error}",
                path.display()
            );
            return Ok(false);
        }
    };
    if receipt.schema_version != RECEIPT_SCHEMA {
        eprintln!(
            "ignoring qualification receipt {} with schema {}, expected {RECEIPT_SCHEMA}",
            path.display(),
            receipt.schema_version
        );
        return Ok(false);
    }
    Ok(receipt.suite == suite
        && receipt.device_input_sha256 == device_input_sha256
        && receipt.device_identity_sha256 == device_identity_sha256)
}

fn worktree_paths(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let output = Command::new("git")
        .args(["-C", path_text(root)?, "worktree", "list", "--porcelain"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git worktree list failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?
        .lines()
        .filter_map(|line| line.strip_prefix("worktree "))
        .map(PathBuf::from)
        .collect())
}

fn copy_artifact(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    let parent = destination
        .parent()
        .ok_or_else(|| format!("artifact path {} has no parent", destination.display()))?;
    fs::create_dir_all(parent)?;
    fs::copy(source, destination)?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("receipt path {} has no parent", path.display()))?;
    fs::create_dir_all(parent)?;
    let mut json = serde_json::to_vec_pretty(value)?;
    json.push(b'\n');
    fs::write(path, json)?;
    Ok(())
}

pub(crate) fn file_sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(hex_digest(Sha256::digest(fs::read(path)?).as_slice()))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn now_milliseconds() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

#[cfg(test)]
mod tests {
    use super::{
        BENCHMARK_EXECUTABLE, BUILD_RECEIPT, PTX, is_device_input, local_build_is_current,
        record_build, restore_build_from_worktrees,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn device_input_filter_excludes_docs_and_timing_authorities() {
        assert!(is_device_input("crates/tuisko-engine/src/lib.rs"));
        assert!(is_device_input("qual/baselines/nvfp4-down-sm120.txt"));
        assert!(is_device_input("xtask/src/main.rs"));
        assert!(is_device_input("Cargo.lock"));
        assert!(!is_device_input("qual/baselines/nvfp4-down-sm120.json"));
        assert!(!is_device_input("docs/performance.md"));
        assert!(!is_device_input("target/cuda/kernel.ptx"));
    }

    fn scaffold_worktrees(label: &str) -> (PathBuf, PathBuf, PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "tuiskollm-performance-artifact-{label}-{}-{nonce}",
            std::process::id()
        ));
        let root = parent.join("root");
        let source = parent.join("source");
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
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["worktree", "add", "--quiet", "-b", "cache-source"])
                .arg(&source)
                .status()
                .unwrap()
                .success()
        );
        for (path, bytes) in [
            (source.join(BENCHMARK_EXECUTABLE), b"executable".as_slice()),
            (source.join(PTX), b"ptx".as_slice()),
        ] {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        record_build(
            &source,
            "inputs".to_string(),
            "resources".to_string(),
            "oxide",
        )
        .unwrap();
        (parent, root, source)
    }

    #[test]
    fn cross_worktree_reuse_verifies_artifact_hashes() {
        let (parent, root, source) = scaffold_worktrees("reuse");

        let restored = restore_build_from_worktrees(&root, "inputs", "resources", "oxide").unwrap();

        assert_eq!(restored, Some(source));
        assert!(local_build_is_current(&root, "inputs", "resources", "oxide").unwrap());
        fs::write(root.join(BENCHMARK_EXECUTABLE), b"tampered").unwrap();
        assert!(!local_build_is_current(&root, "inputs", "resources", "oxide").unwrap());
        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn restore_discards_copies_that_no_longer_match_the_receipt() {
        let (parent, root, _source) = scaffold_worktrees("rehash");
        let ptx = root.join(PTX);
        fs::create_dir_all(ptx.parent().unwrap()).unwrap();
        // writes through this link vanish, standing in for a sibling rebuild mid-copy
        std::os::unix::fs::symlink("/dev/null", &ptx).unwrap();

        let restored = restore_build_from_worktrees(&root, "inputs", "resources", "oxide").unwrap();

        assert_eq!(restored, None);
        assert!(!root.join(BUILD_RECEIPT).is_file());
        fs::remove_dir_all(parent).unwrap();
    }
}
