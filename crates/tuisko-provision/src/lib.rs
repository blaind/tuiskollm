//! Exact pinned Hugging Face snapshot acquisition and cache admission.

use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const HUB_ENDPOINT: &str = "https://huggingface.co";
const QWEN38_REQUIRED_FILES: [RequiredFile; 7] = [
    RequiredFile::new(
        "config.json",
        22_564,
        "b6f6347774036d406eabed6cfffb0fec424ba075",
        "1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c",
    ),
    RequiredFile::new(
        "model.safetensors.index.json",
        164_371,
        "7608ff001dbfc8936318df32aaaaef7c8c9f340d",
        "429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f",
    ),
    RequiredFile::new(
        "model.safetensors",
        22_568_192_096,
        "c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05",
        "c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05",
    ),
    RequiredFile::new(
        "model_mtp.safetensors",
        849_400_392,
        "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
        "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
    ),
    RequiredFile::new(
        "tokenizer.json",
        19_989_325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    RequiredFile::new(
        "chat_template.jinja",
        9_993,
        "a087700658910c336c9ca9f5780a75a3cdd4fcdd",
        "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
    ),
    RequiredFile::new(
        "generation_config.json",
        214,
        "0bc3addd19dc59c5c8899fc1fb887d50b592e7c3",
        "d0d0ed2e37cdfafef4a5067d5ea2407b05f4fb50526e47c008a5b235d50240fb",
    ),
];
const QWEN35_REQUIRED_FILES: [RequiredFile; 5] = [
    RequiredFile::new(
        "config.json",
        5_741,
        "3626408859a3e39b243890ff537e827bfb477991",
        "777188d9226a13c2e1db183655c10854be290e95e26619e17095bdaa45a5e99e",
    ),
    RequiredFile::new(
        "model.safetensors",
        9_361_048_680,
        "a1304acf8325cee90075bf8eb7f5e973ef893c7d497ded92e7d720f0e7178749",
        "a1304acf8325cee90075bf8eb7f5e973ef893c7d497ded92e7d720f0e7178749",
    ),
    RequiredFile::new(
        "tokenizer.json",
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    RequiredFile::new(
        "chat_template.jinja",
        7_756,
        "a585dec894e63da457d9440ec6aa7caa16d20860",
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    RequiredFile::new(
        "generation_config.json",
        115,
        "affcdf188ba614483df3d704e436a8e2da880075",
        "0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e",
    ),
];
const QWEN36_REQUIRED_FILES: [RequiredFile; 9] = [
    RequiredFile::new(
        "config.json",
        58_110,
        "ec1f82c006fd5db6d0b5fa1a59e16fbda9a5c1dd",
        "58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc",
    ),
    RequiredFile::new(
        "hf_quant_config.json",
        35_085,
        "8deefe44cc69cfab994141ea6b9493b03c864061",
        "75fe7cc8d5836b58734e05ee67423a4ce91d602aaad45c8173a1b7597cd57663",
    ),
    RequiredFile::new(
        "model.safetensors.index.json",
        13_726_227,
        "d67403a4e9793c0ba8a136baf14b3b76ec7b32c822267978084895e07ebd8a3e",
        "d67403a4e9793c0ba8a136baf14b3b76ec7b32c822267978084895e07ebd8a3e",
    ),
    RequiredFile::new(
        "model-00001-of-00003.safetensors",
        10_006_877_608,
        "07141c2db92e47bc08777132cd1a0323faf300eab3a7d7c111bc2bf075fda050",
        "07141c2db92e47bc08777132cd1a0323faf300eab3a7d7c111bc2bf075fda050",
    ),
    RequiredFile::new(
        "model-00002-of-00003.safetensors",
        10_003_595_752,
        "6dea9c759a0f941cf9e1cc1501216b0e107966a66c90425050e212efbd053f02",
        "6dea9c759a0f941cf9e1cc1501216b0e107966a66c90425050e212efbd053f02",
    ),
    RequiredFile::new(
        "model-00003-of-00003.safetensors",
        3_413_864_960,
        "9758875fc55e49561165f4a44342b654c123c3e25c6811b34abb83553fb1a164",
        "9758875fc55e49561165f4a44342b654c123c3e25c6811b34abb83553fb1a164",
    ),
    RequiredFile::new(
        "tokenizer.json",
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    RequiredFile::new(
        "chat_template.jinja",
        7_764,
        "a8755d827c0a7b614c246c4060dfd58ab352a8ff",
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    ),
    RequiredFile::new(
        "generation_config.json",
        202,
        "023756cfadf88e5bf69eefeee3e172f38c448d64",
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    ),
];

#[derive(Clone, Copy)]
struct RequiredFile {
    name: &'static str,
    bytes: u64,
    blob: &'static str,
    sha256: &'static str,
}

impl RequiredFile {
    const fn new(name: &'static str, bytes: u64, blob: &'static str, sha256: &'static str) -> Self {
        Self {
            name,
            bytes,
            blob,
            sha256,
        }
    }
}

/// Exact served model whose pinned Hugging Face snapshot can be provisioned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProvisionedModel {
    /// `unsloth/Qwen3.8-27B-NVFP4`.
    Qwen38,
    /// `AxionML/Qwen3.5-9B-NVFP4`.
    Qwen35,
    /// `nvidia/Qwen3.6-35B-A3B-NVFP4`.
    Qwen36,
}

impl ProvisionedModel {
    /// Every exact model with a pinned download manifest.
    pub const ALL: [Self; 3] = [Self::Qwen38, Self::Qwen35, Self::Qwen36];

    /// Exact Hugging Face repository identifier.
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Qwen38 => Qwen38_27B::MODEL_ID,
            Self::Qwen35 => Qwen35_9B::MODEL_ID,
            Self::Qwen36 => Qwen36Moe35B::MODEL_ID,
        }
    }

    /// Immutable Hugging Face revision.
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Qwen38 => Qwen38_27B::REVISION,
            Self::Qwen35 => Qwen35_9B::REVISION,
            Self::Qwen36 => Qwen36Moe35B::REVISION,
        }
    }

    const fn required_files(self) -> &'static [RequiredFile] {
        match self {
            Self::Qwen38 => &QWEN38_REQUIRED_FILES,
            Self::Qwen35 => &QWEN35_REQUIRED_FILES,
            Self::Qwen36 => &QWEN36_REQUIRED_FILES,
        }
    }
}

/// Observable work performed while provisioning one pinned snapshot file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProvisioningStage {
    /// A resumed partial file is being hashed before its HTTP range request.
    Verifying,
    /// Bytes are being received from Hugging Face and appended to the cache blob.
    Downloading,
    /// The complete blob is being synchronized and installed into the snapshot.
    Finalizing,
}

/// Exact per-file and whole-snapshot progress for terminal or API reporting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProvisioningProgress {
    stage: ProvisioningStage,
    file: &'static str,
    file_bytes: u64,
    file_total: u64,
    completed_bytes: u64,
    total_bytes: u64,
}

impl ProvisioningProgress {
    /// Current provisioning stage.
    pub const fn stage(self) -> ProvisioningStage {
        self.stage
    }

    /// Exact filename from the pinned inventory.
    pub const fn file(self) -> &'static str {
        self.file
    }

    /// Bytes processed for the current file and stage.
    pub const fn file_bytes(self) -> u64 {
        self.file_bytes
    }

    /// Exact byte length of the current file.
    pub const fn file_total(self) -> u64 {
        self.file_total
    }

    /// Bytes processed through the current file in the missing-file inventory.
    pub const fn completed_bytes(self) -> u64 {
        self.completed_bytes
    }

    /// Exact logical bytes across all initially missing files.
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

struct FileProgress<'a> {
    required: &'a RequiredFile,
    completed_before: u64,
    total_bytes: u64,
    report: &'a mut dyn FnMut(ProvisioningProgress) -> Result<(), String>,
}

impl FileProgress<'_> {
    fn emit(&mut self, stage: ProvisioningStage, file_bytes: u64) -> Result<(), String> {
        if file_bytes > self.required.bytes {
            return Err(format!(
                "provisioning progress for {} reached {file_bytes} of {} bytes",
                self.required.name, self.required.bytes,
            ));
        }
        let completed_bytes = self
            .completed_before
            .checked_add(file_bytes)
            .ok_or("Hugging Face provisioning byte count overflows")?;
        (self.report)(ProvisioningProgress {
            stage,
            file: self.required.name,
            file_bytes,
            file_total: self.required.bytes,
            completed_bytes,
            total_bytes: self.total_bytes,
        })
    }
}

/// Resolved local snapshot and optional work performed to provision it.
pub struct SnapshotResolution {
    /// Exact admitted snapshot directory.
    pub path: PathBuf,
    /// Download summary, or `None` when the snapshot was already local.
    pub provisioning: Option<Provisioning>,
}

/// Completed snapshot provisioning work.
pub struct Provisioning {
    /// Wall time spent downloading and installing the missing files.
    pub elapsed: Duration,
    /// Number of files missing from the snapshot before provisioning.
    pub files: usize,
    /// Exact logical bytes covered by those files.
    pub bytes: u64,
}

/// Resolves an explicit snapshot or provisions one model's pinned Hugging Face revision.
pub fn resolve_snapshot(
    model: ProvisionedModel,
    explicit: Option<PathBuf>,
) -> Result<SnapshotResolution, String> {
    resolve_snapshot_with_progress(model, explicit, |_| Ok(()))
}

/// Resolves or provisions the pinned revision while reporting exact byte progress.
pub fn resolve_snapshot_with_progress(
    model: ProvisionedModel,
    explicit: Option<PathBuf>,
    mut report: impl FnMut(ProvisioningProgress) -> Result<(), String>,
) -> Result<SnapshotResolution, String> {
    if let Some(path) = explicit {
        return verified_local_resolution(model, path);
    }
    let environment = |name: &str| std::env::var_os(name);
    if let Some(path) = nonempty_environment(&environment, "TUISKO_SNAPSHOT") {
        return verified_local_resolution(model, path.into());
    }

    let cache = hub_cache(&environment)?;
    let snapshot = snapshot_path(&cache, model);
    let missing = match inspect_snapshot(&snapshot, model.required_files())? {
        SnapshotState::Complete => return Ok(local_resolution(snapshot)),
        SnapshotState::Missing(missing) => missing,
    };
    if offline(&environment) {
        return Err(format!(
            "the pinned snapshot is missing {} required file(s) at {} and HF_HUB_OFFLINE is enabled",
            missing.len(),
            snapshot.display(),
        ));
    }

    let started = Instant::now();
    let token = hf_token(&environment)?;
    let total_bytes = missing.iter().map(|file| file.bytes).sum();
    let mut completed_bytes = 0;
    for required in missing.iter().copied() {
        let mut progress = FileProgress {
            required,
            completed_before: completed_bytes,
            total_bytes,
            report: &mut report,
        };
        download_file(model, &cache, &snapshot, token.as_deref(), &mut progress)?;
        completed_bytes = completed_bytes
            .checked_add(required.bytes)
            .ok_or("Hugging Face provisioning byte count overflows")?;
    }
    let SnapshotState::Complete = inspect_snapshot(&snapshot, model.required_files())? else {
        return Err(format!(
            "the Hugging Face download completed without the exact required files at {}",
            snapshot.display(),
        ));
    };

    Ok(SnapshotResolution {
        path: snapshot,
        provisioning: Some(Provisioning {
            elapsed: started.elapsed(),
            files: missing.len(),
            bytes: total_bytes,
        }),
    })
}

fn local_resolution(path: PathBuf) -> SnapshotResolution {
    SnapshotResolution {
        path,
        provisioning: None,
    }
}

fn verified_local_resolution(
    model: ProvisionedModel,
    path: PathBuf,
) -> Result<SnapshotResolution, String> {
    match inspect_snapshot(&path, model.required_files())? {
        SnapshotState::Complete => Ok(local_resolution(path)),
        SnapshotState::Missing(missing) => Err(format!(
            "the explicit snapshot at {} is missing {} required file(s)",
            path.display(),
            missing.len(),
        )),
    }
}

fn download_file(
    model: ProvisionedModel,
    cache: &Path,
    snapshot: &Path,
    token: Option<&str>,
    progress: &mut FileProgress<'_>,
) -> Result<(), String> {
    let required = progress.required;
    let repository = repository_folder(model);
    let repository_root = cache.join(&repository);
    let blob = repository_root.join("blobs").join(required.blob);
    let lock_path = cache
        .join(".locks")
        .join(&repository)
        .join(format!("{}.lock", required.blob));
    create_parent(&lock_path)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| {
            format!(
                "opening Hugging Face cache lock {}: {error}",
                lock_path.display()
            )
        })?;
    lock.lock().map_err(|error| {
        format!(
            "locking Hugging Face cache entry {}: {error}",
            required.name
        )
    })?;
    if exact_file_length(&blob, required.bytes)? {
        progress.emit(ProvisioningStage::Finalizing, required.bytes)?;
        install_snapshot_link(snapshot, required)?;
        return Ok(());
    }
    validate_remote_metadata(model, required, token)?;
    create_parent(&blob)?;
    let incomplete = PathBuf::from(format!("{}.incomplete", blob.display()));
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&incomplete)
        .map_err(|error| format!("opening partial download {}: {error}", incomplete.display()))?;
    let offset = resume_offset(&file, required.bytes, &incomplete)?;

    let mut hasher = Sha256::new();
    if offset != 0 {
        hash_prefix(&mut file, offset, &mut hasher, &incomplete, progress)?;
    }
    if offset < required.bytes {
        progress.emit(ProvisioningStage::Downloading, offset)?;
        append_remote(model, &mut file, offset, token, &mut hasher, progress)?;
    }
    progress.emit(ProvisioningStage::Finalizing, required.bytes)?;
    file.sync_all()
        .map_err(|error| format!("syncing {}: {error}", incomplete.display()))?;
    let actual_bytes = file
        .metadata()
        .map_err(|error| format!("reading {}: {error}", incomplete.display()))?
        .len();
    if actual_bytes != required.bytes {
        return Err(format!(
            "downloaded {} bytes for {}, expected {}",
            actual_bytes, required.name, required.bytes,
        ));
    }
    let digest = hex_digest(hasher.finalize().as_slice());
    if digest != required.sha256 {
        file.set_len(0).map_err(|error| {
            format!(
                "resetting corrupt download {}: {error}",
                incomplete.display()
            )
        })?;
        return Err(format!(
            "downloaded {} has SHA-256 {digest}, expected {}",
            required.name, required.sha256,
        ));
    }
    drop(file);
    fs::rename(&incomplete, &blob)
        .map_err(|error| format!("installing Hugging Face blob {}: {error}", blob.display()))?;
    install_snapshot_link(snapshot, required)
}

fn validate_remote_metadata(
    model: ProvisionedModel,
    required: &RequiredFile,
    token: Option<&str>,
) -> Result<(), String> {
    let agent = agent(0);
    let url = resolve_url(model, required.name);
    let request = authorize(agent.head(&url), token);
    let response = request.call().map_err(|error| {
        format!(
            "reading Hugging Face metadata for {}: {error}",
            required.name
        )
    })?;
    let status = response.status().as_u16();
    if !matches!(status, 200 | 302 | 307) {
        return Err(format!(
            "Hugging Face metadata for {} returned HTTP {status}",
            required.name,
        ));
    }
    require_header(&response, "x-repo-commit", model.revision(), required.name)?;
    require_pinned_etag(
        header_value(&response, "x-linked-etag"),
        header_value(&response, "etag"),
        required,
    )
}

fn header_value<'r>(response: &'r ureq::http::Response<ureq::Body>, name: &str) -> Option<&'r str> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

fn require_pinned_etag(
    linked: Option<&str>,
    etag: Option<&str>,
    required: &RequiredFile,
) -> Result<(), String> {
    let actual = linked.or(etag);
    let normalized =
        actual.map(|value| value.strip_prefix("W/").unwrap_or(value).trim_matches('"'));
    if normalized == Some(required.blob) {
        Ok(())
    } else {
        Err(format!(
            "Hugging Face metadata for {} has etag {actual:?}, expected {}",
            required.name, required.blob,
        ))
    }
}

fn append_remote(
    model: ProvisionedModel,
    file: &mut File,
    offset: u64,
    token: Option<&str>,
    hasher: &mut Sha256,
    progress: &mut FileProgress<'_>,
) -> Result<(), String> {
    let required = progress.required;
    let agent = agent(10);
    let url = resolve_url(model, required.name);
    let mut request = authorize(agent.get(&url), token);
    if offset != 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let response = request
        .call()
        .map_err(|error| format!("downloading {}: {error}", required.name))?;
    let status = response.status().as_u16();
    let expected_status = if offset == 0 { 200 } else { 206 };
    if status != expected_status {
        return Err(format!(
            "downloading {} at byte {offset} returned HTTP {status}, expected {expected_status}",
            required.name,
        ));
    }
    if offset != 0 {
        let expected = format!("bytes {offset}-{}/{}", required.bytes - 1, required.bytes);
        require_header(&response, "content-range", &expected, required.name)?;
    }

    let mut reader = response.into_body().into_reader();
    append_body(&mut reader, file, offset, hasher, progress)
}

fn append_body(
    reader: &mut impl Read,
    file: &mut File,
    offset: u64,
    hasher: &mut Sha256,
    progress: &mut FileProgress<'_>,
) -> Result<(), String> {
    let required = progress.required;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut received = offset;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("receiving {}: {error}", required.name))?;
        if read == 0 {
            break;
        }
        if read as u64 > required.bytes - received {
            return Err(format!(
                "downloading {} received more than the expected {} bytes",
                required.name, required.bytes,
            ));
        }
        file.write_all(&buffer[..read])
            .map_err(|error| format!("writing {}: {error}", required.name))?;
        hasher.update(&buffer[..read]);
        received = received
            .checked_add(read as u64)
            .ok_or("Hugging Face file byte count overflows")?;
        progress.emit(ProvisioningStage::Downloading, received)?;
    }
    Ok(())
}

fn resume_offset(file: &File, limit: u64, path: &Path) -> Result<u64, String> {
    let offset = file
        .metadata()
        .map_err(|error| format!("reading partial download {}: {error}", path.display()))?
        .len();
    if offset <= limit {
        return Ok(offset);
    }
    file.set_len(0)
        .map_err(|error| format!("resetting oversized download {}: {error}", path.display()))?;
    Ok(0)
}

fn hash_prefix(
    file: &mut File,
    bytes: u64,
    hasher: &mut Sha256,
    path: &Path,
    progress: &mut FileProgress<'_>,
) -> Result<(), String> {
    use std::io::{Seek, SeekFrom};

    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("seeking {}: {error}", path.display()))?;
    let mut remaining = bytes;
    let mut hashed = 0;
    let mut buffer = vec![0_u8; 1024 * 1024];
    while remaining != 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        file.read_exact(&mut buffer[..limit])
            .map_err(|error| format!("reading partial download {}: {error}", path.display()))?;
        hasher.update(&buffer[..limit]);
        remaining -= limit as u64;
        hashed += limit as u64;
        progress.emit(ProvisioningStage::Verifying, hashed)?;
    }
    Ok(())
}

fn agent(max_redirects: u32) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .https_only(true)
        .http_status_as_error(false)
        .max_redirects(max_redirects)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(120)))
        .build();
    ureq::Agent::new_with_config(config)
}

fn authorize(
    request: ureq::RequestBuilder<ureq::typestate::WithoutBody>,
    token: Option<&str>,
) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    match token {
        Some(token) => request.header("Authorization", format!("Bearer {token}")),
        None => request,
    }
}

fn require_header(
    response: &ureq::http::Response<ureq::Body>,
    name: &str,
    expected: &str,
    filename: &str,
) -> Result<(), String> {
    let actual = response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_matches('"'));
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "Hugging Face metadata for {filename} has {name} {actual:?}, expected {expected}",
        ))
    }
}

fn exact_file_length(path: &Path, expected: u64) -> Result<bool, String> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.len() == expected => Ok(true),
        Ok(metadata) if metadata.is_file() => Err(format!(
            "Hugging Face blob {} has {} bytes, expected {expected}; delete {} to re-download",
            path.display(),
            metadata.len(),
            path.display(),
        )),
        Ok(_) => Err(format!("{} is not a regular file", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("reading {}: {error}", path.display())),
    }
}

fn install_snapshot_link(snapshot: &Path, required: &RequiredFile) -> Result<(), String> {
    fs::create_dir_all(snapshot)
        .map_err(|error| format!("creating {}: {error}", snapshot.display()))?;
    let link = snapshot.join(required.name);
    let temporary = snapshot.join(format!(".{}.{}.tmp", required.name, std::process::id()));
    if fs::symlink_metadata(&temporary).is_ok() {
        fs::remove_file(&temporary)
            .map_err(|error| format!("removing stale {}: {error}", temporary.display()))?;
    }
    symlink(Path::new("../../blobs").join(required.blob), &temporary)
        .map_err(|error| format!("creating {}: {error}", temporary.display()))?;
    fs::rename(&temporary, &link).map_err(|error| format!("installing {}: {error}", link.display()))
}

fn create_parent(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| format!("creating {}: {error}", parent.display()))
}

fn resolve_url(model: ProvisionedModel, filename: &str) -> String {
    format!(
        "{HUB_ENDPOINT}/{}/resolve/{}/{filename}",
        model.model_id(),
        model.revision(),
    )
}

fn repository_folder(model: ProvisionedModel) -> String {
    format!("models--{}", model.model_id().replace('/', "--"))
}

fn snapshot_path(cache: &Path, model: ProvisionedModel) -> PathBuf {
    cache
        .join(repository_folder(model))
        .join("snapshots")
        .join(model.revision())
}

enum SnapshotState {
    Complete,
    Missing(Vec<&'static RequiredFile>),
}

fn inspect_snapshot(
    snapshot: &Path,
    required_files: &'static [RequiredFile],
) -> Result<SnapshotState, String> {
    let mut missing = Vec::new();
    for required in required_files {
        let path = snapshot.join(required.name);
        match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() == required.bytes => {}
            Ok(metadata) if metadata.is_file() => {
                return Err(format!(
                    "{} has {} bytes, expected {} for the pinned snapshot",
                    path.display(),
                    metadata.len(),
                    required.bytes,
                ));
            }
            Ok(_) => return Err(format!("{} is not a regular file", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing.push(required);
            }
            Err(error) => return Err(format!("reading {}: {error}", path.display())),
        }
    }
    if missing.is_empty() {
        Ok(SnapshotState::Complete)
    } else {
        Ok(SnapshotState::Missing(missing))
    }
}

fn hub_cache(environment: &impl Fn(&str) -> Option<OsString>) -> Result<PathBuf, String> {
    if let Some(cache) = nonempty_environment(environment, "HF_HUB_CACHE") {
        return Ok(cache.into());
    }
    if let Some(home) = nonempty_environment(environment, "HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    if let Some(cache) = nonempty_environment(environment, "XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache).join("huggingface/hub"));
    }
    if let Some(home) = nonempty_environment(environment, "HOME") {
        return Ok(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    Err("cannot locate the Hugging Face cache; set HF_HUB_CACHE or pass SNAPSHOT".into())
}

fn hf_token(environment: &impl Fn(&str) -> Option<OsString>) -> Result<Option<String>, String> {
    if let Some(token) = nonempty_environment(environment, "HF_TOKEN") {
        return Ok(Some(token.to_string_lossy().into_owned()));
    }
    let Some(path) = nonempty_environment(environment, "HF_TOKEN_PATH") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let token = fs::read_to_string(&path)
        .map_err(|error| format!("reading HF_TOKEN_PATH {}: {error}", path.display()))?;
    let token = token.trim();
    if token.is_empty() {
        Ok(None)
    } else {
        Ok(Some(token.to_owned()))
    }
}

fn offline(environment: &impl Fn(&str) -> Option<OsString>) -> bool {
    nonempty_environment(environment, "HF_HUB_OFFLINE").is_some_and(|value| {
        matches!(
            value.to_string_lossy().to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    })
}

fn nonempty_environment(
    environment: &impl Fn(&str) -> Option<OsString>,
    name: &str,
) -> Option<OsString> {
    environment(name).filter(|value| !value.is_empty())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        FileProgress, ProvisionedModel, ProvisioningStage, QWEN38_REQUIRED_FILES, RequiredFile,
        Sha256, SnapshotState, append_body, exact_file_length, hex_digest, hub_cache,
        inspect_snapshot, offline, require_pinned_etag, resume_offset, snapshot_path,
    };
    use crate::resolve_snapshot;
    use sha2::Digest;
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Cursor, Write};
    use std::path::{Path, PathBuf};

    fn environment(values: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let values = values
            .iter()
            .map(|&(name, value)| (name.to_owned(), OsString::from(value)))
            .collect::<BTreeMap<_, _>>();
        move |name| values.get(name).cloned()
    }

    #[test]
    fn hub_environment_precedence_and_pinned_identity_are_exact() {
        for (values, root) in [
            (
                vec![
                    ("HF_HUB_CACHE", "/cache/direct"),
                    ("HF_HOME", "/cache/home"),
                    ("XDG_CACHE_HOME", "/cache/xdg"),
                    ("HOME", "/home/user"),
                ],
                "/cache/direct",
            ),
            (vec![("HF_HOME", "/cache/home")], "/cache/home/hub"),
            (
                vec![("XDG_CACHE_HOME", "/cache/xdg")],
                "/cache/xdg/huggingface/hub",
            ),
            (
                vec![("HOME", "/home/user")],
                "/home/user/.cache/huggingface/hub",
            ),
        ] {
            let cache = hub_cache(&environment(&values)).unwrap();
            for model in ProvisionedModel::ALL {
                let actual = snapshot_path(&cache, model);
                let expected = PathBuf::from(root)
                    .join(format!(
                        "models--{}/snapshots",
                        model.model_id().replace('/', "--")
                    ))
                    .join(model.revision());
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn each_exact_allowlist_and_lengths_define_completeness() {
        for model in ProvisionedModel::ALL {
            let root = TestDirectory::new(model.model_id().replace('/', "-").as_str());
            for required in model.required_files() {
                let file = File::create(root.path().join(required.name)).unwrap();
                file.set_len(required.bytes).unwrap();
            }
            assert!(matches!(
                inspect_snapshot(root.path(), model.required_files()).unwrap(),
                SnapshotState::Complete
            ));

            let missing_name = model.required_files().last().unwrap().name;
            fs::remove_file(root.path().join(missing_name)).unwrap();
            let SnapshotState::Missing(missing) =
                inspect_snapshot(root.path(), model.required_files()).unwrap()
            else {
                panic!("expected one missing file");
            };
            assert_eq!(missing.len(), 1);
            assert_eq!(missing[0].name, missing_name);
        }
    }

    #[test]
    fn wrong_length_is_not_repaired_silently() {
        let root = TestDirectory::new("wrong-length");
        let file = File::create(root.path().join("config.json")).unwrap();
        file.set_len(1).unwrap();
        let error = inspect_snapshot(root.path(), ProvisionedModel::Qwen38.required_files())
            .err()
            .unwrap();
        assert!(error.contains("config.json has 1 bytes, expected 22564"));
    }

    #[test]
    fn standard_hugging_face_offline_values_are_recognized() {
        for value in ["1", "on", "YES", "True"] {
            assert!(offline(&environment(&[("HF_HUB_OFFLINE", value)])));
        }
        assert!(!offline(&environment(&[("HF_HUB_OFFLINE", "0")])));
    }

    #[test]
    fn digest_format_is_lowercase_hex() {
        assert_eq!(hex_digest(&[0, 1, 254, 255]), "0001feff");
    }

    #[test]
    fn progress_is_exact_and_refuses_file_overrun() {
        let required = &QWEN38_REQUIRED_FILES[2];
        let mut events = Vec::new();
        let mut report = |progress| {
            events.push(progress);
            Ok(())
        };
        {
            let mut progress = FileProgress {
                required,
                completed_before: 2 << 30,
                total_bytes: 24 << 30,
                report: &mut report,
            };
            progress
                .emit(ProvisioningStage::Downloading, 1 << 30)
                .unwrap();
        }
        let progress = events[0];
        assert_eq!(progress.stage(), ProvisioningStage::Downloading);
        assert_eq!(progress.file(), "model.safetensors");
        assert_eq!(progress.file_bytes(), 1 << 30);
        assert_eq!(progress.file_total(), required.bytes);
        assert_eq!(progress.completed_bytes(), 3 << 30);
        assert_eq!(progress.total_bytes(), 24 << 30);

        let mut discard = |_| Ok(());
        let mut progress = FileProgress {
            required,
            completed_before: 0,
            total_bytes: required.bytes,
            report: &mut discard,
        };
        assert!(
            progress
                .emit(ProvisioningStage::Downloading, required.bytes + 1)
                .is_err()
        );
    }

    #[test]
    fn explicit_snapshot_paths_are_verified() {
        for model in ProvisionedModel::ALL {
            let root = TestDirectory::new(
                format!("explicit-{}", model.model_id())
                    .replace('/', "-")
                    .as_str(),
            );
            for required in model.required_files() {
                let file = File::create(root.path().join(required.name)).unwrap();
                file.set_len(required.bytes).unwrap();
            }
            let resolution = resolve_snapshot(model, Some(root.path().to_path_buf())).unwrap();
            assert_eq!(resolution.path, root.path());
            assert!(resolution.provisioning.is_none());

            let first = model.required_files().first().unwrap();
            File::create(root.path().join(first.name))
                .unwrap()
                .set_len(1)
                .unwrap();
            let error = resolve_snapshot(model, Some(root.path().to_path_buf()))
                .err()
                .unwrap();
            assert!(error.contains(first.name));

            fs::remove_file(root.path().join(first.name)).unwrap();
            let error = resolve_snapshot(model, Some(root.path().to_path_buf()))
                .err()
                .unwrap();
            assert!(error.contains("missing 1 required file(s)"));
        }
    }

    #[test]
    fn oversized_partial_download_is_truncated_for_retry() {
        let root = TestDirectory::new("oversized");
        let path = root.path().join("blob.incomplete");
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(&[7_u8; 10]).unwrap();
        assert_eq!(resume_offset(&file, 4, &path).unwrap(), 0);
        assert_eq!(file.metadata().unwrap().len(), 0);

        file.write_all(&[7_u8; 3]).unwrap();
        assert_eq!(resume_offset(&file, 4, &path).unwrap(), 3);
        assert_eq!(file.metadata().unwrap().len(), 3);
    }

    #[test]
    fn overlong_body_errors_without_wedging_the_partial_file() {
        let required = RequiredFile::new("small.bin", 4, "blob", "sha");
        let root = TestDirectory::new("overlong");
        let path = root.path().join("small.bin.incomplete");
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
            .unwrap();
        let mut discard = |_| Ok(());
        let mut progress = FileProgress {
            required: &required,
            completed_before: 0,
            total_bytes: required.bytes,
            report: &mut discard,
        };
        let mut hasher = Sha256::new();
        let error = append_body(
            &mut Cursor::new([7_u8; 10]),
            &mut file,
            0,
            &mut hasher,
            &mut progress,
        )
        .err()
        .unwrap();
        assert!(error.contains("small.bin received more than the expected 4 bytes"));
        assert_eq!(file.metadata().unwrap().len(), 0);

        let mut hasher = Sha256::new();
        append_body(
            &mut Cursor::new([7_u8; 4]),
            &mut file,
            0,
            &mut hasher,
            &mut progress,
        )
        .unwrap();
        assert_eq!(file.metadata().unwrap().len(), 4);
    }

    #[test]
    fn plain_etag_backs_up_missing_linked_etag() {
        let required = &QWEN38_REQUIRED_FILES[0];
        let quoted = format!("\"{}\"", required.blob);
        let weak = format!("W/\"{}\"", required.blob);
        require_pinned_etag(Some(&quoted), None, required).unwrap();
        require_pinned_etag(None, Some(&quoted), required).unwrap();
        require_pinned_etag(None, Some(&weak), required).unwrap();
        require_pinned_etag(Some(&quoted), Some("\"junk\""), required).unwrap();

        let error = require_pinned_etag(None, None, required).err().unwrap();
        assert!(error.contains(&format!(
            "metadata for config.json has etag None, expected {}",
            required.blob,
        )));
        assert!(
            require_pinned_etag(Some("\"junk\""), Some(&quoted), required)
                .err()
                .unwrap()
                .contains("junk")
        );
    }

    #[test]
    fn wrong_sized_blob_error_names_the_path_to_delete() {
        let root = TestDirectory::new("blob-size");
        let path = root.path().join("blob");
        File::create(&path).unwrap().set_len(1).unwrap();
        let error = exact_file_length(&path, 4).err().unwrap();
        assert!(error.contains(&format!("delete {} to re-download", path.display())));
    }

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "tuiskollm-hf-{label}-{}-{}",
                std::process::id(),
                std::thread::current().name().unwrap_or("test"),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
