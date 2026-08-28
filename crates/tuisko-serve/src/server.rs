//! Concrete HTTP server and resident scheduler worker.

use crate::request_log::RequestLog;
use crate::response::overloaded_response;
use crate::text_generator::{GenerationEvent, GenerationEvents, TextGenerator};
use crate::{
    ChatCompletionRequest, ChatRequestError, GenerationReply, PreparedChatRequest,
    blocking_response, openai_error, streaming_response,
};
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::{IsTerminal, Write as IoWrite};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{Receiver, Sender, channel, error::TryRecvError, error::TrySendError};
use tokio::sync::oneshot;
use tuisko_engine::{
    EngineError, EngineErrorCode, GenerationStep, MAX_BATCH, Qwen35ResidentMtpBatchGenerator,
    Qwen36ResidentBatchGenerator, Qwen38FlashNextResidentBatchGenerator, ResidentBatchAdmission,
    ResidentLoadPhase, ResidentLoadProgress, ResidentMtpBatchGenerator, ResidentRequestId,
};
use tuisko_frontend::GenerationDefaults;
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext,
};

const REQUEST_BODY_LIMIT: usize = 2 * 1024 * 1024;
const DEFAULT_SEED_SCRAMBLE: u64 = 0x9e37_79b9_7f4a_7c15;
// Bounded so the single-threaded worker never blocks on a stalled client; a full
// lane is treated exactly like a disconnected one.
const GENERATION_REPLY_BUFFER: usize = 32;
const MTP_GENERATION_ROUTE: &str = "mtp-draft-3";
const QWEN35_MTP_GENERATION_ROUTE: &str = "mtp-b1-compact-b2-8";
const COMPACT_GENERATION_ROUTE: &str = "compact-b1-8";
const QWEN38_FLASH_NEXT_SERVED_REASONING_EFFORT: &str = "medium";

/// Startup configuration for the one exact resident server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConfig {
    /// Exact resident model selected for this process.
    pub model: ServerModel,
    /// Admitted pinned snapshot directory.
    pub snapshot: PathBuf,
    /// TCP address on which the OpenAI routes listen.
    pub address: SocketAddr,
}

/// Startup or listener failure from the concrete server.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Thread, runtime, listener, or HTTP serving failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Resident checkpoint or GPU owner failed before the listener opened.
    #[error("resident engine startup failed: {0}")]
    Startup(String),
    /// Resident worker exited without completing its startup handshake.
    #[error("resident engine worker disconnected during startup")]
    StartupDisconnected,
    /// Resident worker loop stopped after startup; serving would leave a zombie listener.
    #[error("resident engine worker failed: {0}")]
    WorkerFailed(String),
}

#[derive(Clone)]
struct AppState {
    jobs: Sender<Job>,
    request_ids: Arc<AtomicU64>,
    worker_ready: Arc<AtomicBool>,
    server_started: Instant,
    model_id: &'static str,
    generation_route: &'static str,
    generation_defaults: GenerationDefaults,
    reasoning_effort: Option<&'static str>,
}

struct Job {
    request: tuisko_engine::ChatGenerationRequest,
    reply: Sender<GenerationReply>,
    log: RequestLog,
}

struct ActiveReply {
    sender: Option<Sender<GenerationReply>>,
    cached_prompt_tokens: usize,
    log: RequestLog,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EnqueueError {
    Full,
    Closed,
}

struct Ready {
    model_id: &'static str,
    generation_route: &'static str,
    generation_defaults: GenerationDefaults,
    reasoning_effort: Option<&'static str>,
    device_name: String,
    checkpoint_admission: Duration,
    weight_load: Duration,
    source_prefault: Duration,
    graph_capture: Duration,
    tensor_count: usize,
    upload_bytes: usize,
    prefault_bytes: usize,
    arena_bytes: usize,
    host_stager_bytes: usize,
    slot_capacity: usize,
    context_capacity: usize,
    detailed_load_timing: bool,
}

/// Exact model loaders compiled into the server.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerModel {
    /// `unsloth/Qwen3.8-27B-NVFP4`.
    Qwen38,
    /// `AxionML/Qwen3.5-9B-NVFP4`.
    Qwen35,
    /// `nvidia/Qwen3.6-35B-A3B-NVFP4`.
    Qwen36,
    /// `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
    Qwen38FlashNext,
}

impl ServerModel {
    /// Every exact model currently served by this executable.
    pub const ALL: [Self; 4] = [
        Self::Qwen38,
        Self::Qwen35,
        Self::Qwen36,
        Self::Qwen38FlashNext,
    ];

    /// Resolves one exact Hugging Face model ID without aliases or discovery.
    pub fn from_model_id(model_id: &str) -> Result<Self, String> {
        match model_id {
            Qwen38_27B::MODEL_ID => Ok(Self::Qwen38),
            Qwen35_9B::MODEL_ID => Ok(Self::Qwen35),
            Qwen36Moe35B::MODEL_ID => Ok(Self::Qwen36),
            Qwen38FlashNext::MODEL_ID => Ok(Self::Qwen38FlashNext),
            _ => Err(format!(
                "unsupported model `{model_id}`; expected one of: {}",
                Self::ALL.map(Self::model_id).join(", ")
            )),
        }
    }

    /// Exact Hugging Face repository admitted by this loader.
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Qwen38 => Qwen38_27B::MODEL_ID,
            Self::Qwen35 => Qwen35_9B::MODEL_ID,
            Self::Qwen36 => Qwen36Moe35B::MODEL_ID,
            Self::Qwen38FlashNext => Qwen38FlashNext::MODEL_ID,
        }
    }

    const fn generation_route(self) -> &'static str {
        match self {
            Self::Qwen38 => MTP_GENERATION_ROUTE,
            Self::Qwen35 => QWEN35_MTP_GENERATION_ROUTE,
            Self::Qwen36 | Self::Qwen38FlashNext => COMPACT_GENERATION_ROUTE,
        }
    }

    const fn reasoning_effort(self) -> Option<&'static str> {
        match self {
            Self::Qwen38 | Self::Qwen35 | Self::Qwen36 => None,
            Self::Qwen38FlashNext => Some(QWEN38_FLASH_NEXT_SERVED_REASONING_EFFORT),
        }
    }
}

impl std::str::FromStr for ServerModel {
    type Err = String;

    fn from_str(model_id: &str) -> Result<Self, Self::Err> {
        Self::from_model_id(model_id)
    }
}

/// Loads the exact resident model, then serves health, model, blocking, and SSE routes.
pub fn run(config: ServerConfig) -> Result<(), ServerError> {
    let startup_start = Instant::now();
    let target = config.model;
    let stdout = std::io::stdout();
    let interactive = stdout.is_terminal();
    let color = interactive && std::env::var_os("NO_COLOR").is_none();
    let (state, ready, worker_failure) = {
        let mut stdout = stdout.lock();
        stdout.write_all(render_loading(target.model_id(), color, interactive).as_bytes())?;
        stdout.flush()?;
        start_worker(
            &config.snapshot,
            target,
            &mut stdout,
            interactive,
            color,
            startup_start,
        )?
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(config.address).await?;
        let output = render_startup(&ready, startup_start.elapsed(), config.address, color);
        let mut stdout = stdout.lock();
        stdout.write_all(output.as_bytes())?;
        stdout.flush()?;
        serve_until_worker_failure(listener, router(state), worker_failure).await
    })
}

async fn serve_until_worker_failure(
    listener: tokio::net::TcpListener,
    router: Router,
    worker_failure: oneshot::Receiver<String>,
) -> Result<(), ServerError> {
    tokio::select! {
        result = axum::serve(listener, router) => result.map_err(ServerError::Io),
        failure = worker_failure => Err(ServerError::WorkerFailed(
            failure.unwrap_or_else(|_| "resident engine worker exited".into()),
        )),
    }
}

fn start_worker(
    snapshot: &Path,
    target: ServerModel,
    output: &mut impl IoWrite,
    interactive: bool,
    color: bool,
    server_started: Instant,
) -> Result<(AppState, Ready, oneshot::Receiver<String>), ServerError> {
    let (jobs_tx, jobs_rx) = channel(MAX_BATCH);
    let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
    let (failure_tx, failure_rx) = oneshot::channel();
    let worker_ready = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(ResidentLoadProgress::new());
    let snapshot = snapshot.to_owned();
    let worker_ready_clone = Arc::clone(&worker_ready);
    let worker_progress = Arc::clone(&progress);
    std::thread::Builder::new()
        .name("tuiskollm-engine".into())
        .spawn(move || {
            engine_worker(
                snapshot,
                target,
                jobs_rx,
                ready_tx,
                failure_tx,
                worker_ready_clone,
                worker_progress,
            )
        })?;
    let mut displayed = None;
    let mut spinner_tick = 1;
    let ready = loop {
        match ready_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(ready) => break ready,
            Err(std_mpsc::RecvTimeoutError::Timeout) if interactive => {
                let snapshot = progress.snapshot();
                if (snapshot.0 == ResidentLoadPhase::Preparing || Some(snapshot) != displayed)
                    && let Some(line) = render_load_progress(
                        snapshot.0,
                        snapshot.1,
                        snapshot.2,
                        spinner_tick,
                        color,
                    )
                {
                    output.write_all(line.as_bytes())?;
                    output.flush()?;
                    displayed = Some(snapshot);
                    spinner_tick = spinner_tick.wrapping_add(1);
                }
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                clear_progress_line(output, interactive)?;
                return Err(ServerError::StartupDisconnected);
            }
        }
    };
    clear_progress_line(output, interactive)?;
    let ready = ready.map_err(ServerError::Startup)?;
    Ok((
        AppState {
            jobs: jobs_tx,
            request_ids: Arc::new(AtomicU64::new(1)),
            worker_ready,
            server_started,
            model_id: ready.model_id,
            generation_route: ready.generation_route,
            generation_defaults: ready.generation_defaults,
            reasoning_effort: ready.reasoning_effort,
        },
        ready,
        failure_rx,
    ))
}
fn engine_worker(
    snapshot: PathBuf,
    target: ServerModel,
    jobs: Receiver<Job>,
    ready: std_mpsc::SyncSender<Result<Ready, String>>,
    failure: oneshot::Sender<String>,
    worker_ready: Arc<AtomicBool>,
    progress: Arc<ResidentLoadProgress>,
) {
    struct ReadinessGuard(Arc<AtomicBool>);
    impl Drop for ReadinessGuard {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _readiness_guard = ReadinessGuard(Arc::clone(&worker_ready));
    match target {
        ServerModel::Qwen38 => start_generator(
            load_qwen38(&snapshot, progress.as_ref()),
            jobs,
            ready,
            failure,
            &worker_ready,
        ),
        ServerModel::Qwen35 => {
            start_generator(load_qwen35(&snapshot), jobs, ready, failure, &worker_ready)
        }
        ServerModel::Qwen36 => {
            start_generator(load_qwen36(&snapshot), jobs, ready, failure, &worker_ready)
        }
        ServerModel::Qwen38FlashNext => start_generator(
            load_qwen38_flash_next(&snapshot, progress.as_ref()),
            jobs,
            ready,
            failure,
            &worker_ready,
        ),
    }
}

fn load_qwen38(
    snapshot: &Path,
    progress: &ResidentLoadProgress,
) -> Result<(ResidentMtpBatchGenerator, Ready), String> {
    let checkpoint_start = Instant::now();
    let admitted = CheckpointSnapshot::<Qwen38_27B>::open(snapshot)
        .map(Arc::new)
        .map_err(|error| format!("admitting {}: {error}", snapshot.display()))?;
    let checkpoint_admission = checkpoint_start.elapsed();
    let tensor_count = admitted.tensor_count();
    let generator =
        ResidentMtpBatchGenerator::from_snapshot_device_zero_with_progress(admitted, progress)
            .map_err(|error| format!("loading the resident text program: {error}"))?;
    let device_name = generator
        .context()
        .device_name()
        .map_err(|error| format!("reading the CUDA device name: {error}"))?;
    let load_stats = generator.load_stats();

    let startup = Ready {
        model_id: Qwen38_27B::MODEL_ID,
        generation_route: ServerModel::Qwen38.generation_route(),
        generation_defaults: generator.generation_defaults(),
        reasoning_effort: ServerModel::Qwen38.reasoning_effort(),
        device_name,
        checkpoint_admission,
        weight_load: Duration::from_nanos(load_stats.weight_load_ns()),
        source_prefault: Duration::from_nanos(load_stats.source_prefault_ns()),
        graph_capture: Duration::from_nanos(load_stats.graph_capture_ns()),
        tensor_count,
        upload_bytes: load_stats.upload_bytes(),
        prefault_bytes: load_stats.prefault_bytes(),
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        slot_capacity: MAX_BATCH,
        context_capacity: generator.context_capacity(),
        detailed_load_timing: true,
    };

    Ok((generator, startup))
}

fn load_qwen35(snapshot: &Path) -> Result<(Qwen35ResidentMtpBatchGenerator, Ready), String> {
    let checkpoint_start = Instant::now();
    let admitted = CheckpointSnapshot::<Qwen35_9B>::open(snapshot)
        .map(Arc::new)
        .map_err(|error| format!("admitting {}: {error}", snapshot.display()))?;
    let checkpoint_admission = checkpoint_start.elapsed();
    let tensor_count = admitted.tensor_count();
    let load_start = Instant::now();
    let generator = Qwen35ResidentMtpBatchGenerator::from_snapshot_device_zero(admitted)
        .map_err(|error| format!("loading the resident Qwen3.5 MTP program: {error}"))?;
    let resident_load = load_start.elapsed();
    let device_name = generator
        .context()
        .device_name()
        .map_err(|error| format!("reading the CUDA device name: {error}"))?;

    let startup = Ready {
        model_id: Qwen35_9B::MODEL_ID,
        generation_route: ServerModel::Qwen35.generation_route(),
        generation_defaults: generator.generation_defaults(),
        reasoning_effort: ServerModel::Qwen35.reasoning_effort(),
        device_name,
        checkpoint_admission,
        weight_load: resident_load,
        source_prefault: Duration::ZERO,
        graph_capture: Duration::ZERO,
        tensor_count,
        upload_bytes: generator.resident_weight_bytes(),
        prefault_bytes: 0,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        slot_capacity: MAX_BATCH,
        context_capacity: generator.context_capacity(),
        detailed_load_timing: false,
    };

    Ok((generator, startup))
}

fn load_qwen36(snapshot: &Path) -> Result<(Qwen36ResidentBatchGenerator, Ready), String> {
    let checkpoint_start = Instant::now();
    let admitted = CheckpointSnapshot::<Qwen36Moe35B>::open(snapshot)
        .map(Arc::new)
        .map_err(|error| format!("admitting {}: {error}", snapshot.display()))?;
    let checkpoint_admission = checkpoint_start.elapsed();
    let tensor_count = admitted.tensor_count();
    let load_start = Instant::now();
    let generator = Qwen36ResidentBatchGenerator::from_snapshot_device_zero(admitted)
        .map_err(|error| format!("loading the resident Qwen3.6 compact program: {error}"))?;
    let resident_load = load_start.elapsed();
    let device_name = generator
        .context()
        .device_name()
        .map_err(|error| format!("reading the CUDA device name: {error}"))?;

    let startup = Ready {
        model_id: Qwen36Moe35B::MODEL_ID,
        generation_route: ServerModel::Qwen36.generation_route(),
        generation_defaults: generator.generation_defaults(),
        reasoning_effort: ServerModel::Qwen36.reasoning_effort(),
        device_name,
        checkpoint_admission,
        weight_load: resident_load,
        source_prefault: Duration::ZERO,
        graph_capture: Duration::ZERO,
        tensor_count,
        upload_bytes: generator.resident_weight_bytes(),
        prefault_bytes: 0,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        slot_capacity: MAX_BATCH,
        context_capacity: generator.context_capacity(),
        detailed_load_timing: false,
    };

    Ok((generator, startup))
}

fn load_qwen38_flash_next(
    snapshot: &Path,
    progress: &ResidentLoadProgress,
) -> Result<(Qwen38FlashNextResidentBatchGenerator, Ready), String> {
    let checkpoint_start = Instant::now();
    let admitted = CheckpointSnapshot::<Qwen38FlashNext>::open(snapshot)
        .map(Arc::new)
        .map_err(|error| format!("admitting {}: {error}", snapshot.display()))?;
    let checkpoint_admission = checkpoint_start.elapsed();
    let tensor_count = admitted.tensor_count();
    let generator = Qwen38FlashNextResidentBatchGenerator::from_snapshot_device_zero_with_progress(
        admitted, progress,
    )
    .map_err(|error| format!("loading the resident Qwen3.8 Flash-Next program: {error}"))?;
    let device_name = generator
        .context()
        .device_name()
        .map_err(|error| format!("reading the CUDA device name: {error}"))?;
    let arena_bytes = generator
        .arena_bytes()
        .map_err(|error| format!("accounting the Qwen3.8 Flash-Next device arenas: {error}"))?;
    if !generator.mapped_primary() {
        return Err("Qwen3.8 Flash-Next serving requires mapped primary expert weights".into());
    }
    let load_stats = generator.load_stats();

    let startup = Ready {
        model_id: Qwen38FlashNext::MODEL_ID,
        generation_route: ServerModel::Qwen38FlashNext.generation_route(),
        generation_defaults: generator.generation_defaults(),
        reasoning_effort: ServerModel::Qwen38FlashNext.reasoning_effort(),
        device_name,
        checkpoint_admission,
        weight_load: load_stats
            .weight_upload()
            .saturating_sub(load_stats.expert_stage()),
        source_prefault: load_stats.expert_stage(),
        graph_capture: load_stats.graph_capture(),
        tensor_count,
        upload_bytes: generator.resident_weight_bytes(),
        prefault_bytes: load_stats.staged_bytes(),
        arena_bytes,
        host_stager_bytes: generator.host_stager_bytes(),
        slot_capacity: generator.slot_capacity(),
        context_capacity: generator.context_capacity(),
        detailed_load_timing: true,
    };

    Ok((generator, startup))
}

/// Publishes readiness for one loaded target, then serves its requests forever.
fn start_generator<G: TextGenerator>(
    loaded: Result<(G, Ready), String>,
    jobs: Receiver<Job>,
    ready: std_mpsc::SyncSender<Result<Ready, String>>,
    failure: oneshot::Sender<String>,
    worker_ready: &AtomicBool,
) {
    let (generator, startup) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    worker_ready.store(true, Ordering::Release);
    if ready.send(Ok(startup)).is_err() {
        return;
    }

    serve_requests(generator, jobs, failure);
}

/// One resident scheduler round loop, identical for every admitted target.
fn serve_requests<G: TextGenerator>(
    mut generator: G,
    mut jobs: Receiver<Job>,
    failure: oneshot::Sender<String>,
) {
    let mut replies = HashMap::new();
    let mut jobs_open = true;
    let mut waiting = Vec::new();
    loop {
        if let Err(error) = cancel_disconnected(&mut generator, &mut replies) {
            fail_all(&mut replies, error.clone());
            jobs.close();
            fail_queued(&mut jobs, &error);
            let _ = failure.send(error);
            break;
        }

        if generator.active_requests() == 0 && jobs_open {
            match jobs.blocking_recv() {
                Some(job) => hold_job(&mut waiting, job),
                None => jobs_open = false,
            }
        }
        while jobs_open && generator.active_requests() + waiting.len() < generator.slot_capacity() {
            match jobs.try_recv() {
                Ok(job) => hold_job(&mut waiting, job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    jobs_open = false;
                    break;
                }
            }
        }
        if !waiting.is_empty() {
            admit_group(&mut generator, &mut replies, &mut waiting);
        }
        if generator.active_requests() == 0 {
            if jobs_open {
                continue;
            }
            break;
        }

        let events = match generator.step() {
            Ok(events) => events,
            Err(error) => {
                let error = error.to_string();
                fail_all(&mut replies, error.clone());
                jobs.close();
                fail_queued(&mut jobs, &error);
                let _ = failure.send(error);
                break;
            }
        };
        for event in events.iter() {
            let request_id = event.request_id();
            let failed = replies.get_mut(&request_id).is_some_and(|reply| {
                if event.steps().any(|step| step.delta.is_some()) {
                    reply.log.observe_output();
                }
                reply
                    .sender
                    .as_ref()
                    .is_none_or(|sender| !try_send_generation_steps(sender, event.steps()))
            });
            if failed && let Some(reply) = replies.get_mut(&request_id) {
                // The active log remains owned until cancellation or completion.
                reply.sender = None;
            }
            if let Some(output) = event.completed()
                && let Some(mut reply) = replies.remove(&request_id)
            {
                if let Some(sender) = reply.sender.take() {
                    let _ = sender.try_send(GenerationReply::Done {
                        output: output.clone(),
                        cached_prompt_tokens: reply.cached_prompt_tokens,
                    });
                }
                reply.log.finish(
                    Some(&output.prompt),
                    output.token_ids.len(),
                    reply.cached_prompt_tokens,
                    output.finish_reason.as_str(),
                    None,
                );
            }
        }
    }
}

fn render_loading(model_id: &str, color: bool, interactive: bool) -> String {
    let (header, loading, reset) = if color {
        ("\x1b[1;36m", "\x1b[1;33m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let newline = if interactive { "" } else { "\n" };
    format!(
        "{header}TuiskoLLM{reset} · {model_id}\n{loading}LOADING{reset} ⠋       preparing resident model…{newline}"
    )
}

fn render_load_progress(
    phase: ResidentLoadPhase,
    submitted_bytes: usize,
    total_bytes: usize,
    spinner_tick: usize,
    color: bool,
) -> Option<String> {
    let finalizing = match phase {
        ResidentLoadPhase::Preparing => {
            const FRAMES: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
            let frame = FRAMES[spinner_tick % FRAMES.len()];
            let (loading, reset) = if color {
                ("\x1b[1;33m", "\x1b[0m")
            } else {
                ("", "")
            };
            return Some(format!(
                "\r\x1b[2K{loading}LOADING{reset} {frame}       preparing resident model…"
            ));
        }
        ResidentLoadPhase::Ready => return None,
        ResidentLoadPhase::Uploading => false,
        ResidentLoadPhase::Finalizing => true,
    };
    Some(render_weight_progress(
        submitted_bytes,
        total_bytes,
        finalizing,
        color,
    ))
}

fn render_weight_progress(
    submitted_bytes: usize,
    total_bytes: usize,
    finalizing: bool,
    color: bool,
) -> String {
    const BAR_WIDTH: usize = 20;
    let submitted_bytes = submitted_bytes.min(total_bytes);
    let filled = submitted_bytes
        .saturating_mul(BAR_WIDTH)
        .checked_div(total_bytes)
        .unwrap_or(0);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled));
    let (loading, reset) = if color {
        ("\x1b[1;33m", "\x1b[0m")
    } else {
        ("", "")
    };
    let suffix = if finalizing { " · finalizing…" } else { "" };
    format!(
        "\r\x1b[2K{loading}LOADING{reset} weights  {bar}  {:.2} / {:.2} GiB{suffix}",
        gibibytes(submitted_bytes),
        gibibytes(total_bytes),
    )
}

fn clear_progress_line(output: &mut impl IoWrite, interactive: bool) -> std::io::Result<()> {
    if interactive {
        output.write_all(b"\r\x1b[2K")?;
        output.flush()?;
    }
    Ok(())
}

fn render_startup(ready: &Ready, total: Duration, address: SocketAddr, color: bool) -> String {
    let mut output = String::new();
    let (ok, ready_label, reset) = if color {
        ("\x1b[32m", "\x1b[1;32m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    writeln!(
        output,
        "{ok}OK{reset} device                 · {}",
        ready.device_name
    )
    .expect("writing to a String cannot fail");
    writeln!(
        output,
        "{ok}OK{reset} checkpoint  {:>8.1} ms · {} tensors",
        milliseconds(ready.checkpoint_admission),
        ready.tensor_count,
    )
    .expect("writing to a String cannot fail");
    if ready.detailed_load_timing {
        writeln!(
            output,
            "{ok}OK{reset} source pages {:>7.1} ms · {:.2} GiB",
            milliseconds(ready.source_prefault),
            gibibytes(ready.prefault_bytes),
        )
        .expect("writing to a String cannot fail");
        writeln!(
            output,
            "{ok}OK{reset} weights      {:>7.1} ms · {:.2} GiB",
            milliseconds(ready.weight_load),
            gibibytes(ready.upload_bytes),
        )
        .expect("writing to a String cannot fail");
        writeln!(
            output,
            "{ok}OK{reset} graphs       {:>7.1} ms",
            milliseconds(ready.graph_capture),
        )
        .expect("writing to a String cannot fail");
    } else {
        writeln!(
            output,
            "{ok}OK{reset} resident    {:>7.1} ms · {:.2} GiB weights and graphs",
            milliseconds(ready.weight_load),
            gibibytes(ready.upload_bytes),
        )
        .expect("writing to a String cannot fail");
    }
    writeln!(
        output,
        "{ready_label}READY{reset}          {:>7.1} ms · http://{address} · {} · {} slots · context {} · {:.2} GiB device · {:.2} MiB pinned",
        milliseconds(total),
        ready.generation_route,
        ready.slot_capacity,
        ready.context_capacity,
        gibibytes(ready.arena_bytes),
        mebibytes(ready.host_stager_bytes),
    )
    .expect("writing to a String cannot fail");
    output
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn gibibytes(bytes: usize) -> f64 {
    bytes as f64 / (1_u64 << 30) as f64
}

fn mebibytes(bytes: usize) -> f64 {
    bytes as f64 / (1_u64 << 20) as f64
}

fn hold_job(waiting: &mut Vec<Job>, job: Job) {
    if job.reply.is_closed() {
        job.log.finish(None, 0, 0, "cancelled", None);
        return;
    }
    waiting.push(job);
}

fn admit_group<G: TextGenerator>(
    generator: &mut G,
    replies: &mut HashMap<ResidentRequestId, ActiveReply>,
    waiting: &mut Vec<Job>,
) {
    let requests = waiting.iter().map(|job| &job.request).collect::<Vec<_>>();
    let admissions = generator.admit_batch(&requests);
    debug_assert_eq!(admissions.len(), waiting.len());
    for (job, admission) in waiting.drain(..).zip(admissions) {
        record_admission(replies, job, admission);
    }
}

fn record_admission(
    replies: &mut HashMap<ResidentRequestId, ActiveReply>,
    job: Job,
    admission: Result<ResidentBatchAdmission, EngineError>,
) {
    let Job { reply, mut log, .. } = job;
    match admission {
        Ok(admission) => {
            log.observe_prompt(admission.prompt_metrics);
            if let Some(output) = admission.completed {
                let _ = reply.try_send(GenerationReply::Done {
                    output: output.clone(),
                    cached_prompt_tokens: admission.device_reused_tokens,
                });
                log.finish(
                    Some(&output.prompt),
                    output.token_ids.len(),
                    admission.device_reused_tokens,
                    output.finish_reason.as_str(),
                    None,
                );
            } else {
                let previous = replies.insert(
                    admission.request_id,
                    ActiveReply {
                        sender: Some(reply),
                        cached_prompt_tokens: admission.device_reused_tokens,
                        log,
                    },
                );
                debug_assert!(previous.is_none(), "resident request identities are unique");
            }
        }
        Err(error) => {
            let message = error.to_string();
            let response = if error.code() == Some(EngineErrorCode::Capacity) {
                GenerationReply::Overloaded(message.clone())
            } else {
                GenerationReply::Rejected(message.clone())
            };
            let _ = reply.try_send(response);
            log.finish(None, 0, 0, "error", Some(&message));
        }
    }
}

fn cancel_disconnected<G: TextGenerator>(
    generator: &mut G,
    replies: &mut HashMap<ResidentRequestId, ActiveReply>,
) -> Result<(), String> {
    let cancelled = generator
        .active_request_ids()
        .filter(|request| {
            replies
                .get(request)
                .is_none_or(|reply| reply.sender.as_ref().is_none_or(Sender::is_closed))
        })
        .collect::<Vec<_>>();
    for request in cancelled {
        let reply = replies.remove(&request);
        match generator.cancel(request) {
            Ok(cancelled) => {
                if let Some(reply) = reply {
                    reply.log.finish(
                        Some(&cancelled.output.prompt),
                        cancelled.output.token_ids.len(),
                        reply.cached_prompt_tokens,
                        "cancelled",
                        None,
                    );
                }
            }
            Err(error) => {
                let message = error.to_string();
                if let Some(reply) = reply {
                    reply.log.finish(None, 0, 0, "error", Some(&message));
                }
                return Err(message);
            }
        }
    }
    Ok(())
}

fn try_send_generation_steps<'a>(
    reply: &Sender<GenerationReply>,
    steps: impl IntoIterator<Item = &'a GenerationStep>,
) -> bool {
    for step in steps {
        if let Some(delta) = &step.delta
            && reply
                .try_send(GenerationReply::Delta(delta.clone()))
                .is_err()
        {
            return false;
        }
    }
    true
}

fn fail_all(replies: &mut HashMap<ResidentRequestId, ActiveReply>, message: String) {
    for (_, mut reply) in replies.drain() {
        if let Some(sender) = reply.sender.take() {
            let _ = sender.try_send(GenerationReply::Failed(message.clone()));
        }
        reply.log.finish(None, 0, 0, "error", Some(&message));
    }
}

fn fail_queued(jobs: &mut Receiver<Job>, message: &str) {
    while let Ok(job) = jobs.try_recv() {
        let _ = job
            .reply
            .try_send(GenerationReply::Failed(message.to_owned()));
        job.log.finish(None, 0, 0, "error", Some(message));
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Response {
    if state.worker_ready.load(Ordering::Acquire) {
        Json(json!({"status": "ok", "generation_route": state.generation_route})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable"})),
        )
            .into_response()
    }
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_id, "object": "model", "owned_by": "tuiskollm"}]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                error.body_text(),
                "invalid_request_error",
            );
        }
    };
    let numeric_id = state.request_ids.fetch_add(1, Ordering::Relaxed);
    let PreparedChatRequest {
        generation,
        stream,
        split_reasoning,
        parse_tools,
        include_usage,
    } = match request.prepare_for(
        numeric_id ^ DEFAULT_SEED_SCRAMBLE,
        state.model_id,
        state.generation_defaults,
        state.reasoning_effort,
    ) {
        Ok(request) => request,
        Err(ChatRequestError::ModelNotFound { requested }) => {
            return openai_error(
                StatusCode::NOT_FOUND,
                format!("model `{requested}` is not served by this process"),
                "model_not_found",
            );
        }
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                error.to_string(),
                "invalid_request_error",
            );
        }
    };
    let accepted = Instant::now();
    let (reply_tx, mut reply_rx) = channel(GENERATION_REPLY_BUFFER);
    if let Err(error) = enqueue_job(
        &state.jobs,
        Job {
            request: generation,
            reply: reply_tx,
            log: RequestLog::new(
                numeric_id,
                accepted,
                state.server_started,
                state.generation_route,
            ),
        },
    ) {
        return enqueue_error_response(error);
    }

    let id = format!("chatcmpl-tuisko-{numeric_id:016x}");
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if stream {
        match reply_rx.recv().await {
            Some(GenerationReply::Rejected(message)) => {
                openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error")
            }
            Some(GenerationReply::Overloaded(message)) => overloaded_response(message),
            Some(first) => streaming_response(
                first,
                reply_rx,
                id,
                created,
                state.model_id,
                split_reasoning,
                parse_tools,
                include_usage,
            ),
            None => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "resident engine worker disconnected".into(),
                "server_error",
            ),
        }
    } else {
        blocking_response(
            reply_rx,
            id,
            created,
            state.model_id,
            split_reasoning,
            parse_tools,
        )
        .await
    }
}

fn enqueue_job(jobs: &Sender<Job>, job: Job) -> Result<(), EnqueueError> {
    match jobs.try_send(job) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(job)) => {
            job.log.finish(
                None,
                0,
                0,
                "error",
                Some("resident inference queue is full"),
            );
            Err(EnqueueError::Full)
        }
        Err(TrySendError::Closed(job)) => {
            job.log.finish(
                None,
                0,
                0,
                "error",
                Some("resident engine worker is unavailable"),
            );
            Err(EnqueueError::Closed)
        }
    }
}

fn enqueue_error_response(error: EnqueueError) -> Response {
    match error {
        EnqueueError::Full => overloaded_response("resident inference queue is full".into()),
        EnqueueError::Closed => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "resident engine worker is unavailable".into(),
            "server_error",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AppState, EnqueueError, Job, QWEN38_FLASH_NEXT_SERVED_REASONING_EFFORT, Ready, ServerError,
        ServerModel, chat_completions, enqueue_job, fail_queued, health, models, record_admission,
        render_loading, render_startup, render_weight_progress, router, serve_until_worker_failure,
        try_send_generation_steps,
    };
    use crate::{ChatCompletionRequest, GenerationReply, SERVED_MODEL};
    use axum::Json;
    use axum::body::to_bytes;
    use axum::extract::State;
    use axum::http::{StatusCode, header::RETRY_AFTER};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::time::Duration;
    use tokio::runtime::Builder;
    use tokio::sync::mpsc::channel;
    use tuisko_engine::{
        ChatGenerationRequest, EngineError, EngineErrorCode, FinishReason, GenerationStep,
        MAX_BATCH,
    };
    use tuisko_frontend::{ChatMessage, GenerationDefaults};
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

    fn runtime() -> tokio::runtime::Runtime {
        Builder::new_current_thread().enable_all().build().unwrap()
    }

    fn job() -> (Job, tokio::sync::mpsc::Receiver<GenerationReply>) {
        let (reply, receiver) = channel(8);
        let started = std::time::Instant::now();
        (
            Job {
                request: ChatGenerationRequest::new(vec![ChatMessage::new("user", "hello")]),
                reply,
                log: crate::request_log::RequestLog::new(1, started, started, "mtp-draft-3"),
            },
            receiver,
        )
    }

    fn state(jobs: tokio::sync::mpsc::Sender<Job>, ready: bool) -> AppState {
        AppState {
            jobs,
            request_ids: Arc::new(AtomicU64::new(1)),
            worker_ready: Arc::new(AtomicBool::new(ready)),
            server_started: std::time::Instant::now(),
            model_id: SERVED_MODEL,
            generation_route: ServerModel::Qwen38.generation_route(),
            generation_defaults: GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            reasoning_effort: ServerModel::Qwen38.reasoning_effort(),
        }
    }

    #[test]
    fn startup_output_is_exact_plain_text_or_terminal_color() {
        let ready = Ready {
            model_id: SERVED_MODEL,
            generation_route: ServerModel::Qwen38.generation_route(),
            generation_defaults: GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            reasoning_effort: ServerModel::Qwen38.reasoning_effort(),
            device_name: "NVIDIA GeForce RTX 5090".into(),
            checkpoint_admission: Duration::from_micros(2_500),
            weight_load: Duration::from_micros(1_604_600),
            source_prefault: Duration::from_micros(53_000),
            graph_capture: Duration::from_micros(83_100),
            tensor_count: 1_968,
            upload_bytes: 19 * (1 << 30),
            prefault_bytes: 18 * (1 << 30),
            arena_bytes: 25 * (1 << 30),
            host_stager_bytes: 16 * (1 << 20),
            slot_capacity: MAX_BATCH,
            context_capacity: 220_000,
            detailed_load_timing: true,
        };
        let address = "127.0.0.1:8000".parse::<SocketAddr>().unwrap();
        let loading = render_loading(SERVED_MODEL, false, false);
        assert_eq!(
            loading,
            "TuiskoLLM · unsloth/Qwen3.8-27B-NVFP4\nLOADING ⠋       preparing resident model…\n"
        );
        assert!(!render_loading(SERVED_MODEL, false, true).ends_with('\n'));

        let plain = render_startup(&ready, Duration::from_micros(1_925_200), address, false);
        let lines = plain.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 6);
        assert_eq!(
            lines[0],
            "OK device                 · NVIDIA GeForce RTX 5090"
        );
        assert!(lines[1].contains("2.5 ms · 1968 tensors"));
        assert!(lines[2].contains("53.0 ms · 18.00 GiB"));
        assert!(lines[3].contains("1604.6 ms · 19.00 GiB"));
        assert!(lines[4].contains("83.1 ms"));
        assert!(lines[5].contains("1925.2 ms · http://127.0.0.1:8000 · mtp-draft-3"));
        assert!(
            lines[5].contains("8 slots · context 220000 · 25.00 GiB device · 16.00 MiB pinned")
        );
        assert!(!plain.contains('\x1b'));

        let colored_loading = render_loading(SERVED_MODEL, true, true);
        assert!(colored_loading.starts_with("\x1b[1;36mTuiskoLLM\x1b[0m"));
        assert!(colored_loading.contains("\x1b[1;33mLOADING\x1b[0m"));

        let progress = render_weight_progress(3 << 30, 4 << 30, false, false);
        assert!(progress.contains("███████████████░░░░░"));
        assert!(progress.contains("3.00 / 4.00 GiB"));
        let finalizing = render_weight_progress(4 << 30, 4 << 30, true, true);
        assert!(finalizing.contains("\x1b[1;33mLOADING\x1b[0m"));
        assert!(finalizing.contains("████████████████████"));
        assert!(finalizing.ends_with("4.00 / 4.00 GiB · finalizing…"));

        let colored = render_startup(&ready, Duration::from_micros(1_925_200), address, true);
        assert!(colored.starts_with("\x1b[32mOK\x1b[0m device"));
        assert!(colored.contains("\x1b[1;32mREADY\x1b[0m"));
    }

    #[test]
    fn qwen35_startup_reports_mtp_compact_route_and_slots() {
        let ready = Ready {
            model_id: Qwen35_9B::MODEL_ID,
            generation_route: ServerModel::Qwen35.generation_route(),
            generation_defaults: GenerationDefaults {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
            },
            reasoning_effort: ServerModel::Qwen35.reasoning_effort(),
            device_name: "NVIDIA GeForce RTX 5090".into(),
            checkpoint_admission: Duration::from_millis(11),
            weight_load: Duration::from_millis(725),
            source_prefault: Duration::ZERO,
            graph_capture: Duration::ZERO,
            tensor_count: 1_234,
            upload_bytes: 6 * (1 << 30),
            prefault_bytes: 0,
            arena_bytes: 7 * (1 << 30),
            host_stager_bytes: 1 << 20,
            slot_capacity: MAX_BATCH,
            context_capacity: 192,
            detailed_load_timing: false,
        };

        let output = render_startup(
            &ready,
            Duration::from_millis(900),
            "127.0.0.1:8000".parse().unwrap(),
            false,
        );
        let lines = output.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 4);
        assert!(lines[2].contains("725.0 ms · 6.00 GiB weights and graphs"));
        assert!(lines[3].contains("mtp-b1-compact-b2-8 · 8 slots · context 192"));
        assert!(!output.contains("source pages"));
    }

    #[test]
    fn bounded_ingress_distinguishes_overload_and_worker_shutdown() {
        let (jobs, receiver) = channel(1);
        enqueue_job(&jobs, job().0).unwrap();
        let full = enqueue_job(&jobs, job().0).unwrap_err();
        assert_eq!(full, EnqueueError::Full);

        drop(receiver);
        let closed = enqueue_job(&jobs, job().0).unwrap_err();
        assert_eq!(closed, EnqueueError::Closed);
    }

    #[test]
    fn resident_capacity_failure_is_retryable() {
        let (job, mut receiver) = job();
        let mut replies = HashMap::new();
        record_admission(
            &mut replies,
            job,
            Err(EngineError::Contract {
                code: EngineErrorCode::Capacity,
                message: "shared KV pages remain active".into(),
            }),
        );

        assert!(replies.is_empty());
        assert!(matches!(
            receiver.try_recv(),
            Ok(GenerationReply::Overloaded(message))
                if message.contains("shared KV pages remain active")
        ));
    }

    #[test]
    fn fatal_worker_failure_drains_queued_replies() {
        let (jobs, mut queued) = channel(2);
        let (first, mut first_reply) = job();
        let (second, mut second_reply) = job();
        jobs.try_send(first).unwrap();
        jobs.try_send(second).unwrap();

        fail_queued(&mut queued, "device launch failed");

        assert!(matches!(
            first_reply.try_recv(),
            Ok(GenerationReply::Failed(message)) if message == "device launch failed"
        ));
        assert!(matches!(
            second_reply.try_recv(),
            Ok(GenerationReply::Failed(message)) if message == "device launch failed"
        ));
        assert!(matches!(
            queued.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn overloaded_handler_paces_retries() {
        runtime().block_on(async {
            let (jobs, _receiver) = channel(1);
            enqueue_job(&jobs, job().0).unwrap();
            let response =
                chat_completions(State(state(jobs, true)), Ok(Json(streaming_request()))).await;

            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "1");
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let error: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"]["type"], "server_overloaded");
        });
    }

    #[test]
    fn worker_failure_stops_serving_with_the_underlying_error() {
        runtime().block_on(async {
            let (jobs, _receiver) = channel(1);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let (failure_tx, failure_rx) = tokio::sync::oneshot::channel();
            let serve = tokio::spawn(serve_until_worker_failure(
                listener,
                router(state(jobs, true)),
                failure_rx,
            ));
            failure_tx.send("device launch failed".into()).unwrap();
            let error = serve.await.unwrap().unwrap_err();
            assert_eq!(
                error.to_string(),
                "resident engine worker failed: device launch failed"
            );

            let (jobs, _receiver) = channel(1);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let (failure_tx, failure_rx) = tokio::sync::oneshot::channel::<String>();
            let serve = tokio::spawn(serve_until_worker_failure(
                listener,
                router(state(jobs, true)),
                failure_rx,
            ));
            drop(failure_tx);
            let error = serve.await.unwrap().unwrap_err();
            assert!(matches!(error, ServerError::WorkerFailed(_)));
            assert_eq!(
                error.to_string(),
                "resident engine worker failed: resident engine worker exited"
            );
        });
    }

    #[test]
    fn mtp_transaction_forwards_every_decoded_delta_in_order() {
        let (reply, mut receiver) = channel(4);
        let steps = [
            GenerationStep {
                token_id: 1,
                delta: Some("one".into()),
                finish_reason: None,
            },
            GenerationStep {
                token_id: 2,
                delta: None,
                finish_reason: None,
            },
            GenerationStep {
                token_id: 3,
                delta: Some(" two".into()),
                finish_reason: None,
            },
            GenerationStep {
                token_id: 4,
                delta: Some(" three".into()),
                finish_reason: Some(FinishReason::Length),
            },
        ];

        assert!(try_send_generation_steps(&reply, &steps));
        let deltas = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|reply| match reply {
                GenerationReply::Delta(delta) => delta,
                _ => panic!("MTP step adapter emitted a non-delta reply"),
            })
            .collect::<Vec<_>>();
        assert_eq!(deltas, ["one", " two", " three"]);
    }

    #[test]
    fn health_and_model_routes_name_the_exact_product() {
        runtime().block_on(async {
            let (jobs, _receiver) = channel(1);
            let ready = health(State(state(jobs, true))).await;
            assert_eq!(ready.status(), StatusCode::OK);
            let body = to_bytes(ready.into_body(), 1 << 20).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["status"], "ok");
            assert_eq!(body["generation_route"], "mtp-draft-3");

            let (jobs, _receiver) = channel(1);
            let unavailable = health(State(state(jobs, false))).await;
            assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
            let (jobs, _receiver) = channel(1);
            let models = models(State(state(jobs, true))).await.0;
            assert_eq!(models["data"][0]["id"], SERVED_MODEL);
            assert_eq!(models["data"][0]["owned_by"], "tuiskollm");
        });
    }

    fn streaming_request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_completion_tokens": 7
        }))
        .unwrap()
    }

    #[test]
    fn exact_model_id_selects_one_concrete_resident_target() {
        for (model, expected, route) in [
            (Qwen38_27B::MODEL_ID, ServerModel::Qwen38, "mtp-draft-3"),
            (
                Qwen35_9B::MODEL_ID,
                ServerModel::Qwen35,
                "mtp-b1-compact-b2-8",
            ),
            (Qwen36Moe35B::MODEL_ID, ServerModel::Qwen36, "compact-b1-8"),
            (
                Qwen38FlashNext::MODEL_ID,
                ServerModel::Qwen38FlashNext,
                "compact-b1-8",
            ),
        ] {
            let target = ServerModel::from_model_id(model).unwrap();
            assert_eq!(target, expected);
            assert_eq!(target.model_id(), model);
            assert_eq!(target.generation_route(), route);
        }
        let error = ServerModel::from_model_id("moving/model").unwrap_err();
        assert!(error.contains("unsupported model `moving/model`"));
        for model in ServerModel::ALL {
            assert!(error.contains(model.model_id()));
        }
    }

    #[test]
    fn qwen38_flash_next_uses_the_zero_preamble_reasoning_default() {
        assert_eq!(QWEN38_FLASH_NEXT_SERVED_REASONING_EFFORT, "medium");
        assert_eq!(
            ServerModel::Qwen38FlashNext.reasoning_effort(),
            Some("medium")
        );
        for model in [
            ServerModel::Qwen38,
            ServerModel::Qwen35,
            ServerModel::Qwen36,
        ] {
            assert_eq!(model.reasoning_effort(), None);
        }
    }

    #[test]
    fn qwen38_flash_next_startup_reports_its_admitted_route() {
        let ready = Ready {
            model_id: Qwen38FlashNext::MODEL_ID,
            generation_route: ServerModel::Qwen38FlashNext.generation_route(),
            generation_defaults: GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            reasoning_effort: ServerModel::Qwen38FlashNext.reasoning_effort(),
            device_name: "NVIDIA GeForce RTX 5090".into(),
            checkpoint_admission: Duration::ZERO,
            weight_load: Duration::ZERO,
            source_prefault: Duration::ZERO,
            graph_capture: Duration::ZERO,
            tensor_count: 2_003,
            upload_bytes: 0,
            prefault_bytes: 0,
            arena_bytes: 29 * (1 << 30),
            host_stager_bytes: 15 * (1 << 20),
            slot_capacity: MAX_BATCH,
            context_capacity: 262_144,
            detailed_load_timing: true,
        };
        let output = render_startup(
            &ready,
            Duration::ZERO,
            "127.0.0.1:8000".parse().unwrap(),
            false,
        );

        assert!(output.contains("compact-b1-8"));
        assert!(output.contains("8 slots"));
        assert!(output.contains("context 262144"));
    }

    #[test]
    fn streaming_handler_enqueues_the_real_request_and_surfaces_worker_errors() {
        runtime().block_on(async {
            let (jobs, mut receiver) = channel(1);
            let state = state(jobs, true);
            let handler = tokio::spawn(chat_completions(
                State(state),
                Ok(Json(streaming_request())),
            ));

            let queued = receiver.recv().await.unwrap();
            assert_eq!(queued.request.max_new_tokens, 7);
            queued
                .reply
                .try_send(GenerationReply::Failed("fixture failure".into()))
                .unwrap();
            let response = handler.await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let body = String::from_utf8(bytes.to_vec()).unwrap();
            let error = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .find_map(|value| value.get("error").cloned())
                .unwrap_or_else(|| panic!("missing streamed error in {body:?}"));
            assert_eq!(error["message"], "fixture failure");
            assert_eq!(error["type"], "server_error");
        });
    }

    #[test]
    fn streaming_admission_rejection_is_a_bad_request_not_a_stream() {
        runtime().block_on(async {
            let (jobs, mut receiver) = channel(1);
            let state = state(jobs, true);
            let handler = tokio::spawn(chat_completions(
                State(state),
                Ok(Json(streaming_request())),
            ));

            let queued = receiver.recv().await.unwrap();
            queued
                .reply
                .try_send(GenerationReply::Rejected(
                    "prompt exceeds the resident context".into(),
                ))
                .unwrap();
            let response = handler.await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let error: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                error["error"]["message"],
                "prompt exceeds the resident context"
            );
            assert_eq!(error["error"]["type"], "invalid_request_error");
        });
    }

    #[test]
    fn streaming_capacity_rejection_is_retryable_not_a_stream() {
        runtime().block_on(async {
            let (jobs, mut receiver) = channel(1);
            let state = state(jobs, true);
            let handler = tokio::spawn(chat_completions(
                State(state),
                Ok(Json(streaming_request())),
            ));

            let queued = receiver.recv().await.unwrap();
            queued
                .reply
                .try_send(GenerationReply::Overloaded(
                    "shared KV pages are busy".into(),
                ))
                .unwrap();
            let response = handler.await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(response.headers()[RETRY_AFTER], "1");
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let error: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"]["type"], "server_overloaded");
        });
    }

    #[test]
    fn streaming_worker_disconnect_before_admission_is_unavailable() {
        runtime().block_on(async {
            let (jobs, mut receiver) = channel(1);
            let state = state(jobs, true);
            let handler = tokio::spawn(chat_completions(
                State(state),
                Ok(Json(streaming_request())),
            ));

            let queued = receiver.recv().await.unwrap();
            drop(queued);
            let response = handler.await.unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        });
    }
}
