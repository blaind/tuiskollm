//! Concrete HTTP server and resident scheduler worker.

use crate::{
    ChatCompletionRequest, ChatRequestError, GenerationReply, PreparedChatRequest, SERVED_MODEL,
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
    GenerationStep, MAX_BATCH, ResidentLoadPhase, ResidentLoadProgress, ResidentMtpBatchGenerator,
    ResidentRequestId,
};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const REQUEST_BODY_LIMIT: usize = 2 * 1024 * 1024;
const DEFAULT_SEED_SCRAMBLE: u64 = 0x9e37_79b9_7f4a_7c15;
// Bounded so the single-threaded worker never blocks on a stalled client; a full
// lane is treated exactly like a disconnected one.
const GENERATION_REPLY_BUFFER: usize = 32;
const GENERATION_ROUTE: &str = "mtp-draft-3";

/// Startup configuration for the one exact resident server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConfig {
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
}

struct Job {
    request: tuisko_engine::ChatGenerationRequest,
    reply: Sender<GenerationReply>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EnqueueError {
    Full,
    Closed,
}

struct Ready {
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
    context_capacity: usize,
}

/// Loads the exact resident model, then serves health, model, blocking, and SSE routes.
pub fn run(config: ServerConfig) -> Result<(), ServerError> {
    let startup_start = Instant::now();
    let stdout = std::io::stdout();
    let interactive = stdout.is_terminal();
    let color = interactive && std::env::var_os("NO_COLOR").is_none();
    let (state, ready, worker_failure) = {
        let mut stdout = stdout.lock();
        stdout.write_all(render_loading(color, interactive).as_bytes())?;
        stdout.flush()?;
        start_worker(&config.snapshot, &mut stdout, interactive, color)?
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
    output: &mut impl IoWrite,
    interactive: bool,
    color: bool,
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
        },
        ready,
        failure_rx,
    ))
}

fn engine_worker(
    snapshot: PathBuf,
    mut jobs: Receiver<Job>,
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
    let generator = (|| {
        let checkpoint_start = Instant::now();
        let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(&snapshot)
            .map(Arc::new)
            .map_err(|error| format!("admitting {}: {error}", snapshot.display()))?;
        let checkpoint_admission = checkpoint_start.elapsed();
        let tensor_count = snapshot.tensor_count();
        let generator = ResidentMtpBatchGenerator::from_snapshot_device_zero_with_progress(
            snapshot,
            progress.as_ref(),
        )
        .map_err(|error| format!("loading the resident text program: {error}"))?;
        let device_name = generator
            .context()
            .device_name()
            .map_err(|error| format!("reading the CUDA device name: {error}"))?;
        Ok::<_, String>((generator, checkpoint_admission, tensor_count, device_name))
    })();
    let (mut generator, checkpoint_admission, tensor_count, device_name) = match generator {
        Ok(result) => result,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let load_stats = generator.load_stats();
    let startup = Ready {
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
        context_capacity: generator.context_capacity(),
    };
    worker_ready.store(true, Ordering::Release);
    if ready.send(Ok(startup)).is_err() {
        return;
    }

    let mut replies = HashMap::new();
    let mut jobs_open = true;
    loop {
        if let Err(error) = cancel_disconnected(&mut generator, &mut replies) {
            fail_all(&mut replies, error.clone());
            let _ = failure.send(error);
            break;
        }

        if generator.active_requests() == 0 && jobs_open {
            match jobs.blocking_recv() {
                Some(job) => admit_job(&mut generator, &mut replies, job),
                None => jobs_open = false,
            }
        }
        while jobs_open && generator.active_requests() < MAX_BATCH {
            match jobs.try_recv() {
                Ok(job) => admit_job(&mut generator, &mut replies, job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    jobs_open = false;
                    break;
                }
            }
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
                let _ = failure.send(error);
                break;
            }
        };
        for event in events.iter() {
            let failed = replies
                .get(&event.request_id)
                .is_some_and(|reply| !try_send_generation_steps(reply, event.steps()));
            if failed {
                // Full or closed: drop the lane so the next cancel pass reaps the request.
                replies.remove(&event.request_id);
            }
            if let Some(output) = &event.completed
                && let Some(reply) = replies.remove(&event.request_id)
            {
                let _ = reply.try_send(GenerationReply::Done(output.clone()));
            }
        }
    }
}

fn render_loading(color: bool, interactive: bool) -> String {
    let (header, loading, reset) = if color {
        ("\x1b[1;36m", "\x1b[1;33m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let newline = if interactive { "" } else { "\n" };
    format!(
        "{header}TuiskoLLM{reset} · {SERVED_MODEL}\n{loading}LOADING{reset} ⠋       preparing resident model…{newline}"
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
    writeln!(
        output,
        "{ready_label}READY{reset}          {:>7.1} ms · http://{address} · {GENERATION_ROUTE} · {MAX_BATCH} slots · context {} · {:.2} GiB device · {:.2} MiB pinned",
        milliseconds(total),
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

fn admit_job(
    generator: &mut ResidentMtpBatchGenerator,
    replies: &mut HashMap<ResidentRequestId, Sender<GenerationReply>>,
    job: Job,
) {
    if job.reply.is_closed() {
        return;
    }
    match generator.admit(&job.request) {
        Ok(admission) => {
            if let Some(output) = admission.completed {
                let _ = job.reply.try_send(GenerationReply::Done(output));
            } else {
                let previous = replies.insert(admission.request_id, job.reply);
                debug_assert!(previous.is_none(), "resident request identities are unique");
            }
        }
        Err(error) => {
            let _ = job
                .reply
                .try_send(GenerationReply::Rejected(error.to_string()));
        }
    }
}

fn cancel_disconnected(
    generator: &mut ResidentMtpBatchGenerator,
    replies: &mut HashMap<ResidentRequestId, Sender<GenerationReply>>,
) -> Result<(), String> {
    let cancelled = generator
        .active_request_ids()
        .filter(|request| replies.get(request).is_none_or(Sender::is_closed))
        .collect::<Vec<_>>();
    for request in cancelled {
        replies.remove(&request);
        generator
            .cancel(request)
            .map_err(|error| error.to_string())?;
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

fn fail_all(replies: &mut HashMap<ResidentRequestId, Sender<GenerationReply>>, message: String) {
    for (_, reply) in replies.drain() {
        let _ = reply.try_send(GenerationReply::Failed(message.clone()));
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
        Json(json!({"status": "ok", "generation_route": GENERATION_ROUTE})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable"})),
        )
            .into_response()
    }
}

async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": SERVED_MODEL, "object": "model", "owned_by": "tuiskollm"}]
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
    } = match request.prepare(numeric_id ^ DEFAULT_SEED_SCRAMBLE) {
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
    let (reply_tx, mut reply_rx) = channel(GENERATION_REPLY_BUFFER);
    if let Err(error) = enqueue_job(
        &state.jobs,
        Job {
            request: generation,
            reply: reply_tx,
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
            Some(first) => streaming_response(
                first,
                reply_rx,
                id,
                created,
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
        blocking_response(reply_rx, id, created, split_reasoning, parse_tools).await
    }
}

fn enqueue_job(jobs: &Sender<Job>, job: Job) -> Result<(), EnqueueError> {
    match jobs.try_send(job) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => Err(EnqueueError::Full),
        Err(TrySendError::Closed(_)) => Err(EnqueueError::Closed),
    }
}

fn enqueue_error_response(error: EnqueueError) -> Response {
    match error {
        EnqueueError::Full => openai_error(
            StatusCode::TOO_MANY_REQUESTS,
            "resident inference queue is full".into(),
            "server_overloaded",
        ),
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
        AppState, EnqueueError, Job, Ready, ServerError, chat_completions, enqueue_job, health,
        models, render_loading, render_startup, render_weight_progress, router,
        serve_until_worker_failure, try_send_generation_steps,
    };
    use crate::{ChatCompletionRequest, GenerationReply, SERVED_MODEL};
    use axum::Json;
    use axum::body::to_bytes;
    use axum::extract::State;
    use axum::http::StatusCode;
    use serde_json::Value;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::time::Duration;
    use tokio::runtime::Builder;
    use tokio::sync::mpsc::channel;
    use tuisko_engine::{ChatGenerationRequest, FinishReason, GenerationStep};
    use tuisko_frontend::ChatMessage;

    fn runtime() -> tokio::runtime::Runtime {
        Builder::new_current_thread().enable_all().build().unwrap()
    }

    fn job() -> (Job, tokio::sync::mpsc::Receiver<GenerationReply>) {
        let (reply, receiver) = channel(8);
        (
            Job {
                request: ChatGenerationRequest::new(vec![ChatMessage::new("user", "hello")]),
                reply,
            },
            receiver,
        )
    }

    fn state(jobs: tokio::sync::mpsc::Sender<Job>, ready: bool) -> AppState {
        AppState {
            jobs,
            request_ids: Arc::new(AtomicU64::new(1)),
            worker_ready: Arc::new(AtomicBool::new(ready)),
        }
    }

    #[test]
    fn startup_output_is_exact_plain_text_or_terminal_color() {
        let ready = Ready {
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
            context_capacity: 220_000,
        };
        let address = "127.0.0.1:8000".parse::<SocketAddr>().unwrap();
        let loading = render_loading(false, false);
        assert_eq!(
            loading,
            "TuiskoLLM · unsloth/Qwen3.8-27B-NVFP4\nLOADING ⠋       preparing resident model…\n"
        );
        assert!(!render_loading(false, true).ends_with('\n'));

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

        let colored_loading = render_loading(true, true);
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
            let models = models().await.0;
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
