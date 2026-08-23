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
use tokio::sync::mpsc::{
    Receiver, Sender, UnboundedSender, channel, error::TryRecvError, error::TrySendError,
    unbounded_channel,
};
use tuisko_engine::{MAX_BATCH, ResidentBatchGenerator, ResidentRequestId};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const REQUEST_BODY_LIMIT: usize = 2 * 1024 * 1024;
const DEFAULT_SEED_SCRAMBLE: u64 = 0x9e37_79b9_7f4a_7c15;

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
}

#[derive(Clone)]
struct AppState {
    jobs: Sender<Job>,
    request_ids: Arc<AtomicU64>,
    worker_ready: Arc<AtomicBool>,
}

struct Job {
    request: tuisko_engine::ChatGenerationRequest,
    reply: UnboundedSender<GenerationReply>,
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
    let (state, ready) = start_worker(&config.snapshot)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(config.address).await?;
        let stdout = std::io::stdout();
        let output = render_startup(
            &ready,
            startup_start.elapsed(),
            config.address,
            stdout.is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        );
        let mut stdout = stdout.lock();
        stdout.write_all(output.as_bytes())?;
        stdout.flush()?;
        axum::serve(listener, router(state)).await?;
        Ok(())
    })
}

fn start_worker(snapshot: &Path) -> Result<(AppState, Ready), ServerError> {
    let (jobs_tx, jobs_rx) = channel(MAX_BATCH);
    let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
    let worker_ready = Arc::new(AtomicBool::new(false));
    let snapshot = snapshot.to_owned();
    let worker_ready_clone = Arc::clone(&worker_ready);
    std::thread::Builder::new()
        .name("tuiskollm-engine".into())
        .spawn(move || engine_worker(snapshot, jobs_rx, ready_tx, worker_ready_clone))?;
    let ready = ready_rx
        .recv()
        .map_err(|_| ServerError::StartupDisconnected)?
        .map_err(ServerError::Startup)?;
    Ok((
        AppState {
            jobs: jobs_tx,
            request_ids: Arc::new(AtomicU64::new(1)),
            worker_ready,
        },
        ready,
    ))
}

fn engine_worker(
    snapshot: PathBuf,
    mut jobs: Receiver<Job>,
    ready: std_mpsc::SyncSender<Result<Ready, String>>,
    worker_ready: Arc<AtomicBool>,
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
        let generator = ResidentBatchGenerator::from_snapshot_device_zero(snapshot)
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
            fail_all(&mut replies, error);
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
                fail_all(&mut replies, error.to_string());
                break;
            }
        };
        for event in events.iter() {
            if let Some(reply) = replies.get(&event.request_id)
                && let Some(delta) = &event.step.delta
            {
                let _ = reply.send(GenerationReply::Delta(delta.clone()));
            }
            if let Some(output) = &event.completed
                && let Some(reply) = replies.remove(&event.request_id)
            {
                let _ = reply.send(GenerationReply::Done(output.clone()));
            }
        }
    }
}

fn render_startup(ready: &Ready, total: Duration, address: SocketAddr, color: bool) -> String {
    let mut output = String::new();
    let (header, ok, ready_label, reset) = if color {
        ("\x1b[1;36m", "\x1b[32m", "\x1b[1;32m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    writeln!(
        output,
        "{header}TuiskoLLM{reset} · {SERVED_MODEL} · {}",
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
        "{ready_label}READY{reset}          {:>7.1} ms · http://{address} · {MAX_BATCH} slots · context {} · {:.2} GiB device · {:.2} MiB pinned",
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
    generator: &mut ResidentBatchGenerator,
    replies: &mut HashMap<ResidentRequestId, UnboundedSender<GenerationReply>>,
    job: Job,
) {
    if job.reply.is_closed() {
        return;
    }
    match generator.admit(&job.request) {
        Ok(admission) => {
            if let Some(output) = admission.completed {
                let _ = job.reply.send(GenerationReply::Done(output));
            } else {
                let previous = replies.insert(admission.request_id, job.reply);
                debug_assert!(previous.is_none(), "resident request identities are unique");
            }
        }
        Err(error) => {
            let _ = job.reply.send(GenerationReply::Rejected(error.to_string()));
        }
    }
}

fn cancel_disconnected(
    generator: &mut ResidentBatchGenerator,
    replies: &mut HashMap<ResidentRequestId, UnboundedSender<GenerationReply>>,
) -> Result<(), String> {
    let cancelled = generator
        .active_request_ids()
        .filter(|request| replies.get(request).is_none_or(UnboundedSender::is_closed))
        .collect::<Vec<_>>();
    for request in cancelled {
        replies.remove(&request);
        generator
            .cancel(request)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn fail_all(
    replies: &mut HashMap<ResidentRequestId, UnboundedSender<GenerationReply>>,
    message: String,
) {
    for (_, reply) in replies.drain() {
        let _ = reply.send(GenerationReply::Failed(message.clone()));
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
        Json(json!({"status": "ok"})).into_response()
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
    let (reply_tx, reply_rx) = unbounded_channel();
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
        streaming_response(
            reply_rx,
            id,
            created,
            split_reasoning,
            parse_tools,
            include_usage,
        )
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
        AppState, EnqueueError, Job, Ready, chat_completions, enqueue_job, health, models,
        render_startup,
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
    use tokio::sync::mpsc::{channel, unbounded_channel};
    use tuisko_engine::ChatGenerationRequest;
    use tuisko_frontend::ChatMessage;

    fn runtime() -> tokio::runtime::Runtime {
        Builder::new_current_thread().enable_all().build().unwrap()
    }

    fn job() -> (Job, tokio::sync::mpsc::UnboundedReceiver<GenerationReply>) {
        let (reply, receiver) = unbounded_channel();
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
        let plain = render_startup(&ready, Duration::from_micros(1_925_200), address, false);
        let lines = plain.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 6);
        assert_eq!(
            lines[0],
            "TuiskoLLM · unsloth/Qwen3.8-27B-NVFP4 · NVIDIA GeForce RTX 5090"
        );
        assert!(lines[1].contains("2.5 ms · 1968 tensors"));
        assert!(lines[2].contains("53.0 ms · 18.00 GiB"));
        assert!(lines[3].contains("1604.6 ms · 19.00 GiB"));
        assert!(lines[4].contains("83.1 ms"));
        assert!(lines[5].contains("1925.2 ms · http://127.0.0.1:8000"));
        assert!(
            lines[5].contains("8 slots · context 220000 · 25.00 GiB device · 16.00 MiB pinned")
        );
        assert!(!plain.contains('\x1b'));

        let colored = render_startup(&ready, Duration::from_micros(1_925_200), address, true);
        assert!(colored.starts_with("\x1b[1;36mTuiskoLLM\x1b[0m"));
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
    fn health_and_model_routes_name_the_exact_product() {
        runtime().block_on(async {
            let (jobs, _receiver) = channel(1);
            let ready = health(State(state(jobs, true))).await;
            assert_eq!(ready.status(), StatusCode::OK);
            let body = to_bytes(ready.into_body(), 1 << 20).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["status"], "ok");

            let (jobs, _receiver) = channel(1);
            let unavailable = health(State(state(jobs, false))).await;
            assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
            let models = models().await.0;
            assert_eq!(models["data"][0]["id"], SERVED_MODEL);
            assert_eq!(models["data"][0]["owned_by"], "tuiskollm");
        });
    }

    #[test]
    fn streaming_handler_enqueues_the_real_request_and_surfaces_worker_errors() {
        runtime().block_on(async {
            let (jobs, mut receiver) = channel(1);
            let state = state(jobs, true);
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_completion_tokens": 7
            }))
            .unwrap();
            let response = chat_completions(State(state), Ok(Json(request))).await;
            assert_eq!(response.status(), StatusCode::OK);

            let queued = receiver.try_recv().unwrap();
            assert_eq!(queued.request.max_new_tokens, 7);
            queued
                .reply
                .send(GenerationReply::Failed("fixture failure".into()))
                .unwrap();
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
}
