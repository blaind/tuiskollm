//! Release-server qualification over the public OpenAI-compatible boundary.

use serde_json::{Value, json};
use std::error::Error;
use std::io;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy)]
struct ServerProfile {
    label: &'static str,
    model: &'static str,
    revision: &'static str,
    generation_route: &'static str,
    reasoning: &'static str,
    tensor_count: usize,
}

const QWEN35: ServerProfile = ServerProfile {
    label: "Qwen3.5",
    model: "AxionML/Qwen3.5-9B-NVFP4",
    revision: "97aef92393f126bf649f310cd40861be8dad3279",
    generation_route: "mtp-b1-compact-b2-8",
    reasoning: "Thinking Process",
    tensor_count: 1_519,
};

const QWEN36: ServerProfile = ServerProfile {
    label: "Qwen3.6",
    model: "nvidia/Qwen3.6-35B-A3B-NVFP4",
    revision: "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
    generation_route: "compact-b1-8",
    reasoning: "Here's",
    tensor_count: 124_468,
};

/// Every sibling target this module can put through its own server qualification.
const SIBLINGS: [ServerProfile; 2] = [QWEN35, QWEN36];

pub(crate) fn qualify_qwen35(executable: &Path, snapshot: &Path) -> Result<(), Box<dyn Error>> {
    qualify(QWEN35, executable, snapshot)
}

pub(crate) fn qualify_qwen36(executable: &Path, snapshot: &Path) -> Result<(), Box<dyn Error>> {
    qualify(QWEN36, executable, snapshot)
}

/// Runs whichever sibling target's qualification the snapshot's pinned revision names.
///
/// The revision lives beside the rest of that target's server facts rather than in the caller,
/// so a checkpoint bump moves one row instead of two files. The server itself resolves the same
/// revision independently from `tuisko-model`, which is what makes a stale row here a visible
/// startup mismatch rather than a silent one.
pub(crate) fn qualify_sibling(executable: &Path, snapshot: &Path) -> Result<(), Box<dyn Error>> {
    let revision = snapshot.file_name().and_then(|name| name.to_str());
    let profile = SIBLINGS
        .into_iter()
        .find(|profile| revision == Some(profile.revision))
        .ok_or_else(|| {
            format!(
                "sibling snapshot revision {revision:?} is not {} or {}; the cross-target check \
                 runs a sibling's own server qualification, so it needs a snapshot that has one",
                QWEN35.revision, QWEN36.revision
            )
        })?;

    qualify(profile, executable, snapshot)
}

fn qualify(
    profile: ServerProfile,
    executable: &Path,
    snapshot: &Path,
) -> Result<(), Box<dyn Error>> {
    if !executable.is_file() {
        return Err(format!("server binary `{}` does not exist", executable.display()).into());
    }
    let snapshot = snapshot.canonicalize().map_err(|error| {
        format!(
            "resolving {} snapshot `{}`: {error}",
            profile.label,
            snapshot.display()
        )
    })?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);

    let child = Command::new(executable)
        .arg("serve")
        .arg(profile.model)
        .arg("--snapshot")
        .arg(&snapshot)
        .arg("--address")
        .arg(address.to_string())
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("starting `{}`: {error}", executable.display()))?;
    let mut server = ServerProcess::new(child);
    let agent = http_agent();
    let base = format!("http://{address}");

    wait_until_ready(profile, &agent, &base, &mut server)?;
    validate_models(profile, &get_json(&agent, &format!("{base}/v1/models"))?)?;

    let request = json!({
        "model": profile.model,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_completion_tokens": 2,
    });
    let blocking = post_json(
        &agent,
        &format!("{base}/v1/chat/completions"),
        &request,
        "application/json",
    )?;
    validate_blocking(profile, &blocking)?;

    let mut streaming_request = request;
    streaming_request["stream"] = Value::Bool(true);
    streaming_request["stream_options"] = json!({"include_usage": true});
    let streaming = post_json(
        &agent,
        &format!("{base}/v1/chat/completions"),
        &streaming_request,
        "text/event-stream",
    )?;
    validate_streaming(profile, &streaming)?;

    let output = server.stop()?;
    validate_startup_output(profile, &output)?;
    println!(
        "{} server qualification passed: health + exact model inventory + blocking + SSE; prompt/completion/total tokens 11/2/13",
        profile.label
    );
    Ok(())
}

fn http_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .timeout_connect(Some(Duration::from_millis(500)))
        .http_status_as_error(false)
        .build();
    ureq::Agent::new_with_config(config)
}

fn wait_until_ready(
    profile: ServerProfile,
    agent: &ureq::Agent,
    base: &str,
    server: &mut ServerProcess,
) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if server.has_exited()? {
            let output = server.stop()?;
            return Err(format!("server exited before readiness{}", render_output(&output)).into());
        }
        let health_url = format!("{base}/health");
        match agent.get(&health_url).call() {
            Ok(response) => {
                let body = read_response("GET", &health_url, "application/json", Ok(response))?;
                validate_health(profile, &body)?;
                return Ok(());
            }
            Err(ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed) => {}
            Err(error) => return Err(format!("GET {health_url}: {error}").into()),
        }
        if Instant::now() >= deadline {
            let output = server.stop()?;
            return Err(format!(
                "server did not become ready within {} seconds{}",
                STARTUP_TIMEOUT.as_secs(),
                render_output(&output)
            )
            .into());
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn get_json(agent: &ureq::Agent, url: &str) -> Result<String, Box<dyn Error>> {
    read_response("GET", url, "application/json", agent.get(url).call())
}

fn post_json(
    agent: &ureq::Agent,
    url: &str,
    body: &Value,
    content_type: &str,
) -> Result<String, Box<dyn Error>> {
    read_response("POST", url, content_type, agent.post(url).send_json(body))
}

fn read_response(
    method: &str,
    url: &str,
    expected_content_type: &str,
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<String, Box<dyn Error>> {
    let response = response.map_err(|error| format!("{method} {url}: {error}"))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|error| format!("reading {method} {url}: {error}"))?;
    if !(200..300).contains(&status) {
        return Err(format!("{method} {url} returned HTTP {status}: {body}").into());
    }
    if !content_type.starts_with(expected_content_type) {
        return Err(format!(
            "{method} {url} returned content type {content_type:?}, expected {expected_content_type:?}"
        )
        .into());
    }
    Ok(body)
}

fn parse_json(label: &str, body: &str) -> Result<Value, String> {
    serde_json::from_str(body).map_err(|error| format!("{label} returned invalid JSON: {error}"))
}

fn validate_health(profile: ServerProfile, body: &str) -> Result<(), String> {
    let value = parse_json("health", body)?;
    let expected = json!({"status": "ok", "generation_route": profile.generation_route});
    if value != expected {
        return Err(format!("health returned {value}, expected {expected}"));
    }
    Ok(())
}

fn validate_models(profile: ServerProfile, body: &str) -> Result<(), String> {
    let value = parse_json("models", body)?;
    let expected = json!({
        "object": "list",
        "data": [{"id": profile.model, "object": "model", "owned_by": "tuiskollm"}],
    });
    if value != expected {
        return Err(format!("models returned {value}, expected {expected}"));
    }
    Ok(())
}

fn validate_blocking(profile: ServerProfile, body: &str) -> Result<(), String> {
    let value = parse_json("blocking completion", body)?;
    let choices = value["choices"]
        .as_array()
        .ok_or_else(|| "blocking completion omitted choices".to_owned())?;
    if value["object"] != "chat.completion"
        || value["model"] != profile.model
        || choices.len() != 1
        || choices[0]["index"] != 0
        || choices[0]["finish_reason"] != "length"
        || choices[0]["message"]["role"] != "assistant"
        || choices[0]["message"]["content"] != ""
        || choices[0]["message"]["reasoning_content"] != profile.reasoning
    {
        return Err(format!(
            "blocking completion did not match the exact {} response: {value}",
            profile.label
        ));
    }
    validate_usage("blocking completion", &value["usage"])
}

fn validate_streaming(profile: ServerProfile, body: &str) -> Result<(), String> {
    let data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    if data.last() != Some(&"[DONE]") {
        return Err("streaming completion did not end with data: [DONE]".into());
    }
    let events = data[..data.len() - 1]
        .iter()
        .map(|event| parse_json("streaming completion event", event))
        .collect::<Result<Vec<_>, _>>()?;
    if events.is_empty() || events[0]["choices"][0]["delta"]["role"] != "assistant" {
        return Err("streaming completion did not begin with the assistant role".into());
    }
    if events
        .iter()
        .any(|event| event["object"] != "chat.completion.chunk" || event["model"] != profile.model)
    {
        return Err("streaming completion emitted the wrong object or model identity".into());
    }

    let reasoning = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect::<String>();
    if reasoning != profile.reasoning {
        return Err(format!(
            "streaming completion produced reasoning {reasoning:?}, expected {:?}",
            profile.reasoning
        ));
    }
    let terminal = events
        .iter()
        .filter(|event| event["choices"][0]["finish_reason"] == "length")
        .count();
    if terminal != 1 {
        return Err(format!(
            "streaming completion emitted {terminal} length terminals, expected one"
        ));
    }
    let usage = events
        .iter()
        .filter(|event| event["choices"].as_array().is_some_and(Vec::is_empty))
        .collect::<Vec<_>>();
    if usage.len() != 1 {
        return Err(format!(
            "streaming completion emitted {} usage-only chunks, expected one",
            usage.len()
        ));
    }
    validate_usage("streaming completion", &usage[0]["usage"])
}

fn validate_usage(label: &str, usage: &Value) -> Result<(), String> {
    let expected = json!({
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 0},
    });
    if usage != &expected {
        return Err(format!(
            "{label} returned usage {usage}, expected {expected}"
        ));
    }
    Ok(())
}

fn validate_startup_output(profile: ServerProfile, output: &Output) -> Result<(), Box<dyn Error>> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let tensor_count = format!("{} tensors", profile.tensor_count);
    for evidence in [
        profile.model,
        tensor_count.as_str(),
        "READY",
        "8 slots · context 262144",
    ] {
        if !stdout.contains(evidence) {
            return Err(format!(
                "server startup omitted {evidence:?}{}",
                render_output(output)
            )
            .into());
        }
    }
    Ok(())
}

fn render_output(output: &Output) -> String {
    format!(
        "\n--- server stdout ---\n{}\n--- server stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

struct ServerProcess {
    child: Option<Child>,
}

impl ServerProcess {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn has_exited(&mut self) -> io::Result<bool> {
        self.child
            .as_mut()
            .expect("server child is present until stop")
            .try_wait()
            .map(|status| status.is_some())
    }

    fn stop(&mut self) -> io::Result<Output> {
        let mut child = self.child.take().expect("server is stopped exactly once");
        if child.try_wait()?.is_none() {
            child.kill()?;
        }
        child.wait_with_output()
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN35, QWEN36, SIBLINGS, ServerProfile, validate_blocking, validate_health,
        validate_models, validate_streaming,
    };
    use serde_json::json;

    fn blocking(profile: ServerProfile) -> String {
        json!({
            "object": "chat.completion",
            "model": profile.model,
            "choices": [{
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": profile.reasoning,
                },
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        })
        .to_string()
    }

    #[test]
    fn exact_json_boundaries_are_admitted_for_both_targets() {
        for profile in [QWEN35, QWEN36] {
            validate_health(
                profile,
                &json!({"status": "ok", "generation_route": profile.generation_route}).to_string(),
            )
            .unwrap();
            validate_models(
                profile,
                &json!({
                    "object": "list",
                    "data": [{"id": profile.model, "object": "model", "owned_by": "tuiskollm"}],
                })
                .to_string(),
            )
            .unwrap();
            validate_blocking(profile, &blocking(profile)).unwrap();
        }
    }

    #[test]
    fn health_requires_the_exact_target_route() {
        let qwen35 = validate_health(
            QWEN35,
            r#"{"status":"ok","generation_route":"compact-b1-8"}"#,
        )
        .unwrap_err();
        assert!(qwen35.contains("mtp-b1-compact-b2-8"));
        let qwen36 = validate_health(
            QWEN36,
            r#"{"status":"ok","generation_route":"mtp-b1-compact-b2-8"}"#,
        )
        .unwrap_err();
        assert!(qwen36.contains("compact-b1-8"));
    }

    #[test]
    fn exact_streaming_boundary_is_admitted_for_both_targets() {
        for profile in [QWEN35, QWEN36] {
            let split = profile.reasoning.len() / 2;
            let body = format!(
                "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({
                    "object": "chat.completion.chunk",
                    "model": profile.model,
                    "choices": [{"delta": {"role": "assistant"}, "finish_reason": null}],
                }),
                json!({
                    "object": "chat.completion.chunk",
                    "model": profile.model,
                    "choices": [{"delta": {"reasoning_content": &profile.reasoning[..split]}, "finish_reason": null}],
                }),
                json!({
                    "object": "chat.completion.chunk",
                    "model": profile.model,
                    "choices": [{"delta": {"reasoning_content": &profile.reasoning[split..]}, "finish_reason": "length"}],
                }),
                json!({
                    "object": "chat.completion.chunk",
                    "model": profile.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 2,
                        "total_tokens": 13,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }),
            );
            validate_streaming(profile, &body).unwrap();
        }
    }

    #[test]
    fn every_sibling_profile_is_reachable_by_its_own_revision() {
        // Dispatch must identify one non-empty revision.
        assert_eq!(SIBLINGS.len(), 2);
        assert_ne!(QWEN35.revision, QWEN36.revision);
        for profile in SIBLINGS {
            assert_eq!(profile.revision.len(), 40);
            assert!(
                profile
                    .revision
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            );
        }
    }

    #[test]
    fn changed_usage_and_incomplete_streams_are_rejected() {
        let mut blocking = serde_json::from_str::<serde_json::Value>(&blocking(QWEN35)).unwrap();
        blocking["usage"]["prompt_tokens"] = json!(10);
        blocking["usage"]["total_tokens"] = json!(12);
        assert!(validate_blocking(QWEN35, &blocking.to_string()).is_err());
        assert!(validate_streaming(QWEN35, "data: {}\n\n").is_err());
    }
}
