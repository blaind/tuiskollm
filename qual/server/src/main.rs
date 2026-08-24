//! External OpenAI HTTP qualification for the exact resident TuiskoLLM server.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use thiserror::Error;
use tuisko_model::{Arch, Qwen38_27B};

const GENERATION_ROUTE: &str = "mtp-draft-3";
const COMPLETION_TOKENS: usize = 8;
const CANCEL_COMPLETION_TOKENS: usize = 128;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const PROMPT: &str = "Reply with exactly the word blue.";

type Result<T> = std::result::Result<T, QualError>;

#[derive(Debug, Error)]
enum QualError {
    #[error("{0}")]
    Contract(String),
    #[error("HTTP transport failed: {0}")]
    Http(#[from] ureq::Error),
    #[error("response body read failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("response JSON failed to decode: {0}")]
    Json(#[from] serde_json::Error),
    #[error("concurrent request thread panicked")]
    ThreadPanic,
}

#[derive(Clone)]
struct Client {
    agent: ureq::Agent,
    base_url: Arc<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Completion {
    content: String,
    finish_reason: String,
    usage: Usage,
}

struct Qualification {
    client: Client,
    checks: usize,
    concurrent_requests: usize,
}

fn main() {
    let base_url = match parse_base_url(std::env::args()) {
        Ok(base_url) => base_url,
        Err(error) => {
            eprintln!("tuisko-server-qual: {error}");
            std::process::exit(2);
        }
    };
    let mut qualification = Qualification::new(base_url);
    match qualification.run() {
        Ok(completion) => {
            let digest = Sha256::digest(completion.content.as_bytes());
            println!(
                "PASS server-http: {} checks; B=1..8 concurrency ({} requests); cancellation and eight-slot recycling; completion sha256={digest:x}",
                qualification.checks, qualification.concurrent_requests,
            );
        }
        Err(error) => {
            eprintln!(
                "FAIL server-http after {} passed checks: {error}",
                qualification.checks
            );
            std::process::exit(1);
        }
    }
}

fn parse_base_url(args: impl IntoIterator<Item = String>) -> Result<String> {
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "tuisko-server-qual".into());
    let base_url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8000".into());
    if args.next().is_some() {
        return Err(QualError::Contract(format!(
            "usage: {program} [http://HOST:PORT]"
        )));
    }
    let base_url = base_url.trim_end_matches('/');
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        return Err(QualError::Contract(
            "server URL must begin with http:// or https://".into(),
        ));
    }
    Ok(base_url.into())
}

impl Client {
    fn new(base_url: String) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
            base_url: base_url.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let mut response = self.agent.get(self.url(path)).call()?;
        expect_status(&response, 200, path)?;
        Ok(response.body_mut().read_json()?)
    }

    fn blocking(&self) -> Result<Completion> {
        self.blocking_request(request(false, COMPLETION_TOKENS), "blocking completion")
    }

    fn blocking_request(&self, request: Value, label: &str) -> Result<Completion> {
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request)?;
        expect_status(&response, 200, label)?;
        let body: Value = response.body_mut().read_json()?;
        validate_blocking(&body)
    }

    fn expect_rejection(&self, request: Value, label: &str) -> Result<()> {
        let response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request)?;
        expect_status(&response, 400, label)
    }

    fn streaming(&self) -> Result<Completion> {
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request(true, COMPLETION_TOKENS))?;
        expect_status(&response, 200, "streaming completion")?;
        expect_event_stream(&response)?;
        let body = response.body_mut().read_to_string()?;
        validate_stream(&parse_sse(&body)?)
    }

    fn disconnect_stream(&self) -> Result<()> {
        let response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .header("connection", "close")
            .send_json(request(true, CANCEL_COMPLETION_TOKENS))?;
        expect_status(&response, 200, "cancellation stream")?;
        expect_event_stream(&response)?;

        let (_, body) = response.into_parts();
        let mut reader = BufReader::new(body.into_reader());
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(QualError::Contract(
                    "cancellation stream ended before its first data event".into(),
                ));
            }
            if line.trim_end().starts_with("data: ") {
                break;
            }
        }
        drop(reader);
        Ok(())
    }
}

impl Qualification {
    fn new(base_url: String) -> Self {
        Self {
            client: Client::new(base_url),
            checks: 0,
            concurrent_requests: 0,
        }
    }

    fn check(&mut self, condition: bool, message: impl Into<String>) -> Result<()> {
        if condition {
            self.checks += 1;
            Ok(())
        } else {
            Err(QualError::Contract(message.into()))
        }
    }

    fn run(&mut self) -> Result<Completion> {
        let health = self.client.get_json("/health")?;
        self.check(
            health == json!({"status": "ok", "generation_route": GENERATION_ROUTE}),
            format!("unexpected /health response: {health}"),
        )?;

        let models = self.client.get_json("/v1/models")?;
        self.check(
            models
                == json!({
                    "object": "list",
                    "data": [{
                        "id": Qwen38_27B::MODEL_ID,
                        "object": "model",
                        "owned_by": "tuiskollm"
                    }]
                }),
            format!("unexpected /v1/models response: {models}"),
        )?;

        let blocking = self.client.blocking()?;
        self.check(!blocking.content.is_empty(), "blocking content is empty")?;

        for (label, stop) in [
            ("empty stop string", json!("")),
            ("empty stop list", json!([])),
        ] {
            let mut request = request(false, COMPLETION_TOKENS);
            request["stop"] = stop;
            let completion = self.client.blocking_request(request, label)?;
            self.check(
                completion == blocking,
                format!(
                    "{label} changed completion semantics: expected={blocking:?}, actual={completion:?}"
                ),
            )?;
        }

        for field in ["max_tokens", "max_completion_tokens"] {
            let mut request = request(false, COMPLETION_TOKENS);
            let request_object = request
                .as_object_mut()
                .expect("the qualification request is an object");
            request_object.remove("max_completion_tokens");
            request_object.insert(field.into(), json!(0));
            self.client
                .expect_rejection(request, &format!("zero {field}"))?;
            self.check(true, format!("zero {field} was not rejected"))?;
        }

        for (label, content) in [
            ("missing user content", None),
            ("null user content", Some(Value::Null)),
        ] {
            let mut request = request(false, COMPLETION_TOKENS);
            let mut message = json!({"role": "user"});
            if let Some(content) = content {
                message["content"] = content;
            }
            request["messages"] = json!([message]);
            self.client.expect_rejection(request, label)?;
            self.check(true, format!("{label} was not rejected"))?;
        }

        let streaming = self.client.streaming()?;
        self.check(
            streaming == blocking,
            format!(
                "blocking and streaming semantics differ: blocking={blocking:?}, streaming={streaming:?}"
            ),
        )?;

        for batch in 1..=8 {
            for completion in concurrent(self.client.clone(), batch)? {
                self.concurrent_requests += 1;
                self.check(
                    completion == blocking,
                    format!(
                        "greedy completion changed during B={batch}: expected={blocking:?}, actual={completion:?}"
                    ),
                )?;
            }
        }

        self.client.disconnect_stream()?;
        self.check(true, "cancellation stream did not disconnect")?;
        for completion in concurrent(self.client.clone(), 8)? {
            self.concurrent_requests += 1;
            self.check(
                completion == blocking,
                format!(
                    "completion changed after cancellation and slot recycling: expected={blocking:?}, actual={completion:?}"
                ),
            )?;
        }

        let health = self.client.get_json("/health")?;
        self.check(
            health == json!({"status": "ok", "generation_route": GENERATION_ROUTE}),
            format!("server was not healthy after cancellation: {health}"),
        )?;
        Ok(blocking)
    }
}

fn request(stream: bool, max_completion_tokens: usize) -> Value {
    let mut request = json!({
        "model": Qwen38_27B::MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    if stream {
        request["stream_options"] = json!({"include_usage": true});
    }
    request
}

fn concurrent(client: Client, count: usize) -> Result<Vec<Completion>> {
    let barrier = Arc::new(Barrier::new(count));
    let handles = (0..count)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let client = client.clone();
            thread::spawn(move || {
                barrier.wait();
                client.blocking()
            })
        })
        .collect::<Vec<_>>();
    handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| QualError::ThreadPanic)?)
        .collect()
}

fn expect_status(
    response: &ureq::http::Response<ureq::Body>,
    expected: u16,
    label: &str,
) -> Result<()> {
    let actual = response.status().as_u16();
    if actual == expected {
        Ok(())
    } else {
        Err(QualError::Contract(format!(
            "{label} returned HTTP {actual}, expected {expected}"
        )))
    }
}

fn expect_event_stream(response: &ureq::http::Response<ureq::Body>) -> Result<()> {
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if content_type.starts_with("text/event-stream") {
        Ok(())
    } else {
        Err(QualError::Contract(format!(
            "stream response content-type is {content_type:?}, expected text/event-stream"
        )))
    }
}

fn validate_blocking(body: &Value) -> Result<Completion> {
    expect_common(body, "chat.completion")?;
    let choices = array(field(body, "choices")?, "blocking choices")?;
    if choices.len() != 1 {
        return Err(QualError::Contract(format!(
            "blocking response has {} choices, expected one",
            choices.len()
        )));
    }
    let choice = &choices[0];
    expect_u64(field(choice, "index")?, "blocking choice index", 0)?;
    let message = object(field(choice, "message")?, "blocking message")?;
    expect_str_value(message.get("role"), "blocking message role", "assistant")?;
    let content = string(
        message
            .get("content")
            .ok_or_else(|| missing("message.content"))?,
        "blocking content",
    )?
    .to_owned();
    let finish_reason = finish_reason(field(choice, "finish_reason")?)?;
    let usage = parse_usage(field(body, "usage")?)?;
    Ok(Completion {
        content,
        finish_reason,
        usage,
    })
}

fn validate_stream(events: &[SseEvent]) -> Result<Completion> {
    if events.len() < 4 {
        return Err(QualError::Contract(format!(
            "stream has {} events, expected role, terminal, usage, and DONE",
            events.len()
        )));
    }
    if !matches!(events.last(), Some(SseEvent::Done)) {
        return Err(QualError::Contract("stream does not end in [DONE]".into()));
    }
    if events[..events.len() - 1]
        .iter()
        .any(|event| matches!(event, SseEvent::Done))
    {
        return Err(QualError::Contract(
            "stream contains [DONE] before the final event".into(),
        ));
    }

    let values = events[..events.len() - 1]
        .iter()
        .map(|event| match event {
            SseEvent::Json(value) => Ok(value),
            SseEvent::Done => unreachable!("early DONE was rejected"),
        })
        .collect::<Result<Vec<_>>>()?;
    let first = values[0];
    expect_common(first, "chat.completion.chunk")?;
    let stream_id = string(field(first, "id")?, "stream id")?;
    let created = integer(field(first, "created")?, "stream created")?;
    let first_choices = array(field(first, "choices")?, "first stream choices")?;
    if first_choices.len() != 1 {
        return Err(QualError::Contract(
            "first stream event does not have one choice".into(),
        ));
    }
    let first_choice = &first_choices[0];
    expect_u64(field(first_choice, "index")?, "first stream index", 0)?;
    expect_null(field(first_choice, "finish_reason")?, "first finish reason")?;
    let first_delta = object(field(first_choice, "delta")?, "first stream delta")?;
    expect_str_value(first_delta.get("role"), "first stream role", "assistant")?;
    expect_null(field(first, "usage")?, "first stream usage")?;

    let mut content = String::new();
    let mut terminal_reason = None;
    let mut usage = None;
    for (index, value) in values.iter().enumerate().skip(1) {
        expect_common(value, "chat.completion.chunk")?;
        expect_str_value(value.get("id"), "stream chunk id", stream_id)?;
        expect_u64(field(value, "created")?, "stream chunk created", created)?;
        let choices = array(field(value, "choices")?, "stream choices")?;
        if choices.is_empty() {
            if index != values.len() - 1 || usage.is_some() {
                return Err(QualError::Contract(
                    "usage-only stream event is not uniquely last before DONE".into(),
                ));
            }
            usage = Some(parse_usage(field(value, "usage")?)?);
            continue;
        }
        if terminal_reason.is_some() {
            return Err(QualError::Contract(
                "ordinary stream event follows the terminal event".into(),
            ));
        }
        if choices.len() != 1 {
            return Err(QualError::Contract(format!(
                "stream event {index} has {} choices, expected one",
                choices.len()
            )));
        }
        expect_null(field(value, "usage")?, "ordinary stream usage")?;
        let choice = &choices[0];
        expect_u64(field(choice, "index")?, "stream choice index", 0)?;
        let delta = object(field(choice, "delta")?, "stream delta")?;
        if let Some(value) = delta.get("content") {
            content.push_str(string(value, "stream content delta")?);
        }
        let reason = field(choice, "finish_reason")?;
        if !reason.is_null() {
            if terminal_reason.is_some() || index == values.len() - 1 {
                return Err(QualError::Contract(
                    "terminal stream event is duplicated or follows usage".into(),
                ));
            }
            if !delta.is_empty() {
                return Err(QualError::Contract(
                    "terminal stream event has a non-empty delta".into(),
                ));
            }
            terminal_reason = Some(finish_reason(reason)?);
        }
    }
    Ok(Completion {
        content,
        finish_reason: terminal_reason
            .ok_or_else(|| QualError::Contract("stream has no terminal finish reason".into()))?,
        usage: usage.ok_or_else(|| QualError::Contract("stream has no usage event".into()))?,
    })
}

#[derive(Debug)]
enum SseEvent {
    Json(Value),
    Done,
}

fn parse_sse(body: &str) -> Result<Vec<SseEvent>> {
    let mut events = Vec::new();
    let mut data: Option<String> = None;
    for line in body.lines().chain(std::iter::once("")) {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if let Some(data) = data.take() {
                if data == "[DONE]" {
                    events.push(SseEvent::Done);
                } else {
                    events.push(SseEvent::Json(serde_json::from_str(&data)?));
                }
            }
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        let value = line.strip_prefix("data: ").ok_or_else(|| {
            QualError::Contract(format!("unsupported SSE field in line {line:?}"))
        })?;
        if data.replace(value.to_owned()).is_some() {
            return Err(QualError::Contract(
                "SSE event contains more than one data field".into(),
            ));
        }
    }
    Ok(events)
}

fn expect_common(value: &Value, object_kind: &str) -> Result<()> {
    let id = string(field(value, "id")?, "completion id")?;
    if !id.starts_with("chatcmpl-tuisko-") {
        return Err(QualError::Contract(format!(
            "completion id {id:?} lacks the Tuisko prefix"
        )));
    }
    expect_str_value(value.get("object"), "response object", object_kind)?;
    expect_str_value(value.get("model"), "response model", Qwen38_27B::MODEL_ID)?;
    if integer(field(value, "created")?, "created timestamp")? == 0 {
        return Err(QualError::Contract("created timestamp is zero".into()));
    }
    Ok(())
}

fn parse_usage(value: &Value) -> Result<Usage> {
    let prompt_tokens = usize_value(field(value, "prompt_tokens")?, "prompt tokens")?;
    let prompt_details = field(value, "prompt_tokens_details")?;
    let cached_tokens = usize_value(
        field(prompt_details, "cached_tokens")?,
        "cached prompt tokens",
    )?;
    let completion_tokens = usize_value(field(value, "completion_tokens")?, "completion tokens")?;
    let total_tokens = usize_value(field(value, "total_tokens")?, "total tokens")?;
    if prompt_tokens == 0 || !(1..=COMPLETION_TOKENS).contains(&completion_tokens) {
        return Err(QualError::Contract(format!(
            "usage has invalid prompt/completion counts {prompt_tokens}/{completion_tokens}"
        )));
    }
    if cached_tokens > prompt_tokens {
        return Err(QualError::Contract(format!(
            "usage has {cached_tokens} cached tokens but only {prompt_tokens} prompt tokens"
        )));
    }
    if total_tokens != prompt_tokens + completion_tokens {
        return Err(QualError::Contract(format!(
            "usage total {total_tokens} does not equal prompt plus completion {}",
            prompt_tokens + completion_tokens
        )));
    }
    Ok(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

fn finish_reason(value: &Value) -> Result<String> {
    let reason = string(value, "finish reason")?;
    if matches!(reason, "stop" | "length") {
        Ok(reason.to_owned())
    } else {
        Err(QualError::Contract(format!(
            "unsupported finish reason {reason:?}"
        )))
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value.get(name).ok_or_else(|| missing(name))
}

fn missing(name: &str) -> QualError {
    QualError::Contract(format!("response is missing {name}"))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| QualError::Contract(format!("{label} is not an object")))
}

fn array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| QualError::Contract(format!("{label} is not an array")))
}

fn string<'a>(value: &'a Value, label: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| QualError::Contract(format!("{label} is not a string")))
}

fn integer(value: &Value, label: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| QualError::Contract(format!("{label} is not an unsigned integer")))
}

fn usize_value(value: &Value, label: &str) -> Result<usize> {
    usize::try_from(integer(value, label)?)
        .map_err(|_| QualError::Contract(format!("{label} does not fit usize")))
}

fn expect_u64(value: &Value, label: &str, expected: u64) -> Result<()> {
    let actual = integer(value, label)?;
    if actual == expected {
        Ok(())
    } else {
        Err(QualError::Contract(format!(
            "{label} is {actual}, expected {expected}"
        )))
    }
}

fn expect_null(value: &Value, label: &str) -> Result<()> {
    if value.is_null() {
        Ok(())
    } else {
        Err(QualError::Contract(format!(
            "{label} is {value}, expected null"
        )))
    }
}

fn expect_str_value(value: Option<&Value>, label: &str, expected: &str) -> Result<()> {
    let actual = value
        .and_then(Value::as_str)
        .ok_or_else(|| QualError::Contract(format!("{label} is not a string")))?;
    if actual == expected {
        Ok(())
    } else {
        Err(QualError::Contract(format!(
            "{label} is {actual:?}, expected {expected:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Completion, QualError, Usage, parse_base_url, parse_sse, validate_blocking, validate_stream,
    };
    use serde_json::json;

    #[test]
    fn base_url_defaults_and_trims_one_argument() {
        assert_eq!(
            parse_base_url(["qual".into()]).unwrap(),
            "http://127.0.0.1:8000"
        );
        assert_eq!(
            parse_base_url(["qual".into(), "http://server:9000/".into()]).unwrap(),
            "http://server:9000"
        );
    }

    #[test]
    fn blocking_and_sse_fixtures_have_identical_semantics() {
        let usage = json!({
            "prompt_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens": 2,
            "total_tokens": 13
        });
        let blocking = json!({
            "id": "chatcmpl-tuisko-0001",
            "object": "chat.completion",
            "created": 1,
            "model": "unsloth/Qwen3.8-27B-NVFP4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "blue"},
                "finish_reason": "stop"
            }],
            "usage": usage
        });
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-tuisko-0002\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"unsloth/Qwen3.8-27B-NVFP4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}],\"usage\":null}\n\n",
            ": keep-alive\n\n",
            "data: {\"id\":\"chatcmpl-tuisko-0002\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"unsloth/Qwen3.8-27B-NVFP4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"bl\"},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-tuisko-0002\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"unsloth/Qwen3.8-27B-NVFP4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ue\"},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-tuisko-0002\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"unsloth/Qwen3.8-27B-NVFP4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-tuisko-0002\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"unsloth/Qwen3.8-27B-NVFP4\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"prompt_tokens_details\":{\"cached_tokens\":7},\"completion_tokens\":2,\"total_tokens\":13}}\n\n",
            "data: [DONE]\n\n"
        );
        let expected = Completion {
            content: "blue".into(),
            finish_reason: "stop".into(),
            usage: Usage {
                prompt_tokens: 11,
                completion_tokens: 2,
                total_tokens: 13,
            },
        };
        assert_eq!(validate_blocking(&blocking).unwrap(), expected);
        assert_eq!(validate_stream(&parse_sse(sse).unwrap()).unwrap(), expected);
    }

    #[test]
    fn sse_parser_refuses_multiple_data_fields() {
        let error = parse_sse("data: one\ndata: two\n\n").unwrap_err();
        assert!(matches!(error, QualError::Contract(_)));
        assert!(error.to_string().contains("more than one data field"));
    }

    #[test]
    fn stream_requires_final_done() {
        let error = validate_stream(&parse_sse(": keep-alive\n\n").unwrap()).unwrap_err();
        assert!(error.to_string().contains("expected role"));
    }
}
