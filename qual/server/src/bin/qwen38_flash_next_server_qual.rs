//! Real-HTTP qualification for the served Qwen3.8 Flash-Next route.
//!
//! It covers transport parity, cancellation, refusal boundaries, bounded ingress, and prompt
//! counts transcribed from the committed frontend fixtures.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

const GENERATION_ROUTE: &str = "compact-b1-8";
const FUNDED_SLOTS: usize = 8;
/// Queue depth the transport holds in front of the funded slots.
const INGRESS_QUEUE: usize = 8;
/// Last visible length served by dense QSA.
const DENSE_BAND: usize = 2_051;
/// Served checkpoint and single-slot page-pool ceiling.
const SERVED_DEPTH: usize = 262_144;
const COMPLETION_TOKENS: usize = 8;
const HELLO_MESSAGE_BOUNDARY_TOKENS: usize = 6;
const THROUGHPUT_COMPLETION_TOKENS: usize = 64;
const CANCEL_COMPLETION_TOKENS: usize = 256;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// One rendered prompt whose exact token count the frontend fixtures pin.
struct PromptFixture {
    name: &'static str,
    messages: Value,
    reasoning_effort: Option<&'static str>,
    enable_thinking: Option<bool>,
    prompt_tokens: usize,
    message_boundary_tokens: usize,
}

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
    cached_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Completion {
    reasoning: String,
    content: String,
    finish_reason: String,
    usage: Usage,
}

fn same_completion_semantics(left: &Completion, right: &Completion) -> bool {
    left.reasoning == right.reasoning
        && left.content == right.content
        && left.finish_reason == right.finish_reason
        && left.usage.prompt_tokens == right.usage.prompt_tokens
        && left.usage.completion_tokens == right.usage.completion_tokens
        && left.usage.total_tokens == right.usage.total_tokens
}

/// What one streamed request cost, measured at the caller's own socket.
#[derive(Clone, Debug)]
struct StreamTiming {
    completion: Completion,
    wall: Duration,
    time_to_first_token: Duration,
    inter_token: Vec<Duration>,
    reasoning_deltas: usize,
    content_deltas: usize,
}

struct Qualification {
    client: Client,
    checks: usize,
    requests: usize,
}

enum Mode {
    Qualify,
    Measure,
}

struct Options {
    base_url: String,
    mode: Mode,
}

fn main() {
    let options = match Options::parse(std::env::args()) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("qwen38-flash-next-server-qual: {error}");
            std::process::exit(2);
        }
    };
    let mut qualification = Qualification::new(options.base_url);
    let outcome = match options.mode {
        Mode::Qualify => qualification.run(),
        Mode::Measure => qualification.measure(),
    };
    match outcome {
        Ok(()) => println!(
            "PASS qwen38-flash-next-server-http: {} checks over {} live requests; {FUNDED_SLOTS} funded slots, {INGRESS_QUEUE}-deep ingress, served depth {SERVED_DEPTH}",
            qualification.checks, qualification.requests,
        ),
        Err(error) => {
            eprintln!(
                "FAIL qwen38-flash-next-server-http after {} passed checks: {error}",
                qualification.checks
            );
            std::process::exit(1);
        }
    }
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let program = args
            .next()
            .unwrap_or_else(|| "qwen38-flash-next-server-qual".into());
        let usage = || format!("usage: {program} [http://HOST:PORT] [--measure]");
        let mut base_url = None;
        let mut measure = false;
        for argument in args {
            if argument == "--measure" {
                if measure {
                    return Err(QualError::Contract(usage()));
                }
                measure = true;
            } else if argument.starts_with('-') || base_url.replace(argument).is_some() {
                return Err(QualError::Contract(usage()));
            }
        }
        let base_url = base_url.unwrap_or_else(|| "http://127.0.0.1:8000".into());
        let base_url = base_url.trim_end_matches('/');
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(QualError::Contract(
                "server URL must begin with http:// or https://".into(),
            ));
        }
        Ok(Self {
            base_url: base_url.into(),
            mode: if measure {
                Mode::Measure
            } else {
                Mode::Qualify
            },
        })
    }
}

/// Wire requests with prompt counts pinned by the frontend fixtures.
fn fixtures() -> Vec<PromptFixture> {
    vec![
        PromptFixture {
            name: "hello-default",
            messages: json!([{"role": "user", "content": "Hello"}]),
            reasoning_effort: None,
            enable_thinking: None,
            prompt_tokens: 11,
            message_boundary_tokens: HELLO_MESSAGE_BOUNDARY_TOKENS,
        },
        PromptFixture {
            name: "hello-medium",
            messages: json!([{"role": "user", "content": "Hello"}]),
            reasoning_effort: Some("medium"),
            enable_thinking: None,
            prompt_tokens: 11,
            message_boundary_tokens: HELLO_MESSAGE_BOUNDARY_TOKENS,
        },
        PromptFixture {
            name: "hello-xhigh",
            messages: json!([{"role": "user", "content": "Hello"}]),
            reasoning_effort: Some("xhigh"),
            enable_thinking: None,
            prompt_tokens: 53,
            message_boundary_tokens: 48,
        },
        PromptFixture {
            name: "hello-low",
            messages: json!([{"role": "user", "content": "Hello"}]),
            reasoning_effort: Some("low"),
            enable_thinking: None,
            prompt_tokens: 41,
            message_boundary_tokens: 36,
        },
        PromptFixture {
            name: "hello-no-thinking",
            messages: json!([{"role": "user", "content": "Hello"}]),
            reasoning_effort: None,
            enable_thinking: Some(false),
            prompt_tokens: 13,
            message_boundary_tokens: 6,
        },
        PromptFixture {
            name: "unicode",
            messages: json!([{"role": "user", "content": "Hei \u{1f680} \u{5317}"}]),
            reasoning_effort: None,
            enable_thinking: None,
            prompt_tokens: 17,
            message_boundary_tokens: 12,
        },
        PromptFixture {
            name: "multi-turn",
            messages: json!([
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "First question"},
                {
                    "role": "assistant",
                    "content": "First answer",
                    "reasoning_content": "earlier thought"
                },
                {"role": "user", "content": "Second question"}
            ]),
            reasoning_effort: None,
            enable_thinking: None,
            prompt_tokens: 43,
            message_boundary_tokens: 38,
        },
    ]
}

impl PromptFixture {
    fn request(&self, stream: bool, max_completion_tokens: usize) -> Value {
        let mut kwargs = serde_json::Map::new();
        if let Some(effort) = self.reasoning_effort {
            kwargs.insert("reasoning_effort".into(), json!(effort));
        }
        if let Some(enable) = self.enable_thinking {
            kwargs.insert("enable_thinking".into(), json!(enable));
        }
        let mut request = json!({
            "model": Qwen38FlashNext::MODEL_ID,
            "messages": self.messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 0,
            "stream": stream,
            "chat_template_kwargs": Value::Object(kwargs),
        });
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        request
    }
}

fn hello(stream: bool, max_completion_tokens: usize) -> Value {
    fixtures()[0].request(stream, max_completion_tokens)
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
        expect_status(&mut response, 200, path)?;
        Ok(response.body_mut().read_json()?)
    }

    fn blocking(&self, request: Value, label: &str) -> Result<Completion> {
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request)?;
        expect_status(&mut response, 200, label)?;
        validate_blocking(&response.body_mut().read_json()?)
    }

    /// Posts a stream and times every event at the caller's socket.
    fn streaming(&self, request: Value, label: &str) -> Result<StreamTiming> {
        let started = Instant::now();
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request)?;
        expect_status(&mut response, 200, label)?;
        expect_event_stream(&response)?;

        let (_, body) = response.into_parts();
        let mut reader = BufReader::new(body.into_reader());
        let mut line = String::new();
        let mut events = Vec::new();
        let mut arrivals = Vec::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let Some(data) = line.trim_end().strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                events.push(None);
                break;
            }
            events.push(Some(serde_json::from_str::<Value>(data)?));
            arrivals.push(started.elapsed());
        }
        let wall = started.elapsed();
        let timing = validate_stream(&events, label)?;
        // The role preamble is not a generated token.
        let mut deltas = arrivals
            .iter()
            .copied()
            .zip(events.iter())
            .filter(|(_, event)| {
                event.as_ref().is_some_and(|event| {
                    let delta = &event["choices"][0]["delta"];
                    delta.get("content").is_some() || delta.get("reasoning_content").is_some()
                })
            })
            .map(|(arrival, _)| arrival)
            .collect::<Vec<_>>();
        deltas.sort_unstable();
        let time_to_first_token = deltas.first().copied().unwrap_or(wall);
        let inter_token = deltas.windows(2).map(|pair| pair[1] - pair[0]).collect();

        Ok(StreamTiming {
            wall,
            time_to_first_token,
            inter_token,
            ..timing
        })
    }

    fn expect_rejection(&self, request: Value, status: u16, label: &str) -> Result<String> {
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .send_json(request)?;
        expect_status(&mut response, status, label)?;
        let body: Value = response.body_mut().read_json()?;
        let error = object(field(&body, "error")?, "error envelope")?;
        Ok(string(
            error
                .get("message")
                .ok_or_else(|| missing("error.message"))?,
            "error message",
        )?
        .to_owned())
    }

    /// Opens a stream, reads its first data event, and drops the socket underneath it.
    fn disconnect_stream(&self) -> Result<()> {
        let mut response = self
            .agent
            .post(self.url("/v1/chat/completions"))
            .header("connection", "close")
            .send_json(hello(true, CANCEL_COMPLETION_TOKENS))?;
        expect_status(&mut response, 200, "cancellation stream")?;
        expect_event_stream(&response)?;

        let (_, body) = response.into_parts();
        let mut reader = BufReader::new(body.into_reader());
        let mut line = String::new();
        let mut seen = 0usize;
        while seen < 3 {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(QualError::Contract(
                    "cancellation stream ended before it produced any tokens".into(),
                ));
            }
            if line.trim_end().starts_with("data: ") {
                seen += 1;
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
            requests: 0,
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

    fn run(&mut self) -> Result<()> {
        self.run_surface()?;
        let reference = self.run_reasoning_default()?;
        self.run_parity()?;
        self.run_refusals(&reference)?;
        self.run_concurrency(&reference)?;
        self.run_batch_independence()?;
        self.run_backpressure()?;
        self.run_cancellation(&reference)?;
        self.run_sibling_models(&reference)
    }

    fn run_surface(&mut self) -> Result<()> {
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
                        "id": Qwen38FlashNext::MODEL_ID,
                        "object": "model",
                        "owned_by": "tuiskollm"
                    }]
                }),
            format!("unexpected /v1/models response: {models}"),
        )
    }

    /// Checks that the served default is the zero-preamble budget.
    fn run_reasoning_default(&mut self) -> Result<Completion> {
        let mut counts = Vec::new();
        for fixture in fixtures() {
            let completion = self.request(fixture.request(false, 1), fixture.name)?;
            self.check(
                completion.usage.prompt_tokens == fixture.prompt_tokens,
                format!(
                    "fixture `{}` rendered {} prompt tokens, the committed frontend fixture is {}",
                    fixture.name, completion.usage.prompt_tokens, fixture.prompt_tokens
                ),
            )?;
            counts.push((fixture.name, completion.usage.prompt_tokens));
        }
        let named = |name: &str| {
            counts
                .iter()
                .find(|(fixture, _)| *fixture == name)
                .map(|(_, tokens)| *tokens)
        };
        let (Some(default), Some(medium), Some(xhigh), Some(low)) = (
            named("hello-default"),
            named("hello-medium"),
            named("hello-xhigh"),
            named("hello-low"),
        ) else {
            return Err(QualError::Contract(
                "the reasoning-effort sweep lost one of its fixtures".into(),
            ));
        };
        self.check(
            default == medium,
            format!(
                "a request naming no reasoning budget rendered {default} prompt tokens; the served default renders {medium}"
            ),
        )?;
        self.check(
            xhigh > default && low > default,
            format!(
                "the served default rendered {default} prompt tokens against xhigh {xhigh} and low {low}; no budget is cheaper than the default, so the default is not the zero-preamble one"
            ),
        )?;
        self.check(
            xhigh - default == 42,
            format!(
                "the checkpoint's own `xhigh` default costs {} preamble tokens over the served default, expected 42",
                xhigh - default
            ),
        )?;

        self.request(hello(false, COMPLETION_TOKENS), "reference completion")
    }

    /// Checks blocking and streaming parity for every prompt fixture.
    fn run_parity(&mut self) -> Result<()> {
        for fixture in fixtures() {
            let blocking = self.request(
                fixture.request(false, COMPLETION_TOKENS),
                &format!("{} blocking", fixture.name),
            )?;
            let streamed = self.stream(
                fixture.request(true, COMPLETION_TOKENS),
                &format!("{} streaming", fixture.name),
            )?;
            self.check(
                same_completion_semantics(&streamed.completion, &blocking),
                format!(
                    "fixture `{}` differs between transports: blocking={blocking:?}, streaming={:?}",
                    fixture.name, streamed.completion
                ),
            )?;
            self.check(
                streamed.completion.usage.cached_tokens == fixture.message_boundary_tokens,
                format!(
                    "fixture `{}` reused {} prompt tokens, expected its {}-token message boundary",
                    fixture.name,
                    streamed.completion.usage.cached_tokens,
                    fixture.message_boundary_tokens,
                ),
            )?;
            self.check(
                blocking.usage.total_tokens
                    == blocking.usage.prompt_tokens + blocking.usage.completion_tokens
                    && blocking.usage.completion_tokens <= COMPLETION_TOKENS
                    && blocking.usage.cached_tokens <= blocking.usage.prompt_tokens,
                format!(
                    "fixture `{}` reported inconsistent usage {:?}",
                    fixture.name, blocking.usage
                ),
            )?;
        }
        Ok(())
    }

    /// Checks refusal boundaries without accepting truncation or state mutation.
    fn run_refusals(&mut self, reference: &Completion) -> Result<()> {
        let message = self.reject(
            exact_length_request(64, SERVED_DEPTH),
            400,
            "over-depth completion",
        )?;
        let (required, capacity) = parse_capacity_refusal(&message)?;
        self.check(
            capacity == SERVED_DEPTH,
            format!("the over-depth refusal named {capacity}, expected {SERVED_DEPTH}"),
        )?;
        self.check(
            required > capacity,
            format!("the over-depth refusal required {required} positions inside {capacity}"),
        )?;

        let selected = self.request(
            long_prompt_request(DENSE_BAND + 160),
            "selected long completion",
        )?;
        self.check(
            selected.usage.prompt_tokens > DENSE_BAND && selected.usage.completion_tokens == 1,
            format!("a selected long prompt reported {:?}", selected.usage),
        )?;

        let inside = self.request(
            exact_length_request(DENSE_BAND - 64, 1),
            "in-band long completion",
        )?;
        self.check(
            inside.usage.prompt_tokens <= DENSE_BAND && inside.usage.completion_tokens == 1,
            format!("an in-band long prompt reported {:?}", inside.usage),
        )?;

        for (label, request, status) in [
            (
                "zero max_completion_tokens",
                json!({
                    "model": Qwen38FlashNext::MODEL_ID,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": 0
                }),
                400,
            ),
            (
                "empty messages",
                json!({"model": Qwen38FlashNext::MODEL_ID, "messages": []}),
                400,
            ),
            (
                "missing user content",
                json!({
                    "model": Qwen38FlashNext::MODEL_ID,
                    "messages": [{"role": "user"}]
                }),
                400,
            ),
            (
                "unknown reasoning budget",
                json!({
                    "model": Qwen38FlashNext::MODEL_ID,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "chat_template_kwargs": {"reasoning_effort": "ultra"}
                }),
                400,
            ),
            (
                "unknown request field",
                json!({
                    "model": Qwen38FlashNext::MODEL_ID,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "nonsense": 1
                }),
                400,
            ),
            (
                "custom stop sequence",
                json!({
                    "model": Qwen38FlashNext::MODEL_ID,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stop": "done"
                }),
                400,
            ),
        ] {
            self.reject(request, status, label)?;
            self.checks += 1;
        }

        let recovery = self.request(hello(false, COMPLETION_TOKENS), "post-refusal completion")?;
        self.check(
            same_completion_semantics(&recovery, reference),
            format!("the served answer changed after the refusal sweep: {recovery:?}"),
        )
    }

    /// Checks concurrent copies preserve the single-request answer.
    fn run_concurrency(&mut self, reference: &Completion) -> Result<()> {
        for lanes in [1usize, 2, 4, 8] {
            let completions = concurrent(self.client.clone(), lanes)?;
            self.requests += lanes;
            for (lane, completion) in completions.iter().enumerate() {
                self.check(
                    same_completion_semantics(completion, reference),
                    format!(
                        "lane {lane} of {lanes} concurrent callers changed the answer: {completion:?}"
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Checks each distinct prompt still produces its solo answer in one batch.
    fn run_batch_independence(&mut self) -> Result<()> {
        let prompts = independence_prompts();
        let mut alone = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            alone.push(self.request(independence_request(prompt), prompt)?);
        }

        let batched = concurrent_prompts(self.client.clone(), &prompts)?;
        self.requests += prompts.len();
        for ((prompt, expected), actual) in prompts.iter().zip(&alone).zip(&batched) {
            self.check(
                same_completion_semantics(actual, expected),
                format!(
                    "`{prompt}` changed under {}-way batching: alone={expected:?}, batched={actual:?}",
                    prompts.len(),
                ),
            )?;
        }
        Ok(())
    }

    /// Checks that transport ingress remains bounded.
    fn run_backpressure(&mut self) -> Result<()> {
        let lanes = FUNDED_SLOTS + INGRESS_QUEUE + 6;
        let statuses = concurrent_statuses(self.client.clone(), lanes)?;
        self.requests += lanes;
        let admitted = statuses.iter().filter(|&&status| status == 200).count();
        let refused = statuses.iter().filter(|&&status| status == 429).count();
        self.check(
            admitted + refused == lanes,
            format!("bounded ingress returned statuses outside 200/429: {statuses:?}"),
        )?;
        self.check(
            refused > 0,
            format!(
                "{lanes} simultaneous callers against {FUNDED_SLOTS} slots and a {INGRESS_QUEUE}-deep queue produced no 429; ingress is not bounded"
            ),
        )?;
        self.check(
            admitted <= FUNDED_SLOTS + INGRESS_QUEUE,
            format!(
                "bounded ingress admitted {admitted} callers, above the funded slots plus their {INGRESS_QUEUE}-deep queue"
            ),
        )
    }

    /// Checks that cancellation restores the admitted message boundary.
    fn run_cancellation(&mut self, reference: &Completion) -> Result<()> {
        for round in 0..3 {
            self.client.disconnect_stream()?;
            self.requests += 1;
            self.checks += 1;
            let health = self.client.get_json("/health")?;
            self.check(
                health == json!({"status": "ok", "generation_route": GENERATION_ROUTE}),
                format!("the server was not healthy after cancellation round {round}: {health}"),
            )?;
            let recovery = self.request(
                hello(false, COMPLETION_TOKENS),
                &format!("post-cancellation completion {round}"),
            )?;
            self.check(
                same_completion_semantics(&recovery, reference),
                format!(
                    "the served answer changed after cancellation round {round}: expected={reference:?}, actual={recovery:?}"
                ),
            )?;
            self.check(
                recovery.usage.cached_tokens == HELLO_MESSAGE_BOUNDARY_TOKENS,
                format!(
                    "cancellation round {round} restored {} cached tokens, expected {HELLO_MESSAGE_BOUNDARY_TOKENS}",
                    recovery.usage.cached_tokens,
                ),
            )?;
        }
        Ok(())
    }

    /// Checks sibling IDs are refused without consuming served capacity.
    fn run_sibling_models(&mut self, reference: &Completion) -> Result<()> {
        for sibling in [
            Qwen38_27B::MODEL_ID,
            Qwen35_9B::MODEL_ID,
            Qwen36Moe35B::MODEL_ID,
            "RadixArk/Qwen3.8-Flash-Next",
        ] {
            let mut request = hello(false, COMPLETION_TOKENS);
            request["model"] = json!(sibling);
            let message = self.reject(request, 404, &format!("sibling model {sibling}"))?;
            self.check(
                message.contains(sibling) && message.contains("not served by this process"),
                format!("the refusal for sibling `{sibling}` was {message:?}"),
            )?;
        }

        let after = self.request(hello(false, COMPLETION_TOKENS), "post-sibling completion")?;
        self.check(
            same_completion_semantics(&after, reference),
            format!("the served answer changed after the sibling-model sweep: {after:?}"),
        )?;

        // Interleave refusals to catch accidental worker admission.
        for round in 0..4 {
            let mut request = hello(false, COMPLETION_TOKENS);
            request["model"] = json!(Qwen36Moe35B::MODEL_ID);
            self.reject(request, 404, "interleaved sibling")?;
            let served = self.request(
                hello(false, COMPLETION_TOKENS),
                &format!("interleaved completion {round}"),
            )?;
            self.check(
                same_completion_semantics(&served, reference),
                format!("interleaving sibling refusals changed round {round}: {served:?}"),
            )?;
        }
        Ok(())
    }

    /// Reports diagnostic real-HTTP timing without blessing a baseline.
    fn measure(&mut self) -> Result<()> {
        self.run_surface()?;
        println!("# Flash-Next served-model measurements (diagnostic, nothing blessed)");
        println!(
            "# route {GENERATION_ROUTE}, {FUNDED_SLOTS} funded slots, served depth {SERVED_DEPTH}"
        );
        println!();
        println!("## Per-request wall time and token consumption (SSE, greedy)");
        println!(
            "Each prompt is measured cold, then immediately warm. Both include expert streaming."
        );
        println!(
            "| prompt | pass | budget | wall | TTFT | ITL median | ITL p90 | prompt tok | cached tok | completion tok | reasoning deltas | content deltas | finish |"
        );
        println!("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |");
        for (label, request, budget) in measurement_requests() {
            let mut cold = None;
            for pass in ["cold", "warm"] {
                let timing = self.stream(request.clone(), &label)?;
                if let Some(cold) = &cold {
                    self.check(
                        same_completion_semantics(&timing.completion, cold),
                        format!("`{label}` changed between cold and warm measurement passes"),
                    )?;
                    self.check(
                        timing.completion.usage.cached_tokens > 0,
                        format!("`{label}` warm measurement reused no prompt tokens"),
                    )?;
                } else {
                    self.check(
                        timing.completion.usage.cached_tokens == 0,
                        format!(
                            "`{label}` cold measurement reused {} prompt tokens",
                            timing.completion.usage.cached_tokens
                        ),
                    )?;
                    cold = Some(timing.completion.clone());
                }
                let (median, p90) = quantiles(&timing.inter_token);
                println!(
                    "| {label} | {pass} | {budget} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    millis(timing.wall),
                    millis(timing.time_to_first_token),
                    millis(median),
                    millis(p90),
                    timing.completion.usage.prompt_tokens,
                    timing.completion.usage.cached_tokens,
                    timing.completion.usage.completion_tokens,
                    timing.reasoning_deltas,
                    timing.content_deltas,
                    timing.completion.finish_reason,
                );
            }
        }

        println!();
        println!("## Throughput across the funded slots");
        println!(
            "| budget | concurrency | wall | completion tok | aggregate tok/s | per-stream tok/s (min / median / max) | TTFT p50 | TTFT p95 | ITL median |"
        );
        println!("| --: | --: | --: | --: | --: | :-- | --: | --: | --: |");
        let warmup = concurrent(self.client.clone(), 1)?;
        self.requests += warmup.len();
        for budget in [COMPLETION_TOKENS, THROUGHPUT_COMPLETION_TOKENS] {
            for lanes in [1usize, 2, 4, 8] {
                let started = Instant::now();
                let timings = concurrent_streams(self.client.clone(), lanes, budget)?;
                let wall = started.elapsed();
                self.requests += lanes;
                let tokens = timings
                    .iter()
                    .map(|timing| timing.completion.usage.completion_tokens)
                    .sum::<usize>();
                let mut rates = timings
                    .iter()
                    .map(|timing| {
                        timing.completion.usage.completion_tokens as f64 / timing.wall.as_secs_f64()
                    })
                    .collect::<Vec<_>>();
                rates.sort_by(f64::total_cmp);
                let mut first_token = timings
                    .iter()
                    .map(|timing| timing.time_to_first_token)
                    .collect::<Vec<_>>();
                first_token.sort_unstable();
                let mut inter_token = timings
                    .iter()
                    .map(|timing| quantiles(&timing.inter_token).0)
                    .collect::<Vec<_>>();
                inter_token.sort_unstable();
                println!(
                    "| {budget} | {lanes} | {} | {tokens} | {:.2} | {:.2} / {:.2} / {:.2} | {} | {} | {} |",
                    millis(wall),
                    tokens as f64 / wall.as_secs_f64(),
                    rates.first().copied().unwrap_or_default(),
                    median_of(&rates),
                    rates.last().copied().unwrap_or_default(),
                    millis(first_token[percentile_index(first_token.len(), 50)]),
                    millis(first_token[percentile_index(first_token.len(), 95)]),
                    millis(duration_median(&inter_token)),
                );
            }
        }

        println!();
        println!("## Reasoning-token consumption at the served `medium` default");
        println!(
            "`reasoning deltas` and `content deltas` are streamed events in each channel, so they \
             count generated tokens that emitted text. `unattributed` is the rest of the exact \
             `completion_tokens` the usage chunk reports: the `</think>` boundary, and any token \
             that completed no character on its own."
        );
        println!(
            "| prompt | prompt tok | completion tok | reasoning deltas | content deltas | unattributed | finish |"
        );
        println!("| :-- | --: | --: | --: | --: | --: | :-- |");
        for prompt in [
            "Reply with exactly the word blue.",
            "What is 17 times 23?",
            "Name the capital of Finland.",
        ] {
            let mut request = hello(true, 512);
            request["messages"] = json!([{"role": "user", "content": prompt}]);
            let timing = self.stream(request, prompt)?;
            let attributed = timing.reasoning_deltas + timing.content_deltas;
            println!(
                "| {prompt} | {} | {} | {} | {} | {} | {} |",
                timing.completion.usage.prompt_tokens,
                timing.completion.usage.completion_tokens,
                timing.reasoning_deltas,
                timing.content_deltas,
                timing
                    .completion
                    .usage
                    .completion_tokens
                    .saturating_sub(attributed),
                timing.completion.finish_reason,
            );
        }
        Ok(())
    }

    fn request(&mut self, request: Value, label: &str) -> Result<Completion> {
        self.requests += 1;
        self.client.blocking(request, label)
    }

    fn stream(&mut self, request: Value, label: &str) -> Result<StreamTiming> {
        self.requests += 1;
        self.client.streaming(request, label)
    }

    fn reject(&mut self, request: Value, status: u16, label: &str) -> Result<String> {
        self.requests += 1;
        self.client.expect_rejection(request, status, label)
    }
}

/// Measurement prompts ordered by length.
fn measurement_requests() -> Vec<(String, Value, usize)> {
    let budgets = [
        ("Hello", 32usize),
        ("Reply with exactly the word blue.", 32),
        ("Summarise the water cycle in one sentence.", 64),
        ("List three prime numbers above one hundred.", 64),
    ];
    let mut requests = budgets
        .into_iter()
        .map(|(prompt, budget)| {
            let mut request = hello(true, budget);
            request["messages"] = json!([{"role": "user", "content": prompt}]);
            (prompt.to_owned(), request, budget)
        })
        .collect::<Vec<_>>();
    // Descending lengths keep a retained repeated-word prompt from becoming the
    // next case's cold prefix authority.
    for tokens in [DENSE_BAND - 32, 1_024, 512] {
        requests.push((
            format!("{tokens}-token prompt"),
            exact_length_request_streaming(tokens, 32),
            32,
        ));
    }
    requests
}

/// Builds a prompt near `tokens`; qualification asserts only its side of the dense boundary.
fn long_prompt_body(tokens: usize) -> String {
    format!("begin.{}", " blue".repeat(tokens.saturating_sub(12)))
}

fn long_prompt_request(tokens: usize) -> Value {
    exact_length_request(tokens, 1)
}

fn exact_length_request(tokens: usize, budget: usize) -> Value {
    let mut request = hello(false, budget);
    request["messages"] = json!([{"role": "user", "content": long_prompt_body(tokens)}]);
    request
}

fn exact_length_request_streaming(tokens: usize, budget: usize) -> Value {
    let mut request = hello(true, budget);
    request["messages"] = json!([{"role": "user", "content": long_prompt_body(tokens)}]);
    request
}

fn concurrent(client: Client, count: usize) -> Result<Vec<Completion>> {
    concurrent_prompts(client, &vec!["Hello".to_owned(); count])
}

fn concurrent_streams(client: Client, count: usize, budget: usize) -> Result<Vec<StreamTiming>> {
    let barrier = Arc::new(Barrier::new(count));
    let handles = (0..count)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let client = client.clone();
            thread::spawn(move || {
                barrier.wait();
                client.streaming(hello(true, budget), "concurrent stream")
            })
        })
        .collect::<Vec<_>>();
    handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| QualError::ThreadPanic)?)
        .collect()
}

fn concurrent_prompts(client: Client, prompts: &[String]) -> Result<Vec<Completion>> {
    let barrier = Arc::new(Barrier::new(prompts.len()));
    let handles = prompts
        .iter()
        .map(|prompt| {
            let barrier = Arc::clone(&barrier);
            let client = client.clone();
            let prompt = prompt.clone();
            thread::spawn(move || {
                let request = independence_request(&prompt);
                barrier.wait();
                client.blocking(request, "concurrent completion")
            })
        })
        .collect::<Vec<_>>();
    handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| QualError::ThreadPanic)?)
        .collect()
}

fn independence_prompts() -> Vec<String> {
    [
        "Name one primary color.",
        "Say hello.",
        "Describe a river in one sentence.",
        "What is two plus two?",
        "List three fruits, separated by commas.",
        "Give one fact about the moon.",
        "Write one short sentence about rain.",
        "Name the largest ocean.",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn independence_request(prompt: &str) -> Value {
    let mut request = hello(false, COMPLETION_TOKENS);
    request["messages"] = json!([{"role": "user", "content": prompt}]);
    request
}

fn concurrent_statuses(client: Client, count: usize) -> Result<Vec<u16>> {
    let barrier = Arc::new(Barrier::new(count));
    let handles = (0..count)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let client = client.clone();
            thread::spawn(move || -> Result<u16> {
                barrier.wait();
                let mut response = client
                    .agent
                    .post(client.url("/v1/chat/completions"))
                    .send_json(hello(false, 32))?;
                let status = response.status().as_u16();
                let _ = response.body_mut().read_to_string();
                Ok(status)
            })
        })
        .collect::<Vec<_>>();
    handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| QualError::ThreadPanic)?)
        .collect()
}

fn quantiles(samples: &[Duration]) -> (Duration, Duration) {
    if samples.is_empty() {
        return (Duration::ZERO, Duration::ZERO);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let median = duration_median(&sorted);
    let p90_index = sorted.len().saturating_mul(9).div_ceil(10) - 1;
    let p90 = sorted[p90_index];
    (median, p90)
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    if len == 0 {
        return 0;
    }

    (len * percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(len - 1)
}

fn duration_median(sorted: &[Duration]) -> Duration {
    match sorted.len() {
        0 => Duration::ZERO,
        len if len % 2 == 0 => (sorted[len / 2 - 1] + sorted[len / 2]) / 2,
        len => sorted[len / 2],
    }
}

fn median_of(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        len if len % 2 == 0 => (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0,
        len => sorted[len / 2],
    }
}

fn millis(duration: Duration) -> String {
    format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
}

fn parse_capacity_refusal(message: &str) -> Result<(usize, usize)> {
    let body = message
        .strip_prefix("[engine.generation] prompt plus processed generation requires ")
        .ok_or_else(|| {
            QualError::Contract(format!(
                "the over-band refusal has an unexpected message {message:?}"
            ))
        })?;
    let (required, capacity) = body
        .split_once(" positions, current resident capacity is ")
        .ok_or_else(|| QualError::Contract("the over-band refusal omitted its limits".into()))?;
    let required = required.parse().map_err(|error| {
        QualError::Contract(format!(
            "required positions {required:?} are invalid: {error}"
        ))
    })?;
    let capacity = capacity.parse().map_err(|error| {
        QualError::Contract(format!(
            "resident capacity {capacity:?} is invalid: {error}"
        ))
    })?;
    Ok((required, capacity))
}

fn validate_blocking(body: &Value) -> Result<Completion> {
    expect_common(body, "chat.completion")?;
    let choices = array(field(body, "choices")?, "blocking choices")?;
    if choices.len() != 1 {
        return Err(QualError::Contract(format!(
            "the blocking response has {} choices, expected one",
            choices.len()
        )));
    }
    let message = object(field(&choices[0], "message")?, "blocking message")?;
    expect_str_value(message.get("role"), "blocking role", "assistant")?;

    Ok(Completion {
        reasoning: message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        content: string(
            message
                .get("content")
                .ok_or_else(|| missing("message.content"))?,
            "blocking content",
        )?
        .to_owned(),
        finish_reason: finish_reason(field(&choices[0], "finish_reason")?)?,
        usage: parse_usage(field(body, "usage")?)?,
    })
}

fn validate_stream(events: &[Option<Value>], label: &str) -> Result<StreamTiming> {
    if !matches!(events.last(), Some(None)) {
        return Err(QualError::Contract(format!(
            "{label} did not end in data: [DONE]"
        )));
    }
    let values = events[..events.len() - 1]
        .iter()
        .map(|event| {
            event
                .as_ref()
                .ok_or_else(|| QualError::Contract(format!("{label} emitted [DONE] early")))
        })
        .collect::<Result<Vec<_>>>()?;
    let first = values
        .first()
        .ok_or_else(|| QualError::Contract(format!("{label} emitted no events")))?;
    expect_common(first, "chat.completion.chunk")?;
    expect_str_value(
        first["choices"][0]["delta"].get("role"),
        "first stream role",
        "assistant",
    )?;

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut reasoning_deltas = 0usize;
    let mut content_deltas = 0usize;
    let mut terminal = None;
    let mut usage = None;
    for value in values.iter().skip(1) {
        expect_common(value, "chat.completion.chunk")?;
        let choices = array(field(value, "choices")?, "stream choices")?;
        if choices.is_empty() {
            usage = Some(parse_usage(field(value, "usage")?)?);
            continue;
        }
        let delta = object(field(&choices[0], "delta")?, "stream delta")?;
        if let Some(chunk) = delta.get("reasoning_content") {
            reasoning.push_str(string(chunk, "stream reasoning delta")?);
            reasoning_deltas += 1;
        }
        if let Some(chunk) = delta.get("content") {
            content.push_str(string(chunk, "stream content delta")?);
            content_deltas += 1;
        }
        let reason = field(&choices[0], "finish_reason")?;
        if !reason.is_null() {
            if terminal.is_some() {
                return Err(QualError::Contract(format!(
                    "{label} emitted more than one terminal event"
                )));
            }
            terminal = Some(finish_reason(reason)?);
        }
    }

    Ok(StreamTiming {
        completion: Completion {
            reasoning,
            content,
            finish_reason: terminal.ok_or_else(|| {
                QualError::Contract(format!("{label} has no terminal finish reason"))
            })?,
            usage: usage
                .ok_or_else(|| QualError::Contract(format!("{label} has no usage event")))?,
        },
        wall: Duration::ZERO,
        time_to_first_token: Duration::ZERO,
        inter_token: Vec::new(),
        reasoning_deltas,
        content_deltas,
    })
}

fn expect_common(value: &Value, object_kind: &str) -> Result<()> {
    let id = string(field(value, "id")?, "completion id")?;
    if !id.starts_with("chatcmpl-tuisko-") {
        return Err(QualError::Contract(format!(
            "the completion id {id:?} lacks the Tuisko prefix"
        )));
    }
    expect_str_value(value.get("object"), "response object", object_kind)?;
    expect_str_value(
        value.get("model"),
        "response model",
        Qwen38FlashNext::MODEL_ID,
    )?;
    if integer(field(value, "created")?, "created timestamp")? == 0 {
        return Err(QualError::Contract("the created timestamp is zero".into()));
    }
    Ok(())
}

fn parse_usage(value: &Value) -> Result<Usage> {
    let prompt_tokens = usize_value(field(value, "prompt_tokens")?, "prompt tokens")?;
    let cached_tokens = usize_value(
        field(field(value, "prompt_tokens_details")?, "cached_tokens")?,
        "cached prompt tokens",
    )?;
    let completion_tokens = usize_value(field(value, "completion_tokens")?, "completion tokens")?;
    let total_tokens = usize_value(field(value, "total_tokens")?, "total tokens")?;
    if prompt_tokens == 0 || completion_tokens == 0 {
        return Err(QualError::Contract(format!(
            "usage reports {prompt_tokens} prompt and {completion_tokens} completion tokens"
        )));
    }
    if total_tokens != prompt_tokens + completion_tokens {
        return Err(QualError::Contract(format!(
            "usage total {total_tokens} is not prompt plus completion {}",
            prompt_tokens + completion_tokens
        )));
    }
    if cached_tokens > prompt_tokens {
        return Err(QualError::Contract(format!(
            "usage reports {cached_tokens} cached tokens against {prompt_tokens} prompt tokens"
        )));
    }
    Ok(Usage {
        prompt_tokens,
        cached_tokens,
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

fn expect_status(
    response: &mut ureq::http::Response<ureq::Body>,
    expected: u16,
    label: &str,
) -> Result<()> {
    let actual = response.status().as_u16();
    if actual == expected {
        return Ok(());
    }
    let body = response
        .body_mut()
        .read_to_string()
        .unwrap_or_else(|error| format!("<response body read failed: {error}>"));
    Err(QualError::Contract(format!(
        "{label} returned HTTP {actual}, expected {expected}: {body}"
    )))
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
            "the stream response content-type is {content_type:?}, expected text/event-stream"
        )))
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value.get(name).ok_or_else(|| missing(name))
}

fn missing(name: &str) -> QualError {
    QualError::Contract(format!("the response is missing {name}"))
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
        DENSE_BAND, HELLO_MESSAGE_BOUNDARY_TOKENS, Options, QualError, SERVED_DEPTH, fixtures,
        long_prompt_body, measurement_requests, median_of, parse_capacity_refusal,
        percentile_index, quantiles, same_completion_semantics, validate_blocking, validate_stream,
    };
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn the_default_fixture_is_the_zero_preamble_one_and_the_others_are_not() {
        let by_name = |name: &str| {
            fixtures()
                .into_iter()
                .find(|fixture| fixture.name == name)
                .map(|fixture| fixture.prompt_tokens)
                .unwrap()
        };
        assert_eq!(by_name("hello-default"), by_name("hello-medium"));
        assert_eq!(by_name("hello-xhigh") - by_name("hello-default"), 42);
        assert!(by_name("hello-low") > by_name("hello-default"));
        assert!(by_name("hello-no-thinking") > by_name("hello-default"));
    }

    #[test]
    fn options_default_and_accept_the_measurement_mode_in_either_order() {
        assert_eq!(
            Options::parse(["qual".into()]).unwrap().base_url,
            "http://127.0.0.1:8000"
        );
        for arguments in [
            ["http://server:9000/", "--measure"],
            ["--measure", "http://server:9000/"],
        ] {
            let options = Options::parse(
                std::iter::once("qual".to_owned()).chain(arguments.into_iter().map(str::to_owned)),
            )
            .unwrap();
            assert_eq!(options.base_url, "http://server:9000");
            assert!(matches!(options.mode, super::Mode::Measure));
        }
        assert!(Options::parse(["qual".into(), "--measure".into(), "--measure".into()]).is_err());
    }

    #[test]
    fn the_over_depth_refusal_is_parsed_for_both_of_its_numbers() {
        let message = format!(
            "[engine.generation] prompt plus processed generation requires 262208 positions, current resident capacity is {SERVED_DEPTH}"
        );
        assert_eq!(
            parse_capacity_refusal(&message).unwrap(),
            (262_208, 262_144)
        );
        let error = parse_capacity_refusal("something else").unwrap_err();
        assert!(matches!(error, QualError::Contract(_)));
    }

    #[test]
    fn blocking_and_streamed_fixtures_decode_to_the_same_completion() {
        let blocking_usage = json!({
            "prompt_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens": 2,
            "total_tokens": 13
        });
        let streamed_usage = json!({
            "prompt_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": HELLO_MESSAGE_BOUNDARY_TOKENS},
            "completion_tokens": 2,
            "total_tokens": 13
        });
        let blocking = validate_blocking(&json!({
            "id": "chatcmpl-tuisko-0001",
            "object": "chat.completion",
            "created": 1,
            "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "The user"
                },
                "finish_reason": "length"
            }],
            "usage": blocking_usage
        }))
        .unwrap();

        let chunk = |delta: serde_json::Value, finish: serde_json::Value| {
            Some(json!({
                "id": "chatcmpl-tuisko-0002",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            }))
        };
        let streamed = validate_stream(
            &[
                chunk(json!({"role": "assistant"}), json!(null)),
                chunk(json!({"reasoning_content": "The"}), json!(null)),
                chunk(json!({"reasoning_content": " user"}), json!(null)),
                chunk(json!({}), json!("length")),
                Some(json!({
                    "id": "chatcmpl-tuisko-0002",
                    "object": "chat.completion.chunk",
                    "created": 2,
                    "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                    "choices": [],
                    "usage": streamed_usage
                })),
                None,
            ],
            "fixture stream",
        )
        .unwrap();

        assert_ne!(streamed.completion, blocking);
        assert!(same_completion_semantics(&streamed.completion, &blocking));
        let mut changed = streamed.completion.clone();
        changed.reasoning.push('!');
        assert!(!same_completion_semantics(&changed, &blocking));
        assert_eq!(streamed.reasoning_deltas, 2);
        assert_eq!(streamed.content_deltas, 0);
        assert!(validate_stream(&[chunk(json!({}), json!(null))], "no DONE").is_err());
    }

    #[test]
    fn the_long_prompt_body_grows_one_token_at_a_time() {
        assert!(long_prompt_body(20).ends_with(" blue"));
        assert_eq!(long_prompt_body(20).matches(" blue").count(), 8);
        assert_eq!(long_prompt_body(4).matches(" blue").count(), 0);
    }

    #[test]
    fn measured_long_prompts_descend_and_fit_the_served_capacity() {
        let long = measurement_requests()
            .into_iter()
            .filter_map(|(label, _, budget)| {
                label
                    .strip_suffix("-token prompt")
                    .and_then(|tokens| tokens.parse::<usize>().ok())
                    .map(|tokens| (tokens, budget))
            })
            .collect::<Vec<_>>();

        assert_eq!(long, [(DENSE_BAND - 32, 32), (1_024, 32), (512, 32)]);
        assert!(
            long.iter()
                .all(|&(prompt, budget)| prompt + budget <= DENSE_BAND)
        );
    }

    #[test]
    fn burst_percentiles_use_nearest_rank() {
        assert_eq!(percentile_index(0, 95), 0);
        for lanes in [1usize, 2, 4, 8] {
            assert_eq!(percentile_index(lanes, 95), lanes - 1);
        }
        assert_eq!(percentile_index(100, 95), 94);
        assert_eq!(percentile_index(8, 50), 3);
    }

    #[test]
    fn quantiles_of_an_empty_sample_are_zero_rather_than_a_panic() {
        assert_eq!(quantiles(&[]), (Duration::ZERO, Duration::ZERO));
        let samples = (1..=10).map(Duration::from_millis).collect::<Vec<_>>();
        let (median, p90) = quantiles(&samples);
        assert_eq!(
            median,
            Duration::from_millis(5) + Duration::from_micros(500)
        );
        assert_eq!(p90, Duration::from_millis(9));
        assert_eq!(median_of(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_of(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_of(&[]), 0.0);
    }
}
