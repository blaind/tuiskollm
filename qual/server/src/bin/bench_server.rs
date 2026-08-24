//! Direct external timing for the exact TuiskoLLM production HTTP boundary.

use serde::Serialize;
use serde_json::{Value, json};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;
use tuisko_model::{Arch, Qwen38_27B};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const STREAM_COMPLETION_TOKENS: usize = 32;
const CONCURRENT_COMPLETION_TOKENS: usize = 8;
const LONG_COMPLETION_TOKENS: usize = 4;
const DEFAULT_SAMPLES: usize = 5;
const LONG_CONTEXTS: [usize; 4] = [4_096, 16_384, 65_536, 178_000];

type Result<T> = std::result::Result<T, BenchError>;

#[derive(Debug, Error)]
enum BenchError {
    #[error("{0}")]
    Contract(String),
    #[error("HTTP transport failed: {0}")]
    Http(#[from] ureq::Error),
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("concurrent benchmark thread panicked")]
    ThreadPanic,
}

#[derive(Clone)]
struct Client {
    agent: ureq::Agent,
    base_url: Arc<str>,
}

struct Options {
    base_url: String,
    output: PathBuf,
    samples: usize,
    long_context: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    suite: &'static str,
    model: &'static str,
    server_url: String,
    authority: &'static str,
    status: &'static str,
    samples_per_case: usize,
    long_context_enabled: bool,
    cases: Vec<CaseReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    in_progress_case: Option<InProgressCase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct CaseReport {
    name: String,
    timing_boundary: &'static str,
    cache_regime: &'static str,
    external_concurrency: usize,
    observations: Vec<Observation>,
    summary: CaseSummary,
}

#[derive(Serialize)]
struct InProgressCase {
    name: String,
    timing_boundary: &'static str,
    cache_regime: &'static str,
    external_concurrency: usize,
    expected_samples: usize,
    observations: Vec<Observation>,
}

#[derive(Clone, Debug, Serialize)]
struct Observation {
    request_count: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    completion_tokens: usize,
    visible_chunks: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_intertoken_ms: Option<f64>,
    e2e_ms: f64,
    completion_tokens_per_second: f64,
}

#[derive(Serialize)]
struct CaseSummary {
    e2e_ms: MetricSummary,
    completion_tokens_per_second: MetricSummary,
    cached_prompt_fraction: MetricSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms: Option<MetricSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_intertoken_ms: Option<MetricSummary>,
}

#[derive(Debug, Serialize)]
struct MetricSummary {
    samples: usize,
    minimum: f64,
    median: f64,
    p95: f64,
    maximum: f64,
}

#[derive(Clone, Copy, Debug)]
struct Usage {
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
}

fn main() {
    let options = match Options::parse(std::env::args()) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("bench-server: {error}");
            std::process::exit(2);
        }
    };
    let client = Client::new(options.base_url.clone());
    let mut report = Report {
        schema_version: 1,
        suite: "server-http",
        model: Qwen38_27B::MODEL_ID,
        server_url: options.base_url.clone(),
        authority: "diagnostic_external_no_clock_evidence",
        status: "running",
        samples_per_case: options.samples,
        long_context_enabled: options.long_context,
        cases: Vec::new(),
        in_progress_case: None,
        error: None,
    };
    if let Err(error) = write_report(&options.output, &report) {
        eprintln!("bench-server: could not initialize report: {error}");
        std::process::exit(1);
    }

    match run(&client, &options, &mut report) {
        Ok(()) => {
            report.status = "complete";
            if let Err(error) = write_report(&options.output, &report) {
                eprintln!("bench-server: could not finalize report: {error}");
                std::process::exit(1);
            }
            println!(
                "PASS server benchmark: {} cases preserved at {} (diagnostic; no clock authority)",
                report.cases.len(),
                options.output.display()
            );
        }
        Err(error) => {
            report.status = "failed";
            report.error = Some(error.to_string());
            if let Err(write_error) = write_report(&options.output, &report) {
                eprintln!("bench-server: additionally failed to preserve report: {write_error}");
            }
            eprintln!(
                "FAIL server benchmark after {} completed cases: {error}; partial report: {}",
                report.cases.len(),
                options.output.display()
            );
            std::process::exit(1);
        }
    }
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let program = args.next().unwrap_or_else(|| "bench-server".into());
        let mut base_url = None;
        let mut output = None;
        let mut samples = DEFAULT_SAMPLES;
        let mut long_context = false;
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--json" => {
                    output = Some(PathBuf::from(args.next().ok_or_else(|| {
                        BenchError::Contract("--json requires a path".into())
                    })?));
                }
                "--samples" => {
                    samples = args
                        .next()
                        .ok_or_else(|| BenchError::Contract("--samples requires a count".into()))?
                        .parse()
                        .map_err(|error| {
                            BenchError::Contract(format!("invalid --samples count: {error}"))
                        })?;
                }
                "--long-context" => long_context = true,
                flag if flag.starts_with("--") => {
                    return Err(BenchError::Contract(format!(
                        "unknown benchmark option {flag:?}"
                    )));
                }
                url if base_url.is_none() => base_url = Some(url.trim_end_matches('/').to_owned()),
                extra => {
                    return Err(BenchError::Contract(format!(
                        "unexpected positional argument {extra:?}"
                    )));
                }
            }
        }
        let base_url = base_url.unwrap_or_else(|| "http://127.0.0.1:8000".into());
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(BenchError::Contract(
                "server URL must begin with http:// or https://".into(),
            ));
        }
        if !(3..=40).contains(&samples) {
            return Err(BenchError::Contract(
                "--samples must be in the exact range 3..=40".into(),
            ));
        }
        let output = output.ok_or_else(|| {
            BenchError::Contract(format!(
                "usage: {program} [http://HOST:PORT] --json target/PATH [--samples N] [--long-context]"
            ))
        })?;
        validate_output_path(&output)?;
        Ok(Self {
            base_url,
            output,
            samples,
            long_context,
        })
    }
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

    fn completion_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn blocking(&self, messages: Value, max_tokens: usize) -> Result<Observation> {
        let started = Instant::now();
        let mut response = self
            .agent
            .post(self.completion_url())
            .send_json(request(messages, max_tokens, false))?;
        expect_status(&response, "blocking benchmark request")?;
        let body: Value = response.body_mut().read_json()?;
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        let usage = parse_usage(field(&body, "usage")?)?;
        if usage.completion_tokens > max_tokens {
            return Err(BenchError::Contract(format!(
                "blocking response reported {} completion tokens above requested {max_tokens}",
                usage.completion_tokens
            )));
        }
        validate_blocking(&body)?;
        Ok(observation(usage, 0, None, None, elapsed, 1))
    }

    fn streaming(&self, messages: Value, max_tokens: usize) -> Result<Observation> {
        let started = Instant::now();
        let response = self
            .agent
            .post(self.completion_url())
            .send_json(request(messages, max_tokens, true))?;
        expect_status(&response, "streaming benchmark request")?;
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !content_type.starts_with("text/event-stream") {
            return Err(BenchError::Contract(format!(
                "streaming benchmark returned content-type {content_type:?}"
            )));
        }

        let (_, body) = response.into_parts();
        let mut reader = BufReader::new(body.into_reader());
        let mut line = String::new();
        let mut first_visible = None;
        let mut terminal = None;
        let mut usage = None;
        let mut visible_chunks = 0;
        let mut done = false;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let Some(data) = line.trim_end_matches(['\r', '\n']).strip_prefix("data: ") else {
                continue;
            };
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            if data == "[DONE]" {
                done = true;
                break;
            }
            let event: Value = serde_json::from_str(data)?;
            if event.get("error").is_some() {
                return Err(BenchError::Contract(format!(
                    "streaming benchmark returned an error event: {event}"
                )));
            }
            let choices = field(&event, "choices")?
                .as_array()
                .ok_or_else(|| BenchError::Contract("stream choices is not an array".into()))?;
            if choices.is_empty() {
                usage = Some(parse_usage(field(&event, "usage")?)?);
                continue;
            }
            if choices.len() != 1 {
                return Err(BenchError::Contract(format!(
                    "stream event has {} choices",
                    choices.len()
                )));
            }
            let choice = &choices[0];
            let delta = field(choice, "delta")?;
            if delta
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| !content.is_empty())
                || delta
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| !content.is_empty())
            {
                first_visible.get_or_insert(elapsed);
                visible_chunks += 1;
            }
            if !field(choice, "finish_reason")?.is_null() {
                terminal = Some(elapsed);
            }
        }
        if !done {
            return Err(BenchError::Contract(
                "stream ended without the final [DONE] event".into(),
            ));
        }
        let usage = usage.ok_or_else(|| {
            BenchError::Contract("stream ended without its usage-only event".into())
        })?;
        if usage.completion_tokens > max_tokens {
            return Err(BenchError::Contract(format!(
                "stream reported {} completion tokens above requested {max_tokens}",
                usage.completion_tokens
            )));
        }
        let ttft_ms = first_visible.ok_or_else(|| {
            BenchError::Contract("stream did not publish visible generated text".into())
        })?;
        let terminal_ms =
            terminal.ok_or_else(|| BenchError::Contract("stream has no terminal event".into()))?;
        let e2e_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let mean_intertoken_ms = (usage.completion_tokens > 1)
            .then(|| (terminal_ms - ttft_ms) / (usage.completion_tokens - 1) as f64);
        Ok(observation(
            usage,
            visible_chunks,
            Some(ttft_ms),
            mean_intertoken_ms,
            e2e_ms,
            1,
        ))
    }
}

fn run(client: &Client, options: &Options, report: &mut Report) -> Result<()> {
    let reusable = reusable_messages();
    let _prime = client.blocking(reusable.clone(), STREAM_COMPLETION_TOKENS)?;
    let warmup = client.streaming(reusable.clone(), STREAM_COMPLETION_TOKENS)?;
    require_full_reuse(&warmup, "reused-stream warmup")?;
    measure_case(
        report,
        options,
        "stream/full-prefix".into(),
        "SSE request start through DONE",
        "reported full prompt reuse",
        1,
        |_: usize| {
            let observation = client.streaming(reusable.clone(), STREAM_COMPLETION_TOKENS)?;
            require_full_reuse(&observation, "reused-stream sample")?;
            Ok(observation)
        },
    )?;

    measure_case(
        report,
        options,
        "stream/low-reuse-256".into(),
        "SSE request start through DONE",
        "reported cached prompt tokens at most 25%",
        1,
        |sample| {
            let messages = fresh_messages(0x1000 + sample, 256);
            let observation = client.streaming(messages, STREAM_COMPLETION_TOKENS)?;
            require_low_reuse(&observation, "fresh-stream sample")?;
            Ok(observation)
        },
    )?;

    for concurrency in 1..=8 {
        measure_case(
            report,
            options,
            format!("blocking/external-concurrency-{concurrency}"),
            "barrier release through all blocking responses",
            "reported cached prompt tokens at most 25% per request",
            concurrency,
            |sample| concurrent_group(client.clone(), concurrency, sample),
        )?;
    }

    if options.long_context {
        for target in LONG_CONTEXTS {
            measure_case(
                report,
                options,
                format!("stream/long-context-{target}"),
                "SSE request start through DONE",
                "reported cached prompt tokens at most 25%",
                1,
                |sample| {
                    let messages = fresh_messages(0x10_0000 + target + sample, target - 32);
                    let observation = client.streaming(messages, LONG_COMPLETION_TOKENS)?;
                    require_low_reuse(&observation, "long-context sample")?;
                    if observation.prompt_tokens.abs_diff(target) > 128 {
                        return Err(BenchError::Contract(format!(
                            "long-context target {target} produced {} prompt tokens, outside the admitted +/-128 calibration band",
                            observation.prompt_tokens
                        )));
                    }
                    Ok(observation)
                },
            )?;
        }
    }
    Ok(())
}

fn concurrent_group(client: Client, concurrency: usize, sample: usize) -> Result<Observation> {
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let handles = (0..concurrency)
        .map(|lane| {
            let barrier = Arc::clone(&barrier);
            let client = client.clone();
            thread::spawn(move || {
                let messages =
                    fresh_messages(0x20_0000 + concurrency * 0x1_0000 + sample * 8 + lane, 256);
                barrier.wait();
                client.blocking(messages, CONCURRENT_COMPLETION_TOKENS)
            })
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    barrier.wait();
    let observations = handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| BenchError::ThreadPanic)?)
        .collect::<Result<Vec<_>>>()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1_000.0;
    for observation in &observations {
        require_low_reuse(observation, "concurrent request")?;
    }
    let prompt_tokens = observations.iter().map(|value| value.prompt_tokens).sum();
    let cached_prompt_tokens = observations
        .iter()
        .map(|value| value.cached_prompt_tokens)
        .sum();
    let completion_tokens = observations
        .iter()
        .map(|value| value.completion_tokens)
        .sum();
    Ok(Observation {
        request_count: concurrency,
        prompt_tokens,
        cached_prompt_tokens,
        completion_tokens,
        visible_chunks: 0,
        ttft_ms: None,
        mean_intertoken_ms: None,
        e2e_ms,
        completion_tokens_per_second: completion_tokens as f64 * 1_000.0 / e2e_ms,
    })
}

fn request(messages: Value, max_tokens: usize, stream: bool) -> Value {
    let mut request = json!({
        "model": Qwen38_27B::MODEL_ID,
        "messages": messages,
        "max_completion_tokens": max_tokens,
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

fn reusable_messages() -> Value {
    json!([{
        "role": "user",
        "content": "Write a numbered list of eight distinct colors, with one color per line and no explanation."
    }])
}

fn fresh_messages(nonce: usize, repeated_tokens: usize) -> Value {
    let content = format!(
        "{nonce:016x} begin unique benchmark prompt.{}",
        " blue".repeat(repeated_tokens)
    );
    json!([{"role": "user", "content": content}])
}

fn observation(
    usage: Usage,
    visible_chunks: usize,
    ttft_ms: Option<f64>,
    mean_intertoken_ms: Option<f64>,
    e2e_ms: f64,
    request_count: usize,
) -> Observation {
    Observation {
        request_count,
        prompt_tokens: usage.prompt_tokens,
        cached_prompt_tokens: usage.cached_tokens,
        completion_tokens: usage.completion_tokens,
        visible_chunks,
        ttft_ms,
        mean_intertoken_ms,
        e2e_ms,
        completion_tokens_per_second: usage.completion_tokens as f64 * 1_000.0 / e2e_ms,
    }
}

fn require_full_reuse(observation: &Observation, label: &str) -> Result<()> {
    if observation.cached_prompt_tokens == observation.prompt_tokens {
        Ok(())
    } else {
        Err(BenchError::Contract(format!(
            "{label} reported {}/{} cached prompt tokens, expected full reuse",
            observation.cached_prompt_tokens, observation.prompt_tokens
        )))
    }
}

fn require_low_reuse(observation: &Observation, label: &str) -> Result<()> {
    if observation.cached_prompt_tokens.saturating_mul(4) <= observation.prompt_tokens {
        Ok(())
    } else {
        Err(BenchError::Contract(format!(
            "{label} reported {}/{} cached prompt tokens, above the 25% low-reuse ceiling",
            observation.cached_prompt_tokens, observation.prompt_tokens
        )))
    }
}

fn measure_case(
    report: &mut Report,
    options: &Options,
    name: String,
    timing_boundary: &'static str,
    cache_regime: &'static str,
    external_concurrency: usize,
    mut measure: impl FnMut(usize) -> Result<Observation>,
) -> Result<()> {
    if report.in_progress_case.is_some() {
        return Err(BenchError::Contract(
            "benchmark attempted to overlap report cases".into(),
        ));
    }
    report.in_progress_case = Some(InProgressCase {
        name,
        timing_boundary,
        cache_regime,
        external_concurrency,
        expected_samples: options.samples,
        observations: Vec::with_capacity(options.samples),
    });
    write_report(&options.output, report)?;
    for sample in 0..options.samples {
        let observation = measure(sample)?;
        report
            .in_progress_case
            .as_mut()
            .expect("in-progress case exists during measurement")
            .observations
            .push(observation);
        write_report(&options.output, report)?;
    }
    let completed = report
        .in_progress_case
        .take()
        .expect("completed case was initialized");
    let summary = summarize_case(&completed.observations)?;
    report.cases.push(CaseReport {
        name: completed.name,
        timing_boundary: completed.timing_boundary,
        cache_regime: completed.cache_regime,
        external_concurrency: completed.external_concurrency,
        observations: completed.observations,
        summary,
    });
    write_report(&options.output, report)
}

fn summarize_case(observations: &[Observation]) -> Result<CaseSummary> {
    if observations.is_empty() {
        return Err(BenchError::Contract(
            "cannot summarize an empty benchmark case".into(),
        ));
    }
    let cached_fractions = observations
        .iter()
        .map(|value| value.cached_prompt_tokens as f64 / value.prompt_tokens as f64)
        .collect::<Vec<_>>();
    Ok(CaseSummary {
        e2e_ms: summarize(observations.iter().map(|value| value.e2e_ms))?,
        completion_tokens_per_second: summarize(
            observations
                .iter()
                .map(|value| value.completion_tokens_per_second),
        )?,
        cached_prompt_fraction: summarize(cached_fractions)?,
        ttft_ms: summarize_optional(observations.iter().map(|value| value.ttft_ms))?,
        mean_intertoken_ms: summarize_optional(
            observations.iter().map(|value| value.mean_intertoken_ms),
        )?,
    })
}

fn summarize(values: impl IntoIterator<Item = f64>) -> Result<MetricSummary> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(BenchError::Contract(
            "metric samples must be nonempty, finite, and nonnegative".into(),
        ));
    }
    values.sort_by(f64::total_cmp);
    let samples = values.len();
    let median = if samples % 2 == 0 {
        (values[samples / 2 - 1] + values[samples / 2]) / 2.0
    } else {
        values[samples / 2]
    };
    let p95_index = ((samples as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(samples - 1);
    Ok(MetricSummary {
        samples,
        minimum: values[0],
        median,
        p95: values[p95_index],
        maximum: values[samples - 1],
    })
}

fn summarize_optional(
    values: impl IntoIterator<Item = Option<f64>>,
) -> Result<Option<MetricSummary>> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.iter().all(Option::is_none) {
        return Ok(None);
    }
    if values.iter().any(Option::is_none) {
        return Err(BenchError::Contract(
            "optional metric inventory differs between samples".into(),
        ));
    }
    summarize(values.into_iter().flatten()).map(Some)
}

fn parse_usage(value: &Value) -> Result<Usage> {
    let prompt_tokens = usize_value(field(value, "prompt_tokens")?, "prompt_tokens")?;
    let details = field(value, "prompt_tokens_details")?;
    let cached_tokens = usize_value(field(details, "cached_tokens")?, "cached_tokens")?;
    let completion_tokens = usize_value(field(value, "completion_tokens")?, "completion_tokens")?;
    let total_tokens = usize_value(field(value, "total_tokens")?, "total_tokens")?;
    if prompt_tokens == 0 || completion_tokens == 0 {
        return Err(BenchError::Contract(
            "benchmark usage must report nonzero prompt and completion tokens".into(),
        ));
    }
    if cached_tokens > prompt_tokens || total_tokens != prompt_tokens + completion_tokens {
        return Err(BenchError::Contract(format!(
            "invalid usage accounting: prompt={prompt_tokens}, cached={cached_tokens}, completion={completion_tokens}, total={total_tokens}"
        )));
    }
    Ok(Usage {
        prompt_tokens,
        cached_tokens,
        completion_tokens,
    })
}

fn validate_blocking(body: &Value) -> Result<()> {
    if field(body, "object")? != "chat.completion" || field(body, "model")? != Qwen38_27B::MODEL_ID
    {
        return Err(BenchError::Contract(format!(
            "blocking response names the wrong product: {body}"
        )));
    }
    let choices = field(body, "choices")?
        .as_array()
        .ok_or_else(|| BenchError::Contract("blocking choices is not an array".into()))?;
    if choices.len() != 1
        || choices[0]["message"]["role"] != "assistant"
        || !choices[0]["message"]["content"].is_string()
    {
        return Err(BenchError::Contract(
            "blocking response has an invalid assistant choice".into(),
        ));
    }
    Ok(())
}

fn expect_status(response: &ureq::http::Response<ureq::Body>, label: &str) -> Result<()> {
    let status = response.status().as_u16();
    if status == 200 {
        Ok(())
    } else {
        Err(BenchError::Contract(format!(
            "{label} returned HTTP {status}"
        )))
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| BenchError::Contract(format!("response omitted {name}")))
}

fn usize_value(value: &Value, label: &str) -> Result<usize> {
    let value = value
        .as_u64()
        .ok_or_else(|| BenchError::Contract(format!("{label} is not an unsigned integer")))?;
    usize::try_from(value).map_err(|_| BenchError::Contract(format!("{label} does not fit usize")))
}

fn validate_output_path(path: &Path) -> Result<()> {
    if path.is_absolute()
        || !matches!(path.components().next(), Some(Component::Normal(first)) if first == "target")
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(BenchError::Contract(
            "benchmark JSON must be a relative path below target/".into(),
        ));
    }
    Ok(())
}

fn write_report(path: &Path, report: &Report) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| BenchError::Contract("benchmark report has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Client, Observation, Options, fresh_messages, parse_usage, summarize, summarize_case,
    };
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn summaries_preserve_even_median_and_nearest_rank_p95() {
        let summary = summarize([4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.median, 2.5);
        assert_eq!(summary.p95, 4.0);
        assert_eq!(summary.maximum, 4.0);
    }

    #[test]
    fn usage_requires_exact_cached_and_total_accounting() {
        let usage = parse_usage(&json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens": 8,
            "total_tokens": 108
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, 80);
        assert!(
            parse_usage(&json!({
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 101},
                "completion_tokens": 8,
                "total_tokens": 108
            }))
            .is_err()
        );
    }

    #[test]
    fn mixed_optional_timing_inventory_is_refused() {
        let observations = [
            observation(Some(1.0)),
            observation(None),
            observation(Some(3.0)),
        ];
        assert!(summarize_case(&observations).is_err());
    }

    #[test]
    fn options_require_target_json_and_three_samples() {
        assert!(Options::parse(["bench".into(), "--json".into(), "report.json".into()]).is_err());
        assert!(
            Options::parse([
                "bench".into(),
                "--json".into(),
                "target/report.json".into(),
                "--samples".into(),
                "2".into()
            ])
            .is_err()
        );
    }

    #[test]
    fn streaming_client_times_the_external_sse_and_reads_cache_accounting() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let fixture = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 16 * 1024];
            let _received = stream.read(&mut request).unwrap();
            let body = concat!(
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"bl\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"ue\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":10},\"completion_tokens\":2,\"total_tokens\":12}}\n\n",
                "data: [DONE]\n\n"
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let client = Client::new(format!("http://{address}"));
        let observation = client.streaming(fresh_messages(1, 4), 2).unwrap();
        fixture.join().unwrap();
        assert_eq!(observation.prompt_tokens, 10);
        assert_eq!(observation.cached_prompt_tokens, 10);
        assert_eq!(observation.completion_tokens, 2);
        assert_eq!(observation.visible_chunks, 2);
        assert!(observation.ttft_ms.is_some());
        assert!(observation.mean_intertoken_ms.is_some());
        assert!(observation.e2e_ms > 0.0);
    }

    fn observation(ttft_ms: Option<f64>) -> Observation {
        Observation {
            request_count: 1,
            prompt_tokens: 10,
            cached_prompt_tokens: 5,
            completion_tokens: 2,
            visible_chunks: 2,
            ttft_ms,
            mean_intertoken_ms: ttft_ms,
            e2e_ms: 4.0,
            completion_tokens_per_second: 500.0,
        }
    }
}
