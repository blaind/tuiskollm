//! Terminal request timing and cache-accounting log lines.

use std::time::{Duration, Instant};
use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics};

pub(crate) struct RequestLog {
    id: u64,
    accepted: Instant,
    accepted_offset: Duration,
    first_output: Option<Instant>,
    prompt_metrics: PromptEncodingMetrics,
    mtp_acceptance: Option<(usize, usize)>,
    route: &'static str,
}

impl RequestLog {
    pub(crate) fn new(
        id: u64,
        accepted: Instant,
        server_started: Instant,
        route: &'static str,
    ) -> Self {
        Self {
            id,
            accepted,
            accepted_offset: accepted.saturating_duration_since(server_started),
            first_output: None,
            prompt_metrics: PromptEncodingMetrics::default(),
            mtp_acceptance: None,
            route,
        }
    }

    pub(crate) fn observe_prompt(&mut self, metrics: PromptEncodingMetrics) {
        self.prompt_metrics = metrics;
    }

    pub(crate) fn observe_output(&mut self) {
        self.first_output.get_or_insert_with(Instant::now);
    }

    pub(crate) fn observe_mtp_acceptance(&mut self, accepted: usize, proposed: usize) {
        debug_assert!(accepted <= proposed);
        self.mtp_acceptance = Some((accepted, proposed));
    }

    pub(crate) fn finish(
        self,
        prompt: Option<&PromptEncoding>,
        generated_tokens: usize,
        cached_prompt_tokens: usize,
        finish_reason: &str,
        error: Option<&str>,
    ) {
        eprintln!(
            "{}",
            self.render_at(
                Instant::now(),
                prompt,
                generated_tokens,
                cached_prompt_tokens,
                finish_reason,
                error,
            )
        );
    }

    fn render_at(
        &self,
        finished: Instant,
        prompt: Option<&PromptEncoding>,
        generated_tokens: usize,
        cached_prompt_tokens: usize,
        finish_reason: &str,
        error: Option<&str>,
    ) -> String {
        let total_ms = finished
            .saturating_duration_since(self.accepted)
            .as_secs_f64()
            * 1_000.0;
        let ttft_ms = self
            .first_output
            .map(|first| first.saturating_duration_since(self.accepted).as_secs_f64() * 1_000.0);
        let (prompt_tokens, frontend_reused) = prompt.map_or((0, 0), |prompt| {
            (prompt.token_ids.len(), prompt.reused_tokens)
        });
        let cached_prompt_tokens = cached_prompt_tokens.min(prompt_tokens);
        let input_tokens = prompt_tokens - cached_prompt_tokens;
        debug_assert_eq!(cached_prompt_tokens + input_tokens, prompt_tokens);
        let cached_percent = if prompt_tokens == 0 {
            0.0
        } else {
            100.0 * cached_prompt_tokens as f64 / prompt_tokens as f64
        };
        let decode_tokens_per_second = ttft_ms
            .filter(|ttft| generated_tokens > 1 && total_ms > *ttft)
            .map_or(0.0, |ttft| {
                (generated_tokens - 1) as f64 / ((total_ms - ttft) / 1_000.0)
            });
        let bpe_tail_tokens = prompt_tokens.saturating_sub(frontend_reused);
        let mtp_acceptance =
            self.mtp_acceptance
                .map_or_else(String::new, |(accepted, proposed)| {
                    let percent = if proposed == 0 {
                        0.0
                    } else {
                        100.0 * accepted as f64 / proposed as f64
                    };
                    format!(", mtp accept {accepted}/{proposed} ({percent:.1}%)")
                });
        let miss = if self.prompt_metrics.miss_reason.is_empty() {
            String::new()
        } else {
            format!(" miss:{}", self.prompt_metrics.miss_reason)
        };
        let error = error.map_or_else(String::new, |message| {
            format!(" ({})", message.replace(['\r', '\n'], " "))
        });

        format!(
            "TuiskoLLM request {}: {:.0} ms (+{total_ms:.1} ms), prompt {prompt_tokens} tok, cached {cached_prompt_tokens} tok ({cached_percent:.1}%), input {input_tokens} tok, gen {generated_tokens} tok, ttft {:.1} ms, decode {decode_tokens_per_second:.1} tok/s{mtp_acceptance}, route {}, render {:.1} ms, encode {:.1} ms, bpe-tail {bpe_tail_tokens} tok{miss}, finish {finish_reason}{error}",
            self.id,
            self.accepted_offset.as_secs_f64() * 1_000.0,
            ttft_ms.unwrap_or(-1.0),
            self.route,
            self.prompt_metrics.render_us as f64 / 1_000.0,
            self.prompt_metrics.encode_us as f64 / 1_000.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::RequestLog;
    use std::time::{Duration, Instant};
    use tuisko_engine::{FinishReason, GeneratedText};
    use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics};

    fn output() -> GeneratedText {
        GeneratedText {
            prompt: PromptEncoding {
                token_ids: vec![1; 100],
                message_boundary_tokens: 96,
                reused_tokens: 75,
                rendered_bytes: 400,
                fresh_bytes: 100,
            },
            token_ids: vec![2; 11],
            text: "answer".into(),
            finish_reason: FinishReason::Length,
        }
    }

    #[test]
    fn terminal_line_derives_every_timing_and_cache_quantity() {
        let started = Instant::now();
        let accepted = started + Duration::from_millis(125);
        let mut log = RequestLog::new(7, accepted, started, "mtp-draft-3");
        log.observe_prompt(PromptEncodingMetrics {
            render_us: 1_500,
            encode_us: 2_250,
            miss_reason: String::new(),
        });
        log.first_output = Some(accepted + Duration::from_millis(200));
        log.observe_mtp_acceptance(23, 30);
        let line = log.render_at(
            accepted + Duration::from_millis(1_200),
            Some(&output().prompt),
            11,
            80,
            "length",
            None,
        );

        assert_eq!(
            line,
            "TuiskoLLM request 7: 125 ms (+1200.0 ms), prompt 100 tok, cached 80 tok (80.0%), input 20 tok, gen 11 tok, ttft 200.0 ms, decode 10.0 tok/s, mtp accept 23/30 (76.7%), route mtp-draft-3, render 1.5 ms, encode 2.2 ms, bpe-tail 25 tok, finish length"
        );
    }

    #[test]
    fn observed_mtp_without_proposals_reports_a_defined_zero_rate() {
        let started = Instant::now();
        let mut log = RequestLog::new(8, started, started, "mtp-draft-3");
        log.observe_mtp_acceptance(0, 0);

        let line = log.render_at(
            started + Duration::from_millis(1),
            None,
            0,
            0,
            "length",
            None,
        );

        assert!(line.contains("mtp accept 0/0 (0.0%)"));
    }

    #[test]
    fn error_without_output_is_explicit_and_never_invents_ttft() {
        let started = Instant::now();
        let log = RequestLog::new(9, started, started, "mtp-draft-3");
        let line = log.render_at(
            started + Duration::from_millis(3),
            None,
            0,
            0,
            "error",
            Some("capacity\nrefused"),
        );

        assert!(line.contains("prompt 0 tok, cached 0 tok (0.0%), input 0 tok"));
        assert!(line.contains("ttft -1.0 ms, decode 0.0 tok/s"));
        assert!(line.ends_with("finish error (capacity refused)"));
        assert!(!line.contains("mtp accept"));
        assert!(!line.contains(['\r', '\n']));
    }

    #[test]
    fn frontend_miss_reason_is_attached_to_bpe_tail() {
        let started = Instant::now();
        let mut log = RequestLog::new(3, started, started, "single-token");
        log.observe_prompt(PromptEncodingMetrics {
            render_us: 0,
            encode_us: 0,
            miss_reason: "cache-empty".into(),
        });
        let mut output = output();
        output.prompt.reused_tokens = 0;
        let line = log.render_at(
            started + Duration::from_millis(1),
            Some(&output.prompt),
            output.token_ids.len(),
            0,
            "stop",
            None,
        );

        assert!(line.contains("bpe-tail 100 tok miss:cache-empty, finish stop"));
    }
}
