//! Terminal request timing and cache-accounting log lines.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};
use tuisko_engine::ResidentMtpGenerationStats;
use tuisko_frontend::{PromptEncoding, PromptEncodingMetrics};

#[derive(Clone, Copy)]
struct PrefillObservation {
    batch: usize,
    native_tokens: usize,
    queue: Duration,
    elapsed: Duration,
}

pub(crate) struct RequestLog {
    id: u64,
    accepted: Instant,
    accepted_offset: Duration,
    first_output: Option<Instant>,
    prompt_metrics: PromptEncodingMetrics,
    prefill: Option<PrefillObservation>,
    mtp_stats: Option<ResidentMtpGenerationStats>,
    route: &'static str,
}

pub(crate) struct ScoringRequestLog {
    id: u64,
    accepted: Instant,
    accepted_offset: Duration,
    batch: usize,
    prompt_tokens: usize,
    min_prompt_tokens: usize,
    max_prompt_tokens: usize,
    common_prefix_tokens: usize,
    evaluated_tokens: usize,
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
            prefill: None,
            mtp_stats: None,
            route,
        }
    }

    pub(crate) fn observe_prompt(&mut self, metrics: PromptEncodingMetrics) {
        self.prompt_metrics = metrics;
    }

    pub(crate) const fn accepted(&self) -> Instant {
        self.accepted
    }

    pub(crate) fn observe_output(&mut self) {
        self.first_output.get_or_insert_with(Instant::now);
    }

    pub(crate) fn observe_prefill(
        &mut self,
        batch: usize,
        native_tokens: usize,
        queue: Duration,
        elapsed: Duration,
    ) {
        self.prefill = Some(PrefillObservation {
            batch,
            native_tokens,
            queue,
            elapsed,
        });
    }

    pub(crate) fn observe_mtp_stats(&mut self, stats: ResidentMtpGenerationStats) {
        debug_assert!(stats.accepted_drafts <= stats.draft_proposals);
        self.mtp_stats = Some(stats);
    }

    pub(crate) fn finish(
        self,
        prompt: Option<&PromptEncoding>,
        generated_tokens: usize,
        cached_prompt_tokens: usize,
        finish_reason: &str,
        error: Option<&str>,
    ) {
        let stderr = std::io::stderr();
        let color = stderr.is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let mut stderr = stderr.lock();
        let _ = writeln!(
            stderr,
            "{}",
            self.render_at(
                Instant::now(),
                prompt,
                generated_tokens,
                cached_prompt_tokens,
                finish_reason,
                error,
                color,
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
        color: bool,
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
        let queue_ms = self
            .prefill
            .map_or(0.0, |prefill| prefill.queue.as_secs_f64() * 1_000.0);
        let prefill = self.prefill.map(|prefill| {
            let milliseconds = prefill.elapsed.as_secs_f64() * 1_000.0;
            let tokens_per_second = if prefill.elapsed.is_zero() {
                0.0
            } else {
                prefill.native_tokens as f64 / prefill.elapsed.as_secs_f64()
            };
            format!(
                "prefill B{} {}/{} ({} tok/s) · queue {queue_ms:.1}ms",
                prefill.batch,
                compact_count(prefill.native_tokens),
                compact_seconds(milliseconds),
                compact_count(tokens_per_second.round() as usize),
            )
        });
        let bpe_tail_tokens = prompt_tokens.saturating_sub(frontend_reused);
        let mtp = self.mtp_stats.map(|stats| {
            let verifications = stats.verification_routes.iter().sum::<usize>();
            let decode_ms = ttft_ms.map_or(0.0, |ttft| (total_ms - ttft).max(0.0));
            let milliseconds_per_verification = if verifications == 0 {
                0.0
            } else {
                decode_ms / verifications as f64
            };
            let outputs_per_verification = if verifications == 0 {
                0.0
            } else {
                stats.verified_outputs as f64 / verifications as f64
            };
            let percent = if stats.draft_proposals == 0 {
                0.0
            } else {
                100.0 * stats.accepted_drafts as f64 / stats.draft_proposals as f64
            };
            (
                format!(" · MTP {percent:.1}%"),
                format!(
                    "verify {verifications} K={}/{}/{}/{} · {milliseconds_per_verification:.1}ms/v · {outputs_per_verification:.2} tok/v · MTP {}/{}",
                    stats.verification_routes[0],
                    stats.verification_routes[1],
                    stats.verification_routes[2],
                    stats.verification_routes[3],
                    stats.accepted_drafts,
                    stats.draft_proposals,
                ),
            )
        });
        let miss = if self.prompt_metrics.miss_reason.is_empty() {
            String::new()
        } else {
            format!(" miss:{}", self.prompt_metrics.miss_reason)
        };
        let error = error.map(|message| message.replace(['\r', '\n'], " "));
        let ttft = ttft_ms.map_or_else(
            || "TTFT pending".into(),
            |milliseconds| format!("TTFT {}", compact_seconds(milliseconds)),
        );
        let (request_color, reset) = if color {
            ("\x1b[1;36m", "\x1b[0m")
        } else {
            ("", "")
        };
        let mtp_summary = mtp
            .as_ref()
            .map_or_else(String::new, |(summary, _)| summary.clone());
        let mut details = vec![
            format!(
                "accepted +{}",
                compact_seconds(self.accepted_offset.as_secs_f64() * 1_000.0)
            ),
            format!("input {}", compact_count(input_tokens)),
        ];
        if let Some(prefill) = prefill {
            details.push(prefill);
        }
        if let Some((_, mtp)) = mtp {
            details.push(mtp);
        }
        details.push(format!("route {}", self.route));
        details.push(format!(
            "frontend {:.1}ms render + {:.1}ms encode",
            self.prompt_metrics.render_us as f64 / 1_000.0,
            self.prompt_metrics.encode_us as f64 / 1_000.0,
        ));
        details.push(format!("BPE tail {}{miss}", compact_count(bpe_tail_tokens)));
        if let Some(error) = error {
            details.push(format!("error {error}"));
        }

        format!(
            "{request_color}REQUEST{reset} {:<7} {} prompt · {} cached ({cached_percent:.1}%) · {} output · {ttft} · {decode_tokens_per_second:.1} tok/s{mtp_summary} · {} · {finish_reason}\n                {}",
            self.id,
            compact_count(prompt_tokens),
            compact_count(cached_prompt_tokens),
            compact_count(generated_tokens),
            compact_seconds(total_ms),
            details.join(" · "),
        )
    }
}

impl ScoringRequestLog {
    pub(crate) fn new(
        id: u64,
        accepted: Instant,
        server_started: Instant,
        prompts: &[Vec<u32>],
        evaluated_tokens: usize,
        route: &'static str,
    ) -> Self {
        let prompt_tokens = prompts.iter().map(Vec::len).sum();
        let min_prompt_tokens = prompts.iter().map(Vec::len).min().unwrap_or_default();
        let max_prompt_tokens = prompts.iter().map(Vec::len).max().unwrap_or_default();
        let common_prefix_tokens = prompts
            .first()
            .map_or(0, |first| {
                first
                    .iter()
                    .enumerate()
                    .take_while(|(index, token)| {
                        prompts
                            .iter()
                            .skip(1)
                            .all(|prompt| prompt.get(*index) == Some(*token))
                    })
                    .count()
            })
            .min(min_prompt_tokens);

        Self {
            id,
            accepted,
            accepted_offset: accepted.saturating_duration_since(server_started),
            batch: prompts.len(),
            prompt_tokens,
            min_prompt_tokens,
            max_prompt_tokens,
            common_prefix_tokens,
            evaluated_tokens,
            route,
        }
    }

    pub(crate) fn native(
        id: u64,
        accepted: Instant,
        server_started: Instant,
        context_tokens: usize,
        continuations: &[Vec<u32>],
    ) -> Self {
        let continuation_tokens = continuations.iter().map(Vec::len).sum::<usize>();
        let min_continuation_tokens = continuations.iter().map(Vec::len).min().unwrap_or_default();
        let max_continuation_tokens = continuations.iter().map(Vec::len).max().unwrap_or_default();

        Self {
            id,
            accepted,
            accepted_offset: accepted.saturating_duration_since(server_started),
            batch: continuations.len(),
            prompt_tokens: context_tokens * continuations.len() + continuation_tokens,
            min_prompt_tokens: context_tokens + min_continuation_tokens,
            max_prompt_tokens: context_tokens + max_continuation_tokens,
            common_prefix_tokens: context_tokens,
            evaluated_tokens: context_tokens + continuation_tokens,
            route: "native-loglikelihood",
        }
    }

    pub(crate) fn finish(
        self,
        scoring_started: Option<Instant>,
        finish_reason: &str,
        error: Option<&str>,
    ) {
        let stderr = std::io::stderr();
        let color = stderr.is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let mut stderr = stderr.lock();
        let _ = writeln!(
            stderr,
            "{}",
            self.render_at(Instant::now(), scoring_started, finish_reason, error, color,)
        );
    }

    fn render_at(
        &self,
        finished: Instant,
        scoring_started: Option<Instant>,
        finish_reason: &str,
        error: Option<&str>,
        color: bool,
    ) -> String {
        let total = finished.saturating_duration_since(self.accepted);
        let queue = scoring_started
            .unwrap_or(finished)
            .saturating_duration_since(self.accepted);
        let scoring = scoring_started.map(|started| finished.saturating_duration_since(started));
        let outputs = usize::from(error.is_none()) * self.batch;
        let (request_color, reset) = if color {
            ("\x1b[1;36m", "\x1b[0m")
        } else {
            ("", "")
        };
        let mut details = vec![
            format!(
                "accepted +{}",
                compact_seconds(self.accepted_offset.as_secs_f64() * 1_000.0)
            ),
            format!("queue {:.1}ms", queue.as_secs_f64() * 1_000.0),
        ];
        if let Some(scoring) = scoring {
            let tokens_per_second = if scoring.is_zero() {
                0.0
            } else {
                self.evaluated_tokens as f64 / scoring.as_secs_f64()
            };
            details.push(format!(
                "score {} ({} tok/s)",
                compact_seconds(scoring.as_secs_f64() * 1_000.0),
                compact_count(tokens_per_second.round() as usize),
            ));
        }
        details.push(format!(
            "lengths {}..{}",
            compact_count(self.min_prompt_tokens),
            compact_count(self.max_prompt_tokens),
        ));
        if self.batch > 1 {
            let common_percent = if self.min_prompt_tokens == 0 {
                0.0
            } else {
                100.0 * self.common_prefix_tokens as f64 / self.min_prompt_tokens as f64
            };
            details.push(format!(
                "common {}/{} ({common_percent:.1}%)",
                compact_count(self.common_prefix_tokens),
                compact_count(self.min_prompt_tokens),
            ));
        }
        details.push(format!("route {}", self.route));
        if let Some(error) = error {
            details.push(format!("error {}", error.replace(['\r', '\n'], " ")));
        }

        format!(
            "{request_color}REQUEST{reset} {:<7} B{} · {} prompt · {} output · {} · {finish_reason}\n                {}",
            self.id,
            self.batch,
            compact_count(self.prompt_tokens),
            compact_count(outputs),
            compact_seconds(total.as_secs_f64() * 1_000.0),
            details.join(" · "),
        )
    }
}

fn compact_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn compact_seconds(milliseconds: f64) -> String {
    format!("{:.1}s", milliseconds / 1_000.0)
}

#[cfg(test)]
mod tests {
    use super::{RequestLog, ScoringRequestLog};
    use std::time::{Duration, Instant};
    use tuisko_engine::{FinishReason, GeneratedText, ResidentMtpGenerationStats};
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
        log.observe_prefill(1, 20, Duration::from_millis(10), Duration::from_millis(100));
        log.observe_mtp_stats(ResidentMtpGenerationStats {
            verification_routes: [0, 0, 0, 10],
            draft_proposals: 30,
            accepted_drafts: 23,
            verified_outputs: 29,
        });
        let line = log.render_at(
            accepted + Duration::from_millis(1_200),
            Some(&output().prompt),
            11,
            80,
            "length",
            None,
            false,
        );

        assert_eq!(
            line,
            "REQUEST 7       100 prompt · 80 cached (80.0%) · 11 output · TTFT 0.2s · 10.0 tok/s · MTP 76.7% · 1.2s · length\n                accepted +0.1s · input 20 · prefill B1 20/0.1s (200 tok/s) · queue 10.0ms · verify 10 K=0/0/0/10 · 100.0ms/v · 2.90 tok/v · MTP 23/30 · route mtp-draft-3 · frontend 1.5ms render + 2.2ms encode · BPE tail 25"
        );

        let colored = log.render_at(
            accepted + Duration::from_millis(1_200),
            Some(&output().prompt),
            11,
            80,
            "length",
            None,
            true,
        );
        assert!(colored.starts_with("\x1b[1;36mREQUEST\x1b[0m 7       "));
    }

    #[test]
    fn observed_mtp_without_proposals_reports_a_defined_zero_rate() {
        let started = Instant::now();
        let mut log = RequestLog::new(8, started, started, "mtp-draft-3");
        log.observe_mtp_stats(ResidentMtpGenerationStats::default());

        let line = log.render_at(
            started + Duration::from_millis(1),
            None,
            0,
            0,
            "length",
            None,
            false,
        );

        assert!(line.contains("MTP 0.0%"));
        assert!(line.contains("MTP 0/0"));
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
            false,
        );

        assert!(line.contains("0 prompt · 0 cached (0.0%) · 0 output"));
        assert!(line.contains("TTFT pending · 0.0 tok/s"));
        assert!(line.ends_with("error capacity refused"));
        assert!(!line.contains("MTP"));
        assert!(!line.contains('\r'));
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
            false,
        );

        assert!(line.contains("BPE tail 100 miss:cache-empty"));
        assert!(line.contains(" · stop\n"));
    }

    #[test]
    fn scoring_line_reports_batch_prefix_queue_and_throughput_without_contents() {
        let started = Instant::now();
        let accepted = started + Duration::from_millis(125);
        let scoring_started = accepted + Duration::from_millis(10);
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 5]];
        let log = ScoringRequestLog::new(11, accepted, started, &prompts, 8, "prompt-scoring");
        let line = log.render_at(
            scoring_started + Duration::from_millis(40),
            Some(scoring_started),
            "length",
            None,
            false,
        );

        assert_eq!(
            line,
            "REQUEST 11      B2 · 8 prompt · 2 output · 0.1s · length\n                accepted +0.1s · queue 10.0ms · score 0.0s (200 tok/s) · lengths 4..4 · common 3/4 (75.0%) · route prompt-scoring"
        );
        assert!(!line.contains("[1, 2, 3"));
    }

    #[test]
    fn native_scoring_line_uses_shared_context_work_and_truthful_route() {
        let started = Instant::now();
        let log = ScoringRequestLog::native(13, started, started, 2, &[vec![3], vec![3, 5]]);
        let line = log.render_at(
            started + Duration::from_millis(100),
            Some(started),
            "length",
            None,
            false,
        );

        assert!(line.contains("score 0.1s (50 tok/s)"));
        assert!(line.contains("lengths 3..4 · common 2/3 (66.7%)"));
        assert!(line.contains("route native-loglikelihood"));
    }

    #[test]
    fn rejected_scoring_line_sanitizes_errors_and_has_no_score_phase() {
        let started = Instant::now();
        let log = ScoringRequestLog::new(12, started, started, &[vec![7, 8]], 2, "prompt-scoring");
        let line = log.render_at(
            started + Duration::from_millis(3),
            None,
            "error",
            Some("capacity\nrefused"),
            false,
        );

        assert!(line.contains("B1 · 2 prompt · 0 output · 0.0s · error"));
        assert!(line.contains("queue 3.0ms · lengths 2..2 · route prompt-scoring"));
        assert!(line.ends_with("error capacity refused"));
        assert!(!line.contains("score "));
        assert!(!line.contains("common"));
        assert!(!line.contains('\r'));
    }
}
