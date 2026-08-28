//! Sequential and grouped prompt-prime evidence for Qwen3.8 Flash-Next.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{
    ChatGenerationRequest, EngineError, GeneratedText, Qwen38FlashNextResidentBatchGenerator,
    SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen38FlashNext};

const WIDTHS: [usize; 4] = [1, 2, 4, 8];
#[cfg(test)]
const SLOTS_ADMITTED: usize = 8;
const BUDGET: usize = 8;
const BURST_PROMPT: &str = "Hello";
const MIXED_PROMPTS: [&str; 8] = [
    "Name one primary color.",
    "Say hello.",
    "Describe a river in one sentence.",
    "What is two plus two?",
    "List three fruits, separated by commas.",
    "Give one fact about the moon.",
    "Write one short sentence about rain.",
    "Name the largest ocean.",
];

/// Failure of the prompt-prime benchmark.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextPromptPrimeBenchmarkError {
    /// Snapshot admission failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Frontend admission failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA setup failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact device was unavailable exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// The admission paths disagreed.
    #[error("Qwen3.8 Flash-Next prompt-prime mismatch: {0}")]
    Mismatch(String),
}

type BenchResult<T> = Result<T, Qwen38FlashNextPromptPrimeBenchmarkError>;

/// One admission path at one width.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextPrimeSample {
    /// Requests admitted.
    pub width: usize,
    /// Admission wall time.
    pub admission: Duration,
    /// All prime rounds.
    pub prime_rounds: usize,
    /// Scalar-tail rounds.
    pub prime_scalar_rounds: usize,
    /// Prompt rows carried.
    pub prime_rows: usize,
}

impl Qwen38FlashNextPrimeSample {
    /// Admission wall time in milliseconds.
    pub fn admission_ms(self) -> f64 {
        self.admission.as_secs_f64() * 1_000.0
    }
}

/// Sequential and grouped admission at one width.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextPrimePair {
    /// One-at-a-time admission.
    pub sequential: Qwen38FlashNextPrimeSample,
    /// Grouped admission.
    pub grouped: Qwen38FlashNextPrimeSample,
}

impl Qwen38FlashNextPrimePair {
    /// Sequential wall time divided by grouped wall time.
    pub fn speedup(self) -> f64 {
        let grouped = self.grouped.admission.as_secs_f64();
        if grouped == 0.0 {
            0.0
        } else {
            self.sequential.admission.as_secs_f64() / grouped
        }
    }
}

/// Prompt-prime sweep and exactness evidence.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextPromptPrimeBenchmark {
    /// Repeated-prompt pairs at every funded width.
    pub burst: Vec<Qwen38FlashNextPrimePair>,
    /// Eight distinct prompts.
    pub mixed: Qwen38FlashNextPrimePair,
    /// Complete outputs compared across admission paths.
    pub compared_sequences: usize,
}

/// Times sequential and grouped admission on the production owner.
pub fn benchmark_qwen38_flash_next_prompt_prime(
    root: &Path,
) -> BenchResult<Qwen38FlashNextPromptPrimeBenchmark> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let mut generator =
        Qwen38FlashNextResidentBatchGenerator::from_snapshot(&context, snapshot, None)?;

    drain(&mut generator, &[greedy_request(BURST_PROMPT, BUDGET)])?;

    let mut burst = Vec::with_capacity(WIDTHS.len());
    for width in WIDTHS {
        let requests = vec![greedy_request(BURST_PROMPT, BUDGET); width];
        burst.push(measure_pair(&mut generator, &requests)?);
    }

    let mixed = MIXED_PROMPTS
        .iter()
        .map(|prompt| greedy_request(prompt, BUDGET))
        .collect::<Vec<_>>();
    let mixed_pair = measure_pair(&mut generator, &mixed)?;
    let compared_sequences = verify_agreement(&mut generator, &mixed)?;

    Ok(Qwen38FlashNextPromptPrimeBenchmark {
        burst,
        mixed: mixed_pair,
        compared_sequences,
    })
}

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextPromptPrimeBenchmarkError {
    Qwen38FlashNextPromptPrimeBenchmarkError::Mismatch(message.into())
}

fn greedy_request(content: &str, maximum: usize) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", content)]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = maximum;
    request
}

fn measure_pair(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> BenchResult<Qwen38FlashNextPrimePair> {
    Ok(Qwen38FlashNextPrimePair {
        sequential: measure_sequential(generator, requests)?,
        grouped: measure_grouped(generator, requests)?,
    })
}

fn measure_sequential(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> BenchResult<Qwen38FlashNextPrimeSample> {
    require_empty(generator)?;
    generator.reset_telemetry();
    let started = Instant::now();
    for request in requests {
        generator.admit(request)?;
    }
    let sample = sample(generator, requests.len(), started.elapsed());
    release_all(generator)?;
    Ok(sample)
}

fn measure_grouped(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> BenchResult<Qwen38FlashNextPrimeSample> {
    require_empty(generator)?;
    generator.reset_telemetry();
    let group = requests.iter().collect::<Vec<_>>();
    let started = Instant::now();
    for admission in generator.admit_batch(&group) {
        admission?;
    }
    let sample = sample(generator, requests.len(), started.elapsed());
    release_all(generator)?;
    Ok(sample)
}

fn sample(
    generator: &Qwen38FlashNextResidentBatchGenerator,
    width: usize,
    admission: Duration,
) -> Qwen38FlashNextPrimeSample {
    let telemetry = generator.telemetry();
    Qwen38FlashNextPrimeSample {
        width,
        admission,
        prime_rounds: telemetry.prime_rounds(),
        prime_scalar_rounds: telemetry.prime_scalar_rounds(),
        prime_rows: telemetry.prime_rows(),
    }
}

fn verify_agreement(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> BenchResult<usize> {
    let mut alone = Vec::with_capacity(requests.len());
    for request in requests {
        alone.push(drain(generator, std::slice::from_ref(request))?.remove(0));
    }
    let grouped = drain(generator, requests)?;
    for (lane, (expected, actual)) in alone.iter().zip(&grouped).enumerate() {
        if expected.prompt.token_ids != actual.prompt.token_ids
            || expected.prompt.rendered_bytes != actual.prompt.rendered_bytes
            || expected.token_ids != actual.token_ids
            || expected.text != actual.text
            || expected.finish_reason != actual.finish_reason
        {
            return Err(mismatch(format!(
                "lane {lane} differs: alone {:?}, grouped {:?}",
                expected.token_ids, actual.token_ids
            )));
        }
    }
    Ok(grouped.len())
}

fn drain(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> BenchResult<Vec<GeneratedText>> {
    require_empty(generator)?;
    let group = requests.iter().collect::<Vec<_>>();
    let mut identities = Vec::with_capacity(requests.len());
    for admission in generator.admit_batch(&group) {
        identities.push(admission?.request_id);
    }
    let mut finished = vec![None; identities.len()];
    while generator.active_requests() > 0 {
        for event in generator.step()?.iter() {
            let lane = identities
                .iter()
                .position(|identity| *identity == event.request_id)
                .ok_or_else(|| mismatch("a round returned an unknown request"))?;
            if let Some(output) = &event.completed {
                finished[lane] = Some(output.clone());
            }
        }
    }

    finished
        .into_iter()
        .enumerate()
        .map(|(lane, output)| output.ok_or_else(|| mismatch(format!("lane {lane} never finished"))))
        .collect()
}

fn require_empty(generator: &Qwen38FlashNextResidentBatchGenerator) -> BenchResult<()> {
    if generator.active_requests() == 0 {
        Ok(())
    } else {
        Err(mismatch("an admission started beside a live request"))
    }
}

fn release_all(generator: &mut Qwen38FlashNextResidentBatchGenerator) -> BenchResult<()> {
    let active = generator.active_request_ids().collect::<Vec<_>>();
    for request in active {
        generator.cancel(request)?;
    }
    Ok(())
}

/// Prints the diagnostic sweep.
pub fn print_qwen38_flash_next_prompt_prime_benchmark(
    report: &Qwen38FlashNextPromptPrimeBenchmark,
) {
    println!("# Qwen3.8 Flash-Next prompt prime - diagnostic, nothing blessed");
    println!(
        "| group | sequential | grouped | speedup | sequential scalar rounds | grouped scalar rounds | rows |"
    );
    println!("| --: | --: | --: | --: | --: | --: | --: |");
    for pair in &report.burst {
        print_pair(pair);
    }
    println!();
    println!("## Eight distinct prompts");
    println!(
        "| group | sequential | grouped | speedup | sequential scalar rounds | grouped scalar rounds | rows |"
    );
    println!("| --: | --: | --: | --: | --: | --: | --: |");
    print_pair(&report.mixed);
    println!();
    println!("{} sequences matched exactly.", report.compared_sequences);
}

fn print_pair(pair: &Qwen38FlashNextPrimePair) {
    println!(
        "| {} | {:.1} ms | {:.1} ms | {:.2}x | {} | {} | {} |",
        pair.sequential.width,
        pair.sequential.admission_ms(),
        pair.grouped.admission_ms(),
        pair.speedup(),
        pair.sequential.prime_scalar_rounds,
        pair.grouped.prime_scalar_rounds,
        pair.grouped.prime_rows,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn qwen38_flash_next_prompt_prime_benchmark_accounting_covers_every_funded_width() {
        assert_eq!(WIDTHS, [1, 2, 4, 8]);
        assert_eq!(MIXED_PROMPTS.len(), SLOTS_ADMITTED);
        let mut prompts = MIXED_PROMPTS.to_vec();
        prompts.sort_unstable();
        prompts.dedup();
        assert_eq!(prompts.len(), SLOTS_ADMITTED);
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.8 Flash-Next snapshot and an exclusive SM120 device"]
    fn qwen38_flash_next_prompt_prime_wide_group_matches_sequential_outputs() -> BenchResult<()> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT")
            .ok_or_else(|| mismatch("set TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT"))?;
        let report = benchmark_qwen38_flash_next_prompt_prime(&PathBuf::from(root))?;
        print_qwen38_flash_next_prompt_prime_benchmark(&report);

        assert_eq!(report.compared_sequences, SLOTS_ADMITTED);
        let wide = report.burst.last().expect("the sweep reaches B=8");
        assert_eq!(wide.grouped.width, SLOTS_ADMITTED);
        assert_eq!(wide.grouped.prime_rows, wide.sequential.prime_rows);
        assert!(wide.grouped.prime_scalar_rounds < wide.sequential.prime_scalar_rounds);
        assert_eq!(
            wide.grouped.prime_rounds - wide.grouped.prime_scalar_rounds,
            wide.sequential.prime_rounds - wide.sequential.prime_scalar_rounds
        );
        Ok(())
    }
}
