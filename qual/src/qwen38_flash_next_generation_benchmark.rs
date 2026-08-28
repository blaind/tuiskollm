//! Diagnostic whole-request Qwen3.8 Flash-Next generation sweep.

use crate::CellWallBudget;
use crate::device_benchmark::{self, DeviceBenchmarkError};
use crate::qwen38_flash_next_golden::{
    load_qwen38_flash_next_golden_boundary, load_qwen38_flash_next_golden_capture,
    qwen38_flash_next_golden_directory,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tuisko_engine::{
    QWEN38_FLASH_NEXT_ATTENTION_LAYERS, Qwen38FlashNextGenerationTelemetry,
    Qwen38FlashNextStreamingRoute, Qwen38FlashNextTextGenerator, SamplingOptions,
};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

/// Warm requests before any measurement.
const WARM_REQUESTS: usize = 3;

/// Measured requests per swept prompt length.
const MEASURED_REQUESTS: usize = 4;

/// Generated tokens per measured request.
const GENERATED_TOKENS: usize = 24;

/// Prompt lengths the sweep distinguishes, chosen so each exercises a different tile ladder.
const SWEPT_PROMPTS: [usize; 5] = [16, 64, 160, 512, 1_120];

/// Selected-route prompt depths.
const LONG_PROMPTS: [usize; 3] = [8_192, 32_768, 131_072];

const LONG_WARM_REQUESTS: usize = 1;
const LONG_MEASURED_REQUESTS: usize = 2;
const LONG_GENERATED_TOKENS: usize = 8;

#[derive(Clone, Copy)]
struct RequestSample {
    telemetry: Qwen38FlashNextGenerationTelemetry,
    time_to_first_token: Duration,
}

/// One measured prompt shape.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextGenerationRouteReport {
    /// Prompt tokens the requests carried.
    pub prompt_tokens: usize,
    /// Tokens each request generated.
    pub generated_tokens: usize,
    /// Prompt tokens the native prefill tiles carried.
    pub native_prefill_tokens: usize,
    /// Prefill tiles and scalar rounds the prime replayed.
    pub prime_tiles: usize,
    /// Single-row rounds the prime's scalar tail replayed.
    pub prime_scalar_rounds: usize,
    /// Median time to first token.
    pub median_ttft: Duration,
    /// Fastest time to first token, which bounds how much of the median is host waiting.
    pub fastest_ttft: Duration,
    /// Median milliseconds per generated token during decode.
    pub median_decode_ms_per_token: f64,
    /// Tokens per second the decode rounds sustained at the median.
    pub decode_tokens_per_second: f64,
    /// Expert hit rate over the decode rounds alone.
    pub decode_hit_rate: f64,
    /// Host-to-device expert bytes one generated token cost during decode.
    pub decode_h2d_bytes_per_token: f64,
    /// Whole-request expert hit rate, prefill included.
    pub request_hit_rate: f64,
    /// Layer rounds that blocked on the streaming publication fence.
    pub publication_stalls: usize,
}

/// Everything the generation sweep observed.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextGenerationBenchmarkReport {
    /// Wall time loading the whole model took.
    pub load: Duration,
    /// Captured executables the program retains.
    pub executables: usize,
    /// Longest sequence a request may reach.
    pub generation_capacity: usize,
    /// Measured prompt shapes, ascending.
    pub routes: Vec<Qwen38FlashNextGenerationRouteReport>,
    /// The eight committed captures replayed end to end, as one real-traffic total.
    pub real_traffic: Qwen38FlashNextGenerationTelemetry,
    /// Tokens the real-traffic pass generated.
    pub real_traffic_tokens: usize,
    /// The same captures on the host-stalled reference route.
    pub stalling_traffic: Qwen38FlashNextGenerationTelemetry,
    /// Tokens the host-stalled pass generated.
    pub stalling_traffic_tokens: usize,
    /// Minimum, median, and maximum SM clocks during the measured sweep.
    pub sm_clocks_mhz: (u32, u32, u32),
    /// Minimum, median, and maximum memory clocks during the measured sweep.
    pub memory_clocks_mhz: (u32, u32, u32),
    /// Telemetry samples captured during the measured sweep.
    pub telemetry_samples: usize,
    /// Whether the measured clocks satisfy the comparable-run policy.
    pub clock_comparable: bool,
}

/// Diagnostic selection and wall controls for the generation sweep.
#[derive(Clone, Debug, Default)]
pub struct Qwen38FlashNextGenerationBenchmarkOptions {
    /// Exact prompt lengths retained; all admitted lengths are the default.
    pub prompt_tokens: Option<Vec<usize>>,
    /// Optional wall limit for one prompt cell.
    pub max_cell_wall: Option<Duration>,
}

/// Loads the model once and sweeps whole generation requests.
pub fn benchmark_qwen38_flash_next_generation(
    root: &Path,
) -> Result<Qwen38FlashNextGenerationBenchmarkReport, DeviceBenchmarkError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let started = std::time::Instant::now();
    let mut generator = Qwen38FlashNextTextGenerator::from_snapshot_device_zero(snapshot)?;
    let load = started.elapsed();

    benchmark_qwen38_flash_next_generation_loaded(
        &mut generator,
        load,
        &Qwen38FlashNextGenerationBenchmarkOptions::default(),
    )
}

/// Sweeps whole requests on an already-constructed generator.
pub fn benchmark_qwen38_flash_next_generation_loaded(
    generator: &mut Qwen38FlashNextTextGenerator,
    load: Duration,
    options: &Qwen38FlashNextGenerationBenchmarkOptions,
) -> Result<Qwen38FlashNextGenerationBenchmarkReport, DeviceBenchmarkError> {
    validate_prompt_selection(options.prompt_tokens.as_deref())?;

    let boundary =
        load_qwen38_flash_next_golden_boundary(&qwen38_flash_next_golden_directory(), 2_100)
            .map_err(|error| DeviceBenchmarkError::Precondition(error.to_string()))?;
    let loaded_prompt = repeated_prompt(&boundary.prompt_ids, LONG_PROMPTS[0]);
    for _ in 0..LONG_WARM_REQUESTS {
        run_request(generator, &loaded_prompt, LONG_GENERATED_TOKENS)?;
    }
    device_benchmark::require_current_process_exclusive()?;
    device_benchmark::validate_loaded_host_clock_policy(
        "qwen3_8_flash_next/generation/request",
        || run_request(generator, &loaded_prompt, LONG_GENERATED_TOKENS).map(|_| ()),
    )?;

    let sampler = device_benchmark::TelemetrySampler::start();
    let mut routes = Vec::with_capacity(SWEPT_PROMPTS.len() + LONG_PROMPTS.len());
    let mut previous_route = None;
    for prompt_tokens in SWEPT_PROMPTS
        .into_iter()
        .filter(|prompt| prompt_selected(*prompt, options.prompt_tokens.as_deref()))
    {
        let estimate = scaled_request_estimate(previous_route, prompt_tokens);
        let report = measure_prompt(generator, prompt_tokens, options.max_cell_wall, estimate)?;
        print_route(&report);
        previous_route = Some((prompt_tokens, report.median_ttft));
        routes.push(report);
    }
    for prompt_tokens in LONG_PROMPTS
        .into_iter()
        .filter(|prompt| prompt_selected(*prompt, options.prompt_tokens.as_deref()))
    {
        let estimate = scaled_request_estimate(previous_route, prompt_tokens);
        let report = measure_long_prompt(
            generator,
            &boundary.prompt_ids,
            prompt_tokens,
            options.max_cell_wall,
            estimate,
        )?;
        print_route(&report);
        previous_route = Some((prompt_tokens, report.median_ttft));
        routes.push(report);
    }

    let (stalling_traffic, stalling_traffic_tokens, real_traffic, real_traffic_tokens) =
        if options.prompt_tokens.is_some() {
            (
                Qwen38FlashNextGenerationTelemetry::default(),
                0,
                Qwen38FlashNextGenerationTelemetry::default(),
                0,
            )
        } else {
            generator.set_streaming_route(Qwen38FlashNextStreamingRoute::Stalling);
            let (stalling, stalling_tokens) = measure_real_traffic(generator)?;
            generator.set_streaming_route(Qwen38FlashNextStreamingRoute::Overlapped);
            let (overlap, overlap_tokens) = measure_real_traffic(generator)?;
            (stalling, stalling_tokens, overlap, overlap_tokens)
        };
    let clocks = sampler.finish()?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen38FlashNextGenerationBenchmarkReport {
        load,
        executables: generator.executables(),
        generation_capacity: generator.context_capacity(),
        routes,
        real_traffic,
        real_traffic_tokens,
        stalling_traffic,
        stalling_traffic_tokens,
        sm_clocks_mhz: (
            clocks.sm_minimum_mhz,
            clocks.sm_median_mhz,
            clocks.sm_maximum_mhz,
        ),
        memory_clocks_mhz: (
            clocks.memory_minimum_mhz,
            clocks.memory_median_mhz,
            clocks.memory_maximum_mhz,
        ),
        telemetry_samples: clocks.samples,
        clock_comparable: clocks.clock_comparable,
    })
}

fn measure_prompt(
    generator: &mut Qwen38FlashNextTextGenerator,
    prompt_tokens: usize,
    max_cell_wall: Option<Duration>,
    estimated_request: Option<Duration>,
) -> Result<Qwen38FlashNextGenerationRouteReport, DeviceBenchmarkError> {
    measure_requests(
        generator,
        synthetic_prompt(prompt_tokens),
        WARM_REQUESTS,
        MEASURED_REQUESTS,
        GENERATED_TOKENS,
        max_cell_wall,
        estimated_request,
    )
}

fn measure_long_prompt(
    generator: &mut Qwen38FlashNextTextGenerator,
    seed: &[u32],
    prompt_tokens: usize,
    max_cell_wall: Option<Duration>,
    estimated_request: Option<Duration>,
) -> Result<Qwen38FlashNextGenerationRouteReport, DeviceBenchmarkError> {
    measure_requests(
        generator,
        repeated_prompt(seed, prompt_tokens),
        LONG_WARM_REQUESTS,
        LONG_MEASURED_REQUESTS,
        LONG_GENERATED_TOKENS,
        max_cell_wall,
        estimated_request,
    )
}

fn measure_requests(
    generator: &mut Qwen38FlashNextTextGenerator,
    prompt: Vec<u32>,
    warm_requests: usize,
    measured_requests: usize,
    generated_tokens: usize,
    max_cell_wall: Option<Duration>,
    mut estimated_request: Option<Duration>,
) -> Result<Qwen38FlashNextGenerationRouteReport, DeviceBenchmarkError> {
    let prompt_tokens = prompt.len();
    let budget = CellWallBudget::new(format!("generation:T={prompt_tokens}"), max_cell_wall);
    for _ in 0..warm_requests {
        budget.admit_next(estimated_request)?;
        let started = std::time::Instant::now();
        run_request(generator, &prompt, generated_tokens)?;
        estimated_request = Some(started.elapsed());
    }

    let mut samples = Vec::with_capacity(measured_requests);
    for _ in 0..measured_requests {
        budget.admit_next(estimated_request)?;
        let started = std::time::Instant::now();
        samples.push(run_request(generator, &prompt, generated_tokens)?);
        estimated_request = Some(started.elapsed());
    }

    let mut times = samples
        .iter()
        .map(|sample| sample.time_to_first_token)
        .collect::<Vec<_>>();
    times.sort_unstable();
    let mut per_token = samples
        .iter()
        .map(|sample| sample.telemetry.decode_ms_per_token())
        .collect::<Vec<_>>();
    per_token.sort_by(f64::total_cmp);

    let last = samples
        .last()
        .copied()
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition(
                "generation benchmark produced no measured requests".to_string(),
            )
        })?
        .telemetry;

    let median_decode_ms_per_token = per_token[per_token.len() / 2];
    let (native_prefill_tokens, _, _) = prompt_plan(prompt_tokens);

    Ok(Qwen38FlashNextGenerationRouteReport {
        prompt_tokens,
        generated_tokens,
        native_prefill_tokens,
        prime_tiles: last.prime_tiles(),
        prime_scalar_rounds: last.prime_scalar_rounds(),
        median_ttft: times[times.len() / 2],
        fastest_ttft: times[0],
        median_decode_ms_per_token,
        decode_tokens_per_second: if median_decode_ms_per_token == 0.0 {
            0.0
        } else {
            1_000.0 / median_decode_ms_per_token
        },
        decode_hit_rate: last.decode_expert_hit_rate(),
        decode_h2d_bytes_per_token: last.decode_expert_h2d_bytes_per_token(),
        request_hit_rate: last.expert_hit_rate(),
        publication_stalls: last.publication_stalls(),
    })
}

fn prompt_selected(prompt: usize, selected: Option<&[usize]>) -> bool {
    selected.is_none_or(|selected| selected.contains(&prompt))
}

fn validate_prompt_selection(selected: Option<&[usize]>) -> Result<(), DeviceBenchmarkError> {
    let Some(selected) = selected else {
        return Ok(());
    };
    if selected.is_empty() {
        return Err(DeviceBenchmarkError::Precondition(
            "generation cell selection is empty".to_string(),
        ));
    }
    for &prompt in selected {
        if !SWEPT_PROMPTS.contains(&prompt) && !LONG_PROMPTS.contains(&prompt) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "generation prompt T={prompt} is not admitted by this sweep"
            )));
        }
    }

    Ok(())
}

fn scaled_request_estimate(
    previous: Option<(usize, Duration)>,
    prompt_tokens: usize,
) -> Option<Duration> {
    let (previous_tokens, duration) = previous?;
    duration.checked_mul(u32::try_from(prompt_tokens.div_ceil(previous_tokens)).ok()?)
}

fn run_request(
    generator: &mut Qwen38FlashNextTextGenerator,
    prompt: &[u32],
    generated_tokens: usize,
) -> Result<RequestSample, DeviceBenchmarkError> {
    let run = generator.qualification_generate_from_tokens(
        prompt,
        generated_tokens,
        SamplingOptions::greedy(),
        0,
    )?;
    if run.token_ids.len() != generated_tokens {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "generation stopped after {} of {generated_tokens} benchmark tokens",
            run.token_ids.len()
        )));
    }
    let time_to_first_token = run.time_to_first_token.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "generation benchmark produced no first-token timing".to_string(),
        )
    })?;
    let telemetry = generator.telemetry();
    validate_request_telemetry(
        prompt.len(),
        generated_tokens,
        run.native_prefill_tokens,
        telemetry,
    )?;

    Ok(RequestSample {
        telemetry,
        time_to_first_token,
    })
}

fn prompt_plan(tokens: usize) -> (usize, usize, usize) {
    let mut remaining = tokens;
    let mut native = 0usize;
    let mut tiles = 0usize;
    for width in [1_024, 128, 64, 32] {
        while remaining >= width {
            native += width;
            remaining -= width;
            tiles += 1;
        }
    }

    (native, tiles, remaining)
}

fn validate_request_telemetry(
    prompt_tokens: usize,
    generated_tokens: usize,
    native_prefill_tokens: usize,
    telemetry: Qwen38FlashNextGenerationTelemetry,
) -> Result<(), DeviceBenchmarkError> {
    validate_request_telemetry_for_tokens(
        prompt_tokens,
        generated_tokens,
        native_prefill_tokens,
        telemetry,
    )
}

fn validate_request_telemetry_for_tokens(
    prompt_tokens: usize,
    generated_tokens: usize,
    native_prefill_tokens: usize,
    telemetry: Qwen38FlashNextGenerationTelemetry,
) -> Result<(), DeviceBenchmarkError> {
    let (native, tiles, scalar) = prompt_plan(prompt_tokens);
    let decode = generated_tokens.saturating_sub(1);
    let rows = prompt_tokens + decode;
    let expert_requests =
        rows * <Qwen38FlashNext as Arch>::LAYERS * Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
    let decode_expert_requests =
        decode * <Qwen38FlashNext as Arch>::LAYERS * Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
    let expected = [
        (native_prefill_tokens, native, "native prefill tokens"),
        (telemetry.prime_rows(), prompt_tokens, "prime rows"),
        (telemetry.prime_tiles(), tiles, "prime tiles"),
        (telemetry.prime_scalar_rounds(), scalar, "scalar rounds"),
        (telemetry.prime_rounds(), tiles + scalar, "prime rounds"),
        (telemetry.decode_rounds(), decode, "decode rounds"),
        (
            telemetry.expert_requests(),
            expert_requests,
            "expert requests",
        ),
        (
            telemetry.decode_expert_requests(),
            decode_expert_requests,
            "decode expert requests",
        ),
        (
            telemetry.embedding_h2d_bytes(),
            rows * <Qwen38FlashNext as Arch>::HIDDEN * size_of::<u16>(),
            "embedding bytes",
        ),
        (
            telemetry.engram_h2d_bytes(),
            rows * Qwen38FlashNext::PLE_EMBED_DIM,
            "engram bytes",
        ),
        (
            telemetry.engram_rows(),
            rows * Qwen38FlashNext::NGRAM_HEADS,
            "engram rows",
        ),
        (
            telemetry.kv_append_bytes(),
            expected_kv_append_bytes(rows),
            "cache append bytes",
        ),
    ];
    for (actual, expected, name) in expected {
        if actual != expected {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "{name} counted {actual}, expected {expected}"
            )));
        }
    }

    Ok(())
}

fn expected_kv_append_bytes(rows: usize) -> usize {
    let per_token = QWEN38_FLASH_NEXT_ATTENTION_LAYERS
        * (2 * <Qwen38FlashNext as Arch>::NUM_KV_HEADS * <Qwen38FlashNext as Arch>::HEAD_DIM
            + Qwen38FlashNext::INDEXER_HEAD_DIM * size_of::<u16>());
    let per_block =
        QWEN38_FLASH_NEXT_ATTENTION_LAYERS * Qwen38FlashNext::INDEXER_HEAD_DIM * size_of::<u16>();

    rows * per_token + rows / Qwen38FlashNext::INDEXER_COMPRESS_RATIO * per_block
}

/// Replays the committed captures from a reset expert cache.
fn measure_real_traffic(
    generator: &mut Qwen38FlashNextTextGenerator,
) -> Result<(Qwen38FlashNextGenerationTelemetry, usize), DeviceBenchmarkError> {
    let stream = Arc::clone(generator.qualification_stream());
    generator.qualification_program_mut().reset_state(&stream)?;
    let directory = qwen38_flash_next_golden_directory();
    let mut total = Qwen38FlashNextGenerationTelemetry::default();
    let mut tokens = 0usize;

    for stem in crate::qwen38_flash_next_golden::QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
        let capture = load_qwen38_flash_next_golden_capture(&directory, stem)
            .map_err(|error| DeviceBenchmarkError::Precondition(error.to_string()))?;
        let run = generator.qualification_generate_from_tokens(
            &capture.prompt_ids,
            capture.generated_ids.len(),
            SamplingOptions::greedy(),
            0,
        )?;
        if run.token_ids.len() != capture.generated_ids.len() {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "{stem} stopped after {} of {} traffic tokens",
                run.token_ids.len(),
                capture.generated_ids.len()
            )));
        }
        validate_request_telemetry_for_tokens(
            capture.prompt_ids.len(),
            capture.generated_ids.len(),
            run.native_prefill_tokens,
            generator.telemetry(),
        )?;
        tokens += run.token_ids.len();
        total.absorb(generator.telemetry());
    }

    Ok((total, tokens))
}

/// A prompt of `tokens` distinct ids, so the routers see a stream that does not repeat itself.
fn synthetic_prompt(tokens: usize) -> Vec<u32> {
    (0..tokens as u32)
        .map(|index| 2_048 + index.wrapping_mul(97) % 200_000)
        .collect()
}

fn repeated_prompt(seed: &[u32], tokens: usize) -> Vec<u32> {
    seed.iter().copied().cycle().take(tokens).collect()
}

fn print_route(report: &Qwen38FlashNextGenerationRouteReport) {
    println!(
        "  {:>6} {:>4} {:>12.2?} {:>12.2?} {:>11.3} {:>9.2} {:>7.4} {:>13.0} {:>6}",
        report.prompt_tokens,
        report.generated_tokens,
        report.median_ttft,
        report.fastest_ttft,
        report.median_decode_ms_per_token,
        report.decode_tokens_per_second,
        report.decode_hit_rate,
        report.decode_h2d_bytes_per_token,
        report.publication_stalls
    );
}

/// Prints one sweep in the house's diagnostic shape.
pub fn print_qwen38_flash_next_generation_benchmark(
    report: &Qwen38FlashNextGenerationBenchmarkReport,
) {
    println!("Qwen3.8 Flash-Next generation sweep");
    println!("  diagnostic only; no performance baseline is admitted");
    println!(
        "  clocks                  {}",
        if report.clock_comparable {
            "controlled"
        } else {
            "uncontrolled; diagnostic only"
        }
    );
    println!("  construction");
    println!("    total                  {:?}", report.load);
    println!("    executables            {}", report.executables);
    println!("    generation capacity    {}", report.generation_capacity);
    println!(
        "    measured SM clocks     {}..{} MHz (median {}, {} samples)",
        report.sm_clocks_mhz.0,
        report.sm_clocks_mhz.2,
        report.sm_clocks_mhz.1,
        report.telemetry_samples
    );
    println!(
        "    measured memory clocks {}..{} MHz (median {})",
        report.memory_clocks_mhz.0, report.memory_clocks_mhz.2, report.memory_clocks_mhz.1
    );
    println!(
        "  {:>6} {:>4} {:>12} {:>12} {:>11} {:>9} {:>7} {:>13} {:>6}",
        "prompt",
        "new",
        "median ttft",
        "fastest",
        "ms/token",
        "tok/s",
        "hit",
        "h2d/token",
        "pubwait"
    );
    for route in &report.routes {
        print_route(route);
    }
    if report.real_traffic_tokens == 0 {
        println!("  real traffic omitted by the diagnostic cell filter");
        return;
    }
    println!("  real traffic: eight unrelated captures from a reset cache");
    println!("    tokens generated       {}", report.real_traffic_tokens);
    println!(
        "    decode expert hit      {:.4}",
        report.real_traffic.decode_expert_hit_rate()
    );
    println!(
        "    decode h2d/token       {:.0} B",
        report.real_traffic.decode_expert_h2d_bytes_per_token()
    );
    println!(
        "    whole-request hit      {:.4}",
        report.real_traffic.expert_hit_rate()
    );
    println!(
        "    decode ms/token        {:.3}",
        report.real_traffic.decode_ms_per_token()
    );
    println!(
        "    prime tiles / scalar   {} / {}",
        report.real_traffic.prime_tiles(),
        report.real_traffic.prime_scalar_rounds()
    );
    println!("  real traffic by publication route");
    println!("    route       tokens  ms/token  residency ms  stalled  in flight");
    for (name, traffic, tokens) in [
        (
            "stalling",
            report.stalling_traffic,
            report.stalling_traffic_tokens,
        ),
        ("overlap", report.real_traffic, report.real_traffic_tokens),
    ] {
        println!(
            "    {name:<10} {tokens:>6} {:>9.3} {:>13.3} {:>8} {:>10}",
            traffic.decode_ms_per_token(),
            traffic.residency_wait().as_secs_f64() * 1_000.0,
            traffic.publication_stalls(),
            traffic.overlapped_rounds()
        );
    }
    println!(
        "    engram rows hashed     {}",
        report.real_traffic.engram_rows()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_flash_next_generation_benchmark_accounting_matches_each_swept_route() {
        for prompt in SWEPT_PROMPTS {
            let (native, tiles, scalar) = prompt_plan(prompt);
            assert_eq!(native + scalar, prompt);
            assert!(scalar < 32);
            assert_eq!(tiles + scalar, expected_prime_rounds(prompt));
        }

        let rows = SWEPT_PROMPTS[0] + GENERATED_TOKENS - 1;
        assert_eq!(
            expected_kv_append_bytes(rows),
            rows * 15_360 + rows / 4 * 3_072
        );
        for prompt in LONG_PROMPTS {
            let (native, tiles, scalar) = prompt_plan(prompt);
            assert_eq!((native, scalar), (prompt, 0));
            assert_eq!(tiles, prompt / 1_024);
        }
        assert_eq!(
            repeated_prompt(&[7, 11, 13], 8),
            [7, 11, 13, 7, 11, 13, 7, 11]
        );
        assert_eq!(WARM_REQUESTS, 3);
        assert_eq!(MEASURED_REQUESTS, 4);
    }

    fn expected_prime_rounds(tokens: usize) -> usize {
        let (_, tiles, scalar) = prompt_plan(tokens);
        tiles + scalar
    }
}
