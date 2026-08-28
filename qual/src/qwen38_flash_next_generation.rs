//! Source-backed Qwen3.8 Flash-Next generation qualification.
//!
//! Represented-value oracles remain the numerical authority. Pinned SGLang captures add an
//! external leading-pair selection cross-check and exact dense-band boundary checks.

use crate::device_benchmark::{self, DeviceBenchmarkError};
use crate::qwen38_flash_next_golden::{
    QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES, QWEN38_FLASH_NEXT_GOLDEN_PROMPTS,
    Qwen38FlashNextGoldenCapture, Qwen38FlashNextGoldenError,
    load_qwen38_flash_next_golden_boundary, load_qwen38_flash_next_golden_capture,
    load_qwen38_flash_next_golden_meta, qwen38_flash_next_golden_directory,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, EngineErrorCode, FinishReason, MAX_BATCH,
    QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, Qwen38FlashNextGenerationTelemetry,
    Qwen38FlashNextSlotState, Qwen38FlashNextTextGenerator, SamplingOptions,
};
use tuisko_frontend::ChatMessage;
use tuisko_gpu::GpuError;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

/// Ranked candidates the gate retains per step, so a divergence can be read rather than guessed.
const RECORDED_CANDIDATES: usize = 8;

/// Tokens each boundary capture asks the reference for.
const BOUNDARY_TOKENS: usize = 8;

/// Failure of the source-backed Flash-Next generation gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),

    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The committed capture set could not be read or admitted.
    #[error(transparent)]
    Golden(#[from] Qwen38FlashNextGoldenError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Generation disagreed with a contract it is required to hold.
    #[error("Flash-Next generation qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextGenerationQualificationError>;

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextGenerationQualificationError {
    Qwen38FlashNextGenerationQualificationError::Mismatch(message.into())
}

/// Teacher-forced and free-running results for one capture.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextCaptureVerdict {
    /// Capture stem.
    pub name: String,
    /// Prompt tokens the capture carries.
    pub prompt_tokens: usize,
    /// Steps scored on the reference's own context.
    pub scored_steps: usize,
    /// Scored steps whose argmax matched the reference's token.
    pub agreed_steps: usize,
    /// Scored steps where the reference's own top two were exactly tied.
    pub tied_steps: Vec<usize>,
    /// Tied steps where we selected the reference's *other* tied candidate.
    pub tied_transpositions: Vec<usize>,
    /// Steps where the local selection fell outside the reference leading pair.
    pub decisive_disagreements: Vec<usize>,
    /// Runner-up selections as `(step, local margin, reference margin)`.
    pub runner_up: Vec<(usize, f64, f64)>,
    /// Smallest nonzero reference margin at which we still agreed, in nats.
    pub tightest_agreement: Option<f64>,
    /// Mean reference top-1 margin, in nats.
    pub mean_margin: Option<f64>,
    /// Tokens a free-running generation matched before its first divergence.
    pub free_running_tokens: usize,
    /// Whether the free-running run matched the whole capture.
    pub free_running_complete: bool,
    /// The first free-running divergence, diagnosed.
    pub free_running_diagnosis: Option<String>,
    /// Prompt tokens the native prefill tiles carried.
    pub native_prefill_tokens: usize,
    /// Whole-request evidence for this capture.
    pub telemetry: Qwen38FlashNextGenerationTelemetry,
}

/// A boundary capture's admitted budget and what happened at it.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextBoundaryVerdict {
    /// Keys the capture's first decode round attends.
    pub visible: usize,
    /// Prompt tokens the capture carries.
    pub prompt_tokens: usize,
    /// Tokens the dense band admits for this prompt.
    pub admitted_tokens: usize,
    /// Steps whose argmax matched the reference on its own context.
    pub agreed_tokens: usize,
    /// Tokens a free-running production run reproduced.
    pub free_running_tokens: usize,
    /// Whether the capture's own eight-token request was refused.
    pub full_request_refused: bool,
    /// Whether one more token than the band admits was refused.
    pub over_budget_refused: bool,
    /// The refusal's own words, when there was one.
    pub refusal: Option<String>,
}

/// Everything the generation gate observed.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextGenerationQualification {
    /// One line naming the external authority the tokens were scored against.
    pub provenance: String,
    /// Wall time loading the whole model took.
    pub load: Duration,
    /// Captured executables the program retains.
    pub executables: usize,
    /// Longest sequence a request may reach.
    pub generation_capacity: usize,
    /// Steps scored against the reference across every prompt capture.
    pub compared_tokens: usize,
    /// Steps whose argmax matched the reference on its own context.
    pub agreed_tokens: usize,
    /// Steps where the reference's own top two were exactly tied.
    pub tied_steps: usize,
    /// Tied steps where we selected the reference's other tied candidate.
    pub tied_transpositions: usize,
    /// Steps outside the reference leading pair. Zero, or the gate failed.
    pub decisive_disagreements: usize,
    /// Steps where we selected the reference's runner-up rather than its own choice.
    pub unresolved_steps: usize,
    /// Widest reference margin among those steps, in nats.
    pub widest_unresolved_margin: f64,
    /// Smallest nonzero reference margin at which we still agreed, in nats.
    pub tightest_agreement: Option<f64>,
    /// Captures a free-running production run reproduced completely.
    pub free_running_complete: usize,
    /// Per-capture verdicts, in file order.
    pub captures: Vec<Qwen38FlashNextCaptureVerdict>,
    /// Per-boundary verdicts, ascending.
    pub boundaries: Vec<Qwen38FlashNextBoundaryVerdict>,
    /// Exact tiling and published-row checks completed.
    pub prefill_checks: usize,
    /// Logit values examined by the prefill checks and diagnostics.
    pub tiling_compared_logits: usize,
    /// Sampled runs compared at an identical seed.
    pub seeded_agreements: usize,
    /// Whether two different seeds produced different token streams.
    pub seeds_separate: bool,
    /// Tokens a mid-generation snapshot and restore reproduced exactly.
    pub rollback_tokens: usize,
    /// Host bytes one slot's restore point holds.
    pub rollback_snapshot_bytes: usize,
    /// Logit values compared between a clean-page and a recycled-page run.
    pub recycled_page_compared_logits: usize,
    /// Pages the lifecycle sweep took and returned.
    pub lifecycle_pages: usize,
    /// Engram rows the host hash addressed while replaying the captures.
    pub engram_rows: usize,
    /// Whole-request telemetry over every capture replayed.
    pub telemetry: Qwen38FlashNextGenerationTelemetry,
    /// Expert hit rate over decode rounds alone, on real token streams.
    pub decode_hit_rate: f64,
    /// Host-to-device expert bytes one generated token cost during decode.
    pub decode_h2d_bytes_per_token: f64,
}

/// Loads the model and scores it against the committed captures.
pub fn qualify_qwen38_flash_next_generation(
    root: &Path,
) -> QualResult<Qwen38FlashNextGenerationQualification> {
    let _preflight = device_benchmark::preflight()?;
    let directory = qwen38_flash_next_golden_directory();
    let meta = load_qwen38_flash_next_golden_meta(&directory)?;
    meta.require_pinned_authority(A::MODEL_ID, A::REVISION)?;
    println!("--- external authority ---");
    println!("  {}", meta.provenance());
    println!("  captures                 {}", directory.display());

    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let started = std::time::Instant::now();
    let mut generator = Qwen38FlashNextTextGenerator::from_snapshot_device_zero(snapshot)?;
    let load = started.elapsed();
    let generation_capacity = generator.context_capacity();
    if generation_capacity != QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
        return Err(mismatch(format!(
            "generation admits {generation_capacity} tokens, expected the proven dense band \
             {QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING}; a generator that admits the funded cache \
             depth would run dense attention outside the band the reference computes"
        )));
    }
    println!("--- construction (diagnostic, nothing blessed) ---");
    println!("  total                    {load:?}");
    println!("  executables              {}", generator.executables());
    println!("  generation capacity      {generation_capacity}");

    // Run structural checks before the capture matrix.
    let (prefill_checks, tiling_compared_logits) = verify_prefill_publication(&mut generator)?;
    let (seeded_agreements, seeds_separate) = verify_seeded_sampling(&mut generator)?;
    let (lifecycle_pages, recycled_page_compared_logits) = verify_slot_lifecycle(&mut generator)?;
    let (rollback_tokens, rollback_snapshot_bytes) = verify_rollback(&mut generator)?;
    verify_eos_and_usage(&mut generator)?;

    let mut captures = Vec::with_capacity(QWEN38_FLASH_NEXT_GOLDEN_PROMPTS.len());
    let mut compared_tokens = 0usize;
    let mut agreed_tokens = 0usize;
    let mut tied_steps = 0usize;
    let mut tied_transpositions = 0usize;
    let mut decisive_disagreements = 0usize;
    let mut free_running_complete = 0usize;
    let mut unresolved_steps = 0usize;
    let mut widest_unresolved_margin = 0.0f64;
    let mut tightest_agreement: Option<f64> = None;
    let mut telemetry = Qwen38FlashNextGenerationTelemetry::default();
    let mut engram_rows = 0usize;
    for stem in QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
        let capture = load_qwen38_flash_next_golden_capture(&directory, stem)?;
        let verdict = replay_capture(&mut generator, stem, &capture, capture.generated_ids.len())?;
        compared_tokens += verdict.scored_steps;
        agreed_tokens += verdict.agreed_steps;
        tied_steps += verdict.tied_steps.len();
        tied_transpositions += verdict.tied_transpositions.len();
        decisive_disagreements += verdict.decisive_disagreements.len();
        free_running_complete += usize::from(verdict.free_running_complete);
        unresolved_steps += verdict.runner_up.len();
        for &(_, _, reference) in &verdict.runner_up {
            if reference.is_finite() {
                widest_unresolved_margin = widest_unresolved_margin.max(reference);
            }
        }
        if let Some(margin) = verdict.tightest_agreement {
            tightest_agreement =
                Some(tightest_agreement.map_or(margin, |best: f64| best.min(margin)));
        }
        engram_rows += verdict.telemetry.engram_rows();
        telemetry = fold(telemetry, verdict.telemetry);
        print_capture_verdict(&verdict);
        captures.push(verdict);
    }
    let classified =
        agreed_tokens + tied_transpositions + unresolved_steps + decisive_disagreements;
    if classified != compared_tokens {
        return Err(mismatch(format!(
            "classified {classified} of {compared_tokens} external selection steps"
        )));
    }
    if decisive_disagreements != 0 {
        return Err(mismatch(format!(
            "{decisive_disagreements} external selections fell outside the reference leading pair"
        )));
    }

    let mut boundaries = Vec::with_capacity(QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES.len());
    for visible in QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES {
        let verdict = replay_boundary(&mut generator, &directory, visible)?;
        print_boundary_verdict(&verdict);
        boundaries.push(verdict);
    }

    let decode_hit_rate = telemetry.decode_expert_hit_rate();
    let decode_h2d_bytes_per_token = telemetry.decode_expert_h2d_bytes_per_token();

    Ok(Qwen38FlashNextGenerationQualification {
        provenance: meta.provenance(),
        load,
        executables: generator.executables(),
        generation_capacity,
        compared_tokens,
        agreed_tokens,
        tied_steps,
        tied_transpositions,
        decisive_disagreements,
        unresolved_steps,
        tightest_agreement,
        free_running_complete,
        widest_unresolved_margin,
        captures,
        boundaries,
        prefill_checks,
        tiling_compared_logits,
        seeded_agreements,
        seeds_separate,
        rollback_tokens,
        rollback_snapshot_bytes,
        recycled_page_compared_logits,
        lifecycle_pages,
        engram_rows,
        telemetry,
        decode_hit_rate,
        decode_h2d_bytes_per_token,
    })
}

/// Scores teacher-forced selections and diagnoses the free-running fork point.
fn replay_capture(
    generator: &mut Qwen38FlashNextTextGenerator,
    name: &str,
    capture: &Qwen38FlashNextGoldenCapture,
    tokens: usize,
) -> QualResult<Qwen38FlashNextCaptureVerdict> {
    let expected = &capture.generated_ids[..tokens];
    let rows = generator.qualification_score_forced_tokens(
        &capture.prompt_ids,
        expected,
        RECORDED_CANDIDATES,
    )?;
    if rows.len() != tokens {
        return Err(mismatch(format!(
            "{name} scored {} steps for {tokens} forced tokens",
            rows.len()
        )));
    }

    let mut agreed_steps = 0usize;
    let mut tied_steps = Vec::new();
    let mut tied_transpositions = Vec::new();
    let mut decisive_disagreements = Vec::new();
    let mut runner_up = Vec::new();
    let mut tightest_agreement: Option<f64> = None;
    for (index, row) in rows.iter().enumerate() {
        let step = &capture.steps[index];
        let (ours, our_logit) = row
            .first()
            .copied()
            .ok_or_else(|| mismatch(format!("{name} step {index} scored an empty row")))?;
        let theirs = expected[index];
        let margin = step.top_margin();
        if margin == Some(0.0) {
            tied_steps.push(index);
        }
        if ours == theirs {
            agreed_steps += 1;
            if let Some(margin) = margin.filter(|margin| *margin > 0.0) {
                tightest_agreement =
                    Some(tightest_agreement.map_or(margin, |best: f64| best.min(margin)));
            }
            continue;
        }

        if step.within_leading_pair(ours) {
            let our_margin = row
                .get(1)
                .map_or(f64::INFINITY, |&(_, next)| f64::from(our_logit - next));
            if margin == Some(0.0) {
                tied_transpositions.push(index);
            } else {
                runner_up.push((index, our_margin, margin.unwrap_or(f64::NAN)));
            }
        } else {
            decisive_disagreements.push(index);
            println!(
                "{}",
                diagnose_step(name, capture, index, row, "teacher-forced")
            );
        }
    }

    let margins = capture.steps[..tokens]
        .iter()
        .filter_map(|step| step.top_margin())
        .collect::<Vec<_>>();

    let run = generator.qualification_generate_from_tokens(
        &capture.prompt_ids,
        tokens,
        SamplingOptions::greedy(),
        RECORDED_CANDIDATES,
    )?;
    let telemetry = generator.telemetry();
    let free_running_tokens = run
        .token_ids
        .iter()
        .zip(expected)
        .take_while(|(ours, theirs)| ours == theirs)
        .count();
    let free_running_complete = free_running_tokens == tokens;
    let free_running_diagnosis = (!free_running_complete).then(|| {
        run.steps.get(free_running_tokens).map_or_else(
            || format!("{name} free-running ended at {free_running_tokens} of {tokens}"),
            |row| diagnose_step(name, capture, free_running_tokens, row, "free-running"),
        )
    });

    Ok(Qwen38FlashNextCaptureVerdict {
        name: name.to_string(),
        prompt_tokens: capture.prompt_ids.len(),
        scored_steps: tokens,
        agreed_steps,
        tied_steps,
        tied_transpositions,
        decisive_disagreements,
        runner_up,
        tightest_agreement,
        mean_margin: (!margins.is_empty())
            .then(|| margins.iter().sum::<f64>() / margins.len() as f64),
        free_running_tokens,
        free_running_complete,
        free_running_diagnosis,
        native_prefill_tokens: run.native_prefill_tokens,
        telemetry,
    })
}

/// Diagnoses one selection difference with both ranked candidate lists.
fn diagnose_step(
    name: &str,
    capture: &Qwen38FlashNextGoldenCapture,
    index: usize,
    ours: &[(u32, f32)],
    pass: &str,
) -> String {
    let mut report = format!("{name} disagreed with the reference at {pass} step {index}.\n");
    let Some(&theirs) = capture.generated_ids.get(index) else {
        report.push_str("  the reference recorded no token at that position\n");
        return report;
    };
    let mine = ours.first().map(|&(token, _)| token);
    report.push_str(&format!(
        "  reference chose {theirs}, we chose {}\n",
        mine.map_or_else(|| "nothing".to_string(), |token| token.to_string())
    ));

    if let Some(step) = capture.steps.get(index) {
        report.push_str("  reference ranked candidates (token, logprob):\n");
        for (rank, (&token, &logprob)) in step
            .top_ids
            .iter()
            .zip(&step.top_logprobs)
            .take(RECORDED_CANDIDATES)
            .enumerate()
        {
            report.push_str(&format!("    {rank}. {token:>7}  {logprob:+.6}\n"));
        }
        match step.top_margin() {
            Some(0.0) => report.push_str(
                "  the reference's top two are EXACTLY TIED, so its token was decided by its \
                 tie-break rule and not by its arithmetic\n",
            ),
            Some(margin) => report.push_str(&format!(
                "  the reference's top-1 margin here is {margin:+.6} nats, so a disagreement \
                 needs a real difference in the logits rather than rounding\n"
            )),
            None => {}
        }
        if let Some(mine) = mine {
            match step.rank_of(mine) {
                Some(rank) => report.push_str(&format!(
                    "  our token sits at reference rank {rank} with logprob {:+.6}\n",
                    step.top_logprobs.get(rank).copied().unwrap_or(f64::NAN)
                )),
                None => report.push_str(
                    "  our token is not in the reference's top thirty-two at all, which is a \
                     much larger disagreement than a reordering\n",
                ),
            }
        }
    }

    report.push_str("  our ranked candidates (token, BF16 logit):\n");
    for (rank, &(token, logit)) in ours.iter().enumerate() {
        report.push_str(&format!("    {rank}. {token:>7}  {logit:+.6}\n"));
    }
    if let (Some(&(_, first)), Some(&(_, second))) = (ours.first(), ours.get(1)) {
        report.push_str(&format!(
            "  our top-1 logit margin is {:+.6}, which is {:.1} BF16 steps at this magnitude\n",
            first - second,
            (first - second) / bf16_ulp(first)
        ));
    }

    report
}

fn require_generation_refusal<T>(result: Result<T, EngineError>, name: &str) -> QualResult<String> {
    let error = match result {
        Err(error) => error,
        Ok(_) => return Err(mismatch(format!("{name} was not refused"))),
    };
    let message = error.to_string();
    if error.code() != Some(EngineErrorCode::Generation) {
        return Err(mismatch(format!(
            "{name} failed with {error} instead of a generation-capacity refusal"
        )));
    }
    if !message.contains(&QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING.to_string()) {
        return Err(mismatch(format!(
            "{name} refusal did not name the {QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING}-token ceiling: {message}"
        )));
    }

    Ok(message)
}

/// Replays one boundary capture inside and immediately outside the dense band.
fn replay_boundary(
    generator: &mut Qwen38FlashNextTextGenerator,
    directory: &Path,
    visible: usize,
) -> QualResult<Qwen38FlashNextBoundaryVerdict> {
    let capture = load_qwen38_flash_next_golden_boundary(directory, visible)?;
    let prompt_tokens = capture.prompt_ids.len();
    let admitted = (QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING + 1).saturating_sub(prompt_tokens);
    let name = format!("boundary-{visible}");

    let full = generator.qualification_generate_from_tokens(
        &capture.prompt_ids,
        BOUNDARY_TOKENS,
        SamplingOptions::greedy(),
        0,
    );
    let refusal = if admitted < BOUNDARY_TOKENS {
        if full.is_ok() {
            return Err(mismatch(format!(
                "{name} admitted {BOUNDARY_TOKENS} tokens with only {admitted} in band"
            )));
        }
        Some(require_generation_refusal(full, &name)?)
    } else {
        full?;
        None
    };

    if admitted == 0 {
        let result = generator.qualification_generate_from_tokens(
            &capture.prompt_ids,
            1,
            SamplingOptions::greedy(),
            0,
        );
        if result.is_ok() {
            return Err(mismatch(format!(
                "{name} admitted one token from an out-of-band {prompt_tokens}-token prompt"
            )));
        }
        let error = require_generation_refusal(result, &name)?;

        return Ok(Qwen38FlashNextBoundaryVerdict {
            visible,
            prompt_tokens,
            admitted_tokens: 0,
            agreed_tokens: 0,
            free_running_tokens: 0,
            full_request_refused: refusal.is_some(),
            over_budget_refused: true,
            refusal: Some(error),
        });
    }

    let verdict = replay_capture(generator, &name, &capture, admitted)?;

    let result = generator.qualification_generate_from_tokens(
        &capture.prompt_ids,
        admitted + 1,
        SamplingOptions::greedy(),
        0,
    );
    if result.is_ok() {
        return Err(mismatch(format!(
            "{name} admitted {} tokens from a {prompt_tokens}-token prompt, one past the \
             {admitted} the dense band funds",
            admitted + 1
        )));
    }
    require_generation_refusal(result, &name)?;

    Ok(Qwen38FlashNextBoundaryVerdict {
        visible,
        prompt_tokens,
        admitted_tokens: admitted,
        agreed_tokens: verdict.agreed_steps,
        free_running_tokens: verdict.free_running_tokens,
        full_request_refused: refusal.is_some(),
        over_budget_refused: true,
        refusal,
    })
}

/// Checks tile-ladder invariance and last-row publication.
fn verify_prefill_publication(
    generator: &mut Qwen38FlashNextTextGenerator,
) -> QualResult<(usize, usize)> {
    let prompt = (0..128u32)
        .map(|index| 1_024 + index * 7)
        .collect::<Vec<_>>();
    let mut checks = 0usize;
    let mut compared = 0usize;

    let reference = prime_tiles(generator, &prompt[..64], &[64])?;
    let split = prime_tiles(generator, &prompt[..64], &[32, 32])?;
    if let Some((index, left, right)) = first_difference(&reference, &split) {
        return Err(mismatch(format!(
            "a 64-token prompt reached different logits through two T=32 tiles than through one \
             T=64 tile: vocabulary index {index} read {left:#06x} then {right:#06x}, so a tile \
             boundary is carrying semantics rather than selecting a route"
        )));
    }
    checks += 1;
    compared += reference.len();

    let wide = prime_tiles(generator, &prompt, &[128])?;
    let split = prime_tiles(generator, &prompt, &[64, 64])?;
    if let Some((index, left, right)) = first_difference(&wide, &split) {
        return Err(mismatch(format!(
            "a 128-token prompt reached different logits through one T=128 tile than through two \
             T=64 tiles: vocabulary index {index} read {left:#06x} then {right:#06x}"
        )));
    }
    checks += 1;
    compared += wide.len();

    // Changing the final token must move the published row.
    for tile in [32usize, 64, 128] {
        let mut altered = prompt[..tile].to_vec();
        let last = altered.len() - 1;
        altered[last] = altered[last].wrapping_add(4_099);
        let base = prime_tiles(generator, &prompt[..tile], &[tile])?;
        let moved = prime_tiles(generator, &altered, &[tile])?;
        if first_difference(&base, &moved).is_none() {
            return Err(mismatch(format!(
                "a T={tile} prefill tile published identical logits after its LAST token changed, \
                 so the tail segment is publishing an earlier row than the one the caller samples; \
                 the logits are plausible and belong to the wrong position"
            )));
        }
        // The preceding history token must also affect the row.
        let mut earlier = prompt[..tile].to_vec();
        earlier[last - 1] = earlier[last - 1].wrapping_add(4_099);
        let shifted = prime_tiles(generator, &earlier, &[tile])?;
        if first_difference(&base, &shifted).is_none() {
            return Err(mismatch(format!(
                "a T={tile} prefill tile ignored a change to its second-to-last token"
            )));
        }
        checks += 2;
        compared += 2 * base.len();
    }

    // Scalar comparisons diagnose the unserved fallback; only native tiling is authoritative.
    for length in [32usize, 64, 128] {
        let tiled = prime_tiles(generator, &prompt[..length], &[length])?;
        let scalar = prime_scalar(generator, &prompt[..length])?;
        let (peak, worst) = row_distance(&tiled, &scalar);
        let spacing = bf16_ulp(peak);
        println!(
            "    tiled vs scalar at {length:>4} tokens: leading {:?} vs {:?}, worst |delta| \
             {worst:.6} ({:.1} BF16 steps) at peak {peak:.4}",
            ranked_tokens(&tiled, 3),
            ranked_tokens(&scalar, 3),
            worst / spacing
        );
        compared += tiled.len();
    }

    Ok((checks, compared))
}

/// Primes a prompt through an explicit ladder of prefill tiles and reads the published row.
fn prime_tiles(
    generator: &mut Qwen38FlashNextTextGenerator,
    prompt: &[u32],
    ladder: &[usize],
) -> QualResult<Vec<u16>> {
    if ladder.iter().sum::<usize>() != prompt.len() {
        return Err(mismatch("a tile ladder does not cover its prompt exactly"));
    }
    let stream = Arc::clone(generator.qualification_stream());
    let model = generator.qualification_program_mut();
    model.recycle_slot(&stream, 0)?;
    model.reserve_slot(&stream, 0, prompt.len())?;
    let mut cursor = 0usize;
    for &rows in ladder {
        let first = u32::try_from(cursor)
            .map_err(|_| mismatch("a tile ladder position exceeds the route width"))?;
        model.prefill_tile(&stream, &prompt[cursor..cursor + rows], first, 0)?;
        cursor += rows;
    }
    let mut logits = vec![0u16; <A as Arch>::VOCAB];
    model.read_logits_into(&stream, 1, &mut logits)?;

    Ok(logits)
}

/// Primes entirely through scalar decode rounds.
fn prime_scalar(
    generator: &mut Qwen38FlashNextTextGenerator,
    prompt: &[u32],
) -> QualResult<Vec<u16>> {
    let stream = Arc::clone(generator.qualification_stream());
    let model = generator.qualification_program_mut();
    model.recycle_slot(&stream, 0)?;
    model.reserve_slot(&stream, 0, prompt.len())?;
    for (position, &token) in prompt.iter().enumerate() {
        let position = u32::try_from(position)
            .map_err(|_| mismatch("a scalar prime position exceeds the route width"))?;
        model.decode_step(&stream, &[token], &[position], &[0])?;
    }
    let mut logits = vec![0u16; <A as Arch>::VOCAB];
    model.read_logits_into(&stream, 1, &mut logits)?;

    Ok(logits)
}

/// The `count` strongest tokens of one BF16 row, descending, ties to the lower token.
fn ranked_tokens(row: &[u16], count: usize) -> Vec<u32> {
    let mut ranked = row
        .iter()
        .enumerate()
        .map(|(token, &bits)| (token as u32, f32::from_bits(u32::from(bits) << 16)))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(count);

    ranked.into_iter().map(|(token, _)| token).collect()
}

/// Peak magnitude across two rows and the largest absolute difference between them.
fn row_distance(left: &[u16], right: &[u16]) -> (f32, f32) {
    let mut peak = 0.0f32;
    let mut worst = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        let left = f32::from_bits(u32::from(left) << 16);
        let right = f32::from_bits(u32::from(right) << 16);
        peak = peak.max(left.abs()).max(right.abs());
        worst = worst.max((left - right).abs());
    }

    (peak, worst)
}

/// Distance between adjacent BF16 values at one magnitude.
fn bf16_ulp(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let exponent = ((value.abs().to_bits() >> 23) & 0xff) as i32 - 127;

    // BF16 keeps seven explicit mantissa bits.
    (2.0f32).powi(exponent - 7)
}

/// First index at which two logit rows differ.
fn first_difference(left: &[u16], right: &[u16]) -> Option<(usize, u16, u16)> {
    left.iter()
        .zip(right)
        .enumerate()
        .find(|(_, (left, right))| left != right)
        .map(|(index, (&left, &right))| (index, left, right))
}

/// Checks repeatability and verifies that the seed affects sampling.
fn verify_seeded_sampling(
    generator: &mut Qwen38FlashNextTextGenerator,
) -> QualResult<(usize, bool)> {
    let prompt = [760u32, 6_511, 314, 9_338, 369];
    let sampled = |seed: u64| SamplingOptions {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 20,
        seed,
        ..SamplingOptions::default()
    };
    let defaults = generator.generation_defaults();
    if defaults.temperature != 1.0 || defaults.top_p != 0.95 || defaults.top_k != 20 {
        return Err(mismatch(format!(
            "the pinned checkpoint states temperature {}, top_p {}, top_k {}; the gate samples \
             at the frontend's own defaults and they have moved",
            defaults.temperature, defaults.top_p, defaults.top_k
        )));
    }

    let mut agreements = 0usize;
    let mut first = Vec::new();
    for pass in 0..2 {
        let run = generator.qualification_generate_from_tokens(&prompt, 24, sampled(7), 0)?;
        if pass == 0 {
            first = run.token_ids;
        } else {
            if run.token_ids != first {
                return Err(mismatch(format!(
                    "two runs at seed 7 produced different token streams: {:?} then {:?}",
                    first, run.token_ids
                )));
            }
            agreements += 1;
        }
    }

    let other = generator.qualification_generate_from_tokens(&prompt, 24, sampled(1_000_003), 0)?;
    let separate = other.token_ids != first;

    // Greedy decoding must also reproduce.
    let greedy =
        generator.qualification_generate_from_tokens(&prompt, 24, SamplingOptions::greedy(), 0)?;
    let repeat =
        generator.qualification_generate_from_tokens(&prompt, 24, SamplingOptions::greedy(), 0)?;
    if greedy.token_ids != repeat.token_ids {
        return Err(mismatch(
            "two greedy runs of the same prompt produced different tokens, so the decode route \
             is not reproducible",
        ));
    }
    agreements += 1;

    Ok((agreements, separate))
}

/// Checks lifecycle conservation and stale-tail masking on recycled pages.
fn verify_slot_lifecycle(
    generator: &mut Qwen38FlashNextTextGenerator,
) -> QualResult<(usize, usize)> {
    let stream = Arc::clone(generator.qualification_stream());
    let probe = (0..96u32)
        .map(|index| 4_096 + index * 13)
        .collect::<Vec<_>>();

    let (clean, clean_pages) = {
        let model = generator.qualification_program_mut();
        model.reset_state(&stream)?;
        model.reserve_slot(&stream, 0, probe.len())?;
        let pages = model.slots().pages(0)?.to_vec();
        model.prefill_tile(&stream, &probe[..64], 0, 0)?;
        model.prefill_tile(&stream, &probe[64..96], 64, 0)?;
        let mut logits = vec![0u16; <A as Arch>::VOCAB];
        model.read_logits_into(&stream, 1, &mut logits)?;
        (logits, pages)
    };

    let model = generator.qualification_program_mut();
    let funded = model.slots().funded_pages();

    // Dirty and recycle pages through every other slot.
    let mut moved = 0usize;
    for slot in 1..MAX_BATCH {
        let filler = (0..192u32)
            .map(|index| 70_000 + slot as u32 * 977 + index * 31)
            .collect::<Vec<_>>();
        let change = model.reserve_slot(&stream, slot, filler.len())?;
        moved += change.acquired_pages;
        if model.slot_state(slot)? != Qwen38FlashNextSlotState::Active {
            return Err(mismatch(format!(
                "slot {slot} is not active after a reservation"
            )));
        }
        model.prefill_tile(&stream, &filler[..128], 0, slot)?;
        model.prefill_tile(&stream, &filler[128..192], 128, slot)?;
        if model.slot_tokens(slot)? != 192 {
            return Err(mismatch(format!(
                "slot {slot} committed {} tokens after 192 were prefilled",
                model.slot_tokens(slot)?
            )));
        }

        let truncated = model.qualification_truncate_slot(&stream, slot, 128)?;
        if truncated.released_pages != 1 || truncated.tokens != 128 {
            return Err(mismatch(format!(
                "truncating slot {slot} from 192 to 128 tokens released {} pages, expected one",
                truncated.released_pages
            )));
        }
        model.recycle_slot(&stream, slot)?;
        moved += model.reserve_slot(&stream, slot, 128)?.acquired_pages;
        model.prefill_tile(&stream, &filler[..128], 0, slot)?;
        let retained = model.retain_slot(slot)?;
        if retained != 128 || model.slot_state(slot)? != Qwen38FlashNextSlotState::Retained {
            return Err(mismatch(format!(
                "slot {slot} retained {retained} tokens and reports {:?}",
                model.slot_state(slot)?
            )));
        }
        let release = model.recycle_slot(&stream, slot)?;
        if release.retained_tokens != 128 {
            return Err(mismatch(format!(
                "recycling retained slot {slot} reported {} retained tokens, expected 128",
                release.retained_tokens
            )));
        }
        if model.slot_state(slot)? != Qwen38FlashNextSlotState::Free {
            return Err(mismatch(format!(
                "slot {slot} is not free after being recycled"
            )));
        }
    }

    if model.slots().free_pages() + model.slots().pages(0)?.len() != funded {
        return Err(mismatch(format!(
            "the page pool holds {} free and {} on slot zero against {funded} funded, so a \
             lifecycle leaked pages",
            model.slots().free_pages(),
            model.slots().pages(0)?.len()
        )));
    }

    model.recycle_slot(&stream, 0)?;
    let contaminant = (0..128u32)
        .map(|index| 120_000 + index * 43)
        .collect::<Vec<_>>();
    moved += model
        .reserve_slot(&stream, 1, contaminant.len())?
        .acquired_pages;
    if model.slots().pages(1)? != clean_pages {
        return Err(mismatch(
            "the contaminating sequence did not receive the clean probe's recycled pages",
        ));
    }
    model.prefill_tile(&stream, &contaminant, 0, 1)?;
    model.recycle_slot(&stream, 1)?;

    model.reserve_slot(&stream, 0, probe.len())?;
    if model.slots().pages(0)? != clean_pages {
        return Err(mismatch(
            "the repeated probe did not receive the pages whose tail was contaminated",
        ));
    }
    model.prefill_tile(&stream, &probe[..64], 0, 0)?;
    model.prefill_tile(&stream, &probe[64..96], 64, 0)?;
    let mut recycled = vec![0u16; <A as Arch>::VOCAB];
    model.read_logits_into(&stream, 1, &mut recycled)?;

    if let Some((index, (&left, &right))) = clean
        .iter()
        .zip(&recycled)
        .enumerate()
        .find(|(_, (left, right))| left != right)
    {
        return Err(mismatch(format!(
            "a sequence on recycled pages produced different logits than the same sequence on \
             clean pages: vocabulary index {index} read {left:#06x} then {right:#06x}, so a \
             round is reading past its own committed length into a previous sequence's cache"
        )));
    }

    Ok((moved, clean.len()))
}

/// Checks token-exact continuation after recurrent-state rollback.
fn verify_rollback(generator: &mut Qwen38FlashNextTextGenerator) -> QualResult<(usize, usize)> {
    let prompt = [
        26_583u32, 310, 55_913, 25, 561, 8_831, 369, 6_037, 3_242, 13,
    ];
    let first =
        generator.qualification_generate_from_tokens(&prompt, 16, SamplingOptions::greedy(), 0)?;

    let midpoint = 8usize;
    let head = generator.qualification_generate_from_tokens(
        &prompt,
        midpoint,
        SamplingOptions::greedy(),
        0,
    )?;
    if head.token_ids != first.token_ids[..midpoint] {
        return Err(mismatch(
            "the same prompt produced different tokens on two greedy runs before any rollback",
        ));
    }

    let stream = Arc::clone(generator.qualification_stream());
    let model = generator.qualification_program_mut();
    let snapshot = model.snapshot_slot(&stream, 0)?;
    let snapshot_bytes = snapshot.byte_len();
    let tokens_at_snapshot = model.slot_tokens(0)?;

    // Advance every recurrent family before restoring.
    let mut position = u32::try_from(tokens_at_snapshot)
        .map_err(|_| mismatch("a rollback position exceeds the route width"))?;
    model.reserve_slot(&stream, 0, tokens_at_snapshot + 8)?;
    for offset in 0..8u32 {
        model.decode_step(&stream, &[11_000 + offset * 97], &[position], &[0])?;
        position += 1;
    }
    if model.slot_tokens(0)? <= tokens_at_snapshot {
        return Err(mismatch(
            "the sequence did not advance past its snapshot, so restoring it would restore \
             nothing and the round trip would prove nothing",
        ));
    }

    model.restore_slot(&stream, &snapshot)?;
    if model.slot_tokens(0)? != tokens_at_snapshot {
        return Err(mismatch(format!(
            "restoring rolled the cache to {} tokens, expected {tokens_at_snapshot}",
            model.slot_tokens(0)?
        )));
    }

    let mut restored = Vec::with_capacity(16 - midpoint);
    let mut logits = vec![0u16; <A as Arch>::VOCAB];
    let mut position = u32::try_from(tokens_at_snapshot)
        .map_err(|_| mismatch("a rollback position exceeds the route width"))?;
    let mut token = first.token_ids[midpoint - 1];
    for _ in midpoint..16 {
        model.reserve_slot(&stream, 0, position as usize + 1)?;
        model.decode_step(&stream, &[token], &[position], &[0])?;
        model.read_logits_into(&stream, 1, &mut logits)?;
        token = greedy_token(&logits)?;
        restored.push(token);
        position += 1;
    }

    if restored != first.token_ids[midpoint..] {
        return Err(mismatch(format!(
            "a restored sequence continued differently from the uninterrupted one: expected \
             {:?}, got {restored:?}; a snapshot that missed one of the GDN history, the GDN \
             recurrent state, the PLE dilated conv state, or the engram two-token carry would \
             continue exactly like this, plausibly but incorrectly",
            &first.token_ids[midpoint..]
        )));
    }

    Ok((restored.len(), snapshot_bytes))
}

/// Checks EOS and usage accounting through the production chat path.
fn verify_eos_and_usage(generator: &mut Qwen38FlashNextTextGenerator) -> QualResult<()> {
    let request = ChatGenerationRequest {
        max_new_tokens: 12,
        sampling: SamplingOptions::greedy(),
        ..ChatGenerationRequest::new(vec![ChatMessage::new("user", "Name one colour.")])
    };
    let mut session = generator.start(&request)?;
    let prompt_tokens = session.prompt_token_ids().len();
    let mut emitted = 0usize;
    let mut finish = None;
    while finish.is_none() {
        let step = session.step()?;
        emitted += 1;
        finish = step.finish_reason;
        if emitted > request.max_new_tokens {
            return Err(mismatch(
                "the session emitted more tokens than its budget without finishing",
            ));
        }
    }
    let output = session.into_output()?;

    if output.token_ids.len() != emitted {
        return Err(mismatch(format!(
            "the session emitted {emitted} steps and reported {} tokens",
            output.token_ids.len()
        )));
    }
    match finish {
        Some(FinishReason::Length) if emitted == request.max_new_tokens => {}
        Some(FinishReason::Stop) => {
            let stops = output
                .token_ids
                .last()
                .copied()
                .ok_or_else(|| mismatch("a stopped session reported no tokens at all"))?;
            if !generator_stop_ids().contains(&stops) {
                return Err(mismatch(format!(
                    "the session finished with reason stop on token {stops}, which is not an \
                     admitted stop id"
                )));
            }
        }
        other => {
            return Err(mismatch(format!(
                "the session finished with {other:?} after {emitted} of {} tokens",
                request.max_new_tokens
            )));
        }
    }
    if output.prompt.token_ids.len() != prompt_tokens {
        return Err(mismatch(
            "the completed output reports a different prompt length than the session did",
        ));
    }

    // A zero-token request finishes before device work.
    let empty = ChatGenerationRequest {
        max_new_tokens: 0,
        ..request
    };
    let session = generator.start(&empty)?;
    if session.finish_reason() != Some(FinishReason::Length) {
        return Err(mismatch("a zero-token request did not finish at admission"));
    }

    Ok(())
}

fn generator_stop_ids() -> [u32; 2] {
    use tuisko_frontend::TokenizedSchema;
    let ids = <Qwen38FlashNext as TokenizedSchema>::EOS_IDS;

    [ids[0], ids[1]]
}

fn greedy_token(logits: &[u16]) -> QualResult<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (token, &bits) in logits.iter().enumerate() {
        let value = f32::from_bits(u32::from(bits) << 16);
        if best.is_none_or(|(_, retained)| value.total_cmp(&retained).is_gt()) {
            best = Some((token as u32, value));
        }
    }

    best.map(|(token, _)| token)
        .ok_or_else(|| mismatch("an empty logit row has no greedy token"))
}

fn fold(
    left: Qwen38FlashNextGenerationTelemetry,
    right: Qwen38FlashNextGenerationTelemetry,
) -> Qwen38FlashNextGenerationTelemetry {
    let mut total = left;
    total.absorb(right);

    total
}

/// Prints one capture's verdict as soon as it is known.
pub fn print_capture_verdict(verdict: &Qwen38FlashNextCaptureVerdict) {
    println!(
        "  {:<14} prompt {:>5}  forced {:>3}/{:<3}  free-running {:>3}/{:<3} {}",
        verdict.name,
        verdict.prompt_tokens,
        verdict.agreed_steps,
        verdict.scored_steps,
        verdict.free_running_tokens,
        verdict.scored_steps,
        if verdict.free_running_complete {
            "COMPLETE"
        } else {
            "forked"
        }
    );
    println!(
        "    native prefill {:>5}   prime wall {:>9.2?}   decode {:>7.3} ms/token",
        verdict.native_prefill_tokens,
        verdict.telemetry.prime_wall(),
        verdict.telemetry.decode_ms_per_token()
    );
    println!(
        "    decode hit {:.4}   h2d/token {:>12.0}   publication stalls {}",
        verdict.telemetry.decode_expert_hit_rate(),
        verdict.telemetry.decode_expert_h2d_bytes_per_token(),
        verdict.telemetry.publication_stalls()
    );
    if let Some(mean) = verdict.mean_margin {
        println!(
            "    reference mean margin {mean:+.4} nats   tightest agreement {}",
            verdict
                .tightest_agreement
                .map_or_else(|| "none".to_string(), |margin| format!("{margin:+.6}"))
        );
    }
    if !verdict.tied_steps.is_empty() {
        println!(
            "    exact reference ties at {:?}; transposed {:?}",
            verdict.tied_steps, verdict.tied_transpositions
        );
    }
    for &(index, ours, theirs) in &verdict.runner_up {
        println!(
            "    step {index:>3} took the reference's runner-up: our margin {ours:+.6}, its \
             margin {theirs:+.6} nats"
        );
    }
    if let Some(diagnosis) = &verdict.free_running_diagnosis {
        for line in diagnosis.lines() {
            println!("    {line}");
        }
    }
}

/// Prints one boundary verdict.
pub fn print_boundary_verdict(verdict: &Qwen38FlashNextBoundaryVerdict) {
    println!(
        "  boundary-{:<5} prompt {:>5}  admitted {}  forced {}  free-running {}  full-request {}  \
         over-budget {}",
        verdict.visible,
        verdict.prompt_tokens,
        verdict.admitted_tokens,
        verdict.agreed_tokens,
        verdict.free_running_tokens,
        if verdict.full_request_refused {
            "REFUSED"
        } else {
            "admitted"
        },
        if verdict.over_budget_refused {
            "REFUSED"
        } else {
            "admitted"
        }
    );
}

/// Prints one qualification report in the house's diagnostic shape.
pub fn print_qwen38_flash_next_generation_report(report: &Qwen38FlashNextGenerationQualification) {
    println!("Qwen3.8 Flash-Next generation external selection cross-check");
    println!("  authority                {}", report.provenance);
    println!("  construction");
    println!("    total                  {:?}", report.load);
    println!("    executables            {}", report.executables);
    println!("    generation capacity    {}", report.generation_capacity);
    println!("  the gate");
    println!(
        "    steps scored           {} over {} captures",
        report.compared_tokens,
        report.captures.len()
    );
    println!(
        "    teacher-forced agreed  {} ({} exact reference ties, {} transposed)",
        report.agreed_tokens, report.tied_steps, report.tied_transpositions
    );
    println!(
        "    decisive disagreements {}",
        report.decisive_disagreements
    );
    println!(
        "    took the runner-up     {} (widest reference margin among them {:+.6} nats)",
        report.unresolved_steps, report.widest_unresolved_margin
    );
    println!(
        "    tightest agreement     {}",
        report
            .tightest_agreement
            .map_or_else(|| "none".to_string(), |margin| format!("{margin:+.6} nats"))
    );
    println!(
        "    free-running complete  {} of {} captures",
        report.free_running_complete,
        report.captures.len()
    );
    for verdict in &report.captures {
        print_capture_verdict(verdict);
    }
    println!("  boundary sweep");
    for verdict in &report.boundaries {
        print_boundary_verdict(verdict);
    }
    println!("  house law");
    println!(
        "    prefill checks         {} over {} logits",
        report.prefill_checks, report.tiling_compared_logits
    );
    println!(
        "    seeded determinism     {} agreements, seeds separate {}",
        report.seeded_agreements, report.seeds_separate
    );
    println!(
        "    rollback round trip    {} tokens, {} B restore point",
        report.rollback_tokens, report.rollback_snapshot_bytes
    );
    println!(
        "    recycled-page logits   {} equal",
        report.recycled_page_compared_logits
    );
    println!("    lifecycle pages moved  {}", report.lifecycle_pages);
    println!("    engram rows hashed     {}", report.engram_rows);
    println!("  real-traffic streaming (diagnostic, nothing blessed)");
    println!(
        "    decode expert hit      {:.4}   against the trace study's 0.85",
        report.decode_hit_rate
    );
    println!(
        "    decode h2d/token       {:.0} B",
        report.decode_h2d_bytes_per_token
    );
    println!(
        "    whole-request hit      {:.4}",
        report.telemetry.expert_hit_rate()
    );
    println!(
        "    decode ms/token        {:.3}   tok/s {:.2}",
        report.telemetry.decode_ms_per_token(),
        report.telemetry.decode_tokens_per_second()
    );
    println!(
        "    prime tiles / scalar   {} / {}",
        report.telemetry.prime_tiles(),
        report.telemetry.prime_scalar_rounds()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_admitted_budget_is_the_dense_bands_own_arithmetic() {
        for (prompt, admitted) in [
            (2_046usize, 6usize),
            (2_047, 5),
            (2_048, 4),
            (2_049, 3),
            (2_050, 2),
            (2_051, 1),
            (2_055, 0),
            (2_099, 0),
        ] {
            let budget = (QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING + 1).saturating_sub(prompt);
            assert_eq!(budget, admitted, "prompt {prompt}");
        }
    }

    #[test]
    fn a_page_holds_sixty_four_tokens_so_the_lifecycle_arithmetic_is_the_one_the_pool_uses() {
        assert_eq!(tuisko_engine::QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS, 64);
        assert_eq!(
            192_usize.div_ceil(tuisko_engine::QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS),
            3
        );
        assert_eq!(
            128_usize.div_ceil(tuisko_engine::QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS),
            2
        );
    }

    #[test]
    fn the_gate_records_enough_candidates_to_diagnose_a_reordering() {
        assert_eq!(RECORDED_CANDIDATES, 8);
        assert_eq!(BOUNDARY_TOKENS, 8);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_generation_matches_the_external_selection_contract()
    -> Result<(), Qwen38FlashNextGenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").ok_or_else(|| {
            Qwen38FlashNextGenerationQualificationError::Mismatch(
                "TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required for the source-backed gate"
                    .to_string(),
            )
        })?;
        let report = qualify_qwen38_flash_next_generation(std::path::Path::new(&root))?;
        print_qwen38_flash_next_generation_report(&report);

        assert_eq!(report.captures.len(), 8);
        assert_eq!(report.compared_tokens, 8 * 64);
        assert_eq!(report.decisive_disagreements, 0);
        assert_eq!(
            report.agreed_tokens + report.tied_transpositions + report.unresolved_steps,
            report.compared_tokens
        );

        assert!(
            report
                .captures
                .iter()
                .all(|capture| capture.decisive_disagreements.is_empty())
        );
        assert_eq!(
            report.generation_capacity,
            QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
        );

        assert_eq!(report.boundaries.len(), 8);
        for verdict in &report.boundaries {
            assert!(
                verdict.full_request_refused,
                "boundary-{} admitted its full eight-token request",
                verdict.visible
            );
            assert!(verdict.over_budget_refused, "boundary-{}", verdict.visible);
            assert_eq!(
                verdict.agreed_tokens, verdict.admitted_tokens,
                "boundary-{}",
                verdict.visible
            );
            assert_eq!(
                verdict.free_running_tokens, verdict.admitted_tokens,
                "boundary-{}",
                verdict.visible
            );
        }
        let admitted = report
            .boundaries
            .iter()
            .map(|verdict| verdict.admitted_tokens)
            .collect::<Vec<_>>();
        assert_eq!(admitted, vec![6, 5, 4, 3, 2, 1, 0, 0]);

        assert_eq!(report.prefill_checks, 8);
        assert_eq!(report.tiling_compared_logits, 11 * <A as Arch>::VOCAB);
        assert!(report.seeded_agreements >= 2);
        assert!(report.seeds_separate);
        assert_eq!(report.rollback_tokens, 8);
        assert!(report.rollback_snapshot_bytes > 0);
        assert_eq!(report.recycled_page_compared_logits, <A as Arch>::VOCAB);
        assert!(report.lifecycle_pages > 0);
        assert!(report.engram_rows > 0);
        assert_eq!(report.engram_rows % A::NGRAM_HEADS, 0);

        assert!(report.decode_hit_rate > 0.0 && report.decode_hit_rate <= 1.0);
        assert!(
            report.decode_h2d_bytes_per_token <= 200_000_000.0,
            "real traffic streamed {} B per generated token, past the 200 MB stop rule",
            report.decode_h2d_bytes_per_token
        );
        let misses = (1.0 - report.decode_hit_rate)
            * (<A as Arch>::LAYERS * A::NUM_EXPERTS_PER_TOKEN) as f64;
        let modelled = misses * tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES as f64;
        assert!(
            (modelled - report.decode_h2d_bytes_per_token).abs() < 1_000_000.0,
            "a hit rate of {} implies {modelled} streamed bytes per token, the telemetry counted {}",
            report.decode_hit_rate,
            report.decode_h2d_bytes_per_token
        );

        Ok(())
    }
}
