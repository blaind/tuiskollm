//! Source-backed exactness and diagnostic timing for Qwen3.8 MTP generation.
//!
//! Greedy speculation must reproduce the pair's target-only token list exactly.

use crate::device_benchmark::{self, DeviceBenchmarkError};
use crate::qwen38_flash_next_golden::{
    QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES, QWEN38_FLASH_NEXT_GOLDEN_PROMPTS,
    Qwen38FlashNextGoldenError, load_qwen38_flash_next_golden_boundary,
    load_qwen38_flash_next_golden_capture, load_qwen38_flash_next_golden_meta,
    qwen38_flash_next_golden_directory,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{
    ChatGenerationRequest, EngineError, QWEN38_FLASH_NEXT_ATTENTION_LAYERS,
    QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, QWEN38_FLASH_NEXT_MTP_ATTENTION_ROUTES,
    QWEN38_FLASH_NEXT_MTP_ROUTES, QWEN38_FLASH_NEXT_MTP_SEGMENTS,
    QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, Qwen38FlashNextMtpAcceptance, Qwen38FlashNextMtpResidency,
    Qwen38FlashNextMtpRoundCost, Qwen38FlashNextMtpTextGenerator, ResidentMtpGenerationStats,
    SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

const IDENTITY_TOKENS: usize = 64;
const BOUNDARY_TOKENS: usize = 8;
const FRESH_PROMPTS: [usize; 6] = [17, 96, 512, 1_600, 2_040, 2_050];
const WARM_PROMPT_TOKENS: usize = 512;
const WARM_TOKENS: usize = 64;
const CODING_TOKENS: usize = 32;
const CODING_CASES: [(&str, &str); 3] = [
    (
        "rust-parser",
        "Write a compact Rust function that parses an unsigned decimal integer without allocation.",
    ),
    (
        "cuda-index",
        "Explain how to compute a row-major CUDA tensor offset from shape, strides, and indices.",
    ),
    (
        "test-review",
        "Review a unit test that checks only a prefix when exact list equality is required.",
    ),
];

/// Failure of the source-backed Qwen3.8 MTP generation gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextMtpGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Resident construction or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),

    /// CUDA ownership or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The committed prompt source could not be admitted.
    #[error(transparent)]
    Golden(#[from] Qwen38FlashNextGoldenError),

    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// An exact composition boundary differed.
    #[error("Qwen3.8 MTP generation qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextMtpGenerationQualificationError>;

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextMtpGenerationQualificationError {
    Qwen38FlashNextMtpGenerationQualificationError::Mismatch(message.into())
}

/// One prompt run through target-only and speculative schedules.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpIdentityCase {
    /// Stable fixture or seam name.
    pub name: String,
    /// Prompt tokens consumed by both schedules.
    pub prompt_tokens: usize,
    /// Requested output budget.
    pub requested_tokens: usize,
    /// Tokens compared exactly.
    pub tokens: usize,
    /// First differing token or list end.
    pub first_divergence: Option<usize>,
    /// Native prefill tokens handled by each schedule.
    pub native_prefill_tokens: usize,
    /// Target-only wall time including prompt prime.
    pub plain: Duration,
    /// Speculative wall time including prompt prime.
    pub speculative: Duration,
    /// Speculative route counters.
    pub stats: ResidentMtpGenerationStats,
    /// Committed outputs per speculative round.
    pub acceptance: Qwen38FlashNextMtpAcceptance,
    /// Speculative stage timings.
    pub cost: Qwen38FlashNextMtpRoundCost,
}

impl Qwen38FlashNextMtpIdentityCase {
    /// Target-only throughput over the complete request.
    pub fn plain_tokens_per_second(&self) -> f64 {
        rate(self.tokens, self.plain)
    }

    /// Speculative throughput over the complete request.
    pub fn speculative_tokens_per_second(&self) -> f64 {
        rate(self.tokens, self.speculative)
    }
}

/// Hot-cache comparison at one fixed source-backed shape.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpWarmComparison {
    /// Prompt tokens in the fixed shape.
    pub prompt_tokens: usize,
    /// Tokens compared exactly.
    pub tokens: usize,
    /// Target-only wall time after warmup.
    pub plain: Duration,
    /// Speculative wall time after warmup.
    pub speculative: Duration,
    /// Speculative acceptance after warmup.
    pub acceptance: Qwen38FlashNextMtpAcceptance,
    /// Speculative stage timings after warmup.
    pub cost: Qwen38FlashNextMtpRoundCost,
}

/// One diagnostic timing through the rendered production chat path.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpCodingCase {
    /// Stable workload name.
    pub name: String,
    /// Rendered prompt tokens.
    pub prompt_tokens: usize,
    /// Generated tokens.
    pub tokens: usize,
    /// Whole-request wall time.
    pub elapsed: Duration,
    /// Generation wall time after prompt prime.
    pub decode: Duration,
    /// Native prefill tokens.
    pub native_prefill_tokens: usize,
    /// Speculative acceptance for the measured pass.
    pub acceptance: Qwen38FlashNextMtpAcceptance,
    /// Speculative stage timings for the measured pass.
    pub cost: Qwen38FlashNextMtpRoundCost,
}

/// One shared-index comparison against the draft's own selection.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpIndexShareTrial {
    /// Prompt tokens primed by both runs.
    pub prompt_tokens: usize,
    /// Acceptance with the target selection reused.
    pub shared: Qwen38FlashNextMtpAcceptance,
    /// Acceptance with draft-side selection.
    pub own: Qwen38FlashNextMtpAcceptance,
    /// Shared-index wall time.
    pub shared_elapsed: Duration,
    /// Draft-selection wall time.
    pub own_elapsed: Duration,
    /// Whether both policies committed identical tokens.
    pub identical: bool,
}

impl Qwen38FlashNextMtpCodingCase {
    /// Whole-request generated-token rate.
    pub fn tokens_per_second(&self) -> f64 {
        rate(self.tokens, self.elapsed)
    }
}

/// Source-backed identity, inventory, replay, and timing evidence.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpGenerationQualification {
    /// Pinned prompt-capture authority.
    pub provenance: String,
    /// Target-plus-draft construction time.
    pub load: Duration,
    /// Draft graph capture time.
    pub draft_capture: Duration,
    /// Draft executable count.
    pub draft_executables: usize,
    /// Target executable count.
    pub target_executables: usize,
    /// Target expert slots in the joint solve.
    pub target_slots: usize,
    /// Draft expert slots in the joint solve.
    pub draft_slots: usize,
    /// Physical pages shared by target and draft mirrors.
    pub physical_pages: usize,
    /// Total target-plus-draft device bytes.
    pub device_bytes: usize,
    /// Exact identity cases.
    pub cases: Vec<Qwen38FlashNextMtpIdentityCase>,
    /// Acceptance across every identity case.
    pub acceptance: Qwen38FlashNextMtpAcceptance,
    /// Stage timings across every identity case.
    pub cost: Qwen38FlashNextMtpRoundCost,
    /// IndexShare comparisons above the dense ceiling.
    pub index_share: Vec<Qwen38FlashNextMtpIndexShareTrial>,
    /// Tokens compared across repeated speculative runs.
    pub replay_compared: usize,
    /// Tokens that changed across repeated speculative runs.
    pub replay_moved: usize,
    /// Fixed-shape hot-cache comparison.
    pub warm: Qwen38FlashNextMtpWarmComparison,
    /// Rendered production chat timings.
    pub coding: Vec<Qwen38FlashNextMtpCodingCase>,
}

impl Qwen38FlashNextMtpGenerationQualification {
    /// Identity cases with equal token lists.
    pub fn identical_cases(&self) -> usize {
        self.cases
            .iter()
            .filter(|case| case.first_divergence.is_none())
            .count()
    }

    /// Tokens compared across all identity cases.
    pub fn compared_tokens(&self) -> usize {
        self.cases.iter().map(|case| case.tokens).sum()
    }

    /// Cases whose request crosses into selected QSA.
    pub fn selective_cases(&self) -> usize {
        self.cases
            .iter()
            .filter(|case| {
                case.prompt_tokens + case.requested_tokens.saturating_sub(1)
                    > QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
            })
            .count()
    }

    /// Aggregate target-only rate over the identity matrix.
    pub fn plain_tokens_per_second(&self) -> f64 {
        rate(
            self.compared_tokens(),
            self.cases.iter().map(|case| case.plain).sum(),
        )
    }

    /// Aggregate speculative rate over the identity matrix.
    pub fn speculative_tokens_per_second(&self) -> f64 {
        rate(
            self.compared_tokens(),
            self.cases.iter().map(|case| case.speculative).sum(),
        )
    }
}

/// Runs the source-backed exactness and diagnostic timing suite.
pub fn qualify_qwen38_flash_next_mtp_generation(
    root: &Path,
) -> QualResult<Qwen38FlashNextMtpGenerationQualification> {
    let _preflight = device_benchmark::preflight()?;
    let golden = qwen38_flash_next_golden_directory();
    let meta = load_qwen38_flash_next_golden_meta(&golden)?;
    meta.require_pinned_authority(A::MODEL_ID, A::REVISION)?;

    let snapshot = Arc::new(CheckpointSnapshot::<A>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let residency = Qwen38FlashNextMtpResidency::build()?;
    let target_slots = residency.target().streaming().slot_count();
    let draft_slots = residency.draft().streaming().slot_count();
    let physical_pages = residency.physical_pages();
    let device_bytes = residency.total_device_bytes()?;

    let started = Instant::now();
    let mut generator = Qwen38FlashNextMtpTextGenerator::from_snapshot(&context, snapshot)?;
    let load = started.elapsed();
    let expected_capacity = physical_pages * ATTENTION_PAGE_SIZE;
    if generator.context_capacity() != expected_capacity {
        return Err(mismatch(format!(
            "MTP generation capacity is {}, expected the funded {expected_capacity}",
            generator.context_capacity(),
        )));
    }
    let draft_stats = generator.program().load_stats();
    let draft_executables = generator.program().executables();
    let target_executables = generator.program().target().executables();
    let expected_draft = QWEN38_FLASH_NEXT_MTP_ROUTES.len()
        + QWEN38_FLASH_NEXT_MTP_ATTENTION_ROUTES.len()
        + QWEN38_FLASH_NEXT_MTP_SEGMENTS;
    let expected_target =
        (QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS + QWEN38_FLASH_NEXT_ATTENTION_LAYERS) * 16;
    if draft_executables != expected_draft || target_executables != expected_target {
        return Err(mismatch(format!(
            "captured draft/target inventory is {draft_executables}/{target_executables}, expected \
             {expected_draft}/{expected_target}"
        )));
    }

    println!("--- Qwen3.8 MTP pair (diagnostic, nothing blessed) ---");
    println!("  provenance               {}", meta.provenance());
    println!("  load                     {load:?}");
    println!("  draft capture            {:?}", draft_stats.capture());
    println!("  draft executables        {draft_executables}");
    println!("  target executables       {target_executables}");
    println!("  target / draft slots     {target_slots} / {draft_slots}");
    println!("  physical pages           {physical_pages}");
    println!("  device bytes             {device_bytes}");

    let mut cases = Vec::new();
    for stem in QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
        let capture = load_qwen38_flash_next_golden_capture(&golden, stem)?;
        cases.push(run_identity_case(
            &mut generator,
            stem.to_string(),
            &capture.prompt_ids,
            IDENTITY_TOKENS,
        )?);
    }
    for visible in QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES {
        let capture = load_qwen38_flash_next_golden_boundary(&golden, visible)?;
        cases.push(run_identity_case(
            &mut generator,
            format!("boundary-{visible}"),
            &capture.prompt_ids,
            BOUNDARY_TOKENS,
        )?);
    }

    let source = load_qwen38_flash_next_golden_boundary(&golden, 2_100)?;
    for prompt_tokens in FRESH_PROMPTS {
        let prompt = source.prompt_ids.get(..prompt_tokens).ok_or_else(|| {
            mismatch(format!(
                "fresh prompt {prompt_tokens} exceeds the {}-token source",
                source.prompt_ids.len()
            ))
        })?;
        cases.push(run_identity_case(
            &mut generator,
            format!("fresh-{prompt_tokens}"),
            prompt,
            dense_budget(prompt_tokens, IDENTITY_TOKENS),
        )?);
    }

    if let Some(case) = cases.iter().find(|case| case.first_divergence.is_some()) {
        return Err(mismatch(format!(
            "{} diverged at token {}",
            case.name,
            case.first_divergence.unwrap_or(case.tokens)
        )));
    }
    let mut acceptance = Qwen38FlashNextMtpAcceptance::default();
    let mut cost = Qwen38FlashNextMtpRoundCost::default();
    for case in &cases {
        print_identity_case(case);
        acceptance.absorb(case.acceptance);
        cost.absorb(case.cost);
    }

    let replay_prompt = &source.prompt_ids[..96];
    let replay_a = generator.qualification_mtp_tokens(replay_prompt, 16)?;
    let replay_b = generator.qualification_mtp_tokens(replay_prompt, 16)?;
    let replay_compared = replay_a.token_ids.len().max(replay_b.token_ids.len());
    let replay_moved = divergence_count(&replay_a.token_ids, &replay_b.token_ids);
    let warm = run_warm_comparison(&mut generator, &source.prompt_ids[..WARM_PROMPT_TOKENS])?;
    let coding = run_coding_cases(&mut generator)?;
    let index_share = run_index_share_trials(&mut generator, &source.prompt_ids)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen38FlashNextMtpGenerationQualification {
        provenance: meta.provenance(),
        load,
        draft_capture: draft_stats.capture(),
        draft_executables,
        target_executables,
        target_slots,
        draft_slots,
        physical_pages,
        device_bytes,
        cases,
        acceptance,
        cost,
        index_share,
        replay_compared,
        replay_moved,
        warm,
        coding,
    })
}

fn run_identity_case(
    generator: &mut Qwen38FlashNextMtpTextGenerator,
    name: String,
    prompt: &[u32],
    requested_tokens: usize,
) -> QualResult<Qwen38FlashNextMtpIdentityCase> {
    let plain = generator.qualification_plain_tokens(prompt, requested_tokens)?;
    let speculative = generator.qualification_mtp_tokens(prompt, requested_tokens)?;
    if plain.native_prefill_tokens != speculative.native_prefill_tokens {
        return Err(mismatch(format!(
            "{name} used {} plain and {} speculative native prefill tokens",
            plain.native_prefill_tokens, speculative.native_prefill_tokens
        )));
    }

    Ok(Qwen38FlashNextMtpIdentityCase {
        name,
        prompt_tokens: prompt.len(),
        requested_tokens,
        tokens: plain.token_ids.len().min(speculative.token_ids.len()),
        first_divergence: first_divergence(&plain.token_ids, &speculative.token_ids),
        native_prefill_tokens: plain.native_prefill_tokens,
        plain: plain.elapsed,
        speculative: speculative.elapsed,
        stats: speculative.stats,
        acceptance: speculative.acceptance,
        cost: speculative.cost,
    })
}

fn run_warm_comparison(
    generator: &mut Qwen38FlashNextMtpTextGenerator,
    prompt: &[u32],
) -> QualResult<Qwen38FlashNextMtpWarmComparison> {
    let warm_plain = generator.qualification_plain_tokens(prompt, WARM_TOKENS)?;
    let warm_mtp = generator.qualification_mtp_tokens(prompt, WARM_TOKENS)?;
    require_equal("warmup", &warm_plain.token_ids, &warm_mtp.token_ids)?;

    let plain = generator.qualification_plain_tokens(prompt, WARM_TOKENS)?;
    let speculative = generator.qualification_mtp_tokens(prompt, WARM_TOKENS)?;
    require_equal("warm comparison", &plain.token_ids, &speculative.token_ids)?;

    Ok(Qwen38FlashNextMtpWarmComparison {
        prompt_tokens: prompt.len(),
        tokens: plain.token_ids.len(),
        plain: plain.elapsed,
        speculative: speculative.elapsed,
        acceptance: speculative.acceptance,
        cost: speculative.cost,
    })
}

fn run_coding_cases(
    generator: &mut Qwen38FlashNextMtpTextGenerator,
) -> QualResult<Vec<Qwen38FlashNextMtpCodingCase>> {
    let mut cases = Vec::with_capacity(CODING_CASES.len());
    for (name, prompt) in CODING_CASES {
        let request = coding_request(prompt);
        let warm = generator.qualification_chat_run(&request)?;
        let measured = generator.qualification_chat_run(&request)?;
        require_equal(name, &warm.token_ids, &measured.token_ids)?;
        cases.push(Qwen38FlashNextMtpCodingCase {
            name: name.to_string(),
            prompt_tokens: measured.prompt_tokens,
            tokens: measured.token_ids.len(),
            elapsed: measured.elapsed,
            decode: measured.decode,
            native_prefill_tokens: measured.native_prefill_tokens,
            acceptance: measured.acceptance,
            cost: measured.cost,
        });
    }

    Ok(cases)
}

fn run_index_share_trials(
    generator: &mut Qwen38FlashNextMtpTextGenerator,
    prompt_ids: &[u32],
) -> QualResult<Vec<Qwen38FlashNextMtpIndexShareTrial>> {
    let mut trials = Vec::new();
    for prompt_tokens in [prompt_ids.len().min(2_048), prompt_ids.len()] {
        if prompt_tokens <= QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
            continue;
        }
        let prompt = &prompt_ids[..prompt_tokens];
        let warm_shared = generator.qualification_mtp_tokens_with_index_share(prompt, 32, true)?;
        let warm_own = generator.qualification_mtp_tokens_with_index_share(prompt, 32, false)?;
        require_equal(
            "IndexShare warmup",
            &warm_shared.token_ids,
            &warm_own.token_ids,
        )?;
        let shared = generator.qualification_mtp_tokens_with_index_share(prompt, 32, true)?;
        let own = generator.qualification_mtp_tokens_with_index_share(prompt, 32, false)?;
        trials.push(Qwen38FlashNextMtpIndexShareTrial {
            prompt_tokens,
            shared: shared.acceptance,
            own: own.acceptance,
            shared_elapsed: shared.elapsed,
            own_elapsed: own.elapsed,
            identical: shared.token_ids == own.token_ids,
        });
    }

    Ok(trials)
}

fn coding_request(content: &str) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", content)]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = CODING_TOKENS;
    request
}

fn dense_budget(prompt_tokens: usize, requested_tokens: usize) -> usize {
    requested_tokens.min(
        QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
            .saturating_add(1)
            .saturating_sub(prompt_tokens),
    )
}

fn first_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

fn divergence_count(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
        + left.len().abs_diff(right.len())
}

fn require_equal(name: &str, left: &[u32], right: &[u32]) -> QualResult<()> {
    if let Some(index) = first_divergence(left, right) {
        return Err(mismatch(format!("{name} diverged at token {index}")));
    }

    Ok(())
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }

    tokens as f64 / elapsed.as_secs_f64()
}

fn print_identity_case(case: &Qwen38FlashNextMtpIdentityCase) {
    println!(
        "  {:<16} prompt {:>4}  tokens {:>2}/{:<2}  plain {:>7.2} tok/s  mtp {:>7.2} tok/s  accept {:.3}",
        case.name,
        case.prompt_tokens,
        case.tokens,
        case.requested_tokens,
        case.plain_tokens_per_second(),
        case.speculative_tokens_per_second(),
        case.acceptance.mean(),
    );
}

/// Prints the diagnostic suite report.
pub fn print_qwen38_flash_next_mtp_generation_report(
    report: &Qwen38FlashNextMtpGenerationQualification,
) {
    let rounds = report.cost.rounds().max(1) as f64;
    println!("--- exact identity ---");
    println!("  cases                    {}", report.cases.len());
    println!("  compared tokens          {}", report.compared_tokens());
    println!(
        "  replay moved/compared    {}/{}",
        report.replay_moved, report.replay_compared
    );
    println!("--- acceptance (diagnostic, nothing blessed) ---");
    println!("  rounds                   {}", report.cost.rounds());
    println!(
        "  distribution 1/2/3/4     {:?}",
        report.acceptance.distribution()
    );
    println!("  mean outputs per round   {:.4}", report.acceptance.mean());
    println!(
        "  whole round              {:.3} ms",
        report.cost.round_ms()
    );
    println!(
        "  verify                   {:.3} ms",
        report.cost.verify().as_secs_f64() * 1_000.0 / rounds
    );
    println!(
        "  restore                  {:.3} ms over {} restores",
        report.cost.restore().as_secs_f64() * 1_000.0 / rounds,
        report.cost.restores()
    );
    println!("--- identity throughput ---");
    println!(
        "  target-only              {:.2} tok/s",
        report.plain_tokens_per_second()
    );
    println!(
        "  speculative              {:.2} tok/s",
        report.speculative_tokens_per_second()
    );
    println!("--- warm fixed shape ---");
    println!(
        "  prompt/tokens            {}/{}",
        report.warm.prompt_tokens, report.warm.tokens
    );
    println!(
        "  target-only              {:.2} tok/s",
        rate(report.warm.tokens, report.warm.plain)
    );
    println!(
        "  speculative              {:.2} tok/s",
        rate(report.warm.tokens, report.warm.speculative)
    );
    println!("--- production chat path ---");
    for case in &report.coding {
        println!(
            "  {:<12} prompt {:>3}  tokens {:>2}  {:>7.2} tok/s  accept {:.3}",
            case.name,
            case.prompt_tokens,
            case.tokens,
            case.tokens_per_second(),
            case.acceptance.mean(),
        );
    }
    println!("--- IndexShare ---");
    for trial in &report.index_share {
        println!(
            "  prompt {:>4}  shared {:.3} in {:?}  own {:.3} in {:?}  identical {}",
            trial.prompt_tokens,
            trial.shared.mean(),
            trial.shared_elapsed,
            trial.own.mean(),
            trial.own_elapsed,
            trial.identical,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_flash_next_mtp_generation_list_identity_is_exact() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2]), Some(2));
        assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
    }

    #[test]
    fn qwen38_flash_next_mtp_generation_case_inventory_is_exact() {
        let dense_boundaries = QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES
            .iter()
            .filter(|visible| **visible <= QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING)
            .count();
        assert_eq!(dense_boundaries, 5);
        assert_eq!(
            QWEN38_FLASH_NEXT_GOLDEN_PROMPTS.len()
                + QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES.len()
                + FRESH_PROMPTS.len(),
            22
        );
        assert_eq!(
            [2_046, 2_047, 2_048, 2_049, 2_050].map(|prompt| dense_budget(prompt, BOUNDARY_TOKENS)),
            [6, 5, 4, 3, 2]
        );
        for prompt in FRESH_PROMPTS {
            assert!(dense_budget(prompt, IDENTITY_TOKENS) > 0);
        }
    }

    #[test]
    fn qwen38_flash_next_mtp_generation_benchmark_accounting_is_pinned() {
        assert_eq!(WARM_PROMPT_TOKENS, 512);
        assert_eq!(WARM_TOKENS, 64);
        assert_eq!(CODING_TOKENS, 32);
        assert_eq!(CODING_CASES.len(), 3);
        assert_eq!(
            QWEN38_FLASH_NEXT_MTP_ROUTES.len()
                + QWEN38_FLASH_NEXT_MTP_ATTENTION_ROUTES.len()
                + QWEN38_FLASH_NEXT_MTP_SEGMENTS,
            10
        );
        assert_eq!(
            (QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS + QWEN38_FLASH_NEXT_ATTENTION_LAYERS) * 16,
            976
        );
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive RTX 5090"]
    fn qwen38_flash_next_mtp_generation_source_backed_identity()
    -> Result<(), Qwen38FlashNextMtpGenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT")
            .ok_or_else(|| mismatch("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required"))?;
        let report = qualify_qwen38_flash_next_mtp_generation(Path::new(&root))?;
        print_qwen38_flash_next_mtp_generation_report(&report);

        assert_eq!(report.identical_cases(), 22);
        assert_eq!(report.cases.len(), 22);
        assert_eq!(report.selective_cases(), 8);
        assert!(report.compared_tokens() > 0);
        assert_eq!(report.replay_moved, 0);
        assert!(report.replay_compared > 0);
        assert_eq!(report.draft_executables, 10);
        assert_eq!(report.target_executables, 976);
        assert_eq!(report.target_slots, 5_578);
        assert_eq!(report.draft_slots, 128);
        assert_eq!(report.physical_pages, 3_765);
        assert!(report.device_bytes > 0);
        assert!(report.acceptance.rounds() > 0);
        assert_eq!(
            report.acceptance.rounds(),
            report.acceptance.distribution().iter().sum::<usize>()
        );
        assert!((1.0..=4.0).contains(&report.acceptance.mean()));
        assert_eq!(report.warm.prompt_tokens, WARM_PROMPT_TOKENS);
        assert!(report.warm.tokens > 0);
        assert!(rate(report.warm.tokens, report.warm.plain) > 0.0);
        assert!(rate(report.warm.tokens, report.warm.speculative) > 0.0);
        assert_eq!(report.coding.len(), CODING_CASES.len());
        assert!(report.coding.iter().all(|case| case.tokens > 0));
        assert!(
            report
                .coding
                .iter()
                .all(|case| case.tokens_per_second() > 0.0)
        );
        assert!(!report.index_share.is_empty());
        assert!(report.index_share.iter().all(|trial| trial.identical));

        Ok(())
    }
}
