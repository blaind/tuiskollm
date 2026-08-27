//! Source-backed qualification for the Qwen3.8 Flash-Next resident program.
//!
//! The suite checks route capture, streaming selections, stable addresses, cache-state
//! invariance, memory accounting, and model-level refusals. Timings are diagnostic only.

use crate::device_benchmark::{self, DeviceBenchmarkError};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tuisko_engine::{
    EngineError, MAX_BATCH, QWEN38_FLASH_NEXT_ATTENTION_LAYERS,
    QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT,
    QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS, QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
    QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS,
    Qwen38FlashNextResidentLayout, Qwen38FlashNextResidentModel, Qwen38FlashNextStepTelemetry,
    Qwen38FlashNextStreamingRoute, StreamingPrimarySource, StreamingResidencyAccounting,
};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

/// Warm passes needed for lazy module scratch to settle.
const WARM_PASSES: usize = 3;

/// Timed steps per measured route.
const MEASURED_STEPS: usize = 8;

/// Failure of the source-backed Qwen3.8 Flash-Next resident-program gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextResidentModelQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),

    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// The composed program disagreed with a contract it is required to hold.
    #[error("Qwen3.8 Flash-Next resident model qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextResidentModelQualificationError>;

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextResidentModelQualificationError {
    Qwen38FlashNextResidentModelQualificationError::Mismatch(message.into())
}

/// Measurement case used by the resident-model diagnostic sweep.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextMeasurementCase {
    /// Decode while positions advance across samples.
    DecodeSweep,
    /// One-row decode reset before every sample.
    DecodeControl,
    /// Causal verification reset before every sample.
    Causal,
    /// Prefill reset before every sample.
    Prefill,
}

impl Qwen38FlashNextMeasurementCase {
    /// Stable report label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DecodeSweep => "decode-sweep",
            Self::DecodeControl => "decode-control",
            Self::Causal => "causal",
            Self::Prefill => "prefill",
        }
    }
}

/// One diagnostic timing and the telemetry that explains it.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextRouteMeasurement {
    /// Route and harness being measured.
    pub case: Qwen38FlashNextMeasurementCase,
    /// Rows carried: `B` for decode, `K` for causal, or `T` for prefill.
    pub rows: usize,
    /// Median forward wall time across the measured steps.
    pub median: Duration,
    /// Fastest measured step, which bounds how much of the median is host overhead.
    pub fastest: Duration,
    /// Tokens per second at the median.
    pub tokens_per_second: f64,
    /// Milliseconds per token at the median.
    pub milliseconds_per_token: f64,
    /// Expert selections the whole stack made.
    pub expert_requests: usize,
    /// Host-to-device expert bytes the step streamed.
    pub expert_h2d_bytes: usize,
    /// Whole-stack expert hit rate over distinct per-round items.
    pub expert_hit_rate: f64,
    /// Per-layer hit rate in stack order.
    pub layer_hit_rates: Vec<f64>,
    /// Per-layer streamed bytes in stack order.
    pub layer_h2d_bytes: Vec<usize>,
    /// Token-embedding bytes uploaded.
    pub embedding_h2d_bytes: usize,
    /// Engram FP8 bytes uploaded and rows hashed.
    pub engram_h2d_bytes: usize,
    /// Bytes appended to the paged K/V planes.
    pub kv_append_bytes: usize,
}

/// Everything the resident-program gate observed.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextResidentModelQualification {
    /// Captured executables the program retains.
    pub executables: usize,
    /// Wall time spent capturing them.
    pub capture: Duration,
    /// Mean capture cost per executable.
    pub capture_per_executable: Duration,
    /// Wall time the weight sweep and expert staging took.
    pub weight_upload: Duration,
    /// Wall time page-locking the pool's host classes took.
    pub host_pin: Duration,
    /// Wall time the whole construction took.
    pub load: Duration,
    /// Device bytes the plan accounts for.
    pub device_resident_bytes: usize,
    /// Page-locked host bytes the plan accounts for.
    pub host_pinned_bytes: usize,
    /// File-backed host bytes the plan accounts for.
    pub host_mapped_bytes: usize,
    /// Free device bytes the driver reported after warmup.
    pub free_device_bytes_after_warmup: usize,
    /// Free device bytes the driver reported after the measured sweep.
    pub free_device_bytes_after_sweep: usize,
    /// Rounds refused by the mapping and position guards.
    pub refused_rounds: usize,
    /// Logits unchanged for a shallow row beside a selected deep row.
    pub isolated_logits: usize,
    /// Logit values inspected for finiteness and responsiveness.
    pub inspected_logits: usize,
    /// Logit values compared across two different cache states.
    pub cache_state_compared_logits: usize,
    /// Largest finite absolute logit the stack produced.
    pub peak_absolute_logit: f32,
    /// Measured cases in execution order.
    pub measurements: Vec<Qwen38FlashNextRouteMeasurement>,
    /// Agreement between a causal span and the same sequential decode.
    pub causal_span: Qwen38FlashNextCausalSpanEvidence,
    /// Agreement when the same round is replayed after rollback.
    pub rollback: Qwen38FlashNextRollbackEvidence,
    /// Agreement between host-stalled and consumer-stream publication waits.
    pub streaming_route: Qwen38FlashNextStreamingRouteEvidence,
}

/// Bitwise evidence for one four-row verification span.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextCausalSpanEvidence {
    /// Logit values compared.
    pub compared_logits: usize,
    /// Values whose represented bits differ.
    pub differing_logits: usize,
    /// Largest represented BF16 step observed.
    pub peak_represented_step: u16,
    /// Rows with the same argmax.
    pub agreeing_rows: usize,
}

/// Bitwise evidence that rollback restores every recurrent state family.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextRollbackEvidence {
    /// Host bytes in one slot restore point.
    pub snapshot_bytes: usize,
    /// Prefix length restored.
    pub restored_tokens: usize,
    /// Logits compared across rollback.
    pub compared_logits: usize,
    /// Logits moved by the no-rollback negative control.
    pub control_moved_logits: usize,
}

/// Bitwise and residency evidence for the two publication routes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextStreamingRouteEvidence {
    /// Logit values compared.
    pub compared_logits: usize,
    /// Cold uploads observed on the host-stalled route.
    pub stalling_misses: usize,
    /// Cold uploads observed on the consumer-stream route.
    pub overlapped_misses: usize,
    /// Consumer-stream rounds observed while publication was still in flight.
    pub rounds_in_flight: usize,
}

/// Loads the whole model, captures every segment, and measures a decode and a prefill route.
pub fn qualify_qwen38_flash_next_resident_model(
    root: &Path,
) -> QualResult<Qwen38FlashNextResidentModelQualification> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    // The plan is checked before the device is touched, so a posture or accounting error costs
    // nothing and never reaches an allocation.
    let plan = Qwen38FlashNextResidentLayout::build()?;
    verify_plan(&plan)?;

    let started = std::time::Instant::now();
    let mut model = Qwen38FlashNextResidentModel::from_snapshot(&context, Arc::clone(&snapshot))?;
    let load = started.elapsed();
    let stats = model.load_stats();

    let expected = (QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS + QWEN38_FLASH_NEXT_ATTENTION_LAYERS) * 16;
    if stats.executables() != expected {
        return Err(mismatch(format!(
            "the resident program captured {} executables, expected {expected}",
            stats.executables(),
        )));
    }
    println!("--- construction (diagnostic, nothing blessed) ---");
    println!("  total                    {:?}", load);
    println!("  weights + expert stage   {:?}", stats.weight_upload());
    println!("  host pin                 {:?}", stats.host_pin());
    println!(
        "  staged items / bytes     {} / {}",
        stats.staged_items(),
        stats.staged_bytes()
    );
    println!("--- captured route inventory and cost ---");
    println!("  executables              {}", stats.executables());
    println!("  capture wall time        {:?}", stats.graph_capture());
    println!(
        "  per executable           {:?}",
        stats.capture_per_executable()
    );

    let stable_base = model.base_address();
    let stable_kv_base = model.kv_base_address();

    let stream = context.new_stream().map_err(GpuError::from)?;
    model.reset_state(&stream)?;
    reserve_probe_slots(&mut model, &stream)?;

    let refused_rounds = verify_route_boundary(&mut model, &stream)?;
    let isolated_logits = verify_route_level_isolation(&mut model, &stream)?;
    let (inspected_logits, peak_absolute_logit) = verify_logits_respond(&mut model, &stream)?;
    let cache_state_compared_logits = verify_cache_state_is_not_numerical(&mut model, &stream)?;
    let streaming_route = verify_streaming_routes(&mut model, &stream)?;
    model.reset_state(&stream)?;
    reserve_probe_slots(&mut model, &stream)?;
    let causal_span = verify_causal_span_is_sequential(&mut model, &stream)?;
    let rollback = verify_rejected_round_rollback(&mut model, &stream)?;

    // Preserve completed route timings if a later route fails.
    let mut measurements = Vec::new();
    for batch in [1usize, MAX_BATCH] {
        measure_decode(&mut model, &stream, batch)?;
    }
    let free_after_warmup = free_device_bytes(&context)?;
    for batch in [1usize, MAX_BATCH] {
        let measurement = measure_decode(&mut model, &stream, batch)?;
        print_measurement(&measurement);
        measurements.push(measurement);
    }

    model.reset_state(&stream)?;
    reserve_probe_slots(&mut model, &stream)?;
    let measurement = measure_decode_control(&mut model, &stream)?;
    print_measurement(&measurement);
    measurements.push(measurement);
    for rows in [1usize, 4] {
        let measurement = measure_causal(&mut model, &stream, rows)?;
        print_measurement(&measurement);
        measurements.push(measurement);
    }

    model.reset_state(&stream)?;
    reserve_probe_slots(&mut model, &stream)?;
    for tile in [32usize, 64] {
        let measurement = measure_prefill(&mut model, &stream, tile)?;
        print_measurement(&measurement);
        measurements.push(measurement);
        model.reset_state(&stream)?;
        reserve_probe_slots(&mut model, &stream)?;
    }

    let free_after_sweep = free_device_bytes(&context)?;

    if model.base_address() != stable_base || model.kv_base_address() != stable_kv_base {
        return Err(mismatch(
            "a resident arena moved across the measured sweep, which every captured segment \
             would have kept pointing at its old address",
        ));
    }
    if let Some(reason) = model.poisoned() {
        return Err(mismatch(format!(
            "the expert cache poisoned itself during the sweep: {reason}"
        )));
    }

    Ok(Qwen38FlashNextResidentModelQualification {
        executables: stats.executables(),
        capture: stats.graph_capture(),
        capture_per_executable: stats.capture_per_executable(),
        weight_upload: stats.weight_upload(),
        host_pin: stats.host_pin(),
        load,
        device_resident_bytes: plan.device_resident_bytes(),
        host_pinned_bytes: plan.host_pinned_bytes(),
        host_mapped_bytes: plan.host_mapped_bytes(),
        free_device_bytes_after_warmup: free_after_warmup,
        free_device_bytes_after_sweep: free_after_sweep,
        refused_rounds,
        isolated_logits,
        inspected_logits,
        peak_absolute_logit,
        cache_state_compared_logits,
        measurements,
        causal_span,
        rollback,
        streaming_route,
    })
}

/// Requires bit-identical logits for cold and LRU-churned expert-cache states.
fn verify_cache_state_is_not_numerical(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<usize> {
    const PROBE: u32 = 7_919;

    model.reset_state(stream)?;
    reserve_probe_slots(model, stream)?;
    model.decode_step(stream, &[PROBE], &[0], &[0])?;
    let cold = model.read_logits(stream, 1)?.to_vec();

    // Other slots churn the shared expert LRU without changing the probe sequence.
    for round in 0..24u32 {
        let tokens = (1..MAX_BATCH)
            .map(|slot| (30_011 + round * 977 + slot as u32 * 131) % 248_320)
            .collect::<Vec<_>>();
        let positions = vec![round; MAX_BATCH - 1];
        let slots = (1..MAX_BATCH).collect::<Vec<_>>();
        model.decode_step(stream, &tokens, &positions, &slots)?;
    }

    // Only the sequence carry is cleared. The cache keeps whatever the churn left in it.
    model.reset_slot(stream, 0)?;
    let second = model.decode_step(stream, &[PROBE], &[0], &[0])?;

    // Refills prove that the second run resolved through a different slot table.
    let refilled = second
        .layers()
        .iter()
        .map(|layer| layer.misses())
        .sum::<usize>();
    if refilled == 0 {
        return Err(mismatch(
            "the churn evicted none of the probe's experts, so comparing the two runs would \
             compare one cache state with itself; the test needs a wider churn to mean anything",
        ));
    }

    let warm = model.read_logits(stream, 1)?;

    if cold.len() != warm.len() {
        return Err(mismatch(
            "the two cache states published different logit widths",
        ));
    }
    if let Some((index, (&left, &right))) = cold
        .iter()
        .zip(warm.iter())
        .enumerate()
        .find(|(_, (left, right))| left != right)
    {
        return Err(mismatch(format!(
            "cache state changed a produced bit: vocabulary index {index} read {left:#06x} from a \
             cold pool and {right:#06x} after the LRU moved its experts, so the indirection table \
             is being read as data rather than as an address"
        )));
    }

    Ok(cold.len())
}

/// Requires both publication routes to produce identical cold-cache logits.
fn verify_streaming_routes(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<Qwen38FlashNextStreamingRouteEvidence> {
    const PROBE: u32 = 4_637;

    if model.streaming_route() != Qwen38FlashNextStreamingRoute::Overlapped {
        return Err(mismatch("the production streaming route is not overlapped"));
    }

    let run = |model: &mut Qwen38FlashNextResidentModel,
               route: Qwen38FlashNextStreamingRoute|
     -> QualResult<(Vec<u16>, usize, usize)> {
        model.set_streaming_route(route);
        model.reset_state(stream)?;
        reserve_probe_slots(model, stream)?;
        let step = model.decode_step(stream, &[PROBE], &[0], &[0])?;
        let logits = model.read_logits(stream, 1)?.to_vec();
        let misses = step.layers().iter().map(|layer| layer.misses()).sum();

        Ok((logits, misses, step.overlapped_rounds()))
    };

    let (stalling, stalling_misses, _) = run(model, Qwen38FlashNextStreamingRoute::Stalling)?;
    let (overlapped, overlapped_misses, rounds_in_flight) =
        run(model, Qwen38FlashNextStreamingRoute::Overlapped)?;

    if overlapped_misses == 0 || stalling_misses != overlapped_misses {
        return Err(mismatch(format!(
            "streaming routes observed {stalling_misses} and {overlapped_misses} cold uploads"
        )));
    }
    if let Some((index, (&left, &right))) = stalling
        .iter()
        .zip(&overlapped)
        .enumerate()
        .find(|(_, (left, right))| left != right)
    {
        return Err(mismatch(format!(
            "streaming route changed vocabulary logit {index}: {left:#06x} versus {right:#06x}"
        )));
    }

    Ok(Qwen38FlashNextStreamingRouteEvidence {
        compared_logits: stalling.len(),
        stalling_misses,
        overlapped_misses,
        rounds_in_flight,
    })
}

/// Requires the causal span to match four decode steps bit for bit.
fn verify_causal_span_is_sequential(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<Qwen38FlashNextCausalSpanEvidence> {
    const SLOT: usize = 0;
    const TOKENS: [u32; 4] = [101, 7_919, 48_127, 199_933];

    let vocab = <A as Arch>::VOCAB;
    let mut sequential = Vec::with_capacity(TOKENS.len() * vocab);

    model.reset_state(stream)?;
    reserve_probe_slots(model, stream)?;
    for (position, &token) in TOKENS.iter().enumerate() {
        model.decode_step(stream, &[token], &[position as u32], &[SLOT])?;
        sequential.extend_from_slice(model.read_logits(stream, 1)?);
    }

    model.reset_state(stream)?;
    reserve_probe_slots(model, stream)?;
    model.verify_step(stream, &TOKENS, 0, SLOT)?;
    let span = model.read_logits(stream, TOKENS.len())?.to_vec();
    if span.len() != sequential.len() {
        return Err(mismatch(format!(
            "the causal span published {} logits, expected {}",
            span.len(),
            sequential.len()
        )));
    }

    let mut differing = 0usize;
    let mut peak_step = 0u16;
    for (&expected, &observed) in sequential.iter().zip(&span) {
        if expected != observed {
            differing += 1;
            peak_step = peak_step.max(expected.abs_diff(observed));
        }
    }
    if differing != 0 {
        return Err(mismatch(format!(
            "the causal span differs in {differing} of {} logits, by up to {peak_step} represented BF16 steps",
            span.len()
        )));
    }

    let mut agreeing_rows = 0usize;
    for row in 0..TOKENS.len() {
        let begin = row * vocab;
        let expected = argmax_row(&sequential[begin..begin + vocab]);
        let observed = argmax_row(&span[begin..begin + vocab]);
        if expected != observed {
            return Err(mismatch(format!(
                "causal span row {row} chose token {observed}, expected {expected}"
            )));
        }
        agreeing_rows += 1;
    }

    Ok(Qwen38FlashNextCausalSpanEvidence {
        compared_logits: span.len(),
        differing_logits: differing,
        peak_represented_step: peak_step,
        agreeing_rows,
    })
}

/// Requires a rejected verification round to leave no recurrent state behind.
fn verify_rejected_round_rollback(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<Qwen38FlashNextRollbackEvidence> {
    const SLOT: usize = 0;
    const PREFIX: [u32; 3] = [1_009, 24_419, 88_651];
    const SPAN: [u32; 4] = [101, 7_919, 48_127, 199_933];

    model.reset_state(stream)?;
    reserve_probe_slots(model, stream)?;
    for (position, &token) in PREFIX.iter().enumerate() {
        model.decode_step(stream, &[token], &[position as u32], &[SLOT])?;
    }

    let first_position = PREFIX.len() as u32;
    let snapshot = model.snapshot_slot(stream, SLOT)?;
    if snapshot.tokens() != PREFIX.len() {
        return Err(mismatch(format!(
            "restore point holds {} tokens, expected {}",
            snapshot.tokens(),
            PREFIX.len()
        )));
    }

    model.verify_step(stream, &SPAN, first_position, SLOT)?;
    let rejected = model.read_logits(stream, SPAN.len())?.to_vec();

    model.restore_slot(stream, &snapshot)?;
    if model.slot_tokens(SLOT)? != PREFIX.len() {
        return Err(mismatch("rollback did not restore the prefix length"));
    }
    model.verify_step(stream, &SPAN, first_position, SLOT)?;
    let replayed = model.read_logits(stream, SPAN.len())?.to_vec();

    let differing = rejected
        .iter()
        .zip(&replayed)
        .filter(|(first, second)| first != second)
        .count();
    if differing != 0 {
        return Err(mismatch(format!(
            "round replay after rollback differs in {differing} of {} logits",
            rejected.len()
        )));
    }

    // A no-op restore would pass the equality above without this control.
    let control_position = first_position + SPAN.len() as u32;
    model.verify_step(stream, &SPAN, control_position, SLOT)?;
    let unrestored = model.read_logits(stream, SPAN.len())?.to_vec();
    let moved = rejected
        .iter()
        .zip(&unrestored)
        .filter(|(first, second)| first != second)
        .count();
    if moved == 0 {
        return Err(mismatch("no-rollback control did not move any logits"));
    }

    Ok(Qwen38FlashNextRollbackEvidence {
        snapshot_bytes: snapshot.byte_len(),
        restored_tokens: snapshot.tokens(),
        compared_logits: rejected.len(),
        control_moved_logits: moved,
    })
}

fn argmax_row(row: &[u16]) -> usize {
    let value = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut best = 0usize;
    for (index, &bits) in row.iter().enumerate().skip(1) {
        if value(bits) > value(row[best]) {
            best = index;
        }
    }

    best
}

/// Reserves every slot driven directly by this suite.
fn reserve_probe_slots(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<()> {
    for slot in 0..MAX_BATCH {
        model.reserve_slot(stream, slot, QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING)?;
    }

    Ok(())
}

/// Checks finite, nondegenerate, input-sensitive logits at the composed boundary.
fn verify_logits_respond(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<(usize, f32)> {
    let mut fingerprints = Vec::new();
    let mut peak = 0.0f32;
    let mut inspected = 0usize;

    for token in [11u32, 5_003, 100_007] {
        model.reset_slot(stream, 0)?;
        model.decode_step(stream, &[token], &[0], &[0])?;
        let logits = model.read_logits(stream, 1)?;
        if logits.len() != <A as Arch>::VOCAB {
            return Err(mismatch(format!(
                "the head published {} logits for one row, expected {}",
                logits.len(),
                <A as Arch>::VOCAB
            )));
        }

        let mut best = (0usize, f32::NEG_INFINITY);
        let mut distinct = std::collections::BTreeSet::new();
        for (index, &bits) in logits.iter().enumerate() {
            let value = f32::from_bits(u32::from(bits) << 16);
            if !value.is_finite() {
                return Err(mismatch(format!(
                    "token {token} produced a non-finite logit at vocabulary index {index}"
                )));
            }
            peak = peak.max(value.abs());
            if value > best.1 {
                best = (index, value);
            }
            distinct.insert(bits);
        }
        inspected += logits.len();

        // A zeroed or never-written head plane collapses to one value across 248,320 entries.
        if distinct.len() < 1_024 {
            return Err(mismatch(format!(
                "token {token} produced only {} distinct logit values over {} entries, which is \
                 what a head reading an unwritten plane looks like",
                distinct.len(),
                logits.len()
            )));
        }
        fingerprints.push((token, best.0));
    }

    // Different inputs must reach different outputs, or the stack is not carrying its input.
    if fingerprints
        .iter()
        .all(|&(_, argmax)| argmax == fingerprints[0].1)
    {
        return Err(mismatch(format!(
            "every probed token produced the same argmax ({}), so the stack is not responding to \
             its input",
            fingerprints[0].1
        )));
    }

    Ok((inspected, peak))
}

/// Checks the accounting the plan is required to reproduce before anything is allocated.
fn verify_plan(plan: &Qwen38FlashNextResidentLayout) -> QualResult<()> {
    if plan.streaming().item_count() != QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT
        || plan.streaming().slot_count() != QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS
    {
        return Err(mismatch(format!(
            "the plan admits {} items over {} slots, expected {QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT} over \
             {QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS}",
            plan.streaming().item_count(),
            plan.streaming().slot_count()
        )));
    }
    if plan.streaming().primary_source() != StreamingPrimarySource::Mapped {
        return Err(mismatch(
            "this box admits the mapped-primary posture: the fully pinned pool is 63.28 GiB \
             against 59.2 GiB of usable RAM and cannot be allocated",
        ));
    }
    if plan.host_pinned_bytes() != 7_585_611_776 || plan.host_mapped_bytes() != 112_869_621_760 {
        return Err(mismatch(format!(
            "the 64 GB posture reports {} pinned and {} mapped bytes, expected 7,585,611,776 and \
             112,869,621,760",
            plan.host_pinned_bytes(),
            plan.host_mapped_bytes()
        )));
    }
    if plan.physical_pages() != QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
        || plan.context_tokens_per_slot() != 32_768
    {
        return Err(mismatch(format!(
            "the funded KV pool is {} pages reaching {} tokens per slot, expected {} and 32,768",
            plan.physical_pages(),
            plan.context_tokens_per_slot(),
            QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
        )));
    }
    if plan.context_tokens_per_slot() <= QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
        return Err(mismatch(
            "the funded KV pool never reaches the selected route",
        ));
    }

    Ok(())
}

/// Proves isolation when one batch mixes dense-band and selected rows.
fn verify_route_level_isolation(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<usize> {
    const SHALLOW_POSITION: u32 = 64;
    const DEEP_POSITION: u32 = 2_060;
    let deep = 0usize;
    let shallow = 1usize;

    model.reserve_slot(stream, deep, DEEP_POSITION as usize + 1)?;
    model.reserve_slot(stream, shallow, SHALLOW_POSITION as usize + 1)?;

    model.reset_slot(stream, shallow)?;
    model.prefill_tile(stream, &[7; 64], 0, shallow)?;
    model.decode_step(stream, &[7], &[SHALLOW_POSITION], &[shallow])?;
    let mut alone = vec![0u16; <A as Arch>::VOCAB];
    model.read_logits_into(stream, 1, &mut alone)?;

    model.reset_slot(stream, shallow)?;
    model.prefill_tile(stream, &[7; 64], 0, shallow)?;
    model.decode_step(
        stream,
        &[7, 7],
        &[DEEP_POSITION, SHALLOW_POSITION],
        &[deep, shallow],
    )?;
    let counts = model.qualification_selected_counts(stream, 2)?;
    if counts[1] as usize != SHALLOW_POSITION as usize + 1 || counts[0] < 2_048 {
        return Err(mismatch(format!(
            "mixed-depth selection published counts {:?}, expected deep >= 2,048 and shallow {}",
            counts,
            SHALLOW_POSITION + 1
        )));
    }

    let mut beside = vec![0u16; 2 * <A as Arch>::VOCAB];
    model.read_logits_into(stream, 2, &mut beside)?;
    for (index, (&left, &right)) in alone.iter().zip(&beside[<A as Arch>::VOCAB..]).enumerate() {
        if left != right {
            return Err(mismatch(format!(
                "a deep neighbor changed shallow logit {index}: {left:#06x} versus {right:#06x}"
            )));
        }
    }

    Ok(alone.len())
}

/// Proves selected routing and the remaining exact refusals.
fn verify_route_boundary(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<usize> {
    let mut refused = 0usize;
    let slot = 0usize;
    let depth = QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING + QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS;

    model.recycle_slot(stream, slot)?;
    let error = model
        .decode_step(stream, &[1], &[0], &[slot])
        .err()
        .ok_or_else(|| mismatch("a step on an unmapped slot was admitted"))?;
    let message = error.to_string();
    if !message.contains("slot 0") || !message.contains("Free") {
        return Err(mismatch(format!(
            "the mapping refusal is incomplete: {error}"
        )));
    }
    refused += 1;

    let error = model
        .decode_step(stream, &[1], &[A::MAX_POSITION_EMBEDDINGS as u32], &[slot])
        .err()
        .ok_or_else(|| mismatch("a step past the position ceiling was admitted"))?;
    let message = error.to_string();
    if !message.contains(&A::MAX_POSITION_EMBEDDINGS.to_string())
        || !message.contains("refused rather than truncated")
    {
        return Err(mismatch(format!(
            "the position-ceiling refusal is incomplete: {message}"
        )));
    }
    refused += 1;

    model.reserve_slot(stream, slot, depth)?;
    let prompt = vec![1u32; 1_024];
    model.prefill_tile(stream, &prompt, 0, slot)?;
    model.prefill_tile(stream, &prompt, 1_024, slot)?;
    for position in 2_048..=2_059 {
        model.decode_step(stream, &[1], &[position], &[slot])?;
    }
    let selected = model.qualification_selected_counts(stream, 1)?;
    if selected != [2_048] {
        return Err(mismatch(format!(
            "the selected route published {selected:?}, expected [2048]"
        )));
    }
    if model.generation_capacity() != A::MAX_POSITION_EMBEDDINGS {
        return Err(mismatch("the resident generator hides funded context"));
    }

    Ok(refused)
}

fn measure_decode(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
    batch: usize,
) -> QualResult<Qwen38FlashNextRouteMeasurement> {
    let tokens = (0..batch)
        .map(|row| (1_024 + row * 97) as u32)
        .collect::<Vec<_>>();
    let slots = (0..batch).collect::<Vec<_>>();
    let rounds = WARM_PASSES + MEASURED_STEPS;
    for &slot in &slots {
        model.reset_slot(stream, slot)?;
        model.reserve_slot(stream, slot, rounds)?;
    }

    for warm in 0..WARM_PASSES {
        let positions = vec![warm as u32; batch];
        model.decode_step(stream, &tokens, &positions, &slots)?;
    }

    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for step in 0..MEASURED_STEPS {
        let positions = vec![(WARM_PASSES + step) as u32; batch];
        samples.push(model.decode_step(stream, &tokens, &positions, &slots)?);
    }

    Ok(summarize(
        Qwen38FlashNextMeasurementCase::DecodeSweep,
        batch,
        &samples,
    ))
}

/// Times one-row decode under the causal span's reset-per-sample harness.
fn measure_decode_control(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
) -> QualResult<Qwen38FlashNextRouteMeasurement> {
    let token = [4_096u32];

    for _ in 0..WARM_PASSES {
        model.reset_slot(stream, 0)?;
        model.decode_step(stream, &token, &[0], &[0])?;
    }

    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for _ in 0..MEASURED_STEPS {
        model.reset_slot(stream, 0)?;
        samples.push(model.decode_step(stream, &token, &[0], &[0])?);
    }

    Ok(summarize(
        Qwen38FlashNextMeasurementCase::DecodeControl,
        1,
        &samples,
    ))
}

/// Times a causal verification span under the same reset-per-sample harness.
fn measure_causal(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
    rows: usize,
) -> QualResult<Qwen38FlashNextRouteMeasurement> {
    let span = (0..rows)
        .map(|row| (4_096 + row * 131) as u32)
        .collect::<Vec<_>>();

    for _ in 0..WARM_PASSES {
        model.reset_slot(stream, 0)?;
        model.verify_step(stream, &span, 0, 0)?;
    }

    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for _ in 0..MEASURED_STEPS {
        model.reset_slot(stream, 0)?;
        samples.push(model.verify_step(stream, &span, 0, 0)?);
    }

    Ok(summarize(
        Qwen38FlashNextMeasurementCase::Causal,
        rows,
        &samples,
    ))
}

fn measure_prefill(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &tuisko_gpu::CudaStream,
    tokens: usize,
) -> QualResult<Qwen38FlashNextRouteMeasurement> {
    let prompt = (0..tokens)
        .map(|token| (2_048 + token) as u32)
        .collect::<Vec<_>>();
    model.reserve_slot(stream, 0, tokens)?;

    for _ in 0..WARM_PASSES {
        model.reset_slot(stream, 0)?;
        model.prefill_tile(stream, &prompt, 0, 0)?;
    }

    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for _ in 0..MEASURED_STEPS {
        model.reset_slot(stream, 0)?;
        samples.push(model.prefill_tile(stream, &prompt, 0, 0)?);
    }

    Ok(summarize(
        Qwen38FlashNextMeasurementCase::Prefill,
        tokens,
        &samples,
    ))
}

fn summarize(
    case: Qwen38FlashNextMeasurementCase,
    rows: usize,
    samples: &[Qwen38FlashNextStepTelemetry],
) -> Qwen38FlashNextRouteMeasurement {
    let mut times = samples
        .iter()
        .map(|sample| sample.forward())
        .collect::<Vec<_>>();
    times.sort_unstable();
    let median = times[times.len() / 2];
    let fastest = times[0];
    let last = samples
        .last()
        .expect("a measured route has at least one step");

    Qwen38FlashNextRouteMeasurement {
        case,
        rows,
        median,
        fastest,
        tokens_per_second: rows as f64 / median.as_secs_f64(),
        milliseconds_per_token: median.as_secs_f64() * 1_000.0 / rows as f64,
        expert_requests: last.expert_requests(),
        expert_h2d_bytes: last.expert_h2d_bytes(),
        expert_hit_rate: last.expert_hit_rate(),
        layer_hit_rates: last.layers().iter().map(|layer| layer.hit_rate()).collect(),
        layer_h2d_bytes: last
            .layers()
            .iter()
            .map(|layer| layer.uploaded_bytes())
            .collect(),
        embedding_h2d_bytes: last.embedding_h2d_bytes(),
        engram_h2d_bytes: last.engram_h2d_bytes(),
        kv_append_bytes: last.kv_append_bytes(),
    }
}

fn free_device_bytes(context: &Arc<CudaContext>) -> QualResult<usize> {
    Ok(device_memory_info(context)?.free_bytes)
}

/// Prints one qualification report in the house's diagnostic shape.
pub fn print_qwen38_flash_next_resident_model_report(
    report: &Qwen38FlashNextResidentModelQualification,
) {
    println!("Qwen3.8 Flash-Next resident model - diagnostic, nothing blessed");
    println!("  construction");
    println!("    total                    {:?}", report.load);
    println!("    weights + expert stage   {:?}", report.weight_upload);
    println!("    host pin                 {:?}", report.host_pin);
    println!("  captured route inventory and cost");
    println!("    executables              {}", report.executables);
    println!("    capture wall time        {:?}", report.capture);
    println!(
        "    per executable           {:?}",
        report.capture_per_executable
    );
    println!("  residency classes");
    println!(
        "    device_resident_bytes    {}",
        report.device_resident_bytes
    );
    println!("    host_pinned_bytes        {}", report.host_pinned_bytes);
    println!("    host_mapped_bytes        {}", report.host_mapped_bytes);
    println!(
        "    free device after warm   {}",
        report.free_device_bytes_after_warmup
    );
    println!(
        "    free device after sweep  {}",
        report.free_device_bytes_after_sweep
    );
    println!("  refused rounds             {}", report.refused_rounds);
    println!("  isolated logits            {}", report.isolated_logits);
    println!("  logits inspected           {}", report.inspected_logits);
    println!(
        "  cache-state logits equal   {}",
        report.cache_state_compared_logits
    );
    println!(
        "  streaming-route logits eq  {} over {} cold uploads, {} rounds in flight",
        report.streaming_route.compared_logits,
        report.streaming_route.overlapped_misses,
        report.streaming_route.rounds_in_flight
    );
    println!(
        "  peak |logit|               {:.4}",
        report.peak_absolute_logit
    );
    println!(
        "  causal span                {} rows, {}/{} differing, peak step {}",
        report.causal_span.agreeing_rows,
        report.causal_span.differing_logits,
        report.causal_span.compared_logits,
        report.causal_span.peak_represented_step
    );
    println!(
        "  rollback                   {} logits equal at {} tokens ({} B); {} control moved",
        report.rollback.compared_logits,
        report.rollback.restored_tokens,
        report.rollback.snapshot_bytes,
        report.rollback.control_moved_logits
    );
    for measurement in &report.measurements {
        print_measurement(measurement);
    }
}

/// Prints one measured case, used both live and in the summary.
pub fn print_measurement(measurement: &Qwen38FlashNextRouteMeasurement) {
    {
        println!(
            "  route case={} rows={}",
            measurement.case.as_str(),
            measurement.rows
        );
        println!("    median forward           {:?}", measurement.median);
        println!("    fastest forward          {:?}", measurement.fastest);
        println!(
            "    tok/s                    {:.2}",
            measurement.tokens_per_second
        );
        println!(
            "    ms/token                 {:.3}",
            measurement.milliseconds_per_token
        );
        println!(
            "    expert requests          {}",
            measurement.expert_requests
        );
        println!(
            "    expert h2d bytes         {}",
            measurement.expert_h2d_bytes
        );
        println!(
            "    expert hit rate          {:.4}",
            measurement.expert_hit_rate
        );
        println!(
            "    embedding h2d bytes      {}",
            measurement.embedding_h2d_bytes
        );
        println!(
            "    engram h2d bytes         {}",
            measurement.engram_h2d_bytes
        );
        println!(
            "    kv append bytes          {}",
            measurement.kv_append_bytes
        );
        let weakest = measurement
            .layer_hit_rates
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1));
        if let Some((layer, rate)) = weakest {
            println!("    weakest layer            {layer} at {rate:.4}");
        }
        for layer in (0..measurement.layer_hit_rates.len()).step_by(8) {
            println!(
                "      layer {:>2}  hit {:.4}  h2d {:>10}",
                layer, measurement.layer_hit_rates[layer], measurement.layer_h2d_bytes[layer]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_segment_inventory_is_the_one_the_plan_derives() {
        assert_eq!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, <A as Arch>::LAYERS + 1);
        assert_eq!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS * 12, 588);
        assert_eq!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS * 4, 196);
    }

    #[test]
    fn the_predicted_per_token_contract_is_the_one_this_suite_reports() {
        // Forty-eight layers make ten routed selections per row.
        let requests = <A as Arch>::LAYERS * A::NUM_EXPERTS_PER_TOKEN;
        assert_eq!(requests, 480);
        assert_eq!(requests * 2_764_800, 1_327_104_000);
        // A modeled 0.85 hit rate leaves 72 misses.
        assert_eq!((requests as f64 * 0.15).round() as usize, 72);
        assert_eq!(72 * 2_764_800, 199_065_600);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn the_source_backed_resident_model_captures_and_decodes()
    -> Result<(), Qwen38FlashNextResidentModelQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").ok_or_else(|| {
            Qwen38FlashNextResidentModelQualificationError::Mismatch(
                "TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT is required for the source-backed gate"
                    .to_string(),
            )
        })?;
        let report = qualify_qwen38_flash_next_resident_model(std::path::Path::new(&root))?;
        print_qwen38_flash_next_resident_model_report(&report);

        assert_eq!(report.executables, 588 + 144 + 196 + 48);
        assert_eq!(report.causal_span.agreeing_rows, 4);
        assert_eq!(report.causal_span.compared_logits, 4 * <A as Arch>::VOCAB);
        assert_eq!(report.causal_span.differing_logits, 0);
        assert_eq!(report.causal_span.peak_represented_step, 0);
        assert_eq!(report.rollback.restored_tokens, 3);
        assert_eq!(report.rollback.compared_logits, 4 * <A as Arch>::VOCAB);
        assert!(report.rollback.snapshot_bytes > 0);
        assert!(report.rollback.control_moved_logits > 0);
        assert_eq!(report.refused_rounds, 2);
        assert_eq!(report.isolated_logits, <A as Arch>::VOCAB);
        assert_eq!(report.inspected_logits, 3 * 248_320);
        assert_eq!(report.cache_state_compared_logits, 248_320);
        assert!(report.peak_absolute_logit.is_finite());
        assert!(report.peak_absolute_logit > 0.0);
        assert_eq!(report.host_pinned_bytes, 7_585_611_776);
        assert_eq!(report.host_mapped_bytes, 112_869_621_760);
        assert_eq!(
            report
                .measurements
                .iter()
                .map(|measurement| (measurement.case, measurement.rows))
                .collect::<Vec<_>>(),
            [
                (Qwen38FlashNextMeasurementCase::DecodeSweep, 1),
                (Qwen38FlashNextMeasurementCase::DecodeSweep, MAX_BATCH),
                (Qwen38FlashNextMeasurementCase::DecodeControl, 1),
                (Qwen38FlashNextMeasurementCase::Causal, 1),
                (Qwen38FlashNextMeasurementCase::Causal, 4),
                (Qwen38FlashNextMeasurementCase::Prefill, 32),
                (Qwen38FlashNextMeasurementCase::Prefill, 64),
            ]
        );
        assert!(
            report
                .measurements
                .iter()
                .all(|measurement| measurement.tokens_per_second.is_finite())
        );
        // Free memory is device-wide, so a display workload can move it while this process keeps
        // its fixed arenas. Refuse growth beyond a bounded desktop allocation.
        let device_growth = report
            .free_device_bytes_after_warmup
            .saturating_sub(report.free_device_bytes_after_sweep);
        assert!(
            device_growth <= 64 * 1024 * 1024,
            "device memory grew by {device_growth} bytes across the measured sweep",
        );

        Ok(())
    }
}
