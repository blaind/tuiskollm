//! Source-backed qualification for the complete Qwen3.6 MTP transformer layer.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16};
use crate::oracles::attention::rope_tables;
use crate::oracles::norm::residual_oracle;
use crate::residual_norm::rms_norm_oracle;
use crate::{
    DeviceBenchmarkError, qualify_qwen36_fp8_attention_qk_prepare, qualify_qwen36_fp8_paged_gqa,
    qualify_qwen36_moe_router, qualify_qwen36_mtp_bf16_attention_output,
    qualify_qwen36_mtp_bf16_fusion, qualify_qwen36_mtp_bf16_moe, qualify_qwen36_mtp_bf16_qkv,
};
use std::path::Path;
use tuisko_engine::{EngineError, MAX_BATCH, Qwen36MtpLayerObservables, Qwen36MtpLayerProgram};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen36Moe35B, Qwen36MtpBindings};

const REALIGN_ROUTES: usize = 4;
const PROMPT_ROUTES: [usize; 3] = [32, 64, 128];
const TABLE_STRIDE: usize = 3;
const PHYSICAL_PAGES: usize = MAX_BATCH * TABLE_STRIDE;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const CACHE_POSITIONS: [u32; MAX_BATCH] = [0, 1, 63, 64, 65, 97, 128, 130];
const INPUT_PATTERN: [f32; 16] = [
    0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.1875, -0.1875, 0.09375,
    -0.09375, 0.046875, -0.046875, 0.015625, -0.015625,
];

/// Failure of the complete source-backed Qwen3.6 MTP layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36MtpLayerQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership, launch, or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with an independent oracle or route contract.
    #[error("Qwen3.6 MTP layer qualification failed: {0}")]
    Mismatch(String),
}

/// Checked graph, seam, oracle, and byte counts for one Qwen3.6 MTP layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36MtpLayerQualification {
    /// Independent represented-value leaf suites completed before composition.
    pub leaf_oracle_suites: usize,
    /// Active residual and normalization values checked mathematically.
    pub boundary_values: usize,
    /// Active values compared with independently launched exact B=1 routes.
    pub route_reference_values: usize,
    /// Complete mutable owner state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Values proved unwritten beyond the prime-only cache-append seam.
    pub prime_sentinel_values: usize,
    /// Causal realignment outputs matched against sequential full routes.
    pub realignment_values: usize,
    /// Device weight samples matched to source and remained immutable.
    pub immutable_source_values: usize,
    /// Exact unchanged source-BF16 MTP weight bytes.
    pub resident_weight_bytes: usize,
    /// Exact represented E4M3 cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Complete owner bytes without padding.
    pub owner_bytes: usize,
    /// Complete single-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Number of immutable draft, prime, realignment, and prompt graphs.
    pub graph_count: usize,
    /// Largest absolute boundary-oracle error.
    pub maximum_absolute_error: f32,
}

struct Norms {
    embedding: Vec<u16>,
    hidden: Vec<u16>,
    input: Vec<u16>,
    post_attention: Vec<u16>,
    final_norm: Vec<u16>,
}

struct Fixture {
    embedding: Vec<u16>,
    target_hidden: Vec<u16>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
}

/// Qualifies draft, prime, causal realignment, and exact prompt-prime graphs.
pub fn qualify_qwen36_mtp_layer(
    root: &Path,
) -> Result<Qwen36MtpLayerQualification, Qwen36MtpLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    run_leaf_oracles(root)?;

    let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(root)?;
    let bindings = Qwen36MtpBindings::bind(&snapshot)?;
    let norms = Norms {
        embedding: bindings.embedding_norm.words().collect(),
        hidden: bindings.hidden_norm.words().collect(),
        input: bindings.input_norm.words().collect(),
        post_attention: bindings.post_attention_norm.words().collect(),
        final_norm: bindings.final_norm.words().collect(),
    };
    let source_samples = immutable_source_samples(&bindings)?;
    let fixture = fixture();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = Qwen36MtpLayerProgram::from_snapshot(&context, &snapshot)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 47 {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "Qwen3.6 MTP owner exposes {} addresses, expected 47",
            stable_addresses.len()
        )));
    }
    verify_accounting(&program)?;
    let immutable_before = program.qualification_immutable_samples(&stream)?;
    compare_exact(
        "immutable source samples",
        &immutable_before,
        &source_samples,
    )?;
    let mut report = Qwen36MtpLayerQualification {
        leaf_oracle_suites: 7,
        boundary_values: 0,
        route_reference_values: 0,
        graph_replay_values: 0,
        prime_sentinel_values: 0,
        realignment_values: 0,
        immutable_source_values: immutable_before.len(),
        resident_weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        owner_bytes: program.owner_bytes(),
        arena_bytes: program.arena_bytes(),
        padding_bytes: program.arena_bytes() - program.owner_bytes(),
        graph_count: program.graph_count(),
        maximum_absolute_error: 0.0,
    };

    let reference = b1_route_references(&program, &stream, &fixture)?;
    for batch in 1..=MAX_BATCH {
        prepare_draft(&program, &stream, batch, &fixture)?;
        program.launch_eager_draft(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;
        verify_metadata(batch, &fixture, &eager)?;
        verify_boundaries(batch, &fixture, &norms, &eager, &mut report)?;
        compare_active(batch, &reference, &eager, &mut report)?;
        verify_inactive(batch, &eager)?;

        prepare_draft(&program, &stream, batch, &fixture)?;
        program.replay_draft(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;
        verify_replay(&eager, &replay, &mut report)?;
        verify_stable(&program, stable_base, &stable_addresses, "draft", batch)?;
    }

    for tokens in 1..=REALIGN_ROUTES {
        verify_prime_route(
            &program,
            &stream,
            tokens,
            &fixture,
            stable_base,
            &stable_addresses,
            &mut report,
        )?;
        verify_realign_route(
            &program,
            &stream,
            tokens,
            &fixture,
            stable_base,
            &stable_addresses,
            &mut report,
        )?;
    }
    for rows in PROMPT_ROUTES {
        verify_prompt_route(
            &program,
            &stream,
            rows,
            &fixture,
            stable_base,
            &stable_addresses,
            &mut report,
        )?;
    }

    let immutable_after = program.qualification_immutable_samples(&stream)?;
    compare_exact(
        "immutable weight samples after every route",
        &immutable_after,
        &immutable_before,
    )?;
    report.immutable_source_values += immutable_after.len();

    verify_no_post_warmup_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn immutable_source_samples(
    bindings: &Qwen36MtpBindings<'_>,
) -> Result<Vec<u16>, Qwen36MtpLayerQualificationError> {
    let qkv = bindings.materialize_qkv()?;
    let mut samples = Vec::with_capacity(17 * 24);
    for bytes in [
        bindings.embedding_norm.bytes(),
        bindings.hidden_norm.bytes(),
        bindings.input_projection.bytes(),
        bindings.input_norm.bytes(),
        &qkv.weight_bf16,
        bindings.query_norm.bytes(),
        bindings.key_norm.bytes(),
        bindings.attention_output_weight.bytes(),
        bindings.post_attention_norm.bytes(),
        bindings.router_weight.bytes(),
        bindings.routed_gate_up_weight.bytes(),
        bindings.routed_down_weight.bytes(),
        bindings.shared_gate_weight.bytes(),
        bindings.shared_up_weight.bytes(),
        bindings.shared_down_weight.bytes(),
        bindings.shared_expert_gate_weight.bytes(),
        bindings.final_norm.bytes(),
    ] {
        samples.extend(sample_bf16_bytes(bytes)?);
    }

    Ok(samples)
}

fn sample_bf16_bytes(bytes: &[u8]) -> Result<Vec<u16>, Qwen36MtpLayerQualificationError> {
    if !bytes.len().is_multiple_of(2) || bytes.len() < 16 {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "immutable BF16 source has invalid byte length {}",
            bytes.len()
        )));
    }
    let words = bytes.len() / 2;
    let mut samples = Vec::with_capacity(24);
    for start in [0, words / 2 - 4, words - 8] {
        for index in start..start + 8 {
            samples.push(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]));
        }
    }

    Ok(samples)
}

fn run_leaf_oracles(root: &Path) -> Result<(), Qwen36MtpLayerQualificationError> {
    macro_rules! oracle {
        ($name:literal, $call:expr) => {
            $call.map_err(|error| {
                Qwen36MtpLayerQualificationError::Mismatch(format!(
                    "independent {} oracle failed: {error}",
                    $name
                ))
            })?;
        };
    }
    oracle!("input fusion", qualify_qwen36_mtp_bf16_fusion(root));
    oracle!("QKV", qualify_qwen36_mtp_bf16_qkv(root));
    oracle!("Q/K preparation", qualify_qwen36_fp8_attention_qk_prepare());
    oracle!("paged GQA", qualify_qwen36_fp8_paged_gqa());
    oracle!(
        "attention output",
        qualify_qwen36_mtp_bf16_attention_output(root)
    );
    oracle!("router", qualify_qwen36_moe_router());
    oracle!("experts", qualify_qwen36_mtp_bf16_moe());
    Ok(())
}

fn verify_accounting(
    program: &Qwen36MtpLayerProgram,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    if program.resident_weight_bytes() != 1_689_281_536
        || program.cache_bytes() != 1_572_864
        || program.workspace_bytes() != 8_402_800
        || program.owner_bytes() != 1_699_257_200
        || program.arena_bytes() != 1_699_257_856
        || program.graph_count() != 19
    {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(
            "Qwen3.6 MTP owner byte or graph accounting differs from the admitted layout"
                .to_string(),
        ));
    }
    Ok(())
}

fn fixture() -> Fixture {
    let hidden = Qwen36Moe35B::HIDDEN;
    let embedding = (0..PROMPT_ROUTES[2] * hidden)
        .map(|index| {
            let row = index / hidden;
            f32_to_bf16(INPUT_PATTERN[(index + row) & 15] * (1.0 - row as f32 * 0.03125))
        })
        .collect();
    let target_hidden = (0..PROMPT_ROUTES[2] * hidden)
        .map(|index| {
            let row = index / hidden;
            f32_to_bf16(INPUT_PATTERN[(index * 5 + row * 3) & 15] * (0.75 + row as f32 * 0.015625))
        })
        .collect();
    let (rope_cos, rope_sin) = rope(&CACHE_POSITIONS);
    let cache_values =
        PHYSICAL_PAGES * Qwen36Moe35B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen36Moe35B::HEAD_DIM;
    Fixture {
        embedding,
        target_hidden,
        rope_cos,
        rope_sin,
        key_pages: vec![0; cache_values],
        value_pages: vec![0; cache_values],
    }
}

fn rope(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    rope_tables(positions, ROTARY_PAIRS, ROTARY_DIM, 10_000_000.0)
}

fn prepare_draft(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    batch: usize,
    fixture: &Fixture,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let values = batch * Qwen36Moe35B::HIDDEN;
    program.load_inputs(
        stream,
        batch,
        &fixture.embedding[..values],
        &fixture.target_hidden[..values],
    )?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    program.load_draft_state(
        stream,
        batch,
        &CACHE_POSITIONS[..batch],
        &fixture.rope_cos[..batch * ROTARY_PAIRS],
        &fixture.rope_sin[..batch * ROTARY_PAIRS],
    )?;
    program.qualification_reset_outputs(stream, 0xa5)?;
    Ok(())
}

fn prepare_realign(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let positions = (0..tokens as u32).collect::<Vec<_>>();
    let (cosine, sine) = rope(&positions);
    let values = tokens * Qwen36Moe35B::HIDDEN;
    program.load_inputs(
        stream,
        tokens,
        &fixture.embedding[..values],
        &fixture.target_hidden[..values],
    )?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    program.load_realign_state(stream, tokens, 0, &positions, &cosine, &sine)?;
    program.qualification_reset_outputs(stream, 0xa5)?;
    Ok(())
}

fn prepare_prompt(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    rows: usize,
    fixture: &Fixture,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let positions = (0..rows as u32).collect::<Vec<_>>();
    let (cosine, sine) = rope(&positions);
    let values = rows * Qwen36Moe35B::HIDDEN;
    program.load_inputs(
        stream,
        rows,
        &fixture.embedding[..values],
        &fixture.target_hidden[..values],
    )?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    program.load_prompt_state(stream, rows, 0, 0, &cosine, &sine)?;
    program.qualification_reset_outputs(stream, 0xa5)?;

    Ok(())
}

fn b1_route_references(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<Qwen36MtpLayerObservables, Qwen36MtpLayerQualificationError> {
    prepare_draft(program, stream, MAX_BATCH, fixture)?;
    for row in 0..MAX_BATCH {
        program.launch_eager_draft_row(stream, row)?;
    }
    Ok(program.qualification_observables(stream)?)
}

fn verify_metadata(
    batch: usize,
    fixture: &Fixture,
    observed: &Qwen36MtpLayerObservables,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let values = batch * Qwen36Moe35B::HIDDEN;
    compare_exact(
        "embedding input",
        &observed.embedding[..values],
        &fixture.embedding[..values],
    )?;
    compare_exact(
        "target hidden input",
        &observed.target_hidden[..values],
        &fixture.target_hidden[..values],
    )?;
    compare_f32_bits(
        "rotary cosine",
        &observed.rope_cos[..batch * ROTARY_PAIRS],
        &fixture.rope_cos[..batch * ROTARY_PAIRS],
    )?;
    compare_f32_bits(
        "rotary sine",
        &observed.rope_sin[..batch * ROTARY_PAIRS],
        &fixture.rope_sin[..batch * ROTARY_PAIRS],
    )?;
    compare_exact(
        "block tables",
        &observed.block_tables,
        &(0..PHYSICAL_PAGES as u32).collect::<Vec<_>>(),
    )?;
    compare_exact(
        "table rows",
        &observed.table_rows[..batch],
        &(0..batch as u32).collect::<Vec<_>>(),
    )?;
    compare_exact(
        "cache positions",
        &observed.cache_positions[..batch],
        &CACHE_POSITIONS[..batch],
    )?;
    compare_exact(
        "causal lengths",
        &observed.lengths[..batch],
        &CACHE_POSITIONS[..batch]
            .iter()
            .map(|position| position + 1)
            .collect::<Vec<_>>(),
    )
}

fn verify_boundaries(
    batch: usize,
    fixture: &Fixture,
    norms: &Norms,
    observed: &Qwen36MtpLayerObservables,
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let hidden = Qwen36Moe35B::HIDDEN;
    for row in 0..batch {
        let begin = row * hidden;
        let end = begin + hidden;
        let embedding =
            rms_norm_oracle::<Qwen36Moe35B>(&fixture.embedding[begin..end], &norms.embedding);
        compare_bf16(
            "embedding RMSNorm",
            &observed.normalized_embedding[begin..end],
            &embedding,
            report,
        )?;
        let target =
            rms_norm_oracle::<Qwen36Moe35B>(&fixture.target_hidden[begin..end], &norms.hidden);
        compare_bf16(
            "hidden RMSNorm",
            &observed.normalized_hidden[begin..end],
            &target,
            report,
        )?;
        let attention =
            rms_norm_oracle::<Qwen36Moe35B>(&observed.residual[begin..end], &norms.input);
        compare_bf16(
            "attention input RMSNorm",
            &observed.attention_normalized[begin..end],
            &attention,
            report,
        )?;
        let post_attention = residual_oracle(
            &observed.residual[begin..end],
            &observed.attention_branch[begin..end],
        );
        compare_exact(
            "attention residual",
            &observed.post_attention_residual[begin..end],
            &post_attention,
        )?;
        let moe = rms_norm_oracle::<Qwen36Moe35B>(&post_attention, &norms.post_attention);
        compare_bf16(
            "post-attention RMSNorm",
            &observed.moe_normalized[begin..end],
            &moe,
            report,
        )?;
        let residual = residual_oracle(
            &observed.post_attention_residual[begin..end],
            &observed.moe_branch[begin..end],
        );
        compare_exact(
            "final residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        let final_norm = rms_norm_oracle::<Qwen36Moe35B>(&residual, &norms.final_norm);
        compare_bf16(
            "final MTP RMSNorm",
            &observed.final_normalized[begin..end],
            &final_norm,
            report,
        )?;
    }
    report.boundary_values += batch * hidden * 7;
    Ok(())
}

fn compare_active(
    batch: usize,
    reference: &Qwen36MtpLayerObservables,
    observed: &Qwen36MtpLayerObservables,
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    macro_rules! same {
        ($field:ident, $width:expr) => {{
            let values = batch * $width;
            compare_exact(
                concat!("exact B=1 reference ", stringify!($field)),
                &observed.$field[..values],
                &reference.$field[..values],
            )?;
            report.route_reference_values += values;
        }};
    }
    macro_rules! same_f32 {
        ($field:ident, $width:expr) => {{
            let values = batch * $width;
            compare_f32_bits(
                concat!("exact B=1 reference ", stringify!($field)),
                &observed.$field[..values],
                &reference.$field[..values],
            )?;
            report.route_reference_values += values;
        }};
    }
    same!(normalized_embedding, Qwen36Moe35B::HIDDEN);
    same!(normalized_hidden, Qwen36Moe35B::HIDDEN);
    same!(residual, Qwen36Moe35B::HIDDEN);
    same!(attention_normalized, Qwen36Moe35B::HIDDEN);
    same!(qkv, Qwen36Moe35B::ATTENTION_QKV_ROWS);
    same_f32!(query, Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS);
    same_f32!(attention, Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS);
    same!(attention_activation, Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS);
    same!(attention_branch, Qwen36Moe35B::HIDDEN);
    same!(post_attention_residual, Qwen36Moe35B::HIDDEN);
    same!(moe_normalized, Qwen36Moe35B::HIDDEN);
    same!(router_logits, Qwen36Moe35B::NUM_EXPERTS);
    same!(expert_indices, Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN);
    same!(routing_weights, Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN);
    same!(
        expert_intermediate,
        (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::INTERMEDIATE
    );
    same!(
        expert_output,
        (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::HIDDEN
    );
    same!(shared_gate_output, 1);
    same!(moe_branch, Qwen36Moe35B::HIDDEN);
    same!(residual_output, Qwen36Moe35B::HIDDEN);
    same!(final_normalized, Qwen36Moe35B::HIDDEN);
    let cache_values = batch
        * TABLE_STRIDE
        * Qwen36Moe35B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen36Moe35B::HEAD_DIM;
    compare_exact(
        "exact B=1 reference key cache",
        &observed.key_pages[..cache_values],
        &reference.key_pages[..cache_values],
    )?;
    compare_exact(
        "exact B=1 reference value cache",
        &observed.value_pages[..cache_values],
        &reference.value_pages[..cache_values],
    )?;
    report.route_reference_values += 2 * cache_values;
    Ok(())
}

fn verify_prime_route(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
    base: u64,
    addresses: &[usize],
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    prepare_realign(program, stream, tokens, fixture)?;
    program.launch_eager_prime(stream, tokens)?;
    let eager = program.qualification_observables(stream)?;
    report.prime_sentinel_values += verify_prime_sentinels(tokens, &eager)?;

    prepare_realign(program, stream, tokens, fixture)?;
    program.replay_prime(stream, tokens)?;
    let replay = program.qualification_observables(stream)?;
    report.prime_sentinel_values += verify_prime_sentinels(tokens, &replay)?;
    verify_replay(&eager, &replay, report)?;
    verify_stable(program, base, addresses, "prime", tokens)
}

fn verify_realign_route(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
    base: u64,
    addresses: &[usize],
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    prepare_realign(program, stream, tokens, fixture)?;
    for row in 0..tokens {
        program.launch_eager_draft_row(stream, row)?;
    }
    let sequential = program.qualification_observables(stream)?;

    prepare_realign(program, stream, tokens, fixture)?;
    program.launch_eager_realign(stream, tokens)?;
    let eager = program.qualification_observables(stream)?;
    verify_realign_final(tokens, &sequential, &eager, report)?;

    prepare_realign(program, stream, tokens, fixture)?;
    program.replay_realign(stream, tokens)?;
    let replay = program.qualification_observables(stream)?;
    verify_realign_final(tokens, &sequential, &replay, report)?;
    verify_replay(&eager, &replay, report)?;
    verify_stable(program, base, addresses, "realign", tokens)
}

fn verify_prompt_route(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
    rows: usize,
    fixture: &Fixture,
    base: u64,
    addresses: &[usize],
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    prepare_prompt(program, stream, rows, fixture)?;
    program.launch_eager_prompt_prime(stream, rows)?;
    let eager = program.qualification_observables(stream)?;
    report.prime_sentinel_values += verify_prime_sentinels(rows, &eager)?;

    prepare_prompt(program, stream, rows, fixture)?;
    program.replay_prompt_prime(stream, rows)?;
    let replay = program.qualification_observables(stream)?;
    report.prime_sentinel_values += verify_prime_sentinels(rows, &replay)?;
    verify_replay(&eager, &replay, report)?;
    verify_stable(program, base, addresses, "prompt prime", rows)
}

fn verify_realign_final(
    tokens: usize,
    sequential: &Qwen36MtpLayerObservables,
    observed: &Qwen36MtpLayerObservables,
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    let row = tokens - 1;
    let begin = row * Qwen36Moe35B::HIDDEN;
    compare_exact(
        "realignment final normalized",
        &observed.final_normalized[begin..begin + Qwen36Moe35B::HIDDEN],
        &sequential.final_normalized[begin..begin + Qwen36Moe35B::HIDDEN],
    )?;
    compare_exact(
        "realignment key cache",
        &observed.key_pages,
        &sequential.key_pages,
    )?;
    compare_exact(
        "realignment value cache",
        &observed.value_pages,
        &sequential.value_pages,
    )?;
    report.realignment_values +=
        Qwen36Moe35B::HIDDEN + observed.key_pages.len() + observed.value_pages.len();
    Ok(())
}

fn verify_prime_sentinels(
    tokens: usize,
    observed: &Qwen36MtpLayerObservables,
) -> Result<usize, Qwen36MtpLayerQualificationError> {
    macro_rules! u16_sentinel {
        ($field:ident) => {
            if observed.$field.iter().any(|&value| value != BF16_SENTINEL) {
                return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
                    "prime K={tokens} crossed `{}`",
                    stringify!($field)
                )));
            }
        };
    }
    u16_sentinel!(attention_activation);
    u16_sentinel!(attention_branch);
    u16_sentinel!(post_attention_residual);
    u16_sentinel!(moe_normalized);
    u16_sentinel!(router_logits);
    u16_sentinel!(expert_indices);
    u16_sentinel!(routing_weights);
    u16_sentinel!(expert_intermediate);
    u16_sentinel!(expert_output);
    u16_sentinel!(shared_gate_output);
    u16_sentinel!(moe_branch);
    u16_sentinel!(residual_output);
    u16_sentinel!(final_normalized);
    if observed
        .attention
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "prime K={tokens} crossed the attention seam"
        )));
    }
    Ok(observed.attention.len()
        + observed.attention_activation.len()
        + observed.attention_branch.len()
        + observed.post_attention_residual.len()
        + observed.moe_normalized.len()
        + observed.router_logits.len()
        + observed.expert_indices.len()
        + observed.routing_weights.len()
        + observed.expert_intermediate.len()
        + observed.expert_output.len()
        + observed.shared_gate_output.len()
        + observed.moe_branch.len()
        + observed.residual_output.len()
        + observed.final_normalized.len())
}

fn verify_replay(
    eager: &Qwen36MtpLayerObservables,
    replay: &Qwen36MtpLayerObservables,
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            compare_exact(
                concat!("graph replay ", stringify!($field)),
                &replay.$field,
                &eager.$field,
            )?;
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            compare_f32_bits(
                concat!("graph replay ", stringify!($field)),
                &replay.$field,
                &eager.$field,
            )?;
        };
    }
    same!(embedding);
    same!(target_hidden);
    same!(normalized_embedding);
    same!(normalized_hidden);
    same!(residual);
    same!(attention_normalized);
    same!(qkv);
    same_f32!(rope_cos);
    same_f32!(rope_sin);
    same!(block_tables);
    same!(table_rows);
    same!(cache_positions);
    same!(lengths);
    same_f32!(query);
    same!(key_pages);
    same!(value_pages);
    same_f32!(attention);
    same!(attention_activation);
    same!(attention_branch);
    same!(post_attention_residual);
    same!(moe_normalized);
    same!(router_logits);
    same!(expert_indices);
    same!(routing_weights);
    same!(expert_intermediate);
    same!(expert_output);
    same!(shared_gate_output);
    same!(moe_branch);
    same!(residual_output);
    same!(final_normalized);
    report.graph_replay_values += observable_values();
    Ok(())
}

fn observable_values() -> usize {
    128 * (6 * Qwen36Moe35B::HIDDEN
        + Qwen36Moe35B::ATTENTION_QKV_ROWS
        + Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS
        + 2 * ROTARY_PAIRS
        + 3)
        + MAX_BATCH
            * (6 * Qwen36Moe35B::HIDDEN
                + 2 * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS
                + Qwen36Moe35B::NUM_EXPERTS
                + 2 * Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN
                + (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1)
                    * (Qwen36Moe35B::INTERMEDIATE + Qwen36Moe35B::HIDDEN)
                + 1)
        + 2 * PHYSICAL_PAGES
            * Qwen36Moe35B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen36Moe35B::HEAD_DIM
        + PHYSICAL_PAGES
}

fn verify_inactive(
    batch: usize,
    observed: &Qwen36MtpLayerObservables,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    macro_rules! tail {
        ($field:ident, $width:expr) => {
            if observed.$field[batch * $width..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
                    "B={batch} modified inactive `{}`",
                    stringify!($field)
                )));
            }
        };
    }
    tail!(normalized_embedding, Qwen36Moe35B::HIDDEN);
    tail!(normalized_hidden, Qwen36Moe35B::HIDDEN);
    tail!(residual, Qwen36Moe35B::HIDDEN);
    tail!(attention_normalized, Qwen36Moe35B::HIDDEN);
    tail!(qkv, Qwen36Moe35B::ATTENTION_QKV_ROWS);
    tail!(attention_activation, Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS);
    tail!(attention_branch, Qwen36Moe35B::HIDDEN);
    tail!(post_attention_residual, Qwen36Moe35B::HIDDEN);
    tail!(moe_normalized, Qwen36Moe35B::HIDDEN);
    tail!(router_logits, Qwen36Moe35B::NUM_EXPERTS);
    tail!(expert_indices, Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN);
    tail!(routing_weights, Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN);
    tail!(
        expert_intermediate,
        (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::INTERMEDIATE
    );
    tail!(
        expert_output,
        (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::HIDDEN
    );
    tail!(shared_gate_output, 1);
    tail!(moe_branch, Qwen36Moe35B::HIDDEN);
    tail!(residual_output, Qwen36Moe35B::HIDDEN);
    tail!(final_normalized, Qwen36Moe35B::HIDDEN);
    for (name, values, width) in [
        (
            "query",
            &observed.query,
            Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS,
        ),
        (
            "attention",
            &observed.attention,
            Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS,
        ),
    ] {
        if values[batch * width..]
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        {
            return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
                "B={batch} modified inactive `{name}`"
            )));
        }
    }
    let active_cache_values = batch
        * TABLE_STRIDE
        * Qwen36Moe35B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen36Moe35B::HEAD_DIM;
    if observed.key_pages[active_cache_values..]
        .iter()
        .chain(&observed.value_pages[active_cache_values..])
        .any(|&value| value != 0)
    {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "B={batch} modified an inactive cache page"
        )));
    }
    Ok(())
}

fn verify_stable(
    program: &Qwen36MtpLayerProgram,
    base: u64,
    addresses: &[usize],
    route: &str,
    width: usize,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    if program.base_address() != base || program.qualification_addresses()? != addresses {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "owner addresses changed after {route} width {width}"
        )));
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    program: &Qwen36MtpLayerProgram,
    stream: &CudaStream,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    for batch in 1..=MAX_BATCH {
        program.replay_draft(stream, batch)?;
    }
    for tokens in 1..=REALIGN_ROUTES {
        program.replay_prime(stream, tokens)?;
        program.replay_realign(stream, tokens)?;
    }
    for rows in PROMPT_ROUTES {
        program.replay_prompt_prime(stream, rows)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            program.replay_draft(stream, batch)?;
        }
        for tokens in [4, 1, 3, 2] {
            program.replay_prime(stream, tokens)?;
            program.replay_realign(stream, tokens)?;
        }
        for rows in [128, 32, 64] {
            program.replay_prompt_prime(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    report: &mut Qwen36MtpLayerQualification,
) -> Result<(), Qwen36MtpLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = bf16_to_f32(actual);
        let expected = bf16_to_f32(expected);
        let error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = 0.015_625f32.max(expected.abs() * 0.005);
        if !actual.is_finite() || error > tolerance {
            return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
                "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), Qwen36MtpLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "{role} length is {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), Qwen36MtpLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "{role} length is {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        return Err(Qwen36MtpLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, PROMPT_ROUTES, REALIGN_ROUTES, observable_values, qualify_qwen36_mtp_layer,
    };
    use std::path::PathBuf;

    #[test]
    fn qwen36_mtp_layer_suite_route_and_byte_inventory_is_exact() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(REALIGN_ROUTES, 4);
        assert_eq!(PROMPT_ROUTES, [32, 64, 128]);
        assert_eq!(observable_values(), 5_208_608);
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and TUISKO_QWEN36_SNAPSHOT"]
    fn qwen36_mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes()
    -> Result<(), super::Qwen36MtpLayerQualificationError> {
        let root = PathBuf::from(
            std::env::var("TUISKO_QWEN36_SNAPSHOT").expect("TUISKO_QWEN36_SNAPSHOT is required"),
        );
        let report = qualify_qwen36_mtp_layer(&root)?;
        assert_eq!(report.leaf_oracle_suites, 7);
        assert_eq!(report.resident_weight_bytes, 1_689_281_536);
        assert_eq!(report.cache_bytes, 1_572_864);
        assert_eq!(report.workspace_bytes, 8_402_800);
        assert_eq!(report.owner_bytes, 1_699_257_200);
        assert_eq!(report.arena_bytes, 1_699_257_856);
        assert_eq!(report.padding_bytes, 656);
        assert_eq!(report.graph_count, 19);
        assert_eq!(report.immutable_source_values, 816);
        assert_eq!(report.boundary_values, 516_096);
        assert_eq!(report.route_reference_values, 9_428_580);
        assert_eq!(report.graph_replay_values, 98_963_552);
        assert_eq!(report.prime_sentinel_values, 4_904_816);
        assert_eq!(report.realignment_values, 12_599_296);
        assert!(report.maximum_absolute_error <= 0.03125);
        Ok(())
    }
}
