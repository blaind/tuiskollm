//! Source-backed qualification for the complete Qwen3.8 MTP draft owner.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{
    DeviceBenchmarkError, qualify_fp8_lm_head, qualify_mtp_bf16_attention_output,
    qualify_mtp_bf16_fusion, qualify_mtp_bf16_mlp, qualify_mtp_bf16_paged_gqa,
    qualify_mtp_bf16_qk_prepare, qualify_mtp_bf16_qkv, qualify_text_endpoint,
};
use std::path::Path;
use tuisko_engine::{EngineError, MAX_BATCH, MtpLayerObservables, MtpLayerProgram};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B};

const REALIGN_ROUTES: usize = 4;
const TABLE_STRIDE: usize = 3;
const PHYSICAL_PAGES: usize = MAX_BATCH * TABLE_STRIDE;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const CACHE_POSITIONS: [u32; MAX_BATCH] = [0, 1, 63, 64, 65, 97, 128, 130];
const INPUT_PATTERN: [f32; 16] = [
    0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.1875, -0.1875, 0.09375,
    -0.09375, 0.046875, -0.046875, 0.015625, -0.015625,
];

/// Failure of the complete source-backed MTP layer gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpLayerQualificationError {
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
    #[error("MTP layer qualification failed: {0}")]
    Mismatch(String),
}

/// Checked graph, seam, oracle, and byte counts for one complete MTP owner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpLayerQualification {
    /// Independent source or mathematical leaf suites completed before composition.
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
    /// Exact unchanged BF16 MTP source bytes.
    pub mtp_weight_bytes: usize,
    /// One source-native FP8 shared LM-head payload.
    pub shared_endpoint_weight_bytes: usize,
    /// Exact represented BF16 cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable non-cache workspace bytes.
    pub workspace_bytes: usize,
    /// Complete owner bytes without padding.
    pub owner_bytes: usize,
    /// Complete single-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Number of immutable draft, prime, and realignment graphs.
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
    key_pages: Vec<u16>,
    value_pages: Vec<u16>,
}

/// Qualifies draft `B=1..=8`, prime `K=1..=4`, and causal realignment.
pub fn qualify_mtp_layer(root: &Path) -> Result<MtpLayerQualification, MtpLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    run_leaf_oracles(root)?;

    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let norms = Norms {
        embedding: bindings.embedding_norm.words().collect(),
        hidden: bindings.hidden_norm.words().collect(),
        input: bindings.input_norm.words().collect(),
        post_attention: bindings.post_attention_norm.words().collect(),
        final_norm: bindings.final_norm.words().collect(),
    };
    let fixture = fixture();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = MtpLayerProgram::from_snapshot(&context, &snapshot)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 42 {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "MTP owner exposes {} addresses, expected 42",
            stable_addresses.len()
        )));
    }
    verify_accounting(&program)?;
    let mut report = MtpLayerQualification {
        leaf_oracle_suites: 8,
        boundary_values: 0,
        route_reference_values: 0,
        graph_replay_values: 0,
        prime_sentinel_values: 0,
        realignment_values: 0,
        mtp_weight_bytes: program.mtp_weight_bytes(),
        shared_endpoint_weight_bytes: program.shared_endpoint_weight_bytes(),
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

    verify_no_post_warmup_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn run_leaf_oracles(root: &Path) -> Result<(), MtpLayerQualificationError> {
    macro_rules! oracle {
        ($name:literal, $call:expr) => {
            $call.map_err(|error| {
                MtpLayerQualificationError::Mismatch(format!(
                    "independent {} oracle failed: {error}",
                    $name
                ))
            })?;
        };
    }
    oracle!("input fusion", qualify_mtp_bf16_fusion(root));
    oracle!("QKV", qualify_mtp_bf16_qkv(root));
    oracle!("Q/K preparation", qualify_mtp_bf16_qk_prepare(root));
    oracle!("paged GQA", qualify_mtp_bf16_paged_gqa());
    oracle!("attention output", qualify_mtp_bf16_attention_output(root));
    oracle!("MLP", qualify_mtp_bf16_mlp(root));
    oracle!("LM head", qualify_fp8_lm_head());
    oracle!("source endpoint", qualify_text_endpoint(root));
    Ok(())
}

fn verify_accounting(program: &MtpLayerProgram) -> Result<(), MtpLayerQualificationError> {
    if program.mtp_weight_bytes() != 849_398_784
        || program.shared_endpoint_weight_bytes() != 1_271_895_040
        || program.resident_weight_bytes() != 2_121_293_824
        || program.cache_bytes() != 6_291_456
        || program.workspace_bytes() != 5_998_816
        || program.owner_bytes() != 2_133_584_096
        || program.arena_bytes() != 2_133_585_152
        || program.graph_count() != 16
    {
        return Err(MtpLayerQualificationError::Mismatch(
            "MTP owner byte or graph accounting differs from the admitted layout".to_string(),
        ));
    }
    Ok(())
}

fn fixture() -> Fixture {
    let hidden = Qwen38_27B::HIDDEN;
    let embedding = (0..MAX_BATCH * hidden)
        .map(|index| {
            let row = index / hidden;
            f32_to_bf16(INPUT_PATTERN[(index + row) & 15] * (1.0 - row as f32 * 0.03125))
        })
        .collect();
    let target_hidden = (0..MAX_BATCH * hidden)
        .map(|index| {
            let row = index / hidden;
            f32_to_bf16(INPUT_PATTERN[(index * 5 + row * 3) & 15] * (0.75 + row as f32 * 0.015625))
        })
        .collect();
    let (rope_cos, rope_sin) = rope(&CACHE_POSITIONS);
    let cache_values =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
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
    let mut cosine = vec![0.0; positions.len() * ROTARY_PAIRS];
    let mut sine = vec![0.0; positions.len() * ROTARY_PAIRS];
    for (row, &position) in positions.iter().enumerate() {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
            let angle = f64::from(position) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[row * ROTARY_PAIRS + pair] = cos as f32;
            sine[row * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn prepare_draft(
    program: &MtpLayerProgram,
    stream: &CudaStream,
    batch: usize,
    fixture: &Fixture,
) -> Result<(), MtpLayerQualificationError> {
    let values = batch * Qwen38_27B::HIDDEN;
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
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    Ok(())
}

fn prepare_realign(
    program: &MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
) -> Result<(), MtpLayerQualificationError> {
    let positions = (0..tokens as u32).collect::<Vec<_>>();
    let (cosine, sine) = rope(&positions);
    let values = tokens * Qwen38_27B::HIDDEN;
    program.load_inputs(
        stream,
        tokens,
        &fixture.embedding[..values],
        &fixture.target_hidden[..values],
    )?;
    program.load_cache(stream, &fixture.key_pages, &fixture.value_pages)?;
    program.load_realign_state(stream, tokens, 0, &positions, &cosine, &sine)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    Ok(())
}

fn b1_route_references(
    program: &MtpLayerProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<MtpLayerObservables, MtpLayerQualificationError> {
    prepare_draft(program, stream, MAX_BATCH, fixture)?;
    for row in 0..MAX_BATCH {
        program.launch_eager_draft_row(stream, row)?;
    }
    Ok(program.qualification_observables(stream)?)
}

fn verify_metadata(
    batch: usize,
    fixture: &Fixture,
    observed: &MtpLayerObservables,
) -> Result<(), MtpLayerQualificationError> {
    let values = batch * Qwen38_27B::HIDDEN;
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
    observed: &MtpLayerObservables,
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for row in 0..batch {
        let begin = row * hidden;
        let end = begin + hidden;
        let embedding =
            rms_norm_oracle::<Qwen38_27B>(&fixture.embedding[begin..end], &norms.embedding);
        compare_bf16(
            "embedding RMSNorm",
            &observed.normalized_embedding[begin..end],
            &embedding,
            report,
        )?;
        let target =
            rms_norm_oracle::<Qwen38_27B>(&fixture.target_hidden[begin..end], &norms.hidden);
        compare_bf16(
            "hidden RMSNorm",
            &observed.normalized_hidden[begin..end],
            &target,
            report,
        )?;
        let attention = rms_norm_oracle::<Qwen38_27B>(&observed.residual[begin..end], &norms.input);
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
        let mlp = rms_norm_oracle::<Qwen38_27B>(&post_attention, &norms.post_attention);
        compare_bf16(
            "post-attention RMSNorm",
            &observed.mlp_normalized[begin..end],
            &mlp,
            report,
        )?;
        let residual = residual_oracle(
            &observed.post_attention_residual[begin..end],
            &observed.mlp_branch[begin..end],
        );
        compare_exact(
            "final residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        let final_norm = rms_norm_oracle::<Qwen38_27B>(&residual, &norms.final_norm);
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
    reference: &MtpLayerObservables,
    observed: &MtpLayerObservables,
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
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
    same!(normalized_embedding, Qwen38_27B::HIDDEN);
    same!(normalized_hidden, Qwen38_27B::HIDDEN);
    same!(residual, Qwen38_27B::HIDDEN);
    same!(attention_normalized, Qwen38_27B::HIDDEN);
    same!(qkv, Qwen38_27B::ATTENTION_QKV_ROWS);
    same_f32!(query, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    same_f32!(attention, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    same!(attention_activation, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    same!(attention_branch, Qwen38_27B::HIDDEN);
    same!(post_attention_residual, Qwen38_27B::HIDDEN);
    same!(mlp_normalized, Qwen38_27B::HIDDEN);
    same!(swiglu, Qwen38_27B::INTERMEDIATE);
    same!(mlp_branch, Qwen38_27B::HIDDEN);
    same!(residual_output, Qwen38_27B::HIDDEN);
    same!(final_normalized, Qwen38_27B::HIDDEN);
    same!(lm_head_activation_codes, Qwen38_27B::HIDDEN);
    same_f32!(lm_head_activation_scales, 1);
    same!(logits, Qwen38_27B::VOCAB);
    let cache_values = batch
        * TABLE_STRIDE
        * Qwen38_27B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38_27B::HEAD_DIM;
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
    program: &MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
    base: u64,
    addresses: &[usize],
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
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
    program: &MtpLayerProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
    base: u64,
    addresses: &[usize],
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
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

fn verify_realign_final(
    tokens: usize,
    sequential: &MtpLayerObservables,
    observed: &MtpLayerObservables,
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
    let row = tokens - 1;
    macro_rules! last {
        ($field:ident, $width:expr) => {{
            let begin = row * $width;
            compare_exact(
                concat!("realignment final ", stringify!($field)),
                &observed.$field[begin..begin + $width],
                &sequential.$field[begin..begin + $width],
            )?;
            report.realignment_values += $width;
        }};
    }
    macro_rules! last_f32 {
        ($field:ident, $width:expr) => {{
            let begin = row * $width;
            compare_f32_bits(
                concat!("realignment final ", stringify!($field)),
                &observed.$field[begin..begin + $width],
                &sequential.$field[begin..begin + $width],
            )?;
            report.realignment_values += $width;
        }};
    }
    last!(final_normalized, Qwen38_27B::HIDDEN);
    last!(lm_head_activation_codes, Qwen38_27B::HIDDEN);
    last_f32!(lm_head_activation_scales, 1);
    last!(logits, Qwen38_27B::VOCAB);
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
    report.realignment_values += observed.key_pages.len() + observed.value_pages.len();
    Ok(())
}

fn verify_prime_sentinels(
    tokens: usize,
    observed: &MtpLayerObservables,
) -> Result<usize, MtpLayerQualificationError> {
    macro_rules! u16_sentinel {
        ($field:ident) => {
            if observed.$field.iter().any(|&value| value != BF16_SENTINEL) {
                return Err(MtpLayerQualificationError::Mismatch(format!(
                    "prime K={tokens} crossed `{}`",
                    stringify!($field)
                )));
            }
        };
    }
    u16_sentinel!(attention_activation);
    u16_sentinel!(attention_branch);
    u16_sentinel!(post_attention_residual);
    u16_sentinel!(mlp_normalized);
    u16_sentinel!(swiglu);
    u16_sentinel!(mlp_branch);
    u16_sentinel!(residual_output);
    u16_sentinel!(final_normalized);
    u16_sentinel!(logits);
    if observed
        .attention
        .iter()
        .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        || observed
            .lm_head_activation_codes
            .iter()
            .any(|&value| value != BYTE_SENTINEL)
        || observed
            .lm_head_activation_scales
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "prime K={tokens} crossed an attention or LM-head seam"
        )));
    }
    Ok(observed.attention.len()
        + observed.attention_activation.len()
        + observed.attention_branch.len()
        + observed.post_attention_residual.len()
        + observed.mlp_normalized.len()
        + observed.swiglu.len()
        + observed.mlp_branch.len()
        + observed.residual_output.len()
        + observed.final_normalized.len()
        + observed.lm_head_activation_codes.len()
        + observed.lm_head_activation_scales.len()
        + observed.logits.len())
}

fn verify_replay(
    eager: &MtpLayerObservables,
    replay: &MtpLayerObservables,
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
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
    same!(mlp_normalized);
    same!(swiglu);
    same!(mlp_branch);
    same!(residual_output);
    same!(final_normalized);
    same!(lm_head_activation_codes);
    same_f32!(lm_head_activation_scales);
    same!(logits);
    report.graph_replay_values += observable_values();
    Ok(())
}

fn observable_values() -> usize {
    MAX_BATCH
        * (13 * Qwen38_27B::HIDDEN
            + Qwen38_27B::ATTENTION_QKV_ROWS
            + 3 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + Qwen38_27B::INTERMEDIATE
            + Qwen38_27B::VOCAB
            + 2 * ROTARY_PAIRS
            + 4)
        + 2 * PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM
        + PHYSICAL_PAGES
}

fn verify_inactive(
    batch: usize,
    observed: &MtpLayerObservables,
) -> Result<(), MtpLayerQualificationError> {
    macro_rules! tail {
        ($field:ident, $width:expr) => {
            if observed.$field[batch * $width..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(MtpLayerQualificationError::Mismatch(format!(
                    "B={batch} modified inactive `{}`",
                    stringify!($field)
                )));
            }
        };
    }
    tail!(normalized_embedding, Qwen38_27B::HIDDEN);
    tail!(normalized_hidden, Qwen38_27B::HIDDEN);
    tail!(residual, Qwen38_27B::HIDDEN);
    tail!(attention_normalized, Qwen38_27B::HIDDEN);
    tail!(qkv, Qwen38_27B::ATTENTION_QKV_ROWS);
    tail!(attention_activation, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    tail!(attention_branch, Qwen38_27B::HIDDEN);
    tail!(post_attention_residual, Qwen38_27B::HIDDEN);
    tail!(mlp_normalized, Qwen38_27B::HIDDEN);
    tail!(swiglu, Qwen38_27B::INTERMEDIATE);
    tail!(mlp_branch, Qwen38_27B::HIDDEN);
    tail!(residual_output, Qwen38_27B::HIDDEN);
    tail!(final_normalized, Qwen38_27B::HIDDEN);
    tail!(logits, Qwen38_27B::VOCAB);
    for (name, values, width) in [
        (
            "query",
            &observed.query,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ),
        (
            "attention",
            &observed.attention,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ),
    ] {
        if values[batch * width..]
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        {
            return Err(MtpLayerQualificationError::Mismatch(format!(
                "B={batch} modified inactive `{name}`"
            )));
        }
    }
    if observed.lm_head_activation_codes[batch * Qwen38_27B::HIDDEN..]
        .iter()
        .any(|&value| value != BYTE_SENTINEL)
        || observed.lm_head_activation_scales[batch..]
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "B={batch} modified inactive LM-head workspace"
        )));
    }
    let active_cache_values = batch
        * TABLE_STRIDE
        * Qwen38_27B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38_27B::HEAD_DIM;
    if observed.key_pages[active_cache_values..]
        .iter()
        .chain(&observed.value_pages[active_cache_values..])
        .any(|&value| value != 0)
    {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "B={batch} modified an inactive cache page"
        )));
    }
    Ok(())
}

fn verify_stable(
    program: &MtpLayerProgram,
    base: u64,
    addresses: &[usize],
    route: &str,
    width: usize,
) -> Result<(), MtpLayerQualificationError> {
    if program.base_address() != base || program.qualification_addresses()? != addresses {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "owner addresses changed after {route} width {width}"
        )));
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    program: &MtpLayerProgram,
    stream: &CudaStream,
) -> Result<(), MtpLayerQualificationError> {
    for batch in 1..=MAX_BATCH {
        program.replay_draft(stream, batch)?;
    }
    for tokens in 1..=REALIGN_ROUTES {
        program.replay_prime(stream, tokens)?;
        program.replay_realign(stream, tokens)?;
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
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

fn residual_oracle(left: &[u16], right: &[u16]) -> Vec<u16> {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| f32_to_bf16(bf16_to_f32(left) + bf16_to_f32(right)))
        .collect()
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    report: &mut MtpLayerQualification,
) -> Result<(), MtpLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = bf16_to_f32(actual);
        let expected = bf16_to_f32(expected);
        let error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        // Match the independently qualified RMSNorm accumulation envelope;
        // the composed inputs can land one BF16 ULP from the FP64 oracle.
        let tolerance = 0.015_625f32.max(expected.abs() * 0.005);
        if !actual.is_finite() || error > tolerance {
            return Err(MtpLayerQualificationError::Mismatch(format!(
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
) -> Result<(), MtpLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(MtpLayerQualificationError::Mismatch(format!(
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
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), MtpLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "{role} length is {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        return Err(MtpLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, REALIGN_ROUTES, observable_values, qualify_mtp_layer};
    use std::path::PathBuf;

    #[test]
    fn mtp_layer_suite_route_and_byte_inventory_is_exact() {
        assert_eq!(
            (1..=MAX_BATCH).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!((1..=REALIGN_ROUTES).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert!(observable_values() > 0);
    }

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes()
    -> Result<(), super::MtpLayerQualificationError> {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_layer(&root)?;
        assert_eq!(report.leaf_oracle_suites, 8);
        assert!(report.boundary_values > 0);
        assert!(report.route_reference_values > 0);
        assert_eq!(report.graph_replay_values, 16 * observable_values());
        assert!(report.prime_sentinel_values > 0);
        assert!(report.realignment_values > 0);
        assert_eq!(report.mtp_weight_bytes, 849_398_784);
        assert_eq!(report.shared_endpoint_weight_bytes, 1_271_895_040);
        assert_eq!(report.cache_bytes, 6_291_456);
        assert_eq!(report.workspace_bytes, 5_998_816);
        assert_eq!(report.owner_bytes, 2_133_584_096);
        assert_eq!(report.arena_bytes, 2_133_585_152);
        assert_eq!(report.padding_bytes, 1_056);
        assert_eq!(report.graph_count, 16);
        assert!(report.maximum_absolute_error.is_finite());
        Ok(())
    }
}
