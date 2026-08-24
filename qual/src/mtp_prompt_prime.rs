//! Source-backed qualification for target-residual MTP prompt priming.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32};
use crate::residual_norm::rms_norm_oracle;
use crate::{
    DeviceBenchmarkError, qualify_mtp_bf16_fusion, qualify_mtp_bf16_qk_prepare,
    qualify_mtp_bf16_qkv,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, LONG_CONTEXT_PHYSICAL_PAGES, MtpPromptPrimeObservables, MtpPromptPrimeProgram,
    MtpPromptPrimeRoute, ResidentModelProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen38_27B, TextEndpointBindings,
};

const ROUTES: [usize; 5] = [1, 32, 64, 128, 1_024];
const TAIL_ROUTES: usize = 31;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const SLOT: usize = 0;
const FUSION_OUTPUT_SAMPLES: [usize; 5] = [0, 1, 511, 4_095, Qwen38_27B::HIDDEN - 1];
const QKV_OUTPUT_SAMPLES: [usize; 5] = [0, 1, 6_143, 12_287, Qwen38_27B::ATTENTION_QKV_ROWS - 1];

/// Failure of the exact source-backed MTP prompt-prime gate.
#[derive(Debug, thiserror::Error)]
pub enum MtpPromptPrimeQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident owner setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership, launch, or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with an independent route or value oracle.
    #[error("MTP prompt-prime qualification failed: {0}")]
    Mismatch(String),
}

/// Exact route, seam, source-oracle, cache, and ownership counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpPromptPrimeQualification {
    /// Independent fusion, QKV, and Q/K mathematical suites completed first.
    pub leaf_oracle_suites: usize,
    /// Exact K=1 and T=32,64,128,1024 owner routes exercised.
    pub prompt_routes: usize,
    /// Repeated-K1 tail lengths exercised without padding.
    pub tail_routes: usize,
    /// Active normalization values compared with an independent host formula.
    pub normalization_oracle_values: usize,
    /// Sampled fusion and QKV outputs compared with source-matrix formulas.
    pub projection_oracle_values: usize,
    /// Complete non-cache workspace values reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Appended and untouched cache values checked at physical-page boundaries.
    pub cache_values: usize,
    /// Exact unchanged BF16 prompt-prime source bytes.
    pub resident_weight_bytes: usize,
    /// Exact represented BF16 MTP cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable device workspace bytes.
    pub workspace_bytes: usize,
    /// Complete owner bytes without padding.
    pub owner_bytes: usize,
    /// Complete device arena bytes.
    pub arena_bytes: usize,
    /// Exact alignment padding bytes.
    pub padding_bytes: usize,
    /// Page-locked graph source bytes.
    pub host_stager_bytes: usize,
    /// Immutable graph entries retained by the owner.
    pub graph_count: usize,
    /// Largest absolute error at a source-backed mathematical boundary.
    pub maximum_absolute_error: f32,
}

struct Source {
    embedding_norm: Vec<u16>,
    hidden_norm: Vec<u16>,
    input_projection: Vec<u16>,
    input_norm: Vec<u16>,
    qkv_weight: Vec<u16>,
}

struct CachePage {
    physical: usize,
    key: Vec<u16>,
    value: Vec<u16>,
}

/// Qualifies exact prompt tiles, scalar tails, target handoff, and owner stability.
pub fn qualify_mtp_prompt_prime(
    root: &Path,
) -> Result<MtpPromptPrimeQualification, MtpPromptPrimeQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    run_leaf_oracles(root)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let embedding_source = TextEndpointBindings::bind_embedding(snapshot.as_ref())?.bytes();
    let bindings = MtpBindings::bind(snapshot.as_ref())?;
    let qkv = bindings.materialize_qkv()?;
    let source = Source {
        embedding_norm: bindings.embedding_norm.words().collect(),
        hidden_norm: bindings.hidden_norm.words().collect(),
        input_projection: bindings.input_projection.words().collect(),
        input_norm: bindings.input_norm.words().collect(),
        qkv_weight: qkv
            .weight_bf16
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .collect(),
    };
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(MtpPromptPrimeQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut target = ResidentModelProgram::from_snapshot(&context, snapshot.clone())?;
    target.activate_kv_slot(SLOT)?;
    target.reserve_kv_slot_tokens(&stream, SLOT, 1_024)?;
    let mut program = MtpPromptPrimeProgram::from_target(&target)?;
    verify_owner(&program)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 24
        || stable_addresses
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 24
    {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "MTP prompt owner exposes {} addresses, expected 24 unique addresses",
            stable_addresses.len()
        )));
    }
    let mut report = MtpPromptPrimeQualification {
        leaf_oracle_suites: 3,
        prompt_routes: 0,
        tail_routes: 0,
        normalization_oracle_values: 0,
        projection_oracle_values: 0,
        graph_replay_values: 0,
        cache_values: 0,
        resident_weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        owner_bytes: program.owner_bytes(),
        arena_bytes: program.arena_bytes(),
        padding_bytes: program.padding_bytes(),
        host_stager_bytes: program.host_stager_bytes(),
        graph_count: program.graph_count(),
        maximum_absolute_error: 0.0,
    };

    for rows in ROUTES {
        let target_hidden = prepare_target(&target, &stream, embedding_source, rows)?;
        let next_tokens = token_ids(1, rows);
        let positions = positions(0, rows)?;
        let (cosine, sine) = rope(&positions);
        let route = program.stage(&stream, rows, SLOT, 0, &next_tokens, &cosine, &sine)?;
        let target_tables = target.qualification_block_tables(&stream)?;

        program.qualification_reset_outputs(&stream, 0xa5)?;
        program.launch_eager(&stream, route)?;
        let eager = program.qualification_observables(&stream)?;
        let eager_cache = read_route_cache(&target, &program, &stream, route)?;
        verify_inputs(
            embedding_source,
            route,
            &next_tokens,
            &target_hidden,
            &target_tables,
            &cosine,
            &sine,
            &eager,
        )?;
        verify_math(route, &source, &eager, &mut report)?;
        verify_cache_boundaries(route, &eager, &eager_cache, &mut report)?;

        program.qualification_reset_outputs(&stream, 0xa5)?;
        program.replay(&stream, route)?;
        let replay = program.qualification_observables(&stream)?;
        let replay_cache = read_route_cache(&target, &program, &stream, route)?;
        compare_observables(rows, &eager, &replay)?;
        compare_cache(rows, &eager_cache, &replay_cache)?;
        report.graph_replay_values += observable_values(&eager);
        report.cache_values += replay_cache
            .iter()
            .map(|page| page.key.len() + page.value.len())
            .sum::<usize>();
        verify_stable(&program, stable_base, &stable_addresses, rows)?;
        report.prompt_routes += 1;
    }

    verify_tails(
        &target,
        &mut program,
        &stream,
        embedding_source,
        &mut report,
    )?;
    verify_no_post_warmup_allocation(&context, &program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn run_leaf_oracles(root: &Path) -> Result<(), MtpPromptPrimeQualificationError> {
    qualify_mtp_bf16_fusion(root).map_err(|error| {
        MtpPromptPrimeQualificationError::Mismatch(format!(
            "independent prompt fusion oracle failed: {error}"
        ))
    })?;
    qualify_mtp_bf16_qkv(root).map_err(|error| {
        MtpPromptPrimeQualificationError::Mismatch(format!(
            "independent prompt QKV oracle failed: {error}"
        ))
    })?;
    qualify_mtp_bf16_qk_prepare(root).map_err(|error| {
        MtpPromptPrimeQualificationError::Mismatch(format!(
            "independent prompt Q/K oracle failed: {error}"
        ))
    })?;
    Ok(())
}

fn verify_owner(
    program: &MtpPromptPrimeProgram<'_>,
) -> Result<(), MtpPromptPrimeQualificationError> {
    if program.resident_weight_bytes() != 251_689_984
        || program.cache_bytes() != 901_251_072
        || program.workspace_bytes() != 117_820_864
        || program.owner_bytes() != 1_270_761_920
        || program.arena_bytes() != 1_270_761_984
        || program.padding_bytes() != 64
        || program.host_stager_bytes() != 10_756_096
        || program.graph_count() != ROUTES.len()
    {
        return Err(MtpPromptPrimeQualificationError::Mismatch(
            "MTP prompt owner accounting differs from the admitted layout".to_string(),
        ));
    }
    Ok(())
}

fn prepare_target(
    target: &ResidentModelProgram,
    stream: &CudaStream,
    embedding_source: &[u8],
    rows: usize,
) -> Result<Vec<u16>, MtpPromptPrimeQualificationError> {
    target.reset_slot(stream, SLOT)?;
    let tokens = token_ids(0, rows);
    let positions = positions(0, rows)?;
    let (cosine, sine) = rope(&positions);
    let embeddings = embedding_rows(embedding_source, &tokens)?;
    target.load_residual(stream, rows, &embeddings)?;
    if rows == 1 {
        target.load_slot_routes(stream, &[SLOT])?;
        let route = target.load_decode_state(stream, 1, &positions, &cosine, &sine)?;
        target.replay(stream, route)?;
        target.read_residual(stream, 1).map_err(Into::into)
    } else {
        let route = target.load_prefill_state(stream, rows, SLOT, &cosine, &sine)?;
        target.replay_prefill(stream, route)?;
        target
            .qualification_prefill_residual(stream, route)
            .map_err(Into::into)
    }
}

fn verify_inputs(
    embedding_source: &[u8],
    route: MtpPromptPrimeRoute,
    next_tokens: &[u32],
    target_hidden: &[u16],
    target_tables: &[u32],
    cosine: &[f32],
    sine: &[f32],
    observed: &MtpPromptPrimeObservables,
) -> Result<(), MtpPromptPrimeQualificationError> {
    let rows = route.rows();
    let hidden_values = rows * Qwen38_27B::HIDDEN;
    let expected_embedding = embedding_rows(embedding_source, next_tokens)?;
    compare_exact(
        "next-token embeddings",
        &observed.embedding[..hidden_values],
        &expected_embedding,
    )?;
    compare_exact(
        "target residual handoff",
        &observed.target_hidden[..hidden_values],
        target_hidden,
    )?;
    compare_exact(
        "target page-table handoff",
        &observed.block_tables,
        target_tables,
    )?;
    compare_exact(
        "slot rows",
        &observed.table_rows[..rows],
        &vec![SLOT as u32; rows],
    )?;
    compare_exact(
        "cache positions",
        &observed.cache_positions[..rows],
        &positions(route.first_position(), rows)?,
    )?;
    compare_f32_bits(
        "MRoPE cosine",
        &observed.rope_cos[..rows * ROTARY_PAIRS],
        cosine,
    )?;
    compare_f32_bits(
        "MRoPE sine",
        &observed.rope_sin[..rows * ROTARY_PAIRS],
        sine,
    )?;
    for (role, inactive) in [
        ("embedding", &observed.embedding[hidden_values..]),
        ("target hidden", &observed.target_hidden[hidden_values..]),
        (
            "normalized embedding",
            &observed.normalized_embedding[hidden_values..],
        ),
        (
            "normalized hidden",
            &observed.normalized_hidden[hidden_values..],
        ),
        ("fusion residual", &observed.residual[hidden_values..]),
        (
            "attention normalized",
            &observed.attention_normalized[hidden_values..],
        ),
    ] {
        require_bf16_sentinel(role, inactive)?;
    }
    require_bf16_sentinel(
        "QKV",
        &observed.qkv[rows * Qwen38_27B::ATTENTION_QKV_ROWS..],
    )?;
    require_f32_sentinel(
        "query",
        &observed.query[rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS..],
    )?;
    require_f32_sentinel("MRoPE cosine", &observed.rope_cos[rows * ROTARY_PAIRS..])?;
    require_f32_sentinel("MRoPE sine", &observed.rope_sin[rows * ROTARY_PAIRS..])?;
    require_u32_sentinel("slot rows", &observed.table_rows[rows..])?;
    require_u32_sentinel("cache positions", &observed.cache_positions[rows..])?;
    Ok(())
}

fn verify_math(
    route: MtpPromptPrimeRoute,
    source: &Source,
    observed: &MtpPromptPrimeObservables,
    report: &mut MtpPromptPrimeQualification,
) -> Result<(), MtpPromptPrimeQualificationError> {
    let rows = route.rows();
    let hidden = Qwen38_27B::HIDDEN;
    let selected = selected_rows(rows);
    for row in 0..rows {
        let begin = row * hidden;
        let end = begin + hidden;
        for (role, actual, input, weight) in [
            (
                "embedding norm",
                &observed.normalized_embedding[begin..end],
                &observed.embedding[begin..end],
                source.embedding_norm.as_slice(),
            ),
            (
                "target-hidden norm",
                &observed.normalized_hidden[begin..end],
                &observed.target_hidden[begin..end],
                source.hidden_norm.as_slice(),
            ),
            (
                "attention input norm",
                &observed.attention_normalized[begin..end],
                &observed.residual[begin..end],
                source.input_norm.as_slice(),
            ),
        ] {
            let expected = rms_norm_oracle::<Qwen38_27B>(input, weight);
            compare_close_bf16(role, actual, &expected, &mut report.maximum_absolute_error)?;
            report.normalization_oracle_values += hidden;
        }
    }

    for row in selected {
        let hidden_begin = row * hidden;
        let fusion_input = observed.normalized_embedding[hidden_begin..hidden_begin + hidden]
            .iter()
            .chain(&observed.normalized_hidden[hidden_begin..hidden_begin + hidden]);
        for &output in &FUSION_OUTPUT_SAMPLES {
            let weight = &source.input_projection[output * 2 * hidden..(output + 1) * 2 * hidden];
            let expected = fusion_input
                .clone()
                .zip(weight)
                .fold(0.0f64, |sum, (&x, &w)| {
                    sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
                });
            require_projection_close(
                "fusion",
                row,
                output,
                observed.residual[hidden_begin + output],
                expected,
                &mut report.maximum_absolute_error,
            )?;
            report.projection_oracle_values += 1;
        }
        let qkv_begin = row * Qwen38_27B::ATTENTION_QKV_ROWS;
        for &output in &QKV_OUTPUT_SAMPLES {
            let weight = &source.qkv_weight[output * hidden..(output + 1) * hidden];
            let expected = observed.attention_normalized[hidden_begin..hidden_begin + hidden]
                .iter()
                .zip(weight)
                .fold(0.0f64, |sum, (&x, &w)| {
                    sum + f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))
                });
            require_projection_close(
                "QKV",
                row,
                output,
                observed.qkv[qkv_begin + output],
                expected,
                &mut report.maximum_absolute_error,
            )?;
            report.projection_oracle_values += 1;
        }
    }
    Ok(())
}

fn verify_cache_boundaries(
    route: MtpPromptPrimeRoute,
    observed: &MtpPromptPrimeObservables,
    pages: &[CachePage],
    report: &mut MtpPromptPrimeQualification,
) -> Result<(), MtpPromptPrimeQualificationError> {
    let page_values = cache_page_values();
    let mut checked = 0usize;
    for page in pages {
        let mut touched = vec![false; page_values];
        for row in 0..route.rows() {
            let position = route.first_position() + row;
            let physical = observed.block_tables
                [route.slot() * LONG_CONTEXT_PHYSICAL_PAGES + position / ATTENTION_PAGE_SIZE]
                as usize;
            if physical != page.physical {
                continue;
            }
            let qkv_row = row * Qwen38_27B::ATTENTION_QKV_ROWS;
            let value_begin =
                qkv_row + Qwen38_27B::ATTENTION_QUERY_ROWS + Qwen38_27B::ATTENTION_KV_ROWS;
            for head in 0..Qwen38_27B::NUM_KV_HEADS {
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let index = Qwen38_27B::HEAD_DIM
                        * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * head)
                        + dimension;
                    touched[index] = true;
                    let expected =
                        observed.qkv[value_begin + head * Qwen38_27B::HEAD_DIM + dimension];
                    if page.value[index] != expected {
                        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                            "T={} value cache page {}, index {index} differs from QKV seam",
                            route.rows(),
                            page.physical
                        )));
                    }
                    if page.key[index] == BF16_SENTINEL {
                        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                            "T={} key cache page {}, index {index} retained the sentinel",
                            route.rows(),
                            page.physical
                        )));
                    }
                }
            }
        }
        for (index, (&key, &value)) in page.key.iter().zip(&page.value).enumerate() {
            if !touched[index] && (key != BF16_SENTINEL || value != BF16_SENTINEL) {
                return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                    "T={} modified inactive cache page {}, index {index}",
                    route.rows(),
                    page.physical
                )));
            }
        }
        checked += 2 * page_values;
    }
    report.cache_values += checked;
    Ok(())
}

fn verify_tails(
    target: &ResidentModelProgram,
    program: &mut MtpPromptPrimeProgram<'_>,
    stream: &CudaStream,
    embedding_source: &[u8],
    report: &mut MtpPromptPrimeQualification,
) -> Result<(), MtpPromptPrimeQualificationError> {
    for tail in 1..=TAIL_ROUTES {
        target.reset_slot(stream, SLOT)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        for position in 0..tail {
            let target_token = [token_id(position)];
            let position_values = [u32::try_from(position).map_err(|_| {
                MtpPromptPrimeQualificationError::Mismatch(
                    "MTP prompt tail position exceeds u32".to_string(),
                )
            })?];
            let (cosine, sine) = rope(&position_values);
            target.load_residual(stream, 1, &embedding_rows(embedding_source, &target_token)?)?;
            target.load_slot_routes(stream, &[SLOT])?;
            let target_route =
                target.load_decode_state(stream, 1, &position_values, &cosine, &sine)?;
            target.replay(stream, target_route)?;
            let route = program.stage(
                stream,
                1,
                SLOT,
                position,
                &[token_id(position + 1)],
                &cosine,
                &sine,
            )?;
            program.replay(stream, route)?;
        }
        let route = MtpPromptPrimeRoute::qualified(1, SLOT, tail - 1)?;
        let observed = program.qualification_observables(stream)?;
        let pages = read_cache_prefix(target, program, stream, tail)?;
        verify_tail_cache(tail, route, &observed, &pages, report)?;
        report.tail_routes += 1;
    }
    Ok(())
}

fn verify_tail_cache(
    tail: usize,
    route: MtpPromptPrimeRoute,
    observed: &MtpPromptPrimeObservables,
    pages: &[CachePage],
    report: &mut MtpPromptPrimeQualification,
) -> Result<(), MtpPromptPrimeQualificationError> {
    let page_values = cache_page_values();
    for page in pages {
        let mut touched = vec![false; page_values];
        for position in 0..tail {
            let physical = observed.block_tables
                [route.slot() * LONG_CONTEXT_PHYSICAL_PAGES + position / ATTENTION_PAGE_SIZE]
                as usize;
            if physical != page.physical {
                continue;
            }
            for head in 0..Qwen38_27B::NUM_KV_HEADS {
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let index = Qwen38_27B::HEAD_DIM
                        * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * head)
                        + dimension;
                    touched[index] = true;
                    if page.key[index] == BF16_SENTINEL || page.value[index] == BF16_SENTINEL {
                        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                            "tail={tail} did not append cache page {}, index {index}",
                            page.physical
                        )));
                    }
                }
            }
        }
        for (index, (&key, &value)) in page.key.iter().zip(&page.value).enumerate() {
            if !touched[index] && (key != BF16_SENTINEL || value != BF16_SENTINEL) {
                return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                    "tail={tail} modified inactive cache page {}, index {index}",
                    page.physical
                )));
            }
        }
        report.cache_values += 2 * page_values;
    }
    Ok(())
}

fn read_route_cache(
    target: &ResidentModelProgram,
    program: &MtpPromptPrimeProgram<'_>,
    stream: &CudaStream,
    route: MtpPromptPrimeRoute,
) -> Result<Vec<CachePage>, MtpPromptPrimeQualificationError> {
    read_cache_range(
        target,
        program,
        stream,
        route.slot(),
        route.first_position(),
        route.rows(),
    )
}

fn read_cache_prefix(
    target: &ResidentModelProgram,
    program: &MtpPromptPrimeProgram<'_>,
    stream: &CudaStream,
    rows: usize,
) -> Result<Vec<CachePage>, MtpPromptPrimeQualificationError> {
    read_cache_range(target, program, stream, SLOT, 0, rows)
}

fn read_cache_range(
    target: &ResidentModelProgram,
    program: &MtpPromptPrimeProgram<'_>,
    stream: &CudaStream,
    slot: usize,
    first_position: usize,
    rows: usize,
) -> Result<Vec<CachePage>, MtpPromptPrimeQualificationError> {
    let mut physical = BTreeSet::new();
    for position in first_position..first_position + rows {
        physical.insert(
            usize::try_from(target.qualification_kv_physical_page(slot, position)?).map_err(
                |_| {
                    MtpPromptPrimeQualificationError::Mismatch(
                        "MTP prompt physical page exceeds usize".to_string(),
                    )
                },
            )?,
        );
    }
    physical
        .into_iter()
        .map(|physical| {
            let (key, value) = program.qualification_cache_page(stream, physical)?;
            Ok(CachePage {
                physical,
                key,
                value,
            })
        })
        .collect()
}

fn compare_observables(
    rows: usize,
    eager: &MtpPromptPrimeObservables,
    replay: &MtpPromptPrimeObservables,
) -> Result<(), MtpPromptPrimeQualificationError> {
    for (role, actual, expected) in [
        ("embedding", &replay.embedding, &eager.embedding),
        ("target hidden", &replay.target_hidden, &eager.target_hidden),
        (
            "normalized embedding",
            &replay.normalized_embedding,
            &eager.normalized_embedding,
        ),
        (
            "normalized hidden",
            &replay.normalized_hidden,
            &eager.normalized_hidden,
        ),
        ("fusion residual", &replay.residual, &eager.residual),
        (
            "attention normalized",
            &replay.attention_normalized,
            &eager.attention_normalized,
        ),
        ("QKV", &replay.qkv, &eager.qkv),
    ] {
        compare_exact(&format!("T={rows} graph {role}"), actual, expected)?;
    }
    for (role, actual, expected) in [
        ("block tables", &replay.block_tables, &eager.block_tables),
        ("slot rows", &replay.table_rows, &eager.table_rows),
        (
            "cache positions",
            &replay.cache_positions,
            &eager.cache_positions,
        ),
    ] {
        compare_exact(&format!("T={rows} graph {role}"), actual, expected)?;
    }
    for (role, actual, expected) in [
        ("MRoPE cosine", &replay.rope_cos, &eager.rope_cos),
        ("MRoPE sine", &replay.rope_sin, &eager.rope_sin),
        ("query", &replay.query, &eager.query),
    ] {
        compare_f32_bits(&format!("T={rows} graph {role}"), actual, expected)?;
    }
    Ok(())
}

fn compare_cache(
    rows: usize,
    eager: &[CachePage],
    replay: &[CachePage],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if eager.len() != replay.len() {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "T={rows} eager/graph cache page counts differ"
        )));
    }
    for (eager, replay) in eager.iter().zip(replay) {
        if eager.physical != replay.physical {
            return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                "T={rows} eager/graph physical cache pages differ"
            )));
        }
        compare_exact(&format!("T={rows} key cache"), &replay.key, &eager.key)?;
        compare_exact(
            &format!("T={rows} value cache"),
            &replay.value,
            &eager.value,
        )?;
    }
    Ok(())
}

fn verify_stable(
    program: &MtpPromptPrimeProgram<'_>,
    base: u64,
    addresses: &[usize],
    rows: usize,
) -> Result<(), MtpPromptPrimeQualificationError> {
    if program.base_address() != base || program.qualification_addresses()? != addresses {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "MTP prompt addresses changed while qualifying rows={rows}"
        )));
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &Arc<CudaContext>,
    program: &MtpPromptPrimeProgram<'_>,
    stream: &CudaStream,
) -> Result<(), MtpPromptPrimeQualificationError> {
    for rows in ROUTES {
        let graph = program.qualification_graph(MtpPromptPrimeRoute::qualified(rows, SLOT, 0)?)?;
        // SAFETY: the borrowed program owns the graph and every allocation it
        // captured, outliving these replays and the synchronize below.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for rows in ROUTES.into_iter().rev() {
            let graph =
                program.qualification_graph(MtpPromptPrimeRoute::qualified(rows, SLOT, 0)?)?;
            // SAFETY: the borrowed program owns the graph and every allocation it
            // captured, outliving these replays and the synchronize below.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "post-warmup prompt graphs changed device memory from {before:?} to {after:?}"
        )));
    }
    Ok(())
}

fn selected_rows(rows: usize) -> Vec<usize> {
    [0, rows / 2, rows - 1]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn require_projection_close(
    role: &str,
    row: usize,
    output: usize,
    actual: u16,
    expected: f64,
    maximum: &mut f32,
) -> Result<(), MtpPromptPrimeQualificationError> {
    let actual = bf16_to_f32(actual);
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.25f32.max(expected.abs() as f32 * 0.015);
    if !actual.is_finite() || error > tolerance {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} row={row}, output={output}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn compare_close_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), MtpPromptPrimeQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = bf16_to_f32(actual);
        let expected = bf16_to_f32(expected);
        let error = (actual - expected).abs();
        *maximum = maximum.max(error);
        // This owner launches the production BF16 residual-norm operation, so
        // use its qualified approximation/rounding envelope at this seam.
        let tolerance = 0.015625f32.max(expected.abs() * 0.005);
        if !actual.is_finite() || error > tolerance {
            return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
                "{role} index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    Ok(())
}

fn token_ids(first: usize, rows: usize) -> Vec<u32> {
    (first..first + rows).map(token_id).collect()
}

fn token_id(position: usize) -> u32 {
    ((position.wrapping_mul(7_919).wrapping_add(101)) % Qwen38_27B::VOCAB) as u32
}

fn positions(first: usize, rows: usize) -> Result<Vec<u32>, MtpPromptPrimeQualificationError> {
    (first..first + rows)
        .map(|position| {
            u32::try_from(position).map_err(|_| {
                MtpPromptPrimeQualificationError::Mismatch(
                    "MTP prompt position exceeds u32".to_string(),
                )
            })
        })
        .collect()
}

fn rope(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; positions.len() * ROTARY_PAIRS];
    let mut sine = vec![0.0; positions.len() * ROTARY_PAIRS];
    for (row, &position) in positions.iter().enumerate() {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
            let (sin, cos) = (f64::from(position) * frequency).sin_cos();
            cosine[row * ROTARY_PAIRS + pair] = cos as f32;
            sine[row * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn embedding_rows(
    source: &[u8],
    token_ids: &[u32],
) -> Result<Vec<u16>, MtpPromptPrimeQualificationError> {
    let mut values = vec![0u16; token_ids.len() * Qwen38_27B::HIDDEN];
    for (row, &token) in token_ids.iter().enumerate() {
        let token = token as usize;
        let begin = token * Qwen38_27B::HIDDEN * 2;
        let end = begin + Qwen38_27B::HIDDEN * 2;
        let source_row = source.get(begin..end).ok_or_else(|| {
            MtpPromptPrimeQualificationError::Mismatch(format!(
                "embedding token {token} is outside source"
            ))
        })?;
        for (target, bytes) in values[row * Qwen38_27B::HIDDEN..(row + 1) * Qwen38_27B::HIDDEN]
            .iter_mut()
            .zip(source_row.as_chunks::<2>().0)
        {
            *target = u16::from_le_bytes(*bytes);
        }
    }
    Ok(values)
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if actual.len() != expected.len() {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} lengths differ: {} versus {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if actual.len() != expected.len() {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} lengths differ: {} versus {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} differs at FP32 value {index}"
        )));
    }
    Ok(())
}

fn require_bf16_sentinel(
    role: &str,
    values: &[u16],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != BF16_SENTINEL) {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} modified inactive BF16 value {index}"
        )));
    }
    Ok(())
}

fn require_f32_sentinel(
    role: &str,
    values: &[f32],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if let Some(index) = values
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} modified inactive FP32 value {index}"
        )));
    }
    Ok(())
}

fn require_u32_sentinel(
    role: &str,
    values: &[u32],
) -> Result<(), MtpPromptPrimeQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != 0xa5a5_a5a5) {
        return Err(MtpPromptPrimeQualificationError::Mismatch(format!(
            "{role} modified inactive U32 value {index}"
        )));
    }
    Ok(())
}

fn cache_page_values() -> usize {
    Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM
}

fn observable_values(observed: &MtpPromptPrimeObservables) -> usize {
    observed.embedding.len()
        + observed.target_hidden.len()
        + observed.normalized_embedding.len()
        + observed.normalized_hidden.len()
        + observed.residual.len()
        + observed.attention_normalized.len()
        + observed.qkv.len()
        + observed.rope_cos.len()
        + observed.rope_sin.len()
        + observed.block_tables.len()
        + observed.table_rows.len()
        + observed.cache_positions.len()
        + observed.query.len()
}

#[cfg(test)]
mod tests {
    use super::{ROUTES, TAIL_ROUTES, qualify_mtp_prompt_prime};
    use std::path::PathBuf;

    #[test]
    fn mtp_prompt_prime_suite_route_inventory_is_exact() {
        assert_eq!(ROUTES, [1, 32, 64, 128, 1_024]);
        assert_eq!(TAIL_ROUTES, 31);
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_prompt_prime_suite_source_values_match_every_route_and_tail() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_prompt_prime(&root).expect("MTP prompt-prime qualification");

        assert_eq!(report.leaf_oracle_suites, 3);
        assert_eq!(report.prompt_routes, ROUTES.len());
        assert_eq!(report.tail_routes, TAIL_ROUTES);
        assert_eq!(report.resident_weight_bytes, 251_689_984);
        assert_eq!(report.cache_bytes, 901_251_072);
        assert_eq!(report.workspace_bytes, 117_820_864);
        assert_eq!(report.owner_bytes, 1_270_761_920);
        assert_eq!(report.arena_bytes, 1_270_761_984);
        assert_eq!(report.padding_bytes, 64);
        assert_eq!(report.host_stager_bytes, 10_756_096);
        assert_eq!(report.graph_count, ROUTES.len());
    }
}
