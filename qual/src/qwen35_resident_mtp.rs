//! Composition qualification for the Qwen3.5 target and MTP resident owners.

use crate::oracles::codecs::bf16_to_f32;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen35MtpPromptRoute, Qwen35ResidentMtpObservables,
    Qwen35ResidentMtpProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, Bf16TextEndpointBindings, CheckpointError, CheckpointSnapshot, Qwen35_9B,
};

const ROTARY_PAIRS: usize = 32;
const PROMPT_ROUTES: [usize; 3] = [32, 64, 128];
const VERIFY_ROUTES: [usize; 4] = [1, 2, 3, 4];
const PAGE_VALUES: usize = 4 * 64 * 256;
const DECODE_PAGES: usize = 3;
const SENTINEL: u8 = 0xA5;

/// Failure of the resident Qwen3.5 MTP composition gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35ResidentMtpQualificationError {
    /// Snapshot admission failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident ownership or routing failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA execution or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact device was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// The composed behavior disagreed with its independent seams.
    #[error("Qwen3.5 resident MTP qualification failed: {0}")]
    Mismatch(String),
}

/// Exact route, lifecycle, and ownership evidence from the composition gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35ResidentMtpQualification {
    /// Exact decode batches with eager/CUDA Graph agreement.
    pub draft_routes: usize,
    /// Complete mutable values compared across eager and graph execution.
    pub graph_replay_values: usize,
    /// Source-backed LM-head values checked with independent FP64 dots.
    pub sampled_logits: usize,
    /// Exact prompt-prime routes with identical cache publication.
    pub prompt_routes: usize,
    /// Prompt K/V values compared after eager and graph execution.
    pub prompt_cache_values: usize,
    /// Exact causal target-verification routes compared eager-to-graph.
    pub target_verify_routes: usize,
    /// Target logits, residuals, and cache words compared exactly across replay modes.
    pub target_verify_values: usize,
    /// Lifecycle transitions retaining identical physical mappings.
    pub lifecycle_transitions: usize,
    /// Stable addresses retained across all routes.
    pub stable_addresses: usize,
    /// Target plus MTP weight bytes with one endpoint.
    pub weight_bytes: usize,
    /// Target plus MTP BF16 cache bytes.
    pub cache_bytes: usize,
    /// Address-stable workspace and table bytes.
    pub workspace_bytes: usize,
    /// Complete device allocation bytes.
    pub arena_bytes: usize,
    /// Fixed page-locked embedding stagers.
    pub host_stager_bytes: usize,
    /// Fixed target plus MTP host page owners.
    pub kv_host_owner_bytes: usize,
}

/// Qualifies exact draft graphs, prompt priming, and mirrored cache lifecycle.
pub fn qualify_qwen35_resident_mtp(
    root: &Path,
) -> Result<Qwen35ResidentMtpQualification, Qwen35ResidentMtpQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let endpoint = Bf16TextEndpointBindings::bind(snapshot.as_ref())?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = Qwen35ResidentMtpProgram::from_snapshot(&context, Arc::clone(&snapshot))?;
    let stable_addresses = program.qualification_addresses()?;
    let layout = program.layout();
    verify_accounting(&program)?;
    let mut report = Qwen35ResidentMtpQualification {
        draft_routes: 0,
        graph_replay_values: 0,
        sampled_logits: 0,
        prompt_routes: 0,
        prompt_cache_values: 0,
        target_verify_routes: 0,
        target_verify_values: 0,
        lifecycle_transitions: 0,
        stable_addresses: stable_addresses.len(),
        weight_bytes: layout.resident_weight_bytes(),
        cache_bytes: layout.cache_bytes(),
        workspace_bytes: layout.workspace_bytes(),
        arena_bytes: layout.arena_bytes(),
        host_stager_bytes: program.host_stager_bytes(),
        kv_host_owner_bytes: program.kv_host_owner_bytes(),
    };

    for batch in 1..=MAX_BATCH {
        prepare_decode(&mut program, &stream, batch)?;
        program.qualification_reset_outputs(&stream, SENTINEL)?;
        program.qualification_launch_draft(&stream, batch)?;
        let eager = program.qualification_observables(&stream, batch)?;
        let cache_values = batch * DECODE_PAGES * PAGE_VALUES;
        let eager_cache = program.qualification_mtp_cache_prefix(&stream, cache_values)?;

        program.qualification_reset_outputs(&stream, SENTINEL)?;
        program.replay_draft(&stream, batch)?;
        let graph = program.qualification_observables(&stream, batch)?;
        let graph_cache = program.qualification_mtp_cache_prefix(&stream, cache_values)?;
        report.graph_replay_values += compare_observables(batch, &eager, &graph)?;
        compare_exact("draft key cache", &eager_cache.0, &graph_cache.0)?;
        compare_exact("draft value cache", &eager_cache.1, &graph_cache.1)?;
        report.graph_replay_values += graph_cache.0.len() + graph_cache.1.len();
        report.sampled_logits += verify_sampled_logits(batch, &graph, endpoint)?;
        report.draft_routes += 1;
        if program.qualification_addresses()? != stable_addresses {
            return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
                "resident addresses changed while qualifying B={batch}"
            )));
        }
    }

    for rows in PROMPT_ROUTES {
        let route = prepare_prompt(&mut program, &stream, rows)?;
        program.qualification_launch_prompt(&stream, route)?;
        let eager = program.qualification_mtp_cache_prefix(&stream, PAGE_VALUES)?;
        program.replay_prompt_prime(&stream, route)?;
        let graph = program.qualification_mtp_cache_prefix(&stream, PAGE_VALUES)?;
        compare_exact("prompt key cache", &eager.0, &graph.0)?;
        compare_exact("prompt value cache", &eager.1, &graph.1)?;
        if graph.0.iter().all(|&word| word == 0) || graph.1.iter().all(|&word| word == 0) {
            return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
                "T={rows} prompt prime left one complete cache plane zero"
            )));
        }
        report.prompt_routes += 1;
        report.prompt_cache_values += graph.0.len() + graph.1.len();
    }

    report.target_verify_values = verify_target_routes(&mut program, &stream)?;
    report.target_verify_routes = VERIFY_ROUTES.len();

    report.lifecycle_transitions = verify_lifecycle(&mut program, &stream)?;
    verify_no_post_warmup_allocation(&mut program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn verify_target_routes(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
) -> Result<usize, Qwen35ResidentMtpQualificationError> {
    const FIRST_POSITION: usize = 32;
    let mut compared = 0;
    for rows in VERIFY_ROUTES {
        prepare_target_prefix(program, stream, FIRST_POSITION + rows)?;
        let token_ids = (0..rows).map(|row| 900 + row as u32).collect::<Vec<_>>();
        let rope_cos = vec![1.0f32; rows * ROTARY_PAIRS];
        let rope_sin = vec![0.0f32; rows * ROTARY_PAIRS];
        program.stage_target_verify(stream, &token_ids, 0, FIRST_POSITION, &rope_cos, &rope_sin)?;
        program.qualification_launch_target_verify(stream, rows)?;
        let eager_logits = program.read_logits(stream, rows)?;
        let eager_residual = program.qualification_target_residual(stream, rows)?;
        let physical_page = usize::try_from(
            program
                .qualification_kv_route(0, FIRST_POSITION)?
                .physical_page(),
        )
        .map_err(|_| {
            Qwen35ResidentMtpQualificationError::Mismatch(
                "target physical page does not fit usize".to_string(),
            )
        })?;
        let eager_cache = program.qualification_target_cache_page(stream, physical_page)?;

        prepare_target_prefix(program, stream, FIRST_POSITION + rows)?;
        program.stage_target_verify(stream, &token_ids, 0, FIRST_POSITION, &rope_cos, &rope_sin)?;
        program.replay_target_verify(stream, rows)?;
        let verify_logits = program.read_logits(stream, rows)?;
        let verify_residual = program.qualification_target_residual(stream, rows)?;
        let verify_cache = program.qualification_target_cache_page(stream, physical_page)?;

        compare_words(
            &format!("K={rows} target verification graph residual"),
            &eager_residual,
            &verify_residual,
        )?;
        compare_exact(
            "target verification graph key cache",
            &eager_cache.0,
            &verify_cache.0,
        )?;
        compare_exact(
            "target verification graph value cache",
            &eager_cache.1,
            &verify_cache.1,
        )?;
        compare_exact(
            "target verification graph logits",
            &eager_logits,
            &verify_logits,
        )?;

        if rows == 1 {
            prepare_target_prefix(program, stream, FIRST_POSITION + 1)?;
            program.stage_target_embeddings(stream, &token_ids)?;
            program.load_decode_state(
                stream,
                &[FIRST_POSITION as u32],
                &[0],
                &[1.0; ROTARY_PAIRS],
                &[0.0; ROTARY_PAIRS],
            )?;
            program.replay_target(stream, 1)?;
            compare_words(
                "K=1 target verification production residual",
                &program.qualification_target_residual(stream, 1)?,
                &verify_residual,
            )?;
            compare_exact(
                "K=1 target verification production logits",
                &program.read_logits(stream, 1)?,
                &verify_logits,
            )?;
            let serial_cache = program.qualification_target_cache_page(stream, physical_page)?;
            compare_exact(
                "K=1 target verification production key cache",
                &serial_cache.0,
                &verify_cache.0,
            )?;
            compare_exact(
                "K=1 target verification production value cache",
                &serial_cache.1,
                &verify_cache.1,
            )?;
        }
        compared += verify_logits.len()
            + verify_residual.len()
            + verify_cache.0.len()
            + verify_cache.1.len();
    }
    Ok(compared)
}

fn prepare_target_prefix(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
    reserved_tokens: usize,
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    const ROWS: usize = 32;
    program.reset_state(stream)?;
    program.activate_kv_slot(0)?;
    program.reserve_kv_slot_tokens(stream, 0, reserved_tokens)?;
    let token_ids = (0..ROWS).map(|row| 700 + row as u32).collect::<Vec<_>>();
    let rope_cos = vec![1.0f32; ROWS * ROTARY_PAIRS];
    let rope_sin = vec![0.0f32; ROWS * ROTARY_PAIRS];
    let route = program.stage_target_prefill(stream, &token_ids, 0, 0, &rope_cos, &rope_sin)?;
    program.replay_target_prefill(stream, route)?;
    Ok(())
}

fn verify_sampled_logits(
    batch: usize,
    observed: &Qwen35ResidentMtpObservables,
    bindings: Bf16TextEndpointBindings<'_>,
) -> Result<usize, Qwen35ResidentMtpQualificationError> {
    const ROWS: [usize; 8] = [0, 1, 17, 257, 4_096, 32_768, 131_071, Qwen35_9B::VOCAB - 1];
    for token in 0..batch {
        let hidden_begin = token * Qwen35_9B::HIDDEN;
        let normalized =
            &observed.mtp.final_normalized[hidden_begin..hidden_begin + Qwen35_9B::HIDDEN];
        for row in ROWS {
            let weight_begin = row * Qwen35_9B::HIDDEN;
            let mut expected = 0.0f64;
            for (column, &activation) in normalized.iter().enumerate() {
                let weight = bindings
                    .lm_head
                    .word(weight_begin + column)
                    .ok_or_else(|| {
                        Qwen35ResidentMtpQualificationError::Mismatch(format!(
                            "LM-head source row {row} ends before column {column}"
                        ))
                    })?;
                expected += f64::from(bf16_to_f32(activation)) * f64::from(bf16_to_f32(weight));
            }
            let actual = bf16_to_f32(observed.logits[token * Qwen35_9B::VOCAB + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            let tolerance = 0.0625f32.max(expected.abs() as f32 * 0.01);
            if error > tolerance {
                return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
                    "B={batch} draft logit token={token}, vocabulary={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
    }

    Ok(batch * ROWS.len())
}

fn prepare_decode(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
    batch: usize,
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    program.reset_state(stream)?;
    let slots = (0..batch).collect::<Vec<_>>();
    for &slot in &slots {
        program.activate_kv_slot(slot)?;
        program.reserve_kv_slot_tokens(stream, slot, 131)?;
    }
    let target_ids = (0..batch).map(|row| 100 + row as u32).collect::<Vec<_>>();
    let draft_ids = (0..batch).map(|row| 200 + row as u32).collect::<Vec<_>>();
    let positions = vec![130u32; batch];
    let rope_cos = vec![1.0f32; batch * ROTARY_PAIRS];
    let rope_sin = vec![0.0f32; batch * ROTARY_PAIRS];
    program.stage_target_embeddings(stream, &target_ids)?;
    program.load_decode_state(stream, &positions, &slots, &rope_cos, &rope_sin)?;
    program.replay_target(stream, batch)?;
    program.stage_mtp_embeddings(stream, &draft_ids)?;

    Ok(())
}

fn prepare_prompt(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
    rows: usize,
) -> Result<Qwen35MtpPromptRoute, Qwen35ResidentMtpQualificationError> {
    program.reset_state(stream)?;
    program.activate_kv_slot(0)?;
    program.reserve_kv_slot_tokens(stream, 0, rows)?;
    let target_ids = (0..rows)
        .map(|row| 300 + (row % 97) as u32)
        .collect::<Vec<_>>();
    let draft_ids = (0..rows)
        .map(|row| 500 + (row % 89) as u32)
        .collect::<Vec<_>>();
    let rope_cos = vec![1.0f32; rows * ROTARY_PAIRS];
    let rope_sin = vec![0.0f32; rows * ROTARY_PAIRS];
    let target_route =
        program.stage_target_prefill(stream, &target_ids, 0, 0, &rope_cos, &rope_sin)?;
    program.replay_target_prefill(stream, target_route)?;
    let route = program.stage_prompt_prime(stream, &draft_ids, 0, 0, &rope_cos, &rope_sin)?;

    Ok(route)
}

fn compare_observables(
    batch: usize,
    eager: &Qwen35ResidentMtpObservables,
    graph: &Qwen35ResidentMtpObservables,
) -> Result<usize, Qwen35ResidentMtpQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            compare_exact(stringify!($field), &eager.mtp.$field, &graph.mtp.$field)?;
        };
    }
    same!(normalized_embedding);
    same!(normalized_hidden);
    same!(residual);
    same!(attention_normalized);
    same!(qkv);
    same!(query);
    same!(attention);
    same!(attention_activation);
    same!(attention_branch);
    same!(post_attention_residual);
    same!(mlp_normalized);
    same!(swiglu);
    same!(mlp_branch);
    same!(residual_output);
    same!(final_normalized);
    compare_exact("shared endpoint logits", &eager.logits, &graph.logits)?;
    for (index, &bits) in graph.logits.iter().enumerate() {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
                "B={batch} draft logit {index} is not finite"
            )));
        }
    }

    Ok(eager.mtp.normalized_embedding.len()
        + eager.mtp.normalized_hidden.len()
        + eager.mtp.residual.len()
        + eager.mtp.attention_normalized.len()
        + eager.mtp.qkv.len()
        + eager.mtp.query.len()
        + eager.mtp.attention.len()
        + eager.mtp.attention_activation.len()
        + eager.mtp.attention_branch.len()
        + eager.mtp.post_attention_residual.len()
        + eager.mtp.mlp_normalized.len()
        + eager.mtp.swiglu.len()
        + eager.mtp.mlp_branch.len()
        + eager.mtp.residual_output.len()
        + eager.mtp.final_normalized.len()
        + eager.logits.len())
}

fn verify_lifecycle(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
) -> Result<usize, Qwen35ResidentMtpQualificationError> {
    const SLOT: usize = 7;
    program.reset_state(stream)?;
    program.activate_kv_slot(SLOT)?;
    program.reserve_kv_slot_tokens(stream, SLOT, 4_097)?;
    let route = program.qualification_kv_route(SLOT, 4_096)?;
    if route.physical_page() != 64 || route.page_offset() != 0 {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "long-context mirror mapped position 4096 to page {} offset {}",
            route.physical_page(),
            route.page_offset()
        )));
    }
    program.truncate_kv_slot_tokens(stream, SLOT, 65)?;
    program.retain_kv_slot(SLOT)?;
    program.activate_kv_slot(SLOT)?;
    program.reserve_kv_slot_tokens(stream, SLOT, 130)?;
    program.recycle_kv_slot(stream, SLOT)?;

    Ok(6)
}

fn verify_no_post_warmup_allocation(
    program: &mut Qwen35ResidentMtpProgram,
    stream: &CudaStream,
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    prepare_decode(program, stream, 1)?;
    program.replay_draft(stream, 1)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..3 {
        program.replay_draft(stream, 1)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn verify_accounting(
    program: &Qwen35ResidentMtpProgram,
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    let layout = program.layout();
    if layout.resident_weight_bytes() != 6_418_401_280
        || layout.cache_bytes() != 9_714_008_064
        || layout.workspace_bytes() != 1_121_302_368
        || layout.owner_bytes() != 17_253_711_712
        || layout.arena_bytes() != 17_253_733_120
        || layout.padding_bytes() != 21_408
        || program.host_stager_bytes() != 2_162_688
        || program.kv_host_owner_bytes() != 270_336
    {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(
            "resident target/MTP ownership differs from the exact accounting".to_string(),
        ));
    }

    Ok(())
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "{role} has {}/{} values",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }

    Ok(())
}

fn compare_words(
    role: &str,
    actual: &[u16],
    expected: &[u16],
) -> Result<(), Qwen35ResidentMtpQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "{role} has {}/{} values",
            actual.len(),
            expected.len()
        )));
    }
    if let Some((index, (&actual, &expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(Qwen35ResidentMtpQualificationError::Mismatch(format!(
            "{role} differs at value {index}: {actual:#06x}/{expected:#06x}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PROMPT_ROUTES, VERIFY_ROUTES, qualify_qwen35_resident_mtp};
    use std::path::PathBuf;

    #[test]
    fn qwen35_resident_mtp_suite_inventory_is_exact() {
        assert_eq!(PROMPT_ROUTES, [32, 64, 128]);
        assert_eq!(VERIFY_ROUTES, [1, 2, 3, 4]);
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive SM120 device"]
    fn qwen35_resident_mtp_suite_composes_draft_prompt_and_mirrored_lifecycle()
    -> Result<(), super::Qwen35ResidentMtpQualificationError> {
        let root = PathBuf::from(
            std::env::var("TUISKO_QWEN35_SNAPSHOT").expect("TUISKO_QWEN35_SNAPSHOT is required"),
        );
        let report = qualify_qwen35_resident_mtp(&root)?;
        assert_eq!(report.draft_routes, 8);
        assert_eq!(report.sampled_logits, 288);
        assert_eq!(report.prompt_routes, 3);
        assert_eq!(report.target_verify_routes, 4);
        assert_eq!(report.target_verify_values, 6_718_464);
        assert_eq!(report.lifecycle_transitions, 6);
        assert_eq!(report.weight_bytes, 6_418_401_280);
        assert_eq!(report.cache_bytes, 9_714_008_064);
        assert_eq!(report.workspace_bytes, 1_121_302_368);
        assert_eq!(report.arena_bytes, 17_253_733_120);
        assert_eq!(report.host_stager_bytes, 2_162_688);
        assert_eq!(report.kv_host_owner_bytes, 270_336);
        Ok(())
    }
}
