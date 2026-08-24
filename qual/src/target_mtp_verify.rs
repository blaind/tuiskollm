//! Source-backed qualification for provisional target MTP verification and commit.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, quantize_oracle,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{qualify_gdn_prepare, qualify_gdn_recurrence, qualify_gdn_state_snapshot};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, ResidentModelProgram, ResidentMtpSegmentedVerifyRoute,
    ResidentMtpVerifyObservables, ResidentMtpVerifyRoute,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38_27B, TextEndpointBindings};

const VERIFY_ROUTES: usize = 4;
const GDN_LAYERS: usize = 48;
const ATTENTION_LAYERS: usize = 16;
const HISTORY_TAPS: usize = 3;
const ROTARY_PAIRS: usize = 32;
const ROTARY_DIM: usize = 64;
const SLOT: usize = 0;
const TOKEN_IDS: [u32; VERIFY_ROUTES] = [101, 7_919, 48_127, 199_933];
const SELECTED_LOGIT_ROWS: [usize; 5] = [0, 1, 31_337, 131_071, Qwen38_27B::VOCAB - 1];

/// Failure of the source-backed target verification gate.
#[derive(Debug, thiserror::Error)]
pub enum TargetMtpVerifyQualificationError {
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
    Precondition(#[from] crate::DeviceBenchmarkError),
    /// Device behavior disagreed with an independent route or value oracle.
    #[error("target MTP verification qualification failed: {0}")]
    Mismatch(String),
}

/// Exact route, seam, replay, persistent-state, and owner counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetMtpVerifyQualification {
    /// Independent represented-value and mathematical leaf suites completed.
    pub leaf_oracle_suites: usize,
    /// Exact K=1..4 provisional routes exercised.
    pub verify_routes: usize,
    /// Exact K=1..4 accepted-prefix commit routes exercised.
    pub commit_routes: usize,
    /// Every exact lane-major `(B=1..8, K=1..4)` route exercised.
    pub segmented_verify_routes: usize,
    /// Per-route commits with distinct accepted counts across active lanes.
    pub segmented_commit_routes: usize,
    /// Final-normalized and selected-logit values checked by source mathematics.
    pub endpoint_oracle_values: usize,
    /// Provisional, recorded, live, and output values reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Live persistent values proved unchanged by provisional execution.
    pub rollback_values: usize,
    /// Live persistent values matched after accepted-prefix replay.
    pub committed_values: usize,
    /// Represented cache bytes compared across eager, graph, and sequential routes.
    pub cache_values: usize,
    /// Complete immutable target verify/commit graph inventory.
    pub graph_count: usize,
    /// Executable instances retained after long-context variant sharing.
    pub graph_executable_count: usize,
    /// Long-context partition variants checked by eager and updated-graph replay.
    pub long_segmented_variant_routes: usize,
    /// Exact resident shared-workspace bytes.
    pub workspace_bytes: usize,
    /// Complete resident and shared-KV arena bytes.
    pub arena_bytes: usize,
    /// Alignment padding across both arenas.
    pub padding_bytes: usize,
    /// Largest absolute error at a source-backed endpoint boundary.
    pub maximum_absolute_error: f32,
}

#[derive(Clone)]
struct Fixture {
    history: Vec<u16>,
    state: Vec<f32>,
}

struct SequentialReference {
    observed: ResidentMtpVerifyObservables,
    cache: (Vec<u8>, Vec<u8>),
}

/// Qualifies exact K=1..4 target verification and accepted-prefix commit.
pub fn qualify_target_mtp_verify(
    root: &Path,
) -> Result<TargetMtpVerifyQualification, TargetMtpVerifyQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    run_leaf_oracles()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let endpoint = TextEndpointBindings::bind(snapshot.as_ref())?;
    let final_norm = endpoint.final_norm.words().collect::<Vec<_>>();
    let lm_head_codes = endpoint.lm_head.codes();
    let lm_head_scales = endpoint.lm_head_scale.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(TargetMtpVerifyQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = ResidentModelProgram::from_snapshot(&context, snapshot.clone())?;
    for slot in 0..MAX_BATCH {
        program.activate_kv_slot(slot)?;
        program.reserve_kv_slot_tokens(&stream, slot, ATTENTION_PAGE_SIZE)?;
    }
    let fixture = fixture(&program);
    verify_owner(&program)?;
    let base = (program.base_address(), program.kv_base_address());
    let addresses = program.qualification_addresses();
    if addresses.len() != 1_168 || addresses.iter().copied().collect::<BTreeSet<_>>().len() != 1_168
    {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "target resident owner exposes {} addresses, expected 1,168 unique addresses",
            addresses.len()
        )));
    }
    let mut report = TargetMtpVerifyQualification {
        leaf_oracle_suites: 3,
        verify_routes: 0,
        commit_routes: 0,
        segmented_verify_routes: 0,
        segmented_commit_routes: 0,
        endpoint_oracle_values: 0,
        graph_replay_values: 0,
        rollback_values: 0,
        committed_values: 0,
        cache_values: 0,
        graph_count: program.target_mtp_graph_count(),
        graph_executable_count: program.target_mtp_graph_executable_count(),
        long_segmented_variant_routes: 0,
        workspace_bytes: program.workspace_bytes(),
        arena_bytes: program.arena_bytes(),
        padding_bytes: program.padding_bytes(),
        maximum_absolute_error: 0.0,
    };
    verify_first_gdn_k1_seams(&mut program, &stream, &fixture)?;
    verify_k1_layer_seams(&mut program, &stream, &fixture)?;

    for tokens in 1..=VERIFY_ROUTES {
        let route = prepare_verify(&mut program, &stream, tokens, &fixture)?;
        program.launch_target_mtp_verify_eager(&stream, route)?;
        let eager = program.qualification_target_mtp_observables(&stream, route)?;
        let eager_cache = cache_page(&program, &stream)?;
        verify_rollback(tokens, &fixture, &eager, &mut report)?;
        verify_record_boundaries(tokens, &eager)?;
        verify_cache_boundaries(tokens, &eager_cache)?;
        verify_endpoint_oracle(
            tokens,
            &eager,
            &final_norm,
            lm_head_codes,
            &lm_head_scales,
            &mut report,
        )?;

        let replay_route = prepare_verify(&mut program, &stream, tokens, &fixture)?;
        program.replay_target_mtp_verify(&stream, replay_route)?;
        let replay = program.qualification_target_mtp_observables(&stream, replay_route)?;
        let replay_cache = cache_page(&program, &stream)?;
        compare_observables("verify graph", &replay, &eager)?;
        compare_exact("verify graph key cache", &replay_cache.0, &eager_cache.0)?;
        compare_exact("verify graph value cache", &replay_cache.1, &eager_cache.1)?;
        report.graph_replay_values += observable_values(&eager);
        report.cache_values += replay_cache.0.len() + replay_cache.1.len();

        let sequential = if tokens == 1 {
            let sequential = sequential_reference(&mut program, &stream, tokens, &fixture, route)?;
            compare_last_provisional(&eager, &sequential.observed)?;
            compare_exact(
                "K=1 sequential key cache",
                &eager_cache.0,
                &sequential.cache.0,
            )?;
            compare_exact(
                "K=1 sequential value cache",
                &eager_cache.1,
                &sequential.cache.1,
            )?;
            report.cache_values += sequential.cache.0.len() + sequential.cache.1.len();
            Some(sequential)
        } else {
            None
        };

        for accepted in 1..=tokens {
            let commit_route = prepare_verify(&mut program, &stream, tokens, &fixture)?;
            program.replay_target_mtp_verify(&stream, commit_route)?;
            program.launch_target_mtp_commit_eager(&stream, commit_route, accepted)?;
            let eager_commit =
                program.qualification_target_mtp_observables(&stream, commit_route)?;
            verify_committed_history(accepted, &fixture, &eager, &eager_commit, &mut report)?;
            if accepted == tokens {
                compare_last_committed(&eager, &eager_commit, &mut report)?;
            }
            if let Some(sequential) = &sequential {
                compare_committed(&eager_commit, &sequential.observed, &mut report)?;
            }

            let graph_commit_route = prepare_verify(&mut program, &stream, tokens, &fixture)?;
            program.replay_target_mtp_verify(&stream, graph_commit_route)?;
            program.replay_target_mtp_commit(&stream, graph_commit_route, accepted)?;
            let graph_commit =
                program.qualification_target_mtp_observables(&stream, graph_commit_route)?;
            compare_observables("commit graph", &graph_commit, &eager_commit)?;
            report.graph_replay_values += observable_values(&eager_commit);
            report.commit_routes += 1;
        }
        report.verify_routes += 1;
        verify_stable(&program, base, &addresses, tokens)?;
    }

    qualify_segmented_routes(
        &mut program,
        &stream,
        &fixture,
        &final_norm,
        lm_head_codes,
        &lm_head_scales,
        base,
        &addresses,
        &mut report,
    )?;
    qualify_long_segmented_variants(&mut program, &stream, &fixture, &mut report)?;

    verify_no_post_warmup_allocation(&mut program, &stream, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn qualify_segmented_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    base: (u64, u64),
    addresses: &[usize],
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    for batch in 1..=MAX_BATCH {
        for tokens in 1..=VERIFY_ROUTES {
            let (route, fixtures) = prepare_segmented(program, stream, batch, tokens, fixture)?;
            program.launch_target_mtp_segmented_verify_eager(stream, route)?;
            let eager = program.qualification_target_mtp_segmented_observables(stream, route)?;
            let eager_cache = segmented_cache_pages(program, stream, batch)?;
            verify_segmented_rollback(batch, &fixtures, &eager, report)?;
            verify_segmented_record_boundaries(batch, tokens, &eager)?;
            for cache in &eager_cache {
                verify_cache_boundaries(tokens, cache)?;
            }
            verify_endpoint_oracle(
                route.rows(),
                &eager,
                final_norm,
                lm_head_codes,
                lm_head_scales,
                report,
            )?;

            let (graph_route, _) = prepare_segmented(program, stream, batch, tokens, fixture)?;
            program.replay_target_mtp_segmented_verify(stream, graph_route)?;
            let replay =
                program.qualification_target_mtp_segmented_observables(stream, graph_route)?;
            let replay_cache = segmented_cache_pages(program, stream, batch)?;
            compare_observables("segmented verify graph", &replay, &eager)?;
            for (lane, (actual, expected)) in replay_cache.iter().zip(&eager_cache).enumerate() {
                compare_exact(
                    &format!("segmented B={batch} K={tokens} lane {lane} key cache"),
                    &actual.0,
                    &expected.0,
                )?;
                compare_exact(
                    &format!("segmented B={batch} K={tokens} lane {lane} value cache"),
                    &actual.1,
                    &expected.1,
                )?;
                report.cache_values += actual.0.len() + actual.1.len();
            }
            report.graph_replay_values += observable_values(&eager);

            let accepted = (0..batch).map(|lane| lane % tokens + 1).collect::<Vec<_>>();
            let (commit_route, commit_fixtures) =
                prepare_segmented(program, stream, batch, tokens, fixture)?;
            program.launch_target_mtp_segmented_verify_eager(stream, commit_route)?;
            let verify =
                program.qualification_target_mtp_segmented_observables(stream, commit_route)?;
            program.commit_target_mtp_segmented(stream, commit_route, &accepted)?;
            let eager_commit =
                program.qualification_target_mtp_segmented_observables(stream, commit_route)?;
            verify_segmented_committed_history(
                batch,
                tokens,
                &accepted,
                &commit_fixtures,
                &verify,
                &eager_commit,
                report,
            )?;

            let (graph_commit_route, _) =
                prepare_segmented(program, stream, batch, tokens, fixture)?;
            let graph = program.qualification_repeated_target_mtp_segmented_graph(
                stream,
                graph_commit_route,
                Some(&accepted),
                1,
            )?;
            graph.launch(stream).map_err(GpuError::from)?;
            let graph_commit = program
                .qualification_target_mtp_segmented_observables(stream, graph_commit_route)?;
            compare_observables(
                "segmented verify/commit graph",
                &graph_commit,
                &eager_commit,
            )?;
            report.graph_replay_values += observable_values(&eager_commit);
            report.segmented_verify_routes += 1;
            report.segmented_commit_routes += 1;
            verify_stable(program, base, addresses, tokens)?;
        }
    }
    Ok(())
}

fn prepare_segmented(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    batch: usize,
    tokens: usize,
    fixture: &Fixture,
) -> Result<(ResidentMtpSegmentedVerifyRoute, Vec<Fixture>), TargetMtpVerifyQualificationError> {
    let fixtures = (0..batch)
        .map(|slot| segmented_fixture(fixture, slot))
        .collect::<Vec<_>>();
    for (slot, lane_fixture) in fixtures.iter().enumerate() {
        program.reset_slot(stream, slot)?;
        program.qualification_load_target_mtp_gdn_slot(
            stream,
            slot,
            &lane_fixture.history,
            &lane_fixture.state,
        )?;
    }
    program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
    let token_ids = (0..batch)
        .flat_map(|lane| {
            TOKEN_IDS
                .iter()
                .take(tokens)
                .map(move |&token| (token + lane as u32 * 8_191) % Qwen38_27B::VOCAB as u32)
        })
        .collect::<Vec<_>>();
    program.stage_target_mtp_segmented_embeddings(stream, &token_ids)?;
    let positions = (0..batch)
        .flat_map(|_| 0..tokens as u32)
        .collect::<Vec<_>>();
    let (cosine, sine) = rope(&positions);
    let slots = (0..batch).collect::<Vec<_>>();
    let first_positions = vec![0; batch];
    let route = program.load_target_mtp_segmented_verify_state(
        stream,
        tokens,
        &slots,
        &first_positions,
        &cosine,
        &sine,
    )?;
    Ok((route, fixtures))
}

fn segmented_fixture(fixture: &Fixture, slot: usize) -> Fixture {
    let history = fixture
        .history
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            value ^ ((slot as u16 + 1).wrapping_mul(0x1111)).rotate_left((index & 3) as u32)
        })
        .collect();
    let state = fixture
        .state
        .iter()
        .enumerate()
        .map(|(index, &value)| value + (slot as f32 + 1.0) * ((index % 7) as f32 - 3.0) / 65_536.0)
        .collect();
    Fixture { history, state }
}

fn segmented_cache_pages(
    program: &ResidentModelProgram,
    stream: &CudaStream,
    batch: usize,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>, TargetMtpVerifyQualificationError> {
    (0..batch)
        .map(|slot| {
            let physical = usize::try_from(program.qualification_kv_physical_page(slot, 0)?)
                .map_err(|_| {
                    TargetMtpVerifyQualificationError::Mismatch(
                        "segmented cache page exceeds usize".into(),
                    )
                })?;
            Ok(program.qualification_cache_page(stream, physical)?)
        })
        .collect()
}

fn verify_segmented_rollback(
    batch: usize,
    fixtures: &[Fixture],
    observed: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let history = fixtures
        .iter()
        .flat_map(|fixture| fixture.history.iter().copied())
        .collect::<Vec<_>>();
    let state = fixtures
        .iter()
        .flat_map(|fixture| fixture.state.iter().copied())
        .collect::<Vec<_>>();
    compare_exact(
        &format!("segmented B={batch} provisional live history"),
        &observed.live_history,
        &history,
    )?;
    compare_f32_bits(
        &format!("segmented B={batch} provisional live state"),
        &observed.live_state,
        &state,
    )?;
    report.rollback_values += history.len() + state.len();
    Ok(())
}

fn verify_segmented_record_boundaries(
    batch: usize,
    tokens: usize,
    observed: &ResidentMtpVerifyObservables,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let projected_stride = batch * VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
    let control_stride = batch * VERIFY_ROUTES * Qwen38_27B::GDN_CONTROL_ROWS;
    for layer in 0..GDN_LAYERS {
        for lane in 0..batch {
            let projected = layer * projected_stride
                + (lane * VERIFY_ROUTES + tokens) * Qwen38_27B::GDN_INPUT_ROWS;
            let projected_end =
                layer * projected_stride + (lane + 1) * VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
            if observed.recorded_projected[projected..projected_end]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
                    "segmented B={batch} K={tokens} modified lane {lane} inactive projected records in GDN layer {layer}"
                )));
            }
            let control = layer * control_stride
                + (lane * VERIFY_ROUTES + tokens) * Qwen38_27B::GDN_CONTROL_ROWS;
            let control_end =
                layer * control_stride + (lane + 1) * VERIFY_ROUTES * Qwen38_27B::GDN_CONTROL_ROWS;
            if observed.recorded_log_decay[control..control_end]
                .iter()
                .chain(&observed.recorded_beta[control..control_end])
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
                    "segmented B={batch} K={tokens} modified lane {lane} inactive control records in GDN layer {layer}"
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_segmented_committed_history(
    batch: usize,
    route_tokens: usize,
    accepted: &[usize],
    fixtures: &[Fixture],
    verify: &ResidentMtpVerifyObservables,
    committed: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let history_per_layer = Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS;
    let projected_layer_stride = batch * VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
    let mut expected = Vec::with_capacity(committed.live_history.len());
    for lane in 0..batch {
        let mut lane_history = fixtures[lane].history.clone();
        for layer in 0..GDN_LAYERS {
            let history_layer = layer * history_per_layer;
            let projected_layer =
                layer * projected_layer_stride + lane * VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
            for token in 0..accepted[lane] {
                let projected_token = projected_layer + token * Qwen38_27B::GDN_INPUT_ROWS;
                for channel in 0..Qwen38_27B::GDN_QKV_ROWS {
                    let history = history_layer + channel * HISTORY_TAPS;
                    lane_history[history] = lane_history[history + 1];
                    lane_history[history + 1] = lane_history[history + 2];
                    lane_history[history + 2] =
                        verify.recorded_projected[projected_token + channel];
                }
            }
        }
        expected.extend(lane_history);
    }
    compare_exact(
        &format!("segmented B={batch} K={route_tokens} committed history"),
        &committed.live_history,
        &expected,
    )?;
    report.committed_values += expected.len();
    Ok(())
}

fn verify_first_gdn_k1_seams(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let route = prepare_verify(program, stream, 1, fixture)?;
    let target = program.qualification_first_gdn_mtp_seams(stream, route, true)?;

    prepare_persistent(program, stream, fixture)?;
    program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
    program.stage_embeddings(stream, &TOKEN_IDS[..1])?;
    program.load_slot_routes(stream, &[SLOT])?;
    let (cosine, sine) = rope(&[0]);
    let _decode_route = program.load_decode_state(stream, 1, &[0], &cosine, &sine)?;
    let decode = program.qualification_first_gdn_mtp_seams(stream, route, false)?;
    for (name, actual, expected) in [
        ("normalized", target.normalized, decode.normalized),
        ("projected", target.projected, decode.projected),
        ("convolved", target.convolved, decode.convolved),
        ("recurrent", target.recurrent, decode.recurrent),
        ("branch", target.branch, decode.branch),
        ("history", target.history, decode.history),
    ] {
        compare_exact(&format!("first GDN K=1 {name}"), &actual, &expected)?;
    }
    for (name, actual, expected) in [
        ("log_decay", target.log_decay, decode.log_decay),
        ("beta", target.beta, decode.beta),
        ("state", target.state, decode.state),
    ] {
        compare_f32_bits(&format!("first GDN K=1 {name}"), &actual, &expected)?;
    }
    Ok(())
}

fn verify_k1_layer_seams(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let route = prepare_verify(program, stream, 1, fixture)?;
    let target = program.qualification_mtp_k1_layer_seams(stream, route, true)?;

    prepare_persistent(program, stream, fixture)?;
    program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
    program.stage_embeddings(stream, &TOKEN_IDS[..1])?;
    program.load_slot_routes(stream, &[SLOT])?;
    let (cosine, sine) = rope(&[0]);
    let _decode_route = program.load_decode_state(stream, 1, &[0], &cosine, &sine)?;
    let decode = program.qualification_mtp_k1_layer_seams(stream, route, false)?;
    if target.len() != decode.len() {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "target visited {} layers, ordinary decode visited {}",
            target.len(),
            decode.len()
        )));
    }
    for (layer, (actual, expected)) in target.iter().zip(&decode).enumerate() {
        for (name, actual, expected) in [
            (
                "mixer branch",
                actual.mixer_branch.as_slice(),
                expected.mixer_branch.as_slice(),
            ),
            (
                "mixer residual",
                actual.mixer_residual.as_slice(),
                expected.mixer_residual.as_slice(),
            ),
            (
                "MLP normalized",
                actual.mlp_normalized.as_slice(),
                expected.mlp_normalized.as_slice(),
            ),
            (
                "MLP branch",
                actual.mlp_branch.as_slice(),
                expected.mlp_branch.as_slice(),
            ),
            (
                "residual",
                actual.residual.as_slice(),
                expected.residual.as_slice(),
            ),
            (
                "next normalized",
                actual.next_normalized.as_slice(),
                expected.next_normalized.as_slice(),
            ),
        ] {
            compare_exact(
                &format!("target/decode K=1 layer {layer} {name}"),
                actual,
                expected,
            )?;
        }
    }
    Ok(())
}

fn sequential_reference(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
    target_route: ResidentMtpVerifyRoute,
) -> Result<SequentialReference, TargetMtpVerifyQualificationError> {
    prepare_persistent(program, stream, fixture)?;
    program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
    let mut residual = Vec::with_capacity(tokens * Qwen38_27B::HIDDEN);
    let mut logits = Vec::with_capacity(tokens * Qwen38_27B::VOCAB);
    for (position, &token) in TOKEN_IDS.iter().take(tokens).enumerate() {
        program.stage_embeddings(stream, &[token])?;
        program.load_slot_routes(stream, &[SLOT])?;
        let (cosine, sine) = rope(&[position as u32]);
        let route = program.load_decode_state(stream, 1, &[position as u32], &cosine, &sine)?;
        program.launch_eager(stream, route)?;
        residual.extend(program.read_residual(stream, 1)?);
        logits.extend(program.read_logits(stream, 1)?);
    }
    let mut observed = program.qualification_target_mtp_observables(stream, target_route)?;
    observed.residual_a = residual;
    observed.logits = logits;
    Ok(SequentialReference {
        observed,
        cache: cache_page(program, stream)?,
    })
}

fn prepare_verify(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    tokens: usize,
    fixture: &Fixture,
) -> Result<ResidentMtpVerifyRoute, TargetMtpVerifyQualificationError> {
    prepare_persistent(program, stream, fixture)?;
    program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
    program.stage_embeddings(stream, &TOKEN_IDS[..tokens])?;
    let positions = (0..tokens as u32).collect::<Vec<_>>();
    let (cosine, sine) = rope(&positions);
    Ok(program.load_target_mtp_verify_state(stream, tokens, SLOT, 0, &cosine, &sine)?)
}

fn prepare_persistent(
    program: &ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<(), TargetMtpVerifyQualificationError> {
    program.reset_slot(stream, SLOT)?;
    program.qualification_load_target_mtp_gdn_slot(
        stream,
        SLOT,
        &fixture.history,
        &fixture.state,
    )?;
    Ok(())
}

fn fixture(program: &ResidentModelProgram) -> Fixture {
    let history_values = program.history_bytes() / MAX_BATCH / size_of::<u16>();
    let state_values = program.state_bytes() / MAX_BATCH / size_of::<f32>();
    let history = (0..history_values)
        .map(|index| ((index as u16).wrapping_mul(17) ^ 0x5aa5).rotate_left((index & 7) as u32))
        .collect();
    let state = (0..state_values)
        .map(|index| ((index.wrapping_mul(13) & 63) as f32 - 31.5) / 8_192.0)
        .collect();
    Fixture { history, state }
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

fn cache_page(
    program: &ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(Vec<u8>, Vec<u8>), TargetMtpVerifyQualificationError> {
    let physical =
        usize::try_from(program.qualification_kv_physical_page(SLOT, 0)?).map_err(|_| {
            TargetMtpVerifyQualificationError::Mismatch("cache page exceeds usize".into())
        })?;
    Ok(program.qualification_cache_page(stream, physical)?)
}

fn run_leaf_oracles() -> Result<(), TargetMtpVerifyQualificationError> {
    qualify_gdn_prepare().map_err(|error| {
        TargetMtpVerifyQualificationError::Mismatch(format!(
            "independent causal GDN prepare oracle failed: {error}"
        ))
    })?;
    qualify_gdn_recurrence().map_err(|error| {
        TargetMtpVerifyQualificationError::Mismatch(format!(
            "independent causal GDN recurrence oracle failed: {error}"
        ))
    })?;
    qualify_gdn_state_snapshot().map_err(|error| {
        TargetMtpVerifyQualificationError::Mismatch(format!(
            "independent GDN snapshot oracle failed: {error}"
        ))
    })?;
    Ok(())
}

fn verify_owner(program: &ResidentModelProgram) -> Result<(), TargetMtpVerifyQualificationError> {
    if program.workspace_bytes() != 923_695_108
        || program.resident_arena_bytes() != 21_258_945_792
        || program.kv_arena_bytes() != 7_210_118_656
        || program.arena_bytes() != 28_469_064_448
        || program.padding_bytes() != 15_676
        || program.target_mtp_graph_count() != 228
        || program.target_mtp_graph_executable_count() != 88
        || program.graph_route_count() != 290
        || program.graph_executable_count() != 150
    {
        return Err(TargetMtpVerifyQualificationError::Mismatch(
            "target verify owner accounting differs from the admitted layout".to_string(),
        ));
    }
    Ok(())
}

fn qualify_long_segmented_variants(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    const LENGTHS: [usize; 6] = [193, 1_025, 4_097, 16_385, 65_537, 131_073];
    program.reserve_kv_slot_tokens(stream, 0, LENGTHS[LENGTHS.len() - 1])?;
    let fixtures = [segmented_fixture(fixture, 0), segmented_fixture(fixture, 1)];
    for (slot, lane_fixture) in fixtures.iter().enumerate() {
        program.qualification_load_target_mtp_gdn_slot(
            stream,
            slot,
            &lane_fixture.history,
            &lane_fixture.state,
        )?;
    }
    let token_ids = [
        TOKEN_IDS[0],
        (TOKEN_IDS[0] + 8_191) % Qwen38_27B::VOCAB as u32,
    ];
    program.stage_target_mtp_segmented_embeddings(stream, &token_ids)?;
    let slots = [0, 1];
    let mut routes = Vec::with_capacity(LENGTHS.len());
    for length in LENGTHS {
        let first_positions = [length - 1, 0];
        let position = u32::try_from(length - 1).map_err(|_| {
            TargetMtpVerifyQualificationError::Mismatch(format!(
                "long-context graph position {length} exceeds u32"
            ))
        })?;
        let positions = [position, 0];
        let (cosine, sine) = rope(&positions);
        program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
        let route = program.load_target_mtp_segmented_verify_state(
            stream,
            1,
            &slots,
            &first_positions,
            &cosine,
            &sine,
        )?;
        program.launch_target_mtp_segmented_verify_eager(stream, route)?;
        let eager = program.qualification_target_mtp_segmented_observables(stream, route)?;

        program.qualification_reset_workspace(stream, BYTE_SENTINEL)?;
        let replay_route = program.load_target_mtp_segmented_verify_state(
            stream,
            1,
            &slots,
            &first_positions,
            &cosine,
            &sine,
        )?;
        program.replay_target_mtp_segmented_verify(stream, replay_route)?;
        let replay =
            program.qualification_target_mtp_segmented_observables(stream, replay_route)?;
        compare_observables(
            &format!("segmented long-context={length} graph variant"),
            &replay,
            &eager,
        )?;
        routes.push((replay_route, first_positions, cosine, sine));
        report.graph_replay_values += observable_values(&eager);
        report.long_segmented_variant_routes += 1;
    }

    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for (route, first_positions, cosine, sine) in routes.iter().rev() {
        let replay_route = program.load_target_mtp_segmented_verify_state(
            stream,
            1,
            &slots,
            first_positions,
            cosine,
            sine,
        )?;
        if replay_route != *route {
            return Err(TargetMtpVerifyQualificationError::Mismatch(
                "long-context graph variant route changed after warmup".to_string(),
            ));
        }
        program.replay_target_mtp_segmented_verify(stream, replay_route)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "device memory changed across long-context graph updates: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

fn verify_rollback(
    tokens: usize,
    fixture: &Fixture,
    observed: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    compare_exact(
        &format!("K={tokens} provisional live history"),
        &observed.live_history,
        &fixture.history,
    )?;
    compare_f32_bits(
        &format!("K={tokens} provisional live state"),
        &observed.live_state,
        &fixture.state,
    )?;
    report.rollback_values += observed.live_history.len() + observed.live_state.len();
    Ok(())
}

fn verify_record_boundaries(
    tokens: usize,
    observed: &ResidentMtpVerifyObservables,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let projected_stride = VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
    let control_stride = VERIFY_ROUTES * Qwen38_27B::GDN_CONTROL_ROWS;
    for layer in 0..GDN_LAYERS {
        let projected = layer * projected_stride + tokens * Qwen38_27B::GDN_INPUT_ROWS;
        if observed.recorded_projected[projected..(layer + 1) * projected_stride]
            .iter()
            .any(|&value| value != BF16_SENTINEL)
        {
            return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
                "K={tokens} modified inactive projected replay rows in GDN layer {layer}"
            )));
        }
        let control = layer * control_stride + tokens * Qwen38_27B::GDN_CONTROL_ROWS;
        if observed.recorded_log_decay[control..(layer + 1) * control_stride]
            .iter()
            .chain(&observed.recorded_beta[control..(layer + 1) * control_stride])
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        {
            return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
                "K={tokens} modified inactive control replay rows in GDN layer {layer}"
            )));
        }
    }
    Ok(())
}

fn verify_cache_boundaries(
    tokens: usize,
    cache: &(Vec<u8>, Vec<u8>),
) -> Result<(), TargetMtpVerifyQualificationError> {
    let page_values = Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let expected = ATTENTION_LAYERS * page_values;
    if cache.0.len() != expected || cache.1.len() != expected {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "K={tokens} cache page has {}/{} key/value bytes, expected {expected}",
            cache.0.len(),
            cache.1.len()
        )));
    }
    for (role, values) in [("key", &cache.0), ("value", &cache.1)] {
        for layer in 0..ATTENTION_LAYERS {
            let layer_base = layer * page_values;
            for head in 0..Qwen38_27B::NUM_KV_HEADS {
                let head_base = layer_base + head * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
                let inactive = head_base + tokens * Qwen38_27B::HEAD_DIM;
                let end = head_base + ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
                if let Some(relative) = values[inactive..end].iter().position(|&value| value != 0) {
                    return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
                        "K={tokens} modified inactive {role} cache byte at layer {layer}, head {head}, offset {}",
                        inactive + relative - layer_base
                    )));
                }
            }
        }
    }
    Ok(())
}

fn compare_last_provisional(
    verify: &ResidentMtpVerifyObservables,
    sequential: &ResidentMtpVerifyObservables,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let history = verify.provisional_history.len();
    let state = verify.provisional_state.len();
    compare_exact(
        "final provisional history",
        &verify.provisional_history,
        &sequential.live_history[sequential.live_history.len() - history..],
    )?;
    compare_f32_bits(
        "final provisional state",
        &verify.provisional_state,
        &sequential.live_state[sequential.live_state.len() - state..],
    )
}

fn verify_committed_history(
    accepted: usize,
    fixture: &Fixture,
    verify: &ResidentMtpVerifyObservables,
    committed: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let history_per_layer = Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS;
    if fixture.history.len() != GDN_LAYERS * history_per_layer
        || committed.live_history.len() != fixture.history.len()
    {
        return Err(TargetMtpVerifyQualificationError::Mismatch(
            "committed history inventory does not match 48 selected GDN rows".to_string(),
        ));
    }
    let projected_stride = VERIFY_ROUTES * Qwen38_27B::GDN_INPUT_ROWS;
    if verify.recorded_projected.len() != GDN_LAYERS * projected_stride {
        return Err(TargetMtpVerifyQualificationError::Mismatch(
            "recorded projection inventory does not match 48 K=1..4 GDN planes".to_string(),
        ));
    }

    let mut expected = fixture.history.clone();
    for layer in 0..GDN_LAYERS {
        let history_layer = layer * history_per_layer;
        let projected_layer = layer * projected_stride;
        for token in 0..accepted {
            let projected_token = projected_layer + token * Qwen38_27B::GDN_INPUT_ROWS;
            for channel in 0..Qwen38_27B::GDN_QKV_ROWS {
                let history = history_layer + channel * HISTORY_TAPS;
                expected[history] = expected[history + 1];
                expected[history + 1] = expected[history + 2];
                expected[history + 2] = verify.recorded_projected[projected_token + channel];
            }
        }
    }
    compare_exact(
        &format!("accepted K={accepted} committed history"),
        &committed.live_history,
        &expected,
    )?;
    report.committed_values += committed.live_history.len();
    Ok(())
}

fn compare_last_committed(
    verify: &ResidentMtpVerifyObservables,
    committed: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let history = verify.provisional_history.len();
    let state = verify.provisional_state.len();
    compare_exact(
        "fully accepted final GDN history",
        &committed.live_history[committed.live_history.len() - history..],
        &verify.provisional_history,
    )?;
    compare_f32_bits(
        "fully accepted final GDN state",
        &committed.live_state[committed.live_state.len() - state..],
        &verify.provisional_state,
    )?;
    report.committed_values += history + state;
    Ok(())
}

fn compare_committed(
    actual: &ResidentMtpVerifyObservables,
    expected: &ResidentMtpVerifyObservables,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    compare_exact(
        "committed live history",
        &actual.live_history,
        &expected.live_history,
    )?;
    compare_f32_bits(
        "committed live state",
        &actual.live_state,
        &expected.live_state,
    )?;
    report.committed_values += actual.live_history.len() + actual.live_state.len();
    Ok(())
}

fn compare_observables(
    role: &str,
    actual: &ResidentMtpVerifyObservables,
    expected: &ResidentMtpVerifyObservables,
) -> Result<(), TargetMtpVerifyQualificationError> {
    for (name, actual, expected) in [
        (
            "residual",
            actual.residual_a.as_slice(),
            expected.residual_a.as_slice(),
        ),
        (
            "final_normalized",
            actual.final_normalized.as_slice(),
            expected.final_normalized.as_slice(),
        ),
        (
            "logits",
            actual.logits.as_slice(),
            expected.logits.as_slice(),
        ),
        (
            "provisional_history",
            actual.provisional_history.as_slice(),
            expected.provisional_history.as_slice(),
        ),
        (
            "recorded_projected",
            actual.recorded_projected.as_slice(),
            expected.recorded_projected.as_slice(),
        ),
        (
            "live_history",
            actual.live_history.as_slice(),
            expected.live_history.as_slice(),
        ),
    ] {
        compare_exact(&format!("{role} {name}"), actual, expected)?;
    }
    for (name, actual, expected) in [
        (
            "provisional_state",
            actual.provisional_state.as_slice(),
            expected.provisional_state.as_slice(),
        ),
        (
            "recorded_log_decay",
            actual.recorded_log_decay.as_slice(),
            expected.recorded_log_decay.as_slice(),
        ),
        (
            "recorded_beta",
            actual.recorded_beta.as_slice(),
            expected.recorded_beta.as_slice(),
        ),
        (
            "live_state",
            actual.live_state.as_slice(),
            expected.live_state.as_slice(),
        ),
    ] {
        compare_f32_bits(&format!("{role} {name}"), actual, expected)?;
    }
    Ok(())
}

fn compare_exact<T: Eq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), TargetMtpVerifyQualificationError> {
    if actual.len() != expected.len() {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "{role} length {} differs from {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), TargetMtpVerifyQualificationError> {
    if actual.len() != expected.len() {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "{role} length {} differs from {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn verify_endpoint_oracle(
    tokens: usize,
    observed: &ResidentMtpVerifyObservables,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    for token in 0..tokens {
        let begin = token * Qwen38_27B::HIDDEN;
        let end = begin + Qwen38_27B::HIDDEN;
        let normalized =
            rms_norm_oracle::<Qwen38_27B>(&observed.residual_a[begin..end], final_norm);
        for (index, (&actual, &expected)) in observed.final_normalized[begin..end]
            .iter()
            .zip(&normalized)
            .enumerate()
        {
            require_endpoint_close(
                "final RMSNorm",
                token * Qwen38_27B::HIDDEN + index,
                bf16_to_f32(actual),
                f64::from(bf16_to_f32(expected)),
                report,
            )?;
        }
        let quantized =
            quantize_oracle(&normalized).map_err(TargetMtpVerifyQualificationError::Mismatch)?;
        for &row in &SELECTED_LOGIT_ROWS {
            let weight_begin = row * Qwen38_27B::HIDDEN;
            let expected = fp8_dot(
                &quantized.codes,
                quantized.scale,
                &lm_head_codes[weight_begin..weight_begin + Qwen38_27B::HIDDEN],
                lm_head_scales[row],
            )?;
            require_endpoint_close(
                "selected LM-head row",
                token * SELECTED_LOGIT_ROWS.len() + row,
                bf16_to_f32(observed.logits[token * Qwen38_27B::VOCAB + row]),
                expected,
                report,
            )?;
        }
    }
    report.endpoint_oracle_values += tokens * (Qwen38_27B::HIDDEN + SELECTED_LOGIT_ROWS.len());
    Ok(())
}

fn fp8_dot(
    activations: &[u8],
    activation_scale: f32,
    weights: &[u8],
    weight_scale: u16,
) -> Result<f64, TargetMtpVerifyQualificationError> {
    let dot = activations
        .iter()
        .zip(weights)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            Ok::<_, String>(
                sum + f64::from(decode_e4m3fn(activation)?) * f64::from(decode_e4m3fn(weight)?),
            )
        })
        .map_err(TargetMtpVerifyQualificationError::Mismatch)?;
    Ok(dot * f64::from(activation_scale) * f64::from(bf16_to_f32(weight_scale)))
}

fn require_endpoint_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    report: &mut TargetMtpVerifyQualification,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    let tolerance = 0.5f32.max(expected.abs() as f32 * 0.03);
    if !actual.is_finite() || error > tolerance {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn observable_values(observed: &ResidentMtpVerifyObservables) -> usize {
    observed.residual_a.len()
        + observed.final_normalized.len()
        + observed.logits.len()
        + observed.provisional_history.len()
        + observed.provisional_state.len()
        + observed.recorded_projected.len()
        + observed.recorded_log_decay.len()
        + observed.recorded_beta.len()
        + observed.live_history.len()
        + observed.live_state.len()
}

fn verify_stable(
    program: &ResidentModelProgram,
    base: (u64, u64),
    addresses: &[usize],
    tokens: usize,
) -> Result<(), TargetMtpVerifyQualificationError> {
    if (program.base_address(), program.kv_base_address()) != base
        || program.qualification_addresses() != addresses
    {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "target owner addresses changed after K={tokens}"
        )));
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    fixture: &Fixture,
) -> Result<(), TargetMtpVerifyQualificationError> {
    let mut routes = Vec::with_capacity(VERIFY_ROUTES);
    for tokens in 1..=VERIFY_ROUTES {
        let route = prepare_verify(program, stream, tokens, fixture)?;
        program.replay_target_mtp_verify(stream, route)?;
        program.replay_target_mtp_commit(stream, route, tokens)?;
        routes.push(route);
    }
    for batch in 1..=MAX_BATCH {
        for tokens in 1..=VERIFY_ROUTES {
            let (route, _) = prepare_segmented(program, stream, batch, tokens, fixture)?;
            let accepted = (0..batch).map(|lane| lane % tokens + 1).collect::<Vec<_>>();
            program.replay_target_mtp_segmented_verify(stream, route)?;
            program.commit_target_mtp_segmented(stream, route, &accepted)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for route in routes.iter().rev() {
            program.replay_target_mtp_verify(stream, *route)?;
            program.replay_target_mtp_commit(stream, *route, route.tokens())?;
        }
        for batch in (1..=MAX_BATCH).rev() {
            for tokens in (1..=VERIFY_ROUTES).rev() {
                let (route, _) = prepare_segmented(program, stream, batch, tokens, fixture)?;
                let accepted = (0..batch).map(|lane| lane % tokens + 1).collect::<Vec<_>>();
                program.replay_target_mtp_segmented_verify(stream, route)?;
                program.commit_target_mtp_segmented(stream, route, &accepted)?;
            }
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(TargetMtpVerifyQualificationError::Mismatch(format!(
            "device memory changed after target warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TargetMtpVerifyQualificationError, qualify_target_mtp_verify};
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the admitted snapshot and an exclusive RTX 5090"]
    fn exact_target_verify_and_commit_match_source_oracles()
    -> Result<(), TargetMtpVerifyQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .map(PathBuf::from)
            .ok_or_else(|| {
                TargetMtpVerifyQualificationError::Mismatch(
                    "TUISKO_SNAPSHOT is required for the source-backed target MTP gate".into(),
                )
            })?;
        let report = qualify_target_mtp_verify(&root)?;
        assert_eq!(report.leaf_oracle_suites, 3);
        assert_eq!(report.verify_routes, 4);
        assert_eq!(report.commit_routes, 10);
        assert_eq!(report.segmented_verify_routes, 32);
        assert_eq!(report.segmented_commit_routes, 32);
        assert_eq!(report.endpoint_oracle_values, 1_896_250);
        assert_eq!(report.graph_count, 228);
        assert_eq!(report.graph_executable_count, 88);
        assert_eq!(report.long_segmented_variant_routes, 6);
        assert_eq!(report.workspace_bytes, 923_695_108);
        assert_eq!(report.arena_bytes, 28_469_064_448);
        assert_eq!(report.padding_bytes, 15_676);
        Ok(())
    }
}
