//! Direct diagnostic timing for one source-backed Qwen3.8-Flash-Next QSA/MoE layer.
//!
//! Every row width is measured on both attention routes with production-shaped metadata.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    MAX_BATCH, Qwen38FlashNextQsaMoeLayerProgram, Qwen38FlashNextQsaRound,
    Qwen38FlashNextQsaRoundStageGraph, Qwen38FlashNextQsaRoute,
    qwen38_flash_next_qsa_block_rotary_rows,
};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_kernels_sm120::{
    SELECTION_RADIX_PASSES, SELECTION_ROW_TILE, selection_block_bucket, selection_ctas_per_row,
};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const WIDTH: usize = A::HC_WIDTH;
const HIDDEN: usize = <A as Arch>::HIDDEN;
const RANK: usize = A::HC_LOWRANK;
const BRANCHES: usize = A::HC_COUNT;
const QKV_ROWS: usize = A::ATTENTION_QKV_ROWS;
const OUTPUT_COLUMNS: usize = A::ATTENTION_OUTPUT_COLUMNS;
const HEAD_DIM: usize = <A as Arch>::HEAD_DIM;
const KV_HEADS: usize = <A as Arch>::NUM_KV_HEADS;
const ROTARY_ELEMENTS: usize = 32;
const INTERMEDIATE: usize = <A as Arch>::INTERMEDIATE;
const EXPERTS: usize = A::NUM_EXPERTS;
const TOP_K: usize = A::NUM_EXPERTS_PER_TOKEN;
const EXPERT_SLOT_BYTES: usize = 2_764_800;
const SOURCE_LAYER: usize = 3;

struct RouteGraph {
    rows: usize,
    route: Qwen38FlashNextQsaRoute,
    context: usize,
    preparation: Qwen38FlashNextQsaRoundStageGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: Qwen38FlashNextQsaMoeLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(
        root: &Path,
        layer: usize,
        repeated_operations: u64,
    ) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = Qwen38FlashNextQsaMoeLayerProgram::from_snapshot(&context, snapshot, layer)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
        program.reset_cache(&stream)?;
        let mut routes = Vec::with_capacity(2 * EXACT_ROUTES.len());
        for route in [
            Qwen38FlashNextQsaRoute::Dense,
            Qwen38FlashNextQsaRoute::Selected,
        ] {
            for rows in EXACT_ROUTES {
                let (table_rows, cache_positions, lengths) = round_metadata(rows, route);
                let rope_cos = vec![1.0; rows * ROTARY_ELEMENTS];
                let rope_sin = vec![0.0; rows * ROTARY_ELEMENTS];
                let block_rotary_rows =
                    qwen38_flash_next_qsa_block_rotary_rows(rows) * ROTARY_ELEMENTS;
                let block_rope_cos = vec![1.0; block_rotary_rows];
                let block_rope_sin = vec![0.0; block_rotary_rows];
                let preparation = program.qualification_round_stage_graph(
                    &stream,
                    rows,
                    Qwen38FlashNextQsaRound {
                        table_rows: &table_rows,
                        cache_positions: &cache_positions,
                        lengths: &lengths,
                        rope_cos: &rope_cos,
                        rope_sin: &rope_sin,
                        block_rope_cos: &block_rope_cos,
                        block_rope_sin: &block_rope_sin,
                    },
                )?;
                if preparation.route() != route {
                    return Err(DeviceBenchmarkError::Precondition(format!(
                        "QSA benchmark staged {:?}, expected {route:?}",
                        preparation.route()
                    )));
                }
                routes.push(RouteGraph {
                    rows,
                    route,
                    context: lengths.iter().copied().max().unwrap_or(0) as usize,
                    preparation,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        rows,
                        route,
                        repeated_operations,
                    )?,
                });
            }
        }

        Ok(Self {
            routes,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the session retains the stage sources and destination program.
                unsafe { route.preparation.graph().launch(&self.stream) }?;
                // SAFETY: the program retains every captured layer allocation through this replay.
                unsafe {
                    self.program
                        .qualification_graph(route.rows, route.route)?
                        .launch(&self.stream)
                }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    fn cases(
        &self,
        repeated_operations: u64,
    ) -> Result<Vec<ExactDeviceCase<'_>>, DeviceBenchmarkError> {
        self.routes
            .iter()
            .map(|route| {
                let (width, mut workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_attention_layer_decode(
                            route.rows as u32,
                            route.context as u64,
                        ),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_attention_layer_prefill(route.rows as u64),
                    )
                };
                workload.context_tokens = Some(route.context as u64);
                Ok(ExactDeviceCase::new(
                    route_name(route.route),
                    format!("{width} ctx={}", route.context),
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.rows, route.route),
                        route.rows as u64,
                        "token",
                    ),
                    self.program.qualification_graph(route.rows, route.route)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
                .with_preparation(route.preparation.graph()))
            })
            .collect()
    }
}

fn benchmark_input() -> Vec<u16> {
    const PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
    (0..MAX_ROWS * WIDTH)
        .map(|index| f32_to_bf16(PATTERN[(index + index / HIDDEN) & 7]))
        .collect()
}

fn route_name(route: Qwen38FlashNextQsaRoute) -> &'static str {
    match route {
        Qwen38FlashNextQsaRoute::Dense => "qwen38_flash_next/qsa_moe/dense_layer",
        Qwen38FlashNextQsaRoute::Selected => "qwen38_flash_next/qsa_moe/selected_layer",
    }
}

fn round_metadata(rows: usize, route: Qwen38FlashNextQsaRoute) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let base = match (route, rows <= MAX_BATCH) {
        (Qwen38FlashNextQsaRoute::Dense, _) => 0,
        (Qwen38FlashNextQsaRoute::Selected, true) => 2_052,
        (Qwen38FlashNextQsaRoute::Selected, false) => 2_048,
    };
    let (table_rows, cache_positions) = if rows <= MAX_BATCH {
        ((0..rows as u32).collect(), vec![base; rows])
    } else {
        (
            vec![0; rows],
            (base..base + rows as u32).collect::<Vec<_>>(),
        )
    };
    let lengths = cache_positions
        .iter()
        .map(|&position| position + 1)
        .collect();

    (table_rows, cache_positions, lengths)
}

fn selection_partial_bytes(rows: usize, blocks: usize) -> usize {
    let bucket = selection_block_bucket(blocks).unwrap_or(blocks);
    let ctas = selection_ctas_per_row(rows, bucket);

    2 * SELECTION_RADIX_PASSES * ctas * 256 * size_of::<u32>()
}

fn selected_positions(length: usize) -> usize {
    let ratio = A::INDEXER_COMPRESS_RATIO;
    let budget_blocks = A::INDEXER_BUDGET / ratio;

    (length / ratio).min(budget_blocks) * ratio + length % ratio
}

/// Bytes one production launch moves at the staged route and context.
fn logical_bytes(rows: usize, route: Qwen38FlashNextQsaRoute) -> usize {
    let word = size_of::<u16>();
    let (_, cache_positions, lengths) = round_metadata(rows, route);
    let visible_keys = lengths.iter().map(|&length| length as usize).sum::<usize>();
    let candidate_blocks = lengths
        .iter()
        .map(|&length| length as usize / A::INDEXER_COMPRESS_RATIO)
        .sum::<usize>();
    let closed_blocks = cache_positions
        .iter()
        .filter(|&&position| (position as usize + 1).is_multiple_of(A::INDEXER_COMPRESS_RATIO))
        .count();

    let bracket_weights = WIDTH + 2 * RANK * WIDTH + BRANCHES * WIDTH;
    let bracket = 2
        * (bracket_weights * word
            + rows * (2 * WIDTH + RANK + HIDDEN + BRANCHES) * word
            + rows * 2 * WIDTH * word);

    let qkv_projection = QKV_ROWS * HIDDEN * word + rows * (HIDDEN + QKV_ROWS) * word;
    let indexer_projection =
        A::INDEXER_ROWS * HIDDEN * word + rows * (HIDDEN + A::INDEXER_ROWS) * word;
    let indexer_prepare = A::INDEXER_HEAD_DIM * word
        + rows
            * (A::INDEXER_ROWS * word
                + A::INDEXER_HEADS * A::INDEXER_HEAD_DIM * size_of::<f32>()
                + A::INDEXER_HEAD_DIM * word
                + 2 * ROTARY_ELEMENTS * size_of::<f32>()
                + 2 * size_of::<u32>());
    let compression_rows = if rows <= MAX_BATCH { rows } else { 1 };
    let indexer_compress = A::INDEXER_HEAD_DIM * word
        + closed_blocks
            * ((A::INDEXER_COMPRESS_RATIO + 1) * A::INDEXER_HEAD_DIM * word
                + 2 * ROTARY_ELEMENTS * size_of::<f32>())
        + compression_rows * 3 * size_of::<u32>();
    // Q/K normalization, MRoPE, and the represented E4M3 append of one key and one value.
    let prepare = 2 * HEAD_DIM * word
        + rows
            * (QKV_ROWS * word
                + 2 * ROTARY_ELEMENTS * size_of::<f32>()
                + OUTPUT_COLUMNS * size_of::<f32>()
                + 2 * KV_HEADS * HEAD_DIM
                + 2 * size_of::<u32>());
    let attention = match route {
        Qwen38FlashNextQsaRoute::Dense => {
            rows * (2 * OUTPUT_COLUMNS * size_of::<f32>() + 2 * size_of::<u32>())
                + visible_keys * 2 * KV_HEADS * HEAD_DIM
        }
        Qwen38FlashNextQsaRoute::Selected => {
            let selected = lengths
                .iter()
                .map(|&length| selected_positions(length as usize))
                .sum::<usize>();
            let score = candidate_blocks * (A::INDEXER_HEAD_DIM * word + size_of::<f32>())
                + rows * A::INDEXER_HEADS * A::INDEXER_HEAD_DIM * size_of::<f32>();
            let tile_rows = if rows <= MAX_BATCH {
                rows
            } else {
                SELECTION_ROW_TILE
            };
            let select = (SELECTION_RADIX_PASSES + 1) * candidate_blocks * size_of::<f32>()
                + rows
                    * selection_partial_bytes(
                        tile_rows,
                        A::MAX_POSITION_EMBEDDINGS / A::INDEXER_COMPRESS_RATIO,
                    )
                + selected * size_of::<u32>();
            let gathered = rows * 2 * OUTPUT_COLUMNS * size_of::<f32>()
                + selected * (2 * <A as Arch>::NUM_ATTENTION_HEADS * HEAD_DIM + size_of::<u32>());

            score + select + gathered
        }
    };
    let gate = rows * (OUTPUT_COLUMNS * size_of::<f32>() + QKV_ROWS * word + OUTPUT_COLUMNS * word);
    let output_projection =
        HIDDEN * OUTPUT_COLUMNS * word + rows * (OUTPUT_COLUMNS + HIDDEN) * word;

    let router = EXPERTS * HIDDEN * word + rows * (HIDDEN + EXPERTS + 2 * TOP_K) * word;
    let routed_weights = rows * TOP_K * EXPERT_SLOT_BYTES;
    let shared_weights = (3 * INTERMEDIATE * HIDDEN + HIDDEN) * word;
    let experts_path = routed_weights
        + shared_weights
        + rows
            * (HIDDEN * word
                + 2 * TOP_K * word
                + TOP_K * INTERMEDIATE * word
                + TOP_K * HIDDEN * word
                + (INTERMEDIATE + HIDDEN + 1) * word
                + HIDDEN * word);

    bracket
        + qkv_projection
        + indexer_projection
        + indexer_prepare
        + indexer_compress
        + prepare
        + attention
        + gate
        + output_projection
        + router
        + experts_path
}

/// Measures every exact graph of one source-backed Qwen3.8-Flash-Next QSA/MoE layer.
pub fn benchmark_qwen38_flash_next_qsa_moe_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, SOURCE_LAYER, options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "qwen38_flash_next/qsa_moe/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "one layer's backbone parameters, indexer planes included, expert pool excluded",
    )?;
    memory.register_owned(
        "qwen38_flash_next/qsa_moe/routed_expert_pool",
        BenchmarkMemoryKind::Weights,
        session.program.pool_arena_bytes(),
        "512 sealed expert slots and their indirection table",
    )?;
    memory.register_owned(
        "qwen38_flash_next/qsa_moe/paged_cache",
        BenchmarkMemoryKind::KvCache,
        session.program.cache_bytes(),
        "512 pages of represented E4M3 K/V and compressed block keys plus the raw-key ring",
    )?;
    memory.register_owned(
        "qwen38_flash_next/qsa_moe/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 working planes, selection scratch, rotary tables, and metadata",
    )?;
    memory.register_owned(
        "qwen38_flash_next/qsa_moe/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.cache_bytes()
            - session.program.workspace_bytes(),
        "QSA arena 256-byte alignment padding",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen38-flash-next-qsa-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust submission/completion, repeated production graph, and repeated-operation graph",
        },
        preflight,
        baseline_sha256,
        options,
        metrics,
        energy_metrics,
        telemetry,
        memory,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, MAX_BATCH, MAX_ROWS, SOURCE_LAYER, logical_bytes, round_metadata, route_name,
        selected_positions,
    };
    use tuisko_engine::Qwen38FlashNextQsaRoute;

    #[test]
    fn the_source_layer_is_a_sparse_attention_one() {
        assert_eq!(SOURCE_LAYER, 3);
        assert!((SOURCE_LAYER + 1).is_multiple_of(4));
    }

    #[test]
    fn accounting_grows_with_every_admitted_route() {
        for route in [
            Qwen38FlashNextQsaRoute::Dense,
            Qwen38FlashNextQsaRoute::Selected,
        ] {
            let mut previous = 0;
            for rows in EXACT_ROUTES {
                let bytes = logical_bytes(rows, route);
                assert!(
                    bytes > previous,
                    "{route:?} logical bytes did not grow at rows={rows}"
                );
                previous = bytes;
            }
        }
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert!(
            logical_bytes(MAX_ROWS, Qwen38FlashNextQsaRoute::Selected)
                > logical_bytes(MAX_ROWS, Qwen38FlashNextQsaRoute::Dense)
        );
    }

    #[test]
    fn every_route_has_production_metadata() {
        for route in [
            Qwen38FlashNextQsaRoute::Dense,
            Qwen38FlashNextQsaRoute::Selected,
        ] {
            for rows in EXACT_ROUTES {
                let (table_rows, cache_positions, lengths) = round_metadata(rows, route);
                assert_eq!(table_rows.len(), rows);
                assert_eq!(cache_positions.len(), rows);
                assert_eq!(lengths.len(), rows);
                assert!(
                    cache_positions
                        .iter()
                        .zip(&lengths)
                        .all(|(&position, &length)| length == position + 1)
                );
                if rows <= MAX_BATCH {
                    assert_eq!(table_rows, (0..rows as u32).collect::<Vec<_>>());
                    assert!(cache_positions.iter().all(|&position| {
                        position
                            == if route == Qwen38FlashNextQsaRoute::Dense {
                                0
                            } else {
                                2_052
                            }
                    }));
                } else {
                    assert_eq!(table_rows, vec![0; rows]);
                    assert!(
                        cache_positions
                            .windows(2)
                            .all(|pair| pair[1] == pair[0] + 1)
                    );
                    let expected_base = if route == Qwen38FlashNextQsaRoute::Dense {
                        0
                    } else {
                        2_048
                    };
                    assert_eq!(cache_positions[0], expected_base);
                }
            }
        }
    }

    #[test]
    fn selected_accounting_keeps_the_exact_tail() {
        assert_eq!(selected_positions(2_051), 2_051);
        assert_eq!(selected_positions(2_052), 2_048);
        assert_eq!(selected_positions(2_053), 2_049);
        assert_eq!(
            route_name(Qwen38FlashNextQsaRoute::Dense),
            "qwen38_flash_next/qsa_moe/dense_layer"
        );
        assert_eq!(
            route_name(Qwen38FlashNextQsaRoute::Selected),
            "qwen38_flash_next/qsa_moe/selected_layer"
        );
    }
}
