//! Direct diagnostic timing for one source-backed Qwen3.8-Flash-Next GDN/MoE layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen38FlashNextGdnMoeLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

type A = Qwen38FlashNext;

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const WIDTH: usize = A::HC_WIDTH;
const HIDDEN: usize = <A as Arch>::HIDDEN;
const RANK: usize = A::HC_LOWRANK;
const BRANCHES: usize = A::HC_COUNT;
const GDN_INPUT_ROWS: usize = A::GDN_INPUT_ROWS;
const GDN_QKV_ROWS: usize = A::GDN_QKV_ROWS;
const GDN_VALUE_ROWS: usize = A::GDN_VALUE_ROWS;
const CONTROLS: usize = A::GDN_CONTROL_ROWS;
const HISTORY_TAPS: usize = <A as Arch>::LINEAR_CONV_KERNEL_DIM - 1;
const STATE_PER_ROW: usize = CONTROLS * <A as Arch>::LINEAR_HEAD_DIM * <A as Arch>::LINEAR_HEAD_DIM;
const INTERMEDIATE: usize = <A as Arch>::INTERMEDIATE;
const EXPERTS: usize = A::NUM_EXPERTS;
const TOP_K: usize = A::NUM_EXPERTS_PER_TOKEN;
const EXPERT_SLOT_BYTES: usize = 2_764_800;

struct RouteGraph {
    rows: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: Qwen38FlashNextGdnMoeLayerProgram,
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
        let program = Qwen38FlashNextGdnMoeLayerProgram::from_snapshot(&context, snapshot, layer)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
        if program.layout().carries_ple_state() {
            program.load_engram_codes(&stream, MAX_ROWS, &benchmark_engram_codes())?;
        }
        program.reset_state(&stream)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| {
                Ok(RouteGraph {
                    rows,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        rows,
                        repeated_operations,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;

        Ok(Self {
            routes,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for rows in EXACT_ROUTES {
                // SAFETY: the program retains every captured layer allocation through this replay.
                unsafe { self.program.qualification_graph(rows)?.launch(&self.stream) }?;
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
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_layer_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_layer_prefill(route.rows as u64),
                    )
                };
                Ok(ExactDeviceCase::new(
                    "qwen38_flash_next/gdn_moe/layer",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    self.program.qualification_graph(route.rows)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
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

fn benchmark_engram_codes() -> Vec<u8> {
    (0..MAX_ROWS * A::NGRAM_HEADS * A::NGRAM_HEAD_DIM)
        .map(|index| ((index.wrapping_mul(37) % 120) as u8).wrapping_add(8))
        .collect()
}

/// Bytes one launch of this layer must move, counting every plane once.
///
/// The routed-expert term is the honest one and it dominates: ten experts per token, each a full
/// 2,764,800-byte slot read out of the pool. At `B=1` that alone is 27.6 MB, which is why a
/// composed decode number can never look like the sum of its kernels' micro-benchmarks.
fn logical_bytes(rows: usize) -> usize {
    let word = size_of::<u16>();
    let state_row_count = if rows <= MAX_BATCH { rows } else { 1 };

    // Two gated-residual brackets: hc_norm, both low-rank projections, the inject plane, and
    // the write-back's read-modify-write of the stream.
    let bracket_weights = WIDTH + 2 * RANK * WIDTH + BRANCHES * WIDTH;
    let bracket = 2
        * (bracket_weights * word
            + rows * (2 * WIDTH + RANK + HIDDEN + BRANCHES) * word
            + rows * 2 * WIDTH * word);

    let input_projection = GDN_INPUT_ROWS * HIDDEN * word + rows * (HIDDEN + GDN_INPUT_ROWS) * word;
    let controls = 2 * CONTROLS * HIDDEN * word;
    let prepare = controls
        + rows
            * (HIDDEN * word
                + GDN_INPUT_ROWS * word
                + GDN_QKV_ROWS * <A as Arch>::LINEAR_CONV_KERNEL_DIM * word
                + 2 * GDN_QKV_ROWS * HISTORY_TAPS * word
                + GDN_QKV_ROWS * word
                + 2 * CONTROLS * size_of::<f32>())
        + state_row_count * size_of::<u32>();
    let recurrence = <A as Arch>::LINEAR_HEAD_DIM * word
        + rows
            * (GDN_QKV_ROWS * word
                + GDN_INPUT_ROWS * word
                + 2 * CONTROLS * size_of::<f32>()
                + 2 * STATE_PER_ROW * size_of::<f32>()
                + GDN_VALUE_ROWS * word)
        + state_row_count * size_of::<u32>();
    let output_projection =
        HIDDEN * GDN_VALUE_ROWS * word + rows * (GDN_VALUE_ROWS + HIDDEN) * word;

    let router = EXPERTS * HIDDEN * word + rows * (HIDDEN + EXPERTS + 2 * TOP_K) * word;
    // Every selected expert's whole slot, plus the shared expert's three planes.
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

    bracket + input_projection + prepare + recurrence + output_projection + router + experts_path
}

/// Measures every exact graph of one source-backed Qwen3.8-Flash-Next GDN/MoE layer.
pub fn benchmark_qwen38_flash_next_gdn_moe_layer(
    root: &Path,
    layer: usize,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, layer, options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "qwen38_flash_next/gdn_moe/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "one layer's backbone parameters, excluding the routed expert pool",
    )?;
    memory.register_owned(
        "qwen38_flash_next/gdn_moe/routed_expert_pool",
        BenchmarkMemoryKind::Weights,
        session.program.pool_arena_bytes(),
        "512 sealed expert slots and the indirection table they resolve through",
    )?;
    memory.register_owned(
        "qwen38_flash_next/gdn_moe/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 working planes plus eight persistent recurrent and convolution slots",
    )?;
    memory.register_owned(
        "qwen38_flash_next/gdn_moe/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.workspace_bytes(),
        "two 256-byte-aligned arenas",
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
            // Layer 1 runs the engram module and no other GDN layer does, so its numbers are
            // not comparable to the other thirty-five and must not be filed under one name.
            suite: if layer == A::PLE_LAYER {
                "bench-qwen38-flash-next-ple-layer"
            } else {
                "bench-qwen38-flash-next-gdn-layer"
            },
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
    use super::{EXACT_ROUTES, EXPERT_SLOT_BYTES, MAX_BATCH, MAX_ROWS, TOP_K, logical_bytes};

    #[test]
    fn the_routed_pool_overtakes_the_backbone_at_six_tokens() {
        // Ten whole expert slots per token is 27.6 MB, which grows with the batch, against a
        // fixed ~155 MB of backbone weights that does not. So a B=1 launch is dominated by
        // weights every token pays for anyway, and the routed pool only becomes the larger
        // term from six tokens on -- which is why a decode number and a prefill number on this
        // layer are bound by different things and must not be read as one curve.
        let routed = |rows: usize| rows * TOP_K * EXPERT_SLOT_BYTES;
        let backbone = logical_bytes(1) - routed(1);

        assert_eq!(routed(1), 27_648_000);
        assert!(backbone > 150_000_000 && backbone < 165_000_000);
        assert!(
            routed(5) < backbone,
            "the pool should still be the smaller term at B=5"
        );
        assert!(
            routed(6) > backbone,
            "the pool should overtake the backbone at B=6"
        );
    }

    #[test]
    fn accounting_grows_with_every_admitted_route() {
        let mut previous = 0;
        for rows in EXACT_ROUTES {
            let bytes = logical_bytes(rows);
            assert!(
                bytes > previous,
                "logical bytes did not grow at rows={rows}"
            );
            previous = bytes;
        }
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert!(logical_bytes(MAX_ROWS) > logical_bytes(MAX_BATCH));
    }
}
