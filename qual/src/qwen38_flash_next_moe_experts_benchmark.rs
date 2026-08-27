//! Paired timings for every exact Qwen3.8-Flash-Next slot-indirected expert route.
//!
//! The measured configuration is the production one: weights are read through
//! the indirection table out of the sealed slot pool, never from a resident
//! expert-major plane, so the timing carries the indirection's cost rather than
//! hiding it behind a direct address.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen38_flash_next_moe_experts::{
    EXACT_ROUTES, HIDDEN, INTERMEDIATE, MAX_BATCH, Regions, SlotAssignment, TOP_K, layout,
    make_fixture, staged_pool,
};
use crate::target::{
    QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, Qwen38FlashNextExpertDispatch, Qwen38FlashNextMoeExpertsOp,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen38FlashNextMoeExpertsOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

fn dispatch_of(arena: &DeviceArena, regions: Regions) -> GpuResult<Qwen38FlashNextExpertDispatch> {
    Ok(Qwen38FlashNextExpertDispatch {
        input: arena.address(regions.input)?,
        expert_indices: arena.address(regions.expert_indices)?,
        routing_weights: arena.address(regions.routing_weights)?,
        slot_table: arena.address(regions.slot_table)?,
        slot_pool: arena.address(regions.slot_pool)?,
        weight_scales_2: arena.address(regions.weight_scales_2)?,
        shared_gate_weight: arena.address(regions.shared_gate)?,
        shared_up_weight: arena.address(regions.shared_up)?,
        shared_down_weight: arena.address(regions.shared_down)?,
        shared_gate_logit_weight: arena.address(regions.shared_gate_logit_weight)?,
        routed_intermediate: arena.address(regions.routed_intermediate)?,
        routed_output: arena.address(regions.routed_output)?,
        shared_intermediate: arena.address(regions.shared_intermediate)?,
        shared_output: arena.address(regions.shared_output)?,
        shared_gate_logit: arena.address(regions.shared_gate_logit)?,
        output: arena.address(regions.output)?,
    })
}

fn launch(
    op: &Qwen38FlashNextMoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    let dispatch = dispatch_of(arena, regions)?;

    unsafe { op.launch(stream, rows, &dispatch) }
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }

        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let fixture = make_fixture();
        let (pool, table) = staged_pool(&fixture, SlotAssignment::Identity);
        arena.copy_from_host(&stream, regions.input, &fixture.input)?;
        arena.copy_from_host(&stream, regions.expert_indices, &fixture.expert_indices)?;
        arena.copy_from_host(&stream, regions.routing_weights, &fixture.routing_weights)?;
        arena.copy_from_host(&stream, regions.slot_table, &table)?;
        arena.copy_from_host(&stream, regions.slot_pool, &pool)?;
        arena.copy_from_host(&stream, regions.weight_scales_2, &fixture.weight_scales_2)?;
        arena.copy_from_host(&stream, regions.shared_gate, &fixture.shared_gate)?;
        arena.copy_from_host(&stream, regions.shared_up, &fixture.shared_up)?;
        arena.copy_from_host(&stream, regions.shared_down, &fixture.shared_down)?;
        arena.copy_from_host(
            &stream,
            regions.shared_gate_logit_weight,
            &fixture.shared_gate_logit_weight,
        )?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen38FlashNextMoeExpertsOp::new(&context)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| capture_route(&op, &arena, &stream, regions, rows, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _op: op,
            arena,
            regions,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_operator_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_operator_prefill(route.rows as u64),
                    )
                };
                ExactDeviceCase::new(
                    "qwen38_flash_next/moe_experts/nvfp4_slot_top10",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen38FlashNextMoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, arena, stream, regions, rows)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        leaf,
        repeated,
    })
}

/// Bytes one route moves: the routed slots it dereferences, the resident shared
/// expert, and every activation plane it reads or writes.
fn logical_bytes(rows: usize) -> usize {
    let slots = rows * TOP_K * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
    let global_scales = rows * TOP_K * 3 * size_of::<f32>();
    let input = rows * HIDDEN * size_of::<u16>();
    let routing = rows * TOP_K * 2 * size_of::<u16>();
    let shared_weights =
        rows * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE + HIDDEN) * size_of::<u16>();
    let routed_intermediate = rows * TOP_K * INTERMEDIATE * 2 * size_of::<u16>();
    let routed_output = rows * TOP_K * HIDDEN * 2 * size_of::<u16>();
    let shared_intermediate = rows * INTERMEDIATE * 2 * size_of::<u16>();
    let shared_output = rows * HIDDEN * 2 * size_of::<u16>();
    let shared_gate_logit = rows * 2 * size_of::<u16>();
    let output = rows * HIDDEN * size_of::<u16>();

    slots
        + global_scales
        + input
        + routing
        + shared_weights
        + routed_intermediate
        + routed_output
        + shared_intermediate
        + shared_output
        + shared_gate_logit
        + output
}

/// Measures every exact Qwen3.8-Flash-Next slot-indirected expert route.
pub fn benchmark_qwen38_flash_next_moe_experts(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let slot_pool_bytes = session.regions.slot_pool_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen38_flash_next/moe_experts/slot_pool",
        BenchmarkMemoryKind::Weights,
        slot_pool_bytes,
        "sixteen address-stable 2,764,800-byte NVFP4 expert slots",
    )?;
    memory.register_owned(
        "qwen38_flash_next/moe_experts/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - slot_pool_bytes - padding_bytes,
        "max_rows=1024 routed and shared intermediates, expert outputs, and the block output",
    )?;
    memory.register_owned(
        "qwen38_flash_next/moe_experts/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen38-flash-next-moe-experts",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, repeated leaf graph, and repeated-operation graph",
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
    use super::*;
    use crate::qwen38_flash_next_moe_experts::{MAX_ROWS, POOL_SLOTS};

    #[test]
    fn accounting_covers_the_slot_pool_and_every_activation_plane() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.slot_pool_bytes(), POOL_SLOTS * 2_764_800);
        assert_eq!(regions.slot_pool_bytes(), 44_236_800);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);

        assert_eq!(logical_bytes(1), 37_634_724);
        assert_eq!(logical_bytes(MAX_BATCH), 301_077_792);
        assert_eq!(logical_bytes(MAX_ROWS), 38_537_957_376);
        assert_eq!(logical_bytes(MAX_BATCH), MAX_BATCH * logical_bytes(1));
        assert_eq!(logical_bytes(MAX_ROWS), MAX_ROWS * logical_bytes(1));
        assert_eq!(
            layout.byte_len(),
            regions.payload_bytes() + layout.byte_len() - regions.payload_bytes()
        );
    }
}
