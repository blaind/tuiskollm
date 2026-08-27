//! Exact Qwen3.8-Flash-Next QSA sigmoid output gate.
//!
//! Each packed query head stores 256 query rows followed by 256 raw gate rows.
//! This op applies `sigmoid(gate)` before `o_proj` and publishes the BF16 seam.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::attention_output::attention_gate_bf16;
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];
const ROUTE_COUNT: usize = MAX_BATCH + PREFILL_TOKENS.len();
const THREADS: u32 = 256;

// Bind the packed gate offset to the exact target geometry.
const _: () = assert!(Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS == 6_144);
const _: () = assert!(Qwen38FlashNext::ATTENTION_QUERY_ROWS == 12_288);
const _: () = assert!(Qwen38FlashNext::ATTENTION_QKV_ROWS == 13_312);
const _: () = assert!(Qwen38FlashNext::HEAD_DIM == 256);
const _: () = assert!(Qwen38FlashNext::HIDDEN == 2_560);

#[cuda_module]
mod kernels {
    use super::*;

    /// Applies the query-paired sigmoid gate for one exact QSA decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_attention_output_gate_bf16<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 6,144-wide row and each of its 256 threads writes
        // exactly 24 columns. The per-value sigmoid and BF16 rounding are the
        // already qualified ones; only the gate's source stride and the number
        // of independent token CTAs are this target's.
        unsafe {
            attention_gate_bf16::<Qwen38FlashNext>(attention, qkv, activation);
        }
    }

    /// Applies the query-paired sigmoid gate for one exact QSA prompt width.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_attention_output_gate_bf16_prefill<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // T=1024 has 6,291,456 independent gated columns; running it as 128
        // B=8 launches would add 127 launch boundaries. One 256-thread CTA
        // still owns each 6,144-column row, so the prompt widths differ from
        // the decode route only in CTA count and every output keeps the same
        // source pair, sigmoid, and BF16 rounding.
        unsafe {
            attention_gate_bf16::<Qwen38FlashNext>(attention, qkv, activation);
        }
    }
}

struct PreparedGate<const TOKENS: usize> {
    gate:
        PreparedLaunch<kernels::__qwen38_flash_next_attention_output_gate_bf16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedGate<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            gate: module
                .prepare_qwen38_flash_next_attention_output_gate_bf16::<TOKENS>(
                    LaunchConfig1D::new(TOKENS as u32, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing Qwen3.8-Flash-Next QSA attention-output gate",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_attention_output_gate_bf16::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching Qwen3.8-Flash-Next QSA attention-output gate",
                    source,
                )
            })
    }
}

struct PreparedPrefillGate<const TOKENS: usize> {
    gate: PreparedLaunch<
        kernels::__qwen38_flash_next_attention_output_gate_bf16_prefill_CudaKernel<TOKENS>,
    >,
}

impl<const TOKENS: usize> PreparedPrefillGate<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next QSA attention-output prefill route T={TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            gate: module
                .prepare_qwen38_flash_next_attention_output_gate_bf16_prefill::<TOKENS>(
                    LaunchConfig1D::new(TOKENS as u32, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing Qwen3.8-Flash-Next QSA attention-output prefill gate",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_attention_output_gate_bf16_prefill::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching Qwen3.8-Flash-Next QSA attention-output prefill gate",
                    source,
                )
            })
    }
}

pub(crate) fn qwen38_flash_next_attention_output_ptx_names() -> [&'static str; ROUTE_COUNT] {
    [
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<1>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<2>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<3>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<4>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<5>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<6>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<7>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_ptx_name::<8>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_attention_output_gate_bf16_prefill_ptx_name::<1_024>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_attention_gate),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextAttentionGateRoutes {
    #[route(1)]
    b1: PreparedGate<1>,
    #[route(2)]
    b2: PreparedGate<2>,
    #[route(3)]
    b3: PreparedGate<3>,
    #[route(4)]
    b4: PreparedGate<4>,
    #[route(5)]
    b5: PreparedGate<5>,
    #[route(6)]
    b6: PreparedGate<6>,
    #[route(7)]
    b7: PreparedGate<7>,
    #[route(8)]
    b8: PreparedGate<8>,
    #[route(32)]
    t32: PreparedPrefillGate<32>,
    #[route(64)]
    t64: PreparedPrefillGate<64>,
    #[route(128)]
    t128: PreparedPrefillGate<128>,
    #[route(1024)]
    t1024: PreparedPrefillGate<1_024>,
}

/// Prepared Qwen3.8-Flash-Next QSA sigmoid output-gate routes for exact `B=1..8` and
/// `T=32,64,128,1024`.
///
/// Stops at the BF16 seam consumed by the `6144 -> 2560` output projection.
pub struct Qwen38FlashNextAttentionGateOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextAttentionGateRoutes,
}

impl Qwen38FlashNextAttentionGateOp {
    /// Loads the embedded module and prepares every exact gate route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry()?;
        let _ = qwen38_flash_next_attention_output_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module(
                "loading Qwen3.8-Flash-Next QSA attention-output gate",
                source,
            )
        })?;

        Ok(Self {
            routes: Qwen38FlashNextAttentionGateRoutes::prepare(&module)?,
            module,
        })
    }

    /// Applies the packed sigmoid gate to the paged-attention output.
    ///
    /// # Safety
    ///
    /// `attention` covers mutable FP32 `[tokens,6144]` in `(head, dimension)`
    /// order; `qkv` covers BF16 `[tokens,13312]` in the fused query/gate, key,
    /// value order; `activation` covers BF16 `[tokens,6144]`. All planes are
    /// aligned, disjoint, context-local, and live through stream completion.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch_gate {
            ($route:expr) => {
                // SAFETY: the caller's pointer contract reaches the entry
                // unchanged.
                unsafe { $route.launch(&self.module, stream, attention, qkv, activation) }
            };
        }

        dispatch_qwen38_flash_next_attention_gate!(&self.routes, tokens, |route| launch_gate!(route), else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next QSA attention output tokens {tokens} must be one of 1..={MAX_BATCH}, 32, 64, 128, or 1024"
            ))) )
    }
}

/// Rejects an architecture whose geometry the emitted gate does not cover.
fn require_geometry() -> GpuResult<()> {
    if Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS != 6_144
        || Qwen38FlashNext::ATTENTION_QUERY_ROWS != 12_288
        || Qwen38FlashNext::ATTENTION_QKV_ROWS != 13_312
        || Qwen38FlashNext::HEAD_DIM != 256
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next geometry is incompatible with its admitted QSA output-gate schedule",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Pins the packed query/gate geometry and exact route inventory.
    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(THREADS, 256);
        assert_eq!(Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS, 6_144);
        assert_eq!(Qwen38FlashNext::ATTENTION_QUERY_ROWS, 12_288);
        assert_eq!(Qwen38FlashNext::ATTENTION_QKV_ROWS, 13_312);
        assert_eq!(Qwen38FlashNext::HEAD_DIM, 256);
        assert!(require_geometry().is_ok());

        // The packed per-head stride is two head widths: query then gate.
        assert_eq!(
            Qwen38FlashNext::ATTENTION_QUERY_ROWS,
            2 * Qwen38FlashNext::NUM_ATTENTION_HEADS * Qwen38FlashNext::HEAD_DIM
        );

        let names = qwen38_flash_next_attention_output_ptx_names();
        assert_eq!(names.len(), ROUTE_COUNT);
        assert_eq!(ROUTE_COUNT, 12);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
        assert_eq!(PREFILL_TOKENS, [32, 64, 128, 1_024]);
    }
}
