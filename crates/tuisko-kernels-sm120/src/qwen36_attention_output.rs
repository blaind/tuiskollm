//! Exact Qwen3.6 gated static-FP8 attention output.

use crate::Qwen36GdnOutputOp;
use crate::device::attention_output::attention_gate_bf16;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const ROUTE_COUNT: usize = MAX_BATCH + PREFILL_TOKENS.len();
const THREADS: u32 = 256;

const _: () = assert!(Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS == 4_096);
const _: () = assert!(Qwen36Moe35B::ATTENTION_QUERY_ROWS == 8_192);
const _: () = assert!(Qwen36Moe35B::ATTENTION_QKV_ROWS == 9_216);
const _: () = assert!(Qwen36Moe35B::HIDDEN == 2_048);

#[cuda_module]
mod kernels {
    use super::*;

    /// Applies the query-paired sigmoid gate and publishes BF16 projection input.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_attention_output_gate_bf16<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 4,096-wide row and each of its 256 threads writes
        // exactly 16 columns. This first route preserves the already qualified
        // head/dimension gate mapping and per-value sigmoid/BF16 arithmetic;
        // only independent tokens scale from one to eight CTAs.
        unsafe {
            attention_gate_bf16::<Qwen36Moe35B>(attention, qkv, activation);
        }
    }

    /// Applies the query-paired gate for one exact Qwen3.6 prompt width.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_attention_output_gate_bf16_prefill<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // T=128 has 524,288 independent gated columns; sixteen B=8 kernels
        // would add 15 launch boundaries. One 256-thread CTA still owns each
        // 4,096-column row, so 128 CTAs remove that launch-latency waste.
        // Each output keeps the same source pair, sigmoid, and BF16 rounding;
        // only the number of independent token CTAs changes.
        unsafe {
            attention_gate_bf16::<Qwen36Moe35B>(attention, qkv, activation);
        }
    }
}

struct PreparedGate<const TOKENS: usize> {
    gate: PreparedLaunch<kernels::__qwen36_attention_output_gate_bf16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedGate<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            gate: module
                .prepare_qwen36_attention_output_gate_bf16::<TOKENS>(LaunchConfig1D::new(
                    TOKENS as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 attention-output gate", source)
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
            .qwen36_attention_output_gate_bf16::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 attention-output gate", source))
    }
}

struct PreparedPrefillGate<const TOKENS: usize> {
    gate: PreparedLaunch<kernels::__qwen36_attention_output_gate_bf16_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillGate<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 attention-output prefill route T={TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            gate: module
                .prepare_qwen36_attention_output_gate_bf16_prefill::<TOKENS>(LaunchConfig1D::new(
                    TOKENS as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 attention-output prefill gate", source)
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
            .qwen36_attention_output_gate_bf16_prefill::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 attention-output prefill gate", source)
            })
    }
}

pub(crate) fn qwen36_attention_output_ptx_names() -> [&'static str; ROUTE_COUNT] {
    [
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<1>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<2>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<3>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<4>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<5>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<6>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<7>(),
        kernels::qwen36_attention_output_gate_bf16_ptx_name::<8>(),
        kernels::qwen36_attention_output_gate_bf16_prefill_ptx_name::<32>(),
        kernels::qwen36_attention_output_gate_bf16_prefill_ptx_name::<64>(),
        kernels::qwen36_attention_output_gate_bf16_prefill_ptx_name::<128>(),
    ]
}

/// Prepared Qwen3.6 gate plus static-FP8 output projection routes for exact
/// `B=1..8` and `T=32,64,128`.
pub struct Qwen36AttentionOutputOp {
    gate_module: kernels::LoadedModule,
    projection: Qwen36GdnOutputOp,
    b1: PreparedGate<1>,
    b2: PreparedGate<2>,
    b3: PreparedGate<3>,
    b4: PreparedGate<4>,
    b5: PreparedGate<5>,
    b6: PreparedGate<6>,
    b7: PreparedGate<7>,
    b8: PreparedGate<8>,
    t32: PreparedPrefillGate<32>,
    t64: PreparedPrefillGate<64>,
    t128: PreparedPrefillGate<128>,
}

impl Qwen36AttentionOutputOp {
    /// Loads the gate module and prepares every exact gated projection route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_attention_output_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let gate_module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 attention-output gate", source))?;

        Ok(Self {
            b1: PreparedGate::prepare(&gate_module)?,
            b2: PreparedGate::prepare(&gate_module)?,
            b3: PreparedGate::prepare(&gate_module)?,
            b4: PreparedGate::prepare(&gate_module)?,
            b5: PreparedGate::prepare(&gate_module)?,
            b6: PreparedGate::prepare(&gate_module)?,
            b7: PreparedGate::prepare(&gate_module)?,
            b8: PreparedGate::prepare(&gate_module)?,
            t32: PreparedPrefillGate::prepare(&gate_module)?,
            t64: PreparedPrefillGate::prepare(&gate_module)?,
            t128: PreparedPrefillGate::prepare(&gate_module)?,
            projection: Qwen36GdnOutputOp::new(context)?,
            gate_module,
        })
    }

    /// Gates paged-attention output, statically quantizes it, and projects it.
    ///
    /// # Safety
    ///
    /// `attention` covers mutable FP32 `[tokens,4096]`; `qkv` covers BF16
    /// `[tokens,9216]`; `activation` covers BF16 `[tokens,4096]`; the code
    /// workspace covers E4M3 `[tokens,4096]`; weights cover E4M3
    /// `[2048,4096]`; and output covers BF16 `[tokens,2048]`. All planes are
    /// aligned, disjoint, context-local, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        activation_codes: *mut u8,
        input_scale: f32,
        weight_codes: *const u8,
        weight_scale: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch_gate {
            ($route:ident) => {
                unsafe {
                    self.$route
                        .launch(&self.gate_module, stream, attention, qkv, activation)
                }
            };
        }

        macro_rules! launch_prefill_gate {
            ($route:ident) => {
                unsafe {
                    self.$route
                        .launch(&self.gate_module, stream, attention, qkv, activation)
                }
            };
        }

        match tokens {
            1 => launch_gate!(b1)?,
            2 => launch_gate!(b2)?,
            3 => launch_gate!(b3)?,
            4 => launch_gate!(b4)?,
            5 => launch_gate!(b5)?,
            6 => launch_gate!(b6)?,
            7 => launch_gate!(b7)?,
            8 => launch_gate!(b8)?,
            32 => launch_prefill_gate!(t32)?,
            64 => launch_prefill_gate!(t64)?,
            128 => launch_prefill_gate!(t128)?,
            _ => {
                return Err(GpuError::invalid_launch(format!(
                    "Qwen3.6 attention output tokens {tokens} must be one of 1..={MAX_BATCH}, 32, 64, or 128"
                )));
            }
        }

        unsafe {
            self.projection.launch(
                stream,
                tokens,
                activation,
                activation_codes,
                input_scale,
                weight_codes,
                weight_scale,
                output,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(THREADS, 256);
        assert_eq!(Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS, 4_096);
        assert_eq!(Qwen36Moe35B::ATTENTION_QUERY_ROWS, 8_192);
        assert_eq!(Qwen36Moe35B::ATTENTION_QKV_ROWS, 9_216);

        let names = qwen36_attention_output_ptx_names();
        assert_eq!(names.len(), ROUTE_COUNT);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
        assert_eq!(PREFILL_TOKENS, [32, 64, 128]);
    }
}
