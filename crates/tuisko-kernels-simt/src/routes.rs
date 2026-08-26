//! Host launchers shared as source by the pre-Blackwell fallback crates.
//!
//! Each `#[cuda_module]` generates its kernel types and launch methods inside
//! the crate that declares it, so the prepared-route scaffolding around them
//! cannot be a plain function: it can only be shared as tokens expanded in the
//! owning module. These macros carry the launcher text that the sm89 and sm86
//! operators were proven to hold character-for-character in common, and they
//! emit nothing that reaches device code.
//!
//! Both macro bodies are `rustfmt`-skipped so the expanded launcher text stays
//! character-identical to the per-crate copies it replaces; reflowing them at
//! macro-arm indentation would rewrite closure bodies for width alone.

/// Declares the exact-batch RMSNorm routes over the calling module's `kernels`.
///
/// The B=1 plain route deliberately binds the concrete `rms_norm_b1` entry that
/// anchors the embedded artifact, while its residual sibling binds the generic
/// entry at `TOKENS = 1`. That asymmetry is the emitted inventory and must not
/// be normalized away.
#[macro_export]
#[rustfmt::skip]
macro_rules! residual_norm_batch_routes {
    () => {
        // Prepared generic entries for one exact batch.
        struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
            plain: PreparedLaunch<kernels::__rms_norm_CudaKernel<A, TOKENS>>,
            residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<A, TOKENS>>,
        }

        // B=1 keeps the concrete plain entry that anchors the embedded module artifact.
        struct PreparedBatchOneRoute {
            plain: PreparedLaunch<kernels::__rms_norm_b1_CudaKernel>,
            residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<Qwen38_27B, 1>>,
        }

        impl PreparedBatchOneRoute {
            fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
                let launch = LaunchConfig1D::new(1, THREADS, 0);
                let plain = module
                    .prepare_rms_norm_b1(launch)
                    .map_err(|source| GpuError::launch("preparing the B=1 RMSNorm kernel", source))?;
                let residual = module
                    .prepare_residual_rms_norm::<Qwen38_27B, 1>(launch)
                    .map_err(|source| {
                        GpuError::launch("preparing the B=1 residual RMSNorm kernel", source)
                    })?;

                Ok(Self { plain, residual })
            }

            unsafe fn launch_plain(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                input: *const u16,
                weight: *const u16,
                output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .rms_norm_b1(
                        stream,
                        &self.plain,
                        input.cast::<u32>(),
                        weight.cast::<u32>(),
                        output.cast::<u32>(),
                    )
                    .map_err(|source| GpuError::launch("launching the B=1 RMSNorm kernel", source))
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn launch_residual(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                residual_input: *const u16,
                branch: *const u16,
                weight: *const u16,
                residual_output: *mut u16,
                normalized_output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .residual_rms_norm::<Qwen38_27B, 1>(
                        stream,
                        &self.residual,
                        residual_input.cast::<u32>(),
                        branch.cast::<u32>(),
                        weight.cast::<u32>(),
                        residual_output.cast::<u32>(),
                        normalized_output.cast::<u32>(),
                    )
                    .map_err(|source| GpuError::launch("launching the B=1 residual RMSNorm kernel", source))
            }
        }

        impl<A: Arch, const TOKENS: usize> PreparedBatchRoute<A, TOKENS> {
            fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
                let blocks = u32::try_from(TOKENS)
                    .map_err(|_| GpuError::invalid_launch("RMSNorm batch exceeds CUDA grid width"))?;
                let launch = LaunchConfig1D::new(blocks, THREADS, 0);
                let plain = module
                    .prepare_rms_norm::<A, TOKENS>(launch)
                    .map_err(|source| GpuError::launch("preparing the RMSNorm kernel", source))?;
                let residual = module
                    .prepare_residual_rms_norm::<A, TOKENS>(launch)
                    .map_err(|source| GpuError::launch("preparing the residual RMSNorm kernel", source))?;

                Ok(Self { plain, residual })
            }

            unsafe fn launch_plain(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                input: *const u16,
                weight: *const u16,
                output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .rms_norm::<A, TOKENS>(
                        stream,
                        &self.plain,
                        input.cast::<u32>(),
                        weight.cast::<u32>(),
                        output.cast::<u32>(),
                    )
                    .map_err(|source| GpuError::launch("launching the RMSNorm kernel", source))
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn launch_residual(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                residual_input: *const u16,
                branch: *const u16,
                weight: *const u16,
                residual_output: *mut u16,
                normalized_output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .residual_rms_norm::<A, TOKENS>(
                        stream,
                        &self.residual,
                        residual_input.cast::<u32>(),
                        branch.cast::<u32>(),
                        weight.cast::<u32>(),
                        residual_output.cast::<u32>(),
                        normalized_output.cast::<u32>(),
                    )
                    .map_err(|source| GpuError::launch("launching the residual RMSNorm kernel", source))
            }
        }
    };
}

/// Declares the exact-batch NVFP4 A16 routes over the calling module's `kernels`.
///
/// `label` names the architecture and operator family exactly as the existing
/// launch errors spell it; the six identifiers are the `#[cuda_module]` items
/// for the concrete B=1 entry and its generic `<A, TOKENS>` sibling.
#[macro_export]
#[rustfmt::skip]
macro_rules! nvfp4_a16_batch_routes {
    (
        label = $label:literal,
        b1 = { $b1_kernel:ident, $b1_prepare:ident, $b1_launch:ident },
        batched = { $kernel:ident, $prepare:ident, $launch:ident },
    ) => {
        struct PreparedBatchOneRoute {
            projection: PreparedLaunch<kernels::$b1_kernel>,
        }

        struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
            projection: PreparedLaunch<kernels::$kernel<A, TOKENS>>,
        }

        impl PreparedBatchOneRoute {
            fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
                let projection = module
                    .$b1_prepare(launch_config())
                    .map_err(|source| GpuError::launch(concat!("preparing ", $label, " B=1"), source))?;

                Ok(Self { projection })
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn launch(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                input: *const u16,
                weight_codes: *const u8,
                weight_scales: *const u8,
                weight_scale_reciprocal: f32,
                output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .$b1_launch(
                        stream,
                        &self.projection,
                        input.cast::<u32>(),
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        weight_scale_reciprocal,
                        output,
                    )
                    .map_err(|source| GpuError::launch(concat!("launching ", $label, " B=1"), source))
            }
        }

        impl<A: Arch, const TOKENS: usize> PreparedBatchRoute<A, TOKENS> {
            fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
                let projection = module
                    .$prepare::<A, TOKENS>(launch_config())
                    .map_err(|source| GpuError::launch(concat!("preparing ", $label), source))?;

                Ok(Self { projection })
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn launch(
                &self,
                module: &kernels::LoadedModule,
                stream: &CudaStream,
                input: *const u16,
                weight_codes: *const u8,
                weight_scales: *const u8,
                weight_scale_reciprocal: f32,
                output: *mut u16,
            ) -> GpuResult<()> {
                module
                    .$launch::<A, TOKENS>(
                        stream,
                        &self.projection,
                        input.cast::<u32>(),
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        weight_scale_reciprocal,
                        output,
                    )
                    .map_err(|source| GpuError::launch(concat!("launching ", $label), source))
            }
        }
    };
}
