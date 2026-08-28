//! Lossless uploads shared by both Qwen3.8-Flash-Next layer shapes.
//!
//! These helpers move source words without conversion. Expert slot assembly only concatenates
//! borrowed byte ranges in kernel address order.

use crate::common::math::product;
use crate::qwen38_flash_next::expert_pool_layout::Qwen38FlashNextExpertPoolRegions;
use crate::{EngineError, EngineResult};
use tuisko_gpu::{ArenaRegion, CudaStream, DeviceArena};
use tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
use tuisko_model::{
    MaterializedQwen38FlashNextExpert, MaterializedQwen38FlashNextExpertPool,
    MaterializedQwen38FlashNextMoe, Qwen38FlashNext, Qwen38FlashNextHyperConnectionBindings,
};

/// The four device planes one gated residual occupies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HyperConnectionRegions {
    pub(crate) norm: ArenaRegion<u16>,
    pub(crate) down: ArenaRegion<u16>,
    pub(crate) up: ArenaRegion<u16>,
    pub(crate) inject: ArenaRegion<u16>,
}

/// The six device planes one layer's MoE half occupies, excluding the routed pool.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeRegions {
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) shared_up_weight: ArenaRegion<u16>,
    pub(crate) shared_down_weight: ArenaRegion<u16>,
    pub(crate) shared_gate_logit_weight: ArenaRegion<u16>,
    pub(crate) expert_weight_scales_2: ArenaRegion<f32>,
}

/// Reinterprets a materialized BF16 plane as little-endian words.
pub(crate) fn bf16_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(EngineError::layout(
            "Qwen3.8-Flash-Next BF16 source plane has an odd byte length",
        ));
    }

    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

/// Uploads one gated residual's four planes, refusing a bracket that cannot write back.
pub(crate) fn upload_hyper_connection(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: HyperConnectionRegions,
    bindings: Qwen38FlashNextHyperConnectionBindings<'_>,
) -> EngineResult<()> {
    arena.copy_from_host(
        stream,
        regions.norm,
        &bindings.hc_norm.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.down,
        &bindings.input_mix_down.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.up,
        &bindings.input_mix_up.words().collect::<Vec<_>>(),
    )?;
    let inject = bindings.block_inject.ok_or_else(|| {
        EngineError::layout(
            "a Qwen3.8-Flash-Next layer hyper-connection must write back into the stream",
        )
    })?;
    arena.copy_from_host(stream, regions.inject, &inject.words().collect::<Vec<_>>())?;

    Ok(())
}

/// Uploads the router, the shared expert, and the per-expert second-stage scales.
pub(crate) fn upload_moe(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: MoeRegions,
    moe: &MaterializedQwen38FlashNextMoe<'_>,
) -> EngineResult<()> {
    arena.copy_from_host(
        stream,
        regions.router_weight,
        &moe.router_weight.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.shared_gate_weight,
        &moe.shared_expert
            .gate_proj_weight
            .words()
            .collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.shared_up_weight,
        &moe.shared_expert.up_proj_weight.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.shared_down_weight,
        &moe.shared_expert
            .down_proj_weight
            .words()
            .collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.shared_gate_logit_weight,
        &moe.shared_expert.gate_weight.words().collect::<Vec<_>>(),
    )?;

    // Three F32 globals per expert, gate then up then down, indexed by expert id exactly as
    // the routed kernels read them. Gate and up share one `weight_scale_2` because they share
    // one fused source plane.
    let mut scales = Vec::with_capacity(moe.experts.expert_count * 3);
    for expert in &moe.experts.experts {
        scales.extend([
            expert.gate_up_weight_scale_2,
            expert.gate_up_weight_scale_2,
            expert.down_weight_scale_2,
        ]);
    }
    arena.copy_from_host(stream, regions.expert_weight_scales_2, &scales)?;

    Ok(())
}

/// Fills the sealed slot arena and publishes the identity assignment over it.
///
/// Expert `e` occupies slot `e`. A qualification that permutes the table publishes a different
/// assignment over the same pool contents, which is what makes the invariance proof meaningful.
pub(crate) fn upload_expert_pool(
    pool: &DeviceArena,
    stream: &CudaStream,
    regions: Qwen38FlashNextExpertPoolRegions,
    experts: &MaterializedQwen38FlashNextExpertPool<'_>,
) -> EngineResult<()> {
    let table = (0..Qwen38FlashNext::NUM_EXPERTS as u32).collect::<Vec<_>>();
    pool.copy_from_host(stream, regions.slot_table, &table)?;

    for (slot, expert) in experts.experts.iter().enumerate() {
        let image = expert_slot_image(expert, &experts.scale_e4m3_swizzled)?;
        let offset = product(
            "Qwen3.8-Flash-Next expert slot offset",
            slot,
            QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES,
        )?;
        pool.copy_slice_from_host(stream, regions.slot_pool, offset, &image)?;
    }

    Ok(())
}

/// One expert's sealed slot image: `down | gate | up | gate_up_scales | down_scales`.
///
/// The packed E2M1 order is the checkpoint's own, which is what makes a production stage a
/// single contiguous read rather than three gathers.
pub(crate) fn expert_slot_image(
    expert: &MaterializedQwen38FlashNextExpert<'_>,
    scales: &[u8],
) -> EngineResult<Vec<u8>> {
    let mut image = Vec::with_capacity(QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES);
    image.extend_from_slice(expert.down_weight_e2m1);
    image.extend_from_slice(expert.gate_weight_e2m1);
    image.extend_from_slice(expert.up_weight_e2m1);
    for extent in [expert.gate_up_scale, expert.down_scale] {
        image.extend_from_slice(
            scales
                .get(extent.offset..extent.offset + extent.bytes)
                .ok_or_else(|| {
                    EngineError::layout(format!(
                        "Qwen3.8-Flash-Next expert {} names a scale extent outside its pool",
                        expert.expert
                    ))
                })?,
        );
    }
    if image.len() != QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next expert {} slot image is {} bytes, expected {QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES}",
            expert.expert,
            image.len()
        )));
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::bf16_words;

    #[test]
    fn bf16_source_words_are_little_endian_and_even() {
        assert_eq!(
            bf16_words(&[0x01, 0x02, 0x03, 0x04]).unwrap(),
            [0x0201, 0x0403]
        );
        assert!(bf16_words(&[0x01]).is_err());
        assert!(bf16_words(&[]).unwrap().is_empty());
    }
}
