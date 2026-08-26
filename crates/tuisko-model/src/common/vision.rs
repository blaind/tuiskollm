//! Vision encoder source bindings shared by every admitted target.

use crate::common::inventory::CheckpointSnapshot;
use crate::{Arch, Bf16View, CheckpointError, CheckpointResult, TensorView};

/// BF16 source planes for one Vision transformer block.
#[derive(Clone, Copy, Debug)]
pub struct VisionBlockBindings<'a> {
    /// Attention output projection bias `[vision_hidden]`.
    pub attention_projection_bias: Bf16View<'a, 1>,
    /// Attention output projection weights `[vision_hidden, vision_hidden]`.
    pub attention_projection_weight: Bf16View<'a, 2>,
    /// Fused query, key, and value bias `[3 * vision_hidden]`.
    pub qkv_bias: Bf16View<'a, 1>,
    /// Fused query, key, and value weights `[3 * vision_hidden, vision_hidden]`.
    pub qkv_weight: Bf16View<'a, 2>,
    /// First MLP projection bias `[vision_intermediate]`.
    pub mlp_fc1_bias: Bf16View<'a, 1>,
    /// First MLP projection weights `[vision_intermediate, vision_hidden]`.
    pub mlp_fc1_weight: Bf16View<'a, 2>,
    /// Second MLP projection bias `[vision_hidden]`.
    pub mlp_fc2_bias: Bf16View<'a, 1>,
    /// Second MLP projection weights `[vision_hidden, vision_intermediate]`.
    pub mlp_fc2_weight: Bf16View<'a, 2>,
    /// First pre-normalization bias `[vision_hidden]`.
    pub norm1_bias: Bf16View<'a, 1>,
    /// First pre-normalization weights `[vision_hidden]`.
    pub norm1_weight: Bf16View<'a, 1>,
    /// Second pre-normalization bias `[vision_hidden]`.
    pub norm2_bias: Bf16View<'a, 1>,
    /// Second pre-normalization weights `[vision_hidden]`.
    pub norm2_weight: Bf16View<'a, 1>,
    /// Zero-based Vision block index.
    pub block: usize,
}

impl<'a> VisionBlockBindings<'a> {
    fn bind_from<A: Arch>(
        block: usize,
        tensor: &mut impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let hidden = A::VISION_HIDDEN as u64;
        let intermediate = A::VISION_INTERMEDIATE as u64;
        let qkv_rows = hidden.checked_mul(3).ok_or_else(|| {
            CheckpointError::source_binding(format!("Vision block-{block} QKV row count overflows"))
        })?;
        let prefix = format!("model.visual.blocks.{block}");

        Ok(Self {
            attention_projection_bias: Bf16View::bind(
                tensor(&format!("{prefix}.attn.proj.bias"))?,
                [hidden],
            )?,
            attention_projection_weight: Bf16View::bind(
                tensor(&format!("{prefix}.attn.proj.weight"))?,
                [hidden, hidden],
            )?,
            qkv_bias: Bf16View::bind(tensor(&format!("{prefix}.attn.qkv.bias"))?, [qkv_rows])?,
            qkv_weight: Bf16View::bind(
                tensor(&format!("{prefix}.attn.qkv.weight"))?,
                [qkv_rows, hidden],
            )?,
            mlp_fc1_bias: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc1.bias"))?,
                [intermediate],
            )?,
            mlp_fc1_weight: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc1.weight"))?,
                [intermediate, hidden],
            )?,
            mlp_fc2_bias: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc2.bias"))?,
                [hidden],
            )?,
            mlp_fc2_weight: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc2.weight"))?,
                [hidden, intermediate],
            )?,
            norm1_bias: Bf16View::bind(tensor(&format!("{prefix}.norm1.bias"))?, [hidden])?,
            norm1_weight: Bf16View::bind(tensor(&format!("{prefix}.norm1.weight"))?, [hidden])?,
            norm2_bias: Bf16View::bind(tensor(&format!("{prefix}.norm2.bias"))?, [hidden])?,
            norm2_weight: Bf16View::bind(tensor(&format!("{prefix}.norm2.weight"))?, [hidden])?,
            block,
        })
    }
}

/// Complete BF16 source family for the admitted Vision encoder.
#[derive(Debug)]
pub struct VisionBindings<'a> {
    /// Transformer blocks in execution order.
    pub blocks: Vec<VisionBlockBindings<'a>>,
    /// Patch projection bias `[vision_hidden]`.
    pub patch_embedding_bias: Bf16View<'a, 1>,
    /// Spatiotemporal patch projection weights.
    pub patch_embedding_weight: Bf16View<'a, 5>,
    /// Learned position embeddings `[vision_positions, vision_hidden]`.
    pub position_embedding: Bf16View<'a, 2>,
    /// Patch-merger normalization bias `[vision_hidden]`.
    pub merger_norm_bias: Bf16View<'a, 1>,
    /// Patch-merger normalization weights `[vision_hidden]`.
    pub merger_norm_weight: Bf16View<'a, 1>,
    /// First patch-merger projection bias `[merged_patch_width]`.
    pub merger_fc1_bias: Bf16View<'a, 1>,
    /// First patch-merger projection weights `[merged_patch_width, merged_patch_width]`.
    pub merger_fc1_weight: Bf16View<'a, 2>,
    /// Text-width patch-merger output bias `[vision_output_hidden]`.
    pub merger_fc2_bias: Bf16View<'a, 1>,
    /// Text-width patch-merger output weights `[vision_output_hidden, merged_patch_width]`.
    pub merger_fc2_weight: Bf16View<'a, 2>,
}

impl<'a> VisionBindings<'a> {
    /// Binds the complete admitted Vision source family.
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(|name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let hidden = A::VISION_HIDDEN as u64;
        let output_hidden = A::VISION_OUTPUT_HIDDEN as u64;
        let merge_area = A::VISION_SPATIAL_MERGE_SIZE
            .checked_mul(A::VISION_SPATIAL_MERGE_SIZE)
            .ok_or_else(|| CheckpointError::source_binding("Vision merge area overflows"))?;
        let merged_width = A::VISION_HIDDEN
            .checked_mul(merge_area)
            .ok_or_else(|| CheckpointError::source_binding("Vision merged width overflows"))?
            as u64;
        let mut blocks = Vec::with_capacity(A::VISION_DEPTH);

        for block in 0..A::VISION_DEPTH {
            blocks.push(VisionBlockBindings::bind_from::<A>(block, &mut tensor)?);
        }

        Ok(Self {
            blocks,
            patch_embedding_bias: Bf16View::bind(
                tensor("model.visual.patch_embed.proj.bias")?,
                [hidden],
            )?,
            patch_embedding_weight: Bf16View::bind(
                tensor("model.visual.patch_embed.proj.weight")?,
                [
                    hidden,
                    A::VISION_INPUT_CHANNELS as u64,
                    A::VISION_TEMPORAL_PATCH_SIZE as u64,
                    A::VISION_PATCH_SIZE as u64,
                    A::VISION_PATCH_SIZE as u64,
                ],
            )?,
            position_embedding: Bf16View::bind(
                tensor("model.visual.pos_embed.weight")?,
                [A::VISION_POSITIONS as u64, hidden],
            )?,
            merger_norm_bias: Bf16View::bind(tensor("model.visual.merger.norm.bias")?, [hidden])?,
            merger_norm_weight: Bf16View::bind(
                tensor("model.visual.merger.norm.weight")?,
                [hidden],
            )?,
            merger_fc1_bias: Bf16View::bind(
                tensor("model.visual.merger.linear_fc1.bias")?,
                [merged_width],
            )?,
            merger_fc1_weight: Bf16View::bind(
                tensor("model.visual.merger.linear_fc1.weight")?,
                [merged_width, merged_width],
            )?,
            merger_fc2_bias: Bf16View::bind(
                tensor("model.visual.merger.linear_fc2.bias")?,
                [output_hidden],
            )?,
            merger_fc2_weight: Bf16View::bind(
                tensor("model.visual.merger.linear_fc2.weight")?,
                [output_hidden, merged_width],
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::test_support::sources::{
        TestArch, append_bf16_tensor, fixture_path, write_safetensors_payload,
    };
    use crate::{CheckpointErrorCode, SafeTensorFile};
    use serde_json::{Value, json};
    use std::fs;

    fn vision_fixture() -> (Value, Vec<u8>) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for block in 0..TestArch::VISION_DEPTH {
            let prefix = format!("model.visual.blocks.{block}");

            for (suffix, shape) in [
                ("attn.proj.bias", vec![4]),
                ("attn.proj.weight", vec![4, 4]),
                ("attn.qkv.bias", vec![12]),
                ("attn.qkv.weight", vec![12, 4]),
                ("mlp.linear_fc1.bias", vec![6]),
                ("mlp.linear_fc1.weight", vec![6, 4]),
                ("mlp.linear_fc2.bias", vec![4]),
                ("mlp.linear_fc2.weight", vec![4, 6]),
                ("norm1.bias", vec![4]),
                ("norm1.weight", vec![4]),
                ("norm2.bias", vec![4]),
                ("norm2.weight", vec![4]),
            ] {
                append_bf16_tensor(
                    &mut header,
                    &mut payload,
                    format!("{prefix}.{suffix}"),
                    shape,
                );
            }
        }

        for (name, shape) in [
            ("model.visual.merger.linear_fc1.bias", vec![16]),
            ("model.visual.merger.linear_fc1.weight", vec![16, 16]),
            ("model.visual.merger.linear_fc2.bias", vec![4]),
            ("model.visual.merger.linear_fc2.weight", vec![4, 16]),
            ("model.visual.merger.norm.bias", vec![4]),
            ("model.visual.merger.norm.weight", vec![4]),
            ("model.visual.patch_embed.proj.bias", vec![4]),
            ("model.visual.patch_embed.proj.weight", vec![4, 3, 2, 2, 2]),
            ("model.visual.pos_embed.weight", vec![8, 4]),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    #[test]
    fn binds_complete_vision_source_contract() {
        let path = fixture_path("vision");
        let (header, payload) = vision_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = VisionBindings::bind_from::<TestArch>(|name| file.tensor(name)).unwrap();

        assert_eq!(bindings.blocks.len(), 2);

        for (block_index, block) in bindings.blocks.iter().enumerate() {
            assert_eq!(block.block, block_index);
            assert_eq!(block.attention_projection_bias.shape(), &[4]);
            assert_eq!(block.attention_projection_weight.shape(), &[4, 4]);
            assert_eq!(block.qkv_bias.shape(), &[12]);
            assert_eq!(block.qkv_weight.shape(), &[12, 4]);
            assert_eq!(block.mlp_fc1_bias.shape(), &[6]);
            assert_eq!(block.mlp_fc1_weight.shape(), &[6, 4]);
            assert_eq!(block.mlp_fc2_bias.shape(), &[4]);
            assert_eq!(block.mlp_fc2_weight.shape(), &[4, 6]);
            assert_eq!(block.norm1_bias.shape(), &[4]);
            assert_eq!(block.norm1_weight.shape(), &[4]);
            assert_eq!(block.norm2_bias.shape(), &[4]);
            assert_eq!(block.norm2_weight.shape(), &[4]);
        }

        assert_eq!(
            bindings.blocks[0].attention_projection_bias.word(0),
            Some(1)
        );
        assert_eq!(bindings.blocks[1].norm2_weight.word(0), Some(24));
        assert_eq!(bindings.patch_embedding_bias.shape(), &[4]);
        assert_eq!(bindings.patch_embedding_bias.word(0), Some(31));
        assert_eq!(bindings.patch_embedding_weight.shape(), &[4, 3, 2, 2, 2]);
        assert_eq!(bindings.patch_embedding_weight.word(0), Some(32));
        assert_eq!(bindings.position_embedding.shape(), &[8, 4]);
        assert_eq!(bindings.position_embedding.word(0), Some(33));
        assert_eq!(bindings.merger_norm_bias.shape(), &[4]);
        assert_eq!(bindings.merger_norm_weight.shape(), &[4]);
        assert_eq!(bindings.merger_fc1_bias.shape(), &[16]);
        assert_eq!(bindings.merger_fc1_weight.shape(), &[16, 16]);
        assert_eq!(bindings.merger_fc2_bias.shape(), &[4]);
        assert_eq!(bindings.merger_fc2_weight.shape(), &[4, 16]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_vision_tensor_dtype_and_shape_mismatches() {
        for (label, tensor_name, expected, dtype_mismatch) in [
            (
                "vision-dtype",
                "model.visual.pos_embed.weight",
                "dtype `F32`, expected `BF16`",
                true,
            ),
            (
                "vision-shape",
                "model.visual.blocks.1.attn.qkv.weight",
                "shape [6, 8], expected [12, 4]",
                false,
            ),
        ] {
            let path = fixture_path(label);
            let (mut header, payload) = vision_fixture();

            if dtype_mismatch {
                header[tensor_name]["dtype"] = json!("F32");
                header[tensor_name]["shape"] = json!([4, 4]);
            } else {
                header[tensor_name]["shape"] = json!([6, 8]);
            }

            write_safetensors_payload(&path, header, &payload);
            let file = SafeTensorFile::open(&path).unwrap();
            let error = VisionBindings::bind_from::<TestArch>(|name| file.tensor(name))
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Tensor);
            assert!(error.to_string().contains(tensor_name), "{error}");
            assert!(error.to_string().contains(expected), "{error}");

            fs::remove_file(path).unwrap();
        }
    }
}
