//! Qwen3.8-27B lossless materialization into runtime-native host layouts.

use crate::Arch;
use crate::common::inventory::CheckpointSnapshot;
use crate::common::materialized::{MaterializedMemory, sealed};
use crate::common::mtp::MaterializedMtpQkv;
use crate::common::routes::require_full_attention_layer;
use crate::common::scale_swizzle::{PlaneGatherer, host_shape};
use crate::common::source_binding::{SourceLayerBinding, sealed as binding_sealed};
use crate::qwen38::bindings::{FullAttentionQkvBindings, MtpBindings};
use crate::{CheckpointError, CheckpointResult};

/// Runtime-native fused QKV planes in query/gate, key, value row order.
#[derive(Debug)]
pub struct MaterializedFullAttentionQkv {
    /// Losslessly gathered E4M3 weights `[rows, columns]`.
    pub weight_e4m3: Vec<u8>,
    /// Losslessly gathered little-endian BF16 row scales `[rows, 1]`.
    pub scale_bf16: Vec<u8>,
    /// Fused query/gate, key, and value row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl binding_sealed::Sealed for FullAttentionQkvBindings<'_> {}

impl<'a, A: Arch> SourceLayerBinding<'a, A> for FullAttentionQkvBindings<'a> {
    type Materialized = MaterializedFullAttentionQkv;

    fn bind(snapshot: &'a CheckpointSnapshot<A>, layer: usize) -> CheckpointResult<Self> {
        Self::bind::<A>(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Self::materialize(self)
    }
}

impl sealed::Sealed for MaterializedFullAttentionQkv {}

impl MaterializedMemory for MaterializedFullAttentionQkv {
    fn host_bytes(&self) -> usize {
        self.weight_e4m3.len() + self.scale_bf16.len()
    }
}

impl FullAttentionQkvBindings<'_> {
    /// Gathers the non-contiguous source planes without requantizing represented values.
    pub fn materialize(self) -> CheckpointResult<MaterializedFullAttentionQkv> {
        require_full_attention_layer(self.layer, self.layer_count, self.full_attention_interval)?;

        let [query_rows, columns] = host_shape(
            self.query_gate_weight.shape(),
            "full-attention query/gate weights",
        )?;
        let [key_rows, key_columns] =
            host_shape(self.key_weight.shape(), "full-attention key weights")?;
        let [value_rows, value_columns] =
            host_shape(self.value_weight.shape(), "full-attention value weights")?;

        let query_scale_shape = host_shape(
            self.query_gate_scale.shape(),
            "full-attention query/gate scales",
        )?;
        let key_scale_shape = host_shape(self.key_scale.shape(), "full-attention key scales")?;
        let value_scale_shape =
            host_shape(self.value_scale.shape(), "full-attention value scales")?;

        if key_rows != value_rows
            || columns != key_columns
            || columns != value_columns
            || query_scale_shape != [query_rows, 1]
            || key_scale_shape != [key_rows, 1]
            || value_scale_shape != [value_rows, 1]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} full-attention QKV source planes have incompatible shapes",
                self.layer
            )));
        }

        let rows = query_rows
            .checked_add(key_rows)
            .and_then(|rows| rows.checked_add(value_rows))
            .ok_or_else(|| {
                CheckpointError::source_binding(format!(
                    "layer-{} full-attention QKV row count overflows",
                    self.layer
                ))
            })?;

        let weight_e4m3 = PlaneGatherer::gather(
            [
                self.query_gate_weight.codes(),
                self.key_weight.codes(),
                self.value_weight.codes(),
            ],
            &format!("layer-{} full-attention QKV weights", self.layer),
        )?;
        let scale_bf16 = PlaneGatherer::gather(
            [
                self.query_gate_scale.bytes(),
                self.key_scale.bytes(),
                self.value_scale.bytes(),
            ],
            &format!("layer-{} full-attention QKV scales", self.layer),
        )?;

        Ok(MaterializedFullAttentionQkv {
            weight_e4m3,
            scale_bf16,
            rows,
            columns,
            layer: self.layer,
        })
    }
}

impl MtpBindings<'_> {
    /// Gathers the non-contiguous draft QKV planes without changing BF16 words.
    pub fn materialize_qkv(&self) -> CheckpointResult<MaterializedMtpQkv> {
        let [query_rows, columns] =
            host_shape(self.query_gate_weight.shape(), "MTP query/gate weights")?;
        let [key_rows, key_columns] = host_shape(self.key_weight.shape(), "MTP key weights")?;
        let [value_rows, value_columns] =
            host_shape(self.value_weight.shape(), "MTP value weights")?;

        if key_rows != value_rows || columns != key_columns || columns != value_columns {
            return Err(CheckpointError::source_binding(
                "MTP QKV source planes have incompatible shapes",
            ));
        }

        let rows = query_rows
            .checked_add(key_rows)
            .and_then(|rows| rows.checked_add(value_rows))
            .ok_or_else(|| CheckpointError::source_binding("MTP QKV row count overflows"))?;

        let weight_bf16 = PlaneGatherer::gather(
            [
                self.query_gate_weight.bytes(),
                self.key_weight.bytes(),
                self.value_weight.bytes(),
            ],
            "MTP QKV weights",
        )?;

        Ok(MaterializedMtpQkv {
            weight_bf16,
            rows,
            columns,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::common::source_binding::SourceLayerBinding;
    use crate::{MaterializedMemory, Qwen38_27B};

    use crate::CheckpointErrorCode;
    use crate::common::test_support::sources::{bf16_bytes, bf16_view, fp8_view};
    use crate::qwen38::bindings::FullAttentionQkvBindings;

    #[test]
    fn full_attention_qkv_materialization_gathers_exact_source_words() {
        let query_shape = [4, 3];
        let kv_shape = [1, 3];
        let query_scale_shape = [4, 1];
        let kv_scale_shape = [1, 1];
        let query_weight = (0x10..0x1c).collect::<Vec<_>>();
        let key_weight = (0x30..0x33).collect::<Vec<_>>();
        let value_weight = (0x50..0x53).collect::<Vec<_>>();
        let query_scale = bf16_bytes(&[0x3f80, 0x4000, 0x4040, 0x4080]);
        let key_scale = bf16_bytes(&[0x40a0]);
        let value_scale = bf16_bytes(&[0x40c0]);
        let bindings = FullAttentionQkvBindings {
            query_gate_weight: fp8_view("query", &query_shape, &query_weight),
            key_weight: fp8_view("key", &kv_shape, &key_weight),
            value_weight: fp8_view("value", &kv_shape, &value_weight),
            query_gate_scale: bf16_view("query-scale", &query_scale_shape, &query_scale),
            key_scale: bf16_view("key-scale", &kv_scale_shape, &key_scale),
            value_scale: bf16_view("value-scale", &kv_scale_shape, &value_scale),
            layer: 3,
            layer_count: 8,
            full_attention_interval: 4,
        };

        let error = FullAttentionQkvBindings {
            layer: 4,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("does not use the admitted full-attention")
        );

        let materialized = bindings.materialize().unwrap();
        let query_end = query_weight.len();
        let key_end = query_end + key_weight.len();
        let query_scale_end = query_scale.len();
        let key_scale_end = query_scale_end + key_scale.len();

        assert_eq!(&materialized.weight_e4m3[..query_end], query_weight);
        assert_eq!(&materialized.weight_e4m3[query_end..key_end], key_weight);
        assert_eq!(&materialized.weight_e4m3[key_end..], value_weight);
        assert_eq!(&materialized.scale_bf16[..query_scale_end], query_scale);
        assert_eq!(
            &materialized.scale_bf16[query_scale_end..key_scale_end],
            key_scale
        );
        assert_eq!(&materialized.scale_bf16[key_scale_end..], value_scale);
        assert_eq!(materialized.host_bytes(), 30);
        assert_eq!((materialized.rows, materialized.columns), (6, 3));
        assert_eq!(materialized.layer, 3);

        let via_trait =
            <FullAttentionQkvBindings<'_> as SourceLayerBinding<'_, Qwen38_27B>>::materialize(
                bindings,
            )
            .unwrap();

        assert_eq!(via_trait.weight_e4m3, materialized.weight_e4m3);
        assert_eq!(via_trait.scale_bf16, materialized.scale_bf16);
        assert_eq!(via_trait.host_bytes(), materialized.host_bytes());
        assert_eq!((via_trait.rows, via_trait.columns), (6, 3));
    }

    #[test]
    fn full_attention_qkv_materialization_rejects_incompatible_shapes() {
        let query_shape = [4, 3];
        let key_shape = [1, 2];
        let value_shape = [1, 3];
        let query_scale_shape = [4, 1];
        let kv_scale_shape = [1, 1];
        let query_weight = vec![0x10; 12];
        let key_weight = vec![0x20; 2];
        let value_weight = vec![0x30; 3];
        let query_scale = bf16_bytes(&[0x3f80; 4]);
        let key_scale = bf16_bytes(&[0x3f80]);
        let value_scale = bf16_bytes(&[0x3f80]);
        let error = FullAttentionQkvBindings {
            query_gate_weight: fp8_view("query", &query_shape, &query_weight),
            key_weight: fp8_view("key", &key_shape, &key_weight),
            value_weight: fp8_view("value", &value_shape, &value_weight),
            query_gate_scale: bf16_view("query-scale", &query_scale_shape, &query_scale),
            key_scale: bf16_view("key-scale", &kv_scale_shape, &key_scale),
            value_scale: bf16_view("value-scale", &kv_scale_shape, &value_scale),
            layer: 3,
            layer_count: 8,
            full_attention_interval: 4,
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("incompatible shapes"));
    }
}
