//! Qwen3.8-27B split-shard snapshot inventory admission.

use crate::CheckpointSnapshot;
use crate::common::config_util::validate_config;
use crate::common::inventory::{
    CONFIG_FILE, INDEX_FILE, MODEL_FILE, Shard, read_index, require_count, validate_file_length,
    validate_revision,
};
use crate::{Arch, CheckpointError, CheckpointResult, SafeTensorFile};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::Path;

const MTP_FILE: &str = "model_mtp.safetensors";

pub(crate) const QWEN38_INVENTORY: SplitInventorySpec = SplitInventorySpec {
    model_bytes: 22_568_192_096,
    model_tensors: 1_953,
    mtp_bytes: 849_400_392,
    mtp_tensors: 15,
    index_bytes: 164_371,
    index_entries: 1_968,
};

#[derive(Clone, Copy)]
pub(crate) struct SplitInventorySpec {
    model_bytes: u64,
    model_tensors: usize,
    mtp_bytes: u64,
    mtp_tensors: usize,
    index_bytes: u64,
    index_entries: usize,
}

fn validate_weight_map(
    index_path: &Path,
    weight_map: BTreeMap<String, String>,
    model: &SafeTensorFile,
    mtp: &SafeTensorFile,
    spec: SplitInventorySpec,
) -> CheckpointResult<BTreeMap<String, Shard>> {
    let mut tensors = BTreeMap::new();
    let mut model_entries = 0;
    let mut mtp_entries = 0;

    for (name, file) in weight_map {
        let shard = match file.as_str() {
            MODEL_FILE => {
                model.tensor(&name)?;
                model_entries += 1;
                Shard(0)
            }
            MTP_FILE => {
                mtp.tensor(&name)?;
                mtp_entries += 1;
                Shard(1)
            }
            _ => {
                return Err(CheckpointError::inventory(format!(
                    "{} maps tensor `{name}` to unsupported shard `{file}`",
                    index_path.display()
                )));
            }
        };

        tensors.insert(name, shard);
    }

    require_count(index_path, MODEL_FILE, model_entries, spec.model_tensors)?;
    require_count(index_path, MTP_FILE, mtp_entries, spec.mtp_tensors)?;

    Ok(tensors)
}

impl<A: Arch> CheckpointSnapshot<A> {
    pub(crate) fn open_split_with_spec(
        root: &Path,
        spec: SplitInventorySpec,
    ) -> CheckpointResult<Self> {
        validate_revision::<A>(root)?;
        validate_config::<A>(&root.join(CONFIG_FILE))?;

        let index_path = root.join(INDEX_FILE);
        let model_path = root.join(MODEL_FILE);
        let mtp_path = root.join(MTP_FILE);

        validate_file_length(&index_path, spec.index_bytes)?;
        validate_file_length(&model_path, spec.model_bytes)?;
        validate_file_length(&mtp_path, spec.mtp_bytes)?;

        let index = read_index(&index_path)?;
        let total_bytes = spec
            .model_bytes
            .checked_add(spec.mtp_bytes)
            .ok_or_else(|| CheckpointError::inventory("checkpoint shard lengths overflow"))?;

        require_count(
            &index_path,
            "entries",
            index.weight_map.len(),
            spec.index_entries,
        )?;
        require_count(
            &index_path,
            "metadata.total_size",
            index.metadata.total_size,
            total_bytes,
        )?;

        let model = SafeTensorFile::open(&model_path)?;
        let mtp = SafeTensorFile::open(&mtp_path)?;

        require_count(
            &model_path,
            "tensors",
            model.tensor_count(),
            spec.model_tensors,
        )?;
        require_count(&mtp_path, "tensors", mtp.tensor_count(), spec.mtp_tensors)?;

        let tensors = validate_weight_map(&index_path, index.weight_map, &model, &mtp, spec)?;

        Ok(Self {
            root: root.to_owned(),
            inventory_path: index_path,
            tensors,
            shards: vec![model, mtp],
            arch: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config_util::test_quantization_config;
    use crate::common::inventory::{
        CheckpointSnapshot, ExpectedTensor, INDEX_FILE, MODEL_FILE, validate_exact_tensors,
    };
    use crate::common::test_builder::SafeTensorTestBuilder;
    use crate::{Arch, CheckpointErrorCode, DType, SafeTensorFile};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 1;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 1;
        const VOCAB: usize = 1;
        const LAYERS: usize = 64;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 1;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 1;
        const VISION_HIDDEN: usize = 1;
        const VISION_INTERMEDIATE: usize = 1;
        const VISION_NUM_HEADS: usize = 1;
        const VISION_POSITIONS: usize = 1;
        const VISION_OUTPUT_HIDDEN: usize = 1;
        const VISION_INPUT_CHANNELS: usize = 1;
        const VISION_PATCH_SIZE: usize = 1;
        const VISION_SPATIAL_MERGE_SIZE: usize = 1;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 1;
    }

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        weight_map: BTreeMap<String, String>,
        spec: SplitInventorySpec,
    }

    impl Fixture {
        fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "tuisko-inventory-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            let root = base.join(TestArch::REVISION);
            fs::create_dir_all(&root).unwrap();
            write_config(&root);

            let model_path = root.join(MODEL_FILE);
            let mtp_path = root.join(MTP_FILE);
            write_safetensors(&model_path, &["main.a", "main.b"]);
            write_safetensors(&mtp_path, &["mtp.a"]);

            let model_bytes = fs::metadata(&model_path).unwrap().len();
            let mtp_bytes = fs::metadata(&mtp_path).unwrap().len();
            let total_bytes = model_bytes + mtp_bytes;
            let weight_map = BTreeMap::from([
                ("main.a".to_owned(), MODEL_FILE.to_owned()),
                ("main.b".to_owned(), MODEL_FILE.to_owned()),
                ("mtp.a".to_owned(), MTP_FILE.to_owned()),
            ]);
            let index_bytes = write_index(&root, total_bytes, &weight_map);

            Self {
                base,
                root,
                weight_map,
                spec: SplitInventorySpec {
                    model_bytes,
                    model_tensors: 2,
                    mtp_bytes,
                    mtp_tensors: 1,
                    index_bytes,
                    index_entries: 3,
                },
            }
        }

        fn rewrite_index(&mut self, total_bytes: u64) {
            self.spec.index_bytes = write_index(&self.root, total_bytes, &self.weight_map);
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn write_config(root: &Path) {
        let layer_types = (0usize..64)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let config = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "bfloat16",
            "head_dim": 1,
            "image_token_id": 248056,
            "language_model_only": false,
            "model_type": "qwen3_5",
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "quantization_config": test_quantization_config(),
            "text_config": {
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 1,
                "hidden_size": 1,
                "intermediate_size": 1,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 1,
                "linear_key_head_dim": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_value_head_dim": 1,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 1,
                "num_hidden_layers": 64,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "vocab_size": 1
            },
            "video_token_id": 248057,
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 1,
                "dtype": "bfloat16",
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1,
                "in_channels": 1,
                "initializer_range": 0.02,
                "intermediate_size": 1,
                "model_type": "qwen3_5_vision",
                "num_heads": 1,
                "num_position_embeddings": 1,
                "out_hidden_size": 1,
                "patch_size": 1,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        });

        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, names: &[&str]) {
        let mut fixture = SafeTensorTestBuilder::new();

        for name in names {
            fixture.add_raw(*name, DType::U8, &[1], 0);
        }

        fixture.write(path);
    }

    fn write_index(root: &Path, total_bytes: u64, weight_map: &BTreeMap<String, String>) -> u64 {
        let index = json!({
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map
        });
        let bytes = serde_json::to_vec(&index).unwrap();
        fs::write(root.join(INDEX_FILE), &bytes).unwrap();
        bytes.len() as u64
    }

    #[test]
    fn exact_inventory_rejects_descriptor_drift() {
        let fixture = Fixture::new();
        let model = SafeTensorFile::open(&fixture.root.join(MODEL_FILE)).unwrap();
        let expected = BTreeMap::from([(
            String::from("main.a"),
            ExpectedTensor {
                dtype: DType::Bf16,
                shape: vec![1],
            },
        )]);

        let error = validate_exact_tensors(&model, &expected).unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("expected BF16 [1]"));
    }

    #[test]
    fn admits_complete_inventory_and_routes_tensors() {
        let fixture = Fixture::new();

        let snapshot =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .unwrap();

        assert_eq!(snapshot.root(), fixture.root);
        assert_eq!(snapshot.tensor_count(), 3);
        assert_eq!(snapshot.tensor("main.b").unwrap().bytes, &[0]);
        assert_eq!(snapshot.tensor("mtp.a").unwrap().bytes, &[0]);
        assert_eq!(
            snapshot
                .adjacent_tensor_bytes("main.a", "main.b", "main pair")
                .unwrap(),
            &[0, 0]
        );
    }

    #[test]
    fn rejects_tensor_span_across_shards() {
        let fixture = Fixture::new();
        let snapshot =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .unwrap();

        let error = snapshot
            .adjacent_tensor_bytes("main.b", "mtp.a", "cross-shard pair")
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("split across checkpoint shards"));
    }

    #[test]
    fn rejects_file_length_mismatch() {
        let mut fixture = Fixture::new();
        fixture.spec.model_bytes += 1;

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Inventory);
        assert!(error.to_string().contains("model.safetensors has"));
    }

    #[test]
    fn rejects_inventory_count_mismatch() {
        let fixture = Fixture::new();
        let mut index_spec = fixture.spec;
        index_spec.index_entries += 1;
        let mut model_spec = fixture.spec;
        model_spec.model_tensors += 1;
        let mut mtp_spec = fixture.spec;
        mtp_spec.mtp_tensors += 1;

        for (spec, field) in [
            (index_spec, "entries"),
            (model_spec, "model.safetensors tensors"),
            (mtp_spec, "model_mtp.safetensors tensors"),
        ] {
            let error = CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, spec)
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Inventory);
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_index_metadata_mismatch() {
        let mut fixture = Fixture::new();
        fixture.rewrite_index(fixture.spec.model_bytes + fixture.spec.mtp_bytes + 1);

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap()
                .to_string();

        assert!(error.contains("metadata.total_size"));
    }

    #[test]
    fn rejects_wrong_shard_assignment() {
        let mut fixture = Fixture::new();
        fixture
            .weight_map
            .insert("mtp.a".to_owned(), MODEL_FILE.to_owned());
        fixture.rewrite_index(fixture.spec.model_bytes + fixture.spec.mtp_bytes);

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(
            error
                .to_string()
                .contains("model.safetensors is missing tensor `mtp.a`")
        );
    }

    #[test]
    fn rejects_unpinned_revision_path() {
        let fixture = Fixture::new();
        let root = fixture.base.join("other-revision");

        let error = CheckpointSnapshot::<TestArch>::open_split_with_spec(&root, fixture.spec)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Revision);
        assert!(error.to_string().contains("expected \"test-revision\""));
    }
}
