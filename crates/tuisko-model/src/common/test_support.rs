//! Synthetic source and config fixtures shared by the split target test modules.

/// Synthetic safetensors sources shared by the target binding and materialization tests.
pub(crate) mod sources {
    use crate::common::scale_swizzle::{SCALE_TILE_GROUPS, SCALE_TILE_ROWS};
    use crate::{Arch, Bf16View, DType, F32View, Fp8E4M3View, TensorView, U8View};
    use serde_json::{Value, json};
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    pub(crate) struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 4;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 8;
        const VOCAB: usize = 3;
        const LAYERS: usize = 1;
        const FULL_ATTENTION_INTERVAL: usize = 1;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 1;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 2;
        const VISION_HIDDEN: usize = 4;
        const VISION_INTERMEDIATE: usize = 6;
        const VISION_NUM_HEADS: usize = 2;
        const VISION_POSITIONS: usize = 8;
        const VISION_OUTPUT_HIDDEN: usize = 4;
        const VISION_INPUT_CHANNELS: usize = 3;
        const VISION_PATCH_SIZE: usize = 2;
        const VISION_SPATIAL_MERGE_SIZE: usize = 2;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Nvfp4Arch;

    impl Arch for Nvfp4Arch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 32;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 16;
        const VOCAB: usize = 3;
        const LAYERS: usize = 64;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 4;
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

    pub(crate) fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-bindings-{label}-{}-{}.safetensors",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    pub(crate) fn write_safetensors_payload(path: &Path, header: Value, payload: &[u8]) {
        let mut header = serde_json::to_vec(&header).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = File::create(path).unwrap();

        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
    }

    pub(crate) fn append_bf16_tensor(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        name: impl Into<String>,
        shape: Vec<usize>,
    ) {
        let begin = payload.len();
        let elements = shape.iter().product::<usize>();
        let word = u16::try_from(header.len() + 1).unwrap().to_le_bytes();

        for _ in 0..elements {
            payload.extend_from_slice(&word);
        }

        header.insert(
            name.into(),
            json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    pub(crate) const ROWS: usize = 128;
    pub(crate) const GROUPS: usize = 8;
    pub(crate) const COLUMNS: usize = GROUPS * 16;
    pub(crate) const PACKED_COLUMNS: usize = COLUMNS / 2;

    pub(crate) fn u8_view<'a>(
        name: &'a str,
        shape: &'a [u64; 2],
        bytes: &'a [u8],
    ) -> U8View<'a, 2> {
        U8View::bind(
            TensorView {
                name,
                dtype: DType::U8,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    pub(crate) fn fp8_view<'a>(
        name: &'a str,
        shape: &'a [u64; 2],
        bytes: &'a [u8],
    ) -> Fp8E4M3View<'a, 2> {
        Fp8E4M3View::bind(
            TensorView {
                name,
                dtype: DType::Fp8E4M3,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    pub(crate) fn bf16_view<'a>(
        name: &'a str,
        shape: &'a [u64; 2],
        bytes: &'a [u8],
    ) -> Bf16View<'a, 2> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    pub(crate) fn bf16_vector<'a>(
        name: &'a str,
        shape: &'a [u64; 1],
        bytes: &'a [u8],
    ) -> Bf16View<'a, 1> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    pub(crate) fn bf16_volume<'a>(
        name: &'a str,
        shape: &'a [u64; 3],
        bytes: &'a [u8],
    ) -> Bf16View<'a, 3> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    pub(crate) fn f32_scalar_view<'a>(name: &'a str, bytes: &'a [u8; 4]) -> F32View<'a, 0> {
        F32View::bind(
            TensorView {
                name,
                dtype: DType::F32,
                shape: &[],
                bytes,
                data_range: 0..4,
            },
            [],
        )
        .unwrap()
    }

    pub(crate) fn bf16_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    pub(crate) fn scale_codes(seed: usize) -> Vec<u8> {
        scale_codes_for(ROWS, seed)
    }

    pub(crate) fn scale_codes_for(rows: usize, seed: usize) -> Vec<u8> {
        (0..rows * GROUPS)
            .map(|index| ((index * 37 + seed) % 0x7f) as u8)
            .collect()
    }

    pub(crate) fn block_scale_oracle(source: &[u8], rows: usize, groups: usize) -> Vec<u8> {
        let mut expected = Vec::with_capacity(source.len());

        for row_tile in 0..rows / SCALE_TILE_ROWS {
            for group_tile in 0..groups / SCALE_TILE_GROUPS {
                for row_mod32 in 0..32 {
                    for row_quartile in 0..4 {
                        for scale_lane in 0..SCALE_TILE_GROUPS {
                            let row = row_tile * SCALE_TILE_ROWS + row_quartile * 32 + row_mod32;
                            let group = group_tile * SCALE_TILE_GROUPS + scale_lane;
                            expected.push(source[row * groups + group]);
                        }
                    }
                }
            }
        }

        expected
    }
}

/// Synthetic `config.json` documents shared by the target config tests.
pub(crate) mod configs {
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(crate) static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-model-{label}-{}-{}.json",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    pub(crate) fn write_config(label: &str, config: &Value) -> PathBuf {
        let path = fixture_path(label);
        fs::write(&path, serde_json::to_vec(config).unwrap()).unwrap();
        path
    }
}
