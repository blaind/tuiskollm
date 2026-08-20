use crate::{Arch, Bf16View, CheckpointResult, CheckpointSnapshot, Fp8E4M3View, TensorView};

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LM_HEAD_SCALE: &str = "lm_head.weight_scale";

#[derive(Clone, Copy, Debug)]
pub struct TextEndpointBindings<'a> {
    pub embedding: Bf16View<'a, 2>,
    pub final_norm: Bf16View<'a, 1>,
    pub lm_head: Fp8E4M3View<'a, 2>,
    pub lm_head_scale: Bf16View<'a, 2>,
}

impl<'a> TextEndpointBindings<'a> {
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(|name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let vocab = A::VOCAB as u64;
        let hidden = A::HIDDEN as u64;

        let embedding = Bf16View::bind(tensor(EMBEDDING)?, [vocab, hidden])?;
        let final_norm = Bf16View::bind(tensor(FINAL_NORM)?, [hidden])?;
        let lm_head = Fp8E4M3View::bind(tensor(LM_HEAD)?, [vocab, hidden])?;
        let lm_head_scale = Bf16View::bind(tensor(LM_HEAD_SCALE)?, [vocab, 1])?;

        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            lm_head_scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TextEndpointBindings;
    use crate::{Arch, SafeTensorFile};
    use serde_json::{Value, json};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 4;
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
    }

    fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-bindings-{label}-{}-{}.safetensors",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn endpoint_header() -> Value {
        json!({
            "model.language_model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [3, 4],
                "data_offsets": [0, 24]
            },
            "model.language_model.norm.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [24, 32]
            },
            "lm_head.weight": {
                "dtype": "F8_E4M3",
                "shape": [3, 4],
                "data_offsets": [32, 44]
            },
            "lm_head.weight_scale": {
                "dtype": "BF16",
                "shape": [3, 1],
                "data_offsets": [44, 50]
            }
        })
    }

    fn write_safetensors(path: &Path, header: Value) {
        let mut header = serde_json::to_vec(&header).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = File::create(path).unwrap();
        let payload = (0u8..50).collect::<Vec<_>>();

        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload).unwrap();
    }

    #[test]
    fn binds_exact_text_endpoint_contract() {
        let path = fixture_path("valid");
        write_safetensors(&path, endpoint_header());
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name)).unwrap();

        assert_eq!(bindings.embedding.shape(), &[3, 4]);
        assert_eq!(bindings.embedding.word(0), Some(0x0100));
        assert_eq!(bindings.final_norm.shape(), &[4]);
        assert_eq!(bindings.lm_head.shape(), &[3, 4]);
        assert_eq!(bindings.lm_head.codes(), &(32u8..44).collect::<Vec<_>>());
        assert_eq!(bindings.lm_head_scale.shape(), &[3, 1]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_endpoint_dtype_mismatch() {
        let path = fixture_path("dtype");
        let mut header = endpoint_header();
        header["lm_head.weight"]["dtype"] = json!("U8");
        write_safetensors(&path, header);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name))
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("lm_head.weight"));
        assert!(error.contains("dtype `U8`, expected `F8_E4M3`"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_endpoint_shape_mismatch() {
        let path = fixture_path("shape");
        let mut header = endpoint_header();
        header["lm_head.weight_scale"]["shape"] = json!([3]);
        write_safetensors(&path, header);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name))
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("lm_head.weight_scale"));
        assert!(error.contains("shape [3], expected [3, 1]"));

        fs::remove_file(path).unwrap();
    }
}
