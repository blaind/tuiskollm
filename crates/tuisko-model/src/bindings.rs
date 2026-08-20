use crate::{
    Arch, Bf16View, CheckpointError, CheckpointResult, CheckpointSnapshot, F32View, Fp8E4M3View,
    TensorView, U8View,
};

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LM_HEAD_SCALE: &str = "lm_head.weight_scale";

// These are source-codec facts, not architecture geometry.
const NVFP4_MLP_LAYER_END: usize = 56;
const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;

/// Exact packed gate/up source planes for one NVFP4 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4GateUpBindings<'a> {
    /// Packed E2M1 gate weights `[intermediate, hidden / 2]`.
    pub gate_weight: U8View<'a, 2>,
    /// Packed E2M1 up weights `[intermediate, hidden / 2]`.
    pub up_weight: U8View<'a, 2>,
    /// E4M3 gate block scales `[intermediate, hidden / 16]`.
    pub gate_scale: Fp8E4M3View<'a, 2>,
    /// E4M3 up block scales `[intermediate, hidden / 16]`.
    pub up_scale: Fp8E4M3View<'a, 2>,
    /// Shared finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Shared finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Decoder layer owning these planes.
    pub layer: usize,
}

impl<'a> Nvfp4GateUpBindings<'a> {
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(layer, |name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_nvfp4_mlp_layer::<A>(layer)?;

        let intermediate = A::INTERMEDIATE as u64;
        let packed_columns = codec_columns(A::HIDDEN, E2M1_VALUES_PER_BYTE, "packed E2M1")?;
        let scale_columns = codec_columns(A::HIDDEN, NVFP4_GROUP_SIZE, "E4M3 block-scale")?;
        let prefix = format!("model.language_model.layers.{layer}.mlp");

        let gate_weight = tensor(&format!("{prefix}.gate_proj.weight_packed"))?;
        let up_weight = tensor(&format!("{prefix}.up_proj.weight_packed"))?;
        let gate_scale = tensor(&format!("{prefix}.gate_proj.weight_scale"))?;
        let up_scale = tensor(&format!("{prefix}.up_proj.weight_scale"))?;

        require_adjacent(layer, "packed gate/up weights", &gate_weight, &up_weight)?;
        require_adjacent(layer, "gate/up scale planes", &gate_scale, &up_scale)?;

        let gate_weight = U8View::bind(gate_weight, [intermediate, packed_columns])?;
        let up_weight = U8View::bind(up_weight, [intermediate, packed_columns])?;
        let gate_scale = Fp8E4M3View::bind(gate_scale, [intermediate, scale_columns])?;
        let up_scale = Fp8E4M3View::bind(up_scale, [intermediate, scale_columns])?;

        validate_nvfp4_scales(layer, "gate", gate_scale.codes())?;
        validate_nvfp4_scales(layer, "up", up_scale.codes())?;

        let gate_input_divisor =
            positive_f32(tensor(&format!("{prefix}.gate_proj.input_global_scale"))?)?;
        let up_input_divisor =
            positive_f32(tensor(&format!("{prefix}.up_proj.input_global_scale"))?)?;
        let gate_weight_divisor =
            positive_f32(tensor(&format!("{prefix}.gate_proj.weight_global_scale"))?)?;
        let up_weight_divisor =
            positive_f32(tensor(&format!("{prefix}.up_proj.weight_global_scale"))?)?;

        require_same_divisor(
            layer,
            "input_global_scale",
            gate_input_divisor,
            up_input_divisor,
        )?;
        require_same_divisor(
            layer,
            "weight_global_scale",
            gate_weight_divisor,
            up_weight_divisor,
        )?;

        Ok(Self {
            gate_weight,
            up_weight,
            gate_scale,
            up_scale,
            input_scale_divisor: gate_input_divisor,
            weight_scale_divisor: gate_weight_divisor,
            layer,
        })
    }
}

/// Exact packed down-projection source planes for one NVFP4 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4DownBindings<'a> {
    /// Packed E2M1 weights `[hidden, intermediate / 2]`.
    pub weight: U8View<'a, 2>,
    /// E4M3 block scales `[hidden, intermediate / 16]`.
    pub scale: Fp8E4M3View<'a, 2>,
    /// Finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Decoder layer owning these planes.
    pub layer: usize,
}

impl<'a> Nvfp4DownBindings<'a> {
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(layer, |name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_nvfp4_mlp_layer::<A>(layer)?;

        let hidden = A::HIDDEN as u64;
        let packed_columns = codec_columns(A::INTERMEDIATE, E2M1_VALUES_PER_BYTE, "packed E2M1")?;
        let scale_columns = codec_columns(A::INTERMEDIATE, NVFP4_GROUP_SIZE, "E4M3 block-scale")?;
        let prefix = format!("model.language_model.layers.{layer}.mlp.down_proj");

        let weight = U8View::bind(
            tensor(&format!("{prefix}.weight_packed"))?,
            [hidden, packed_columns],
        )?;
        let scale = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.weight_scale"))?,
            [hidden, scale_columns],
        )?;

        validate_nvfp4_scales(layer, "down", scale.codes())?;

        let input_scale_divisor = positive_f32(tensor(&format!("{prefix}.input_global_scale"))?)?;
        let weight_scale_divisor = positive_f32(tensor(&format!("{prefix}.weight_global_scale"))?)?;

        Ok(Self {
            weight,
            scale,
            input_scale_divisor,
            weight_scale_divisor,
            layer,
        })
    }
}

/// Shape- and dtype-checked source views for the text input and output endpoints.
#[derive(Clone, Copy, Debug)]
pub struct TextEndpointBindings<'a> {
    /// BF16 token embedding matrix `[vocab, hidden]`.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 final RMSNorm weights `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
    /// FP8 E4M3 language-model head `[vocab, hidden]`.
    pub lm_head: Fp8E4M3View<'a, 2>,
    /// Per-vocabulary-row BF16 language-model head scales `[vocab, 1]`.
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

fn require_nvfp4_mlp_layer<A: Arch>(layer: usize) -> CheckpointResult<()> {
    if layer >= A::LAYERS || layer >= NVFP4_MLP_LAYER_END {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted NVFP4 MLP source contract"
        )));
    }

    Ok(())
}

fn codec_columns(width: usize, divisor: usize, role: &str) -> CheckpointResult<u64> {
    if !width.is_multiple_of(divisor) {
        return Err(CheckpointError::source_binding(format!(
            "architecture width {width} is not divisible by the {role} divisor {divisor}"
        )));
    }

    Ok((width / divisor) as u64)
}

fn require_adjacent(
    layer: usize,
    role: &str,
    first: &TensorView<'_>,
    second: &TensorView<'_>,
) -> CheckpointResult<()> {
    if first.data_range.end != second.data_range.start {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} {role} are not source-adjacent"
        )));
    }

    Ok(())
}

fn validate_nvfp4_scales(layer: usize, role: &str, scales: &[u8]) -> CheckpointResult<()> {
    if scales
        .iter()
        .any(|&scale| scale & 0x80 != 0 || scale == 0x7f)
    {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale plane contains a negative or NaN E4M3FN code"
        )));
    }

    Ok(())
}

fn positive_f32(tensor: TensorView<'_>) -> CheckpointResult<f32> {
    let view = F32View::bind(tensor, [1])?;
    let value = view.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "tensor `{}` must contain a finite positive divisor, observed {value}",
            view.name()
        )));
    }

    Ok(value)
}

fn require_same_divisor(layer: usize, role: &str, gate: f32, up: f32) -> CheckpointResult<()> {
    if gate.to_bits() != up.to_bits() {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} gate/up {role} words differ and cannot share one fused operator"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Nvfp4DownBindings, Nvfp4GateUpBindings, TextEndpointBindings, require_adjacent,
        require_nvfp4_mlp_layer, validate_nvfp4_scales,
    };
    use crate::{Arch, CheckpointErrorCode, DType, SafeTensorFile, TensorView};
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

    #[derive(Clone, Copy)]
    struct Nvfp4Arch;

    impl Arch for Nvfp4Arch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 32;
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
        let payload = (0u8..50).collect::<Vec<_>>();

        write_safetensors_payload(path, header, &payload);
    }

    fn write_safetensors_payload(path: &Path, header: Value, payload: &[u8]) {
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

    fn nvfp4_mlp_fixture(layer: usize) -> (Value, Vec<u8>) {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let down = format!("{prefix}.down_proj");
        let mut payload = vec![0x38; 888];

        payload[0..4].copy_from_slice(&3.0f32.to_le_bytes());
        payload[4..8].copy_from_slice(&0.125f32.to_le_bytes());
        payload[8..12].copy_from_slice(&3.0f32.to_le_bytes());
        payload[12..16].copy_from_slice(&0.125f32.to_le_bytes());
        payload[592..596].copy_from_slice(&19.0f32.to_le_bytes());
        payload[596..600].copy_from_slice(&3_376.0f32.to_le_bytes());

        (
            json!({
                format!("{prefix}.gate_proj.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[0,4]
                },
                format!("{prefix}.gate_proj.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[4,8]
                },
                format!("{prefix}.up_proj.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[8,12]
                },
                format!("{prefix}.up_proj.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[12,16]
                },
                format!("{prefix}.gate_proj.weight_packed"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[16,272]
                },
                format!("{prefix}.up_proj.weight_packed"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[272,528]
                },
                format!("{prefix}.gate_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[528,560]
                },
                format!("{prefix}.up_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[560,592]
                },
                format!("{down}.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[592,596]
                },
                format!("{down}.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[596,600]
                },
                format!("{down}.weight_packed"): {
                    "dtype":"U8", "shape":[32,8], "data_offsets":[600,856]
                },
                format!("{down}.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[32,1], "data_offsets":[856,888]
                }
            }),
            payload,
        )
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
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("lm_head.weight"));
        assert!(error.to_string().contains("dtype `U8`, expected `F8_E4M3`"));

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

    #[test]
    fn binds_exact_nvfp4_mlp_source_contract() {
        let path = fixture_path("nvfp4-mlp");
        let (header, payload) = nvfp4_mlp_fixture(55);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let gate_up =
            Nvfp4GateUpBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name)).unwrap();
        let down = Nvfp4DownBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name)).unwrap();

        assert_eq!(gate_up.gate_weight.shape(), &[16, 16]);
        assert_eq!(gate_up.up_weight.shape(), &[16, 16]);
        assert_eq!(gate_up.gate_scale.shape(), &[16, 2]);
        assert_eq!(gate_up.up_scale.shape(), &[16, 2]);
        assert_eq!(gate_up.input_scale_divisor.to_bits(), 3.0f32.to_bits());
        assert_eq!(gate_up.weight_scale_divisor.to_bits(), 0.125f32.to_bits());
        assert_eq!(down.weight.shape(), &[32, 8]);
        assert_eq!(down.scale.shape(), &[32, 1]);
        assert_eq!(down.input_scale_divisor.to_bits(), 19.0f32.to_bits());
        assert_eq!(down.weight_scale_divisor.to_bits(), 3_376.0f32.to_bits());
        assert_eq!((gate_up.layer, down.layer), (55, 55));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn nvfp4_layer_route_is_exact() {
        for (layer, admitted) in [(0, true), (55, true), (56, false), (63, false), (64, false)] {
            assert_eq!(
                require_nvfp4_mlp_layer::<Nvfp4Arch>(layer).is_ok(),
                admitted,
                "layer {layer}"
            );
        }
    }

    #[test]
    fn rejects_nvfp4_gate_up_with_different_divisors() {
        let path = fixture_path("nvfp4-divisor");
        let (header, mut payload) = nvfp4_mlp_fixture(55);
        payload[8..12].copy_from_slice(&4.0f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = Nvfp4GateUpBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name))
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("input_global_scale words differ")
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_nvfp4_scale_codes() {
        let error = validate_nvfp4_scales(55, "gate", &[0x38, 0x7f])
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("negative or NaN E4M3FN code"));
    }

    #[test]
    fn rejects_nonadjacent_fused_source_planes() {
        let first = TensorView {
            name: "gate",
            dtype: DType::U8,
            shape: &[1],
            bytes: &[0],
            data_range: 0..1,
        };
        let second = TensorView {
            name: "up",
            dtype: DType::U8,
            shape: &[1],
            bytes: &[0],
            data_range: 2..3,
        };

        let error = require_adjacent(55, "packed gate/up weights", &first, &second)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("not source-adjacent"));
    }
}
