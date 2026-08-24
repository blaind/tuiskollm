//! Host benchmarks for checkpoint admission and NVFP4 scale materialization.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tuisko_model::{
    Arch, Bf16View, DType, Fp8E4M3View, FullAttentionQkvBindings, Nvfp4DownBindings,
    Nvfp4GateUpBindings, Qwen38_27B, SafeTensorFile, TensorView, U8View, validate_config,
};

const TENSOR_COUNT: usize = 1_953;
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    config: PathBuf,
    safetensors: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tuisko-checkpoint-bench-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();

        let config = root.join("config.json");
        let safetensors = root.join("model.safetensors");
        write_config(&config);
        write_safetensors(&safetensors);

        Self {
            root,
            config,
            safetensors,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn write_config(path: &Path) {
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
        "head_dim": 256,
        "image_token_id": 248056,
        "ignored_padding": "x".repeat(20_000),
        "language_model_only": false,
        "model_type": "qwen3_5",
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "quantization_config": serde_json::from_str::<Value>(include_str!(
            "../fixtures/quantization-config.json"
        )).unwrap(),
        "text_config": {
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "vocab_size": 248320
        },
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_vision",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053
    });

    fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
}

fn write_safetensors(path: &Path) {
    let mut header = serde_json::Map::new();

    for offset in 0..TENSOR_COUNT {
        header.insert(
            format!("tensor.{offset:04}"),
            json!({
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [offset, offset + 1]
            }),
        );
    }

    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();

    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }

    let mut file = File::create(path).unwrap();

    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0; TENSOR_COUNT]).unwrap();
}

fn checkpoint_benches(criterion: &mut Criterion) {
    let fixture = Fixture::new();

    criterion.bench_function("checkpoint/config-admission", |bencher| {
        bencher
            .iter(|| validate_config::<Qwen38_27B>(black_box(fixture.config.as_path())).unwrap());
    });

    let mut group = criterion.benchmark_group("checkpoint/safetensors-header-admission");
    group.sample_size(60);
    group.throughput(Throughput::Elements(TENSOR_COUNT as u64));
    group.bench_function(TENSOR_COUNT.to_string(), |bencher| {
        bencher.iter(|| {
            let file = SafeTensorFile::open(black_box(fixture.safetensors.as_path())).unwrap();

            black_box(file.tensor_count());
        });
    });
    group.finish();

    nvfp4_materialization_benches(criterion);
    qkv_materialization_bench(criterion);
}

fn tensor_view<'a>(
    name: &'a str,
    dtype: DType,
    shape: &'a [u64],
    bytes: &'a [u8],
) -> TensorView<'a> {
    TensorView {
        name,
        dtype,
        shape,
        bytes,
        data_range: 0..bytes.len() as u64,
    }
}

fn nvfp4_materialization_benches(criterion: &mut Criterion) {
    let hidden = Qwen38_27B::HIDDEN as u64;
    let intermediate = Qwen38_27B::INTERMEDIATE as u64;

    let gate_weight = vec![0x5a; (intermediate * (hidden / 2)) as usize];
    let up_weight = vec![0x5a; (intermediate * (hidden / 2)) as usize];
    let gate_scale = vec![0x30; (intermediate * (hidden / 16)) as usize];
    let up_scale = vec![0x30; (intermediate * (hidden / 16)) as usize];
    let down_weight = vec![0x5a; (hidden * (intermediate / 2)) as usize];
    let down_scale = vec![0x30; (hidden * (intermediate / 16)) as usize];

    let gate_shape = [intermediate, hidden / 2];
    let gate_scale_shape = [intermediate, hidden / 16];
    let down_shape = [hidden, intermediate / 2];
    let down_scale_shape = [hidden, intermediate / 16];

    let gate_up = Nvfp4GateUpBindings {
        gate_weight: U8View::bind(
            tensor_view("gate", DType::U8, &gate_shape, &gate_weight),
            gate_shape,
        )
        .unwrap(),
        up_weight: U8View::bind(
            tensor_view("up", DType::U8, &gate_shape, &up_weight),
            gate_shape,
        )
        .unwrap(),
        gate_scale: Fp8E4M3View::bind(
            tensor_view("gate_scale", DType::Fp8E4M3, &gate_scale_shape, &gate_scale),
            gate_scale_shape,
        )
        .unwrap(),
        up_scale: Fp8E4M3View::bind(
            tensor_view("up_scale", DType::Fp8E4M3, &gate_scale_shape, &up_scale),
            gate_scale_shape,
        )
        .unwrap(),
        input_scale_divisor: 1.0,
        weight_scale_divisor: 1.0,
        layer: 0,
    };

    let down = Nvfp4DownBindings {
        weight: U8View::bind(
            tensor_view("down", DType::U8, &down_shape, &down_weight),
            down_shape,
        )
        .unwrap(),
        scale: Fp8E4M3View::bind(
            tensor_view("down_scale", DType::Fp8E4M3, &down_scale_shape, &down_scale),
            down_scale_shape,
        )
        .unwrap(),
        input_scale_divisor: 1.0,
        weight_scale_divisor: 1.0,
        layer: 0,
    };

    let mut group = criterion.benchmark_group("checkpoint/nvfp4-scale-materialization");
    group.sample_size(60);
    group.throughput(Throughput::Bytes(gate_scale.len() as u64 * 2));
    group.bench_function("gate-up", |bencher| {
        bencher.iter(|| black_box(gate_up.materialize().unwrap()));
    });
    group.throughput(Throughput::Bytes(down_scale.len() as u64));
    group.bench_function("down", |bencher| {
        bencher.iter(|| black_box(down.materialize().unwrap()));
    });
    group.finish();
}

fn qkv_materialization_bench(criterion: &mut Criterion) {
    let query_rows = Qwen38_27B::ATTENTION_QUERY_ROWS as u64;
    let kv_rows = Qwen38_27B::ATTENTION_KV_ROWS as u64;
    let hidden = Qwen38_27B::HIDDEN as u64;
    let query_shape = [query_rows, hidden];
    let kv_shape = [kv_rows, hidden];
    let query_scale_shape = [query_rows, 1];
    let kv_scale_shape = [kv_rows, 1];
    let query_weight = vec![0x20; (query_rows * hidden) as usize];
    let key_weight = vec![0x30; (kv_rows * hidden) as usize];
    let value_weight = vec![0x40; (kv_rows * hidden) as usize];
    let query_scale = [0x80, 0x3f].repeat(query_rows as usize);
    let key_scale = [0x80, 0x3f].repeat(kv_rows as usize);
    let value_scale = [0x80, 0x3f].repeat(kv_rows as usize);
    let bindings = FullAttentionQkvBindings::from_views::<Qwen38_27B>(
        3,
        [
            Fp8E4M3View::bind(
                tensor_view("query", DType::Fp8E4M3, &query_shape, &query_weight),
                query_shape,
            )
            .unwrap(),
            Fp8E4M3View::bind(
                tensor_view("key", DType::Fp8E4M3, &kv_shape, &key_weight),
                kv_shape,
            )
            .unwrap(),
            Fp8E4M3View::bind(
                tensor_view("value", DType::Fp8E4M3, &kv_shape, &value_weight),
                kv_shape,
            )
            .unwrap(),
        ],
        [
            Bf16View::bind(
                tensor_view("query-scale", DType::Bf16, &query_scale_shape, &query_scale),
                query_scale_shape,
            )
            .unwrap(),
            Bf16View::bind(
                tensor_view("key-scale", DType::Bf16, &kv_scale_shape, &key_scale),
                kv_scale_shape,
            )
            .unwrap(),
            Bf16View::bind(
                tensor_view("value-scale", DType::Bf16, &kv_scale_shape, &value_scale),
                kv_scale_shape,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let bytes = query_weight.len()
        + key_weight.len()
        + value_weight.len()
        + query_scale.len()
        + key_scale.len()
        + value_scale.len();
    let mut group = criterion.benchmark_group("checkpoint/full-attention-qkv-materialization");
    group.sample_size(40);
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("weights", |bencher| {
        bencher.iter(|| black_box(bindings.materialize().unwrap()));
    });
    group.finish();
}

criterion_group!(benches, checkpoint_benches);
criterion_main!(benches);
