//! Real-snapshot text frontend benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::env;
use std::hint::black_box;
use std::path::Path;
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, TextFrontend};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

fn text_frontend(criterion: &mut Criterion) {
    let root = env::var_os("TUISKO_SNAPSHOT")
        .expect("set TUISKO_SNAPSHOT to the pinned Hugging Face snapshot directory");
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&root)).unwrap();
    let frontend = TextFrontend::open(&snapshot).unwrap();
    let options = ChatTemplateOptions {
        enable_thinking: Some(false),
    };

    let short_messages = [ChatMessage::new("user", "Hello")];
    let long_text = "Hello! café naïve 中文 テスト тест 🚀 ".repeat(64);
    let long_messages = [ChatMessage::new("user", long_text)];
    let short_rendered = frontend
        .render_chat(&short_messages, true, options)
        .unwrap();
    let long_rendered = frontend.render_chat(&long_messages, true, options).unwrap();
    let long_ids = frontend.encode(&long_rendered).unwrap();

    let mut render = criterion.benchmark_group("frontend/render_chat");
    render.bench_function("hello", |bencher| {
        bencher.iter(|| {
            frontend
                .render_chat(black_box(&short_messages), true, options)
                .unwrap()
        });
    });
    render.finish();

    let mut encode = criterion.benchmark_group("frontend/encode");
    for (name, rendered) in [("hello", &short_rendered), ("unicode-long", &long_rendered)] {
        encode.throughput(Throughput::Bytes(rendered.len() as u64));
        encode.bench_with_input(
            BenchmarkId::from_parameter(name),
            rendered,
            |bencher, text| {
                bencher.iter(|| frontend.encode(black_box(text)).unwrap());
            },
        );
    }
    encode.finish();

    let mut decode = criterion.benchmark_group("frontend/decode");
    decode.throughput(Throughput::Elements(long_ids.len() as u64));
    decode.bench_function("batch-unicode-long", |bencher| {
        bencher.iter(|| frontend.decode(black_box(&long_ids), true).unwrap());
    });
    decode.bench_function("stream-unicode-long", |bencher| {
        bencher.iter(|| {
            let mut decoder = frontend.streaming_decoder();
            for &token in black_box(&long_ids) {
                black_box(decoder.push(token).unwrap());
            }
            black_box(decoder.finish());
            decoder
        });
    });
    decode.bench_function("prefix-redecode-unicode-long", |bencher| {
        bencher.iter(|| {
            let mut prefix = Vec::with_capacity(long_ids.len());
            let mut text = String::new();
            for &token in black_box(&long_ids) {
                prefix.push(token);
                text = frontend.decode(&prefix, true).unwrap();
            }
            black_box(text)
        });
    });
    decode.finish();
}

criterion_group!(benches, text_frontend);
criterion_main!(benches);
