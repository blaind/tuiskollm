//! Real-snapshot text frontend benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::env;
use std::hint::black_box;
use std::path::Path;
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, TextFrontend, TextFrontendOptions};
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

    let cache_disabled = TextFrontend::open_with_options(
        &snapshot,
        TextFrontendOptions {
            prompt_cache_capacity: 0,
        },
    )
    .unwrap();
    let identical_cache = TextFrontend::open(&snapshot).unwrap();
    let partial_cache = TextFrontend::open_with_options(
        &snapshot,
        TextFrontendOptions {
            prompt_cache_capacity: 4,
        },
    )
    .unwrap();
    let shared_user = "Explain this Unicode text: café 中文 テスト 🚀. ".repeat(32);
    let shared_messages = vec![
        ChatMessage::new("user", shared_user),
        ChatMessage::new("assistant", "It combines several Unicode scripts."),
    ];
    let mut variants = Vec::new();
    for index in 0..8 {
        let mut messages = shared_messages.clone();
        messages.push(ChatMessage::new(
            "user",
            format!("Give variation {index} without changing the earlier messages."),
        ));
        variants.push(messages);
    }
    let expected_tokens = cache_disabled
        .encode_chat(&variants[0], options)
        .unwrap()
        .len() as u64;
    identical_cache.encode_chat(&variants[0], options).unwrap();
    let identical_probe = identical_cache
        .encode_chat_with_report(&variants[0], options)
        .unwrap();
    assert_eq!(
        identical_probe.reused_tokens,
        identical_probe.token_ids.len()
    );
    for messages in variants.iter().take(4) {
        partial_cache.encode_chat(messages, options).unwrap();
    }
    let partial_probe = partial_cache
        .encode_chat_with_report(&variants[4], options)
        .unwrap();
    assert!(partial_probe.reused_tokens > 0);
    assert!(partial_probe.fresh_bytes > 0);

    let mut encode_chat = criterion.benchmark_group("frontend/encode_chat");
    encode_chat.throughput(Throughput::Elements(expected_tokens));
    encode_chat.bench_function("cache-disabled", |bencher| {
        bencher.iter(|| {
            cache_disabled
                .encode_chat(black_box(&variants[0]), options)
                .unwrap()
        });
    });
    encode_chat.bench_function("identical-hit", |bencher| {
        bencher.iter(|| {
            identical_cache
                .encode_chat_with_report(black_box(&variants[0]), options)
                .unwrap()
        });
    });
    let mut variant = 0;
    encode_chat.bench_function("partial-hit", |bencher| {
        bencher.iter(|| {
            let messages = &variants[variant % variants.len()];
            variant += 1;
            partial_cache
                .encode_chat_with_report(black_box(messages), options)
                .unwrap()
        });
    });
    encode_chat.finish();

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
