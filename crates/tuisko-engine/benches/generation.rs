//! Host generation-control benchmarks over the pinned frontend.

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::env;
use std::hint::black_box;
use std::path::Path;
use tuisko_engine::{ChatGenerationRequest, GenerationSession, SamplingOptions};
use tuisko_frontend::{ChatMessage, TextFrontend};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

fn generation(criterion: &mut Criterion) {
    let root = env::var("TUISKO_SNAPSHOT").expect("set TUISKO_SNAPSHOT to the pinned snapshot");
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&root)).unwrap();
    let frontend = TextFrontend::open(&snapshot).unwrap();
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = 1;
    let prompt_tokens = GenerationSession::start(&frontend, &request)
        .unwrap()
        .prompt_token_ids()
        .len() as u64;
    let selected = frontend.encode("Hello").unwrap()[0];
    let mut logits = vec![0xbf80; Qwen38_27B::VOCAB];
    logits[selected as usize] = 0x3f80;

    let mut start = criterion.benchmark_group("engine/generation/start");
    start.throughput(Throughput::Elements(prompt_tokens));
    start.bench_function("identical-prefix", |bencher| {
        bencher
            .iter(|| GenerationSession::start(black_box(&frontend), black_box(&request)).unwrap())
    });
    start.finish();

    let mut step = criterion.benchmark_group("engine/generation/step");
    step.throughput(Throughput::Elements(Qwen38_27B::VOCAB as u64));
    step.bench_function("greedy", |bencher| {
        bencher.iter_batched(
            || GenerationSession::start(&frontend, &request).unwrap(),
            |mut session| session.accept_logits(black_box(&logits)).unwrap(),
            BatchSize::SmallInput,
        )
    });
    step.finish();
}

criterion_group!(benches, generation);
criterion_main!(benches);
