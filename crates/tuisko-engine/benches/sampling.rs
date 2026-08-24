//! Host sampling microbenchmarks over the complete vocabulary.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tuisko_engine::{Sampler, SamplingOptions};
use tuisko_model::{Arch, Qwen38_27B};

const STOP_IDS: [u32; 2] = [248_046, 248_044];

fn bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn sampling(criterion: &mut Criterion) {
    let mut seed = 0xc0ff_ee12_3456_7890_u64;
    let mut logits = Vec::with_capacity(Qwen38_27B::VOCAB);
    for _ in 0..Qwen38_27B::VOCAB {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (seed >> 40) as f32 / (1u32 << 24) as f32;
        logits.push(bf16(-8.0 + 16.0 * unit));
    }
    logits[17] = bf16(10.0);

    let mut group = criterion.benchmark_group("engine/sampling");
    group.throughput(Throughput::Elements(Qwen38_27B::VOCAB as u64));

    let mut greedy = Sampler::new(SamplingOptions::greedy(), &STOP_IDS).unwrap();
    group.bench_function("greedy", |bencher| {
        bencher.iter(|| greedy.sample(black_box(&logits)).unwrap())
    });

    let mut default = Sampler::new(SamplingOptions::default(), &STOP_IDS).unwrap();
    group.bench_function("top-k20-top-p095", |bencher| {
        bencher.iter(|| default.sample(black_box(&logits)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, sampling);
criterion_main!(benches);
