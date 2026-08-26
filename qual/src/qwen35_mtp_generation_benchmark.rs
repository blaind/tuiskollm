//! Direct host-completion timing for one production Qwen3.5 MTP request.

use crate::device_benchmark::{
    BenchmarkMemoryKind, DeviceBenchmarkError, DeviceBenchmarkOptions, DeviceBenchmarkReport,
    MemoryRecorder,
};
use crate::harness::benchmark_session::{
    MtpGreedyBenchmarkSpec, admit_greedy_mtp_benchmark, greedy_request, open_device_zero,
    run_greedy_mtp_benchmark,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::Qwen35ResidentMtpTextGenerator;
use tuisko_model::{CheckpointSnapshot, Qwen35_9B};

const OUTPUT_TOKENS: usize = 8;

const SPEC: MtpGreedyBenchmarkSpec = MtpGreedyBenchmarkSpec {
    batch_refusal: "Qwen3.5 MTP generation benchmark admits only B=1",
    energy_refusal: "Qwen3.5 MTP request energy belongs to the full-server gate",
    warmup_width_drift: "Qwen3.5 MTP committed width changed during warmup",
    sample_width_drift: "Qwen3.5 MTP committed width changed between samples",
    missing_warmup_request: "Qwen3.5 MTP benchmark requires one warmup request",
    unwarmed_round: "Qwen3.5 MTP round was not warmed",
    output_drift: "Qwen3.5 MTP output changed between samples",
    k4_refusal: "Qwen3.5 benchmark did not execute a draft-three/K=4 round",
    round_metric: "qwen3_5/generation/mtp_greedy_round",
    request_metric: "qwen3_5/generation/mtp_greedy_request",
    suite: "bench-qwen35-mtp-generation",
    classification: "performance_sensitive_model",
    timing_scope: "direct Rust host completion for Qwen3.5 prompt prime, draft-three, exact-K target verification/commit, MTP realignment, and streaming control",
};

/// Measures full Qwen3.5 MTP requests and one draft-three transaction directly.
pub fn benchmark_qwen35_mtp_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let mut preamble = admit_greedy_mtp_benchmark(options, &SPEC)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let context = open_device_zero()?;
    let mut generator = Qwen35ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
    register_memory(&mut preamble.memory, &generator)?;
    preamble.memory.capture("after_setup")?;

    run_greedy_mtp_benchmark(
        &mut generator,
        &greedy_request(OUTPUT_TOKENS),
        preamble,
        options,
        &SPEC,
    )
}

fn register_memory(
    memory: &mut MemoryRecorder,
    generator: &Qwen35ResidentMtpTextGenerator,
) -> Result<(), DeviceBenchmarkError> {
    let layout = generator.qualification_program().layout();
    for (name, kind, bytes, description) in [
        (
            "qwen35_generation_mtp/weights",
            BenchmarkMemoryKind::Weights,
            layout.resident_weight_bytes(),
            "32 target layers, one BF16 MTP layer, and one shared BF16 endpoint",
        ),
        (
            "qwen35_generation_mtp/cache",
            BenchmarkMemoryKind::KvCache,
            layout.cache_bytes(),
            "target and mirrored MTP 262,144-position BF16 K/V pools",
        ),
        (
            "qwen35_generation_mtp/workspace",
            BenchmarkMemoryKind::Workspace,
            layout.workspace_bytes(),
            "target/MTP working planes plus exact device-resident GDN snapshots",
        ),
        (
            "qwen35_generation_mtp/padding",
            BenchmarkMemoryKind::Other,
            layout.padding_bytes(),
            "256-byte arena alignment",
        ),
    ] {
        memory.register_owned(name, kind, bytes, description)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OUTPUT_TOKENS, SPEC};

    #[test]
    fn qwen35_mtp_generation_suite_benchmark_uses_one_complete_k4_request() {
        assert_eq!(OUTPUT_TOKENS, 8);
    }

    #[test]
    fn qwen35_mtp_generation_report_identity_is_pinned() {
        assert_eq!(SPEC.suite, "bench-qwen35-mtp-generation");
        assert_eq!(SPEC.classification, "performance_sensitive_model");
        assert_eq!(SPEC.round_metric, "qwen3_5/generation/mtp_greedy_round");
        assert_eq!(SPEC.request_metric, "qwen3_5/generation/mtp_greedy_request");
    }
}
