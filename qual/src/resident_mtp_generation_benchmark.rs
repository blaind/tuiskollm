//! Direct host-completion timing for one production greedy MTP request.

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
use tuisko_engine::{ResidentMtpProgram, ResidentMtpTextGenerator};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const OUTPUT_TOKENS: usize = 8;

const SPEC: MtpGreedyBenchmarkSpec = MtpGreedyBenchmarkSpec {
    batch_refusal: "resident greedy MTP generation benchmark admits only B=1",
    energy_refusal: "resident greedy MTP request energy is deferred to the full-server gate",
    warmup_width_drift: "resident greedy MTP round committed count changed during warmup",
    sample_width_drift: "resident greedy MTP round committed count changed between samples",
    missing_warmup_request: "resident greedy MTP benchmark requires at least one warmup request",
    unwarmed_round: "resident greedy MTP benchmark did not warm one speculative round",
    output_drift: "resident greedy MTP benchmark output changed between samples",
    k4_refusal: "resident greedy MTP benchmark did not execute a draft-three/K=4 round",
    round_metric: "qwen3_8/generation/mtp_greedy_round",
    request_metric: "qwen3_8/generation/mtp_greedy_request",
    suite: "bench-generation-mtp-greedy",
    classification: "performance_sensitive_model",
    timing_scope: "direct Rust host completion for prompt prime, greedy draft-three, target verify/commit, MTP realignment, and streaming control through the production owner",
};

/// Measures complete frontend-to-target/MTP greedy requests without summing leaf medians.
pub fn benchmark_resident_mtp_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let mut preamble = admit_greedy_mtp_benchmark(options, &SPEC)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = open_device_zero()?;
    let mut generator = ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
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

pub(crate) fn register_memory(
    memory: &mut MemoryRecorder,
    generator: &ResidentMtpTextGenerator,
) -> Result<(), DeviceBenchmarkError> {
    register_program_memory(memory, generator.qualification_program())
}

pub(crate) fn register_program_memory(
    memory: &mut MemoryRecorder,
    program: &ResidentMtpProgram,
) -> Result<(), DeviceBenchmarkError> {
    let target = program.target();
    for (name, kind, bytes, description) in [
        (
            "generation_mtp/target_weights",
            BenchmarkMemoryKind::Weights,
            target.resident_weight_bytes(),
            "64 exact target layers plus shared final norm and LM head",
        ),
        (
            "generation_mtp/target_gdn_history",
            BenchmarkMemoryKind::Other,
            target.history_bytes(),
            "48 layers * 8 persistent causal-history slots",
        ),
        (
            "generation_mtp/target_gdn_state",
            BenchmarkMemoryKind::Other,
            target.state_bytes(),
            "48 layers * 8 persistent recurrent-state slots",
        ),
        (
            "generation_mtp/target_kv_cache",
            BenchmarkMemoryKind::KvCache,
            target.cache_bytes(),
            "16 target layers sharing the exact 3,438-page pool",
        ),
        (
            "generation_mtp/target_kv_tables",
            BenchmarkMemoryKind::Other,
            target.kv_table_bytes(),
            "8 target slot rows * 3,438 page entries",
        ),
        (
            "generation_mtp/target_workspace",
            BenchmarkMemoryKind::Workspace,
            target.workspace_bytes(),
            "target address-stable decode, prefill, and verification workspace",
        ),
        (
            "generation_mtp/target_tensor_maps",
            BenchmarkMemoryKind::Other,
            target.descriptor_bytes(),
            "eight dense target layers * four address-bound tensor maps",
        ),
        (
            "generation_mtp/target_padding",
            BenchmarkMemoryKind::Other,
            target.padding_bytes(),
            "target resident and KV arena alignment",
        ),
        (
            "generation_mtp/mtp_weights",
            BenchmarkMemoryKind::Weights,
            program.resident_weight_bytes(),
            "one unchanged source-BF16 MTP weight set sharing the target endpoint",
        ),
        (
            "generation_mtp/mtp_kv_cache",
            BenchmarkMemoryKind::KvCache,
            program.cache_bytes(),
            "one BF16 MTP K/V mirror using the target page lifecycle",
        ),
        (
            "generation_mtp/mtp_workspace",
            BenchmarkMemoryKind::Workspace,
            program.workspace_bytes(),
            "prompt, continuation, verification realignment, and LM-head seams",
        ),
        (
            "generation_mtp/mtp_padding",
            BenchmarkMemoryKind::Other,
            program.padding_bytes(),
            "two 256-byte-aligned MTP arenas",
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
    fn resident_mtp_generation_benchmark_uses_one_complete_k4_request() {
        assert_eq!(OUTPUT_TOKENS, 8);
    }

    #[test]
    fn resident_mtp_generation_report_identity_is_pinned() {
        assert_eq!(SPEC.suite, "bench-generation-mtp-greedy");
        assert_eq!(SPEC.classification, "performance_sensitive_model");
        assert_eq!(SPEC.round_metric, "qwen3_8/generation/mtp_greedy_round");
        assert_eq!(SPEC.request_metric, "qwen3_8/generation/mtp_greedy_request");
    }
}
