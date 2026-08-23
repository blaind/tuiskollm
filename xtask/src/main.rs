//! Repository build and qualification gates.

mod gpu_target;
mod perf_artifact;
mod perf_iteration;
mod performance;
mod remote;

use gpu_target::BuildTargetProfile;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const RESIDUAL_NORM_RESOURCE_BASELINE: &str = "qual/baselines/residual-norm-sm120.txt";
const QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-residual-norm-sm120.txt";
const QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-swiglu-sm120.txt";
const QWEN35_NVFP4_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-down-sm120.txt";
const QWEN35_NVFP4_QKV_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-qkv-sm120.txt";
const FP8_QKV_RESOURCE_BASELINE: &str = "qual/baselines/fp8-qkv-sm120.txt";
const FP8_GDN_INPUT_RESOURCE_BASELINE: &str = "qual/baselines/fp8-gdn-input-sm120.txt";
const FP8_LM_HEAD_RESOURCE_BASELINE: &str = "qual/baselines/fp8-lm-head-sm120.txt";
const FP8_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/fp8-swiglu-sm120.txt";
const FP8_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/fp8-down-sm120.txt";
const NVFP4_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/nvfp4-swiglu-sm120.txt";
const NVFP4_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/nvfp4-down-sm120.txt";
const GDN_PREPARE_RESOURCE_BASELINE: &str = "qual/baselines/gdn-prepare-sm120.txt";
const GDN_RECURRENCE_RESOURCE_BASELINE: &str = "qual/baselines/gdn-recurrence-sm120.txt";
const GDN_OUTPUT_RESOURCE_BASELINE: &str = "qual/baselines/gdn-output-sm120.txt";
const ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/attention-qk-prepare-sm120.txt";
const QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-attention-qk-prepare-sm120.txt";
const PAGED_GQA_RESOURCE_BASELINE: &str = "qual/baselines/paged-gqa-sm120.txt";
const LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE: &str =
    "qual/baselines/long-context-paged-gqa-sm120.txt";
const ATTENTION_OUTPUT_RESOURCE_BASELINE: &str = "qual/baselines/attention-output-sm120.txt";
const RESIDENT_MODEL_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
];
const SM120_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_QKV_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
];
const NVFP4_MLP_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
];
const DENSE_FP8_MLP_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const DENSE_FP8_GDN_LAYER_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const FULL_ATTENTION_LAYER_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const TEXT_ENDPOINT_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
];
const PTX: &str = "target/cuda/tuisko_kernels_sm120.ptx";
const CUDA_OXIDE_BUILD_TARGET: &str = "target/cuda-oxide-build-sm120";
const CUDA_OXIDE_TEST_TARGET: &str = "target/cuda-oxide-test";
const CUDA_OXIDE_REPOSITORY: &str = "https://github.com/NVlabs/cuda-oxide.git";
const CUDA_OXIDE_REVISION: &str = "1f4d813719012d384f2db12b88efc9314c8bf50c";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PerformanceSuite {
    ResidualNorm,
    Fp8Qkv,
    Fp8GdnInput,
    Fp8LmHead,
    Fp8SwiGlu,
    Fp8Down,
    Nvfp4SwiGlu,
    Nvfp4Down,
    GdnPrepare,
    GdnRecurrence,
    GdnOutput,
    AttentionQkPrepare,
    PagedGqa,
    LongContextPagedGqa,
    AttentionOutput,
}

const PERFORMANCE_SUITES: [PerformanceSuite; 15] = [
    PerformanceSuite::ResidualNorm,
    PerformanceSuite::Fp8Qkv,
    PerformanceSuite::Fp8GdnInput,
    PerformanceSuite::Fp8LmHead,
    PerformanceSuite::Fp8SwiGlu,
    PerformanceSuite::Fp8Down,
    PerformanceSuite::Nvfp4SwiGlu,
    PerformanceSuite::Nvfp4Down,
    PerformanceSuite::GdnPrepare,
    PerformanceSuite::GdnRecurrence,
    PerformanceSuite::GdnOutput,
    PerformanceSuite::AttentionQkPrepare,
    PerformanceSuite::PagedGqa,
    PerformanceSuite::LongContextPagedGqa,
    PerformanceSuite::AttentionOutput,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OptimizationSuite {
    Leaf(PerformanceSuite),
    Nvfp4Mlp,
    DenseFp8Mlp,
    DenseFp8GdnLayer,
    FullAttentionLayer,
    TextEndpoint,
    ResidentModel,
    ResidentLongContextModel,
}

const COMPOSED_PERFORMANCE_SUITES: [OptimizationSuite; 7] = [
    OptimizationSuite::Nvfp4Mlp,
    OptimizationSuite::DenseFp8Mlp,
    OptimizationSuite::DenseFp8GdnLayer,
    OptimizationSuite::FullAttentionLayer,
    OptimizationSuite::TextEndpoint,
    OptimizationSuite::ResidentModel,
    OptimizationSuite::ResidentLongContextModel,
];

impl PerformanceSuite {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::ResidualNorm => "residual-norm",
            Self::Fp8Qkv => "fp8-qkv",
            Self::Fp8GdnInput => "fp8-gdn-input",
            Self::Fp8LmHead => "fp8-lm-head",
            Self::Fp8SwiGlu => "fp8-swiglu",
            Self::Fp8Down => "fp8-down",
            Self::Nvfp4SwiGlu => "nvfp4-swiglu",
            Self::Nvfp4Down => "nvfp4-down",
            Self::GdnPrepare => "gdn-prepare",
            Self::GdnRecurrence => "gdn-recurrence",
            Self::GdnOutput => "gdn-output",
            Self::AttentionQkPrepare => "attention-qk-prepare",
            Self::PagedGqa => "paged-gqa",
            Self::LongContextPagedGqa => "long-context-paged-gqa",
            Self::AttentionOutput => "attention-output",
        }
    }

    const fn resource_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => RESIDUAL_NORM_RESOURCE_BASELINE,
            Self::Fp8Qkv => FP8_QKV_RESOURCE_BASELINE,
            Self::Fp8GdnInput => FP8_GDN_INPUT_RESOURCE_BASELINE,
            Self::Fp8LmHead => FP8_LM_HEAD_RESOURCE_BASELINE,
            Self::Fp8SwiGlu => FP8_SWIGLU_RESOURCE_BASELINE,
            Self::Fp8Down => FP8_DOWN_RESOURCE_BASELINE,
            Self::Nvfp4SwiGlu => NVFP4_SWIGLU_RESOURCE_BASELINE,
            Self::Nvfp4Down => NVFP4_DOWN_RESOURCE_BASELINE,
            Self::GdnPrepare => GDN_PREPARE_RESOURCE_BASELINE,
            Self::GdnRecurrence => GDN_RECURRENCE_RESOURCE_BASELINE,
            Self::GdnOutput => GDN_OUTPUT_RESOURCE_BASELINE,
            Self::AttentionQkPrepare => ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
            Self::PagedGqa => PAGED_GQA_RESOURCE_BASELINE,
            Self::LongContextPagedGqa => LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
            Self::AttentionOutput => ATTENTION_OUTPUT_RESOURCE_BASELINE,
        }
    }

    const fn performance_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => "qual/baselines/residual-norm-sm120.json",
            Self::Fp8Qkv => "qual/baselines/fp8-qkv-sm120.json",
            Self::Fp8GdnInput => "qual/baselines/fp8-gdn-input-sm120.json",
            Self::Fp8LmHead => "qual/baselines/fp8-lm-head-sm120.json",
            Self::Fp8SwiGlu => "qual/baselines/fp8-swiglu-sm120.json",
            Self::Fp8Down => "qual/baselines/fp8-down-sm120.json",
            Self::Nvfp4SwiGlu => "qual/baselines/nvfp4-swiglu-sm120.json",
            Self::Nvfp4Down => "qual/baselines/nvfp4-down-sm120.json",
            Self::GdnPrepare => "qual/baselines/gdn-prepare-sm120.json",
            Self::GdnRecurrence => "qual/baselines/gdn-recurrence-sm120.json",
            Self::GdnOutput => "qual/baselines/gdn-output-sm120.json",
            Self::AttentionQkPrepare => "qual/baselines/attention-qk-prepare-sm120.json",
            Self::PagedGqa => "qual/baselines/paged-gqa-sm120.json",
            Self::LongContextPagedGqa => "qual/baselines/long-context-paged-gqa-sm120.json",
            Self::AttentionOutput => "qual/baselines/attention-output-sm120.json",
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "residual-norm" => Ok(Self::ResidualNorm),
            "fp8-qkv" => Ok(Self::Fp8Qkv),
            "fp8-gdn-input" => Ok(Self::Fp8GdnInput),
            "fp8-lm-head" => Ok(Self::Fp8LmHead),
            "fp8-swiglu" => Ok(Self::Fp8SwiGlu),
            "fp8-down" => Ok(Self::Fp8Down),
            "nvfp4-swiglu" => Ok(Self::Nvfp4SwiGlu),
            "nvfp4-down" => Ok(Self::Nvfp4Down),
            "gdn-prepare" => Ok(Self::GdnPrepare),
            "gdn-recurrence" => Ok(Self::GdnRecurrence),
            "gdn-output" => Ok(Self::GdnOutput),
            "attention-qk-prepare" => Ok(Self::AttentionQkPrepare),
            "paged-gqa" => Ok(Self::PagedGqa),
            "long-context-paged-gqa" => Ok(Self::LongContextPagedGqa),
            "attention-output" => Ok(Self::AttentionOutput),
            _ => Err(format!("unknown performance suite `{value}`").into()),
        }
    }

    fn qualify(self, root: &Path) -> Result<(), Box<dyn Error>> {
        match self {
            Self::ResidualNorm => qualify_residual_norm(root),
            Self::Fp8Qkv => qualify_fp8_qkv(root),
            Self::Fp8GdnInput => qualify_fp8_gdn_input(root),
            Self::Fp8LmHead => qualify_fp8_lm_head(root),
            Self::Fp8SwiGlu => qualify_fp8_swiglu(root),
            Self::Fp8Down => qualify_fp8_down(root),
            Self::Nvfp4SwiGlu => qualify_nvfp4_swiglu(root),
            Self::Nvfp4Down => qualify_nvfp4_down(root),
            Self::GdnPrepare => qualify_gdn_prepare(root),
            Self::GdnRecurrence => qualify_gdn_recurrence(root),
            Self::GdnOutput => qualify_gdn_output(root),
            Self::AttentionQkPrepare => qualify_attention_qk_prepare(root),
            Self::PagedGqa => qualify_paged_gqa(root),
            Self::LongContextPagedGqa => qualify_long_context_paged_gqa(root),
            Self::AttentionOutput => qualify_attention_output(root),
        }
    }
}

impl OptimizationSuite {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        if let Ok(suite) = PerformanceSuite::parse(value) {
            return Ok(Self::Leaf(suite));
        }
        COMPOSED_PERFORMANCE_SUITES
            .iter()
            .copied()
            .find(|suite| suite.name() == value)
            .ok_or_else(|| format!("unknown optimization suite `{value}`").into())
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Leaf(suite) => suite.name(),
            Self::Nvfp4Mlp => "nvfp4-mlp",
            Self::DenseFp8Mlp => "dense-fp8-mlp",
            Self::DenseFp8GdnLayer => "dense-fp8-gdn-layer",
            Self::FullAttentionLayer => "full-attention-layer",
            Self::TextEndpoint => "text-endpoint",
            Self::ResidentModel => "resident-model",
            Self::ResidentLongContextModel => "resident-long-context-model",
        }
    }

    const fn requires_snapshot(self) -> bool {
        !matches!(self, Self::Leaf(_))
    }

    fn resource_baselines(self) -> Vec<&'static str> {
        match self {
            Self::Leaf(suite) => vec![suite.resource_baseline()],
            Self::Nvfp4Mlp => NVFP4_MLP_RESOURCE_BASELINES.to_vec(),
            Self::DenseFp8Mlp => DENSE_FP8_MLP_RESOURCE_BASELINES.to_vec(),
            Self::DenseFp8GdnLayer => DENSE_FP8_GDN_LAYER_RESOURCE_BASELINES.to_vec(),
            Self::FullAttentionLayer => FULL_ATTENTION_LAYER_RESOURCE_BASELINES.to_vec(),
            Self::TextEndpoint => TEXT_ENDPOINT_RESOURCE_BASELINES.to_vec(),
            Self::ResidentModel | Self::ResidentLongContextModel => {
                RESIDENT_MODEL_RESOURCE_BASELINES.to_vec()
            }
        }
    }

    const fn performance_baseline(self) -> &'static str {
        match self {
            Self::Leaf(suite) => suite.performance_baseline(),
            Self::Nvfp4Mlp => "qual/baselines/nvfp4-mlp-sm120.json",
            Self::DenseFp8Mlp => "qual/baselines/dense-fp8-mlp-sm120.json",
            Self::DenseFp8GdnLayer => "qual/baselines/dense-fp8-gdn-layer-sm120.json",
            Self::FullAttentionLayer => "qual/baselines/full-attention-layer-sm120.json",
            Self::TextEndpoint => "qual/baselines/text-endpoint-sm120.json",
            Self::ResidentModel => "qual/baselines/resident-model-sm120.json",
            Self::ResidentLongContextModel => {
                "qual/baselines/resident-long-context-model-sm120.json"
            }
        }
    }

    fn qualify(self, root: &Path, snapshot: Option<&OsStr>) -> Result<(), Box<dyn Error>> {
        let snapshot_arguments = || -> Result<[std::ffi::OsString; 1], Box<dyn Error>> {
            Ok([snapshot
                .ok_or_else(|| format!("{} requires the admitted snapshot path", self.name()))?
                .to_os_string()])
        };
        match self {
            Self::Leaf(suite) => suite.qualify(root),
            Self::Nvfp4Mlp => qualify_nvfp4_mlp(root, &snapshot_arguments()?),
            Self::DenseFp8Mlp => qualify_dense_fp8_mlp(root, &snapshot_arguments()?),
            Self::DenseFp8GdnLayer => qualify_dense_fp8_gdn_layer(root, &snapshot_arguments()?),
            Self::FullAttentionLayer => qualify_full_attention_layer(root, &snapshot_arguments()?),
            Self::TextEndpoint => qualify_text_endpoint(root, &snapshot_arguments()?),
            Self::ResidentModel | Self::ResidentLongContextModel => {
                qualify_resident_model(root, &snapshot_arguments()?)
            }
        }
    }

    fn dependency_cone(self) -> Vec<Self> {
        use OptimizationSuite::{
            DenseFp8GdnLayer, DenseFp8Mlp, FullAttentionLayer, Nvfp4Mlp, ResidentLongContextModel,
            ResidentModel, TextEndpoint,
        };
        use PerformanceSuite::{
            AttentionOutput, AttentionQkPrepare, Fp8Down, Fp8GdnInput, Fp8LmHead, Fp8Qkv,
            Fp8SwiGlu, GdnOutput, GdnPrepare, GdnRecurrence, LongContextPagedGqa, Nvfp4Down,
            Nvfp4SwiGlu, PagedGqa, ResidualNorm,
        };

        let downstream = match self {
            Self::ResidentModel | Self::ResidentLongContextModel => &[][..],
            Self::Leaf(LongContextPagedGqa) => &[ResidentLongContextModel],
            Self::Leaf(Nvfp4SwiGlu | Nvfp4Down) | Self::Nvfp4Mlp => {
                &[Nvfp4Mlp, ResidentModel, ResidentLongContextModel]
            }
            Self::Leaf(Fp8LmHead) | Self::TextEndpoint => {
                &[TextEndpoint, ResidentModel, ResidentLongContextModel]
            }
            Self::Leaf(Fp8GdnInput | GdnPrepare | GdnRecurrence | GdnOutput)
            | Self::DenseFp8GdnLayer => {
                &[DenseFp8GdnLayer, ResidentModel, ResidentLongContextModel]
            }
            Self::Leaf(Fp8Qkv | AttentionQkPrepare | PagedGqa | AttentionOutput)
            | Self::FullAttentionLayer => {
                &[FullAttentionLayer, ResidentModel, ResidentLongContextModel]
            }
            Self::Leaf(Fp8SwiGlu | Fp8Down) | Self::DenseFp8Mlp => &[
                DenseFp8Mlp,
                DenseFp8GdnLayer,
                FullAttentionLayer,
                ResidentModel,
                ResidentLongContextModel,
            ],
            Self::Leaf(ResidualNorm) => &[
                Nvfp4Mlp,
                DenseFp8Mlp,
                DenseFp8GdnLayer,
                FullAttentionLayer,
                TextEndpoint,
                ResidentModel,
                ResidentLongContextModel,
            ],
        };
        let mut cone = vec![self];
        for suite in downstream {
            if !cone.contains(suite) {
                cone.push(*suite);
            }
        }

        cone
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(command) = arguments.next() else {
        return Err("usage: cargo run -p xtask -- <bootstrap-cuda-oxide|build-sm120|build-server|qualify-...|bench-...|gate-...|perf|profile|remote>".into());
    };
    let remaining = arguments.collect::<Vec<_>>();
    let root = workspace_root()?;

    match command.to_str() {
        Some("bootstrap-cuda-oxide") if remaining.is_empty() => bootstrap_cuda_oxide(root),
        Some("build-sm120") if remaining.is_empty() => build_sm120(root),
        Some("build-residual-norm") => build_residual_norm(root, &remaining),
        Some("build-residual-bench") => build_residual_bench(root, &remaining),
        Some("build-server") if remaining.is_empty() => build_server(root),
        Some("qualify-frontend") => qualify_frontend(root, &remaining),
        Some("qualify-generation") => qualify_generation(root, &remaining),
        Some("qualify-residual-norm") if remaining.is_empty() => qualify_residual_norm(root),
        Some("qualify-qwen35-residual-norm") if remaining.is_empty() => {
            qualify_qwen35_residual_norm(root)
        }
        Some("qualify-qwen35-nvfp4-swiglu") if remaining.is_empty() => {
            qualify_qwen35_nvfp4_swiglu(root)
        }
        Some("qualify-qwen35-nvfp4-down") if remaining.is_empty() => {
            qualify_qwen35_nvfp4_down(root)
        }
        Some("qualify-qwen35-nvfp4-qkv") if remaining.is_empty() => qualify_qwen35_nvfp4_qkv(root),
        Some("qualify-qwen35-nvfp4-mlp") => qualify_qwen35_nvfp4_mlp(root, &remaining),
        Some("qualify-qwen35-attention-qk-prepare") if remaining.is_empty() => {
            qualify_qwen35_attention_qk_prepare(root)
        }
        Some("qualify-fp8-qkv") if remaining.is_empty() => qualify_fp8_qkv(root),
        Some("qualify-fp8-gdn-input") if remaining.is_empty() => qualify_fp8_gdn_input(root),
        Some("qualify-fp8-lm-head") if remaining.is_empty() => qualify_fp8_lm_head(root),
        Some("qualify-fp8-swiglu") if remaining.is_empty() => qualify_fp8_swiglu(root),
        Some("qualify-fp8-down") if remaining.is_empty() => qualify_fp8_down(root),
        Some("qualify-nvfp4-swiglu") if remaining.is_empty() => qualify_nvfp4_swiglu(root),
        Some("qualify-nvfp4-down") if remaining.is_empty() => qualify_nvfp4_down(root),
        Some("qualify-nvfp4-mlp") => qualify_nvfp4_mlp(root, &remaining),
        Some("qualify-gdn-prepare") if remaining.is_empty() => qualify_gdn_prepare(root),
        Some("qualify-gdn-recurrence") if remaining.is_empty() => qualify_gdn_recurrence(root),
        Some("qualify-gdn-output") if remaining.is_empty() => qualify_gdn_output(root),
        Some("qualify-attention-qk-prepare") if remaining.is_empty() => {
            qualify_attention_qk_prepare(root)
        }
        Some("qualify-paged-gqa") if remaining.is_empty() => qualify_paged_gqa(root),
        Some("qualify-long-context-paged-gqa") if remaining.is_empty() => {
            qualify_long_context_paged_gqa(root)
        }
        Some("qualify-attention-output") if remaining.is_empty() => qualify_attention_output(root),
        Some("qualify-dense-fp8-mlp") => qualify_dense_fp8_mlp(root, &remaining),
        Some("qualify-dense-fp8-gdn-layer") => qualify_dense_fp8_gdn_layer(root, &remaining),
        Some("qualify-full-attention-layer") => qualify_full_attention_layer(root, &remaining),
        Some("qualify-resident-model") => qualify_resident_model(root, &remaining),
        Some("qualify-resident-generation") => qualify_resident_generation(root, &remaining),
        Some("qualify-resident-batch-generation") => {
            qualify_resident_batch_generation(root, &remaining)
        }
        Some("qualify-text-endpoint") => qualify_text_endpoint(root, &remaining),
        Some("bench-startup") => bench_startup(root, &remaining),
        Some("bench-residual-norm") => bench_residual_norm(root, &remaining),
        Some("bench-qwen35-residual-norm") => bench_qwen35_residual_norm(root, &remaining),
        Some("bench-qwen35-nvfp4-swiglu") => bench_qwen35_nvfp4_swiglu(root, &remaining),
        Some("bench-qwen35-nvfp4-down") => bench_qwen35_nvfp4_down(root, &remaining),
        Some("bench-qwen35-nvfp4-qkv") => bench_qwen35_nvfp4_qkv(root, &remaining),
        Some("bench-qwen35-nvfp4-mlp") => bench_qwen35_nvfp4_mlp(root, &remaining),
        Some("bench-qwen35-attention-qk-prepare") => {
            bench_qwen35_attention_qk_prepare(root, &remaining)
        }
        Some("bench-fp8-qkv") => bench_fp8_qkv(root, &remaining),
        Some("bench-fp8-gdn-input") => bench_fp8_gdn_input(root, &remaining),
        Some("bench-fp8-lm-head") => bench_fp8_lm_head(root, &remaining),
        Some("bench-fp8-swiglu") => bench_fp8_swiglu(root, &remaining),
        Some("bench-fp8-down") => bench_fp8_down(root, &remaining),
        Some("bench-nvfp4-swiglu") => bench_nvfp4_swiglu(root, &remaining),
        Some("bench-nvfp4-down") => bench_nvfp4_down(root, &remaining),
        Some("bench-nvfp4-mlp") => bench_nvfp4_mlp(root, &remaining),
        Some("bench-gdn-prepare") => bench_gdn_prepare(root, &remaining),
        Some("bench-gdn-recurrence") => bench_gdn_recurrence(root, &remaining),
        Some("bench-gdn-output") => bench_gdn_output(root, &remaining),
        Some("bench-attention-qk-prepare") => bench_attention_qk_prepare(root, &remaining),
        Some("bench-paged-gqa") => bench_paged_gqa(root, &remaining),
        Some("bench-long-context-paged-gqa") => bench_long_context_paged_gqa(root, &remaining),
        Some("bench-attention-output") => bench_attention_output(root, &remaining),
        Some("bench-dense-fp8-mlp") => bench_dense_fp8_mlp(root, &remaining),
        Some("bench-dense-fp8-gdn-layer") => bench_dense_fp8_gdn_layer(root, &remaining),
        Some("bench-full-attention-layer") => bench_full_attention_layer(root, &remaining),
        Some("bench-resident-model") => bench_resident_model(root, &remaining),
        Some("bench-resident-long-context-model") => {
            bench_resident_long_context_model(root, &remaining)
        }
        Some("bench-text-endpoint") => bench_text_endpoint(root, &remaining),
        Some("gate-residual-norm") if remaining.is_empty() => gate_residual_norm(root),
        Some("gate-qwen35-residual-norm") if remaining.is_empty() => {
            gate_qwen35_residual_norm(root)
        }
        Some("gate-qwen35-nvfp4-swiglu") if remaining.is_empty() => gate_qwen35_nvfp4_swiglu(root),
        Some("gate-qwen35-nvfp4-down") if remaining.is_empty() => gate_qwen35_nvfp4_down(root),
        Some("gate-qwen35-nvfp4-qkv") if remaining.is_empty() => gate_qwen35_nvfp4_qkv(root),
        Some("gate-qwen35-attention-qk-prepare") if remaining.is_empty() => {
            gate_qwen35_attention_qk_prepare(root)
        }
        Some("gate-fp8-qkv") if remaining.is_empty() => gate_fp8_qkv(root),
        Some("gate-fp8-gdn-input") if remaining.is_empty() => gate_fp8_gdn_input(root),
        Some("gate-fp8-lm-head") if remaining.is_empty() => gate_fp8_lm_head(root),
        Some("gate-fp8-swiglu") if remaining.is_empty() => gate_fp8_swiglu(root),
        Some("gate-fp8-down") if remaining.is_empty() => gate_fp8_down(root),
        Some("gate-nvfp4-swiglu") if remaining.is_empty() => gate_nvfp4_swiglu(root),
        Some("gate-nvfp4-down") if remaining.is_empty() => gate_nvfp4_down(root),
        Some("gate-gdn-prepare") if remaining.is_empty() => gate_gdn_prepare(root),
        Some("gate-gdn-recurrence") if remaining.is_empty() => gate_gdn_recurrence(root),
        Some("gate-gdn-output") if remaining.is_empty() => gate_gdn_output(root),
        Some("gate-attention-qk-prepare") if remaining.is_empty() => {
            gate_attention_qk_prepare(root)
        }
        Some("gate-paged-gqa") if remaining.is_empty() => gate_paged_gqa(root),
        Some("gate-long-context-paged-gqa") if remaining.is_empty() => {
            gate_long_context_paged_gqa(root)
        }
        Some("gate-attention-output") if remaining.is_empty() => gate_attention_output(root),
        Some("perf") => perf(root, &remaining),
        Some("profile") => profile(root, &remaining),
        Some("remote") => remote::run(root, &remaining),
        Some(known)
            if matches!(
                known,
                "bootstrap-cuda-oxide"
                    | "build-sm120"
                    | "build-residual-norm"
                    | "build-residual-bench"
                    | "build-server"
                    | "qualify-residual-norm"
                    | "qualify-qwen35-residual-norm"
                    | "qualify-qwen35-attention-qk-prepare"
                    | "qualify-fp8-qkv"
                    | "qualify-fp8-gdn-input"
                    | "qualify-fp8-lm-head"
                    | "qualify-fp8-swiglu"
                    | "qualify-fp8-down"
                    | "qualify-nvfp4-swiglu"
                    | "qualify-nvfp4-down"
                    | "qualify-gdn-prepare"
                    | "qualify-gdn-recurrence"
                    | "qualify-gdn-output"
                    | "qualify-attention-qk-prepare"
                    | "qualify-paged-gqa"
                    | "qualify-long-context-paged-gqa"
                    | "qualify-attention-output"
                    | "gate-residual-norm"
                    | "gate-qwen35-residual-norm"
                    | "gate-qwen35-attention-qk-prepare"
                    | "gate-fp8-qkv"
                    | "gate-fp8-gdn-input"
                    | "gate-fp8-lm-head"
                    | "gate-fp8-swiglu"
                    | "gate-fp8-down"
                    | "gate-nvfp4-swiglu"
                    | "gate-nvfp4-down"
                    | "gate-gdn-prepare"
                    | "gate-gdn-recurrence"
                    | "gate-gdn-output"
                    | "gate-attention-qk-prepare"
                    | "gate-paged-gqa"
                    | "gate-long-context-paged-gqa"
                    | "gate-attention-output"
            ) =>
        {
            Err(format!("`{known}` takes no arguments").into())
        }
        _ => Err(format!("unknown xtask command `{}`", command.to_string_lossy()).into()),
    }
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest has no workspace parent".into())
}

fn bootstrap_cuda_oxide(root: &Path) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    if !source.join(".git").is_dir() {
        fs::create_dir_all(
            source
                .parent()
                .ok_or("cuda-oxide source path has no parent")?,
        )?;
        run_visible(
            Command::new("git")
                .args(["clone", "--no-checkout", CUDA_OXIDE_REPOSITORY])
                .arg(&source),
        )?;
        run_visible(Command::new("git").arg("-C").arg(&source).args([
            "checkout",
            "--detach",
            CUDA_OXIDE_REVISION,
        ]))?;
    }
    require_cuda_oxide_revision(&source)?;

    let driver_target = root.join("target/cuda-oxide-driver");
    run_visible(
        Command::new("cargo")
            .arg("+nightly-2026-04-03")
            .arg("build")
            .arg("--manifest-path")
            .arg(source.join("Cargo.toml"))
            .args(["--package", "cargo-oxide", "--target-dir"])
            .arg(&driver_target)
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let wrapper = driver_target.join("debug/cargo-oxide");
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    run_visible(
        Command::new(&wrapper)
            .arg("setup")
            .current_dir(&source)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS"),
    )?;

    println!(
        "cuda-oxide ready: {} at {}",
        CUDA_OXIDE_REVISION,
        wrapper.display()
    );
    Ok(())
}

fn build_sm120(root: &Path) -> Result<(), Box<dyn Error>> {
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let resource_baselines =
        perf_artifact::resource_baselines_sha256(root, SM120_RESOURCE_BASELINES)?;
    if perf_artifact::local_build_is_current(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    )? {
        println!("reusing verified worktree-local SM120 benchmark artifacts");
        return gate_sm120_resources(root);
    }
    if let Some(source) = perf_artifact::restore_build_from_worktrees(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    )? {
        println!(
            "restored verified SM120 benchmark artifacts from {}",
            source.display()
        );
        gate_sm120_resources(root)?;
        return Ok(());
    }

    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-device",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)?;
    perf_artifact::record_build(root, device_inputs, resource_baselines, CUDA_OXIDE_REVISION)
}

fn build_sm120_for_performance(root: &Path) -> Result<(), Box<dyn Error>> {
    require_performance_device_idle()?;
    build_sm120(root)?;
    wait_for_device_idle()
}

fn build_startup_benchmark(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-startup",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)
}

fn build_server(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuiskollm",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)?;

    let binary = root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm");
    if !binary.is_file() {
        return Err(format!("server build omitted `{}`", binary.display()).into());
    }
    println!("server binary: {}", binary.display());
    Ok(())
}

fn gate_sm120_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm(root)?;
    gate_qwen35_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_fp8_gdn_input(root)?;
    gate_fp8_lm_head(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_gdn_output(root)?;
    gate_attention_qk_prepare(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_long_context_paged_gqa(root)?;
    gate_attention_output(root)
}

fn build_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let gpu = parse_build_gpu(arguments, "build-residual-norm")?;
    let prepared = prepare_remote_qualify(root, gpu, "residual_norm::tests")?;
    gate_residual_norm_target(root, gpu)?;
    println!(
        "{} ({}, compute capability {}) residual-norm qualification artifact: {}",
        gpu.key(),
        gpu.device_name(),
        gpu.compute_capability_text(),
        prepared.executable.display()
    );
    Ok(())
}

fn build_residual_bench(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let gpu = parse_build_gpu(arguments, "build-residual-bench")?;
    build_residual_benchmark_target(root, gpu)?;
    let prepared = prepare_remote_benchmark(root, gpu, PerformanceSuite::ResidualNorm)?;
    println!(
        "{} residual-norm benchmark artifact: {} (resource baseline {})",
        gpu.key(),
        prepared.executable.display(),
        prepared.generator_baseline_sha256,
    );
    Ok(())
}

fn parse_build_gpu(
    arguments: &[std::ffi::OsString],
    command: &str,
) -> Result<gpu_target::GpuTarget, Box<dyn Error>> {
    let [flag, value] = arguments else {
        return Err(
            format!("usage: cargo run -p xtask -- {command} --gpu <5090|4090|3090>").into(),
        );
    };
    if flag != "--gpu" {
        return Err(format!("{command} requires `--gpu <5090|4090|3090>`").into());
    }
    Ok(gpu_target::GpuTarget::parse(&value.to_string_lossy())?)
}

pub(crate) fn build_residual_benchmark_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        return build_sm120(root);
    }
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            gpu.oxide_arch(),
            "--cargo-target-dir",
            gpu.oxide_build_target(),
            "--device-codegen-crate",
            gpu.kernel_crate(),
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-device",
            "--release",
            "--no-default-features",
            "--features",
            gpu.qualification_feature(),
        ],
    )?;
    gate_residual_norm_target(root, gpu)
}

fn qualify_frontend(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-frontend SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--bin",
                "qualify-frontend",
                "--",
            ])
            .arg(snapshot),
    )
}

fn qualify_generation(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--features",
                "engine",
                "--bin",
                "qualify-generation",
                "--",
            ])
            .arg(snapshot),
    )
}

fn qualify_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "residual_norm::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_residual_norm(root)
}

fn qualify_qwen35_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "residual_norm::tests::qwen35_exact_batches_match_independent_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_qwen35_residual_norm(root)
}

fn qualify_qwen35_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "qwen35_nvfp4_swiglu",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_qwen35_nvfp4_swiglu(root)
}

fn qualify_qwen35_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "qwen35_nvfp4_down",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_qwen35_nvfp4_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "qwen35_nvfp4_qkv",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_qwen35_nvfp4_qkv(root)
}

fn qualify_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_qkv",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_qkv(root)
}

fn qualify_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_swiglu",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_swiglu(root)
}

fn qualify_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_down",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_down(root)
}

fn qualify_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "nvfp4_swiglu",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_nvfp4_swiglu(root)
}

fn qualify_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "nvfp4_down",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_nvfp4_down(root)
}

fn qualify_nvfp4_mlp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)
}

fn qualify_qwen35_nvfp4_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-nvfp4-mlp SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "nvfp4_mlp::tests::qwen35_source_layer0_matches_complete_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "gdn_prepare::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_gdn_prepare(root)
}

fn qualify_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "gdn_recurrence::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_gdn_recurrence(root)
}

fn qualify_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "gdn_output::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_gdn_output(root)
}

fn qualify_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "attention_qk_prepare",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_attention_qk_prepare(root)
}

fn qualify_qwen35_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "attention_qk_prepare::tests::qwen35_",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "attention_qk_prepare_benchmark::tests::qwen35_",
            "--nocapture",
        ],
    )?;
    gate_qwen35_attention_qk_prepare(root)
}

fn qualify_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "paged_gqa_suite_",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ],
    )?;
    gate_paged_gqa(root)
}

fn qualify_long_context_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "long_context_paged_gqa",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_long_context_paged_gqa(root)
}

fn qualify_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "attention_output::tests",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_attention_output(root)
}

fn qualify_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-dense-fp8-mlp SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "dense_fp8_mlp::tests::source_layer60_matches_complete_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_dense_fp8_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-dense-fp8-gdn-layer SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "dense_fp8_gdn_layer::tests::source_layer60_matches_complete_seam_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_gdn_input(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_gdn_output(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-full-attention-layer SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "full_attention_layer::tests::source_layer63_matches_complete_seam_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_attention_output(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-resident-model SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn qualify_resident_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-resident-generation SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn qualify_resident_batch_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT".into(),
        );
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn gate_resident_model_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_fp8_gdn_input(root)?;
    gate_fp8_lm_head(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_gdn_output(root)?;
    gate_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_long_context_paged_gqa(root)?;
    gate_attention_output(root)
}

fn qualify_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_gdn_input",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_gdn_input(root)
}

fn qualify_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "fp8_lm_head",
            "--include-ignored",
            "--nocapture",
        ],
    )?;
    gate_fp8_lm_head(root)
}

fn qualify_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-text-endpoint SNAPSHOT".into());
    };
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            "tuisko-kernels-sm120",
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
            "--",
            "text_endpoint::tests::source_endpoint_matches_independent_oracles_and_graph_replay",
            "--include-ignored",
            "--nocapture",
        ],
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_lm_head(root)
}

fn bench_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::ResidualNorm, arguments)
}

fn bench_qwen35_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg("qwen35-residual-norm")
            .args(arguments)
            .env(
                "TUISKO_GENERATOR_BASELINE_SHA256",
                sha256(&fs::read(
                    root.join(QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE),
                )?),
            ),
    )
}

fn bench_qwen35_nvfp4_swiglu(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg("qwen35-nvfp4-swiglu")
            .args(arguments)
            .env(
                "TUISKO_GENERATOR_BASELINE_SHA256",
                sha256(&fs::read(root.join(QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE))?),
            ),
    )
}

fn bench_qwen35_nvfp4_down(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg("qwen35-nvfp4-down")
            .args(arguments)
            .env(
                "TUISKO_GENERATOR_BASELINE_SHA256",
                sha256(&fs::read(root.join(QWEN35_NVFP4_DOWN_RESOURCE_BASELINE))?),
            ),
    )
}

fn bench_qwen35_nvfp4_qkv(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg("qwen35-nvfp4-qkv")
            .args(arguments)
            .env(
                "TUISKO_GENERATOR_BASELINE_SHA256",
                sha256(&fs::read(root.join(QWEN35_NVFP4_QKV_RESOURCE_BASELINE))?),
            ),
    )
}

fn bench_qwen35_nvfp4_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-nvfp4-mlp SNAPSHOT [options]".into(),
        );
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE))?);
    baselines.extend_from_slice(&fs::read(root.join(QWEN35_NVFP4_DOWN_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("qwen35-nvfp4-mlp")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_qwen35_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg("qwen35-attention-qk-prepare")
            .args(arguments)
            .env(
                "TUISKO_GENERATOR_BASELINE_SHA256",
                sha256(&fs::read(
                    root.join(QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE),
                )?),
            ),
    )
}

fn bench_fp8_qkv(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Qkv, arguments)
}

fn bench_fp8_gdn_input(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8GdnInput, arguments)
}

fn bench_fp8_lm_head(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8LmHead, arguments)
}

fn bench_fp8_swiglu(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8SwiGlu, arguments)
}

fn bench_fp8_down(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Down, arguments)
}

fn bench_nvfp4_swiglu(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Nvfp4SwiGlu, arguments)
}

fn bench_nvfp4_down(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Nvfp4Down, arguments)
}

fn bench_nvfp4_mlp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-nvfp4-mlp SNAPSHOT [options]".into());
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(NVFP4_SWIGLU_RESOURCE_BASELINE))?);
    baselines.extend_from_slice(&fs::read(root.join(NVFP4_DOWN_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("nvfp4-mlp")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_gdn_prepare(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnPrepare, arguments)
}

fn bench_gdn_recurrence(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnRecurrence, arguments)
}

fn bench_gdn_output(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnOutput, arguments)
}

fn bench_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::AttentionQkPrepare, arguments)
}

fn bench_paged_gqa(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::PagedGqa, arguments)
}

fn bench_long_context_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::LongContextPagedGqa, arguments)
}

fn bench_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::AttentionOutput, arguments)
}

fn bench_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-dense-fp8-mlp SNAPSHOT [options]".into());
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(FP8_SWIGLU_RESOURCE_BASELINE))?);
    baselines.extend_from_slice(&fs::read(root.join(FP8_DOWN_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("dense-fp8-mlp")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_dense_fp8_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-dense-fp8-gdn-layer SNAPSHOT [options]".into(),
        );
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    for baseline in [
        FP8_GDN_INPUT_RESOURCE_BASELINE,
        GDN_PREPARE_RESOURCE_BASELINE,
        GDN_RECURRENCE_RESOURCE_BASELINE,
        GDN_OUTPUT_RESOURCE_BASELINE,
        FP8_SWIGLU_RESOURCE_BASELINE,
        FP8_DOWN_RESOURCE_BASELINE,
    ] {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    run_visible(
        Command::new(executable)
            .arg("dense-fp8-gdn-layer")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-full-attention-layer SNAPSHOT [options]".into(),
        );
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = Vec::new();
    for baseline in [
        RESIDUAL_NORM_RESOURCE_BASELINE,
        FP8_QKV_RESOURCE_BASELINE,
        ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        PAGED_GQA_RESOURCE_BASELINE,
        ATTENTION_OUTPUT_RESOURCE_BASELINE,
        FP8_SWIGLU_RESOURCE_BASELINE,
        FP8_DOWN_RESOURCE_BASELINE,
    ] {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    run_visible(
        Command::new(executable)
            .arg("full-attention-layer")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_resident_model_variant(root, arguments, "bench-resident-model", "resident-model")
}

fn bench_startup(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-startup SNAPSHOT [options]".into());
    };
    require_performance_device_idle()?;
    build_startup_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-startup");
    if !executable.is_file() {
        return Err(format!(
            "startup benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(Command::new(executable).arg(snapshot).args(options))
}

fn bench_resident_long_context_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_resident_model_variant(
        root,
        arguments,
        "bench-resident-long-context-model",
        "resident-long-context-model",
    )
}

fn bench_resident_model_variant(
    root: &Path,
    arguments: &[std::ffi::OsString],
    command: &str,
    suite: &str,
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(format!("usage: cargo run -p xtask -- {command} SNAPSHOT [options]").into());
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = Vec::new();
    for baseline in RESIDENT_MODEL_RESOURCE_BASELINES {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    run_visible(
        Command::new(executable)
            .arg(suite)
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-text-endpoint SNAPSHOT [options]".into());
    };
    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut baselines = fs::read(root.join(RESIDUAL_NORM_RESOURCE_BASELINE))?;
    baselines.extend_from_slice(&fs::read(root.join(FP8_LM_HEAD_RESOURCE_BASELINE))?);
    run_visible(
        Command::new(executable)
            .arg("text-endpoint")
            .arg(snapshot)
            .args(options)
            .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines)),
    )
}

fn bench_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    run_benchmark_suite(root, suite, arguments)
}

fn run_benchmark_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut command = Command::new(&executable);
    command.arg(suite.name()).args(arguments).env(
        "TUISKO_GENERATOR_BASELINE_SHA256",
        sha256(&fs::read(root.join(suite.resource_baseline()))?),
    );
    run_visible(&mut command)?;

    Ok(())
}

fn run_optimization_benchmark(
    root: &Path,
    suite: OptimizationSuite,
    snapshot: Option<&OsStr>,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if let OptimizationSuite::Leaf(leaf) = suite {
        return run_benchmark_suite(root, leaf, arguments);
    }
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let snapshot =
        snapshot.ok_or_else(|| format!("{} requires the admitted snapshot path", suite.name()))?;
    let mut baselines = Vec::new();
    for baseline in suite.resource_baselines() {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    let mut command = Command::new(executable);
    command
        .arg(suite.name())
        .arg(snapshot)
        .args(arguments)
        .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines));
    run_visible(&mut command)
}

fn profile(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [scope, snapshot, remaining @ ..] = arguments else {
        return Err("usage: cargo run -p xtask -- profile resident-model SNAPSHOT [--batch B] [--replays N] [--tool nsys|ncu] [--kernel REGEX] [--output-dir PATH]".into());
    };
    if scope != "resident-model" {
        return Err(format!("unknown profile scope `{}`", scope.to_string_lossy()).into());
    }
    let mut batch = 1u32;
    let mut replays = 3u64;
    let mut tool = "nsys";
    let mut kernel = None;
    let mut output_dir = None;
    let mut options = remaining.iter();
    while let Some(argument) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| format!("`{}` requires a value", argument.to_string_lossy()))?;
        match argument.to_str().ok_or("profile argument is not UTF-8")? {
            "--batch" => batch = value.to_str().ok_or("batch is not UTF-8")?.parse()?,
            "--replays" => replays = value.to_str().ok_or("replays is not UTF-8")?.parse()?,
            "--tool" => tool = value.to_str().ok_or("profile tool is not UTF-8")?,
            "--kernel" => kernel = Some(value.to_str().ok_or("kernel filter is not UTF-8")?),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            option => return Err(format!("unknown profile option `{option}`").into()),
        }
    }
    if !(1..=8).contains(&batch) || replays == 0 {
        return Err("resident profile requires `--batch 1..=8` and nonzero `--replays`".into());
    }
    if !matches!(tool, "nsys" | "ncu") {
        return Err("resident profile tool must be `nsys` or `ncu`".into());
    }
    if tool == "ncu" && kernel.is_none() {
        return Err("resident `ncu` profiling requires `--kernel REGEX`".into());
    }

    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let output_dir =
        output_dir.unwrap_or_else(|| root.join(format!("target/profiles/resident-model-b{batch}")));
    fs::create_dir_all(&output_dir)?;
    let stem = format!("resident-model-b{batch}");
    let graph_dot = output_dir.join(format!("{stem}-graph.dot"));
    let manifest = output_dir.join(format!("{stem}-semantic.json"));
    let profile_prefix = output_dir.join(format!("{stem}-{tool}"));
    let warmup_launches = if tool == "ncu" { 1 } else { 16 };
    let profile_arguments = [
        "profile-resident-model".into(),
        snapshot.clone(),
        "--batch".into(),
        batch.to_string().into(),
        "--warmup-launches".into(),
        warmup_launches.to_string().into(),
        "--captured-replays".into(),
        replays.to_string().into(),
        "--graph-dot".into(),
        graph_dot.as_os_str().to_os_string(),
        "--manifest".into(),
        manifest.as_os_str().to_os_string(),
    ];
    let tool_path = cuda_tool(tool);
    let mut command = Command::new(&tool_path);
    if tool == "nsys" {
        command.args([
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
        ]);
        command.arg("--output").arg(&profile_prefix);
    } else {
        command.args([
            "--profile-from-start=off",
            "--target-processes=all",
            "--set=full",
            "--launch-count=1",
            "--force-overwrite",
        ]);
        command
            .arg("--kernel-name")
            .arg(kernel.expect("ncu kernel filter checked"))
            .arg("--export")
            .arg(&profile_prefix);
    }
    command.arg(&executable).args(&profile_arguments);
    run_visible(&mut command)?;

    if tool == "nsys" {
        let report = profile_prefix.with_extension("nsys-rep");
        let sqlite = output_dir.join(format!("{stem}-nsys.sqlite"));
        run_visible(
            Command::new(&tool_path)
                .args(["export", "--type=sqlite", "--force-overwrite=true"])
                .arg("--output")
                .arg(&sqlite)
                .arg(&report),
        )?;
        postprocess_resident_nsys(&sqlite, &manifest, &output_dir, &stem)?;
    }

    let tool_identity = run_captured(Command::new(&tool_path).arg("--version"))?;
    let git_commit = command_text("git", &["-C", path_text(root)?, "rev-parse", "HEAD"])?;
    let git_status = command_text("git", &["-C", path_text(root)?, "status", "--short"])?;
    let device = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw.instant",
            "--format=csv,noheader",
        ],
    )?;
    let metadata = serde_json::json!({
        "schema_version": 1,
        "scope": "resident-model",
        "batch_size": batch,
        "captured_replays": replays,
        "warmup_launches": warmup_launches,
        "tool": tool,
        "tool_path": tool_path,
        "tool_identity": tool_identity.trim(),
        "binary_sha256": sha256(&fs::read(&executable)?),
        "git_commit": git_commit.trim(),
        "git_status": git_status.lines().collect::<Vec<_>>(),
        "device_after_capture": device.trim(),
        "graph_dot": graph_dot,
        "semantic_manifest": manifest,
    });
    let metadata_path = output_dir.join(format!("{stem}-{tool}-metadata.json"));
    let mut json = serde_json::to_vec_pretty(&metadata)?;
    json.push(b'\n');
    fs::write(&metadata_path, json)?;
    println!("profile artifacts: {}", output_dir.display());

    Ok(())
}

#[derive(Clone)]
struct ProfileKernelSample {
    start_ns: u64,
    end_ns: u64,
}

struct ProfileNodeSamples {
    graph_node_id: u64,
    kernel: String,
    samples: Vec<ProfileKernelSample>,
}

struct SemanticProfileNode {
    layer: Option<usize>,
    component: String,
    source_route: String,
    kernel_family: String,
}

fn postprocess_resident_nsys(
    sqlite: &Path,
    manifest: &Path,
    output_dir: &Path,
    stem: &str,
) -> Result<(), Box<dyn Error>> {
    let manifest_json: serde_json::Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let expected_nodes = manifest_json["graph_kernel_nodes"]
        .as_u64()
        .ok_or("resident semantic manifest omits graph_kernel_nodes")?
        as usize;
    let expected_replays = manifest_json["captured_replays"]
        .as_u64()
        .ok_or("resident semantic manifest omits captured_replays")?
        as usize;
    let semantic_nodes = semantic_profile_nodes(&manifest_json)?;
    if semantic_nodes.len() != expected_nodes {
        return Err(format!(
            "semantic manifest expands to {} nodes, expected {expected_nodes}",
            semantic_nodes.len()
        )
        .into());
    }
    let connection = Connection::open_with_flags(sqlite, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let graph_id = connection
        .query_row(
            "SELECT graphId
             FROM CUPTI_ACTIVITY_KIND_KERNEL
             WHERE graphId IS NOT NULL
             GROUP BY graphId
             HAVING COUNT(DISTINCT graphNodeId) = ?1
             ORDER BY graphId
             LIMIT 1",
            [expected_nodes as i64],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .ok_or_else(|| format!("Nsight SQLite has no graph with {expected_nodes} kernel nodes"))?;
    let mut statement = connection.prepare(
        "SELECT k.graphNodeId, k.start, k.end, s.value
         FROM CUPTI_ACTIVITY_KIND_KERNEL k
         JOIN StringIds s ON s.id = k.shortName
         WHERE k.graphId = ?1
         ORDER BY k.graphNodeId, k.start",
    )?;
    let rows = statement.query_map([graph_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    let mut by_node = BTreeMap::<u64, ProfileNodeSamples>::new();
    for row in rows {
        let (graph_node_id, start_ns, end_ns, kernel) = row?;
        let graph_node_id = u64::try_from(graph_node_id)?;
        let start_ns = u64::try_from(start_ns)?;
        let end_ns = u64::try_from(end_ns)?;
        let entry = by_node
            .entry(graph_node_id)
            .or_insert_with(|| ProfileNodeSamples {
                graph_node_id,
                kernel: kernel.clone(),
                samples: Vec::new(),
            });
        if entry.kernel != kernel {
            return Err(format!(
                "graph node {graph_node_id} changed kernel from `{}` to `{kernel}`",
                entry.kernel
            )
            .into());
        }
        entry.samples.push(ProfileKernelSample { start_ns, end_ns });
    }
    let nodes = by_node.into_values().collect::<Vec<_>>();
    if nodes.len() != expected_nodes {
        return Err(format!(
            "Nsight graph {graph_id} contains {} kernel nodes, expected {expected_nodes}",
            nodes.len()
        )
        .into());
    }
    for (ordinal, (node, semantic)) in nodes.iter().zip(&semantic_nodes).enumerate() {
        if node.samples.len() != expected_replays {
            return Err(format!(
                "graph node {} has {} captured samples, expected {expected_replays}",
                node.graph_node_id,
                node.samples.len()
            )
            .into());
        }
        if !node.kernel.contains(&semantic.kernel_family) {
            return Err(format!(
                "semantic node {} expects kernel family `{}`, Nsight observed `{}`",
                ordinal + 1,
                semantic.kernel_family,
                node.kernel
            )
            .into());
        }
    }

    let mut graph_spans = vec![0.0; expected_replays];
    let mut graph_kernel_sums = vec![0.0; expected_replays];
    for replay in 0..expected_replays {
        let start = nodes
            .iter()
            .map(|node| node.samples[replay].start_ns)
            .min()
            .expect("nonempty graph node inventory");
        let end = nodes
            .iter()
            .map(|node| node.samples[replay].end_ns)
            .max()
            .expect("nonempty graph node inventory");
        graph_spans[replay] = (end - start) as f64 / 1_000.0;
        graph_kernel_sums[replay] = nodes
            .iter()
            .map(|node| {
                let sample = &node.samples[replay];
                (sample.end_ns - sample.start_ns) as f64 / 1_000.0
            })
            .sum();
    }
    write_profile_node_csv(output_dir, stem, &nodes, &semantic_nodes)?;
    write_profile_stage_csv(
        output_dir,
        stem,
        &nodes,
        &semantic_nodes,
        expected_replays,
        mean(&graph_spans),
    )?;
    write_profile_layer_csv(
        output_dir,
        stem,
        &nodes,
        &semantic_nodes,
        expected_replays,
        mean(&graph_spans),
    )?;
    let mut replay_csv = String::from("replay,graph_span_us,kernel_sum_us,gaps_us,gap_percent\n");
    for replay in 0..expected_replays {
        let gaps = graph_spans[replay] - graph_kernel_sums[replay];
        replay_csv.push_str(&format!(
            "{},{:.3},{:.3},{:.3},{:.4}\n",
            replay + 1,
            graph_spans[replay],
            graph_kernel_sums[replay],
            gaps,
            gaps / graph_spans[replay] * 100.0,
        ));
    }
    fs::write(
        output_dir.join(format!("{stem}-replay-timings.csv")),
        replay_csv,
    )?;
    println!(
        "profile closure: {:.3} us graph, {:.3} us kernels, {:.3} us gaps ({:.2}%)",
        mean(&graph_spans),
        mean(&graph_kernel_sums),
        mean(&graph_spans) - mean(&graph_kernel_sums),
        (mean(&graph_spans) - mean(&graph_kernel_sums)) / mean(&graph_spans) * 100.0,
    );

    Ok(())
}

fn semantic_profile_nodes(
    manifest: &serde_json::Value,
) -> Result<Vec<SemanticProfileNode>, Box<dyn Error>> {
    let stages = manifest["stages"]
        .as_array()
        .ok_or("resident semantic manifest omits stages")?;
    let mut nodes = Vec::new();
    for stage in stages {
        let layer = stage["layer"].as_u64().map(|layer| layer as usize);
        let component = stage["component"]
            .as_str()
            .ok_or("resident semantic stage omits component")?;
        let source_route = stage["source_route"]
            .as_str()
            .ok_or("resident semantic stage omits source_route")?;
        let families = stage["kernel_families"]
            .as_array()
            .ok_or("resident semantic stage omits kernel_families")?;
        for family in families {
            nodes.push(SemanticProfileNode {
                layer,
                component: component.to_string(),
                source_route: source_route.to_string(),
                kernel_family: family
                    .as_str()
                    .ok_or("resident semantic kernel family is not text")?
                    .to_string(),
            });
        }
    }

    Ok(nodes)
}

fn write_profile_node_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "ordinal,graph_node_id,layer,component,source_route,kernel,mean_us,median_us,min_us,max_us\n",
    );
    for (ordinal, (node, semantic)) in nodes.iter().zip(semantic).enumerate() {
        let values = node
            .samples
            .iter()
            .map(|sample| (sample.end_ns - sample.start_ns) as f64 / 1_000.0)
            .collect::<Vec<_>>();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}\n",
            ordinal + 1,
            node.graph_node_id,
            semantic
                .layer
                .map_or_else(|| "-".to_string(), |layer| layer.to_string()),
            semantic.component,
            semantic.source_route,
            csv_field(&node.kernel),
            mean(&values),
            median(&values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
        ));
    }
    fs::write(output_dir.join(format!("{stem}-node-timings.csv")), csv)?;
    Ok(())
}

fn write_profile_stage_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
    replays: usize,
    graph_mean_us: f64,
) -> Result<(), Box<dyn Error>> {
    let mut groups = Vec::<(Option<usize>, String, String, Vec<f64>, usize)>::new();
    for (node, semantic) in nodes.iter().zip(semantic) {
        let new_group = groups.last().is_none_or(|(layer, component, route, _, _)| {
            *layer != semantic.layer
                || component != &semantic.component
                || route != &semantic.source_route
        });
        if new_group {
            groups.push((
                semantic.layer,
                semantic.component.clone(),
                semantic.source_route.clone(),
                vec![0.0; replays],
                0,
            ));
        }
        let group = groups.last_mut().expect("stage group was inserted");
        group.4 += 1;
        for replay in 0..replays {
            group.3[replay] +=
                (node.samples[replay].end_ns - node.samples[replay].start_ns) as f64 / 1_000.0;
        }
    }
    let mut csv = String::from(
        "ordinal,layer,component,source_route,kernel_nodes,mean_us,median_us,min_us,max_us,graph_share_percent\n",
    );
    for (ordinal, (layer, component, route, values, kernel_nodes)) in groups.iter().enumerate() {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.4}\n",
            ordinal + 1,
            layer.map_or_else(|| "-".to_string(), |layer| layer.to_string()),
            component,
            route,
            kernel_nodes,
            mean(values),
            median(values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
            mean(values) / graph_mean_us * 100.0,
        ));
    }
    fs::write(output_dir.join(format!("{stem}-stage-timings.csv")), csv)?;
    Ok(())
}

fn write_profile_layer_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
    replays: usize,
    graph_mean_us: f64,
) -> Result<(), Box<dyn Error>> {
    let mut layers = BTreeMap::<i32, (String, Vec<f64>)>::new();
    for (node, semantic) in nodes.iter().zip(semantic) {
        let layer = semantic.layer.map_or_else(
            || {
                if semantic.component == "input_norm" {
                    -1
                } else {
                    64
                }
            },
            |layer| layer as i32,
        );
        let entry = layers
            .entry(layer)
            .or_insert_with(|| (semantic.source_route.clone(), vec![0.0; replays]));
        for replay in 0..replays {
            entry.1[replay] +=
                (node.samples[replay].end_ns - node.samples[replay].start_ns) as f64 / 1_000.0;
        }
    }
    let mut csv =
        String::from("layer,source_route,mean_us,median_us,min_us,max_us,graph_share_percent\n");
    for (layer, (route, values)) in layers {
        let label = match layer {
            -1 => "input".to_string(),
            64 => "endpoint".to_string(),
            _ => layer.to_string(),
        };
        csv.push_str(&format!(
            "{label},{route},{:.3},{:.3},{:.3},{:.3},{:.4}\n",
            mean(&values),
            median(&values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
            mean(&values) / graph_mean_us * 100.0,
        ));
    }
    fs::write(output_dir.join(format!("{stem}-layer-timings.csv")), csv)?;
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn perf(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some(mode) = arguments.first() else {
        return Err(
            "usage: cargo run -p xtask -- perf <smoke|leaf|energy|gate|candidate|check|bless|iterate|diagnose-diff> ...".into(),
        );
    };
    let mode = mode.to_str().ok_or("perf mode is not UTF-8")?;
    if mode == "diagnose-diff" {
        return diagnose_performance_report(root, &arguments[1..]);
    }
    if mode == "iterate" {
        let options = parse_performance_iteration(&arguments[1..])?;
        return run_performance_iteration(root, options);
    }
    if mode == "bless" {
        let suite = arguments
            .get(1)
            .ok_or("usage: cargo run -p xtask -- perf bless SUITE [SNAPSHOT]")?;
        let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
        let snapshot = if suite.requires_snapshot() {
            let [_, _, snapshot] = arguments else {
                return Err(format!("perf bless {} requires SNAPSHOT", suite.name()).into());
            };
            Some(snapshot.as_os_str())
        } else {
            if arguments.len() != 2 {
                return Err(format!("perf bless {} takes no SNAPSHOT", suite.name()).into());
            }
            None
        };
        require_performance_device_idle()?;
        return bless_optimization_suite(root, suite, snapshot);
    }
    if matches!(mode, "candidate" | "check") {
        let suite = arguments.get(1).ok_or(
            "usage: cargo run -p xtask -- perf candidate|check SUITE [SNAPSHOT] [options]",
        )?;
        let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
        let cone = suite.dependency_cone();
        let needs_snapshot = cone.iter().any(|suite| suite.requires_snapshot());
        let mut remaining = &arguments[2..];
        let snapshot = if needs_snapshot {
            let (snapshot, options) = remaining.split_first().ok_or_else(|| {
                format!(
                    "perf {mode} {} requires SNAPSHOT for its composed dependency cone",
                    suite.name()
                )
            })?;
            remaining = options;
            Some(snapshot.as_os_str())
        } else {
            None
        };
        if mode == "check" && !remaining.is_empty() {
            return Err("`perf check` uses the complete authoritative suite defaults".into());
        }
        require_performance_device_idle()?;
        return run_optimization_cone(root, mode, suite, &cone, snapshot, remaining);
    }
    if arguments.len() != 1 {
        return Err(format!("`perf {mode}` takes no additional arguments").into());
    }

    let options = match mode {
        "smoke" => vec!["--samples".into(), "3".into()],
        "leaf" | "gate" => Vec::new(),
        "energy" => vec!["--energy-seconds".into(), "2".into()],
        _ => return Err(format!("unknown perf mode `{mode}`").into()),
    };
    require_performance_device_idle()?;
    if mode == "gate" {
        for suite in PERFORMANCE_SUITES {
            suite.qualify(root)?;
        }
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    run_performance_suites(root, mode, &options, mode == "gate")
}

struct PerformanceIterationOptions {
    suite: PerformanceSuite,
    batch_size: u32,
    hypothesis: String,
}

fn parse_performance_iteration(
    arguments: &[std::ffi::OsString],
) -> Result<PerformanceIterationOptions, Box<dyn Error>> {
    let Some((suite, remaining)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- perf iterate SUITE --batch B --hypothesis TEXT".into(),
        );
    };
    let suite = PerformanceSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
    let mut batch_size = None;
    let mut hypothesis = None;
    let mut options = remaining.iter();
    while let Some(option) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| format!("`{}` requires a value", option.to_string_lossy()))?;
        match option.to_str().ok_or("perf iterate option is not UTF-8")? {
            "--batch" if batch_size.is_none() => {
                batch_size = Some(value.to_str().ok_or("batch is not UTF-8")?.parse()?);
            }
            "--hypothesis" if hypothesis.is_none() => {
                hypothesis = Some(
                    value
                        .to_str()
                        .ok_or("performance hypothesis is not UTF-8")?
                        .trim()
                        .to_string(),
                );
            }
            option => {
                return Err(format!("unknown or duplicate perf iterate option `{option}`").into());
            }
        }
    }
    let batch_size = batch_size.ok_or("perf iterate requires `--batch B`")?;
    if !(1..=8).contains(&batch_size) {
        return Err("perf iterate requires `--batch 1..=8`".into());
    }
    let hypothesis = hypothesis.ok_or("perf iterate requires `--hypothesis TEXT`")?;
    if hypothesis.is_empty() {
        return Err("perf iterate requires a nonempty hypothesis".into());
    }

    Ok(PerformanceIterationOptions {
        suite,
        batch_size,
        hypothesis,
    })
}

fn run_performance_iteration(
    root: &Path,
    options: PerformanceIterationOptions,
) -> Result<(), Box<dyn Error>> {
    let suite = options.suite;
    let baseline = root.join(suite.performance_baseline());
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let mut recorder = perf_iteration::IterationRecorder::start(
        root,
        suite.name(),
        options.batch_size,
        options.hypothesis,
        device_inputs.clone(),
        &baseline,
    )?;

    let started = Instant::now();
    let device_identity = match require_performance_device_idle()
        .and_then(|()| performance_device_identity_sha256())
    {
        Ok(identity) => {
            recorder.record_stage("preflight", "passed", started.elapsed());
            identity
        }
        Err(error) => {
            recorder.record_stage("preflight", "refused", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    };

    let started = Instant::now();
    match perf_artifact::qualification_is_current(
        root,
        suite.name(),
        &device_inputs,
        &device_identity,
    ) {
        Ok(true) => recorder.record_stage("qualification", "reused", started.elapsed()),
        Ok(false) => {
            if let Err(error) = suite.qualify(root).and_then(|()| {
                perf_artifact::record_qualification(
                    root,
                    suite.name(),
                    device_inputs.clone(),
                    device_identity.clone(),
                )
            }) {
                recorder.record_stage("qualification", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
            recorder.record_stage("qualification", "passed", started.elapsed());
        }
        Err(error) => {
            recorder.record_stage("qualification", "failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    }

    let started = Instant::now();
    let resource_baselines =
        match perf_artifact::resource_baselines_sha256(root, SM120_RESOURCE_BASELINES) {
            Ok(value) => value,
            Err(error) => {
                recorder.record_stage("build", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
        };
    match perf_artifact::local_build_is_current(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    ) {
        Ok(true) => recorder.record_stage("build", "reused", started.elapsed()),
        Ok(false) => {
            if let Err(error) = build_sm120(root) {
                recorder.record_stage("build", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
            recorder.record_stage("build", "passed", started.elapsed());
        }
        Err(error) => {
            recorder.record_stage("build", "failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    }

    let started = Instant::now();
    let benchmark_arguments = [
        "--batch".into(),
        options.batch_size.to_string().into(),
        "--json".into(),
        recorder.report_path().into_os_string(),
    ];
    if let Err(error) =
        wait_for_device_idle().and_then(|()| run_benchmark_suite(root, suite, &benchmark_arguments))
    {
        recorder.record_stage("benchmark", "refused_or_failed", started.elapsed());
        return fail_performance_iteration(recorder, error);
    }
    recorder.record_stage("benchmark", "passed", started.elapsed());

    let started = Instant::now();
    let diagnostic = match performance::diagnose(&recorder.report_path(), &baseline) {
        Ok(diagnostic) => diagnostic,
        Err(error) => {
            recorder.record_stage("comparison", "refused_or_failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    };
    recorder.record_stage("comparison", "diagnostic_only", started.elapsed());
    let output = recorder.succeed(diagnostic)?;
    println!(
        "diagnostic optimization iteration preserved at {}",
        output.display()
    );
    Ok(())
}

fn fail_performance_iteration(
    recorder: perf_iteration::IterationRecorder,
    error: Box<dyn Error>,
) -> Result<(), Box<dyn Error>> {
    let error_text = error.to_string();
    match recorder.fail(error.as_ref()) {
        Ok(output) => eprintln!(
            "failed or refused optimization iteration preserved at {}",
            output.display()
        ),
        Err(record_error) => {
            return Err(format!(
                "{error_text}; additionally failed to preserve the iteration record: {record_error}"
            )
            .into());
        }
    }
    Err(error)
}

fn diagnose_performance_report(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let (suite, report, output) = match arguments {
        [suite, report] => (suite, report, None),
        [suite, report, option, output] if option == "--json" => (suite, report, Some(output)),
        _ => {
            return Err(
                "usage: cargo run -p xtask -- perf diagnose-diff SUITE REPORT [--json OUTPUT]"
                    .into(),
            );
        }
    };
    let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
    let report = resolve_repository_path(root, report);
    let output = match output {
        Some(path) => resolve_target_output(root, path)?,
        None => root.join(format!(
            "target/benchmarks/perf-diagnostic/{}.json",
            suite.name()
        )),
    };
    let diagnostic = performance::diagnose(&report, &root.join(suite.performance_baseline()))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(&diagnostic)?;
    json.push(b'\n');
    fs::write(&output, json)?;
    println!("diagnostic comparison JSON: {}", output.display());
    Ok(())
}

fn resolve_repository_path(root: &Path, path: &OsStr) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn resolve_target_output(root: &Path, path: &OsStr) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        || !path.starts_with("target")
    {
        return Err("diagnostic output must be a repository-relative path under `target/`".into());
    }
    Ok(root.join(path))
}

fn run_performance_suites(
    root: &Path,
    mode: &str,
    options: &[std::ffi::OsString],
    compare: bool,
) -> Result<(), Box<dyn Error>> {
    for (index, suite) in PERFORMANCE_SUITES.into_iter().enumerate() {
        if index != 0 {
            wait_for_device_idle()?;
        }
        let report = performance_report_path(root, mode, suite);
        let mut arguments = options.to_vec();
        arguments.push("--json".into());
        arguments.push(path_text(&report)?.into());
        run_benchmark_suite(root, suite, &arguments)?;
    }
    if compare {
        for suite in PERFORMANCE_SUITES {
            let report = performance_report_path(root, mode, suite);
            performance::compare(&report, &root.join(suite.performance_baseline()))?;
        }
    }

    Ok(())
}

fn run_optimization_cone(
    root: &Path,
    mode: &str,
    root_suite: OptimizationSuite,
    cone: &[OptimizationSuite],
    snapshot: Option<&OsStr>,
    options: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let device_identity = performance_device_identity_sha256()?;
    if mode == "check" {
        let mut qualified = Vec::new();
        for suite in cone.iter().copied() {
            let authority = if suite == OptimizationSuite::ResidentLongContextModel {
                OptimizationSuite::ResidentModel
            } else {
                suite
            };
            if !qualified.contains(&authority) {
                authority.qualify(root, snapshot)?;
                if let OptimizationSuite::Leaf(leaf) = authority {
                    perf_artifact::record_qualification(
                        root,
                        leaf.name(),
                        device_inputs.clone(),
                        device_identity.clone(),
                    )?;
                }
                qualified.push(authority);
            }
        }
    } else {
        root_suite.qualify(root, snapshot)?;
        if let OptimizationSuite::Leaf(leaf) = root_suite {
            perf_artifact::record_qualification(root, leaf.name(), device_inputs, device_identity)?;
        }
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    for (index, suite) in cone.iter().copied().enumerate() {
        if index != 0 {
            wait_for_device_idle()?;
        }
        let report = optimization_report_path(root, mode, root_suite, suite);
        let mut arguments = options.to_vec();
        arguments.push("--json".into());
        arguments.push(path_text(&report)?.into());
        run_optimization_benchmark(root, suite, snapshot, &arguments)?;
        if mode == "check" {
            performance::compare(&report, &root.join(suite.performance_baseline()))?;
        }
    }

    Ok(())
}

fn wait_for_device_idle() -> Result<(), Box<dyn Error>> {
    let timeout = Duration::from_secs(10);
    let deadline = Instant::now() + timeout;
    loop {
        let row = command_text(
            "nvidia-smi",
            &[
                "-i",
                "0",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
        )?;
        let (utilization, memory_mib) = parse_idle_sample(&row)?;
        if utilization == 0 && memory_mib <= 1_024 {
            return require_performance_device_idle();
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "device zero remained busy for {} seconds between performance suites: utilization={utilization}%, memory={memory_mib} MiB",
                timeout.as_secs()
            )
            .into());
        }

        std::thread::sleep(Duration::from_millis(100));
    }
}

fn require_performance_device_idle() -> Result<(), Box<dyn Error>> {
    if env::var_os("CUDA_VISIBLE_DEVICES").is_some_and(|value| value != "0") {
        return Err(
            "performance commands require CUDA_VISIBLE_DEVICES to be unset or exactly `0`".into(),
        );
    }
    let row = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let (device, utilization, memory_mib) = parse_performance_device_sample(&row)?;
    let expected = gpu_target::GpuTarget::Sm120.device_name();
    if device != expected {
        return Err(format!(
            "performance commands require device zero to be `{expected}`, found `{device}`"
        )
        .into());
    }
    if utilization != 0 || memory_mib > 1_024 {
        return Err(format!(
            "device zero is busy before performance setup: utilization={utilization}%, memory={memory_mib} MiB"
        )
        .into());
    }
    let processes = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let pids = parse_compute_pids(&processes)?;
    if !pids.is_empty() {
        return Err(format!(
            "device zero has foreign compute processes before performance setup: {pids:?}"
        )
        .into());
    }

    Ok(())
}

fn performance_device_identity_sha256() -> Result<String, Box<dyn Error>> {
    let identity = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )?;
    Ok(sha256(identity.trim().as_bytes()))
}

fn parse_performance_device_sample(row: &str) -> Result<(String, u32, u64), Box<dyn Error>> {
    let fields = row.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [device, utilization, memory_mib] = fields.as_slice() else {
        return Err(format!(
            "unexpected nvidia-smi performance preflight row `{}`",
            row.trim()
        )
        .into());
    };

    Ok((
        (*device).to_string(),
        utilization.parse()?,
        memory_mib.parse()?,
    ))
}

fn parse_compute_pids(output: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != "No running processes found")
        .map(|line| {
            line.parse().map_err(|error| {
                format!("unexpected nvidia-smi compute PID `{line}`: {error}").into()
            })
        })
        .collect()
}

fn parse_idle_sample(row: &str) -> Result<(u32, u64), Box<dyn Error>> {
    let fields = row.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [utilization, memory_mib] = fields.as_slice() else {
        return Err(format!("unexpected nvidia-smi idle row `{}`", row.trim()).into());
    };

    Ok((utilization.parse()?, memory_mib.parse()?))
}

fn sass_function_body<'a>(sass: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("Function : {name}");
    let body = &sass[sass.find(&marker)? + marker.len()..];

    Some(body.split("\n\t\tFunction :").next().unwrap_or(body))
}

fn bless_optimization_suite(
    root: &Path,
    suite: OptimizationSuite,
    snapshot: Option<&OsStr>,
) -> Result<(), Box<dyn Error>> {
    suite.qualify(root, snapshot)?;
    if let OptimizationSuite::Leaf(leaf) = suite {
        perf_artifact::record_qualification(
            root,
            leaf.name(),
            perf_artifact::device_input_sha256(root)?,
            performance_device_identity_sha256()?,
        )?;
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    let report = optimization_report_path(root, "bless", suite, suite);
    run_optimization_benchmark(
        root,
        suite,
        snapshot,
        &["--json".into(), path_text(&report)?.into()],
    )?;
    performance::bless(&report, &root.join(suite.performance_baseline()))
}

fn performance_report_path(root: &Path, mode: &str, suite: PerformanceSuite) -> PathBuf {
    root.join(format!(
        "target/benchmarks/perf-{mode}/{}.json",
        suite.name()
    ))
}

fn optimization_report_path(
    root: &Path,
    mode: &str,
    root_suite: OptimizationSuite,
    suite: OptimizationSuite,
) -> PathBuf {
    root.join(format!(
        "target/benchmarks/perf-{mode}/{}/{}.json",
        root_suite.name(),
        suite.name()
    ))
}

fn run_oxide(root: &Path, arguments: &[&str]) -> Result<(), Box<dyn Error>> {
    run_oxide_with_env(root, arguments, None)
}

fn run_oxide_with_env(
    root: &Path,
    arguments: &[&str],
    environment: Option<(&str, &OsStr)>,
) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    require_cuda_oxide_revision(&source)?;
    let wrapper = root.join("target/cuda-oxide-driver/debug/cargo-oxide");
    if !wrapper.is_file() {
        return Err(
            "cuda-oxide is not bootstrapped; run `cargo run -p xtask -- bootstrap-cuda-oxide`"
                .into(),
        );
    }
    let backend = local_backend(root)?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    fs::create_dir_all(root.join("target/tmp"))?;
    let mut command = Command::new(wrapper);
    command
        .args(arguments)
        .current_dir(root)
        .env("CARGO_HOME", task_cargo_home(root))
        .env("CUDA_OXIDE_BACKEND", backend)
        .env("CUDA_OXIDE_SOURCE", source)
        .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
        .env_remove("RUSTFLAGS")
        .env("TMPDIR", root.join("target/tmp"));
    if let Some((name, value)) = environment {
        command.env(name, value);
    }

    run_visible(&mut command)
}

/// Same as `run_oxide` but captures stdout so the caller can locate
/// build artifacts (remote prepare builds with `--no-run`).
fn run_oxide_capture(root: &Path, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    require_cuda_oxide_revision(&source)?;
    let wrapper = root.join("target/cuda-oxide-driver/debug/cargo-oxide");
    if !wrapper.is_file() {
        return Err(
            "cuda-oxide is not bootstrapped; run `cargo run -p xtask -- bootstrap-cuda-oxide`"
                .into(),
        );
    }
    let backend = local_backend(root)?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    fs::create_dir_all(root.join("target/tmp"))?;
    run_captured(
        Command::new(wrapper)
            .args(arguments)
            .current_dir(root)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CUDA_OXIDE_BACKEND", backend)
            .env("CUDA_OXIDE_SOURCE", source)
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS")
            .env("TMPDIR", root.join("target/tmp")),
    )
}

/// Runs a command with stdout piped (stderr inherited) and returns stdout.
fn run_captured(command: &mut Command) -> Result<String, Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let output = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(format!("{program} failed with {}", output.status).into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A compiled qual test executable ready to ship to a gate pod.
pub struct RemoteQualify {
    /// Path to the compiled qual test executable.
    pub executable: PathBuf,
    /// Arguments to run the executable with (libtest filters + flags).
    pub test_args: Vec<String>,
}

/// A compiled benchmark executable and its matching resource identity.
pub(crate) struct RemoteBenchmark {
    /// Path to the compiled `bench-device` executable.
    pub executable: PathBuf,
    /// Hash of the suite's checked static-resource baseline.
    pub generator_baseline_sha256: String,
}

/// Locates the already-built benchmark executable and binds it to one suite's resources.
pub(crate) fn prepare_remote_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    suite: PerformanceSuite,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    let resource_baseline = match suite {
        PerformanceSuite::ResidualNorm => gpu.residual_resource_baseline(),
        PerformanceSuite::Nvfp4SwiGlu => gpu
            .nvfp4_swiglu_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no NVFP4 SwiGLU resource baseline", gpu.key()))?,
        PerformanceSuite::Nvfp4Down => gpu
            .nvfp4_down_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no NVFP4 down resource baseline", gpu.key()))?,
        PerformanceSuite::Fp8Qkv => gpu
            .fp8_qkv_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no FP8 QKV resource baseline", gpu.key()))?,
        _ => suite.resource_baseline(),
    };
    prepare_remote_benchmark_with_baselines(root, gpu, &[resource_baseline])
}

/// Locates the composed NVFP4 MLP benchmark and binds it to all leaf resources it launches.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_nvfp4_mlp_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(
        root,
        gpu,
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            NVFP4_SWIGLU_RESOURCE_BASELINE,
            NVFP4_DOWN_RESOURCE_BASELINE,
        ],
    )
}

/// Locates the composed full-attention benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_full_attention_layer_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(
        root,
        gpu,
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_QKV_RESOURCE_BASELINE,
            ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
            PAGED_GQA_RESOURCE_BASELINE,
            ATTENTION_OUTPUT_RESOURCE_BASELINE,
            FP8_SWIGLU_RESOURCE_BASELINE,
            FP8_DOWN_RESOURCE_BASELINE,
        ],
    )
}

/// Locates the complete resident-model benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_resident_model_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(root, gpu, RESIDENT_MODEL_RESOURCE_BASELINES)
}

fn prepare_remote_benchmark_with_baselines(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    baselines: &[&str],
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    let built = root
        .join(gpu.oxide_build_target())
        .join("release/bench-device");
    if !built.is_file() {
        return Err(format!("benchmark executable is missing at {}", built.display()).into());
    }
    let artifact_name = format!("bench-device-{}", gpu.key());
    let executable = strip_remote_artifact(root, &built, &artifact_name)?;
    let mut resources = Vec::new();
    for baseline in baselines {
        resources.extend_from_slice(&fs::read(root.join(baseline))?);
    }

    Ok(RemoteBenchmark {
        executable,
        generator_baseline_sha256: sha256(&resources),
    })
}

/// Compiles the qual test executable for a remote `qualify-*` gate.
///
/// No GPU is needed locally: the cuda-oxide codegen and ptxas steps run
/// at compile time; the pod only needs the driver to execute the binary.
pub(crate) fn prepare_remote_qualify(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    test_filter: &str,
) -> Result<RemoteQualify, Box<dyn Error>> {
    let test_args = [
        test_filter,
        "--include-ignored",
        "--nocapture",
        "--test-threads=1",
    ];
    let arguments = vec![
        "test".to_string(),
        "--arch".to_string(),
        gpu.oxide_arch().to_string(),
        "--cargo-target-dir".to_string(),
        gpu.oxide_test_target().to_string(),
        "--device-codegen-crate".to_string(),
        gpu.kernel_crate().to_string(),
        "--".to_string(),
        "--package".to_string(),
        "tuisko-qual".to_string(),
        "--release".to_string(),
        "--lib".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        gpu.qualification_feature().to_string(),
        "--message-format=json-render-diagnostics".to_string(),
        "--no-run".to_string(),
        "--".to_string(),
        test_args[0].to_string(),
        test_args[1].to_string(),
        test_args[2].to_string(),
        test_args[3].to_string(),
    ];
    let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    let messages = run_oxide_capture(root, &argument_refs)?;
    let built = messages
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message.get("reason").and_then(serde_json::Value::as_str) == Some("compiler-artifact")
        })
        .filter(|message| {
            message
                .pointer("/target/name")
                .and_then(serde_json::Value::as_str)
                == Some("tuisko_qual")
        })
        .filter_map(|message| {
            message
                .get("executable")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from)
        })
        .next_back()
        .ok_or("cargo did not report the tuisko-qual test executable")?;
    if !built.is_file() {
        return Err(format!(
            "cargo reported a missing qualification executable at {}",
            built.display()
        )
        .into());
    }
    let artifact_name = format!("tuisko-qual-{}", gpu.key());
    let executable = strip_remote_artifact(root, &built, &artifact_name)?;
    println!("remote test binary: {}", executable.display());
    Ok(RemoteQualify {
        executable,
        test_args: test_args.into_iter().map(String::from).collect(),
    })
}

fn strip_remote_artifact(
    root: &Path,
    source: &Path,
    name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let directory = root.join("target/remote-artifacts");
    fs::create_dir_all(&directory)?;
    let output = directory.join(name);
    run_visible(
        Command::new("strip")
            .args(["--strip-all", "-o"])
            .arg(&output)
            .arg(source),
    )?;
    let source_bytes = fs::metadata(source)?.len();
    let output_bytes = fs::metadata(&output)?.len();
    println!(
        "remote artifact: {} -> {} bytes ({:.1}% smaller)",
        source_bytes,
        output_bytes,
        100.0 * (source_bytes.saturating_sub(output_bytes)) as f64 / source_bytes as f64,
    );

    Ok(output)
}

fn run_visible(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(format!("{program} failed with {status}").into());
    }

    Ok(())
}

fn require_cuda_oxide_revision(source: &Path) -> Result<(), Box<dyn Error>> {
    let commit = command_text("git", &["-C", path_text(source)?, "rev-parse", "HEAD"])?;
    if commit.trim() != CUDA_OXIDE_REVISION {
        return Err(format!(
            "cuda-oxide source is at {}, expected {}",
            commit.trim(),
            CUDA_OXIDE_REVISION
        )
        .into());
    }

    Ok(())
}

fn gate_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm_target(root, gpu_target::GpuTarget::Sm120)
}

pub(crate) fn gate_residual_norm_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(gpu.residual_resource_baseline()),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(gpu.ptx_path());
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let plain = entries
        .iter()
        .filter(|entry| entry.name == "rms_norm_b1" || entry.name.starts_with("rms_norm_TID_"))
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| entry.name.starts_with("residual_rms_norm_TID_"))
        .collect::<Vec<_>>();
    require_count("plain RMSNorm", plain.len(), 8)?;
    require_count("residual RMSNorm", residual.len(), 8)?;

    for entry in plain.iter().chain(&residual) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        sm120_gate_artifact(root)?.resources.clone()
    } else {
        let temporary = root.join("target/tmp");
        fs::create_dir_all(&temporary)?;
        let cubin = temporary.join(format!("residual-norm-{}-gate.cubin", gpu.key()));
        require_success(
            &cuda_tool("ptxas"),
            &[
                OsStr::new("-O3"),
                OsStr::new("--gpu-name"),
                OsStr::new(gpu.oxide_arch()),
                ptx_path.as_os_str(),
                OsStr::new("--output-file"),
                cubin.as_os_str(),
            ],
        )?;
        let output = require_success(
            &cuda_tool("cuobjdump"),
            &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
        )?;
        parse_resources(&String::from_utf8(output.stdout)?)?
    };
    let mut plain_registers = Vec::new();
    let mut residual_registers = Vec::new();
    let mut shared = Vec::new();

    for entry in plain {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted plain entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        plain_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    for entry in residual {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted residual entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        residual_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "{} residual-norm gate passed: 8 plain + 8 residual entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        gpu.key(),
        plain_registers,
        residual_registers,
        shared
    );
    Ok(())
}

fn gate_qwen35_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let plain = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_rms_norm_TID_"))
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_residual_rms_norm_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 plain RMSNorm", plain.len(), 8)?;
    require_count("Qwen3.5 residual RMSNorm", residual.len(), 8)?;

    for entry in plain.iter().chain(&residual) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut plain_registers = Vec::with_capacity(plain.len());
    let mut residual_registers = Vec::with_capacity(residual.len());
    let mut shared = Vec::with_capacity(plain.len() + residual.len());

    for (family, entries, registers) in [
        ("plain", &plain, &mut plain_registers),
        ("residual", &residual, &mut residual_registers),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!("cuobjdump omitted Qwen3.5 {family} entry `{}`", entry.name)
            })?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["MUFU.RSQ", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5 residual-norm gate passed: 8 plain + 8 residual entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, RSQ/BF16 pack present",
        plain_registers, residual_registers, shared
    );
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn gate_nvfp4_swiglu_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    let baseline_path = gpu
        .nvfp4_swiglu_resource_baseline()
        .ok_or_else(|| format!("GPU {} has no NVFP4 SwiGLU resource baseline", gpu.key()))?;
    gate_nvfp4_a16_target(
        root,
        gpu,
        baseline_path,
        "nvfp4_swiglu_a16_b1",
        "nvfp4_swiglu_a16_TID_",
        "nvfp4-swiglu",
        "NVFP4 SwiGLU",
    )
}

#[cfg(feature = "remote")]
pub(crate) fn gate_nvfp4_down_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    let baseline_path = gpu
        .nvfp4_down_resource_baseline()
        .ok_or_else(|| format!("GPU {} has no NVFP4 down resource baseline", gpu.key()))?;
    gate_nvfp4_a16_target(
        root,
        gpu,
        baseline_path,
        "nvfp4_down_a16_b1",
        "nvfp4_down_a16_TID_",
        "nvfp4-down",
        "NVFP4 down",
    )
}

#[cfg(feature = "remote")]
#[allow(clippy::too_many_arguments)]
fn gate_nvfp4_a16_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    baseline_path: &str,
    singleton_name: &str,
    route_prefix: &str,
    artifact_stem: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(gpu.ptx_path());
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned {} release build first",
            ptx_path.display(),
            gpu.oxide_arch()
        )
    })?;
    let entries = parse_entries(&ptx);
    let routes = entries
        .iter()
        .filter(|entry| entry.name == singleton_name || entry.name.starts_with(route_prefix))
        .collect::<Vec<_>>();
    require_count(label, routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 1") {
            return Err(format!(
                "entry `{}` lost its 256-thread/one-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join(format!("{artifact_stem}-{}-gate.cubin", gpu.key()));
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new(gpu.oxide_arch()),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();

    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted {} NVFP4 entry `{}`",
                gpu.key(),
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "{} {label} gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        gpu.key(),
        registers,
        shared
    );
    Ok(())
}

fn gate_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_QKV_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name == "quantize_activation_e4m3")
        .collect::<Vec<_>>();
    let qkv = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    let qkv_t16 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_qkv_mma_t16")
        .collect::<Vec<_>>();
    let qkv_prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_mma_TID_"))
        .collect::<Vec<_>>();
    let qkv_t1024 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_qkv_mma_t1024")
        .collect::<Vec<_>>();
    require_count("FP8 activation quantization", quantize.len(), 1)?;
    require_count("FP8 QKV", qkv.len(), 8)?;
    require_count("FP8 QKV T=16", qkv_t16.len(), 1)?;
    require_count("FP8 QKV T=32/64/128", qkv_prefill.len(), 3)?;
    require_count("FP8 QKV T=1024", qkv_t1024.len(), 1)?;

    for entry in quantize.iter().chain(&qkv) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !qkv_t16[0].body.contains(".reqntid 64, 1, 1")
        || !qkv_t16[0].body.contains(".minnctapersm 4")
    {
        return Err("FP8 QKV T=16 lost its 64-thread/four-CTA launch bounds".into());
    }
    for entry in qkv_prefill.iter().chain(&qkv_t1024) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA prefill launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in qkv_t16.iter().chain(&qkv_prefill).chain(&qkv_t1024) {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV MMA entry `{}`", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "FP8 QKV MMA entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted FP8 activation quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut qkv_registers = Vec::new();
    let mut qkv_shared = Vec::new();
    for entry in qkv {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_registers.push(resource.registers);
        qkv_shared.push(resource.shared);
    }
    qkv_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &qkv_registers)?;
    let qkv_t16_resource = resources
        .get(qkv_t16[0].name)
        .ok_or("cuobjdump omitted FP8 QKV T=16")?;
    require_spill_free(qkv_t16[0].name, qkv_t16_resource)?;
    require_registers(
        &baseline,
        "qkv_t16_registers",
        &[qkv_t16_resource.registers],
    )?;
    let mut qkv_prefill_registers = Vec::new();
    let mut qkv_prefill_shared = Vec::new();
    for entry in qkv_prefill.iter().chain(&qkv_t1024) {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV prefill entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_prefill_registers.push(resource.registers);
        qkv_prefill_shared.push(resource.shared);
    }
    qkv_prefill_registers.sort_unstable();
    if baseline.contains_key("qkv_prefill_registers") {
        require_registers(&baseline, "qkv_prefill_registers", &qkv_prefill_registers)?;
    }

    println!(
        "FP8 QKV gate passed: 1 quantize + 8 decode + 1 T=16 + 4 prefill projection entries, REG {} / {:?} / {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?} / {} / {:?}",
        quantize_resource.registers,
        qkv_registers,
        qkv_t16_resource.registers,
        qkv_prefill_registers,
        quantize_resource.shared,
        qkv_shared,
        qkv_t16_resource.shared,
        qkv_prefill_shared,
    );
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn gate_fp8_qkv_sm89(root: &Path) -> Result<(), Box<dyn Error>> {
    let gpu = gpu_target::GpuTarget::Sm89;
    let baseline_path = gpu
        .fp8_qkv_resource_baseline()
        .ok_or("SM89 has no FP8 QKV resource baseline")?;
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(gpu.ptx_path());
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned SM89 release build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name == "quantize_activation_e4m3")
        .collect::<Vec<_>>();
    let qkv = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    require_count("SM89 FP8 activation quantization", quantize.len(), 1)?;
    require_count("SM89 FP8 QKV", qkv.len(), 8)?;

    for entry in quantize.iter().chain(&qkv) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-qkv-sm89-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new(gpu.oxide_arch()),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted SM89 FP8 activation quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut qkv_registers = Vec::new();
    let mut qkv_shared = Vec::new();
    for entry in qkv {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted SM89 FP8 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_registers.push(resource.registers);
        qkv_shared.push(resource.shared);
    }
    qkv_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &qkv_registers)?;

    println!(
        "4090 FP8 QKV gate passed: 1 quantize + 8 decode entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}",
        quantize_resource.registers, qkv_registers, quantize_resource.shared, qkv_shared,
    );
    Ok(())
}

fn gate_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_GDN_INPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let gdn_input = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_gdn_input_TID_"))
        .collect::<Vec<_>>();
    require_count("FP8 GDN input", gdn_input.len(), 8)?;

    for entry in &gdn_input {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = &sm120_gate_artifact(root)?.resources;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in gdn_input {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 GDN input entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "gdn_input_registers", &registers)?;

    println!(
        "FP8 GDN input gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let lm_head = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_lm_head_TID_"))
        .collect::<Vec<_>>();
    require_count("FP8 LM head", lm_head.len(), 8)?;

    for entry in &lm_head {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = &sm120_gate_artifact(root)?.resources;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in lm_head {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 LM-head entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;

    println!(
        "FP8 LM-head gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_quantize_TID_"))
        .collect::<Vec<_>>();
    let decode = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_decode_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.name,
                "fp8_swiglu_mma_t32" | "fp8_swiglu_mma_t64" | "fp8_swiglu_mma_t128"
            )
        })
        .collect::<Vec<_>>();
    require_count("dense-FP8 SwiGLU quantization", quantize.len(), 1)?;
    require_count("dense-FP8 SwiGLU decode", decode.len(), 8)?;
    require_count("dense-FP8 SwiGLU prefill", prefill.len(), 3)?;

    for entry in quantize.iter().chain(&decode).chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in &prefill {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "dense-FP8 SwiGLU entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }

    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 SwiGLU quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut decode_registers = Vec::new();
    for entry in &decode {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        decode_registers.push(resource.registers);
    }
    decode_registers.sort_unstable();
    require_registers(&baseline, "decode_registers", &decode_registers)?;

    let mut prefill_registers = Vec::new();
    for entry in &prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
    }
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prefill_registers", &prefill_registers)?;

    println!(
        "dense-FP8 SwiGLU gate passed: 1 quantize + 8 decode + 3 prefill entries, REG {} / {:?} / {:?}, STACK:0 LOCAL:0",
        quantize_resource.registers, decode_registers, prefill_registers
    );
    Ok(())
}

fn gate_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_DOWN_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_quantize_TID_"))
        .collect::<Vec<_>>();
    let down = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_TID_"))
        .collect::<Vec<_>>();
    require_count("dense-FP8 down quantization", quantize.len(), 1)?;
    require_count("dense-FP8 down projection", down.len(), 8)?;

    for entry in quantize.iter().chain(&down) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = &sm120_gate_artifact(root)?.resources;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 down quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut down_registers = Vec::new();
    let mut shared = Vec::new();
    for entry in down {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        down_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    down_registers.sort_unstable();
    require_registers(&baseline, "down_registers", &down_registers)?;

    println!(
        "dense-FP8 down gate passed: 1 quantize + 8 projection entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}",
        quantize_resource.registers, down_registers, quantize_resource.shared, shared,
    );
    Ok(())
}

fn gate_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(NVFP4_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let a16 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_swiglu_a16_t"))
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_swiglu_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("NVFP4 SwiGLU A16", a16.len(), 4)?;
    require_count("NVFP4 activation quantization", quantize.len(), 5)?;
    require_count("NVFP4 SwiGLU W4A4", w4a4.len(), 5)?;

    for entry in &a16 {
        let minimum_ctas = if entry.name == "nvfp4_swiglu_a16_t1" {
            2
        } else {
            1
        };
        if !entry.body.contains(".reqntid 256, 1, 1")
            || !entry
                .body
                .contains(&format!(".minnctapersm {minimum_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its retained 256-thread/{minimum_ctas}-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its retained 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;

    let mut a16_registers = Vec::new();
    let mut a16_shared = Vec::new();
    for entry in a16 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 A16 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        a16_registers.push(resource.registers);
        a16_shared.push(resource.shared);
    }
    a16_registers.sort_unstable();
    a16_shared.sort_unstable();
    require_registers(&baseline, "a16_registers", &a16_registers)?;
    require_registers(&baseline, "a16_shared_bytes", &a16_shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted NVFP4 quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_uniform_value(&baseline, "quantize_shared_bytes", &quantize_shared)?;

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 W4A4 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 W4A4 SASS `{}`", entry.name))?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    require_registers(&baseline, "w4a4_registers", &w4a4_registers)?;
    require_uniform_value(&baseline, "w4a4_shared_bytes", &w4a4_shared)?;

    println!(
        "NVFP4 SwiGLU gate passed: 4 A16 + 5 quantize + 5 W4A4 entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        a16_registers, quantize_registers, w4a4_registers, a16_shared, quantize_shared, w4a4_shared,
    );
    Ok(())
}

fn gate_qwen35_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let a16 = entries
        .iter()
        .filter(|entry| {
            entry.name == "qwen35_nvfp4_swiglu_a16_t1"
                || entry.name == "qwen35_nvfp4_swiglu_a16_t2"
                || entry.name.starts_with("qwen35_nvfp4_swiglu_a16_TID_")
        })
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_swiglu_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 SwiGLU A16", a16.len(), 4)?;
    require_count("Qwen3.5 NVFP4 activation quantization", quantize.len(), 8)?;
    require_count("Qwen3.5 NVFP4 SwiGLU W4A4", w4a4.len(), 8)?;

    for entry in &a16 {
        let minimum_ctas = if entry.name == "qwen35_nvfp4_swiglu_a16_t1" {
            2
        } else {
            1
        };
        if !entry.body.contains(".reqntid 256, 1, 1")
            || !entry
                .body
                .contains(&format!(".minnctapersm {minimum_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its retained 256-thread/{minimum_ctas}-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its retained 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;

    let mut a16_registers = Vec::new();
    let mut a16_shared = Vec::new();
    for entry in a16 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 A16 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        a16_registers.push(resource.registers);
        a16_shared.push(resource.shared);
    }
    a16_registers.sort_unstable();
    a16_shared.sort_unstable();
    require_registers(&baseline, "a16_registers", &a16_registers)?;
    require_registers(&baseline, "a16_shared_bytes", &a16_shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    quantize_shared.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_uniform_value(&baseline, "quantize_shared_bytes", &quantize_shared)?;

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 W4A4 entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 W4A4 SASS `{}`", entry.name))?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    w4a4_shared.sort_unstable();
    require_registers(&baseline, "w4a4_registers", &w4a4_registers)?;
    require_uniform_value(&baseline, "w4a4_shared_bytes", &w4a4_shared)?;

    println!(
        "Qwen3.5 NVFP4 SwiGLU gate passed: 4 A16 + 8 quantize + 8 W4A4 entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        a16_registers, quantize_registers, w4a4_registers, a16_shared, quantize_shared, w4a4_shared,
    );
    Ok(())
}

fn gate_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_PREPARE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let control = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_control_exact_TID_"))
        .collect::<Vec<_>>();
    let convolution = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_convolution_exact_TID_"))
        .collect::<Vec<_>>();
    require_count("GDN control", control.len(), 8)?;
    require_count("GDN convolution", convolution.len(), 8)?;

    for entry in &control {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &convolution {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = &sm120_gate_artifact(root)?.resources;
    let mut control_registers = Vec::new();
    let mut convolution_registers = Vec::new();
    let mut control_shared = Vec::new();
    let mut convolution_shared = Vec::new();

    for entry in control {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        control_registers.push(resource.registers);
        control_shared.push(resource.shared);
    }
    for entry in convolution {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        convolution_registers.push(resource.registers);
        convolution_shared.push(resource.shared);
    }
    control_registers.sort_unstable();
    convolution_registers.sort_unstable();
    require_registers(&baseline, "control_registers", &control_registers)?;
    require_registers(&baseline, "convolution_registers", &convolution_registers)?;

    println!(
        "GDN prepare gate passed: 8 control + 8 convolution entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}",
        control_registers, convolution_registers, control_shared, convolution_shared
    );
    Ok(())
}

fn gate_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_RECURRENCE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let recurrence = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_recurrence_exact_TID_"))
        .collect::<Vec<_>>();
    require_count("GDN recurrence", recurrence.len(), 8)?;
    for entry in &recurrence {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    let resources = &sm120_gate_artifact(root)?.resources;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in recurrence {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "recurrence_registers", &registers)?;
    println!(
        "GDN recurrence gate passed: 8 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_output_quantize"))
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_output_projection_TID_"))
        .collect::<Vec<_>>();
    require_count("GDN output quantization", quantize.len(), 1)?;
    require_count("GDN output projection", projection.len(), 8)?;
    for entry in quantize.iter().chain(&projection) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    let resources = &sm120_gate_artifact(root)?.resources;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted GDN output quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    let mut projection_registers = Vec::new();
    let mut projection_shared = Vec::new();
    for entry in projection {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        projection_registers.push(resource.registers);
        projection_shared.push(resource.shared);
    }
    projection_registers.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    println!(
        "GDN output gate passed: 1 quantize + 8 projection entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}",
        quantize_resource.registers,
        projection_registers,
        quantize_resource.shared,
        projection_shared,
    );
    Ok(())
}

fn gate_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "attention_qk_prepare_exact_TID_",
        Some("attention_qk_prepare_prefill_exact_TID_"),
        "attention Q/K prepare",
        "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
        "E4M3",
    )
}

fn gate_qwen35_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "qwen35_attention_qk_prepare_exact_TID_",
        None,
        "Qwen3.5 attention Q/K prepare",
        "F2FP.BF16.F32.PACK_AB",
        "BF16",
    )
}

fn gate_attention_qk_prepare_target(
    root: &Path,
    baseline_path: &str,
    entry_prefix: &str,
    prefill_prefix: Option<&str>,
    label: &str,
    cache_instruction: &str,
    cache_label: &str,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let prepare = entries
        .iter()
        .filter(|entry| entry.name.starts_with(entry_prefix))
        .collect::<Vec<_>>();
    let prefill = prefill_prefix.map_or_else(Vec::new, |prefix| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    });
    require_count(label, prepare.len(), 8)?;
    if prefill_prefix.is_some() {
        require_count("attention Q/K prefill preparation", prefill.len(), 4)?;
    }
    for entry in prepare.iter().chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut decode_registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, registers) in [
        (&prepare, &mut decode_registers),
        (&prefill, &mut prefill_registers),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["MUFU.RSQ", "SHFL.BFLY", cache_instruction] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    decode_registers.sort_unstable();
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prepare_registers", &decode_registers)?;
    if baseline.contains_key("prefill_prepare_registers") {
        require_registers(&baseline, "prefill_prepare_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "{label} gate passed: {} decode + {} prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, RSQ/SHFL/{cache_label} present",
        prepare.len(),
        prefill.len(),
        decode_registers,
        prefill_registers,
        shared
    );
    Ok(())
}

fn gate_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(PAGED_GQA_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let attention = entries
        .iter()
        .filter(|entry| entry.name.starts_with("paged_gqa_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_shared_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_partials = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_partitioned_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_reductions = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_partitioned_reduce_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("paged GQA", attention.len(), 8)?;
    require_count("shared prefill paged GQA", prefill.len(), 3)?;
    require_count("partitioned prefill paged GQA", prefill_partials.len(), 2)?;
    require_count(
        "partitioned prefill paged GQA reduction",
        prefill_reductions.len(),
        2,
    )?;
    for entry in &attention {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!("entry `{}` lost dynamic shared memory", entry.name).into());
        }
    }
    for entry in &prefill_partials {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!("entry `{}` lost dynamic shared memory", entry.name).into());
        }
    }
    for entry in &prefill_reductions {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_partial_registers = Vec::new();
    let mut prefill_reduce_registers = Vec::new();
    let mut shared = Vec::new();
    for entry in attention {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill_partials {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_partial_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill_reductions {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_reduce_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("MUFU.EX2") {
            return Err(format!("entry `{}` lost required `MUFU.EX2` SASS", entry.name).into());
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    prefill_partial_registers.sort_unstable();
    prefill_reduce_registers.sort_unstable();
    require_registers(&baseline, "attention_registers", &registers)?;
    if baseline.contains_key("prefill_shared_registers") {
        require_registers(&baseline, "prefill_shared_registers", &prefill_registers)?;
    }
    if baseline.contains_key("prefill_partition_registers") {
        require_registers(
            &baseline,
            "prefill_partition_registers",
            &prefill_partial_registers,
        )?;
    }
    if baseline.contains_key("prefill_reduce_registers") {
        require_registers(
            &baseline,
            "prefill_reduce_registers",
            &prefill_reduce_registers,
        )?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "paged GQA gate passed: 8 decode + 3 shared + 2 partition + 2 reduction entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/SHFL/EX2/LDGSTS present",
        registers, prefill_registers, prefill_partial_registers, prefill_reduce_registers, shared
    );
    Ok(())
}

fn gate_long_context_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let partials = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("long_context_paged_gqa_partial_exact_TID_")
        })
        .collect::<Vec<_>>();
    let reductions = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("long_context_paged_gqa_reduce_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("long-context paged GQA partial", partials.len(), 8)?;
    require_count("long-context paged GQA reduction", reductions.len(), 8)?;
    for entry in partials.iter().chain(&reductions) {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &reductions {
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!(
                "entry `{}` lost its dynamic partition-weight workspace",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut partial_registers = Vec::new();
    let mut reduction_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, registers, instructions) in [
        (
            &partials,
            &mut partial_registers,
            &["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2"][..],
        ),
        (
            &reductions,
            &mut reduction_registers,
            &["SHFL.BFLY", "MUFU.EX2", "STS", "LDS", "BAR.SYNC"][..],
        ),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    partial_registers.sort_unstable();
    reduction_registers.sort_unstable();
    require_registers(&baseline, "partial_registers", &partial_registers)?;
    require_registers(&baseline, "reduction_registers", &reduction_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "long-context paged GQA gate passed: 8 partial + 8 reduction entries, REG partial {:?}, reduction {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/SHFL/EX2/dynamic-shared present",
        partial_registers, reduction_registers, shared
    );
    Ok(())
}

fn gate_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(ATTENTION_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path)?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("attention_gate_quantize_exact_TID_"))
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("attention_output_projection_TID_"))
        .collect::<Vec<_>>();
    require_count("attention-output gate quantization", quantize.len(), 1)?;
    require_count("attention-output projection", projection.len(), 8)?;
    for entry in quantize.iter().chain(&projection) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted attention-output gate quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    let quantize_sass = sass_function_body(sass, quantize[0].name)
        .ok_or("cuobjdump omitted attention-output gate quantization SASS")?;
    for instruction in ["MUFU.EX2", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C"] {
        if !quantize_sass.contains(instruction) {
            return Err(format!(
                "entry `{}` lost required `{instruction}` SASS",
                quantize[0].name
            )
            .into());
        }
    }
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    require_uniform_value(
        &baseline,
        "quantize_shared_bytes",
        &[quantize_resource.shared],
    )?;

    let mut projection_registers = Vec::new();
    let mut projection_shared = Vec::new();
    for entry in projection {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.DOWN"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        projection_registers.push(resource.registers);
        projection_shared.push(resource.shared);
    }
    projection_registers.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_uniform_value(&baseline, "projection_shared_bytes", &projection_shared)?;

    println!(
        "attention output gate passed: 1 quantize + 8 projection entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}, EX2/E4M3/SHFL present",
        quantize_resource.registers,
        projection_registers,
        quantize_resource.shared,
        projection_shared,
    );
    Ok(())
}

fn gate_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(NVFP4_DOWN_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let routes = entries
        .iter()
        .filter(|entry| {
            entry.name == "nvfp4_down_a16_b1" || entry.name.starts_with("nvfp4_down_a16_TID_")
        })
        .collect::<Vec<_>>();
    require_count("NVFP4 down", routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 down entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!("cuobjdump omitted NVFP4 down SASS `{}`", entry.name).into());
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "NVFP4 down gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_qwen35_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_DOWN_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_down_a16_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 down", routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(
                format!("cuobjdump omitted Qwen3.5 NVFP4 down SASS `{}`", entry.name).into(),
            );
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5 NVFP4 down gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_qwen35_nvfp4_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_QKV_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let ptx_path = root.join(PTX);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned release device build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_qkv_a16_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 QKV", routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(
                format!("cuobjdump omitted Qwen3.5 NVFP4 QKV SASS `{}`", entry.name).into(),
            );
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5 NVFP4 QKV gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn verify_generator_stamp(
    root: &Path,
    baseline: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let backend = backend_path(root)?;
    let source = cuda_oxide_source(root, &backend)?;
    let commit = command_text("git", &["-C", path_text(&source)?, "rev-parse", "HEAD"])?;
    let changes = command_text(
        "git",
        &[
            "-C",
            path_text(&source)?,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )?;
    if !changes.trim().is_empty() {
        return Err("cuda-oxide source has tracked changes; restore the pinned checkout".into());
    }
    require_stamp(baseline, "cuda_oxide_commit", commit.trim())?;
    let rustc = require_success(Path::new("rustc"), &[OsStr::new("-vV")])?;
    let (rustc_release, rustc_commit) = parse_rustc_identity(&String::from_utf8(rustc.stdout)?)?;
    require_stamp(baseline, "rustc_release", &rustc_release)?;
    require_stamp(baseline, "rustc_commit", &rustc_commit)?;

    let ptxas = cuda_tool("ptxas");
    let cuobjdump = cuda_tool("cuobjdump");
    let ptxas_identity = cuda_toolkit_identity(&ptxas)?;
    let cuobjdump_identity = cuda_toolkit_identity(&cuobjdump)?;
    if ptxas_identity != cuobjdump_identity {
        return Err(format!(
            "CUDA tools come from different Toolkit versions: {} reports release {} / V{}, while {} reports release {} / V{}",
            ptxas.display(),
            ptxas_identity.release,
            ptxas_identity.version,
            cuobjdump.display(),
            cuobjdump_identity.release,
            cuobjdump_identity.version,
        )
        .into());
    }
    require_stamp(baseline, "cuda_toolkit_release", &ptxas_identity.release)?;
    require_stamp(baseline, "cuda_toolkit_version", &ptxas_identity.version)?;

    let lock = fs::read_to_string(root.join("Cargo.lock"))?;
    let expected_commit = baseline
        .get("cuda_oxide_commit")
        .ok_or("baseline is missing `cuda_oxide_commit`")?;
    if !lock.contains(&format!("rev={expected_commit}")) {
        return Err("Cargo.lock does not contain the stamped cuda-oxide revision".into());
    }

    Ok(())
}

fn backend_path(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_BACKEND") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = local_backend(root) {
        return Ok(path);
    }
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .ok_or("set CUDA_OXIDE_BACKEND or CARGO_HOME")?;

    Ok(cargo_home
        .join("cuda-oxide")
        .join("librustc_codegen_cuda.so"))
}

fn cuda_oxide_source(root: &Path, backend: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_SOURCE") {
        return Ok(PathBuf::from(path));
    }
    for ancestor in backend.ancestors() {
        if ancestor.join(".git").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }
    if let Some(parent) = backend.parent() {
        let cached = parent.join("src");
        if cached.join(".git").exists() {
            return Ok(cached);
        }
    }
    let local = local_cuda_oxide_source(root);
    if local.join(".git").exists() {
        return Ok(local);
    }

    Err("could not locate cuda-oxide source; set CUDA_OXIDE_SOURCE".into())
}

fn local_cuda_oxide_source(root: &Path) -> PathBuf {
    root.join("target/cuda-oxide-source")
}

fn task_cargo_home(root: &Path) -> PathBuf {
    root.join("target/cargo-home")
}

fn encoded_backend_rustflags(root: &Path, source: &Path) -> Result<String, Box<dyn Error>> {
    let sysroot = command_text("rustc", &["--print", "sysroot"])?;
    let cargo_home = task_cargo_home(root);
    let prefixes = [
        (source, "/cuda-oxide"),
        (cargo_home.as_path(), "/cargo-home"),
        (Path::new(sysroot.trim()), "/rust-toolchain"),
        (root, "/tuiskollm"),
    ];
    let flags = prefixes
        .into_iter()
        .map(|(path, replacement)| {
            Ok(format!(
                "--remap-path-prefix={}={replacement}",
                path_text(path)?
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(flags.join("\u{1f}"))
}

fn local_backend(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let rustc = command_text("rustc", &["-vV"])?;
    let host = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or("rustc -vV omitted its host triple")?;
    let path = local_cuda_oxide_source(root)
        .join("crates/rustc-codegen-cuda/target")
        .join(host)
        .join("debug/librustc_codegen_cuda.so");
    if !path.is_file() {
        return Err(format!("cuda-oxide backend does not exist at {}", path.display()).into());
    }

    Ok(path)
}

fn cuda_tool(name: &str) -> PathBuf {
    let home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let candidate = home.join("bin").join(name);
    if candidate.is_file() {
        candidate
    } else {
        PathBuf::from(name)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CudaToolkitIdentity {
    release: String,
    version: String,
}

fn cuda_toolkit_identity(tool: &Path) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let output = require_success(tool, &[OsStr::new("--version")])?;

    parse_cuda_toolkit_identity(&String::from_utf8(output.stdout)?)
}

fn parse_cuda_toolkit_identity(text: &str) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let identity = text
        .lines()
        .find_map(|line| line.strip_prefix("Cuda compilation tools, release "))
        .ok_or("CUDA tool omitted its release identity")?;
    let (release, version) = identity
        .split_once(", V")
        .ok_or("CUDA tool emitted an invalid release identity")?;

    Ok(CudaToolkitIdentity {
        release: release.to_string(),
        version: version.to_string(),
    })
}

fn parse_rustc_identity(text: &str) -> Result<(String, String), Box<dyn Error>> {
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name}: ")))
            .map(str::to_string)
            .ok_or_else(|| format!("rustc -vV omitted `{name}`"))
    };

    Ok((field("release")?, field("commit-hash")?))
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let arguments = arguments.iter().map(OsStr::new).collect::<Vec<_>>();
    let output = require_success(Path::new(program), &arguments)?;

    Ok(String::from_utf8(output.stdout)?)
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

fn require_success(program: &Path, arguments: &[&OsStr]) -> Result<Output, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            program.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(output)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_baseline(text: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut fields = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid baseline line `{line}`"))?;
        fields.insert(key.to_string(), value.to_string());
    }

    Ok(fields)
}

fn require_stamp(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &str,
) -> Result<(), Box<dyn Error>> {
    let expected = baseline
        .get(key)
        .ok_or_else(|| format!("baseline is missing `{key}`"))?;
    if expected != actual {
        return Err(format!(
            "generator stamp `{key}` is `{actual}`, expected `{expected}`; re-baseline separately"
        )
        .into());
    }

    Ok(())
}

struct Entry<'a> {
    name: &'a str,
    body: &'a str,
}

fn parse_entries(ptx: &str) -> Vec<Entry<'_>> {
    let marker = ".visible .entry ";
    let offsets = ptx.match_indices(marker).collect::<Vec<_>>();
    offsets
        .iter()
        .enumerate()
        .filter_map(|(index, (offset, _))| {
            let begin = offset + marker.len();
            let end = offsets
                .get(index + 1)
                .map(|(offset, _)| *offset)
                .unwrap_or(ptx.len());
            let body = &ptx[begin..end];
            let name_end = body.find('(')?;

            Some(Entry {
                name: body[..name_end].trim(),
                body,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Resource {
    registers: u32,
    stack: u32,
    shared: u32,
    local: u32,
}

struct Sm120GateArtifact {
    root: PathBuf,
    cubin: PathBuf,
    resources: BTreeMap<String, Resource>,
    sass: OnceLock<Result<String, String>>,
}

static SM120_GATE_ARTIFACT: OnceLock<Result<Sm120GateArtifact, String>> = OnceLock::new();

fn sm120_gate_artifact(root: &Path) -> Result<&'static Sm120GateArtifact, Box<dyn Error>> {
    let artifact = SM120_GATE_ARTIFACT
        .get_or_init(|| build_sm120_gate_artifact(root).map_err(|error| error.to_string()));
    let artifact = match artifact {
        Ok(artifact) => artifact,
        Err(error) => return Err(error.clone().into()),
    };
    if artifact.root != root {
        return Err(format!(
            "one xtask process cannot resource-check SM120 artifacts from both `{}` and `{}`",
            artifact.root.display(),
            root.display()
        )
        .into());
    }

    Ok(artifact)
}

fn build_sm120_gate_artifact(root: &Path) -> Result<Sm120GateArtifact, Box<dyn Error>> {
    let ptx = root.join(PTX);
    if !ptx.is_file() {
        return Err(format!(
            "could not read {}; run the pinned release device build first",
            ptx.display()
        )
        .into());
    }
    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("sm120-resource-gates.cubin");
    require_success(
        &cuda_tool("ptxas"),
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new("sm_120a"),
            ptx.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let output = require_success(
        &cuda_tool("cuobjdump"),
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;

    Ok(Sm120GateArtifact {
        root: root.to_path_buf(),
        cubin,
        resources: parse_resources(&String::from_utf8(output.stdout)?)?,
        sass: OnceLock::new(),
    })
}

impl Sm120GateArtifact {
    fn sass(&self) -> Result<&str, Box<dyn Error>> {
        let sass = self.sass.get_or_init(|| {
            require_success(
                &cuda_tool("cuobjdump"),
                &[OsStr::new("--dump-sass"), self.cubin.as_os_str()],
            )
            .and_then(|output| String::from_utf8(output.stdout).map_err(Into::into))
            .map_err(|error| error.to_string())
        });
        match sass {
            Ok(sass) => Ok(sass),
            Err(error) => Err(error.clone().into()),
        }
    }
}

fn parse_resources(text: &str) -> Result<BTreeMap<String, Resource>, Box<dyn Error>> {
    let mut resources = BTreeMap::new();
    let mut function = None;
    for line in text.lines() {
        if let Some(name) = line
            .trim()
            .strip_prefix("Function ")
            .and_then(|name| name.strip_suffix(':'))
        {
            function = Some(name.to_string());
            continue;
        }
        let Some(name) = function.take() else {
            continue;
        };
        let fields = line
            .split_whitespace()
            .filter_map(|field| field.split_once(':'))
            .collect::<BTreeMap<_, _>>();
        let field = |key: &str| -> Result<u32, Box<dyn Error>> {
            Ok(fields
                .get(key)
                .ok_or_else(|| format!("resource line for `{name}` is missing `{key}`"))?
                .parse()?)
        };
        let resource = Resource {
            registers: field("REG")?,
            stack: field("STACK")?,
            shared: field("SHARED")?,
            local: field("LOCAL")?,
        };
        resources.insert(name, resource);
    }

    Ok(resources)
}

fn require_count(family: &str, actual: usize, expected: usize) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(format!(
            "{family} emitted {actual} entries, expected {expected}; zero entries is a silent generic-instantiation failure"
        )
        .into());
    }

    Ok(())
}

fn require_spill_free(name: &str, resource: &Resource) -> Result<(), Box<dyn Error>> {
    if resource.stack != 0 || resource.local != 0 {
        return Err(format!(
            "entry `{name}` uses STACK:{} LOCAL:{}",
            resource.stack, resource.local
        )
        .into());
    }

    Ok(())
}

fn require_registers(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &[u32],
) -> Result<(), Box<dyn Error>> {
    let actual = actual
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    require_stamp(baseline, key, &actual)
}

fn require_uniform_value(
    baseline: &BTreeMap<String, String>,
    key: &str,
    actual: &[u32],
) -> Result<(), Box<dyn Error>> {
    let Some((&first, remaining)) = actual.split_first() else {
        return Err(format!("resource inventory `{key}` is empty").into());
    };
    if remaining.iter().any(|value| *value != first) {
        return Err(format!("resource inventory `{key}` is not uniform: {actual:?}").into());
    }

    require_stamp(baseline, key, &first.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        COMPOSED_PERFORMANCE_SUITES, OptimizationSuite, PERFORMANCE_SUITES, PerformanceSuite,
        SM120_RESOURCE_BASELINES, parse_compute_pids, parse_cuda_toolkit_identity, parse_entries,
        parse_idle_sample, parse_performance_device_sample, parse_performance_iteration,
        parse_resources, parse_rustc_identity, require_count, require_uniform_value,
        resolve_target_output, sass_function_body,
    };
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    #[test]
    fn parses_hashed_and_concrete_entries() {
        let ptx = ".visible .entry rms_norm_b1()\n.reqntid 512, 1, 1\n\
                   .visible .entry residual_rms_norm_TID_abc()\n.reqntid 512, 1, 1\n";
        let entries = parse_entries(ptx);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "rms_norm_b1");
        assert_eq!(entries[1].name, "residual_rms_norm_TID_abc");
    }

    #[test]
    fn parses_cuobjdump_resource_lines() {
        let resources = parse_resources(
            " Function rms_norm_b1:\n  REG:20 STACK:0 SHARED:1088 LOCAL:0 CONSTANT[0]:920\n",
        )
        .unwrap();
        let resource = resources["rms_norm_b1"];

        assert_eq!(resource.registers, 20);
        assert_eq!(resource.stack, 0);
        assert_eq!(resource.shared, 1_088);
        assert_eq!(resource.local, 0);
    }

    #[test]
    fn zero_generic_entries_fail_loudly() {
        let error = require_count("plain RMSNorm", 0, 8).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("silent generic-instantiation failure")
        );
    }

    #[test]
    fn parses_readable_compiler_identities() {
        let rustc = parse_rustc_identity(
            "rustc 1.96.0-nightly (55e86c996 2026-04-02)\n\
             commit-hash: 55e86c996809902e8bbad512cfb4d2c18be446d9\n\
             release: 1.96.0-nightly\n",
        )
        .unwrap();
        let cuda = parse_cuda_toolkit_identity(
            "Cuda compilation tools, release 13.3, V13.3.73\n\
             Build cuda_13.3.r13.3/compiler.38244171_0\n",
        )
        .unwrap();

        assert_eq!(rustc.0, "1.96.0-nightly");
        assert_eq!(rustc.1, "55e86c996809902e8bbad512cfb4d2c18be446d9");
        assert_eq!(cuda.release, "13.3");
        assert_eq!(cuda.version, "13.3.73");
    }

    #[test]
    fn shared_memory_contract_requires_one_value() {
        let baseline = BTreeMap::from([("shared_bytes".to_string(), "1088".to_string())]);

        require_uniform_value(&baseline, "shared_bytes", &[1_088; 16]).unwrap();
        assert!(require_uniform_value(&baseline, "shared_bytes", &[1_088, 1_024]).is_err());
    }

    #[test]
    fn performance_suite_names_select_the_complete_inventory() {
        let names = PERFORMANCE_SUITES
            .iter()
            .map(|suite| suite.name())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "residual-norm",
                "fp8-qkv",
                "fp8-gdn-input",
                "fp8-lm-head",
                "fp8-swiglu",
                "fp8-down",
                "nvfp4-swiglu",
                "nvfp4-down",
                "gdn-prepare",
                "gdn-recurrence",
                "gdn-output",
                "attention-qk-prepare",
                "paged-gqa",
                "long-context-paged-gqa",
                "attention-output",
            ]
        );
        for suite in PERFORMANCE_SUITES {
            assert_eq!(
                PerformanceSuite::parse(suite.name()).unwrap().name(),
                suite.name()
            );
        }
        assert!(PerformanceSuite::parse("unknown").is_err());
    }

    #[test]
    fn sm120_build_receipt_covers_every_resource_gate() {
        assert_eq!(
            SM120_RESOURCE_BASELINES,
            [
                "qual/baselines/residual-norm-sm120.txt",
                "qual/baselines/qwen35-residual-norm-sm120.txt",
                "qual/baselines/fp8-qkv-sm120.txt",
                "qual/baselines/fp8-gdn-input-sm120.txt",
                "qual/baselines/fp8-lm-head-sm120.txt",
                "qual/baselines/fp8-swiglu-sm120.txt",
                "qual/baselines/fp8-down-sm120.txt",
                "qual/baselines/nvfp4-swiglu-sm120.txt",
                "qual/baselines/qwen35-nvfp4-swiglu-sm120.txt",
                "qual/baselines/nvfp4-down-sm120.txt",
                "qual/baselines/qwen35-nvfp4-down-sm120.txt",
                "qual/baselines/qwen35-nvfp4-qkv-sm120.txt",
                "qual/baselines/gdn-prepare-sm120.txt",
                "qual/baselines/gdn-recurrence-sm120.txt",
                "qual/baselines/gdn-output-sm120.txt",
                "qual/baselines/attention-qk-prepare-sm120.txt",
                "qual/baselines/qwen35-attention-qk-prepare-sm120.txt",
                "qual/baselines/paged-gqa-sm120.txt",
                "qual/baselines/long-context-paged-gqa-sm120.txt",
                "qual/baselines/attention-output-sm120.txt",
            ]
        );
    }

    #[test]
    fn composed_performance_inventory_and_dependency_cones_are_exact() {
        let names = COMPOSED_PERFORMANCE_SUITES
            .iter()
            .map(|suite| suite.name())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "nvfp4-mlp",
                "dense-fp8-mlp",
                "dense-fp8-gdn-layer",
                "full-attention-layer",
                "text-endpoint",
                "resident-model",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("nvfp4-down")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            [
                "nvfp4-down",
                "nvfp4-mlp",
                "resident-model",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("fp8-down")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            [
                "fp8-down",
                "dense-fp8-mlp",
                "dense-fp8-gdn-layer",
                "full-attention-layer",
                "resident-model",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("long-context-paged-gqa")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            ["long-context-paged-gqa", "resident-long-context-model"]
        );
    }

    #[test]
    fn parses_idle_device_samples() {
        assert_eq!(parse_idle_sample("0, 234\n").unwrap(), (0, 234));
        assert_eq!(parse_idle_sample("69, 1024").unwrap(), (69, 1_024));
        assert!(parse_idle_sample("0, 234, 1").is_err());
    }

    #[test]
    fn parses_exact_performance_preflight_samples_and_processes() {
        assert_eq!(
            parse_performance_device_sample("NVIDIA GeForce RTX 5090, 0, 503\n").unwrap(),
            ("NVIDIA GeForce RTX 5090".to_string(), 0, 503)
        );
        assert!(parse_performance_device_sample("RTX 5090, 0").is_err());
        assert_eq!(parse_compute_pids("\n").unwrap(), Vec::<u32>::new());
        assert_eq!(parse_compute_pids("123\n456\n").unwrap(), vec![123, 456]);
        assert!(parse_compute_pids("N/A\n").is_err());
    }

    #[test]
    fn perf_iteration_requires_one_exact_batch_and_hypothesis() {
        let options = parse_performance_iteration(&[
            OsString::from("nvfp4-down"),
            OsString::from("--batch"),
            OsString::from("1"),
            OsString::from("--hypothesis"),
            OsString::from("coalesce B=1 loads"),
        ])
        .unwrap();
        assert_eq!(options.suite, PerformanceSuite::Nvfp4Down);
        assert_eq!(options.batch_size, 1);
        assert_eq!(options.hypothesis, "coalesce B=1 loads");

        assert!(
            parse_performance_iteration(&[
                OsString::from("nvfp4-down"),
                OsString::from("--batch"),
                OsString::from("9"),
                OsString::from("--hypothesis"),
                OsString::from("invalid batch"),
            ])
            .is_err()
        );
    }

    #[test]
    fn diagnostic_output_is_confined_to_the_ignored_target_tree() {
        let root = std::path::Path::new("/repository");
        assert_eq!(
            resolve_target_output(root, std::ffi::OsStr::new("target/report.json")).unwrap(),
            root.join("target/report.json")
        );
        assert!(resolve_target_output(root, std::ffi::OsStr::new("AGENTS.md")).is_err());
        assert!(resolve_target_output(root, std::ffi::OsStr::new("target/../AGENTS.md")).is_err());
        assert!(resolve_target_output(root, std::ffi::OsStr::new("/tmp/report.json")).is_err());
    }

    #[test]
    fn isolates_one_sass_function() {
        let sass = "Function : first\nQMMA.16832.F32.E4M3.E4M3 R0, R1, R2, R3;\n\
                    \t\tFunction : second\nNOP;\n";

        assert!(
            sass_function_body(sass, "first")
                .unwrap()
                .contains("QMMA.16832.F32.E4M3.E4M3")
        );
        assert!(!sass_function_body(sass, "second").unwrap().contains("QMMA"));
        assert!(sass_function_body(sass, "missing").is_none());
    }
}
