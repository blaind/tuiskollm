//! Exact resident and shared-KV arena plan for the Qwen3.8 text model.

mod program;

pub use program::{
    ResidentDecodeRoute, ResidentLoadMode, ResidentLoadStats, ResidentModelProgram,
    ResidentMtpSegmentedVerifyRoute, ResidentMtpVerifyRoute, ResidentPrefillRoute,
};
#[cfg(feature = "qualification")]
pub use program::{
    ResidentEmbeddingStageGraph, ResidentLongContextObservables, ResidentModelObservables,
    ResidentMtpGdnObservables, ResidentMtpLayerObservables, ResidentMtpSegmentedStageGraph,
    ResidentMtpVerifyObservables, ResidentPrefillStageGraph,
};

use crate::common::math::{checked_sum, product};
use crate::{
    EngineError, EngineResult, KvCacheCodec, MAX_BATCH, SharedPagedKvLayout,
    qwen38::long_context_kv_layout::MAX_CONTEXT_TOKENS,
};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::{
    LONG_CONTEXT_GQA_MAX_PARTITIONS, PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES,
};
use tuisko_model::{Arch, NVFP4_MLP_LAYER_END, Qwen38_27B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
const TARGET_VERIFY_TOKENS: usize = 4;
const TARGET_VERIFY_ROWS: usize = MAX_BATCH * TARGET_VERIFY_TOKENS;
const GDN_LAYER_COUNT: usize = 48;
pub(crate) const MAX_ROWS: usize = 1_024;

/// Exact mixer and MLP source route owned by one decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentLayerKind {
    /// Early-layer GDN mixer followed by packed NVFP4 MLP weights.
    Nvfp4Gdn,
    /// Early-layer full attention followed by packed NVFP4 MLP weights.
    Nvfp4Attention,
    /// Late-layer GDN mixer followed by dense E4M3 MLP weights.
    DenseFp8Gdn,
    /// Late-layer full attention followed by dense E4M3 MLP weights.
    DenseFp8Attention,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GdnWeights {
    pub(super) input_norm: ArenaRegion<u16>,
    pub(super) input_weight_codes: ArenaRegion<u8>,
    pub(super) input_weight_scales: ArenaRegion<u16>,
    pub(super) control_weights: ArenaRegion<u16>,
    pub(super) a_log: ArenaRegion<u16>,
    pub(super) dt_bias: ArenaRegion<u16>,
    pub(super) convolution_weights: ArenaRegion<u16>,
    pub(super) recurrent_norm: ArenaRegion<u16>,
    pub(super) output_weight_codes: ArenaRegion<u8>,
    pub(super) output_weight_scales: ArenaRegion<u16>,
    pub(super) post_attention_norm: ArenaRegion<u16>,
}

impl GdnWeights {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let input_weights = product("resident GDN input weights", A::GDN_INPUT_ROWS, A::HIDDEN)?;
        let control_weights = product(
            "resident GDN control weights",
            product("resident GDN A/B control rows", 2, A::GDN_CONTROL_ROWS)?,
            A::HIDDEN,
        )?;
        let convolution_weights = product(
            "resident GDN convolution weights",
            A::GDN_QKV_ROWS,
            A::LINEAR_CONV_KERNEL_DIM,
        )?;
        let output_weights = product("resident GDN output weights", A::HIDDEN, A::GDN_VALUE_ROWS)?;

        Ok(Self {
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            input_weight_codes: builder.reserve(input_weights, ALIGNMENT)?,
            input_weight_scales: builder.reserve(A::GDN_INPUT_ROWS, ALIGNMENT)?,
            control_weights: builder.reserve(control_weights, ALIGNMENT)?,
            a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            convolution_weights: builder.reserve(convolution_weights, ALIGNMENT)?,
            recurrent_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weights, ALIGNMENT)?,
            output_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.input_norm,
            self.input_weight_codes,
            self.input_weight_scales,
            self.control_weights,
            self.a_log,
            self.dt_bias,
            self.convolution_weights,
            self.recurrent_norm,
            self.output_weight_codes,
            self.output_weight_scales,
            self.post_attention_norm,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident GDN weight bytes", |spans| self.push_spans(spans))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionWeights {
    pub(super) input_norm: ArenaRegion<u16>,
    pub(super) qkv_weight_codes: ArenaRegion<u8>,
    pub(super) qkv_weight_scales: ArenaRegion<u16>,
    pub(super) query_norm: ArenaRegion<u16>,
    pub(super) key_norm: ArenaRegion<u16>,
    pub(super) output_weight_codes: ArenaRegion<u8>,
    pub(super) output_weight_scales: ArenaRegion<u16>,
    pub(super) post_attention_norm: ArenaRegion<u16>,
}

impl AttentionWeights {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let qkv_weights = product(
            "resident attention QKV weights",
            A::ATTENTION_QKV_ROWS,
            A::HIDDEN,
        )?;
        let output_weights = product(
            "resident attention output weights",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;

        Ok(Self {
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            qkv_weight_codes: builder.reserve(qkv_weights, ALIGNMENT)?,
            qkv_weight_scales: builder.reserve(A::ATTENTION_QKV_ROWS, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weights, ALIGNMENT)?,
            output_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.input_norm,
            self.qkv_weight_codes,
            self.qkv_weight_scales,
            self.query_norm,
            self.key_norm,
            self.output_weight_codes,
            self.output_weight_scales,
            self.post_attention_norm,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident attention weight bytes", |spans| {
            self.push_spans(spans)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MixerWeights {
    Gdn(GdnWeights),
    Attention(AttentionWeights),
}

impl MixerWeights {
    fn push_spans(self, spans: &mut Vec<Span>) {
        match self {
            Self::Gdn(weights) => weights.push_spans(spans),
            Self::Attention(weights) => weights.push_spans(spans),
        }
    }

    fn byte_len(self) -> EngineResult<usize> {
        match self {
            Self::Gdn(weights) => weights.byte_len(),
            Self::Attention(weights) => weights.byte_len(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Nvfp4MlpWeights {
    pub(super) gate_weight_codes: ArenaRegion<u8>,
    pub(super) up_weight_codes: ArenaRegion<u8>,
    pub(super) gate_up_weight_scales: ArenaRegion<u8>,
    pub(super) down_weight_codes: ArenaRegion<u8>,
    pub(super) down_weight_scales: ArenaRegion<u8>,
}

impl Nvfp4MlpWeights {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let branch_codes = product(
            "resident NVFP4 gate/up branch codes",
            A::INTERMEDIATE,
            A::HIDDEN / 2,
        )?;
        let gate_up_scales = product(
            "resident NVFP4 gate/up scales",
            product("resident NVFP4 gate/up scale rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN / NVFP4_GROUP,
        )?;
        let down_codes = product("resident NVFP4 down codes", A::HIDDEN, A::INTERMEDIATE / 2)?;
        let down_scales = product(
            "resident NVFP4 down scales",
            A::HIDDEN,
            A::INTERMEDIATE / NVFP4_GROUP,
        )?;

        Ok(Self {
            gate_weight_codes: builder.reserve(branch_codes, ALIGNMENT)?,
            up_weight_codes: builder.reserve(branch_codes, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(gate_up_scales, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_codes, ALIGNMENT)?,
            down_weight_scales: builder.reserve(down_scales, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.gate_weight_codes,
            self.up_weight_codes,
            self.gate_up_weight_scales,
            self.down_weight_codes,
            self.down_weight_scales,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident NVFP4 MLP weight bytes", |spans| {
            self.push_spans(spans)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DenseFp8MlpWeights {
    pub(super) gate_up_weight_codes: ArenaRegion<u8>,
    pub(super) gate_up_weight_scales: ArenaRegion<u16>,
    pub(super) down_weight_codes: ArenaRegion<u8>,
    pub(super) down_weight_scales: ArenaRegion<u16>,
}

impl DenseFp8MlpWeights {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let gate_up_weights = product(
            "resident dense-FP8 gate/up weights",
            product("resident dense-FP8 gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weights = product(
            "resident dense-FP8 down weights",
            A::HIDDEN,
            A::INTERMEDIATE,
        )?;

        Ok(Self {
            gate_up_weight_codes: builder.reserve(gate_up_weights, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(2 * A::INTERMEDIATE, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weights, ALIGNMENT)?,
            down_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.gate_up_weight_codes,
            self.gate_up_weight_scales,
            self.down_weight_codes,
            self.down_weight_scales,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident dense-FP8 MLP weight bytes", |spans| {
            self.push_spans(spans)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MlpWeights {
    Nvfp4(Nvfp4MlpWeights),
    DenseFp8(DenseFp8MlpWeights),
}

impl MlpWeights {
    fn push_spans(self, spans: &mut Vec<Span>) {
        match self {
            Self::Nvfp4(weights) => weights.push_spans(spans),
            Self::DenseFp8(weights) => weights.push_spans(spans),
        }
    }

    fn byte_len(self) -> EngineResult<usize> {
        match self {
            Self::Nvfp4(weights) => weights.byte_len(),
            Self::DenseFp8(weights) => weights.byte_len(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GdnPersistent {
    pub(super) history: ArenaRegion<u16>,
    pub(super) state: ArenaRegion<f32>,
}

impl GdnPersistent {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let history = product(
            "resident GDN history",
            product("resident GDN history rows", MAX_BATCH, A::GDN_QKV_ROWS)?,
            A::LINEAR_CONV_KERNEL_DIM
                .checked_sub(1)
                .ok_or_else(|| EngineError::layout("GDN convolution width is zero"))?,
        )?;
        let state = product(
            "resident GDN state",
            product("resident GDN state heads", MAX_BATCH, A::GDN_CONTROL_ROWS)?,
            product(
                "resident GDN state head matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;

        Ok(Self {
            history: builder.reserve(history, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(spans, self.history, self.state);
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PersistentState {
    Gdn(GdnPersistent),
    Attention,
}

impl PersistentState {
    fn push_spans(self, spans: &mut Vec<Span>) {
        match self {
            Self::Gdn(state) => state.push_spans(spans),
            Self::Attention => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResidentLayerLayout {
    kind: ResidentLayerKind,
    pub(super) mixer: MixerWeights,
    pub(super) mlp: MlpWeights,
    pub(super) persistent: PersistentState,
}

impl ResidentLayerLayout {
    fn reserve(builder: &mut ArenaLayout, layer: usize) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let attention = (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL);
        let nvfp4 = layer < NVFP4_MLP_LAYER_END;
        let mixer = if attention {
            MixerWeights::Attention(AttentionWeights::reserve(builder)?)
        } else {
            MixerWeights::Gdn(GdnWeights::reserve(builder)?)
        };
        let mlp = if nvfp4 {
            MlpWeights::Nvfp4(Nvfp4MlpWeights::reserve(builder)?)
        } else {
            MlpWeights::DenseFp8(DenseFp8MlpWeights::reserve(builder)?)
        };
        let persistent = if attention {
            PersistentState::Attention
        } else {
            PersistentState::Gdn(GdnPersistent::reserve(builder)?)
        };
        let kind = match (nvfp4, attention) {
            (true, false) => ResidentLayerKind::Nvfp4Gdn,
            (true, true) => ResidentLayerKind::Nvfp4Attention,
            (false, false) => ResidentLayerKind::DenseFp8Gdn,
            (false, true) => ResidentLayerKind::DenseFp8Attention,
        };

        Ok(Self {
            kind,
            mixer,
            mlp,
            persistent,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        self.mixer.push_spans(spans);
        self.mlp.push_spans(spans);
        self.persistent.push_spans(spans);
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EndpointWeights {
    pub(super) final_norm: ArenaRegion<u16>,
    pub(super) lm_head_codes: ArenaRegion<u8>,
    pub(super) lm_head_scales: ArenaRegion<u16>,
}

impl EndpointWeights {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let lm_head = product("resident LM-head weights", A::VOCAB, A::HIDDEN)?;

        Ok(Self {
            final_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            lm_head_codes: builder.reserve(lm_head, ALIGNMENT)?,
            lm_head_scales: builder.reserve(A::VOCAB, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.final_norm,
            self.lm_head_codes,
            self.lm_head_scales,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident endpoint weight bytes", |spans| {
            self.push_spans(spans)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SharedWorkspace {
    pub(super) residual_a: ArenaRegion<u16>,
    pub(super) residual_b: ArenaRegion<u16>,
    pub(super) mixer_residual: ArenaRegion<u16>,
    pub(super) mixer_normalized: ArenaRegion<u16>,
    pub(super) mlp_normalized: ArenaRegion<u16>,
    pub(super) activation_codes: ArenaRegion<u8>,
    pub(super) activation_scales: ArenaRegion<f32>,
    pub(super) nvfp4_activation_codes: ArenaRegion<u8>,
    pub(super) nvfp4_activation_scales: ArenaRegion<u8>,
    pub(super) projected: ArenaRegion<u16>,
    pub(super) state_rows: ArenaRegion<u32>,
    pub(super) log_decay: ArenaRegion<f32>,
    pub(super) beta: ArenaRegion<f32>,
    pub(super) convolved: ArenaRegion<u16>,
    pub(super) recurrent_output: ArenaRegion<u16>,
    pub(super) rope_cos: ArenaRegion<f32>,
    pub(super) rope_sin: ArenaRegion<f32>,
    pub(super) table_rows: ArenaRegion<u32>,
    pub(super) cache_positions: ArenaRegion<u32>,
    pub(super) lengths: ArenaRegion<u32>,
    pub(super) query: ArenaRegion<f32>,
    pub(super) partial_maximum: ArenaRegion<f32>,
    pub(super) partial_denominator: ArenaRegion<f32>,
    pub(super) partial_numerator: ArenaRegion<f32>,
    pub(super) prefill_partials: ArenaRegion<f32>,
    pub(super) recurrent_plane: ArenaRegion<f32>,
    pub(super) attention: ArenaRegion<f32>,
    pub(super) mixer_branch: ArenaRegion<u16>,
    pub(super) swiglu: ArenaRegion<u16>,
    pub(super) mlp_branch: ArenaRegion<u16>,
    pub(super) logits: ArenaRegion<u16>,
    pub(super) provisional_history: ArenaRegion<u16>,
    pub(super) provisional_state: ArenaRegion<f32>,
    pub(super) provisional_state_row: ArenaRegion<u32>,
    pub(super) recorded_projected: ArenaRegion<u16>,
    pub(super) recorded_log_decay: ArenaRegion<f32>,
    pub(super) recorded_beta: ArenaRegion<f32>,
}

impl SharedWorkspace {
    fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38_27B;
        let row_hidden = product("resident row-hidden workspace", MAX_ROWS, A::HIDDEN)?;
        let row_intermediate = product(
            "resident row-intermediate workspace",
            MAX_ROWS,
            A::INTERMEDIATE,
        )?;
        let row_projected = product("resident projected workspace", MAX_ROWS, A::GDN_INPUT_ROWS)?;
        let row_control = product("resident control workspace", MAX_ROWS, A::GDN_CONTROL_ROWS)?;
        let row_qkv = product("resident convolved workspace", MAX_ROWS, A::GDN_QKV_ROWS)?;
        let row_value = product(
            "resident recurrent-output workspace",
            MAX_ROWS,
            A::GDN_VALUE_ROWS,
        )?;
        let row_attention = product(
            "resident attention workspace",
            MAX_ROWS,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let decode_attention_partials = product(
            "resident attention partial workspace",
            product(
                "resident attention partial rows",
                MAX_BATCH,
                A::NUM_ATTENTION_HEADS,
            )?,
            LONG_CONTEXT_GQA_MAX_PARTITIONS,
        )?;
        let attention_numerator = product(
            "resident attention partial numerator workspace",
            decode_attention_partials,
            A::HEAD_DIM,
        )?;
        let batch_logits = product("resident logits workspace", TARGET_VERIFY_ROWS, A::VOCAB)?;
        let provisional_history = product(
            "resident provisional GDN history",
            MAX_BATCH,
            product(
                "resident provisional GDN history row",
                A::GDN_QKV_ROWS,
                A::LINEAR_CONV_KERNEL_DIM - 1,
            )?,
        )?;
        let provisional_state = product(
            "resident provisional GDN state",
            product(
                "resident provisional GDN state rows",
                MAX_BATCH,
                A::GDN_CONTROL_ROWS,
            )?,
            product(
                "resident provisional GDN head matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;
        let recorded_projected = product(
            "resident target projected replay",
            GDN_LAYER_COUNT,
            product(
                "resident target projected replay rows",
                TARGET_VERIFY_ROWS,
                A::GDN_INPUT_ROWS,
            )?,
        )?;
        let recorded_control = product(
            "resident target control replay",
            GDN_LAYER_COUNT,
            product(
                "resident target control replay rows",
                TARGET_VERIFY_ROWS,
                A::GDN_CONTROL_ROWS,
            )?,
        )?;

        Ok(Self {
            residual_a: builder.reserve(row_hidden, ALIGNMENT)?,
            residual_b: builder.reserve(row_hidden, ALIGNMENT)?,
            mixer_residual: builder.reserve(row_hidden, ALIGNMENT)?,
            mixer_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            activation_codes: builder.reserve(row_intermediate, ALIGNMENT)?,
            activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            nvfp4_activation_codes: builder.reserve(row_intermediate / 2, ALIGNMENT)?,
            nvfp4_activation_scales: builder.reserve(row_intermediate / NVFP4_GROUP, ALIGNMENT)?,
            projected: builder.reserve(row_projected, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            log_decay: builder.reserve(row_control, ALIGNMENT)?,
            beta: builder.reserve(row_control, ALIGNMENT)?,
            convolved: builder.reserve(row_qkv, ALIGNMENT)?,
            recurrent_output: builder.reserve(row_value, ALIGNMENT)?,
            rope_cos: builder.reserve(MAX_ROWS * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MAX_ROWS * 32, ALIGNMENT)?,
            table_rows: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            cache_positions: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            lengths: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            query: builder.reserve(row_attention, ALIGNMENT)?,
            partial_maximum: builder.reserve(decode_attention_partials, ALIGNMENT)?,
            partial_denominator: builder.reserve(decode_attention_partials, ALIGNMENT)?,
            partial_numerator: builder.reserve(attention_numerator, ALIGNMENT)?,
            // The admitted T=1024/P=4 macro route retains 24 heads × 1024 rows ×
            // four 256-wide partition payloads, which is the largest exact prefill tile.
            prefill_partials: builder.reserve(
                PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES / size_of::<f32>(),
                ALIGNMENT,
            )?,
            // The prefill recurrence publishes its scaled recurrent rows here
            // for the parallel RMS/gate epilogue; the plane is scratch between
            // the paired kernels of one GDN layer.
            recurrent_plane: builder.reserve(row_value, ALIGNMENT)?,
            attention: builder.reserve(row_attention, ALIGNMENT)?,
            mixer_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            swiglu: builder.reserve(row_intermediate, ALIGNMENT)?,
            mlp_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            logits: builder.reserve(batch_logits, ALIGNMENT)?,
            provisional_history: builder.reserve(provisional_history, ALIGNMENT)?,
            provisional_state: builder.reserve(provisional_state, ALIGNMENT)?,
            provisional_state_row: builder.reserve(1, ALIGNMENT)?,
            recorded_projected: builder.reserve(recorded_projected, ALIGNMENT)?,
            recorded_log_decay: builder.reserve(recorded_control, ALIGNMENT)?,
            recorded_beta: builder.reserve(recorded_control, ALIGNMENT)?,
        })
    }

    fn push_spans(self, spans: &mut Vec<Span>) {
        push_regions!(
            spans,
            self.residual_a,
            self.residual_b,
            self.mixer_residual,
            self.mixer_normalized,
            self.mlp_normalized,
            self.activation_codes,
            self.activation_scales,
            self.nvfp4_activation_codes,
            self.nvfp4_activation_scales,
            self.projected,
            self.state_rows,
            self.log_decay,
            self.beta,
            self.convolved,
            self.recurrent_output,
            self.rope_cos,
            self.rope_sin,
            self.table_rows,
            self.cache_positions,
            self.lengths,
            self.query,
            self.partial_maximum,
            self.partial_denominator,
            self.partial_numerator,
            self.prefill_partials,
            self.recurrent_plane,
            self.attention,
            self.mixer_branch,
            self.swiglu,
            self.mlp_branch,
            self.logits,
            self.provisional_history,
            self.provisional_state,
            self.provisional_state_row,
            self.recorded_projected,
            self.recorded_log_decay,
            self.recorded_beta,
        );
    }

    fn byte_len(self) -> EngineResult<usize> {
        region_sum("resident shared workspace bytes", |spans| {
            self.push_spans(spans)
        })
    }
}

/// Checked exact-target layout for all resident text weights, state, caches, and shared scratch.
#[derive(Clone, Debug)]
pub struct ResidentModelLayout {
    builder: ArenaLayout,
    pub(super) kv_layout: SharedPagedKvLayout,
    pub(super) layers: Vec<ResidentLayerLayout>,
    pub(super) endpoint: EndpointWeights,
    pub(super) workspace: SharedWorkspace,
    resident_weight_bytes: usize,
    history_bytes: usize,
    state_bytes: usize,
    cache_bytes: usize,
    kv_table_bytes: usize,
    workspace_bytes: usize,
}

impl ResidentModelLayout {
    /// Plans the exact admitted 64-layer model and its shared 220K E4M3 KV pool.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen38_27B;
        require_exact_geometry()?;
        let mut builder = ArenaLayout::new();
        let mut layers = Vec::with_capacity(A::LAYERS);
        for layer in 0..A::LAYERS {
            layers.push(ResidentLayerLayout::reserve(&mut builder, layer)?);
        }
        let endpoint = EndpointWeights::reserve(&mut builder)?;
        let workspace = SharedWorkspace::reserve(&mut builder)?;
        let kv_layout = SharedPagedKvLayout::build(KvCacheCodec::E4m3)?;
        let resident_weight_bytes =
            layers
                .iter()
                .try_fold(endpoint.byte_len()?, |total, layer| {
                    checked_sum(
                        "resident model weight bytes",
                        total,
                        checked_sum(
                            "resident layer weight bytes",
                            layer.mixer.byte_len()?,
                            layer.mlp.byte_len()?,
                        )?,
                    )
                })?;
        let history_bytes = gdn_history_bytes()?;
        let state_bytes = gdn_state_bytes()?;
        let cache_bytes = kv_layout.cache_bytes();
        let kv_table_bytes = kv_layout.block_table_bytes();
        let workspace_bytes = workspace.byte_len()?;
        let layout = Self {
            builder,
            kv_layout,
            layers,
            endpoint,
            workspace,
            resident_weight_bytes,
            history_bytes,
            state_bytes,
            cache_bytes,
            kv_table_bytes,
            workspace_bytes,
        };
        layout.validate_regions()?;

        Ok(layout)
    }

    /// Number of exact decoder layers in the owner plan.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the exact source route for `layer`, or `None` outside `0..64`.
    pub fn layer_kind(&self, layer: usize) -> Option<ResidentLayerKind> {
        self.layers.get(layer).map(|layer| layer.kind)
    }

    /// Complete resident plus shared-KV device allocation bytes, including padding.
    pub const fn arena_bytes(&self) -> usize {
        self.resident_arena_bytes() + self.kv_arena_bytes()
    }

    /// Weight, GDN-state, and shared-workspace arena bytes including padding.
    pub const fn resident_arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Shared page tables and E4M3 K/V planes in their address-stable arena.
    pub const fn kv_arena_bytes(&self) -> usize {
        self.kv_layout.arena_bytes()
    }

    /// Source-backed norm, projection, MLP, final-norm, and LM-head bytes on device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Address-stable causal-convolution history bytes across all 48 GDN layers.
    pub const fn history_bytes(&self) -> usize {
        self.history_bytes
    }

    /// Address-stable recurrent matrix-state bytes across all 48 GDN layers.
    pub const fn state_bytes(&self) -> usize {
        self.state_bytes
    }

    /// Represented E4M3 key/value bytes across all 16 attention layers.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Stable device block-table bytes for all eight 220K-capable slot rows.
    pub const fn kv_table_bytes(&self) -> usize {
        self.kv_table_bytes
    }

    /// One address-stable workspace shared sequentially by all layers and the endpoint.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights, persistent state/cache, and shared workspace without padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes
            + self.history_bytes
            + self.state_bytes
            + self.cache_bytes
            + self.kv_table_bytes
            + self.workspace_bytes
    }

    /// Alignment bytes not owned by any typed region.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes() - self.owner_bytes()
    }

    /// mmap-backed BF16 embedding bytes intentionally excluded from device residency.
    pub fn source_mapped_embedding_bytes(&self) -> EngineResult<usize> {
        product(
            "source-mapped embedding bytes",
            product(
                "source-mapped embedding elements",
                Qwen38_27B::VOCAB,
                Qwen38_27B::HIDDEN,
            )?,
            size_of::<u16>(),
        )
    }

    /// Maximum logical context admitted for one slot in the shared page pool.
    pub const fn context_capacity(&self) -> usize {
        MAX_CONTEXT_TOKENS
    }

    fn validate_regions(&self) -> EngineResult<()> {
        let mut spans = Vec::new();
        for layer in &self.layers {
            layer.push_spans(&mut spans);
        }
        self.endpoint.push_spans(&mut spans);
        self.workspace.push_spans(&mut spans);
        spans.sort_unstable_by_key(|span| span.offset);

        for span in &spans {
            if !span.offset.is_multiple_of(ALIGNMENT) {
                return Err(EngineError::layout(format!(
                    "resident region offset {} is not {ALIGNMENT}-byte aligned",
                    span.offset
                )));
            }
            let end = checked_sum("resident region end", span.offset, span.bytes)?;
            if end > self.resident_arena_bytes() {
                return Err(EngineError::layout(format!(
                    "resident region {0}..{end} exceeds arena {1}",
                    span.offset,
                    self.resident_arena_bytes()
                )));
            }
        }
        for adjacent in spans.windows(2) {
            let first_end = checked_sum(
                "resident adjacent region end",
                adjacent[0].offset,
                adjacent[0].bytes,
            )?;
            if first_end > adjacent[1].offset {
                return Err(EngineError::layout("resident arena regions overlap"));
            }
        }
        let represented_bytes = spans.iter().try_fold(0usize, |total, span| {
            checked_sum("represented resident bytes", total, span.bytes)
        })?;
        let resident_owner_bytes = self
            .owner_bytes()
            .checked_sub(self.kv_layout.owner_bytes())
            .ok_or_else(|| EngineError::layout("resident KV ownership exceeds total ownership"))?;
        if represented_bytes != resident_owner_bytes {
            return Err(EngineError::layout(format!(
                "represented resident regions own {represented_bytes} bytes, accounting owns {}",
                resident_owner_bytes
            )));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct Span {
    offset: usize,
    bytes: usize,
}

macro_rules! push_regions {
    ($spans:expr, $($region:expr),+ $(,)?) => {{
        $(push_region($spans, $region);)+
    }};
}
use push_regions;

fn push_region<T: Copy>(spans: &mut Vec<Span>, region: ArenaRegion<T>) {
    spans.push(Span {
        offset: region.offset_bytes(),
        bytes: region.byte_len(),
    });
}

fn region_sum(name: &str, push: impl FnOnce(&mut Vec<Span>)) -> EngineResult<usize> {
    let mut spans = Vec::new();
    push(&mut spans);
    spans
        .iter()
        .try_fold(0usize, |total, span| checked_sum(name, total, span.bytes))
}

fn require_exact_geometry() -> EngineResult<()> {
    type A = Qwen38_27B;
    if A::LAYERS != 64
        || A::FULL_ATTENTION_INTERVAL != 4
        || NVFP4_MLP_LAYER_END != 56
        || !A::HIDDEN.is_multiple_of(NVFP4_GROUP)
        || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP)
    {
        return Err(EngineError::layout(
            "resident model geometry does not match the admitted Qwen3.8 source routes",
        ));
    }

    Ok(())
}

fn gdn_history_bytes() -> EngineResult<usize> {
    type A = Qwen38_27B;
    let gdn_layers = A::LAYERS - A::LAYERS / A::FULL_ATTENTION_INTERVAL;
    product(
        "all GDN history bytes",
        product(
            "all GDN history values",
            product(
                "per-layer GDN history values",
                product("GDN history rows", MAX_BATCH, A::GDN_QKV_ROWS)?,
                A::LINEAR_CONV_KERNEL_DIM - 1,
            )?,
            gdn_layers,
        )?,
        size_of::<u16>(),
    )
}

fn gdn_state_bytes() -> EngineResult<usize> {
    type A = Qwen38_27B;
    let gdn_layers = A::LAYERS - A::LAYERS / A::FULL_ATTENTION_INTERVAL;
    product(
        "all GDN state bytes",
        product(
            "all GDN state values",
            product(
                "per-layer GDN state values",
                product("GDN state heads", MAX_BATCH, A::GDN_CONTROL_ROWS)?,
                product(
                    "GDN state head matrix",
                    A::LINEAR_HEAD_DIM,
                    A::LINEAR_HEAD_DIM,
                )?,
            )?,
            gdn_layers,
        )?,
        size_of::<f32>(),
    )
}

#[cfg(test)]
mod tests {
    use super::{PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, ResidentLayerKind, ResidentModelLayout};

    #[test]
    fn exact_layer_route_inventory_is_complete() {
        let layout = ResidentModelLayout::build().unwrap();
        let mut counts = [0; 4];
        for layer in 0..layout.layer_count() {
            let kind = layout.layer_kind(layer).unwrap();
            counts[match kind {
                ResidentLayerKind::Nvfp4Gdn => 0,
                ResidentLayerKind::Nvfp4Attention => 1,
                ResidentLayerKind::DenseFp8Gdn => 2,
                ResidentLayerKind::DenseFp8Attention => 3,
            }] += 1;
            let attention = (layer + 1).is_multiple_of(4);
            assert_eq!(
                attention,
                matches!(
                    kind,
                    ResidentLayerKind::Nvfp4Attention | ResidentLayerKind::DenseFp8Attention
                ),
                "layer {layer}",
            );
        }

        assert_eq!(layout.layer_count(), 64);
        assert_eq!(counts, [42, 14, 6, 2]);
        assert_eq!(layout.layer_kind(64), None);
    }

    #[test]
    fn resident_byte_accounting_is_exact() {
        let layout = ResidentModelLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 19_103_682_560);
        assert_eq!(layout.history_bytes(), 23_592_960);
        assert_eq!(layout.state_bytes(), 1_207_959_552);
        assert_eq!(layout.cache_bytes(), 7_210_008_576);
        assert_eq!(layout.kv_table_bytes(), 110_016);
        assert_eq!(layout.workspace_bytes(), 948_860_932);
        assert_eq!(
            layout.source_mapped_embedding_bytes().unwrap(),
            2_542_796_800
        );
        assert_eq!(layout.context_capacity(), 220_000);
        assert_eq!(layout.owner_bytes(), 28_494_214_596);
    }

    #[test]
    fn one_shared_workspace_covers_every_exact_decode_and_prefill_route() {
        let layout = ResidentModelLayout::build().unwrap();

        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024] {
            assert!(rows * 5_120 <= layout.workspace.residual_a.len());
            assert!(rows * 17_408 <= layout.workspace.activation_codes.len());
            assert!(rows * 17_408 / 2 <= layout.workspace.nvfp4_activation_codes.len());
            assert!(rows * 17_408 / 16 <= layout.workspace.nvfp4_activation_scales.len());
            assert!(rows * 16_384 <= layout.workspace.projected.len());
            assert!(rows * 10_240 <= layout.workspace.convolved.len());
            assert!(rows * 6_144 <= layout.workspace.query.len());
            assert!(rows * 6_144 <= layout.workspace.attention.len());
            assert!(rows * 17_408 <= layout.workspace.swiglu.len());
        }
        assert!(8 * 24 * 860 <= layout.workspace.partial_maximum.len());
        assert!(8 * 24 * 860 * 256 <= layout.workspace.partial_numerator.len());
        assert_eq!(
            layout.workspace.prefill_partials.byte_len(),
            PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES,
        );
        assert_eq!(layout.workspace.recurrent_plane.len(), 1_024 * 48 * 128);
        assert_eq!(layout.workspace.logits.len(), 8 * 4 * 248_320);
        assert_eq!(layout.workspace.provisional_history.len(), 8 * 10_240 * 3);
        assert_eq!(layout.workspace.provisional_state.len(), 8 * 48 * 128 * 128);
        assert_eq!(layout.workspace.provisional_state_row.len(), 1);
        assert_eq!(
            layout.workspace.recorded_projected.len(),
            48 * 8 * 4 * 16_384,
        );
        assert_eq!(layout.workspace.recorded_log_decay.len(), 48 * 8 * 4 * 48);
        assert_eq!(layout.workspace.recorded_beta.len(), 48 * 8 * 4 * 48);
        assert_eq!(layout.resident_arena_bytes(), 21_284_111_616);
        assert_eq!(layout.kv_arena_bytes(), 7_210_118_656);
        assert_eq!(layout.padding_bytes(), 15_676);
        assert_eq!(layout.arena_bytes(), 28_494_230_272);
    }
}
