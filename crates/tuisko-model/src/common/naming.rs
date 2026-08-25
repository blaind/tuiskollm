//! Checkpoint-wide tensor key constants and layer key formatting shared by every admitted target.

/// Root of every decoder layer's tensor keys.
const LAYER_ROOT: &str = "model.language_model.layers";

pub(crate) const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
pub(crate) const FINAL_NORM: &str = "model.language_model.norm.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
pub(crate) const LM_HEAD_SCALE: &str = "lm_head.weight_scale";
pub(crate) const MTP_LAYER: usize = 0;

/// Tensor key prefix owned by one decoder layer.
pub(crate) fn layer_prefix(layer: usize) -> String {
    format!("{LAYER_ROOT}.{layer}")
}

/// Tensor key prefix owned by one submodule of one decoder layer.
pub(crate) fn layer_module_prefix(layer: usize, module: &str) -> String {
    format!("{LAYER_ROOT}.{layer}.{module}")
}

/// Tensor key of one decoder layer's pre-mixer RMSNorm weights.
pub(crate) fn input_layernorm(layer: usize) -> String {
    format!("{LAYER_ROOT}.{layer}.input_layernorm.weight")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_keys_match_the_pinned_checkpoint_spelling() {
        assert_eq!(layer_prefix(0), "model.language_model.layers.0");
        assert_eq!(layer_prefix(63), "model.language_model.layers.63");
        assert_eq!(
            layer_module_prefix(7, "mlp"),
            "model.language_model.layers.7.mlp"
        );
        assert_eq!(
            layer_module_prefix(7, "mlp.down_proj"),
            "model.language_model.layers.7.mlp.down_proj"
        );
        assert_eq!(
            layer_module_prefix(3, "self_attn"),
            "model.language_model.layers.3.self_attn"
        );
        assert_eq!(
            input_layernorm(4),
            "model.language_model.layers.4.input_layernorm.weight"
        );
    }
}
