pub trait Arch: Copy + 'static {
    const MODEL_ID: &'static str;
    const REVISION: &'static str;
    const HIDDEN: usize;
    const INTERMEDIATE: usize;
    const VOCAB: usize;
    const LAYERS: usize;
    const FP8_LAYER: usize;
    const FULL_ATTENTION_INTERVAL: usize;
    const ATTENTION_HEADS: usize;
    const ATTENTION_KV_HEADS: usize;
    const ATTENTION_HEAD_DIM: usize;
    const GDN_KEY_HEADS: usize;
    const GDN_VALUE_HEADS: usize;
    const GDN_KEY_HEAD_DIM: usize;
    const GDN_VALUE_HEAD_DIM: usize;
    const GDN_CONV_KERNEL: usize;

    const ATTENTION_QUERY_ROWS: usize = 2 * Self::ATTENTION_HEADS * Self::ATTENTION_HEAD_DIM;
    const ATTENTION_KV_ROWS: usize = Self::ATTENTION_KV_HEADS * Self::ATTENTION_HEAD_DIM;
    const ATTENTION_QKV_ROWS: usize = Self::ATTENTION_QUERY_ROWS + 2 * Self::ATTENTION_KV_ROWS;
    const GDN_QK_ROWS: usize = Self::GDN_KEY_HEADS * Self::GDN_KEY_HEAD_DIM;
    const GDN_VALUE_ROWS: usize = Self::GDN_VALUE_HEADS * Self::GDN_VALUE_HEAD_DIM;
    const GDN_QKV_ROWS: usize = 2 * Self::GDN_QK_ROWS + Self::GDN_VALUE_ROWS;
    const GDN_INPUT_ROWS: usize = Self::GDN_QKV_ROWS + Self::GDN_VALUE_ROWS;
    const GDN_CONTROL_ROWS: usize = Self::GDN_VALUE_HEADS;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38_27bNvfp4;

impl Arch for Qwen38_27bNvfp4 {
    const MODEL_ID: &'static str = "unsloth/Qwen3.8-27B-NVFP4";
    const REVISION: &'static str = "16b6615af3548b88e2d8e382457bc705b00479cf";
    const HIDDEN: usize = 5_120;
    const INTERMEDIATE: usize = 17_408;
    const VOCAB: usize = 248_320;
    const LAYERS: usize = 64;
    const FP8_LAYER: usize = 56;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const ATTENTION_HEADS: usize = 24;
    const ATTENTION_KV_HEADS: usize = 4;
    const ATTENTION_HEAD_DIM: usize = 256;
    const GDN_KEY_HEADS: usize = 16;
    const GDN_VALUE_HEADS: usize = 48;
    const GDN_KEY_HEAD_DIM: usize = 128;
    const GDN_VALUE_HEAD_DIM: usize = 128;
    const GDN_CONV_KERNEL: usize = 4;
}

#[cfg(test)]
mod tests {
    use super::{Arch, Qwen38_27bNvfp4};

    #[test]
    fn qwen38_profile_matches_checkpoint_geometry() {
        type A = Qwen38_27bNvfp4;

        for (field, actual, expected) in [
            ("hidden", A::HIDDEN, 5_120),
            ("intermediate", A::INTERMEDIATE, 17_408),
            ("vocab", A::VOCAB, 248_320),
            ("layers", A::LAYERS, 64),
            ("fp8_layer", A::FP8_LAYER, 56),
            ("full_attention_interval", A::FULL_ATTENTION_INTERVAL, 4),
            ("attention_heads", A::ATTENTION_HEADS, 24),
            ("attention_kv_heads", A::ATTENTION_KV_HEADS, 4),
            ("attention_head_dim", A::ATTENTION_HEAD_DIM, 256),
            ("gdn_key_heads", A::GDN_KEY_HEADS, 16),
            ("gdn_value_heads", A::GDN_VALUE_HEADS, 48),
            ("gdn_key_head_dim", A::GDN_KEY_HEAD_DIM, 128),
            ("gdn_value_head_dim", A::GDN_VALUE_HEAD_DIM, 128),
            ("gdn_conv_kernel", A::GDN_CONV_KERNEL, 4),
            ("attention_query_rows", A::ATTENTION_QUERY_ROWS, 12_288),
            ("attention_kv_rows", A::ATTENTION_KV_ROWS, 1_024),
            ("attention_qkv_rows", A::ATTENTION_QKV_ROWS, 14_336),
            ("gdn_qk_rows", A::GDN_QK_ROWS, 2_048),
            ("gdn_value_rows", A::GDN_VALUE_ROWS, 6_144),
            ("gdn_qkv_rows", A::GDN_QKV_ROWS, 10_240),
            ("gdn_input_rows", A::GDN_INPUT_ROWS, 16_384),
            ("gdn_control_rows", A::GDN_CONTROL_ROWS, 48),
        ] {
            assert_eq!(actual, expected, "{field}");
        }
    }

    #[test]
    fn qwen38_profile_matches_checkpoint_identity() {
        assert_eq!(Qwen38_27bNvfp4::MODEL_ID, "unsloth/Qwen3.8-27B-NVFP4");
        assert_eq!(
            Qwen38_27bNvfp4::REVISION,
            "16b6615af3548b88e2d8e382457bc705b00479cf"
        );
    }
}
