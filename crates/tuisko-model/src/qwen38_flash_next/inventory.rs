//! Exact 206-shard inventory for Qwen3.8-Flash-Next.
//!
//! Fourteen fixed shards hold non-routed tensors; 192 shards each hold 128 routed experts.
//! File and tensor-payload byte totals are pinned separately.

use crate::CheckpointSnapshot;
use crate::common::inventory::{
    CONFIG_FILE, ExpectedTensor, INDEX_FILE, Shard, add_expected, add_modelopt_linear,
    add_vision_expected_tensors, dimension, read_index, require_count, validate_expected_tensor,
    validate_file_length, validate_revision,
};
use crate::common::naming::{EMBEDDING, LM_HEAD, layer_prefix};
use crate::common::schema::validate_config;
use crate::qwen38_flash_next::bindings::HYPER_CONNECTION_MIXER;
use crate::{Arch, CheckpointError, CheckpointResult, DType, Qwen38FlashNext, SafeTensorFile};
use std::collections::BTreeMap;
use std::fs;
use std::marker::PhantomData;
use std::path::Path;

/// Routed experts held by one expert shard.
const EXPERTS_PER_SHARD: usize = 128;
/// Expert shards owned by one decoder layer.
const EXPERT_SHARD_BLOCKS: usize = 4;
/// Root of the MTP draft block.
const MTP_ROOT: &str = "mtp";

const QWEN38_FLASH_NEXT_FIXED_SHARDS: [FixedShardSpec; 14] = [
    FixedShardSpec::new("model-bf16-00001.safetensors", 1_273_114_461, 385),
    FixedShardSpec::new("model-bf16-00010.safetensors", 323_564_201, 58),
    FixedShardSpec::new("model-bf16-00011.safetensors", 10_727_387_912, 821),
    FixedShardSpec::new("model-bf16-00012.safetensors", 3_683_689_888, 170),
    FixedShardSpec::new("model-plefp8-00000.safetensors", 5_200_027_199, 13),
    FixedShardSpec::new("model-plefp8-00001.safetensors", 5_200_027_202, 13),
    FixedShardSpec::new("model-plefp8-00002.safetensors", 5_200_027_198, 13),
    FixedShardSpec::new("model-plefp8-00003.safetensors", 5_200_027_189, 13),
    FixedShardSpec::new("model-plefp8-00004.safetensors", 5_200_027_190, 13),
    FixedShardSpec::new("model-plefp8-00005.safetensors", 5_200_027_190, 13),
    FixedShardSpec::new("model-plefp8-00006.safetensors", 5_200_027_190, 13),
    FixedShardSpec::new("model-plefp8-00007.safetensors", 5_200_027_190, 13),
    FixedShardSpec::new("model-plefp8-00008.safetensors", 5_200_027_190, 13),
    FixedShardSpec::new("model-plefp8-00009.safetensors", 4_400_023_163, 12),
];

pub(crate) const QWEN38_FLASH_NEXT_INVENTORY: ShardedInventorySpec = ShardedInventorySpec {
    index_bytes: 34_258_807,
    index_entries: 296_475,
    snapshot_bytes: 135_195_303_851,
    payload_bytes: 135_156_121_594,
    expert_shard_tensors: 1_536,
    expert_shard_bytes: 67_987_279_488,
    fixed: QWEN38_FLASH_NEXT_FIXED_SHARDS,
};

#[derive(Clone, Copy)]
pub(crate) struct ShardedInventorySpec {
    index_bytes: u64,
    index_entries: usize,
    /// `metadata.total_size`: the summed length of all 206 shard files, framing included.
    snapshot_bytes: u64,
    /// Summed length of every tensor's payload, framing excluded.
    payload_bytes: u64,
    expert_shard_tensors: usize,
    expert_shard_bytes: u64,
    fixed: [FixedShardSpec; 14],
}

#[derive(Clone, Copy)]
pub(crate) struct FixedShardSpec {
    file: &'static str,
    file_bytes: u64,
    tensors: usize,
}

impl FixedShardSpec {
    const fn new(file: &'static str, file_bytes: u64, tensors: usize) -> Self {
        Self {
            file,
            file_bytes,
            tensors,
        }
    }
}

/// Shard file holding `block`'s 128 routed experts for `layer`.
fn expert_shard_file(layer: usize, block: usize) -> String {
    let first = block * EXPERTS_PER_SHARD;

    format!(
        "layer-{layer:05}-experts-{first:04}-{last:04}.safetensors",
        last = first + EXPERTS_PER_SHARD - 1
    )
}

/// Length of one shard file, for the summed snapshot-byte gate.
fn shard_bytes(path: &Path) -> CheckpointResult<u64> {
    Ok(fs::metadata(path)
        .map_err(|source| CheckpointError::io("reading metadata for", path, source))?
        .len())
}

/// Layer and routed-expert index a tensor key names, if it is a routed-expert plane.
fn routed_expert_position(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let (expert, _) = rest.strip_prefix("mlp.experts.")?.split_once('.')?;

    Some((layer.parse().ok()?, expert.parse().ok()?))
}

fn qwen38_flash_next_expected_tensors<A: Arch>()
-> CheckpointResult<BTreeMap<String, ExpectedTensor>> {
    type F = Qwen38FlashNext;

    if A::FULL_ATTENTION_INTERVAL == 0 {
        return Err(CheckpointError::inventory(
            "Qwen3.8-Flash-Next full-attention interval must be nonzero",
        ));
    }

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let vocab = dimension(A::VOCAB, "vocabulary size")?;
    let expert_intermediate = dimension(A::INTERMEDIATE, "expert intermediate width")?;
    let mut expected = BTreeMap::new();

    for name in [EMBEDDING, LM_HEAD] {
        add_expected(&mut expected, name, DType::Bf16, vec![vocab, hidden])?;
    }
    add_hyper_connection(&mut expected, HYPER_CONNECTION_MIXER, false)?;

    for layer in 0..A::LAYERS {
        let prefix = layer_prefix(layer);

        for module in ["attn_hyper_connection", "mlp_hyper_connection"] {
            add_hyper_connection(&mut expected, &format!("{prefix}.{module}"), true)?;
        }
        add_moe::<A>(&mut expected, &format!("{prefix}.mlp"))?;

        for expert in 0..F::NUM_EXPERTS {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");

            for projection in ["gate_proj", "up_proj"] {
                add_modelopt_linear(
                    &mut expected,
                    &format!("{expert_prefix}.{projection}"),
                    expert_intermediate,
                    hidden,
                )?;
            }
            add_modelopt_linear(
                &mut expected,
                &format!("{expert_prefix}.down_proj"),
                hidden,
                expert_intermediate,
            )?;
        }

        if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            add_sparse_attention::<A>(&mut expected, &format!("{prefix}.self_attn"))?;
        } else {
            add_gated_deltanet::<A>(&mut expected, &format!("{prefix}.linear_attn"))?;
        }
        if layer == F::PLE_LAYER {
            add_engram(&mut expected, &format!("{prefix}.ple"))?;
        }
    }

    add_mtp::<A>(&mut expected)?;
    add_vision_expected_tensors::<A>(&mut expected)?;

    Ok(expected)
}

/// One gated residual. The model-level and MTP mixers collapse the four branches without
/// writing back, so they carry no `block_inject_weight`.
fn add_hyper_connection(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
    combines: bool,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

    let width = dimension(F::HC_WIDTH, "hyper-connection width")?;
    let lowrank = dimension(F::HC_LOWRANK, "hyper-connection rank")?;

    add_expected(
        expected,
        format!("{prefix}.hc_norm.weight"),
        DType::Bf16,
        vec![width],
    )?;
    add_expected(
        expected,
        format!("{prefix}.input_mix_weight_down.weight"),
        DType::Bf16,
        vec![lowrank, width],
    )?;
    add_expected(
        expected,
        format!("{prefix}.input_mix_weight_up.weight"),
        DType::Bf16,
        vec![width, lowrank],
    )?;

    if !combines {
        return Ok(());
    }

    add_expected(
        expected,
        format!("{prefix}.block_inject_weight.weight"),
        DType::Bf16,
        vec![dimension(F::HC_COUNT, "hyper-connection branches")?, width],
    )
}

/// The router, the shared expert, and its gate. Routed experts are added by the caller
/// because only the target's decoder layers shard them.
fn add_moe<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let shared = dimension(F::SHARED_EXPERT_INTERMEDIATE, "shared expert width")?;

    add_expected(
        expected,
        format!("{prefix}.gate.weight"),
        DType::Bf16,
        vec![dimension(F::NUM_EXPERTS, "expert count")?, hidden],
    )?;
    add_expected(
        expected,
        format!("{prefix}.shared_expert_gate.weight"),
        DType::Bf16,
        vec![1, hidden],
    )?;
    for (projection, rows, columns) in [
        ("gate_proj", shared, hidden),
        ("up_proj", shared, hidden),
        ("down_proj", hidden, shared),
    ] {
        add_expected(
            expected,
            format!("{prefix}.shared_expert.{projection}.weight"),
            DType::Bf16,
            vec![rows, columns],
        )?;
    }

    Ok(())
}

/// One `qwen_sparse_attention` layer: gated GQA plus the block-selection indexer.
fn add_sparse_attention<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let head_dim = dimension(A::HEAD_DIM, "attention head width")?;

    for (projection, rows, columns) in [
        (
            "q_proj",
            dimension(A::ATTENTION_QUERY_ROWS, "attention query rows")?,
            hidden,
        ),
        (
            "k_proj",
            dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?,
            hidden,
        ),
        (
            "v_proj",
            dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?,
            hidden,
        ),
        (
            "o_proj",
            hidden,
            dimension(A::ATTENTION_OUTPUT_COLUMNS, "attention output columns")?,
        ),
    ] {
        add_expected(
            expected,
            format!("{prefix}.{projection}.weight"),
            DType::Bf16,
            vec![rows, columns],
        )?;
    }
    for name in ["q_norm", "k_norm"] {
        add_expected(
            expected,
            format!("{prefix}.{name}.weight"),
            DType::Bf16,
            vec![head_dim],
        )?;
    }

    let indexer_dim = dimension(F::INDEXER_HEAD_DIM, "indexer head width")?;
    add_expected(
        expected,
        format!("{prefix}.indexer.index_qk_proj.weight"),
        DType::Bf16,
        vec![
            dimension(F::INDEXER_ROWS, "indexer projection rows")?,
            hidden,
        ],
    )?;
    for name in ["q_layernorm", "k_layernorm"] {
        add_expected(
            expected,
            format!("{prefix}.indexer.{name}.weight"),
            DType::Bf16,
            vec![indexer_dim],
        )?;
    }

    Ok(())
}

/// One `linear_attention` layer. Four separate input projections, not the Qwen3-Next packing.
fn add_gated_deltanet<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
) -> CheckpointResult<()> {
    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let control_rows = dimension(A::GDN_CONTROL_ROWS, "GDN control rows")?;
    let qkv_rows = dimension(A::GDN_QKV_ROWS, "GDN QKV rows")?;
    let value_rows = dimension(A::GDN_VALUE_ROWS, "GDN value rows")?;

    for (projection, rows, columns) in [
        ("in_proj_qkv", qkv_rows, hidden),
        ("in_proj_z", value_rows, hidden),
        ("in_proj_a", control_rows, hidden),
        ("in_proj_b", control_rows, hidden),
        ("out_proj", hidden, value_rows),
    ] {
        add_expected(
            expected,
            format!("{prefix}.{projection}.weight"),
            DType::Bf16,
            vec![rows, columns],
        )?;
    }
    for name in ["A_log", "dt_bias"] {
        add_expected(
            expected,
            format!("{prefix}.{name}"),
            DType::Bf16,
            vec![control_rows],
        )?;
    }
    add_expected(
        expected,
        format!("{prefix}.conv1d.weight"),
        DType::Bf16,
        vec![
            qkv_rows,
            1,
            dimension(A::LINEAR_CONV_KERNEL_DIM, "convolution width")?,
        ],
    )?;
    add_expected(
        expected,
        format!("{prefix}.norm.weight"),
        DType::Bf16,
        vec![dimension(A::LINEAR_HEAD_DIM, "GDN head width")?],
    )
}

/// The single engram (PLE) block: gate projections, the dilated short convolution, the three
/// I64 hash constants, and the 128 FP8 table shards with their one shared scale.
fn add_engram(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

    let width = dimension(F::HC_WIDTH, "hyper-connection width")?;
    let embed = dimension(F::PLE_EMBED_DIM, "engram embedding width")?;

    add_expected(
        expected,
        format!("{prefix}.key_proj.weight"),
        DType::Bf16,
        vec![width, embed],
    )?;
    add_expected(
        expected,
        format!("{prefix}.value_proj.weight"),
        DType::Bf16,
        vec![embed, embed],
    )?;
    for name in ["norm_key", "norm_query", "norm_conv"] {
        add_expected(
            expected,
            format!("{prefix}.{name}.weight"),
            DType::Bf16,
            vec![width],
        )?;
    }
    add_expected(
        expected,
        format!("{prefix}.conv1d.weight"),
        DType::Bf16,
        vec![
            width,
            1,
            dimension(F::PLE_CONV_KERNEL, "engram convolution width")?,
        ],
    )?;

    let table = format!("{prefix}.ple_embedding");
    let heads = dimension(F::NGRAM_HEADS, "engram heads")?;
    add_expected(
        expected,
        format!("{table}.layer_multipliers"),
        DType::I64,
        vec![dimension(F::NGRAM_SIZE, "engram n-gram size")?],
    )?;
    for name in ["ngram_heads_offsets", "ngram_heads_vocab_sizes"] {
        add_expected(expected, format!("{table}.{name}"), DType::I64, vec![heads])?;
    }

    let rows = dimension(F::NGRAM_SHARD_ROWS, "engram shard rows")?;
    let head_dim = dimension(F::NGRAM_HEAD_DIM, "engram head width")?;
    for shard in 0..F::NGRAM_SHARDS {
        add_expected(
            expected,
            format!("{table}.ngram_embedding.shard_{shard}.weight"),
            DType::Fp8E4M3,
            vec![rows, head_dim],
        )?;
    }
    add_expected(
        expected,
        format!("{table}.ngram_embedding.weight_scale"),
        DType::Bf16,
        vec![1],
    )
}

/// The draft block. The ModelOpt run ignored every `mtp*` module, so its 512-expert pool
/// stays BF16 and stays fused as `gate_up_proj` / `down_proj`.
fn add_mtp<A: Arch>(expected: &mut BTreeMap<String, ExpectedTensor>) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let experts = dimension(F::NUM_EXPERTS, "expert count")?;
    let expert_intermediate = dimension(A::INTERMEDIATE, "expert intermediate width")?;
    let gate_up_rows = expert_intermediate
        .checked_mul(2)
        .ok_or_else(|| CheckpointError::inventory("MTP expert gate/up rows overflow"))?;

    add_expected(
        expected,
        format!("{MTP_ROOT}.pre_fc_norm_embedding.weight"),
        DType::Bf16,
        vec![hidden],
    )?;
    add_expected(
        expected,
        format!("{MTP_ROOT}.pre_fc_norm_hidden.weight"),
        DType::Bf16,
        vec![dimension(F::HC_WIDTH, "hyper-connection width")?],
    )?;
    for name in ["fc_embedding", "fc_hidden"] {
        add_expected(
            expected,
            format!("{MTP_ROOT}.{name}.weight"),
            DType::Bf16,
            vec![hidden, hidden],
        )?;
    }
    add_hyper_connection(
        expected,
        &format!("{MTP_ROOT}.hyper_connection_mixer"),
        false,
    )?;

    for layer in 0..A::MTP_LAYERS {
        let prefix = format!("{MTP_ROOT}.layers.{layer}");

        for module in ["attn_hyper_connection", "mlp_hyper_connection"] {
            add_hyper_connection(expected, &format!("{prefix}.{module}"), true)?;
        }
        add_sparse_attention::<A>(expected, &format!("{prefix}.self_attn"))?;
        add_moe::<A>(expected, &format!("{prefix}.mlp"))?;
        add_expected(
            expected,
            format!("{prefix}.mlp.experts.gate_up_proj"),
            DType::Bf16,
            vec![experts, gate_up_rows, hidden],
        )?;
        add_expected(
            expected,
            format!("{prefix}.mlp.experts.down_proj"),
            DType::Bf16,
            vec![experts, hidden, expert_intermediate],
        )?;
    }

    Ok(())
}

/// Summed payload of every expected tensor, framing excluded.
fn expected_payload_bytes(expected: &BTreeMap<String, ExpectedTensor>) -> CheckpointResult<u64> {
    let mut total = 0u64;

    for (name, descriptor) in expected {
        let elements = descriptor
            .shape
            .iter()
            .try_fold(1u64, |elements, extent| elements.checked_mul(*extent))
            .and_then(|elements| elements.checked_mul(descriptor.dtype.byte_width()))
            .ok_or_else(|| {
                CheckpointError::inventory(format!("tensor `{name}` payload bytes overflow"))
            })?;
        total = total.checked_add(elements).ok_or_else(|| {
            CheckpointError::inventory("Qwen3.8-Flash-Next payload bytes overflow")
        })?;
    }

    Ok(total)
}

fn validate_sharded_weight_map(
    index_path: &Path,
    weight_map: BTreeMap<String, String>,
    order: &BTreeMap<String, usize>,
    shards: &[SafeTensorFile],
    expected: &BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<BTreeMap<String, Shard>> {
    let mut tensors = BTreeMap::new();

    for (name, file) in weight_map {
        let shard_index = order.get(&file).copied().ok_or_else(|| {
            CheckpointError::inventory(format!(
                "{} maps tensor `{name}` to unsupported shard `{file}`",
                index_path.display()
            ))
        })?;
        let descriptor = expected.get(&name).ok_or_else(|| {
            CheckpointError::inventory(format!(
                "{} contains unexpected tensor `{name}`",
                index_path.display()
            ))
        })?;

        if let Some((layer, expert)) = routed_expert_position(&name) {
            let owner = expert_shard_file(layer, expert / EXPERTS_PER_SHARD);

            if file != owner {
                return Err(CheckpointError::inventory(format!(
                    "{} places tensor `{name}` in shard `{file}`, expected `{owner}`",
                    index_path.display()
                )));
            }
        }

        validate_expected_tensor(&shards[shard_index], &name, descriptor)?;
        tensors.insert(name, Shard(shard_index));
    }

    for name in expected.keys() {
        if !tensors.contains_key(name) {
            return Err(CheckpointError::inventory(format!(
                "{} is missing tensor `{name}`",
                index_path.display()
            )));
        }
    }

    Ok(tensors)
}

impl<A: Arch> CheckpointSnapshot<A> {
    pub(crate) fn open_sharded(root: &Path, spec: ShardedInventorySpec) -> CheckpointResult<Self> {
        validate_revision::<A>(root)?;
        validate_config::<A>(&root.join(CONFIG_FILE))?;

        let index_path = root.join(INDEX_FILE);
        validate_file_length(&index_path, spec.index_bytes)?;
        let index = read_index(&index_path)?;
        require_count(
            &index_path,
            "entries",
            index.weight_map.len(),
            spec.index_entries,
        )?;
        require_count(
            &index_path,
            "metadata.total_size",
            index.metadata.total_size,
            spec.snapshot_bytes,
        )?;
        if index.metadata.total_parameters.is_some() {
            return Err(CheckpointError::inventory(format!(
                "{} declares metadata.total_parameters, which this contract does not carry",
                index_path.display()
            )));
        }

        let mut order = BTreeMap::new();
        let mut shards = Vec::new();
        let mut snapshot_bytes = 0u64;

        for fixed in spec.fixed {
            let path = root.join(fixed.file);
            validate_file_length(&path, fixed.file_bytes)?;
            let file = SafeTensorFile::open(&path)?;
            require_count(&path, "tensors", file.tensor_count(), fixed.tensors)?;
            order.insert(String::from(fixed.file), shards.len());
            shards.push(file);
            snapshot_bytes = snapshot_bytes
                .checked_add(fixed.file_bytes)
                .ok_or_else(|| CheckpointError::inventory("snapshot file bytes overflow"))?;
        }

        let mut expert_bytes = 0u64;
        for layer in 0..A::LAYERS {
            for block in 0..EXPERT_SHARD_BLOCKS {
                let name = expert_shard_file(layer, block);
                let path = root.join(&name);
                let bytes = shard_bytes(&path)?;
                let file = SafeTensorFile::open(&path)?;
                require_count(
                    &path,
                    "tensors",
                    file.tensor_count(),
                    spec.expert_shard_tensors,
                )?;
                expert_bytes = expert_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| CheckpointError::inventory("expert shard bytes overflow"))?;
                order.insert(name, shards.len());
                shards.push(file);
            }
        }
        require_count(
            &index_path,
            "expert shard bytes",
            expert_bytes,
            spec.expert_shard_bytes,
        )?;
        snapshot_bytes = snapshot_bytes
            .checked_add(expert_bytes)
            .ok_or_else(|| CheckpointError::inventory("snapshot file bytes overflow"))?;
        require_count(
            &index_path,
            "snapshot bytes",
            snapshot_bytes,
            spec.snapshot_bytes,
        )?;

        let expected = qwen38_flash_next_expected_tensors::<A>()?;
        require_count(
            &index_path,
            "expected tensors",
            expected.len(),
            spec.index_entries,
        )?;
        require_count(
            &index_path,
            "payload bytes",
            expected_payload_bytes(&expected)?,
            spec.payload_bytes,
        )?;
        let tensors =
            validate_sharded_weight_map(&index_path, index.weight_map, &order, &shards, &expected)?;

        Ok(Self {
            root: root.to_owned(),
            inventory_path: index_path,
            tensors,
            shards,
            arch: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::inventory::ExpectedTensor;
    use crate::{DType, Qwen38FlashNext};
    use std::collections::BTreeMap;

    #[test]
    fn qwen38_flash_next_shard_table_covers_the_pinned_snapshot() {
        type A = Qwen38FlashNext;

        let expert_shards = A::LAYERS * EXPERT_SHARD_BLOCKS;
        let fixed_tensors = QWEN38_FLASH_NEXT_FIXED_SHARDS
            .iter()
            .map(|shard| shard.tensors)
            .sum::<usize>();
        let fixed_bytes = QWEN38_FLASH_NEXT_FIXED_SHARDS
            .iter()
            .map(|shard| shard.file_bytes)
            .sum::<u64>();

        assert_eq!(EXPERT_SHARD_BLOCKS * EXPERTS_PER_SHARD, A::NUM_EXPERTS);
        assert_eq!(expert_shards, 192);
        assert_eq!(QWEN38_FLASH_NEXT_FIXED_SHARDS.len() + expert_shards, 206);
        assert_eq!(
            fixed_tensors + expert_shards * QWEN38_FLASH_NEXT_INVENTORY.expert_shard_tensors,
            QWEN38_FLASH_NEXT_INVENTORY.index_entries
        );
        assert_eq!(
            fixed_bytes + QWEN38_FLASH_NEXT_INVENTORY.expert_shard_bytes,
            QWEN38_FLASH_NEXT_INVENTORY.snapshot_bytes
        );
        assert_ne!(
            QWEN38_FLASH_NEXT_INVENTORY.snapshot_bytes, QWEN38_FLASH_NEXT_INVENTORY.payload_bytes,
            "metadata.total_size counts shard framing; the payload total must not absorb it"
        );

        assert_eq!(
            expert_shard_file(0, 0),
            "layer-00000-experts-0000-0127.safetensors"
        );
        assert_eq!(
            expert_shard_file(47, 3),
            "layer-00047-experts-0384-0511.safetensors"
        );
        assert_eq!(
            routed_expert_position("model.language_model.layers.7.mlp.experts.300.up_proj.weight"),
            Some((7, 300))
        );
        assert_eq!(
            routed_expert_position("model.language_model.layers.7.mlp.gate.weight"),
            None
        );
        assert_eq!(
            routed_expert_position("mtp.layers.0.mlp.experts.gate_up_proj"),
            None
        );
    }

    #[test]
    fn qwen38_flash_next_inventory_is_bijective_and_byte_exact() {
        let expected = qwen38_flash_next_expected_tensors::<Qwen38FlashNext>().unwrap();
        let mut dtype_counts = BTreeMap::new();

        for descriptor in expected.values() {
            *dtype_counts
                .entry(descriptor.dtype.as_str())
                .or_insert(0usize) += 1;
        }

        assert_eq!(expected.len(), QWEN38_FLASH_NEXT_INVENTORY.index_entries);
        assert_eq!(
            dtype_counts,
            BTreeMap::from([
                ("BF16", 1_432),
                ("F32", 147_456),
                ("F8_E4M3", 73_856),
                ("I64", 3),
                ("U8", 73_728),
            ])
        );
        assert_eq!(
            expected_payload_bytes(&expected).unwrap(),
            QWEN38_FLASH_NEXT_INVENTORY.payload_bytes
        );
    }

    #[test]
    fn qwen38_flash_next_inventory_pins_every_new_subsystem_plane() {
        let expected = qwen38_flash_next_expected_tensors::<Qwen38FlashNext>().unwrap();

        for (name, dtype, shape) in [
            (
                "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                DType::Bf16,
                vec![320, 10_240],
            ),
            (
                "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
                DType::Bf16,
                vec![4, 10_240],
            ),
            (
                "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
                DType::Bf16,
                vec![640, 2_560],
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                DType::Bf16,
                vec![12_288, 2_560],
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                DType::Bf16,
                vec![10_240, 1, 4],
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                DType::I64,
                vec![3],
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                DType::Fp8E4M3,
                vec![2_500_012, 160],
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
                DType::Bf16,
                vec![1],
            ),
            (
                "model.language_model.layers.47.mlp.experts.511.gate_proj.weight",
                DType::U8,
                vec![640, 1_280],
            ),
            (
                "model.language_model.layers.47.mlp.experts.511.down_proj.weight_scale",
                DType::Fp8E4M3,
                vec![2_560, 40],
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                DType::Bf16,
                vec![512, 1_280, 2_560],
            ),
            (
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                DType::Bf16,
                vec![10_240, 320],
            ),
        ] {
            assert_eq!(expected[name], ExpectedTensor { dtype, shape }, "{name}");
        }

        // No classic per-layer norms and no final norm survive in this architecture.
        assert!(!expected.contains_key("model.language_model.norm.weight"));
        assert!(!expected.contains_key("model.language_model.layers.0.input_layernorm.weight"));
        assert!(
            !expected.contains_key("model.language_model.layers.0.post_attention_layernorm.weight")
        );
        // Layer routing: 12 sparse-attention layers, 36 gated-DeltaNet layers, one engram.
        assert!(!expected.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(
            !expected.contains_key("model.language_model.layers.3.linear_attn.in_proj_qkv.weight")
        );
        assert!(!expected.contains_key("model.language_model.layers.0.ple.key_proj.weight"));
        // The MTP pool is ignored by the ModelOpt run and stays BF16 and fused.
        assert!(!expected.contains_key("mtp.layers.0.mlp.experts.0.gate_proj.weight"));
    }

    /// Partitions every inventory entry by binding family; only 31 MTP tensors remain deferred.
    #[test]
    fn every_non_mtp_tensor_is_claimed_by_an_admitted_source_binding() {
        fn binding_family(name: &str) -> &'static str {
            if name.starts_with(MTP_ROOT) {
                return "mtp";
            }
            if name.starts_with("model.visual.") {
                return "vision";
            }
            if name == EMBEDDING || name == LM_HEAD || name.starts_with(HYPER_CONNECTION_MIXER) {
                return "endpoints";
            }
            if name.contains("_hyper_connection.") {
                return "hyper-connections";
            }
            if name.contains(".mlp.experts.") {
                return "routed experts";
            }
            if name.contains(".mlp.") {
                return "moe";
            }
            if name.contains(".self_attn.") {
                return "sparse_attention";
            }
            if name.contains(".linear_attn.") {
                return "gdn";
            }
            if name.contains(".ple.") {
                return "engram";
            }

            "unclaimed"
        }

        let expected = qwen38_flash_next_expected_tensors::<Qwen38FlashNext>().unwrap();
        let mut counts = BTreeMap::new();

        for name in expected.keys() {
            *counts.entry(binding_family(name)).or_insert(0usize) += 1;
        }

        assert_eq!(
            counts,
            BTreeMap::from([
                ("endpoints", 5),
                ("engram", 138),
                ("gdn", 324),
                ("hyper-connections", 384),
                ("moe", 240),
                ("mtp", 31),
                ("sparse_attention", 108),
                ("routed experts", 294_912),
                ("vision", 333),
            ])
        );
        assert_eq!(
            counts.values().sum::<usize>(),
            QWEN38_FLASH_NEXT_INVENTORY.index_entries
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete Flash-Next checkpoint"]
    fn qwen38_flash_next_snapshot_inventory_is_byte_exact() {
        let root = std::env::var("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").unwrap();
        let snapshot = CheckpointSnapshot::<Qwen38FlashNext>::open(Path::new(&root)).unwrap();

        assert_eq!(
            snapshot.tensor_count(),
            QWEN38_FLASH_NEXT_INVENTORY.index_entries
        );
        assert_eq!(
            snapshot
                .tensor("model.language_model.layers.47.mlp.experts.511.gate_proj.weight")
                .unwrap()
                .shape,
            [640, 1_280]
        );
        assert_eq!(
            snapshot
                .tensor("model.language_model.layers.1.ple.ple_embedding.layer_multipliers")
                .unwrap()
                .dtype,
            DType::I64
        );
    }
}
