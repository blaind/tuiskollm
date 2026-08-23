use super::{
    AttentionWeights, DenseFp8MlpWeights, EndpointWeights, GdnPersistent, GdnWeights, MixerWeights,
    MlpWeights, Nvfp4MlpWeights, PersistentState, ResidentModelLayout, SharedWorkspace,
};
use crate::{EngineError, EngineResult};
use std::collections::BTreeSet;
use tuisko_gpu::ArenaRegion;

/// Device allocation selected by one startup initialization entry.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResidentUploadArena {
    /// Weights, persistent GDN state, and shared execution workspace.
    Resident,
    /// Shared page tables and represented attention cache.
    Kv,
}

/// Host preparation required before one destination can be initialized.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResidentUploadPreparation {
    /// Validated represented bytes can be copied without reordering.
    BorrowedSource,
    /// Multiple validated source tensors are gathered into resident order.
    GatheredSource,
    /// Represented scale codes are losslessly reordered into kernel order.
    SwizzledSource,
    /// Small runtime metadata is derived directly on the host.
    HostDerived,
    /// Runtime, cache, or alignment bytes begin as zero.
    Zero,
}

/// One exactly covered destination range in the resident startup plan.
#[derive(Clone, Debug)]
pub struct ResidentUploadEntry {
    arena: ResidentUploadArena,
    offset: usize,
    bytes: usize,
    preparation: ResidentUploadPreparation,
    role: String,
    source: UploadSource,
    padding: bool,
}

impl ResidentUploadEntry {
    /// Device allocation containing this range.
    pub const fn arena(&self) -> ResidentUploadArena {
        self.arena
    }

    /// Byte offset from the selected allocation's stable base address.
    pub const fn offset_bytes(&self) -> usize {
        self.offset
    }

    /// Exact number of destination bytes initialized by this entry.
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Preparation required to produce the entry's represented bytes.
    pub const fn preparation(&self) -> ResidentUploadPreparation {
        self.preparation
    }

    /// Stable exact-target role used in diagnostics and reports.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Whether this entry initializes alignment bytes outside typed ownership.
    pub const fn is_padding(&self) -> bool {
        self.padding
    }
}

/// Complete, non-overlapping initialization contract for both model allocations.
#[derive(Clone, Debug)]
pub struct ResidentUploadPlan {
    entries: Vec<ResidentUploadEntry>,
    weight_bytes: usize,
    host_derived_bytes: usize,
    zeroed_owner_bytes: usize,
    padding_bytes: usize,
    initialized_bytes: usize,
}

impl ResidentUploadPlan {
    /// Derives the exact startup initialization inventory from a checked model layout.
    pub fn build(layout: &ResidentModelLayout) -> EngineResult<Self> {
        let mut entries = Vec::new();
        for (layer, layout) in layout.layers.iter().enumerate() {
            push_mixer_weights(&mut entries, layer, layout.mixer);
            push_mlp_weights(&mut entries, layer, layout.mlp);
            if let PersistentState::Gdn(state) = layout.persistent {
                push_gdn_persistent(&mut entries, layer, state);
            }
        }
        push_endpoint_weights(&mut entries, layout.endpoint);
        push_workspace(&mut entries, layout.workspace);
        push_kv(&mut entries, layout);
        let entries = insert_padding(entries, layout)?;
        validate_unique_sources(&entries)?;
        validate_coverage(&entries, layout)?;

        let weight_bytes = bytes_where(&entries, |entry| entry.source.is_weight())?;
        let host_derived_bytes = bytes_where(&entries, |entry| {
            matches!(entry.source, UploadSource::Metadata(_))
        })?;
        let zeroed_owner_bytes = bytes_where(&entries, |entry| {
            matches!(entry.source, UploadSource::Zero) && !entry.padding
        })?;
        let padding_bytes = bytes_where(&entries, |entry| entry.padding)?;
        let resident_metadata_bytes = layout
            .workspace
            .state_rows
            .byte_len()
            .checked_add(layout.workspace.table_rows.byte_len())
            .ok_or_else(|| EngineError::layout("resident upload metadata bytes overflow"))?;
        let expected_host_derived = resident_metadata_bytes
            .checked_add(layout.kv_layout.block_table_bytes())
            .ok_or_else(|| EngineError::layout("resident upload metadata bytes overflow"))?;
        let expected_zeroed_owner = layout
            .history_bytes()
            .checked_add(layout.state_bytes())
            .and_then(|bytes| bytes.checked_add(layout.cache_bytes()))
            .and_then(|bytes| bytes.checked_add(layout.workspace_bytes()))
            .and_then(|bytes| bytes.checked_sub(resident_metadata_bytes))
            .ok_or_else(|| EngineError::layout("resident upload zeroed bytes overflow"))?;
        let initialized_bytes = bytes_where(&entries, |_| true)?;

        require_equal(
            "resident upload weight bytes",
            weight_bytes,
            layout.resident_weight_bytes(),
        )?;
        require_equal(
            "resident upload host-derived bytes",
            host_derived_bytes,
            expected_host_derived,
        )?;
        require_equal(
            "resident upload zeroed owner bytes",
            zeroed_owner_bytes,
            expected_zeroed_owner,
        )?;
        require_equal(
            "resident upload padding bytes",
            padding_bytes,
            layout.padding_bytes(),
        )?;
        require_equal(
            "resident upload initialized bytes",
            initialized_bytes,
            layout.arena_bytes(),
        )?;

        Ok(Self {
            entries,
            weight_bytes,
            host_derived_bytes,
            zeroed_owner_bytes,
            padding_bytes,
            initialized_bytes,
        })
    }

    /// Every destination entry, sorted by allocation and byte offset.
    pub fn entries(&self) -> &[ResidentUploadEntry] {
        &self.entries
    }

    /// Source-backed represented weight bytes written exactly once.
    pub const fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Runtime metadata bytes derived on the host.
    pub const fn host_derived_bytes(&self) -> usize {
        self.host_derived_bytes
    }

    /// Typed runtime, workspace, and cache bytes initialized to zero.
    pub const fn zeroed_owner_bytes(&self) -> usize {
        self.zeroed_owner_bytes
    }

    /// Alignment bytes explicitly initialized outside typed ownership.
    pub const fn padding_bytes(&self) -> usize {
        self.padding_bytes
    }

    /// Complete bytes initialized across both address-stable allocations.
    pub const fn initialized_bytes(&self) -> usize {
        self.initialized_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum UploadSource {
    Weight(WeightSource),
    Metadata(MetadataSource),
    Zero,
}

impl UploadSource {
    const fn is_weight(self) -> bool {
        matches!(self, Self::Weight(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum WeightSource {
    Gdn {
        layer: usize,
        plane: GdnPlane,
    },
    Attention {
        layer: usize,
        plane: AttentionPlane,
    },
    Nvfp4Mlp {
        layer: usize,
        plane: Nvfp4MlpPlane,
    },
    DenseFp8Mlp {
        layer: usize,
        plane: DenseFp8MlpPlane,
    },
    Endpoint(EndpointPlane),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum GdnPlane {
    InputNorm,
    InputWeightCodes,
    InputWeightScales,
    ControlWeights,
    ALog,
    DtBias,
    ConvolutionWeights,
    RecurrentNorm,
    OutputWeightCodes,
    OutputWeightScales,
    PostAttentionNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AttentionPlane {
    InputNorm,
    QkvWeightCodes,
    QkvWeightScales,
    QueryNorm,
    KeyNorm,
    OutputWeightCodes,
    OutputWeightScales,
    PostAttentionNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Nvfp4MlpPlane {
    GateWeightCodes,
    UpWeightCodes,
    GateUpWeightScales,
    DownWeightCodes,
    DownWeightScales,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DenseFp8MlpPlane {
    GateUpWeightCodes,
    GateUpWeightScales,
    DownWeightCodes,
    DownWeightScales,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum EndpointPlane {
    FinalNorm,
    LmHeadCodes,
    LmHeadScales,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum MetadataSource {
    StateRows,
    TableRows,
    KvBlockTables,
}

fn push_mixer_weights(entries: &mut Vec<ResidentUploadEntry>, layer: usize, mixer: MixerWeights) {
    match mixer {
        MixerWeights::Gdn(weights) => push_gdn_weights(entries, layer, weights),
        MixerWeights::Attention(weights) => push_attention_weights(entries, layer, weights),
    }
}

fn push_gdn_weights(entries: &mut Vec<ResidentUploadEntry>, layer: usize, weights: GdnWeights) {
    let prefix = format!("layers.{layer}.gdn");
    macro_rules! push {
        ($field:ident, $plane:ident, $preparation:ident) => {
            push_weight(
                entries,
                weights.$field,
                WeightSource::Gdn {
                    layer,
                    plane: GdnPlane::$plane,
                },
                ResidentUploadPreparation::$preparation,
                format!("{prefix}.{}", stringify!($field)),
            );
        };
    }
    push!(input_norm, InputNorm, BorrowedSource);
    push!(input_weight_codes, InputWeightCodes, BorrowedSource);
    push!(input_weight_scales, InputWeightScales, BorrowedSource);
    push!(control_weights, ControlWeights, GatheredSource);
    push!(a_log, ALog, BorrowedSource);
    push!(dt_bias, DtBias, BorrowedSource);
    push!(convolution_weights, ConvolutionWeights, BorrowedSource);
    push!(recurrent_norm, RecurrentNorm, BorrowedSource);
    push!(output_weight_codes, OutputWeightCodes, BorrowedSource);
    push!(output_weight_scales, OutputWeightScales, BorrowedSource);
    push!(post_attention_norm, PostAttentionNorm, BorrowedSource);
}

fn push_attention_weights(
    entries: &mut Vec<ResidentUploadEntry>,
    layer: usize,
    weights: AttentionWeights,
) {
    let prefix = format!("layers.{layer}.attention");
    macro_rules! push {
        ($field:ident, $plane:ident, $preparation:ident) => {
            push_weight(
                entries,
                weights.$field,
                WeightSource::Attention {
                    layer,
                    plane: AttentionPlane::$plane,
                },
                ResidentUploadPreparation::$preparation,
                format!("{prefix}.{}", stringify!($field)),
            );
        };
    }
    push!(input_norm, InputNorm, BorrowedSource);
    push!(qkv_weight_codes, QkvWeightCodes, GatheredSource);
    push!(qkv_weight_scales, QkvWeightScales, GatheredSource);
    push!(query_norm, QueryNorm, BorrowedSource);
    push!(key_norm, KeyNorm, BorrowedSource);
    push!(output_weight_codes, OutputWeightCodes, BorrowedSource);
    push!(output_weight_scales, OutputWeightScales, BorrowedSource);
    push!(post_attention_norm, PostAttentionNorm, BorrowedSource);
}

fn push_mlp_weights(entries: &mut Vec<ResidentUploadEntry>, layer: usize, mlp: MlpWeights) {
    match mlp {
        MlpWeights::Nvfp4(weights) => push_nvfp4_mlp_weights(entries, layer, weights),
        MlpWeights::DenseFp8(weights) => push_dense_fp8_mlp_weights(entries, layer, weights),
    }
}

fn push_nvfp4_mlp_weights(
    entries: &mut Vec<ResidentUploadEntry>,
    layer: usize,
    weights: Nvfp4MlpWeights,
) {
    let prefix = format!("layers.{layer}.nvfp4_mlp");
    macro_rules! push {
        ($field:ident, $plane:ident, $preparation:ident) => {
            push_weight(
                entries,
                weights.$field,
                WeightSource::Nvfp4Mlp {
                    layer,
                    plane: Nvfp4MlpPlane::$plane,
                },
                ResidentUploadPreparation::$preparation,
                format!("{prefix}.{}", stringify!($field)),
            );
        };
    }
    push!(gate_weight_codes, GateWeightCodes, BorrowedSource);
    push!(up_weight_codes, UpWeightCodes, BorrowedSource);
    push!(gate_up_weight_scales, GateUpWeightScales, SwizzledSource);
    push!(down_weight_codes, DownWeightCodes, BorrowedSource);
    push!(down_weight_scales, DownWeightScales, SwizzledSource);
}

fn push_dense_fp8_mlp_weights(
    entries: &mut Vec<ResidentUploadEntry>,
    layer: usize,
    weights: DenseFp8MlpWeights,
) {
    let prefix = format!("layers.{layer}.dense_fp8_mlp");
    macro_rules! push {
        ($field:ident, $plane:ident) => {
            push_weight(
                entries,
                weights.$field,
                WeightSource::DenseFp8Mlp {
                    layer,
                    plane: DenseFp8MlpPlane::$plane,
                },
                ResidentUploadPreparation::BorrowedSource,
                format!("{prefix}.{}", stringify!($field)),
            );
        };
    }
    push!(gate_up_weight_codes, GateUpWeightCodes);
    push!(gate_up_weight_scales, GateUpWeightScales);
    push!(down_weight_codes, DownWeightCodes);
    push!(down_weight_scales, DownWeightScales);
}

fn push_endpoint_weights(entries: &mut Vec<ResidentUploadEntry>, endpoint: EndpointWeights) {
    macro_rules! push {
        ($field:ident, $plane:ident) => {
            push_weight(
                entries,
                endpoint.$field,
                WeightSource::Endpoint(EndpointPlane::$plane),
                ResidentUploadPreparation::BorrowedSource,
                concat!("endpoint.", stringify!($field)).into(),
            );
        };
    }
    push!(final_norm, FinalNorm);
    push!(lm_head_codes, LmHeadCodes);
    push!(lm_head_scales, LmHeadScales);
}

fn push_gdn_persistent(entries: &mut Vec<ResidentUploadEntry>, layer: usize, state: GdnPersistent) {
    push_zero(
        entries,
        ResidentUploadArena::Resident,
        state.history,
        format!("layers.{layer}.gdn.history"),
    );
    push_zero(
        entries,
        ResidentUploadArena::Resident,
        state.state,
        format!("layers.{layer}.gdn.state"),
    );
}

macro_rules! push_zero_workspace {
    ($entries:expr, $workspace:expr, $($field:ident),+ $(,)?) => {{
        $(push_zero(
            $entries,
            ResidentUploadArena::Resident,
            $workspace.$field,
            concat!("workspace.", stringify!($field)).into(),
        );)+
    }};
}

fn push_workspace(entries: &mut Vec<ResidentUploadEntry>, workspace: SharedWorkspace) {
    push_zero_workspace!(
        entries,
        workspace,
        residual_a,
        residual_b,
        mixer_residual,
        mixer_normalized,
        mlp_normalized,
        activation_codes,
        activation_scales,
        nvfp4_activation_codes,
        nvfp4_activation_scales,
        projected,
        log_decay,
        beta,
        convolved,
        recurrent_output,
        rope_cos,
        rope_sin,
        cache_positions,
        lengths,
        query,
        partial_maximum,
        partial_denominator,
        partial_numerator,
        attention,
        mixer_branch,
        swiglu,
        mlp_branch,
        logits,
    );
    push_metadata(
        entries,
        ResidentUploadArena::Resident,
        workspace.state_rows,
        MetadataSource::StateRows,
        "workspace.state_rows",
    );
    push_metadata(
        entries,
        ResidentUploadArena::Resident,
        workspace.table_rows,
        MetadataSource::TableRows,
        "workspace.table_rows",
    );
}

fn push_kv(entries: &mut Vec<ResidentUploadEntry>, layout: &ResidentModelLayout) {
    push_metadata(
        entries,
        ResidentUploadArena::Kv,
        layout.kv_layout.block_tables(),
        MetadataSource::KvBlockTables,
        "kv.block_tables",
    );
    for (layer, regions) in layout.kv_layout.layers().iter().enumerate() {
        push_zero(
            entries,
            ResidentUploadArena::Kv,
            regions.key.data,
            format!("kv.layers.{layer}.key_codes"),
        );
        push_zero(
            entries,
            ResidentUploadArena::Kv,
            regions.value.data,
            format!("kv.layers.{layer}.value_codes"),
        );
        if let Some(scales) = regions.key.scales {
            push_zero(
                entries,
                ResidentUploadArena::Kv,
                scales,
                format!("kv.layers.{layer}.key_scales"),
            );
        }
        if let Some(scales) = regions.value.scales {
            push_zero(
                entries,
                ResidentUploadArena::Kv,
                scales,
                format!("kv.layers.{layer}.value_scales"),
            );
        }
    }
}

fn push_weight<T: Copy>(
    entries: &mut Vec<ResidentUploadEntry>,
    region: ArenaRegion<T>,
    source: WeightSource,
    preparation: ResidentUploadPreparation,
    role: String,
) {
    push_entry(
        entries,
        ResidentUploadArena::Resident,
        region,
        UploadSource::Weight(source),
        preparation,
        role,
    );
}

fn push_metadata<T: Copy>(
    entries: &mut Vec<ResidentUploadEntry>,
    arena: ResidentUploadArena,
    region: ArenaRegion<T>,
    source: MetadataSource,
    role: &str,
) {
    push_entry(
        entries,
        arena,
        region,
        UploadSource::Metadata(source),
        ResidentUploadPreparation::HostDerived,
        role.into(),
    );
}

fn push_zero<T: Copy>(
    entries: &mut Vec<ResidentUploadEntry>,
    arena: ResidentUploadArena,
    region: ArenaRegion<T>,
    role: String,
) {
    push_entry(
        entries,
        arena,
        region,
        UploadSource::Zero,
        ResidentUploadPreparation::Zero,
        role,
    );
}

fn push_entry<T: Copy>(
    entries: &mut Vec<ResidentUploadEntry>,
    arena: ResidentUploadArena,
    region: ArenaRegion<T>,
    source: UploadSource,
    preparation: ResidentUploadPreparation,
    role: String,
) {
    entries.push(ResidentUploadEntry {
        arena,
        offset: region.offset_bytes(),
        bytes: region.byte_len(),
        preparation,
        role,
        source,
        padding: false,
    });
}

fn insert_padding(
    entries: Vec<ResidentUploadEntry>,
    layout: &ResidentModelLayout,
) -> EngineResult<Vec<ResidentUploadEntry>> {
    let mut completed = Vec::new();
    for (arena, arena_bytes) in [
        (ResidentUploadArena::Resident, layout.resident_arena_bytes()),
        (ResidentUploadArena::Kv, layout.kv_arena_bytes()),
    ] {
        let mut arena_entries = entries
            .iter()
            .filter(|entry| entry.arena == arena)
            .cloned()
            .collect::<Vec<_>>();
        arena_entries.sort_unstable_by_key(|entry| entry.offset);
        let mut cursor = 0usize;
        for entry in arena_entries {
            if entry.offset < cursor {
                return Err(EngineError::layout(format!(
                    "resident upload entry `{}` at {} overlaps an earlier destination ending at {cursor}",
                    entry.role, entry.offset
                )));
            }
            if cursor < entry.offset {
                completed.push(padding_entry(arena, cursor, entry.offset - cursor));
            }
            cursor = entry
                .offset
                .checked_add(entry.bytes)
                .ok_or_else(|| EngineError::layout("resident upload destination overflows"))?;
            if cursor > arena_bytes {
                return Err(EngineError::layout(format!(
                    "resident upload entry `{}` ends at {cursor}, beyond {arena_bytes}",
                    entry.role
                )));
            }
            completed.push(entry);
        }
        if cursor < arena_bytes {
            completed.push(padding_entry(arena, cursor, arena_bytes - cursor));
        }
    }
    completed.sort_unstable_by_key(|entry| (entry.arena, entry.offset));
    Ok(completed)
}

fn padding_entry(arena: ResidentUploadArena, offset: usize, bytes: usize) -> ResidentUploadEntry {
    ResidentUploadEntry {
        arena,
        offset,
        bytes,
        preparation: ResidentUploadPreparation::Zero,
        role: format!("{}.padding.{offset}", arena.as_str()),
        source: UploadSource::Zero,
        padding: true,
    }
}

impl ResidentUploadArena {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Kv => "kv",
        }
    }
}

fn validate_unique_sources(entries: &[ResidentUploadEntry]) -> EngineResult<()> {
    let mut weights = BTreeSet::new();
    let mut metadata = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for entry in entries {
        if !roles.insert(entry.role.as_str()) {
            return Err(EngineError::layout(format!(
                "resident upload role `{}` is duplicated",
                entry.role
            )));
        }
        match entry.source {
            UploadSource::Weight(source) if !weights.insert(source) => {
                return Err(EngineError::layout(format!(
                    "resident upload weight source {source:?} is duplicated"
                )));
            }
            UploadSource::Metadata(source) if !metadata.insert(source) => {
                return Err(EngineError::layout(format!(
                    "resident upload metadata source {source:?} is duplicated"
                )));
            }
            UploadSource::Weight(_) | UploadSource::Metadata(_) | UploadSource::Zero => {}
        }
    }
    Ok(())
}

fn validate_coverage(
    entries: &[ResidentUploadEntry],
    layout: &ResidentModelLayout,
) -> EngineResult<()> {
    for (arena, expected_bytes) in [
        (ResidentUploadArena::Resident, layout.resident_arena_bytes()),
        (ResidentUploadArena::Kv, layout.kv_arena_bytes()),
    ] {
        let mut cursor = 0usize;
        for entry in entries.iter().filter(|entry| entry.arena == arena) {
            if entry.offset != cursor {
                return Err(EngineError::layout(format!(
                    "resident upload {arena:?} coverage jumps from {cursor} to {}",
                    entry.offset
                )));
            }
            cursor = cursor
                .checked_add(entry.bytes)
                .ok_or_else(|| EngineError::layout("resident upload coverage overflows"))?;
        }
        require_equal("resident upload arena coverage", cursor, expected_bytes)?;
    }
    Ok(())
}

fn bytes_where(
    entries: &[ResidentUploadEntry],
    predicate: impl Fn(&ResidentUploadEntry) -> bool,
) -> EngineResult<usize> {
    entries
        .iter()
        .filter(|entry| predicate(entry))
        .try_fold(0usize, |total, entry| {
            total
                .checked_add(entry.bytes)
                .ok_or_else(|| EngineError::layout("resident upload byte sum overflows"))
        })
}

fn require_equal(name: &str, actual: usize, expected: usize) -> EngineResult<()> {
    if actual != expected {
        return Err(EngineError::layout(format!(
            "{name} are {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ResidentUploadArena, ResidentUploadPlan, ResidentUploadPreparation};
    use crate::ResidentModelLayout;

    #[test]
    fn exact_upload_inventory_accounts_for_every_destination_byte() {
        let layout = ResidentModelLayout::build().unwrap();
        let plan = ResidentUploadPlan::build(&layout).unwrap();

        assert_eq!(plan.weight_bytes(), 19_103_682_560);
        assert_eq!(plan.host_derived_bytes(), 110_080);
        assert_eq!(plan.zeroed_owner_bytes(), 8_617_875_040);
        assert_eq!(plan.padding_bytes(), 16_544);
        assert_eq!(plan.initialized_bytes(), layout.arena_bytes());
    }

    #[test]
    fn weight_preparation_inventory_is_exact() {
        let layout = ResidentModelLayout::build().unwrap();
        let plan = ResidentUploadPlan::build(&layout).unwrap();
        let count = |preparation| {
            plan.entries()
                .iter()
                .filter(|entry| entry.preparation() == preparation && !entry.is_padding())
                .count()
        };

        assert_eq!(count(ResidentUploadPreparation::BorrowedSource), 779);
        assert_eq!(count(ResidentUploadPreparation::GatheredSource), 80);
        assert_eq!(count(ResidentUploadPreparation::SwizzledSource), 112);
        assert_eq!(count(ResidentUploadPreparation::HostDerived), 3);
        assert_eq!(count(ResidentUploadPreparation::Zero), 48 * 2 + 27 + 16 * 2);
    }

    #[test]
    fn entries_are_contiguous_and_non_overlapping_in_both_arenas() {
        let layout = ResidentModelLayout::build().unwrap();
        let plan = ResidentUploadPlan::build(&layout).unwrap();
        for (arena, expected) in [
            (ResidentUploadArena::Resident, layout.resident_arena_bytes()),
            (ResidentUploadArena::Kv, layout.kv_arena_bytes()),
        ] {
            let mut cursor = 0;
            for entry in plan.entries().iter().filter(|entry| entry.arena() == arena) {
                assert_eq!(entry.offset_bytes(), cursor, "{}", entry.role());
                cursor += entry.byte_len();
            }
            assert_eq!(cursor, expected);
        }
    }
}
