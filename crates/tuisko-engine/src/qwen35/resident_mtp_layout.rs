//! Aggregate ownership for the exact Qwen3.5 target and MTP resident programs.

use crate::qwen35::mtp_kv_layout::Qwen35MtpKvLayout;
use crate::{
    EngineError, EngineResult, LayerMemoryLayout, Qwen35MtpLayerLayout, Qwen35ResidentModelLayout,
};

/// Exact target, MTP weight, cache-mirror, and workspace byte accounting.
#[derive(Clone, Debug)]
pub struct Qwen35ResidentMtpLayout {
    target: Qwen35ResidentModelLayout,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
    arena_bytes: usize,
}

impl Qwen35ResidentMtpLayout {
    /// Accounts the shared target endpoint and separate MTP BF16 cache mirror.
    pub fn build() -> EngineResult<Self> {
        let target = Qwen35ResidentModelLayout::build()?;
        let mtp = Qwen35MtpLayerLayout::build_for_external_cache()?;
        let cache = Qwen35MtpKvLayout::build()?;
        let resident_weight_bytes = sum(
            "Qwen3.5 resident MTP weight bytes",
            target.resident_weight_bytes(),
            mtp.resident_weight_bytes(),
        )?;
        let cache_bytes = sum(
            "Qwen3.5 resident MTP cache bytes",
            target.cache_bytes(),
            cache.cache_bytes(),
        )?;
        let workspace_bytes = target
            .workspace_bytes()
            .checked_add(mtp.workspace_bytes())
            .and_then(|bytes| bytes.checked_add(cache.block_table_bytes()))
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident MTP workspace bytes overflow"))?;
        let arena_bytes = target
            .arena_bytes()
            .checked_add(mtp.arena_bytes())
            .and_then(|bytes| bytes.checked_add(cache.arena_bytes()))
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident MTP arena bytes overflow"))?;

        Ok(Self {
            target,
            resident_weight_bytes,
            cache_bytes,
            workspace_bytes,
            arena_bytes,
        })
    }

    /// Complete target-model layout shared by generation.
    pub const fn target(&self) -> &Qwen35ResidentModelLayout {
        &self.target
    }

    /// Target plus source-BF16 MTP weight bytes, with one shared endpoint.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Target cache plus the separate MTP BF16 mirror.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable target/MTP workspaces and MTP block tables.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Complete device allocation across all stable arenas.
    pub const fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }

    /// Represented and typed bytes excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Aggregate alignment padding.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes - self.owner_bytes()
    }

    /// Maximum context admitted by both page-table owners.
    pub const fn context_capacity(&self) -> usize {
        self.target.context_capacity()
    }
}

impl LayerMemoryLayout for Qwen35ResidentMtpLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes()
    }

    fn cache_bytes(&self) -> usize {
        self.cache_bytes()
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes()
    }
}

fn sum(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::Qwen35ResidentMtpLayout;

    #[test]
    fn qwen35_resident_mtp_accounting_is_self_consistent() {
        let layout = Qwen35ResidentMtpLayout::build().unwrap();

        assert_eq!(layout.context_capacity(), 262_144);
        assert_eq!(
            layout.arena_bytes(),
            layout.owner_bytes() + layout.padding_bytes()
        );
        assert_eq!(
            layout.resident_weight_bytes() - layout.target().resident_weight_bytes(),
            486_581_248
        );
        assert_eq!(
            layout.cache_bytes() - layout.target().cache_bytes(),
            1_073_741_824
        );
    }
}
