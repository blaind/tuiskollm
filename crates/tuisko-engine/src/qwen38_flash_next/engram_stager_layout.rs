//! Address-stable Qwen3.8-Flash-Next engram staging layout.
//!
//! The mapped 47.68 GiB table is gathered per step. The layout owns one pinned
//! stager and one stable device plane sized for the widest admitted round.

use crate::common::math::product;
use crate::{EngineError, EngineResult, LayerMemoryLayout, StreamingResidencyAccounting};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN, Qwen38FlashNext};

type F = Qwen38FlashNext;

const ALIGNMENT: usize = 256;

/// Token counts one exact staging round admits.
///
/// `T=1` is decode; the remaining values are prefill tiles. Other widths are
/// refused because padding would hash tokens outside the sequence.
pub const QWEN38_FLASH_NEXT_ENGRAM_WIDTHS: [usize; 5] = [1, 32, 64, 128, 1_024];

/// Checked regions and byte counts for Qwen3.8-Flash-Next engram staging.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextEngramStagerLayout {
    builder: ArenaLayout,
    embedding: ArenaRegion<u8>,
    max_tokens: usize,
    token_bytes: usize,
    stager_bytes: usize,
    row_scratch_bytes: usize,
    table_bytes: usize,
}

impl Qwen38FlashNextEngramStagerLayout {
    /// Plans the widest admitted round's stager, device plane, and row scratch.
    pub fn build() -> EngineResult<Self> {
        let max_tokens = max_engram_tokens();
        let token_bytes = product(
            "Flash-Next engram token bytes",
            F::NGRAM_HEADS,
            F::NGRAM_HEAD_DIM,
        )?;

        if token_bytes != F::PLE_EMBED_DIM {
            return Err(EngineError::layout(format!(
                "Flash-Next engram stages {token_bytes} bytes per token but the PLE embedding is {} wide",
                F::PLE_EMBED_DIM
            )));
        }

        let plane_bytes = product("Flash-Next engram plane", max_tokens, token_bytes)?;
        let row_scratch_bytes = product(
            "Flash-Next engram row scratch",
            product(
                "Flash-Next engram round rows",
                max_tokens,
                QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN,
            )?,
            size_of::<i64>(),
        )?;
        let table_bytes = product(
            "Flash-Next engram table",
            product(
                "Flash-Next engram table rows",
                F::NGRAM_SHARDS,
                F::NGRAM_SHARD_ROWS,
            )?,
            F::NGRAM_HEAD_DIM,
        )?;
        let mut builder = ArenaLayout::new();
        let embedding = builder.reserve(plane_bytes, ALIGNMENT)?;

        Ok(Self {
            builder,
            embedding,
            max_tokens,
            token_bytes,
            stager_bytes: plane_bytes,
            row_scratch_bytes,
            table_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    /// Tokens in the widest admitted round.
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Bytes one token contributes to the staged plane: one FP8 row per engram head.
    pub const fn token_bytes(&self) -> usize {
        self.token_bytes
    }

    /// Page-locked bytes the host gather writes into.
    pub const fn stager_bytes(&self) -> usize {
        self.stager_bytes
    }

    /// Device bytes of the stable engram plane, excluding alignment padding.
    pub const fn plane_bytes(&self) -> usize {
        self.embedding.byte_len()
    }

    /// Host heap bytes holding one round's row indices; neither pinned nor device-visible.
    pub const fn row_scratch_bytes(&self) -> usize {
        self.row_scratch_bytes
    }

    /// File-backed bytes of the engram table, mapped and gathered rather than held.
    pub const fn table_bytes(&self) -> usize {
        self.table_bytes
    }

    /// Allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Staged bytes one round of `tokens` occupies.
    pub fn round_bytes(&self, tokens: usize) -> EngineResult<usize> {
        require_qwen38_flash_next_engram_width(tokens)?;
        product("Flash-Next engram round bytes", tokens, self.token_bytes)
    }

    /// Row indices one round of `tokens` addresses.
    pub fn round_rows(&self, tokens: usize) -> EngineResult<usize> {
        require_qwen38_flash_next_engram_width(tokens)?;
        product(
            "Flash-Next engram round rows",
            tokens,
            QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN,
        )
    }

    pub(crate) const fn embedding(&self) -> ArenaRegion<u8> {
        self.embedding
    }
}

impl LayerMemoryLayout for Qwen38FlashNextEngramStagerLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    // The engram's weights are the mapped table; nothing this tier owns is a resident weight.
    fn resident_weight_bytes(&self) -> usize {
        0
    }

    fn cache_bytes(&self) -> usize {
        0
    }

    fn workspace_bytes(&self) -> usize {
        self.plane_bytes()
    }
}

impl StreamingResidencyAccounting for Qwen38FlashNextEngramStagerLayout {
    fn device_resident_bytes(&self) -> usize {
        self.arena_bytes()
    }

    fn host_pinned_bytes(&self) -> usize {
        self.stager_bytes
    }

    fn host_mapped_bytes(&self) -> usize {
        self.table_bytes
    }
}

/// Refuses a round width that has no admitted route.
pub fn require_qwen38_flash_next_engram_width(tokens: usize) -> EngineResult<()> {
    if !QWEN38_FLASH_NEXT_ENGRAM_WIDTHS.contains(&tokens) {
        return Err(EngineError::route(format!(
            "Flash-Next engram round of {tokens} tokens is not an admitted T=1/32/64/128/1024 route"
        )));
    }

    Ok(())
}

const fn max_engram_tokens() -> usize {
    let mut widest = 0;
    let mut index = 0;

    while index < QWEN38_FLASH_NEXT_ENGRAM_WIDTHS.len() {
        if QWEN38_FLASH_NEXT_ENGRAM_WIDTHS[index] > widest {
            widest = QWEN38_FLASH_NEXT_ENGRAM_WIDTHS[index];
        }

        index += 1;
    }

    widest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineErrorCode;

    #[test]
    fn engram_staging_byte_accounting_is_exact() {
        let layout = Qwen38FlashNextEngramStagerLayout::build().unwrap();

        assert_eq!(layout.max_tokens(), 1_024);
        assert_eq!(layout.token_bytes(), 2_560);
        assert_eq!(layout.plane_bytes(), 2_621_440);
        assert_eq!(layout.stager_bytes(), 2_621_440);
        assert_eq!(layout.row_scratch_bytes(), 131_072);
        // 128 shards of 2,500,012 rows by 160 FP8 bytes: the whole 51,200,245,760-byte table.
        assert_eq!(layout.table_bytes(), 51_200_245_760);
        assert_eq!(layout.arena_bytes(), 2_621_440);
    }

    #[test]
    fn the_three_residency_classes_are_reported_separately() {
        let layout = Qwen38FlashNextEngramStagerLayout::build().unwrap();

        assert_eq!(layout.device_resident_bytes(), 2_621_440);
        assert_eq!(layout.host_pinned_bytes(), 2_621_440);
        assert_eq!(layout.host_mapped_bytes(), 51_200_245_760);

        // The table remains mapped because it dwarfs both held classes.
        assert!(layout.host_mapped_bytes() > 10_000 * layout.host_pinned_bytes());
        assert_eq!(layout.resident_weight_bytes(), 0);
        assert_eq!(layout.workspace_bytes(), layout.plane_bytes());
        assert_eq!(layout.cache_bytes(), 0);
    }

    #[test]
    fn only_the_exact_route_widths_are_admitted() {
        let layout = Qwen38FlashNextEngramStagerLayout::build().unwrap();

        for rows in QWEN38_FLASH_NEXT_ENGRAM_WIDTHS {
            assert_eq!(layout.round_bytes(rows).unwrap(), rows * 2_560);
            assert_eq!(layout.round_rows(rows).unwrap(), rows * 16);
            assert!(layout.round_bytes(rows).unwrap() <= layout.stager_bytes());
        }

        for rows in [0, 2, 8, 31, 33, 127, 129, 512, 1_023, 1_025, usize::MAX] {
            let error = require_qwen38_flash_next_engram_width(rows).err().unwrap();

            assert_eq!(error.code(), Some(EngineErrorCode::Route));
            assert!(error.to_string().contains("not an admitted"), "{error}");
        }
    }

    #[test]
    fn the_plane_is_aligned_and_fills_its_arena() {
        let layout = Qwen38FlashNextEngramStagerLayout::build().unwrap();

        assert_eq!(layout.embedding().offset_bytes() % ALIGNMENT, 0);
        assert_eq!(
            layout.embedding().offset_bytes() + layout.embedding().byte_len(),
            layout.arena_bytes()
        );
    }
}
