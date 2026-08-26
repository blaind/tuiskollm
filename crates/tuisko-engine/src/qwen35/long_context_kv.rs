//! Address-stable device and host ownership for Qwen3.5 BF16 KV pages.

use crate::Qwen35LongContextKvLayout;
use crate::common::paged_kv::{PagedKvBinding, PagedKvCacheStorage};

pub(crate) type Qwen35AttentionKvBinding = PagedKvBinding;

/// Fixed shared BF16 KV allocation and its allocation-free page lifecycle.
pub type Qwen35LongContextKvProgram = PagedKvCacheStorage<Qwen35LongContextKvLayout>;

#[cfg(test)]
mod tests {
    use crate::{QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS};

    #[test]
    fn exact_context_and_page_inventory_are_consistent() {
        assert_eq!(QWEN35_MAX_CONTEXT_TOKENS, 262_144);
        assert_eq!(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES * 64, 262_144);
    }
}
