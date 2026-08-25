//! Address-stable device and host ownership for Qwen3.6 E4M3 KV pages.

use crate::Qwen36LongContextKvLayout;
use crate::common::paged_kv::{PagedKvBinding, PagedKvCacheStorage};

pub(crate) type Qwen36AttentionKvBinding = PagedKvBinding;

/// Fixed shared E4M3 KV allocation and its allocation-free page lifecycle.
pub type Qwen36LongContextKvProgram = PagedKvCacheStorage<Qwen36LongContextKvLayout>;

#[cfg(test)]
mod tests {
    use crate::{QWEN36_LONG_CONTEXT_PHYSICAL_PAGES, QWEN36_MAX_CONTEXT_TOKENS};

    #[test]
    fn exact_context_and_page_inventory_are_consistent() {
        assert_eq!(QWEN36_MAX_CONTEXT_TOKENS, 262_144);
        assert_eq!(QWEN36_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(QWEN36_LONG_CONTEXT_PHYSICAL_PAGES * 64, 262_144);
    }
}
