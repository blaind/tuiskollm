//! Address-stable Qwen3.5 MTP cache storage with mirrored page ownership.

use crate::common::paged_kv::PagedKvCacheStorage;
use crate::qwen35::mtp_kv_layout::Qwen35MtpKvLayout;

/// Separate BF16 MTP cache whose logical page ownership mirrors the target.
pub(crate) type Qwen35MtpKvProgram = PagedKvCacheStorage<Qwen35MtpKvLayout>;

#[cfg(test)]
mod tests {
    use crate::MAX_BATCH;

    #[test]
    fn qwen35_mtp_mirror_has_eight_stable_rows() {
        assert_eq!(MAX_BATCH, 8);
    }
}
