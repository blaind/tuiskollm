use crate::residual_norm::residual_norm_ptx_names;

/// Stable semantic inventory of every admitted SM89 entry.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    residual_norm_ptx_names().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::kernel_ptx_names;
    use std::collections::BTreeSet;

    #[test]
    fn inventory_has_no_missing_or_duplicate_entries() {
        let names = kernel_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 16);
        assert_eq!(unique.len(), names.len());
    }
}
