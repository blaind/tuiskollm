//! Typed logit and hidden bank range slicing shared by every compact generator.

use crate::MAX_BATCH;
use std::ops::Range;

/// Contiguous `columns`-wide row `index` of one typed bank.
pub(crate) fn row(index: usize, columns: usize) -> Range<usize> {
    let begin = index * columns;
    begin..begin + columns
}

/// Compact `rows`-row span that follows the bank's eight per-slot rows.
pub(crate) fn compact(rows: usize, columns: usize) -> Range<usize> {
    let begin = MAX_BATCH * columns;
    begin..begin + rows * columns
}
