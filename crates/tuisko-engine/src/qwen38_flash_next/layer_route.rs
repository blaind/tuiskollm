//! Exact row admission and the dense-QSA visible-length ceiling.
//!
//! Composed layers admit only `B=1..8` and `T=32/64/128/1024`. Dense QSA matches selected
//! attention through 2,051 visible keys; larger requests are refused rather than truncated.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_model::Qwen38FlashNext;

/// Prefill tile widths every Qwen3.8-Flash-Next kernel family admits, ascending.
pub const QWEN38_FLASH_NEXT_PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

/// Largest row count any admitted Qwen3.8-Flash-Next route carries.
pub const QWEN38_FLASH_NEXT_MAX_ROWS: usize = 1_024;

/// Largest visible key count for which dense QSA is the reference's own function.
///
/// At 2,052 visible keys, 513 four-token blocks exceed the indexer's 512-block budget.
pub const QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING: usize =
    Qwen38FlashNext::INDEXER_BUDGET + Qwen38FlashNext::INDEXER_COMPRESS_RATIO - 1;

const _: () = assert!(QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING == 2_051);

/// Which admitted graph a row count selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextRowRoute {
    /// Exact decode batch, `1..=MAX_BATCH`.
    Decode(usize),
    /// Exact prefill tile, one of [`QWEN38_FLASH_NEXT_PREFILL_ROWS`].
    Prefill(usize),
}

impl Qwen38FlashNextRowRoute {
    /// Rows this route carries.
    pub const fn rows(self) -> usize {
        match self {
            Self::Decode(rows) | Self::Prefill(rows) => rows,
        }
    }

    /// Index of this route inside its own captured-graph array.
    ///
    /// Decode graphs are stored densely by batch, prefill graphs by position in
    /// [`QWEN38_FLASH_NEXT_PREFILL_ROWS`], so a composed owner can index its two arrays without a
    /// second search.
    pub const fn graph_index(self) -> usize {
        match self {
            Self::Decode(rows) => rows - 1,
            Self::Prefill(rows) => match rows {
                32 => 0,
                64 => 1,
                128 => 2,
                _ => 3,
            },
        }
    }
}

/// Resolves one admitted row count, refusing every width no graph was captured for.
pub fn qwen38_flash_next_row_route(rows: usize) -> EngineResult<Qwen38FlashNextRowRoute> {
    if (1..=MAX_BATCH).contains(&rows) {
        return Ok(Qwen38FlashNextRowRoute::Decode(rows));
    }
    if QWEN38_FLASH_NEXT_PREFILL_ROWS.contains(&rows) {
        return Ok(Qwen38FlashNextRowRoute::Prefill(rows));
    }

    Err(EngineError::route(format!(
        "Qwen3.8-Flash-Next row count {rows} is not an admitted B=1..={MAX_BATCH} or T=32/64/128/1024 route"
    )))
}

/// Admits one QSA layer request, refusing any visible length dense attention cannot answer.
///
/// `visible` is the total count of keys the *last* query in this round may attend: prompt plus
/// everything already generated, including the token being produced. A round is admitted only if
/// its widest query stays inside the proven band, because the mask is shared by every query in
/// the round.
///
/// This refuses; it never truncates. The message names the ceiling and the offending length so a
/// caller can surface a real limit rather than a mysterious short answer.
pub fn require_qwen38_flash_next_dense_qsa_visible(visible: usize) -> EngineResult<()> {
    if visible == 0 {
        return Err(EngineError::route(
            "Qwen3.8-Flash-Next QSA visible length must include at least the current token",
        ));
    }
    if visible > QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
        return Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next QSA visible length {visible} exceeds the dense-equivalent ceiling \
             {QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING}; above it the reference's indexer drops at \
             least one four-token block, so a dense route is not numerically admissible and the \
             request is refused rather than truncated or silently run dense"
        )));
    }

    Ok(())
}

/// Admits a whole QSA prefill tile by its last query's visible length.
///
/// A tile of `rows` tokens ending at absolute position `last_position` (zero-based) makes its
/// final query see `last_position + 1` keys. Callers hold positions, not visible counts, so this
/// converts once and keeps the arithmetic in one place.
pub fn require_qwen38_flash_next_dense_qsa_round(
    rows: usize,
    last_position: usize,
) -> EngineResult<()> {
    let route = qwen38_flash_next_row_route(rows)?;
    let visible = last_position.checked_add(1).ok_or_else(|| {
        EngineError::route(
            "Qwen3.8-Flash-Next QSA position overflows its visible-length conversion",
        )
    })?;
    require_qwen38_flash_next_dense_qsa_visible(visible)?;
    let _ = route;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, QWEN38_FLASH_NEXT_MAX_ROWS,
        QWEN38_FLASH_NEXT_PREFILL_ROWS, Qwen38FlashNextRowRoute, qwen38_flash_next_row_route,
        require_qwen38_flash_next_dense_qsa_round, require_qwen38_flash_next_dense_qsa_visible,
    };
    use crate::{EngineErrorCode, MAX_BATCH};

    #[test]
    fn every_admitted_row_count_resolves_to_a_distinct_graph_slot() {
        let mut decode = Vec::new();
        for rows in 1..=MAX_BATCH {
            let route = qwen38_flash_next_row_route(rows).unwrap();
            assert_eq!(route, Qwen38FlashNextRowRoute::Decode(rows));
            assert_eq!(route.rows(), rows);
            decode.push(route.graph_index());
        }
        assert_eq!(decode, (0..MAX_BATCH).collect::<Vec<_>>());

        let mut prefill = Vec::new();
        for rows in QWEN38_FLASH_NEXT_PREFILL_ROWS {
            let route = qwen38_flash_next_row_route(rows).unwrap();
            assert_eq!(route, Qwen38FlashNextRowRoute::Prefill(rows));
            prefill.push(route.graph_index());
        }
        assert_eq!(prefill, vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_admitted_route_table_is_exactly_twelve_wide() {
        let admitted = (0..=2_048)
            .filter(|rows| qwen38_flash_next_row_route(*rows).is_ok())
            .collect::<Vec<_>>();

        assert_eq!(admitted, vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(admitted.len(), 12);
        assert_eq!(*admitted.last().unwrap(), QWEN38_FLASH_NEXT_MAX_ROWS);
    }

    #[test]
    fn unadmitted_row_counts_are_refused_by_route_code() {
        // The neighbours of every boundary, plus zero and one past the widest tile.
        for rows in [0, 9, 31, 33, 63, 65, 127, 129, 1_023, 1_025] {
            let error = qwen38_flash_next_row_route(rows).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
            assert!(error.to_string().contains(&rows.to_string()));
        }
    }

    #[test]
    fn the_dense_ceiling_is_sharp_at_2051() {
        assert_eq!(QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, 2_051);

        // Every visible count the proof covers is admitted, including the exact boundary.
        for visible in [1, 4, 2_048, 2_050, 2_051] {
            require_qwen38_flash_next_dense_qsa_visible(visible).unwrap();
        }
        // 2,052 is where n_blocks reaches 513 against a 512 budget.
        for visible in [2_052, 2_053, 4_096, 262_144] {
            let error = require_qwen38_flash_next_dense_qsa_visible(visible).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn a_refused_length_is_never_silently_truncated() {
        let error = require_qwen38_flash_next_dense_qsa_visible(2_052).unwrap_err();
        let message = error.to_string();

        // The refusal names both numbers and says what it refuses to do instead.
        assert!(message.contains("2052"));
        assert!(message.contains("2051"));
        assert!(message.contains("refused rather than truncated"));
    }

    #[test]
    fn a_zero_length_round_is_refused() {
        let error = require_qwen38_flash_next_dense_qsa_visible(0).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Route));
    }

    #[test]
    fn a_round_is_admitted_by_its_widest_query() {
        // A T=1024 tile whose last token sits at position 2,050 sees 2,051 keys: admitted.
        require_qwen38_flash_next_dense_qsa_round(1_024, 2_050).unwrap();
        // One token later the same tile would need selection.
        assert!(require_qwen38_flash_next_dense_qsa_round(1_024, 2_051).is_err());
        // The row count is admitted first, so an unadmitted width fails on its own terms.
        let error = require_qwen38_flash_next_dense_qsa_round(1_000, 0).unwrap_err();
        assert!(error.to_string().contains("not an admitted"));
    }
}
