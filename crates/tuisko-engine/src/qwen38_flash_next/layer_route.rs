//! Exact row admission and QSA route selection.
//!
//! Composed layers admit only `B=1..8` and `T=32/64/128/1024`. Dense QSA matches selected
//! attention through 2,051 visible keys; longer requests take the selected route.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_model::Qwen38FlashNext;

/// Prefill tile widths every Qwen3.8-Flash-Next kernel family admits, ascending.
pub const QWEN38_FLASH_NEXT_PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

/// Consecutive rows one speculative verification round admits.
pub const QWEN38_FLASH_NEXT_CAUSAL_ROWS: [usize; 4] = [1, 2, 3, 4];

/// Largest row count any admitted Qwen3.8-Flash-Next route carries.
pub const QWEN38_FLASH_NEXT_MAX_ROWS: usize = 1_024;

/// Largest visible key count for which dense QSA is the reference's own function.
///
/// At 2,052 visible keys, 513 four-token blocks exceed the indexer's 512-block budget.
pub const QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING: usize =
    Qwen38FlashNext::INDEXER_BUDGET + Qwen38FlashNext::INDEXER_COMPRESS_RATIO - 1;

/// Largest visible key count any QSA route serves.
pub const QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING: usize = Qwen38FlashNext::MAX_POSITION_EMBEDDINGS;

const _: () = assert!(QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING == 2_051);
const _: () = assert!(QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING == 262_144);

/// Which admitted graph a row count selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextRowRoute {
    /// Exact decode batch, `1..=MAX_BATCH`.
    Decode(usize),
    /// Exact prefill tile, one of [`QWEN38_FLASH_NEXT_PREFILL_ROWS`].
    Prefill(usize),
}

/// Whether rows are independent sequences or consecutive positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextRoundShape {
    /// Independent one-token sequences.
    Batch,
    /// Consecutive positions of one sequence.
    Causal,
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

/// Resolves one admitted causal verification span.
pub fn qwen38_flash_next_causal_route(rows: usize) -> EngineResult<Qwen38FlashNextRowRoute> {
    if QWEN38_FLASH_NEXT_CAUSAL_ROWS.contains(&rows) {
        return Ok(Qwen38FlashNextRowRoute::Decode(rows));
    }

    Err(EngineError::route(format!(
        "Qwen3.8-Flash-Next causal round {rows} is outside the admitted K=1..=4 span"
    )))
}

/// Attention route that computes the reference function at a visible length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextQsaRoute {
    /// Plain causal GQA inside the identity-selection band.
    Dense,
    /// Block selection followed by gathered attention.
    Selected,
}

impl Qwen38FlashNextQsaRoute {
    /// Whether the route runs selection and gathered attention.
    pub const fn selective(self) -> bool {
        matches!(self, Self::Selected)
    }

    /// Widens a shared batch route if either input needs selection.
    pub const fn widen(self, other: Self) -> Self {
        match (self, other) {
            (Self::Dense, Self::Dense) => Self::Dense,
            _ => Self::Selected,
        }
    }
}

/// Classifies one QSA layer request by visible key count.
pub fn qwen38_flash_next_qsa_route(visible: usize) -> EngineResult<Qwen38FlashNextQsaRoute> {
    if visible == 0 {
        return Err(EngineError::route(
            "Qwen3.8-Flash-Next QSA visible length must include at least the current token",
        ));
    }
    if visible > QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING {
        return Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next QSA visible length {visible} exceeds the checkpoint's \
             {QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING}-position ceiling and is refused rather than \
             truncated"
        )));
    }
    if visible > QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
        return Ok(Qwen38FlashNextQsaRoute::Selected);
    }

    Ok(Qwen38FlashNextQsaRoute::Dense)
}

/// Classifies a whole QSA tile by its last query's visible length.
pub fn qwen38_flash_next_qsa_round_route(
    rows: usize,
    last_position: usize,
) -> EngineResult<Qwen38FlashNextQsaRoute> {
    qwen38_flash_next_row_route(rows)?;
    let visible = last_position.checked_add(1).ok_or_else(|| {
        EngineError::route(
            "Qwen3.8-Flash-Next QSA position overflows its visible-length conversion",
        )
    })?;

    qwen38_flash_next_qsa_route(visible)
}

/// Admits a dense-only owner while selection wiring remains outside its scope.
pub fn require_qwen38_flash_next_dense_qsa_visible(visible: usize) -> EngineResult<()> {
    match qwen38_flash_next_qsa_route(visible)? {
        Qwen38FlashNextQsaRoute::Dense => Ok(()),
        Qwen38FlashNextQsaRoute::Selected => Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next QSA visible length {visible} requires the selected route"
        ))),
    }
}

/// Admits one dense-only tile by its last query position.
pub fn require_qwen38_flash_next_dense_qsa_round(
    rows: usize,
    last_position: usize,
) -> EngineResult<()> {
    match qwen38_flash_next_qsa_round_route(rows, last_position)? {
        Qwen38FlashNextQsaRoute::Dense => Ok(()),
        Qwen38FlashNextQsaRoute::Selected => Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next QSA position {last_position} requires the selected route"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN38_FLASH_NEXT_CAUSAL_ROWS, QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING,
        QWEN38_FLASH_NEXT_MAX_ROWS, QWEN38_FLASH_NEXT_PREFILL_ROWS,
        QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING, Qwen38FlashNextQsaRoute, Qwen38FlashNextRowRoute,
        qwen38_flash_next_causal_route, qwen38_flash_next_qsa_round_route,
        qwen38_flash_next_qsa_route, qwen38_flash_next_row_route,
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
    fn causal_routes_are_exactly_the_verification_span() {
        let admitted = (0..=8)
            .filter(|rows| qwen38_flash_next_causal_route(*rows).is_ok())
            .collect::<Vec<_>>();

        assert_eq!(admitted, QWEN38_FLASH_NEXT_CAUSAL_ROWS);
        for rows in QWEN38_FLASH_NEXT_CAUSAL_ROWS {
            assert_eq!(
                qwen38_flash_next_causal_route(rows).unwrap(),
                Qwen38FlashNextRowRoute::Decode(rows)
            );
        }
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

        // The identity band stays dense, including the exact boundary.
        for visible in [1, 4, 2_048, 2_050, 2_051] {
            assert_eq!(
                qwen38_flash_next_qsa_route(visible).unwrap(),
                Qwen38FlashNextQsaRoute::Dense
            );
        }
        // Selection starts when 513 blocks exceed the 512-block budget.
        for visible in [2_052, 2_053, 4_096, 262_144] {
            assert_eq!(
                qwen38_flash_next_qsa_route(visible).unwrap(),
                Qwen38FlashNextQsaRoute::Selected
            );
        }
    }

    #[test]
    fn the_position_ceiling_is_refused_without_truncation() {
        assert_eq!(QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING, 262_144);
        let error =
            qwen38_flash_next_qsa_route(QWEN38_FLASH_NEXT_QSA_VISIBLE_CEILING + 1).unwrap_err();
        let message = error.to_string();

        assert_eq!(error.code(), Some(EngineErrorCode::Route));
        assert!(message.contains("262145"));
        assert!(message.contains("262144"));
        assert!(message.contains("refused rather than truncated"));
    }

    #[test]
    fn a_zero_length_round_is_refused() {
        let error = qwen38_flash_next_qsa_route(0).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Route));
    }

    #[test]
    fn a_round_takes_the_route_its_widest_query_needs() {
        assert_eq!(
            qwen38_flash_next_qsa_round_route(1_024, 2_050).unwrap(),
            Qwen38FlashNextQsaRoute::Dense
        );
        assert_eq!(
            qwen38_flash_next_qsa_round_route(1_024, 2_051).unwrap(),
            Qwen38FlashNextQsaRoute::Selected
        );
        let error = qwen38_flash_next_qsa_round_route(1_000, 0).unwrap_err();
        assert!(error.to_string().contains("not an admitted"));
    }

    #[test]
    fn a_mixed_batch_widens_to_selection() {
        let route = [1_usize, 2_051, 2_052, 64]
            .into_iter()
            .try_fold(Qwen38FlashNextQsaRoute::Dense, |widest, visible| {
                qwen38_flash_next_qsa_route(visible).map(|route| widest.widen(route))
            })
            .unwrap();

        assert_eq!(route, Qwen38FlashNextQsaRoute::Selected);
        assert!(route.selective());
        assert!(!Qwen38FlashNextQsaRoute::Dense.selective());
    }
}
