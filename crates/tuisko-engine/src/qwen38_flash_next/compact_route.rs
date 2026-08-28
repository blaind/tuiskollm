//! Plain-value policy for mapping stable physical slots to dense decode rows.
//!
//! Survivors retain admission order and physical slots. Each nonempty round selects its exact
//! `B=1..8` graph without padding.

use crate::qwen38_flash_next::layer_route::{Qwen38FlashNextRowRoute, qwen38_flash_next_row_route};
use crate::{EngineError, EngineResult, MAX_BATCH};

/// One compact decode round: `rows` rows, where row `r` drives physical slot `slots[r]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextCompactRound {
    slots: [usize; MAX_BATCH],
    rows: usize,
}

impl Qwen38FlashNextCompactRound {
    /// Empty round used when no request has a pending replay.
    pub const EMPTY: Self = Self {
        slots: [usize::MAX; MAX_BATCH],
        rows: 0,
    };

    /// Rows this round carries.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Whether this round has any device work to do.
    pub const fn is_empty(self) -> bool {
        self.rows == 0
    }

    /// Physical slots in row order.
    pub fn slots(&self) -> &[usize] {
        &self.slots[..self.rows]
    }

    /// Row that drives one physical slot, if this round drives it at all.
    pub fn row_of(&self, slot: usize) -> Option<usize> {
        self.slots().iter().position(|&owned| owned == slot)
    }

    /// Captured graph selected by this width, or none for an empty round.
    pub fn route(self) -> Option<EngineResult<Qwen38FlashNextRowRoute>> {
        (self.rows > 0).then(|| qwen38_flash_next_row_route(self.rows))
    }
}

/// Packs pending active slots into a dense round and rejects invalid or duplicate slots.
pub fn qwen38_flash_next_compact_round(
    active: &[usize],
    pending: &[bool],
) -> EngineResult<Qwen38FlashNextCompactRound> {
    if active.len() != pending.len() {
        return Err(EngineError::route(format!(
            "a Flash-Next compact round was given {} active slots and {} pending flags",
            active.len(),
            pending.len()
        )));
    }
    if active.len() > MAX_BATCH {
        return Err(EngineError::route(format!(
            "a Flash-Next compact round names {} active slots, more than the {MAX_BATCH} funded",
            active.len()
        )));
    }
    let mut seen = [false; MAX_BATCH];
    let mut round = Qwen38FlashNextCompactRound::EMPTY;
    for (&slot, &pending) in active.iter().zip(pending) {
        let occupied = seen.get_mut(slot).ok_or_else(|| {
            EngineError::route(format!(
                "a Flash-Next compact round names slot {slot}, outside 0..{MAX_BATCH}"
            ))
        })?;
        if *occupied {
            return Err(EngineError::route(format!(
                "a Flash-Next compact round names slot {slot} twice; one sequence cannot occupy \
                 two rows of one round"
            )));
        }
        *occupied = true;
        if pending {
            round.slots[round.rows] = slot;
            round.rows += 1;
        }
    }

    Ok(round)
}

/// Drops retired entries while preserving survivor order and physical slots.
pub fn qwen38_flash_next_compact_survivors(
    active: &[usize],
    retired: &[bool],
) -> EngineResult<([usize; MAX_BATCH], usize)> {
    if active.len() != retired.len() {
        return Err(EngineError::route(format!(
            "a Flash-Next retirement was given {} active slots and {} retired flags",
            active.len(),
            retired.len()
        )));
    }
    if active.len() > MAX_BATCH {
        return Err(EngineError::route(format!(
            "a Flash-Next retirement names {} active slots, more than the {MAX_BATCH} funded",
            active.len()
        )));
    }
    let mut survivors = [usize::MAX; MAX_BATCH];
    let mut surviving = 0;
    for (&slot, &retired) in active.iter().zip(retired) {
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "a Flash-Next retirement names slot {slot}, outside 0..{MAX_BATCH}"
            )));
        }
        if !retired {
            survivors[surviving] = slot;
            surviving += 1;
        }
    }

    Ok((survivors, surviving))
}

/// Selects the lowest-numbered free physical slot deterministically.
pub fn qwen38_flash_next_admission_slot(occupied: [bool; MAX_BATCH]) -> Option<usize> {
    occupied.iter().position(|&occupied| !occupied)
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen38FlashNextCompactRound, qwen38_flash_next_admission_slot,
        qwen38_flash_next_compact_round, qwen38_flash_next_compact_survivors,
    };
    use crate::qwen38_flash_next::layer_route::Qwen38FlashNextRowRoute;
    use crate::{EngineErrorCode, MAX_BATCH};

    /// One row of the route policy table: an active order and its pending flags, and the exact
    /// dense round they describe.
    struct RoundCase {
        name: &'static str,
        active: &'static [usize],
        pending: &'static [bool],
        rows: &'static [usize],
    }

    const ROUND_TABLE: &[RoundCase] = &[
        RoundCase {
            name: "an idle scheduler has no round",
            active: &[],
            pending: &[],
            rows: &[],
        },
        RoundCase {
            name: "one just-admitted request replays nothing",
            active: &[0],
            pending: &[false],
            rows: &[],
        },
        RoundCase {
            name: "one live request is a B=1 round on its own slot",
            active: &[0],
            pending: &[true],
            rows: &[0],
        },
        RoundCase {
            name: "eight live requests fill every row in admission order",
            active: &[0, 1, 2, 3, 4, 5, 6, 7],
            pending: &[true; 8],
            rows: &[0, 1, 2, 3, 4, 5, 6, 7],
        },
        RoundCase {
            name: "a hole in the middle is a hole in the slots, not in the rows",
            active: &[0, 2, 5],
            pending: &[true, true, true],
            rows: &[0, 2, 5],
        },
        RoundCase {
            name: "noncontiguous survivors keep their admission order",
            active: &[7, 1, 4],
            pending: &[true, true, true],
            rows: &[7, 1, 4],
        },
        RoundCase {
            name: "a request admitted mid-flight sits out exactly one round",
            active: &[3, 6, 2],
            pending: &[true, false, true],
            rows: &[3, 2],
        },
        RoundCase {
            name: "a round where only the newest request is pending is B=1",
            active: &[0, 1, 2],
            pending: &[false, false, true],
            rows: &[2],
        },
        RoundCase {
            name: "seven of eight pending is a B=7 round",
            active: &[0, 1, 2, 3, 4, 5, 6, 7],
            pending: &[true, true, true, false, true, true, true, true],
            rows: &[0, 1, 2, 4, 5, 6, 7],
        },
    ];

    #[test]
    fn the_route_table_packs_every_admitted_scheduler_state() {
        for case in ROUND_TABLE {
            let round = qwen38_flash_next_compact_round(case.active, case.pending)
                .unwrap_or_else(|error| panic!("{}: {error}", case.name));

            assert_eq!(round.slots(), case.rows, "{}", case.name);
            assert_eq!(round.rows(), case.rows.len(), "{}", case.name);
            assert_eq!(round.is_empty(), case.rows.is_empty(), "{}", case.name);
            for (row, &slot) in case.rows.iter().enumerate() {
                assert_eq!(round.row_of(slot), Some(row), "{}", case.name);
            }
        }
    }

    #[test]
    fn every_nonempty_round_selects_a_captured_decode_graph() {
        for case in ROUND_TABLE {
            let round = qwen38_flash_next_compact_round(case.active, case.pending).unwrap();
            match round.route() {
                None => assert!(round.is_empty(), "{}", case.name),
                Some(route) => assert_eq!(
                    route.unwrap(),
                    Qwen38FlashNextRowRoute::Decode(case.rows.len()),
                    "{}",
                    case.name
                ),
            }
        }
    }

    #[test]
    fn a_round_never_names_one_slot_twice() {
        // One sequence may advance only once per round.
        let error = qwen38_flash_next_compact_round(&[3, 1, 3], &[true, true, true]).unwrap_err();

        assert_eq!(error.code(), Some(EngineErrorCode::Route));
        assert!(error.to_string().contains("twice"));
        // Reject duplicates even when only one duplicate is pending.
        assert!(qwen38_flash_next_compact_round(&[3, 1, 3], &[true, true, false]).is_err());
    }

    #[test]
    fn a_round_outside_the_eight_funded_slots_is_refused() {
        for (active, pending) in [
            (vec![MAX_BATCH], vec![true]),
            (vec![0, 99], vec![true, true]),
            ((0..=MAX_BATCH).collect(), vec![true; MAX_BATCH + 1]),
        ] {
            let error = qwen38_flash_next_compact_round(&active, &pending).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
        let error = qwen38_flash_next_compact_round(&[0, 1], &[true]).unwrap_err();
        assert!(error.to_string().contains("pending flags"));
    }

    #[test]
    fn survivors_close_the_hole_without_moving_anyone_else() {
        let cases: &[(&[usize], &[bool], &[usize])] = &[
            (&[0, 1, 2, 3], &[false, false, false, false], &[0, 1, 2, 3]),
            (&[0, 1, 2, 3], &[true, false, true, false], &[1, 3]),
            (&[7, 1, 4], &[false, true, false], &[7, 4]),
            (&[5], &[true], &[]),
            (&[], &[], &[]),
        ];
        for (active, retired, expected) in cases {
            let (survivors, surviving) =
                qwen38_flash_next_compact_survivors(active, retired).unwrap();

            assert_eq!(&survivors[..surviving], *expected, "{active:?}");
            assert_eq!(surviving, expected.len(), "{active:?}");
            // Keep the unused tail sentinel-filled.
            assert!(
                survivors[surviving..]
                    .iter()
                    .all(|&slot| slot == usize::MAX)
            );
        }
    }

    #[test]
    fn a_grouped_prompt_prime_narrows_as_prompts_finish() {
        let slots = [0usize, 1, 2];
        let tails = [11usize, 3, 7];
        let mut widths = Vec::new();
        let mut rounds = Vec::new();
        for position in 0..11 {
            let pending = tails.map(|tail| position < tail);
            let round = qwen38_flash_next_compact_round(&slots, &pending).unwrap();
            widths.push(round.rows());
            rounds.push(round.slots().to_vec());
        }

        assert_eq!(widths, [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]);
        assert_eq!(rounds[0], [0, 1, 2]);
        assert_eq!(rounds[3], [0, 2]);
        assert_eq!(rounds[7], [0]);
        assert!(widths.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(
            qwen38_flash_next_compact_round(&[0, 1], &[false, false])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_retirement_and_the_round_that_produced_it_agree_on_their_lengths() {
        let error = qwen38_flash_next_compact_survivors(&[0, 1], &[true]).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Route));
        assert!(qwen38_flash_next_compact_survivors(&[MAX_BATCH], &[false]).is_err());
        assert!(
            qwen38_flash_next_compact_survivors(&[0; MAX_BATCH + 1], &[false; MAX_BATCH + 1])
                .is_err()
        );
    }

    #[test]
    fn admission_takes_the_lowest_free_slot() {
        let cases = [
            ([false; MAX_BATCH], Some(0)),
            (
                [true, false, true, false, true, false, true, false],
                Some(1),
            ),
            ([true, true, true, true, true, true, true, false], Some(7)),
            ([true; MAX_BATCH], None),
        ];
        for (occupied, expected) in cases {
            assert_eq!(
                qwen38_flash_next_admission_slot(occupied),
                expected,
                "{occupied:?}"
            );
        }
    }

    #[test]
    fn a_full_lifecycle_reproduces_its_slots_from_the_admission_sequence_alone() {
        // Survivors keep slots while new requests fill retirement holes.
        let mut occupied = [false; MAX_BATCH];
        let mut active = Vec::new();
        for _ in 0..4 {
            let slot = qwen38_flash_next_admission_slot(occupied).unwrap();
            occupied[slot] = true;
            active.push(slot);
        }
        assert_eq!(active, vec![0, 1, 2, 3]);

        let retired = [false, true, false, true];
        let (survivors, surviving) =
            qwen38_flash_next_compact_survivors(&active, &retired).unwrap();
        for (&slot, &retired) in active.iter().zip(&retired) {
            occupied[slot] = !retired;
        }
        active = survivors[..surviving].to_vec();
        assert_eq!(active, vec![0, 2]);

        for _ in 0..2 {
            let slot = qwen38_flash_next_admission_slot(occupied).unwrap();
            occupied[slot] = true;
            active.push(slot);
        }
        assert_eq!(active, vec![0, 2, 1, 3]);

        // Fresh primes are not pending decode rows.
        let round = qwen38_flash_next_compact_round(&active, &[true, true, false, false]).unwrap();
        assert_eq!(round.slots(), [0, 2]);
        assert_eq!(round.row_of(0), Some(0));
        assert_eq!(round.row_of(2), Some(1));
        assert_eq!(round.row_of(1), None);
    }

    #[test]
    fn the_empty_round_carries_no_slot_and_no_graph() {
        let round = Qwen38FlashNextCompactRound::EMPTY;

        assert!(round.is_empty());
        assert_eq!(round.rows(), 0);
        assert!(round.slots().is_empty());
        assert!(round.route().is_none());
        assert_eq!(round.row_of(0), None);
    }
}
