//! Host engram hashing and zero-copy FP8 table addressing.
//!
//! [`Qwen38FlashNextEngramCarry`] treats EOS as a segment boundary. Multiplication uses the model's mod-2^64
//! rule after token admission proves every represented product fits in signed 64 bits.

use crate::qwen38_flash_next::engram::Qwen38FlashNextEngramHashConstants;
use crate::{Arch, CheckpointError, CheckpointResult, Qwen38FlashNext};

type F = Qwen38FlashNext;

/// Longest n-gram hashed, and therefore the shift count.
const NGRAM_SIZE: usize = F::NGRAM_SIZE;
/// Independently hashed lookup heads.
const NGRAM_HEADS: usize = F::NGRAM_HEADS;
/// Heads sharing one n-gram order; head `h` hashes order `2 + h / HEADS_PER_NGRAM`.
const HEADS_PER_NGRAM: usize = F::HEADS_PER_NGRAM;

/// Tokens of history carried between streaming calls.
pub const QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN: usize = NGRAM_SIZE - 1;

/// Token that terminates an engram segment.
pub const QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN: u32 = F::EOS_TOKEN_ID;

/// Rows one token addresses: one per engram head.
pub const QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN: usize = NGRAM_HEADS;

/// Token history carried between streaming calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextEngramCarry {
    /// Previous tokens, most recent first: `previous[k - 1]` is `x[t - k]`.
    previous: [u32; QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN],
}

impl Qwen38FlashNextEngramCarry {
    /// The carry at sequence start: `[eos, eos]`.
    pub const fn start() -> Self {
        Self {
            previous: [QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN; QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN],
        }
    }

    /// The carry a stream reaches after consuming `tokens` from a sequence start.
    pub fn after(tokens: &[u32]) -> CheckpointResult<Self> {
        let mut carry = Self::start();

        for token in tokens.iter().copied() {
            admit_qwen38_flash_next_engram_token(token)?;
            carry.push(token);
        }

        Ok(carry)
    }

    /// Raw previous tokens, most recent first.
    pub const fn previous(&self) -> [u32; QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN] {
        self.previous
    }

    /// Returns the current token and its within-segment history.
    pub const fn shifts(&self, token: u32) -> [u32; NGRAM_SIZE] {
        let mut shifts = [QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN; NGRAM_SIZE];
        let mut position = 1;

        shifts[0] = token;

        while position <= QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN {
            let previous = self.previous[position - 1];

            shifts[position] = previous;

            if previous == QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN {
                break;
            }

            position += 1;
        }

        shifts
    }

    /// Advances the carry past `token`.
    const fn push(&mut self, token: u32) {
        let mut position = QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN - 1;

        while position > 0 {
            self.previous[position] = self.previous[position - 1];
            position -= 1;
        }

        self.previous[0] = token;
    }
}

impl Default for Qwen38FlashNextEngramCarry {
    fn default() -> Self {
        Self::start()
    }
}

/// The engram row law over one admitted constant set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextEngramRowHasher {
    constants: Qwen38FlashNextEngramHashConstants,
}

impl Qwen38FlashNextEngramRowHasher {
    /// Binds the law to one admitted constant set.
    pub const fn new(constants: Qwen38FlashNextEngramHashConstants) -> Self {
        Self { constants }
    }

    /// The constants this hasher addresses rows with.
    pub const fn constants(&self) -> &Qwen38FlashNextEngramHashConstants {
        &self.constants
    }

    /// Returns one token's rows in head order and advances the carry.
    pub fn rows(
        &self,
        carry: &mut Qwen38FlashNextEngramCarry,
        token: u32,
    ) -> CheckpointResult<[i64; NGRAM_HEADS]> {
        admit_qwen38_flash_next_engram_token(token)?;

        let shifts = carry.shifts(token);
        let mut hashes = [0i64; NGRAM_SIZE];
        let mut running = 0i64;

        carry.push(token);

        // Heads 0..7 read the bigram hash; heads 8..15 read the trigram hash.
        for ((hash, shift), multiplier) in hashes
            .iter_mut()
            .zip(shifts)
            .zip(self.constants.layer_multipliers().iter().copied())
        {
            debug_assert!(
                i64::from(shift).checked_mul(multiplier).is_some(),
                "admitted token/multiplier product must fit in i64"
            );
            running ^= i64::from(shift).wrapping_mul(multiplier);
            *hash = running;
        }

        let mut rows = [0i64; NGRAM_HEADS];

        for (head, ((row, vocab), offset)) in rows
            .iter_mut()
            .zip(self.constants.head_vocab_sizes().iter().copied())
            .zip(self.constants.head_offsets().iter().copied())
            .enumerate()
        {
            *row = hashes[1 + head / HEADS_PER_NGRAM] % vocab + offset;
        }

        Ok(rows)
    }

    /// Rows for a whole token window, token-major and head-minor.
    ///
    /// `rows` must hold exactly [`QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN`] entries per token; token `t`'s head
    /// `h` lands at `rows[t * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN + h]`. The carry advances across the window,
    /// so feeding a window in one call and feeding it in pieces address identical rows.
    pub fn stream_rows(
        &self,
        carry: &mut Qwen38FlashNextEngramCarry,
        tokens: &[u32],
        rows: &mut [i64],
    ) -> CheckpointResult<()> {
        let expected = tokens
            .len()
            .checked_mul(QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN)
            .ok_or_else(|| row_error("engram row count overflows this host"))?;

        if rows.len() != expected {
            return Err(row_error(format!(
                "engram row destination holds {} rows, expected {expected} for {} tokens",
                rows.len(),
                tokens.len()
            )));
        }

        for token in tokens.iter().copied() {
            admit_qwen38_flash_next_engram_token(token)?;
        }

        let (destinations, remainder) =
            rows.as_chunks_mut::<QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN>();
        debug_assert!(remainder.is_empty());

        for (token, destination) in tokens.iter().copied().zip(destinations.iter_mut()) {
            destination.copy_from_slice(&self.rows(carry, token)?);
        }

        Ok(())
    }
}

/// Borrowed engram table addressed by row.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextEngramTable<'a> {
    shards: &'a [&'a [u8]],
    shard_rows: usize,
    head_dim: usize,
    token_bytes: usize,
    constants: Qwen38FlashNextEngramHashConstants,
}

impl<'a> Qwen38FlashNextEngramTable<'a> {
    /// Views equal-sized shards in index order as one table.
    pub fn new(
        shards: &'a [&'a [u8]],
        shard_rows: usize,
        head_dim: usize,
        constants: Qwen38FlashNextEngramHashConstants,
    ) -> CheckpointResult<Self> {
        if shards.is_empty() || shard_rows == 0 || head_dim == 0 {
            return Err(row_error(format!(
                "engram table needs shards of nonzero rows and width, given {} shards of {shard_rows} rows by {head_dim}",
                shards.len()
            )));
        }

        let shard_bytes = shard_rows
            .checked_mul(head_dim)
            .ok_or_else(|| row_error("engram shard bytes overflow this host"))?;

        shard_bytes
            .checked_mul(shards.len())
            .ok_or_else(|| row_error("engram table bytes overflow this host"))?;
        let token_bytes = constants
            .head_vocab_sizes()
            .len()
            .checked_mul(head_dim)
            .ok_or_else(|| row_error("engram token bytes overflow this host"))?;

        for (shard, codes) in shards.iter().enumerate() {
            if codes.len() != shard_bytes {
                return Err(row_error(format!(
                    "engram shard {shard} holds {} bytes, expected {shard_bytes}",
                    codes.len()
                )));
            }
        }

        Ok(Self {
            shards,
            shard_rows,
            head_dim,
            token_bytes,
            constants,
        })
    }

    /// The admitted hash law this table is addressed by.
    pub const fn constants(&self) -> Qwen38FlashNextEngramHashConstants {
        self.constants
    }

    /// Width contributed by one head, and therefore by one row.
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Rows held by one shard.
    pub const fn shard_rows(&self) -> usize {
        self.shard_rows
    }

    /// Engram lookup heads, and therefore rows, per token.
    pub const fn heads(&self) -> usize {
        self.constants.head_vocab_sizes().len()
    }

    /// Rows the borrowed shards address, in shard-index order.
    pub const fn table_rows(&self) -> usize {
        self.shards.len() * self.shard_rows
    }

    /// Bytes one token contributes to a staged engram plane: one row per head.
    pub const fn token_bytes(&self) -> usize {
        self.token_bytes
    }

    /// Returns one runtime row's unmodified FP8 code bytes.
    pub fn row_codes(&self, row: i64) -> CheckpointResult<&'a [u8]> {
        let rows = self.table_rows();
        let index = usize::try_from(row)
            .ok()
            .filter(|index| *index < rows)
            .ok_or_else(|| row_error(format!("engram row {row} is outside 0..{rows}")))?;
        let offset = (index % self.shard_rows) * self.head_dim;

        Ok(&self.shards[index / self.shard_rows][offset..offset + self.head_dim])
    }

    /// Gathers rows into caller-owned storage without decoding them.
    pub fn gather_rows(&self, rows: &[i64], destination: &mut [u8]) -> CheckpointResult<()> {
        let expected = rows
            .len()
            .checked_mul(self.head_dim)
            .ok_or_else(|| row_error("engram gather bytes overflow this host"))?;

        if destination.len() != expected {
            return Err(row_error(format!(
                "engram gather destination holds {} bytes, expected {expected} for {} rows",
                destination.len(),
                rows.len()
            )));
        }

        for (row, slot) in rows
            .iter()
            .copied()
            .zip(destination.chunks_exact_mut(self.head_dim))
        {
            slot.copy_from_slice(self.row_codes(row)?);
        }

        Ok(())
    }
}

/// Refuses a token id outside the vocabulary the overflow proof covers.
pub fn admit_qwen38_flash_next_engram_token(token: u32) -> CheckpointResult<()> {
    if token as usize >= F::VOCAB {
        return Err(row_error(format!(
            "engram token id {token} is outside vocabulary 0..{}",
            F::VOCAB
        )));
    }

    Ok(())
}

fn row_error(message: impl Into<String>) -> CheckpointError {
    CheckpointError::source_binding(format!("Qwen3.8-Flash-Next {}", message.into()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::test_support::sources::fixture_path;
    use crate::qwen38_flash_next::engram::Qwen38FlashNextEngramConstantBindings;
    use crate::qwen38_flash_next::engram::tests::{
        MULTIPLIERS, OFFSETS, PREFIX, VOCAB_SIZES, engram_constant_fixture,
    };
    use crate::{CheckpointErrorCode, SafeTensorFile};
    use std::fs;

    /// A deterministic token stream with EOS injected at a fixed cadence.
    ///
    /// Reproducible without a dependency: the sequence is fully determined by `seed`.
    pub(crate) fn token_stream(seed: u64, len: usize, eos_every: usize) -> Vec<u32> {
        let mut state = seed;

        (0..len)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);

                if eos_every > 0 && index % eos_every == eos_every - 1 {
                    QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN
                } else {
                    ((state >> 33) % (F::VOCAB as u64 - 1)) as u32
                }
            })
            .collect()
    }

    /// Independent whole-sequence oracle that scans for the previous EOS.
    fn literal_shifts(tokens: &[u32], position: usize) -> [u32; NGRAM_SIZE] {
        let segment_start = (0..position)
            .rev()
            .find(|&index| tokens[index] == QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN)
            .map_or(0, |index| index + 1);
        let mut shifts = [QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN; NGRAM_SIZE];

        for (shift, back) in shifts.iter_mut().zip(0..NGRAM_SIZE) {
            if position - segment_start >= back {
                *shift = tokens[position - back];
            }
        }

        shifts
    }

    fn literal_rows(constants: &Qwen38FlashNextEngramHashConstants, tokens: &[u32]) -> Vec<i64> {
        let mut rows = Vec::new();

        for position in 0..tokens.len() {
            let shifts = literal_shifts(tokens, position);
            let terms = [0, 1, 2].map(|k| {
                i64::from(shifts[k])
                    .checked_mul(constants.layer_multipliers()[k])
                    .expect("no admitted product overflows")
            });
            let bigram = terms[0] ^ terms[1];
            let trigram = bigram ^ terms[2];

            for head in 0..NGRAM_HEADS {
                let hash = if head < HEADS_PER_NGRAM {
                    bigram
                } else {
                    trigram
                };

                rows.push(
                    hash % constants.head_vocab_sizes()[head] + constants.head_offsets()[head],
                );
            }
        }

        rows
    }

    fn hasher() -> Qwen38FlashNextEngramRowHasher {
        Qwen38FlashNextEngramRowHasher::new(Qwen38FlashNextEngramHashConstants::compute().unwrap())
    }

    fn stream(hasher: &Qwen38FlashNextEngramRowHasher, tokens: &[u32]) -> Vec<i64> {
        let mut rows = vec![0i64; tokens.len() * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];
        let mut carry = Qwen38FlashNextEngramCarry::start();

        hasher
            .stream_rows(&mut carry, tokens, &mut rows)
            .expect("admitted tokens hash");

        rows
    }

    #[test]
    fn the_carry_reproduces_the_literal_segment_law() {
        let hasher = hasher();

        // Cover no EOS and several mid-sequence EOS cadences.
        for (seed, eos_every) in [(1, 0), (2, 7), (3, 3), (4, 2), (5, 1)] {
            let tokens = token_stream(seed, 96, eos_every);

            assert_eq!(
                stream(&hasher, &tokens),
                literal_rows(hasher.constants(), &tokens),
                "seed {seed}, EOS every {eos_every}"
            );
        }
    }

    #[test]
    fn segment_boundaries_and_short_prompts_shift_exactly() {
        let carry = Qwen38FlashNextEngramCarry::start();
        const EOS: u32 = QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN;

        // (d) a prompt shorter than context_len = 2: every missing shift reads EOS.
        assert_eq!(carry.shifts(11), [11, EOS, EOS]);
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11]).unwrap().shifts(12),
            [12, 11, EOS]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11, 12])
                .unwrap()
                .shifts(13),
            [13, 12, 11]
        );

        // (c) an EOS at position 0 terminates a segment that never held a token.
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[EOS])
                .unwrap()
                .shifts(11),
            [11, EOS, EOS]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[EOS, 11])
                .unwrap()
                .shifts(12),
            [12, 11, EOS]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[EOS, 11, 12])
                .unwrap()
                .shifts(13),
            [13, 12, 11]
        );

        // (b) a mid-sequence EOS: the EOS itself still sees its own segment, the token after it
        // sees nothing, and the segment refills one shift at a time.
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11, 12])
                .unwrap()
                .shifts(EOS),
            [EOS, 12, 11]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11, 12, EOS])
                .unwrap()
                .shifts(13),
            [13, EOS, EOS]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11, 12, EOS, 13])
                .unwrap()
                .shifts(14),
            [14, 13, EOS]
        );
        assert_eq!(
            Qwen38FlashNextEngramCarry::after(&[11, 12, EOS, 13, 14])
                .unwrap()
                .shifts(15),
            [15, 14, 13]
        );
    }

    #[test]
    fn a_prefill_window_and_its_streamed_rounds_address_identical_rows() {
        let hasher = hasher();

        for (seed, eos_every) in [(11, 0), (12, 5), (13, 2), (14, 1)] {
            let tokens = token_stream(seed, 71, eos_every);
            let expected = stream(&hasher, &tokens);

            // Every split the house's route widths can produce, including a decode tail of
            // one-token rounds after a prefill window.
            for round in [1, 2, 3, 8, 17, 70] {
                let mut carry = Qwen38FlashNextEngramCarry::start();
                let mut streamed = Vec::new();

                for window in tokens.chunks(round) {
                    let mut rows =
                        vec![0i64; window.len() * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];

                    hasher.stream_rows(&mut carry, window, &mut rows).unwrap();
                    streamed.extend_from_slice(&rows);
                }

                assert_eq!(streamed, expected, "seed {seed} in rounds of {round}");
            }
        }
    }

    #[test]
    fn resuming_from_a_two_token_carry_matches_a_full_recompute() {
        let hasher = hasher();
        let tokens = token_stream(21, 64, 4);
        let expected = stream(&hasher, &tokens);

        for split in [1, 2, 3, 4, 5, 33, 63] {
            // A resumed sequence knows only its previous two tokens, never the whole history.
            let mut carry = Qwen38FlashNextEngramCarry::after(&tokens[..split]).unwrap();
            let mut rows =
                vec![0i64; (tokens.len() - split) * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];

            hasher
                .stream_rows(&mut carry, &tokens[split..], &mut rows)
                .unwrap();

            assert_eq!(
                rows,
                expected[split * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN..],
                "split {split}"
            );
        }
    }

    #[test]
    fn every_row_lands_inside_its_own_head_vocabulary() {
        let hasher = hasher();
        let constants = *hasher.constants();
        let tokens = token_stream(31, 256, 6);
        let rows = stream(&hasher, &tokens);

        for token_rows in rows
            .as_chunks::<QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN>()
            .0
        {
            for (head, row) in token_rows.iter().copied().enumerate() {
                let offset = constants.head_offsets()[head];

                assert!(
                    row >= offset && row < offset + constants.head_vocab_sizes()[head],
                    "head {head} row {row}"
                );
                assert!(row < constants.total_vocab());
            }
        }
    }

    #[test]
    fn the_two_hashed_orders_split_the_heads_in_half() {
        let hasher = hasher();
        // Two streams whose current and previous tokens agree but whose token before those
        // differs: the bigram heads cannot see the difference, the trigram heads must.
        let mut bigram_equal = Qwen38FlashNextEngramCarry::after(&[11, 12]).unwrap();
        let mut bigram_other = Qwen38FlashNextEngramCarry::after(&[13, 12]).unwrap();
        let left = hasher.rows(&mut bigram_equal, 14).unwrap();
        let right = hasher.rows(&mut bigram_other, 14).unwrap();

        for head in 0..HEADS_PER_NGRAM {
            assert_eq!(left[head], right[head], "bigram head {head}");
        }
        for head in HEADS_PER_NGRAM..NGRAM_HEADS {
            assert_ne!(left[head], right[head], "trigram head {head}");
        }
    }

    #[test]
    fn no_admitted_token_wraps_the_i64_products() {
        let constants = Qwen38FlashNextEngramHashConstants::compute().unwrap();

        for token in [
            0,
            1,
            F::VOCAB as i64 - 1,
            QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN as i64,
        ] {
            for multiplier in constants.layer_multipliers().iter().copied() {
                let product = token
                    .checked_mul(multiplier)
                    .expect("admitted product must not overflow");

                assert_eq!(product, token.wrapping_mul(multiplier));
                assert!(product >= 0);
            }
        }
    }

    #[test]
    fn refuses_a_token_outside_the_admitted_vocabulary() {
        let hasher = hasher();
        let mut carry = Qwen38FlashNextEngramCarry::start();
        let mut rows = [0i64; QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];

        admit_qwen38_flash_next_engram_token(F::VOCAB as u32 - 1).unwrap();

        for token in [F::VOCAB as u32, u32::MAX] {
            let error = Qwen38FlashNextEngramCarry::after(&[token]).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);

            let before = carry;
            let error = hasher.rows(&mut carry, token).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("outside vocabulary"), "{error}");
            assert_eq!(carry, before, "a refused token must not advance the carry");

            let error = hasher
                .stream_rows(&mut carry, &[token], &mut rows)
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("outside vocabulary"), "{error}");
        }

        let mut carry = Qwen38FlashNextEngramCarry::after(&[7]).unwrap();
        let before = carry;
        let mut rows = [-1; 2 * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];
        let error = hasher
            .stream_rows(&mut carry, &[8, F::VOCAB as u32], &mut rows)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert_eq!(carry, before, "a refused window must not advance the carry");
        assert!(rows.iter().all(|row| *row == -1));
    }

    #[test]
    fn refuses_a_row_destination_that_is_not_one_row_per_head_per_token() {
        let hasher = hasher();
        let mut carry = Qwen38FlashNextEngramCarry::start();

        for rows in [
            QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN - 1,
            QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN + 1,
            0,
        ] {
            let error = hasher
                .stream_rows(&mut carry, &[11], &mut vec![0i64; rows])
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("expected 16"), "{error}");
        }
    }

    #[test]
    fn refuses_a_table_view_the_shard_arithmetic_cannot_address() {
        let constants = Qwen38FlashNextEngramHashConstants::compute().unwrap();
        let short = vec![0u8; 7];
        let exact = vec![0u8; 8];
        let shards = [exact.as_slice(), short.as_slice()];

        let error = Qwen38FlashNextEngramTable::new(&shards, 4, 2, constants)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error.to_string().contains("shard 1 holds 7 bytes"),
            "{error}"
        );

        for (rows, width) in [(0, 2), (4, 0)] {
            let error = Qwen38FlashNextEngramTable::new(&shards[..1], rows, width, constants)
                .err()
                .unwrap();

            assert!(
                error.to_string().contains("nonzero rows and width"),
                "{error}"
            );
        }

        let error = Qwen38FlashNextEngramTable::new(&[], 4, 2, constants)
            .err()
            .unwrap();

        assert!(error.to_string().contains("given 0 shards"), "{error}");

        let empty = [&[][..]];
        let error = Qwen38FlashNextEngramTable::new(&empty, 1, usize::MAX, constants)
            .err()
            .unwrap();

        assert!(
            error.to_string().contains("token bytes overflow"),
            "{error}"
        );

        // One shard of four rows addresses four rows.
        let table = Qwen38FlashNextEngramTable::new(&shards[..1], 4, 2, constants).unwrap();

        assert_eq!(table.table_rows(), 4);
        assert_eq!(table.token_bytes(), 32);
        assert_eq!(table.heads(), NGRAM_HEADS);
    }

    #[test]
    fn the_checkpoints_own_buffers_address_the_same_rows_as_the_computed_law() {
        let path = fixture_path("qwen38_flash_next-engram-hash-constants");
        engram_constant_fixture(MULTIPLIERS, VOCAB_SIZES, OFFSETS).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextEngramConstantBindings::bind_from(PREFIX, |name| file.tensor(name))
                .unwrap();

        // The bound constants are the gated stored words; the hasher must not restate them.
        assert_eq!(
            bindings.layer_multipliers.values().collect::<Vec<_>>(),
            bindings.constants.layer_multipliers()
        );

        let bound = Qwen38FlashNextEngramRowHasher::new(bindings.constants);
        let tokens = token_stream(41, 48, 5);

        assert_eq!(bound, hasher());
        assert_eq!(stream(&bound, &tokens), stream(&hasher(), &tokens));

        fs::remove_file(path).unwrap();
    }
}
