//! Qwen3.8-Flash-Next engram hash constants and checkpoint validation.
//!
//! Admission recomputes the target-local law and requires exact agreement with all three I64
//! buffers stored in the checkpoint.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::naming::layer_prefix;
use crate::{Arch, CheckpointError, CheckpointResult, I64View, Qwen38FlashNext, TensorView};

type F = Qwen38FlashNext;

/// Longest n-gram hashed, and therefore the multiplier count.
const NGRAM_SIZE: usize = F::NGRAM_SIZE;
/// Independently hashed lookup heads.
const NGRAM_HEADS: usize = F::NGRAM_HEADS;

/// SplitMix64 increment.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64 first finalizer multiplier.
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
/// SplitMix64 second finalizer multiplier.
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// `_build_layer_multipliers` seed.
const MULTIPLIER_SEED: u64 = 1_234;
/// Per-PLE-layer seed stride. This checkpoint has one PLE layer, so it contributes nothing.
const MULTIPLIER_SEED_STRIDE: u64 = 10_007;
/// Zero-based index of the PLE layer among the model's PLE layers, not among its decoder layers.
const PLE_LAYER_INDEX: u64 = 0;

/// Engram row-addressing constants derived independently of checkpoint bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextEngramHashConstants {
    /// Per-n-gram-position hash multipliers, always odd.
    layer_multipliers: [i64; NGRAM_SIZE],
    /// Per-head prime moduli, ascending.
    head_vocab_sizes: [i64; NGRAM_HEADS],
    /// Per-head exclusive prefix sums of `head_vocab_sizes`.
    head_offsets: [i64; NGRAM_HEADS],
    /// Summed addressable rows before alignment padding.
    total_vocab: i64,
    /// Table rows after alignment padding; the padding rows are never addressable.
    padded_rows: usize,
}

impl Qwen38FlashNextEngramHashConstants {
    /// Computes the complete law from the architecture's own constants.
    pub fn compute() -> CheckpointResult<Self> {
        let layer_multipliers = compute_layer_multipliers()?;
        let head_vocab_sizes = compute_head_vocab_sizes()?;
        let head_offsets = compute_head_offsets(&head_vocab_sizes)?;
        let total_vocab = head_offsets[NGRAM_HEADS - 1]
            .checked_add(head_vocab_sizes[NGRAM_HEADS - 1])
            .ok_or_else(|| law_error("summed engram vocabulary overflows"))?;
        let padded_rows = usize::try_from(total_vocab)
            .ok()
            .and_then(|total| total.checked_next_multiple_of(F::NGRAM_VOCAB_DIVISOR))
            .ok_or_else(|| law_error("padded engram vocabulary overflows this host"))?;
        let shard_rows = F::NGRAM_SHARDS
            .checked_mul(F::NGRAM_SHARD_ROWS)
            .ok_or_else(|| law_error("engram shard row count overflows"))?;

        if padded_rows != shard_rows {
            return Err(law_error(format!(
                "computed engram table has {padded_rows} rows but the checkpoint's {} shards of {} rows hold {shard_rows}",
                F::NGRAM_SHARDS,
                F::NGRAM_SHARD_ROWS
            )));
        }

        Ok(Self {
            layer_multipliers,
            head_vocab_sizes,
            head_offsets,
            total_vocab,
            padded_rows,
        })
    }

    /// Builds a smaller structurally valid law for qualification fixtures.
    #[cfg(feature = "qualification")]
    pub fn for_qualification(head_vocab_sizes: [i64; NGRAM_HEADS]) -> CheckpointResult<Self> {
        if head_vocab_sizes.iter().any(|size| *size <= 0) {
            return Err(law_error(
                "qualification engram vocabularies must be positive",
            ));
        }

        let layer_multipliers = compute_layer_multipliers()?;
        let head_offsets = compute_head_offsets(&head_vocab_sizes)?;
        let total_vocab = head_offsets[NGRAM_HEADS - 1]
            .checked_add(head_vocab_sizes[NGRAM_HEADS - 1])
            .ok_or_else(|| law_error("summed qualification engram vocabulary overflows"))?;
        let padded_rows = usize::try_from(total_vocab)
            .ok()
            .and_then(|total| total.checked_next_multiple_of(F::NGRAM_VOCAB_DIVISOR))
            .ok_or_else(|| law_error("qualification engram vocabulary padding overflows"))?;

        Ok(Self {
            layer_multipliers,
            head_vocab_sizes,
            head_offsets,
            total_vocab,
            padded_rows,
        })
    }

    /// Per-position hash multipliers.
    pub const fn layer_multipliers(&self) -> &[i64; NGRAM_SIZE] {
        &self.layer_multipliers
    }

    /// Per-head prime moduli.
    pub const fn head_vocab_sizes(&self) -> &[i64; NGRAM_HEADS] {
        &self.head_vocab_sizes
    }

    /// Per-head exclusive row offsets.
    pub const fn head_offsets(&self) -> &[i64; NGRAM_HEADS] {
        &self.head_offsets
    }

    /// Addressable rows before alignment padding.
    pub const fn total_vocab(&self) -> i64 {
        self.total_vocab
    }

    /// Physical rows after alignment padding.
    pub const fn padded_rows(&self) -> usize {
        self.padded_rows
    }
}

/// Three I64 buffers validated against the computed hash law.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextEngramConstantBindings<'a> {
    /// Stored per-n-gram-position multipliers `[ngram_size]`.
    pub layer_multipliers: I64View<'a, 1>,
    /// Stored per-head prime moduli `[ngram_heads]`.
    pub head_vocab_sizes: I64View<'a, 1>,
    /// Stored per-head row offsets `[ngram_heads]`.
    pub head_offsets: I64View<'a, 1>,
    /// The computed law these buffers were admitted against.
    pub constants: Qwen38FlashNextEngramHashConstants,
}

/// Tensor-key prefix of the single admitted PLE layer.
pub(crate) fn engram_table_prefix(layer: usize, ple_layer: usize) -> CheckpointResult<String> {
    if layer != ple_layer {
        return Err(law_error(format!(
            "layer {layer} carries no engram; only layer {ple_layer} does"
        )));
    }

    Ok(format!("{}.ple.ple_embedding", layer_prefix(layer)))
}

impl<'a> Qwen38FlashNextEngramConstantBindings<'a> {
    /// Binds and gates the engram's three I64 buffers from the admitted snapshot.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(&engram_table_prefix(layer, F::PLE_LAYER)?, |name| {
            snapshot.tensor(name)
        })
    }

    /// Binds and gates the three buffers rooted at one `ple_embedding` prefix.
    pub(crate) fn bind_from(
        prefix: &str,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let constants = Qwen38FlashNextEngramHashConstants::compute()?;
        let layer_multipliers = I64View::bind(
            tensor(&format!("{prefix}.layer_multipliers"))?,
            [NGRAM_SIZE as u64],
        )?;
        let head_vocab_sizes = I64View::bind(
            tensor(&format!("{prefix}.ngram_heads_vocab_sizes"))?,
            [NGRAM_HEADS as u64],
        )?;
        let head_offsets = I64View::bind(
            tensor(&format!("{prefix}.ngram_heads_offsets"))?,
            [NGRAM_HEADS as u64],
        )?;

        require_stored_words(
            "layer_multipliers",
            &layer_multipliers,
            &constants.layer_multipliers,
        )?;
        require_stored_words(
            "ngram_heads_vocab_sizes",
            &head_vocab_sizes,
            &constants.head_vocab_sizes,
        )?;
        require_stored_words(
            "ngram_heads_offsets",
            &head_offsets,
            &constants.head_offsets,
        )?;

        Ok(Self {
            layer_multipliers,
            head_vocab_sizes,
            head_offsets,
            constants,
        })
    }
}

/// Refuses a stored buffer that disagrees with the computed law at any position.
fn require_stored_words(
    buffer: &str,
    stored: &I64View<'_, 1>,
    computed: &[i64],
) -> CheckpointResult<()> {
    for (index, expected) in computed.iter().copied().enumerate() {
        let actual = stored
            .value(index)
            .expect("validated shape covers every computed word");

        if actual != expected {
            return Err(CheckpointError::source_binding(format!(
                "engram `{buffer}[{index}]` is {actual}, but the checkpoint's own hash law computes {expected}"
            )));
        }
    }

    Ok(())
}

/// Computes odd layer multipliers with mod-2^64 SplitMix arithmetic.
fn compute_layer_multipliers() -> CheckpointResult<[i64; NGRAM_SIZE]> {
    let vocab = u64::try_from(F::VOCAB).map_err(|_| law_error("vocabulary exceeds u64"))?;
    let multiplier_max = i64::MAX as u64 / vocab;
    let half_bound = multiplier_max / 2;

    if half_bound == 0 {
        return Err(law_error("engram multiplier bound collapses to zero"));
    }

    let base_seed =
        MULTIPLIER_SEED.wrapping_add(MULTIPLIER_SEED_STRIDE.wrapping_mul(PLE_LAYER_INDEX));
    let mut multipliers = [0i64; NGRAM_SIZE];

    for (position, multiplier) in multipliers.iter_mut().enumerate() {
        let stride = SPLITMIX_GAMMA.wrapping_mul(position as u64 + 1);
        let word = 2 * (splitmix64(base_seed.wrapping_add(stride)) % half_bound) + 1;

        *multiplier = i64::try_from(word)
            .map_err(|_| law_error("engram multiplier exceeds the signed 64-bit range"))?;
    }

    Ok(multipliers)
}

/// The SplitMix64 finalizer, all arithmetic mod 2^64.
fn splitmix64(value: u64) -> u64 {
    let mut word = value.wrapping_add(SPLITMIX_GAMMA);
    word = (word ^ (word >> 30)).wrapping_mul(SPLITMIX_M1);
    word = (word ^ (word >> 27)).wrapping_mul(SPLITMIX_M2);

    word ^ (word >> 31)
}

/// Finds the first `NGRAM_HEADS` primes at or above `NGRAM_VOCAB_BASE`.
fn compute_head_vocab_sizes() -> CheckpointResult<[i64; NGRAM_HEADS]> {
    let mut sizes = [0i64; NGRAM_HEADS];
    let mut candidate = u64::try_from(F::NGRAM_VOCAB_BASE)
        .map_err(|_| law_error("engram vocabulary base exceeds u64"))?
        - 1;

    for size in sizes.iter_mut() {
        loop {
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| law_error("engram prime search overflows"))?;

            if is_prime(candidate) {
                break;
            }
        }

        *size = i64::try_from(candidate)
            .map_err(|_| law_error("engram head vocabulary exceeds the signed 64-bit range"))?;
    }

    Ok(sizes)
}

/// Exclusive prefix sums over the per-head vocabularies, in head order.
fn compute_head_offsets(vocab_sizes: &[i64; NGRAM_HEADS]) -> CheckpointResult<[i64; NGRAM_HEADS]> {
    let mut offsets = [0i64; NGRAM_HEADS];
    let mut running = 0i64;

    for (offset, size) in offsets.iter_mut().zip(vocab_sizes) {
        *offset = running;
        running = running
            .checked_add(*size)
            .ok_or_else(|| law_error("engram head offsets overflow"))?;
    }

    Ok(offsets)
}

/// Trial division over roughly 170 candidates near 2e7.
fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    if candidate.is_multiple_of(2) {
        return candidate == 2;
    }

    let mut divisor = 3u64;

    while divisor.saturating_mul(divisor) <= candidate {
        if candidate.is_multiple_of(divisor) {
            return false;
        }

        divisor += 2;
    }

    true
}

fn law_error(message: impl Into<String>) -> CheckpointError {
    CheckpointError::source_binding(format!("Qwen3.8-Flash-Next {}", message.into()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::test_builder::SafeTensorTestBuilder;
    use crate::common::test_support::sources::{fixture_path, write_safetensors_payload};
    use crate::{CheckpointErrorCode, DType, SafeTensorFile};
    use std::fs;

    pub(crate) const PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding";

    /// Exact checkpoint words, independent of `compute()`.
    pub(crate) const MULTIPLIERS: [i64; 3] =
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
    pub(crate) const VOCAB_SIZES: [i64; 16] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    pub(crate) const OFFSETS: [i64; 16] = [
        0,
        20_000_003,
        40_000_026,
        60_000_059,
        80_000_106,
        100_000_165,
        120_000_228,
        140_000_297,
        160_000_374,
        180_000_455,
        200_000_548,
        220_000_655,
        240_000_802,
        260_000_955,
        280_001_114,
        300_001_275,
    ];

    pub(crate) fn engram_constant_fixture(
        multipliers: [i64; 3],
        vocab_sizes: [i64; 16],
        offsets: [i64; 16],
    ) -> SafeTensorTestBuilder {
        let mut fixture = SafeTensorTestBuilder::new();

        for (name, words) in [
            ("layer_multipliers", multipliers.as_slice()),
            ("ngram_heads_vocab_sizes", vocab_sizes.as_slice()),
            ("ngram_heads_offsets", offsets.as_slice()),
        ] {
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();

            fixture.add_with(
                format!("{PREFIX}.{name}"),
                DType::I64,
                &[words.len()],
                |index| bytes[index],
            );
        }

        fixture
    }

    #[test]
    fn engram_table_prefix_admits_only_the_single_ple_layer() {
        assert_eq!(
            engram_table_prefix(F::PLE_LAYER, F::PLE_LAYER).unwrap(),
            "model.language_model.layers.1.ple.ple_embedding"
        );

        for layer in [0, 2, 47] {
            let error = engram_table_prefix(layer, F::PLE_LAYER).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("carries no engram"), "{error}");
        }
    }

    #[test]
    fn computed_law_reproduces_the_pinned_engram_constants() {
        let constants = Qwen38FlashNextEngramHashConstants::compute().unwrap();

        assert_eq!(constants.layer_multipliers, MULTIPLIERS);
        assert_eq!(constants.head_vocab_sizes, VOCAB_SIZES);
        assert_eq!(constants.head_offsets, OFFSETS);
        assert_eq!(constants.total_vocab, 320_001_446);
        assert_eq!(constants.padded_rows, 320_001_536);
    }

    #[cfg(feature = "qualification")]
    #[test]
    fn qualification_law_preserves_multipliers_and_derives_its_row_space() {
        let vocabularies = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        let constants =
            Qwen38FlashNextEngramHashConstants::for_qualification(vocabularies).unwrap();

        assert_eq!(constants.layer_multipliers, MULTIPLIERS);
        assert_eq!(constants.head_vocab_sizes, vocabularies);
        assert_eq!(constants.total_vocab, 381);
        assert_eq!(constants.padded_rows, 384);

        let mut invalid = vocabularies;
        invalid[7] = 0;
        assert!(Qwen38FlashNextEngramHashConstants::for_qualification(invalid).is_err());
    }

    #[test]
    fn multiplier_law_keeps_every_token_product_inside_the_signed_range() {
        let constants = Qwen38FlashNextEngramHashConstants::compute().unwrap();
        let multiplier_max = i64::MAX / F::VOCAB as i64;
        let largest_token = F::VOCAB as i64 - 1;

        for (position, multiplier) in constants.layer_multipliers.iter().copied().enumerate() {
            assert_eq!(multiplier % 2, 1, "multiplier {position} must be odd");
            assert!(multiplier < multiplier_max, "multiplier {position} bound");
            largest_token
                .checked_mul(multiplier)
                .expect("no token/multiplier product may overflow i64");
        }

        // Pin the maximum admitted token/multiplier product.
        assert_eq!(
            largest_token * constants.layer_multipliers[0],
            5_886_047_582_964_040_311
        );
    }

    #[test]
    fn head_offsets_are_the_exclusive_prefix_sums_of_ascending_primes() {
        let constants = Qwen38FlashNextEngramHashConstants::compute().unwrap();
        let mut running = 0i64;

        for (head, size) in constants.head_vocab_sizes.iter().copied().enumerate() {
            assert!(
                is_prime(size as u64),
                "head-{head} vocabulary must be prime"
            );
            assert!(size > F::NGRAM_VOCAB_BASE as i64 - 1);
            assert_eq!(constants.head_offsets[head], running);

            if head > 0 {
                assert!(size > constants.head_vocab_sizes[head - 1]);
            }

            running += size;
        }

        assert_eq!(running, constants.total_vocab);
        assert!(constants.padded_rows.is_multiple_of(F::NGRAM_VOCAB_DIVISOR));
        assert_eq!(constants.padded_rows - constants.total_vocab as usize, 90);
    }

    #[test]
    fn binds_engram_constants_that_agree_with_the_computed_law() {
        let path = fixture_path("qwen38_flash_next-engram-constants");
        engram_constant_fixture(MULTIPLIERS, VOCAB_SIZES, OFFSETS).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextEngramConstantBindings::bind_from(PREFIX, |name| file.tensor(name))
                .unwrap();

        assert_eq!(
            bindings.layer_multipliers.values().collect::<Vec<_>>(),
            MULTIPLIERS
        );
        assert_eq!(
            bindings.head_vocab_sizes.values().collect::<Vec<_>>(),
            VOCAB_SIZES
        );
        assert_eq!(bindings.head_offsets.values().collect::<Vec<_>>(), OFFSETS);
        assert_eq!(
            bindings.constants,
            Qwen38FlashNextEngramHashConstants::compute().unwrap()
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_engram_constants_that_disagree_with_the_computed_law() {
        // One-off drift in each buffer: a multiplier made even, a composite modulus, and an
        // offset that no longer closes the prefix sum.
        let mut drifted_multipliers = MULTIPLIERS;
        drifted_multipliers[1] += 1;
        let mut drifted_vocab = VOCAB_SIZES;
        drifted_vocab[3] += 2;
        let mut drifted_offsets = OFFSETS;
        drifted_offsets[15] -= 1;

        for (label, multipliers, vocab_sizes, offsets, expected) in [
            (
                "qwen38_flash_next-engram-multiplier",
                drifted_multipliers,
                VOCAB_SIZES,
                OFFSETS,
                "`layer_multipliers[1]` is 20109073645366",
            ),
            (
                "qwen38_flash_next-engram-vocab",
                MULTIPLIERS,
                drifted_vocab,
                OFFSETS,
                "`ngram_heads_vocab_sizes[3]` is 20000049",
            ),
            (
                "qwen38_flash_next-engram-offset",
                MULTIPLIERS,
                VOCAB_SIZES,
                drifted_offsets,
                "`ngram_heads_offsets[15]` is 300001274",
            ),
        ] {
            let path = fixture_path(label);
            engram_constant_fixture(multipliers, vocab_sizes, offsets).write(&path);
            let file = SafeTensorFile::open(&path).unwrap();
            let error =
                Qwen38FlashNextEngramConstantBindings::bind_from(PREFIX, |name| file.tensor(name))
                    .err()
                    .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains(expected), "{error}");
            assert!(
                error.to_string().contains("own hash law computes"),
                "{error}"
            );

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn refuses_engram_constant_buffers_that_are_not_i64_of_the_admitted_length() {
        let path = fixture_path("qwen38_flash_next-engram-shape");
        let (mut header, payload) =
            engram_constant_fixture(MULTIPLIERS, VOCAB_SIZES, OFFSETS).into_parts();
        header[&format!("{PREFIX}.ngram_heads_offsets")]["shape"] = serde_json::json!([8, 2]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error =
            Qwen38FlashNextEngramConstantBindings::bind_from(PREFIX, |name| file.tensor(name))
                .err()
                .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("shape [8, 2], expected [16]"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn prime_predicate_agrees_with_the_boundary_cases_the_search_walks() {
        for (candidate, prime) in [
            (0, false),
            (1, false),
            (2, true),
            (3, true),
            (4, false),
            (9, false),
            // The search starts above this prime.
            (19_999_999, true),
            (20_000_003, true),
            (20_000_004, false),
            (20_000_171, true),
        ] {
            assert_eq!(is_prime(candidate), prime, "{candidate}");
        }

        assert_ne!(
            Qwen38FlashNextEngramHashConstants::compute()
                .unwrap()
                .head_vocab_sizes[0],
            19_999_999
        );
    }
}
