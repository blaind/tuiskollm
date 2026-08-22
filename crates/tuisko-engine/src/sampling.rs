//! Exact-vocabulary CPU sampling.

use crate::{EngineError, EngineResult};
use std::cmp::Ordering;
use tuisko_model::{Arch, Qwen38_27B};

const DEFAULT_TOP_K: usize = 20;
const DEFAULT_TOP_P: f32 = 0.95;
const RNG_SCRAMBLE: u64 = 0x9e37_79b9_7f4a_7c15;

/// Per-request sampling controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    /// Softmax temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Cumulative probability retained after top-k selection.
    pub top_p: f32,
    /// Highest-logit candidates considered before top-p truncation.
    pub top_k: usize,
    /// Deterministic random seed for sampled decoding.
    pub seed: u64,
}

impl SamplingOptions {
    /// Exact greedy route.
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            seed: 0,
        }
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
            seed: 0,
        }
    }
}

/// One selected token and its registered-stop status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleDecision {
    /// Selected vocabulary token.
    pub token_id: u32,
    /// Whether the token is one of the admitted generation stop IDs.
    pub stopped: bool,
}

/// Stateful sampler for one generation request.
#[derive(Debug)]
pub struct Sampler {
    options: SamplingOptions,
    stop_ids: [u32; 2],
    random: XorShift64,
}

impl Sampler {
    /// Validates one request's options and stop-token pair.
    pub fn new(options: SamplingOptions, stop_ids: [u32; 2]) -> EngineResult<Self> {
        validate_options(options)?;
        validate_stop_ids(stop_ids)?;

        Ok(Self {
            options,
            stop_ids,
            random: XorShift64::new(options.seed),
        })
    }

    /// Selects one token from a complete BF16 vocabulary row.
    pub fn sample(&mut self, logits: &[u16]) -> EngineResult<SampleDecision> {
        require_vocabulary_row(logits)?;
        let token = match sampling_route(self.options) {
            SamplingRoute::Greedy => greedy_index(logits)?,
            SamplingRoute::TopKTopP => sampled_index(logits, self.options, &mut self.random)?,
        };
        let token_id =
            u32::try_from(token).map_err(|_| EngineError::sampling("sampled token exceeds u32"))?;

        Ok(SampleDecision {
            token_id,
            stopped: self.stop_ids.contains(&token_id),
        })
    }

    /// Validated controls used by this request.
    pub const fn options(&self) -> SamplingOptions {
        self.options
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplingRoute {
    Greedy,
    TopKTopP,
}

fn sampling_route(options: SamplingOptions) -> SamplingRoute {
    if options.temperature == 0.0 || options.top_k == 1 {
        SamplingRoute::Greedy
    } else {
        SamplingRoute::TopKTopP
    }
}

fn validate_options(options: SamplingOptions) -> EngineResult<()> {
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(EngineError::sampling(
            "temperature must be finite and non-negative",
        ));
    }
    if !options.top_p.is_finite() || !(0.0..=1.0).contains(&options.top_p) {
        return Err(EngineError::sampling("top_p must be finite and in 0..=1"));
    }
    if !(1..=Qwen38_27B::VOCAB).contains(&options.top_k) {
        return Err(EngineError::sampling(format!(
            "top_k must be in 1..={}",
            Qwen38_27B::VOCAB
        )));
    }

    Ok(())
}

fn validate_stop_ids(stop_ids: [u32; 2]) -> EngineResult<()> {
    if stop_ids[0] == stop_ids[1] {
        return Err(EngineError::sampling(
            "generation stop-token identifiers must be distinct",
        ));
    }
    for token in stop_ids {
        if usize::try_from(token).map_or(true, |token| token >= Qwen38_27B::VOCAB) {
            return Err(EngineError::sampling(format!(
                "generation stop token {token} is outside vocabulary 0..{}",
                Qwen38_27B::VOCAB
            )));
        }
    }

    Ok(())
}

fn require_vocabulary_row(logits: &[u16]) -> EngineResult<()> {
    if logits.len() != Qwen38_27B::VOCAB {
        return Err(EngineError::sampling(format!(
            "sampling requires {} BF16 logits, got {}",
            Qwen38_27B::VOCAB,
            logits.len()
        )));
    }

    Ok(())
}

fn greedy_index(logits: &[u16]) -> EngineResult<usize> {
    let mut best = None;
    for (token, &bits) in logits.iter().enumerate() {
        let value = finite_logit(bits, token)?;
        if best.is_none_or(|(_, best_value): (usize, f32)| value.total_cmp(&best_value).is_gt()) {
            best = Some((token, value));
        }
    }

    best.map(|(token, _)| token)
        .ok_or_else(|| EngineError::sampling("sampling received an empty logit row"))
}

fn sampled_index(
    logits: &[u16],
    options: SamplingOptions,
    random: &mut XorShift64,
) -> EngineResult<usize> {
    let top = top_k_select(logits, options.top_k)?;
    let maximum = top[0].1;
    let weights = top
        .iter()
        .map(|&(_, value)| ((value - maximum) / options.temperature).exp() as f64)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let cutoff = options.top_p.max(f32::MIN_POSITIVE) as f64 * total;
    let mut retained_total = 0.0;
    let mut retained = 0;
    for &weight in &weights {
        retained_total += weight;
        retained += 1;
        if retained_total >= cutoff {
            break;
        }
    }
    if retained == 0 || !retained_total.is_finite() || retained_total <= 0.0 {
        return Err(EngineError::sampling(
            "sampling distribution has no finite probability mass",
        ));
    }

    let threshold = random.unit_f64() * retained_total;
    let mut cumulative = 0.0;
    for (&(token, _), &weight) in top.iter().zip(&weights).take(retained) {
        cumulative += weight;
        if cumulative >= threshold {
            return Ok(token);
        }
    }

    Ok(top[retained - 1].0)
}

fn top_k_select(logits: &[u16], k: usize) -> EngineResult<Vec<(usize, f32)>> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (token, &bits) in logits.iter().enumerate() {
        let value = finite_logit(bits, token)?;
        if top.len() == k && !value.total_cmp(&top[top.len() - 1].1).is_gt() {
            continue;
        }
        let insertion =
            top.partition_point(|&(_, retained)| retained.total_cmp(&value) != Ordering::Less);
        top.insert(insertion, (token, value));
        if top.len() > k {
            top.pop();
        }
    }

    Ok(top)
}

fn finite_logit(bits: u16, token: usize) -> EngineResult<f32> {
    let value = f32::from_bits(u32::from(bits) << 16);
    if !value.is_finite() {
        return Err(EngineError::sampling(format!(
            "sampling received a non-finite logit at token {token}"
        )));
    }

    Ok(value)
}

#[derive(Debug)]
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = seed ^ RNG_SCRAMBLE;
        Self(if state == 0 { RNG_SCRAMBLE } else { state })
    }

    fn unit_f64(&mut self) -> f64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        (value >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TOP_K, DEFAULT_TOP_P, RNG_SCRAMBLE, SampleDecision, Sampler, SamplingOptions,
        SamplingRoute, XorShift64, sampling_route, top_k_select,
    };
    use crate::EngineErrorCode;
    use std::cmp::Ordering;
    use tuisko_model::{Arch, Qwen38_27B};

    const STOP_IDS: [u32; 2] = [248_046, 248_044];

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn logits() -> Vec<u16> {
        vec![bf16(-4.0); Qwen38_27B::VOCAB]
    }

    #[test]
    fn sampling_routes_are_exact() {
        let cases = [
            (0.0, 20, SamplingRoute::Greedy),
            (1.0, 1, SamplingRoute::Greedy),
            (f32::MIN_POSITIVE, 2, SamplingRoute::TopKTopP),
            (1.0, 20, SamplingRoute::TopKTopP),
        ];
        for (temperature, top_k, expected) in cases {
            assert_eq!(
                sampling_route(SamplingOptions {
                    temperature,
                    top_k,
                    ..SamplingOptions::default()
                }),
                expected
            );
        }
    }

    #[test]
    fn defaults_match_the_checkpoint_generation_contract() {
        let options = SamplingOptions::default();

        assert_eq!(options.temperature, 1.0);
        assert_eq!(options.top_p, DEFAULT_TOP_P);
        assert_eq!(options.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn greedy_keeps_the_first_maximum_and_marks_stop_tokens() {
        let mut row = logits();
        row[17] = bf16(3.0);
        row[STOP_IDS[0] as usize] = bf16(3.0);
        let mut sampler = Sampler::new(SamplingOptions::greedy(), STOP_IDS).unwrap();

        assert_eq!(
            sampler.sample(&row).unwrap(),
            SampleDecision {
                token_id: 17,
                stopped: false
            }
        );

        row[STOP_IDS[0] as usize] = bf16(4.0);
        assert_eq!(
            sampler.sample(&row).unwrap(),
            SampleDecision {
                token_id: STOP_IDS[0],
                stopped: true
            }
        );
    }

    #[test]
    fn top_k_matches_an_independent_full_sort() {
        let mut seed = 0x8b8b_8b8b_8b8b_8b8b_u64;
        let mut row = Vec::with_capacity(4_096);
        for _ in 0..4_096 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            row.push(bf16(((seed >> 40) as i32 - 32_768) as f32 / 1_024.0));
        }
        let mut expected = row
            .iter()
            .enumerate()
            .map(|(token, &bits)| (token, f32::from_bits(u32::from(bits) << 16)))
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        expected.truncate(20);

        assert_eq!(top_k_select(&row, 20).unwrap(), expected);
    }

    #[test]
    fn sampled_route_never_leaves_top_k() {
        let mut row = logits();
        row[3] = bf16(3.0);
        row[5] = bf16(2.0);
        row[7] = bf16(1.0);
        for seed in 0..64 {
            let mut sampler = Sampler::new(
                SamplingOptions {
                    top_p: 1.0,
                    top_k: 2,
                    seed,
                    ..SamplingOptions::default()
                },
                STOP_IDS,
            )
            .unwrap();
            assert!(matches!(sampler.sample(&row).unwrap().token_id, 3 | 5));
        }
    }

    #[test]
    fn deterministic_seed_pins_the_sampled_token() {
        let mut row = logits();
        row[3] = bf16(2.0);
        row[5] = bf16(1.0);
        let options = SamplingOptions {
            top_p: 1.0,
            top_k: 2,
            seed: 7,
            ..SamplingOptions::default()
        };
        let mut sampler = Sampler::new(options, STOP_IDS).unwrap();

        assert_eq!(sampler.sample(&row).unwrap().token_id, 5);
    }

    #[test]
    fn invalid_options_and_rows_keep_sampling_error_context() {
        let bad_options = [
            SamplingOptions {
                temperature: f32::NAN,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                temperature: -1.0,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_p: 1.1,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_p: f32::NAN,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_k: 0,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_k: Qwen38_27B::VOCAB + 1,
                ..SamplingOptions::default()
            },
        ];
        for options in bad_options {
            assert_eq!(
                Sampler::new(options, STOP_IDS).unwrap_err().code(),
                Some(EngineErrorCode::Sampling)
            );
        }
        for stop_ids in [[1, 1], [0, Qwen38_27B::VOCAB as u32]] {
            assert_eq!(
                Sampler::new(SamplingOptions::default(), stop_ids)
                    .unwrap_err()
                    .code(),
                Some(EngineErrorCode::Sampling)
            );
        }

        let mut sampler = Sampler::new(SamplingOptions::greedy(), STOP_IDS).unwrap();
        let short = vec![0; Qwen38_27B::VOCAB - 1];
        assert_eq!(
            sampler.sample(&short).unwrap_err().code(),
            Some(EngineErrorCode::Sampling)
        );

        let mut non_finite = logits();
        non_finite[37] = 0x7fc0;
        let error = sampler.sample(&non_finite).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Sampling));
        assert!(error.to_string().contains("token 37"));
    }

    #[test]
    fn random_draw_stays_in_the_unit_interval() {
        let mut random = XorShift64::new(9);
        for _ in 0..1_024 {
            let value = random.unit_f64();
            assert!(value >= 0.0);
            assert!(value < 1.0);
        }

        let mut formerly_zero = XorShift64::new(RNG_SCRAMBLE);
        assert!(formerly_zero.unit_f64() > 0.0);
    }

    #[test]
    fn oracle_sort_declares_its_tie_order() {
        let mut values = [(2usize, 1.0f32), (1, 1.0), (3, -0.0), (4, 0.0)];
        values.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        assert_eq!(values, [(1, 1.0), (2, 1.0), (4, 0.0), (3, -0.0)]);
        assert_eq!(1.0f32.total_cmp(&1.0), Ordering::Equal);
    }
}
