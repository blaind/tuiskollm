//! Pinned SGLang captures for the Qwen3.8 Flash-Next generation cross-check.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Directory the committed captures live in, relative to the `tuisko-qual` package root.
pub const QWEN38_FLASH_NEXT_GOLDEN_DIRECTORY: &str = "fixtures/qwen38-flash-next-golden-sglang";

/// Environment variable that relocates the capture set.
pub const QWEN38_FLASH_NEXT_GOLDEN_ENV: &str = "TUISKO_QWEN38_FLASH_NEXT_GOLDEN";

const GOLDEN_CANDIDATES: usize = 32;
const GOLDEN_PROMPT_STEPS: usize = 64;
const GOLDEN_BOUNDARY_STEPS: usize = 8;

/// Prompt captures, in file order.
pub const QWEN38_FLASH_NEXT_GOLDEN_PROMPTS: [&str; 8] = [
    "prompt-00",
    "prompt-01",
    "prompt-02",
    "prompt-03",
    "prompt-04",
    "prompt-05",
    "prompt-06",
    "prompt-07",
];

/// Boundary captures, by the first decode round's visible length.
pub const QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES: [usize; 8] =
    [2_047, 2_048, 2_049, 2_050, 2_051, 2_052, 2_056, 2_100];

/// Failure to read or admit the committed capture set.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextGoldenError {
    /// A capture file could not be read.
    #[error("could not read Flash-Next golden capture {path}: {source}")]
    Read {
        /// Capture that could not be read.
        path: PathBuf,
        /// Underlying I/O failure.
        source: std::io::Error,
    },

    /// A capture file was not the shape the gate reads.
    #[error("Flash-Next golden capture {path} is malformed: {source}")]
    Parse {
        /// Capture that could not be parsed.
        path: PathBuf,
        /// Underlying deserialization failure.
        source: serde_json::Error,
    },

    /// The capture set does not describe the checkpoint under test.
    #[error("Flash-Next golden capture set is inadmissible: {0}")]
    Inadmissible(String),
}

type GoldenResult<T> = Result<T, Qwen38FlashNextGoldenError>;

/// Provenance of one capture set.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38FlashNextGoldenMeta {
    /// Checkpoint repository the reference ran.
    pub model: String,
    /// Checkpoint revision, which must be the one under test.
    pub revision: String,
    /// Reference engine.
    pub engine: String,
    /// Reference engine version, exactly as it reported itself.
    pub engine_version: String,
    /// Tensor-parallel width the reference ran at.
    pub tensor_parallel_size: usize,
    /// Hardware the reference ran on.
    pub hardware: String,
    /// Whether the reference offloaded PLE to the CPU.
    pub ple_cpu_offload: Option<bool>,
    /// Ranked candidates recorded per step.
    pub topk_logprobs: usize,
    /// Whether the capture is greedy.
    pub greedy: bool,
    /// Context width the reference was configured for.
    pub max_model_len: usize,
}

impl Qwen38FlashNextGoldenMeta {
    /// Refuses captures outside the pinned reference authority.
    pub fn require_pinned_authority(&self, model: &str, revision: &str) -> GoldenResult<()> {
        let expected = [
            (self.model == model, "model"),
            (self.revision == revision, "revision"),
            (self.engine == "sglang", "engine"),
            (
                self.engine_version == "0.0.0.dev1+gd91c3682b",
                "engine version",
            ),
            (self.tensor_parallel_size == 1, "tensor-parallel width"),
            (self.hardware == "1x NVIDIA B300 SXM6 AC", "hardware"),
            (self.ple_cpu_offload.is_none(), "PLE CPU offload"),
            (self.topk_logprobs == GOLDEN_CANDIDATES, "candidate count"),
            (self.greedy, "sampling mode"),
            (self.max_model_len == 8_192, "context width"),
        ];
        if let Some((_, field)) = expected.into_iter().find(|(matches, _)| !matches) {
            return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                "capture {field} does not match the pinned SGLang authority"
            )));
        }

        Ok(())
    }

    /// One line naming the authority, for the report.
    pub fn provenance(&self) -> String {
        format!(
            "{} {} on {} (TP{}), {} @ {}, top-{} logprobs, greedy",
            self.engine,
            self.engine_version,
            self.hardware,
            self.tensor_parallel_size,
            self.model,
            self.revision,
            self.topk_logprobs
        )
    }
}

/// Ranked candidates in either committed capture schema.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum RawCandidates {
    /// Parallel-array form: bare token ids, with logprobs in a sibling field.
    Ids(Vec<u32>),
    /// Paired form: `[token, logprob]` per candidate.
    Pairs(Vec<(u32, f64)>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStep {
    top_ids: RawCandidates,
    #[serde(default)]
    top_logprobs: Vec<f64>,
    #[serde(default)]
    chosen: Option<u32>,
}

/// One recorded step of a capture.
#[derive(Clone, Debug, Deserialize)]
#[serde(from = "RawStep")]
pub struct Qwen38FlashNextGoldenStep {
    /// Ranked candidate token ids, most probable first.
    pub top_ids: Vec<u32>,
    /// Log probabilities paired with `top_ids`.
    pub top_logprobs: Vec<f64>,
    /// Token the reference selected, when it recorded one.
    pub chosen: Option<u32>,
}

impl From<RawStep> for Qwen38FlashNextGoldenStep {
    fn from(raw: RawStep) -> Self {
        match raw.top_ids {
            RawCandidates::Ids(top_ids) => Self {
                top_ids,
                top_logprobs: raw.top_logprobs,
                chosen: raw.chosen,
            },
            RawCandidates::Pairs(pairs) => Self {
                top_ids: pairs.iter().map(|&(token, _)| token).collect(),
                top_logprobs: pairs.iter().map(|&(_, logprob)| logprob).collect(),
                chosen: raw.chosen,
            },
        }
    }
}

impl Qwen38FlashNextGoldenStep {
    /// Rank of one token in this step's recorded candidates.
    pub fn rank_of(&self, token: u32) -> Option<usize> {
        self.top_ids
            .iter()
            .position(|&candidate| candidate == token)
    }

    /// Log probability the reference assigned one token, when it recorded one.
    pub fn logprob_of(&self, token: u32) -> Option<f64> {
        self.rank_of(token)
            .and_then(|rank| self.top_logprobs.get(rank).copied())
    }

    /// Whether `token` reaches the second-ranked probability, including boundary ties.
    pub fn within_leading_pair(&self, token: u32) -> bool {
        let Some(&second) = self.top_logprobs.get(1) else {
            return self.rank_of(token).is_some_and(|rank| rank < 2);
        };
        self.logprob_of(token)
            .is_some_and(|logprob| logprob >= second)
    }

    /// Gap between the two most probable candidates.
    pub fn top_margin(&self) -> Option<f64> {
        match (self.top_logprobs.first(), self.top_logprobs.get(1)) {
            (Some(&first), Some(&second)) => Some(first - second),
            _ => None,
        }
    }
}

/// One capture: a prompt, what the reference generated from it, and every step's candidates.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen38FlashNextGoldenCapture {
    /// Prompt token ids, already tokenized by the reference.
    pub prompt_ids: Vec<u32>,
    /// Tokens the reference generated, in order.
    pub generated_ids: Vec<u32>,
    /// Ranked candidates per generated token.
    pub steps: Vec<Qwen38FlashNextGoldenStep>,
    /// Decoded continuation, present in the prompt captures.
    #[serde(default)]
    pub text: Option<String>,
    /// Keys the reference's first *decode* round attends, present in the boundary captures.
    #[serde(default)]
    pub total_length_at_first_step: Option<usize>,
}

impl Qwen38FlashNextGoldenCapture {
    /// Checks step counts, ranked candidates, and recorded choices.
    pub fn require_self_consistent(&self, name: &str) -> GoldenResult<()> {
        if self.steps.len() != self.generated_ids.len() {
            return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                "{name} recorded {} steps for {} generated tokens",
                self.steps.len(),
                self.generated_ids.len()
            )));
        }
        if self.generated_ids.is_empty() {
            return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                "{name} records no generated tokens"
            )));
        }
        for (field, tokens) in [
            ("prompt", self.prompt_ids.as_slice()),
            ("generated", self.generated_ids.as_slice()),
        ] {
            if let Some(&token) = tokens
                .iter()
                .find(|&&token| token as usize >= Qwen38FlashNext::VOCAB)
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} carries {field} token {token} outside 0..{}",
                    Qwen38FlashNext::VOCAB
                )));
            }
        }
        for (index, step) in self.steps.iter().enumerate() {
            if step.top_ids.len() != GOLDEN_CANDIDATES
                || step.top_logprobs.len() != GOLDEN_CANDIDATES
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} records {} candidate ids and {} logprobs, expected \
                     {GOLDEN_CANDIDATES} of each",
                    step.top_ids.len(),
                    step.top_logprobs.len()
                )));
            }
            if step
                .top_ids
                .iter()
                .enumerate()
                .any(|(left, token)| step.top_ids[left + 1..].contains(token))
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} repeats a candidate token"
                )));
            }
            if let Some(&token) = step
                .top_ids
                .iter()
                .find(|&&token| token as usize >= Qwen38FlashNext::VOCAB)
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} carries candidate {token} outside 0..{}",
                    Qwen38FlashNext::VOCAB
                )));
            }
            if step.top_logprobs.iter().any(|value| !value.is_finite()) {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} records a non-finite logprob"
                )));
            }
            if step.top_logprobs.windows(2).any(|pair| pair[0] < pair[1]) {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} candidates are not probability-ranked"
                )));
            }
            if step.rank_of(self.generated_ids[index]).is_none() {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} omits generated token {} from its candidates",
                    self.generated_ids[index]
                )));
            }
            if step
                .rank_of(self.generated_ids[index])
                .is_none_or(|rank| rank >= 2)
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} generated token {} outside its leading pair",
                    self.generated_ids[index]
                )));
            }
            if let Some(chosen) = step.chosen
                && chosen != self.generated_ids[index]
            {
                return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                    "{name} step {index} recorded {chosen} as chosen but generated {}",
                    self.generated_ids[index]
                )));
            }
        }
        if self.prompt_ids.is_empty() {
            return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                "{name} carries an empty prompt"
            )));
        }

        Ok(())
    }

    /// Steps whose top two reference candidates are tied.
    pub fn exact_ties(&self) -> Vec<usize> {
        self.steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| (step.top_margin() == Some(0.0)).then_some(index))
            .collect()
    }
}

/// Resolves the capture directory: the environment override, else the committed default.
pub fn qwen38_flash_next_golden_directory() -> PathBuf {
    match std::env::var_os(QWEN38_FLASH_NEXT_GOLDEN_ENV) {
        Some(directory) => PathBuf::from(directory),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(QWEN38_FLASH_NEXT_GOLDEN_DIRECTORY),
    }
}

/// Reads the capture set's provenance.
pub fn load_qwen38_flash_next_golden_meta(
    directory: &Path,
) -> GoldenResult<Qwen38FlashNextGoldenMeta> {
    read_json(&directory.join("meta.json"))
}

/// Reads one capture by stem, without the `.json`.
pub fn load_qwen38_flash_next_golden_capture(
    directory: &Path,
    stem: &str,
) -> GoldenResult<Qwen38FlashNextGoldenCapture> {
    let capture: Qwen38FlashNextGoldenCapture = read_json(&directory.join(format!("{stem}.json")))?;
    capture.require_self_consistent(stem)?;
    if QWEN38_FLASH_NEXT_GOLDEN_PROMPTS.contains(&stem) {
        if capture.steps.len() != GOLDEN_PROMPT_STEPS || capture.text.is_none() {
            return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
                "{stem} must carry {GOLDEN_PROMPT_STEPS} steps and decoded text"
            )));
        }
    } else if stem.starts_with("boundary-") && capture.steps.len() != GOLDEN_BOUNDARY_STEPS {
        return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
            "{stem} must carry {GOLDEN_BOUNDARY_STEPS} boundary steps"
        )));
    }

    Ok(capture)
}

/// Reads one boundary capture and checks its name is the length it says it probes.
pub fn load_qwen38_flash_next_golden_boundary(
    directory: &Path,
    visible: usize,
) -> GoldenResult<Qwen38FlashNextGoldenCapture> {
    let stem = format!("boundary-{visible}");
    let capture = load_qwen38_flash_next_golden_capture(directory, &stem)?;
    let recorded = capture.total_length_at_first_step.ok_or_else(|| {
        Qwen38FlashNextGoldenError::Inadmissible(format!(
            "{stem} records no total_length_at_first_step, so what it probes is only its filename"
        ))
    })?;
    if recorded != visible {
        return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
            "{stem} probes visible length {recorded}, not the {visible} its name claims"
        )));
    }
    if recorded != capture.prompt_ids.len() + 1 {
        return Err(Qwen38FlashNextGoldenError::Inadmissible(format!(
            "{stem} claims visible length {recorded} for a {}-token prompt; the first decode \
             round sees one key more than the prompt, so these cannot both be true",
            capture.prompt_ids.len()
        )));
    }

    Ok(capture)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> GoldenResult<T> {
    let text =
        std::fs::read_to_string(path).map_err(|source| Qwen38FlashNextGoldenError::Read {
            path: path.to_path_buf(),
            source,
        })?;

    serde_json::from_str(&text).map_err(|source| Qwen38FlashNextGoldenError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_capture_set_is_present_and_names_its_authority() {
        let directory = qwen38_flash_next_golden_directory();
        let meta = load_qwen38_flash_next_golden_meta(&directory).unwrap();

        assert_eq!(meta.engine, "sglang");
        assert_eq!(meta.engine_version, "0.0.0.dev1+gd91c3682b");
        assert_eq!(meta.tensor_parallel_size, 1);
        assert_eq!(meta.topk_logprobs, 32);
        assert!(meta.greedy);
        assert_eq!(meta.revision, "7b719225242aacd3dbd3f9407468c2ee9a9d2594");
        assert!(meta.provenance().contains("B300"));
    }

    #[test]
    fn a_capture_set_from_another_revision_is_refused() {
        let meta =
            load_qwen38_flash_next_golden_meta(&qwen38_flash_next_golden_directory()).unwrap();

        meta.require_pinned_authority(
            "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        )
        .unwrap();
        let error = meta
            .require_pinned_authority("RadixArk/Qwen3.8-Flash-Next-NVFP4", "deadbeef")
            .unwrap_err();
        assert!(error.to_string().contains("revision"));

        let mut offloaded = meta;
        offloaded.ple_cpu_offload = Some(true);
        let error = offloaded
            .require_pinned_authority(
                "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
            )
            .unwrap_err();
        assert!(error.to_string().contains("PLE CPU offload"));
    }

    #[test]
    fn malformed_ranked_candidates_are_refused() {
        let directory = qwen38_flash_next_golden_directory();
        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();

        capture.steps[0].top_ids.pop();
        let error = capture.require_self_consistent("short").unwrap_err();
        assert!(error.to_string().contains("expected 32"), "{error}");

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        capture.steps[0].top_ids[1] = capture.steps[0].top_ids[0];
        let error = capture.require_self_consistent("duplicate").unwrap_err();
        assert!(error.to_string().contains("repeats"), "{error}");

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        capture.steps[0].top_logprobs.swap(0, 1);
        let error = capture.require_self_consistent("unsorted").unwrap_err();
        assert!(error.to_string().contains("probability-ranked"), "{error}");

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        capture.steps[0].top_logprobs[0] = f64::NAN;
        let error = capture.require_self_consistent("non-finite").unwrap_err();
        assert!(error.to_string().contains("non-finite"), "{error}");

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        let replacement = (0..Qwen38FlashNext::VOCAB as u32)
            .find(|token| !capture.steps[0].top_ids.contains(token))
            .unwrap();
        capture.steps[0].top_ids[0] = replacement;
        let error = capture.require_self_consistent("missing").unwrap_err();
        assert!(
            error.to_string().contains("omits generated token"),
            "{error}"
        );

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        capture.steps[0].top_ids[0] = u32::MAX;
        let error = capture
            .require_self_consistent("outside-vocab")
            .unwrap_err();
        assert!(error.to_string().contains("outside 0..248320"), "{error}");

        let mut capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-00").unwrap();
        let third = capture.steps[0].top_ids[2];
        capture.generated_ids[0] = third;
        capture.steps[0].chosen = Some(third);
        let error = capture.require_self_consistent("not-leading").unwrap_err();
        assert!(
            error.to_string().contains("outside its leading pair"),
            "{error}"
        );
    }

    #[test]
    fn every_prompt_capture_is_sixty_four_self_consistent_greedy_steps() {
        let directory = qwen38_flash_next_golden_directory();
        for stem in QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
            let capture = load_qwen38_flash_next_golden_capture(&directory, stem).unwrap();

            assert_eq!(capture.generated_ids.len(), 64, "{stem}");
            assert_eq!(capture.steps.len(), 64, "{stem}");
            assert!(capture.text.is_some(), "{stem}");
            for step in &capture.steps {
                assert_eq!(step.top_ids.len(), 32, "{stem}");
                assert_eq!(step.top_logprobs.len(), 32, "{stem}");
                assert!(step.chosen.is_some(), "{stem}");
            }
        }
    }

    #[test]
    fn the_reference_tie_inventory_is_pinned() {
        let directory = qwen38_flash_next_golden_directory();
        let expected = [
            ("prompt-00", vec![1, 2]),
            ("prompt-04", vec![13]),
            ("prompt-05", vec![4]),
            ("prompt-07", vec![11, 44]),
        ];

        for stem in QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
            let capture = load_qwen38_flash_next_golden_capture(&directory, stem).unwrap();
            let expected = expected
                .iter()
                .find_map(|(candidate, ties)| (*candidate == stem).then_some(ties.as_slice()))
                .unwrap_or_default();
            assert_eq!(capture.exact_ties(), expected, "{stem}");
        }

        let capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-04").unwrap();
        let step = &capture.steps[13];
        assert_eq!(&step.top_ids[..2], &[7_806, 3_095]);
        assert_eq!(capture.generated_ids[13], 3_095);

        let capture = load_qwen38_flash_next_golden_capture(&directory, "prompt-07").unwrap();
        let step = &capture.steps[55];
        assert_eq!(step.top_logprobs[1], step.top_logprobs[2]);
        assert!(step.within_leading_pair(step.top_ids[2]));
    }

    #[test]
    fn no_prompt_capture_ever_selected_a_stop_token() {
        // Every capture runs its full sixty-four tokens, so a run of ours that stops early has
        // diverged rather than finished, so the gate must not compare only a prefix.
        use tuisko_frontend::TokenizedSchema;
        let stops = <tuisko_model::Qwen38FlashNext as TokenizedSchema>::EOS_IDS;
        assert_eq!(stops, [248_046, 248_044]);

        let directory = qwen38_flash_next_golden_directory();
        for stem in QWEN38_FLASH_NEXT_GOLDEN_PROMPTS {
            let capture = load_qwen38_flash_next_golden_capture(&directory, stem).unwrap();
            for &token in &capture.generated_ids {
                assert!(
                    !stops.contains(&token),
                    "{stem} generated stop token {token}"
                );
            }
        }
    }

    #[test]
    fn every_boundary_capture_probes_one_past_its_prompt() {
        let directory = qwen38_flash_next_golden_directory();
        for visible in QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES {
            let capture = load_qwen38_flash_next_golden_boundary(&directory, visible).unwrap();

            assert_eq!(capture.prompt_ids.len() + 1, visible);
            assert_eq!(capture.generated_ids.len(), 8);
            assert_eq!(capture.steps.len(), 8);
            // The boundary captures pack their candidates as pairs; both families must reach the
            // reader as ranked ids beside ranked logprobs or the two halves of the sweep would be
            // scored by different rules.
            for step in &capture.steps {
                assert_eq!(step.top_ids.len(), 32, "boundary-{visible}");
                assert_eq!(step.top_logprobs.len(), 32, "boundary-{visible}");
                assert!(step.top_margin().is_some(), "boundary-{visible}");
            }
        }
    }

    #[test]
    fn the_boundary_sweep_straddles_the_dense_ceiling_in_both_directions() {
        // Five inside, three outside. A sweep entirely on one side of a threshold proves the
        // threshold is never reached rather than that it is right.
        let ceiling = 2_051usize;
        let inside = QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES
            .iter()
            .filter(|&&visible| visible <= ceiling)
            .count();

        assert_eq!(inside, 5);
        assert_eq!(QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES.len() - inside, 3);
        assert!(QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES.contains(&ceiling));
        assert!(QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES.contains(&(ceiling + 1)));
    }

    #[test]
    fn a_missing_capture_names_the_file_rather_than_the_directory() {
        let error = load_qwen38_flash_next_golden_capture(
            &qwen38_flash_next_golden_directory(),
            "prompt-99",
        )
        .unwrap_err();

        assert!(error.to_string().contains("prompt-99.json"), "{error}");
    }
}
