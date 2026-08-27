//! Construct-once Flash-Next benchmark sweeps.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tuisko_engine::Qwen38FlashNextTextGenerator;
use tuisko_model::{CheckpointSnapshot, Qwen38FlashNext};
use tuisko_qual::{
    BenchmarkCellSelector, BenchmarkFingerprint, Qwen38FlashNextGenerationBenchmarkOptions,
    Qwen38FlashNextResidentBaselineCache, benchmark_device_preflight,
    benchmark_qwen38_flash_next_generation_loaded,
    benchmark_qwen38_flash_next_resident_model_loaded, parse_benchmark_cells,
    print_qwen38_flash_next_generation_benchmark, print_qwen38_flash_next_resident_benchmark,
    run_benchmark_sweeps,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = Options::parse(std::env::args().skip(1))?;
    benchmark_device_preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(
        &options.snapshot,
    )?);
    let started = std::time::Instant::now();
    let mut generator = Qwen38FlashNextTextGenerator::from_snapshot_device_zero(snapshot)?;
    let load = started.elapsed();
    println!("Flash-Next construct-once runner: model loaded in {load:.2?}");

    let fingerprint = options.fingerprint()?;
    let sweeps = options
        .sweeps
        .iter()
        .filter(|sweep| match sweep {
            Sweep::Resident => options.run_resident(),
            Sweep::Generation => options.run_generation(),
        })
        .map(|sweep| sweep.as_str())
        .collect::<Vec<_>>();
    run_benchmark_sweeps(&mut generator, sweeps, |sweep, generator| {
        match sweep {
            "resident" => {
                let cache = options
                    .baseline_cache
                    .as_deref()
                    .zip(fingerprint.as_ref())
                    .map(|(path, fingerprint)| Qwen38FlashNextResidentBaselineCache {
                        path,
                        fingerprint,
                    });
                let resident_cells = options.resident_cells();
                let (model, stream) = generator.qualification_program_and_stream();
                let report = benchmark_qwen38_flash_next_resident_model_loaded(
                    model,
                    stream,
                    load,
                    cache,
                    &resident_cells,
                )?;
                print_qwen38_flash_next_resident_benchmark(&report);
            }
            "generation" => {
                let report = benchmark_qwen38_flash_next_generation_loaded(
                    generator,
                    load,
                    &Qwen38FlashNextGenerationBenchmarkOptions {
                        prompt_tokens: options.generation_prompts()?,
                        max_cell_wall: options.max_cell_wall,
                    },
                )?;
                print_qwen38_flash_next_generation_benchmark(&report);
            }
            _ => unreachable!("sweeps were validated by Options::parse"),
        }
        Ok::<_, Box<dyn Error>>(())
    })?;

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sweep {
    Resident,
    Generation,
}

impl Sweep {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Generation => "generation",
        }
    }
}

struct Options {
    snapshot: PathBuf,
    sweeps: Vec<Sweep>,
    cells: Vec<BenchmarkCellSelector>,
    max_cell_wall: Option<Duration>,
    baseline_cache: Option<PathBuf>,
    base_sha: Option<String>,
    cuda_oxide_commit: Option<String>,
}

impl Options {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let snapshot = PathBuf::from(arguments.next().ok_or(
            "usage: bench-qwen38-flash-next SNAPSHOT [--sweeps resident,generation] [options]",
        )?);
        let mut options = Self {
            snapshot,
            sweeps: vec![Sweep::Resident, Sweep::Generation],
            cells: Vec::new(),
            max_cell_wall: None,
            baseline_cache: None,
            base_sha: None,
            cuda_oxide_commit: None,
        };
        while let Some(argument) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("`{argument}` requires a value"))?;
            match argument.as_str() {
                "--sweeps" => options.sweeps = parse_sweeps(&value)?,
                "--cells" => options.cells = parse_benchmark_cells(Some(&value))?,
                "--max-cell-seconds" => {
                    let seconds = value.parse::<f64>()?;
                    if !seconds.is_finite() || seconds <= 0.0 {
                        return Err("`--max-cell-seconds` must be finite and positive".into());
                    }
                    options.max_cell_wall = Some(Duration::from_secs_f64(seconds));
                }
                "--baseline-cache" => options.baseline_cache = Some(PathBuf::from(value)),
                "--base-sha" => options.base_sha = Some(value),
                "--cuda-oxide-commit" => options.cuda_oxide_commit = Some(value),
                _ => return Err(format!("unknown argument `{argument}`").into()),
            }
        }
        if options.baseline_cache.is_some()
            && (!options.sweeps.contains(&Sweep::Resident)
                || options.base_sha.is_none()
                || options.cuda_oxide_commit.is_none())
        {
            return Err("`--baseline-cache` requires the resident sweep, `--base-sha`, and the generator stamp".into());
        }
        if let Some(cell) = options.cells.iter().find(|cell| {
            !matches!(
                cell.route.as_str(),
                "decode" | "verify" | "prefill" | "generation" | "prompt"
            )
        }) {
            return Err(format!("no requested sweep recognizes cell `{}`", cell.original).into());
        }
        let selected_cell_runs = (options.sweeps.contains(&Sweep::Resident)
            && options.run_resident())
            || (options.sweeps.contains(&Sweep::Generation) && options.run_generation());
        if !options.cells.is_empty() && !selected_cell_runs {
            return Err("no requested sweep matches `--cells`".into());
        }

        Ok(options)
    }

    fn fingerprint(&self) -> Result<Option<BenchmarkFingerprint>, Box<dyn Error>> {
        let Some(_) = self.baseline_cache else {
            return Ok(None);
        };
        let revision = snapshot_revision(&self.snapshot)?;
        Ok(Some(BenchmarkFingerprint::record(
            revision,
            self.cuda_oxide_commit.as_deref().unwrap_or_default(),
            self.base_sha.as_deref().unwrap_or_default(),
        )?))
    }

    fn resident_cells(&self) -> Vec<BenchmarkCellSelector> {
        self.cells
            .iter()
            .filter(|cell| matches!(cell.route.as_str(), "decode" | "verify" | "prefill"))
            .cloned()
            .collect()
    }

    fn generation_prompts(&self) -> Result<Option<Vec<usize>>, Box<dyn Error>> {
        if self.cells.is_empty() {
            return Ok(None);
        }
        self.cells
            .iter()
            .filter(|cell| matches!(cell.route.as_str(), "generation" | "prompt"))
            .map(|cell| {
                cell.shape
                    .strip_prefix("T=")
                    .ok_or_else(|| format!("generation cell `{}` must use T=N", cell.original))?
                    .parse::<usize>()
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
            .map_err(Into::into)
    }

    fn run_resident(&self) -> bool {
        self.cells.is_empty() || !self.resident_cells().is_empty()
    }

    fn run_generation(&self) -> bool {
        self.cells.is_empty()
            || self
                .cells
                .iter()
                .any(|cell| matches!(cell.route.as_str(), "generation" | "prompt"))
    }
}

fn parse_sweeps(value: &str) -> Result<Vec<Sweep>, Box<dyn Error>> {
    let mut sweeps = Vec::new();
    for name in value.split(',') {
        let sweep = match name {
            "resident" => Sweep::Resident,
            "generation" => Sweep::Generation,
            _ => return Err(format!("unknown sweep `{name}`").into()),
        };
        if sweeps.contains(&sweep) {
            return Err(format!("sweep `{name}` is repeated").into());
        }
        sweeps.push(sweep);
    }
    if sweeps.is_empty() {
        return Err("`--sweeps` must be nonempty".into());
    }

    Ok(sweeps)
}

fn snapshot_revision(snapshot: &Path) -> Result<String, Box<dyn Error>> {
    let revision = snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("snapshot path has no UTF-8 revision component")?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("snapshot path must end in its 40-character revision".into());
    }

    Ok(revision.to_string())
}
