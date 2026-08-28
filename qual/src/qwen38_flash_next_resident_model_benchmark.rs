//! End-to-end timing for the Qwen3.8 Flash-Next resident program.
//!
//! A step includes 49 graph replays and 48 host-resolved streaming rounds, so this measures the
//! whole host-observed boundary. Results are diagnostic and are not leaf-graph baselines.

use crate::device_benchmark::DeviceBenchmarkError;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tuisko_engine::{
    MAX_BATCH, QWEN38_FLASH_NEXT_ATTENTION_LAYERS, QWEN38_FLASH_NEXT_PREFILL_ROWS,
    QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, Qwen38FlashNextResidentModel,
    Qwen38FlashNextStepTelemetry,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError};
use tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

/// Warm passes before any measurement; three, for the lazy module-scratch release.
const WARM_PASSES: usize = 3;

/// Timed steps per measured route.
const MEASURED_STEPS: usize = 16;

const DECODE_ROWS: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteAccounting {
    layer_rounds: usize,
    expert_requests: usize,
    expert_bytes_routed: usize,
    embedding_h2d_bytes: usize,
    engram_h2d_bytes: usize,
    engram_rows: usize,
    kv_append_bytes: usize,
    segment_replays: usize,
    expert_readbacks: usize,
}

/// One measured route's wall-clock summary and the telemetry that explains it.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextResidentRouteReport {
    /// `"decode"` or `"prefill"`.
    pub kind: &'static str,
    /// Rows the route carried.
    pub rows: usize,
    /// Median whole-step wall time.
    pub median: Duration,
    /// Fastest measured step.
    pub fastest: Duration,
    /// Slowest measured step.
    pub slowest: Duration,
    /// Tokens per second at the median.
    pub tokens_per_second: f64,
    /// Milliseconds per token at the median.
    pub milliseconds_per_token: f64,
    /// Microseconds per decoder layer at the median, for reading against the layer medians.
    pub microseconds_per_layer: f64,
    /// Expert selections the whole stack made.
    pub expert_requests: usize,
    /// Host-to-device expert bytes the step streamed.
    pub expert_h2d_bytes: usize,
    /// Streamed bytes per token.
    pub expert_h2d_bytes_per_token: usize,
    /// Whole-stack expert hit rate.
    pub expert_hit_rate: f64,
    /// Per-layer hit rate in stack order.
    pub layer_hit_rates: Vec<f64>,
    /// Per-layer streamed bytes in stack order.
    pub layer_h2d_bytes: Vec<usize>,
    /// Token-embedding bytes uploaded.
    pub embedding_h2d_bytes: usize,
    /// Engram FP8 bytes uploaded.
    pub engram_h2d_bytes: usize,
    /// Bytes appended to the paged K/V planes.
    pub kv_append_bytes: usize,
}

/// The whole diagnostic sweep.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextResidentBenchmarkReport {
    /// Captured executables the program retains.
    pub executables: usize,
    /// Wall time capturing them took.
    pub capture: Duration,
    /// Wall time the weight sweep and expert staging took.
    pub weight_upload: Duration,
    /// Wall time page-locking the pool's host classes took.
    pub host_pin: Duration,
    /// Wall time the whole construction took.
    pub load: Duration,
    /// Every measured route, decode ascending then prefill ascending.
    pub routes: Vec<Qwen38FlashNextResidentRouteReport>,
}

/// Loads the model once and sweeps every admitted decode batch and prefill tile.
pub fn benchmark_qwen38_flash_next_resident_model(
    root: &Path,
) -> Result<Qwen38FlashNextResidentBenchmarkReport, DeviceBenchmarkError> {
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;

    let started = std::time::Instant::now();
    let mut model = Qwen38FlashNextResidentModel::from_snapshot(&context, snapshot)?;
    let load = started.elapsed();
    let stats = model.load_stats();

    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut routes = Vec::new();

    for batch in DECODE_ROWS {
        model.reset_state(&stream)?;
        routes.push(measure_decode(&mut model, &stream, batch)?);
    }
    for tile in QWEN38_FLASH_NEXT_PREFILL_ROWS {
        model.reset_state(&stream)?;
        routes.push(measure_prefill(&mut model, &stream, tile)?);
    }

    Ok(Qwen38FlashNextResidentBenchmarkReport {
        executables: stats.executables(),
        capture: stats.graph_capture(),
        weight_upload: stats.weight_upload(),
        host_pin: stats.host_pin(),
        load,
        routes,
    })
}

fn measure_decode(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &CudaStream,
    batch: usize,
) -> Result<Qwen38FlashNextResidentRouteReport, DeviceBenchmarkError> {
    let tokens = (0..batch)
        .map(|row| (1_024 + row * 97) as u32)
        .collect::<Vec<_>>();
    let slots = (0..batch).collect::<Vec<_>>();
    let rounds = WARM_PASSES + MEASURED_STEPS;
    for &slot in &slots {
        model.reserve_slot(stream, slot, rounds)?;
    }

    for warm in 0..WARM_PASSES {
        model.decode_step(stream, &tokens, &vec![warm as u32; batch], &slots)?;
    }
    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for step in 0..MEASURED_STEPS {
        let positions = vec![(WARM_PASSES + step) as u32; batch];
        samples.push(model.decode_step(stream, &tokens, &positions, &slots)?);
    }

    summarize("decode", batch, &samples)
}

fn measure_prefill(
    model: &mut Qwen38FlashNextResidentModel,
    stream: &CudaStream,
    tile: usize,
) -> Result<Qwen38FlashNextResidentRouteReport, DeviceBenchmarkError> {
    let prompt = (0..tile)
        .map(|token| (2_048 + token) as u32)
        .collect::<Vec<_>>();
    model.reserve_slot(stream, 0, tile)?;

    for _ in 0..WARM_PASSES {
        model.reset_slot(stream, 0)?;
        model.prefill_tile(stream, &prompt, 0, 0)?;
    }
    let mut samples = Vec::with_capacity(MEASURED_STEPS);
    for _ in 0..MEASURED_STEPS {
        model.reset_slot(stream, 0)?;
        samples.push(model.prefill_tile(stream, &prompt, 0, 0)?);
    }

    summarize("prefill", tile, &samples)
}

fn summarize(
    kind: &'static str,
    rows: usize,
    samples: &[Qwen38FlashNextStepTelemetry],
) -> Result<Qwen38FlashNextResidentRouteReport, DeviceBenchmarkError> {
    if samples.is_empty() {
        return Err(DeviceBenchmarkError::Precondition(
            "resident benchmark route produced no samples".to_string(),
        ));
    }
    for sample in samples {
        validate_route_accounting(sample, rows)?;
    }

    let mut times = samples
        .iter()
        .map(Qwen38FlashNextStepTelemetry::forward)
        .collect::<Vec<_>>();
    times.sort_unstable();
    let median = times[times.len() / 2];
    let last = samples
        .last()
        .expect("a measured route has at least one step");
    let layers = last.layers().len().max(1);

    Ok(Qwen38FlashNextResidentRouteReport {
        kind,
        rows,
        median,
        fastest: times[0],
        slowest: times[times.len() - 1],
        tokens_per_second: rows as f64 / median.as_secs_f64(),
        milliseconds_per_token: median.as_secs_f64() * 1_000.0 / rows as f64,
        microseconds_per_layer: median.as_secs_f64() * 1_000_000.0 / layers as f64,
        expert_requests: last.expert_requests(),
        expert_h2d_bytes: last.expert_h2d_bytes(),
        expert_h2d_bytes_per_token: last.expert_h2d_bytes() / rows,
        expert_hit_rate: last.expert_hit_rate(),
        layer_hit_rates: last.layers().iter().map(|layer| layer.hit_rate()).collect(),
        layer_h2d_bytes: last
            .layers()
            .iter()
            .map(|layer| layer.uploaded_bytes())
            .collect(),
        embedding_h2d_bytes: last.embedding_h2d_bytes(),
        engram_h2d_bytes: last.engram_h2d_bytes(),
        kv_append_bytes: last.kv_append_bytes(),
    })
}

fn expected_route_accounting(rows: usize) -> RouteAccounting {
    let expert_requests = rows * Qwen38FlashNext::LAYERS * Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;

    RouteAccounting {
        layer_rounds: Qwen38FlashNext::LAYERS,
        expert_requests,
        expert_bytes_routed: expert_requests * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES,
        embedding_h2d_bytes: rows * Qwen38FlashNext::HIDDEN * size_of::<u16>(),
        engram_h2d_bytes: rows * Qwen38FlashNext::PLE_EMBED_DIM,
        engram_rows: rows * Qwen38FlashNext::NGRAM_HEADS,
        kv_append_bytes: rows
            * QWEN38_FLASH_NEXT_ATTENTION_LAYERS
            * 2
            * Qwen38FlashNext::NUM_KV_HEADS
            * Qwen38FlashNext::HEAD_DIM,
        segment_replays: QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS,
        expert_readbacks: Qwen38FlashNext::LAYERS,
    }
}

fn validate_route_accounting(
    sample: &Qwen38FlashNextStepTelemetry,
    rows: usize,
) -> Result<(), DeviceBenchmarkError> {
    let observed = RouteAccounting {
        layer_rounds: sample.streaming_rounds(),
        expert_requests: sample.expert_requests(),
        expert_bytes_routed: sample.expert_bytes_routed(),
        embedding_h2d_bytes: sample.embedding_h2d_bytes(),
        engram_h2d_bytes: sample.engram_h2d_bytes(),
        engram_rows: sample.engram_rows(),
        kv_append_bytes: sample.kv_append_bytes(),
        segment_replays: sample.segment_replays(),
        expert_readbacks: sample.expert_readbacks(),
    };
    let expected = expected_route_accounting(rows);
    if sample.rows() != rows || observed != expected {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "resident benchmark accounting mismatch for {rows} rows: observed {observed:?}, expected {expected:?}"
        )));
    }
    for (layer, round) in sample.layers().iter().enumerate() {
        if round.layer() != layer
            || round.requests() != rows * Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN
        {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "resident benchmark layer {layer} reported id {} and {} requests",
                round.layer(),
                round.requests()
            )));
        }
    }

    Ok(())
}

/// Prints one sweep in the house's diagnostic shape.
pub fn print_qwen38_flash_next_resident_benchmark(report: &Qwen38FlashNextResidentBenchmarkReport) {
    println!("Qwen3.8 Flash-Next resident model - end-to-end sweep");
    println!("  DIAGNOSTIC ONLY. This command cannot bless a performance baseline.");
    println!("  Host-observed wall times: they include the per-round readback and the require");
    println!("  stall, so they are not comparable with per-layer captured-replay medians.");
    println!();
    println!("  construction");
    println!("    total                    {:?}", report.load);
    println!("    weights + expert stage   {:?}", report.weight_upload);
    println!("    host pin                 {:?}", report.host_pin);
    println!("    executables captured     {}", report.executables);
    println!("    capture wall time        {:?}", report.capture);
    println!();
    println!(
        "  {:<8} {:>5} {:>12} {:>10} {:>12} {:>10} {:>8} {:>14}",
        "route", "rows", "median ms", "tok/s", "ms/token", "us/layer", "hit", "h2d/token"
    );
    for route in &report.routes {
        println!(
            "  {:<8} {:>5} {:>12.3} {:>10.2} {:>12.3} {:>10.1} {:>8.4} {:>14}",
            route.kind,
            route.rows,
            route.median.as_secs_f64() * 1_000.0,
            route.tokens_per_second,
            route.milliseconds_per_token,
            route.microseconds_per_layer,
            route.expert_hit_rate,
            route.expert_h2d_bytes_per_token
        );
    }
    println!();
    for route in &report.routes {
        println!("  {} rows={} per-layer streaming", route.kind, route.rows);
        let weakest = route
            .layer_hit_rates
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1));
        if let Some((layer, rate)) = weakest {
            println!("    weakest layer {layer} at {rate:.4}");
        }
        for layer in (0..route.layer_hit_rates.len()).step_by(8) {
            println!(
                "      layer {:>2}  hit {:.4}  h2d {:>10}",
                layer, route.layer_hit_rates[layer], route.layer_h2d_bytes[layer]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_flash_next_resident_model_benchmark_accounting_covers_every_route_and_boundary() {
        assert_eq!(DECODE_ROWS, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(QWEN38_FLASH_NEXT_PREFILL_ROWS, [32, 64, 128, 1_024]);

        assert_eq!(
            expected_route_accounting(1),
            RouteAccounting {
                layer_rounds: 48,
                expert_requests: 480,
                expert_bytes_routed: 1_327_104_000,
                embedding_h2d_bytes: 5_120,
                engram_h2d_bytes: 2_560,
                engram_rows: 16,
                kv_append_bytes: 12_288,
                segment_replays: 49,
                expert_readbacks: 48,
            }
        );
        assert_eq!(expected_route_accounting(8).expert_requests, 3_840);
        assert_eq!(expected_route_accounting(1_024).kv_append_bytes, 12_582_912);
    }
}
