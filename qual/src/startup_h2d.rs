use crate::DeviceBenchmarkError;
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, EXPECTED_DEVICE_NAME};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tuisko_gpu::{CudaContext, DeviceBuffer, GpuError, GpuTimer, PinnedHostBuffer};

const BYTES_PER_COPY: usize = 256 * 1024 * 1024;
const COPIES_PER_SAMPLE: usize = 16;
const WARMUPS_PER_ROUTE: usize = 2;
const SAMPLE_COUNT: usize = 7;

#[derive(Clone, Copy, Debug)]
enum H2dRoute {
    Pageable,
    Pinned,
}

impl H2dRoute {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pageable => "pageable",
            Self::Pinned => "pinned",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct H2dSample {
    route: String,
    device_ms: f64,
    host_submit_ms: f64,
    host_completion_ms: f64,
    device_gib_s: f64,
    host_completion_gib_s: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct H2dRouteReport {
    route: String,
    samples: Vec<H2dSample>,
    median_device_gib_s: f64,
    median_host_completion_gib_s: f64,
}

/// Contiguous host-to-device calibration taken in an isolated fresh process.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct H2dCalibrationReport {
    device_name: String,
    compute_capability: String,
    bytes_per_copy: usize,
    copies_per_sample: usize,
    warmups_per_route: usize,
    routes: Vec<H2dRouteReport>,
}

pub(crate) fn measure_h2d_calibration() -> Result<H2dCalibrationReport, DeviceBenchmarkError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let device_name = context.device_name().map_err(GpuError::from)?;
    let compute_capability = context.compute_capability().map_err(GpuError::from)?;
    if device_name != EXPECTED_DEVICE_NAME || compute_capability != EXPECTED_COMPUTE_CAPABILITY {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "device zero is {device_name} with compute capability {}.{}, expected {EXPECTED_DEVICE_NAME} with compute capability {}.{}",
            compute_capability.0,
            compute_capability.1,
            EXPECTED_COMPUTE_CAPABILITY.0,
            EXPECTED_COMPUTE_CAPABILITY.1,
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut pageable = vec![0_u8; BYTES_PER_COPY];
    for (index, byte) in pageable.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    let pinned = PinnedHostBuffer::from_slice(&context, &pageable).map_err(GpuError::from)?;
    let mut destination =
        DeviceBuffer::<u8>::zeroed(&stream, BYTES_PER_COPY).map_err(GpuError::from)?;
    stream.synchronize().map_err(GpuError::from)?;

    for _ in 0..WARMUPS_PER_ROUTE {
        // SAFETY: both sources remain immutable and alive through the synchronization below.
        unsafe {
            destination
                .copy_from_host_async_unchecked(&stream, &pageable)
                .map_err(GpuError::from)?;
            destination
                .copy_from_pinned_host_async(&stream, &pinned)
                .map_err(GpuError::from)?;
        }
        stream.synchronize().map_err(GpuError::from)?;
    }

    let timer = GpuTimer::new(&context)?;
    let mut pageable_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut pinned_samples = Vec::with_capacity(SAMPLE_COUNT);
    for sample in 0..SAMPLE_COUNT {
        let routes = if sample.is_multiple_of(2) {
            [H2dRoute::Pageable, H2dRoute::Pinned]
        } else {
            [H2dRoute::Pinned, H2dRoute::Pageable]
        };
        for route in routes {
            let timing = timer.measure_with_host(&stream, || {
                for _ in 0..COPIES_PER_SAMPLE {
                    // SAFETY: the selected source stays immutable and alive until the timing end
                    // event completes all preceding copies on this stream.
                    unsafe {
                        match route {
                            H2dRoute::Pageable => {
                                destination.copy_from_host_async_unchecked(&stream, &pageable)?
                            }
                            H2dRoute::Pinned => {
                                destination.copy_from_pinned_host_async(&stream, &pinned)?
                            }
                        }
                    }
                }
                Ok(())
            })?;
            let sample = make_sample(
                route,
                timing.device,
                timing.host_submit,
                timing.host_completion,
            )?;
            match route {
                H2dRoute::Pageable => pageable_samples.push(sample),
                H2dRoute::Pinned => pinned_samples.push(sample),
            }
        }
    }
    let observed = destination.to_host_vec(&stream).map_err(GpuError::from)?;
    if observed != pageable {
        return Err(DeviceBenchmarkError::Precondition(
            "H2D calibration destination differs from its deterministic source".into(),
        ));
    }

    Ok(H2dCalibrationReport {
        device_name,
        compute_capability: format!("{}.{}", compute_capability.0, compute_capability.1),
        bytes_per_copy: BYTES_PER_COPY,
        copies_per_sample: COPIES_PER_SAMPLE,
        warmups_per_route: WARMUPS_PER_ROUTE,
        routes: vec![
            summarize_route(H2dRoute::Pageable, pageable_samples)?,
            summarize_route(H2dRoute::Pinned, pinned_samples)?,
        ],
    })
}

fn make_sample(
    route: H2dRoute,
    device: Duration,
    host_submit: Duration,
    host_completion: Duration,
) -> Result<H2dSample, DeviceBenchmarkError> {
    let bytes = BYTES_PER_COPY
        .checked_mul(COPIES_PER_SAMPLE)
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition("H2D calibration byte count overflows".into())
        })?;
    let device_gib_s = throughput_gib_s(bytes, device)?;
    let host_completion_gib_s = throughput_gib_s(bytes, host_completion)?;
    Ok(H2dSample {
        route: route.as_str().into(),
        device_ms: milliseconds(device),
        host_submit_ms: milliseconds(host_submit),
        host_completion_ms: milliseconds(host_completion),
        device_gib_s,
        host_completion_gib_s,
    })
}

fn summarize_route(
    route: H2dRoute,
    samples: Vec<H2dSample>,
) -> Result<H2dRouteReport, DeviceBenchmarkError> {
    if samples.len() != SAMPLE_COUNT
        || samples.iter().any(|sample| {
            sample.route != route.as_str()
                || !sample.device_ms.is_finite()
                || sample.device_ms <= 0.0
                || !sample.host_submit_ms.is_finite()
                || sample.host_submit_ms <= 0.0
                || !sample.host_completion_ms.is_finite()
                || sample.host_completion_ms <= 0.0
                || !sample.device_gib_s.is_finite()
                || sample.device_gib_s <= 0.0
                || !sample.host_completion_gib_s.is_finite()
                || sample.host_completion_gib_s <= 0.0
        })
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "{} H2D calibration samples are incomplete or invalid",
            route.as_str()
        )));
    }
    let median_device_gib_s = median(samples.iter().map(|sample| sample.device_gib_s));
    let median_host_completion_gib_s =
        median(samples.iter().map(|sample| sample.host_completion_gib_s));
    Ok(H2dRouteReport {
        route: route.as_str().into(),
        samples,
        median_device_gib_s,
        median_host_completion_gib_s,
    })
}

fn throughput_gib_s(bytes: usize, elapsed: Duration) -> Result<f64, DeviceBenchmarkError> {
    let throughput = bytes as f64 / (1_u64 << 30) as f64 / elapsed.as_secs_f64();
    if !throughput.is_finite() || throughput <= 0.0 {
        return Err(DeviceBenchmarkError::Precondition(
            "H2D calibration produced a non-positive throughput".into(),
        ));
    }
    Ok(throughput)
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(crate) fn print_h2d_calibration(report: &H2dCalibrationReport) {
    eprintln!();
    eprintln!(
        "contiguous H2D calibration · {} MiB x {} copies/sample",
        report.bytes_per_copy / (1 << 20),
        report.copies_per_sample,
    );
    eprintln!("source       device GiB/s   host-completion GiB/s");
    for route in &report.routes {
        eprintln!(
            "{:<12} {:>12.2} {:>23.2}",
            route.route, route.median_device_gib_s, route.median_host_completion_gib_s,
        );
    }
}

pub(crate) fn pinned_h2d_gib_s(report: &H2dCalibrationReport) -> f64 {
    report
        .routes
        .iter()
        .find(|route| route.route == H2dRoute::Pinned.as_str())
        .expect("H2D calibration always contains its pinned route")
        .median_host_completion_gib_s
}

#[cfg(test)]
mod tests {
    use super::{BYTES_PER_COPY, COPIES_PER_SAMPLE, SAMPLE_COUNT, WARMUPS_PER_ROUTE};

    #[test]
    fn calibration_uses_repeated_long_transfer_windows() {
        assert_eq!(BYTES_PER_COPY, 256 * 1024 * 1024);
        assert_eq!(BYTES_PER_COPY * COPIES_PER_SAMPLE, 4 * 1024 * 1024 * 1024);
        assert_eq!(WARMUPS_PER_ROUTE, 2);
        assert_eq!(SAMPLE_COUNT, 7);
    }
}
