//! CUDA-event timing for device-only measurements.

use crate::{CudaContext, CudaEvent, CudaStream, GpuError, GpuResult};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Paired host and device durations from one submitted work interval.
#[derive(Clone, Copy, Debug)]
pub struct GpuTiming {
    /// CUDA-event time between the stream records.
    pub device: Duration,
    /// Host time spent submitting the measured work.
    pub host_submit: Duration,
    /// Host time through completion of the recorded device work.
    pub host_completion: Duration,
}

/// A reusable pair of timing-enabled CUDA events.
///
/// Measurements take `&mut self` because one event pair can only describe one
/// interval: overlapping or re-entrant measurements would re-record the same
/// events and report an interval nobody asked for.
pub struct GpuTimer {
    start: CudaEvent,
    end: CudaEvent,
}

impl GpuTimer {
    /// Creates timing events in `context`.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
        let start = context
            .new_event(Some(flags))
            .map_err(|source| GpuError::driver("creating the start timing event", source))?;
        let end = context
            .new_event(Some(flags))
            .map_err(|source| GpuError::driver("creating the end timing event", source))?;

        Ok(Self { start, end })
    }

    /// Measures only the device work enqueued by `record` on `stream`.
    pub fn measure<F>(&mut self, stream: &CudaStream, record: F) -> GpuResult<Duration>
    where
        F: FnOnce() -> GpuResult<()>,
    {
        Ok(self.measure_with_host(stream, record)?.device)
    }

    /// Measures paired host submission, host completion, and CUDA-event time.
    pub fn measure_with_host<F>(&mut self, stream: &CudaStream, record: F) -> GpuResult<GpuTiming>
    where
        F: FnOnce() -> GpuResult<()>,
    {
        if self.start.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "CUDA timer and measured stream belong to different contexts",
            ));
        }

        self.start
            .record(stream)
            .map_err(|source| GpuError::driver("recording the start timing event", source))?;
        let host_start = Instant::now();
        record()?;
        let host_submit = host_start.elapsed();
        self.end
            .record(stream)
            .map_err(|source| GpuError::driver("recording the end timing event", source))?;
        self.end
            .synchronize()
            .map_err(|source| GpuError::driver("waiting for the end timing event", source))?;
        let host_completion = host_start.elapsed();
        let milliseconds = self
            .start
            .elapsed_ms(&self.end)
            .map_err(|source| GpuError::driver("reading CUDA event time", source))?;

        Ok(GpuTiming {
            device: Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0),
            host_submit,
            host_completion,
        })
    }
}
