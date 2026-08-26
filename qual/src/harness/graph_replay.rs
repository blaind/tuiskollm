//! Captured-graph replay mechanics: replay loops, address stability, and the post-warmup
//! device-heap gate.
//!
//! Capture, sentinel resets, and observation stay in each suite: they bind that suite's regions,
//! routes, and fixtures. Only the replay loops and the memory-counter comparison are shared, and
//! repetition counts remain caller-supplied parameters (`AGENTS.md`: repetition counts are part
//! of measurement identity).

use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult, device_memory_info};

/// Index of the first recorded base address that moved.
pub(crate) fn first_moved_address(before: &[usize], after: &[usize]) -> Option<usize> {
    before
        .iter()
        .zip(after)
        .position(|(before, after)| before != after)
}

/// Launches one captured graph `replays` times on `stream` without synchronizing.
///
/// # Safety
///
/// Every allocation the graph captured must be owned by the caller and outlive both the replays
/// and the synchronization that follows them.
pub(crate) unsafe fn replay(
    graph: &CudaGraph,
    stream: &CudaStream,
    replays: usize,
) -> GpuResult<()> {
    for _ in 0..replays {
        // SAFETY: the caller guarantees the captured allocations outlive these replays.
        unsafe { graph.launch(stream) }?;
    }

    Ok(())
}

/// Runs the post-warmup device-heap gate over already-captured graphs.
///
/// Warms every graph once, snapshots the driver's memory counters, replays `rounds` reversed
/// passes over the same graphs, and snapshots again. Returns the standard drift description when
/// the counters differ so the caller can raise it as its own mismatch.
///
/// # Safety
///
/// Every allocation these graphs captured must be owned by the caller and outlive this call.
pub(crate) unsafe fn post_warmup_allocation_drift(
    context: &CudaContext,
    stream: &CudaStream,
    graphs: &[CudaGraph],
    rounds: usize,
) -> GpuResult<Option<String>> {
    for graph in graphs {
        // SAFETY: the caller guarantees the captured allocations outlive this warmup.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..rounds {
        for graph in graphs.iter().rev() {
            // SAFETY: the caller guarantees the captured allocations outlive these replays.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;

    Ok((before != after)
        .then(|| format!("device memory changed after warmup: before={before:?}, after={after:?}")))
}

#[cfg(test)]
mod tests {
    use super::first_moved_address;

    #[test]
    fn stable_addresses_report_no_movement() {
        let recorded = [0x1000_usize, 0x2000, 0x3000];
        assert_eq!(first_moved_address(&recorded, &recorded), None);
    }

    #[test]
    fn a_moved_address_reports_its_recording_order() {
        let before = [0x1000_usize, 0x2000, 0x3000];
        let after = [0x1000_usize, 0x2000, 0x3080];
        assert_eq!(first_moved_address(&before, &after), Some(2));
    }
}
