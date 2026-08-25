//! Exact CUDA Graph capture loops shared by every resident program.

use crate::{EngineError, EngineResult};
use tuisko_gpu::{CudaGraph, CudaStream, GpuResult};

/// Captures one graph per admitted exact width `1..=N`, in ascending order.
pub(crate) fn capture_batch_graphs<const N: usize, F>(
    stream: &CudaStream,
    inventory: &'static str,
    mut launch: F,
) -> EngineResult<[CudaGraph; N]>
where
    F: FnMut(usize) -> GpuResult<()>,
{
    let mut graphs = Vec::with_capacity(N);
    for width in 1..=N {
        graphs.push(CudaGraph::capture(stream, || launch(width))?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout(inventory))
}

/// Captures one graph per listed exact route, in the listed order.
pub(crate) fn capture_route_graphs<const N: usize, F>(
    stream: &CudaStream,
    routes: [usize; N],
    inventory: &'static str,
    mut launch: F,
) -> EngineResult<[CudaGraph; N]>
where
    F: FnMut(usize) -> GpuResult<()>,
{
    let mut graphs = Vec::with_capacity(N);
    for route in routes {
        graphs.push(CudaGraph::capture(stream, || launch(route))?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout(inventory))
}
