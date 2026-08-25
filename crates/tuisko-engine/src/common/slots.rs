//! Compact-batch slot, capacity, and device admission checks shared by every target.

use crate::{EngineError, EngineResult, MAX_BATCH};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, GpuError};

pub(crate) fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "batch {batch} is outside the exact range 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

pub(crate) fn first_free_slot(occupied: [bool; MAX_BATCH]) -> Option<usize> {
    occupied.iter().position(|&occupied| !occupied)
}

pub(crate) fn device_zero_context() -> EngineResult<Arc<CudaContext>> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(EngineError::route(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    Ok(context)
}

pub(crate) fn require_generation_capacity(
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    context_capacity: usize,
) -> EngineResult<usize> {
    if prompt_tokens == 0 {
        return Err(EngineError::generation(
            "resident generation requires a nonempty prompt",
        ));
    }
    let evaluated = prompt_tokens
        .checked_add(maximum_new_tokens.saturating_sub(1))
        .ok_or_else(|| EngineError::generation("generation token budget overflows"))?;
    if evaluated > context_capacity {
        return Err(EngineError::generation(format!(
            "prompt plus processed generation requires {evaluated} positions, current resident capacity is {context_capacity}"
        )));
    }
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use super::require_generation_capacity;
    use crate::EngineErrorCode;

    #[test]
    fn long_context_capacity_counts_only_processed_generated_tokens() {
        require_generation_capacity(220_000, 1, 220_000).unwrap();
        require_generation_capacity(1, 220_000, 220_000).unwrap();
        require_generation_capacity(220_000, 0, 220_000).unwrap();

        for (prompt, generated) in [(0, 1), (220_000, 2), (2, 220_000), (usize::MAX, 2)] {
            assert_eq!(
                require_generation_capacity(prompt, generated, 220_000)
                    .unwrap_err()
                    .code(),
                Some(EngineErrorCode::Generation)
            );
        }
    }
}
