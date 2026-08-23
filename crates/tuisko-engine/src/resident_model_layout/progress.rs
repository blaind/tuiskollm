//! Lock-free startup progress shared with the serving thread.

use crate::{EngineError, EngineResult};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Current phase of exact resident-model construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResidentLoadPhase {
    /// The upload plan, device owners, and operators are being prepared.
    Preparing = 0,
    /// Resident weight and metadata copies are being submitted.
    Uploading = 1,
    /// All bytes were submitted and the arenas and graphs are being finalized.
    Finalizing = 2,
    /// The resident program is ready to serve requests.
    Ready = 3,
}

/// Monotonic resident-loading counters polled outside the engine hot path.
#[derive(Debug, Default)]
pub struct ResidentLoadProgress {
    phase: AtomicU8,
    submitted_bytes: AtomicUsize,
    total_bytes: AtomicUsize,
}

impl ResidentLoadProgress {
    /// Creates a progress owner in the preparation phase.
    pub const fn new() -> Self {
        Self {
            phase: AtomicU8::new(ResidentLoadPhase::Preparing as u8),
            submitted_bytes: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
        }
    }

    /// Reads the monotonic phase and byte counters without blocking the loader.
    pub fn snapshot(&self) -> (ResidentLoadPhase, usize, usize) {
        let phase = match self.phase.load(Ordering::Acquire) {
            0 => ResidentLoadPhase::Preparing,
            1 => ResidentLoadPhase::Uploading,
            2 => ResidentLoadPhase::Finalizing,
            3 => ResidentLoadPhase::Ready,
            _ => unreachable!("resident load phase is written only by this type"),
        };
        (
            phase,
            self.submitted_bytes.load(Ordering::Relaxed),
            self.total_bytes.load(Ordering::Relaxed),
        )
    }

    pub(super) fn begin_upload(&self, total_bytes: usize) {
        self.submitted_bytes.store(0, Ordering::Relaxed);
        self.total_bytes.store(total_bytes, Ordering::Relaxed);
        self.phase
            .store(ResidentLoadPhase::Uploading as u8, Ordering::Release);
    }

    pub(super) fn submit(&self, bytes: usize) -> EngineResult<()> {
        let total = self.total_bytes.load(Ordering::Relaxed);
        self.submitted_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |submitted| {
                submitted
                    .checked_add(bytes)
                    .filter(|&updated| updated <= total)
            })
            .map(|_| ())
            .map_err(|submitted| {
                EngineError::layout(format!(
                    "resident progress cannot add {bytes} submitted bytes to {submitted} of {total}"
                ))
            })
    }

    pub(super) fn finish_upload(&self) -> EngineResult<()> {
        let submitted = self.submitted_bytes.load(Ordering::Relaxed);
        let total = self.total_bytes.load(Ordering::Relaxed);
        if submitted != total {
            return Err(EngineError::layout(format!(
                "resident progress ended at {submitted} of {total} submitted bytes"
            )));
        }
        self.phase
            .store(ResidentLoadPhase::Finalizing as u8, Ordering::Release);
        Ok(())
    }

    pub(super) fn finish(&self) {
        self.phase
            .store(ResidentLoadPhase::Ready as u8, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::{ResidentLoadPhase, ResidentLoadProgress};

    #[test]
    fn progress_is_monotonic_and_exact() {
        let progress = ResidentLoadProgress::new();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Preparing);

        progress.begin_upload(12);
        progress.submit(5).unwrap();
        progress.submit(7).unwrap();
        assert_eq!(progress.snapshot().1, 12);
        progress.finish_upload().unwrap();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Finalizing);
        progress.finish();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Ready);
    }

    #[test]
    fn progress_refuses_overflow_and_incomplete_finalization() {
        let progress = ResidentLoadProgress::new();
        progress.begin_upload(8);
        assert!(progress.submit(9).is_err());
        progress.submit(7).unwrap();
        assert!(progress.finish_upload().is_err());
    }
}
