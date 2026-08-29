//! Lock-free startup progress shared with the serving thread.

use crate::{EngineError, EngineResult};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Current phase of exact resident-model construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResidentLoadPhase {
    /// Checkpoint admission and target dispatch are still in progress.
    Preparing = 0,
    /// Exact resident layouts and upload plans are being built.
    Planning = 1,
    /// Address-stable device arenas are being allocated.
    Allocating = 2,
    /// Concrete CUDA operators and host stagers are being prepared.
    PreparingOperators = 3,
    /// Resident weight and metadata copies are being submitted.
    Uploading = 4,
    /// Immutable target graphs are being captured.
    CapturingTargetGraphs = 5,
    /// The MTP block's source weights are being submitted.
    UploadingMtp = 6,
    /// Concrete MTP operators and host stagers are being prepared.
    PreparingMtpOperators = 7,
    /// Immutable MTP graphs are being captured.
    CapturingMtpGraphs = 8,
    /// All resident owners are being finalized for serving.
    Finalizing = 9,
    /// The resident program is ready to serve requests.
    Ready = 10,
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
            1 => ResidentLoadPhase::Planning,
            2 => ResidentLoadPhase::Allocating,
            3 => ResidentLoadPhase::PreparingOperators,
            4 => ResidentLoadPhase::Uploading,
            5 => ResidentLoadPhase::CapturingTargetGraphs,
            6 => ResidentLoadPhase::UploadingMtp,
            7 => ResidentLoadPhase::PreparingMtpOperators,
            8 => ResidentLoadPhase::CapturingMtpGraphs,
            9 => ResidentLoadPhase::Finalizing,
            10 => ResidentLoadPhase::Ready,
            _ => unreachable!("resident load phase is written only by this type"),
        };
        (
            phase,
            self.submitted_bytes.load(Ordering::Relaxed),
            self.total_bytes.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn plan(&self) {
        self.set_phase(ResidentLoadPhase::Planning);
    }

    pub(crate) fn allocate(&self) {
        self.set_phase(ResidentLoadPhase::Allocating);
    }

    pub(crate) fn prepare_operators(&self) {
        self.set_phase(ResidentLoadPhase::PreparingOperators);
    }

    pub(crate) fn begin_upload(&self, total_bytes: usize) {
        self.submitted_bytes.store(0, Ordering::Relaxed);
        self.total_bytes.store(total_bytes, Ordering::Relaxed);
        self.phase
            .store(ResidentLoadPhase::Uploading as u8, Ordering::Release);
    }

    pub(crate) fn submit(&self, bytes: usize) -> EngineResult<()> {
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

    pub(crate) fn finish_upload(&self) -> EngineResult<()> {
        self.finish_upload_as(ResidentLoadPhase::Finalizing)
    }

    pub(crate) fn finish_target_upload(&self) -> EngineResult<()> {
        self.finish_upload_as(ResidentLoadPhase::CapturingTargetGraphs)
    }

    pub(crate) fn capture_target_graphs(&self) {
        self.set_phase(ResidentLoadPhase::CapturingTargetGraphs);
    }

    pub(crate) fn begin_mtp_upload(&self) {
        self.set_phase(ResidentLoadPhase::UploadingMtp);
    }

    pub(crate) fn finish_mtp_upload(&self) -> EngineResult<()> {
        self.finish_upload_as(ResidentLoadPhase::PreparingMtpOperators)
    }

    pub(crate) fn capture_mtp_graphs(&self) {
        self.set_phase(ResidentLoadPhase::CapturingMtpGraphs);
    }

    pub(crate) fn finalize(&self) {
        self.set_phase(ResidentLoadPhase::Finalizing);
    }

    fn finish_upload_as(&self, phase: ResidentLoadPhase) -> EngineResult<()> {
        let submitted = self.submitted_bytes.load(Ordering::Relaxed);
        let total = self.total_bytes.load(Ordering::Relaxed);
        if submitted != total {
            return Err(EngineError::layout(format!(
                "resident progress ended at {submitted} of {total} submitted bytes"
            )));
        }
        self.set_phase(phase);
        Ok(())
    }

    pub(crate) fn finish(&self) {
        self.set_phase(ResidentLoadPhase::Ready);
    }

    fn set_phase(&self, phase: ResidentLoadPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::{ResidentLoadPhase, ResidentLoadProgress};

    #[test]
    fn progress_is_monotonic_and_exact() {
        let progress = ResidentLoadProgress::new();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Preparing);

        progress.plan();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Planning);
        progress.allocate();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::Allocating);
        progress.prepare_operators();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::PreparingOperators);
        progress.begin_upload(12);
        progress.submit(5).unwrap();
        progress.submit(7).unwrap();
        assert_eq!(progress.snapshot().1, 12);
        progress.finish_target_upload().unwrap();
        assert_eq!(
            progress.snapshot().0,
            ResidentLoadPhase::CapturingTargetGraphs
        );
        progress.begin_mtp_upload();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::UploadingMtp);
        progress.finish_mtp_upload().unwrap();
        assert_eq!(
            progress.snapshot().0,
            ResidentLoadPhase::PreparingMtpOperators
        );
        progress.capture_mtp_graphs();
        assert_eq!(progress.snapshot().0, ResidentLoadPhase::CapturingMtpGraphs);
        progress.finalize();
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
