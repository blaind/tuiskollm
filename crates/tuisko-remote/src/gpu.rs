//! GPU identity required by one remote run.

/// Exact RunPod GPU identity and CUDA capability expected by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuTarget {
    device_name: &'static str,
    compute_capability: &'static str,
}

impl GpuTarget {
    /// Creates a target from an exact `nvidia-smi` device name and capability.
    pub const fn new(device_name: &'static str, compute_capability: &'static str) -> Self {
        Self {
            device_name,
            compute_capability,
        }
    }

    pub(crate) const fn device_name(self) -> &'static str {
        self.device_name
    }

    pub(crate) fn expected_identity(self) -> String {
        format!("{}, {}", self.device_name, self.compute_capability)
    }
}
