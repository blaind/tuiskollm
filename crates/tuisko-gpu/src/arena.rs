//! Typed layout and single-allocation ownership for address-stable workspaces.

use crate::{CudaStream, DeviceBuffer, DeviceCopy, GpuError, GpuResult, PinnedHostBuffer};
use std::marker::PhantomData;
use std::mem::{align_of, size_of, size_of_val};

/// A typed byte range reserved within a device arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaRegion<T> {
    offset: usize,
    len: usize,
    bytes: usize,
    alignment: usize,
    element: PhantomData<fn() -> T>,
}

impl<T> ArenaRegion<T> {
    /// Byte offset from the arena base address.
    pub const fn offset_bytes(self) -> usize {
        self.offset
    }

    /// Number of typed elements in this region.
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether this region contains no elements.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Number of bytes occupied by this region.
    pub const fn byte_len(self) -> usize {
        self.bytes
    }

    /// Required device-address alignment in bytes.
    pub const fn alignment(self) -> usize {
        self.alignment
    }
}

/// Monotonic builder for non-overlapping, explicitly aligned arena regions.
#[derive(Clone, Debug, Default)]
pub struct ArenaLayout {
    cursor: usize,
}

impl ArenaLayout {
    /// Creates an empty layout.
    pub const fn new() -> Self {
        Self { cursor: 0 }
    }

    /// Reserves `len` elements with the requested byte alignment.
    pub fn reserve<T: DeviceCopy>(
        &mut self,
        len: usize,
        alignment: usize,
    ) -> GpuResult<ArenaRegion<T>> {
        if size_of::<T>() == 0 {
            return Err(GpuError::arena(
                "arena regions cannot contain zero-sized element types",
            ));
        }
        if !alignment.is_power_of_two() {
            return Err(GpuError::arena(format!(
                "arena region alignment {alignment} is not a power of two"
            )));
        }
        if alignment < align_of::<T>() {
            return Err(GpuError::arena(format!(
                "arena region alignment {alignment} is smaller than the element alignment {}",
                align_of::<T>()
            )));
        }

        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            GpuError::arena(format!(
                "arena byte count overflows for {len} elements of {} bytes",
                size_of::<T>()
            ))
        })?;
        let mask = alignment - 1;
        let offset = self
            .cursor
            .checked_add(mask)
            .map(|value| value & !mask)
            .ok_or_else(|| GpuError::arena("arena alignment overflows"))?;
        self.cursor = offset
            .checked_add(bytes)
            .ok_or_else(|| GpuError::arena("arena size overflows"))?;

        Ok(ArenaRegion {
            offset,
            len,
            bytes,
            alignment,
            element: PhantomData,
        })
    }

    /// Total bytes required by all regions reserved so far.
    pub const fn byte_len(&self) -> usize {
        self.cursor
    }

    /// Whether this layout contains no bytes.
    pub const fn is_empty(&self) -> bool {
        self.cursor == 0
    }
}

/// One address-stable device allocation partitioned by checked typed regions.
pub struct DeviceArena {
    storage: DeviceBuffer<u8>,
    bytes: usize,
}

impl DeviceArena {
    /// Allocates and zeroes the storage required by `layout`.
    pub fn zeroed(stream: &CudaStream, layout: &ArenaLayout) -> GpuResult<Self> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        let storage = DeviceBuffer::zeroed(stream, layout.byte_len())
            .map_err(|source| GpuError::driver("allocating a zeroed device arena", source))?;

        Ok(Self {
            storage,
            bytes: layout.byte_len(),
        })
    }

    /// Returns the stable base device address.
    pub fn base_address(&self) -> u64 {
        self.storage.cu_deviceptr()
    }

    /// Returns the allocation size in bytes.
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns whether this arena owns no device bytes.
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Returns the checked device address of `region`.
    pub fn address<T: DeviceCopy>(&self, region: ArenaRegion<T>) -> GpuResult<*mut T> {
        let end = region
            .offset
            .checked_add(region.bytes)
            .ok_or_else(|| GpuError::arena("arena region end overflows"))?;
        if end > self.bytes {
            return Err(GpuError::arena(format!(
                "arena region {}..{end} exceeds a {}-byte allocation",
                region.offset, self.bytes
            )));
        }

        let offset = u64::try_from(region.offset)
            .map_err(|_| GpuError::arena("arena region offset exceeds the device address width"))?;
        let address = self
            .base_address()
            .checked_add(offset)
            .ok_or_else(|| GpuError::arena("arena device address overflows"))?;
        let element_alignment = u64::try_from(align_of::<T>())
            .map_err(|_| GpuError::arena("element alignment exceeds the device address width"))?;
        let region_alignment = u64::try_from(region.alignment)
            .map_err(|_| GpuError::arena("region alignment exceeds the device address width"))?;

        if address % element_alignment != 0 || address % region_alignment != 0 {
            return Err(GpuError::arena(format!(
                "arena address {address:#x} does not satisfy {}-byte region alignment",
                region.alignment
            )));
        }

        let address = usize::try_from(address)
            .map_err(|_| GpuError::arena("device address exceeds the host pointer width"))?;

        Ok(address as *mut T)
    }

    /// Enqueues a byte fill over one checked region.
    pub fn fill<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        value: u8,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "filling a device arena")?;
        let address = self.address(region)? as u64;
        if region.bytes == 0 {
            return Ok(());
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: `address..address + bytes` was checked against this live allocation.
        unsafe {
            cuda_core::memory::memset_d8_async(address, value, region.bytes, stream.cu_stream())
        }
        .map_err(|source| GpuError::driver("filling a device arena region", source))
    }

    /// Copies the complete arena into a host byte vector.
    pub fn to_host_vec(&self, stream: &CudaStream) -> GpuResult<Vec<u8>> {
        self.require_stream_context(stream, "copying a device arena to the host")?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        self.storage
            .to_host_vec(stream)
            .map_err(|source| GpuError::driver("copying a device arena to the host", source))
    }

    /// Copies one complete typed region from a host slice.
    pub fn copy_from_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[T],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a host slice into a device arena")?;
        if source.len() != region.len {
            return Err(GpuError::arena(format!(
                "host source has {} elements for an arena region of {} elements",
                source.len(),
                region.len
            )));
        }

        self.copy_prefix_from_host(stream, region, source)
    }

    /// Copies a typed host slice into the beginning of one region.
    pub fn copy_prefix_from_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[T],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a host slice into a device arena")?;
        if source.len() > region.len {
            return Err(GpuError::arena(format!(
                "host source has {} elements for an arena region of {} elements",
                source.len(),
                region.len
            )));
        }
        let address = self.address(region)? as u64;
        let bytes = size_of_val(source);
        if bytes == 0 {
            return Ok(());
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the checked region and source slice both cover exactly `region.bytes`.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr(),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a host slice into a device arena", source))?;
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing a device arena upload", source))
    }

    /// Enqueues a pinned-host prefix upload without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// `source` must remain allocated and immutable until the copy completes. If the copy is
    /// captured in a CUDA Graph, that requirement extends through the final graph replay.
    pub unsafe fn copy_prefix_from_pinned_host_async<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &PinnedHostBuffer<T>,
        len: usize,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a pinned host prefix into a device arena")?;
        if source.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "pinned host source and arena stream must share one CUDA context",
            ));
        }
        if len > source.len() || len > region.len {
            return Err(GpuError::arena(format!(
                "pinned upload length {len} exceeds source {} or region {} elements",
                source.len(),
                region.len
            )));
        }
        let address = self.address(region)? as u64;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena("pinned upload byte count overflows"))?;
        if bytes == 0 {
            return Ok(());
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the checked arena region covers `bytes`; the caller owns the pinned-source
        // lifetime and immutability contract documented by this method.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr(),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a pinned host prefix into an arena", source))
    }

    /// Copies one complete typed region into an owned host vector.
    pub fn copy_to_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
    ) -> GpuResult<Vec<T>> {
        self.copy_prefix_to_host(stream, region, region.len)
    }

    /// Copies the beginning of one region into an owned host vector.
    pub fn copy_prefix_to_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        len: usize,
    ) -> GpuResult<Vec<T>> {
        self.require_stream_context(stream, "copying a device arena region to the host")?;
        if len > region.len {
            return Err(GpuError::arena(format!(
                "requested {len} elements from an arena region of {} elements",
                region.len
            )));
        }
        let address = self.address(region)? as u64;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena("arena copy byte count overflows"))?;
        let mut host = Vec::with_capacity(len);
        if bytes == 0 {
            return Ok(host);
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the checked region and reserved vector capacity both cover `bytes`.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                address,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a device arena region to the host", source))?;
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing a device arena download", source))?;
        // SAFETY: the completed copy initialized all `len` elements.
        unsafe { host.set_len(len) };

        Ok(host)
    }

    fn require_stream_context(&self, stream: &CudaStream, operation: &str) -> GpuResult<()> {
        if self.storage.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(format!(
                "{operation} requires the allocation and stream to share one CUDA context"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ArenaLayout;
    use crate::GpuErrorCode;

    #[test]
    fn layout_aligns_non_overlapping_typed_regions() {
        let mut layout = ArenaLayout::new();
        let bytes = layout.reserve::<u8>(13, 1).unwrap();
        let words = layout.reserve::<u32>(4, 256).unwrap();
        let halves = layout.reserve::<u16>(3, 8).unwrap();

        assert_eq!((bytes.offset_bytes(), bytes.byte_len()), (0, 13));
        assert_eq!((words.offset_bytes(), words.byte_len()), (256, 16));
        assert_eq!((halves.offset_bytes(), halves.byte_len()), (272, 6));
        assert_eq!(layout.byte_len(), 278);
    }

    #[test]
    fn layout_rejects_invalid_alignment() {
        let mut layout = ArenaLayout::new();

        for alignment in [0, 3, 6] {
            let error = layout.reserve::<u8>(1, alignment).err().unwrap();

            assert_eq!(error.code(), GpuErrorCode::Arena);
            assert!(error.to_string().contains("not a power of two"));
        }

        let error = layout.reserve::<u32>(1, 2).err().unwrap();

        assert_eq!(error.code(), GpuErrorCode::Arena);
        assert!(
            error
                .to_string()
                .contains("smaller than the element alignment 4")
        );
    }

    #[test]
    fn layout_rejects_element_and_size_overflow() {
        let mut layout = ArenaLayout::new();
        let zero_sized = layout.reserve::<()>(1, 1).err().unwrap();
        let overflow = layout.reserve::<u64>(usize::MAX, 8).err().unwrap();

        assert_eq!(zero_sized.code(), GpuErrorCode::Arena);
        assert!(zero_sized.to_string().contains("zero-sized element"));
        assert_eq!(overflow.code(), GpuErrorCode::Arena);
        assert!(overflow.to_string().contains("byte count overflows"));
    }
}
