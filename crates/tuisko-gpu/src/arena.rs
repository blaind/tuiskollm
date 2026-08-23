//! Typed layout and single-allocation ownership for address-stable workspaces.

use crate::{CudaStream, DeviceBuffer, DeviceCopy, GpuError, GpuResult, PinnedHostBuffer};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, size_of_val};
use std::ops::Range;
use std::sync::Arc;

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

/// Uninitialized device allocation available only to checked startup writes.
pub struct LoadingDeviceArena {
    storage: DeviceBuffer<u8>,
    stream: Arc<CudaStream>,
    bytes: usize,
    initialized: InitializationCoverage,
}

impl LoadingDeviceArena {
    /// Allocates the storage required by `layout` without enqueueing a full-arena memset.
    pub fn allocate(stream: &Arc<CudaStream>, layout: &ArenaLayout) -> GpuResult<Self> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the loading-arena CUDA context", source))?;
        // SAFETY: this type exposes only writes until `seal` proves complete initialization and
        // synchronizes their stream. No safe operation can read the returned storage beforehand.
        let storage = unsafe { DeviceBuffer::uninitialized_async(stream, layout.byte_len()) }
            .map_err(|source| {
                GpuError::driver("allocating an uninitialized device arena", source)
            })?;

        Ok(Self {
            storage,
            stream: stream.clone(),
            bytes: layout.byte_len(),
            initialized: InitializationCoverage::new(layout.byte_len()),
        })
    }

    /// Enqueues a byte fill over one not-yet-initialized destination range.
    pub fn fill_async(
        &mut self,
        stream: &CudaStream,
        destination: Range<usize>,
        value: u8,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "filling a loading arena")?;
        self.initialized.require_available(&destination)?;
        let bytes = destination.end - destination.start;
        if bytes == 0 {
            return Ok(());
        }
        let address = self.address(destination.start)?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the loading-arena CUDA context", source))?;
        // SAFETY: coverage validation proved this byte range is inside the live allocation.
        unsafe { cuda_core::memory::memset_d8_async(address, value, bytes, stream.cu_stream()) }
            .map_err(|source| GpuError::driver("filling a loading-arena range", source))?;
        self.initialized.record(destination);
        Ok(())
    }

    /// Enqueues a pinned-host copy into one not-yet-initialized destination range.
    ///
    /// `source_offset` and the destination length select the source byte range.
    ///
    /// # Safety
    ///
    /// The selected pinned source range must remain allocated and immutable until this stream
    /// reaches the copy. An event wait or stream synchronization must precede buffer reuse.
    pub unsafe fn copy_from_pinned_host_async(
        &mut self,
        stream: &CudaStream,
        destination: Range<usize>,
        source: &PinnedHostBuffer<u8>,
        source_offset: usize,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying into a loading arena")?;
        if source.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "pinned host source and loading arena must share one CUDA context",
            ));
        }
        self.initialized.require_available(&destination)?;
        let bytes = destination.end - destination.start;
        let source_end = source_offset
            .checked_add(bytes)
            .ok_or_else(|| GpuError::arena("loading-arena source range overflows"))?;
        if source_end > source.len() {
            return Err(GpuError::arena(format!(
                "pinned source range {source_offset}..{source_end} exceeds {} bytes",
                source.len()
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        let address = self.address(destination.start)?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the loading-arena CUDA context", source))?;
        // SAFETY: the caller retains the pinned bytes; both checked ranges cover `bytes`.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr().add(source_offset),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a pinned loading-arena range", source))?;
        self.initialized.record(destination);
        Ok(())
    }

    /// Synchronizes all initialization writes and exposes the complete arena to runtime owners.
    pub fn seal(self, stream: &CudaStream) -> GpuResult<DeviceArena> {
        self.require_stream_context(stream, "sealing a loading arena")?;
        self.initialized.require_complete()?;
        stream.synchronize().map_err(|source| {
            GpuError::driver("synchronizing loading-arena initialization", source)
        })?;
        Ok(DeviceArena {
            storage: self.storage,
            bytes: self.bytes,
        })
    }

    /// Complete allocation bytes that must be initialized before sealing.
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    fn address(&self, offset: usize) -> GpuResult<u64> {
        let offset = u64::try_from(offset).map_err(|_| {
            GpuError::arena("loading-arena offset exceeds the device address width")
        })?;
        self.storage
            .cu_deviceptr()
            .checked_add(offset)
            .ok_or_else(|| GpuError::arena("loading-arena device address overflows"))
    }

    fn require_stream_context(&self, stream: &CudaStream, operation: &str) -> GpuResult<()> {
        if self.storage.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(format!(
                "{operation} requires the allocation and stream to share one CUDA context"
            )));
        }
        if self.stream.cu_stream() != stream.cu_stream() {
            return Err(GpuError::context(format!(
                "{operation} requires the loading arena's allocation stream"
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct InitializationCoverage {
    bytes: usize,
    ranges: BTreeMap<usize, usize>,
}

impl InitializationCoverage {
    const fn new(bytes: usize) -> Self {
        Self {
            bytes,
            ranges: BTreeMap::new(),
        }
    }

    fn require_available(&self, range: &Range<usize>) -> GpuResult<()> {
        if range.start > range.end || range.end > self.bytes {
            return Err(GpuError::arena(format!(
                "loading-arena initialization range {}..{} exceeds {} bytes",
                range.start, range.end, self.bytes
            )));
        }
        if range.is_empty() {
            return Ok(());
        }
        if let Some((&start, &end)) = self.ranges.range(..=range.start).next_back()
            && end > range.start
        {
            return Err(overlap_error(range, start, end));
        }
        if let Some((&start, &end)) = self.ranges.range(range.start..).next()
            && start < range.end
        {
            return Err(overlap_error(range, start, end));
        }
        Ok(())
    }

    fn record(&mut self, range: Range<usize>) {
        if !range.is_empty() {
            self.ranges.insert(range.start, range.end);
        }
    }

    fn require_complete(&self) -> GpuResult<()> {
        let mut cursor = 0;
        for (&start, &end) in &self.ranges {
            if start != cursor {
                return Err(GpuError::arena(format!(
                    "loading arena is uninitialized at {cursor}..{start}"
                )));
            }
            cursor = end;
        }
        if cursor != self.bytes {
            return Err(GpuError::arena(format!(
                "loading arena is uninitialized at {cursor}..{}",
                self.bytes
            )));
        }
        Ok(())
    }
}

fn overlap_error(range: &Range<usize>, start: usize, end: usize) -> GpuError {
    GpuError::arena(format!(
        "loading-arena initialization {}..{} overlaps {start}..{end}",
        range.start, range.end
    ))
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

    /// Enqueues a byte fill over one checked typed subrange.
    pub fn fill_slice<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        start: usize,
        len: usize,
        value: u8,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "filling a device arena subrange")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("arena fill subrange overflows"))?;
        if end > region.len {
            return Err(GpuError::arena(format!(
                "arena fill subrange {start}..{end} exceeds a region of {} elements",
                region.len
            )));
        }
        let element_bytes = size_of::<T>();
        let byte_start = start
            .checked_mul(element_bytes)
            .ok_or_else(|| GpuError::arena("arena fill byte offset overflows"))?;
        let bytes = len
            .checked_mul(element_bytes)
            .ok_or_else(|| GpuError::arena("arena fill byte count overflows"))?;
        if bytes == 0 {
            return Ok(());
        }
        let address = (self.address(region)? as u64)
            .checked_add(u64::try_from(byte_start).map_err(|_| {
                GpuError::arena("arena fill byte offset exceeds the device address width")
            })?)
            .ok_or_else(|| GpuError::arena("arena fill device address overflows"))?;

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the typed element subrange was checked inside the live region.
        unsafe { cuda_core::memory::memset_d8_async(address, value, bytes, stream.cu_stream()) }
            .map_err(|source| GpuError::driver("filling a device arena subrange", source))
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
        self.copy_slice_from_host(stream, region, 0, source)
    }

    /// Copies a typed host slice into one checked subrange of a region.
    pub fn copy_slice_from_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        start: usize,
        source: &[T],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a host slice into a device arena")?;
        let end = start
            .checked_add(source.len())
            .ok_or_else(|| GpuError::arena("arena upload subrange overflows"))?;
        if end > region.len {
            return Err(GpuError::arena(format!(
                "host source has {} elements; arena upload subrange {start}..{end} exceeds a region of {} elements",
                source.len(),
                region.len
            )));
        }
        let byte_start = start
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena("arena upload byte offset overflows"))?;
        let address = (self.address(region)? as u64)
            .checked_add(u64::try_from(byte_start).map_err(|_| {
                GpuError::arena("arena upload byte offset exceeds the device address width")
            })?)
            .ok_or_else(|| GpuError::arena("arena upload device address overflows"))?;
        let bytes = size_of_val(source);
        if bytes == 0 {
            return Ok(());
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the checked typed subrange and source slice both cover `bytes`.
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
        self.copy_slice_to_host(stream, region, 0, len)
    }

    /// Copies one checked typed subrange into an owned host vector.
    pub fn copy_slice_to_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        start: usize,
        len: usize,
    ) -> GpuResult<Vec<T>> {
        self.require_stream_context(stream, "copying a device arena region to the host")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("arena download subrange overflows"))?;
        if end > region.len {
            return Err(GpuError::arena(format!(
                "arena download subrange {start}..{end} exceeds a region of {} elements",
                region.len
            )));
        }
        let byte_start = start
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena("arena download byte offset overflows"))?;
        let address = (self.address(region)? as u64)
            .checked_add(u64::try_from(byte_start).map_err(|_| {
                GpuError::arena("arena download byte offset exceeds the device address width")
            })?)
            .ok_or_else(|| GpuError::arena("arena download device address overflows"))?;
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

    /// Copies the beginning of one region into an existing host slice.
    pub fn copy_prefix_to_host_slice<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        destination: &mut [T],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a device arena region to the host")?;
        if destination.len() > region.len {
            return Err(GpuError::arena(format!(
                "host destination has {} elements for an arena region of {} elements",
                destination.len(),
                region.len
            )));
        }
        let address = self.address(region)? as u64;
        let bytes = size_of_val(destination);
        if bytes == 0 {
            return Ok(());
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the arena CUDA context", source))?;
        // SAFETY: the checked region and initialized destination both cover `bytes`.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                destination.as_mut_ptr(),
                address,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a device arena region to the host", source))?;
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing a device arena download", source))
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
    use super::{ArenaLayout, DeviceArena, InitializationCoverage, LoadingDeviceArena};
    use crate::{CudaContext, GpuErrorCode, PinnedHostBuffer};

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

    #[test]
    fn loading_coverage_accepts_one_out_of_order_write_per_byte() {
        let mut coverage = InitializationCoverage::new(16);

        for range in [8..12, 0..4, 12..16, 4..8] {
            coverage.require_available(&range).unwrap();
            coverage.record(range);
        }

        coverage.require_complete().unwrap();
    }

    #[test]
    fn loading_coverage_rejects_gaps_overlaps_and_out_of_bounds() {
        let mut coverage = InitializationCoverage::new(16);
        coverage.require_available(&(4..12)).unwrap();
        coverage.record(4..12);

        for range in [0..5, 8..16, 12..17] {
            let error = coverage.require_available(&range).unwrap_err();
            assert_eq!(error.code(), GpuErrorCode::Arena);
        }
        let reversed = std::ops::Range { start: 14, end: 13 };
        assert_eq!(
            coverage.require_available(&reversed).unwrap_err().code(),
            GpuErrorCode::Arena
        );
        let error = coverage.require_complete().unwrap_err();
        assert_eq!(error.code(), GpuErrorCode::Arena);
        assert!(error.to_string().contains("uninitialized at 0..4"));
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn loading_arena_seals_only_after_every_byte_is_initialized() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let bytes = layout.reserve::<u8>(8, 256).unwrap();
        let mut source = PinnedHostBuffer::zeroed(&context, 4).unwrap();
        source.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
        let mut loading = LoadingDeviceArena::allocate(&stream, &layout).unwrap();

        loading.fill_async(&stream, 0..4, 0xa5).unwrap();
        // SAFETY: `source` remains immutable and alive through `seal`, which synchronizes the copy.
        unsafe {
            loading
                .copy_from_pinned_host_async(&stream, 4..8, &source, 0)
                .unwrap();
        }
        let arena = loading.seal(&stream).unwrap();

        assert_eq!(
            arena.copy_to_host(&stream, bytes).unwrap(),
            [0xa5, 0xa5, 0xa5, 0xa5, 1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn typed_subrange_copies_preserve_neighboring_values() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let values = layout.reserve::<u32>(8, 256).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();
        let stable_address = arena.address(values).unwrap();

        arena.fill(&stream, values, 0xa5).unwrap();
        arena
            .copy_slice_from_host(&stream, values, 3, &[11, 22])
            .unwrap();

        assert_eq!(
            arena.copy_slice_to_host(&stream, values, 3, 2).unwrap(),
            [11, 22]
        );
        assert_eq!(
            arena.copy_to_host(&stream, values).unwrap(),
            [
                0xa5a5_a5a5,
                0xa5a5_a5a5,
                0xa5a5_a5a5,
                11,
                22,
                0xa5a5_a5a5,
                0xa5a5_a5a5,
                0xa5a5_a5a5
            ]
        );
        assert_eq!(arena.address(values).unwrap(), stable_address);

        for error in [
            arena
                .copy_slice_from_host(&stream, values, 7, &[1, 2])
                .unwrap_err(),
            arena
                .copy_slice_from_host(&stream, values, usize::MAX, &[1])
                .unwrap_err(),
            arena.copy_slice_to_host(&stream, values, 7, 2).unwrap_err(),
            arena
                .copy_slice_to_host(&stream, values, usize::MAX, 1)
                .unwrap_err(),
        ] {
            assert_eq!(error.code(), GpuErrorCode::Arena);
        }
    }
}
