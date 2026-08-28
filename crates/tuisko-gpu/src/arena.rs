//! Typed layout and single-allocation ownership for address-stable workspaces.

use crate::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, GpuError, GpuResult, PinnedHostBuffer,
    VmmSegmentClass, VmmSegmentManifest,
};
use cuda_core::vmm::{Mapping, PhysicalAllocation, VirtualReservation};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, size_of_val};
use std::ops::Range;
use std::sync::Arc;

/// The base alignment the CUDA driver guarantees for every device allocation.
const DRIVER_BASE_ALIGNMENT: usize = 256;

/// Queries the exact device's minimum VMM physical-allocation granularity.
pub fn vmm_allocation_granularity(stream: &CudaStream) -> GpuResult<usize> {
    bind_context(stream, "binding the VMM granularity-query context")?;
    cuda_core::vmm::allocation_granularity(stream.context().cu_device())
        .map_err(|source| GpuError::driver("querying VMM allocation granularity", source))
}

#[cfg(debug_assertions)]
fn next_layout_nonce() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A typed byte range reserved within a device arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaRegion<T> {
    offset: usize,
    len: usize,
    bytes: usize,
    alignment: usize,
    /// Identity of the reserving layout, checked against the arena in debug builds.
    #[cfg(debug_assertions)]
    layout: u64,
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
#[derive(Clone, Debug)]
pub struct ArenaLayout {
    cursor: usize,
    max_alignment: usize,
    /// Identity stamped into every region this layout reserves and into the arenas
    /// allocated from it, so debug builds reject regions from a different layout.
    #[cfg(debug_assertions)]
    nonce: u64,
}

impl Default for ArenaLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaLayout {
    /// Creates an empty layout.
    pub fn new() -> Self {
        Self {
            cursor: 0,
            max_alignment: 1,
            #[cfg(debug_assertions)]
            nonce: next_layout_nonce(),
        }
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
        self.max_alignment = self.max_alignment.max(alignment);

        Ok(ArenaRegion {
            offset,
            len,
            bytes,
            alignment,
            #[cfg(debug_assertions)]
            layout: self.nonce,
            element: PhantomData,
        })
    }

    /// Rejects a real allocation base that cannot satisfy the strictest reserved alignment.
    fn require_base_alignment(&self, base: u64) -> GpuResult<()> {
        if self.max_alignment <= DRIVER_BASE_ALIGNMENT {
            return Ok(());
        }
        let alignment = u64::try_from(self.max_alignment).map_err(|_| {
            GpuError::arena("arena layout alignment exceeds the device address width")
        })?;
        if !base.is_multiple_of(alignment) {
            return Err(GpuError::arena(format!(
                "arena base address {base:#x} does not satisfy the layout's {}-byte maximum alignment; the driver guarantees only {DRIVER_BASE_ALIGNMENT} bytes",
                self.max_alignment
            )));
        }
        Ok(())
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
    storage: ArenaStorage,
    context: Arc<CudaContext>,
    bytes: usize,
    #[cfg(debug_assertions)]
    layout: u64,
}

/// Uninitialized device allocation available only to checked startup writes.
///
/// Every write and the final seal run on the retained allocation stream, so no
/// caller can select a stream that would order them differently.
pub struct LoadingDeviceArena {
    storage: ArenaStorage,
    stream: Arc<CudaStream>,
    bytes: usize,
    initialized: InitializationCoverage,
    #[cfg(debug_assertions)]
    layout: u64,
}

enum ArenaStorage {
    Legacy(DeviceBuffer<u8>),
    Vmm(VmmArenaStorage),
}

struct VmmArenaStorage {
    segments: Vec<VmmSegmentBacking>,
    reservation: VirtualReservation,
    granularity: usize,
    parkable_bytes: usize,
    context: Arc<CudaContext>,
}

struct VmmSegmentBacking {
    mapping: Option<Mapping>,
    allocation: Option<PhysicalAllocation>,
    offset: usize,
    bytes: usize,
    class: VmmSegmentClass,
}

struct NewVmmBacking {
    mapping: Mapping,
    allocation: PhysicalAllocation,
}

impl ArenaStorage {
    fn base_address(&self) -> u64 {
        match self {
            Self::Legacy(storage) => storage.cu_deviceptr(),
            Self::Vmm(storage) => storage.reservation.base(),
        }
    }
}

impl VmmArenaStorage {
    fn allocate(stream: &CudaStream, manifest: &VmmSegmentManifest) -> GpuResult<Self> {
        bind_context(stream, "binding the VMM arena CUDA context")?;
        let device = stream.context().cu_device();
        let granularity = vmm_allocation_granularity(stream)?;
        if granularity != manifest.granularity() {
            return Err(GpuError::arena(format!(
                "VMM manifest granularity {} does not match device granularity {granularity}",
                manifest.granularity()
            )));
        }
        let reservation = VirtualReservation::new(manifest.reservation_bytes(), granularity)
            .map_err(|source| GpuError::driver("reserving VMM arena addresses", source))?;
        let mut segments = Vec::with_capacity(manifest.segments().len());
        for segment in manifest.segments() {
            let allocation = PhysicalAllocation::new(device, segment.bytes())
                .map_err(|source| GpuError::driver("allocating VMM arena backing", source))?;
            let va = reservation
                .base()
                .checked_add(u64::try_from(segment.offset()).map_err(|_| {
                    GpuError::arena("VMM segment offset exceeds the device address width")
                })?)
                .ok_or_else(|| GpuError::arena("VMM segment address overflows"))?;
            let mapping = Mapping::new(va, segment.bytes(), &allocation, 0)
                .map_err(|source| GpuError::driver("mapping VMM arena backing", source))?;
            cuda_core::vmm::set_access(va, segment.bytes(), &[device])
                .map_err(|source| GpuError::driver("granting VMM arena access", source))?;
            segments.push(VmmSegmentBacking {
                mapping: Some(mapping),
                allocation: Some(allocation),
                offset: segment.offset(),
                bytes: segment.bytes(),
                class: segment.class(),
            });
        }

        Ok(Self {
            segments,
            reservation,
            granularity,
            parkable_bytes: manifest.parkable_bytes(),
            context: stream.context().clone(),
        })
    }

    fn park(&mut self, stream: &CudaStream) -> GpuResult<usize> {
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing before VMM arena park", source))?;
        for segment in &mut self.segments {
            if segment.class == VmmSegmentClass::Parkable {
                drop(segment.mapping.take());
                drop(segment.allocation.take());
            }
        }
        Ok(self.parkable_bytes)
    }

    fn resume(&mut self, stream: &CudaStream) -> GpuResult<usize> {
        bind_context(stream, "binding the VMM arena resume context")?;
        let device = stream.context().cu_device();
        let mut replacements = Vec::new();
        let mut resumed_bytes = 0usize;
        for (index, segment) in self.segments.iter().enumerate() {
            if segment.class != VmmSegmentClass::Parkable {
                continue;
            }
            if segment.mapping.is_some() && segment.allocation.is_some() {
                continue;
            }
            if segment.mapping.is_some() || segment.allocation.is_some() {
                return Err(GpuError::arena(
                    "parkable VMM segment retained partial physical ownership",
                ));
            }
            let allocation = PhysicalAllocation::new(device, segment.bytes)
                .map_err(|source| GpuError::driver("allocating resumed VMM backing", source))?;
            let va = self
                .reservation
                .base()
                .checked_add(u64::try_from(segment.offset).map_err(|_| {
                    GpuError::arena("resumed VMM segment offset exceeds device width")
                })?)
                .ok_or_else(|| GpuError::arena("resumed VMM segment address overflows"))?;
            let mapping = Mapping::new(va, segment.bytes, &allocation, 0)
                .map_err(|source| GpuError::driver("mapping resumed VMM backing", source))?;
            cuda_core::vmm::set_access(va, segment.bytes, &[device])
                .map_err(|source| GpuError::driver("granting resumed VMM access", source))?;
            replacements.push((
                index,
                NewVmmBacking {
                    mapping,
                    allocation,
                },
            ));
            resumed_bytes = resumed_bytes
                .checked_add(segment.bytes)
                .ok_or_else(|| GpuError::arena("resumed VMM byte count overflows"))?;
        }
        for (index, replacement) in replacements {
            let segment = &mut self.segments[index];
            segment.mapping = Some(replacement.mapping);
            segment.allocation = Some(replacement.allocation);
        }
        Ok(resumed_bytes)
    }

    fn is_parked(&self) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.class == VmmSegmentClass::Parkable && segment.mapping.is_none())
    }
}

impl Drop for VmmArenaStorage {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
    }
}

impl LoadingDeviceArena {
    /// Allocates the storage required by `layout` without enqueueing a full-arena memset.
    pub fn allocate(stream: &Arc<CudaStream>, layout: &ArenaLayout) -> GpuResult<Self> {
        bind_context(stream, "binding the loading-arena CUDA context")?;
        // SAFETY: this type exposes only writes until `seal` proves complete initialization and
        // synchronizes their stream. No safe operation can read the returned storage beforehand.
        let storage = unsafe { DeviceBuffer::uninitialized_async(stream, layout.byte_len()) }
            .map_err(|source| {
                GpuError::driver("allocating an uninitialized device arena", source)
            })?;
        layout.require_base_alignment(storage.cu_deviceptr())?;

        Ok(Self {
            storage: ArenaStorage::Legacy(storage),
            stream: stream.clone(),
            bytes: layout.byte_len(),
            initialized: InitializationCoverage::new(layout.byte_len()),
            #[cfg(debug_assertions)]
            layout: layout.nonce,
        })
    }

    /// Allocates checked VMM storage without initializing its typed layout bytes.
    pub fn allocate_vmm(
        stream: &Arc<CudaStream>,
        layout: &ArenaLayout,
        manifest: &VmmSegmentManifest,
    ) -> GpuResult<Self> {
        if manifest.arena_bytes() != layout.byte_len() {
            return Err(GpuError::arena(format!(
                "VMM manifest covers {} arena bytes for a {}-byte layout",
                manifest.arena_bytes(),
                layout.byte_len()
            )));
        }
        let storage = ArenaStorage::Vmm(VmmArenaStorage::allocate(stream, manifest)?);
        layout.require_base_alignment(storage.base_address())?;
        Ok(Self {
            storage,
            stream: stream.clone(),
            bytes: layout.byte_len(),
            initialized: InitializationCoverage::new(layout.byte_len()),
            #[cfg(debug_assertions)]
            layout: layout.nonce,
        })
    }

    /// Enqueues a byte fill over one not-yet-initialized destination range.
    pub fn fill_async(&mut self, destination: Range<usize>, value: u8) -> GpuResult<()> {
        self.initialized.require_available(&destination)?;
        let bytes = destination.end - destination.start;
        if bytes == 0 {
            return Ok(());
        }
        let address = self.address(destination.start)?;
        self.bind_context()?;
        // SAFETY: coverage validation proved this byte range is inside the live allocation.
        unsafe {
            cuda_core::memory::memset_d8_async(address, value, bytes, self.stream.cu_stream())
        }
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
    /// The selected pinned source range must remain allocated and immutable until the retained
    /// stream reaches the copy. An event wait or synchronization must precede buffer reuse.
    pub unsafe fn copy_from_pinned_host_async(
        &mut self,
        destination: Range<usize>,
        source: &PinnedHostBuffer<u8>,
        source_offset: usize,
    ) -> GpuResult<()> {
        if source.context().as_ref() != self.stream.context().as_ref() {
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
        self.bind_context()?;
        // SAFETY: the caller retains the pinned bytes; both checked ranges cover `bytes`.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr().add(source_offset),
                bytes,
                self.stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a pinned loading-arena range", source))?;
        self.initialized.record(destination);
        Ok(())
    }

    /// Enqueues a host-slice copy into one not-yet-initialized destination range.
    ///
    /// # Safety
    ///
    /// `source` must remain allocated and immutable until the retained stream reaches the copy.
    /// The caller must synchronize that stream before the source can be released or changed.
    pub unsafe fn copy_from_host_async<T: DeviceCopy>(
        &mut self,
        destination: Range<usize>,
        source: &[T],
    ) -> GpuResult<()> {
        self.initialized.require_available(&destination)?;
        let bytes = destination.end - destination.start;
        if bytes != size_of_val(source) {
            return Err(GpuError::arena(format!(
                "host source has {} bytes for loading-arena destination {}..{} ({} bytes)",
                size_of_val(source),
                destination.start,
                destination.end,
                bytes,
            )));
        }
        if bytes == 0 {
            return Ok(());
        }
        let address = self.address(destination.start)?;
        self.bind_context()?;
        // SAFETY: the caller owns the source lifetime; coverage validation proved the destination
        // range is inside the live allocation and both ranges contain exactly `bytes`.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr(),
                bytes,
                self.stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a host slice into a loading arena", source))?;
        self.initialized.record(destination);
        Ok(())
    }

    /// Synchronizes all initialization writes and exposes the complete arena to runtime owners.
    pub fn seal(self) -> GpuResult<DeviceArena> {
        self.initialized.require_complete()?;
        self.stream.synchronize().map_err(|source| {
            GpuError::driver("synchronizing loading-arena initialization", source)
        })?;
        Ok(DeviceArena {
            storage: self.storage,
            context: self.stream.context().clone(),
            bytes: self.bytes,
            #[cfg(debug_assertions)]
            layout: self.layout,
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
            .base_address()
            .checked_add(offset)
            .ok_or_else(|| GpuError::arena("loading-arena device address overflows"))
    }

    fn bind_context(&self) -> GpuResult<()> {
        bind_context(&self.stream, "binding the loading-arena CUDA context")
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

/// Drains the context after a failed download so the enqueued copy cannot outlive the
/// host destination this frame is about to release, recording that failure as well.
fn drain_failed_download(stream: &CudaStream) {
    stream.context().record_err(stream.context().synchronize());
}

fn overlap_error(range: &Range<usize>, start: usize, end: usize) -> GpuError {
    GpuError::arena(format!(
        "loading-arena initialization {}..{} overlaps {start}..{end}",
        range.start, range.end
    ))
}

/// Makes `stream`'s context current, naming the failing operation.
fn bind_context(stream: &CudaStream, operation: &'static str) -> GpuResult<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|source| GpuError::driver(operation, source))
}

/// Enqueues one host-to-device copy and drains `stream` before returning.
///
/// # Safety
///
/// `source..source + bytes` must be a live host range, and `address..address + bytes` must lie
/// inside a device allocation that stays live until `stream` completes the copy.
unsafe fn upload<T>(
    stream: &CudaStream,
    address: u64,
    source: *const T,
    bytes: usize,
    operation: &'static str,
) -> GpuResult<()> {
    bind_context(stream, "binding the arena CUDA context")?;
    // SAFETY: the caller proved both ranges cover `bytes`.
    unsafe { cuda_core::memory::memcpy_htod_async(address, source, bytes, stream.cu_stream()) }
        .map_err(|source| GpuError::driver(operation, source))?;
    stream
        .synchronize()
        .map_err(|source| GpuError::driver("synchronizing a device arena upload", source))
}

/// Enqueues one device-to-host copy and drains `stream` before returning.
///
/// # Safety
///
/// `destination..destination + bytes` must be a live writable host range, and
/// `address..address + bytes` must lie inside a device allocation that stays live until
/// `stream` completes the copy.
unsafe fn download<T>(
    stream: &CudaStream,
    destination: *mut T,
    address: u64,
    bytes: usize,
    operation: &'static str,
) -> GpuResult<()> {
    bind_context(stream, "binding the arena CUDA context")?;
    // SAFETY: the caller proved both ranges cover `bytes`.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(destination, address, bytes, stream.cu_stream())
    }
    .map_err(|source| GpuError::driver(operation, source))?;
    stream.synchronize().map_err(|source| {
        drain_failed_download(stream);
        GpuError::driver("synchronizing a device arena download", source)
    })
}

impl DeviceArena {
    /// Allocates and zeroes the storage required by `layout`.
    ///
    /// Synchronizes `stream` before returning, so later operations observe the
    /// zeroed bytes from any stream in the same context.
    pub fn zeroed(stream: &CudaStream, layout: &ArenaLayout) -> GpuResult<Self> {
        bind_context(stream, "binding the arena CUDA context")?;
        let storage = DeviceBuffer::zeroed(stream, layout.byte_len())
            .map_err(|source| GpuError::driver("allocating a zeroed device arena", source))?;
        layout.require_base_alignment(storage.cu_deviceptr())?;
        // The zeroing memset is only enqueued on `stream`, while later arena
        // operations accept any same-context stream; draining `stream` here
        // keeps a cross-stream reader from observing uninitialized memory.
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing device arena zeroing", source))?;

        Ok(Self {
            storage: ArenaStorage::Legacy(storage),
            context: stream.context().clone(),
            bytes: layout.byte_len(),
            #[cfg(debug_assertions)]
            layout: layout.nonce,
        })
    }

    /// Allocates and zeroes a checked VMM-backed arena.
    pub fn zeroed_vmm(
        stream: &CudaStream,
        layout: &ArenaLayout,
        manifest: &VmmSegmentManifest,
    ) -> GpuResult<Self> {
        if manifest.arena_bytes() != layout.byte_len() {
            return Err(GpuError::arena(format!(
                "VMM manifest covers {} arena bytes for a {}-byte layout",
                manifest.arena_bytes(),
                layout.byte_len()
            )));
        }
        let storage = ArenaStorage::Vmm(VmmArenaStorage::allocate(stream, manifest)?);
        layout.require_base_alignment(storage.base_address())?;
        if layout.byte_len() != 0 {
            // SAFETY: every byte in the typed layout is mapped and writable on this context.
            unsafe {
                cuda_core::memory::memset_d8_async(
                    storage.base_address(),
                    0,
                    layout.byte_len(),
                    stream.cu_stream(),
                )
            }
            .map_err(|source| GpuError::driver("zeroing a VMM device arena", source))?;
            stream
                .synchronize()
                .map_err(|source| GpuError::driver("synchronizing VMM arena zeroing", source))?;
        }
        Ok(Self {
            storage,
            context: stream.context().clone(),
            bytes: layout.byte_len(),
            #[cfg(debug_assertions)]
            layout: layout.nonce,
        })
    }

    /// Returns the stable base device address.
    pub fn base_address(&self) -> u64 {
        self.storage.base_address()
    }

    /// Returns the allocation size in bytes.
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Physical device bytes currently mapped by this arena.
    pub fn mapped_physical_bytes(&self) -> usize {
        match &self.storage {
            ArenaStorage::Legacy(_) => self.bytes,
            ArenaStorage::Vmm(storage) => storage
                .segments
                .iter()
                .filter(|segment| segment.allocation.is_some())
                .map(|segment| segment.bytes)
                .sum(),
        }
    }

    /// Returns whether this arena owns no device bytes.
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Releases every parkable VMM mapping after draining the supplied stream.
    pub fn park(&mut self, stream: &CudaStream) -> GpuResult<usize> {
        self.require_stream_context(stream, "parking a VMM device arena")?;
        match &mut self.storage {
            ArenaStorage::Vmm(storage) if !storage.is_parked() => storage.park(stream),
            ArenaStorage::Vmm(_) => Err(GpuError::arena("VMM device arena is already parked")),
            ArenaStorage::Legacy(_) => Err(GpuError::arena(
                "legacy device arena cannot preserve addresses while parked",
            )),
        }
    }

    /// Recreates every released parkable mapping at its retained virtual address.
    pub fn resume(&mut self, stream: &CudaStream) -> GpuResult<usize> {
        self.require_stream_context(stream, "resuming a VMM device arena")?;
        match &mut self.storage {
            ArenaStorage::Vmm(storage) => storage.resume(stream),
            ArenaStorage::Legacy(_) => Err(GpuError::arena(
                "legacy device arena cannot resume retained addresses",
            )),
        }
    }

    /// Minimum allocation granularity when this arena is VMM-backed.
    pub fn vmm_granularity(&self) -> Option<usize> {
        match &self.storage {
            ArenaStorage::Vmm(storage) => Some(storage.granularity),
            ArenaStorage::Legacy(_) => None,
        }
    }

    /// Physical bytes released by [`Self::park`] for a VMM-backed arena.
    pub fn parkable_bytes(&self) -> usize {
        match &self.storage {
            ArenaStorage::Vmm(storage) => storage.parkable_bytes,
            ArenaStorage::Legacy(_) => 0,
        }
    }

    /// Returns the checked device address of `region`.
    pub fn address<T: DeviceCopy>(&self, region: ArenaRegion<T>) -> GpuResult<*mut T> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            region.layout, self.layout,
            "arena region was reserved from a different arena layout"
        );
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

    /// Returns the checked device address and byte count of one typed element subrange.
    ///
    /// `label` names the operation in every rejection this validation can produce.
    fn subrange_address<T: DeviceCopy>(
        &self,
        region: ArenaRegion<T>,
        start: usize,
        len: usize,
        label: &str,
    ) -> GpuResult<(u64, usize)> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena(format!("{label} subrange overflows")))?;
        if end > region.len {
            return Err(GpuError::arena(format!(
                "{label} subrange {start}..{end} exceeds a region of {} elements",
                region.len
            )));
        }
        let byte_start = start
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena(format!("{label} byte offset overflows")))?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::arena(format!("{label} byte count overflows")))?;
        let address = (self.address(region)? as u64)
            .checked_add(u64::try_from(byte_start).map_err(|_| {
                GpuError::arena(format!(
                    "{label} byte offset exceeds the device address width"
                ))
            })?)
            .ok_or_else(|| GpuError::arena(format!("{label} device address overflows")))?;

        Ok((address, bytes))
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

        bind_context(stream, "binding the arena CUDA context")?;
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
        // Empty fills still validate region identity and bounds.
        let (address, bytes) = self.subrange_address(region, start, len, "arena fill")?;
        if bytes == 0 {
            return Ok(());
        }

        bind_context(stream, "binding the arena CUDA context")?;
        // SAFETY: the typed element subrange was checked inside the live region.
        unsafe { cuda_core::memory::memset_d8_async(address, value, bytes, stream.cu_stream()) }
            .map_err(|source| GpuError::driver("filling a device arena subrange", source))
    }

    /// Copies the complete arena into a host byte vector.
    pub fn to_host_vec(&self, stream: &CudaStream) -> GpuResult<Vec<u8>> {
        self.require_stream_context(stream, "copying a device arena to the host")?;
        let mut host = vec![0u8; self.bytes];
        if !host.is_empty() {
            // SAFETY: the host vector and checked live arena both cover `self.bytes` bytes.
            unsafe {
                download(
                    stream,
                    host.as_mut_ptr(),
                    self.base_address(),
                    self.bytes,
                    "copying a device arena to the host",
                )?;
            }
        }
        Ok(host)
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

    /// Copies exact source bytes into one complete typed region.
    pub fn copy_region_bytes_from_host<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[u8],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying source bytes into a device arena")?;
        if source.len() != region.byte_len() {
            return Err(GpuError::arena(format!(
                "host source has {} bytes for an arena region of {} bytes",
                source.len(),
                region.byte_len()
            )));
        }
        if source.is_empty() {
            return Ok(());
        }
        let address = self.address(region)? as u64;
        // SAFETY: the checked typed region and source slice both cover the exact byte count.
        unsafe {
            upload(
                stream,
                address,
                source.as_ptr(),
                source.len(),
                "copying source bytes into a device arena",
            )
        }
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
        let (address, bytes) =
            self.subrange_address(region, start, source.len(), "arena upload")?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: the checked typed subrange and source slice both cover `bytes`.
        unsafe {
            upload(
                stream,
                address,
                source.as_ptr(),
                bytes,
                "copying a host slice into a device arena",
            )
        }
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
        unsafe { self.copy_slice_from_pinned_host_async(stream, region, 0, source, 0, len) }
    }

    /// Enqueues a checked pinned-host subrange upload without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// `source` must remain allocated and immutable until the copy completes. Captured copies
    /// require the source and destination addresses to remain stable through final replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_slice_from_pinned_host_async<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        destination_start: usize,
        source: &PinnedHostBuffer<T>,
        source_start: usize,
        len: usize,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a pinned host prefix into a device arena")?;
        if source.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "pinned host source and arena stream must share one CUDA context",
            ));
        }
        let destination_end = destination_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("pinned upload destination range overflows"))?;
        let source_end = source_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("pinned upload source range overflows"))?;
        if destination_end > region.len || source_end > source.len() {
            return Err(GpuError::arena(format!(
                "pinned upload ranges {source_start}..{source_end} and {destination_start}..{destination_end} exceed source {} or region {} elements",
                source.len(),
                region.len
            )));
        }
        let (address, bytes) =
            self.subrange_address(region, destination_start, len, "pinned upload")?;
        if bytes == 0 {
            return Ok(());
        }

        bind_context(stream, "binding the arena CUDA context")?;
        // SAFETY: the checked arena region covers `bytes`; the caller owns the pinned-source
        // lifetime and immutability contract documented by this method.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                address,
                source.as_ptr().add(source_start),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying a pinned host prefix into an arena", source))
    }

    /// Enqueues a checked device subrange download into pinned host memory.
    ///
    /// # Safety
    ///
    /// `destination` must remain allocated and must not be read or mutated until the stream
    /// reaches the copy. Captured copies require stable source and destination addresses.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_slice_to_pinned_host_async<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source_start: usize,
        destination: &mut PinnedHostBuffer<T>,
        destination_start: usize,
        len: usize,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a device arena into pinned host memory")?;
        if destination.context().as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "pinned host destination and arena stream must share one CUDA context",
            ));
        }
        let source_end = source_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("pinned download source range overflows"))?;
        let destination_end = destination_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("pinned download destination range overflows"))?;
        if source_end > region.len || destination_end > destination.len() {
            return Err(GpuError::arena(format!(
                "pinned download ranges {source_start}..{source_end} and {destination_start}..{destination_end} exceed region {} or destination {} elements",
                region.len,
                destination.len()
            )));
        }
        let (address, bytes) =
            self.subrange_address(region, source_start, len, "pinned download")?;
        if bytes == 0 {
            return Ok(());
        }

        bind_context(
            stream,
            "binding the arena CUDA context for a pinned download",
        )?;
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                destination.as_mut_ptr().add(destination_start),
                address,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying an arena into pinned host memory", source))
    }

    /// Enqueues a typed prefix copy between two checked arena regions.
    ///
    /// # Safety
    ///
    /// Both arenas must remain live until the stream reaches the copy. If the copy is captured in
    /// a CUDA Graph, both arena allocations and their addresses must remain stable through the
    /// final graph replay. Source and destination prefixes must not overlap unless identical.
    pub unsafe fn copy_prefix_from_arena_async<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        destination: ArenaRegion<T>,
        source_arena: &DeviceArena,
        source: ArenaRegion<T>,
        len: usize,
    ) -> GpuResult<()> {
        unsafe {
            self.copy_slice_from_arena_async(stream, destination, 0, source_arena, source, 0, len)
        }
    }

    /// Enqueues a typed subrange copy between two checked arena regions.
    ///
    /// # Safety
    ///
    /// Both arenas must remain live until the stream reaches the copy. Captured copies require
    /// stable arena addresses through their final replay. The selected subranges must not overlap
    /// unless their addresses are identical.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_slice_from_arena_async<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        destination: ArenaRegion<T>,
        destination_start: usize,
        source_arena: &DeviceArena,
        source: ArenaRegion<T>,
        source_start: usize,
        len: usize,
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying between device arenas")?;
        source_arena.require_stream_context(stream, "copying between device arenas")?;
        let destination_end = destination_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("device-copy destination subrange overflows"))?;
        let source_end = source_start
            .checked_add(len)
            .ok_or_else(|| GpuError::arena("device-copy source subrange overflows"))?;
        if destination_end > destination.len || source_end > source.len {
            return Err(GpuError::arena(format!(
                "device copy selects destination {destination_start}..{destination_end} of {} and source {source_start}..{source_end} of {} elements",
                destination.len, source.len,
            )));
        }
        let (destination_address, bytes) = self.subrange_address(
            destination,
            destination_start,
            len,
            "device-copy destination",
        )?;
        let (source_address, _) =
            source_arena.subrange_address(source, source_start, len, "device-copy source")?;
        if bytes == 0 {
            return Ok(());
        }
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| GpuError::arena("device-copy byte count exceeds the address width"))?;
        let destination_limit = destination_address
            .checked_add(bytes_u64)
            .ok_or_else(|| GpuError::arena("device-copy destination range overflows"))?;
        let source_limit = source_address
            .checked_add(bytes_u64)
            .ok_or_else(|| GpuError::arena("device-copy source range overflows"))?;
        if destination_address != source_address
            && destination_address < source_limit
            && source_address < destination_limit
        {
            return Err(GpuError::arena(
                "device-copy source and destination subranges overlap",
            ));
        }

        bind_context(stream, "binding the device-copy CUDA context")?;
        // SAFETY: both checked typed prefixes cover `bytes`; the caller owns the asynchronous
        // lifetimes and non-overlap contract documented by this method.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                destination_address,
                source_address,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("copying between device arenas", source))
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
        let (address, bytes) = self.subrange_address(region, start, len, "arena download")?;
        let mut host = Vec::with_capacity(len);
        if bytes == 0 {
            return Ok(host);
        }

        // SAFETY: the checked region and reserved vector capacity both cover `bytes`.
        unsafe {
            download(
                stream,
                host.as_mut_ptr(),
                address,
                bytes,
                "copying a device arena region to the host",
            )?;
        }
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
        self.copy_slice_to_host_slice(stream, region, 0, destination)
    }

    /// Copies one checked typed subrange into an existing host slice.
    pub fn copy_slice_to_host_slice<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        start: usize,
        destination: &mut [T],
    ) -> GpuResult<()> {
        self.require_stream_context(stream, "copying a device arena region to the host")?;
        let (address, bytes) =
            self.subrange_address(region, start, destination.len(), "arena download")?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: the checked region and initialized destination both cover `bytes`.
        unsafe {
            download(
                stream,
                destination.as_mut_ptr(),
                address,
                bytes,
                "copying a device arena region to the host",
            )
        }
    }

    fn require_stream_context(&self, stream: &CudaStream, operation: &str) -> GpuResult<()> {
        if self.context.as_ref() != stream.context().as_ref() {
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
    use crate::{
        CudaContext, CudaGraph, GpuErrorCode, PinnedHostBuffer, VmmSegmentClass,
        VmmSegmentManifest, device_memory_info, vmm_allocation_granularity,
    };

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
        assert_eq!(layout.max_alignment, 256);
        #[cfg(debug_assertions)]
        assert_ne!(ArenaLayout::new().nonce, ArenaLayout::new().nonce);
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

        loading.fill_async(0..4, 0xa5).unwrap();
        // SAFETY: `source` remains immutable and alive through `seal`, which synchronizes the copy.
        unsafe {
            loading
                .copy_from_pinned_host_async(4..8, &source, 0)
                .unwrap();
        }
        let arena = loading.seal().unwrap();

        assert_eq!(
            arena.copy_to_host(&stream, bytes).unwrap(),
            [0xa5, 0xa5, 0xa5, 0xa5, 1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn region_byte_upload_preserves_exact_little_endian_words() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let words = layout.reserve::<u16>(2, 256).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();

        arena
            .copy_region_bytes_from_host(&stream, words, &[0x80, 0x3f, 0x00, 0xbf])
            .unwrap();
        assert_eq!(
            arena.copy_to_host(&stream, words).unwrap(),
            [0x3f80, 0xbf00]
        );
        assert!(
            arena
                .copy_region_bytes_from_host(&stream, words, &[0; 3])
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn vmm_park_resume_preserves_addresses_and_captured_graph_replay() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let granularity = vmm_allocation_granularity(&stream).unwrap();
        let mut layout = ArenaLayout::new();
        let values = layout.reserve::<u32>(granularity / 4, 256).unwrap();
        let manifest =
            VmmSegmentManifest::uniform(layout.byte_len(), granularity, VmmSegmentClass::Parkable)
                .unwrap();
        let mut arena = DeviceArena::zeroed_vmm(&stream, &layout, &manifest).unwrap();
        let base_address = arena.base_address();
        let values_address = arena.address(values).unwrap();
        let graph = CudaGraph::capture(&stream, || arena.fill(&stream, values, 0x5a)).unwrap();

        // SAFETY: the graph references only `arena`, whose captured virtual address remains
        // reserved through both launches and is mapped during each replay.
        unsafe {
            graph.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert!(
            arena
                .copy_to_host(&stream, values)
                .unwrap()
                .iter()
                .all(|value| *value == 0x5a5a_5a5a)
        );

        let mapped_memory = device_memory_info(&context).unwrap();
        assert_eq!(arena.park(&stream).unwrap(), granularity);
        let parked_memory = device_memory_info(&context).unwrap();
        assert!(parked_memory.free_bytes >= mapped_memory.free_bytes + granularity);
        assert_eq!(arena.mapped_physical_bytes(), 0);
        assert_eq!(arena.base_address(), base_address);
        assert_eq!(arena.address(values).unwrap(), values_address);
        assert_eq!(arena.resume(&stream).unwrap(), granularity);
        let resumed_memory = device_memory_info(&context).unwrap();
        assert!(resumed_memory.free_bytes + granularity <= parked_memory.free_bytes);
        assert_eq!(arena.mapped_physical_bytes(), granularity);
        assert_eq!(arena.base_address(), base_address);
        assert_eq!(arena.address(values).unwrap(), values_address);

        // SAFETY: resume remapped the graph's captured virtual address before replay.
        unsafe {
            graph.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert!(
            arena
                .copy_to_host(&stream, values)
                .unwrap()
                .iter()
                .all(|value| *value == 0x5a5a_5a5a)
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
        // SAFETY: the selected same-arena ranges are disjoint and the stream is synchronized by
        // the following host download.
        unsafe {
            arena
                .copy_slice_from_arena_async(&stream, values, 0, &arena, values, 3, 2)
                .unwrap();
        }

        let mut copied = [0; 2];
        arena
            .copy_slice_to_host_slice(&stream, values, 3, &mut copied)
            .unwrap();
        assert_eq!(copied, [11, 22]);
        assert_eq!(
            arena.copy_slice_to_host(&stream, values, 3, 2).unwrap(),
            [11, 22]
        );
        assert_eq!(
            arena.copy_to_host(&stream, values).unwrap(),
            [
                11,
                22,
                0xa5a5_a5a5,
                11,
                22,
                0xa5a5_a5a5,
                0xa5a5_a5a5,
                0xa5a5_a5a5
            ]
        );
        assert_eq!(arena.address(values).unwrap(), stable_address);

        let mut pinned_source = PinnedHostBuffer::zeroed(&context, 4).unwrap();
        pinned_source
            .as_mut_slice()
            .copy_from_slice(&[31, 41, 59, 26]);
        let mut pinned_destination = PinnedHostBuffer::zeroed(&context, 4).unwrap();
        // SAFETY: both pinned buffers remain live and untouched through the synchronization.
        unsafe {
            arena
                .copy_slice_from_pinned_host_async(&stream, values, 5, &pinned_source, 1, 2)
                .unwrap();
            arena
                .copy_slice_to_pinned_host_async(&stream, values, 5, &mut pinned_destination, 1, 2)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(pinned_destination.as_slice(), &[0, 41, 59, 0]);

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
            arena
                .copy_slice_to_host_slice(&stream, values, 7, &mut [0; 2])
                .unwrap_err(),
        ] {
            assert_eq!(error.code(), GpuErrorCode::Arena);
        }
        // SAFETY: these calls deliberately exercise checked range rejection before enqueue.
        let overlap = unsafe {
            arena
                .copy_slice_from_arena_async(&stream, values, 0, &arena, values, 1, 2)
                .unwrap_err()
        };
        // SAFETY: this deliberately exercises checked destination bounds before enqueue.
        let out_of_bounds = unsafe {
            arena
                .copy_slice_from_arena_async(&stream, values, 7, &arena, values, 0, 2)
                .unwrap_err()
        };
        assert_eq!(overlap.code(), GpuErrorCode::Arena);
        assert_eq!(out_of_bounds.code(), GpuErrorCode::Arena);
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn zero_length_fill_validates_bounds_and_debug_layout_identity() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let values = layout.reserve::<u32>(8, 256).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();

        let error = arena
            .fill_slice(&stream, values, values.len() + 1, 0, 0)
            .unwrap_err();
        assert_eq!(error.code(), GpuErrorCode::Arena);

        #[cfg(debug_assertions)]
        {
            let mut foreign_layout = ArenaLayout::new();
            let foreign = foreign_layout.reserve::<u32>(8, 256).unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                arena.fill_slice(&stream, foreign, 0, 0, 0)
            }));
            assert!(result.is_err());
        }
    }
}
