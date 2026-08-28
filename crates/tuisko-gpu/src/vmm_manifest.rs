//! Checked physical-segment planning for a future VMM-backed arena.

use crate::{GpuError, GpuResult};
use std::ops::Range;

/// Physical lifetime assigned to one or more adjacent VMM granules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmmSegmentClass {
    /// Backing must remain present for the loaded and parked owner.
    Resident,
    /// Backing may be released while the owner is parked.
    Parkable,
}

/// One coalesced granularity-aligned virtual-address segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmmSegment {
    offset: usize,
    bytes: usize,
    class: VmmSegmentClass,
}

impl VmmSegment {
    /// Byte offset from the arena reservation base.
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Granularity-aligned byte length.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Physical lifetime of this segment.
    pub const fn class(&self) -> VmmSegmentClass {
        self.class
    }
}

/// Complete granule classification for one virtual-address reservation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmmSegmentManifest {
    arena_bytes: usize,
    reservation_bytes: usize,
    granularity: usize,
    segments: Vec<VmmSegment>,
}

impl VmmSegmentManifest {
    /// Classifies every granule from declared resident and parkable byte ranges.
    ///
    /// A mixed granule is resident. Every arena byte must be covered by exactly one class; only
    /// reservation tail padding is implicitly parkable.
    pub fn build(
        arena_bytes: usize,
        granularity: usize,
        resident_ranges: &[Range<usize>],
        parkable_ranges: &[Range<usize>],
    ) -> GpuResult<Self> {
        if arena_bytes == 0 {
            return Err(GpuError::arena("a VMM arena cannot be empty"));
        }
        if !granularity.is_power_of_two() {
            return Err(GpuError::arena(format!(
                "VMM granularity {granularity} is not a power of two"
            )));
        }
        for range in resident_ranges.iter().chain(parkable_ranges) {
            if range.start >= range.end || range.end > arena_bytes {
                return Err(GpuError::arena(format!(
                    "resident VMM range {}..{} is empty or exceeds arena bytes {arena_bytes}",
                    range.start, range.end
                )));
            }
        }
        if resident_ranges.iter().any(|resident| {
            parkable_ranges
                .iter()
                .any(|parkable| resident.start < parkable.end && parkable.start < resident.end)
        }) {
            return Err(GpuError::arena(
                "resident and parkable VMM byte ranges overlap",
            ));
        }
        let mut parkable_ranges = parkable_ranges.to_vec();
        parkable_ranges.sort_unstable_by_key(|range| range.start);
        let mut declared_ranges = resident_ranges
            .iter()
            .chain(&parkable_ranges)
            .cloned()
            .collect::<Vec<_>>();
        declared_ranges.sort_unstable_by_key(|range| range.start);
        let reservation_bytes = arena_bytes
            .checked_add(granularity - 1)
            .map(|bytes| bytes & !(granularity - 1))
            .ok_or_else(|| GpuError::arena("VMM reservation size overflows"))?;
        let mut segments: Vec<VmmSegment> = Vec::new();
        for offset in (0..reservation_bytes).step_by(granularity) {
            let end = offset + granularity;
            if offset < arena_bytes && !covers(&declared_ranges, offset..end.min(arena_bytes)) {
                return Err(GpuError::arena(format!(
                    "VMM granule {offset}..{end} contains undeclared arena bytes"
                )));
            }
            let class = if resident_ranges
                .iter()
                .any(|range| range.start < end && offset < range.end)
            {
                VmmSegmentClass::Resident
            } else if offset >= arena_bytes
                || covers(&parkable_ranges, offset..end.min(arena_bytes))
            {
                VmmSegmentClass::Parkable
            } else {
                return Err(GpuError::arena(format!(
                    "VMM granule {offset}..{end} is neither resident nor wholly parkable"
                )));
            };
            if let Some(last) = segments.last_mut()
                && last.class == class
                && last.offset + last.bytes == offset
            {
                last.bytes += granularity;
            } else {
                segments.push(VmmSegment {
                    offset,
                    bytes: granularity,
                    class,
                });
            }
        }

        Ok(Self {
            arena_bytes,
            reservation_bytes,
            granularity,
            segments,
        })
    }

    /// Typed-layout bytes before VMM tail rounding.
    pub const fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }

    /// Virtual reservation and physical coverage bytes after tail rounding.
    pub const fn reservation_bytes(&self) -> usize {
        self.reservation_bytes
    }

    /// Minimum physical allocation and mapping granularity.
    pub const fn granularity(&self) -> usize {
        self.granularity
    }

    /// Coalesced complete segment inventory in ascending address order.
    pub fn segments(&self) -> &[VmmSegment] {
        &self.segments
    }

    /// Physical bytes releasable while parked.
    pub fn parkable_bytes(&self) -> usize {
        self.segments
            .iter()
            .filter(|segment| segment.class == VmmSegmentClass::Parkable)
            .map(|segment| segment.bytes)
            .sum()
    }
}

fn covers(ranges: &[Range<usize>], required: Range<usize>) -> bool {
    let mut cursor = required.start;
    for range in ranges {
        if range.end <= cursor {
            continue;
        }
        if range.start > cursor {
            return false;
        }
        cursor = cursor.max(range.end);
        if cursor >= required.end {
            return true;
        }
    }
    cursor >= required.end
}

#[cfg(test)]
mod tests {
    use super::{VmmSegmentClass, VmmSegmentManifest};

    #[test]
    fn mixed_granules_remain_resident_and_adjacent_classes_coalesce() {
        let manifest = VmmSegmentManifest::build(
            1_000,
            256,
            &[100..300, 800..900],
            &[0..100, 300..800, 900..1_000],
        )
        .unwrap();

        assert_eq!(manifest.reservation_bytes(), 1_024);
        assert_eq!(manifest.parkable_bytes(), 256);
        assert_eq!(manifest.segments().len(), 3);
        assert_eq!(manifest.segments()[0].class(), VmmSegmentClass::Resident);
        assert_eq!(manifest.segments()[0].bytes(), 512);
        assert_eq!(manifest.segments()[1].class(), VmmSegmentClass::Parkable);
        assert_eq!(manifest.segments()[1].offset(), 512);
        assert_eq!(manifest.segments()[2].class(), VmmSegmentClass::Resident);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let whole = std::iter::once(0..1_024).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(0, 256, &[], &[]).is_err());
        assert!(VmmSegmentManifest::build(1_024, 96, &[], &whole).is_err());
        let empty = std::iter::once(0..0).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(1_024, 256, &empty, &whole).is_err());
        let outside = std::iter::once(0..1_025).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(1_024, 256, &outside, &[]).is_err());
        let first = std::iter::once(0..256).collect::<Vec<_>>();
        let overlapping = std::iter::once(128..1_024).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(1_024, 256, &first, &overlapping).is_err());
        let gap_after_first = std::iter::once(512..1_024).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(1_024, 256, &first, &gap_after_first).is_err());
        let middle = std::iter::once(100..200).collect::<Vec<_>>();
        let after_middle = std::iter::once(256..1_024).collect::<Vec<_>>();
        assert!(VmmSegmentManifest::build(1_024, 256, &middle, &after_middle).is_err());
    }
}
