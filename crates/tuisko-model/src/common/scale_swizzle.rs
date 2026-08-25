//! Source plane gathering and `BlockScaleK16M128x4` swizzling shared by every admitted target.

use crate::{CheckpointError, CheckpointResult};
use rayon::prelude::*;
use std::sync::OnceLock;

pub(crate) const SCALE_TILE_ROWS: usize = 128;
pub(crate) const SCALE_TILE_GROUPS: usize = 4;
const SCALE_TILE_BYTES: usize = SCALE_TILE_ROWS * SCALE_TILE_GROUPS;

const PARALLEL_SWIZZLE_MIN_BYTES: usize = 1 << 20;
const PARALLEL_GATHER_MIN_BYTES: usize = 1 << 20;
const MAX_MATERIALIZATION_WORKERS: usize = 16;

static MATERIALIZATION_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

/// Worker bound used by target-size NVFP4 scale materialization.
pub fn nvfp4_scale_materialization_workers() -> usize {
    materialization_workers()
}

pub(crate) fn materialization_workers() -> usize {
    std::thread::available_parallelism()
        .map(|workers| workers.get())
        .unwrap_or(1)
        .min(MAX_MATERIALIZATION_WORKERS)
}

pub(crate) fn host_shape(shape: &[u64; 2], role: &str) -> CheckpointResult<[usize; 2]> {
    let rows = usize::try_from(shape[0]).map_err(|_| {
        CheckpointError::source_binding(format!("{role} row count exceeds this host"))
    })?;
    let columns = usize::try_from(shape[1]).map_err(|_| {
        CheckpointError::source_binding(format!("{role} column count exceeds this host"))
    })?;

    Ok([rows, columns])
}

pub(crate) fn gather_source_planes<const N: usize>(
    planes: [&[u8]; N],
    role: &str,
) -> CheckpointResult<Vec<u8>> {
    let bytes = planes.iter().try_fold(0usize, |bytes, plane| {
        bytes
            .checked_add(plane.len())
            .ok_or_else(|| CheckpointError::source_binding(format!("{role} length overflows")))
    })?;

    let mut gathered = Vec::new();

    gathered.try_reserve_exact(bytes).map_err(|_| {
        CheckpointError::source_binding(format!("{role} cannot reserve {bytes} host bytes"))
    })?;

    if bytes >= PARALLEL_GATHER_MIN_BYTES && materialization_workers() > 1 {
        match planes.as_slice() {
            [first, second] => {
                materialization_pool(role)?.install(|| {
                    first
                        .par_iter()
                        .copied()
                        .chain(second.par_iter().copied())
                        .collect_into_vec(&mut gathered);
                });
                return Ok(gathered);
            }
            [first, second, third] => {
                materialization_pool(role)?.install(|| {
                    first
                        .par_iter()
                        .copied()
                        .chain(second.par_iter().copied())
                        .chain(third.par_iter().copied())
                        .collect_into_vec(&mut gathered);
                });
                return Ok(gathered);
            }
            _ => {}
        }
    }

    for plane in planes {
        gathered.extend_from_slice(plane);
    }

    Ok(gathered)
}

pub(crate) fn swizzle_scale_planes(
    planes: &[&[u8]],
    rows_per_plane: usize,
    groups_per_row: usize,
    layer: usize,
    role: &str,
) -> CheckpointResult<Vec<u8>> {
    let rows = rows_per_plane.checked_mul(planes.len()).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} fused scale row count overflows"
        ))
    })?;

    if rows == 0 || !rows.is_multiple_of(SCALE_TILE_ROWS) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale rows {rows} are not tiled by {SCALE_TILE_ROWS}"
        )));
    }

    if groups_per_row == 0 || !groups_per_row.is_multiple_of(SCALE_TILE_GROUPS) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale groups {groups_per_row} are not tiled by {SCALE_TILE_GROUPS}"
        )));
    }

    let plane_len = rows_per_plane.checked_mul(groups_per_row).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} source scale length overflows"
        ))
    })?;
    let output_len = rows.checked_mul(groups_per_row).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} materialized scale length overflows"
        ))
    })?;

    if planes.iter().any(|plane| plane.len() != plane_len) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} source scale plane length does not match its shape"
        )));
    }

    let mut swizzled = vec![0; output_len];
    let scale_tiles_per_row = groups_per_row / SCALE_TILE_GROUPS;
    let swizzle_tile = |(tile_index, destination): (usize, &mut [u8])| {
        swizzle_scale_tile(
            destination,
            tile_index,
            scale_tiles_per_row,
            planes,
            rows_per_plane,
            groups_per_row,
        );
    };

    if output_len >= PARALLEL_SWIZZLE_MIN_BYTES {
        materialization_pool(&format!("layer-{layer} NVFP4 {role} scales"))?.install(|| {
            swizzled
                .par_chunks_mut(SCALE_TILE_BYTES)
                .enumerate()
                .for_each(swizzle_tile);
        });
    } else {
        swizzled
            .chunks_mut(SCALE_TILE_BYTES)
            .enumerate()
            .for_each(swizzle_tile);
    }

    Ok(swizzled)
}

fn swizzle_scale_tile(
    destination: &mut [u8],
    tile_index: usize,
    scale_tiles_per_row: usize,
    planes: &[&[u8]],
    rows_per_plane: usize,
    groups_per_row: usize,
) {
    let persistent_tile = tile_index / scale_tiles_per_row;
    let scale_tile = tile_index % scale_tiles_per_row;
    let source_group = scale_tile * SCALE_TILE_GROUPS;

    // Each 512-byte destination tile is independent. Writing by its 32 contiguous
    // 16-byte rows avoids the old per-byte division and scattered store while preserving
    // the exact BlockScaleK16M128x4 address mapping.
    for row_mod32 in 0..32 {
        let destination_row = &mut destination[row_mod32 * 16..(row_mod32 + 1) * 16];
        for row_quartile in 0..4 {
            let row = persistent_tile * SCALE_TILE_ROWS + row_quartile * 32 + row_mod32;
            let source_plane = row / rows_per_plane;
            let source_row = row % rows_per_plane;
            let source = source_row * groups_per_row + source_group;
            destination_row[row_quartile * 4..(row_quartile + 1) * 4]
                .copy_from_slice(&planes[source_plane][source..source + 4]);
        }
    }
}

pub(crate) fn materialization_pool(role: &str) -> CheckpointResult<&'static rayon::ThreadPool> {
    let pool = MATERIALIZATION_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(materialization_workers())
            .thread_name(|index| format!("tuisko-materialize-{index}"))
            .build()
            .map_err(|error| error.to_string())
    });
    pool.as_ref().map_err(|error| {
        CheckpointError::source_binding(format!(
            "{role} cannot start bounded materialization workers: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointErrorCode;
    use crate::common::test_support::sources::{GROUPS, ROWS};

    #[test]
    fn parallel_gather_preserves_three_plane_order_exactly() {
        let first = (0..1 << 20).map(|index| index as u8).collect::<Vec<_>>();
        let second = (0..4_097)
            .map(|index| (index as u8).wrapping_mul(3))
            .collect::<Vec<_>>();
        let third = (0..8_191)
            .map(|index| (index as u8).wrapping_mul(5))
            .collect::<Vec<_>>();

        let gathered = gather_source_planes([&first, &second, &third], "test QKV").unwrap();
        let second_end = first.len() + second.len();

        assert_eq!(&gathered[..first.len()], first.as_slice());
        assert_eq!(&gathered[first.len()..second_end], second.as_slice());
        assert_eq!(&gathered[second_end..], third.as_slice());
    }

    #[test]
    fn parallel_gather_preserves_two_plane_order_exactly() {
        let first = (0..1 << 20).map(|index| index as u8).collect::<Vec<_>>();
        let second = (0..8_191)
            .map(|index| (index as u8).wrapping_mul(5))
            .collect::<Vec<_>>();

        let gathered = gather_source_planes([&first, &second], "test fused planes").unwrap();

        assert_eq!(&gathered[..first.len()], first.as_slice());
        assert_eq!(&gathered[first.len()..], second.as_slice());
    }

    #[test]
    fn scale_layout_rejects_incompatible_geometry() {
        for (rows, groups, message) in [
            (127, 8, "scale rows 127 are not tiled by 128"),
            (128, 6, "scale groups 6 are not tiled by 4"),
        ] {
            let error = swizzle_scale_planes(&[&[]], rows, groups, 55, "test")
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains(message), "{error}");
        }

        let error = swizzle_scale_planes(&[&[]], ROWS, GROUPS, 55, "test")
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("plane length does not match"));
    }
}
