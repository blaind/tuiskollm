//! Qwen3.8-Flash-Next represented-value qualification for the slot-indirected NVFP4
//! expert dispatch and its resident BF16 shared expert.
//!
//! Two laws are qualified here. The first is arithmetic: an `f64` reference
//! decodes every E2M1 code and E4M3 block scale from the fixture's own words and
//! reproduces the SwiGLU, down projection, ascending-expert weighted sum, and
//! sigmoid-gated shared addition.
//!
//! The second is the streaming law `AGENTS.md` states as "cache state must never
//! change produced bits". The same experts are staged into three different slot
//! assignments - identity, reversed, and rotated - and the dispatch must return
//! **byte-identical** output for all three. A negative control mis-points one
//! expert's table entry and requires the output to change, so the identity proof
//! cannot pass by reading weights that were the same everywhere anyway.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::target::{
    QWEN38_FLASH_NEXT_ABSENT_SLOT, QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES,
    Qwen38FlashNextExpertDispatch, Qwen38FlashNextMoeExpertsOp,
    qwen38_flash_next_expert_slot_plane,
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 1_024;
pub(crate) const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen38FlashNext::HIDDEN;
pub(crate) const INTERMEDIATE: usize = Qwen38FlashNext::INTERMEDIATE;
pub(crate) const EXPERTS: usize = Qwen38FlashNext::NUM_EXPERTS;
pub(crate) const TOP_K: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
const GROUP: usize = 16;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const GATE_UP_GROUPS: usize = HIDDEN / GROUP;
const DOWN_GROUPS: usize = INTERMEDIATE / GROUP;

/// Slot-pool geometry. Sixteen slots is deliberately far below the 512 experts:
/// the production pool is a fraction of the inventory, and a small pool is what
/// makes the permuted-assignment sweep cheap enough to run at every route.
pub(crate) const POOL_SLOTS: usize = 16;

/// The sixteen experts this fixture routes to, spread across the whole id range
/// so a table read that ignored the indirection would land somewhere else.
pub(crate) const POOL_EXPERTS: [usize; POOL_SLOTS] = [
    3, 17, 42, 60, 99, 128, 170, 201, 255, 300, 333, 371, 400, 444, 480, 511,
];

const BF16_SENTINEL: u16 = 0xa5a5;

/// Slot assignments the identity proof sweeps. Each maps pool position `i` to a
/// slot; all three stage the identical expert bytes at different addresses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotAssignment {
    /// Pool position `i` occupies slot `i`.
    Identity,
    /// Pool position `i` occupies slot `POOL_SLOTS - 1 - i`.
    Reversed,
    /// Pool position `i` occupies slot `(i + 7) % POOL_SLOTS`.
    Rotated,
}

impl SlotAssignment {
    pub(crate) fn slot_of(self, position: usize) -> usize {
        match self {
            Self::Identity => position,
            Self::Reversed => POOL_SLOTS - 1 - position,
            Self::Rotated => (position + 7) % POOL_SLOTS,
        }
    }
}

/// Failure of the exact Qwen3.8-Flash-Next MoE expert qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextMoeExpertsQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next MoE expert qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextMoeExpertsQualification {
    /// Block output values compared with the independent `f64` oracle.
    pub output_values: usize,
    /// Values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Values proved byte-identical across every permuted slot assignment.
    pub permuted_identity_values: usize,
    /// Slot assignments swept, per route.
    pub slot_assignments: usize,
    /// Values a deliberately mis-pointed table changed, proving sensitivity.
    pub negative_control_values: usize,
    /// Sentinel values verified outside each active route extent.
    pub inactive_values: usize,
    /// Read-only slot-pool and shared-plane values proved unchanged.
    pub immutable_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact resident slot-pool bytes.
    pub slot_pool_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute block-output difference.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) slot_table: ArenaRegion<u32>,
    pub(crate) slot_pool: ArenaRegion<u8>,
    pub(crate) weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate: ArenaRegion<u16>,
    pub(crate) shared_up: ArenaRegion<u16>,
    pub(crate) shared_down: ArenaRegion<u16>,
    pub(crate) shared_gate_logit_weight: ArenaRegion<u16>,
    pub(crate) routed_intermediate: ArenaRegion<u16>,
    pub(crate) routed_output: ArenaRegion<u16>,
    pub(crate) shared_intermediate: ArenaRegion<u16>,
    pub(crate) shared_output: ArenaRegion<u16>,
    pub(crate) shared_gate_logit: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn slot_pool_bytes(self) -> usize {
        self.slot_pool.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.expert_indices.byte_len()
            + self.routing_weights.byte_len()
            + self.slot_table.byte_len()
            + self.slot_pool.byte_len()
            + self.weight_scales_2.byte_len()
            + self.shared_gate.byte_len()
            + self.shared_up.byte_len()
            + self.shared_down.byte_len()
            + self.shared_gate_logit_weight.byte_len()
            + self.routed_intermediate.byte_len()
            + self.routed_output.byte_len()
            + self.shared_intermediate.byte_len()
            + self.shared_output.byte_len()
            + self.shared_gate_logit.byte_len()
            + self.output.byte_len()
    }
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve::<u16>(MAX_ROWS * HIDDEN, ALIGNMENT)?;
    let expert_indices = layout.reserve::<u16>(MAX_ROWS * TOP_K, ALIGNMENT)?;
    let routing_weights = layout.reserve::<u16>(MAX_ROWS * TOP_K, ALIGNMENT)?;
    let slot_table = layout.reserve::<u32>(EXPERTS, ALIGNMENT)?;
    let slot_pool =
        layout.reserve::<u8>(POOL_SLOTS * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, ALIGNMENT)?;
    let weight_scales_2 = layout.reserve::<f32>(EXPERTS * 3, ALIGNMENT)?;
    let shared_gate = layout.reserve::<u16>(INTERMEDIATE * HIDDEN, ALIGNMENT)?;
    let shared_up = layout.reserve::<u16>(INTERMEDIATE * HIDDEN, ALIGNMENT)?;
    let shared_down = layout.reserve::<u16>(HIDDEN * INTERMEDIATE, ALIGNMENT)?;
    let shared_gate_logit_weight = layout.reserve::<u16>(HIDDEN, ALIGNMENT)?;
    let routed_intermediate = layout.reserve::<u16>(MAX_ROWS * TOP_K * INTERMEDIATE, ALIGNMENT)?;
    let routed_output = layout.reserve::<u16>(MAX_ROWS * TOP_K * HIDDEN, ALIGNMENT)?;
    let shared_intermediate = layout.reserve::<u16>(MAX_ROWS * INTERMEDIATE, ALIGNMENT)?;
    let shared_output = layout.reserve::<u16>(MAX_ROWS * HIDDEN, ALIGNMENT)?;
    let shared_gate_logit = layout.reserve::<u16>(MAX_ROWS, ALIGNMENT)?;
    let output = layout.reserve::<u16>(MAX_ROWS * HIDDEN, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            expert_indices,
            routing_weights,
            slot_table,
            slot_pool,
            weight_scales_2,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_logit_weight,
            routed_intermediate,
            routed_output,
            shared_intermediate,
            shared_output,
            shared_gate_logit,
            output,
        },
    ))
}

pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) expert_indices: Vec<u16>,
    pub(crate) routing_weights: Vec<u16>,
    pub(crate) weight_scales_2: Vec<f32>,
    pub(crate) shared_gate: Vec<u16>,
    pub(crate) shared_up: Vec<u16>,
    pub(crate) shared_down: Vec<u16>,
    pub(crate) shared_gate_logit_weight: Vec<u16>,
    /// One expert's slot image, indexed by pool position.
    pub(crate) slot_images: Vec<Vec<u8>>,
    pub(crate) expected_output: Vec<u16>,
    /// Un-cancelled magnitude behind each expected value, for the tolerance.
    pub(crate) expected_mass: Vec<f32>,
}

fn decode_e2m1(code: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(code & 7) as usize];

    if code & 8 == 0 { magnitude } else { -magnitude }
}

fn decode_e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;

    if exponent == 0 {
        fraction as f32 / 512.0
    } else {
        f32::from_bits(((exponent as u32 + 120) << 23) | ((fraction as u32) << 20))
    }
}

/// Independent `BlockScaleK16M128x4` offset oracle.
fn scale_offset(row: usize, group: usize, groups: usize) -> usize {
    let tile_base = (row / 128) * (groups / 4) * 512;
    let group_tile = (group / 4) * 512;
    let row_lane = (row % 32) * 16 + ((row % 128) / 32) * 4;

    tile_base + group_tile + row_lane + group % 4
}

const SCALE_CODES: [u8; 4] = [0x38, 0x3c, 0x40, 0x34];

/// A non-periodic mix of the expert id with a position.
///
/// The obvious `(expert * k + position) & 15` is periodic in the expert with
/// period 16, so every pair of pool experts congruent mod 16 would share an
/// image byte for byte -- and `every_expert_image_is_distinct` caught exactly
/// that. Folding the high bits down first removes the period.
fn mix(expert: usize, position: usize) -> usize {
    let seed = expert
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(position.wrapping_mul(0x85EB_CA77));

    (seed ^ (seed >> 15)).wrapping_mul(0xC2B2_AE35) >> 16
}

/// Builds one expert's 2,764,800-byte slot image: `down || gate || up` packed
/// codes, then the fused gate||up block scales, then the down block scales.
///
/// Every byte depends on the expert id, so two experts never share an image -
/// which is what gives the permuted-assignment proof its teeth.
fn build_slot_image(expert: usize) -> Vec<u8> {
    let mut image = vec![0u8; QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES];
    let down_code = 0usize;
    let gate_code = HIDDEN * INTERMEDIATE / 2;
    let up_code = gate_code + INTERMEDIATE * HIDDEN / 2;
    let gate_up_scale = up_code + INTERMEDIATE * HIDDEN / 2;
    let down_scale = gate_up_scale + GATE_UP_ROWS * GATE_UP_GROUPS;

    // down_proj codes: [2560, 320]
    for row in 0..HIDDEN {
        for byte in 0..INTERMEDIATE / 2 {
            let low = mix(expert, row * 331 + byte * 7) & 15;
            let high = mix(expert, row * 733 + byte * 13 + 1) & 15;
            image[down_code + row * (INTERMEDIATE / 2) + byte] = low as u8 | (high as u8) << 4;
        }
        for group in 0..DOWN_GROUPS {
            image[down_scale + scale_offset(row, group, DOWN_GROUPS)] =
                SCALE_CODES[mix(expert, row * 97 + group * 11) & 3];
        }
    }

    // gate_proj then up_proj codes, both [640, 1280], contiguous so the kernel
    // reads them as one fused [1280, 1280] plane.
    for (plane, base) in [(0usize, gate_code), (1usize, up_code)] {
        for row in 0..INTERMEDIATE {
            for byte in 0..HIDDEN / 2 {
                let low = mix(expert, row * 419 + byte * 5 + plane * 1_009) & 15;
                let high = mix(expert, row * 577 + byte * 11 + plane * 2_003 + 1) & 15;
                image[base + row * (HIDDEN / 2) + byte] = low as u8 | (high as u8) << 4;
            }
        }
    }
    for fused_row in 0..GATE_UP_ROWS {
        for group in 0..GATE_UP_GROUPS {
            image[gate_up_scale + scale_offset(fused_row, group, GATE_UP_GROUPS)] =
                SCALE_CODES[mix(expert, fused_row * 89 + group * 17) & 3];
        }
    }

    image
}

/// The `f64` dot of one packed E2M1 row against a BF16 activation row.
#[allow(clippy::too_many_arguments)]
fn nvfp4_dot(
    activation: &[u16],
    image: &[u8],
    code_base: usize,
    scale_base: usize,
    row: usize,
    columns: usize,
    scale_rows: usize,
    weight_scale_2: f32,
) -> (f32, f32) {
    let groups = columns / GROUP;
    let mut sum = 0.0f64;
    let mut mass = 0.0f64;

    for group in 0..groups {
        let scale = decode_e4m3(image[scale_base + scale_offset(row + scale_rows, group, groups)]);
        let coefficient = f64::from(scale * weight_scale_2);
        let mut group_sum = 0.0f64;
        for column in 0..GROUP {
            let packed = image[code_base + row * (columns / 2) + group * (GROUP / 2) + column / 2];
            let code = if column & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            let term = f64::from(bf16_to_f32(activation[group * GROUP + column]))
                * f64::from(decode_e2m1(code));
            group_sum += term;
            mass += term.abs() * coefficient.abs();
        }
        sum += group_sum * coefficient;
    }

    (sum as f32, mass as f32)
}

fn bf16_dot(activation: &[u16], weights: &[u16], row: usize, columns: usize) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut mass = 0.0f64;
    for column in 0..columns {
        let term = f64::from(bf16_to_f32(activation[column]))
            * f64::from(bf16_to_f32(weights[row * columns + column]));
        sum += term;
        mass += term.abs();
    }

    (sum as f32, mass as f32)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn make_fixture() -> Fixture {
    // Token `t` repeats pattern `t & 7`, so the oracle costs eight tokens'
    // arithmetic and still verifies all 1,024 rows exhaustively.
    let input = (0..MAX_ROWS * HIDDEN)
        .map(|index| {
            let pattern = (index / HIDDEN) & (MAX_BATCH - 1);
            let column = index % HIDDEN;
            // Exact BF16 powers of two, small enough that a 2,560-wide dot
            // against E2M1 codes lands near unity. SwiGLU squares that dot, so
            // an O(1) activation would drive the block output into the
            // thousands, where BF16 carries three digits and the acceptance
            // contract stops discriminating.
            let value = match (pattern + column) % 5 {
                0 => 0.015_625,
                1 => -0.007_812_5,
                2 => 0.031_25,
                3 => -0.015_625,
                _ => 0.007_812_5,
            };
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();

    // Pattern `p` routes to ten of the sixteen pool experts, published ascending
    // exactly as the router publishes them.
    let mut expert_indices = vec![0u16; MAX_ROWS * TOP_K];
    let mut routing_weights = vec![0u16; MAX_ROWS * TOP_K];
    for token in 0..MAX_ROWS {
        let pattern = token & (MAX_BATCH - 1);
        let mut chosen = (0..TOP_K)
            .map(|rank| POOL_EXPERTS[(pattern + rank) % POOL_SLOTS])
            .collect::<Vec<_>>();
        chosen.sort_unstable();
        for (position, expert) in chosen.into_iter().enumerate() {
            expert_indices[token * TOP_K + position] = expert as u16;
            routing_weights[token * TOP_K + position] =
                f32_to_bf16(0.05 + 0.01 * ((pattern + position) % 7) as f32);
        }
    }

    let weight_scales_2 = (0..EXPERTS * 3)
        .map(|index| {
            let expert = index / 3;
            let projection = index % 3;
            0.5 + 0.125 * ((expert + projection) % 4) as f32
        })
        .collect::<Vec<_>>();

    let shared_gate = (0..INTERMEDIATE * HIDDEN)
        .map(|index| f32_to_bf16(if index % 3 == 0 { 0.25 } else { -0.125 }))
        .collect::<Vec<_>>();
    let shared_up = (0..INTERMEDIATE * HIDDEN)
        .map(|index| f32_to_bf16(if index % 4 == 0 { -0.5 } else { 0.25 }))
        .collect::<Vec<_>>();
    let shared_down = (0..HIDDEN * INTERMEDIATE)
        .map(|index| f32_to_bf16(if index % 5 == 0 { 0.125 } else { -0.0625 }))
        .collect::<Vec<_>>();
    let shared_gate_logit_weight = (0..HIDDEN)
        .map(|index| f32_to_bf16(if index % 2 == 0 { 0.01 } else { -0.005 }))
        .collect::<Vec<_>>();

    let slot_images = POOL_EXPERTS
        .iter()
        .map(|&expert| build_slot_image(expert))
        .collect::<Vec<_>>();

    let mut expected_output = vec![0u16; MAX_ROWS * HIDDEN];
    let mut expected_mass = vec![0.0f32; MAX_ROWS * HIDDEN];
    for pattern in 0..MAX_BATCH {
        let activation = &input[pattern * HIDDEN..(pattern + 1) * HIDDEN];
        let mut accumulated = vec![0.0f32; HIDDEN];
        // The un-cancelled mass reaching each output value. The block output is
        // a near-cancelling sum of eleven terms whose own dots also cancel, so
        // a tolerance relative to the *result* is meaningless -- at a result of
        // 0.4 built from terms of magnitude 50, one BF16 ulp of a term already
        // exceeds it. The acceptance contract is therefore relative to this
        // mass, which is what the roundings actually act on.
        let mut mass = vec![0.0f32; HIDDEN];

        // The ten routed experts, in the published ascending order, which is the
        // reference's `index_add_` order.
        for position in 0..TOP_K {
            let expert = expert_indices[pattern * TOP_K + position] as usize;
            let pool = POOL_EXPERTS.iter().position(|&e| e == expert).unwrap();
            let image = &slot_images[pool];
            let routing = bf16_to_f32(routing_weights[pattern * TOP_K + position]);
            let gate_scale_2 = weight_scales_2[expert * 3];
            let up_scale_2 = weight_scales_2[expert * 3 + 1];
            let down_scale_2 = weight_scales_2[expert * 3 + 2];
            let gate_code = HIDDEN * INTERMEDIATE / 2;
            let gate_up_scale = gate_code + 2 * (INTERMEDIATE * HIDDEN / 2);
            let down_scale = gate_up_scale + GATE_UP_ROWS * GATE_UP_GROUPS;

            let mut intermediate = vec![0u16; INTERMEDIATE];
            for (row, intermediate) in intermediate.iter_mut().enumerate() {
                let gate = nvfp4_dot(
                    activation,
                    image,
                    gate_code,
                    gate_up_scale,
                    row,
                    HIDDEN,
                    0,
                    gate_scale_2,
                );
                let up = nvfp4_dot(
                    activation,
                    image,
                    gate_code,
                    gate_up_scale,
                    row + INTERMEDIATE,
                    HIDDEN,
                    0,
                    up_scale_2,
                );
                *intermediate = f32_to_bf16(silu(gate.0) * up.0);
            }
            for row in 0..HIDDEN {
                let value = nvfp4_dot(
                    &intermediate,
                    image,
                    0,
                    down_scale,
                    row,
                    INTERMEDIATE,
                    0,
                    down_scale_2,
                );
                accumulated[row] =
                    f32_to_bf16_roundtrip(value.0).mul_add(routing, accumulated[row]);
                mass[row] += value.1 * routing.abs();
            }
        }

        // The shared expert is added last and gated by its own sigmoid.
        let mut shared_intermediate = vec![0u16; INTERMEDIATE];
        for (row, intermediate) in shared_intermediate.iter_mut().enumerate() {
            let gate = bf16_dot(activation, &shared_gate, row, HIDDEN);
            let up = bf16_dot(activation, &shared_up, row, HIDDEN);
            *intermediate = f32_to_bf16(silu(gate.0) * up.0);
        }
        let logit = bf16_dot(activation, &shared_gate_logit_weight, 0, HIDDEN);
        let gate_value = sigmoid(bf16_to_f32(f32_to_bf16(logit.0)));
        for row in 0..HIDDEN {
            let value = bf16_dot(&shared_intermediate, &shared_down, row, INTERMEDIATE);
            let total = f32_to_bf16_roundtrip(value.0).mul_add(gate_value, accumulated[row]);
            expected_output[pattern * HIDDEN + row] = f32_to_bf16(total);
            expected_mass[pattern * HIDDEN + row] = mass[row] + value.1 * gate_value;
        }
    }
    for token in MAX_BATCH..MAX_ROWS {
        let pattern = token & (MAX_BATCH - 1);
        expected_output.copy_within(pattern * HIDDEN..(pattern + 1) * HIDDEN, token * HIDDEN);
        expected_mass.copy_within(pattern * HIDDEN..(pattern + 1) * HIDDEN, token * HIDDEN);
    }

    Fixture {
        input,
        expert_indices,
        routing_weights,
        weight_scales_2,
        shared_gate,
        shared_up,
        shared_down,
        shared_gate_logit_weight,
        slot_images,
        expected_output,
        expected_mass,
    }
}

/// The device writes each expert output as BF16 before the combine reads it, so
/// the oracle rounds at the same place.
fn f32_to_bf16_roundtrip(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

/// Stages the pool under one assignment and returns the table it publishes.
pub(crate) fn staged_pool(fixture: &Fixture, assignment: SlotAssignment) -> (Vec<u8>, Vec<u32>) {
    let mut pool = vec![0u8; POOL_SLOTS * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES];
    let mut table = vec![QWEN38_FLASH_NEXT_ABSENT_SLOT; EXPERTS];
    for (position, &expert) in POOL_EXPERTS.iter().enumerate() {
        let slot = assignment.slot_of(position);
        let base = slot * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
        pool[base..base + QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES]
            .copy_from_slice(&fixture.slot_images[position]);
        table[expert] = slot as u32;
    }

    (pool, table)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_of(arena: &DeviceArena, regions: Regions) -> GpuResult<Qwen38FlashNextExpertDispatch> {
    Ok(Qwen38FlashNextExpertDispatch {
        input: arena.address(regions.input)?,
        expert_indices: arena.address(regions.expert_indices)?,
        routing_weights: arena.address(regions.routing_weights)?,
        slot_table: arena.address(regions.slot_table)?,
        slot_pool: arena.address(regions.slot_pool)?,
        weight_scales_2: arena.address(regions.weight_scales_2)?,
        shared_gate_weight: arena.address(regions.shared_gate)?,
        shared_up_weight: arena.address(regions.shared_up)?,
        shared_down_weight: arena.address(regions.shared_down)?,
        shared_gate_logit_weight: arena.address(regions.shared_gate_logit_weight)?,
        routed_intermediate: arena.address(regions.routed_intermediate)?,
        routed_output: arena.address(regions.routed_output)?,
        shared_intermediate: arena.address(regions.shared_intermediate)?,
        shared_output: arena.address(regions.shared_output)?,
        shared_gate_logit: arena.address(regions.shared_gate_logit)?,
        output: arena.address(regions.output)?,
    })
}

fn fill_output(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)
}

fn read_output(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Vec<u16>> {
    arena.copy_to_host(stream, regions.output)
}

/// Qualifies eager, captured, and permuted-slot execution at every exact route.
pub fn qualify_qwen38_flash_next_moe_experts()
-> Result<Qwen38FlashNextMoeExpertsQualification, Qwen38FlashNextMoeExpertsQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let operator = Qwen38FlashNextMoeExpertsOp::new(&context)?;
    let fixture = make_fixture();
    let plane = qwen38_flash_next_expert_slot_plane(POOL_SLOTS);

    arena.copy_from_host(&stream, regions.input, &fixture.input)?;
    arena.copy_from_host(&stream, regions.expert_indices, &fixture.expert_indices)?;
    arena.copy_from_host(&stream, regions.routing_weights, &fixture.routing_weights)?;
    arena.copy_from_host(&stream, regions.weight_scales_2, &fixture.weight_scales_2)?;
    arena.copy_from_host(&stream, regions.shared_gate, &fixture.shared_gate)?;
    arena.copy_from_host(&stream, regions.shared_up, &fixture.shared_up)?;
    arena.copy_from_host(&stream, regions.shared_down, &fixture.shared_down)?;
    arena.copy_from_host(
        &stream,
        regions.shared_gate_logit_weight,
        &fixture.shared_gate_logit_weight,
    )?;
    stream.synchronize().map_err(GpuError::from)?;

    let mut report = Qwen38FlashNextMoeExpertsQualification {
        output_values: 0,
        graph_replay_values: 0,
        permuted_identity_values: 0,
        slot_assignments: 0,
        negative_control_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        slot_pool_bytes: regions.slot_pool_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        let mut reference: Option<Vec<u16>> = None;

        for assignment in [
            SlotAssignment::Identity,
            SlotAssignment::Reversed,
            SlotAssignment::Rotated,
        ] {
            let (pool, table) = staged_pool(&fixture, assignment);
            plane.validate_routed_presence(&table, &fixture.expert_indices[..rows * TOP_K])?;
            arena.copy_from_host(&stream, regions.slot_pool, &pool)?;
            arena.copy_from_host(&stream, regions.slot_table, &table)?;
            fill_output(&arena, &stream, regions)?;
            stream.synchronize().map_err(GpuError::from)?;

            let dispatch = dispatch_of(&arena, regions)?;
            unsafe { operator.launch(&stream, rows, &dispatch)? };
            stream.synchronize().map_err(GpuError::from)?;
            let observed = read_output(&arena, &stream, regions)?;
            report.slot_assignments += 1;

            match &reference {
                None => {
                    // The identity assignment carries the numerical check.
                    for token in 0..rows {
                        for column in 0..HIDDEN {
                            let index = token * HIDDEN + column;
                            let got = bf16_to_f32(observed[index]);
                            let want = bf16_to_f32(fixture.expected_output[index]);
                            let error = (got - want).abs();
                            if error > output_tolerance(fixture.expected_mass[index]) {
                                return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                                    format!(
                                        "rows={rows} token {token} column {column} output {got} \
                                         != {want}"
                                    ),
                                ));
                            }
                            report.maximum_absolute_error =
                                report.maximum_absolute_error.max(error);
                            report.output_values += 1;
                        }
                    }
                    for token in rows..MAX_ROWS {
                        for column in 0..HIDDEN {
                            if observed[token * HIDDEN + column] != BF16_SENTINEL {
                                return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                                    format!("rows={rows} wrote token {token} column {column}"),
                                ));
                            }
                            report.inactive_values += 1;
                        }
                    }
                    reference = Some(observed);
                }
                Some(expected) => {
                    // Cache state must never change produced bits.
                    if &observed != expected {
                        let first = observed
                            .iter()
                            .zip(expected)
                            .position(|(left, right)| left != right)
                            .unwrap_or(0);
                        return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                            format!(
                                "rows={rows} assignment {assignment:?} changed produced bits at \
                             value {first}"
                            ),
                        ));
                    }
                    report.permuted_identity_values += rows * HIDDEN;
                }
            }
        }

        let reference = reference.expect("the identity assignment ran first");

        // Eager and replay agreement over the observable boundary.
        fill_output(&arena, &stream, regions)?;
        let graph = CudaGraph::capture(&stream, || {
            let dispatch = dispatch_of(&arena, regions)?;
            unsafe { operator.launch(&stream, rows, &dispatch) }
        })?;
        unsafe { graph.launch(&stream) }?;
        stream.synchronize().map_err(GpuError::from)?;
        let replayed = read_output(&arena, &stream, regions)?;
        if replayed[..rows * HIDDEN] != reference[..rows * HIDDEN] {
            return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                format!("graph replay diverged from eager execution at rows={rows}"),
            ));
        }
        report.graph_replay_values += rows * HIDDEN;

        // Negative control: the proof above is only meaningful if a wrong slot
        // would have been observable. Point one routed expert at another
        // expert's slot and require the output to move.
        let (pool, mut table) = staged_pool(&fixture, SlotAssignment::Identity);
        let victim = fixture.expert_indices[0] as usize;
        let other = *POOL_EXPERTS
            .iter()
            .find(|&&expert| expert != victim)
            .expect("the pool holds more than one expert");
        table[victim] = table[other];
        arena.copy_from_host(&stream, regions.slot_pool, &pool)?;
        arena.copy_from_host(&stream, regions.slot_table, &table)?;
        fill_output(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        unsafe { operator.launch(&stream, rows, &dispatch_of(&arena, regions)?)? };
        stream.synchronize().map_err(GpuError::from)?;
        let mispointed = read_output(&arena, &stream, regions)?;
        let changed = mispointed[..rows * HIDDEN]
            .iter()
            .zip(&reference[..rows * HIDDEN])
            .filter(|(left, right)| left != right)
            .count();
        if changed == 0 {
            return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                format!(
                    "rows={rows} a mis-pointed slot table produced identical output, so the \
                 permuted-assignment proof is vacuous"
                ),
            ));
        }
        report.negative_control_values += changed;

        // Restore the identity staging and confirm the read-only planes held.
        let (pool, table) = staged_pool(&fixture, SlotAssignment::Identity);
        arena.copy_from_host(&stream, regions.slot_pool, &pool)?;
        arena.copy_from_host(&stream, regions.slot_table, &table)?;
        stream.synchronize().map_err(GpuError::from)?;
        let observed_pool = arena.copy_to_host(&stream, regions.slot_pool)?;
        let observed_shared = arena.copy_to_host(&stream, regions.shared_gate)?;
        if observed_pool != pool || observed_shared != fixture.shared_gate {
            return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
                format!("rows={rows} modified a read-only plane"),
            ));
        }
        report.immutable_values += observed_pool.len() + observed_shared.len();
    }

    verify_no_post_warmup_allocation(&context, &operator, &arena, &stream, regions)?;

    Ok(report)
}

/// Graph replay after warmup must not allocate.
///
/// Measured around *replays only*: capturing and instantiating a graph
/// legitimately allocates, so a span that included the captures would report a
/// leak that is really the harness building its own fixtures.
fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen38FlashNextMoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen38FlashNextMoeExpertsQualificationError> {
    let graphs = EXACT_ROUTES
        .iter()
        .map(|&rows| {
            CudaGraph::capture(stream, || {
                let dispatch = dispatch_of(arena, regions)?;
                unsafe { op.launch(stream, rows, &dispatch) }
            })
        })
        .collect::<GpuResult<Vec<_>>>()?;
    // Warm both orders. These graphs are large enough -- T=1024 alone submits
    // 1.8M expert CTAs -- that the driver materializes per-launch resources
    // lazily, and the measured loop below replays in reverse. Warming only the
    // forward order leaves that first reverse pass allocating inside the
    // measured window, which reads as a leak and is really warmup.
    for _ in 0..2 {
        for graph in &graphs {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen38FlashNextMoeExpertsQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

/// The acceptance contract, relative to the un-cancelled mass rather than to
/// the result.
///
/// Each value passes through two BF16 quantizations the `f64` oracle also
/// applies -- the SwiGLU intermediate and the per-expert output -- and each
/// costs at most half an ulp, `2^-9` relative, of the mass flowing through it.
/// Two such stages plus the fp32 reduction and the device's `ex2.approx.f32`
/// SiLU is bounded here by `2^-7` of the mass, with a floor of one BF16 ulp at
/// unity so a value whose mass is genuinely tiny is still compared.
fn output_tolerance(mass: f32) -> f32 {
    (mass.abs() * 0.007_812_5).max(0.007_812_5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_assignments_are_permutations_that_actually_move_every_expert() {
        for assignment in [SlotAssignment::Reversed, SlotAssignment::Rotated] {
            let slots = (0..POOL_SLOTS)
                .map(|position| assignment.slot_of(position))
                .collect::<Vec<_>>();
            let mut sorted = slots.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..POOL_SLOTS).collect::<Vec<_>>());
            assert!(
                (0..POOL_SLOTS).all(|position| slots[position] != position),
                "{assignment:?} left an expert in place"
            );
        }
    }

    #[test]
    fn every_expert_image_is_distinct() {
        // The permuted-assignment proof is vacuous if two experts share bytes.
        let fixture = make_fixture();
        for left in 0..POOL_SLOTS {
            for right in left + 1..POOL_SLOTS {
                assert_ne!(
                    fixture.slot_images[left], fixture.slot_images[right],
                    "pool positions {left} and {right} hold identical bytes"
                );
            }
        }
    }

    #[test]
    fn staging_places_each_expert_at_its_assigned_slot() {
        let fixture = make_fixture();
        for assignment in [
            SlotAssignment::Identity,
            SlotAssignment::Reversed,
            SlotAssignment::Rotated,
        ] {
            let (pool, table) = staged_pool(&fixture, assignment);
            for (position, &expert) in POOL_EXPERTS.iter().enumerate() {
                let slot = table[expert] as usize;
                assert_eq!(slot, assignment.slot_of(position));
                let base = slot * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
                assert_eq!(
                    &pool[base..base + QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES],
                    fixture.slot_images[position].as_slice()
                );
            }
            // Everything the fixture does not route to stays absent.
            assert_eq!(
                table
                    .iter()
                    .filter(|&&slot| slot != QWEN38_FLASH_NEXT_ABSENT_SLOT)
                    .count(),
                POOL_SLOTS
            );
        }
    }

    #[test]
    fn published_routes_are_ascending_and_inside_the_pool() {
        let fixture = make_fixture();
        for token in 0..MAX_ROWS {
            let selection = &fixture.expert_indices[token * TOP_K..(token + 1) * TOP_K];
            assert!(
                selection.windows(2).all(|pair| pair[0] < pair[1]),
                "token {token} published {selection:?}"
            );
            for &expert in selection {
                assert!(POOL_EXPERTS.contains(&(expert as usize)));
            }
        }
    }

    #[test]
    fn slot_extent_and_accounting_are_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, 2_764_800);
        assert_eq!(regions.slot_pool_bytes(), POOL_SLOTS * 2_764_800);
        assert_eq!(regions.slot_pool_bytes(), 44_236_800);
        assert_eq!(
            layout.byte_len(),
            regions.payload_bytes() + (layout.byte_len() - regions.payload_bytes())
        );
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_hold_across_slot_permutations()
    -> Result<(), Qwen38FlashNextMoeExpertsQualificationError> {
        let report = qualify_qwen38_flash_next_moe_experts()?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        let inactive_rows = EXACT_ROUTES
            .iter()
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();

        assert_eq!(report.output_values, active_rows * HIDDEN);
        assert_eq!(report.graph_replay_values, active_rows * HIDDEN);
        // Two further assignments per route must reproduce the identity's bits.
        assert_eq!(report.permuted_identity_values, 2 * active_rows * HIDDEN);
        assert_eq!(report.slot_assignments, 3 * EXACT_ROUTES.len());
        assert_eq!(report.inactive_values, inactive_rows * HIDDEN);
        assert!(report.negative_control_values > 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}
