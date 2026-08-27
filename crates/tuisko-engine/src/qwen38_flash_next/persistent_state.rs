//! Slot-owned recurrent state for Qwen3.8-Flash-Next layers.
//!
//! Every GDN layer owns BF16 convolution history and FP32 recurrence state. Layer 1 also owns the
//! PLE convolution's nine-column dilated history. The engram's two token ids remain host state.

use crate::common::math::{product, sum};
use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Alignment every Qwen3.8-Flash-Next arena region carries.
pub(crate) const ALIGNMENT: usize = 256;

/// Slot-owned GDN carries: causal convolution history and the FP32 recurrent state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextGdnPersistent {
    /// `[MAX_BATCH, GDN_QKV_ROWS, LINEAR_CONV_KERNEL_DIM - 1]` BF16 convolution history.
    pub(crate) history: ArenaRegion<u16>,
    /// `[MAX_BATCH, GDN_CONTROL_ROWS, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM]` FP32 recurrent state.
    pub(crate) state: ArenaRegion<f32>,
}

impl Qwen38FlashNextGdnPersistent {
    pub(crate) fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        let columns = A::LINEAR_CONV_KERNEL_DIM.checked_sub(1).ok_or_else(|| {
            EngineError::layout("Qwen3.8-Flash-Next GDN convolution width is zero")
        })?;
        let history = product(
            "Qwen3.8-Flash-Next GDN history",
            product(
                "Qwen3.8-Flash-Next GDN history rows",
                MAX_BATCH,
                A::GDN_QKV_ROWS,
            )?,
            columns,
        )?;
        let state = product(
            "Qwen3.8-Flash-Next GDN state",
            product(
                "Qwen3.8-Flash-Next GDN state heads",
                MAX_BATCH,
                A::GDN_CONTROL_ROWS,
            )?,
            product(
                "Qwen3.8-Flash-Next GDN state head matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;

        Ok(Self {
            history: builder.reserve(history, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
        })
    }

    /// Bytes this carry holds across every slot.
    pub(crate) const fn byte_len(self) -> usize {
        self.history.byte_len() + self.state.byte_len()
    }

    /// Per-slot element counts, in `(history, state)` order, for snapshot and restore.
    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn slot_widths(self) -> (usize, usize) {
        (self.history.len() / MAX_BATCH, self.state.len() / MAX_BATCH)
    }
}

/// Slot-owned PLE dilated short-convolution state, reserved only where a PLE module runs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextPlePersistent {
    /// `[MAX_BATCH, HC_WIDTH, PLE_CONV_STATE_LEN]` BF16, nine columns per channel.
    pub(crate) conv_state: ArenaRegion<u16>,
}

impl Qwen38FlashNextPlePersistent {
    pub(crate) fn reserve(builder: &mut ArenaLayout) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        let conv_state = product(
            "Qwen3.8-Flash-Next PLE conv state",
            product("Qwen3.8-Flash-Next PLE conv rows", MAX_BATCH, A::HC_WIDTH)?,
            A::PLE_CONV_STATE_LEN,
        )?;

        Ok(Self {
            conv_state: builder.reserve(conv_state, ALIGNMENT)?,
        })
    }

    pub(crate) const fn byte_len(self) -> usize {
        self.conv_state.byte_len()
    }

    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn slot_width(self) -> usize {
        self.conv_state.len() / MAX_BATCH
    }
}

/// The device carries one Qwen3.8-Flash-Next decoder layer owns, by layer kind.
///
/// `Qsa` holds none because the shared KV pool owns its paged K/V and indexer keys.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Qwen38FlashNextPersistentState {
    /// The 35 GDN layers that carry no PLE module.
    Gdn(Qwen38FlashNextGdnPersistent),
    /// Decoder layer 1: GDN carries plus the PLE dilated convolution state.
    GdnWithPle(Qwen38FlashNextGdnPersistent, Qwen38FlashNextPlePersistent),
    /// The 12 sparse-attention layers.
    Qsa,
}

impl Qwen38FlashNextPersistentState {
    /// Reserves the carries decoder `layer` owns.
    pub(crate) fn reserve(builder: &mut ArenaLayout, layer: usize) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        if layer >= A::LAYERS {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next layer {layer} is outside 0..{}",
                A::LAYERS
            )));
        }
        if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            return Ok(Self::Qsa);
        }

        let gdn = Qwen38FlashNextGdnPersistent::reserve(builder)?;
        if layer == A::PLE_LAYER {
            let ple = Qwen38FlashNextPlePersistent::reserve(builder)?;
            return Ok(Self::GdnWithPle(gdn, ple));
        }

        Ok(Self::Gdn(gdn))
    }

    /// Bytes these carries hold across every slot.
    pub(crate) fn byte_len(self) -> EngineResult<usize> {
        match self {
            Self::Gdn(gdn) => Ok(gdn.byte_len()),
            Self::GdnWithPle(gdn, ple) => sum(
                "Qwen3.8-Flash-Next layer-1 persistent state",
                &[gdn.byte_len(), ple.byte_len()],
            ),
            Self::Qsa => Ok(0),
        }
    }

    /// The GDN carries, when this layer has them.
    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn gdn(self) -> Option<Qwen38FlashNextGdnPersistent> {
        match self {
            Self::Gdn(gdn) | Self::GdnWithPle(gdn, _) => Some(gdn),
            Self::Qsa => None,
        }
    }

    /// The PLE carry, which exactly one layer has.
    pub(crate) const fn ple(self) -> Option<Qwen38FlashNextPlePersistent> {
        match self {
            Self::GdnWithPle(_, ple) => Some(ple),
            Self::Gdn(_) | Self::Qsa => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, Qwen38FlashNextPersistentState};
    use tuisko_gpu::ArenaLayout;
    use tuisko_model::{Arch, Qwen38FlashNext};

    type A = Qwen38FlashNext;

    #[test]
    fn exactly_one_layer_carries_a_ple_conv_state() {
        let mut with_ple = Vec::new();
        for layer in 0..A::LAYERS {
            let mut builder = ArenaLayout::new();
            let state = Qwen38FlashNextPersistentState::reserve(&mut builder, layer).unwrap();
            if state.ple().is_some() {
                with_ple.push(layer);
            }
        }

        assert_eq!(with_ple, vec![A::PLE_LAYER]);
        assert_eq!(with_ple, vec![1]);
    }

    #[test]
    fn the_layer_kinds_partition_the_stack_thirty_six_to_twelve() {
        let mut gdn = 0usize;
        let mut qsa = 0usize;
        for layer in 0..A::LAYERS {
            let mut builder = ArenaLayout::new();
            match Qwen38FlashNextPersistentState::reserve(&mut builder, layer).unwrap() {
                Qwen38FlashNextPersistentState::Gdn(_)
                | Qwen38FlashNextPersistentState::GdnWithPle(..) => {
                    gdn += 1;
                }
                Qwen38FlashNextPersistentState::Qsa => qsa += 1,
            }
        }

        assert_eq!((gdn, qsa), (36, 12));
        assert_eq!(gdn + qsa, A::LAYERS);
    }

    #[test]
    fn the_ple_conv_state_is_nine_dilated_columns_not_three() {
        let mut builder = ArenaLayout::new();
        let state = Qwen38FlashNextPersistentState::reserve(&mut builder, A::PLE_LAYER).unwrap();
        let ple = state.ple().unwrap();

        // (kernel - 1) * dilation = 3 * 3, never the GDN convolution's kernel - 1 = 3.
        assert_eq!(A::PLE_CONV_STATE_LEN, 9);
        assert_ne!(A::PLE_CONV_STATE_LEN, A::LINEAR_CONV_KERNEL_DIM - 1);
        assert_eq!(ple.slot_width(), A::HC_WIDTH * 9);
        assert_eq!(ple.slot_width() * 2, 184_320);
        assert_eq!(ple.byte_len(), 184_320 * MAX_BATCH);
        assert_eq!(ple.byte_len(), 1_474_560);
    }

    #[test]
    fn gdn_carries_reproduce_the_pinned_per_slot_geometry() {
        let mut builder = ArenaLayout::new();
        let state = Qwen38FlashNextPersistentState::reserve(&mut builder, 0).unwrap();
        let gdn = state.gdn().unwrap();
        let (history, recurrent) = gdn.slot_widths();

        // 10,240 x 3 BF16 history, 48 x 128 x 128 FP32 state, 3 MiB per slot.
        assert_eq!(history * 2, 61_440);
        assert_eq!(recurrent * 4, 3_145_728);
        assert_eq!(gdn.byte_len(), (61_440 + 3_145_728) * MAX_BATCH);
    }

    #[test]
    fn a_sparse_attention_layer_reserves_no_recurrent_plane() {
        let mut builder = ArenaLayout::new();
        let state = Qwen38FlashNextPersistentState::reserve(&mut builder, 3).unwrap();

        assert!(matches!(state, Qwen38FlashNextPersistentState::Qsa));
        assert_eq!(state.byte_len().unwrap(), 0);
        assert_eq!(builder.byte_len(), 0);
        assert!(state.gdn().is_none());
        assert!(state.ple().is_none());
    }

    #[test]
    fn the_whole_stack_costs_the_resident_layout_plan_s_prediction() {
        let mut total = 0usize;
        for layer in 0..A::LAYERS {
            let mut builder = ArenaLayout::new();
            total += Qwen38FlashNextPersistentState::reserve(&mut builder, layer)
                .unwrap()
                .byte_len()
                .unwrap();
        }

        // 17,694,720 history + 905,969,664 state + 1,474,560 PLE convolution.
        assert_eq!(total, 925_138_944);
    }

    #[test]
    fn an_out_of_range_layer_is_refused() {
        let mut builder = ArenaLayout::new();
        assert!(Qwen38FlashNextPersistentState::reserve(&mut builder, A::LAYERS).is_err());
    }
}
