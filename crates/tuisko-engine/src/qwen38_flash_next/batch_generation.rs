//! Compact Qwen3.8 Flash-Next generation over eight physical slots.
//!
//! Requests retain their slots while pending rows pack densely. Prime and decode reuse the
//! qualified model entries. Separate compact and per-slot logit banks prevent row/slot aliasing.

use crate::common::banks::{compact, row};
use crate::common::progress::ResidentLoadProgress;
use crate::common::slots::{device_zero_context, require_generation_capacity};
use crate::common::text_generator::ModelProgram;
use crate::qwen38_flash_next::compact_route::{
    Qwen38FlashNextCompactRound, qwen38_flash_next_admission_slot, qwen38_flash_next_compact_round,
    qwen38_flash_next_compact_survivors,
};
use crate::qwen38_flash_next::resident_model::{
    Qwen38FlashNextResidentLoadStats, Qwen38FlashNextResidentModel, Qwen38FlashNextStepTelemetry,
};
use crate::qwen38_flash_next::text_generation::{Qwen38FlashNextGenerationTelemetry, prime_prompt};
use crate::{
    ChatGenerationRequest, EngineError, EngineResult, GenerationSession, LayerMemoryLayout,
    MAX_BATCH, ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents,
    ResidentCancellation, ResidentRequestId,
};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;
use tuisko_frontend::{GenerationDefaults, TextFrontend};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38FlashNext};

/// Per-slot rows plus one compact download block.
const LOGIT_BANK_ROWS: usize = 2 * MAX_BATCH;

/// Measured decode cost for one exact batch width.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextBatchWidthTelemetry {
    rounds: usize,
    tokens: usize,
    forward: Duration,
    expert_requests: usize,
    expert_hits: usize,
    expert_misses: usize,
    expert_h2d_bytes: usize,
    embedding_h2d_bytes: usize,
    engram_h2d_bytes: usize,
    engram_rows: usize,
    kv_append_bytes: usize,
}

impl Qwen38FlashNextBatchWidthTelemetry {
    /// Decode rounds run at this width.
    pub const fn rounds(self) -> usize {
        self.rounds
    }

    /// Tokens those rounds committed, which is `rounds * width`.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Wall time those rounds took together.
    pub const fn forward(self) -> Duration {
        self.forward
    }

    /// Mean wall time one round at this width took.
    pub fn round_ms(self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }

        self.forward.as_secs_f64() * 1_000.0 / self.rounds as f64
    }

    /// Tokens per second the rounds at this width sustained.
    pub fn tokens_per_second(self) -> f64 {
        if self.forward.is_zero() {
            return 0.0;
        }

        self.tokens as f64 / self.forward.as_secs_f64()
    }

    /// Expert selections the rounds at this width made.
    pub const fn expert_requests(self) -> usize {
        self.expert_requests
    }

    /// Expert hit rate over distinct per-round items at this width.
    pub fn expert_hit_rate(self) -> f64 {
        let resolved = self.expert_hits + self.expert_misses;
        if resolved == 0 {
            return 0.0;
        }

        self.expert_hits as f64 / resolved as f64
    }

    /// Host-to-device expert bytes one token cost at this width.
    pub fn expert_h2d_bytes_per_token(self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }

        self.expert_h2d_bytes as f64 / self.tokens as f64
    }

    /// Host-to-device bytes one token cost across all four staged families.
    pub fn h2d_bytes_per_token(self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }

        let total = self.expert_h2d_bytes + self.embedding_h2d_bytes + self.engram_h2d_bytes;

        total as f64 / self.tokens as f64
    }

    /// Engram rows the host hash addressed at this width.
    pub const fn engram_rows(self) -> usize {
        self.engram_rows
    }

    /// Engram FP8 bytes uploaded at this width.
    pub const fn engram_h2d_bytes(self) -> usize {
        self.engram_h2d_bytes
    }

    /// Token-embedding bytes uploaded at this width.
    pub const fn embedding_h2d_bytes(self) -> usize {
        self.embedding_h2d_bytes
    }

    /// Bytes appended to the paged K/V planes at this width.
    pub const fn kv_append_bytes(self) -> usize {
        self.kv_append_bytes
    }

    fn observe(&mut self, step: &Qwen38FlashNextStepTelemetry) {
        self.rounds += 1;
        self.tokens += step.rows();
        self.forward += step.forward();
        self.expert_requests += step.expert_requests();
        self.expert_h2d_bytes += step.expert_h2d_bytes();
        self.embedding_h2d_bytes += step.embedding_h2d_bytes();
        self.engram_h2d_bytes += step.engram_h2d_bytes();
        self.engram_rows += step.engram_rows();
        self.kv_append_bytes += step.kv_append_bytes();
        for layer in step.layers() {
            self.expert_hits += layer.hits();
            self.expert_misses += layer.misses();
        }
    }
}

/// Decode evidence split by the width of the round that produced it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNextBatchTelemetry {
    widths: [Qwen38FlashNextBatchWidthTelemetry; MAX_BATCH],
    admissions: usize,
    retirements: usize,
    cancellations: usize,
}

impl Qwen38FlashNextBatchTelemetry {
    /// Evidence for the rounds run at exactly `width` rows.
    pub fn at(&self, width: usize) -> EngineResult<Qwen38FlashNextBatchWidthTelemetry> {
        self.widths
            .get(width.wrapping_sub(1))
            .copied()
            .ok_or_else(|| {
                EngineError::route(format!(
                    "Flash-Next batch width {width} is outside 1..={MAX_BATCH}"
                ))
            })
    }

    /// Every width in ascending order, including the ones no round reached.
    pub const fn widths(&self) -> &[Qwen38FlashNextBatchWidthTelemetry; MAX_BATCH] {
        &self.widths
    }

    /// Decode rounds the scheduler ran at every width together.
    pub fn rounds(&self) -> usize {
        self.widths.iter().map(|width| width.rounds).sum()
    }

    /// Tokens every decode round committed.
    pub fn tokens(&self) -> usize {
        self.widths.iter().map(|width| width.tokens).sum()
    }

    /// Mean rows a decode round carried, which is the occupancy the scheduler actually reached.
    pub fn mean_width(&self) -> f64 {
        let rounds = self.rounds();
        if rounds == 0 {
            return 0.0;
        }

        self.tokens() as f64 / rounds as f64
    }

    /// Requests admitted into a physical slot.
    pub const fn admissions(&self) -> usize {
        self.admissions
    }

    /// Requests that reached their own finish reason.
    pub const fn retirements(&self) -> usize {
        self.retirements
    }

    /// Requests whose caller went away before they finished.
    pub const fn cancellations(&self) -> usize {
        self.cancellations
    }

    fn observe(&mut self, step: &Qwen38FlashNextStepTelemetry) -> EngineResult<()> {
        let width = self
            .widths
            .get_mut(step.rows().wrapping_sub(1))
            .ok_or_else(|| {
                EngineError::route(format!(
                    "a Flash-Next decode round reported {} rows, outside 1..={MAX_BATCH}",
                    step.rows()
                ))
            })?;
        width.observe(step);

        Ok(())
    }
}

/// Frontend, resident program, stream, and logit banks behind the served Flash-Next model.
pub struct Qwen38FlashNextResidentBatchGenerator {
    frontend: TextFrontend,
    program: Qwen38FlashNextResidentModel,
    stream: Arc<CudaStream>,
    logits: PinnedHostBuffer<u16>,
    sessions: [Option<Qwen38FlashNextBatchSession>; MAX_BATCH],
    active_slots: [usize; MAX_BATCH],
    active: usize,
    next_request_id: u64,
    batch: Qwen38FlashNextBatchTelemetry,
}

struct Qwen38FlashNextBatchSession {
    request_id: ResidentRequestId,
    control: GenerationSession,
    pending_token: Option<u32>,
    next_position: u32,
}

impl Qwen38FlashNextResidentBatchGenerator {
    /// Opens the served Flash-Next program on device zero, reporting load progress.
    pub fn from_snapshot_device_zero_with_progress(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, Some(progress))
    }

    /// Opens the served Flash-Next program on device zero.
    pub fn from_snapshot_device_zero(
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        let context = device_zero_context()?;
        Self::from_snapshot(&context, snapshot, None)
    }

    /// Loads one resident program shared by eight physical request slots.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let frontend = TextFrontend::open(snapshot.as_ref())?;
        let program = Qwen38FlashNextResidentModel::from_snapshot_with_progress(
            context,
            Arc::clone(&snapshot),
            progress,
        )?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let logit_values = Qwen38FlashNext::VOCAB
            .checked_mul(LOGIT_BANK_ROWS)
            .ok_or_else(|| EngineError::layout("Flash-Next compact logit banks overflow"))?;
        let logits = PinnedHostBuffer::zeroed(context, logit_values).map_err(GpuError::from)?;

        Ok(Self {
            frontend,
            program,
            stream,
            logits,
            sessions: std::array::from_fn(|_| None),
            active_slots: [usize::MAX; MAX_BATCH],
            active: 0,
            next_request_id: 1,
            batch: Qwen38FlashNextBatchTelemetry::default(),
        })
    }

    /// Admits the lowest free slot and primes it without mutating siblings.
    pub fn admit(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> EngineResult<ResidentBatchAdmission> {
        let control = GenerationSession::start(&self.frontend, request)?;
        // Admission follows the qualified route, not the deeper KV allocation.
        let required_positions = require_generation_capacity(
            control.prompt_token_ids().len(),
            request.max_new_tokens,
            ModelProgram::context_capacity(&self.program),
        )?;
        let request_id = ResidentRequestId::from_raw(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| EngineError::generation("Flash-Next request identity overflows"))?;
        let prompt_tokens = control.prompt_token_ids().len();
        let prompt_metrics = control.prompt_metrics().clone();
        if control.finish_reason().is_some() {
            return Ok(ResidentBatchAdmission {
                request_id,
                prompt_tokens,
                device_reused_tokens: 0,
                native_prefill_tokens: 0,
                prompt_metrics,
                completed: Some(control.into_output()?),
            });
        }

        let occupied = std::array::from_fn(|slot| self.sessions[slot].is_some());
        let slot = qwen38_flash_next_admission_slot(occupied).ok_or_else(|| {
            EngineError::route(format!(
                "all {MAX_BATCH} Flash-Next generation slots are active"
            ))
        })?;
        // Clear every slot-owned state family left by the previous tenant.
        self.program.recycle_slot(&self.stream, slot)?;
        let native_prefill_tokens = match self
            .program
            .reserve_slot(&self.stream, slot, required_positions)
            .and_then(|_| {
                prime_prompt(
                    &mut self.program,
                    &self.stream,
                    control.prompt_token_ids(),
                    slot,
                )
            }) {
            Ok(tokens) => {
                if let Err(error) = self.program.read_logits_into(
                    &self.stream,
                    1,
                    &mut self.logits[slot_logits(slot)],
                ) {
                    self.program.recycle_slot(&self.stream, slot)?;
                    return Err(error);
                }
                tokens
            }
            Err(error) => {
                // Reservation or prime failure may leave pages owned.
                self.program.recycle_slot(&self.stream, slot)?;
                return Err(error);
            }
        };
        let next_position = u32::try_from(prompt_tokens)
            .map_err(|_| EngineError::generation("prompt length exceeds the position width"))?;
        self.sessions[slot] = Some(Qwen38FlashNextBatchSession {
            request_id,
            control,
            pending_token: None,
            next_position,
        });
        self.active_slots[self.active] = slot;
        self.active += 1;
        self.batch.admissions += 1;

        Ok(ResidentBatchAdmission {
            request_id,
            prompt_tokens,
            device_reused_tokens: 0,
            native_prefill_tokens,
            prompt_metrics,
            completed: None,
        })
    }

    /// Replays every pending token in one compact round, then samples one event per request.
    pub fn step(&mut self) -> EngineResult<ResidentBatchEvents> {
        if self.active == 0 {
            return Err(EngineError::generation(
                "cannot step an empty Flash-Next generation scheduler",
            ));
        }
        self.replay_pending()?;

        let mut events = std::array::from_fn(|_| None);
        let mut retired = [false; MAX_BATCH];
        let active = self.active;
        for (index, event) in events[..active].iter_mut().enumerate() {
            let slot = self.active_slots[index];
            let logits = slot_logits(slot);
            let step = {
                let session = self.sessions[slot].as_mut().ok_or_else(|| {
                    EngineError::generation("active Flash-Next slot has no generation session")
                })?;
                let step = session.control.accept_logits(&self.logits[logits])?;
                if step.finish_reason.is_none() {
                    session.pending_token = Some(step.token_id);
                }
                step
            };
            let request_id = self.sessions[slot]
                .as_ref()
                .expect("active Flash-Next session survived sampling")
                .request_id;
            let completed = if step.finish_reason.is_some() {
                let session = self.sessions[slot]
                    .take()
                    .expect("terminal Flash-Next session exists");
                self.program.recycle_slot(&self.stream, slot)?;
                retired[index] = true;
                self.batch.retirements += 1;
                Some(session.control.into_output()?)
            } else {
                None
            };
            *event = Some(ResidentBatchEvent {
                request_id,
                step,
                completed,
            });
        }
        let (survivors, surviving) =
            qwen38_flash_next_compact_survivors(&self.active_slots[..active], &retired[..active])?;
        self.active_slots = survivors;
        self.active = surviving;

        Ok(ResidentBatchEvents::from_events(events, active))
    }

    /// Cancels one request, releases its pages, and clears its carries.
    pub fn cancel(&mut self, request_id: ResidentRequestId) -> EngineResult<ResidentCancellation> {
        let index = self.active_slots[..self.active]
            .iter()
            .position(|&slot| {
                self.sessions[slot]
                    .as_ref()
                    .is_some_and(|session| session.request_id == request_id)
            })
            .ok_or_else(|| {
                EngineError::generation("Flash-Next cancellation request is not active")
            })?;
        let slot = self.active_slots[index];
        let session = self.sessions[slot]
            .take()
            .expect("cancelled Flash-Next slot owns a session");
        let mut retired = [false; MAX_BATCH];
        retired[index] = true;
        let (survivors, surviving) = qwen38_flash_next_compact_survivors(
            &self.active_slots[..self.active],
            &retired[..self.active],
        )?;
        self.active_slots = survivors;
        self.active = surviving;
        self.program.recycle_slot(&self.stream, slot)?;
        self.batch.cancellations += 1;

        Ok(ResidentCancellation {
            request_id,
            output: session.control.cancel()?,
            device_retained_tokens: 0,
        })
    }

    /// Requests currently holding a physical slot.
    pub const fn active_requests(&self) -> usize {
        self.active
    }

    /// Active request identities in stable compact-row order.
    pub fn active_request_ids(&self) -> impl Iterator<Item = ResidentRequestId> + '_ {
        self.active_slots[..self.active].iter().map(|&slot| {
            self.sessions[slot]
                .as_ref()
                .expect("active Flash-Next slot owns a session")
                .request_id
        })
    }

    /// Concurrent requests funded by the slot and carry layouts.
    pub const fn slot_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// The proven dense band a served request may reach, not the funded cache depth.
    pub fn context_capacity(&self) -> usize {
        ModelProgram::context_capacity(&self.program)
    }

    /// Device bytes across the resident, paged-cache, expert, and engram arenas.
    pub fn arena_bytes(&self) -> EngineResult<usize> {
        self.program.layout().total_device_bytes()
    }

    /// Source-backed weights this program uploaded to the device.
    pub fn resident_weight_bytes(&self) -> usize {
        self.program.layout().resident_weight_bytes()
    }

    /// Page-locked staging, engram, and logit-bank bytes this owner holds.
    pub fn host_stager_bytes(&self) -> usize {
        self.program.host_stager_bytes() + self.logits.num_bytes()
    }

    /// Construction evidence: upload, expert staging, and graph capture.
    pub const fn load_stats(&self) -> Qwen38FlashNextResidentLoadStats {
        self.program.load_stats()
    }

    /// Whether the packed primary extent is borrowed from the checkpoint mapping.
    pub fn mapped_primary(&self) -> bool {
        !self
            .program
            .layout()
            .streaming()
            .primary_source()
            .is_pinned()
    }

    /// Streaming and timing evidence folded over every request the scheduler has run.
    pub const fn telemetry(&self) -> Qwen38FlashNextGenerationTelemetry {
        self.program.generation_telemetry()
    }

    /// Decode evidence split by the width of the round that produced it.
    pub const fn batch_telemetry(&self) -> Qwen38FlashNextBatchTelemetry {
        self.batch
    }

    /// Restarts both telemetry accumulators, so a measurement can exclude its own warm-up.
    pub fn reset_telemetry(&mut self) {
        self.program.reset_generation_telemetry();
        self.batch = Qwen38FlashNextBatchTelemetry::default();
    }

    /// Checkpoint-admitted sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.frontend.generation_defaults()
    }

    /// CUDA context shared by every arena, stream, graph, and pinned buffer.
    pub const fn context(&self) -> &Arc<CudaContext> {
        self.program.context()
    }

    #[cfg(feature = "qualification")]
    /// Physical slot currently owning one active request.
    pub fn qualification_slot(&self, request_id: ResidentRequestId) -> Option<usize> {
        self.sessions.iter().position(|session| {
            session
                .as_ref()
                .is_some_and(|session| session.request_id == request_id)
        })
    }

    #[cfg(feature = "qualification")]
    /// The round the next [`Self::step`] would replay, without running it.
    pub fn qualification_round(&self) -> EngineResult<Qwen38FlashNextCompactRound> {
        self.pending_round()
    }

    #[cfg(feature = "qualification")]
    /// The two token ids one physical slot's engram carry is holding.
    pub fn qualification_engram_carry(
        &self,
        slot: usize,
    ) -> EngineResult<[u32; tuisko_model::QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN]> {
        self.program.qualification_engram_carry(slot)
    }

    #[cfg(feature = "qualification")]
    /// Tokens one physical slot's paged cache currently covers.
    pub fn qualification_slot_tokens(&self, slot: usize) -> EngineResult<usize> {
        self.program.slot_tokens(slot)
    }

    #[cfg(feature = "qualification")]
    /// Pages no slot currently owns, which is what a refused admission would have drawn from.
    pub fn qualification_free_pages(&self) -> usize {
        self.program.slots().free_pages()
    }

    #[cfg(feature = "qualification")]
    /// Stable retained device and pinned-logit addresses.
    pub fn qualification_addresses(&self) -> [usize; 3] {
        [
            self.program.base_address() as usize,
            self.program.kv_base_address() as usize,
            self.logits.as_ptr().addr(),
        ]
    }

    /// The dense round the pending subset of the active order describes.
    fn pending_round(&self) -> EngineResult<Qwen38FlashNextCompactRound> {
        let active = &self.active_slots[..self.active];
        let mut pending = [false; MAX_BATCH];
        for (flag, &slot) in pending[..self.active].iter_mut().zip(active) {
            *flag = self
                .sessions
                .get(slot)
                .and_then(|session| session.as_ref())
                .ok_or_else(|| {
                    EngineError::generation("active Flash-Next slot has no pending session")
                })?
                .pending_token
                .is_some();
        }

        qwen38_flash_next_compact_round(active, &pending[..self.active])
    }

    /// Runs one compact decode round and scatters its logits into the per-slot bank.
    fn replay_pending(&mut self) -> EngineResult<()> {
        let round = self.pending_round()?;
        if round.is_empty() {
            return Ok(());
        }
        let rows = round.rows();
        let mut tokens = [0u32; MAX_BATCH];
        let mut positions = [0u32; MAX_BATCH];
        for (row, &slot) in round.slots().iter().enumerate() {
            let session = self.sessions[slot]
                .as_ref()
                .expect("a pending row names a slot the round already resolved");
            tokens[row] = session
                .pending_token
                .expect("a pending row names a slot holding a token");
            positions[row] = session.next_position;
        }

        let step = self.program.decode_step(
            &self.stream,
            &tokens[..rows],
            &positions[..rows],
            round.slots(),
        )?;
        self.program.observe_decode_round(&step);
        self.batch.observe(&step)?;
        self.program.read_logits_into(
            &self.stream,
            rows,
            &mut self.logits[compact_logits(rows)],
        )?;
        for (row, &slot) in round.slots().iter().enumerate() {
            self.logits
                .copy_within(compact_row(row), slot * Qwen38FlashNext::VOCAB);
            let session = self.sessions[slot]
                .as_mut()
                .expect("a pending row names a slot holding a session");
            session.pending_token = None;
            session.next_position = session
                .next_position
                .checked_add(1)
                .ok_or_else(|| EngineError::generation("generation position overflows"))?;
        }

        Ok(())
    }
}

fn slot_logits(slot: usize) -> Range<usize> {
    row(slot, Qwen38FlashNext::VOCAB)
}

fn compact_logits(rows: usize) -> Range<usize> {
    compact(rows, Qwen38FlashNext::VOCAB)
}

fn compact_row(row: usize) -> Range<usize> {
    let begin = (MAX_BATCH + row) * Qwen38FlashNext::VOCAB;
    begin..begin + Qwen38FlashNext::VOCAB
}

#[cfg(test)]
mod tests {
    use super::{
        LOGIT_BANK_ROWS, MAX_BATCH, Qwen38FlashNextBatchTelemetry, compact_logits, compact_row,
        slot_logits,
    };
    use std::mem::size_of;
    use tuisko_model::{Arch, Qwen38FlashNext};

    #[test]
    fn compact_owner_byte_inventory_is_exact() {
        let layout = crate::Qwen38FlashNextResidentLayout::build().unwrap();
        let stagers = crate::QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS
            * (Qwen38FlashNext::HIDDEN * size_of::<u16>()
                + Qwen38FlashNext::NGRAM_HEADS * Qwen38FlashNext::NGRAM_HEAD_DIM)
            + 2 * LOGIT_BANK_ROWS * Qwen38FlashNext::VOCAB * size_of::<u16>();

        assert_eq!(layout.total_device_bytes().unwrap(), 30_675_307_776);
        assert_eq!(stagers, 23_756_800);
    }

    #[test]
    fn the_two_logit_banks_never_overlap() {
        // Compact downloads must not overwrite a slot awaiting sampling.
        for slot in 0..MAX_BATCH {
            let per_slot = slot_logits(slot);
            assert!(per_slot.end <= MAX_BATCH * Qwen38FlashNext::VOCAB);
            for row in 0..MAX_BATCH {
                let compact = compact_row(row);
                assert!(compact.start >= per_slot.end || compact.end <= per_slot.start);
            }
        }
        assert_eq!(
            compact_logits(MAX_BATCH).end,
            LOGIT_BANK_ROWS * Qwen38FlashNext::VOCAB
        );
        assert_eq!(compact_logits(1).start, compact_row(0).start);
    }

    #[test]
    fn every_compact_download_row_is_one_whole_vocabulary_row() {
        for row in 0..MAX_BATCH {
            assert_eq!(compact_row(row).len(), Qwen38FlashNext::VOCAB);
        }
        for rows in 1..=MAX_BATCH {
            assert_eq!(compact_logits(rows).len(), rows * Qwen38FlashNext::VOCAB);
        }
    }

    #[test]
    fn batch_telemetry_addresses_only_admitted_widths() {
        let telemetry = Qwen38FlashNextBatchTelemetry::default();

        for width in 1..=MAX_BATCH {
            assert_eq!(telemetry.at(width).unwrap().rounds(), 0);
        }
        assert!(telemetry.at(0).is_err());
        assert!(telemetry.at(MAX_BATCH + 1).is_err());
        assert_eq!(telemetry.rounds(), 0);
        assert_eq!(telemetry.tokens(), 0);
        assert_eq!(telemetry.mean_width(), 0.0);
    }

    #[test]
    fn an_unreached_width_reports_zero_rather_than_a_division() {
        let width = Qwen38FlashNextBatchTelemetry::default().at(4).unwrap();

        assert_eq!(width.round_ms(), 0.0);
        assert_eq!(width.tokens_per_second(), 0.0);
        assert_eq!(width.expert_hit_rate(), 0.0);
        assert_eq!(width.expert_h2d_bytes_per_token(), 0.0);
        assert_eq!(width.h2d_bytes_per_token(), 0.0);
    }
}
