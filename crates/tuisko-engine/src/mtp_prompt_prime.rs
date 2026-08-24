//! Exact target-residual handoff and prompt priming for the Qwen3.8 MTP layer.

use crate::mtp_prompt_prime_layout::{
    MTP_PROMPT_TILE_CAPACITY, MtpPromptPrimeLayout, MtpPromptPrimeRegions,
};
use crate::{
    EngineError, EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, ResidentModelProgram,
};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{MtpBf16FusionOp, MtpBf16QkPrepareOp, MtpBf16QkvOp, ResidualNormOp};
use tuisko_model::{Arch, MtpBindings, Qwen38_27B, TextEndpointBindings};

const ROTARY_PAIRS: usize = 32;
const PROMPT_ROUTES: [usize; 5] = [1, 32, 64, 128, MTP_PROMPT_TILE_CAPACITY];

/// Exact MTP prompt-prime graph selected by one checked metadata staging call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the MTP prompt-prime route must be replayed with the staged inputs"]
pub struct MtpPromptPrimeRoute {
    rows: usize,
    slot: usize,
    first_position: usize,
}

impl MtpPromptPrimeRoute {
    /// Number of aligned target residual and next-token embedding rows.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Stable resident slot whose page-table row receives MTP cache entries.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// Absolute cache position of the first primed row.
    pub const fn first_position(self) -> usize {
        self.first_position
    }

    #[cfg(feature = "qualification")]
    /// Constructs one exact route for qualification graph lookup.
    pub fn qualified(rows: usize, slot: usize, first_position: usize) -> EngineResult<Self> {
        require_route(rows)?;
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "MTP prompt qualification slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        Ok(Self {
            rows,
            slot,
            first_position,
        })
    }
}

/// Source-backed prompt-prime owner borrowing the target addresses captured by its graphs.
pub struct MtpPromptPrimeProgram<'a> {
    // Graphs retain target/local arena, pinned-host, and module addresses, so they drop first.
    graphs: [CudaGraph; PROMPT_ROUTES.len()],
    arena: DeviceArena,
    _fusion: MtpBf16FusionOp,
    _norm: ResidualNormOp<Qwen38_27B>,
    _qkv: MtpBf16QkvOp,
    _qk_prepare: MtpBf16QkPrepareOp,
    embedding_stager: PinnedHostBuffer<u16>,
    table_rows_stager: PinnedHostBuffer<u32>,
    positions_stager: PinnedHostBuffer<u32>,
    rope_cos_stager: PinnedHostBuffer<f32>,
    rope_sin_stager: PinnedHostBuffer<f32>,
    target: &'a ResidentModelProgram,
    context: Arc<CudaContext>,
    layout: MtpPromptPrimeLayout,
    base_address: u64,
}

#[derive(Clone, Copy)]
struct Pointers {
    embedding: *mut u16,
    target_hidden: *mut u16,
    embedding_norm: *const u16,
    hidden_norm: *const u16,
    normalized_embedding: *mut u16,
    normalized_hidden: *mut u16,
    input_projection: *const u16,
    residual: *mut u16,
    input_norm: *const u16,
    attention_normalized: *mut u16,
    qkv_weight: *const u16,
    qkv: *mut u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *mut f32,
    rope_sin: *mut f32,
    block_tables: *mut u32,
    table_rows: *mut u32,
    cache_positions: *mut u32,
    query: *mut f32,
    key_pages: *mut u16,
    value_pages: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: MtpPromptPrimeRegions) -> GpuResult<Self> {
        Ok(Self {
            embedding: arena.address(regions.embedding)?,
            target_hidden: arena.address(regions.target_hidden)?,
            embedding_norm: arena.address(regions.embedding_norm)?.cast_const(),
            hidden_norm: arena.address(regions.hidden_norm)?.cast_const(),
            normalized_embedding: arena.address(regions.normalized_embedding)?,
            normalized_hidden: arena.address(regions.normalized_hidden)?,
            input_projection: arena.address(regions.input_projection)?.cast_const(),
            residual: arena.address(regions.residual)?,
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            attention_normalized: arena.address(regions.attention_normalized)?,
            qkv_weight: arena.address(regions.qkv_weight)?.cast_const(),
            qkv: arena.address(regions.qkv)?,
            query_norm: arena.address(regions.query_norm)?.cast_const(),
            key_norm: arena.address(regions.key_norm)?.cast_const(),
            rope_cos: arena.address(regions.rope_cos)?,
            rope_sin: arena.address(regions.rope_sin)?,
            block_tables: arena.address(regions.block_tables)?,
            table_rows: arena.address(regions.table_rows)?,
            cache_positions: arena.address(regions.cache_positions)?,
            query: arena.address(regions.query)?,
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> Vec<usize> {
        vec![
            self.embedding.addr(),
            self.target_hidden.addr(),
            self.embedding_norm.addr(),
            self.hidden_norm.addr(),
            self.normalized_embedding.addr(),
            self.normalized_hidden.addr(),
            self.input_projection.addr(),
            self.residual.addr(),
            self.input_norm.addr(),
            self.attention_normalized.addr(),
            self.qkv_weight.addr(),
            self.qkv.addr(),
            self.query_norm.addr(),
            self.key_norm.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.block_tables.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.query.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    fusion: &'a MtpBf16FusionOp,
    norm: &'a ResidualNormOp<Qwen38_27B>,
    qkv: &'a MtpBf16QkvOp,
    qk_prepare: &'a MtpBf16QkPrepareOp,
}

#[derive(Clone, Copy)]
struct Stagers<'a> {
    embedding: &'a PinnedHostBuffer<u16>,
    table_rows: &'a PinnedHostBuffer<u32>,
    positions: &'a PinnedHostBuffer<u32>,
    rope_cos: &'a PinnedHostBuffer<f32>,
    rope_sin: &'a PinnedHostBuffer<f32>,
}

impl<'a> MtpPromptPrimeProgram<'a> {
    /// Loads the exact prompt-prime BF16 source family and captures all admitted routes.
    pub fn from_target(target: &'a ResidentModelProgram) -> EngineResult<Self> {
        let context = target.context().clone();
        let mtp = MtpBindings::bind(target.snapshot().as_ref())?;
        let qkv = mtp.materialize_qkv()?;
        let layout = MtpPromptPrimeLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;

        arena.copy_region_bytes_from_host(
            &stream,
            regions.embedding_norm,
            mtp.embedding_norm.bytes(),
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.hidden_norm, mtp.hidden_norm.bytes())?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.input_projection,
            mtp.input_projection.bytes(),
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.input_norm, mtp.input_norm.bytes())?;
        arena.copy_region_bytes_from_host(&stream, regions.qkv_weight, &qkv.weight_bf16)?;
        arena.copy_region_bytes_from_host(&stream, regions.query_norm, mtp.query_norm.bytes())?;
        arena.copy_region_bytes_from_host(&stream, regions.key_norm, mtp.key_norm.bytes())?;

        let fusion = MtpBf16FusionOp::new(&context)?;
        let norm = ResidualNormOp::new(&context)?;
        let qkv_op = MtpBf16QkvOp::new(&context)?;
        let qk_prepare = MtpBf16QkPrepareOp::new(&context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(
            &context,
            product(
                "MTP prompt embedding stager elements",
                MTP_PROMPT_TILE_CAPACITY,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let table_rows_stager =
            PinnedHostBuffer::zeroed(&context, MTP_PROMPT_TILE_CAPACITY).map_err(GpuError::from)?;
        let positions_stager =
            PinnedHostBuffer::zeroed(&context, MTP_PROMPT_TILE_CAPACITY).map_err(GpuError::from)?;
        let rotary_values = product(
            "MTP prompt rotary stager elements",
            MTP_PROMPT_TILE_CAPACITY,
            ROTARY_PAIRS,
        )?;
        let rope_cos_stager =
            PinnedHostBuffer::zeroed(&context, rotary_values).map_err(GpuError::from)?;
        let rope_sin_stager =
            PinnedHostBuffer::zeroed(&context, rotary_values).map_err(GpuError::from)?;
        let pointers = Pointers::bind(&arena, regions)?;
        let ops = Ops {
            fusion: &fusion,
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
        };
        let stagers = Stagers {
            embedding: &embedding_stager,
            table_rows: &table_rows_stager,
            positions: &positions_stager,
            rope_cos: &rope_cos_stager,
            rope_sin: &rope_sin_stager,
        };
        let graphs = capture_routes(&stream, target, &arena, regions, pointers, ops, stagers)?;
        let base_address = arena.base_address();

        Ok(Self {
            graphs,
            arena,
            _fusion: fusion,
            _norm: norm,
            _qkv: qkv_op,
            _qk_prepare: qk_prepare,
            embedding_stager,
            table_rows_stager,
            positions_stager,
            rope_cos_stager,
            rope_sin_stager,
            target,
            context,
            layout,
            base_address,
        })
    }

    /// Stages next-token embeddings and exact MTP cache metadata for one prompt route.
    ///
    /// Synchronization prevents mutation of page-locked sources still referenced by a prior
    /// replay. Target prefill for these residual rows must already be enqueued on the same stream.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        first_position: usize,
        next_token_ids: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<MtpPromptPrimeRoute> {
        require_route(rows)?;
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "MTP prompt slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if next_token_ids.len() != rows {
            return Err(EngineError::layout(format!(
                "MTP prompt has {} next-token identifiers, expected {rows}",
                next_token_ids.len()
            )));
        }
        let rotary_values = product("MTP prompt rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "MTP prompt rotary planes have {}/{} values, expected {rotary_values}",
                rope_cos.len(),
                rope_sin.len()
            )));
        }
        self.target
            .validate_mtp_prompt_cache(slot, first_position, rows)?;
        stream.synchronize().map_err(GpuError::from)?;

        let embedding = TextEndpointBindings::bind_embedding(self.target.snapshot().as_ref())?;
        for (row, &token) in next_token_ids.iter().enumerate() {
            let token = usize::try_from(token).map_err(|_| {
                EngineError::route("MTP prompt token identifier exceeds host width")
            })?;
            if token >= Qwen38_27B::VOCAB {
                return Err(EngineError::route(format!(
                    "MTP prompt token {token} is outside vocabulary 0..{}",
                    Qwen38_27B::VOCAB
                )));
            }
            copy_embedding_row(
                embedding.bytes(),
                token,
                &mut self.embedding_stager
                    [row * Qwen38_27B::HIDDEN..(row + 1) * Qwen38_27B::HIDDEN],
            )?;
            self.table_rows_stager[row] = u32::try_from(slot)
                .map_err(|_| EngineError::layout("MTP prompt slot exceeds u32"))?;
            self.positions_stager[row] = u32::try_from(
                first_position
                    .checked_add(row)
                    .ok_or_else(|| EngineError::route("MTP prompt position overflows"))?,
            )
            .map_err(|_| EngineError::route("MTP prompt position exceeds u32"))?;
        }
        self.rope_cos_stager[..rotary_values].copy_from_slice(rope_cos);
        self.rope_sin_stager[..rotary_values].copy_from_slice(rope_sin);

        Ok(MtpPromptPrimeRoute {
            rows,
            slot,
            first_position,
        })
    }

    /// Replays the immutable handoff-and-prime graph matching the staged route.
    pub fn replay(&self, stream: &CudaStream, route: MtpPromptPrimeRoute) -> EngineResult<()> {
        // SAFETY: this MtpPromptPrimeProgram owns every local captured allocation
        // (arena, pinned stagers, op modules), borrows the target arenas for
        // lifetime 'a, and drops the graphs first.
        unsafe { self.graph(route.rows)?.launch(stream) }?;
        Ok(())
    }

    /// Clears both complete represented BF16 MTP cache planes.
    pub fn reset_cache(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.key_pages, 0)?;
        self.arena.fill(stream, regions.value_pages, 0)?;
        Ok(())
    }

    /// CUDA context shared by target, prompt owner, and all captured graphs.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every exact route.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Exact unchanged BF16 prompt-prime source bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact represented BF16 cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Address-stable device workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete device owner bytes without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.layout.owner_bytes()
    }

    /// Complete device arena including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Exact alignment padding in the device arena.
    pub const fn padding_bytes(&self) -> usize {
        self.layout.padding_bytes()
    }

    /// Page-locked host bytes retained by all graph upload nodes.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
            + self.table_rows_stager.num_bytes()
            + self.positions_stager.num_bytes()
            + self.rope_cos_stager.num_bytes()
            + self.rope_sin_stager.num_bytes()
    }

    /// Exact graph inventory: scalar K1 plus four prompt tile widths.
    pub const fn graph_count(&self) -> usize {
        PROMPT_ROUTES.len()
    }

    /// Checked prompt-prime layout.
    pub const fn layout(&self) -> &MtpPromptPrimeLayout {
        &self.layout
    }

    fn graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        let index = route_index(rows).ok_or_else(|| {
            EngineError::route(format!(
                "MTP prompt rows {rows} are outside exact K=1 or T=32,64,128,1024"
            ))
        })?;
        Ok(&self.graphs[index])
    }

    #[cfg(feature = "qualification")]
    /// Launches the production handoff and prime boundary eagerly.
    pub fn launch_eager(
        &self,
        stream: &CudaStream,
        route: MtpPromptPrimeRoute,
    ) -> EngineResult<()> {
        let regions = self.layout.regions();
        launch_route(
            stream,
            route.rows,
            self.target,
            &self.arena,
            regions,
            Pointers::bind(&self.arena, regions)?,
            self.ops(),
            self.stagers(),
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns every local and captured target handoff address.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = Pointers::bind(&self.arena, self.layout.regions())?.addresses();
        addresses.extend(self.target.qualification_mtp_prompt_source_addresses()?);
        Ok(addresses)
    }

    #[cfg(feature = "qualification")]
    /// Returns the production graph for direct prompt-prime timing.
    pub fn qualification_graph(&self, route: MtpPromptPrimeRoute) -> EngineResult<&CudaGraph> {
        self.graph(route.rows)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated production prompt-prime work for amortized direct timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        route: MtpPromptPrimeRoute,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        if operations == 0 {
            return Err(EngineError::route(
                "MTP prompt repeated graph requires at least one operation",
            ));
        }
        let regions = self.layout.regions();
        let pointers = Pointers::bind(&self.arena, regions)?;
        let ops = self.ops();
        let stagers = self.stagers();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(
                    stream,
                    route.rows,
                    self.target,
                    &self.arena,
                    regions,
                    pointers,
                    ops,
                    stagers,
                )?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Fills every mutable prompt seam and both cache planes with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.embedding,
            regions.target_hidden,
            regions.normalized_embedding,
            regions.normalized_hidden,
            regions.residual,
            regions.attention_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        self.arena.fill(stream, regions.qkv, byte)?;
        self.arena.fill(stream, regions.rope_cos, byte)?;
        self.arena.fill(stream, regions.rope_sin, byte)?;
        self.arena.fill(stream, regions.block_tables, byte)?;
        self.arena.fill(stream, regions.table_rows, byte)?;
        self.arena.fill(stream, regions.cache_positions, byte)?;
        self.arena.fill(stream, regions.query, byte)?;
        self.arena.fill(stream, regions.key_pages, byte)?;
        self.arena.fill(stream, regions.value_pages, byte)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads complete maximum-route workspaces so inactive and guard values remain observable.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<MtpPromptPrimeObservables> {
        let regions = self.layout.regions();
        Ok(MtpPromptPrimeObservables {
            embedding: self.arena.copy_to_host(stream, regions.embedding)?,
            target_hidden: self.arena.copy_to_host(stream, regions.target_hidden)?,
            normalized_embedding: self
                .arena
                .copy_to_host(stream, regions.normalized_embedding)?,
            normalized_hidden: self.arena.copy_to_host(stream, regions.normalized_hidden)?,
            residual: self.arena.copy_to_host(stream, regions.residual)?,
            attention_normalized: self
                .arena
                .copy_to_host(stream, regions.attention_normalized)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            rope_cos: self.arena.copy_to_host(stream, regions.rope_cos)?,
            rope_sin: self.arena.copy_to_host(stream, regions.rope_sin)?,
            block_tables: self.arena.copy_to_host(stream, regions.block_tables)?,
            table_rows: self.arena.copy_to_host(stream, regions.table_rows)?,
            cache_positions: self.arena.copy_to_host(stream, regions.cache_positions)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads one physical BF16 cache page from each MTP plane.
    pub fn qualification_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        if physical_page >= LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::route(format!(
                "MTP prompt physical page {physical_page} is outside 0..{LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        let values = cache_page_values()?;
        let start = product("MTP prompt cache-page offset", physical_page, values)?;
        let regions = self.layout.regions();
        Ok((
            self.arena
                .copy_slice_to_host(stream, regions.key_pages, start, values)?,
            self.arena
                .copy_slice_to_host(stream, regions.value_pages, start, values)?,
        ))
    }

    #[cfg(feature = "qualification")]
    fn ops(&self) -> Ops<'_> {
        Ops {
            fusion: &self._fusion,
            norm: &self._norm,
            qkv: &self._qkv,
            qk_prepare: &self._qk_prepare,
        }
    }

    #[cfg(feature = "qualification")]
    fn stagers(&self) -> Stagers<'_> {
        Stagers {
            embedding: &self.embedding_stager,
            table_rows: &self.table_rows_stager,
            positions: &self.positions_stager,
            rope_cos: &self.rope_cos_stager,
            rope_sin: &self.rope_sin_stager,
        }
    }
}

#[cfg(feature = "qualification")]
/// Every externally observable non-cache seam in the maximum prompt workspace.
pub struct MtpPromptPrimeObservables {
    /// Next-token embedding rows staged from the source table.
    pub embedding: Vec<u16>,
    /// Raw target residual rows copied device-to-device.
    pub target_hidden: Vec<u16>,
    /// Embedding rows after the dedicated pre-fusion norm.
    pub normalized_embedding: Vec<u16>,
    /// Target residual rows after the dedicated pre-fusion norm.
    pub normalized_hidden: Vec<u16>,
    /// BF16 fusion-projection output rows.
    pub residual: Vec<u16>,
    /// Fused rows after the MTP attention input norm.
    pub attention_normalized: Vec<u16>,
    /// Gathered query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Captured MRoPE cosine plane.
    pub rope_cos: Vec<f32>,
    /// Captured MRoPE sine plane.
    pub rope_sin: Vec<f32>,
    /// Complete current target logical-to-physical page mapping.
    pub block_tables: Vec<u32>,
    /// Per-row selected stable slot.
    pub table_rows: Vec<u32>,
    /// Per-row absolute cache append position.
    pub cache_positions: Vec<u32>,
    /// Prepared FP32 query rows.
    pub query: Vec<f32>,
}

fn capture_routes(
    stream: &CudaStream,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: MtpPromptPrimeRegions,
    pointers: Pointers,
    ops: Ops<'_>,
    stagers: Stagers<'_>,
) -> EngineResult<[CudaGraph; PROMPT_ROUTES.len()]> {
    let mut graphs = Vec::with_capacity(PROMPT_ROUTES.len());
    for rows in PROMPT_ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, rows, target, arena, regions, pointers, ops, stagers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("MTP prompt graph inventory has wrong cardinality"))
}

#[allow(clippy::too_many_arguments)]
fn launch_route(
    stream: &CudaStream,
    rows: usize,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: MtpPromptPrimeRegions,
    pointers: Pointers,
    ops: Ops<'_>,
    stagers: Stagers<'_>,
) -> GpuResult<()> {
    let hidden_values = rows
        .checked_mul(Qwen38_27B::HIDDEN)
        .ok_or_else(|| GpuError::invalid_launch("MTP prompt hidden count overflows"))?;
    let rotary_values = rows
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| GpuError::invalid_launch("MTP prompt rotary count overflows"))?;
    // SAFETY: the owner retains all five pinned sources and both device arenas at fixed addresses
    // until every captured graph is dropped. Exact route counts bound every copy and leaf launch.
    unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.embedding,
            stagers.embedding,
            hidden_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.table_rows,
            stagers.table_rows,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.cache_positions,
            stagers.positions,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_cos,
            stagers.rope_cos,
            rotary_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_sin,
            stagers.rope_sin,
            rotary_values,
        )?;
        target.enqueue_mtp_prompt_handoff(
            stream,
            rows,
            arena,
            regions.target_hidden,
            regions.block_tables,
        )?;
        ops.fusion.launch(
            stream,
            rows,
            pointers.embedding,
            pointers.target_hidden,
            pointers.embedding_norm,
            pointers.hidden_norm,
            pointers.normalized_embedding,
            pointers.normalized_hidden,
            pointers.input_projection,
            pointers.residual,
        )?;
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual,
            pointers.input_norm,
            pointers.attention_normalized,
        )?;
        ops.qkv.launch(
            stream,
            rows,
            pointers.attention_normalized,
            pointers.qkv_weight,
            pointers.qkv,
        )?;
        ops.qk_prepare.launch(
            stream,
            rows,
            pointers.qkv,
            pointers.query_norm,
            pointers.key_norm,
            pointers.rope_cos,
            pointers.rope_sin,
            pointers.block_tables,
            pointers.table_rows,
            LONG_CONTEXT_PHYSICAL_PAGES,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
        )?;
    }
    Ok(())
}

const fn route_index(rows: usize) -> Option<usize> {
    match rows {
        1 => Some(0),
        32 => Some(1),
        64 => Some(2),
        128 => Some(3),
        MTP_PROMPT_TILE_CAPACITY => Some(4),
        _ => None,
    }
}

fn require_route(rows: usize) -> EngineResult<()> {
    if route_index(rows).is_none() {
        return Err(EngineError::route(format!(
            "MTP prompt rows {rows} are outside exact K=1 or T=32,64,128,1024"
        )));
    }
    Ok(())
}

fn copy_embedding_row(source: &[u8], token: usize, destination: &mut [u16]) -> EngineResult<()> {
    if destination.len() != Qwen38_27B::HIDDEN {
        return Err(EngineError::layout(format!(
            "MTP prompt embedding destination has {} words, expected {}",
            destination.len(),
            Qwen38_27B::HIDDEN
        )));
    }
    let word_begin = product("MTP prompt embedding row offset", token, Qwen38_27B::HIDDEN)?;
    let byte_begin = product("MTP prompt embedding byte offset", word_begin, 2)?;
    let byte_len = product("MTP prompt embedding row bytes", Qwen38_27B::HIDDEN, 2)?;
    let byte_end = byte_begin
        .checked_add(byte_len)
        .ok_or_else(|| EngineError::layout("MTP prompt embedding byte range overflows"))?;
    let row = source.get(byte_begin..byte_end).ok_or_else(|| {
        EngineError::layout(format!(
            "MTP prompt embedding row {token} is outside source"
        ))
    })?;
    for (target, bytes) in destination.iter_mut().zip(row.as_chunks::<2>().0) {
        *target = u16::from_le_bytes(*bytes);
    }
    Ok(())
}

#[cfg(feature = "qualification")]
fn cache_page_values() -> EngineResult<usize> {
    product(
        "MTP prompt cache page values",
        product(
            "MTP prompt cache page heads",
            Qwen38_27B::NUM_KV_HEADS,
            tuisko_kernels_sm120::ATTENTION_PAGE_SIZE,
        )?,
        Qwen38_27B::HEAD_DIM,
    )
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{PROMPT_ROUTES, require_route, route_index};

    #[test]
    fn exact_prompt_route_inventory_is_complete() {
        assert_eq!(PROMPT_ROUTES, [1, 32, 64, 128, 1_024]);
        for (index, rows) in PROMPT_ROUTES.into_iter().enumerate() {
            assert_eq!(route_index(rows), Some(index));
            assert!(require_route(rows).is_ok());
        }
        for rows in [0, 2, 8, 16, 31, 33, 1_023, 1_025] {
            assert!(require_route(rows).is_err());
        }
    }
}
