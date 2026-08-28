# Plan: GPU memory offload — model unload/load and durable-state park/resume

Status: implementation in progress. Last updated 2026-08-28.

The A1 host lifecycle and B0 checked segment manifest are implemented. The B0 VMM
release/remap/same-graph feasibility probe passes on the exact RTX 5090, including the checked
free-memory delta. The current branch also contains the B1 VMM arena migration, B2 compact pinned
durable-state owner/type-state transition, and B3 serving routes and host tests. A and B remain
unqualified until their complete numerical, resource, artifact, failure, and performance gates
run; implementation or a representative primitive probe is not exact-model device authority.

This plan covers only the complete product target,
`unsloth/Qwen3.8-27B-NVFP4` at revision
`16b6615af3548b88e2d8e382457bc705b00479cf`, served by the Qwen3.8 target-plus-MTP
resident generator on one SM120 RTX 5090. The shared worker plumbing may remain generic where that
does not make an unfinished lifecycle operation constructible, but the new routes are rejected as
unsupported for every other `ServerModel` until that exact target has its own lifecycle gate.

The resident server currently retains all CUDA ownership for its process lifetime. A co-tenant
cannot use that memory without killing the process, which also loses the retained prefix-cache
state and pays the complete reload cost. This plan adds two distinct capabilities:

- **A — unload/load**: drop the Qwen3.8 target, MTP, graphs, CUDA context reference, pinned
  stagers, frontend, and admitted snapshot; explicitly reload them later or auto-reload for the
  first new chat request.
- **B — park/resume**: preserve the generator and its captured virtual addresses while copying
  only durable target/MTP cache and recurrent state to pinned host memory, releasing the
  corresponding physical device mappings, and leaving weights plus scratch mapped.

`unload` does not preserve prefix-cache state. `park` does. Neither feature claims persistence
across process exit.

## 1. Exact production memory ledger

The current Qwen3.8 serving owner contains a target program and an incremental MTP program. All
figures below are checked assertions, not `nvidia-smi` estimates.

Sources:

- `crates/tuisko-engine/src/qwen38/resident_model_layout.rs`
- `crates/tuisko-engine/src/qwen38/resident_mtp_layout.rs`
- `ResidentMtpBatchGenerator::device_owner_bytes()`

| Region | Allocation bytes | GiB | Park treatment |
| --- | ---: | ---: | --- |
| Target represented weights | 19,103,682,560 | 17.792 | stays mapped |
| Target GDN causal history | 23,592,960 | 0.022 | mirror live rows; release eligible granules |
| Target GDN recurrent state | 1,207,959,552 | 1.125 | mirror live rows; release eligible granules |
| Target shared scratch | 948,860,932 | 0.884 | stays mapped; not durable state |
| Target resident-arena allocation, including padding | 21,284,111,616 | 19.822 | mixed |
| Target E4M3 KV data | 7,210,008,576 | 6.715 | mirror in-use pages |
| Target KV tables | 110,016 | 0.0001 | mirror or reconstruct and verify exactly |
| Target KV-arena allocation, including padding | 7,210,118,656 | 6.715 | release whole backing |
| **Target allocation total** | **28,494,230,272** | **26.537** | — |
| MTP represented weights | 849,398,784 | 0.791 | stays mapped |
| MTP scratch and route metadata | 122,904,032 | 0.114 | stays mapped |
| MTP main-arena allocation, including padding | 972,303,104 | 0.906 | stays mapped |
| MTP BF16 KV-arena allocation | 901,251,072 | 0.839 | mirror in-use pages; release whole backing |
| **MTP allocation total** | **1,873,554,176** | **1.745** | — |
| **Production generator allocation total** | **30,367,784,448** | **28.282** | — |

The target KV allocation is **one shared physical pool with 220,000 token positions total**. It is
not eight independent 220K pools. Each of the eight block-table rows can address the shared pool,
and one slot may consume most or all of it. There is therefore no fixed `KV bytes / 8` per-slot
ownership figure.

The maximum durable device payload before VMM-granule rounding is:

```
target KV allocation       7,210,118,656
target history + state     1,231,552,512
MTP KV allocation            901,251,072
                           -------------
                           9,342,922,240 bytes = 8.701 GiB
```

The exact parkable allocation count is not declared until MR-B0 queries the RTX 5090's minimum
VMM granularity and classifies every target resident-arena granule. Granules containing any weight
byte remain mapped. Target and MTP KV arenas are separate allocations and can be released in full.

An observed loaded process used 30,404 MiB in `nvidia-smi`. The exact arena ownership above is
28,961 MiB when rounded to the nearest MiB, leaving about 1,443 MiB for context, modules, graph
executables, and other driver ownership in that run. This observation is diagnostic, not a
portable floor or baseline.

## 2. Lifecycle API and state model

### 2.1 Routes

| Route | Success | Defined failures |
| --- | --- | --- |
| `POST /v1/unload` | `200 {"object":"model_unload","model":...,"state":"unloaded","was_loaded":bool,"was_parked":bool,"device_allocation_bytes_released":N}` | `409 transition_in_progress` or unsupported target; `503 worker_dead` |
| `POST /v1/load` | `200 {"object":"model_load","model":...,"state":"loaded","reloaded":bool,"load_ms":M}` | `409 model_parked`, transition, or unsupported target; `500 load_failed`; `503 worker_dead` |
| `POST /v1/park` | `200 {"object":"model_park","model":...,"state":"parked","host_bytes":N,"device_allocation_bytes_released":M,"park_ms":P}` | `409 unloaded`, transition, or unsupported target; `500 park_failed`; `503 worker_dead` |
| `POST /v1/resume` | `200 {"object":"model_resume","model":...,"state":"loaded","resumed":bool,"restore_ms":R}` | `409 unloaded`, transition, or unsupported target; `500 resume_failed`; `503 worker_dead` |
| `GET /health` | `200` while the HTTP process and engine worker are alive; includes `model_state` | `503 worker_dead` |
| `GET /ready` | `200` only in `Loaded`; includes `model_state` | `503` in every other live state |

`GET /v1/models` remains unchanged. OpenAI routes remain unchanged except for their documented
behavior while the model is unloaded, loading, parked, or transitioning.

An unloaded chat triggers one load attempt. Failure maps to retryable `503 model_load_failed` for
that chat and every chat admitted during the attempt; the explicit `/v1/load` caller receives the
administrative `500 load_failed` response in the table above.

The default listener is loopback. If the configured listener is non-loopback, lifecycle routes
must be disabled unless an explicit admin bearer token is configured; chat authorization is a
separate server concern. The token is never logged. This prevents an unauthenticated remote client
from repeatedly unloading or reloading the model.

### 2.2 Stable states and transitions

The engine worker owns the concrete generator state. Its published ingress state is changed only
under a short mutex: handlers reserve transitions around `try_send`, and the worker publishes
their completion or rollback under the same mutex. An admin transition and a concurrently
arriving chat therefore have a single defined order.

| Current state | Operation | Result |
| --- | --- | --- |
| `Loaded` | chat | enqueue normally |
| `Loaded` | unload | close the ingress fence, finish already accepted chats, then `Unloaded` |
| `Loaded` | load | idempotent `200 reloaded:false` |
| `Loaded` | park | close the ingress fence, finish already accepted chats, then `Parked` |
| `Unloaded` | chat | first chat starts one load; chats admitted during `Loading` pend |
| `Unloaded` | load | enter `Loading`, then `Loaded` or return to `Unloaded` on failure |
| `Unloaded` | unload | idempotent `200 was_loaded:false` |
| `Unloaded` | park/resume | `409` |
| `Parked` | chat | `503 model_parked`; never auto-resume |
| `Parked` | resume | enter `Resuming`, then `Loaded` or return to `Parked` on failure |
| `Parked` | park | idempotent `200` |
| `Parked` | unload | drop host mirror and remaining device/VA ownership, then `Unloaded` |
| `Parked` | load | `409 model_parked`; caller must resume or unload first |
| any transition | lifecycle operation | `409 transition_in_progress` |
| any transition | chat | `503` with the current transition and `Retry-After` |

The transient states are `Loading`, `Unloading`, `Parking`, and `Resuming`. A worker failure is a
terminal `Dead` state and preserves the existing process-exit behavior.

### 2.3 Exact ingress fence and queue behavior

The current worker performs continuous batching and calls `try_recv` while requests are active.
An admin job therefore cannot rely on being dequeued only at idle.

`AppState` instead owns `Arc<Mutex<Ingress>>`, containing the published lifecycle state and job
sender. Under this mutex:

1. a chat checks the state and enqueues only if that state admits chats;
2. unload/park changes `Loaded` to its transition state before enqueueing its admin job;
3. a failed admin enqueue rolls the state back before releasing the mutex; and
4. no later chat can land behind the unload/park fence.

Chats successfully enqueued before the fence are accepted work and finish normally. The worker
may dequeue the admin job while chats are active, stores it as `held_admin`, stops intake behind
it, and processes it only after `active_requests() == 0`. This preserves continuous batching and
gives unload/park a race-free boundary without draining unrelated concurrent arrivals.

During auto-reload, the triggering chat is held locally and the bounded channel can hold
`MAX_BATCH` additional chats, so that path permits at most `MAX_BATCH + 1` pending chats. An
explicit `/v1/load` holds an admin job locally and permits at most `MAX_BATCH` pended chats. Both
limits are recorded in tests and API documentation. Making every path total exactly eight would
require a separate admission permit and is not part of A.

## 3. Feature A — complete unload/load and auto-reload

### 3.1 Ownership and worker implementation

The Qwen3.8 worker becomes a concrete state machine holding one of:

- `Loaded(ResidentMtpBatchGenerator)`;
- `Unloaded`;
- the transient load/unload states; or
- after B lands, `Parked(ParkedQwen38Generator)`.

Other `ServerModel` arms retain the existing one-shot `start_generator` path. This avoids widening
`TextGenerator` with lifecycle methods that are not qualified for those exact targets.

Unload runs only after the ingress fence and scheduler-idle gate. It records the already checked
`ResidentMtpBatchGenerator::device_owner_bytes()`, drops the generator, and replies only after all
RAII destructors finish. That drop releases target and MTP arenas, graphs, modules, streams,
pinned stagers, frontend, and the admitted mmap-backed snapshot. No CUDA context reference may be
retained by the worker's reload closure.

The response reports exact allocation ownership released, not a claim about the instantaneous
global `nvidia-smi` delta. The resource qualification independently measures the resulting device
floor.

Load calls a closure that owns only the pinned snapshot path and immutable server configuration:

```rust
FnMut(Arc<ResidentLoadProgress>)
    -> Result<(ResidentMtpBatchGenerator, Ready), LoadFailure>
```

Each attempt receives a fresh `ResidentLoadProgress`; the startup-only monotonic progress object is
not reused. A separate scoped reporter polls runtime progress while the worker is synchronously in
the loader. Stdout retains the startup banner contract; runtime lifecycle messages go to stderr.

On runtime load failure, partially constructed CUDA ownership drops, the triggering and currently
pended chats receive the same structured failure, and the state returns to `Unloaded`. One request
causes at most one load attempt. A later chat or explicit `/v1/load` may retry; there is no internal
retry loop. Startup load failure still terminates startup as today.

### 3.2 Host tests

Use the existing crate-private fake-generator pattern and HTTP router tests. Required cases:

- ingress ordering proves no chat can enqueue behind an unload/park fence;
- an admin dequeued during active generation is held until idle;
- unload waits for already accepted chats and is idempotent;
- unload drops the generator before replying and publishes `Unloaded`;
- explicit load is idempotent while loaded;
- load failure drops partial ownership, fails all load-pended chats, and permits a later retry;
- auto-reload admits the triggering chat after a successful load;
- auto-reload admits at most `MAX_BATCH + 1` chats, explicit load admits at most `MAX_BATCH`, and
  excess ingress receives `429`;
- chat while parked returns `503` without starting resume;
- unload while parked reaches `Unloaded` and drops the host mirror;
- `/health` tests liveness while `/ready` tests model readiness;
- unsupported targets reject lifecycle routes without changing their existing worker path;
- non-loopback lifecycle-route admission requires the configured admin token; and
- every admin and chat error maps to the documented OpenAI/custom JSON and status code.

Run the standard host gates from `AGENTS.md`.

### 3.3 Device lifecycle qualification

Add a Qwen3.8 server-lifecycle suite selected together with its sibling accounting tests:

1. run the exact device preflight before opening the source snapshot;
2. load the production target-plus-MTP generator and record checked owner bytes;
3. run one greedy golden request and retain its output;
4. unload and assert the generator was dropped;
5. measure the post-unload device floor against an explicit authority and absolute tolerance;
6. submit the same chat, allowing one auto-reload;
7. require bit-identical greedy token output and logits at the declared observable boundary; and
8. assert the complete target+MTP allocation count after reload.

The initial floor authority and any later change to it are separate resource-baseline commits.
An exclusive local RTX 5090 is required. With explicit owner permission, `xtask remote` may satisfy
the numerical and resource gate but cannot bless timing.

### 3.4 MR split

- **MR-A1 — host behavior:** Qwen3.8 reloadable worker, ingress fence, routes, liveness/readiness,
  admin protection, and host tests. Other exact targets remain on the one-shot path.
- **MR-A2 — device authority:** lifecycle qualification, resource authority, README endpoint
  documentation, and A marked implemented only after the local or permitted remote numerical and
  resource gate passes.

## 4. Feature B — durable-state park/resume

### 4.1 Preserved state

Park is allowed only when the scheduler is idle. At that boundary there are no active chat
sessions, but inactive retained prefixes may own device state. “Conversation survives” means those
retained prefixes remain reusable; TuiskoLLM does not add a new server-side conversation object.

The pinned mirror contains a typed manifest and only semantically live data:

- target E4M3 K/V payloads for every physical page owned by a retained slot;
- target device block-table words, checked against the independent host page-route owner;
- target GDN history and FP32 recurrent rows for every retained slot;
- MTP BF16 K/V payloads for every physical page owned by a retained slot; and
- the physical page identifiers, slot generations, token counts, and checksums required to restore
  those bytes to the same logical ownership.

MTP device block tables live in the always-mapped MTP main arena. They are not copied, but park and
resume checksum them against the independent host page-route owner and require them to remain
unchanged.

The host `RetainedMtpSlot` token vectors, LRU clock, page-route owners, and frontend remain inside
the parked generator. Unused cache pages and inactive state rows are not copied; resume initializes
them to the production reset value before they can become observable.

Target and MTP scratch workspaces are not durable state and are not mirrored in phase 1. They stay
mapped. A later workspace-offload optimization is a separate measured feature with an explicit
initialization/liveness manifest.

Pinned allocation is all-or-nothing and occurs before any mapping is removed. Park failure leaves
the original loaded generator usable and publishes `Loaded`; it never exposes a partial mirror.
Worst-case mirror payload is approximately 8.7 GiB plus bounded manifest/alignment overhead, but
the response reports the exact allocation.

### 4.2 VMM ownership model

Captured graphs retain target, MTP, cache, state, and workspace virtual addresses. Ordinary
`cuMemFree`/`cuMemAlloc` cannot preserve those addresses. The preferred implementation reserves
stable VA ranges and changes only their physical backing.

The cuda-oxide revision already exposes `PhysicalAllocation`, `VirtualReservation`, `Mapping`,
`set_access`, and `allocation_granularity`. MR-B0 therefore answers feasibility, not API
availability.

`tuisko-gpu` owns a new model-independent checked VMM arena wrapper with explicit drop order:

1. query the exact device's minimum allocation granularity;
2. reserve one granularity-aligned VA range per current arena;
3. create granularity-aligned physical allocations for a checked segment manifest;
4. map them at `reserved_base + existing_region_offset` coverage;
5. call `cuMemSetAccess` before any upload or capture; and
6. drop mappings before their physical allocations and the VA reservation.

The invariant is relative, not equality with a pointer from an earlier `cuMemAlloc` run: every
typed region must still resolve to `reserved_base + its checked layout offset`. Graphs are captured
only after the VMM-backed owner is fully mapped and initialized.

The target KV arena and MTP KV arena each have separate backing and are wholly releasable. Target
GDN history/state are interleaved with layer weights at 256-byte layout alignment. MR-B0 classifies
minimum-granularity chunks as:

- **resident:** contains any weight or always-mapped workspace byte;
- **parkable:** contains only history/state bytes and padding; or
- **invalid:** overlaps an unowned gap in a way the checked manifest did not declare.

Mixed weight/state boundary granules remain resident. The exact releasable byte count is therefore:

```
target KV allocation
+ MTP KV allocation
+ target resident-arena granules classified parkable
```

Park performs, in order: idle/fence confirmation, complete mirror allocation, D2H copies,
stream/context synchronization, checksum validation, mapping drop, and physical-handle drop. The
VA reservations and resident mappings remain alive.

Resume creates new physical backing, maps it at the retained VAs, sets access, initializes all
non-restored bytes to the production reset value, restores mirrored bytes to their original
physical-page/state-row positions, synchronizes, validates checksums and device tables, and only
then exposes a loaded generator. A resume failure drops only newly created partial backing and
retains the complete host mirror so a later resume can retry.

The fallback is full arena reallocation plus complete graph recapture. It is not selected merely
because capture appears cheap: MR-B0 must first prove VMM unsupported or invalid on the exact
RTX 5090 route. If selected, the plan and gates are revised before implementation because the
current no-recapture and parked-owner invariants would no longer apply.

### 4.3 Type-state boundary

Park/resume does not widen the shared `TextGenerator` trait. Qwen3.8 uses target-specific owners:

- `ResidentMtpBatchGenerator` is the loaded, launch-capable owner;
- `ParkedQwen38Generator` owns graphs, stable VA reservations, resident mappings, host metadata,
  and the complete mirror, but exposes no admission or replay method; and
- consuming `park`/`resume` transitions are the only way to move between them.

This prevents a kernel or graph replay from being constructed while a required VA range is
unmapped.

### 4.4 Device gates

MR-B0 and every production B MR require an exclusive RTX 5090 preflight. The gates are:

- **VMM feasibility:** query support/granularity; map, initialize, capture a representative graph,
  replay, unmap and release backing, prove the free-memory delta, create new backing at the same VA,
  set access, restore, and replay the unchanged graph.
- **Independent byte oracle:** compare independently packed host bytes with every restored in-use
  target E4M3 page, MTP BF16 page, target block-table word, GDN BF16 history value, and GDN FP32
  state value. No decode or requantization occurs.
- **Numerical:** eager and CUDA Graph replay agree after resume at every observable boundary;
  pre-park and post-resume greedy logits and token output are bit-identical.
- **Inventory:** all admitted target and MTP exact routes, including every `B=1..8` entry, remain
  present. Under VMM, graph executable identities and captured node inventory do not change across
  park/resume.
- **Artifacts/resources:** generated PTX/SASS semantic inventories are unchanged, launch bounds
  remain preserved, stack/local memory remains zero, and released bytes equal the checked VMM
  segment manifest within an explicit driver tolerance.
- **Failure rollback:** injected host-allocation, copy, map, access, and restore failures prove the
  state returns to a usable `Loaded` or retryable `Parked` owner as specified.
- **Performance:** time the production park and resume operations directly with their real
  allocations, streams, page population, cache regime, and exact route matrix. Do not sum copy
  medians. Validate loaded clocks under sustained production-graph load before a long matrix.

The `xtask qualify-*` selection must include the numerical/device oracle and its sibling benchmark
accounting tests. If it selects multiple device tests, run them sequentially with one test thread
or sequence them inside one test so their CUDA preflights cannot race.

Remote timing remains diagnostic and cannot bless a baseline.

### 4.5 MR split

- **MR-B0 — inventory and feasibility:** host-generated typed region/granule manifest plus an
  exclusive-device VMM release/remap/same-graph probe. Record the VMM-versus-recapture decision.
- **MR-B1 — VMM arena ownership:** add the checked `tuisko-gpu` owner and migrate Qwen3.8 target
  plus MTP arenas without park behavior. Rerun the complete existing Qwen3.8 numerical, graph,
  artifact, resource, and performance dependency cone.
- **MR-B2 — engine park/resume:** add the type-state owners, durable mirror, rollback behavior,
  independent byte oracle, eager/graph agreement, inventory checks, and lifecycle benchmark.
- **MR-B3 — serving:** expose park/resume through the worker state machine, add HTTP and fence
  tests, run the server golden/resource qualification, and document the routes in `README.md`.

## 5. Decisions and deferred work

- Endpoint names remain the explicit lifecycle verbs. They are custom administrative routes, not
  claims of OpenAI-standard lifecycle behavior.
- Unload and park finish chats accepted before their ingress fence. Later chats are rejected during
  the transition; no racy channel drain is used.
- Auto-reload is enabled only from `Unloaded`. Park never auto-resumes.
- Auto-reload admits at most `MAX_BATCH + 1` chats; explicit load admits at most `MAX_BATCH`.
- B phase 1 parks the whole generator's retained durable state. Per-slot park is deferred.
- The parked mirror is process-local and is destroyed by unload or process exit.
- Workspace offload and persistence of parked state across restarts are separate features.

## 6. Sequencing and current gate status

| Step | Kind | Status |
| --- | --- | --- |
| A: MR-A1 host plumbing and tests | host | implemented |
| A: MR-A2 lifecycle qualification | device | pending |
| B: MR-B0 host manifest work | host | implemented |
| B: MR-B0 VMM feasibility probe | device | passed locally on the exact RTX 5090 |
| B: MR-B1 VMM ownership | host + device | host implementation complete; device gate pending |
| B: MR-B2 engine park/resume | host + device | host implementation complete; device gate pending |
| B: MR-B3 serving | host + device | host/server tests complete; exact-model device gate pending |

Device refusal recorded 2026-08-28 20:54 UTC: `nvidia-smi` showed 30,404 MiB used and
96–97% utilization bursts from a process outside this sandbox's PID namespace; its process table
was empty and no local `:8000` listener existed. That failed both the no-foreign-compute and
pre-used-memory gates. The device later became exclusively idle. `cargo run -p xtask --
build-sm120` passed its artifact and resource checks, and the isolated B0 probe proved mapping
release, its free-memory delta, same-address remap, and unchanged CUDA Graph replay. No complete A2
or exact-model B1/B2/B3 device qualification is reported passed. `xtask remote` requires explicit
owner permission, must follow the lease and cleanup rules in `docs/remote-gates.md`, and remains
non-authoritative for performance.
