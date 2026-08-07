# Python-native `cudnn.pygraph` and pluggable execution backends

## What this is

`cudnn.pygraph` is a Python-native graph class: graph structure (nodes,
tensors, op parameters) lives in Python with full introspection, and execution
dispatches through pluggable backends — python DSL engines and the cuDNN C++
backend. The C++ graph builder is internal
(`cudnn._pybind_module.backend_graph`) and is reached exclusively through
lowering.

```
cudnn.pygraph (Python IR)  →  create_execution_plans()  →  Router  →  routed plan list
  nodes / tensors / params        (route here,               PlanConfig(engine_id, knobs):
  fully introspectable            lazy lowering)             python engines + one backend entry
```

Why: python-DSL engines (CuTe-DSL / cuTile style GEMM and attention fusions)
need to *see* the graph to decide whether and how to run it. Previously that
required monkey-patching the pybind class and recording calls; now the graph
is natively introspectable and an engine is one file implementing
`BaseEngine`.

## Architecture

### Graph IR

- `graph_types.Tensor`, `nodes.Node`, `_pygraph.pygraph` — an engine-agnostic
  op DAG. Input/output **port names equal the C++ pybind kwarg names**,
  everywhere.
- Three declarative op mechanisms cover 100% of the C++ op surface:
  `_POINTWISE_TENSOR_ARGS` (54 uniform pointwise ops; `mode` == method name),
  `_STRUCTURED_OPS` (25 ops: norms, reduction, block-scale, MoE, conv,
  structural — one table entry each: ports, attrs, outputs, shape-infer),
  `_CAPTURED_OPS` (6 SDPA variants, ~130 kwargs: generic capture over an
  explicit per-op schema carrying positional order, output-direction kwargs,
  and conditional outputs). `matmul` is explicit for positional ergonomics.

### Backend contract (`engines/`)

- `BaseEngine`: `propose_plans(graph) → [PlanConfig]` (several knob configs
  per engine), `build_plan(graph, plan, ctx) → CompiledPlan` (the expensive
  JIT step, once per graph/plan, cached on the graph),
  `CompiledPlan.execute(graph, uid_to_data, ExecutionContext)` with explicit
  handle/stream/workspace/overrides. `uid_to_data` is the caller's variant
  pack, exactly as the classic backend receives it; engines that address
  buffers by port name call `resolve_node_buffers(graph, uid_to_data)`
  (`engines/base.py`), which joins the pack with each node's wired ports —
  strict missing-buffer validation, torch tensors detached once (DLPack/CAI
  refuse `requires_grad` export) — into per-node `NodeBuffers`
  (`{port_name: caller buffer}`). Simple eager engines implement
  `execute()` only.
- Every engine has a stable `engine_id` in one flat id space
  (`engines/engine_ids.py`: backend `[0, 10_000)`, C++ OSS `[10_000, 20_000)`,
  python `[20_000, …)` with a `FAMILY_BLOCK`-wide block per family, out-of-tree
  `30_000+`) — reproducible pinning/autotune. Engines do not DECLARE their id:
  the manifest holds every slot and `instantiate()` hands each factory the ids
  its engines are to use, so an engine cannot claim a number it was not given
  and the whole space is readable in one file.
- An engine declines a graph ONLY via `NotImplementedError`,
  `cudnn.cudnnGraphNotSupportedError`, or `ImportError`; anything else is an
  engine bug and propagates. `ImportError` counts because lowering imports are
  deferred past `check_support()` (see *Import boundaries*), so a missing
  optional dependency can only surface at build time — without it, a host
  lacking the `cutedsl` extra would lose graphs the backend could have served.
- The contract oracle is `TorchMatmulEngine` in
  `test/python/test_engine_router.py` (pure PyTorch, CPU): the dispatch
  contract is proven end to end without a GPU, and no oracle ships in the
  package.
- `GdnCuTileEngine` executes the single-node `gdn` and `gdn_bwd` ops (Gated
  DeltaNet linear attention) via the cuTile chunked kernels. Both ops are
  THD-only: token-packed `[total_T, heads, dim]` tensors with a required
  `cu_seqlens` (a dense batch is `[0, T, 2T, ...]`). `gdn_bwd` takes the
  forward inputs plus `dO` (and optionally `d_final_state`) and produces
  `dQ/dK/dV/dG/dBeta` (+ `d_initial_state` iff `initial_state` is given);
  the cumulative gate and intra-chunk WY matrix are recomputed inside the
  engine, so the graph contract carries no forward intermediates. Both are
  python-engine-only ops: they have no cuDNN backend lowering, so routing
  them to the backend entry raises `cudnnGraphNotSupportedError` at
  lowering. The kernels live in `cudnn.linear_attention.cutile.kernels.gdn_chunk_cutile`;
  the torch custom op `cudnn.linear_attention.ops.gated_delta_net` is a thin
  adapter that builds and executes cached `gdn`/`gdn_bwd` graphs (the SDPA
  op pattern), so it inherits whatever engine the planner selects. The
  optional `use_qk_l2norm` attribute asks the engine to L2-normalize the q/k
  rows in-kernel; `GdnFrostEngine` (the SM100/SM103 forward default) declines
  such graphs, the cuTile engine serves them.
- `KdaFrostEngine` / `KdaCuTileEngine` do the same for the single-node
  `kda` / `kda_bwd` ops (Kimi Delta Attention). KDA is GDN with a
  per-key-channel decay: its `g` is the log-space vector gate
  `[total_T, HV, K]` (GDN's is the scalar `[total_T, HV]`); `beta` stays
  scalar. The FROST engine (`cudnn.linear_attention.frost.kda_engine`) is
  the forward default on SM100/SM103; the node's `use_qk_l2norm` attribute
  (in-kernel L2-normalization of q/k — the KDA model's feature map) passes
  through to the kernel (without the in-kernel norm the caller owns the q/k
  conditioning). It declines `kda_bwd` (its backward
  kernel is a stub), so gradients route to the cuTile engine
  (`cudnn.linear_attention.cutile.kernels.kda_chunk_cutile`); the torch op
  is `cudnn.linear_attention.ops.kimi_delta_attention`.
- Gated DeltaNet v2 (`gdn2` / `gdn2_bwd`) has channel-wise gates — `g`/`beta`
  `[total_T, HO, K]` plus a NEW per-value write gate `w` `[total_T, HO, V]`.
  GDN-2 has **no** cuTile engine; `Gdn2FrostEngine`
  (`cudnn.linear_attention.frost.gdn2_engine`, SM100/SM103) is its only
  engine, passes the `use_qk_l2norm` attribute through to the kernel (like
  `KdaFrostEngine`), and declines `gdn2_bwd` (stub backward kernel), so the
  op (`cudnn.linear_attention.ops.gated_delta_net_v2`) is forward-only for
  now.
- The FROST engines are pure pass-through: `check_support` requires the
  kernel-native dtypes (fp32 gates — io-dtype `beta`/`w` for GDN-2 — int32
  `cu_seqlens`, fp32-or-bf16 state ports with matching initial/final dtypes,
  fp32 state gradients) and execute hands the caller's buffers straight to
  the kernels, carving any scratch it needs out of the explicit workspace as
  DLPack views. The cuTile engines follow the same buffer contract: outputs
  are written in place (the caller's output buffers, required in the
  kernel-native dtypes, are planted under the pipelines' terminal workspace
  names), and their chunk-index tables are built on device from
  `cu_seqlens`, so execution stays sync-free. Buffers only need
  `__cuda_array_interface__` or `__dlpack__`; the torch custom ops do the
  dtype normalization on their side.

### The manifest: classify, then let the engine decide

`engines/manifest.py` answers only what an engine cannot answer about itself
without being imported. Everything else is the engine's own `check_support()`.

- **A family is a KIND OF GRAPH** — roughly the backend's operation-graph mode,
  at a granularity of our choosing — not a group of engines that ship together.
  Every graph belongs to exactly one family or to none, so engines within a
  family compete and engines across families never do.
- **Classification is a lookup.** `_ANCHOR_NODE_TO_FAMILY` maps the node types
  that NAME a family; `family_for(graph)` is a function, so "two families
  claimed this graph" is not a case that can arise. A graph naming two (a
  matmul and an sdpa together) belongs to neither and goes to the backend.
- **The table holds anchors, not an envelope.** Node types absent from it —
  `POINTWISE`, `REDUCTION`, anything added tomorrow — are ignored when
  classifying, so `matmul + pointwise` is a gemm graph. Whether a family can
  serve the WHOLE graph is its analyzer's judgment. A coarser copy of that
  judgment here is what `closed_under` was, and it promised RESHAPE support
  nothing implemented.
- **`family_for(graph)` is a pure property of the graph** — no `sm`, no
  environment. What kind of graph something is cannot depend on which machine
  is asking. Availability is separate (`EngineFamily.offered_ids`).
- **What the manifest does NOT decide:** architecture. An engine's
  `Capabilities` declares an arch RANGE (`sm_lo`/`sm_hi`, `major*10 + minor`),
  because an sm100 kernel serves the sm100 LINE — enumerating the members that
  exist today silently declines the ones that ship later.
- **Maturity is per engine** (`EngineSlot.opt_in`, gated by
  `CUDNN_FRONTEND_ENABLE_FROST_ENGINES`), so one implementation can graduate
  while a sibling matures. It lives in the manifest rather than on the engine
  class because the gate must answer without importing the engine.
- **`register_backend()` is the out-of-tree escape hatch and nothing else.**
  In-tree engines are discovered; ids below `OUT_OF_TREE_ID_BASE` are rejected.

### Facts: one description per graph, shared

A family may name an `analyzer` — a `("module", "callable")` pair, kept as
strings so matching stays import-free. Planning resolves it and attaches the
record to the graph; engines read that record back rather than parsing again.

- **Attached after the freeze.** Planning runs `_finalize_backend_layout()` →
  `_freeze()` → `_attach_facts()`. Analyzing a graph that can still change
  means chasing every mutation point — the layout the backend infers, a dtype
  set between two `validate()` calls — and missing one leaves facts describing
  a graph that is gone. `_facts_for()` memoises only a frozen graph, so there
  is no invalidation rule to get wrong.
- **Keyed by the analyzer itself**, so the ranking (which resolves it from
  `EngineFamily.analyzer`) and the engine (which passes the callable it already
  imports) reach ONE record with no name to keep in sync. Two families MAY
  share an analyzer; SDPA forward and backward do.
- **Facts describe, capabilities judge.** `SdpaGraphFacts` records
  `has_bias=True` as a fact, never an error; each engine's `Capabilities` row
  does the rejecting in `mismatch()`. A shared parser that starts rejecting
  becomes an if-ladder that must know every kernel.
- **Framework-neutral vocabulary**: `cudnn.data_type`, not `torch.dtype`;
  device from `cudnn.create_device_properties()`, the backend's own descriptor.
  Facts are what every engine of a family reads, so expressing them in one
  framework's types would make dispatch require that framework.

### Handle, stream, and device

There is no ambient device state in dispatch. Three separate things:

- **Handle** — `execute(..., handle=h)` if given, else the graph's own. The
  handle is what carries the stream.
- **Stream** — `_resolve_stream(handle)` is `cudnn.get_stream(handle)`. With no
  handle it is `None`, and kernel wrappers then use the default stream. A
  failed query on a SUPPLIED handle raises rather than silently falling back to
  another stream: running on the wrong stream is a correctness bug, not a
  degradation. So it is a fallback for *no handle*, never a fallback for a
  handle whose stream could not be read.
- **Device** — the analyzer reads compute capability and SM count from
  `cudnn.create_device_properties()`, the same serialisable descriptor the C++
  deviceless-AoT path uses, rather than from `torch.cuda.current_device()`.
  Engines re-check the arch in `check_support()` regardless.

### Import boundaries

Deciding whether an engine COULD serve a graph must not cost the machinery that
would serve it — importing the CuTe DSL is ~1 s and 357 modules, and paying it
only to decline is why `closed_under` existed.

- Package `__init__`s under `cudnn/sdpa` are lazy (PEP 562) and
  `EngineSpec.lower` resolves its DSL adapter at build time, so
  `import cudnn.sdpa.graph_analyzer` costs 9 ms and 2 modules rather than
  1059 ms and 381.
- The graph API pulls no framework at all: describing and validating a graph
  imports neither torch nor cutlass.
- A missing or too-old DSL is a DECLINE at `check_support()`, probed without
  executing the module (`importlib.util.find_spec`, `importlib.metadata`).
  `CUTEDSL_MIN_VERSION` in `frost/buffers.py` is the floor; `pyproject`'s extra
  deliberately does NOT pin it, since that would make cudnn-frontend
  incompatible with anything holding the DSL back.
- `test/python/test_import_boundaries.py` holds all of this, in a fresh
  interpreter, measuring the delta against an empty one.

### Router and the one plan list

- `create_execution_plans()` collects both sides and ranks them into ONE list:
  the python engines of the graph's family that claim it (from
  `engines/manifest.py`; `register_backend()` adds out-of-tree ones) and the
  backend's own ranked
  `(engine_id, knobs)` recommendation from `backend_plan_entries()`.
  `engines/heuristics.py::heuristics_sort` decides the order.
- The list IS `graph.plans`, and the classic at-index APIs
  (`get_execution_plan_count()`, `get_plan_name_at_index()`,
  `build_plan_at_index()`, `execute_plan_at_index()`,
  `get_workspace_size_plan_at_index()`) address it, so code that loops over the
  plan count picks up python engines with no change.
- A backend entry carries the `cpp_index` it holds in the lowered graph's own
  plan list, so building it is one `build_plan_at_index`. Backend engine sets
  are still never statically enumerated: they are discovered per graph at plan
  time, and `backend_plan_entries()` returns `[]` when the backend declines the
  graph or is not installed — the backend participates in the ranking, it is
  not a hard dependency.
- A Router places the backend's entries by calling `backend_plan_entries()`
  (answered once per graph) and putting the result where it wants; nothing
  rewrites the list it returns, so a routed index means what the Router said.
  `BACKEND_HEURISTIC_ENGINE_ID` names one thing only: the delegating entry that
  method appends under `heur_mode.OPENSOURCE`, where the backend picks among
  candidates it never exposes as plans.
- `build_plans()` walks the list from the selected index and takes the first
  entry that builds; a decline advances to the next. `select_plan(i)` pins,
  and a pinned decline raises instead of running something else.
- Ranking policy is pluggable at three levels: subclass `Router`, per-graph
  `router=`, process-wide `default_router` — plus `heuristics_sort` itself,
  which is the seam a real cost model replaces.

## Key invariants

- **uid ownership**: the Python IR owns the whole uid namespace; every uid is
  pushed explicitly to C++ and a post-build assertion fails loudly on
  violation (C++ auto-assignment never runs for Python-built graphs — its
  enumeration order is nondeterministic for multi-output ops). A user uid
  landing on an auto-assigned one steals it (the holder is renumbered);
  user-user collisions raise.
- **Pure-python or pure-C++**: a graph routed to a python engine never
  touches C++ on the execute path; mixed construction is unsupported.
  (Explicitly querying the backend plan space lowers the backend entry on
  demand — that is the caller asking for the backend.)
- **One-shot planning**: a second `create_execution_plans()` raises (the
  classic C++ graph never supported re-planning — it appends engine configs
  by accident). Switch plans with `select_plan()`; plan differently by
  building a new graph.
- **Whole-surface freeze**: after lowering/planning, every public mutation
  path raises — op builders and fluent setters, direct attribute writes on
  `Tensor`/`Node`/`GraphContext`, dict writes on node ports/params
  (MappingProxy), in-place dim/stride edits (sealed to tuples). Inspection
  stays fully readable. A mutation in the mutable window after `validate()`
  invalidates the validation. Planning freezes BEFORE it analyses, so facts
  never describe a graph that can still change.
- **Output layout contract**: only USER-assigned output dim/stride are pushed
  to the lowered graph; IR-inferred strides are provisional (row-major) and
  the backend keeps its classic per-op layout inference (e.g. channels-last
  conv). A unified layout resolver across python/cuDNN candidates belongs to
  the heuristics follow-up.
- **Classic parity**: the public `cudnn.pygraph` surface behaves as before —
  `cudnnGraphNotSupportedError` at `validate()`, conditional outputs return
  `None`, torch dtypes/`torch.Size` accepted, ragged (THD) offsets and
  multipliers on outputs, serialize/deserialize passthrough, plan queries
  delegate to the lowered graph.

## Naming

- `cudnn.pygraph` — THE public graph class (Python IR), implemented in
  `cudnn/_pygraph.py`.
- `cudnn._pybind_module.backend_graph` — the internal C++ builder the IR
  lowers to (renamed from its pre-flip public name to avoid two things called
  `pygraph`).

## Testing the backend path

The `test_native_backend_lowering.py` suite builds graphs natively, lowers,
executes on GPU, and checks numerics against torch references. Dispatch-level
assertions (`selected_engine is None`, backend plans created, lowered graph
present) prove the execution went through the cuDNN backend plan path rather
than a python engine; kernel identity below the backend API is deliberately
not asserted (kernel names are backend-internal and version-dependent).

## Follow-ups (separate MRs)

- Heuristics/ranking: pluggable Router policy + typed plan representation.
- DSL engine integration (the cuTile matmul engine lives in this track).
- Structural cleanup: lifecycle state objects, a `CudnnBackendAdapter` to
  remove `selected_engine is None` branching, lowering extracted to its own
  module, op-identity dedup (NodeType vs registry keys), longer-term a typed
  `OpSpec` as the single per-op source for builder/validation/lowering.
