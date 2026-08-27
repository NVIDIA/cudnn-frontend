# Python-native `cudnn.pygraph` and pluggable execution backends

## What this is

`cudnn.pygraph` is a Python-native graph class: graph structure (nodes,
tensors, op parameters) lives in Python with full introspection, and execution
dispatches over python DSL engines and the cuDNN C++ backend alike. The C++
graph builder is internal (`cudnn._pybind_module.backend_graph`) and is reached
exclusively through lowering.

```text
cudnn.pygraph (Python IR)  →  create_execution_plans()  →  rank  →  ONE ranked plan list
  nodes / tensors / params        (freeze, analyze,         (the graph's    PlanConfig(engine_id,
  fully introspectable            query the backend)         family)        knobs) — python engines
                                                                            AND backend entries
```

Why: python-DSL engines (CuTe-DSL / cuTile style GEMM and attention fusions)
need to *see* the graph to decide whether and how to run it. Previously that
required monkey-patching the pybind class and recording calls; now the graph
is natively introspectable and an engine is one file implementing
`BaseEngine`.

**"Pluggable" means the library ships a table, not that callers hand engines
over at runtime.** There is no registration call: `engines/manifest.py` is the
only way a python engine exists. An engine handed over at runtime could not be
ranked anyway — it declares no `Capabilities`, so nothing could enumerate its
configs or place it against the backend.

### The dispatch tree

```text
create_execution_plans([heur_mode.A, ...])                    _pygraph.py
│
├─ validate()                    lowers + freezes any graph the backend CAN lower
├─ _finalize_backend_layout()    backend layout inference lands, or records a decline
├─ _freeze()                     whole public surface sealed
├─ _attach_facts()               family_for → resolve_analyzer → ONE parse, hung on the graph
│
└─ heuristics.rank(graph, _candidate_engines(), backend_plan_entries(), modes)
   │                 │                           │
   │                 │                           └─ _create_backend_plans(): one C++
   │                 │                              create_execution_plans PER MODE, spans
   │                 │                              recorded → every entry carries its mode
   │                 │                              (+ the untagged delegating entry)
   │                 └─ manifest.engines_for(graph) — the family's offered slots, nothing else
   │
   ├─ family_for(graph) → resolve_heuristics(family)
   │     declares none → _unranked: accepting engines, then the backend
   │
   └─ <family>.recommend(modes, facts, offered, backend_plans)
      │                                    e.g. sdpa/fwd/heuristics.py
      ├─ A  → per eligible cell: a measured rule (_sm120_tiles) names the config,
      │       runners-up behind it; a cell with one point per axis contributes one
      │       entry. Placed against the backend's A block by _MEASURED_BEHIND.
      ├─ FALLBACK → the config expected to build, + the backend's FALLBACK block
      ├─ OPENSOURCE → our candidates, then the delegating entry (see below)
      └─ dedup by (engine_id, knobs), first position wins

= graph.plans, position for position.  build_plans() walks it.
```

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

- `BaseEngine`: `check_support(graph)` (accept, or decline by raising),
  `build_plan(graph, plan, ctx) → CompiledPlan` (the expensive
  JIT step, once per graph/plan, cached on the graph),
  `CompiledPlan.execute(graph, operands, ExecutionContext)` with explicit
  handle/stream/workspace. Dynamic-shape overrides are a backend-path feature:
  a python plan is compiled for the shapes the graph declared, so `execute()`
  refuses them rather than silently running a different problem. Simple eager
  engines implement `execute()` only.

#### The variant pack is normalized once

`graph.execute()` converts whatever the caller passed — a torch tensor, a
`DeviceView`, any `__dlpack__` / `__cuda_array_interface__` producer, or a bare
device address — into a `VariantPack` at the top, and everything below reads that.
**Do not add a branch on what the caller's object is.** This exists because
there used to be two such branches: the backend path accepted a bare address
(`_native_var_pack`'s `if type(d) is int: return d`) while an engine got the
object untouched and `frost.buffers.probe` refused it. One public call, two
answers, decided by which plan the heuristics happened to pick — which the
caller does not control.

`VariantPack` carries the caller-filled uids ascending and the operands
themselves, held in a C type (`pygraph/variant_pack.cpp`) as one `DLTensor`
each. That type both consumes `__dlpack_c_exchange_api__` — the C function
table a producer publishes on its type, which is how one crossing reads the
whole pack — and implements it, so a kernel reads an operand through the same
fast path it has for a framework tensor. `address` is the pointer array
`_execute_with_raw_ptrs` takes.

The pack's vocabulary distinguishes the position from the thing at it:
`pack.index_of(tensor_or_uid)` gives an operand's POSITION, and
`pack.operands(indices)` turns positions into `OperandBuffer`s — one caller
buffer described (pointer, shape, stride, dtype), non-owning. Resolve positions
once; ask for buffers per call.

Each operand's OWN dim/stride/data_type is what the pack holds, deliberately
not the graph's declaration: the two may differ and one engine relies on it —
`frost_gemm` takes its M/N/K from the buffers, so a plan built for one problem
size runs another bit-exactly. **Read the IR port for the shape the plan was
built for; read the pack for the shape about to run.**

Two rules that are easy to break by accident:

- **The pointer array is per call.** Two threads may execute one graph
  concurrently with different buffers. A shared array hands each thread the
  other's pointers, and each pointer in it is individually valid, so the failure
  is a wrong number rather than a raise.
- **The operand order has exactly one source, never a union.** The lowered
  graph's variant-pack template when the graph has one — only C++ sees every
  user slot, since a tensor's `ragged_offset` is an operand but hangs off the
  `Tensor` rather than off a node port, and the slots the graph fills itself
  (pass-by-value scalars, slice replacement destinations, workspace
  modifications) must be excluded. The python IR only for the python-only ops
  that never lower. The two sides do not have to agree: each indexes the layout
  it was handed.

`is_virtual` on a python-only graph does not mean "the caller supplies
nothing" — it is a statement about the backend's lowering, and a gdn graph marks
its own `O` virtual while the caller passes a buffer for it. So the layout there
is every wired port, and an unfilled slot is an optional port the caller did not
request.

`CompiledPlan.takes_variant_pack` is the migration flag. An engine that has not
set it still receives the caller's `{uid: buffer}` map and reaches ports through
`resolve_node_buffers`; the flag and that function both go once the last engine
has moved. `execute()` builds the `Tensor`s only for a plan that sets the flag —
measured, normalizing for a plan that will not read the result costs more than
it saves.

#### What a per-execute path costs

An engine owns its internals, and this section does not change that. It exists
because the default outcome is expensive: an engine that re-derives its per-call
facts lands around **40 µs of host time per execute**, and for a single-kernel
op that is most of what the caller pays. The same kernel with those facts read
once is **20**. Both numbers are `frost_gemm` at 256×256×128 bf16, host
enqueue, min over 25 reps of a 64-call burst from a drained queue.

The budget it has to fit in, all measured on SM100:

| | µs |
|---|---|
| `cuLaunchKernelEx`, untraced | 1.85 |
| one CuTe-DSL entry | ~3.6 |
| `graph.execute()` entry + `_normalize` | ~8 |
| **everything else is the engine's** | |

Do not read a per-call cost out of an nsys trace: CUPTI adds ~2.2 µs per traced
API call, which is more than the call.

**Split the facts by when they are decided.** Operand roles and majors, packing
factors, alignment requirements, output shape rules, which outputs need a seed —
all fixed when the kernel compiled. M/N/K, strides and pointers arrive per call.
Read the first set into a table at build (`gemm/frost/recipe.py` is the worked
example) and let the call read the table. That alone is 44 → 35.

**Then lower the table into one closure per plan**, with its constants captured
and the operand structure flattened into the loop headers, so the call does no
attribute lookup and takes no branch the build already settled. That is 35 → 20.
Two rules make it safe:

- **The lowered path never raises, and it is the only path that runs.** What it
  refuses it hands to a checker that reads the same table, names the rule and
  raises — it launches nothing. A graph the closure cannot serve at all is
  declined when the engine is asked to support it, so it goes to the backend
  rather than to a second executor.
- **So a refusal is the answer, not a slower route.** The set of calls the
  closure refuses should equal the set of illegal calls; a legal call it will
  not serve is a bug. Keeping a reference executor instead would buy a
  differential that catches divergence but never a misconception the two share —
  which is exactly how an axis-order bug survived one here. The tests that
  matter are against intended semantics and against the BACKEND, at the shapes
  where two encodings coincide.

**A loop over a flat table gets almost all of it, so do not hand-unroll per
flavor.** Measured three ways on the same plan and buffers: interpreting the
table 35.8, looping over it flattened 19.7, a hand-written straight line with the
structure unrolled 17.5. The loop is worth 45%; unrolling adds 12% and costs one
closure body per operand shape — six flavors, six bodies to keep in agreement.
One loop over `arg_plan` (the launch argument order as data) serves aux, extra
outputs, multi-GEMM and block scale at 22 µs each, down from 39–50. Source
codegen off the same table is how to buy the last 12% back later, for every
flavor at once rather than for the one that was worth hand-writing.

This is a pattern to copy, not a framework to import. Sharing the code across
engines would couple their kernels' ABIs, which is the thing engine autonomy
buys; sharing the shape of the solution costs nothing.

**Costs that are easy to miss, each measured:**

- A `from x import y` inside a per-call function: **1.1 µs**. It was 65% of
  what `_check_plan_device` cost.
- `torch.Tensor.permute()`: **1.4 µs** per call, per operand.
- Rebuilding a `{id(tensor): buffer}` map to look operands back up, when the
  operand order was settled at build and a list index would do.
- Recomputing a pure function of values every call. `tensor_alignment`'s
  layout half is **1.5 µs** and memoizes on `(shape, stride, elem_bytes)` —
  values, so there is nothing to invalidate; only the pointer half is per call.
- Reading an operand through the exchange vtable is **0.08 µs** against 1.5 for
  the python attribute walk. Framework neutrality is not what costs.

**Measure from a drained queue, and sweep the burst size.** A number that is
flat in the burst size is host-bound; one that climbs with it is the device
rate, and back-to-back timing reads the device rate whenever host and device
are close.
- **An engine does not propose its own plans.** Which configs to try, in what
  order, and where the backend's entries belong is one comparison across every
  candidate, and no engine can make it from the inside — it sees neither its
  siblings nor the backend. That decision lives in `engines/heuristics.py` and
  the per-family hook it dispatches to (see *Ranking and the one plan list*).
- Every engine has a stable `engine_id` in one flat id space
  (`engines/engine_ids.py`: backend `[0, 10_000)`, C++ OSS `[10_000, 20_000)`,
  python `[20_000, …)` with a `FAMILY_BLOCK`-wide block per family) —
  reproducible pinning/autotune. Engines do not DECLARE their id: the manifest
  holds every slot and `instantiate()` hands each factory the ids its engines
  are to use, so an engine cannot claim a number it was not given, the whole
  space is readable in one file, and any id decodes back to a family and a slot
  (`manifest.engine_for_id`) with nothing registered first — which is what lets
  `create_execution_plan(engine_id, knobs)` replay an autotune result.
- An engine declines a graph ONLY via `NotImplementedError`,
  `cudnn.cudnnGraphNotSupportedError`, or `ImportError`; anything else is an
  engine bug and propagates. `ImportError` counts because lowering imports are
  deferred past `check_support()` (see *Import boundaries*), so a missing
  optional dependency can only surface at build time — without it, a host
  lacking the `cutedsl` extra would lose graphs the backend could have served.
- The contract is proven end to end without a GPU in
  `test/python/test_dispatch.py`, with stand-in engines injected through the
  manifest — the same path production uses. Those engines do no arithmetic:
  what dispatch is responsible for is reaching the engine and resolving the
  caller's buffers, and checking a result against `torch.matmul` would put a
  torch reference implementation of matmul inside a dispatch test.
- `GdnCuTileEngine` executes the single-node `gdn` and `gdn_bwd` ops (Gated
  DeltaNet linear attention) via the cuTile chunked kernels. Both ops are
  THD-only: token-packed `[total_T, heads, dim]` tensors with a required
  `cu_seqlens` (a dense batch is `[0, T, 2T, ...]`). `gdn_bwd` takes the
  forward inputs plus `dO` (and optionally `d_final_state`) and produces
  `dQ/dK/dV/dG/dBeta` (+ `d_initial_state` iff `initial_state` is given,
  + `d_a_log`/`d_dt_bias` iff the node carries `safe_gate` — the gate
  transform's parameter gradients, with `dG`/`dBeta` then in raw-logit
  space under `safe_gate`/`use_beta_sigmoid`);
  the cumulative gate and intra-chunk WY matrix are recomputed inside the
  engine, so the graph contract carries no forward intermediates. Both are
  python-engine-only ops: they have no cuDNN backend lowering, so routing
  them to the backend entry raises `cudnnGraphNotSupportedError` at
  lowering. The kernels live in `cudnn.linear_attention.cutile.kernels.gdn`;
  the torch custom op `cudnn.linear_attention.ops.gated_delta_net` is a thin
  adapter that builds and executes cached `gdn`/`gdn_bwd` graphs (the SDPA
  op pattern), so it inherits whatever engine the planner selects. The
  optional `use_qk_l2norm` attribute asks the engine to L2-normalize the q/k
  rows; `GdnFrostEngine` (the SM100/SM103 default, serving both `gdn` and
  `gdn_bwd` on the FROST chunked kernels) serves it through a workspace
  helper kernel (normalized q/k copies + saved inverse norms, with the
  backward Jacobian projection applied in place after the head-group fold),
  and likewise serves `safe_gate` (in-kernel raw-logit gate transform, with
  `d_a_log`/`d_dt_bias` produced by a deterministic reduction helper) and
  `use_beta_sigmoid`; the cuTile engine remains the fallback for non-128
  head dims.
- `KdaFrostEngine` / `KdaCuTileEngine` do the same for the single-node
  `kda` / `kda_bwd` ops (Kimi Delta Attention). KDA is GDN with a
  per-key-channel decay: its `g` is the log-space vector gate
  `[total_T, HV, K]` (GDN's is the scalar `[total_T, HV]`); `beta` stays
  scalar. The FROST engine (`cudnn.linear_attention.frost.kda_engine`) is
  the forward default on SM100/SM103; the node's `use_qk_l2norm` attribute
  (in-kernel L2-normalization of q/k — the KDA model's feature map) passes
  through to the kernel (without the in-kernel norm the caller owns the q/k
  conditioning). It serves `kda_bwd` on the FROST backward kernel,
  regenerating the per-chunk state checkpoints with a recompute pass when
  the graph does not provide them; the cuTile engine
  (`cudnn.linear_attention.cutile.kernels.kda`) is the
  fallback slot. The torch op is
  `cudnn.linear_attention.ops.kimi_delta_attention`.
- Gated DeltaNet v2 (`gdn2` / `gdn2_bwd`) has channel-wise gates — `g`/`beta`
  `[total_T, HO, K]` plus a NEW per-value write gate `w` `[total_T, HO, V]`.
  GDN-2 has **no** cuTile engine; `Gdn2FrostEngine`
  (`cudnn.linear_attention.frost.gdn2_engine`, SM100/SM103) is its only
  engine, passes the `use_qk_l2norm` attribute through to the kernel (like
  `KdaFrostEngine`), and serves `gdn2_bwd` the same way (checkpoint
  recompute when the series is absent); the op is
  `cudnn.linear_attention.ops.gated_delta_net_v2`.
- The FROST engines are pure pass-through: `check_support` requires the
  kernel-native dtypes (fp32 gates — io-dtype `beta`/`w` for GDN-2 — int32
  or int64 `cu_seqlens`, fp32-or-bf16 state ports with matching
  initial/final dtypes, fp32 state gradients for GDN/KDA and io-dtype
  `dBeta`/`dW` for GDN-2) and execute hands the caller's buffers straight to
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
- **A family may name a `heuristics` hook** — like `analyzer`, a
  `("module", "callable")` pair kept as strings so the coarse key stays
  import-free. It is handed the facts, the family's offered ids and the
  backend's entries, and what it returns IS the plan list.
- **There is no registration call.** The manifest is the only way a python
  engine exists, and `_candidate_engines()` is the graph's family and nothing
  else. An engine handed over at runtime could never be ranked anyway: it
  declares no `Capabilities`, so nothing can enumerate its configs or place it
  against the backend — it was an entry point into the plan list, not into the
  decision. Tests inject their fakes as a manifest family, so they reach
  dispatch the way real engines do.

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

### Ranking and the one plan list

- `create_execution_plans()` gathers the inputs — the parsed facts, the family's
  offered ids, and the backend's own `(engine_id, knobs)` recommendation from
  `backend_plan_entries()` — and hands all of it to the graph's family in ONE
  call (`engines/heuristics.py::rank` → the family's `recommend`). What comes
  back IS the plan list, position for position. There is no second merge step:
  splitting the decision is what forced the previous design to concatenate the
  two sides and call it ranking.
- **The backend's entries arrive tagged with the mode that produced them.**
  `_create_backend_plans()` asks C++ one heuristic mode at a time and records
  `get_execution_plan_count()` after each, so a family can say "the backend's
  mode-A entries ahead of ours, its fallbacks behind". C++ appends each query
  to the same plan list, which is exactly what makes the boundaries readable —
  no C++ change was needed.
- **`heur_mode.OPENSOURCE` is mode A without the backend's recommendation**:
  the python engines ARE the open-source implementation. Combine it to measure
  coverage — `[OPENSOURCE, A, FALLBACK]` tries every python config first and
  still has the backend behind it, so a graph that runs on a backend plan is
  one no python engine covers.
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
- **A plan's identity is `(engine_id, knobs)`, never its `cpp_index`.** The
  index only says where one backend query happened to put it, so deduplicating
  on it lets `[A, A]`, or one config both modes return, through twice — and an
  autotuner would build and time the same config twice.
- **Whether a mode SUCCEEDED is tracked per call, not read off the plan
  spans.** An OPENSOURCE query registers a C++ OSS candidate without adding a
  plan, so it contributes no span; judging by spans would rethrow a later
  mode's failure and discard the delegate that successful query earned.
- The family places the backend's entries wherever it wants; nothing rewrites
  the list it returns, so a ranked index means what the heuristics said.
  `BACKEND_HEURISTIC_ENGINE_ID` names one thing only: the delegating entry
  `backend_plan_entries()` appends, where the backend picks among OSS
  candidates it never exposes as plans and which therefore cannot be
  enumerated. It carries no mode. It leads the BACKEND's entries, because
  `Graph::build_plans` tries it before its own engine configs — but it does NOT
  lead the family's, because it is not a pure OSS entry: if the C++ OSS engine
  declines, that same call falls through to the native configs already
  enqueued. Ahead of the family's OPENSOURCE block it would answer an
  OSS-coverage question with a native kernel.
- `build_plans()` walks the list from the selected index and takes the first
  entry that builds; a decline advances to the next. `select_plan(i)` pins,
  and a pinned decline raises instead of running something else.
- Ranking policy has ONE home: the family's `heuristics` hook, replaceable per
  family. Which side leads is meant to be a measurement — a cell timed slower
  than the backend follows it — not a default. Any cell that has not been timed
  keeps the historical order, and the code says so where it is written.
- **A recommendation always names a concrete config.** A knob field is `None`
  only where the capability row declares no domain for that axis. `None` never
  means "engine, pick for me" — that reading is what let the same choice be
  made in two places, once in the ranking and once in the DSL adapter, and
  drift. `sdpa/fwd/heuristics.py::_sm120_tiles` is the worked example of a
  rule: it reads facts only, names its choice first, and puts the rest of the
  domain behind it for a caller that autotunes. To add one — write the
  function, list the cell in `_TILE_RULE_CELLS`, and put the measurement in the
  commit.

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
  stays fully readable. `validate()` lowers and freezes any graph the backend
  CAN lower (classic error timing), so the mutable-after-validate window is
  exactly the ops with no backend node (GDN/KDA/…); a mutation in it
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

### What each machine actually covers

A dispatch change needs three runs, because the suites SKIP rather than fail on
the wrong arch — a green sweep on one box says nothing about the others:

| target | covers |
|---|---|
| any GPU (CPU-only logic) | `test_dispatch.py`, `test_graph_native.py`, `test_import_boundaries.py` — the ranking contract, one-shot planning, the at-index APIs, manifest classification |
| SM100 | every FROST SDPA-forward cell except sm120; the FROST GEMM family; `linear_attention` (GDN / KDA / GDN-2), which is where the cuTile-vs-FROST pin lives |
| SM120 | `sdpa_fwd_prefill_sm120` and its tile rule; `test_mhas_v2` routing tallies |

Enumerate the device with `torch.cuda.get_device_properties(i).major` and pin
it with `CUDA_VISIBLE_DEVICES` — CUDA's device order is not `nvidia-smi`'s, and
defaulting to device 0 is how an SM100 suite silently skips in full.

## Follow-ups (separate MRs)

- More per-family tuning rules on top of the ranking frame, each with the
  measurements behind it. `sdpa/fwd/heuristics.py::_sm120_tiles` and
  `_pack_gqa_wins` are the two today; every other cell falls back to its
  capability row's sole point per axis, which is the honest answer while
  nobody has timed it.
- FALLBACK is one config per cell today — the smallest tile the row admits, the
  config that asks least of the device. Picking the handful that between them
  cover the plane needs measurements; the TODO is in `_mode_fallback`.
- `_MEASURED_BEHIND` is an empty set: which side leads is meant to be a
  measurement, and an untimed cell keeps the order this dispatch has always
  had. A cost model that can compare a python config against a cuDNN engine on
  a common currency (predicted time) turns that set into a number.
- DSL engine integration (the cuTile matmul engine lives in this track).
- Structural cleanup: lifecycle state objects, a `CudnnBackendAdapter` to
  remove `selected_engine is None` branching, lowering extracted to its own
  module, op-identity dedup (NodeType vs registry keys), longer-term a typed
  `OpSpec` as the single per-op source for builder/validation/lowering.
