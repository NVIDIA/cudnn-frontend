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
  `CompiledPlan.execute(graph, tensor_data, ExecutionContext)` with explicit
  handle/stream/workspace/overrides. Simple eager engines implement
  `execute()` only.
- Every engine owns a stable `engine_id` in a reserved region
  (`PYTHON_ENGINE_ID_BASE = 1 << 20`) — reproducible pinning/autotune.
- An engine declines a graph ONLY via `NotImplementedError` or
  `cudnn.cudnnGraphNotSupportedError`; anything else is an engine bug and
  propagates.
- `ReferenceMatmulEngine` (pure PyTorch) is the in-tree contract oracle; real
  DSL engines land as separate PRs, one file each.

### Router and the two plan-index spaces

- The Router returns the routed plan list: python `PlanConfig` entries plus
  AT MOST ONE backend delegating entry (`BACKEND_HEURISTIC_ENGINE_ID`). The
  final output is validated regardless of Router implementation (registered
  ids only, one sentinel max, never empty).
- **Routed space**: `graph.plans`, selected with `select_plan()`. Indices are
  stable — the backend entry is one index forever and never expands in place.
- **Backend space**: the cuDNN backend's own plans, discovered per graph from
  the lowered graph and addressed via the classic
  `get_execution_plan_count()` / `*_plan_at_index()` APIs (pure delegation).
  The frontend never statically enumerates backend engines — backend engine
  sets vary by version and are discovered at plan time.
- Concrete backend engine configs as first-class routed entries need a typed
  plan representation — heuristics/autotune follow-up scope, together with
  ranking policy (the Router is pluggable at three levels: subclass,
  per-graph `router=`, process-wide `default_router`).

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
  invalidates the validation.
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
