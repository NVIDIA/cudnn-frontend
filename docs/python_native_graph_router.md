# Python-native Graph + Backend Router

A backend-agnostic Python graph IR with a first-class **Router** that dispatches
execution to interchangeable backends (a native DSL engine, or the cuDNN Graph
backend). This is a concrete implementation of the *Python API Engine and Graph
API Unification Proposal* (Frontend v1 sync-up).

```
Python Graph API  ->  create_execution_plans()  ->  Router  ->  Selected backend
 (build ops, no                (route here,                     (python DSLs /
  backend commit)              lazy lowering)                    reference / cuDNN Graph)
```

## Why

Two prior efforts converged on the same need — a Python-visible graph an engine
can consume:

- A **Python-native graph IR** (`Node`/`Tensor`/`NativeGraph`) that keeps all
  structure in Python (full introspection, no C++ round-trip to inspect).
- A **native DSL fusion engine** that today reconstructs the graph by
  monkey-patching `cudnn.pygraph` and recording op calls into side tables —
  fragile, import-order sensitive, `id()`-keyed.

The recorder exists only because pybind's `cudnn.pygraph` doesn't expose its
structure to Python. Once the IR is the source of truth, the recorder is
deleted and every backend consumes `graph.nodes` directly.

## Layers (kept separate on purpose)

1. **Graph IR** — `graph_types.Tensor`, `nodes.Node`, `graph_native.NativeGraph`.
   Engine-agnostic op DAG with dim/stride/dtype/reordering and per-op params.
   The shared contract for *all* backends.
2. **Backend contract** — `engines.BaseEngine`: `check_support()` / `execute()`
   / `get_workspace_size()`, plus a stable `engine_id`. What every python engine
   implements.
3. **Router** — `engines.Router`: at `create_execution_plans()` time, builds the
   ranked **plan list** (see below).

A backend's own *lowered IR* (e.g. a GEMM engine's fusion spec) is **private to
that backend** — it lowers from `graph.nodes` internally. Simple backends (see
`ReferenceMatmulEngine`) consume `graph.nodes` directly with no lowered IR.

## One flat engine-id space (cuDNN is not one engine)

cuDNN's backend is not a single engine — it's a namespace of engine-configs
(small ids `0..N`, each with knobs). Python engines join that **same flat id
space** in a reserved high region (`engine_ids.PYTHON_ENGINE_ID_BASE`, `1<<20`),
each declaring a **stable** `engine_id` it owns (so ids don't shift with
registration order — autotune results and pinned plans stay reproducible).

A heuristics query therefore returns one flat ranked list of
`PlanConfig(engine_id, knobs)` mixing both, e.g. `[(1048576, knobs), (1, knobs),
(5, knobs), (1048577, knobs), (19, knobs)]`. Dispatch is a single predicate on
the id — `is_python_engine(engine_id)` → run via the python registry; otherwise
lower to the cuDNN C++ backend. There is **no** "cuDNN as one BaseEngine" wrapper
and no `if native else cpp` fork: one plan list, one id-keyed dispatch.

Rule of thumb: distinct algorithm → distinct `engine_id`; tuning within an
algorithm → knobs.

## Routing at plan-creation time

Per the proposal (and Anerudhan's feedback), plan selection happens at
`create_execution_plans()`, **not** at graph construction:

- `build_operation_graph()` is backend-agnostic (validate only, no lowering).
- `create_execution_plans()` runs the Router → `self._plans` (the ranked list).
  Nothing is lowered here; a plan is built lazily when selected.
- `get_execution_plan_count()` / `select_plan(i)` expose the list for autotune.
- `check_support()` / `build_plans()` / `get_workspace_size()` / `execute()`
  dispatch on the selected plan's id: python engine, else lower to cuDNN.

### Phasing of the plan list

This PR builds the list as **supporting python engines (by `engine_id`) + one
trailing cuDNN entry** (`CUDNN_HEURISTIC_ENGINE_ID`, "let cuDNN heuristics
pick"). That concat is a placeholder: it is later replaced by reading the true
per-engine cuDNN configs via `get_engine_and_knobs_at_index()` and a real
heuristics-driven ranking merge — at which point the list literally contains
`eng=1, eng=5, eng=19` interleaved with the python ids. 2163 already prototyped
the mixed-list idea: its `heur_mode.TBD` sentinel lives in the same list as
`heur_mode.A`.

## Front door: `cudnn.pygraph` is engine-aware in place

Users don't switch classes. `__init__.py` augments the pybind `cudnn.pygraph`
**in place** (`pygraph_engines.install`) — the same sanctioned mechanism it
already uses (`pygraph.execute = _execute`). So `g = cudnn.pygraph(...)` is
unchanged for every existing sample, yet transparently routes to a registered
python engine:

```python
import cudnn
from cudnn.engines import ReferenceMatmulEngine

g = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
A = g.tensor(dim=[M, K], stride=[K, 1], data_type=cudnn.data_type.FLOAT)
B = g.tensor(dim=[K, N], stride=[N, 1], data_type=cudnn.data_type.FLOAT)
C = g.matmul(A, B); C.set_output(True)
g.register_backend(ReferenceMatmulEngine())   # opt-in today; a global registry gives it for free
g.execute({A: a, B: b, C: c})                 # routed to the python engine
```

How it stays safe:
- A per-graph mirror (WeakKeyDictionary, since pybind instances reject arbitrary
  attrs) records a `Node`/`Tensor` IR alongside the real C++ calls for a curated
  *represented* set (matmul + common pointwise, mirrored via the `NativeGraph`
  builders so the recorded op is exactly what engines consume).
- Every other op-builder is auto-wrapped to flag the graph **opaque** — the safe
  direction: it only *disables* the python path, never changes classic output.
- The lifecycle routes to a python engine iff one is registered AND the whole
  graph is represented AND it supports the graph; otherwise it delegates to the
  untouched C++ path (verified byte-identical: a classic matmul runs the same
  with and without the augmentation).

Current form is **eager** (the C++ graph is still built as ops are added).
Lazy / pure-python (never touch cuDNN) is the follow-up — it needs a structured
builder per op, because multi-tensor-return ops (sdpa, norms) can't be mirrored
generically. Coverage grows op-by-op; a fully-represented graph then skips C++.

`NativeGraph` remains as the equivalent standalone/greenfield authoring object
(`g = NativeGraph(); g.register_backend(...)`) sharing the same IR + engines.

## Scope of this PR (foundation only)

Included: the IR, `BaseEngine`, `Router`, the CPU `ReferenceMatmulEngine`
(CI-testable oracle), the optional `MatmulCuTileEngine`, and node builders for
block-scale / MoE / reduction so a fusion backend can represent them.

Also included: the in-place `cudnn.pygraph` front-door (`pygraph_engines`) for
matmul + common pointwise.

Deferred (follow-up MRs):

- **Lazy / pure-python**: structured builders per op so a fully-represented
  graph never builds the C++ graph (current front-door is eager).
- **Widen the represented set** on `cudnn.pygraph` (sdpa, block-scale, MoE,
  reduction) so more graphs are engine-eligible; grows op-by-op.
- **Global backend registry** so engines apply with zero `register_backend`
  call (fully transparent benefit).
- **DSL fusion backend** (e.g. the CuTe GEMM engine) ported to consume
  `graph.nodes` and registered as a `BaseEngine`.
- **Attention / other DSL backends**.
- **cuDNN lowering** (`_lower_to_cpp`) for the block-scale / MoE / reduction node
  types (today they are backend-path ops only).
- **Cost/benchmark-driven Router** ranking (and interleaving the true per-engine
  cuDNN configs) beyond the current python-engines-then-cuDNN concat.

## Open question (from the proposal)

Direct backend-invocation paths (wrapper APIs, custom PyTorch extensions calling
a backend directly) bypass the graph abstraction and are not yet unified with
the routing model.
