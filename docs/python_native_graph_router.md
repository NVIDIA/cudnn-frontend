# Python-native Graph + Backend Router

A backend-agnostic Python graph IR with a first-class **Router** that dispatches
execution to interchangeable backends (a native DSL engine, or the cuDNN Graph
backend). This is a concrete implementation of the *Python API Engine and Graph
API Unification Proposal* (Frontend v1 sync-up).

```
Python Graph API  ->  create_execution_plans()  ->  Router  ->  Selected backend
 (build ops, no                (route here,                     (QDSL / CTM / Triton /
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
   / `get_workspace_size()` + a `priority`. What every backend implements.
3. **Router** — `engines.Router`: at `create_execution_plans()` time, picks the
   first registered backend whose `check_support()` accepts the graph (ascending
   priority); `None` ⇒ fall back to the cuDNN Graph backend via lazy lowering.

A backend's own *lowered IR* (e.g. a GEMM engine's fusion spec) is **private to
that backend** — it lowers from `graph.nodes` internally. Simple backends (see
`ReferenceMatmulEngine`) consume `graph.nodes` directly with no lowered IR.

## Routing at plan-creation time

Per the proposal (and Anerudhan's feedback), backend selection happens at
`create_execution_plans()`, **not** at graph construction:

- `build_operation_graph()` is now backend-agnostic (validate only, no lowering).
- `create_execution_plans()` runs the Router, then lowers to cuDNN *only if* the
  cuDNN path was chosen.
- `check_support()` / `build_plans()` / `get_workspace_size()` / `execute()`
  dispatch on the selected backend (`None` ⇒ cuDNN).

## Usage

```python
import cudnn
from cudnn import NativeGraph
from cudnn.engines import ReferenceMatmulEngine

g = NativeGraph()
g.register_backend(ReferenceMatmulEngine())     # add candidate backend(s)
C = g.matmul(a, b)                              # torch tensors auto-bound
g.execute({C: c})                              # Router picks a backend; else cuDNN
assert g.selected_engine.name == "reference_matmul"
```

No registered backend ⇒ the classic cuDNN path is used transparently.

## Scope of this PR (foundation only)

Included: the IR, `BaseEngine`, `Router`, the CPU `ReferenceMatmulEngine`
(CI-testable oracle), the optional `MatmulCuTileEngine`, and node builders for
block-scale / MoE / reduction so a fusion backend can represent them.

Deferred (follow-up MRs):

- **`NativeGraph.from_pygraph()`** — populate the IR from an existing
  `cudnn.pygraph` (interim: reuse the op-recording hook to emit `Node`/`Tensor`;
  long-term: a C++/pybind reflection API). Currently raises `NotImplementedError`.
- **DSL fusion backend** (e.g. the CuTe GEMM engine) ported to consume
  `graph.nodes` and registered as a `BaseEngine`.
- **Attention / other DSL backends**.
- **cuDNN lowering** (`_lower_to_cpp`) for the block-scale / MoE / reduction node
  types (today they are backend-path ops only).
- **Cost/benchmark-driven Router** policy beyond first-supporting.

## Open question (from the proposal)

Direct backend-invocation paths (wrapper APIs, custom PyTorch extensions calling
a backend directly) bypass the graph abstraction and are not yet unified with
the routing model.
