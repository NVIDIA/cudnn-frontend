# FROST engines -- design and contributor guide

FROST engines are pure-Python execution engines for `cudnn.pygraph` graphs,
JIT-compiled with the CuTe DSL. They are not a side channel: the library ships
a static table of the engines it was built with
(`python/cudnn/engines/manifest.py`), and `create_execution_plans()` ranks
every engine that claims the graph into the SAME plan list as the cuDNN
backend's own plans. One list, one index space, no registration call, no
monkey-patching of the graph class.

While an engine matures its manifest row is marked `opt_in=True`, so it is
offered only with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`. That flag is a
per-engine maturity gate, not an architecture switch: an engine graduates by
flipping one field once it has the arch coverage and the benchmarks to justify
serving graphs unasked. Engines that are the only implementation of their
operation (GDN/KDA/GDN2 -- the backend has no lowering for those nodes at all)
are never gated.

This document is the contract for the FROST side of that mechanism. If you are
an agent or a human adding an engine, a kernel, a knob, or an op: read "The
rules" at the bottom first, then the section for the layer you are touching.
`docs/python_graph_and_execution_backends.md` covers the graph IR and the
backend contract from the frontend's side; this file covers what a FROST
engine owes it.


## The lifecycle, end to end

Nothing here is FROST-specific API -- it is the ordinary `cudnn.pygraph`
sequence, and the FROST engine is one entry of the list it produces:

```python
import torch
import cudnn

b, h, s, d = 2, 8, 256, 128
dims = (b, h, s, d)
strides = (s * h * d, d, h * d, 1)  # BSHD-physical

g = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)
q = g.tensor(dim=dims, stride=strides, name="q")
k = g.tensor(dim=dims, stride=strides, name="k")
v = g.tensor(dim=dims, stride=strides, name="v")
o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=d**-0.5, is_inference=True, use_causal_mask=True)
o.set_output(True).set_dim(dims).set_stride(strides)

g.validate()
g.build_operation_graph()
g.create_execution_plans([cudnn.heur_mode.A])   # ONE ranked list: backend + python

names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
# ['eng10_...', 'eng8_...', 'sdpa_fwd_prefill_sm100_d128', '..._d256', '..._d512']

g.select_plan(names.index("sdpa_fwd_prefill_sm100_d128"))   # optional: pin a FROST entry
g.check_support()
g.build_plans()                     # the JIT compile happens here
g.selected_engine                   # FrostSdpaFwdEngine(name='sdpa_fwd_prefill_sm100_d128', engine_id=20300)

q_gpu = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)
k_gpu = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)
v_gpu = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)
o_gpu = torch.empty(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)

ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")   # 16384 here
g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
```

Reading that sequence line by line:

- **`create_execution_plans()` is the dispatch stage.** It hands the graph's
  family, its engine ids and the backend's own recommendation to
  `engines.heuristics.rank`, which returns ONE list (`graph.plans`, a
  `list[PlanConfig(engine_id, knobs)]`). Planning is one-shot: a second call
  raises, because the classic C++ graph never supported re-planning. To plan
  differently, build a new graph.
- **Which plan runs is read through the classic API.** `graph.plans`,
  `get_execution_plan_count()` and `get_plan_name_at_index(i)` all address that
  one list, and so do `build_plan_at_index` / `execute_plan_at_index` /
  `get_workspace_size_plan_at_index`. There is no FROST-specific query and no
  prefix: a python plan is an entry like any other, at whatever position the
  ranking gave it. `graph.selected_engine` is the `BaseEngine` object behind
  the selected entry, or `None` when a backend plan is selected.
- **`select_plan(i)` is optional and strict.** Without it, the walk starts at
  index 0, at whatever the family's `recommend()` put first. With it,
  `build_plans()` starts at `i` and a decline there raises instead of quietly
  running something else.
- **`build_plans()` walks the list.** It tries the selected entry, and on a
  decline (`NotImplementedError` / `cudnn.cudnnGraphNotSupportedError`, logged
  at info level) moves to the next one. Any other exception is a bug in that
  engine and propagates rather than costing the user a kernel silently.
- **`deselect_engines(names)`** is the classic C++ API and still means what it
  always meant: bar every plan whose name contains one of these substrings.
  It now applies to the whole ranked list -- a python engine name bars that
  engine exactly as a backend engine name bars a backend plan -- and it is
  still forwarded to the lowered C++ graph.
- **`get_workspace_size()` is honest.** For a python plan it returns
  `CompiledPlan.get_workspace_size()`, the executor's real requirement (for
  the dense graph above: 0 — every SM100/SM120 adapter is `lse_optional`, so
  a stats-less inference graph compiles the LSE store out and binds no dummy
  buffer at any level). `execute()` forwards the caller's
  buffer through `ExecutionContext.workspace` and the executor carves its
  scratch out of it in 128-byte-aligned chunks, never touching bytes at or
  beyond the reported size -- no hidden per-execute allocation, stable
  pointers, CUDA-graph friendly. A missing or undersized workspace raises with
  the required size in the message; executors that report 0 never touch the
  buffer.

One documented exception to that accounting: `dense_flex` layout
normalization. A graph whose Q/K/V/O tensors are not BSHD-compact is served by
normalizing them to the kernel's compact-BSHD storage -- gather copies for the
inputs and a scatter-back scratch for O -- and those staging temporaries are
torch-allocated per execute, NOT drawn from the reported workspace.
BSHD-compact callers are zero-copy and unaffected.


## How an engine joins the graph API

By being a row in `python/cudnn/engines/manifest.py`. That is the whole
mechanism:

```python
EngineFamily(
    FROST_SDPA_FWD_ID_BASE, "frost_sdpa_fwd",
    "cudnn.sdpa.fwd.engine", "FrostSdpaFwdEngines",
    slots={"sdpa_fwd_prefill_sm100_d128": EngineSlot(0, opt_in=True), ...},
    analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    heuristics=("cudnn.sdpa.fwd.heuristics", "recommend"),
),
```

- A family is **pure data**: strings and ints, zero imports of engine code.
  `import cudnn` must never pay the CuTe-DSL import (~1.2 s) merely to know an
  engine exists. `analyzer` and `heuristics` are `(module, callable)` pairs for
  the same reason, resolved only when something needs to rank.
- **A family is a KIND OF GRAPH**, not a group of engines that ship together.
  `_ANCHOR_NODE_TO_FAMILY` maps a node type to the one family that serves that
  kind of graph, so a graph belongs to exactly one family or to none, and
  engines across families never compete. Node types absent from that table
  (POINTWISE, REDUCTION) are ignored when classifying, which is what makes
  `matmul + relu` a gemm graph.
- `slots` is the single source of every python engine id. `instantiate()` HANDS
  each engine the id its slot assigns; engines declare none of their own, so an
  engine cannot claim a number the manifest did not give it, and an id always
  decodes back to a family and a slot (`engine_for_id`). A shipped slot is fixed
  forever -- append the next free one, never reorder or reuse.
- `EngineSlot(opt_in=True)` withholds an engine until it matures. Per engine,
  not per family, so one implementation can graduate while a sibling waits, and
  the gate answers without importing anything.
- An unimportable engine is an ABSENT engine, not an error: `instantiate()`
  logs at info level and returns `[]`, so a missing optional dependency can
  never break planning for everyone else.
- **There is no registration call.** The manifest is the only way a python
  engine exists. An engine handed over at runtime could never be ranked anyway:
  it declares no `Capabilities`, so nothing can enumerate its configs or place
  it against the backend -- it was an entry point into the plan list, not into
  the decision.

### The coarse key is a NECESSARY condition, never the verdict

Candidate selection is two-stage. Stage 1 is the family's node-type key,
matched in microseconds with nothing imported -- a GEMM family is not imported
for an SDPA graph. Stage 2 is the engine's own `check_support()`, reached only
now, and that is the only thing that decides. Stage 1 deliberately does NOT
describe what an engine can do; a coarser copy of that judgment here would be a
second thing to maintain and a place to lie, which is exactly what
`closed_under` was before it was deleted (it lied about RESHAPE).

Keeping stage 1 a FILTER is the same argument this document makes below
against a central opset enum (see "Opsets"), and it is the reason the manifest
is not an authoritative pattern matcher: an authoritative table would be a
second matcher to maintain, and every time an engine widened its envelope
someone would have to remember to widen the table too. Several engines of one
family may all claim a graph and all stay in the ranked list -- a python
engine's claim is "I execute this whole graph", so claims compete rather than
conflict.

### Ids are identity; positions are rank

`python/cudnn/engines/engine_ids.py` defines ONE flat integer id space,
segmented by provider:

```
[0,      10_000)   cuDNN backend engines (ids assigned by the backend)
[10_000, 20_000)   C++-side OSS engines -- reserved, not populated
[20_000, ...)      python engines, one FAMILY_BLOCK-wide block per family
                   (engine_ids.py holds the bases; MANIFEST holds the slots)
```

An `engine_id` is the engine's stable IDENTITY and the key the build walk
dispatches on. A plan's POSITION in `graph.plans` is its RANK. The two are
independent: any engine may sit at any position, and re-ranking never
renumbers anything. That is what makes an autotune result replayable -- it is
`(engine_id, knobs)`, and it must mean the same thing in the next version.

Consequences you must respect when adding engines:

- Ids are append-only. A shipped `engine_id` is never renumbered or reused.
  `EngineFamily.slots` is keyed by the engine's shipped NAME rather than by its
  position in `ENGINE_SPECS`, because that position is preference order and may
  change.
- Each family owns `[engine_id, engine_id + FAMILY_BLOCK)`. Declared intervals
  are what make disjointness checkable -- an arbitrary predicate is not.
- `create_execution_plans()` validates the final ranked list: an entry naming a
  python engine this graph cannot dispatch to raises rather than being dropped.


## The engine contract (`cudnn/engines/base.py`)

```
check_support(graph)              -> None, or raise to decline
build_plan(graph, plan, ctx)      -> CompiledPlan      (the expensive JIT step)

CompiledPlan.get_workspace_size() -> int
CompiledPlan.execute(graph, uid_to_data, ctx)          (the hot path)
```

- **Declining is typed.** `NotImplementedError` or
  `cudnn.cudnnGraphNotSupportedError` mean "not mine"; anything else is a bug
  and propagates. Engines whose internal analyzers raise `ValueError` for
  "cannot express this graph" translate it at the engine boundary -- see
  `FrostGemmEngine.check_support` and `FrostSdpaFwdEngine._decline_reason`.
- **An engine does not propose its own plans.** Which configs of an engine are
  worth running, and in what order against the backend's, is a comparison only
  a party with the whole picture can make -- an engine cannot see its siblings,
  and neither side of the FROST/backend split can place the other. That
  judgment lives in the family's `heuristics.recommend`; the engine's job is to
  say whether a config it is HANDED is servable (`check_support`) and to build
  it. `PlanConfig.knobs` reaches `build_plan` verbatim, which is what makes an
  autotune result replayable as `(engine_id, knobs)`.
- **`build_plan` runs once per (graph, plan)** at `build_plans()` time and the
  compiled artifact lives on the graph, so one engine instance is safely
  reusable across graphs.
- **What a runtime value cannot change belongs in a build-time table.** An
  operand's role and major, each output's shape rule and required alignment,
  which outputs are reductions -- all settled when the kernel compiled, and
  deciding them again per call is most of what a python execute path costs
  (measured: 40-50 -> 20-22 us for one gemm, across six flavors).
  `gemm/frost/recipe.py` is the worked example: one table, captured into a
  closure that loops over it flat. Even what the kernel's parameter list looks
  like is a table entry (`arg_plan`), which is why one loop serves every flavor
  -- a call path per flavor is how two of them disagree. That closure never
  raises and it is the ONLY thing that launches: what it refuses goes to a
  checker that reads the same table, names the rule and raises without running
  anything, and a graph it cannot serve at all is declined at `check_support`.
  A second executor kept for diagnostics is still a second answer to what the
  graph computes, and a differential between two readings of one plan cannot
  catch a misconception they share.
- **`ExecutionContext` carries handle, stream and workspace explicitly.** No
  engine may hard-code a stream, reach into private graph state, or allocate
  hidden workspace. `uid_to_data` is the caller's variant pack (tensor uid ->
  device buffer), exactly as the classic backend receives it.
- **The pack's vocabulary is `index` and `OperandBuffer`,** and the two are not
  the same thing. `pack.index_of(tensor_or_uid)` gives an operand's POSITION in
  the pack; `pack.operands(indices)` turns positions into `OperandBuffer`s --
  one caller buffer described (pointer, shape, stride, dtype), non-owning, and
  itself a DLPack producer. An engine resolves positions once at first execute
  and asks for buffers per call.

`python/cudnn/gemm/frost/engine.py` is the worked example, deliberately thin:
`check_support` delegates to `probe_supported` and
`build_plan` wraps `build_gemm_plan(graph)` in a `CompiledPlan`
whose `execute` keys the kernel's own operands out of the variant pack by uid
and validates the caller's workspace. All the analysis and codegen stayed in
`graph_analyzer.py` / `compiler.py`; the engine file is only the contract
around them.


## Where things live

Engine code lives with its operation under `python/cudnn/<op>/`. The frost
directory holds only the shared kernel-authoring framework -- no dispatch, no
op code, ever.

```
python/cudnn/
  engines/                      the dispatch mechanism (not FROST-specific)
    manifest.py                 the static engine table: which engines this
                                build ships, matched by a coarse node-type key
    engine_ids.py               the flat id space and its segments
    base.py                     BaseEngine / PlanConfig / CompiledPlan /
                                ExecutionContext / resolve_node_buffers
    heuristics.py               rank(graph, engines, backend_plans, modes) --
                                the one entry point; delegates to the family's
                                own recommend() (cudnn/<op>/<pass>/heuristics.py)

  frost/                        kernel-authoring framework ONLY
    template_loader.py          (path, TemplateParams) -> uniquely-named
                                kernel module
    tile_dsl/                   shared CuTe-DSL primitives (barriers, TMA,
                                MMA, softmax pieces, masks). Import by dotted
                                path: cudnn.frost.tile_dsl.<module>

  sdpa/                         ALL sdpa engines, every arch / pass / phase
    graph_analyzer.py           facts extraction: graph.nodes -> SdpaGraphFacts,
                                engine-agnostic, parsed once per graph, cached;
                                plus the shared variant-pack binding helpers
    fwd/
      engine.py                 BaseEngine wrappers + the engine-id offsets
      engines.py                Capabilities + EngineSpec table + SdpaFwdKnobs
                                + mismatch/analyze_for/build
      api_dsl.py                DSL adapters (APIBase). Arch-free filename:
                                APIs differ by PASS (fwd vs bwd), never by
                                sm version or head dim
      config_sm100.py           TemplateParams + per-geometry Cfg + raising
                                validation
      config_sm120.py           TemplateParams + supported SM120 tile/layout
                                vocabulary + raising validation
      kernels/
        prefill_d256_f16_sm100.py     naming: <phase>_d<dim>_<dtype-family>_sm<arch>.py
        prefill_d512_f16_sm100.py
        prefill_f16_sm120.py
        _common_sm100.py
        thd_sm100.py
    bwd/                        future: same shape, its own api_dsl.py

  gemm/frost/                   engine.py + graph_analyzer.py + compiler.py
                                + kernel_templates/
```

Two levels under the pass directory, always. The coverage axes (arch, phase,
head dim, dtype family) are encoded in filenames and engine names, never in
directory depth. A dimension may be omitted from a filename when one kernel
implementation covers multiple dimensions. A new reader should be able to list
`sdpa/fwd/kernels/` and see the whole coverage matrix on one screen.

As a layer stack (each layer talks only to its neighbors):

```
+----------------------------------------------------------------+
| user code             pygraph build, create_execution_plans,    |
|                       select_plan / deselect_engines, execute   |
+----------------------------------------------------------------+
| cudnn/engines/        manifest (what exists, and which family   |
|                       a graph belongs to) + rank (delegates to  |
|                       the family) + BaseEngine / PlanConfig /   |
|                       CompiledPlan                              |
+----------------------------------------------------------------+
| engine modules        one per op + pass (cudnn/sdpa/fwd,        |
|                       cudnn/gemm/frost): facts analyzer,        |
|                       Capabilities/EngineSpec, knob vocabulary, |
|                       lower(), APIBase adapter,                 |
|                       TemplateParams -> Cfg                     |
+----------------------------------------------------------------+
| kernel templates      CuTe-DSL sources; frost/template_loader   |
|                       makes one specialized module per          |
|                       TemplateParams                            |
+----------------------------------------------------------------+
```


## The flow, end to end

What each line of the lifecycle actually does, across the layers:

```
user code                       cudnn/engines                    engine module (cudnn/sdpa/fwd)
---------                       -------------                    -----------------------------
import cudnn                    nothing imported: the manifest
                                is strings and ints

g.sdpa(...)                     (nothing -- graph.nodes
                                records the op natively)

g.create_execution_plans([A])   family_for(graph): node types
                                  -> 0 or 1 families
                                _freeze(); _attach_facts()  --> analyze(graph) -> facts
                                                                  (parsed once, on the graph)
                                instantiate(family): import --> FrostSdpaFwdEngines()
                                  NOW, ids handed in
                                graph.backend_plan_entries()
                                  -> the backend's own ranked
                                     (engine_id, knobs), one
                                     create_execution_plans
                                     per heur_mode, tagged
                                rank(graph, engines,
                                     backend_plans, modes)
                                  -> resolve_heuristics()   --> recommend(modes, facts,
                                                                  offered, backend_plans)
                                                                  mismatch(capabilities,
                                                                    facts, knobs) per cell
                                  -> ONE ranked list = graph.plans

g.select_plan(i)   (optional)   pin index i; the walk is strict

g.check_support()               the selected entry's engine
                                re-affirms                  --> check_support(graph)

g.build_plans()                 walk from the selected index;
                                a decline advances (unless
                                pinned), else finalize
                                build_plan(graph, cfg, ctx) --> lower(spec, facts, knobs):
                                                                  TemplateParams =
                                                                    facts-derived semantics
                                                                    + knob choices
                                                                  load_template(path, params)
                                                                    -> specialized module
                                                                  module.compile(shapes)
                                CompiledPlan cached on graph      (CuTe JIT)

g.get_workspace_size()          CompiledPlan.get_workspace_size()

g.execute(vp, workspace)        ExecutionContext(handle,
                                stream, workspace)          --> resolve variant pack
                                                                -> kernel launch
```

And the same flow as data (which record feeds which decision):

```
                static, per engine          per graph, cached
                Capabilities                SdpaGraphFacts <--analyze-- graph.nodes
                       \                       /
                        v                     v
  SdpaFwdKnobs ------> mismatch(capabilities, facts, knobs)
  (PlanConfig.knobs)        |
                            +-- reason string --> the engine declines
                            |                     (NotImplementedError)
                            +-- None (eligible) --> lower(spec, facts, knobs)
                                                        |
                                                        v
                                             TemplateParams (frozen)
                                                        |
                                          load_template(path, params)
                                                        |
                                                        v
                                    specialized kernel module (one per params)
                                                        |
                                             module.compile(shapes)
                                                        |
                                                        v
                                     executor(variant_pack) -> kernel launch
```

Reading the two diagrams together: everything left of `lower()` is cheap and
compile-free (facts extraction plus field comparisons -- safe to run on every
graph, which is why `recommend()` may run it for every cell); everything right
of it is the expensive JIT, reached only at `build_plans()` for the entry the
walk lands on. The records travel one way: facts and the plan's knobs feed
eligibility; eligibility plus the engine's defaults produce TemplateParams;
TemplateParams produces exactly one specialized module.


## Eligibility: Facts, Capabilities, Knobs, TemplateParams

Four records with distinct jobs:

| record | describes | one entry is | lifetime |
|---|---|---|---|
| `SdpaGraphFacts` | what the graph asks for | a single concrete value ("this graph is bf16") | per graph, runtime |
| `Capabilities` | what one ENGINE can serve | an acceptance set or rule ("dtypes = {fp16, bf16}", "skv % 128") | static, per engine row |
| knobs (requested) | tuning the plan carries | one requested value per knob | per `PlanConfig` |
| `TemplateParams` | how one template instance compiles | one chosen value per compile-time switch | per compiled template |

**Facts** (`graph_analyzer.analyze`) describe what the graph asks for:
geometry, dtype, masks, sink/THD/stats, requested features (bias, dropout,
paged KV, ...). Extraction never judges supportedness. A malformed graph (K/V
shape mismatch, padding mask without seq_len_kv, ...) sets `facts.invalid` and
every engine refuses. `analyze` returning `None` means "not my operation".
Parsed once per graph and cached (a weak-keyed cache), so N engines of an op
share one parse.

**Capabilities** (`fwd/engines.py`) declare, in the same vocabulary, the
envelope one ENGINE can serve -- including the tuning-knob domains it honors.
An engine is a lowering strategy that may span several kernels (and several
engines may share a kernel); its row declares what its lowering can actually
deliver.

**The eligibility check reads both facts and the plan's knobs:**

```python
def analyze_for(spec, graph, knobs=None):
    facts = ga.analyze(graph)
    if facts is None:
        return None, "graph is not a single sdpa() forward node"
    return facts, mismatch(spec.capabilities, facts, knobs)
```

Both checks run before any compile, field by field, returning the first
human-readable reason the engine cannot serve the request (or `None`). The
engine turns a reason into a decline:
`raise NotImplementedError(f"{self.name}: {reason}")`. Engines with different
feature envelopes (one has paged-KV, another has sinks) are just different
rows; there is no shared if-ladder that must know about every engine.

**TemplateParams is the output of a successful match, not an input.** After
the check passes, the engine's `lower` hook assembles the frozen record the
loader injects into the kernel template: graph-derived semantics plus knob
choices (requested values where given, engine defaults otherwise). The
architecture config's validation (`config_sm100._validate_params` /
`make_cfg_*`, `config_sm120.validate_params`) re-validates that record and
raises `ValueError`, but that is a backstop: reaching it means a
`Capabilities` row is dishonest, not that a user did something wrong.

### Feature interactions: the box and the notches

Field-wise capabilities describe an axis-aligned box in feature space (the
product of per-axis sets: dtypes x masks x layouts x ...). Real support
surfaces are almost boxes, with NOTCHES cut where feature conjunctions break
(bottom-right + SWA; THD + stats; a future mxfp8 + SWA + dropout). Both are
expressible, with one discipline separating them:

- The box is pure data: per-axis fields on `Capabilities`. Covers most of the
  surface; adding an engine is writing a row, not logic.
- A notch is a rule in `mismatch()` gated by a conjunction flag on the row
  (e.g. `padded_stats: bool` — padding mask + generate_stats needs the
  per-batch LSE trim). The matcher encodes the SHAPE of the interaction once;
  each engine's row supplies the VERDICT. When a future
  kernel supports the conjunction, flip its flag -- never edit the matcher.
  This is what keeps interaction checks from regressing into a per-engine
  if-ladder: shared code may know about kinds of interactions, never about
  specific engines.
- Escape hatch for a truly one-engine oddity (if one ever exists): an optional
  `extra_mismatch(facts, requested) -> reason | None` hook on its
  `EngineSpec` -- engine-local code, same reason-string contract, still run
  before lowering so the "ValueError past eligibility is a capabilities bug"
  rule holds. Not built until a constraint needs it.


## The knob channel (no global enum)

The C++ backend forces one global `knobType_t` enum on every engine of every
op -- an ABI constraint, not a design ideal. Here knobs are scoped at three
levels, with no shared vocabulary at all:

- **Vocabulary per operation.** Each op defines a typed, frozen dataclass:
  `cudnn.sdpa.fwd.engines.SdpaFwdKnobs(sched_policy=None, tile_m=None,
  tile_n=None, cga=None, pack_gqa=None)`, where `None` means "no
  preference". SDPA's knobs cannot collide with GEMM's; fields have real
  types instead of enum-plus-int64.
- **Domains per engine.** Each `Capabilities` row advertises the values its
  lowering honors: `sched_policies = {NATURAL}`, `tile_ms = {128}`,
  `tile_ns = {128}`, `cgas = {2}`, `pack_gqas = {False}`. Two engines of the
  same op may honor different subsets.
- **Per plan, not per graph.** A knob set rides on `PlanConfig.knobs`, so a
  tuning choice is part of the plan's identity: a family that wants several
  tunings ranked emits several `PlanConfig`s from `recommend()`, each with its
  own knobs, and the caller picks one with `select_plan(i)`. The knobs reach
  `check_support`'s `mismatch()` and then `build_plan` verbatim, and each
  distinct `TemplateParams` compiles into its own module, so two graphs in one
  process can run different tunings of the same engine.

**A recommendation always names a CONCRETE config.** `recommend()` fills every
knob axis on which the cell declares a domain; `None` means the capability row
declares no domain at all, so there is nothing for the engine to honour. It
never means "engine, pick for me" -- that reading is what let the same choice
be made twice, once in the ranking and once inside the adapter, and drift.

**There is no user-facing knob setter yet.** The old
`graph.set_engine_knobs(...)` was part of the deleted monkey-patch layer and
has no replacement. The plumbing that makes a request expressible is in place
(`PlanConfig.knobs` -> `mismatch()` -> `lower()`); what is missing is the
user-facing producer.

**A knob is honored or the engine is ineligible -- never silently degraded.**
If a kernel cannot run the requested scheduler policy, the answer is "this
engine cannot serve this plan", not "ran with a different policy". A knob
object of the wrong operation's type is rejected outright.

Generic discoverability survives without the enum: knob domains are ordinary
dataclass fields on `Capabilities`, so "list every engine and the knobs it
honors" is a `dataclasses.fields()` walk over the spec table.


## Engines are lowering strategies, not kernels

The engine-to-kernel mapping is many-to-many by design:

- One engine serves several dtypes through one template: the d512 engine
  lowers fp16 and bf16 graphs to `prefill_d512_f16_sm100.py` with different
  TemplateParams.
- One engine can drive several kernels: `EngineSpec.lower` is a hook
  `(spec, facts, knobs) -> executor`. The default (`lower_dsl_prefill`)
  compiles one template, but an engine may pick between kernels (decode vs
  prefill by S_q) or chain launches -- the THD path already runs an
  O-descriptor builder kernel before the main one.


## Heuristics are per operation

`cudnn/engines/heuristics.py` is the single entry point, and it decides
nothing:

```python
def rank(graph, engines, backend_plans, modes=None) -> List[PlanConfig]:
    family = manifest.family_for(graph)
    recommend = manifest.resolve_heuristics(family)   # the family's own rules
    facts = graph._facts_for(manifest.resolve_analyzer(family))
    return recommend(modes, facts, {e.name: e.engine_id for e in engines}, backend_plans)
```

Ranking knowledge is op-specific -- what makes one SDPA engine beat another
(seqlen regime, GQA ratio, causal fraction) is meaningless for GEMM -- so it
lives in `cudnn/<op>/<pass>/heuristics.py`, the analogue of the C++ per-op heur
files such as `jit_engine_heur_sdpa.cpp`. A family that declares no
`heuristics` hook falls back to one default plan per accepting engine, ahead of
the backend's.

The family is the smallest scope that can rank, and that is the whole reason
this seam exists rather than an engine-side `propose_plans`:

- **`recommend(modes, facts, offered, backend_plans) -> [PlanConfig]` returns
  `graph.plans`, position for position.** Nothing downstream reorders it.
- **It places BOTH sides.** The backend's entries arrive tagged with the
  `heur_mode` that produced them, so the family says, per mode, whether its own
  configs lead or follow. That is a measurement, not a preference: whether a
  FROST cell beats the backend's kernel on a given arch is a number someone
  timed.
- **Each mode contributes a block, and the blocks concatenate** in the caller's
  order. `[A, FALLBACK]` therefore puts every tuned candidate -- both sides' --
  ahead of every fallback.
  - **A**: candidates worth running, best guess first, runners-up behind it for
    a caller that autotunes.
  - **FALLBACK**: the config expected to build where mode A's choice may not.
    Nothing here is chosen for speed.
  - **OPENSOURCE**: mode A without the backend's recommendation -- these cells
    ARE the open-source implementation. Combine it (`[OPENSOURCE, A, FALLBACK]`)
    to measure coverage: a graph that ends up on a backend plan is one FROST
    does not cover.
  - **B**: answered as A until a family has a wider search to give.
- Knob precedence: **user request > heuristic proposal > engine default** --
  same rule at every level: a proposal outside the engine's `Capabilities`
  domain is a heuristic bug; a request outside it makes that plan ineligible.


## Opsets: graph-to-operation mapping is implicit and multi-valued

The C++ backend maps each graph to exactly one opset (a central pattern
matcher with precedence), and engines register against opsets. There is no
opset enum here, deliberately -- the same reasoning as the knob enum:

- **In code, an opset is one module** (op + pass granularity):
  `cudnn.sdpa.fwd.engines` is the SDPA-forward opset, `cudnn.gemm.frost` the
  GEMM one. A manifest row points at it; the module is imported only when the
  coarse key matches, so the DSL import cost is paid by the graphs that might
  use it and nobody else. (GDN, GDN-2 and KDA are single-node ops with no
  backend lowering at all: their FROST engines are the only way to execute
  them, and they are rows like any other.)
- **Each op's analyzer IS its pattern matcher.** `analyze()` returning facts
  means "this graph is an instance of my operation"; returning `None` means
  "not mine". Membership is a predicate the op owns, not an entry in a central
  table someone must maintain. The manifest's node-type key does NOT overrule
  it -- it only decides whether the question is worth an import (see "The
  coarse key is a NECESSARY condition"). The per-op facts cache keeps this
  cheap: N engines of an op share one parse.
- **Multiple ops may claim one graph.** Nothing prevents two ops' analyzers
  from both matching (a matmul graph claimed by gemm engines and by a future
  fused-epilogue op's engines). This is coherent because a FROST engine's
  `build_plan` executes the ENTIRE graph: a claim is a complete alternative
  execution strategy, never a partition -- so claims compete, they cannot
  conflict. All claims land in the same flat ranked list, resolved by the
  family's `recommend()`, with `select_plan` / `deselect_engines` as the user's
  overrides.
- **Today a graph belongs to at most ONE family**, so no cross-op comparison
  arises: `_ANCHOR_NODE_TO_FAMILY` names one family or none. When two families
  can claim one graph, they must return a common currency -- estimated cost --
  rather than resolving by precedence (that is the opset enum reborn). The
  shape (flat manifest, whole-graph claims, per-family rankers with comparable
  scores) is what must not regress.


## Engine naming

```
sdpa_<pass>_<phase>_sm<arch>[_d<dqk>[x<dv>]][_<quantization>]

sdpa_fwd_prefill_sm100_d512
sdpa_fwd_prefill_sm100_d256
sdpa_fwd_prefill_sm100_d128_fp8
sdpa_fwd_prefill_sm120
sdpa_fwd_decode_sm100_d256          (future)
sdpa_bwd_sm100_d128                 (future)
```

- The name is what `get_plan_name_at_index()` reports and what
  `deselect_engines()` matches on (substring), so it is user-visible API: do
  not rename a shipped engine.
- dtype is deliberately NOT part of the name: one cell's engine serves every
  dtype its kernel handles (fp16 and bf16 today, via `Capabilities.dtypes`),
  so users do not switch engine strings when they flip precision. A
  quantization suffix (`_fp8`, `_mxfp8`) marks a genuinely different kernel,
  not a dtype flip.
- `sm<arch>` may name an architecture FAMILY (`sm120` = SM120 + SM121) when
  one row serves several compute capabilities. The row's `Capabilities.arches`
  set is the source of truth for exactly which; the name never enumerates
  minors.
- Head dimensions never appear in engine names: one engine per
  arch x dtype family accepts a DOMAIN of dimensions and its lowering picks
  the kernel flavor (the smallest native shape covering the graph).
  `Capabilities.d_shapes` (native flavor shapes) plus `d_pad_multiple`
  (envelope alignment; 0 = exact shapes only) are the source of truth for
  that domain.
- No version counters. If a genuinely distinct second engine ever serves the
  same cell, give it a descriptive variant suffix (e.g. `_cga4`), not a number.
- Names are for humans; `engine_id` is for machines. Pin by index
  (`select_plan`) or replay by id -- never by parsing a name.
- `cudnn.sdpa.fwd.engines.engine_name(arch=..., fp8=..., mxfp8=...)` computes
  the family names (test/user convenience).


## Kernel templates and TemplateParams

Compile-time kernel parameters travel as a frozen `TemplateParams` dataclass
(graph-derived semantics + chosen knob values). The shared loader
(`frost/template_loader.py`) executes each kernel template into a
uniquely-named module per `(path, params)`, injecting them as the module
global `FROST_TEMPLATE_PARAMS` before the body runs. Multiple parameter sets
coexist in one process; nothing is reloaded, nothing is popped from
`sys.modules`, no `sys.path` entries are added, no environment variables are
read or written.

A kernel template:

- imports shared primitives by dotted path
  (`from cudnn.frost.tile_dsl.mma import mma_ss`) -- never via `sys.path`
  manipulation or bare top-level names;
- builds its config once at import:
  `PARAMS = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())` then
  `CFG, _TMA = make_cfg_d<dim>(PARAMS)` -- the plain-import default keeps
  `python kernels/<file>.py` (standalone benchmark) working;
- treats `CFG.*` fields as compile-time constants (`cutlass.const_expr`,
  `cutlass.range_constexpr`) so each parameter set traces to specialized code;
- exposes `compile(b, qh, kh, sq, skv) -> callable` with an `@lru_cache`
  (per-shape cache; the per-parameter split already happened at module load).

Asserts:

- Anything derived from user input raises `ValueError` inside `make_cfg_*` --
  never a module-level `assert` (stripped under `python -O`, and an
  import-time crash is undebuggable from the frontend).
- Hardware-invariant geometry checks in a template use a raising helper (see
  `_require` in `prefill_d512_f16_sm100.py`) and must be unreachable for any
  parameter set the engine's capabilities admit.
- Never `assert api.check_support()` -- it raises on failure and the assert is
  stripped under `-O`; call it plainly.


## Adding coverage, cheapest first

1. **New dtype an existing template already handles**: add it to the row's
   `Capabilities.dtypes`. **New geometry**: one new `EngineSpec` row plus its
   permanent `EngineSlot` in the family's manifest entry.
2. **New knob**: add the field to the op's knobs dataclass (`None` default),
   the domain to `Capabilities`, the check line to `mismatch()`, and the merge
   in `lower`. To make it reachable, teach the family's `recommend()` which
   value to name, and emit the runners-up behind it. Update the gating tests.
3. **New kernel (decode, fp8, new arch)**: write the template in
   `<op>/<pass>/kernels/` following the naming grammar; add its `make_cfg_*`
   to the config module (a new `config_sm90.py` for a new arch); add the spec
   row with an honest `Capabilities` and its id offset. `api_dsl.py` and
   `graph_analyzer.py` should not need changes.
4. **New pass (bwd)**: new `<op>/bwd/` with its own `api_dsl.py` (the tensor
   contract differs), reusing the shared analyzer facts and the loader; a new
   `BaseEngine` subclass and a new `EngineFamily` with a fresh id block
   (`FROST_SDPA_BWD_ID_BASE` is already reserved).
5. **New op**: new `python/cudnn/<op>/` with the same layers (analyzer facts,
   engines/capabilities, kernels), a `BaseEngine` subclass, and one
   `EngineFamily` in `cudnn/engines/manifest.py` naming its module, factory,
   slots, analyzer and heuristics -- plus its anchor node types in
   `_ANCHOR_NODE_TO_FAMILY`. There is no out-of-tree route; the manifest is the
   only way a python engine exists.


## The rules (read before changing anything)

1. **Facts never judge; capabilities never parse.** The analyzer describes,
   the engine rows accept or reject. If you find yourself writing an if-ladder
   that knows about specific kernels inside shared code, stop -- that logic
   belongs in a `Capabilities` row.
2. **Every kernel constraint appears in its engine's `Capabilities`.** A
   `ValueError` escaping a kernel template or `make_cfg_*` after the
   eligibility check passed is a bug in the capabilities row, and there must
   be a test for the constraint (accept AND reject).
3. **Decline with the typed exceptions only.** `NotImplementedError` /
   `cudnn.cudnnGraphNotSupportedError` mean "not mine" and advance the build
   walk; anything else is an engine bug and must propagate.
4. **`engine_id` is forever.** Append-only, never renumbered, never reused,
   always inside the block the manifest reserves for the family. Ranking may
   move a plan's position freely; it may never move its id.
5. **A knob is honored or the engine is ineligible.** Never substitute, never
   silently degrade. Precedence when heuristics land:
   user request > heuristic proposal > engine default.
6. **No global vocabularies.** Knob dataclasses are per operation; no shared
   enum, no shared registry of knob meanings; the manifest's node-type key is
   a filter, never an authoritative opset table.
7. **No environment variables for configuration.** The single flag that
   exists, `CUDNN_FRONTEND_ENABLE_FROST_ENGINES`, gates nothing but engine
   MATURITY (`opt_in=True` rows), and it is read in exactly one place
   (`engines/manifest.py`). It never tunes behaviour, selects a kernel, or
   changes a contract; parameters travel as typed dataclasses through the
   loader.
8. **No monkey-patching and no import side effects.** The graph API is native
   Python (`graph.nodes` already records everything); engines are found by
   data, not by an import that mutates `cudnn.pygraph`. An engine module must
   be importable without side effects beyond defining its classes.
9. **No module-level asserts on anything a user could trip.** Raise
   `ValueError` in validation functions; keep `assert` for programmer
   invariants inside tile_dsl at most.
10. **Two directory levels under the pass, maximum.** Coverage axes go in
    filenames and engine names, never in directory depth.
11. **Identifiers are keyed by op geometry, not model names.**
    `make_cfg_d512`, not `make_cfg_dsv4`; model provenance goes in comments
    only.
12. **Tests gate everything.** New capabilities field -> accept and reject
    tests. New knob -> gating tests (in-domain, out-of-domain, wrong
    vocabulary). New engine -> a manifest test that the row exists and a
    frontend-integration test that it appears in `graph.plans` and runs when
    pinned. Mark test modules `pytest.mark.L0` -- the default pytest addopts
    is `-m L0` and unmarked tests silently never run. Run
    `pre-commit run --all-files` (black, 160 cols) before pushing.
13. **Keep this document true.** If code and this contract disagree and you
    change the code, change this file in the same commit.
