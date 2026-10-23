# Python IR routing invariants

`cudnn.pygraph` stores an inspectable Python operation DAG, then ranks the in-tree Python engines for that graph family together with cuDNN backend plans. These invariants keep routing deterministic and prevent a plan from observing a graph different from the graph it accepted.

## Operation declaration mechanisms

The Python IR covers the public Graph operation surface through three source-owned registries plus explicit methods:

- `_POINTWISE_TENSOR_ARGS` installs uniform pointwise builders whose mode matches the method name.
- `_STRUCTURED_OPS` describes named input ports, scalar attributes, outputs, and shape inference for convolution, normalization, reduction, block scaling, MoE, and structural operations.
- `_CAPTURED_OPS` records explicit schemas for SDPA variants with many conditional inputs and outputs.
- `matmul` remains explicit because its positional calling convention is independently useful.

Input and output port names match the C++ binding keyword names. An engine should inspect `NodeType`, named ports, parameter values, tensor metadata, and graph context instead of depending on insertion order.

## Engine contract

A `BaseEngine` checks support and builds a `PlanConfig` selected by family heuristics. Building a plan performs expensive specialization once. A compiled plan executes with caller-provided tensor data and an `ExecutionContext` containing handle, stream, workspace, and dynamic overrides.

`python/cudnn/engines/manifest.py` is the only engine registry. Applications cannot add engines at runtime. Each family owns a fixed engine-ID block, and slots must remain stable for reproducible selection and cache identity. An engine declines only with one of the documented decline exception types. Any other exception is an implementation defect and propagates rather than silently trying another engine.

## Routed and backend plan spaces

The routed list contains eligible in-tree Python plans and cuDNN backend entries. `select_plan()` addresses this combined list. A backend entry may refer to a specific backend-plan index or delegate to backend heuristics when the backend does not expose its candidates. A routed index must never be interpreted as a backend plan index.

Family heuristic hooks may return only engine IDs owned by that family. A `BACKEND` marker places the backend block relative to Python candidates; the common assembler expands the marker, combines heuristic-mode blocks, and removes duplicate `(engine_id, knobs)` entries.

## UID ownership

The Python IR owns the graph UID namespace. User-assigned UIDs are pushed explicitly when creating the backend graph representation. Two user-assigned tensors cannot share a UID. When a user UID collides with an automatically assigned UID, the automatically assigned tensor is renumbered so the explicit user contract wins.

Bind each required non-virtual UID exactly once at execution. UID equality does not establish shape, stride, dtype, device, or lifetime compatibility.

## Pure execution paths

A graph selected for a Python engine executes without creating a C++ backend graph representation. A graph selected for the backend creates that representation from the complete Python IR and delegates planning and execution. Mixed construction, where some nodes are built directly in C++ and others in the Python IR, is unsupported.

Explicitly querying backend plan methods requests backend graph creation even when Python engines are available. Applications should not use those methods merely to inspect a Python-engine plan.

## One-shot planning and freeze

`create_execution_plans()` is one-shot. Use `select_plan()` to switch among already proposed entries; construct a new graph to run routing with different policy.

Creating the graph representation or planning freezes the graph's entire public mutation surface: graph operation builders, fluent tensor and node setters, direct metadata assignment, node input/output/parameter mappings, and in-place dimension or stride edits. Inspection remains available. A mutation after validation but before freeze invalidates validation and requires validation again.

## Output layout ownership

Only output dimensions and strides explicitly assigned by the user are authoritative when creating the graph representation. Python-inferred strides are provisional. When no explicit layout is supplied, the cuDNN backend retains operation-specific inference, such as preserving a convolution layout family.

An engine that requires a concrete output layout must either accept the user-assigned layout or establish its own documented output contract during support checking. It must not assume every provisional Python row-major stride was a user requirement.

## Prove the selected path

Backend-path tests should check numerics and dispatch state: `graph.selected_engine` is `None`, backend plans exist, and a backend graph representation was created. Python-engine tests should assert the selected manifest engine and verify that unsupported graphs are rejected. Do not assert backend kernel names because those names are internal and version-dependent.

Current sources: [Python graph implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L44) and [engine contract](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/engines/base.py#L1).

## Availability

Python IR routing is unreleased behavior in this checkout. Manifest slots,
family heuristics, UID collision handling, freeze boundaries, plan
compatibility, and pure-Python execution contracts can change before release.
