# API layers and graph lifecycle

cuDNN Frontend has several user-facing layers that share operations but own different amounts of lifecycle and integration behavior.

## API layers

`cudnn.Graph` is a PyTorch-oriented wrapper. It owns a `cudnn.pygraph`, runs lifecycle stages at context exit, allocates workspace and outputs, and makes the built graph callable.

`cudnn.pygraph` is the explicit Python IR. It owns tensor and node objects, validates and freezes them, ranks manifest-discovered Python engines with cuDNN backend plans, and binds caller resources at execution.

`cudnn_frontend::graph::Graph` is the modern header-only C++ API. It exposes graph attributes, operation construction, planning, filters, serialization, CUDA Graph integration, and explicit execution resources.

Decorators in `cudnn.graph` build and cache `cudnn.Graph` instances. `cudnn.ops` and `cudnn.experimental.ops` register PyTorch operators. Optional OSS packages call architecture-specific CUTLASS DSL kernels directly or participate through engine selection.

The legacy C++ builder API constructs backend descriptors individually. It remains a compatibility surface, not the preferred learning path.

## Graph lifecycle

| Stage | Result |
| --- | --- |
| Describe the IR | Tensors, operations, attributes, and output intent are explicit. |
| Validate | Graph invariants and inferred metadata are checked. |
| Create the graph representation | User attributes become cuDNN backend operation descriptors. |
| Create and rank plans | Heuristics discover engine configurations, including modes A, B, and fallback where requested. |
| Check and build | Unsupported candidates are rejected and reusable executable plans are compiled. |
| Bind and execute | Runtime data and workspace are bound without changing graph structure. |

The graph freezes when creating the graph representation or planning needs stable identity. Create a new graph for incompatible metadata or policy; do not mutate tensor UIDs, names, shapes, or node attributes behind a built plan.

See [Python IR routing invariants](/concepts/python-ir-routing) for UID ownership, plan index spaces, engine decline semantics, freeze boundaries, output layout ownership, and dispatch testing.

Python IR routing and pure-Python engines are unreleased in this checkout; the
[routing page](/concepts/python-ir-routing#availability) records their current
availability.
