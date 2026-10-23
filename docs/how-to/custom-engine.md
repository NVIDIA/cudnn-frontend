# Inspect automatic Python engine selection

`cudnn.pygraph` ranks the Python engines shipped in the library together with cuDNN backend plans. Applications select among the resulting plans; they do not register engines or replace the routing policy at runtime.

## Before you begin

Use the explicit `cudnn.pygraph` lifecycle and construct a graph that is supported by the installed cuDNN release. Python-engine routing remains an unreleased interface, so do not persist plan indices across frontend versions.

## Inspect the graph

Graph inspection is available before planning:

<Markdown src="/snippets/generated/python/custom_engine/inspection.mdx" />

Use named ports, tensor metadata, and `NodeType` when diagnosing engine support. Insertion order is not an execution contract.

## Inspect automatic selection

<Markdown src="/snippets/generated/python/custom_engine/routing.mdx" />

`create_execution_plans()` discovers candidate Python engines from the in-tree manifest and queries the backend for the requested heuristic modes. Mode B asks the backend for a broader search; Python engine families use their mode-A recommendations because they do not expose a separate mode-B candidate set. `graph.plans` contains the combined ranked list.

`graph.selected_engine` identifies an in-tree Python engine. A value of `None` means the selected entry is owned by the cuDNN backend. `select_plan()` pins an index strictly: if that entry cannot build, the call fails instead of silently moving to another plan.

## Execute the selected plan

<Markdown src="/snippets/generated/python/custom_engine/execution.mdx" />

The caller owns tensor buffers, workspace, handles, and stream lifetime. The selected implementation must preserve the graph's declared output and asynchronous execution contract.

## Add an engine to the library

Library contributors add an engine to `python/cudnn/engines/manifest.py`, give it a stable family slot, implement exact support and build contracts, and add family-level analyzer and heuristic hooks when ranking requires them. Runtime engine registration is intentionally unsupported: a caller-provided engine would not have a manifest capability record or a stable plan identity.

See [Python IR routing invariants](/concepts/python-ir-routing) for the dispatch model. The retained Fern source contains the complete contributor contract in `docs/utilities/python_graph_and_execution_backends.md`.

## Expected result

The sample prints whether the first ranked plan belongs to an in-tree Python engine or the cuDNN backend, executes that plan, and checks its result against PyTorch.
