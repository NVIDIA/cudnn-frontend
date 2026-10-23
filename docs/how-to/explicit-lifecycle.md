# Control the explicit `cudnn.pygraph` lifecycle

Use the explicit lifecycle when you need to inspect graph state, filter backend plans, route to Python engines, serialize, or control runtime buffers yourself.

Python IR routing, its routed/backend index spaces, and its one-shot planning
contract are unreleased in this checkout. See
[Python IR routing availability](/concepts/python-ir-routing#availability).

## Before you begin

Complete [Build and extend a graph](/tutorials/first-graph) first. Use
`cudnn.pygraph` here only when the high-level context manager does not expose
the plan inspection, routing, serialization, or buffer control you need.

## Build in stages

The stages follow one ordered path:

1. `graph.validate()` checks Python IR and infers properties; the backend path may create the graph representation here so unsupported graphs can fail early.
2. `graph.build_operation_graph()` finalizes backend operation descriptors and makes backend engine queries available.
3. `graph.create_execution_plans()` builds one ranked list from eligible in-tree Python engines and the backend candidates returned for the requested heuristic modes.
4. `graph.select_plan()` chooses an entry from that combined list. Backend at-index methods operate in a separate index space.
5. `graph.check_support()` lets the selected engine reject the graph before compilation.
6. `graph.build_plans()` compiles the selected entry once. Planning and graph mutation are one-shot; construct a new graph to re-plan.
7. `graph.get_workspace_size()` determines required temporary storage.
8. `graph.execute()` binds every non-virtual tensor and enqueues work on the handle stream.

The [engine-selection sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/custom_engine.py) exercises routed
planning. The [`graph.execute()` sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/execute_basic.py)
uses the explicit backend lifecycle, binds runtime buffers, and executes a
matmul. The higher-level [matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/matmul_basic.py) is a
contrast: `cudnn.Graph` performs the build stages inside its wrapper.

## Find the failing stage

Invalid graph metadata normally raises `ValueError`. A legal graph that no selected engine supports raises `cudnnGraphNotSupportedError`. CUDA, handle, compilation, and execution failures propagate as runtime errors. Do not catch all errors and silently fall back; only explicit unsupported signals are safe for routing.

## Expected result

The `graph.execute()` sample creates the graph representation, queries
heuristic candidates, builds a supported plan, allocates the reported
workspace, and executes with explicit bindings. The engine-selection sample
inspects the combined routed list, identifies whether the selected entry is an
in-tree Python engine or the backend, and verifies the result against PyTorch.
