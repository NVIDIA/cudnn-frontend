# Select, filter, and autotune plans

Create candidate plans, remove configurations that violate application requirements, and benchmark the remaining plans with representative buffers.

## Before you begin

Complete the [explicit lifecycle](/how-to/explicit-lifecycle) with a graph that has
more than one supported candidate. Define workspace, determinism, numerical,
and latency policy before filtering or timing candidates.

## Keep index spaces separate

Python `graph.plans` is the combined list of eligible in-tree Python plans and cuDNN backend candidates. `select_plan()` chooses from that list. Backend methods such as `build_plan_at_index()` use the backend's separate plan index space.

## Filter backend candidates

After `build_operation_graph()`, use engine, behavior-note, numerical-note, workspace, and shared-memory filters. A filter changes the eligible set; it does not prove numerical acceptability. Keep determinism, down-conversion, and reduced-precision notes aligned with application policy.

## Autotune

Allocate the maximum required workspace for candidates, warm up each plan, time multiple executions on the intended stream, synchronize at measurement boundaries, and choose a stable statistic. Use representative shapes and pointer alignment. Re-run when the backend, driver, GPU, or graph metadata changes.

<Markdown src="/snippets/generated/cpp/plan_inspection/autotune.mdx" />

## Control compilation

Build one selected plan or use the documented parallel build policy. SM carveout and engine knobs can change support and resource requirements. Persist a choice only with a compatibility key that includes the graph and target environment.

References: [`create_execution_plans`](/api/python/create-execution-plans), [`deselect_workspace_greater_than`](/api/cpp/deselect-workspace-greater-than), and [`autotune`](/api/cpp/autotune).

## Verification limits

The canonical sample exercises C++ autotuning and plan inspection. A focused
executable example for Python routed-plan index mapping is not yet published.
On CUDA 13.1, cuDNN 9.15.1, and SM100, the aggregate autotune workspace helper
dereferenced null slots after a partial all-plan build; the sample therefore
uses checked per-plan workspace queries.

## Expected result

The C++ sample filters supported candidates, allocates enough workspace for
each timed plan, autotunes representative buffers, and executes the selected
plan. Python uses the routed and backend index spaces described above; this
repository does not yet publish a focused executable Python autotune sample.
