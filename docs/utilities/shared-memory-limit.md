# Limiting Execution Plan Shared Memory

Use `Graph::deselect_shared_mem_greater_than()` to set the maximum dynamic shared memory, in bytes, allowed for an execution plan.

Set the limit before querying configurations if it should affect backend engine configuration:

```cpp
REQUIRE(graph.build_operation_graph(handle).is_good());

graph.deselect_shared_mem_greater_than(98 * 1024);
REQUIRE(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());
REQUIRE(graph.check_support().is_good());
```

For direct-engine workflows, set the limit before `Graph::get_knobs_for_engine()` and `Graph::create_execution_plan()`.

The limit is copied to the heuristic or engine descriptor when that descriptor is queried. Changing the limit afterward does not regenerate existing engine configurations or resize their pipeline stages. Query new configurations after changing the limit if backend configuration must use the new value.

The existing frontend filter remains active: execution plan configurations reporting more shared-memory use than the limit are deselected. This filter also applies when the limit is set after a query, but filtering cannot reduce the shared-memory use of an existing configuration.

## Default and Version Behavior

If `deselect_shared_mem_greater_than()` is not called, the frontend does not send a shared-memory-limit attribute to cuDNN. The backend continues to use its default device limit, preserving previous behavior.

With cuDNN 9.27 or later headers and a cuDNN 9.27 or later runtime, the frontend also sends an explicitly set positive limit during heuristic, forced-engine, and knob queries. Backends that support the attribute can use the limit while configuring an engine, such as when choosing a pipeline-stage count.

When built with headers older than cuDNN 9.27 or used with a runtime older than cuDNN 9.27, the new backend attribute is not sent. `deselect_shared_mem_greater_than()` retains its earlier behavior of filtering the configurations returned by the query.

See the [restricted shared-memory matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/matmul/matmuls.cpp) for a complete graph workflow.
