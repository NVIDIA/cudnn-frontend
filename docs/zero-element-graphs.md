# Zero-element (no-op) graphs

Graphs whose tensors contain a dimension of size 0 — for example SDPA with a
batch size of 0 — are supported by the frontend as no-ops
(see [issue #101](https://github.com/NVIDIA/cudnn-frontend/issues/101)).

The cuDNN backend does not accept tensor descriptors with 0-sized dimensions.
Instead of lowering such graphs, the frontend detects them during `validate()`:
if the graph references at least one zero-element tensor and **every**
non-virtual output tensor is zero-element, no output byte would ever be
written, so the graph is flagged as a zero-element no-op. For such graphs:

- `validate()`, `build_operation_graph()`, `create_execution_plans()`,
  `check_support()`, `build_plans()`, and `build()` all succeed without
  touching the backend.
- `get_workspace_size()` reports 0.
- `execute()` returns success without launching any work. Pointers in the
  variant pack are ignored (zero-element tensors may have null pointers).
- `populate_cuda_graph()` leaves the provided CUDA graph empty (an empty CUDA
  graph is valid and instantiable), and `update_cuda_graph()` is a no-op.
- `Graph::is_zero_element_graph()` (C++) / `pygraph.is_zero_element_graph()`
  (Python) report whether the graph was flagged.

## Example

```cpp
namespace fe = cudnn_frontend;
fe::graph::Graph graph;
graph.set_io_data_type(fe::DataType_t::HALF)
    .set_intermediate_data_type(fe::DataType_t::FLOAT)
    .set_compute_data_type(fe::DataType_t::FLOAT);

int64_t b = 0, h = 4, s_q = 64, s_kv = 64, d = 64;  // batch size 0
auto Q = graph.tensor(fe::graph::Tensor_attributes()
                          .set_dim({b, h, s_q, d})
                          .set_stride({h * s_q * d, s_q * d, d, 1}));
// ... K, V ...

auto [O, Stats] = graph.sdpa(Q, K, V, sdpa_options);
O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1});

// The full pipeline succeeds; execute() is a no-op.
auto status = graph.build(handle, {fe::HeurMode_t::A});
```

## Unsupported mixes

A graph that mixes zero-element tensors with **non-zero-element** output
tensors is rejected at `validate()` with `GRAPH_NOT_SUPPORTED` and a clear
message. For example, a matmul contracting over a dimension of size 0
(`[1, m, 0] x [1, 0, n] -> [1, m, n]`) would require zero-filling the output,
which cuDNN does not do.

## Limitations

- Plan serialization (`serialize(std::vector<uint8_t>&)` /
  `deserialize(handle, data)`) is not supported for zero-element no-op graphs,
  as there is no execution plan to serialize. JSON graph (structure)
  serialization is unaffected: a graph deserialized from JSON is re-detected as
  a no-op during its `validate()`.
- Runtime shape overrides (`override_uids` / `override_shapes` /
  `override_strides`) cannot be applied to a graph built as a zero-element
  no-op; build the graph with non-zero shapes instead.
