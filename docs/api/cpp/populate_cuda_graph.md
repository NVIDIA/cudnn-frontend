# populate_cuda_graph

Populate an empty CUDA graph with the cuDNN graph execution nodes for the selected plan.

## C++ synopsis

```cpp
auto status = graph.populate_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph);
```

## Parameters

- `handle` (`cudnnHandle_t`, required): cuDNN handle.

- `variant_pack` (`std::unordered_map<Tensor_attributes::uid_t, void*>` or tensor-pointer map, required): Runtime pointer map.

- `workspace` (`void*`, required): Workspace pointer.

- `cudnn_cuda_graph` (`cudaGraph_t`, required): Empty CUDA graph to populate.

## Returns

- `status` (`fe::error_t`): `OK` when the CUDA graph is populated successfully.

## Errors

Most C++ lifecycle calls return `fe::error_t`. Check `status.is_good()` or `status.is_bad()` before continuing. Errors can report invalid graph attributes, unsupported backend graphs, failed heuristic queries, failed execution-plan creation, CUDA/cuDNN backend failures, invalid variant packs, invalid plan indices, unsupported runtime overrides, or execution failures.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** 8.5+ for the C++ frontend; Python wrapper paths require 9.12+
- **Family CUDA:** A CUDA toolkit supported by the installed cuDNN release
- **Family GPU:** Architecture supported by the selected cuDNN engine
- **Family platforms:** Linux (executed); Windows (source-supported, not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

The implementation requires the CUDA graph to be empty and appends frontend workspace setup nodes plus a backend child graph for the selected plan.

On the recorded CUDA 13.1, cuDNN 9.15.1, and SM100 environment, execution
reached cuDNN and returned
`CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API`. The required backend,
driver, and runtime combination has not been verified, so no executable sample
is published for that environment.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L861) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L861).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.populate_cuda_graph` reference](/api/python/populate-cuda-graph) for Python exceptions, wrapper behavior, and current availability constraints.
