# update_cuda_graph

Update an existing CUDA graph populated by cuDNN with new runtime pointers and workspace.

## C++ synopsis

```cpp
auto status = graph.update_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph);
```

## Parameters

- `handle` (`cudnnHandle_t`, required): cuDNN handle.

- `variant_pack` (`std::unordered_map<Tensor_attributes::uid_t, void*>` or tensor-pointer map, required): Updated runtime pointer map.

- `workspace` (`void*`, required): Updated workspace pointer.

- `cudnn_cuda_graph` (`cudaGraph_t`, required): CUDA graph to update.

## Returns

- `status` (`fe::error_t`): `OK` when the CUDA graph is updated successfully.

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

The implementation validates the graph shape, updates workspace setup nodes, recreates the backend variant pack, and calls the backend CUDA graph update API.

On the recorded CUDA 13.1, cuDNN 9.15.1, and SM100 environment, the prerequisite
population call returned `CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API`, so
the update path could not be exercised.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L701) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L701).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.update_cuda_graph` reference](/api/python/update-cuda-graph) for Python exceptions, wrapper behavior, and current availability constraints.
