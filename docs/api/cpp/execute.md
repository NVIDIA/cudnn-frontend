# execute

Execute the selected built plan with a variant pack and workspace buffer.

## C++ synopsis

```cpp
std::unordered_map<int64_t, void*> variant_pack = {
    {a_uid, A_gpu.devPtr}, {b_uid, B_gpu.devPtr}, {c_uid, C_gpu.devPtr},
};
auto status = graph.execute(handle, variant_pack, workspace.devPtr);
```

## Parameters

- `handle` (`cudnnHandle_t`, required): cuDNN handle used for execution.

- `variant_pack` (`std::unordered_map<int64_t, void*>` or `std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>`, required): Runtime pointer map for non-virtual tensors.

- `workspace` (`void*`, required): Device workspace pointer.

- Optional override metadata: Runtime UID, shape, and stride vectors for dynamic or override-shape execution.

## Returns

- `status` (`fe::error_t`): `OK` when the selected plan executes successfully.

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

The C++ implementation prepares a variant-pack template after plan build and fills runtime user pointers at execution time. Missing UIDs are reported as `INVALID_VARIANT_PACK`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L681) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L681).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/execute_basic.cc)

<Code
  src="../../samples/cpp/execute_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.execute` reference](/api/python/execute) for Python exceptions, wrapper behavior, and current availability constraints.
