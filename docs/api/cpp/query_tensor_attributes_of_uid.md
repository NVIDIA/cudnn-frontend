# query_tensor_attributes_of_uid

Return tensor attributes associated with a frontend UID.

## C++ synopsis

```cpp
fe::graph::Tensor_attributes tensor;
auto status = graph.query_tensor_attributes_of_uid(uid, tensor);
```

## Parameters

- `uid` (`int64_t`, required): Tensor UID to query.

- `tensor` (`fe::graph::Tensor_attributes&`, output): Receives a copy of the matching tensor attributes.

## Returns

- `status` (`fe::error_t`): `OK` when a matching tensor is found. `INVALID_VALUE` is returned when no matching UID exists.

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

The C++ implementation searches graph outputs, graph inputs, and deserialized tensor properties.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2370) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3615).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/query_tensor_attributes_of_uid_basic.cc)

<Code
  src="../../samples/cpp/query_tensor_attributes_of_uid_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.query_tensor_attributes_of_uid` reference](/api/python/query-tensor-attributes-of-uid) for Python exceptions, wrapper behavior, and current availability constraints.
