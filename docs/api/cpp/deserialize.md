# deserialize

Restore graph and plan state from serialized data.

## C++ synopsis

```cpp
auto status = graph.deserialize(handle, data);
auto json_status = graph.deserialize(json_object);
```

## Parameters

- `handle` (`cudnnHandle_t`, required for serialized byte data): cuDNN handle used to rebuild backend plan state.

- `data` (`std::vector<uint8_t> const&`, required): Serialized bytes.

- `json_object` (`nlohmann::json const&`, optional overload): JSON graph data for graph-format deserialization.

## Returns

- `status` (`fe::error_t`): `OK` when deserialization succeeds.

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

Byte deserialization rebuilds execution-plan state, restores pass-by-value and workspace metadata, prepares the variant-pack template, and performs warmup. JSON-library-disabled builds report unsupported deserialization for byte data.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2128) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2128).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/deserialize_basic.cc)

<Code
  src="../../samples/cpp/deserialize_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.deserialize` reference](/api/python/deserialize) for Python exceptions, wrapper behavior, and current availability constraints.
