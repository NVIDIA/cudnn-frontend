# serialize

Serialize graph and plan state to a list of bytes.

## C++ synopsis

```cpp
std::vector<uint8_t> data;
auto status = graph.serialize(data);

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
nlohmann::json j;
graph.serialize(j);
#endif
```

## Parameters

- `data` (`std::vector<uint8_t>&`, output): Receives UBJSON-encoded serialized graph data.

- `j` (`nlohmann::json&`, output overload): Receives the JSON object representation used internally before UBJSON conversion. This overload is available only when the frontend is compiled with JSON support.

## Returns

- `status` (`fe::error_t`): `OK` when byte-vector serialization succeeds.

- `void`: The JSON overload fills the supplied `nlohmann::json` object and does not return `error_t`.

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

When `CUDNN_FRONTEND_SKIP_JSON_LIB` is defined, byte-vector serialization returns `GRAPH_NOT_SUPPORTED`, and the JSON overload is not compiled.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2013) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2013).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/serialize_basic.cc)

<Code
  src="../../samples/cpp/serialize_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.serialize` reference](/api/python/serialize) for Python exceptions, wrapper behavior, and current availability constraints.
