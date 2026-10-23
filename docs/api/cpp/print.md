# print

Return a JSON representation of the graph as a string.

## C++ synopsis

```cpp
std::string graph_json = graph.print();
```

## Parameters

This method takes no parameters.

## Returns

- `graph_json` (`std::string`): Serialized JSON text for the graph when JSON support is compiled in. If the frontend is compiled with `CUDNN_FRONTEND_SKIP_JSON_LIB`, the returned string states that `print` is unavailable.

## Errors

`print()` returns a string and does not return `error_t`. JSON serialization issues are compile-time or build-configuration concerns in this implementation.

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

`print()` is intended for debugging and inspection. It does not build, validate, or execute the graph.

Use `serialize()` when you need the binary UBJSON representation used for graph serialization and deserialization workflows.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3283) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3283).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/print.mdx" />

## Appendix: Python equivalent

There is no direct Python method named `print()` documented in `docs/api/python`.
