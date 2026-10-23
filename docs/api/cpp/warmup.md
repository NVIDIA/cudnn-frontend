# warmup

Run a fake CUDA graph capture warmup for a built or deserialized graph.

## C++ synopsis

```cpp
auto status = graph.warmup(handle);
```

## Parameters

- `handle` (`cudnnHandle_t`, required): cuDNN handle whose stream is used for the fake capture.

## Returns

- `status` (`fe::error_t`): `OK` when the warmup completes or when warmup is skipped because CUDA graph capture is already active.

## Errors

The returned `error_t` can report CUDA stream, CUDA graph, allocation, or graph execution setup failures. The warmup intentionally ignores the status of the dummy graph execution itself.

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

`warmup()` is an advanced helper used by deserialization. It creates a temporary CUDA graph capture with dummy pointers so backend descriptors and execution paths are initialized before normal execution.

The method allocates a temporary 1 MiB device buffer and restores the original cuDNN stream before returning. Most users do not need to call it manually.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1906) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1906).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/serialize_basic.cc)

<Code
  src="../../samples/cpp/serialize_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

There is no direct Python graph binding documented for `warmup()` in `docs/api/python`.
