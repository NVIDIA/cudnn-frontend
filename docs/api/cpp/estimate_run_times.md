# estimate_run_times

Historical entry for cuDNN cost-model run-time estimates. The current C++
`Graph` class does not expose this method.

## C++ synopsis

```cpp
// Not exposed by the current C++ Graph class.
```

## Parameters

- `times` (`std::vector<float>&`, output): Receives predicted run times in milliseconds.

## Returns

- `status` (`fe::error_t`): `OK` when at least one engine config can be estimated. A failed overall estimate returns a bad status, commonly `HEURISTIC_QUERY_FAILED`.

## Errors

Most C++ lifecycle calls return `fe::error_t`. Check `status.is_good()` or `status.is_bad()` before continuing. Errors can report invalid graph attributes, unsupported backend graphs, failed heuristic queries, failed execution-plan creation, CUDA/cuDNN backend failures, invalid variant packs, invalid plan indices, unsupported runtime overrides, or execution failures.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Not exposed
- **API stability:** Historical
- **API language:** cpp
- **API-specific availability:** Not exposed by the current Python binding or C++ Graph class
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

The signature is retained only as a historical lookup entry. No current cuDNN
version claim applies while the method is absent from the public C++ class.


{/* BEGIN GENERATED API SOURCES */}
Current absence evidence: [inspected public surface](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L53) and [inspected implementation surface](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L53).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.estimate_run_times` reference](/api/python/estimate-run-times) for Python exceptions, wrapper behavior, and current availability constraints.
