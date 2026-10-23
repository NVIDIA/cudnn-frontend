# estimate_run_times

Historical Python entry for cuDNN cost-model run-time estimates. The current
Python binding does not expose this method.

## Python synopsis

```python
# Not exposed by the current Python binding.
```

## Parameters

- This method takes no Python arguments.

## Returns

- `times` (`list[float]`): Predicted execution times in milliseconds, one per engine config produced by `create_execution_plans(...)`. Per-config estimate failures appear as positive infinity.

## Raises

- `TypeError`: Raised immediately when Python arguments cannot be converted to the bound C++ types, for example an invalid enum, a non-sequence list argument, or a non-integer plan index.

- `ValueError`: Raised for frontend validation errors that map to `std::invalid_argument`, such as missing tensor attributes, shape-deduction failures, invalid tensor names, or invalid variant-pack metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised when a well-formed graph or plan candidate is not supported by the selected heuristics, backend implementation, architecture, cuDNN version, or graph format.

- `RuntimeError`: Raised for cuDNN backend API failures, CUDA API failures, handle errors, graph execution failures, invalid values reported by the frontend, NVRTC compilation failures, or invalid device state.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Not exposed
- **API stability:** Historical
- **API language:** python
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

Historical versions described cost-model predictions rather than measured
wall-clock timings. The current Python and C++ Graph surfaces expose no such
method, so no current cuDNN version contract applies.


{/* BEGIN GENERATED API SOURCES */}
Current absence evidence: [inspected public surface](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2577) and [inspected implementation surface](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2577).
{/* END GENERATED API SOURCES */}

## Examples

No executable Python sample is published because the current binding has no
`estimate_run_times` attribute. The C++ counterpart is retained only as a
historical lookup entry and is also absent from the current Graph class.

## Appendix: C++ equivalent

See the [C++ `Graph::estimate_run_times` reference](/api/cpp/estimate-run-times) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
