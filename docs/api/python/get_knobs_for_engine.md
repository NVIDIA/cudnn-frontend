# get_knobs_for_engine

Return the tunable knob descriptors for a backend engine.

## Python synopsis

```python
knobs = graph.get_knobs_for_engine(engine_id)
```

## Parameters

- `engine_id` (`int`, required): Global backend engine ID to inspect.

## Returns

- `knobs` (`list[cudnn.knob]`): Knob descriptors. Each descriptor exposes fields such as knob type, minimum value, maximum value, and stride according to the Python binding.

## Raises

- `TypeError`: Raised immediately when Python arguments cannot be converted to the bound C++ types, for example an invalid enum, a non-sequence list argument, or a non-integer plan index.

- `ValueError`: Raised for frontend validation errors that map to `std::invalid_argument`, such as missing tensor attributes, shape-deduction failures, invalid tensor names, or invalid variant-pack metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised when a well-formed graph or plan candidate is not supported by the selected heuristics, backend implementation, architecture, cuDNN version, or graph format.

- `RuntimeError`: Raised for cuDNN backend API failures, CUDA API failures, handle errors, graph execution failures, invalid values reported by the frontend, NVRTC compilation failures, or invalid device state.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
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

Call this after `build_operation_graph()`. Some engines may not expose knobs for the current operation graph or may be rejected by the backend; custom-plan code should handle errors per engine.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1349) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L516).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/get_knobs_for_engine_basic.py)

<Code
  src="../../samples/python/get_knobs_for_engine_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::get_knobs_for_engine` reference](/api/cpp/get-knobs-for-engine) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
