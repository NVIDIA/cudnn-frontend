# get_knobs_for_engine

Return the tunable knob descriptors for a backend engine.

## C++ synopsis

```cpp
std::vector<fe::Knob> knobs;
auto status = graph.get_knobs_for_engine(engine_id, knobs);
```

## Parameters

- `engine_id` (`int64_t`, required): Global backend engine ID.

- `knobs` (`std::vector<fe::Knob>&`, output): Receives knob descriptors.

## Returns

- `status` (`fe::error_t`): `OK` when knob descriptors are queried successfully.

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

A bad status for one engine does not imply the graph is invalid. Custom-plan loops commonly skip that engine and continue.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2621) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3333).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/get_knobs_for_engine_basic.cc)

<Code
  src="../../samples/cpp/get_knobs_for_engine_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.get_knobs_for_engine` reference](/api/python/get-knobs-for-engine) for Python exceptions, wrapper behavior, and current availability constraints.
