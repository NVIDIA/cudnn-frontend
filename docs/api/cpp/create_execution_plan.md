# create_execution_plan

Create a single execution-plan candidate from an explicit backend engine ID and knob values.

## C++ synopsis

```cpp
std::unordered_map<fe::KnobType_t, int64_t> knobs;
auto status = graph.create_execution_plan(engine_id, knobs);
```

## Parameters

- `engine_id` (`int64_t`, required): Global backend engine ID.

- `knobs` (`std::unordered_map<fe::KnobType_t, int64_t>`, required): Knob values used to create the engine config.

## Returns

- `status` (`fe::error_t`): `OK` when the engine, knob choices, and engine config are created and enqueued.

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

The operation graph must already be built. Backend engines or knob choices can be rejected, so custom-plan loops usually continue after a bad status.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1374) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3401).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/create_execution_plan_basic.cc)

<Code
  src="../../samples/cpp/create_execution_plan_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.create_execution_plan` reference](/api/python/create-execution-plan) for Python exceptions, wrapper behavior, and current availability constraints.
