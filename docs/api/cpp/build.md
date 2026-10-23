# build

Run the common graph build sequence as a shortcut.

## C++ synopsis

```cpp
auto status = graph.build(handle, {fe::HeurMode_t::A});
auto aot_status = graph.build({fe::HeurMode_t::A});
```

## Parameters

- `handle` (`cudnnHandle_t`, optional overload): cuDNN handle for ordinary backend graph-representation creation.

- `modes` (`std::vector<fe::HeurMode_t>`, required): Heuristic modes.

- `policy` (`fe::BuildPlanPolicy_t`, default `HEURISTICS_CHOICE`): Plan build policy.

- `do_multithreaded_builds` (`bool`, default `false`): Whether to build plans in parallel where supported.

## Returns

- `status` (`fe::error_t`): `OK` when validate, operation graph creation, heuristics query, support check, and plan build all succeed.

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

The no-handle C++ overload is intended for deviceless ahead-of-time flows with device properties configured on the graph.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2654) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3474).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/build_basic.cc)

<Code
  src="../../samples/cpp/build_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.build` reference](/api/python/build) for Python exceptions, wrapper behavior, and current availability constraints.
