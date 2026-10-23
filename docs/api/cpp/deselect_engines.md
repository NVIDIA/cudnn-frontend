# deselect_engines

Remove candidate plans whose backend engine names match barred name prefixes or strings.

## C++ synopsis

```cpp
graph.deselect_engines({"eng4_"});
```

## Parameters

- `engine_names` (`std::vector<std::string>`, required): Engine names or prefixes to bar from the candidate set.

## Returns

- `graph` (`fe::graph::Graph&`): The graph object, enabling fluent chaining.

## Errors

These filter helpers return `Graph&` and do not return `error_t`. Invalid enum or string types are compile-time issues in C++. If filtering leaves no viable candidate, the later `check_support()` or `build_plans()` call reports the failure.

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

The method stores barred names in the plan list; subsequent support checks and builds honor the filter.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2905) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2905).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/deselect_engines_basic.cc)

<Code
  src="../../samples/cpp/deselect_engines_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.deselect_engines` reference](/api/python/deselect-engines) for Python exceptions, wrapper behavior, and current availability constraints.
