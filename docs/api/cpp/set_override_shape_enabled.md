# set_override_shape_enabled

Enable or disable runtime shape override support for a graph.

## C++ synopsis

```cpp
graph.set_override_shape_enabled(true);
```

## Parameters

- `is_enabled` (`bool`, required): `true` marks the cuDNN operation graph as accepting runtime tensor shape or stride overrides. `false` disables that mode.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_override_shape_enabled()` itself returns `Graph&` and does not return `error_t`. Runtime override mistakes are reported by later lifecycle calls, workspace queries, or execution. For example, the C++ overloads that query workspace with `override_uids`, `override_shapes`, and `override_strides` require matching vector lengths and require cuDNN backend support for runtime override queries.

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

Set this option before building the operation graph. Override shapes are passed later to overloads such as `get_workspace_size(handle, ..., override_uids, override_shapes, override_strides)` and `execute(..., override_uids, override_shapes, override_strides)`.

This flag does not replace tensor metadata. The graph still needs enough static tensor information to validate and select plans.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2334) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3534).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_override_shape_enabled()`. Use the `cudnn.pygraph(is_override_shape_enabled=True)` constructor argument.
