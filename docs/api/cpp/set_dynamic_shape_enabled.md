# set_dynamic_shape_enabled

Enable or disable dynamic-shape graph behavior.

## C++ synopsis

```cpp
graph.set_dynamic_shape_enabled(true);
```

## Parameters

- `is_enabled` (`bool`, required): `true` enables dynamic-shape graph creation on the cuDNN operation graph. `false` disables it.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_dynamic_shape_enabled()` itself returns `Graph&` and does not return `error_t`. Later validation fails if dynamic shapes are enabled with a cuDNN backend older than 9.4.

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

Dynamic shape must be enabled when the graph uses a `KernelCache`. It should be set before `validate()`, `build_operation_graph()`, `create_execution_plans()`, or the shortcut `build()` sequence.

Dynamic shape does not allocate buffers or change tensor dimensions by itself. Runtime buffers and, where applicable, runtime shape overrides are still provided during workspace queries and execution.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2332) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3527).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_dynamic_shape_enabled()`. Use the `cudnn.pygraph(is_dynamic_shape_enabled=True)` constructor argument, or pass `kernel_cache` to the constructor when available.
