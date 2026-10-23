# set_sm_version

Set the target GPU SM version used by support checks and OSS engine paths.

## C++ synopsis

```cpp
graph.set_sm_version(90);
```

## Parameters

- `version` (`int32_t`, required): Target SM architecture version, such as `80`, `89`, `90`, or `100`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_sm_version()` itself returns `Graph&` and does not return `error_t`. Unsupported architecture choices are reported later by `check_support()` or plan-building calls.

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

When no target SM version is set, `check_support()` attempts to populate it from the current CUDA device. Setting the value explicitly is useful for ahead-of-time flows or for checking support against a target architecture.

This setting is distinct from `set_sm_count()`: SM version selects the architecture family, while SM count describes the amount of hardware.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2338) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3559).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_sm_version()`. Use the `cudnn.pygraph(sm_version=...)` constructor argument.
