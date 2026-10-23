# set_device_properties

Attach explicit device properties for heuristic query and deviceless build flows.

## C++ synopsis

```cpp
auto device_properties = std::make_shared<fe::DeviceProperties>();
device_properties->set_device_id(0);
graph.set_device_properties(device_properties);
```

## Parameters

- `device_prop` (`std::shared_ptr<const fe::DeviceProperties>`, required): Device properties descriptor wrapper. `DeviceProperties` is defined in `include/cudnn_frontend/backend/device_properties.h`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_device_properties()` itself returns `Graph&` and does not return `error_t`. Backend descriptor initialization or compatibility problems are reported later by heuristic queries and build calls. `DeviceProperties::status()` reports a backend descriptor failure when the feature is unavailable in the linked cuDNN version.

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

Device properties are passed to the cuDNN heuristic query path. They are useful for ahead-of-time compilation or when planning for a device described by a descriptor rather than the current CUDA handle.

When device properties are set, the no-handle graph build overloads can be used for deviceless planning where supported by the selected operations and cuDNN version.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2146) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3547).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_device_properties()`. Use the `cudnn.pygraph(device_property=...)` constructor argument when the Python binding exposes a `DeviceProperties` object.
