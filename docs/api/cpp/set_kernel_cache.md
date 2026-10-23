# set_kernel_cache

Attach a cuDNN frontend kernel cache to a C++ graph.

## C++ synopsis

```cpp
auto cache = std::make_shared<fe::KernelCache>();
graph.set_dynamic_shape_enabled(true)
     .set_kernel_cache(cache);
```

## Parameters

- `cache` (`std::shared_ptr<fe::KernelCache>`, required): Kernel cache object stored by the graph. `KernelCache` is defined by the C++ frontend backend descriptor wrapper in `include/cudnn_frontend/backend/kernel_cache.h`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_kernel_cache()` itself returns `Graph&` and does not return `error_t`. The graph validation path reports configuration errors later. In particular, `validate()` rejects kernel caching when dynamic shape is disabled, and rejects dynamic shapes or kernel caching when the runtime cuDNN backend version is older than 9.4.

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

Kernel caching is a dynamic-shape feature. Call `set_dynamic_shape_enabled(true)` before validating and building the graph.

The cache is passed to execution-plan construction and is built from the cuDNN operation graph during `build_plans()` when dynamic shapes are enabled and the cache has not already been finalized. The graph keeps a `shared_ptr`, so the cache can be shared with other graphs or kept by the caller for reuse.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2340) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3541).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_kernel_cache()`. The Python `cudnn.pygraph()` constructor accepts a `kernel_cache` argument in this repository and enables dynamic shape on the underlying C++ graph when it is provided.
