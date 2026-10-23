# tensor_like

Create a C++ graph tensor by cloning metadata from another tensor attribute object.

## C++ synopsis

```cpp
auto X_copy = graph->tensor_like(X, "X_copy");
```

## Parameters

- `X` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Existing tensor attributes to copy. Shape, stride, dtype, virtual state, pass-by-value state, ragged-offset metadata, and reordering metadata are copied.

- `name` (`std::string`, default empty): Name assigned to the cloned tensor. The original name is not copied.

## Returns

- `X_copy` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Cloned tensor attributes registered as a graph input. The clone receives a fresh frontend UID because UIDs are internal graph-representation details.

## Errors

`Graph::tensor_like(...)` returns tensor attributes immediately and does not return `error_t`. Invalid or incomplete cloned metadata is reported later by graph lifecycle calls.

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

Use `tensor_like` when separate graph inputs should share the same metadata but have independent variant-pack entries. It intentionally clears the UID and replaces the name.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2395) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3645).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/tensor_like_basic.cc)

<Code
  src="../../samples/cpp/tensor_like_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.tensor_like` reference](/api/python/tensor-like) for Python exceptions, wrapper behavior, and current availability constraints.
