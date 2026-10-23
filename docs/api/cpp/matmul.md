# matmul

Append a matrix multiplication operation to a C++ frontend graph.

## C++ synopsis

```cpp
std::shared_ptr<fe::graph::Tensor_attributes>
fe::graph::Graph::matmul(
    std::shared_ptr<fe::graph::Tensor_attributes> A,
    std::shared_ptr<fe::graph::Tensor_attributes> B,
    fe::graph::Matmul_attributes attributes);
```

Typical use:

```cpp
auto C = graph->matmul(
    A,
    B,
    fe::graph::Matmul_attributes()
        .set_name("matmul")
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_padding_value(0.0f));
```

## Parameters

- `A` (`std::shared_ptr<Tensor_attributes>`, required): Left input tensor. The
  innermost logical dimension is the contracting `K` dimension; the preceding
  dimension is `M`.

- `B` (`std::shared_ptr<Tensor_attributes>`, required): Right input tensor. The
  second-to-last logical dimension is the contracting `K` dimension and must
  match `A`; the last dimension is `N`.

- `attributes` (`fe::graph::Matmul_attributes`, required): Attribute object
  defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Sets the accumulation and
  compute type. Mixed-precision examples commonly use `FLOAT`.

- `attributes.set_padding_value(float)`: Sets the value used by padded backend
  matmul paths. Dense examples normally use `0.0f`.

- `attributes.set_name(std::string)`: Sets the operation name.

## Returns

- `C` (`std::shared_ptr<Tensor_attributes>`): Output tensor attributes for the
  matmul result. The inferred logical shape is usually the broadcast batch shape
  followed by `(M, N)`. Mark it as an output and set the desired IO dtype when it
  must be written by `execute()`.

## Errors

`Graph::matmul` appends the node and returns tensor attributes. Shape, dtype,
layout, and support failures are reported later through lifecycle `error_t`
statuses from `validate`, `build_operation_graph`, `check_support`, or
`build_plans`.

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

`A`, `B`, and `C` do not need identical strides. Each tensor must use a layout
supported by the selected engine. Output dtype is independent from compute dtype.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L464) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/matmul.h#L202).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/matmul_basic.cc)

<Code
  src="../../samples/cpp/matmul_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.matmul` reference](/api/python/matmul) for Python exceptions, wrapper behavior, and current availability constraints.
