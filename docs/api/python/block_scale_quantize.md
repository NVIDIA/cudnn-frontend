# block_scale_quantize

Append a block-scale quantization operation to a PyGraph.

## Python synopsis

```python
Y, scale = graph.block_scale_quantize(
    input,
    block_size,
    axis=None,
    transpose=False,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Tensor to quantize. The input tensor keeps its own shape, stride, and dtype metadata; the selected backend plan determines which low-precision source and output layouts are supported.

- `block_size` (`int`, required): Number of elements per quantization block along `axis`. Repository tests commonly use `32` for FP8 block-scale paths and `16` for FP4 paths.

- `axis` (`int or None`, default `None`): Axis along which quantization blocks are formed. Negative axis values follow Python indexing conventions. `None` leaves the axis unset on `Block_scale_quantize_attributes`.

- `transpose` (`bool`, default `False`): Requests a transposed quantized output layout when the backend supports that layout. This changes how the output and generated scale tensor should be interpreted, not the input tensor metadata.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for scale generation and conversion. If `NOT_SET`, the binding does not set an explicit compute type on the C++ attributes.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Quantized output graph tensor. Mark it as an output and set its dtype, shape, stride, or reordering metadata when inference is ambiguous or when a specific FP8/FP4 storage layout is required.

- `scale` (`cudnn.tensor`): Generated block-scale graph tensor. It must be materialized with `set_output(True)` if host code or a later separately built graph needs the scale factors. Examples using MXFP8 usually set the scale dtype to `FP8_E8M0` and may set `reordering_type=cudnn.tensor_reordering.F8_128x4`.

## Raises

- `TypeError`: Raised immediately when `input`, `block_size`, `axis`, `transpose`, `compute_data_type`, or `name` cannot be converted to the bound C++ types.

- `ValueError`: Raised by `graph.validate()` for unsupported block metadata, inconsistent output metadata, incompatible dimensions, or malformed tensor attributes.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or `graph.build_plans()` when the requested block-scale quantization dtype, layout, reordering, architecture, cuDNN version, or backend engine is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: FP8, MXFP8, NVFP4, and block scaling**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** See local constraints
- **Family release status:** Stable or experimental by operation
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** CUDA 12 or later; CUDA 13 used for this checkout
- **Family GPU:** Hopper for FP8; Blackwell for MXFP8/NVFP4 and SM100 kernels
- **Family platforms:** Linux (partially executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** C++ FP8 and MXFP8 attention samples executed on SM100; unsupported block-scaling paths are reported locally
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

`block_scale_quantize` appends a graph node and returns graph tensor attributes; it does not execute or allocate storage. Runtime buffers are supplied later through the graph execution variant pack.

The output tensor and scale tensor do not need to have the same dtype or stride as `input`. They need metadata that describes their own runtime storage and is accepted by the selected backend plan.

Block-scale quantization is layout-sensitive. Document `axis`, `block_size`, scale tensor dimensions, strides, and reordering together in examples because those values define how blocks map to the scale tensor.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1183) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L308).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MXFP8 block-scaled matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/block_scaled_matmul.py), which dequantizes two inputs, performs a matmul, and block-quantizes the result.

## Appendix: C++ equivalent

See the [C++ `Graph::block_scale_quantize` reference](/api/cpp/block-scale-quantize) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
