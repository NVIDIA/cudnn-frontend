# block_scale_dequantize

Append a block-scale dequantization operation to a PyGraph.

## Python synopsis

```python
Y = graph.block_scale_dequantize(
    input,
    descale,
    block_size,
    is_negative_scale=False,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Quantized tensor to dequantize. Its shape, stride, dtype, and optional reordering metadata describe the runtime low-precision storage.

- `descale` (`cudnn.tensor`, required): Block-scale tensor applied to `input`. Its shape, stride, dtype, and reordering must match the block layout implied by `block_size` and the selected backend plan.

- `block_size` (`list[int]`, required): Per-axis block sizes used to interpret `descale`. For a matrix input shaped like `[B, M, K]`, examples use `[1, block_size]` when scaling blocks along `K`; for `[B, K, N]`, examples use `[block_size, 1]`.

- `is_negative_scale` (`bool`, default `False`): Indicates whether scale values can be negative. Set this only for formats or workflows that intentionally encode signed scale factors.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for applying the scale and converting the result. If `NOT_SET`, the binding leaves the C++ attribute unset.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Dequantized output graph tensor. Set its dtype explicitly when the downstream operation expects a particular compute or IO type.

## Raises

- `TypeError`: Raised immediately when `input`, `descale`, `block_size`, `is_negative_scale`, `compute_data_type`, or `name` cannot be converted to the bound C++ types.

- `ValueError`: Raised by `graph.validate()` for malformed block sizes, incompatible scale tensor metadata, unsupported output metadata, or inconsistent tensor dimensions.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or `graph.build_plans()` when the requested block-scale dequantization dtype, layout, reordering, architecture, cuDNN version, or backend engine is unsupported.

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

`block_scale_dequantize` appends a graph node and returns a graph tensor attribute. It does not change the memory layout of the input buffer in place; it describes a graph operation whose output tensor has its own metadata.

The `input`, `descale`, and output tensors do not need identical dtype or stride. Each tensor needs metadata that describes its own runtime storage and is compatible with the block layout.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1161) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L289).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MXFP8 block-scaled matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/block_scaled_matmul.py), which uses `F8_128x4`-reordered E8M0 scales to dequantize both matmul inputs.

## Appendix: C++ equivalent

See the [C++ `Graph::block_scale_dequantize` reference](/api/cpp/block-scale-dequantize) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
