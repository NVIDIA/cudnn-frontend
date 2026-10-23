# conv_fprop

Append a forward convolution operation to a PyGraph.

## Python synopsis

```python
Y = graph.conv_fprop(
    image,
    weight,
    padding=[],
    stride=[],
    dilation=[],
    convolution_mode=cudnn.convolution_mode.CROSS_CORRELATION,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)

Y = graph.conv_fprop(
    image,
    weight,
    pre_padding=[],
    post_padding=[],
    stride=[],
    dilation=[],
    convolution_mode=cudnn.convolution_mode.CROSS_CORRELATION,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

The first overload copies `padding` to both pre-padding and post-padding. Use the second overload when the two sides differ.

## Parameters

- `image` (`cudnn.tensor`, required): Input activation tensor. Its rank, channel axis, spatial axes, dtype, and stride must describe a cuDNN convolution layout supported by the selected backend plan.

- `weight` (`cudnn.tensor`, required): Filter tensor. Its output-channel, input-channel, and spatial dimensions must be compatible with `image`, padding, stride, dilation, and convolution mode.

- `padding` (`list[int]`, default `[]`): Symmetric spatial padding. The list length should match the number of spatial dimensions. Use this overload instead of `pre_padding` and `post_padding`.

- `pre_padding` and `post_padding` (`list[int]`, default `[]`): Asymmetric spatial padding before and after each spatial dimension. Use these parameters instead of `padding` when the two sides differ.

- `stride` (`list[int]`, default `[]`): Spatial stride. Empty means the attribute remains empty and backend/default handling applies.

- `dilation` (`list[int]`, default `[]`): Spatial dilation. Empty means the attribute remains empty and backend/default handling applies.

- `convolution_mode` (`cudnn.convolution_mode`, default `CROSS_CORRELATION`): cuDNN convolution mode forwarded to `Conv_fprop_attributes`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type. Mixed-precision examples commonly use `FLOAT` with FP16 or BF16 tensors.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Forward convolution output graph tensor. Mark it as an output and set dtype, shape, or stride if the surrounding graph cannot infer the required metadata.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, enum values, or padding/stride/dilation list types.

- `ValueError`: Raised by `graph.validate()` when convolution dimensions, padding lengths, tensor metadata, or output metadata are inconsistent.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or `graph.build_plans()` when no backend engine supports the requested dtype, layout, architecture, fusion, or convolution attributes.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Convolution forward and gradients**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** See selected engine support
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** Engine-dependent
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Forward, data-gradient, and weight-gradient samples executed in Python and C++ on SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

`conv_fprop` appends a graph node and returns a graph tensor attribute; it does not execute immediately.

`image`, `weight`, and `Y` do not need identical strides. Use strides that describe each tensor's actual runtime memory layout, then let support checking select a compatible engine.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L998) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L222).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/conv_fprop_basic.py)

<Code
  src="../../samples/python/conv_fprop_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::conv_fprop` reference](/api/cpp/conv-fprop) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
