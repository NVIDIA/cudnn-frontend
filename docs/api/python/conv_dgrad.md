# conv_dgrad

Append a data-gradient convolution operation to a PyGraph.

## Python synopsis

```python
dX = graph.conv_dgrad(
    loss,
    filter,
    padding=[],
    stride=[],
    dilation=[],
    convolution_mode=cudnn.convolution_mode.CROSS_CORRELATION,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)

dX = graph.conv_dgrad(
    loss,
    filter,
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

- `loss` (`cudnn.tensor`, required): Upstream output-gradient tensor, commonly named `dY`.

- `filter` (`cudnn.tensor`, required): Forward filter tensor. Its channel and spatial dimensions must be compatible with `loss` and the requested input-gradient tensor metadata.

- `padding` (`list[int]`, default `[]`): Symmetric spatial padding. The list length should match the number of spatial dimensions. Use this overload instead of `pre_padding` and `post_padding`.

- `pre_padding` and `post_padding` (`list[int]`, default `[]`): Asymmetric spatial padding before and after each spatial dimension.

- `stride` (`list[int]`, default `[]`): Spatial stride from the corresponding forward convolution.

- `dilation` (`list[int]`, default `[]`): Spatial dilation from the corresponding forward convolution.

- `convolution_mode` (`cudnn.convolution_mode`, default `CROSS_CORRELATION`): cuDNN convolution mode forwarded to `Conv_dgrad_attributes`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type. Mixed-precision examples commonly use `FLOAT` with FP16 or BF16 tensors.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `dX` (`cudnn.tensor`): Input-gradient graph tensor. Data-gradient examples often set `dX` output dimensions and strides explicitly because the frontend may not infer the intended input shape from `loss` and `filter` alone.

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

`conv_dgrad` appends a graph node and returns a graph tensor attribute; it does not execute immediately.

`loss`, `filter`, and `dX` do not need identical strides. Set `dX` stride to the runtime layout of the input-gradient buffer.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1095) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L245).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/conv_dgrad_basic.py)

<Code
  src="../../samples/python/conv_dgrad_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::conv_dgrad` reference](/api/cpp/conv-dgrad) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
