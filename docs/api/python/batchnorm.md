# batchnorm

Append training batch normalization to a PyGraph.

## Python synopsis

```python
Y, mean, inv_variance, out_running_mean, out_running_var = graph.batchnorm(
    input, scale, bias, in_running_mean, in_running_var, epsilon, momentum,
    peer_stats=[], compute_data_type=cudnn.data_type.NOT_SET, name="")
```

## Parameters

- `input` (`cudnn.tensor`): Input tensor. Supported rank, normalized axes, dtype, and layout are selected by the backend plan.

- `scale` (`cudnn.tensor`): Scale/gamma tensor, usually broadcast over non-normalized dimensions.

- `bias` (`cudnn.tensor`): Bias/beta tensor broadcast-compatible with the normalized output.

- `in_running_mean` (`cudnn.tensor`): Running mean tensor read by batch normalization.

- `in_running_var` (`cudnn.tensor`): Running variance tensor read by batch normalization.

- `epsilon` (`cudnn.tensor`): Small numeric constant used in the denominator. Examples commonly use a pass-by-value `FLOAT` tensor.

- `momentum` (`cudnn.tensor`): Running-stat momentum tensor, commonly pass-by-value `FLOAT`.

- `peer_stats` (`list[cudnn.tensor]`, default `[]`): Optional peer statistics tensors for batch-normalization modes that aggregate statistics across peers.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for normalization math. Common mixed-precision examples use `FLOAT` with BF16 or FP16 IO tensors.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Normalized output tensor.

- `mean` (`cudnn.tensor`): Saved mean tensor for backward.

- `inv_variance` (`cudnn.tensor`): Saved inverse variance tensor for backward.

- `out_running_mean` (`cudnn.tensor or None`): Updated running mean when running-stat inputs and momentum are active.

- `out_running_var` (`cudnn.tensor or None`): Updated running variance when running-stat inputs and momentum are active.

## Raises

- `TypeError`: Raised immediately when tensor arguments, enum values, lists such as `peer_stats`, or keyword arguments cannot be converted to the bound C++ types.

- `ValueError`: Raised by `graph.validate()` when tensor metadata, statistics shapes, broadcast shapes, phase-dependent outputs, or saved-stat relationships are inconsistent.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or `graph.build_plans()` when the selected normalization operation, dtype, layout, fusion, architecture, or cuDNN backend plan is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Normalization families**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** Engine-dependent
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** LayerNorm and RMSNorm samples executed in Python and C++; additional C++ normalization samples executed on SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

The operation appends nodes to `graph` and returns graph tensor attributes. It does not execute immediately. Mark tensors that must be materialized with `set_output(True)` and set output dtype, shape, or stride when inference is ambiguous.

Scale and bias tensors often use lower-rank or singleton dimensions for broadcasting. They do not need the same stride as `input`; they need strides that describe their own runtime storage and satisfy the selected backend plan.

Saved statistics that feed a backward API, such as `mean` and `inv_variance`, must be materialized or otherwise preserved when forward and backward are built as separate graphs.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L221) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L24).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API in the current checkout.

## Appendix: C++ equivalent

See the [C++ `Graph::batchnorm` reference](/api/cpp/batchnorm) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
