# rope_backward

Append a rotary positional embedding (RoPE) backward operation to a PyGraph.

## Python synopsis

```python
dX = graph.rope_backward(
    dY,
    freqs,
    output_scale=1.0,
    rope_dim=0,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `dY` (`cudnn.tensor`, required): Upstream gradient of the RoPE output. Its logical shape should match the corresponding forward `rope` output.

- `freqs` (`cudnn.tensor`, required): Frequency tensor used by the matching forward operation. It should have the same metadata convention as the forward `freqs` tensor.

- `output_scale` (`float`, default `1.0`): Scale multiplier. Use the same value as the matching forward `rope` call for correct gradient flow.

- `rope_dim` (`int`, default `0`): Number of rotary dimensions. Use the same value as the matching forward `rope` call.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for the transform.

- `name` (`str`, default `""`): Operation name.

## Returns

- `dX` (`cudnn.tensor`): Gradient tensor with respect to the forward `input`. The C++ graph marks this output non-virtual by default.

## Raises

- `TypeError`: Raised immediately for invalid tensor, scalar, enum, or name arguments.

- `ValueError`: Raised by `graph.validate()` for incompatible shapes, frequency metadata, or output metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the RoPE backward dtype, layout, rank, architecture, or fusion is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: SDPA and RoPE**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** Standalone sample requires cuDNN 9.24+; syntax-checked with cuDNN 9.15.1
- **Family release status:** Stable with option-specific constraints
- **Family cuDNN:** Option and implementation dependent
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** SM80+ with narrower limits by implementation
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** SDPA, masking, cache, dropout, and selected backward samples executed on SM100; standalone RoPE samples were compiled or syntax-checked because they require cuDNN 9.24+
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

When RoPE feeds an SDPA graph, backward often computes gradients with `sdpa_backward` first and then maps the rotated `dQ` or `dK` tensors back through `rope_backward`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L336) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L365).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/rope_backward_basic.py)

<Code
  src="../../samples/python/rope_backward_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::rope_backward` reference](/api/cpp/rope-backward) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
