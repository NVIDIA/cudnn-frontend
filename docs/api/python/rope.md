# rope

Append a rotary positional embedding (RoPE) forward operation to a PyGraph.

## Python synopsis

```python
Y = graph.rope(
    input,
    freqs,
    output_scale=1.0,
    rope_dim=0,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Tensor to rotate. Attention examples use shapes such as `(B, H, S, D)`.

- `freqs` (`cudnn.tensor`, required): Frequency tensor. Repository tests use `FLOAT` tensors broadcastable to the input, for example shape `(S, 1, 1, D)` with stride `(D, D, D, 1)`.

- `output_scale` (`float`, default `1.0`): Scalar multiplied into the RoPE output.

- `rope_dim` (`int`, default `0`): Number of rotary dimensions. `0` means use the full input head dimension. When smaller than the head dimension, the backend rotates the last `rope_dim` dimensions and passes through the remaining leading dimensions with scaling.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for the transform.

- `name` (`str`, default `""`): Operation name.

## Returns

- `Y` (`cudnn.tensor`): RoPE output tensor. It has the same logical shape as `input` unless backend attributes specify otherwise.

## Raises

- `TypeError`: Raised immediately for invalid tensor, scalar, enum, or name arguments.

- `ValueError`: Raised by `graph.validate()` for incompatible shapes, frequency metadata, or output metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the RoPE dtype, layout, rank, architecture, or fusion is unsupported.

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

RoPE outputs are frequently fed directly into `sdpa`. In that case, the rotated query and key tensors must still satisfy the SDPA shape, dtype, stride, and head-dimension constraints.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L327) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/norm.cpp#L348).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/rope_basic.py)

<Code
  src="../../samples/python/rope_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::rope` reference](/api/cpp/rope) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
