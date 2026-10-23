# moe_grouped_matmul_bwd

Append a mixture-of-experts grouped matmul weight-gradient operation to a PyGraph.

## Python synopsis

```python
dweight = graph.moe_grouped_matmul_bwd(
    doutput,
    token,
    first_token_offset,
    compute_data_type=cudnn.data_type.FLOAT,
    name="",
)
```

## Parameters

- `doutput` (`cudnn.tensor`, required): Upstream grouped matmul output gradient. Repository tests use shape `[1, T, N]`.

- `token` (`cudnn.tensor`, required): Forward token tensor grouped by expert. Repository tests use shape `[1, T, H]` with row-major stride `[T * H, H, 1]`.

- `first_token_offset` (`cudnn.tensor`, required): `INT32` tensor that gives the first token offset for each expert. It must match the token grouping used by the forward operation.

- `compute_data_type` (`cudnn.data_type`, default `FLOAT`): Compute and accumulation type. BF16 examples use `FLOAT`.

- `name` (`str`, default `""`): Operation name.

## Returns

- `dweight` (`cudnn.tensor`): Expert weight-gradient tensor. Repository tests infer shape `[E, H, N]` and column-major stride `[H * N, 1, H]`.

## Raises

- `TypeError`: Raised immediately for invalid tensor, enum, or name arguments.

- `ValueError`: Raised by `graph.validate()` for inconsistent expert offsets, shapes, strides, or output metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the dtype, architecture, expert layout, cuDNN version, cublasLt version, or mode is unsupported.

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

Repository tests gate `moe_grouped_matmul_bwd` on cuDNN 9.22.0 or later and cublasLt 13.5 or later. Treat those as requirements unless a local support probe proves otherwise for the installed build.

The operation computes per-expert `dweight` from `token` and `doutput` according to `first_token_offset`; it does not consume the forward `weight` tensor.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1311) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L450).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MoE grouped weight-gradient sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/moe_grouped_matmul_backward.py). It executes the repository test layout and checks every expert against a dense PyTorch reference when the required cuDNN and cublasLt versions are available.

## Appendix: C++ equivalent

See the [C++ `Graph::moe_grouped_matmul_bwd` reference](/api/cpp/moe-grouped-matmul-bwd) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
