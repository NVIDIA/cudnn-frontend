# moe_grouped_matmul

Append a mixture-of-experts grouped matmul forward operation to a PyGraph.

## Python synopsis

```python
output = graph.moe_grouped_matmul(
    token,
    weight,
    first_token_offset,
    token_index=None,
    token_ks=None,
    mode=cudnn.moe_grouped_matmul_mode.NONE,
    compute_data_type=cudnn.data_type.FLOAT,
    top_k=0,
    name="",
)
```

## Parameters

- `token` (`cudnn.tensor`, required): Token matrix grouped by expert. Repository tests use shape `[1, T, H]` with row-major stride `[T * H, H, 1]`.

- `weight` (`cudnn.tensor`, required): Expert weight tensor. Repository tests use shape `[E, H, N]` and column-major stride `[H * N, 1, H]`.

- `first_token_offset` (`cudnn.tensor`, required): `INT32` tensor that gives the first token offset for each expert. Repository tests use shape `[E, 1, 1]` and stride `[1, 1, 1]`.

- `token_index` (`cudnn.tensor` or `None`, default `None`): INT32 tensor with shape `[1, routed_rows, 1]`. Required for `GATHER` and `SCATTER`; omitted for `NONE`.

- `token_ks` (`cudnn.tensor` or `None`, default `None`): INT32 routing tensor with shape `[1, routed_rows, 1]`. Required for `SCATTER` and omitted for `NONE` or `GATHER`.

- `mode` (`cudnn.moe_grouped_matmul_mode`, default `NONE`): `NONE` consumes rows already grouped by expert. `GATHER` gathers source-token rows through `token_index`. `SCATTER` consumes routing metadata and scatters results using `token_index`, `token_ks`, and `top_k`.

- `compute_data_type` (`cudnn.data_type`, default `FLOAT`): Compute and accumulation type. BF16 examples use `FLOAT`.

- `top_k` (`int`, default `0`): Positive top-k routing value required for `SCATTER`. It is not attached to the backend descriptor for `NONE` or `GATHER`.

- `name` (`str`, default `""`): Operation name.

## Returns

- `output` (`cudnn.tensor`): Forward grouped matmul output. The backend uses `[1, token_index.dim[1], N]` for `GATHER` and `[1, token.dim[1], N]` otherwise, with row-major stride. Before creating the graph representation, the Python IR currently exposes a provisional token-row shape for every mode; do not treat that provisional Gather dimension as authoritative.

## Raises

- `TypeError`: Raised immediately for invalid tensor, enum, integer, or name arguments.

- `ValueError`: Raised by Python argument or graph metadata checks for malformed declared shapes, strides, modes, or output metadata. Runtime values such as offset monotonicity and index ranges are caller-owned data contracts and are not exhaustively inspected by `graph.validate()`.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the dtype, architecture, expert layout, or mode is unsupported.

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

The grouped matmul layout is specialized. Token order and `first_token_offset` define which token rows are multiplied by each expert, so document and validate those metadata together with the runtime token buffers. Gather output row count comes from `token_index`; other modes use the Token row count.

Source reports cuDNN 9.15 as the descriptor minimum. Current Python tests require 9.18 for forward and 9.22 plus cublasLt 13.5 for backward. Use those test gates conservatively until the source conflict is resolved.

The forward sample could not execute with the recorded cuDNN 9.15.1 backend.
Before creating the graph representation, Python IR shape inference exposes a provisional Token row count
for every mode; the backend uses the TokenIndex row count for `GATHER`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1284) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L428).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/moe_grouped_matmul.py). It is adapted from the current numerical test and executes only when the documented backend minimum is present.

## Appendix: C++ equivalent

See the [C++ `Graph::moe_grouped_matmul` reference](/api/cpp/moe-grouped-matmul) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
