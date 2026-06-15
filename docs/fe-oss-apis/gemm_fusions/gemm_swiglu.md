# GEMM + SwiGLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**GEMM + SwiGLU fusion**: A persistent, batched dense GEMM fused with a SwiGLU epilogue on NVIDIA Blackwell GPUs (SM100+), implemented with CUTLASS/CUTE. It produces both the full GEMM output `AB12` and a SwiGLU-projected tensor `C` in a single pass.

This API supports two modes:
1. **Standard mode**: High-precision GEMM with SwiGLU epilogue
2. **Quantized mode** (block-scaled): Low-precision GEMM using block scaling supporting FP4 and FP8 data types

### Shapes

- Inputs:
  - `A`: shape `(M, K, L)`
  - `B`: shape `(N, K, L)`
- Outputs:
  - `AB12`: shape `(M, N, L)` – full GEMM result
  - `C`: shape `(M, N/2, L)` – SwiGLU-projected result

    `L` is the batch dimension.

### Equations

- GEMM (per batch l):

$$
AB12[m, n, l] = \alpha \sum_{k} A[m, k, l] \, B[n, k, l]
$$

- SwiGLU epilogue (performed by pairing 32-column blocks along `N`):

  Let block size `G = 32`. For each pair of consecutive 32-wide column blocks in `AB12`:
  - Input block:  `X_b = AB12[:, 2*b*G : 2*b*G + G, :]`
  - Gate block:   `G_b = AB12[:, 2*b*G + G : 2*b*G + 2*G, :]`

$$
C[:, \, bG:(b+1)G, \, :] = X_b \cdot \operatorname{swish}(G_b), \quad
\operatorname{swish}(x) = x \cdot \sigma(x)
$$

Notes:
- The `alpha` scaling is applied before the SwiGLU; both `X_b` and `G_b` are from the scaled GEMM results.
- `AB12` stores the entire scaled GEMM output (both input and gate blocks), while `C` stores the fused SwiGLU-projected result with half the columns.
- **N divisibility requirement**: `N` must be divisible by 64 (two consecutive 32-column blocks) to ensure proper pairing for the SwiGLU operation.

### Diagram

```text
 A (MxKxL)     B (NxKxL)
      |              |
      \__ GEMM (per L): AB12 = alpha * A @ B  ______________________
                            AB12 (MxNxL)                            \
                            |                                        \
                            |  Pair 32-col blocks along N:           |
                            |   [X0 | G0 | X1 | G1 | ...]           |
                            |    |     |    |     |                  |
                            |    \_swish(G_b)<____/                  |
                            |           |                            |
                            \___ C[:, b*32:(b+1)*32, :] = X_b * swish(G_b)
                                              C (MxN/2xL)
```

---

## API Usage

### High-level wrapper (Standard Mode)

```python
result = gemm_swiglu_wrapper_sm100(
    a_tensor,
    b_tensor,
    alpha=1.0,
    c_major="m",
    ab12_dtype=torch.float32,
    c_dtype=torch.float16,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    stream=None,
)
ab12, c, sfc, amax = result
# Note: sfc and amax are always None in standard mode
# Key access: result["ab12_tensor"], result["c_tensor"]
```

### High-level wrapper (Quantized Mode)

When scale factor tensors are provided, the wrapper uses the block-scaled quantized kernel.

```python
result = gemm_swiglu_wrapper_sm100(
    a_tensor,
    b_tensor,
    alpha=1.0,
    c_major="m",
    ab12_dtype=torch.bfloat16,
    c_dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    # Quantization parameters
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    norm_const_tensor=norm_const_tensor,  # Required when c_dtype is fp8
    sf_vec_size=16,
    vector_f32=False,
    ab12_stages=4,
    stream=None,
)
ab12, c, sfc, amax = result
# Key access: result["ab12_tensor"], result["c_tensor"], result["sfc_tensor"], result["amax_tensor"]
```

### Class API (Standard Mode)

```python
gemm = GemmSwigluSm100(
    sample_a,
    sample_b,
    sample_ab12,
    sample_c,
    alpha=1.0,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=None,
)
assert gemm.check_support()
gemm.compile()
gemm.execute(
    a_tensor,
    b_tensor,
    ab12_tensor,
    c_tensor,
    alpha=1.0,
    current_stream=None,
)
```

### Class API (Quantized Mode)

```python
gemm = GemmSwigluSm100(
    sample_a,
    sample_b,
    sample_ab12,
    sample_c,
    alpha=1.0,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=None,
    # Quantization parameters
    sample_sfa=sample_sfa,
    sample_sfb=sample_sfb,
    sample_amax=sample_amax,          # Required for fp4 inputs with bf16 output
    sample_sfc=sample_sfc,            # Required when c_dtype is fp8
    sample_norm_const=sample_norm_const,  # Required when c_dtype is fp8
    sf_vec_size=16,
    vector_f32=False,
    ab12_stages=4,
)
assert gemm.check_support()
gemm.compile()
gemm.execute(
    a_tensor,
    b_tensor,
    ab12_tensor,
    c_tensor,
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    amax_tensor=amax_tensor,
    sfc_tensor=sfc_tensor,
    norm_const_tensor=norm_const_tensor,
    alpha=1.0,
    current_stream=None,
)
```

---

## Parameters

### Input/Output tensors

- Input tensor **A**: `a_tensor` (wrapper) or `sample_a`, `a_tensor` (class)
  - Shape: `(M, K, L)`
  - Stride: `(1, M, M·K)` for `m`-major or `(K, 1, M·K)` for `k`-major
    - Quantized mode: Must be `k`-major for FP4 inputs
  - Dtype (`ab_dtype`):
    - Standard mode: `{float16, bfloat16, float32, float8_e4m3fn, float8_e5m2}`
    - Quantized mode: `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`
      - `uint8` is interpreted as packed FP4 (two FP4 values per byte)
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b`, `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Stride: `(1, N, N·K)` for `n`-major or `(K, 1, N·K)` for `k`-major
  - Dtype (`ab_dtype`): Must match `A`
- Output tensor **AB12**: `result["ab12_tensor"]` (wrapper) or `sample_ab12`, `ab12_tensor` (class)
  - Shape: `(M, N, L)`
  - Stride: `(1, M, M·N)` for `m`-major or `(N, 1, M·N)` for `n`-major. Provided as `c_major` argument for wrapper
    - Quantized mode: Must be `n`-major for FP4 outputs
  - Dtype (`ab12_dtype`, provided as `ab12_dtype` argument for wrapper):
    - Standard mode: `{float32, float16, bfloat16}` if `acc_dtype == float32`, `{float16, bfloat16}` if `acc_dtype == float16`
    - Quantized mode: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}`
- Output tensor **C**: `result["c_tensor"]` (wrapper) or `sample_c`, `c_tensor` (class)
  - Shape: `(M, N/2, L)`
  - Stride: `(1, M, M·N/2)` for `m`-major or `(N/2, 1, M·N/2)` for `n`-major. Must match with `AB12`
  - Dtype (`c_dtype`, provided as `c_dtype` argument for wrapper):
    - Standard mode: `{float16, bfloat16}`
    - Quantized mode: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}`
- **Quantization-specific tensors**
  - Input tensor **SFA** (A scale factor): `sfa_tensor` (wrapper) or `sample_sfa`, `sfa_tensor` (class)
    - Shape: `(32, 4, ceil(M/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
    - Dtype: `{float8_e8m0fnu, float8_e4m3fn}`
  - Input tensor **SFB** (B scale factor): `sfb_tensor` (wrapper) or `sample_sfb`, `sfb_tensor` (class)
    - Shape: `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
    - Dtype: Must match `SFA`
  - Output tensor **SFC** (C scale factor, **Optional**): `result["sfc_tensor"]` (wrapper) or `sample_sfc`, `sfc_tensor` (class)
    - Shape: `(32, 4, ceil(M/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), L)`
    - Dtype: Must match `SFA`
    - **Required when**: `c_dtype ∈ {float8_e4m3fn, float8_e5m2}`
  - Output tensor **AMAX** (**Optional**): `result["amax_tensor"]` (wrapper) or `sample_amax`, `amax_tensor` (class)
    - Shape: `(1,)`
    - Dtype: `float32`
    - **Required when**: `ab_dtype` is FP4 and `c_dtype == bfloat16`
  - Input tensor **Norm Const** (**Optional**): `norm_const_tensor` (wrapper) or `sample_norm_const`, `norm_const_tensor` (class)
    - Shape: `(1,)`
    - Dtype: `float32`
    - **Required when**: `c_dtype ∈ {float8_e4m3fn, float8_e5m2}`

### Common parameters

- `alpha: float`
  - Scalar multiplier applied to the GEMM result before SwiGLU.
  - Default: `1.0`
- `acc_dtype: torch.dtype`
  - Accumulator dtype.
  - Standard mode: `{float32, float16}`. Default: `torch.float32`
  - Quantized mode: Must be `float32`
- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`. Default: `(128, 128)`
  - `TILE_M ∈ {128, 256}`
  - Standard mode: `TILE_N ∈ {32, 64, ..., 224, 256}`
  - Quantized mode: `TILE_N ∈ {64, 128, 192, 256}`
- `cluster_shape_mn: Tuple[int, int] | None`
  - Thread Block cluster shape `(CLUSTER_M, CLUSTER_N)`
  - Constraints: positive powers of 2, `CLUSTER_M*CLUSTER_N ≤ 16`.
  - Default: `(1,1)` if `mma_tiler_mn[0] != 256` else `(2,2)`.
- CUDA stream (`current_stream` in class API, `stream` in wrapper)
- **Quantization-specific parameters**
  - `sf_vec_size: int`
    - Scale factor vector size (number of elements per scale factor)
    - Allowed values: `{16, 32}`. Default: `16`
    - Constraints:
      - FP8 inputs require `sf_vec_size=32` with `sf_dtype=float8_e8m0fnu`
      - FP4 inputs do not support `sf_vec_size=32` with `sf_dtype=float8_e4m3fn`
  - `vector_f32: bool`
    - Enable packed f32 operations for improved performance
    - Default: `False`
  - `ab12_stages: int`
    - Number of pipeline stages for AB12 output
    - Default: `4`

### Wrapper-specific parameters: `gemm_swiglu_wrapper_sm100`

- `a_tensor`, `b_tensor`: see Input/Output tensors
- `c_major: str`: see Input/Output tensors. Default: `"n"`
- `ab12_dtype: torch.dtype`: see Input/Output tensors. Default: `torch.float32`
- `c_dtype: torch.dtype`: see Input/Output tensors. Default: `torch.float16`
- `sfa_tensor`, `sfb_tensor`, `norm_const_tensor`: see Quantization-specific tensors
- `sf_vec_size`, `vector_f32`, `ab12_stages`: see Quantization-specific parameters

### Wrapper return values

Returns a `TupleDict` with fixed keys:

- `ab12_tensor`: Intermediate GEMM output
- `c_tensor`: SwiGLU output
- `sfc_tensor`: Output scale factors (or `None` when not applicable)
- `amax_tensor`: Max-abs output (or `None` when not applicable)

Tuple unpacking order is always:
`(ab12_tensor, c_tensor, sfc_tensor, amax_tensor)`.

- **Standard mode**: `sfc_tensor is None` and `amax_tensor is None`
- **Quantized mode**: `sfc_tensor` and/or `amax_tensor` are populated based on dtype/configuration

### Class-specific parameters

#### `GemmSwigluSm100` (constructor)

- `sample_a`, `sample_b`, `sample_ab12`, `sample_c` – see Input/Output tensors
- `sample_sfa`, `sample_sfb`, `sample_sfc`, `sample_amax`, `sample_norm_const` – see Scale factor tensors (quantized mode)

#### `GemmSwigluSm100.execute`

- `a_tensor`, `b_tensor`, `ab12_tensor`, `c_tensor` – see Input/Output tensors. Must have same layout as sample tensors provided in constructor.
- `sfa_tensor`, `sfb_tensor`, `sfc_tensor`, `amax_tensor`, `norm_const_tensor` – see Scale factor tensors (quantized mode)

---

## Support surface and constraints

### Layouts and strides

- `AB12` and `C` must have the same major order.
- `A`, `B`, `AB12` must be 16-byte aligned along the contiguous dimension.
- For FP4 inputs (quantized mode): `A` and `B` must be `k`-major, `AB12` must be `n`-major.

### Dtypes

#### Standard mode

- `A`/`B` must have the same dtype.
- `ab12_dtype ∈ {float8_e4m3fn, float8_e5m2}` is currently disabled
- `acc_dtype == float16` is only supported with `ab_dtype ∈ {float16, float8_e4m3fn, float8_e5m2}`
- `ab12_dtype ∈ {float32}` requires `acc_dtype == float32`

#### Quantized mode

The quantized kernel supports the following configurations:

| Format | ab_dtype | sf_dtype | sf_vec_size | Notes |
|--------|-----------|----------|-------------|-------|
| **MXFP4** | `float4_e2m1fn_x2` or `uint8` | `float8_e8m0fnu` | 16 | Standard MX FP4 |
| **MXFP4** | `float4_e2m1fn_x2` or `uint8` | `float8_e4m3fn` | 16 | NVF4 variant |
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | Standard MX FP8 |

Additional constraints:
- `acc_dtype` must be `float32`
- Not compatible with FP8 c_dtype. BF16 `c_dtype` is expected.
- For MXFP8 inputs, ab12_dtype` should be float16 or bfloat16.
- When `c_dtype ∈ {float8_e4m3fn, float8_e5m2}`: `sfc_tensor` and `norm_const_tensor` are required
- When `ab_dtype` is FP4 and `c_dtype == bfloat16`: `amax_tensor` is required
- `c_dtype` and `ab12_dtype` cannot both be `float32`

### Tiling and cluster

- Using `TILE_M == 256` requires `mma_tiler_mn[0] == 256` (enables 2-CTA instructions).
- If `TILE_M == 128` and `cluster_shape_mn != (1, 1)`, `mma_tiler_mn` must be exactly `(128, 128)`.
- If `mma_tiler_mn[0] == 256`, `CLUSTER_M` must be divisible by 2
- Standard mode: If `mma_tiler_mn[0] != 256`, `cluster_shape_mn` must be `(1, 1)`.

### Environment

- Requires CUDA with SM100+ compute capability

---

## Usage examples

For usage examples, see test cases in `test/python/fe_api/test_gemm_swiglu.py`
