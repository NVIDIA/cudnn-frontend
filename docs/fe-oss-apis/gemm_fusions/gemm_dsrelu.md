# GEMM + dsReLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Block-scaled GEMM + dsReLU backward fusion**: A persistent, batched dense GEMM on NVIDIA Blackwell GPUs (SM100+) that supports block-scaled FP4 and FP8 inputs and produces both the backward output `D` and the probability gradient `dprob` in a single kernel launch.

- **Inputs**: quantized `A` and `B`, the forward/intermediate tensor `C`, scale-factor tensors `SFA` and `SFB`, and a per-row probability tensor `prob`
- **Outputs**: backward output `D`, probability gradient `dprob`, and optional output scale factors `SFD` / `Amax`

### Shapes

- **Inputs**
  - `A`: shape `(M, K, L)`
  - `B`: shape `(N, K, L)`
  - `C`: shape `(M, N, L)`
  - `SFA`: shape `(32, 4, ceil_div(M, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - `SFB`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - `prob`: shape `(M, 1, L)`

- **Outputs**
  - `D`: shape `(M, N, L)`
  - `dprob`: shape `(M, 1, L)`
  - `SFD`: shape `(32, 4, ceil_div(M, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), L)` when `D` is FP8
  - `Amax`: shape `(1,)` when FP4 input is written to fp16/bf16/fp32 output

`L` is the batch dimension.

### Equations

Let `A_hat` and `B_hat` denote the dequantized inputs from `(A, SFA)` and `(B, SFB)`.

$$
G[m, n, l] = \alpha \sum_k A\_hat[m, k, l] \, B\_hat[n, k, l]
$$

$$
D[m, n, l] = \mathrm{prob}[m, 0, l] \cdot 2 \cdot C[m, n, l] \cdot \mathrm{relu}(G[m, n, l])
$$

$$
\mathrm{dprob}[m, 0, l] = \sum_n C[m, n, l] \cdot \mathrm{relu}(G[m, n, l])^2
$$

As with the forward `srelu` kernel, FP8 `D` also emits `SFD` using the provided `norm_const_tensor`, and FP4 input written to fp16/bf16/fp32 `D` can emit `Amax`.

### Diagram

```text
A (MxKxL), SFA                   B (NxKxL), SFB
     |  dequantize                    |  dequantize
     v                                v
   A_hat                           B_hat
          \__ GEMM over K ___________________
                                             \
                                              G (MxNxL)
                                              |
                        C (MxNxL) ------------+
                                              |
                                              | relu(G), 2*C*prob
                                              +--> D (MxNxL)
                                              |
                                              +--> dprob (Mx1xL)
                                              |
                                  +-----------+-----------+
                                  |                       |
                                  v                       v
                                 SFD                    Amax
```

---

## API Usage

### High-level wrapper

```python
from cudnn import gemm_dsrelu_wrapper_sm100

result = gemm_dsrelu_wrapper_sm100(
    a_tensor,
    b_tensor,
    c_tensor,
    sfa_tensor,
    sfb_tensor,
    prob_tensor,
    alpha=1.0,
    d_major="n",
    d_dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    norm_const_tensor=None,
    sf_vec_size=16,
    vector_f32=False,
    stream=None,
)

d, dprob, amax, sfd = result
```

### Class API

```python
from cudnn import GemmDsreluSm100

op = GemmDsreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d=d,
    sample_dprob=dprob,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_prob=prob,
    sample_sfd=sfd,
    sample_amax=amax,
    sample_norm_const=norm_const,
    alpha=1.0,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
)
assert op.check_support()
op.compile()
op.execute(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
    d_tensor=d,
    dprob_tensor=dprob,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    prob_tensor=prob,
    sfd_tensor=sfd,
    amax_tensor=amax,
    norm_const_tensor=norm_const,
    current_stream=None,
)
```

---

## Parameters

### Input/Output tensors

- Input tensor **A**: `a_tensor` (wrapper) or `sample_a` / `a_tensor` (class)
  - Shape: `(M, K, L)`
  - Dtype: `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`
  - Note: `uint8` is interpreted as packed `float4_e2m1fn_x2` (FP4x2) data, not integer quantization
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b` / `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Dtype: Must match `A`
- Input tensor **C**: `c_tensor` (wrapper) or `sample_c` / `c_tensor` (class)
  - Shape: `(M, N, L)`
  - Dtype: `{float16, bfloat16, float32}`
- Input tensor **SFA**: `sfa_tensor` (wrapper) or `sample_sfa` / `sfa_tensor` (class)
  - Shape: `(32, 4, ceil_div(M, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn}`
- Input tensor **SFB**: `sfb_tensor` (wrapper) or `sample_sfb` / `sfb_tensor` (class)
  - Shape: `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - Dtype: Must match `SFA`
- Input tensor **prob**: `prob_tensor` (wrapper) or `sample_prob` / `prob_tensor` (class)
  - Shape: `(M, 1, L)`
  - Dtype: `float32`
- Output tensor **D**: `result["d_tensor"]` (wrapper) or `sample_d` / `d_tensor` (class)
  - Shape: `(M, N, L)`
  - Dtype: `{float16, bfloat16, float32, float8_e4m3fn, float8_e5m2}`
- Output tensor **dprob**: `result["dprob_tensor"]` (wrapper) or `sample_dprob` / `dprob_tensor` (class)
  - Shape: `(M, 1, L)`
  - Dtype: `float32`
- Output tensor **SFD**: `result["sfd_tensor"]` (wrapper) or `sample_sfd` / `sfd_tensor` (class)
  - Shape: `(32, 4, ceil_div(M, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), L)`
  - Dtype: Must match `SFA`
  - Required when `D` is FP8
- Output tensor **Amax**: `result["amax_tensor"]` (wrapper) or `sample_amax` / `amax_tensor` (class)
  - Shape: `(1,)`
  - Dtype: `float32`
  - Allocated by the wrapper for FP4 input with fp16/bf16/fp32 `D`
- Input tensor **Norm Const**: `norm_const_tensor` (wrapper) or `sample_norm_const` / `norm_const_tensor` (class)
  - Shape: `(1,)`
  - Dtype: `float32`
  - Required when `D` is FP8

### Common parameters

- `alpha: float`
  - Scalar multiplier applied to the GEMM result before the dsReLU backward epilogue. Default: `1.0`
- `acc_dtype: torch.dtype`
  - Accumulator dtype. Only `torch.float32` is supported
- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`
  - `TILE_M ∈ {128, 256}`
  - `TILE_N ∈ {64, 128, 192, 256}`
- `cluster_shape_mn: Tuple[int, int] | None`
  - Thread-block cluster shape
  - Default: `(2, 1)` when `TILE_M == 256`, else `(1, 1)`
- `sf_vec_size: int`
  - Scale-factor vector size. Allowed values: `{16, 32}`
- `vector_f32: bool`
  - Enables vectorized f32 operations for supported configurations
- `d_major: str` (wrapper only)
  - Output layout for both `C` and `D`
  - Must be either `"m"` or `"n"`
- CUDA stream (`current_stream` in class API, `stream` in wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `d_tensor`
- `dprob_tensor`
- `amax_tensor`
- `sfd_tensor`

Tuple unpacking order is: `(d_tensor, dprob_tensor, amax_tensor, sfd_tensor)`.

---

## Support surface and constraints

### Layouts

- `A` may be `m`-major or `k`-major
- `B` may be `n`-major or `k`-major
- `C` and `D` must share the same layout
- The wrapper exposes the output layout as `d_major ∈ {"m", "n"}`

### Dtypes

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, and `SFD` must have the same dtype
- `sf_vec_size == 32` is unsupported with `sf_dtype == float8_e4m3fn`
- FP8 input requires `sf_vec_size == 32`
- FP4 input with FP8 `D` is unsupported
- FP8 `D` requires both `SFD` and `norm_const_tensor`

### Environment

- Requires CUDA with SM100+ compute capability

---

## Usage examples

For end-to-end usage and regression coverage, see:

- `test/python/fe_api/test_gemm_dsrelu.py`
- `test/python/fe_api/test_gemm_dsrelu_utils.py`
