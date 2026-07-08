# GEMM + dsReLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Block-scaled GEMM + dsReLU backward fusion**: A persistent, batched dense GEMM on NVIDIA Blackwell GPUs (SM100+) that supports block-scaled FP4 and FP8 inputs and produces both the backward output `D` and the probability gradient `dprob` in a single kernel launch.

- **Inputs**: quantized `A` and `B`, the forward/intermediate tensor `C`, scale-factor tensors `SFA` and `SFB`, and a per-row probability tensor `prob`
- **Outputs**: backward output `D`, probability gradient `dprob`, and optional `Amax`

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

As with the forward `srelu` kernel, FP4 input written to fp16/bf16/fp32 `D`
can emit `Amax`. The FP8 `D`/SFD epilogue is reserved by the API but is not
implemented by the current kernel.

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
                                              v
                                             Amax
```

---

## API Usage

The existing APIs in this section use Torch tensors. The JAX API is described
separately below because its public arrays are row-major and name their logical
axis order explicitly.

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

### JAX API

```python
import jax.numpy as jnp
from cudnn.jax import gemm_dsrelu_wrapper_sm100

result = gemm_dsrelu_wrapper_sm100(
    a,                       # (L, M, K), float4_e2m1fn or FP8
    b,                       # (L, N, K), same dtype as A
    c,                       # (L, M, N)
    sfa,                     # (L, ceil(M/128), rest_k, 32, 4, 4)
    sfb,                     # (L, ceil(N/128), rest_k, 32, 4, 4)
    prob,                    # (L, 1, M), float32
    d_dtype=jnp.bfloat16,
    sf_vec_size=16,          # use 32 for MXFP8 + E8M0 scales
)
```

`GemmDsreluSm100` provides the corresponding class API and accepts optional
`sample_d` and `sample_dprob` output exemplars. The high-level wrapper is JIT
compiled and infers its outputs.

JAX arrays use compact row-major storage:

- `a_layout="LMK"` and `"LKM"` map public A shapes `(L,M,K)` and `(L,K,M)`
  to kernel axes `(M,K,L)`.
- `b_layout="LNK"` and `"LKN"` map public B shapes `(L,N,K)` and `(L,K,N)`
  to kernel axes `(N,K,L)`.
- `d_layout="LMN"` consumes C and produces D as `(L,M,N)`; `"LNM"` uses
  `(L,N,M)`.
- SFA/SFB use public shape `(L, tiles, rest, 32, 4, 4)`. The adapter maps
  this row-major representation to the packed six-dimensional kernel ABI.
- `prob` and the inferred `dprob_tensor` use public shape `(L,1,M)`.

Here `rest_k = ceil(ceil(K / sf_vec_size) / 4)`.

Native JAX `float4_e2m1fn` inputs return an initialized `amax_tensor` with
shape `(1,)`. JAX currently rejects FP8 `D`, `norm_const_tensor`, and SFD
generation because the kernel's SFD epilogue is not implemented. `D` must use
`float16`, `bfloat16`, or `float32`.

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
  - Functional dtypes: `{float16, bfloat16, float32}`
- Output tensor **dprob**: `result["dprob_tensor"]` (wrapper) or `sample_dprob` / `dprob_tensor` (class)
  - Shape: `(M, 1, L)`
  - Dtype: `float32`
- Output tensor **SFD**: `result["sfd_tensor"]` (wrapper) or `sample_sfd` / `sfd_tensor` (class)
  - Shape: `(32, 4, ceil_div(M, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), L)`
  - Dtype: Must match `SFA`
  - Reserved for FP8 `D`; generation is not implemented
- Output tensor **Amax**: `result["amax_tensor"]` (wrapper) or `sample_amax` / `amax_tensor` (class)
  - Shape: `(1,)`
  - Dtype: `float32`
  - Allocated by the wrapper for FP4 input with fp16/bf16/fp32 `D`
- Input tensor **Norm Const**: `norm_const_tensor` (wrapper) or `sample_norm_const` / `norm_const_tensor` (class)
  - Shape: `(1,)`
  - Dtype: `float32`
  - Reserved for FP8 `D`; currently unsupported

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
- FP8 `D` and SFD generation are not implemented

### Environment

- Requires CUDA with SM100+ compute capability
- The JAX API requires a homogeneous local SM100-family GPU and resolves
  occupancy before CuTe lowering. Device-free compilation is not supported.

---

## Usage examples

For end-to-end usage and regression coverage, see:

- `test/python/fe_api/test_gemm_dsrelu.py`
- `test/python/fe_api/test_gemm_dsrelu_utils.py`
- `test/python/fe_api/test_jax_gemm_relu.py`
