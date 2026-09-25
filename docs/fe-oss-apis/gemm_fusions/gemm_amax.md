# GEMM + Amax (SM100)

**This is an experimental API and subject to change.**

## Overview

**Block-scaled GEMM + amax**: A persistent, batched dense GEMM on NVIDIA Blackwell GPUs (SM100+) that supports low-precision inputs (FP8, FP4) with per-block scale factors, producing the full GEMM output `C` and global amax reduction. Implemented with CUTLASS/CUTE.

- **Inputs**: quantized `A` and `B` (FP8 or FP4), and corresponding scale-factor tensors `SFA` and `SFB` that dequantize along the `K` dimension in groups of size `sf_vec_size`.
- **Outputs**: full GEMM result `C` and `Amax`.

### Shapes

- **Inputs**
  - `A`: shape `(M, K, L)`
  - `B`: shape `(N, K, L)`
  - `SFA`: shape `(32, 4, ceil_div(M, 128), 4, ceil_div(K, 4·sf_vec_size), L)`
  - `SFB`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(K, 4·sf_vec_size), L)`

- **Outputs**
  - `C`: shape `(M, N, L)`
  - `Amax`: shape `(1, 1, 1)`

`L` is the batch dimension.

### Equations

Let block size along `K` be `sf_vec_size ∈ {16, 32}`. Dequantization is performed using the provided scale factors for groups of `sf_vec_size` along `K` (per `M`/`N` blocks defined by the atom tiling):

$$
\hat{A}[m, k, l] = \operatorname{dequantize}(A[m, k, l], \text{SFA}, \text{sf_vec_size})
$$

$$
\hat{B}[n, k, l] = \operatorname{dequantize}(B[n, k, l], \text{SFB}, \text{sf_vec_size})
$$

$$
C[m, n, l] = \sum_{k} \hat{A}[m, k, l] \, \hat{B}[n, k, l]
$$

$$
\mathrm{Amax} = \max_{m, n, l} |C[m, n, l]|
$$


### Diagram

```text
A (MxKxL), SFA                   B (NxKxL), SFB
     |  dequantize(.; SFA)            |  dequantize(.; SFB)
     v                                v
   A_hat (MxKxL)                   B_hat (NxKxL)
          \__ GEMM over K ______________________
                                                  \
                                                   C (MxNxL or packed)
                                                   |
                                                   +-- reduce: Amax = max |C|
                                                   |
                                                   v
                                              Amax (1x1x1)
```

## API Usage

The tensor parameters are type-erased: torch tensors and JAX arrays are both accepted. torch is only imported when torch tensors/dtypes are passed, and jax only when JAX arrays are passed. Dtype parameters (`c_dtype`, `acc_dtype`) accept torch dtypes, numpy/ml_dtypes dtypes, dtype name strings, or `cutlass` types.

### High-level wrapper
```python
result = gemm_amax_wrapper_sm100(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    c_major="n",
    c_dtype=torch.float32,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    sf_vec_size=32,
    stream=None,
)
c, amax = result
# Key access: result["c_tensor"], result["amax_tensor"]
```

### Class API
```python
from cuda.bindings import driver as cuda

op = GemmAmaxSm100(
    sample_a=a,
    sample_b=b,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_c=c,
    sample_amax=amax,
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    sf_vec_size=32,
)
assert op.check_support()
op.compile()
op.execute(a, b, sfa, sfb, c, amax, current_stream=None)
```

### Using JAX arrays

Two integration levels are available:

1. **`gemm_amax_jax_sm100`** (recommended for jitted programs) — an XLA custom call built on `cudnn.jax.call` (CuTeDSL's native `cutlass.jax` bridge). The kernel runs on XLA's compute stream (correctly ordered with surrounding ops), outputs are XLA-managed fresh arrays, and the call composes with `jax.jit`. No manual synchronization is needed. Requires the `jax` dependency group (`pip install --group jax`; jax >= 0.5).

```python
from cudnn import gemm_amax_jax_sm100

@jax.jit
def quantized_matmul(a, b, sfa, sfb):
    c, amax = gemm_amax_jax_sm100(a, b, sfa, sfb, c_dtype=jnp.float32, sf_vec_size=32)
    return c, amax
```

Calling it eagerly works but re-traces the custom call on every invocation; call it from inside a jitted function in hot loops.

2. **The eager entry points below** (`gemm_amax_wrapper_sm100`, `GemmAmaxSm100`) also accept JAX arrays via DLPack. In hot loops prefer the **class API with pre-allocated output buffers** (~15 µs CPU per launch) over the wrapper — per-call `jnp` output allocation in the wrapper costs hundreds of µs of XLA dispatch.

JAX arrays are always row-major (C-contiguous) and immutable-by-contract, so the JAX contract differs from torch in a few ways (all entry points):

- `A`/`B` must be k-major `(M, K, 1)` / `(N, K, 1)` C-contiguous arrays, and the batch dim must be `L == 1` (batch-outermost layouts are not expressible as JAX arrays).
- `C` supports `c_major="n"` only (the wrapper raises for `"m"`).
- `SFA`/`SFB` are passed in the **physical C-contiguous atom shape** `(L, ceil_div(MN, 128), ceil_div(K, 4·sf_vec_size), 32, 4, 4)` — byte-identical in memory to the torch-style logical view above (which is this allocation permuted by `(3, 4, 1, 5, 2, 0)`). Both forms are accepted for either framework.
- Packed fp4 has no JAX dtype; the intended vehicle is a `uint8` container tensor (`(M, K // 2, 1)`), but that container path is currently disabled kernel-side for torch and JAX alike. FP8 flavors (`float8_e4m3fn`, `float8_e5m2`, `float8_e8m0fnu` via ml_dtypes) are fully supported.
- Eager entry points only: with no explicit stream, the kernel launches on the **CUDA legacy default stream**, which XLA does not track. `jax.block_until_ready(...)` your inputs before calling, and synchronize the device (or the stream you passed) before reading the outputs. (`gemm_amax_jax_sm100` has neither caveat — XLA orders it on its own stream.)
- The eager wrapper allocates outputs with `jnp.empty`/`jnp.full` and the kernel writes into them via DLPack. This is outside JAX's functional model: eager use only — do not call the wrapper under `jax.jit` or with donated buffers. Use `gemm_amax_jax_sm100` under `jit`.

```python
import jax, jax.numpy as jnp
import ml_dtypes
import numpy as np
from cudnn import gemm_amax_wrapper_sm100

m, n, k, sf_vec_size = 512, 256, 256, 32
a = jax.device_put(np.random.randn(m, k, 1).astype(ml_dtypes.float8_e5m2))
b = jax.device_put(np.random.randn(n, k, 1).astype(ml_dtypes.float8_e5m2))
sfa = jax.device_put(np.ones((1, m // 128, k // (4 * sf_vec_size), 32, 4, 4), dtype=ml_dtypes.float8_e8m0fnu))
sfb = jax.device_put(np.ones((1, n // 128, k // (4 * sf_vec_size), 32, 4, 4), dtype=ml_dtypes.float8_e8m0fnu))
jax.block_until_ready((a, b, sfa, sfb))

c, amax = gemm_amax_wrapper_sm100(a, b, sfa, sfb, c_dtype=jnp.float32, sf_vec_size=sf_vec_size)

from cuda.bindings import runtime as cudart
cudart.cudaDeviceSynchronize()  # kernel ran on the legacy stream, outside XLA's tracking
```

---

## Parameters

### Input/Output tensors
- Input tensor **A**: `a_tensor` (wrapper) or `sample_a`/`a_tensor` (class)
  - Shape: `(M, K, L)`
  - Stride: `(1, M, M·K)` for `m`-major or `(K, 1, M·K)` for `k`-major
  - Dtype: `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}` (`uint8` is interpreted as packed fp4x2)
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b`/`b_tensor` (class)
  - Shape: `(N, K, L)`
  - Stride: `(1, N, N·K)` for `n`-major or `(K, 1, N·K)` for `k`-major
  - Dtype: Must match `A`
- Input tensor **SFA**: `sfa_tensor` (wrapper) or `sample_sfa`/`sfa_tensor` (class)
  - Shape: `(ATOM_M0, ATOM_M1, ceil_div(M, ATOM_M0·ATOM_M1), ATOM_K, ceil_div(K, ATOM_K·sf_vec_size), L)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn, int8}` (`int8` is interpreted as `float8_e8m0fnu`)
- Input tensor **SFB**: `sfb_tensor` (wrapper) or `sample_sfb`/`sfb_tensor` (class)
  - Shape: `(ATOM_M0, ATOM_M1, ceil_div(N, ATOM_M0·ATOM_M1), ATOM_K, ceil_div(K, ATOM_K·sf_vec_size), L)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn, int8}` (`int8` is interpreted as `float8_e8m0fnu`)
- Output tensor **C**: `result["c_tensor"]` (wrapper) or `sample_c`/`c_tensor` (class)
  - Shape: `(M, N, L)`
  - Stride: `(1, M, M·N)` for `m`-major or `(N, 1, M·N)` for `n`-major. Provided as `c_major` argument for wrapper
  - Dtype: `{float32, float16, bfloat16, float8_e5m2, float8_e4m3fn, float4_e2m1fn_x2, uint8}`. Provided as `c_dtype` argument for wrapper
- Output tensor **Amax**: `result["amax_tensor"]` (wrapper) or `sample_amax`/`amax_tensor` (class)
  - Shape: `(1, 1, 1)`
  - Dtype: `float32`

### Common parameters
- `acc_dtype: torch.dtype`
  - Accumulator dtype. Default: `torch.float32` (only supported value)
- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`. Default: `(128, 128)`
  - `TILE_M ∈ {128}`; `TILE_M = 256` is currently disabled
  - `TILE_N ∈ {128, 256}`
- `cluster_shape_mn: Tuple[int, int]`
  - Thread Block cluster shape `(CLUSTER_M, CLUSTER_N)`. Default: `(1, 1)`
  - Constraints: values in `{1, 2, 4}`
- `sf_vec_size: int`
  - Size of K-group per scale factor: `{16, 32}`. Default: `32`
- CUDA stream (`current_stream` in class API, `stream` in wrapper)

### Wrapper-specific parameters: `gemm_amax_wrapper_sm100`
- `a_tensor`, `b_tensor`, `sfa_tensor`, `sfb_tensor`: see Input/Output tensors
- `c_major: str`:  see Input/Output tensors. Default: `"n"`
- `c_dtype: torch.dtype`:  see Input/Output tensors. Default: `torch.float32`

### Wrapper return values

Returns a `TupleDict` with keys:

- `c_tensor`: GEMM output tensor `C`
- `amax_tensor`: Max-abs reduction output

Tuple unpacking order is: `(c_tensor, amax_tensor)`.

### Class-specific parameters: `GemmAmaxSm100`

#### `GemmAmaxSm100` (constructor)
- `sample_a`, `sample_b`, `sample_sfa`, `sample_sfb`, `sample_c`, `sample_amax`: see Input/Output tensors

#### `GemmAmaxSm100.execute`
- `a_tensor`, `b_tensor`, `sfa_tensor`, `sfb_tensor`, `c_tensor`, `amax_tensor`: see Input/Output tensors

---

## Support surface and constraints

### Layouts and strides

- For `A/B ∈ {float4_e2m1fn_x2, uint8}` (packed FP4), `A` and `B` must be `k`-major.
- For `C ∈ {float4_e2m1fn_x2, uint8}` (packed FP4), `C` must be `n`-major.
- For all `float4_e2m1fn_x2`/`uint8` cases, the innermost tensor dimension will be divided by 2 due to 2x packing. i.e. `A` would be shaped `(M, K // 2, L)` instead of `(M, K, L)`.
- `A`, `B`, `C` must be 16-byte aligned along the contiguous dimension.

### Dtypes

- `A`/`B` must have the same dtype.
- `sf_vec_size ∈ {16, 32}` with coupling:
  - `sf_dtype == float8_e4m3fn` is unsupported with `sf_vec_size == 32`
  - `A/B ∈ {float8_e4m3fn, float8_e5m2}` is unsupported with `sf_vec_size == 16`
- `A/B ∈ FP8` and `C ∈ FP8` together are currently disabled
- `C ∈ {float4_e2m1fn_x2, uint8}` requires `A/B ∈ {float4_e2m1fn_x2, uint8}`

### Tiling and cluster

- `A/B ∈ {float4_e2m1fn_x2, uint8}` and `N_tile == 256` requires `K > 128`
- `mma_tiler_mn == (128, 256)`, `sf_vec_size == 16`, `C ∈ {float32, float16, bfloat16}` is currently disabled

### Shapes and divisibility

- `SFA/SFB` shapes must follow the atom tiling and `sf_vec_size` rules above
- When `C` is packed FP4, use `(M, ceil_div(N, 2), L)` and `n`-major strides

### Environment

- Requires CUDA with SM100+ compute capability
- All tensors must reside on the same CUDA device

### JAX-specific constraints

- `L == 1`; `A`/`B` k-major; `C` n-major only
- `SFA`/`SFB` in the physical atom shape `(L, MN', K', 32, 4, 4)` (see "Using JAX arrays")
- The wrapper/class entry points are eager-only (use `gemm_amax_jax_sm100` under `jax.jit`); synchronize before reading outputs

---

## Usage examples

For usage examples, see test cases in `test/python/fe_api/gemm/test_gemm_amax.py` (torch) and `test/python/fe_api/gemm/test_gemm_amax_jax.py` (JAX)
