# Grouped GEMM + GLU + Hadamard (SM100)

**This is an experimental API and subject to change.**

## Overview

**Grouped GEMM + GLU + Hadamard fusion**: A contiguous grouped block-scaled GEMM fused with a GLU epilogue, a 16-wide Hadamard transform, and per-expert `amax` reduction on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This frontend integration is currently wired for the fp4 input path.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **GLU epilogue** using per-row `prob`
3. **Hadamard transform** across the post-GLU output
4. **Per-expert amax reduction** on the final output

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: weight tensor across all groups, shape `(N, K, L)`
  - `SFA`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - `SFB`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - `padded_offsets`: cumulative padded group ends, shape `(L,)`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`
  - `bias` (optional): per-expert bias tensor, shape `(N, L)` with stride `(1, N)`
  - `Hadamard`: fixed transform matrix, shape `(16, 16)`

- **Outputs**
  - `C`: intermediate GEMM result before GLU/Hadamard, shape `(valid_m, N, 1)`
  - `D`: output after GLU and Hadamard, shape `(valid_m, N / 2, 1)`
  - `Amax`: per-expert amax, shape `(L, 1)` when `D` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Equations

For rows belonging to expert `g`:

$$
C[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB)
$$

Split the `N` dimension into consecutive 32-column gate/up blocks:

$$
G_b = C[:, 2bG:(2b+1)G], \quad U_b = C[:, (2b+1)G:(2b+2)G], \quad G = 32
$$

For **SwiGLU** (`act_func="swiglu"`):

$$
X[:, bG:(b+1)G] = \mathrm{prob} \cdot U_b \cdot \left(G_b \cdot \sigma(G_b)\right)
$$

For **GeGLU** (`act_func="geglu"`):

$$
X[:, bG:(b+1)G] = \mathrm{prob} \cdot (U_b + 1) \cdot G_b \cdot \sigma(1.702 \cdot G_b)
$$

Apply the fixed Hadamard matrix `H` of size `16 x 16` blockwise over the output:

$$
D = X \cdot H
$$

When `D` is fp16/bf16, the kernel also emits per-expert `Amax`.

### Diagram

```text
A (valid_m×K×1), SFA     B (N×K×L), SFB       padded_offsets
          |                      |                    |
          |     dequantize       |                    |
          +----------+-----------+                    |
                     v                                v
                 Grouped GEMM over expert ranges --> group idx
                     |
                     | * alpha[group_idx]
                     v
                 C (valid_m×N×1)
                     |
                     | GLU over paired 32-col blocks
                     | with per-row prob
                     v
                 X (valid_m×N/2×1)
                     |
                     | blockwise Hadamard(16)
                     v
                 D (valid_m×N/2×1)
                     |
                     v
                 Amax (L×1)
```

---

## API Usage

### High-level wrapper

```python
from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

result = grouped_gemm_glu_hadamard_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    bias_tensor=bias,
    acc_dtype=torch.float32,
    c_dtype=torch.bfloat16,
    d_dtype=torch.bfloat16,
    cd_major="n",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
    m_aligned=256,
    act_func="swiglu",
    current_stream=None,
)

c_tensor, d_tensor, amax_tensor = result
```

The wrapper constructs the fixed Hadamard matrix internally.

### Class API

```python
from cudnn import GroupedGemmGluHadamardSm100

op = GroupedGemmGluHadamardSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d=d,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_prob=prob,
    sample_amax=amax,
    sample_bias=bias,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
    m_aligned=256,
    act_func="swiglu",
)
assert op.check_support()
op.compile()
op.execute(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
    d_tensor=d,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    amax_tensor=amax,
    bias_tensor=bias,
    current_stream=None,
)
```

You may optionally pass a custom `sample_hadamard` / `hadamard_tensor`, but the API normalizes it to the fixed `16 x 16` bf16 contiguous layout expected by the kernel. If you do not provide one, the default is the fixed kernel matrix.

---

## Parameters

### Input/Output tensors

- Input tensor **A**: `a_tensor` (wrapper) or `sample_a` / `a_tensor` (class)
  - Shape: `(valid_m, K, 1)`
  - Layout: must be `k`-major
  - Dtype: `{float4_e2m1fn_x2, uint8}`
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b` / `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Layout: must be `k`-major
  - Dtype: must match `A`
- Input tensor **SFA**: `sfa_tensor` (wrapper) or `sample_sfa` / `sfa_tensor` (class)
  - Shape: `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn}`
- Input tensor **SFB**: `sfb_tensor` (wrapper) or `sample_sfb` / `sfb_tensor` (class)
  - Shape: `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - Dtype: must match `SFA`
- Input tensor **padded_offsets**
  - Shape: `(L,)`
  - Dtype: `int32`
- Input tensor **alpha**
  - Shape: `(L,)`
  - Dtype: `float32`
- Input tensor **prob**
  - Shape: `(valid_m, 1, 1)`
  - Dtype: `float32`
- Input tensor **bias** (optional)
  - Shape: `(N, L)`
  - Stride: `(1, N)`
  - Dtype: `{float16, bfloat16, float32}`
- Input tensor **Hadamard** (optional in class API)
  - Shape: `(16, 16)`
  - Dtype: `bfloat16`
  - Layout: normalized to a contiguous `16 x 16` bf16 tensor before compile/execute
- Output tensor **C**: `result["c_tensor"]` (wrapper) or `sample_c` / `c_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float16, bfloat16}`
- Output tensor **D**: `result["d_tensor"]` (wrapper) or `sample_d` / `d_tensor` (class)
  - Shape: `(valid_m, N / 2, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float16, bfloat16}`
- Output tensor **Amax**: `result["amax_tensor"]` (wrapper) or `sample_amax` / `amax_tensor` (class)
  - Shape: `(L, 1)`
  - Dtype: `float32`

### Common parameters

- `acc_dtype: torch.dtype`
  - Only `torch.float32` is supported
- `mma_tiler_mn: Tuple[int, int]`
  - Must be `(256, 256)`
- `cluster_shape_mn: Tuple[int, int] | None`
  - Default: `(2, 1)`
- `sf_vec_size: int`
  - Allowed values: `{16, 32}`
- `vector_f32: bool`
  - Enables vectorized f32 operations for supported configurations
- `m_aligned: int`
  - Must equal the kernel fixed pad size `256`
- `act_func: str`
  - Allowed values: `{"swiglu", "geglu"}`
- CUDA stream (`current_stream` in class API and wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `c_tensor`
- `d_tensor`
- `amax_tensor`

Tuple unpacking order is: `(c_tensor, d_tensor, amax_tensor)`.

---

## Support surface and constraints

- Only dense contiguous grouped weights are exposed in this frontend integration.
- The wrapper constructs the fixed Hadamard matrix internally.
- `A` and `B` must be fp4 input tensors.
- `D` is currently supported for `{float16, bfloat16}`.
- `N` must be divisible by `64`.
- `N / 2` must be divisible by `16`.
- `m_aligned` must be `256`.
- `expert_cnt` must be `<= 1024`.
- The kernel requires SM100+.
