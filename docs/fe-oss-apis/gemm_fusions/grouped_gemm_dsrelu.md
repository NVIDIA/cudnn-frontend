# Grouped GEMM + dsReLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Grouped GEMM + dsReLU backward fusion**: A grouped block-scaled GEMM fused with a probability-gradient backward epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. The API supports dense contiguous weights and discrete per-expert weight allocations. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **dsReLU backward epilogue** using the forward/intermediate tensor `C`
3. **Optional output quantization** through `SFD_row` / `SFD_col` or `Amax`

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: dense weight tensor across all groups, shape `(N, K, L)`, or discrete per-expert tensors addressed by `b_ptrs`
  - `C`: forward/intermediate tensor, shape `(valid_m, N, 1)`
  - `SFA`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - `SFB`: dense scale-factor tensor, shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`, or discrete per-expert tensors addressed by `sfb_ptrs`
  - `padded_offsets`: cumulative padded group ends, shape `(L,)`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`

- **Outputs**
  - `D_row`: row output after dsReLU, shape `(valid_m, N, 1)`
  - `D_col`: column output after dsReLU, shape `(valid_m, N, 1)`
  - `dprob`: probability gradient, shape `(valid_m, 1, 1)`
  - `SFD_row`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), 1)` when `D_row` is FP8
  - `SFD_col`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(valid_m, sf_vec_size), 4), 1)` when `D_row`/`D_col` is FP8
  - `Amax`: shape `(L, 1)` when `D_row` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Equations

For rows belonging to expert `g`:

$$
G[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB)
$$

$$
D\_{row}[m, n] = \mathrm{prob}[m, 0, 0] \cdot 2 \cdot C[m, n, 0] \cdot \mathrm{relu}(G[m, n])
$$

$$
\mathrm{dprob}[m, 0, 0] = \sum_n C[m, n, 0] \cdot \mathrm{relu}(G[m, n])^2
$$

`D_col` stores the companion column-quantized output used by the grouped kernel family. When FP8 output is enabled, the kernel also emits `SFD_row` and `SFD_col`. When fp16/bf16 output is used, the kernel can emit per-expert `Amax`.

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
                 G (valid_m×N×1)
                     |
       C (valid_m×N×1)+
                     |
                     +--> D_row / D_col
                     |
                     +--> dprob
                     |
          +----------+-----------+
          |                      |
          v                      v
      SFD_row/SFD_col          Amax
```

---

## API Usage

### High-level wrapper

```python
from cudnn import grouped_gemm_dsrelu_wrapper_sm100

result = grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    norm_const_tensor=norm_const,
    acc_dtype=torch.float32,
    d_dtype=torch.bfloat16,
    cd_major="n",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
    current_stream=None,
)

d_row, d_col, dprob, dbias, amax, sfd_row, sfd_col = result
```

### Discrete-weight wrapper

```python
result = grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor=a,
    c_tensor=c,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    b_ptrs=b_ptrs,          # int64 device tensor of per-expert B pointers
    sfb_ptrs=sfb_ptrs,      # int64 device tensor of per-expert SFB pointers
    n=N,
    b_dtype=torch.float4_e2m1fn_x2,
    b_major="k",
    d_dtype=torch.bfloat16,
)
```

### Class API

```python
from cudnn import GroupedGemmDsreluSm100

op = GroupedGemmDsreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d_row=d_row,
    sample_d_col=d_col,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_prob=prob,
    sample_dprob=dprob,
    sample_sfd_row=sfd_row,
    sample_sfd_col=sfd_col,
    sample_amax=amax,
    sample_norm_const=norm_const,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
)
assert op.check_support()
op.compile()
op.execute(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
    d_row_tensor=d_row,
    d_col_tensor=d_col,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    dprob_tensor=dprob,
    sfd_row_tensor=sfd_row,
    sfd_col_tensor=sfd_col,
    amax_tensor=amax,
    norm_const_tensor=norm_const,
    current_stream=None,
)
```

---

## Parameters

### Input/Output tensors

- Input tensor **A**: `a_tensor` (wrapper) or `sample_a` / `a_tensor` (class)
  - Shape: `(valid_m, K, 1)`
  - Layout: must be `k`-major
  - Dtype: `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`
  - Note: `uint8` is interpreted as packed `float4_e2m1fn_x2` (FP4x2) data, not integer quantization
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b` / `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Layout: must be `k`-major
  - Dtype: must match `A`
- Discrete input **B pointers**: `b_ptrs` (wrapper) or `num_experts` / `b_shape` / `b_dtype` (class)
  - `b_ptrs`: 1-D `int64` CUDA tensor containing one data pointer per expert
  - `n` and `b_dtype` are required in wrapper discrete mode
  - `b_major` may be `"k"` or `"n"` for supported FP8 cases; FP4 uses `"k"`
- Input tensor **C**: `c_tensor` (wrapper) or `sample_c` / `c_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}`
- Input tensor **SFA**: `sfa_tensor` (wrapper) or `sample_sfa` / `sfa_tensor` (class)
  - Shape: `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn}`
- Input tensor **SFB**: `sfb_tensor` (wrapper) or `sample_sfb` / `sfb_tensor` (class)
  - Shape: `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - Dtype: must match `SFA`
- Discrete input **SFB pointers**: `sfb_ptrs`
  - 1-D `int64` CUDA tensor containing one scale-factor pointer per expert
- Input tensor **padded_offsets**
  - Shape: `(L,)`
  - Dtype: `int32`
- Input tensor **alpha**
  - Shape: `(L,)`
  - Dtype: `float32`
- Input tensor **prob**
  - Shape: `(valid_m, 1, 1)`
  - Dtype: `float32`
- Output tensor **D_row**: `result["d_row_tensor"]` (wrapper) or `sample_d_row` / `d_row_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype:
    - FP4 input: `{float16, bfloat16, float32}`
    - FP8 input: `{float8_e4m3fn, float8_e5m2}`
- Output tensor **D_col**: `result["d_col_tensor"]` (wrapper) or `sample_d_col` / `d_col_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must match `D_row`
  - Dtype: must match `D_row`
- Output tensor **dprob**: `result["dprob_tensor"]` (wrapper) or `sample_dprob` / `dprob_tensor` (class)
  - Shape: `(valid_m, 1, 1)`
  - Dtype: `float32`
- Output tensors **SFD_row** / **SFD_col**
  - Dtypes: must match `SFA`
  - Generated when `D_row` / `D_col` uses an FP8 dtype
- Output tensor **Amax**
  - Shape: `(L, 1)`
  - Dtype: `float32`
  - Generated when `D_row` / `D_col` uses `float16` or `bfloat16`
- Input tensor **Norm Const**
  - Shape: `(1,)`
  - Dtype: `float32`
  - Required when `SFD_row` / `SFD_col` are generated for FP8 output

### Common parameters

- `acc_dtype: torch.dtype`
  - Only `torch.float32` is supported
- `mma_tiler_mn: Tuple[int, int]`
  - `TILE_M` depends on the 1-CTA / 2-CTA mode
  - `TILE_N ∈ {128, 256}`
- `cluster_shape_mn: Tuple[int, int] | None`
  - Default: `(2, 1)` when `TILE_M == 256`, else `(1, 1)`
- `sf_vec_size: int`
  - Allowed values: `{16, 32}`
- `vector_f32: bool`
  - Enables vectorized f32 operations for supported configurations
- `m_aligned: int`
  - Must equal the kernel fixed pad size `256`
- `cd_major: str` (wrapper only)
  - Specifies the major dimension for `C` and `D` tensors
  - Only `"n"` (n-major layout) is supported
- `discrete_col_sfd: bool`
  - Enables the discrete column-scale-factor path used by grouped FP8
- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `d_row_tensor`
- `d_col_tensor`
- `dprob_tensor`
- `dbias_tensor`
- `amax_tensor`
- `sfd_row_tensor`
- `sfd_col_tensor`

Tuple unpacking order is: `(d_row_tensor, d_col_tensor, dprob_tensor, dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor)`.

---

## Support surface and constraints

### Layouts

- `A` must be `k`-major
- `B` must be `k`-major
- Discrete `B` supports `b_major="k"` and supported FP8 `b_major="n"` configurations
- `C`, `D_row`, and `D_col` must be `n`-major
- The wrapper only supports `cd_major="n"`

### Dtypes

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- Scale-factor dtype constraint: `sf_vec_size == 32` is unsupported when `sf_dtype == float8_e4m3fn`
- Input dtype constraint: FP8 `A`/`B` inputs require `sf_vec_size == 32`
- Grouped FP8 currently requires `discrete_col_sfd=True`
- Grouped `dsrelu` requires the kernel-supported `k`-major `B` layout

### Shapes and environment

- `m_aligned` must be `256`
- Requires CUDA with SM100+ compute capability

---

## Usage examples

For end-to-end usage and regression coverage, see:

- `test/python/fe_api/test_grouped_gemm_dsrelu.py`
- `test/python/fe_api/test_grouped_gemm_dsrelu_utils.py`
