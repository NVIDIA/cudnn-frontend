# Grouped GEMM + GLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Unified Grouped GEMM + GLU fusion**: A block-scaled grouped GEMM fused with a GLU epilogue (SwiGLU or GeGLU) on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads. Implemented with CUTLASS/CUTE.

This is a **unified API** that supports both weight layout modes:
- **Dense mode**: All expert weights packed into a single contiguous `(N, K, L)` tensor
- **Discrete mode**: Per-expert weight pointers (no weight stacking required)

And both activation functions:
- **SwiGLU**: `act_func="swiglu"` (default)
- **GeGLU**: `act_func="geglu"`

Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).

This kernel performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4, FP8) with per-block scale factors across multiple expert groups
2. **GLU activation**: Fused SwiGLU or GeGLU activation applied to the GEMM output
3. **Optional quantized output**: Produces row and column scale factors for downstream quantization

### Shapes

### Equations

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B` (dense): weight tensor across all groups, shape `(N, K, L)`
  - `B` (discrete): per-expert weight pointers, `b_ptrs` shape `(num_experts,)` of int64
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB` (dense): scale factor tensor for B, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
  - `SFB` (discrete): per-expert SFB pointers, `sfb_ptrs` shape `(num_experts,)` of int64
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `bias` (optional): per-expert bias tensor, shape `(N, L)` with stride `(1, N)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `C`: intermediate GEMM result, shape `(valid_m, N, 1)`
  - `D`: row-quantized GLU output, shape `(valid_m, N/2, 1)`
  - `D_col`: column-quantized GLU output, shape `(valid_m, N/2, 1)`
  - `SFD_row`: row scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil((N/2)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
  - `amax`: per-group amax (when `d_dtype` is bf16/fp16), shape `(L, 1)`

**Step 1: Block-scaled grouped GEMM** (per group `g` with rows `m` in `[padded_offsets[g-1], padded_offsets[g])`):

$$
C[m, n] = \alpha_g \sum_{k} \text{dequantize}(A[m, k], \text{SFA}) \cdot \text{dequantize}(B[n, k, g], \text{SFB})
$$

**Step 2: GLU epilogue** (performed by pairing 32-column blocks along `N`):

Let block size `G = 32`. For each pair of consecutive 32-wide column blocks:
- Input block: `X_b = C[:, 2·b·G : 2·b·G + G]`
- Gate block: `G_b = C[:, 2·b·G + G : 2·b·G + 2·G]`

For **SwiGLU** (`act_func="swiglu"`):

$$
D[:, bG:(b+1)G] = \text{prob} \cdot X_b \cdot \text{swish}(G_b), \quad \text{swish}(x) = x \cdot \sigma(x)
$$

For **GeGLU** (`act_func="geglu"`):

$$
D[:, bG:(b+1)G] = \text{prob} \cdot (X_b + 1) \cdot G_b \cdot \sigma(1.702 \cdot G_b)
$$

**Step 3: Optional output quantization** (when SFD outputs are generated):

$$
\text{SFD\_row}[m, n] = \text{norm\_const} \cdot \max_{k \in \text{block}} |D[m, k]| \cdot \text{rcp\_max}
$$

$$
D_{\text{quantized}}[m, n] = D[m, n] \cdot \frac{\text{norm\_const}}{\text{SFD\_row}[m, n]}
$$

### Diagram

```text
 A (valid_m×K×1)    B (N×K×L) or b_ptrs    padded_offsets
 SFA                SFB or sfb_ptrs              |
   |                 |                           |
   |    +------------+                           |
   |    |                                        |
   v    v                                        v
  Dequantize → Grouped GEMM (per group ranges) → Select B[:,:,group_idx]
                    |
                    | × alpha[group_idx]
                    v
               C (valid_m×N×1)
                    |
                    | Pair 32-col blocks: [X0|G0|X1|G1|...]
                    |     X_b × swish(G_b)  [SwiGLU]
                    |     (X_b+1) × x·σ(1.702·G_b)  [GeGLU]
                    v
                    | × prob
                    v
               D (valid_m×N/2×1)
                    |
         +----------+-----------+
         |                      |
         v                      v
    Row Quantize           Col Quantize
         |                      |
         v                      v
    D_row, SFD_row        D_col, SFD_col
```

---

## API Usage

### High-level Wrapper

**Dense mode:**

```python
from cudnn import grouped_gemm_glu_wrapper_sm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

outputs = grouped_gemm_glu_wrapper_sm100(
    a_tensor=a,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    bias_tensor=bias,
    # Dense mode weights:
    b_tensor=b,
    sfb_tensor=sfb,
    # Common:
    norm_const_tensor=norm_const,
    prob_tensor=prob,
    acc_dtype=torch.float32,
    c_dtype=torch.bfloat16,
    d_dtype=torch.bfloat16,
    cd_major="n",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
    act_func="swiglu",
    current_stream=stream,
)

# dictionary access:
c = outputs["c_tensor"]
d = outputs["d_tensor"]
d_col = outputs["d_col_tensor"]
amax = outputs["amax_tensor"]
sfd_row = outputs["sfd_row_tensor"]
sfd_col = outputs["sfd_col_tensor"]

# or tuple unpacking:
c, d, d_col, amax, sfd_row, sfd_col = outputs
```

**Discrete mode:**

```python
outputs = grouped_gemm_glu_wrapper_sm100(
    a_tensor=a,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    # Discrete mode weights:
    b_ptrs=b_ptrs,       # int64 tensor of per-expert B data pointers
    sfb_ptrs=sfb_ptrs,   # int64 tensor of per-expert SFB data pointers
    n=n_dim,             # B weight N dimension
    b_dtype=torch.uint8, # B weight data type
    b_major="k",         # B tensor major dimension
    # Common:
    norm_const_tensor=norm_const,
    prob_tensor=prob,
    act_func="geglu",    # GeGLU activation
    current_stream=stream,
)
```

`bias_tensor` must use the kernel layout expected by the fused bias path: shape `(N, L)` and stride `(1, N)`.

### Class API

**Dense mode:**

```python
from cudnn import GroupedGemmGluSm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

api = GroupedGemmGluSm100(
    sample_a=a,
    sample_c=c,
    sample_d=d,
    sample_sfa=sfa,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_d_col=d_col,
    sample_bias=bias,
    # Dense mode:
    sample_b=b,
    sample_sfb=sfb,
    # Optional quantization outputs
    sample_sfd_row=sfd_row,
    sample_sfd_col=sfd_col,
    sample_amax=amax,
    sample_norm_const=norm_const,
    sample_prob=prob,
    # Configuration
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    act_func="swiglu",
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a, c_tensor=c, d_tensor=d,
    sfa_tensor=sfa, padded_offsets=padded_offsets, alpha_tensor=alpha,
    b_tensor=b, sfb_tensor=sfb, bias_tensor=bias,
    d_col_tensor=d_col, sfd_row_tensor=sfd_row, sfd_col_tensor=sfd_col,
    amax_tensor=amax, norm_const_tensor=norm_const, prob_tensor=prob,
    current_stream=stream,
)
```

`sample_bias` and runtime `bias_tensor` must both use shape `(N, L)` and stride `(1, N)`.

**Discrete mode:**

```python
api = GroupedGemmGluSm100(
    sample_a=a,
    sample_c=c,
    sample_d=d,
    sample_sfa=sfa,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_d_col=d_col,
    # Discrete mode:
    num_experts=num_experts,
    b_shape=(n, k),
    b_dtype=torch.uint8,
    # Configuration
    act_func="geglu",
    b_major="k",
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a, c_tensor=c, d_tensor=d,
    sfa_tensor=sfa, padded_offsets=padded_offsets, alpha_tensor=alpha,
    b_ptrs=b_ptrs, sfb_ptrs=sfb_ptrs,
    d_col_tensor=d_col, prob_tensor=prob,
    current_stream=stream,
)
```

---

## Parameters

### Weight Mode

The weight mode is auto-detected from constructor arguments:
- **Dense**: Provide `sample_b` and `sample_sfb` (contiguous weight tensors)
- **Discrete**: Provide `num_experts`, `b_shape`, and `b_dtype` (per-expert pointer mode)

Providing both or neither raises `ValueError`.

### Input/Output Tensors

- **Input tensor A**: `a_tensor` / `sample_a`
  - Shape: `(valid_m, K, 1)`
  - Stride: `(K, 1, valid_m·K)` -- must be K-major
  - Dtype (`ab_dtype`): `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`

- **Input tensor B** (dense mode): `b_tensor` / `sample_b`
  - Shape: `(N, K, L)` where `L = num_groups`
  - Stride: `(K, 1, N·K)` -- must be K-major
  - Dtype: Must match A

- **Input B pointers** (discrete mode): `b_ptrs`
  - Shape: `(num_experts,)` -- 1-D int64 device tensor of per-expert B data pointers
  - Build via: `torch.tensor([b.data_ptr() for b in experts], dtype=torch.int64, device="cuda")`

- **Output tensor C**: returned in wrapper dict or `c_tensor` in class
  - Shape: `(valid_m, N, 1)`
  - Stride: `(N, 1, valid_m·N)` -- **must be N-major**
  - Dtype (`c_dtype`): `{float16, bfloat16}` for FP4 inputs; `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` otherwise

- **Output tensor D**: `d_tensor` / `sample_d`
  - Shape: `(valid_m, N/2, 1)`
  - Stride: `(N/2, 1, valid_m·N/2)` -- **must be N-major**
  - Dtype (`d_dtype`): `{bfloat16, float32}` for FP4 inputs; `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` otherwise

- **Output tensor D_col**: `d_col_tensor` / `sample_d_col`
  - Shape: `(valid_m, N/2, 1)` -- must match D dtype and stride

- **Scale factor tensors**: Same as contiguous swiglu (SFA, SFB, SFD_row, SFD_col)
  - SFB (discrete mode): use `sfb_ptrs` (1-D int64 device tensor of per-expert SFB pointers)

- **Group offsets**: `padded_offsets` -- shape `(L,)`, dtype `int32`

- **Scaling tensors**: `alpha` shape `(L,)`, `prob` shape `(valid_m, 1, 1)`, `amax` shape `(L, 1)`, `norm_const` shape `(1,)`

### Common Parameters

- `acc_dtype`: Must be `torch.float32`
- `mma_tiler_mn`: Kernel tile size `(TILE_M, TILE_N)`. Default: `(256, 256)`
  - `TILE_M ∈ {128, 256}`
  - `TILE_N = 256`
- `cluster_shape_mn`: Thread Block cluster shape. Default: `(2, 1)` when `TILE_M=256`, `(1, 1)` otherwise
- `sf_vec_size`: Scale factor vector size. `{16, 32}`. Default: `16`
- `vector_f32`: Enable packed f32 operations. Default: `False`
- `m_aligned`: Must be `256` (FIX_PAD_SIZE). Default: `256`
- `discrete_col_sfd`: Generate discrete col-major scale factors. Default: `False`
- `act_func`: Activation function. `"swiglu"` (default) or `"geglu"`
- `b_major` (discrete only): B tensor major dimension. `"k"` (default) or `"n"`. Must be `"k"` for FP4.

### Wrapper-specific Parameters

- `c_dtype`: Intermediate C tensor data type. Default: `torch.bfloat16`
- `d_dtype`: Output D tensor data type. Default: `torch.bfloat16`
- `cd_major`: Must be `"n"`. Default: `"n"`
- `n` (discrete only): B weight N dimension (full N before GLU split)
- `b_dtype` (discrete only): B weight data type

### Wrapper Return Values

Returns a `TupleDict` (dictionary + tuple unpacking):
- `c_tensor`: Intermediate GEMM result
- `d_tensor`: Row-quantized GLU output
- `d_col_tensor`: Column-quantized GLU output
- `amax_tensor`: Per-group amax (when `d_dtype` is bf16/fp16)
- `sfd_row_tensor`: Row scale factors (when SFD enabled)
- `sfd_col_tensor`: Column scale factors (when SFD enabled)

---

## Support Surface and Constraints

### Layouts and Strides

- `A` must be **K-major**
- `B` must be **K-major** (dense mode). For discrete mode: K-major or N-major (K-major required for FP4)
- `C`, `D`, `D_col` must be **N-major**
- All tensors must be **16-byte aligned** along the contiguous dimension

### Data Types

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|---------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | `{float16, bfloat16, float8_e4m3fn, float8_e5m2}` |
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e4m3fn`, `float8_e8m0fnu`} | {16, 32} | `{bfloat16, float32}` |

### Additional Type Constraints

- `A` and `B` must have the same dtype
- Scale factor tensors (SFA, SFB, SFD_row, SFD_col) must have the same dtype
- `D` and `D_col` must have the same dtype
- `bias` must be one of `{float16, bfloat16, float32}`
- `bias` must have shape `(N, L)` and stride `(1, N)`
- For non-bias paths, FP4 `ab_dtype` with `sf_vec_size=16` and `d_dtype=float32` is not supported
- FP4 `ab_dtype` requires `c_dtype` in `{float16, bfloat16}`

### Shapes and Divisibility

- `N` must be divisible by 64 (two consecutive 32-column blocks for GLU pairing)
- Expert count must be `<= 1024`
- Each group's M dimension is aligned to `m_aligned` (256)
- All supported kernel configurations require `mma_tiler_mn[1] == 256`

### Environment

- Requires CUDA with **SM100+ compute capability** (Blackwell GPUs)

---

## Usage Examples

For usage examples, see test cases in `test/python/fe_api/test_grouped_gemm_glu.py` (dense mode, unified API) and `test/python/fe_api/test_discrete_grouped_gemm_swiglu.py` (discrete mode).
