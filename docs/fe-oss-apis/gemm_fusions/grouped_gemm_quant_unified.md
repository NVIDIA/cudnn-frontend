# Grouped GEMM + Quant -- Unified (SM100)

**This is an experimental API and subject to change.**

## Overview

**Unified Grouped GEMM + Quant fusion**: A block-scaled grouped GEMM with output quantization and per-row gating on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads. Implemented with CUTLASS/CUTE.
Used for FC2 (forward down-projection) and dFC1 (backward FC1 GEMMs).

This kernel uses the unified `BlockScaledMoEGroupedGemmQuantKernel` which supports the `MoEWeightMode` abstraction:

- **Dense mode** (`MoEWeightMode.DENSE`): all expert weights packed into a single contiguous `(N, K, L)` tensor
- **Discrete mode** (`MoEWeightMode.DISCRETE`): each expert weight and scale-factor tensor provided through per-expert device pointer arrays

Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).

This kernel performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4, FP8) with per-block scale factors across multiple expert groups
2. **Per-row gating**: Multiplies output by per-row gating probability
3. **Optional quantized output**: Produces row and column scale factors for downstream quantization

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B` (dense): weight tensor across all groups, shape `(N, K, L)`
  - `B` (discrete): per-expert weight tensors, each with shape `(N, K)`, passed via `b_ptrs`
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB` (dense): scale factor tensor for B, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
  - `SFB` (discrete): per-expert scale factor tensors, passed via `sfb_ptrs`
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`. **Required.**
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `D`: row-quantized output, shape `(valid_m, N, 1)`
  - `D_col`: column-quantized output, shape `(valid_m, N, 1)`
  - `SFD_row`: row scale factors (when SFD outputs are enabled), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(N/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when SFD outputs are enabled), shape `(32, 4, ceil(N/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
  - `amax`: per-group amax (when `d_dtype` is bf16/float16), shape `(L, 1)`

### Equations

**Step 1: Block-scaled grouped GEMM** (per group `g` with rows `m` in `[padded_offsets[g-1], padded_offsets[g])`):

$$
\text{ref}[m, n] = \alpha_g \sum_{k} \text{dequantize}(A[m, k], \text{SFA}) \cdot \text{dequantize}(B[n, k, g], \text{SFB})
$$

**Step 2: Per-row gating**:

$$
D[m, n] = \text{prob}[m] \cdot \text{ref}[m, n]
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
 A (valid_m×K×1)    B (N×K×L)           padded_offsets
 SFA                SFB                      |
   |                 |                       |
   |    +------------+                       |
   |    |                                    |
   v    v                                    v
  Dequantize → Grouped GEMM (per group ranges) → Select B[:,:,group_idx]
                    |
                    | × alpha[group_idx]
                    v
               ref (valid_m×N×1)
                    |
                    | × prob
                    v
               D (valid_m×N×1)
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

```python
from cudnn import grouped_gemm_quant_wrapper_sm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

outputs = grouped_gemm_quant_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    norm_const_tensor=norm_const,
    prob_tensor=prob,
    acc_dtype=torch.float32,
    d_dtype=torch.bfloat16,
    cd_major="n",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
    current_stream=stream,
)

# dictionary access:
d = outputs["d_tensor"]
d_col = outputs["d_col_tensor"]  # None for bf16/fp16/fp32 outputs; tensor only for low-precision outputs
amax = outputs["amax_tensor"]
sfd_row = outputs["sfd_row_tensor"]
sfd_col = outputs["sfd_col_tensor"]

# or tuple unpacking:
d, d_col, amax, sfd_row, sfd_col = outputs  # d_col is None for bf16/fp16/fp32 outputs
```

### Class API

```python
from cudnn import GroupedGemmQuantSm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

api = GroupedGemmQuantSm100(
    sample_a=a,
    sample_b=b,
    sample_d=d,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_d_col=d_col,
    sample_sfd_row=sfd_row,
    sample_sfd_col=sfd_col,
    sample_amax=amax,
    sample_norm_const=norm_const,
    sample_prob=prob,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a, b_tensor=b, d_tensor=d,
    sfa_tensor=sfa, sfb_tensor=sfb,
    padded_offsets=padded_offsets, alpha_tensor=alpha,
    d_col_tensor=d_col, sfd_row_tensor=sfd_row, sfd_col_tensor=sfd_col,
    amax_tensor=amax, norm_const_tensor=norm_const, prob_tensor=prob,
    current_stream=stream,
)
```

---

## Parameters

### Input/Output Tensors

- **Input tensor A**: `a_tensor` / `sample_a`
  - Shape: `(valid_m, K, 1)`, Stride: K-major
  - Dtype: `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`

- **Input tensor B**: `b_tensor` / `sample_b`
  - Shape: `(N, K, L)`, Stride: K-major (FP8 also supports N-major)
  - Dtype: Must match A

- **Output tensor D**: `d_tensor` / `sample_d`
  - Shape: `(valid_m, N, 1)`, Stride: N-major
  - Dtype: `{float16, bfloat16, float32}` for FP4; `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` for FP8

- **Output tensor D_col**: `d_col_tensor` / `sample_d_col`
  - Shape/Dtype: Must match D
  - Wrapper behavior: returned only when `d_dtype ∈ {float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`; for `bfloat16`, `float16`, and `float32`, `outputs["d_col_tensor"]` is `None`

- **Input tensor prob**: `prob_tensor` / `sample_prob`
  - Shape: `(valid_m, 1, 1)`, dtype: `float32`
  - **Required**: pass ones tensor if no gating needed

- **Scale factor tensors**: SFA, SFB, SFD_row, SFD_col -- block-scaled 6-D layout
- **Group offsets**: `padded_offsets` shape `(L,)`, dtype `int32`
- **Scaling tensors**: `alpha` shape `(L,)`, `amax` shape `(L, 1)`, `norm_const` shape `(1,)`

### Common Parameters

- `acc_dtype`: Must be `torch.float32`
- `mma_tiler_mn`: Default `(256, 256)`; supported tiles are `TILE_M ∈ {128, 256}` and `TILE_N = 256`
- `cluster_shape_mn`: Default `(2, 1)` when `TILE_M=256`, `(1, 1)` otherwise
- `sf_vec_size`: `{16, 32}`. Default: `16`
- `vector_f32`: Default: `False`
- `m_aligned`: Must be `256`
- `discrete_col_sfd`: Default: `False`

### Wrapper Return Values

Returns `TupleDict`: `d_tensor`, `d_col_tensor` (optional; `None` for `bfloat16`/`float16`/`float32` outputs), `amax_tensor`, `sfd_row_tensor`, `sfd_col_tensor`

---

## Support Surface and Constraints

### Data Types

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|---------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` |
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e4m3fn`, `float8_e8m0fnu`} | {16, 32} | `{float16, bfloat16, float32}` |

### Key Constraints

- `A` and `B` must have same dtype; `D` and `D_col` must have same dtype
- All scale factor tensors must have same dtype
- Expert count `<= 1024`; M aligned to 256
- SM100+ compute capability required
- `prob_tensor` is unconditionally required

---

## Usage Examples

For usage examples, see `test/python/fe_api/test_grouped_gemm_quant.py` + `test/python/fe_api/test_grouped_gemm_quant_utils.py` (dense and discrete unified API coverage)
