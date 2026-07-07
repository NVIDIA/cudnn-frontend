# Grouped GEMM + Quant (SM100)

**This is an experimental API and subject to change.**

**Legacy dense-only API note:** This page documents the older dense-only grouped quant API. For new integrations, prefer the unified [Grouped GEMM + Quant (Unified)](grouped_gemm_quant_unified.md) page.

## Overview

**Grouped GEMM + Quant fusion**: A contiguous grouped block-scaled GEMM with output quantization on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads. Implemented with CUTLASS/CUTE.
Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).
Used for FC2 (forward down-projection) and dFC1 (backward FC1 GEMMs).

This kernel performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4, FP8) with per-block scale factors across multiple expert groups
2. **Per-row gating**: Multiplies output by per-row gating probability
3. **Optional quantized output**: Produces row and column scale factors for downstream quantization

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: weight tensor across all groups, shape `(N, K, L)`
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB`: scale factor tensor for B, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`. Required.
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `D`: row-quantized output, shape `(valid_m, N, 1)`
  - `D_col`: column-quantized output, shape `(valid_m, N, 1)`
  - `SFD_row`: row scale factors (when SFD outputs are enabled, i.e. FP8 inputs), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(N/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when SFD outputs are enabled, i.e. FP8 inputs), shape `(32, 4, ceil(N/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
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

Let $\text{rcp\_max} = 1 / q_{\max}$, where $q_{\max}$ is the maximum representable value of the output data type (e.g., 448 for FP8 E4M3, 57344 for FP8 E5M2).

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
    norm_const_tensor=norm_const,  # Required when SFD outputs are enabled (FP8 inputs)
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
d = outputs["d_tensor"]             # row-quantized output
d_col = outputs["d_col_tensor"]     # None for bf16/fp16/fp32 outputs; tensor only for low-precision outputs
amax = outputs["amax_tensor"]       # per-group amax (when d_dtype is bf16/float16)
sfd_row = outputs["sfd_row_tensor"] # row scale factors (when SFD outputs are enabled, FP8 inputs)
sfd_col = outputs["sfd_col_tensor"] # column scale factors (when SFD outputs are enabled, FP8 inputs)

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
    # Optional quantization outputs
    sample_sfd_row=sfd_row,        # Required when SFD outputs are enabled
    sample_sfd_col=sfd_col,        # Required when SFD outputs are enabled
    sample_amax=amax,              # Required for bf16 output with FP4 input
    sample_norm_const=norm_const,  # Required when SFD outputs are enabled
    sample_prob=prob,              # Per-row gating probabilities (required)
    # Configuration
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
    a_tensor=a,
    b_tensor=b,
    d_tensor=d,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    d_col_tensor=d_col,
    sfd_row_tensor=sfd_row,
    sfd_col_tensor=sfd_col,
    amax_tensor=amax,
    norm_const_tensor=norm_const,
    prob_tensor=prob,
    current_stream=stream,
)
```

---

## Parameters

### Input/Output Tensors

- **Input tensor A**: `a_tensor` (wrapper) or `sample_a`, `a_tensor` (class)
  - Shape: `(valid_m, K, 1)`
  - Stride: `(K, 1, valid_m·K)` – must be K-major
  - Dtype (`ab_dtype`): `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`
    - `uint8` is interpreted as packed FP4 (two FP4 values per byte)

- **Input tensor B**: `b_tensor` (wrapper) or `sample_b`, `b_tensor` (class)
  - Shape: `(N, K, L)` where `L = num_groups`
  - Stride: `(K, 1, N·K)` – must be K-major
  - Dtype (`ab_dtype`): Must match A

- **Output tensor D**: `d_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N, 1)`
  - Stride: `(N, 1, valid_m·N)` – **must be N-major**
  - Dtype (`d_dtype`): `{float16, bfloat16, float32}` for FP4 inputs; `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` otherwise

- **Output tensor D_col**: `d_col_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N, 1)`
  - Stride: `(N, 1, valid_m·N)` – must match D (N-major)
  - Dtype: Must match D
  - Wrapper behavior: returned only when `d_dtype ∈ {float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`; for `bfloat16`, `float16`, and `float32`, `outputs["d_col_tensor"]` is `None`

- **Input tensor prob**: `prob_tensor` (wrapper) or `sample_prob` (class)
  - Shape: `(valid_m, 1, 1)`
  - Dtype: `float32`
  - **Required**: the kernel unconditionally multiplies output by per-row gating probability. Pass a tensor of ones when no gating is needed.

- **Scale factor tensors**
  - **SFA** (A scale factor): `sfa_tensor` (wrapper) or `sample_sfa`, `sfa_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
    - Dtype (`sf_dtype`): `{float8_e8m0fnu, float8_e4m3fn}`
  - **SFB** (B scale factor): `sfb_tensor` (wrapper) or `sample_sfb`, `sfb_tensor` (class)
    - Shape: `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
    - Dtype: Must match SFA
  - **SFD_row** (D row scale factor, optional): `sfd_row_tensor` (wrapper) or `sample_sfd_row`, `sfd_row_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil(N/sf_vec_size)/4), 1)`
    - Dtype: Must match SFA
    - **Required when**: SFD outputs are enabled (FP8 inputs)
  - **SFD_col** (D column scale factor, optional): `sfd_col_tensor` (wrapper) or `sample_sfd_col`, `sfd_col_tensor` (class)
    - Shape: `(32, 4, ceil(N/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
    - Dtype: Must match SFA
    - **Required when**: SFD outputs are enabled (FP8 inputs)

- **Group offsets**
  - **padded_offsets**: Cumulative sum of aligned group M sizes
    - Shape: `(L,)` where `L = num_groups`
    - Dtype: `int32`
    - `padded_offsets[-1]` equals `valid_m`; each offset is a multiple of `m_aligned`

- **Scaling tensors**
  - **alpha**: Per-group scaling factors
    - Shape: `(L,)` where `L = num_groups`
    - Dtype: `float32`
  - **amax** (optional): Per-group max absolute values
    - Shape: `(L, 1)`
    - Dtype: `float32`
    - **Required when**: `d_dtype ∈ {bfloat16, float16}`
  - **norm_const** (optional): Normalization constant for FP8 quantization
    - Shape: `(1,)`
    - Dtype: `float32`
    - **Required when**: `sfd_row_tensor`/`sfd_col_tensor` are provided (FP8 inputs)

### Common Parameters

- `acc_dtype: torch.dtype`
  - Accumulator dtype. Must be `torch.float32`

- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`. Default: `(256, 256)`
  - `TILE_M ∈ {128, 256}`
  - `TILE_N = 256`

- `cluster_shape_mn: Tuple[int, int] | None`
  - Thread Block cluster shape `(CLUSTER_M, CLUSTER_N)`
  - Constraints: positive powers of 2, both ≤ 4, `CLUSTER_M × CLUSTER_N ≤ 16`
  - Default: `(2, 1)` when `TILE_M=256`, `(1, 1)` otherwise

- `sf_vec_size: int`
  - Scale factor vector size (number of elements per scale factor)
  - Allowed values: `{16, 32}`. Default: `16`

- `vector_f32: bool`
  - Enable packed f32 operations for improved performance
  - Default: `False`

- `m_aligned: int`
  - Internal constant equal to `FIX_PAD_SIZE` (256); cannot be changed (the implementation raises `ValueError` for any other value)
  - Must be divisible by `mma_tiler_mn[0]`

- `discrete_col_sfd: bool`
  - If True, generate discrete column scale factors grouped by expert tiles
  - Only applies when `sfd_row_tensor`, `sfd_col_tensor`, and `norm_const_tensor` are provided
  - Default: `False`

- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper-specific Parameters: `grouped_gemm_quant_wrapper_sm100`

- `d_dtype: torch.dtype`: Output D tensor data type. Default: `torch.bfloat16`
- `cd_major: str`: Major dimension for D tensors. Must be `"n"` (only N-major layout is supported). Default: `"n"`

### Wrapper Return Values

Returns a `TupleDict` - a dictionary-like object that also supports tuple unpacking and integer indexing.

**Dictionary keys** (also the tuple unpacking order):
- `d_tensor`: Row-quantized output
- `d_col_tensor`: Optional column-quantized output; `None` when `d_dtype ∈ {bfloat16, float16, float32}`
- `amax_tensor`: Per-group amax (when `d_dtype ∈ {bfloat16, float16}`)
- `sfd_row_tensor`: Row scale factors (when SFD outputs are enabled)
- `sfd_col_tensor`: Column scale factors (when SFD outputs are enabled)

### Class-specific Parameters

#### `GroupedGemmQuantSm100` (constructor)

- `sample_a`, `sample_b`, `sample_d`, `sample_sfa`, `sample_sfb`, `sample_padded_offsets`, `sample_alpha`, `sample_d_col`, `sample_sfd_row`, `sample_sfd_col`, `sample_amax`, `sample_norm_const`, `sample_prob` – see Input/Output tensors
  - Note: `sample_sfd_row`, `sample_sfd_col`, `sample_norm_const` must be all `None` or all not `None`

#### `GroupedGemmQuantSm100.execute`

- `a_tensor`, `b_tensor`, `d_tensor`, `sfa_tensor`, `sfb_tensor`, `padded_offsets`, `alpha_tensor`, `d_col_tensor`, `sfd_row_tensor`, `sfd_col_tensor`, `amax_tensor`, `norm_const_tensor`, `prob_tensor` – see Input/Output tensors. Must have same layout as sample tensors provided in constructor.

---

## Support Surface and Constraints

### Layouts and Strides

- `A` must be **K-major** (contiguous along K dimension)
- `B` must be **K-major** (contiguous along K dimension)
- `D` and `D_col` must be **N-major** (contiguous along N dimension)
- All tensors must be **16-byte aligned** along the contiguous dimension

### Data Types

#### Input/Weight Types (ab_dtype)

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|-------------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` |
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e4m3fn`, `float8_e8m0fnu`} | {16, 32} | `{float16, bfloat16, float32}` |

#### Additional Type Constraints

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- `D` and `D_col` must have the same dtype
- `acc_dtype` must be `float32`
- `sf_dtype=float8_e4m3fn` is incompatible with `sf_vec_size=32`
- FP8 `ab_dtype` is incompatible with `sf_vec_size=16`
- FP4 `ab_dtype` with `sf_vec_size=16` and `d_dtype=float32` is not supported

### Scale Factor Output Requirements

- When `sfd_row_tensor`/`sfd_col_tensor` are provided (FP8 inputs):
  - `sfd_row_tensor`, `sfd_col_tensor`, and `norm_const_tensor` are **all required**
  - These must be provided together (all None or all not None)

- When `d_dtype ∈ {bfloat16, float16}`:
  - `amax_tensor` is required for tracking per-group max values

### Tiling and Cluster

- `mma_tiler_mn[0] = 256` enables 2-CTA instructions automatically (`use_2cta_instrs=True`)
- When `use_2cta_instrs=True`: `cluster_shape_mn[0]` must be divisible by 2
- `m_aligned` must be divisible by `mma_tiler_mn[0]` to prevent tiles from spanning multiple groups

### Shapes and Divisibility

- `padded_offsets` length `L` is the expert count and must be `<= 1024`
- Each group's M dimension is aligned to `m_aligned`
- `valid_m = padded_offsets[-1]` determines the actual tensor M dimension
- Scale factor tensor shapes follow the MMA atom tiling pattern: `(32, 4, ceil(dim/128), 4, ceil(K_groups/4), L)`

### Environment

- Requires CUDA with **SM100+ compute capability** (Blackwell GPUs)

---

## Usage Examples

For usage examples, see test cases in `test/python/fe_api/test_grouped_gemm_quant.py` + `test/python/fe_api/test_grouped_gemm_quant_utils.py`
