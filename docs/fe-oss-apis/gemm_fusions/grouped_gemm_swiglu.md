# Grouped GEMM + SwiGLU (SM100)

**This is an experimental API and subject to change.**

**Legacy contiguous-only API note:** This page documents the older contiguous-only SwiGLU API. For new integrations, prefer the unified [Grouped GEMM + GLU](grouped_gemm_glu.md) API, which covers dense and discrete weight layouts.

## Overview

**Grouped GEMM + SwiGLU fusion**: A contiguous grouped block-scaled GEMM fused with a SwiGLU epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads. Implemented with CUTLASS/CUTE.
Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).

This kernel performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4, FP8) with per-block scale factors across multiple expert groups
2. **SwiGLU activation**: Fused activation applied to the GEMM output
3. **Optional quantized output**: Produces row and column scale factors for downstream quantization

### Shapes

### Equations

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: weight tensor across all groups, shape `(N, K, L)`
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB`: scale factor tensor for B, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `C`: intermediate GEMM result, shape `(valid_m, N, 1)`
  - `D`: row-quantized SwiGLU output, shape `(valid_m, N/2, 1)`
  - `D_col`: column-quantized SwiGLU output, shape `(valid_m, N/2, 1)`
  - `SFD_row`: row scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil((N/2)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
  - `amax`: per-group amax (when `d_dtype` is bf16/fp16), shape `(L, 1)`

**Step 1: Block-scaled grouped GEMM** (per group `g` with rows `m` in `[padded_offsets[g-1], padded_offsets[g])`):

$$
C[m, n] = \alpha_g \sum_{k} \text{dequantize}(A[m, k], \text{SFA}) \cdot \text{dequantize}(B[n, k, g], \text{SFB})
$$

**Step 2: SwiGLU epilogue** (performed by pairing 32-column blocks along `N`):

Let block size `G = 32`. For each pair of consecutive 32-wide column blocks:
- Input block: `X_b = C[:, 2·b·G : 2·b·G + G]`
- Gate block: `G_b = C[:, 2·b·G + G : 2·b·G + 2·G]`

$$
D[:, bG:(b+1)G] = \text{prob} \cdot X_b \cdot \text{swish}(G_b), \quad \text{swish}(x) = x \cdot \sigma(x)
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
               C (valid_m×N×1)
                    |
                    | Pair 32-col blocks: [X0|G0|X1|G1|...]
                    |     X_b × swish(G_b)
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

```python
from cudnn import grouped_gemm_swiglu_wrapper_sm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

outputs = grouped_gemm_swiglu_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    norm_const_tensor=norm_const,  # Required when SFD outputs are enabled (FP8 inputs)
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
    current_stream=stream,
)

# dictionary access:
c = outputs["c_tensor"] # intermediate GEMM result
d = outputs["d_tensor"] # row-quantized SwiGLU output
d_col = outputs["d_col_tensor"] # column-quantized SwiGLU output
amax = outputs["amax_tensor"] # per-group amax (when d_dtype is bf16)
sfd_row = outputs["sfd_row_tensor"] # row scale factors (when d_dtype is FP8)
sfd_col = outputs["sfd_col_tensor"] # column scale factors (when d_dtype is FP8)

# or tuple unpacking:
c, d, d_col, amax, sfd_row, sfd_col = outputs
```

### Class API

```python
from cudnn import GroupedGemmSwigluSm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

api = GroupedGemmSwigluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
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
    sample_prob=prob,              # Optional gating probabilities
    # Configuration
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
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

- **Output tensor C**: `c_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N, 1)`
  - Stride: `(N, 1, valid_m·N)` – **must be N-major**
  - Dtype (`c_dtype`): `{float16, bfloat16}` for FP4 inputs; `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` otherwise

- **Output tensor D**: `d_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N/2, 1)`
  - Stride: `(N/2, 1, valid_m·N/2)` – **must be N-major**
  - Dtype (`d_dtype`): `{bfloat16, float32}` for FP4 inputs; `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` otherwise

- **Output tensor D_col**: `d_col_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N/2, 1)`
  - Stride: `(N/2, 1, valid_m·N/2)` – must match D (N-major)
  - Dtype: Must match D

- **Scale factor tensors**
  - **SFA** (A scale factor): `sfa_tensor` (wrapper) or `sample_sfa`, `sfa_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
    - Dtype (`sf_dtype`): `{float8_e8m0fnu, float8_e4m3fn}`
  - **SFB** (B scale factor): `sfb_tensor` (wrapper) or `sample_sfb`, `sfb_tensor` (class)
    - Shape: `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
    - Dtype: Must match SFA
  - **SFD_row** (D row scale factor, optional): `sfd_row_tensor` (wrapper) or `sample_sfd_row`, `sfd_row_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), 1)`
    - Dtype: Must match SFA
    - **Required when**: SFD outputs are enabled (FP8 inputs)
  - **SFD_col** (D column scale factor, optional): `sfd_col_tensor` (wrapper) or `sample_sfd_col`, `sfd_col_tensor` (class)
    - Shape: `(32, 4, ceil((N/2)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
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
  - **prob** (optional): Per-row gating probabilities
    - Shape: `(valid_m, 1, 1)`
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
  - `TILE_M ∈ {64, 128, 256}`
  - `TILE_N ∈ {128, 256}`

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
  - Alignment requirement for group M dimension
  - Must equal `FIX_PAD_SIZE` (256) and be divisible by `mma_tiler_mn[0]`
  - Default: `256`

- `discrete_col_sfd: bool`
  - If True, generate discrete column scale factors grouped by expert tiles
  - Only applies when `sfd_row_tensor`, `sfd_col_tensor`, and `norm_const_tensor` are provided
  - Default: `False`

- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper-specific Parameters: `grouped_gemm_swiglu_wrapper_sm100`

- `c_dtype: torch.dtype`: Intermediate C tensor data type. Default: `torch.bfloat16`
- `d_dtype: torch.dtype`: Output D tensor data type. Default: `torch.bfloat16`
- `cd_major: str`: Major dimension for C and D tensors. Must be `"n"` (only N-major layout is supported). Default: `"n"`

### Wrapper Return Values

Returns a `TupleDict` - a dictionary-like object that also supports tuple unpacking and integer indexing.

**Dictionary keys** (also the tuple unpacking order):
- `c_tensor`: Intermediate GEMM result
- `d_tensor`: Row-quantized SwiGLU output
- `d_col_tensor`: Column-quantized SwiGLU output
- `amax_tensor`: Per-group amax (when `d_dtype ∈ {bfloat16, float16}`)
- `sfd_row_tensor`: Row scale factors (when SFD outputs are enabled)
- `sfd_col_tensor`: Column scale factors (when SFD outputs are enabled)

### Class-specific Parameters

#### `GroupedGemmSwigluSm100` (constructor)

- `sample_a`, `sample_b`, `sample_c`, `sample_d`, `sample_sfa`, `sample_sfb`, `sample_padded_offsets`, `sample_alpha`, `sample_d_col`, `sample_sfd_row`, `sample_sfd_col`, `sample_amax`, `sample_norm_const`, `sample_prob` – see Input/Output tensors
  - Note: `sample_sfd_row`, `sample_sfd_col`, `sample_norm_const` must be all `None` or all not `None`

#### `GroupedGemmSwigluSm100.execute`

- `a_tensor`, `b_tensor`, `c_tensor`, `d_tensor`, `sfa_tensor`, `sfb_tensor`, `padded_offsets`, `alpha_tensor`, `d_col_tensor`, `sfd_row_tensor`, `sfd_col_tensor`, `amax_tensor`, `norm_const_tensor`, `prob_tensor` – see Input/Output tensors. Must have same layout as sample tensors provided in constructor.

---

## Support Surface and Constraints

### Layouts and Strides

- `A` and `B` must be **K-major** (contiguous along K dimension)
- `C`, `D`, and `D_col` must be **N-major** (contiguous along N dimension)
- All tensors must be **16-byte aligned** along the contiguous dimension

### Data Types

#### Input/Weight Types (ab_dtype)

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|-------------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | `{float16, bfloat16, float8_e4m3fn, float8_e5m2}` | 
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e4m3fn`, `float8_e8m0fnu`} | {16, 32} | `{bfloat16, float32}` |

#### Additional Type Constraints

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- `D` and `D_col` must have the same dtype
- `acc_dtype` must be `float32`
- FP4 `ab_dtype` with `sf_vec_size=16` and `d_dtype=float32` is not supported
- FP8 `ab_dtype` with `mma_tiler_mn[1]=128` and FP8 `d_dtype` is not supported
- FP4 `ab_dtype` is not compatible with FP8 `c_dtype`

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

- `N` must be divisible by 64 (two consecutive 32-column blocks for SwiGLU pairing)
- `padded_offsets` length `L` is the expert count and must be `<= 1024`
- Each group's M dimension is aligned to `m_aligned`
- `valid_m = padded_offsets[-1]` determines the actual tensor M dimension
- Scale factor tensor shapes follow the MMA atom tiling pattern: `(32, 4, ceil(dim/128), 4, ceil(K_groups/4), L)`

### Environment

- Requires CUDA with **SM100+ compute capability** (Blackwell GPUs)

---

## Usage Examples

For usage examples, see test cases in `test/python/fe_api/test_grouped_gemm_swiglu.py` + `test/python/fe_api/test_grouped_gemm_swiglu_utils.py`
