# Discrete Grouped GEMM + SwiGLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Discrete Grouped GEMM + SwiGLU fusion**: A block-scaled grouped GEMM fused with a SwiGLU/GeGLU epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE workloads where each expert weight lives in a separate allocation.

Unlike the contiguous grouped API (single packed `B` tensor with shape `(N, K, L)`), this API passes per-expert pointers:

- `b_ptrs`: device `int64` tensor of B pointers (one pointer per expert)
- `sfb_ptrs`: device `int64` tensor of SFB pointers (one pointer per expert)

Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).

This kernel performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4/FP8) using per-expert B and SFB pointers
2. **GLU epilogue**: `act_func="swiglu"` or `act_func="geglu"` applied to GEMM output
3. **Optional quantized output**: Produces row/column scale factors for downstream quantization

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B_g`: expert-`g` weight tensor referenced by `b_ptrs[g]`, logical shape `(N, K)` (or `(N, K, 1)`)
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB_g`: expert-`g` B scale tensor referenced by `sfb_ptrs[g]`, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)` (optional input)
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `C`: intermediate GEMM result, shape `(valid_m, N, 1)`
  - `D`: row-quantized GLU output, shape `(valid_m, N/2, 1)`
  - `D_col`: column-quantized GLU output, shape `(valid_m, N/2, 1)`
  - `SFD_row`: row scale factors (when SFD outputs are enabled; wrapper auto-enables this for FP8-input configs), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when SFD outputs are enabled; wrapper auto-enables this for FP8-input configs), shape `(32, 4, ceil((N/2)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
  - `amax`: per-group amax (optional; wrapper provides it when `d_dtype` is bf16/fp16), shape `(L, 1)`

### Equations

**Step 1: Block-scaled grouped GEMM** (per group `g` with rows `m` in `[padded_offsets[g-1], padded_offsets[g])`):

$$
C[m, n] = \alpha_g \sum_{k} \text{dequantize}(A[m, k], \text{SFA}) \cdot \text{dequantize}(B_g[n, k], \text{SFB}_g)
$$

**Step 2: GLU epilogue** (performed by pairing 32-column blocks along `N`, equations shown for `act_func="swiglu"`):

Let block size `G = 32`. For each pair of consecutive 32-wide column blocks:
- Gate block: `Gate_b = C[:, 2·b·G : 2·b·G + G]`
- Up block: `Up_b = C[:, 2·b·G + G : 2·b·G + 2·G]`

$$
D[:, bG:(b+1)G] = \text{prob} \cdot Up_b \cdot \text{swish}(Gate_b), \quad \text{swish}(x) = x \cdot \sigma(x)
$$

For `act_func="geglu"`, the gate nonlinearity is GeGLU instead of SwiGLU.

**Step 3: Optional output quantization** (when SFD outputs are generated):

$$
\text{SFD\_row}[m, n] = \text{norm\_const} \cdot \max_{k \in \text{block}} |D[m, k]| \cdot \text{rcp\_max}
$$

$$
D_{\text{quantized}}[m, n] = D[m, n] \cdot \frac{\text{norm\_const}}{\text{SFD\_row}[m, n]}
$$

### Diagram

```text
 A (valid_m×K×1)           per-expert B/SFB pointers            padded_offsets
 SFA                          b_ptrs, sfb_ptrs                        |
   |                                  |                                |
   |        +-------------------------+                                |
   |        |                                                          |
   v        v                                                          v
Dequantize → Grouped GEMM (expert selected by row range) → C (valid_m×N×1)
                                |
                                | Pair 32-col blocks: [Gate0|Up0|Gate1|Up1|...]
                                |   act_func in {swiglu, geglu}
                                v
                              D (valid_m×N/2×1)
                                |
                     +----------+-----------+
                     |                      |
                     v                      v
                Row Quantize           Col Quantize
                     |                      |
                     v                      v
                D, SFD_row            D_col, SFD_col
```

---

## API Usage

### High-level Wrapper

```python
from cudnn import discrete_grouped_gemm_swiglu_wrapper_sm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

b_ptrs = torch.tensor([b.data_ptr() for b in b_list], dtype=torch.int64, device="cuda")
sfb_ptrs = torch.tensor([sfb.data_ptr() for sfb in sfb_list], dtype=torch.int64, device="cuda")

outputs = discrete_grouped_gemm_swiglu_wrapper_sm100(
    a_tensor=a_tensor,
    b_ptrs=b_ptrs,
    sfa_tensor=sfa_tensor,
    sfb_ptrs=sfb_ptrs,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha_tensor,
    n=n,                              # logical full N before GLU split
    b_dtype=b_dtype,                  # dtype of per-expert B tensors
    norm_const_tensor=norm_const,     # required when SFD outputs are enabled
    prob_tensor=prob_tensor,          # optional
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
    act_func="swiglu",                # or "geglu"
    b_major="k",                      # or "n" (fp8 only)
    current_stream=stream,
)

# dictionary access:
c = outputs["c_tensor"]               # intermediate GEMM result
d = outputs["d_tensor"]               # row-quantized GLU output
d_col = outputs["d_col_tensor"]       # column-quantized GLU output
amax = outputs["amax_tensor"]         # per-group amax (when d_dtype is bf16/fp16)
sfd_row = outputs["sfd_row_tensor"]   # row scale factors (when enabled)
sfd_col = outputs["sfd_col_tensor"]   # column scale factors (when enabled)

# or tuple unpacking:
c, d, d_col, amax, sfd_row, sfd_col = outputs
```

### Class API

```python
from cudnn import DiscreteGroupedGemmSwigluSm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

api = DiscreteGroupedGemmSwigluSm100(
    sample_a=sample_a,
    num_experts=num_experts,
    b_shape=(n, k),                   # logical (N, K) for one expert
    b_dtype=b_dtype,
    sample_c=sample_c,
    sample_d=sample_d,
    sample_sfa=sample_sfa,
    sample_padded_offsets=sample_padded_offsets,
    sample_alpha=sample_alpha,
    sample_d_col=sample_d_col,
    # Optional quantization outputs
    sample_sfd_row=sample_sfd_row,
    sample_sfd_col=sample_sfd_col,
    sample_amax=sample_amax,
    sample_norm_const=sample_norm_const,
    sample_prob=sample_prob,          # optional
    # Configuration
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=False,
    m_aligned=256,
    discrete_col_sfd=False,
    act_func="swiglu",                # or "geglu"
    b_major="k",                      # or "n" (fp8 only)
)
assert api.check_support()
api.compile()  # descriptor-driven; no runtime tensors required
api.execute(
    a_tensor=a_tensor,
    b_ptrs=b_ptrs,
    c_tensor=c_tensor,
    d_tensor=d_tensor,
    sfa_tensor=sfa_tensor,
    sfb_ptrs=sfb_ptrs,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha_tensor,
    d_col_tensor=d_col_tensor,
    sfd_row_tensor=sfd_row_tensor,
    sfd_col_tensor=sfd_col_tensor,
    amax_tensor=amax_tensor,
    norm_const_tensor=norm_const_tensor,
    prob_tensor=prob_tensor,
    current_stream=stream,
)
```

---

## Parameters

### Input/Output Tensors

- **Input tensor A**: `a_tensor` (wrapper) or `sample_a`, `a_tensor` (class)
  - Shape: `(valid_m, K, 1)`
  - Stride: `(K, 1, valid_m*K)` - **must be K-major**
  - Dtype (`ab_dtype`): `{float4_e2m1fn_x2, uint8, float8_e4m3fn, float8_e5m2}`
    - `uint8` is interpreted as packed FP4 (two FP4 values per byte)

- **Input tensor B pointers**: `b_ptrs` (wrapper/class execute)
  - Shape: `(L,)` where `L = num_experts`
  - Dtype: `int64`, CUDA device tensor
  - Each pointer must reference one expert B tensor with logical shape `(N, K)` (or `(N, K, 1)`) and dtype `b_dtype`
  - Expert B layout is controlled by `b_major` (`"k"` or `"n"`)

- **Input tensor SFB pointers**: `sfb_ptrs` (wrapper/class execute)
  - Shape: `(L,)` where `L = num_experts`
  - Dtype: `int64`, CUDA device tensor
  - Each pointer must reference one expert SFB tensor with shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`

- **Output tensor C**: `c_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N, 1)`
  - Stride: `(N, 1, valid_m*N)` - **must be N-major**
  - Dtype (`c_dtype`):
    - FP4 inputs: `{float16, bfloat16}`
    - FP8 inputs: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`

- **Output tensor D**: `d_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N/2, 1)`
  - Stride: `(N/2, 1, valid_m*(N/2))` - **must be N-major**
  - Dtype (`d_dtype`):
    - FP4 inputs: `{float16, bfloat16, float32}`
    - FP8 inputs: `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`

- **Output tensor D_col**: `d_col_tensor` (class) or returned in wrapper dict
  - Shape: `(valid_m, N/2, 1)`
  - Stride: `(N/2, 1, valid_m*(N/2))` - must match D (N-major)
  - Dtype: Must match D

- **Input tensor prob** (optional): `prob_tensor` (wrapper/class)
  - Shape: `(valid_m, 1, 1)`
  - Dtype: `float32`

- **Scale factor tensors**
  - **SFA** (A scale factor): `sfa_tensor` (wrapper) or `sample_sfa`, `sfa_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
    - Dtype (`sf_dtype`): `{float8_e8m0fnu, float8_e4m3fn}`
  - **SFD_row** (optional): `sfd_row_tensor` (wrapper) or `sample_sfd_row`, `sfd_row_tensor` (class)
    - Shape: `(32, 4, ceil(valid_m/128), 4, ceil(ceil((N/2)/sf_vec_size)/4), 1)`
    - Dtype: Must match SFA
    - **Required when**: SFD outputs are enabled
  - **SFD_col** (optional): `sfd_col_tensor` (wrapper) or `sample_sfd_col`, `sfd_col_tensor` (class)
    - Shape: `(32, 4, ceil((N/2)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
    - Dtype: Must match SFA
    - **Required when**: SFD outputs are enabled

- **Group offsets**
  - **padded_offsets**: Cumulative sum of aligned group M sizes
    - Shape: `(L,)` where `L = num_experts`
    - Dtype: `int32`
    - `padded_offsets[-1]` equals `valid_m`; each offset is a multiple of `m_aligned`

- **Scaling tensors**
  - **alpha**: Per-group scaling factors
    - Shape: `(L,)`
    - Dtype: `float32`
  - **amax** (optional): Per-group max absolute values
    - Shape: `(L, 1)`
    - Dtype: `float32`
    - If provided, updated in-place; wrapper auto-allocates it when `d_dtype in {bfloat16, float16}`
  - **norm_const** (optional): Normalization constant for FP8 quantization
    - Shape: `(1,)`
    - Dtype: `float32`
    - **Required when**: `sfd_row_tensor`/`sfd_col_tensor` are provided

### Common Parameters

- `acc_dtype: torch.dtype`
  - Accumulator dtype. Must be `torch.float32`

- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`. Default: `(256, 256)`
  - `TILE_M in {128, 256}`
  - `TILE_N = 256`

- `cluster_shape_mn: Tuple[int, int] | None`
  - Thread block cluster shape `(CLUSTER_M, CLUSTER_N)`
  - Constraints: positive powers of 2, both <= 4, `CLUSTER_M * CLUSTER_N <= 16`
  - Default: `(2, 1)` when `TILE_M=256`, `(1, 1)` otherwise

- `sf_vec_size: int`
  - Scale factor vector size
  - Allowed values: `{16, 32}`. Default: `16`

- `vector_f32: bool`
  - Enable packed f32 operations
  - Default: `False`

- `m_aligned: int`
  - Alignment requirement for group M dimension
  - Must equal `FIX_PAD_SIZE` (256) and be divisible by `mma_tiler_mn[0]`
  - Default: `256`

- `discrete_col_sfd: bool`
  - If True, generate discrete column scale factors grouped by expert tiles
  - Only applies when SFD outputs are enabled
  - Default: `False`

- `act_func: str`
  - Activation function. Valid values: `"swiglu"`, `"geglu"`
  - Default: `"swiglu"`

- `b_major: str`
  - Expert B layout. Valid values: `"k"`, `"n"`
  - FP4 inputs require `"k"`
  - Default: `"k"`

- CUDA stream (`current_stream` in class API and wrapper)

### Wrapper-specific Parameters: `discrete_grouped_gemm_swiglu_wrapper_sm100`

- `n: int`: Logical full N dimension for expert B (before GLU halves to `N/2`)
- `b_dtype: torch.dtype`: Dtype of expert B tensors referenced by `b_ptrs`
- `c_dtype: torch.dtype`: Intermediate C tensor dtype. Default: `torch.bfloat16`
- `d_dtype: torch.dtype`: Output D tensor dtype. Default: `torch.bfloat16`
- `cd_major: str`: Major dimension for C and D tensors. Must be `"n"`

### Wrapper Return Values

Returns a `TupleDict` - a dictionary-like object that also supports tuple unpacking and integer indexing.

**Dictionary keys** (also tuple unpacking order):
- `c_tensor`: Intermediate GEMM result
- `d_tensor`: Row-quantized GLU output
- `d_col_tensor`: Column-quantized GLU output
- `amax_tensor`: Per-group amax (when `d_dtype in {bfloat16, float16}`)
- `sfd_row_tensor`: Row scale factors (when SFD outputs are enabled)
- `sfd_col_tensor`: Column scale factors (when SFD outputs are enabled)

### Class-specific Parameters

#### `DiscreteGroupedGemmSwigluSm100` (constructor)

- `sample_a`, `num_experts`, `b_shape`, `b_dtype`, `sample_c`, `sample_d`, `sample_sfa`, `sample_padded_offsets`, `sample_alpha`, `sample_d_col`, `sample_sfd_row`, `sample_sfd_col`, `sample_amax`, `sample_norm_const`, `sample_prob` - see Input/Output tensors
  - Note: `sample_sfd_row`, `sample_sfd_col`, `sample_norm_const` must be all `None` or all not `None`
  - `b_shape` must be logical `(N, K)` for one expert (pass logical `K`, not packed `K/2`, for FP4)

#### `DiscreteGroupedGemmSwigluSm100.execute`

- `a_tensor`, `b_ptrs`, `c_tensor`, `d_tensor`, `sfa_tensor`, `sfb_ptrs`, `padded_offsets`, `alpha_tensor`, `d_col_tensor`, `sfd_row_tensor`, `sfd_col_tensor`, `amax_tensor`, `norm_const_tensor`, `prob_tensor` - see Input/Output tensors. Layouts must match constructor sample descriptors.

---

## Support Surface and Constraints

### Layouts and Strides

- `A` must be **K-major** (contiguous along K dimension)
- Expert `B` layout is selected by `b_major`:
  - `b_major="k"`: K-major
  - `b_major="n"`: N-major (FP8 configs only)
- `C`, `D`, and `D_col` must be **N-major**
- All tensors must be **16-byte aligned** along the contiguous dimension

### Data Types

#### Input/Weight Types (ab_dtype)

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|---------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `float8_e8m0fnu` | 32 | `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}` |
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e8m0fnu`, `float8_e4m3fn`} | {16, 32} | `{float16, bfloat16, float32}` |

#### Additional Type Constraints

- `b_dtype` must match `A` dtype
- `SFA`, `SFD_row`, and `SFD_col` must share dtype
- `D` and `D_col` must have the same dtype
- `acc_dtype` must be `float32`
- `sf_dtype=float8_e4m3fn` with `sf_vec_size=32` is not supported
- FP8 `ab_dtype` with `sf_vec_size=16` is not supported
- FP4 `ab_dtype` with `sf_vec_size=16` and `d_dtype=float32` is not supported
- FP4 `ab_dtype` requires `c_dtype in {float16, bfloat16}`
- FP4 `ab_dtype` requires `b_major="k"`

### Scale Factor Output Requirements

- When `sfd_row_tensor`/`sfd_col_tensor` are provided:
  - `sfd_row_tensor`, `sfd_col_tensor`, and `norm_const_tensor` are **all required**
  - These must be provided together (all `None` or all not `None`)

- `amax_tensor` is optional:
  - If provided, it is updated in-place with per-group maxima
  - Wrapper auto-allocates it when `d_dtype in {bfloat16, float16}`

### Tiling and Cluster

- `mma_tiler_mn[0] = 256` enables 2-CTA instructions (`use_2cta_instrs=True`)
- `mma_tiler_mn[0] = 128` uses the non-2CTA instruction path
- When `use_2cta_instrs=True`: `cluster_shape_mn[0]` must be divisible by 2
- `m_aligned` must be divisible by `mma_tiler_mn[0]`
- `m_aligned` must equal `FIX_PAD_SIZE=256`

### Shapes and Divisibility

- `N` is consumed in paired 32-column blocks by the GLU epilogue (use `N` divisible by 64)
- `padded_offsets` length `L` is expert count and must be `<= 1024`
- `valid_m = padded_offsets[-1]` determines actual M size
- `b_ptrs` and `sfb_ptrs` must be CUDA `int64` tensors with shape `(L,)`

### Environment

- Requires CUDA with **SM100+ compute capability** (Blackwell GPUs)

---

## Usage Examples

For runnable examples and validation, see:
- `test/python/fe_api/grouped_gemm/test_discrete_grouped_gemm_swiglu.py`
- `test/python/fe_api/grouped_gemm/test_discrete_grouped_gemm_swiglu_utils.py`
