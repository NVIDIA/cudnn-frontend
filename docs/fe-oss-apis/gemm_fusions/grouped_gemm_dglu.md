# Grouped GEMM + dGLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**Unified Grouped GEMM + dGLU fusion**: one public class and wrapper select a
plain BF16 or legacy block-scaled grouped GEMM fused with a dGLU backward
epilogue (dSwiGLU or dGeGLU) on NVIDIA Blackwell GPUs (SM100+). The operation
is implemented with CUTLASS/CuTe DSL.

This is a **unified API** that supports both weight layout modes:
- **Dense mode**: All expert weights packed into a single contiguous `(N, K, L)` tensor
- **Discrete mode**: Per-expert weight pointers (no weight stacking required)

And both backward activation functions:
- **dSwiGLU**: `act_func="dswiglu"` (default)
- **dGeGLU**: `act_func="dgeglu"`

Groups are contiguous in the M dimension and described by `padded_offsets` (cumulative aligned end offsets).

### Backend dispatch

| Operand contract | Selected backend |
| --- | --- |
| `A` and `B` are BF16 | BF16 |
| matching supported FP4/FP8 `A` and `B` plus scale descriptors | block-scaled |

Mixed families and unsupported pairs are rejected before allocation or
compilation. Each backend's argument contract is described below.

## BF16 contract

Pass `sfa_tensor=None`, `sfb_tensor=None` (or `sfb_ptrs=None`), and
`norm_const_tensor=None`; keep `sf_vec_size=16`, `discrete_col_sfd=False`, and
`epilogue_op=None`. BF16 also uses the source GeGLU constants
`geglu_alpha=1.702`, `glu_clamp_max=7`, and `glu_clamp_min=-7`.

### Tensors, layouts, and equation

For padded rows `M`, reduction dimension `K`, compact gradient width `N`, and
`L` experts, BF16 uses:

- `A`: `(M, K, 1)`, stride `(K, 1, M*K)`, BF16;
- dense `B`: `(N, K, L)`, K-major stride `(K, 1, N*K)`, BF16;
- discrete `b_ptrs`: contiguous CUDA int64 pointers to expert `(N, K)` BF16
  matrices, with `n=N`, `b_dtype=torch.bfloat16`, and `b_major="k"` or `"n"`;
- `C`: `(M, 2N, 1)`, stride `(2N, 1, 2M*N)`, BF16/FP16/FP32;
- `padded_offsets`: `(L,)` int32 cumulative 256-aligned ends;
- `alpha`, `beta`: `(L,)` FP32; `prob`: `(M, 1, 1)` FP32;
- caller-zeroed `dprob`: `(M, 1, 1)`, stride `(1, 1, 1)`, FP32;
- `D_row`: `(M, 2N, 1)`, stride `(2N, 1, 2M*N)`, BF16/FP16/FP32;
- caller-zeroed optional `dbias`: `(L, 2N, 1)`, stride `(2N, 1, 1)`, BF16.

For expert `g`, the compact GEMM gradient and scaled forward activation are

$$
R_g = \alpha_g^2 A_g B_g^T, \qquad X_g = \beta_g C_g.
$$

Split `X` into alternating 32-wide gate/input blocks. For dSwiGLU, with
`s = sigmoid(gate)`:

$$
d\mathrm{input} = R\,\mathrm{prob}\,(\mathrm{gate}\,s),
$$

$$
d\mathrm{gate} = R\,\mathrm{prob}\,\mathrm{input}\,s
                  (1 + \mathrm{gate}(1-s)).
$$

For dGeGLU, distinguish raw values, clamped activation values, and the source's
value-bearing filters:

```text
raw_gate = gate(X)
raw_input = input(X)
clamped_gate = min(raw_gate, 7)
clamped_input = clamp(raw_input, -7, 7)
gate_filter = raw_gate if raw_gate <= 7 else 0
input_filter = raw_input if -7 <= raw_input <= 7 else 0
s = sigmoid(1.702 * clamped_gate)
```

With the default `linear_offset=1`, the kernel computes

$$
d\mathrm{gate} = R\,\mathrm{prob}\,
                  (\mathrm{clamped\_input}+\mathrm{linear\_offset})\,s
                  (1 + 1.702\,\mathrm{clamped\_gate}(1-s))\,
                  \mathrm{gate\_filter},
$$

$$
d\mathrm{input} = R\,\mathrm{prob}\,\mathrm{clamped\_gate}\,s\,
                   \mathrm{input\_filter}.
$$

`dprob` accumulates the row sum of the matching unscaled activation times `R`;
`dbias` is the per-expert row reduction of interleaved `D_row`. The
pointer-array tensor is stream-recorded, while every pointed allocation must
remain alive and unchanged until that stream completes.

The wrapper return order is exactly `d_row_tensor`, `d_col_tensor`,
`dprob_tensor`, `dbias_tensor`, `amax_tensor`, `sfd_row_tensor`,
`sfd_col_tensor`. On BF16, `d_col_tensor`, `amax_tensor`, `sfd_row_tensor`, and
`sfd_col_tensor` are `None`; `dbias_tensor` is `None` unless
`generate_dbias=True`.

## Block-scaled contract

The block-scaled backend performs:
1. **Block-scaled grouped GEMM**: Low-precision GEMM (FP4, FP8) with per-block scale factors across multiple expert groups
2. **dGLU backward epilogue**: Fused backward computation using the forward `C` tensor (input/gate interleaved)
3. **Optional quantized output**: Produces row and column scale factors for downstream quantization

### Shapes

### Equations

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B` (dense): weight tensor across all groups, shape `(N, K, L)`
  - `B` (discrete): per-expert weight pointers, `b_ptrs` shape `(num_experts,)` of int64
  - `C`: forward intermediate tensor with interleaved input/gate blocks, shape `(valid_m, 2N, 1)`
  - `SFA`: scale factor tensor for A, shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)`
  - `SFB` (dense): scale factor tensor for B, shape `(32, 4, ceil(N/128), 4, ceil(ceil(K/sf_vec_size)/4), L)`
  - `SFB` (discrete): per-expert SFB pointers, `sfb_ptrs` shape `(num_experts,)` of int64
  - `padded_offsets`: cumulative sum of aligned group M sizes, shape `(L,)`. `valid_m = padded_offsets[-1]`
  - `alpha`: per-group scaling factors for GEMM, shape `(L,)`
  - `beta`: per-group scaling factors for `C`, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`
  - `norm_const`: normalization constant for FP8 quantization, shape `(1,)`
- **Outputs**
  - `D_row`: row-quantized dGLU output, shape `(valid_m, 2N, 1)`
  - `D_col`: column-quantized dGLU output, shape `(valid_m, 2N, 1)`
  - `dprob`: gradient of `prob`, shape `(valid_m, 1, 1)`. Must be zero-initialized.
  - `dbias` (optional): per-expert bias gradient tensor, shape `(L, 2N, 1)`
  - `SFD_row`: row scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil(valid_m/128), 4, ceil(ceil((2N)/sf_vec_size)/4), 1)`
  - `SFD_col`: column scale factors (when `d_dtype` is FP8), shape `(32, 4, ceil((2N)/128), 4, ceil(ceil(valid_m/sf_vec_size)/4), 1)`
  - `amax`: per-group amax (when `d_dtype` is bf16/float16), shape `(L, 2, 1)`

**Step 1: Block-scaled grouped GEMM** (per group `g` with rows `m` in `[padded_offsets[g-1], padded_offsets[g])`):

$$
\text{ref}[m, n] = \alpha_g^2 \sum_{k} \text{dequantize}(A[m, k], \text{SFA}) \cdot \text{dequantize}(B[n, k, g], \text{SFB})
$$

**Step 2: dGLU backward epilogue** (performed with 32-column interleaving along `2N`):

- Scale `C` by `beta_g` per group and deinterleave into input/gate halves by 32-wide blocks.

For **dSwiGLU** (`act_func="dswiglu"`):
- `swish = gate * sigmoid(gate)`
- `dprob += sum(swish * input * ref)` over 32-column chunks
- `ab = ref * prob * swish`
- `dswiglu = ref * prob * input * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))`
- Interleave `[ab, dswiglu]` back into `D_row`/`D_col` with 32-column blocks.

For **dGeGLU** (`act_func="dgeglu"`): Uses `sigmoid(1.702 * gate)` scaling in the backward computation.

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
  Dequantize → Grouped GEMM (per group ranges) → ref
                    |
                    | × alpha[group_idx]
                    v
               ref (valid_m×N×1)
                    |
 C (valid_m×2N×1) --× beta[group_idx]--> deinterleave 32-col blocks
                    |                    |
                    |                swish, sigmoid
                    |                    |
                    +--> dprob (sum over blocks)
                    |
                    +--> ab, dswiglu → interleave → D (valid_m×2N×1)
                                      |
                         +-----------+-----------+
                         |                       |
                         v                       v
                    Row Quantize            Col Quantize
                         |                       |
                         v                       v
                    D_row, SFD_row         D_col, SFD_col
```

---

## API usage

### BF16

#### High-level wrapper

```python
import cudnn
import torch

dprob.zero_()
out = cudnn.grouped_gemm_dglu_wrapper_sm100(
    a_tensor=a,
    c_tensor=c,
    sfa_tensor=None,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    beta_tensor=beta,
    prob_tensor=prob,
    dprob_tensor=dprob,
    b_tensor=b,
    sfb_tensor=None,
    act_func="dswiglu",
    generate_dbias=True,
    use_dynamic_sched=True,
)
d_row, d_col, dprob, dbias, amax, sfd_row, sfd_col = out

# Discrete mode replaces the dense weight arguments.
out = cudnn.grouped_gemm_dglu_wrapper_sm100(
    a_tensor=a, c_tensor=c, sfa_tensor=None,
    padded_offsets=padded_offsets, alpha_tensor=alpha, beta_tensor=beta,
    prob_tensor=prob, dprob_tensor=dprob,
    b_ptrs=b_ptrs, sfb_ptrs=None, n=N, b_dtype=torch.bfloat16,
    act_func="dgeglu",
)
```

#### Class API

```python
op = cudnn.GroupedGemmDgluSm100(
    sample_a=a, sample_c=c, sample_d_row=d_row, sample_d_col=None,
    sample_sfa=None, sample_padded_offsets=padded_offsets,
    sample_alpha=alpha, sample_beta=beta, sample_prob=prob,
    sample_dprob=dprob, sample_dbias=dbias,
    sample_b=b, sample_sfb=None, act_func="dswiglu",
)
assert op.check_support()
op.compile()
dprob.zero_()
dbias.zero_()
op.execute(
    a_tensor=a, c_tensor=c, d_row_tensor=d_row, d_col_tensor=None,
    sfa_tensor=None, padded_offsets=padded_offsets, alpha_tensor=alpha,
    beta_tensor=beta, prob_tensor=prob, dprob_tensor=dprob,
    dbias_tensor=dbias, b_tensor=b, sfb_tensor=None,
)
```

`use_dynamic_sched=False` selects static scheduling; `True` caches a dynamic-M
callable for compatible shapes. Cache keys retain compile-sensitive layouts,
dtypes, activation, dbias policy, scheduler, tiles/clusters, features, and
overlap margin.

### Block-scaled

#### High-level wrapper

**Dense mode:**

```python
from cudnn import grouped_gemm_dglu_wrapper_sm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

outputs = grouped_gemm_dglu_wrapper_sm100(
    a_tensor=a,
    c_tensor=c,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    beta_tensor=beta,
    prob_tensor=prob,
    dprob_tensor=dprob,
    generate_dbias=True,
    # Dense mode weights:
    b_tensor=b,
    sfb_tensor=sfb,
    # Common:
    norm_const_tensor=norm_const,
    acc_dtype=torch.float32,
    d_dtype=torch.bfloat16,
    cd_major="n",
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    act_func="dswiglu",
    epilogue_op=None,
    current_stream=stream,
)

# dictionary access:
d_row = outputs["d_row_tensor"]
d_col = outputs["d_col_tensor"]
dprob = outputs["dprob_tensor"]
dbias = outputs["dbias_tensor"]
amax = outputs["amax_tensor"]
sfd_row = outputs["sfd_row_tensor"]
sfd_col = outputs["sfd_col_tensor"]

# or tuple unpacking:
d_row, d_col, dprob, dbias, amax, sfd_row, sfd_col = outputs
```

**Discrete mode:**

```python
outputs = grouped_gemm_dglu_wrapper_sm100(
    a_tensor=a,
    c_tensor=c,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    beta_tensor=beta,
    prob_tensor=prob,
    dprob_tensor=dprob,
    # Discrete mode weights:
    b_ptrs=b_ptrs,
    sfb_ptrs=sfb_ptrs,
    n=n_dim,
    b_dtype=torch.uint8,
    b_major="k",
    # Common:
    act_func="dgeglu",
    current_stream=stream,
)
```

#### Class API

**Dense mode:**

```python
from cudnn import GroupedGemmDgluSm100
from cuda.bindings import driver as cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

api = GroupedGemmDgluSm100(
    sample_a=a,
    sample_c=c,
    sample_d_row=d_row,
    sample_d_col=d_col,
    sample_sfa=sfa,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_beta=beta,
    sample_prob=prob,
    sample_dprob=dprob,
    sample_dbias=dbias,
    # Dense mode:
    sample_b=b,
    sample_sfb=sfb,
    # Optional quantization outputs
    sample_sfd_row=sfd_row,
    sample_sfd_col=sfd_col,
    sample_amax=amax,
    sample_norm_const=norm_const,
    # Configuration
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    act_func="dswiglu",
    epilogue_op=None,
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a, c_tensor=c, d_row_tensor=d_row, d_col_tensor=d_col,
    sfa_tensor=sfa, padded_offsets=padded_offsets, alpha_tensor=alpha,
    beta_tensor=beta, prob_tensor=prob, dprob_tensor=dprob, dbias_tensor=dbias,
    b_tensor=b, sfb_tensor=sfb,
    sfd_row_tensor=sfd_row, sfd_col_tensor=sfd_col,
    amax_tensor=amax, norm_const_tensor=norm_const,
    current_stream=stream,
)
```

In the class API, dbias generation is specialized at compile time: if `sample_dbias` is omitted, `dbias_tensor` must also be omitted at `execute()`.

**Discrete mode:**

```python
api = GroupedGemmDgluSm100(
    sample_a=a,
    sample_c=c,
    sample_d_row=d_row,
    sample_d_col=d_col,
    sample_sfa=sfa,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_beta=beta,
    sample_prob=prob,
    sample_dprob=dprob,
    # Discrete mode:
    num_experts=num_experts,
    b_shape=(n, k),
    b_dtype=torch.uint8,
    # Configuration
    act_func="dgeglu",
    b_major="k",
    epilogue_op="relu",
)
assert api.check_support()
api.compile()
api.execute(
    a_tensor=a, c_tensor=c, d_row_tensor=d_row, d_col_tensor=d_col,
    sfa_tensor=sfa, padded_offsets=padded_offsets, alpha_tensor=alpha,
    beta_tensor=beta, prob_tensor=prob, dprob_tensor=dprob,
    b_ptrs=b_ptrs, sfb_ptrs=sfb_ptrs,
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
  - Stride: K-major or N-major. Must be K-major for FP4.
  - Dtype: Must match A

- **Input B pointers** (discrete mode): `b_ptrs`
  - Shape: `(num_experts,)` -- 1-D int64 device tensor of per-expert B data pointers

- **Input tensor C**: `c_tensor` / `sample_c`
  - Shape: `(valid_m, 2N, 1)` -- forward activation with interleaved input/gate
  - Stride: `(2N, 1, valid_m·2N)` -- **must be N-major**
  - Dtype: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}`

- **Output tensor D_row**: `d_row_tensor` / `sample_d_row`
  - Shape: `(valid_m, 2N, 1)`
  - Stride: `(2N, 1, valid_m·2N)` -- **must be N-major**
  - Dtype (`d_dtype`): `{bfloat16, float32}` for FP4; `{float8_e4m3fn, float8_e5m2}` for FP8

- **Output tensor D_col**: `d_col_tensor` / `sample_d_col`
  - Shape: `(valid_m, 2N, 1)` -- must match D_row dtype and stride

- **Input tensor prob**: `prob_tensor` / `sample_prob`
  - Shape: `(valid_m, 1, 1)`, dtype: `float32`

- **Output tensor dprob**: `dprob_tensor` / `sample_dprob`
  - Shape: `(valid_m, 1, 1)`, dtype: `float32`
  - Must be zero-initialized

- **Scaling tensors**: `alpha` shape `(L,)`, `beta` shape `(L,)`, `amax` shape `(L, 2, 1)`, `norm_const` shape `(1,)`

### Common Parameters

- `acc_dtype`: Must be `torch.float32`
- `mma_tiler_mn`: Kernel tile size. Default: `(256, 256)`
  - `TILE_M ∈ {128, 256}`
  - `TILE_N = 256`
- `cluster_shape_mn`: Thread Block cluster shape. Default: `(2, 1)` when `TILE_M=256`, `(1, 1)` otherwise
- `sf_vec_size`: Scale factor vector size. `{16, 32}`. Default: `16`
- `vector_f32`: Enable packed f32 operations. Default: `False`
- `m_aligned`: Must be `256`. Default: `256`
- `discrete_col_sfd`: Generate discrete col-major scale factors. Default: `False`
- `act_func`: Backward activation function. `"dswiglu"` (default) or `"dgeglu"`
- `b_major` (discrete only): B tensor major dimension. `"k"` (default) or `"n"`. Must be `"k"` for FP4.
- `epilogue_op`: Optional post-processing. `None` (default), `"identity"`, `"relu"`, or `"srelu"`

### Wrapper-specific Parameters

- `d_dtype`: Output D tensor data type. Default: `torch.bfloat16`
- `cd_major`: Must be `"n"`. Default: `"n"`
- `n` (discrete only): B weight N dimension
- `b_dtype` (discrete only): B weight data type

### Wrapper Return Values

Returns a `TupleDict` (dictionary + tuple unpacking):
- `d_row_tensor`: Row-quantized dGLU output
- `d_col_tensor`: Column-quantized dGLU output
- `dprob_tensor`: Gradient of prob
- `amax_tensor`: Per-group amax (when `d_dtype` is bf16/fp16)
- `sfd_row_tensor`: Row scale factors (when SFD enabled)
- `sfd_col_tensor`: Column scale factors (when SFD enabled)

---

## Support Surface and Constraints

### Layouts and Strides

- `A` must be **K-major**
- `B` must be **K-major** (dense) or K/N-major (discrete). Must be K-major for FP4.
- `C`, `D_row`, `D_col` must be **N-major**
- All tensors must be **16-byte aligned** along the contiguous dimension

### Data Types

| Format | ab_dtype | sf_dtype | sf_vec_size | d_dtype |
|--------|----------|----------|-------------|---------|
| **MXFP8** | `float8_e4m3fn` or `float8_e5m2` | `{float8_e8m0fnu, float8_e4m3fn}` | 32 | `{float8_e4m3fn, float8_e5m2}` |
| **NVF4** | `float4_e2m1fn_x2` or `uint8` | {`float8_e4m3fn`, `float8_e8m0fnu`} | {16, 32} | `{bfloat16, float32}` |

### Additional Type Constraints

- `A` and `B` must have the same dtype
- Scale factor tensors must have the same dtype
- `D_row` and `D_col` must have the same dtype
- `dbias` must be `bfloat16`
- `sf_dtype=float8_e4m3fn` is incompatible with `sf_vec_size=32`
- FP8 `c_dtype` with `vector_f32=True` is not supported
- FP4 `ab_dtype` only supports `d_dtype` in `{bfloat16, float32}`
- For non-dbias paths, FP4 `ab_dtype` with `sf_vec_size=16` and `d_dtype=float32` is not supported
- FP8 `ab_dtype` only supports `d_dtype` in `{float8_e4m3fn, float8_e5m2}`

### Shapes and Divisibility

- `N` must be divisible by 32 (32-column blocks for input/gate interleaving)
- Expert count must be `<= 1024`
- Each group's M dimension is aligned to `m_aligned` (256)
- In the class API, `dbias` is compiled in only when `sample_dbias` is provided; passing a runtime `dbias_tensor` without `sample_dbias` raises `ValueError`
- `use_single_group_runtime_offsets=True` is supported only by the block-scaled
  kernel with exactly one expert. In this mode the kernel derives
  `padded_offsets[0]` from runtime `A.shape[0]` and does not load its value from
  device memory; the argument must still be an int32 tensor with shape `(1,)`.

### Environment

- Requires CUDA with **SM100+ compute capability** (Blackwell GPUs)

---

## Usage Examples

For usage examples, see test cases in `test/python/fe_api/grouped_gemm/test_grouped_gemm_dglu.py` (dense mode, unified API) and `test/python/fe_api/grouped_gemm/test_discrete_grouped_gemm_dswiglu.py` (discrete mode).
