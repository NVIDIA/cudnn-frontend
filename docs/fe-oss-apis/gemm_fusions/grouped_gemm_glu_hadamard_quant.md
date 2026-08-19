# Grouped GEMM + GLU + Hadamard + Quant (SM100)

**This is an experimental API and subject to change.**

## JAX support

JAX arrays are **not supported**: this fusion is block-scaled-only and its mandatory scale-factor inputs use an MMA-interleaved layout with no row-major (JAX) equivalent. JAX inputs raise a clear `ValueError` at the entry points. The API is otherwise type-erased and torch-lazy.

## Overview

**Grouped GEMM + GLU + Hadamard + Quant fusion**: A contiguous grouped block-scaled GEMM fused with a GLU/SReLU epilogue, optional RHT (Hadamard transform) output, and optional NVFP4 output quantization on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This frontend integration is currently wired for the FP4 input path and exposes the quantized Hadamard fusion under the operation name:

- `GroupedGemmGluHadamardQuantSm100`
- `grouped_gemm_glu_hadamard_quant_wrapper_sm100`

This kernel performs:

1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **GLU or SReLU epilogue** using per-row `prob`
3. **Optional NVFP4 quantization** of the post-activation `D` output
4. **Optional RHT output** in bf16 or NVFP4 form

### Shapes

Let `N_out = N / 2` for `act_func="swiglu"` or `"geglu"` and `N_out = N` for `act_func="srelu"`. Let `SF(rows, cols)` denote the swizzled scale-factor layout `(32, 4, ceil_div(rows, 128), 4, ceil_div(ceil_div(cols, sf_vec_size), 4), 1)`.

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: weight tensor across all groups, shape `(N, K, L)` in dense mode
  - `b_ptrs` / `sfb_ptrs`: int64 CUDA pointer arrays, shape `(L,)`, in discrete mode
  - `SFA`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - `SFB`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)` in dense mode
  - `padded_offsets`: cumulative padded group ends, shape `(L,)`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`
  - `bias` (optional): per-expert bias tensor, shape `(N, L)` with stride `(1, N)`

- **Outputs**
  - `C`: intermediate GEMM result before activation/clamping, shape `(valid_m, N, 1)`
  - `D`: post-activation output, logical shape `(valid_m, N_out, 1)`
  - `SFD`: swizzled e4m3 scale factors for NVFP4 `D`, shape `SF(valid_m, N_out)`, present only when `D` is NVFP4
  - `RHT`: optional Hadamard-transform output, logical shape `(valid_m, N_out, 1)`
  - `SFRHT`: e4m3 scale factors for NVFP4 `RHT`, present only when `RHT` is NVFP4
    - Rowwise RHT: swizzled shape `SF(valid_m, N_out)`
    - Colwise RHT: swizzled shape `SF(N_out, valid_m)`

For packed NVFP4 output tensors (`torch.float4_e2m1fn_x2`), the physical tensor stores two logical values per byte along the innermost dimension. The wrapper therefore allocates packed `D` and `RHT` tensors with physical second dimension `N_out / 2`. Raw `torch.uint8` tensors are not accepted as a packed FP4 container by this fusion. RHT data is always stored in the same logical `(m, feature)` orientation as `D`; `rht_rowwise` only changes the transform axis and, for quantized RHT, the `SFRHT` scale domain.

`L` is the expert count. `valid_m` is the `M` extent of `a_tensor`; by contract it must match the final cumulative padded offset in `padded_offsets`.

### Equations

For rows belonging to expert `g`:

$$
C[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB) + \mathrm{bias}_g[n]
$$

The `bias` term is omitted when `bias_tensor=None`. `C` stores this unclamped GEMM-plus-bias value.

Split the `N` dimension into consecutive 32-column gate/up blocks:

$$
G_b = C[:, 2bG:(2b+1)G], \quad U_b = C[:, (2b+1)G:(2b+2)G], \quad G = 32
$$

When `glu_limit` is set for `swiglu` or `geglu`, both `G_b` and `U_b` are clamped to `[-glu_limit, glu_limit]` before activation. Let `gamma` be `glu_alpha` when it is set and not `1.0`; otherwise `gamma = 1`.

For **SwiGLU** (`act_func="swiglu"`):

$$
D[:, bG:(b+1)G] = \gamma \cdot \mathrm{prob} \cdot U_b \cdot \left(G_b \cdot \sigma(G_b)\right)
$$

For **GeGLU** (`act_func="geglu"`):

$$
D[:, bG:(b+1)G] = \gamma \cdot \mathrm{prob} \cdot (U_b + 1) \cdot G_b \cdot \sigma(1.702 \cdot G_b)
$$

For **SReLU** (`act_func="srelu"`), the kernel does not split `N` into gate/up halves:

$$
D = \mathrm{prob} \cdot \mathrm{ReLU}(C)^2
$$

When requested, the RHT output applies a fixed 16-wide orthonormal Hadamard transform to bf16-rounded `D`, either across feature blocks (`rht_rowwise=True`) or across token blocks (`rht_rowwise=False`).

When `D` or `RHT` is NVFP4, the kernel emits packed e2m1 data plus e4m3 scale factors. `norm_const` and `rht_norm_const` are the corresponding global encode scales.

### Diagram

```text
A (valid_m x K x 1), SFA     B (N x K x L), SFB       padded_offsets
          |                         |                         |
          |       dequantize        |                         |
          +-------------+-----------+                         |
                        v                                     v
                    Grouped GEMM over expert ranges ------> group idx
                        |
                        | * alpha[group_idx]
                        v
                    C (valid_m x N x 1)
                        |
                        | GLU over paired 32-col blocks, or SReLU
                        | with per-row prob
                        v
                    D (valid_m x N_out x 1)
                        |
               +--------+---------+
               |                  |
               v                  v
          optional NVFP4     optional Hadamard/RHT
          D + SFD            RHT, optional SFRHT
```

---

## API Usage

### High-level wrapper

```python
from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

result = grouped_gemm_glu_hadamard_quant_wrapper_sm100(
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
    d_dtype=torch.float4_e2m1fn_x2,
    cd_major="n",
    rht_output=True,
    rht_dtype=torch.float4_e2m1fn_x2,
    rht_rowwise=False,
    norm_const=norm_const,
    rht_norm_const=rht_norm_const,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    sf_fp8_dtype_override=None,
    vector_f32=False,
    m_aligned=256,
    act_func="swiglu",
    current_stream=None,
)

c_tensor = result["c_tensor"]
d_tensor = result["d_tensor"]
sfd_tensor = result["sfd_tensor"]
rht_tensor = result["rht_tensor"]
sfrht_tensor = result["sfrht_tensor"]
```

Set `rht_output=False` to skip the Hadamard/RHT output. Set `d_dtype=torch.bfloat16` or `rht_dtype=torch.bfloat16` to request unquantized bf16 outputs for the corresponding path.

### Class API

```python
from cudnn import GroupedGemmGluHadamardQuantSm100

op = GroupedGemmGluHadamardQuantSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d=d,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_prob=prob,
    sample_sfd=sfd,
    sample_rht=rht,
    sample_sfrht=sfrht,
    sample_bias=bias,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    sf_fp8_dtype_override=None,
    vector_f32=False,
    m_aligned=256,
    act_func="swiglu",
    rht_rowwise=False,
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
    sfd_tensor=sfd,
    rht_tensor=rht,
    sfrht_tensor=sfrht,
    bias_tensor=bias,
    norm_const=norm_const,
    rht_norm_const=rht_norm_const,
    current_stream=None,
)
```

### Discrete weight mode

The wrapper also accepts per-expert discrete weight allocations:

```python
result = grouped_gemm_glu_hadamard_quant_wrapper_sm100(
    a_tensor=a,
    b_ptrs=b_ptrs,
    sfa_tensor=sfa,
    sfb_ptrs=sfb_ptrs,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    n=N,
    b_dtype=torch.float4_e2m1fn_x2,
    b_major="k",
)
```

`b_ptrs` and `sfb_ptrs` must be contiguous int64 CUDA tensors containing device pointers for each expert. Each `b_ptrs` entry points to a logical `(N, K)` FP4 expert weight allocation; each `sfb_ptrs` entry points to that expert's scale-factor allocation with logical shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`.

---

## Parameters

### Input/output tensors

- Input tensor **A**: `a_tensor` (wrapper) or `sample_a` / `a_tensor` (class)
  - Shape: `(valid_m, K, 1)`
  - Layout: must be `k`-major
  - Dtype: `float4_e2m1fn_x2`
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b` / `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Layout: must be `k`-major
  - Dtype: must match `A`
- Input tensor **B pointers**: `b_ptrs` (wrapper execute) or `num_experts` / `b_shape` / `b_dtype` (class construction)
  - Shape: `(L,)`
  - Dtype: `int64`
  - Device: CUDA
  - `b_dtype` must match `A`; FP4 discrete mode requires `b_major="k"`
- Input tensor **SFA**: `sfa_tensor` (wrapper) or `sample_sfa` / `sfa_tensor` (class)
  - Shape: `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - Dtype: `{float8_e8m0fnu, float8_e4m3fn}`
  - Set `sf_fp8_dtype_override="e5m3"` to reinterpret `float8_e4m3fn` storage as UE5M3 on Rubin.
- Input tensor **SFB**: `sfb_tensor` (wrapper) or `sample_sfb` / `sfb_tensor` (class)
  - Shape: `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`
  - Dtype: must match `SFA`
- Input tensor **SFB pointers**: `sfb_ptrs` in discrete mode
  - Shape: `(L,)`
  - Dtype: `int64`
  - Device: CUDA
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
- Output tensor **C**
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float16, bfloat16}`
- Output tensor **D**
  - Logical shape: `(valid_m, N_out, 1)`
  - Layout: must be `n`-major
  - Dtype: `{bfloat16, float4_e2m1fn_x2}`
  - NVFP4 `D` requires `SFD`
- Output tensor **SFD** (present only with NVFP4 `D`)
  - Shape: `SF(valid_m, N_out)` = `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(N_out, sf_vec_size), 4), 1)`
  - Layout: swizzled scale-factor layout matching `SFA`
  - Dtype: `float8_e4m3fn`
- Output tensor **RHT** (optional)
  - Logical shape: `(valid_m, N_out, 1)`
  - Layout: must be `n`-major
  - Dtype: `{bfloat16, float4_e2m1fn_x2}`
  - NVFP4 `RHT` requires `SFRHT`
- Output tensor **SFRHT** (present only with NVFP4 `RHT`)
  - Shape: `SF(valid_m, N_out)` when `rht_rowwise=True`; `SF(N_out, valid_m)` when `rht_rowwise=False`
  - Layout: swizzled scale-factor layout
  - Dtype: `float8_e4m3fn`

### Configuration

- `act_func`: `"swiglu"`, `"geglu"`, or `"srelu"`
- `cd_major`: must be `"n"`
- `mma_tiler_mn`: must be `(256, 256)`
- `cluster_shape_mn`: cluster dimensions; for this fixed 2-CTA tiler, `cluster_shape_mn[0]` must be `2` and `cluster_shape_mn[1]` must be a positive power of two no larger than `4`
- `sf_vec_size`: must be `16`
- `sf_fp8_dtype_override`: `None` uses the scale format implied by `SFA`/`SFB` dtype. `"e5m3"` reinterprets `torch.float8_e4m3fn` SFA/SFB storage as UE5M3 input scale factors; this is Rubin-only and does not convert tensor contents.
- `m_aligned`: must be `256`
- `rht_rowwise`: selects feature-blocked Hadamard/RHT (`True`) or token-blocked Hadamard/RHT (`False`); for quantized RHT, the scale grid follows the selected axis
- `glu_alpha`: optional final output scale for `swiglu`/`geglu`
- `glu_limit`: optional clamp limit applied to both gate and up blocks for `swiglu`/`geglu`
- `norm_const`: global encode scale for NVFP4 `D`
- `rht_norm_const`: global encode scale for NVFP4 `RHT`

### Constraints

- Requires SM100 or newer.
- `N` must be divisible by `64`.
- `K` must satisfy the FP4 K-major 16-byte alignment requirement, so `K` must be divisible by `32`.
- `valid_m` / `a_tensor.shape[0]` must be divisible by `256`; `padded_offsets` must contain cumulative 256-aligned expert ends and end at `valid_m`.
- `N_out` must be divisible by `32`.
- NVFP4 quantization requires `N_out` divisible by `128`.
- NVFP4 quantization is not supported with `act_func="srelu"`.
- `sf_fp8_dtype_override="e5m3"` requires Rubin (SM107) and `SFA`/`SFB` tensors stored as `torch.float8_e4m3fn`.
- `SFRHT` in colwise quantized mode uses the transposed scale domain `SF(N_out, valid_m)`.
- `expert_cnt` must be `<= 1024`.
- Dense and discrete weight modes are mutually exclusive.
