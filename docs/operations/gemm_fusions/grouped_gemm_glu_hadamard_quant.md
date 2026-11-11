# Grouped GEMM + GLU + Hadamard + Quant (SM100)

**This is an experimental API and subject to change.**

## JAX support

**Canonical FP4 (dense weights):** the `jax.jit`-compatible entry point `cudnn.grouped_gemm_glu_hadamard_quant_jax_forward` takes packed FP4 A `(m, k/2)` and B `(experts, n, k/2)` as `uint8` with two values per byte, already MMA-packed `sfa_tensor`/`sfb_tensor` bytes (physical or flat; E4M3 with `sf_vec_size=16` for NVFP4, or E8M0), int32 `padded_offsets`, fp32 alpha and prob `(m,)` fp32/bf16. Output modes are dtype driven as in Torch, with `cutlass.Float4E2M1FN` selecting NVFP4 (Torch's packed dtype has no JAX equivalent): NVFP4 D and rowwise RHT come back as packed `uint8` `(m, n_out/2)` with physical 6-D E4M3 scale buffers, and colwise RHT as flat packed data `(m*n_out/2,)` and scales `(m*n_out/sf_vec_size,)`. `norm_const`, `rht_norm_const`, `glu_alpha` and `glu_limit` are compile-time constants of the traced call: pass Python numbers (or mark them static), since a traced JAX value raises `ConcretizationTypeError`. A scale computed inside the jitted program is not supported. Rows at or past `padded_offsets[-1]` are unspecified. `grouped_gemm_glu_hadamard_quant_wrapper_sm100` dispatches JAX inputs with a dense `b_tensor` to it. SM100+, with the Rubin kernel on SM107, and the default 256x256 tile; needs `nvidia-cutlass-dsl >= 4.8`. Bias, dynamic scheduling, discrete (`b_ptrs`) weights and the class API raise `ValueError` for JAX.

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

### Gate/up branch storage

Both the wrapper and class accept an append-only `glu_layout` option. The
existing `"interleaved_32"` default keeps its 256x256 two-CTA geometry.
`"gate_then_up"` accepts conventional `[gate | up]` weights, matching bias
and saved C, without a persistent weight repack. It uses the existing kernel
with a nested TMA view and a local scale-quarter permutation before MMA.
D, RHT and their scale tensors retain their natural post-activation feature order.

The canonical branch option currently requires Torch on SM100/SM103,
`act_func="swiglu"`, NVFP4 E2M1 operands with E4M3 scales, `sf_vec_size=16`,
K-major weights, BF16 saved C, and `N/2` divisible by 128. Set
`mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1)` for this option.
Both dense and discrete expert weights are supported. The existing output
modes remain available: BF16 or NVFP4 D, BF16/NVFP4 rowwise RHT, and NVFP4
colwise RHT. Expert token offsets still require 256-row alignment.
JAX, Rubin and UE5M3 input-scale overrides do not gain this branch option.

Capability metadata is explicit:
`supports_glu_layouts=("interleaved_32", "gate_then_up")`,
`supports_gate_then_up_precisions=("nvfp4",)`, and
`supports_gate_then_up_mma_tilers=((128, 256),)`.
These branch-order capabilities are separate from tensor rank/stride support.

### Natural tensor ranks (Torch)

The wrapper and class advertise `supports_canonical_layouts=True` and accept
contiguous A `(M, K/2)`, dense B `(L, N, K/2)`, and prob `(M,)` for packed NVFP4.
The shapes use physical bytes for FP4; logical K is twice the last extent.
Discrete weights keep the existing pointer-array contract.

SFA and dense SFB can be contiguous buffers containing the **already MMA-packed
scale bytes**, at any rank. Their lengths are respectively
`512 * ceil_div(M, 128) * ceil_div(K, 4 * sf_vec_size)` and
`512 * ceil_div(N, 128) * ceil_div(K, 4 * sf_vec_size) * L`.
Flat storage does not mean logical row-major scale order: the caller must provide
the same packed bytes as the legacy six-dimensional view. No scale repacking,
weight permutation, or arithmetic change is performed by this adapter.

For rank-two A, the wrapper returns C `(M, N)`, D and rowwise RHT `(M, N_out)`
(or `(M, N_out/2)` for packed NVFP4), and flat contiguous rowwise scale buffers.
Colwise RHT and its scales keep their existing flat per-expert representation.
The class similarly accepts caller-owned natural-rank outputs; contiguous
rowwise scale output buffers may have any rank. Trace-time views normalize
these tensors for the existing kernel, without per-execution tensor reshapes.

Natural tensor ranks work with both `glu_layout="interleaved_32"` and
`glu_layout="gate_then_up"`. Colwise RHT scales use the same standard
GEMM-blocked output order for either branch layout. Legacy input ranks and
their wrapper output shapes remain supported.

### Legacy kernel-facing shapes

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
  - `RHT rowwise`: optional Hadamard-transform output across feature blocks, logical shape `(valid_m, N_out, 1)`
  - `SFRHT rowwise`: scale factors for NVFP4 rowwise `RHT`, shape `SF(valid_m, N_out)`, present only when rowwise `RHT` is NVFP4
  - `RHT colwise`: optional NVFP4 Hadamard-transform output across token blocks, flattened in per-expert transposed order
  - `SFRHT colwise`: scale factors for NVFP4 colwise `RHT`, flattened in per-expert ragged order

For packed NVFP4 output tensors (`torch.float4_e2m1fn_x2`), the physical tensor stores two logical values per byte along the innermost dimension. The wrapper therefore allocates packed `D` and rowwise `RHT` tensors with physical second dimension `N_out / 2`. Raw `torch.uint8` tensors are not accepted as a packed FP4 container by this fusion.

Colwise `RHT` is NVFP4-only. Its data tensor is a flat physical tensor of length `valid_m * N_out / 2`; after FP4 unpacking, each expert segment is logical `(N_out, expert_m)` with adjacent token values packed together. Expert segments are concatenated in expert order. `SFRHT colwise` is flat length `valid_m * N_out / sf_vec_size`; each expert segment contains standard GEMM-blocked scale bytes, starting at `N_out * expert_m_prefix / sf_vec_size`.

For logical column scales `S[r, q]`, where `r` indexes features and `q`
indexes 16-token blocks, let `C = expert_m / 64`. The byte offset within
one expert's scale segment is:

```text
512 * ((r / 128) * C + q / 4) + 16 * (r % 32) + 4 * ((r / 32) % 4) + q % 4
```

Divisions are integer divisions. Physical row-major axes are
`(N_out/128, expert_m/64, 32, 4, 4)`. Torch and JAX use this contract on both
Blackwell and Rubin; the kernel writes these bytes directly without a repack.

**Output-format migration:** earlier versions flattened the logical CuTe axes
`(32, 4, N_out/128, 4, expert_m/64)` in row-major order. That format was
decodable by its matching reference but differed from GEMM consumers' scale
packing. Callers decoding that old order must now consume standard GEMM-blocked
bytes. Public API signatures, tensor lengths, logical scale values, RHT data
and rowwise outputs are unchanged.

`L` is the expert count. `valid_m` is the `M` extent of `a_tensor`; by contract it must match the final cumulative padded offset in `padded_offsets`.

### Equations

For rows belonging to expert `g`:

$$
C[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB) + \mathrm{bias}_g[n]
$$

The `bias` term is omitted when `bias_tensor=None`. `C` stores this unclamped GEMM-plus-bias value.

With the default `glu_layout="interleaved_32"`, split the `N` dimension into
consecutive 32-column gate/up blocks. For `"gate_then_up"`, use the first
`N/2` columns as gate and the remaining `N/2` columns as up; the activation
equations below are unchanged. The default block indexing is:

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

When requested, the RHT output applies a fixed 16-wide orthonormal Hadamard transform to bf16-rounded `D`. Rowwise RHT transforms feature blocks. Colwise RHT transforms token blocks and writes NVFP4 data in per-expert transposed order.

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
    rht_colwise_dtype=torch.float4_e2m1fn_x2,
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
rht_colwise_tensor = result["rht_colwise_tensor"]
sfrht_colwise_tensor = result["sfrht_colwise_tensor"]
```

Leave both `rht_rowwise_dtype` and `rht_colwise_dtype` unset to skip the Hadamard/RHT output. Set `rht_rowwise_dtype=torch.bfloat16` for unquantized rowwise RHT, `rht_rowwise_dtype=torch.float4_e2m1fn_x2` for quantized rowwise RHT, or `rht_colwise_dtype=torch.float4_e2m1fn_x2` for wgrad-compatible colwise RHT.

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
    sample_rht_colwise=rht_colwise,
    sample_sfrht_colwise=sfrht_colwise,
    sample_bias=bias,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    sf_fp8_dtype_override=None,
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
    sfd_tensor=sfd,
    rht_colwise_tensor=rht_colwise,
    sfrht_colwise_tensor=sfrht_colwise,
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
- Output tensor **RHT rowwise** (optional)
  - Logical shape: `(valid_m, N_out, 1)`
  - Layout: must be `n`-major
  - Dtype: `{bfloat16, float4_e2m1fn_x2}`
  - NVFP4 rowwise `RHT` requires `SFRHT rowwise`
- Output tensor **SFRHT rowwise** (present only with NVFP4 rowwise `RHT`)
  - Shape: `SF(valid_m, N_out)`
  - Layout: swizzled scale-factor layout
  - Dtype: `float8_e4m3fn`, independent of `SFA`/`SFB` dtype
- Output tensor **RHT colwise** (optional)
  - Logical unpacked shape: per-expert `(N_out, expert_m)` segments concatenated in expert order
  - Physical shape: 1-D packed FP4 tensor with `valid_m * N_out / 2` elements
  - Dtype: `float4_e2m1fn_x2`
  - NVFP4 colwise `RHT` always requires `SFRHT colwise`
- Output tensor **SFRHT colwise** (present with colwise `RHT`)
  - Shape: `(valid_m * N_out / sf_vec_size,)`
  - Layout: standard GEMM 128x4 blocked scale bytes per expert, concatenated in expert order; see the physical offset formula above
  - Dtype: `float8_e4m3fn`, independent of `SFA`/`SFB` dtype

### Configuration

- `act_func`: `"swiglu"`, `"geglu"`, or `"srelu"`
- `cd_major`: must be `"n"`
- `mma_tiler_mn`: must be `(256, 256)`
- `cluster_shape_mn`: cluster dimensions; for this fixed 2-CTA tiler, `cluster_shape_mn[0]` must be `2` and `cluster_shape_mn[1]` must be a positive power of two no larger than `4`
- `sf_vec_size`: must be `16`
- `sf_fp8_dtype_override`: `None` uses the scale format implied by `SFA`/`SFB` dtype. `"e5m3"` reinterprets `torch.float8_e4m3fn` SFA/SFB storage as UE5M3 input scale factors; this is Rubin-only and does not convert tensor contents.
- `m_aligned`: must be `256`
- `rht_rowwise_dtype`: unset to skip rowwise RHT, `torch.bfloat16` for unquantized rowwise RHT, or `torch.float4_e2m1fn_x2` for NVFP4 rowwise RHT
- `rht_colwise_dtype`: unset to skip colwise RHT, or `torch.float4_e2m1fn_x2` for NVFP4 colwise RHT in per-expert transposed order
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
- At most one of rowwise and colwise RHT may be requested.
- Colwise RHT is NVFP4-only and uses flat per-expert ragged scale storage.
- `expert_cnt` must be `<= 1024`.
- Dense and discrete weight modes are mutually exclusive.
