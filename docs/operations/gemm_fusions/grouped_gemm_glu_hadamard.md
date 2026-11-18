# Grouped GEMM + GLU + Hadamard (SM100)

**This is an experimental API and subject to change.**

## JAX support

**Canonical FP4 (dense weights):** the `jax.jit`-compatible entry point `cudnn.grouped_gemm_glu_hadamard_jax_forward` takes packed FP4 A `(m, k/2)` and B `(experts, n, k/2)` as `uint8` with two values per byte, already MMA-packed `sfa_tensor`/`sfb_tensor` bytes (physical or flat; E4M3 with `sf_vec_size=16` for NVFP4, or E8M0), int32 `padded_offsets`, fp32 alpha and prob `(m,)` fp32/bf16. It returns a `TupleDict` with the eager wrapper's keys: the pre-activation C `(m, n)`, D `(m, n/2)` (or `(m, n)` for `act_func="srelu"`) in a BF16/FP16 `d_dtype`, and the per-expert amax and post-Hadamard amax `(experts, 1)`. Rows at or past `padded_offsets[-1]` are unspecified. `grouped_gemm_glu_hadamard_wrapper_sm100` dispatches JAX inputs with a dense `b_tensor` to it. SM100+, as in Torch, with the default 256x256 tile. Bias, dynamic scheduling, discrete (`b_ptrs`) weights and the class API raise `ValueError` for JAX.

## Overview

**Grouped GEMM + GLU + Hadamard fusion**: A contiguous grouped block-scaled GEMM fused with a GLU epilogue, a 16-wide Hadamard transform for post-RHT amax computation, and per-expert `amax` reductions on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This frontend integration is currently wired for the fp4 input path.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **GLU epilogue** using per-row `prob` with SwiGLU, GeGLU, SiTU-GLU, or SReLU
3. **Hadamard transform** over 16-token groups of the post-GLU output
4. **Per-expert amax reductions** before and after the Hadamard transform

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
  - `D`: activation output before the Hadamard transform, shape `(valid_m, N / 2, 1)` for GLU activations and `(valid_m, N, 1)` for SReLU
  - `Amax`: per-expert amax of `D`, shape `(L, 1)` when `D` is fp16/bf16
  - `PostRhtAmax`: per-expert amax after the normalized Hadamard transform, shape `(L, 1)` when `D` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Canonical layouts

The wrapper and API class also accept contiguous `A` `(valid_m, K)`, dense `B`
`(L, N, K)`, and `prob` `(valid_m,)` with `torch.float4_e2m1fn_x2` operands. Scale
buffers may be contiguous tensors of any rank, provided their bytes are already
packed in the required MMA-tiled physical order. With 2-D `A`, the wrapper returns
`C` `(valid_m, N)` and a 2-D `D`. Legacy layouts are unchanged. The wrapper
advertises `supports_canonical_layouts`.

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

For **SiTU-GLU** (`act_func="situglu"`):

$$
X[:, bG:(b+1)G] = \mathrm{prob}\,
    \left[\beta_1\tanh(G_b/\beta_1)\sigma(G_b)\right]
    \left[\beta_2\tanh(U_b/\beta_2)\right].
$$

Here `beta_1 = situ_beta1` and `beta_2 = situ_beta2`, with defaults `4.0` and `25.0`. `situ_beta1` specializes the compiled kernel and is part of its cache key. `situ_beta2` is a runtime FP32 scalar, so changing it does not create a new compiled-kernel cache entry.

For **SReLU** (`act_func="srelu"`), `N` is not split into gate/up halves and `X` has `N` columns:

$$
X[m, n] = \mathrm{prob}[m] \cdot \mathrm{relu}(\hat C[m, n])^2, \qquad \hat C = \mathrm{round}_{c\_dtype}(C).
$$

$\hat C$ is `C` as it is stored in the returned `c_tensor`. See [SReLU from the saved C](#srelu-from-the-saved-c).

#### Tanh soft clamp (`tanh_clamp_scale`)

With `act_func="srelu"`, passing `tanh_clamp_scale=s` (a positive finite float; `None`, the default, keeps the equation above) soft-clamps the ReLU output before squaring, bounding `X` by $s^2 \cdot |\mathrm{prob}|$:

$$
X[m, n] = \mathrm{prob}[m] \cdot \left( s \cdot \tanh\!\left( \frac{\mathrm{relu}(\hat C[m, n])}{s} \right) \right)^2
$$

This is the same expression, evaluated in the same order, as the plain [grouped_gemm_srelu](/operations/Matmul/gemm_fusions/grouped_gemm_srelu) kernel; its notes on the approximate `tanh` apply here unchanged. `s` is a trace-time constant: it is part of the wrapper cache key and distinct values compile distinct kernels. The matching backward kernel, [grouped_gemm_dsrelu](/operations/Matmul/gemm_fusions/grouped_gemm_dsrelu), must be built with the same `s`.

The returned `D` is `X`. For NVFP4 quantization, the kernel also applies the normalized fixed Hadamard matrix `H` of size `16 x 16` over 16-token groups within each expert and reduces its absolute maximum:

$$
\mathrm{PostRhtAmax}_g = \max \left|\mathrm{round}_{bf16}\!\left(\mathrm{RHT}_{16}(D_g)\right)\right|.
$$

When `D` is fp16/bf16, the kernel emits both `Amax`, computed from the untransformed `D`, and `PostRhtAmax`. The transformed values are not materialized as another output tensor; the post-RHT amax is intended for the downstream NVFP4 quantization step.

Both reductions describe the values that step consumes, not the FP32 values in the epilogue registers:

- `Amax` is the amax of `D` as stored in `d_dtype`. Round-to-nearest can lift the largest element above its FP32 value; an amax taken before the rounding would then be too small and the element would saturate when `D` is quantized with it.
- `PostRhtAmax` is the amax of the transformed values after rounding to BF16, because the consumer's transform produces BF16. The transform reads `D` as stored.

#### Randomized Hadamard sign mask (`rht_sign_mask`)

A consumer that quantizes with a *randomized* Hadamard transform negates a fixed subset of each 16-token block before transforming it. `rht_sign_mask` is that subset as a 16-bit integer: bit `i` negates input element `i` of every block. `PostRhtAmax` is then the amax for that transform:

$$
\mathrm{RHT}_{16}(x) = \tfrac{1}{4} H \, \mathrm{diag}(\sigma) \, x, \qquad \sigma_i = \begin{cases} -1 & \text{bit } i \text{ of rht\_sign\_mask set} \\ +1 & \text{otherwise.} \end{cases}
$$

The default `0` is the plain transform. The mask changes only `PostRhtAmax`; `C`, `D` and `Amax` keep the same bytes. It must equal the mask the consumer uses: the amax of the plain transform can differ from that of the signed one in either direction, which either clips or wastes range when the consumer quantizes with it. It is a trace-time constant and part of the wrapper cache key. With the class API it cannot be combined with `sample_hadamard`; fold the signs into that tensor instead.

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
                 D (valid_m×N/2×1)
                     |\
                     | +--> Amax (L×1)
                     |
                     +----> normalized RHT(16) --> PostRhtAmax (L×1)
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
    act_func="situglu",
    situ_beta1=4.0,
    situ_beta2=25.0,
    current_stream=None,
)

c_tensor, d_tensor, amax_tensor, post_rht_amax_tensor = result
```

The wrapper constructs the fixed Hadamard matrix internally.

`tanh_clamp_scale` and `rht_sign_mask` follow `current_stream` in the wrapper signature. Pass them by keyword:

```python
result = grouped_gemm_glu_hadamard_wrapper_sm100(
    ...,
    act_func="srelu",
    tanh_clamp_scale=16.0,
    rht_sign_mask=0xA5C3,
)
```

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
    sample_post_rht_amax=post_rht_amax,
    sample_bias=bias,
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    vector_f32=False,
    m_aligned=256,
    act_func="situglu",
    situ_beta1=4.0,
    situ_beta2=25.0,
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
    post_rht_amax_tensor=post_rht_amax,
    bias_tensor=bias,
    situ_beta1=4.0,
    situ_beta2=25.0,
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
  - Shape: `(valid_m, N / 2, 1)` for GLU activations; `(valid_m, N, 1)` for SReLU
  - Layout: must be `n`-major
  - Dtype: `{float16, bfloat16}`
- Output tensor **Amax**: `result["amax_tensor"]` (wrapper) or `sample_amax` / `amax_tensor` (class)
  - Shape: `(L, 1)`
  - Dtype: `float32`
- Output tensor **PostRhtAmax**: `result["post_rht_amax_tensor"]` (wrapper) or `sample_post_rht_amax` / `post_rht_amax_tensor` (class)
  - Shape: `(L, 1)`
  - Dtype: `float32`
  - Semantics: per-expert amax after normalized RHT(16), for downstream NVFP4 quantization

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
  - Allowed values: `{"swiglu", "geglu", "situglu", "srelu"}`
- `situ_beta1: float`
  - Positive finite gate tanh scale for SiTU-GLU; default `4.0`
  - Compile-time specialized and included in the wrapper cache key
- `situ_beta2: float`
  - Positive finite up-branch tanh scale for SiTU-GLU; default `25.0`
  - Runtime FP32 scalar; changing it reuses the compiled beta1 specialization
- `use_tmem_post_rht_amax: bool`
  - Default `False`: `PostRhtAmax` is reduced from `D` in shared memory with a fast Walsh-Hadamard transform
  - `True`: the transform runs as an MMA against the Hadamard tensor and is reduced from TMEM
  - Both paths return the same `PostRhtAmax` bytes in the tested configurations; the choice is a performance knob
- `tanh_clamp_scale: float | None`
  - Soft-clamp scale `s` for `act_func="srelu"`; default `None` (unclamped). See [Tanh soft clamp](#tanh-soft-clamp-tanh_clamp_scale)
  - Must be finite and positive; rejected for every other `act_func`
  - Compile-time specialized and included in the wrapper cache key
- `rht_sign_mask: int`
  - 16-bit sign mask of the consumer's randomized Hadamard transform; default `0`. See [Randomized Hadamard sign mask](#randomized-hadamard-sign-mask-rht_sign_mask)
  - Compile-time specialized and included in the wrapper cache key
  - Class API: cannot be combined with `sample_hadamard`
- CUDA stream (`current_stream` in class API and wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `c_tensor`
- `d_tensor`
- `amax_tensor`
- `post_rht_amax_tensor`

Tuple unpacking order is: `(c_tensor, d_tensor, amax_tensor, post_rht_amax_tensor)`.

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
- `tanh_clamp_scale` requires `act_func="srelu"`.
- `tanh_clamp_scale`, `rht_sign_mask` and `use_tmem_post_rht_amax` are not supported on the JAX block-scaled path.

## Numerical behavior

### SReLU from the saved C

The SReLU epilogue rounds the FP32 accumulator to `c_dtype` and evaluates the activation, in FP32, from that rounded value. `D` is therefore exactly what a consumer computes from the returned `c_tensor`:

```python
c = result["c_tensor"].float()
d = (torch.relu(c) ** 2 * prob).to(d_dtype)   # same bytes as result["d_tensor"]
```

This is what lets the dSReLU backward and activation recomputation, which only have the saved `C`, regenerate `D` bit for bit. With `tanh_clamp_scale` set, the forward and dSReLU kernels go through one shared clamp routine for the same purpose; a host reference using a different `tanh` agrees only to a tolerance.

**Change from earlier releases.** The SReLU epilogue used to evaluate the activation from the unrounded FP32 accumulator. `D` now differs from those releases wherever rounding `C` to `c_dtype` changes the value: for `c_dtype=bfloat16`, up to one part in $2^{8}$ of `C`, so about one part in $2^{7}$ of `D`. The GLU activations (`swiglu`, `geglu`, `situglu`) are unchanged. The same change applies to `act_func="srelu"` in [grouped_gemm_glu_hadamard_quant](/operations/Matmul/gemm_fusions/grouped_gemm_glu_hadamard_quant).

`Amax` and `PostRhtAmax` also changed, for every activation: they are now reduced from the stored `D` and from the BF16-rounded transform (see [Equations](#equations)). Earlier releases reduced the FP32 values, so either amax can move by up to one rounding step of BF16 (or of `d_dtype` for `Amax`).

### Determinism

Tested scope, `act_func="srelu"` with FP4 inputs, BF16 `C`/`D`, `tanh_clamp_scale` and `rht_sign_mask` set, both `PostRhtAmax` paths, unequal expert sizes including an empty expert:

- **Repeatability:** `C`, `D`, `Amax` and `PostRhtAmax` keep identical bytes across repeated calls with fixed inputs and shapes, with concurrent work on another stream. The per-expert amax reductions are maxima, which do not depend on reduction order.
- **Batch invariance:** not established. `Amax` and `PostRhtAmax` are whole-expert reductions, and `PostRhtAmax` depends on how tokens fall into 16-token blocks, so both change when an expert's rows change or move.
- Results are not expected to match across GPU architectures or software versions, and the approximate `tanh` of the clamped path is not bitwise equal to a host `tanh`.

See [Determinism support](/operations/determinism) for the status table.
