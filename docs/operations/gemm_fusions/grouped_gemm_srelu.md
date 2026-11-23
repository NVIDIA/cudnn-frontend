# Grouped GEMM + sReLU (SM100)

**This is an experimental API and subject to change.**

## JAX support

**Canonical block-scaled (dense weights):** the `jax.jit`-compatible entry point `cudnn.grouped_gemm_srelu_jax_forward` takes A `(m, k)`, B `(experts, n, k)`, already MMA-packed `sfa_tensor`/`sfb_tensor` bytes (physical or flat), int32 `padded_offsets`, fp32 alpha and prob `(m,)` fp32/bf16. A/B are MXFP8 with E8M0 scales (`sf_vec_size=32`; uint8 accepted), or packed FP4. Packed FP4 operands travel as `uint8` with two values per byte (JAX has no packed FP4 dtype), so K is twice the array extent; their scale buffers are E4M3 with `sf_vec_size=16` for NVFP4, or E8M0. It returns a `TupleDict` with the eager wrapper's keys: the pre-activation C `(m, n)` and D `(m, n)`. A BF16/FP16 `d_dtype` also returns the per-expert amax `(experts, 1)`; an FP8 `d_dtype`, which needs MXFP8 inputs and `norm_const_tensor` `(1,)`, also returns D_col and the physical 6-D SFD buffers. Outputs are not initialized, so rows at or past `padded_offsets[-1]` are unspecified. `grouped_gemm_srelu_wrapper_sm100` dispatches JAX inputs with a dense `b_tensor` to it; pass `sf_vec_size`. SM100+, as in Torch. Bias, dynamic scheduling, `b_major="n"`, discrete (`b_ptrs`) weights and the `GroupedGemmSreluSm100` class raise `ValueError` for JAX.

JAX inputs go through the same option checks as the block-scaled Torch path: `act_input` must be `"acc"`, and `tanh_mode="accurate"` with `tanh_clamp_scale` set raises `ValueError`. The [BF16 backend](#bf16-backend) takes torch tensors only: BF16 JAX arrays raise `ValueError` in the wrapper, before anything is allocated, and in the `GroupedGemmSreluSm100` constructor.

## Overview

**Grouped GEMM + sReLU fusion**: A grouped block-scaled GEMM fused with a probability-gated squared-ReLU epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. The API supports dense contiguous weights and discrete per-expert weight allocations. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **sReLU epilogue** using per-row `prob`
3. **Optional output quantization** through `SFD_row` / `SFD_col` or `Amax`

BF16 `A`/`B` select a separate BF16 kernel behind the same wrapper and class, on SM100 and
SM103 and for torch tensors only: no scale factors, no quantization, and `C` is the BF16
pre-activation that the backward reloads. See [BF16 backend](#bf16-backend).

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: dense weight tensor across all groups, shape `(N, K, L)`, or discrete per-expert tensors addressed by `b_ptrs`
  - `SFA`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - `SFB`: dense scale-factor tensor, shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`, or discrete per-expert tensors addressed by `sfb_ptrs`
  - `padded_offsets`: cumulative padded group ends, shape `(L,)`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`

- **Outputs**
  - `C`: intermediate GEMM result, shape `(valid_m, N, 1)`
  - `D`: row output after sReLU, shape `(valid_m, N, 1)`
  - `D_col`: column output after sReLU, shape `(valid_m, N, 1)`
  - `SFD_row`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), 1)` when `D` is FP8
  - `SFD_col`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(valid_m, sf_vec_size), 4), 1)` when `D` is FP8
  - `Amax`: shape `(L, 1)` when `D` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Equations

For rows belonging to expert `g`:

$$
C[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB)
$$

$$
D[m, n] = \mathrm{prob}[m, 0, 0] \cdot \mathrm{relu}(C[m, n])^2
$$

#### Tanh soft clamp (`tanh_clamp_scale`)

Passing `tanh_clamp_scale=s` (a positive float; `None`, the default, keeps the equation above)
soft-clamps the ReLU output before squaring, bounding `D` by $s^2 \cdot \mathrm{prob}$:

$$
D[m, n] = \mathrm{prob}[m, 0, 0] \cdot \left( s \cdot \tanh\!\left( \frac{\mathrm{relu}(C[m, n])}{s} \right) \right)^2
$$

`s` is baked into the compiled kernel, so distinct values compile distinct kernels -- it is meant
to be constant for a whole training job, not a per-call argument.

The block-scaled epilogue is evaluated in FP32 with the approximate `tanh`, MUFU
`tanh.approx.f32`. Measured on sm_103 against a
correctly-rounded reference, over a sweep of $[0, 64]$ plus saturated inputs, its maximum
absolute error is $7.7\times10^{-6}$ (the exact variant measures $1.0\times10^{-7}$ on the same
sweep). Since $D = c^2$ with $c \le s$, that propagates to at most $2s^2 \cdot 7.7\times10^{-6}$
absolutely -- about $1.6\times10^{-2}$ at $s = 32$ -- and to roughly $1.5\times10^{-5}$
relative in the saturated tail. $t$ is capped at 1 before use, so $c \le s$ and the output bound
$s^2 \cdot \mathrm{prob}$ hold structurally rather than by relying on the range of the
approximate instruction.

That instruction is the block-scaled kernel's only `tanh`, so with `tanh_clamp_scale` set the
block-scaled backend requires `tanh_mode="approx"`, the default. The BF16 backend uses the same
instruction by default and also offers an accurate FP32 `tanh`; see [BF16 backend](#bf16-backend).

When `D` is FP8, even that small error is enough to carry a value across an output-format
rounding boundary, so individual elements can land one ULP away from a reference computed with
an exact `tanh` (12.5% relative for `e4m3`). Measured incidence on this kernel's test matrix is
a few elements in $5\times10^{5}$. This does not affect the unclamped path, where the kernel and
a reference both evaluate $\mathrm{relu}(x)^2$ and therefore quantize identically.

The block-scaled kernel maps a NaN pre-activation to +0, clamped or not: the clamped path
saturates `tanh` with `mul.rn.sat.f32`, and the plain path's relu is a select. The BF16 backend
propagates NaN instead.

The matching backward kernel takes the same `tanh_clamp_scale` -- see
[grouped_gemm_dsrelu](/operations/Matmul/gemm_fusions/grouped_gemm_dsrelu). Both must be built with the same `s` and the same
`tanh_mode`, or the saved pre-activation is differentiated against the wrong nonlinearity.

`D_col` stores the same logical output in the column-quantized path used by the grouped kernel family. When `D` is FP8, the kernel also emits `SFD_row` and `SFD_col`. When `D` is fp16/bf16, the kernel can emit per-expert `Amax`.

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
                     | relu(C)^2 * prob
                     v
             D / D_col (valid_m×N×1)
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
from cudnn import grouped_gemm_srelu_wrapper_sm100

result = grouped_gemm_srelu_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
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
    current_stream=None,
)

c, d, d_col, amax, sfd_row, sfd_col = result
```

### Discrete-weight wrapper

```python
result = grouped_gemm_srelu_wrapper_sm100(
    a_tensor=a,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    b_ptrs=b_ptrs,          # int64 device tensor of per-expert B pointers
    sfb_ptrs=sfb_ptrs,      # int64 device tensor of per-expert SFB pointers
    n=N,
    b_dtype=torch.float4_e2m1fn_x2,
    b_major="k",
    prob_tensor=prob,
    c_dtype=torch.bfloat16,
    d_dtype=torch.bfloat16,
)
```

### Class API

The constructor records the configuration; backend selection and validation happen in
`check_support()` (also called by `compile()` and the workspace-size query). Existing positional
parameters keep their order. The facade exposes the documented methods; backend tensor-descriptor
attributes are not forwarded. Call `check_support()` explicitly when validation must happen before
compilation or execution.

```python
from cudnn import GroupedGemmSreluSm100

op = GroupedGemmSreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
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
    current_stream=None,
)
```

The constructor only records its arguments. `check_support()` selects the backend from the
operand dtypes, builds it and validates the configuration; `compile()` and
`scratch_workspace_bytes()` run it first when it has not run yet. Configuration errors therefore
surface at `check_support()`, not at construction: the constructor itself rejects only JAX arrays
and unsupported tensor frameworks. The block-scaled backend owns its scratch, so its
`scratch_workspace_bytes()` returns 0 and `execute()` raises `ValueError` when given a
`workspace`. The BF16 backend's `execute()` takes a caller-owned workspace; see
[Class-API workspace](#class-api-workspace).

---

## Parameters

### Input/Output tensors

These are the block-scaled backend's tensors; the BF16 backend's are listed in
[BF16 arguments](#bf16-arguments).

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
  - Required
- Output tensor **C**: `result["c_tensor"]` (wrapper) or `sample_c` / `c_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`
- Output tensor **D**: `result["d_tensor"]` (wrapper) or `sample_d` / `d_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype:
    - FP4 input: `{float16, bfloat16, float32}`
    - FP8 input: `{float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2}`
- Output tensor **D_col**: `result["d_col_tensor"]` (wrapper) or `sample_d_col` / `d_col_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must match `D`
  - Dtype: must match `D`
- Output tensors **SFD_row** / **SFD_col**
  - Dtypes: must match `SFA`
  - Required when FP8 output scale factors are generated
- Output tensor **Amax**
  - Shape: `(L, 1)`
  - Dtype: `float32`
- Input tensor **Norm Const**
  - Shape: `(1,)`
  - Dtype: `float32`
  - Required when FP8 output scale factors are generated

### Common parameters

- `acc_dtype: torch.dtype`
  - Only `torch.float32` is supported
- `mma_tiler_mn: Tuple[int, int]`
  - `TILE_M` depends on the 1-CTA / 2-CTA mode
  - Block-scaled: `TILE_N ∈ {128, 256}`; BF16: see [BF16 arguments](#bf16-arguments)
- `cluster_shape_mn: Tuple[int, int] | None`
  - Default: `(2, 1)` when `TILE_M == 256`, else `(1, 1)`
- `sf_vec_size: int`
  - Allowed values: `{16, 32}`
- `vector_f32: bool`
  - Enables vectorized f32 operations for supported configurations
- `m_aligned: int`
  - Must equal the kernel fixed pad size `256`
- `discrete_col_sfd: bool`
  - Enables the discrete column-scale-factor path used by grouped FP8
- `tanh_clamp_scale: float | None`
  - `None` (default): plain `relu(x) ** 2`; `s > 0`: `(s * tanh(relu(x) / s)) ** 2`, with `s`
    baked into the compiled kernel
- `tanh_mode: str`
  - `"approx"` (default) or `"accurate"`; any other value, `None` included, raises `ValueError`
  - With `tanh_clamp_scale` set, `"approx"` is MUFU `tanh.approx.f32` (about $2^{-11}$ relative
    error, capped at 1) on both backends, and `"accurate"` is a Cephes-style FP32 `tanh`
    (relative error below $3\times10^{-7}$) on the BF16 backend only; the block-scaled backend
    raises `ValueError` for it
  - Without `tanh_clamp_scale` no `tanh` is evaluated, and both backends ignore the value
  - The backward must use the same value; see
    [grouped_gemm_dsrelu](/operations/Matmul/gemm_fusions/grouped_gemm_dsrelu)
  - NaN pre-activations propagate to `D` on the BF16 backend in every mode; the block-scaled
    kernel maps them to +0, clamped or not
- `act_input: str`
  - BF16 only: `"acc"` (default) takes the activation from the FP32 accumulator, `"saved"` from
    the stored BF16 `C`; see [BF16 backend](#bf16-backend)
  - The block-scaled backend accepts only `"acc"`
- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `c_tensor`
- `d_tensor`
- `d_col_tensor`
- `amax_tensor`
- `sfd_row_tensor`
- `sfd_col_tensor`

Tuple unpacking order is: `(c_tensor, d_tensor, d_col_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor)`.

On the BF16 backend only `c_tensor` and `d_tensor` are set; the other four are `None`.

---

## Support surface and constraints

### Layouts

- `A` must be `k`-major
- Block-scaled: `B` must be `k`-major
- Block-scaled: discrete `B` supports `b_major="k"` and supported FP8 `b_major="n"` configurations
- BF16: dense `B` is `k`-major; discrete `B` takes `b_major="k"` or `"n"`; see [BF16 arguments](#bf16-arguments)
- `C`, `D`, and `D_col` must be `n`-major
- The wrapper only supports `cd_major="n"`

### Dtypes

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- Scale-factor dtype constraint: `sf_vec_size == 32` is unsupported when `sf_dtype == float8_e4m3fn`
- Input dtype constraint: FP8 `A`/`B` inputs require `sf_vec_size == 32`
- Grouped FP8 currently requires `discrete_col_sfd=True`
- BF16 `A` and `B` select the [BF16 backend](#bf16-backend); pairing a BF16 operand with a
  block-scaled one raises `ValueError`

### Shapes and environment

- `prob_tensor` is required
- `m_aligned` must be `256`
- Requires CUDA with SM100+ compute capability; the BF16 backend runs on SM100 and SM103 only

---

## BF16 backend

When `A` and `B` (`b_tensor`, or `b_dtype` in discrete mode) are `torch.bfloat16`, the wrapper and
`GroupedGemmSreluSm100` run a BF16 grouped GEMM with a fused weighted squared-ReLU epilogue. It
has no scale factors, no `alpha`, no bias and no output quantization. It runs on SM100 and SM103
and takes torch tensors only.

For rows of expert `g`:

$$
Z[m, n] = \sum_k A[m, k] \cdot B[n, k, g] \quad \text{(BF16 operands, FP32 accumulation)}
$$

$$
C[m, n] = \mathrm{bf16}(Z[m, n]), \qquad D[m, n] = \mathrm{bf16}\big(\mathrm{prob}[m] \cdot \mathrm{srelu}(z_{in}[m, n])\big)
$$

with $z_{in} = Z$ for `act_input="acc"` (default) or $z_{in} = \mathrm{float}(C)$ for
`act_input="saved"`, and

- plain (`tanh_clamp_scale=None`): $\mathrm{srelu}(z) = r \cdot r$ with $r = \max(z, 0)$;
- tanh-clamped (`tanh_clamp_scale=s`): $t = \tanh(\max(z, 0) \cdot \mathrm{fp32}(1/s))$,
  $c = s \cdot t$ and $\mathrm{srelu}(z) = c \cdot c$.

The epilogue is inline PTX with every multiply rounded to nearest, so the result does not depend
on the compiler's fast-math choices. `prob` may have any sign.

`tanh_mode` selects the tanh-clamped path's `tanh`:

- `"approx"` (default): MUFU `tanh.approx.f32`, about $2^{-11}$ relative error, capped at 1. It is
  the instruction the block-scaled kernel uses.
- `"accurate"`: a Cephes-style FP32 `tanh`, an odd polynomial for arguments below 0.625 and
  $1 - 2/(e^{2u} + 1)$ above (`ex2.approx`, and `rcp.approx` refined by one Newton step). Its
  relative error against an exact `tanh` is below $3\times10^{-7}$. Only the BF16 backend has it.

The plain squared-ReLU evaluates no `tanh`: it accepts either value of `tanh_mode` and builds one
kernel for both. A NaN pre-activation propagates to `D` in every mode, since the relu is a
NaN-propagating `max`; the block-scaled kernel maps it to +0 instead.

`C` is always written. With `act_input="saved"` the forward evaluates the activation at exactly
that saved BF16 value. With `act_input="acc"` (default), `D` comes from the unrounded
FP32 accumulator and is slightly more accurate. The plain squared-ReLU with `act_input="saved"` is
bit for bit `r = torch.relu(c.float()); ((r * r) * prob).to(torch.bfloat16)` of the returned `C`:
FP32 math on `float(C)`, rounded once. The all-BF16 eager form
`(torch.relu(c) * torch.relu(c) * prob).to(torch.bfloat16)` rounds `relu(c) ** 2` to BF16 before
the weighting and can differ by one BF16 ulp.

### BF16 arguments

| Argument | BF16 meaning | Default |
| --- | --- | --- |
| `a_tensor` / `sample_a` | `(valid_m, K, 1)` BF16, `k`-major | required |
| `b_tensor` / `sample_b` | dense weights: `(N, K, L)` BF16, `k`-major, stride `(K, 1, N * K)` (an `[L, N, K]` tensor `w` passes as `w.permute(1, 2, 0)`) | dense or discrete weights required |
| `b_ptrs`, `n`, `b_dtype`, `b_major` (wrapper); `num_experts`, `b_shape=(N, K)`, `b_dtype`, `b_major` (class) | discrete weights: a contiguous int64 CUDA tensor of per-expert BF16 pointers, each to a row-major `[N, K]` (`b_major="k"`) or `[K, N]` (`b_major="n"`) matrix | `b_major="k"` |
| `padded_offsets` / `sample_padded_offsets` | `(L,)` int32 cumulative row ends | required |
| `prob_tensor` / `sample_prob` | `(valid_m, 1, 1)` float32 | required |
| `sample_c` / `c_tensor`, `sample_d` / `d_tensor` (class) | `(valid_m, N, 1)` BF16, `n`-major; the wrapper allocates them | required |
| `tanh_clamp_scale` | `None`: plain; `s > 0`: tanh-clamped, baked into the kernel | `None` |
| `tanh_mode` | `"approx"` or `"accurate"` | `"approx"` |
| `act_input` | `"acc"` or `"saved"` | `"acc"` |
| `acc_dtype`, `c_dtype`, `d_dtype` | FP32 accumulation and BF16 `C` and `D` only; other values raise `ValueError` | FP32, BF16, BF16 |
| `mma_tiler_mn` | `(256, tile_n)` for a 2-CTA tile or `(128, tile_n)` for a 1-CTA tile, `tile_n` a multiple of 32 up to 256 | `(256, 256)` |
| `cluster_shape_mn` | powers of two, at most 16 CTAs; `cluster_shape_mn[0]` is 2 for a 256-row tile, 1 or 2 for a 128-row tile | `(2, 1)` for a 256-row tile, else `(1, 1)` |
| `use_dynamic_sched` | dynamic tile scheduler | `False` |
| `m_aligned`, `cd_major` | fixed | `256`, `"n"` |
| `current_stream` | launch stream; the launch always runs on the operands' device, and the caller's device stays current | torch's current stream on the operands' device |
| `workspace` (class `execute()`, keyword-only) | caller-owned scratch; see [Class-API workspace](#class-api-workspace) | required |

`sfa_tensor`, `sfb_tensor`, `sfb_ptrs`, `alpha_tensor`, `bias_tensor`, `norm_const_tensor`, an
`sf_vec_size` other than 16, `vector_f32=True` and `discrete_col_sfd=True` are block-scaled only
and raise `ValueError` on BF16, as do the class's `sample_sfa`, `sample_sfb`, `sample_alpha`,
`sample_bias`, `sample_d_col`, `sample_sfd_row`, `sample_sfd_col`, `sample_amax` and
`sample_norm_const`. In dense mode the wrapper also rejects `n` and `b_dtype`. Through the class
API pass `sfa_tensor=None` and `alpha_tensor=None` to `execute()` and leave its other block-scaled
arguments unset. The wrapper returns `c_tensor` and `d_tensor`; the other keys are `None`.

### BF16 constraints

- SM100 and SM103 (compute capability 10.0 and 10.3) only: `check_support()` raises
  `RuntimeError` on any other GPU, Rubin (SM107) included.
- torch tensors only; JAX arrays raise `ValueError`.
- CuTe DSL 4.8.0 or newer, the version the BF16 kernels are qualified on (they use
  `cutlass.experimental`). Below it, `check_support()` and the wrapper raise `ImportError` naming
  the installed version, and the BF16 kernel modules are never imported; the block-scaled backend
  is unaffected.
- `valid_m`, the row count of `A`, is a multiple of 256, and so is every expert's row range (empty
  experts are allowed). `padded_offsets[-1]` may be smaller than `valid_m`: rows at or past it are
  not written, so `A` may carry padding rows past the routed total.
- `N % 32 == 0`; `N` need not be a multiple of the tile N (`N = 1856` runs a partial last tile).
  `K % 8 == 0`. `1 <= L <= 1024`. `valid_m * max(N, K)` and `L * N * K` stay below $2^{31}$.
- Every operand is on the same CUDA device. Data pointers are 16-byte aligned; the discrete pointer
  array is 8-byte aligned and its entries are 16-byte aligned.

### Device-data contract

`padded_offsets` values and `b_ptrs` entries are a **device-data contract**: the kernel reads them
on device, and neither `check_support()` nor `execute()` copies them to the host. Offsets must be
non-decreasing multiples of 256 with the last in `(0, valid_m]`, and pointers non-null and 16-byte
aligned; malformed values are undefined behavior, as for any raw-pointer interface. Set
`CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1` before importing cudnn (it is read once, at
import) to have `execute()` check those values with a blocking host read; that mode raises
`RuntimeError` while the launch stream is capturing a CUDA graph. The API records the pointer
array on the launch stream; keep every expert weight it points to alive and unchanged until that
stream completes.

### Class-API workspace

The BF16 `execute()` allocates nothing. It carves its scratch from `workspace=`, a keyword-only
argument after `current_stream`:

- **Size.** At least `op.scratch_workspace_bytes()` bytes; that call runs `check_support()` first
  when needed. The scratch holds the per-expert B TMA descriptors (discrete weights) and the
  dynamic scheduler's tile counter, rounded up to a multiple of 128 bytes and never 0: 128 bytes
  for dense weights with either scheduler, and for discrete weights $128 L$ bytes, or
  $128 L + 128$ with `use_dynamic_sched=True`.
- **Placement.** Contiguous, 128-byte aligned and on the operands' CUDA device: a torch `uint8`
  tensor from `torch.empty`, or any DLPack device buffer.
- **No overlap.** The carved bytes must not overlap any operand. `execute()` checks every operand
  passed to it, `b_ptrs` included, from host metadata before the launch; disjoint views of one
  allocation are accepted. The check does not read the `b_ptrs` entries, so keeping the weights
  they point to clear of the workspace is up to the caller.
- **Reuse.** Every launch rewrites the descriptors and resets the counter. Launches that can
  overlap, on different streams or in CUDA graphs replayed concurrently, need distinct
  workspaces; launches ordered on one stream can share one.
- **Always checked.** `workspace` is required, and validated, even when `a_tensor` has no rows.

The wrapper allocates this scratch on the launch stream for every call, so a cached wrapper plan
holds no device memory and concurrent wrapper calls never share scratch.

```python
import torch
from cudnn import GroupedGemmSreluSm100

# a: (valid_m, K, 1) BF16; b: (N, K, L) BF16 with stride (K, 1, N * K);
# padded_offsets: (L,) int32; prob: (valid_m, 1, 1) float32
c = torch.empty(valid_m, N, 1, dtype=torch.bfloat16, device=a.device)
d = torch.empty_like(c)
op = GroupedGemmSreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d=d,
    sample_padded_offsets=padded_offsets,
    sample_prob=prob,
    tanh_clamp_scale=16.0,
)
assert op.check_support()  # selects the BF16 backend; raises on an unsupported configuration
op.compile()
workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device=a.device)
op.execute(
    a_tensor=a,
    sfa_tensor=None,
    padded_offsets=padded_offsets,
    alpha_tensor=None,
    c_tensor=c,
    d_tensor=d,
    b_tensor=b,
    prob_tensor=prob,
    workspace=workspace,
)
```

### CUDA graphs

Apart from the debug mode above, the BF16 wrapper and `execute()` make no host reads and no stream
synchronizations, so both can be captured in a CUDA graph once the kernel is compiled: by an eager
wrapper call with the same configuration, or by `compile()` for the class API. A replay reuses the
captured addresses, and the kernel reads `padded_offsets` and the `b_ptrs` entries on device at
every launch, so values written into those same tensors between replays take effect. Each
captured wrapper call allocates its own scratch from the graph's memory pool. A class-API capture
keeps using its workspace for as long as the graph can replay, and graphs that can replay
concurrently need distinct workspaces.

### Example: Nemotron 3 expert FC1

```python
import torch
from cudnn import grouped_gemm_srelu_wrapper_sm100

E, K, N = 16, 2688, 1856  # experts per rank, hidden size, expert FFN size
counts = [3072] * E  # routed rows per expert, each padded to a multiple of 256
valid_m = sum(counts)
padded_offsets = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
x = torch.randn(valid_m, K, device="cuda", dtype=torch.bfloat16)
prob = torch.rand(valid_m, 1, 1, device="cuda")  # routing probability of each row
fc1 = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * K**-0.5 for _ in range(E)]

out = grouped_gemm_srelu_wrapper_sm100(
    a_tensor=x.unsqueeze(-1),
    b_ptrs=torch.tensor([w.data_ptr() for w in fc1], dtype=torch.int64, device="cuda"),
    n=N,
    b_dtype=torch.bfloat16,
    padded_offsets=padded_offsets,
    prob_tensor=prob,
    tanh_clamp_scale=None,  # Nemotron 3: plain relu(z) ** 2; e.g. 16.0 for the tanh-clamped variant
)
z_save, h = out["c_tensor"], out["d_tensor"]  # (valid_m, N, 1) BF16 each
```

---

## Usage examples

For end-to-end usage and regression coverage, see:

- `test/python/gemm/cutedsl/test_grouped_gemm_srelu.py`
- `test/python/gemm/cutedsl/test_grouped_gemm_srelu_utils.py`
- `test/python/gemm/cutedsl/test_grouped_gemm_srelu_bf16.py` (BF16 backend)
- `test/python/gemm/cutedsl/test_grouped_gemm_srelu_bf16_utils.py` (BF16 references)
