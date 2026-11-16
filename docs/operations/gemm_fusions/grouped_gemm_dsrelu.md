# Grouped GEMM + dsReLU (SM100)

**This is an experimental API and subject to change.**

## JAX support

**Canonical block-scaled (dense weights):** `grouped_gemm_dsrelu_jax_sm100` takes keyword `b_tensor` `(experts, n, k)` and `sfb_tensor` with canonical A `(m, k)`, saved C `(m, n)`, prob `(m,)` fp32/bf16 and already MMA-packed scale bytes (physical or flat). A/B are MXFP8 with E8M0 scales (`sf_vec_size=32`), or packed FP4. Packed FP4 operands travel as `uint8` with two values per byte (JAX has no packed FP4 dtype), so K is twice the array extent; their scale buffers are E4M3 with `sf_vec_size=16` for NVFP4, or E8M0. It returns a `TupleDict` with the eager wrapper's keys. MXFP8 inputs take an FP8 `d_dtype` and `norm_const_tensor`, and return D_row, D_col, D_srelu and the three physical 6-D SFD buffers. FP4 inputs take a BF16/FP16 `d_dtype` and return D_row, D_srelu and the per-expert amax `(experts, 1)`. dprob `(m,)` is zero-initialized for accumulation; other rows at or past `padded_offsets[-1]` are unspecified. `grouped_gemm_dsrelu_wrapper_sm100` dispatches JAX inputs with a dense `b_tensor` to it; pass `sf_vec_size` and leave `dprob_tensor` unset. dbias, dSReLU reuse, deterministic reduction, `tanh_clamp_scale`, dynamic scheduling and `generate_d_srelu=False` raise `ValueError` (the entry always writes D_srelu). SM100+, as in Torch.

**Discrete FP8 weights:** supports **JAX arrays** in the discrete (b_ptrs) FP8 configurations: pointer arrays as int64 (jax x64 mode) or packed uint8 (8 bytes per pointer), scale-factor tensors in the physical C-contiguous atom shape `(L, MN', K', 32, 4, 4)` (the kernel rebuilds SF layouts from the GEMM shapes and reads only the base pointer), outputs allocated as C-contiguous `jnp` arrays. Dense weight mode and packed-fp4 A/B are not expressible as JAX arrays and raise clear errors. The wrapper is eager, on the CUDA legacy default stream: `block_until_ready` inputs, synchronize before reading outputs; keep weight arrays alive until the kernel completes.

For jitted JAX programs use the `jax.jit`-compatible XLA custom-call entry point `grouped_gemm_dsrelu_jax_sm100` (built on `cutlass.jax.cutlass_call`; discrete FP8 mode, `sf_vec_size=32`): all outputs (d/SFD tensors, `dprob`, and with `generate_dbias=True` `dbias`) are XLA-managed donated zero-initialized buffers — no manual synchronization. Under tracing the `padded_offsets` *values* cannot be host-validated, and the weight/scale buffers behind the pointer arrays must stay alive and unmoved across every execution of the traced computation.

The [BF16 backend](#bf16-backend) takes torch tensors only: BF16 JAX arrays raise `ValueError` in the wrapper, before anything is allocated, and in `GroupedGemmDsreluSm100.check_support()`.

## Overview

**Grouped GEMM + dsReLU backward fusion**: A grouped block-scaled GEMM fused with a probability-gradient backward epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. The API supports dense contiguous weights and discrete per-expert weight allocations. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **dsReLU backward epilogue** using the forward/intermediate tensor `C`
3. **Optional output quantization** through `SFD_row` / `SFD_col` or `Amax`

BF16 `A`/`B` select a separate BF16 kernel behind the same wrapper and class, on SM100 and
SM103 and for torch tensors only: no scale factors, no quantization, and `C` is the BF16
pre-activation the [sReLU forward](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#bf16-backend)
saved. See [BF16 backend](#bf16-backend).

### Shapes

- **Inputs**
  - `A`: contiguous activation tensor across all groups, shape `(valid_m, K, 1)`
  - `B`: dense weight tensor across all groups, shape `(N, K, L)`, or discrete per-expert tensors addressed by `b_ptrs`
  - `C`: forward/intermediate tensor, shape `(valid_m, N, 1)`
  - `SFA`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), 1)`
  - `SFB`: dense scale-factor tensor, shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(K, sf_vec_size), 4), L)`, or discrete per-expert tensors addressed by `sfb_ptrs`
  - `padded_offsets`: cumulative padded group ends, shape `(L,)`
  - `alpha`: per-group scaling factors, shape `(L,)`
  - `prob`: per-row gating probabilities, shape `(valid_m, 1, 1)`

- **Outputs**
  - `D_row`: row output after dsReLU, shape `(valid_m, N, 1)`
  - `D_col`: column output after dsReLU, shape `(valid_m, N, 1)`
  - `dprob`: probability gradient, shape `(valid_m, 1, 1)`
  - `D_srelu`: recomputed forward output, shape `(valid_m, N, 1)`, unless `generate_d_srelu=False`
  - `SFD_row`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), 1)` when `D_row` is FP8
  - `SFD_col`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(valid_m, sf_vec_size), 4), 1)` when `D_row`/`D_col` is FP8
  - `SFD_col_d_srelu`: shape of `SFD_col`, when `D_srelu` is FP8
  - `Amax`: shape `(L, 1)` when `D_row` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Equations

For rows belonging to expert `g`:

$$
G[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB)
$$

$$
D\_{row}[m, n] = \mathrm{prob}[m, 0, 0] \cdot 2 \cdot G[m, n] \cdot \mathrm{relu}(C[m, n, 0])
$$

$$
\mathrm{dprob}[m, 0, 0] = \sum_n G[m, n] \cdot \mathrm{relu}(C[m, n, 0])^2
$$

#### Tanh soft clamp (`tanh_clamp_scale`)

`tanh_clamp_scale=s` differentiates the soft-clamped forward epilogue
(see [grouped_gemm_srelu](/operations/Matmul/gemm_fusions/grouped_gemm_srelu)) instead of the plain one. Writing
$t = \tanh(\mathrm{relu}(C)/s)$ and $b = s\,t$, both equations keep their shape with
$\mathrm{relu}(C)$ replaced by $b$, and the row output picks up the chain-rule factor
$1 - t^2$:

$$
D\_{row}[m, n] = \mathrm{prob}[m, 0, 0] \cdot 2 \cdot G[m, n] \cdot b[m, n] \cdot \left(1 - t[m, n]^2\right)
$$

$$
\mathrm{dprob}[m, 0, 0] = \sum_n G[m, n] \cdot b[m, n]^2
$$

Both kernels must be built with the same `s` and the same `tanh_mode`, or the saved
pre-activation is differentiated against the wrong nonlinearity. `s` is baked in at compile time, so distinct values compile
distinct kernels. Determinism is unaffected: the clamp changes only the per-element value that
feeds the dprob accumulation, never the slotting or reduction order, so
[deterministic dprob](#deterministic-dprob) stays bit-exact for the clamped path too.

`D_col` stores the companion column-quantized output used by the grouped kernel family. When FP8 output is enabled, the kernel also emits `SFD_row` and `SFD_col`. When fp16/bf16 output is used, the kernel can emit per-expert `Amax`.

`D_srelu` is the recomputed forward output, $\mathrm{prob}[m, 0, 0] \cdot \mathrm{relu}(C[m, n, 0])^2$
($\mathrm{prob} \cdot b^2$ when soft-clamped), for callers that recompute the FC2 input instead
of saving it. The wrapper writes it unless `generate_d_srelu=False`, and the class API when it is
built with `sample_d_srelu`; with FP8 output it comes with its column scale factors
`SFD_col_d_srelu`.

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
                 G (valid_m×N×1)
                     |
       C (valid_m×N×1)+
                     |
                     +--> D_row / D_col
                     |
                     +--> dprob
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
from cudnn import grouped_gemm_dsrelu_wrapper_sm100

result = grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor=a,
    b_tensor=b,
    c_tensor=c,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    norm_const_tensor=norm_const,
    acc_dtype=torch.float32,
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

d_row, d_col, d_srelu, dprob, dbias, amax, sfd_row, sfd_col, sfd_col_d_srelu = result
```

### Discrete-weight wrapper

```python
result = grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor=a,
    c_tensor=c,
    sfa_tensor=sfa,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    b_ptrs=b_ptrs,          # int64 device tensor of per-expert B pointers
    sfb_ptrs=sfb_ptrs,      # int64 device tensor of per-expert SFB pointers
    n=N,
    b_dtype=torch.float4_e2m1fn_x2,
    b_major="k",
    d_dtype=torch.bfloat16,
)
```

### Class API

```python
from cudnn import GroupedGemmDsreluSm100

op = GroupedGemmDsreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=c,
    sample_d_row=d_row,
    sample_d_col=d_col,
    sample_sfa=sfa,
    sample_sfb=sfb,
    sample_padded_offsets=padded_offsets,
    sample_alpha=alpha,
    sample_prob=prob,
    sample_dprob=dprob,
    sample_sfd_row=sfd_row,
    sample_sfd_col=sfd_col,
    sample_amax=amax,
    sample_norm_const=norm_const,
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
    d_row_tensor=d_row,
    d_col_tensor=d_col,
    sfa_tensor=sfa,
    sfb_tensor=sfb,
    padded_offsets=padded_offsets,
    alpha_tensor=alpha,
    prob_tensor=prob,
    dprob_tensor=dprob,
    sfd_row_tensor=sfd_row,
    sfd_col_tensor=sfd_col,
    amax_tensor=amax,
    norm_const_tensor=norm_const,
    current_stream=None,
)
```

The constructor only records its arguments. `check_support()` selects the backend from the
operand dtypes, builds it and validates the configuration; `compile()` and
`scratch_workspace_bytes()` run it first when it has not run yet. Configuration errors therefore
surface at `check_support()`, not at construction: the constructor itself rejects only
unsupported tensor frameworks. The block-scaled backend owns its scratch, so its
`scratch_workspace_bytes()` returns 0 and `execute()` raises `ValueError` when given a
`workspace`. The BF16 backend's `execute()` takes a caller-owned workspace; see
[BF16 backend](#bf16-backend).

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
- Input tensor **C**: `c_tensor` (wrapper) or `sample_c` / `c_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype: `{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}`
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
- Output tensor **D_row**: `result["d_row_tensor"]` (wrapper) or `sample_d_row` / `d_row_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must be `n`-major
  - Dtype:
    - FP4 input: `{float16, bfloat16, float32}`
    - FP8 input: `{float8_e4m3fn, float8_e5m2}`
- Output tensor **D_col**: `result["d_col_tensor"]` (wrapper) or `sample_d_col` / `d_col_tensor` (class)
  - Shape: `(valid_m, N, 1)`
  - Layout: must match `D_row`
  - Dtype: must match `D_row`
- Output tensor **dprob**: `result["dprob_tensor"]` (wrapper) or `sample_dprob` / `dprob_tensor` (class)
  - Shape (wrapper): `(valid_m, 1, 1)`, whether or not `deterministic` is set — the per-N-tile
    workspace and the reduction into it are internal
  - Shape: `(valid_m, 1, 1)` in both modes. Under `deterministic=True` the class API also takes
    `sample_dprob_workspace` / `dprob_workspace_tensor` (`dprob_workspace_shape(valid_m, n)`,
    float32); after `execute()` the caller calls `reduce_dprob_workspace(slots, dprob_tensor)`
    and zeroes the slots. The BF16 `execute()` finishes that reduction itself, and the same
    sequence is correct there; see [dprob reduction](#dprob-reduction)
  - Dtype: `float32`
- Output tensor **D_srelu**: `result["d_srelu_tensor"]` (wrapper) or `sample_d_srelu` / `d_srelu_tensor` (class)
  - The recomputed forward output; omitted with `generate_d_srelu=False` (wrapper) or without
    `sample_d_srelu` (class)
  - Shape: `(valid_m, N, 1)`
  - Dtype: must match `D_row`
- Output tensors **SFD_row** / **SFD_col**
  - Dtypes: must match `SFA`
  - Generated when `D_row` / `D_col` uses an FP8 dtype
- Output tensor **SFD_col_d_srelu**: `result["sfd_col_d_srelu_tensor"]` (wrapper) or `sample_sfd_col_d_srelu` / `sfd_col_d_srelu_tensor` (class)
  - Column scale factors of an FP8 `D_srelu`, shaped like `SFD_col`
  - Dtype: must match `SFA`
- Output tensor **Amax**
  - Shape: `(L, 1)`
  - Dtype: `float32`
  - Generated when `D_row` / `D_col` uses `float16` or `bfloat16`
- Input tensor **Norm Const**
  - Shape: `(1,)`
  - Dtype: `float32`
  - Required when `SFD_row` / `SFD_col` are generated for FP8 output

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
- `cd_major: str` (wrapper only)
  - Specifies the major dimension for `C` and `D` tensors
  - Only `"n"` (n-major layout) is supported
- `discrete_col_sfd: bool`
  - Enables the discrete column-scale-factor path used by grouped FP8
- `deterministic: bool | None`
  - Makes `dprob` bit-exact run to run — see [Deterministic dprob](#deterministic-dprob), and
    [dprob reduction](#dprob-reduction) for the BF16 backend
  - Wrapper: `None` (default) follows `torch.use_deterministic_algorithms`
  - Class API: plain `bool`, default `False`; the class never reads torch's flag
- `tanh_mode: str`
  - `"approx"` (default) or `"accurate"`; any other value, `None` included, raises `ValueError`
  - Must match the forward pass that produced `C`, like `tanh_clamp_scale`: the backward recomputes
    `t` and `D_srelu` from `C`
  - With `tanh_clamp_scale` set, `"approx"` is MUFU `tanh.approx.f32` (capped at 1) on both
    backends, and `"accurate"` is the BF16 backend's Cephes-style FP32 `tanh`; the block-scaled
    backend raises `ValueError` for it
  - Without `tanh_clamp_scale` no `tanh` is evaluated, and both backends ignore the value
  - A NaN in `C` propagates on the BF16 backend in every mode; the block-scaled kernel maps it to
    +0, clamped or not
- `generate_d_srelu: bool` (wrapper)
  - `True` (default): also write `d_srelu_tensor`, the recomputed forward output
    `prob * srelu(C)` (the FC2 input, for callers that recompute it rather than save it)
  - `False` skips that write (and, block-scaled, its column scale factors) and returns
    `d_srelu_tensor=None`; the JAX dense path rejects it
  - The class API keys the same choice on `sample_d_srelu`
- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `d_row_tensor`
- `d_col_tensor`
- `d_srelu_tensor` (`None` with `generate_d_srelu=False`)
- `dprob_tensor`
- `dbias_tensor`
- `amax_tensor`
- `sfd_row_tensor`
- `sfd_col_tensor`
- `sfd_col_d_srelu_tensor`

Tuple unpacking order is: `(d_row_tensor, d_col_tensor, d_srelu_tensor, dprob_tensor, dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor, sfd_col_d_srelu_tensor)`.

On the BF16 backend `d_col_tensor`, `dbias_tensor`, `amax_tensor` and the three scale-factor
tensors are `None`.

---

## Support surface and constraints

### Layouts

- `A` must be `k`-major
- Block-scaled: `B` must be `k`-major
- Block-scaled: discrete `B` supports `b_major="k"` and supported FP8 `b_major="n"` configurations
- BF16: dense `B` may be `k`- or `n`-major; discrete `B` takes `b_major="k"` or `"n"`; see [BF16 arguments](#bf16-arguments)
- `C`, `D_row`, and `D_col` must be `n`-major
- The wrapper only supports `cd_major="n"`

### Dtypes

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- Scale-factor dtype constraint: `sf_vec_size == 32` is unsupported when `sf_dtype == float8_e4m3fn`
- Input dtype constraint: FP8 `A`/`B` inputs require `sf_vec_size == 32`
- Grouped FP8 currently requires `discrete_col_sfd=True`
- Block-scaled: grouped `dsrelu` requires the kernel-supported `k`-major `B` layout
- BF16 `A` and `B` select the [BF16 backend](#bf16-backend); pairing a BF16 operand with a
  block-scaled one raises `ValueError`

### Shapes and environment

- `m_aligned` must be `256`
- Requires CUDA with SM100+ compute capability; the BF16 backend runs on SM100 and SM103 only

---

## BF16 backend

When `A` and `B` (`b_tensor`, or `b_dtype` in discrete mode) are `torch.bfloat16`, the wrapper and
`GroupedGemmDsreluSm100` run a BF16 grouped GEMM fused with the backward of the
[BF16 sReLU forward](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#bf16-backend). It has no
scale factors, no `alpha`, no `dbias` and no output quantization. It runs on SM100 and SM103 and
takes torch tensors only.

`A` is the gradient of the FC2 output, `(valid_m, D, 1)` with `D` the hidden size, and `B` the
FC2 weight viewed as `(N, D, L)`; `C` is the BF16 pre-activation the forward returned as its
`c_tensor`. For rows of expert `g`, with $\mathrm{dH}$ kept in FP32 and never rounded:

$$
\mathrm{dH}[m, n] = \sum_k A[m, k] \cdot B[n, k, g], \qquad z = \mathrm{float}(C[m, n])
$$

- plain (`tanh_clamp_scale=None`): $r = \max(z, 0)$, $a = r \cdot r$ and
  $D_{\mathrm{row}} = \mathrm{bf16}\big((\mathrm{dH} \cdot \mathrm{prob}[m]) \cdot (r + r)\big)$;
- tanh-clamped (`tanh_clamp_scale=s`): $t = \tanh(\max(z, 0) \cdot \mathrm{fp32}(1/s))$,
  $c = s \cdot t$, $a = c \cdot c$ and
  $D_{\mathrm{row}} = \mathrm{bf16}\big(((\mathrm{dH} \cdot \mathrm{prob}[m]) \cdot (c + c)) \cdot (1 - t^2)\big)$;
- $\mathrm{dprob}[m] \mathrel{+}= \sum_n a \cdot \mathrm{dH}$ in FP32;
- $D_{\mathrm{srelu}} = \mathrm{bf16}(a \cdot \mathrm{prob}[m])$ (`d_srelu_tensor`) when
  `generate_d_srelu=True`.

`t`, `c`, `a` and `a * prob` are the forward kernel's own instructions (the same inline PTX and
`tanh_mode`), so `tanh_clamp_scale` and `tanh_mode` must match the forward's. When the forward ran
with `act_input="saved"`, `d_srelu_tensor` equals the forward's `d_tensor` bit for bit. The plain
`D_row` rounds exactly like the FP32 eager expression `((dH * prob) * 2) * torch.relu(c.float())`
at the same `dH`. A NaN in `C` propagates to `D_row`, `d_srelu` and that row's `dprob`, as in the
forward.

### dprob reduction

`deterministic` keeps its meaning: the wrapper's default `None` follows
`torch.use_deterministic_algorithms`, while the class defaults to `False` and never reads torch's
flag.

- `False`: every N tile adds its row partial into `dprob` with an FP32 atomic, so the tiles of a
  row are summed in completion order and `dprob` can differ run to run in the last bits. `D_row`
  and `d_srelu` are unaffected.
- `True`: each (row, N tile) partial goes to its own slot of a
  `dprob_workspace_shape(valid_m, n, mma_tiler_mn, cluster_shape_mn)` FP32 buffer, each epilogue
  thread summing its row of a tile in a fixed order. A second small kernel then adds each row's
  slots to `dprob` as a strict left fold in increasing tile order and zeroes them. `dprob` is
  bit-exact run to run and independent of tile scheduling.

Both modes accumulate into `dprob_tensor`: zero it before each call (the wrapper allocates zeros
when no `dprob_tensor` is passed). Through the class API, `deterministic=True` needs
`sample_dprob_workspace` / `dprob_workspace_tensor` of that shape: FP32, row-major and
zero-initialized once. Pass it to every `execute()`, and omit it without `deterministic=True`.
The BF16 `execute()` completes the reduction itself and leaves the slots zeroed for the next call,
so the result is already in `dprob_tensor`. Code written for both backends can keep the
block-scaled sequence, which is correct here as well:

```python
op.execute(..., dprob_tensor=dprob, dprob_workspace_tensor=slots)
GroupedGemmDsreluSm100.reduce_dprob_workspace(slots, dprob)  # BF16: adds zero
slots.zero_()  # block-scaled: required before the next execute(); BF16: already zero
```

With an explicit `current_stream`, pass it to `reduce_dprob_workspace` as well and zero the slots
on that stream. **The return-value form `reduce_dprob_workspace(slots)` is not portable**: after a
BF16 `execute()` it returns zeros, not `dprob`. The slot buffer is separate from the `workspace=`
scratch and is not counted by `scratch_workspace_bytes()`.

Through the wrapper, deterministic mode costs a zero-filled slot buffer per call and the fold
kernel; atomic mode needs neither. The BF16 wrapper allocates its outputs, the zero-filled
`dprob` and slots, and its scratch on `current_stream`.

### BF16 arguments

| Argument | BF16 meaning | Default |
| --- | --- | --- |
| `a_tensor` / `sample_a` | `(valid_m, D, 1)` BF16, `k`-major: the gradient of the FC2 output | required |
| `c_tensor` / `sample_c` | `(valid_m, N, 1)` BF16, `n`-major: the pre-activation the forward saved | required |
| `b_tensor` / `sample_b` | dense weights: `(N, D, L)` BF16, `k`-major (stride `(D, 1, N * D)`) or `n`-major (stride `(1, N, N * D)`, an `[L, D, N]` FC2 weight) | dense or discrete weights required |
| `b_ptrs`, `n`, `b_dtype`, `b_major` (wrapper); `num_experts`, `b_shape=(N, D)`, `b_dtype`, `b_major` (class) | discrete weights: a contiguous int64 CUDA tensor of per-expert BF16 pointers; `b_major="n"` for row-major `[D, N]` FC2 weights, `"k"` for `[N, D]` | `b_major="k"` |
| `padded_offsets` / `sample_padded_offsets` | `(L,)` int32 cumulative row ends | required |
| `prob_tensor` / `sample_prob` | `(valid_m, 1, 1)` float32, the forward's `prob` | required |
| `dprob_tensor` / `sample_dprob` | `(valid_m, 1, 1)` float32, accumulated into | wrapper: allocates zeros; class: required |
| `sample_d_row` / `d_row_tensor` (class) | `(valid_m, N, 1)` BF16, `n`-major; the wrapper allocates it | required |
| `sample_d_srelu` / `d_srelu_tensor` (class) | `(valid_m, N, 1)` BF16, `n`-major; written exactly when given at construction | `None` |
| `generate_d_srelu` (wrapper) | write `d_srelu_tensor` | `True` |
| `use_dsrelu_reuse` | accepted: the BF16 epilogue always computes `relu(C)^2` (clamped: `c^2`) once for both `d_srelu` and `dprob`, so both values build the same kernel | `False` |
| `tanh_clamp_scale`, `tanh_mode` | as in the forward, and equal to the forward's values | `None`, `"approx"` |
| `deterministic` | see [dprob reduction](#dprob-reduction) | `None` (wrapper), `False` (class) |
| `sample_dprob_workspace` / `dprob_workspace_tensor` (class) | the deterministic slot buffer; see [dprob reduction](#dprob-reduction) | `None` |
| `acc_dtype`, `d_dtype` | FP32 accumulation and BF16 `D_row` only; other values raise `ValueError` | FP32, BF16 |
| `mma_tiler_mn`, `cluster_shape_mn`, `use_dynamic_sched` | as in the [forward](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#bf16-arguments) | `(256, 256)`; `(2, 1)` for a 256-row tile, else `(1, 1)`; `False` |
| `m_aligned`, `cd_major` | fixed | `256`, `"n"` |
| `current_stream` | launch stream; the launch always runs on the operands' device, and the caller's device stays current | torch's current stream on the operands' device |
| `workspace` (class `execute()`, keyword-only) | caller-owned scratch under the forward's [workspace contract](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#class-api-workspace) | required |

`sfa_tensor`, `sfb_tensor`, `sfb_ptrs`, `alpha_tensor`, `norm_const_tensor`,
`generate_dbias=True`, an `sf_vec_size` other than 16, `vector_f32=True` and
`discrete_col_sfd=True` are block-scaled only and raise `ValueError` on BF16, as do the
class's `sample_d_col`, `sample_sfa`, `sample_sfb`, `sample_alpha`, `sample_dbias`,
`sample_dbias_workspace`, `sample_sfd_row`, `sample_sfd_col`, `sample_sfd_col_d_srelu`,
`sample_amax` and `sample_norm_const`. In dense mode the wrapper also rejects `n` and `b_dtype`.
Through the class API pass `d_col_tensor=None`, `sfa_tensor=None` and `alpha_tensor=None` to
`execute()` and leave its other block-scaled arguments unset. The wrapper returns
`d_row_tensor`, `d_srelu_tensor` and `dprob_tensor`; the other keys are `None`.

### BF16 constraints

The forward's [BF16 constraints](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#bf16-constraints),
[device-data contract](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#device-data-contract),
[class-API workspace](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#class-api-workspace) and
[CUDA-graph notes](/operations/Matmul/gemm_fusions/grouped_gemm_srelu#cuda-graphs) apply unchanged,
with `D` as the reduction dimension:

- SM100 and SM103 only, torch tensors only.
- `valid_m` and every expert's row range are multiples of 256; `N % 32 == 0` (a partial last N
  tile is allowed), `D % 8 == 0`, `1 <= L <= 1024`, and `valid_m * max(N, D, slots)` and
  `L * N * D` stay below $2^{31}$, where `slots` is the deterministic slot count.
- `padded_offsets` and `b_ptrs` values are read only on device; set
  `CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1` before importing cudnn to check them.
- Rows at or past `padded_offsets[-1]` of `D_row` and `d_srelu` are not written, and their
  `dprob` keeps its value.
- `execute()` makes no host reads; in deterministic mode a captured graph holds the fold kernel
  too, and the deterministic slot buffer, like the workspace, must not be shared by launches that
  can overlap.

### Example: Nemotron 3 expert FC2 backward

```python
import torch
from cudnn import grouped_gemm_dsrelu_wrapper_sm100

# Continues the FC1 example of grouped_gemm_srelu: E, K, N, valid_m, padded_offsets,
# prob and z_save (its c_tensor) come from there.
D = K  # FC2 maps the expert FFN width N back to the hidden size
grad_y = torch.randn(valid_m, D, device="cuda", dtype=torch.bfloat16)  # gradient of the FC2 output
fc2 = [torch.randn(D, N, device="cuda", dtype=torch.bfloat16) * N**-0.5 for _ in range(E)]

out = grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor=grad_y.unsqueeze(-1),
    c_tensor=z_save,
    b_ptrs=torch.tensor([w.data_ptr() for w in fc2], dtype=torch.int64, device="cuda"),
    n=N,
    b_dtype=torch.bfloat16,
    b_major="n",  # each FC2 weight is a row-major [D, N]
    padded_offsets=padded_offsets,
    prob_tensor=prob,
    tanh_clamp_scale=None,  # must match the forward, as must tanh_mode
    deterministic=True,
    generate_d_srelu=False,  # True to recompute the FC2 input instead of saving it
)
grad_z, grad_prob = out["d_row_tensor"], out["dprob_tensor"]
```

The class API with deterministic `dprob`, here with dense weights:

```python
from cudnn import GroupedGemmDsreluSm100

a = grad_y.unsqueeze(-1)
b = torch.stack(fc2).permute(2, 1, 0)  # (N, D, L), the n-major view of an [L, D, N] FC2 weight
d_row = torch.empty(valid_m, N, 1, dtype=torch.bfloat16, device="cuda")
dprob = torch.zeros(valid_m, 1, 1, dtype=torch.float32, device="cuda")
slots_shape = GroupedGemmDsreluSm100.dprob_workspace_shape(valid_m, N)  # default tile and cluster
slots = torch.zeros(slots_shape, dtype=torch.float32, device="cuda")
op = GroupedGemmDsreluSm100(
    sample_a=a,
    sample_b=b,
    sample_c=z_save,
    sample_d_row=d_row,
    sample_padded_offsets=padded_offsets,
    sample_prob=prob,
    sample_dprob=dprob,
    sample_dprob_workspace=slots,
    deterministic=True,
)
assert op.check_support()
op.compile()
workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
op.execute(
    a_tensor=a,
    c_tensor=z_save,
    d_row_tensor=d_row,
    d_col_tensor=None,
    sfa_tensor=None,
    padded_offsets=padded_offsets,
    alpha_tensor=None,
    prob_tensor=prob,
    dprob_tensor=dprob,
    b_tensor=b,
    dprob_workspace_tensor=slots,
    workspace=workspace,
)
# dprob holds the reduced result and slots are zero again; zero dprob before the next call.
```

---

## Deterministic dprob

This section describes the block-scaled kernel; the BF16 kernel's mechanism is in
[dprob reduction](#dprob-reduction).

`dprob` is a float reduction rather than a single write per element, and by default it is not
reproducible run to run. It is non-deterministic at two levels:

1. **Within a CTA.** The N-subtile loop is traversed forward or reversed depending on the
   accumulator pipeline phase, which varies between runs. A running fp32 sum over a flipping
   order is not reproducible, because float addition is not associative.
2. **Across CTAs.** Every N-tile atomically accumulates into the same `dprob[token]`, so the
   summation order follows tile scheduling.

`deterministic=True` fixes both. **Neither fix is sufficient on its own** — fixing only the
cross-CTA atomic still leaves a divergent result.

1. Each subtile's partial goes into a slot indexed by the actual subtile, then the slots are
   summed in canonical order after the loop.
2. `dprob` is given one slot per N-tile, so each `(token, tile_n)` pair has exactly one
   writer, and those slots are reduced with `torch.sum` in fixed order.

Left unset, the flag follows torch:

```python
# Process-wide, along with every other deterministic algorithm:
torch.use_deterministic_algorithms(True)

# Or explicitly, per call site, independent of the torch setting:
result = cudnn.grouped_gemm_dsrelu_wrapper_sm100(..., deterministic=True)
```

`dprob_tensor` keeps its `(valid_m, 1, 1)` shape either way — the per-N-tile workspace and
the reduction into it are internal to the wrapper. The class API keeps that shape too and takes
the slots as separate scratch, `sample_dprob_workspace` / `dprob_workspace_tensor`; after
`execute()` the caller reduces them into `dprob_tensor` with `reduce_dprob_workspace` and zeroes
them (see [dprob reduction](#dprob-reduction)).

**Cost.** `grid_n ×` the `dprob` workspace, one reduction kernel, and `subtile_cnt` extra
registers per epilogue thread — and the last of those only for tile shapes that overlap the
accumulator, since that is what reverses the subtile loop. Deterministic and
non-deterministic configurations compile and cache separately.

`grid_n` is the number of N-tiles the scheduler can emit, `ceil_div(n, TILE_N × cluster_n) ×
cluster_n` — which is *not* `ceil_div(n, TILE_N)` unless `cluster_n` is 1, because the
scheduler counts whole clusters and then expands to CTAs.

**`dbias` is covered too, by a different mechanism.** By default the kernel accumulates it
across *M*-tiles with bf16 atomics (`red.global.add.noftz.bf16x2`) in an order set by tile
scheduling — a separate contention axis from `dprob`'s, since every N column is owned outright
by one `(tile_n, subtile)` pair.

Under `deterministic=True` the kernel instead writes one slot per `(absolute M-block, n)`,
which has exactly one writer, and the reduction sums those per expert. Groups are padded to a
multiple of `m_aligned`, so no M-block straddles two experts and each expert owns a contiguous
block range `[padded_offsets[e-1] / cta_tile_m, padded_offsets[e] / cta_tile_m)`. The workspace
is `ceil_div(valid_m, cta_tile_m) × n_out` **bf16** — 2 MiB at `valid_m=64k`, `n=2048`.

The slots stay bf16 deliberately. Reproducibility comes from the single writer and the
fixed-order reduction, not from a wider accumulator, so fp32 slots would double the memory and
split one packed `bf16x2` store into two scalar ones for no determinism benefit. Accuracy still
improves over the default: there each M-tile's atomic rounds the *running* sum, here each slot
rounds once and the segment matmul accumulates them in fp32.

That segment sum is a one-hot matmul rather than `index_add_`/`scatter_add_`, which are
themselves non-deterministic on CUDA, and rather than a per-expert slice, which would need
`padded_offsets` on the host — a sync in the training loop.

**The output arguments are identical in both modes.** `dprob` is `(valid_m, 1, 1)` float32 and
`dbias` is `(expert_cnt, n_out, 1)` bf16 whether or not the flag is set. What the flag adds is
scratch, and only for the class API: `sample_dprob_workspace` / `dprob_workspace_tensor` and
`sample_dbias_workspace` / `dbias_workspace_tensor`, sized by `dprob_workspace_shape(valid_m, n)`
and `dbias_workspace_shape(valid_m, n)`, reduced afterwards by `reduce_dprob_workspace` and
`reduce_dbias_workspace`. Use those rather than reducing by hand — a plain sum over dim 0 of the
dbias slots is wrong, and wrong quietly. The wrapper allocates and reduces both for you.

`check_support()` rejects `deterministic=True` with `m_aligned % (cta_tile_m × cluster_m) != 0`:
the slot index is only single-writer if the scheduler emits no M-tile past an expert's range,
which needs that division to be exact. Every supported shape satisfies it — `m_aligned` is
pinned to 256 and `cta_tile_m × cluster_m` is 128 or 256 — but a wider cluster would alias one
expert's phantom tiles onto the next expert's slots.

Every other output — `d_row`, `d_col`, `d_srelu`, the scale factors — is a single write per
element and is reproducible either way.

**Streams.** `dprob`, `dbias` and `amax` are accumulated into, so the wrapper initialises them
on `current_stream` rather than on torch's current stream; otherwise the memset is unordered
against the kernel and the guarantee is void whenever the caller runs on its own stream. Those
buffers are therefore allocated on `current_stream` too; the write-only outputs still come from
torch's stream and are `record_stream`-ed onto `current_stream` instead. A caller driving the class
API directly owns both of these itself.

---

## Usage examples

For end-to-end usage and regression coverage, see:

- `test/python/gemm/cutedsl/test_grouped_gemm_dsrelu.py`
- `test/python/gemm/cutedsl/test_grouped_gemm_dsrelu_utils.py`
- `test/python/gemm/cutedsl/test_grouped_gemm_dsrelu_bf16.py` (BF16 backend)
- `test/python/gemm/cutedsl/test_grouped_gemm_srelu_bf16_utils.py` (BF16 references)
