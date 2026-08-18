# Grouped GEMM + dsReLU (SM100)

**This is an experimental API and subject to change.**

## JAX support

Supports **JAX arrays** in the discrete (b_ptrs) FP8 configurations: pointer arrays as int64 (jax x64 mode) or packed uint8 (8 bytes per pointer), scale-factor tensors in the physical C-contiguous atom shape `(L, MN', K', 32, 4, 4)` (the kernel rebuilds SF layouts from the GEMM shapes and reads only the base pointer), outputs allocated as C-contiguous `jnp` arrays. Dense weight mode and packed-fp4 A/B are not expressible as JAX arrays and raise clear errors. The wrapper is eager, on the CUDA legacy default stream: `block_until_ready` inputs, synchronize before reading outputs; keep weight arrays alive until the kernel completes.

For jitted JAX programs use the `jax.jit`-compatible XLA custom-call entry point `grouped_gemm_dsrelu_jax_sm100` (built on `cudnn.jax.call`; discrete FP8 mode, `sf_vec_size=32`): all outputs (d/SFD tensors, `dprob`, and with `generate_dbias=True` `dbias`) are XLA-managed donated zero-initialized buffers — no manual synchronization. Under tracing the `padded_offsets` *values* cannot be host-validated, and the weight/scale buffers behind the pointer arrays must stay alive and unmoved across every execution of the traced computation.

## Overview

**Grouped GEMM + dsReLU backward fusion**: A grouped block-scaled GEMM fused with a probability-gradient backward epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE-style workloads. The API supports dense contiguous weights and discrete per-expert weight allocations. Groups are contiguous in the `M` dimension and described by `padded_offsets`.

This kernel performs:
1. **Block-scaled grouped GEMM** over contiguous expert ranges
2. **dsReLU backward epilogue** using the forward/intermediate tensor `C`
3. **Optional output quantization** through `SFD_row` / `SFD_col` or `Amax`

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
  - `SFD_row`: shape `(32, 4, ceil_div(valid_m, 128), 4, ceil_div(ceil_div(N, sf_vec_size), 4), 1)` when `D_row` is FP8
  - `SFD_col`: shape `(32, 4, ceil_div(N, 128), 4, ceil_div(ceil_div(valid_m, sf_vec_size), 4), 1)` when `D_row`/`D_col` is FP8
  - `Amax`: shape `(L, 1)` when `D_row` is fp16/bf16

`L` is the expert count and `valid_m = padded_offsets[-1]`.

### Equations

For rows belonging to expert `g`:

$$
G[m, n] = \alpha_g \sum_k \mathrm{dequantize}(A[m, k], SFA) \cdot \mathrm{dequantize}(B[n, k, g], SFB)
$$

$$
D\_{row}[m, n] = \mathrm{prob}[m, 0, 0] \cdot 2 \cdot C[m, n, 0] \cdot \mathrm{relu}(G[m, n])
$$

$$
\mathrm{dprob}[m, 0, 0] = \sum_n C[m, n, 0] \cdot \mathrm{relu}(G[m, n])^2
$$

`D_col` stores the companion column-quantized output used by the grouped kernel family. When FP8 output is enabled, the kernel also emits `SFD_row` and `SFD_col`. When fp16/bf16 output is used, the kernel can emit per-expert `Amax`.

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

d_row, d_col, dprob, dbias, amax, sfd_row, sfd_col = result
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

---

## Parameters

### Input/Output tensors

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
    float32) and the caller calls `reduce_dprob_workspace` after `execute()`
  - Dtype: `float32`
- Output tensors **SFD_row** / **SFD_col**
  - Dtypes: must match `SFA`
  - Generated when `D_row` / `D_col` uses an FP8 dtype
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
  - `TILE_N ∈ {128, 256}`
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
  - Makes `dprob` bit-exact run to run — see [Deterministic dprob](#deterministic-dprob)
  - Wrapper: `None` (default) follows `torch.use_deterministic_algorithms`
  - Class API: plain `bool`, default `False`
- CUDA stream (`current_stream` in class API, `current_stream` in wrapper)

### Wrapper return values

Returns a `TupleDict` with keys:

- `d_row_tensor`
- `d_col_tensor`
- `dprob_tensor`
- `dbias_tensor`
- `amax_tensor`
- `sfd_row_tensor`
- `sfd_col_tensor`

Tuple unpacking order is: `(d_row_tensor, d_col_tensor, dprob_tensor, dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor)`.

---

## Support surface and constraints

### Layouts

- `A` must be `k`-major
- `B` must be `k`-major
- Discrete `B` supports `b_major="k"` and supported FP8 `b_major="n"` configurations
- `C`, `D_row`, and `D_col` must be `n`-major
- The wrapper only supports `cd_major="n"`

### Dtypes

- `A` and `B` must have the same dtype
- `SFA`, `SFB`, `SFD_row`, and `SFD_col` must have the same dtype
- Scale-factor dtype constraint: `sf_vec_size == 32` is unsupported when `sf_dtype == float8_e4m3fn`
- Input dtype constraint: FP8 `A`/`B` inputs require `sf_vec_size == 32`
- Grouped FP8 currently requires `discrete_col_sfd=True`
- Grouped `dsrelu` requires the kernel-supported `k`-major `B` layout

### Shapes and environment

- `m_aligned` must be `256`
- Requires CUDA with SM100+ compute capability

---

## Deterministic dprob

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
the reduction into it are internal to the wrapper. The class API is lower level: pass
`sample_dprob` / `dprob_tensor` carrying one slot per N-tile and reduce over dim 1 yourself.

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

- `test/python/fe_api/grouped_gemm/test_grouped_gemm_dsrelu.py`
- `test/python/fe_api/grouped_gemm/test_grouped_gemm_dsrelu_utils.py`
