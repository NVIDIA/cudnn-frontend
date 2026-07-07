# RMSNorm + RHT + Amax (SM100)

**This is an experimental API and subject to change.**

## Overview

**RMSNorm + RHT + amax**: A fused CUTE DSL kernel for NVIDIA Blackwell GPUs (SM100+) that applies RMS normalization, a block-diagonal Hadamard transform with fixed block size `16`, and a per-CTA `amax` reduction.

This frontend integration exposes the kernel as a standard FE-OSS Python API with:
- a class API (`RmsNormRhtAmaxSm100`)
- a wrapper API (`rmsnorm_rht_amax_wrapper_sm100`)
- an explicit JAX API under `cudnn.jax`
- a framework-neutral operation signature in `op.py` and a CUTE DSL implementation in `kernel.py`
- grouped-gemm-style regression coverage for compile/execute, wrapper use, and cache reuse

## Shapes

- **Inputs**
  - `X`: activation tensor, shape `(M, N)`
  - `W`: RMSNorm scale tensor, shape `(N,)`

- **Outputs**
  - `O`: fused RMSNorm + RHT output tensor, shape `(M, N)`
  - `Amax`: per-CTA max-abs tensor, shape `(M / rows_per_cta,)`

`rows_per_cta` is the number of rows reduced into each `amax` element.

## Equations

For each row `m`:

$$
\mathrm{RMS}(X_m) = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} X[m, n]^2 + \epsilon}
$$

$$
Y[m, n] = \frac{X[m, n]}{\mathrm{RMS}(X_m)} \cdot W[n]
$$

Then apply the fixed Hadamard transform blockwise over 16-wide chunks:

$$
O[m, b] = Y[m, b] \cdot H_{16} / \sqrt{16}
$$

where `H_16` is the `16 x 16` Hadamard matrix and `b` indexes each 16-element block in the hidden dimension.

For each CTA covering `rows_per_cta` rows:

$$
\mathrm{Amax}[c] = \max |O|
$$

over every element produced by that CTA.

## API Usage

### High-level wrapper

```python
from cudnn import rmsnorm_rht_amax_wrapper_sm100

result = rmsnorm_rht_amax_wrapper_sm100(
    x_tensor=x,
    w_tensor=w,
    eps=1e-5,
    num_threads=None,   # optional override
    rows_per_cta=None,  # optional override
    current_stream=None,
)

o_tensor, amax_tensor = result
```

When no overrides are supplied, the wrapper uses the upstream-tuned thread table when available and an upstream-style `rows_per_cta` heuristic.

### Class API

```python
from cudnn import RmsNormRhtAmaxSm100

op = RmsNormRhtAmaxSm100(
    sample_x=x,
    sample_w=w,
    sample_o=o,
    sample_amax=amax,
    eps=1e-5,
    num_threads=128,
    rows_per_cta=2,
)
assert op.check_support()
op.compile()
op.execute(
    x_tensor=x,
    w_tensor=w,
    o_tensor=o,
    amax_tensor=amax,
    current_stream=None,
)
```

The PyTorch class preserves caller-owned output buffers. During
`check_support()`, the framework-neutral operation validates the complete
input and output exemplar signature and resolves the launch configuration.
`compile()` then constructs the CUTE DSL kernel from that resolved
configuration.

### JAX API

The CUDA 13 JAX packages require Python 3.11 or newer. Install the JAX integration explicitly:

```bash
pip install 'nvidia-cudnn-frontend[jax]'
```

```python
import jax
from cudnn.jax import RmsNormRhtAmaxSm100, rmsnorm_rht_amax_sm100

o, amax = rmsnorm_rht_amax_sm100(
    x,
    w,
    eps=1e-5,
    num_threads=128,
    rows_per_cta=2,
)

# The class API specializes from abstract placeholders, then accepts arrays
# when called under JAX transformations.
x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
w_spec = jax.ShapeDtypeStruct(w.shape, w.dtype)
op = RmsNormRhtAmaxSm100(x_spec, w_spec, rows_per_cta=2)
o, amax = jax.jit(op)(x, w)
```

The function API is JIT-compiled and treats `eps`, `num_threads`, and
`rows_per_cta` as static compilation options. Use `RmsNormRhtAmaxSm100`
directly when the caller needs to control JAX transformations explicitly. A
JAX array or another object exposing shape and dtype may be used instead of a
`ShapeDtypeStruct` exemplar; only its metadata is retained. Unlike PyTorch,
JAX derives output descriptors in its adapter and lets XLA allocate the
results. Callers may instead provide both `sample_o` and `sample_amax` as
explicit output placeholders. In either case, the common operation validates
the complete signature through `check_support()`.

By default, JAX support validation requires every local GPU to meet the
operation's SM100 minimum. AOT or remote compilation without a compatible
local target device requires an explicit process-wide opt-out before tracing:

```python
import cudnn.jax

cudnn.jax.disable_device_compatibility_checks(True)
lowered = rmsnorm_rht_amax_sm100.lower(x_spec, w_spec)
```

Pass `False` to re-enable the checks. Changing this setting does not
invalidate existing JAX compilation-cache entries, so configure it before the
first `jax.jit` or `lower` call for a signature.

The current JAX adapter models its buffers as row-major with the public and
kernel axes in the same order. Additional JAX-specific axis mappings can be
added without changing the framework-neutral operation signature.

## Parameters

### PyTorch input and output tensors

- `x_tensor` / `sample_x`
  - Shape: `(M, N)`
  - Layout: row-major contiguous
  - Dtype: `torch.bfloat16`
- `w_tensor` / `sample_w`
  - Shape: `(N,)`
  - Layout: contiguous
  - Dtype: `torch.bfloat16`
- `o_tensor` / `sample_o`
  - Shape: `(M, N)`
  - Layout: row-major contiguous
  - Dtype: `torch.bfloat16`
- `amax_tensor` / `sample_amax`
  - Shape: `(M / rows_per_cta,)`
  - Dtype: `torch.float32`

The JAX API uses the same shapes and layouts with `jax.numpy.bfloat16` for
`X`, `W`, and `O`, and `jax.numpy.float32` for `Amax`.

### Common parameters

- `eps: float`
  - RMSNorm epsilon. Default: `1e-5`
- `num_threads: Optional[int]`
  - Threads per CTA. If omitted, the API uses the upstream-tuned table when possible, otherwise a valid fallback search.
- `rows_per_cta: Optional[int]`
  - Rows processed by each CTA. If omitted, the wrapper uses the upstream-style heuristic over `{2, 4, 8}`.

The PyTorch `execute()` and wrapper APIs additionally accept the optional
CUDA stream argument `current_stream`. JAX execution uses the stream supplied
by XLA.

### Return values

The PyTorch wrapper returns a `TupleDict` with keys:

- `o_tensor`
  - Shape: `(M, N)`
  - Layout: row-major contiguous
  - Dtype: `torch.bfloat16`
- `amax_tensor`
  - Shape: `(M / rows_per_cta,)`
  - Dtype: `torch.float32`

Tuple unpacking order is `(o_tensor, amax_tensor)`.

The class API writes into its caller-provided output tensors and returns
`None`.

## Support surface and constraints

- Requires SM100+.
- `N` must be divisible by `16`.
- `N` must be divisible by the resolved `num_threads`.
- `EPT = N / num_threads` must be at least `8` and divisible by `8`.
- `M` must be divisible by `rows_per_cta`.
- `X`, `W`, and `O` use bf16; `amax` uses float32.
- PyTorch input and output tensors must be 16-byte aligned.
- The frontend integration matches the upstream RMSNorm kernel semantics; it does not expose full LayerNorm mean/bias behavior.

## Verification

Focused correctness and cache coverage live in:

- `test/python/fe_api/test_rmsnorm_rht_amax.py`
- `test/python/fe_api/test_jax_rmsnorm_rht_amax.py`
