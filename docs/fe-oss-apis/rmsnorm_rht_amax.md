# RMSNorm + RHT + Amax (SM100)

**This is an experimental API and subject to change.**

## Overview

**RMSNorm + RHT + amax**: A fused CUTE DSL kernel for NVIDIA Blackwell GPUs (SM100+) that applies RMS normalization, a block-diagonal Hadamard transform with fixed block size `16`, and a per-CTA `amax` reduction.

This frontend integration exposes the kernel as a standard FE-OSS Python API with:
- a class API (`RmsNormRhtAmaxSm100`)
- a PyTorch functional API (`cudnn.rmsnorm_rht_amax_sm100`)
- the existing wrapper API (`rmsnorm_rht_amax_wrapper_sm100`)
- a JAX class API (`cudnn.jax.RmsNormRhtAmaxSm100`)
- a JAX functional API (`cudnn.jax.rmsnorm_rht_amax_sm100`)
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

### Comparable framework APIs

Both bindings implement the same forward computation and expose comparable
inputs, options, and result roles. This operation currently uses the same
function name in both namespaces; return types, lifecycle, supported layouts,
and stream handling remain framework-specific.

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

Install the CuTe DSL dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

```python
from cudnn import rmsnorm_rht_amax_sm100

result = rmsnorm_rht_amax_sm100(
    x,
    weight,
    eps=1e-5,
    num_threads=None,
    rows_per_cta=None,
    current_stream=None,
)

output, amax = result
```

The PyTorch functional API returns `TupleDict(output=..., amax=...)`.

`````

`````{tab-item} JAX
:sync: jax

Install the JAX-specific optional dependencies:

```bash
pip install nvidia-cudnn-frontend[jax]
```

The JAX optional dependency set requires Python 3.11 or newer.

```python
import jax
from cudnn.jax import rmsnorm_rht_amax_sm100

eps = 1e-5
num_threads = 128
rows_per_cta = 2

@jax.jit
def run(x, weight):
    return rmsnorm_rht_amax_sm100(
        x,
        weight,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )

output, amax = run(x, weight)
```

The JAX function returns `TupleDict(output=..., amax=...)`, registered as a
JAX pytree with the same key and unpacking behavior as the PyTorch result.

The JAX class API mirrors the sample-signature style without retaining sample
arrays:

```python
from cudnn.jax import RmsNormRhtAmaxSm100

op = RmsNormRhtAmaxSm100(
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.ShapeDtypeStruct(weight.shape, weight.dtype),
    eps=eps,
    num_threads=num_threads,
    rows_per_cta=rows_per_cta,
)
assert op.check_support()
run = jax.jit(op)
output, amax = run(x, weight)
```

JAX `check_support()` validates sample shape, dtype, and static launch
configuration. The final SM100 device check belongs to lowering/runtime because
abstract samples such as `ShapeDtypeStruct` do not have a device placement.

`````

``````

The JAX API requires concrete `M` and `N` during tracing and does
not yet define autodiff, `vmap`, or automatic partitioning rules. `eps`,
`num_threads`, and `rows_per_cta` are static compilation state; close them over
as above or list them in `jax.jit(static_argnames=...)`. JAX always uses XLA's
runtime stream; Torch optionally accepts `current_stream`. See
[CuTe DSL + JAX support for FE-OSS APIs](cutedsl-jax-design.md) for the design,
workspace model, rollout plan, and current limitations.

### Additional PyTorch APIs

The functional API above is additive. The existing PyTorch wrapper and class
APIs remain first class and retain their original contracts.

#### Existing high-level wrapper

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

#### Class API

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

## Parameters

### Input and output tensors

- `x` / `x_tensor` / `sample_x`
  - Shape: `(M, N)`
  - Layout: row-major contiguous
  - Dtype: bfloat16 (`torch.bfloat16` or `jax.numpy.bfloat16`)
- `weight` / `w_tensor` / `sample_w`
  - Shape: `(N,)`
  - Layout: contiguous
  - Dtype: bfloat16 (`torch.bfloat16` or `jax.numpy.bfloat16`)
- `output` / `o_tensor` / `sample_o`
  - Shape: `(M, N)`
  - Layout: row-major contiguous
  - Dtype: bfloat16 (`torch.bfloat16` or `jax.numpy.bfloat16`)
- `amax` / `amax_tensor` / `sample_amax`
  - Shape: `(M / rows_per_cta,)`
  - Dtype: float32 (`torch.float32` or `jax.numpy.float32`)

### Common parameters

- `eps: float`
  - RMSNorm epsilon. Default: `1e-5`
- `num_threads: Optional[int]`
  - Threads per CTA. If omitted, the API uses the upstream-tuned table when possible, otherwise a valid fallback search.
- `rows_per_cta: Optional[int]`
  - Rows processed by each CTA. If omitted, the wrapper uses the upstream-style heuristic over `{2, 4, 8}`.
- Torch-only CUDA stream (`current_stream`); JAX always uses XLA's stream

### Return values

The PyTorch functional API returns a `TupleDict` with keys:
- `output`
- `amax`

The JAX functional API returns a `TupleDict` with keys `output` and `amax`.

The legacy wrapper returns a `TupleDict` with keys:
- `o_tensor`
- `amax_tensor`

Tuple unpacking follows each key order.

## Support surface and constraints

- Requires SM100+.
- `N` must be divisible by `16`.
- `N` must be divisible by the resolved `num_threads`.
- `EPT = N / num_threads` must be at least `8` and divisible by `8`.
- `M` must be divisible by `rows_per_cta`.
- Inputs and output are currently bf16 only.
- The JAX API requires concrete shapes and compact row-major inputs.
- The frontend integration matches the upstream RMSNorm kernel semantics; it does not expose full LayerNorm mean/bias behavior.

## Verification

Focused correctness and cache coverage live in:
- `test/python/fe_api/test_rmsnorm_rht_amax.py`
