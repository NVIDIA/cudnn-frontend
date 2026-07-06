# FE-OSS APIs Overview 

**FE-OSS APIs are experimental and subject to change.**

Design work for tracing FE-OSS CuTe DSL kernels from JAX is documented in
[CuTe DSL + JAX support for FE-OSS APIs](cutedsl-jax-design.md).

This folder documents the Python FE APIs implemented under `python/cudnn`. For details on currently implemented operations, see:
- [GEMM + Amax](gemm_fusions/gemm_amax.md)
- [GEMM + SwiGLU](gemm_fusions/gemm_swiglu.md)
- [GEMM + sReLU](gemm_fusions/gemm_srelu.md)
- [GEMM + dsReLU](gemm_fusions/gemm_dsrelu.md)
- [Grouped GEMM + GLU (Unified)](gemm_fusions/grouped_gemm_glu.md)
- [Grouped GEMM + GLU + Hadamard](gemm_fusions/grouped_gemm_glu_hadamard.md)
- [Grouped GEMM + dGLU (Unified)](gemm_fusions/grouped_gemm_dglu.md)
- [Grouped GEMM + SwiGLU (Legacy, Contiguous-only)](gemm_fusions/grouped_gemm_swiglu.md)
- [Grouped GEMM + dSwiGLU (Legacy, Contiguous-only)](gemm_fusions/grouped_gemm_dswiglu.md)
- [Grouped GEMM + sReLU (Unified)](gemm_fusions/grouped_gemm_srelu.md)
- [Grouped GEMM + dsReLU (Unified)](gemm_fusions/grouped_gemm_dsrelu.md)
- [Discrete Grouped GEMM + SwiGLU](gemm_fusions/discrete_grouped_gemm_swiglu.md)
- [Discrete Grouped GEMM + dSwiGLU](gemm_fusions/discrete_grouped_gemm_dswiglu.md)
- [Grouped GEMM + Quant (Legacy, Dense-only)](gemm_fusions/grouped_gemm_quant.md)
- [Grouped GEMM + Quant (Unified)](gemm_fusions/grouped_gemm_quant_unified.md)
- [Grouped GEMM + Wgrad](gemm_fusions/grouped_gemm_wgrad.md)
- [Block Sparse Attention (BSA)](bsa.md)
- [Native Sparse Attention (NSA)](nsa.md)
- [RMSNorm + RHT + Amax](rmsnorm_rht_amax.md)
- [SDPA Forward FE OSS API (SM100, D=256)](../operations/Attention.md#sdpa-forward-fe-oss-sm100-d256)
- [SDPA Backward FE OSS API (SM100, D=256)](../operations/Attention.md#sdpa-backward-fe-oss-sm100-d256)
- [RMSNorm + SiLU](rmsnorm_silu.md)

## Installation and setup

All Frontend OSS APIs come installed with the `nvidia-cudnn-frontend` package. However, each API may require additional optional dependencies defined in the `pyproject.toml` file. For instance, GEMM + Amax and GEMM + SwiGLU require the `cute-dsl` optional dependency, which can be installed via:
```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

After installation, you can import the APIs directly from the `cudnn` package, i.e. `from cudnn import {your_operation}`

The JAX integration uses a separate optional dependency set:

```bash
pip install nvidia-cudnn-frontend[jax]
```

This optional dependency set requires Python 3.11 or newer, matching the JAX
CUDA package requirement; the base frontend package retains its broader Python
support.

Importing `cudnn` does not load JAX or CuTe DSL. Importing `cudnn.jax`, or
accessing `cudnn.jax` after importing `cudnn`, is the explicit JAX opt-in. It
validates JAX and CuTe DSL availability and points missing installations to the
`jax` extra, including checking `cutlass.jax.is_available()` and reporting
CUTLASS's minimum supported JAX version when unavailable. It then loads the JAX
operation wrappers and shared CuTe DSL bridge. Architecture-specific kernel
modules remain deferred until an operation is traced. Each implemented
operation keeps `api.py` as its Torch binding and a sibling `jax.py` as its JAX
binding.

## API Usage

PyTorch remains the default FE-OSS interface and preserves its existing
wrappers and classes. Some operations also provide a functional JAX API under
`cudnn.jax`. Each operation-backed JAX wrapper has a callable class with the
aligned Torch class name; the two DSA layout helpers remain functional-only.
Supported bindings also retain aligned functional names across the two
namespaces:

```python
from cudnn import rmsnorm_rht_amax_sm100
from cudnn.jax import rmsnorm_rht_amax_sm100

from cudnn import gemm_swiglu_wrapper_sm100
from cudnn.jax import gemm_swiglu_wrapper_sm100
```

JAX class constructors accept array-like samples, immediately reduce them to
shape/dtype descriptors, and do not retain the sample arrays. Actual arrays are
passed when the object is called. The object is intentionally not pre-jitted,
so applications retain control over JIT, sharding, donation, and placement:

```python
import jax
from cudnn.jax import RmsNormRhtAmaxSm100

op = RmsNormRhtAmaxSm100(
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.ShapeDtypeStruct(weight.shape, weight.dtype),
)
op.check_support()
output, amax = jax.jit(op)(x, weight)
```

Compile-time configuration becomes immutable on the first call because JAX
caches by callable identity. Construct a new operation object to change static
options after tracing.

Where both bindings exist, they should offer comparable operation semantics and
recognizable inputs, options, and results. Exact names, signatures, defaults,
layouts, result containers, lifecycle controls, and supported domains may
differ by framework. JAX does not replace or narrow the PyTorch APIs, and
backend selection is never inferred from array types.

### 1. High-level wrapper

- Single pythonic function call
- Allocates and returns output tensors
- Returns outputs as a `TupleDict` (supports both dictionary-style key access and tuple unpacking)
- No explicit compilation step – internally caches compiled kernels via a simple dictionary lookup
- When to use:
  - Fast prototyping and common cases
  - You want automatic allocation and minimal boilerplate
  - You are okay with the library managing the compiled-kernel cache

```python
from cudnn import {your_operation}_wrapper
result = {your_operation}_wrapper(
    inputs,
    ...,
    config_options,
    ...,
    stream=None,
)

# Dictionary-style access (recommended)
primary_output = result["output_tensor_name"]

# Tuple unpacking (order follows documented wrapper output keys)
out0, out1 = result
```

### 2. Class API

- Explicit lifecycle with compile and execute steps
- Reusable object with underlying compiled kernel for multiple executions
- Requires preallocated output tensors
- When to use:
  - You need to reuse a compiled kernel across many calls
  - You want explicit control over compilation and lifecycle management
  
```python
from cudnn import {your_operation}

op = {your_operation}(
    sample_inputs,
    ...,
    sample_outputs,
    ...,
    config_options,
    ...
)
op.compile()
op.execute(
    inputs,
    ...
    outputs,
    ...
    current_stream=None,
)
```
Methods:
- `check_support()` – validates target problem configuration (i.e. tensor shapes, tensor strides, dtypes, tiling/cluster/kernel configurations, environment, etc.)
- `compile()` – compiles the kernel with the provided sample tensors and parameters.
- `execute(inputs, ..., outputs, ..., current_stream)` – runs the kernel with the provided inputs and outputs.
  
## Common Parameters and Conventions

- CUDA stream (`current_stream` in class API, `stream` in wrapper)
  - The cuda stream to use for operation kernel execution.
  - Default: None (uses default stream)


## File structure and examples

- All FE OSS APIs are implemented in the `python/cudnn` directory.
- Correctness tests/samples are implemented in the `test/python/fe_api` directory.
