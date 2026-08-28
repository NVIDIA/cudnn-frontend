# FE-OSS APIs Overview 

**FE-OSS APIs are experimental and subject to change.**

The GEMM CuTeDSL APIs are type-erased and torch-lazy: torch is imported only when torch tensors are passed. JAX arrays are additionally accepted wherever the kernel's tensor layouts are expressible as row-major arrays (each API's page has a "JAX support" section with its exact contract):

- **Dense fusions** (amax, swiglu, srelu, dsrelu): full JAX eager support, plus `jax.jit`-compatible XLA custom-call entry points for all four (built on `cudnn.jax.call` / CuTeDSL's native `cutlass.jax` bridge; see `gemm_amax.md` "Using JAX arrays").
- **Grouped / discrete-grouped**: JAX eager support in discrete (pointer-array) weight modes — unfused grouped GEMM, glu/dglu (BF16), dsrelu (FP8), wgrad (BF16), and discrete-grouped swiglu/dswiglu (FP8) — plus a `jax.jit`-compatible `*_jax_sm100` entry point for each of those same families (built on `cudnn.jax.call`; each API page documents its exact jit contract). Dense weight mode, column-major bias layouts, and kernels whose scale factors are MMA-permuted tensor arguments (grouped swiglu/srelu/quant/dswiglu, glu_hadamard, block-scaled glu/dglu/wgrad backends) reject JAX with clear errors.
- **proj_rope_mxfp8**: JAX eager support on both input paths with `w_out_in=True` (the transposed [in, out] weight view is torch-only), plus the `jax.jit`-compatible `gemm_proj_rope_mxfp8_jax_sm100` entry point.

This folder documents the Python FE APIs implemented under `python/cudnn`. For details on currently implemented operations, see:
- [FLA Integration Shims](fla.md)
- [GEMM + Amax](gemm_fusions/gemm_amax.md)
- [GEMM + RoPE + MXFP8 Projection](gemm_fusions/gemm_proj_rope_mxfp8.md)
- [GEMM + SwiGLU](gemm_fusions/gemm_swiglu.md)
- [GEMM + sReLU](gemm_fusions/gemm_srelu.md)
- [GEMM + dsReLU](gemm_fusions/gemm_dsrelu.md)
- [Grouped GEMM (BF16)](gemm_fusions/grouped_gemm.md)
- [Grouped GEMM + GLU (Unified)](gemm_fusions/grouped_gemm_glu.md)
- [Grouped GEMM + GLU + Hadamard](gemm_fusions/grouped_gemm_glu_hadamard.md)
- [Grouped GEMM + GLU + Hadamard + Quant](gemm_fusions/grouped_gemm_glu_hadamard_quant.md)
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
- [HSTU Attention (Blackwell SM100/SM103)](attention/hstu.md)
- [Native Sparse Attention (NSA)](nsa.md)
- [CSA Fused Compressor](csa.md)
- [RMSNorm + RHT + Amax](rmsnorm_rht_amax.md)
- [SDPA Backward (SM120)](attention/sdpa_bwd_sm120.md)
- [RMSNorm + SiLU](rmsnorm_silu.md)

## Installation and setup

All Frontend OSS APIs come installed with the `nvidia-cudnn-frontend` package. However, each API may require additional optional dependencies defined in the `pyproject.toml` file. For instance, GEMM + Amax, GEMM + SwiGLU, and the grouped GEMM APIs require the `cutedsl` optional dependency, which can be installed via:
```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

The `cutedsl` extra is framework-neutral (nvidia-cutlass-dsl, cuda-python, apache-tvm-ffi). Install your tensor framework separately — from a checkout, the PEP 735 dependency groups pin the right companion packages:
```bash
pip install --group torch   # torch + torch-c-dlpack-ext
pip install --group jax     # jax >= 0.5 (XLA entry points via cutlass.jax, shipped with nvidia-cutlass-dsl)
```
(For the published wheel, `pip install torch torch-c-dlpack-ext` or `pip install "jax>=0.5"` directly.)

After installation, you can import the APIs directly from the `cudnn` package, i.e. `from cudnn import {your_operation}`

## API Usage

Each operation exposes two APIs:

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

- Determinism
  - Several kernels reduce with cross-CTA atomics, so those outputs are not bit-exact run to
    run. Kernels react to `torch.use_deterministic_algorithms(True)`: they switch to a
    deterministic path where one exists, and otherwise raise (or warn, under
    `warn_only`) rather than silently return a non-reproducible result.
  - Where a deterministic path exists it can also be selected per call with
    `deterministic=True`, independent of the torch setting. See
    [Grouped GEMM + dsReLU](gemm_fusions/grouped_gemm_dsrelu.md#deterministic-dprob).


## File structure and examples

- All FE OSS APIs are implemented in the `python/cudnn` directory.
- Correctness tests/samples are implemented in the `test/python/fe_api` directory.
