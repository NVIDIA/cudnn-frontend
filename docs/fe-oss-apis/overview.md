# FE-OSS APIs Overview 

**FE-OSS APIs are experimental and subject to change.**

This folder documents the Python FE APIs implemented under `python/cudnn`. For details on currently implemented operations, see:
- [GEMM + Amax](gemm_fusions/gemm_amax.md)
- [GEMM + RoPE + MXFP8 Projection](gemm_fusions/gemm_proj_rope_mxfp8.md)
- [GEMM + SwiGLU](gemm_fusions/gemm_swiglu.md)
- [GEMM + sReLU](gemm_fusions/gemm_srelu.md)
- [GEMM + dsReLU](gemm_fusions/gemm_dsrelu.md)
- [Grouped GEMM (BF16)](gemm_fusions/grouped_gemm.md)
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
- [CSA Fused Compressor](csa.md)
- [RMSNorm + RHT + Amax](rmsnorm_rht_amax.md)
- [SDPA Forward FE OSS API (SM100, D=256)](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#sdpa-forward-fe-oss-sm100-d256)
- [SDPA Backward FE OSS API (SM100, D=256)](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#sdpa-backward-fe-oss-sm100-d256)
- [RMSNorm + SiLU](rmsnorm_silu.md)

## Installation and setup

All Frontend OSS APIs come installed with the `nvidia-cudnn-frontend` package. However, each API may require additional optional dependencies defined in the `pyproject.toml` file. For instance, GEMM + Amax, GEMM + SwiGLU, and the grouped GEMM APIs require the `cutedsl` optional dependency, which can be installed via:
```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

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


## File structure and examples

- All FE OSS APIs are implemented in the `python/cudnn` directory.
- Correctness tests/samples are implemented in the `test/python/fe_api` directory.
