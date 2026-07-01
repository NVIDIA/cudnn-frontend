# cuDNN Python Frontend

This folder exposes the Python Frontend Graph APIs and the high-level Graph wrapper, along with several frontend-only, ready-to-use APIs.

- **Graph API**: Low-level primitives for building, compiling, and executing cuDNN operation graphs in Python.
- **Graph Wrapper (`Graph`)**: A convenience layer that reduces boilerplate, manages workspace and tensor mapping, and makes execution ergonomic.
- **Frontend-only APIs**: Individual turnkey kernels with Python-first APIs

## Directory Structure

A simplified view of package structure:

```
pyproject.toml                       # Project metadata and dependencies. Optional dependencies for frontend-only APIs are registered here.
python/cudnn/
├── __init__.py                     # Top-level exports (Graph, graph, jit, wrappers, kernels)
├── jax/                            # Lazy JAX facade and shared CuTe/JAX adapter
├── graph.py                        # Low-level graph helpers (graph, jit, graph_cache)
├── wrapper.py                      # High-level Graph wrapper class
├── datatypes.py                    # Data type conversions and helpers
├── api_base.py                     # Abstract API base class for frontend-only APIs
├── {frontend-only-api-name}/
│   ├── __init__.py                 # Torch-first lazy framework selection
│   ├── api.py                      # PyTorch API implementation
│   ├── jax.py                      # Optional JAX functional API
│   └── {kernel_name}.py            # Kernel implementation, i.e CuteDSL
test/python/                        # Test files
└── fe_api/                         # Test files for frontend-only APIs
```

## Adding new frontend-only APIs

The review unit is a user-visible operation, which may use one or more
main/helper CuTe kernels. PyTorch remains the default interface; JAX is an
optional functional namespace. When both bindings exist, they should implement
comparable functionality on their documented overlapping domain and use
familiar terminology where practical:

```python
from cudnn import my_operation_wrapper
torch_result = my_operation_wrapper(torch_inputs, ...)

from cudnn.jax import my_operation
jax_result = my_operation(jax_inputs, ...)
```

To add a new frontend-only API:

1. Document logical inputs and outputs, supported shapes/dtypes/layouts,
   workspace, aliasing, initialization, and transformation behavior.
2. Add the CuTe implementation and preserve the existing PyTorch class/wrapper
   conventions and compatibility behavior.
3. If JAX support is in scope, add a functional adapter using
   `cutlass.jax.cutlass_call`. It must infer outputs/workspace from abstract
   metadata, accept XLA's stream, and avoid Torch imports or host reads during
   tracing.
4. Prefer recognizable operation, operand, option, and result names across
   frameworks, but document intentional differences. Exact Python names,
   signatures, defaults, layouts, result containers, and supported domains are
   not required to match.
5. Test each framework's lifecycle and compare numerical behavior on the domain
   they share. JAX coverage should include `eval_shape`, `jit`, lowering, and
   execution on supported hardware.
6. During review, consider whether a new or modified PyTorch operation should
   update JAX. Static lint or LLM review may report likely gaps or drift, but is
   advisory rather than a public API contract or merge gate.
7. Keep `api.py` as the PyTorch implementation and place an optional JAX
   implementation in `jax.py` beside it. The operation package selects Torch
   when available and falls back to JAX only in a JAX-only installation;
   `cudnn.jax` remains the deterministic JAX facade. Register JAX dependencies
   only in the optional extra.

Do not use tensor-type dispatch or a traced `target=` argument. JAX does not
emulate the mutable `APIBase.compile()` / `execute()` lifecycle, and adding or
updating a PyTorch operation does not require a JAX binding.

**Currently implemented frontend-only APIs**:
- `GEMM + Amax`
- `RMSNorm + RHT + Amax`
- `GEMM + SwiGLU`
- `GEMM + sReLU`
- `GEMM + dsReLU`
- `Grouped Gemm + GLU (Unified)`
- `Grouped Gemm + GLU + Hadamard`
- `Grouped Gemm + dGLU (Unified)`
- `Grouped Gemm + SwiGLU (Legacy, Contiguous-only)`
- `Grouped Gemm + dSwiglu (Legacy, Contiguous-only)`
- `Grouped Gemm + sReLU (Contiguous-only)`
- `Grouped Gemm + dsReLU (Contiguous-only)`
- `Discrete Grouped Gemm + SwiGLU`
- `Discrete Grouped Gemm + dSwiglu`
- `Grouped Gemm + Quant (Legacy, Dense-only)`
- `Grouped Gemm + Quant (Unified)`
- `Grouped Gemm + Wgrad`
- `SDPA Forward (SM100, D=256)`
- `SDPA Backward (SM100, D=256)`

**In progress frontend-only APIs**:
- GEMM + Dswiglu
- GEMM + RoPE
- Native Sparse Attention (NSA)

## Discrete grouped API notes

The discrete grouped APIs (`DiscreteGroupedGemmSwigluSm100` and `DiscreteGroupedGemmDswigluSm100`) use per-expert pointer arrays instead of a packed `B` tensor:

- Runtime pointer inputs are CUDA `torch.int64` tensors (`b_ptrs`, `sfb_ptrs`) with shape `(num_experts,)`.
- `compile()` is no-arg and compiles from descriptors captured in the constructor.
- For CUDA graph capture, call `compile()` before capture and capture only `execute()` with preallocated tensors.
