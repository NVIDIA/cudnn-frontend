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
├── frontend/                       # Internal target-support and kernel-ownership catalog
├── jax/                            # Optional functional JAX APIs
├── graph.py                        # Low-level graph helpers (graph, jit, graph_cache)
├── wrapper.py                      # High-level Graph wrapper class
├── datatypes.py                    # Data type conversions and helpers
├── api_base.py                     # Abstract API base class for frontend-only APIs
├── {frontend-only-api-name}/
│   ├── __init__.py                 # Frontend-only API class
│   └── api.py                      # High-level API implementation
│   └── {kernel_name}.py            # Kernel implementation, i.e CuteDSL
test/python/                        # Test files
└── fe_api/                         # Test files for frontend-only APIs
```

## Adding new frontend-only APIs

The intended unit is a semantic operation variant, which may own one or more
main/helper CuTe kernels. The existing Torch conventions remain canonical; JAX
is an optional additive namespace. JAX-enabled operations expose an aligned
high-level name in both namespaces while retaining the legacy Torch wrapper:

```python
from cudnn import my_operation
torch_result = my_operation(torch_inputs, ...)

from cudnn.jax import my_operation
jax_result = my_operation(jax_inputs, ...)

from cudnn import my_operation_wrapper  # unchanged compatibility API
```

To add a new frontend-only API:

1. Define the semantic operation: logical inputs and outputs, common option
   names/defaults, shape/dtype/layout inference, support domain, workspace,
   aliasing, and initialization behavior.
2. Add the CuTe implementation. Every `@cute.kernel`, including helper kernels,
   must be owned by the semantic operation's exact `kernel_anchors` entry.
3. Preserve or add the canonical Torch class/wrapper API following existing
   conventions. Add an aligned Torch high-level function whose symbol exactly
   matches the semantic operation and JAX symbol. Do not replace the legacy
   `TupleDict`, stream controls, output-buffer lifecycle, or compatibility
   behavior to make it resemble JAX.
4. Add the functional JAX adapter using `cutlass.jax.cutlass_call`. It must infer
   outputs/workspace from abstract metadata, accept XLA's stream, and avoid
   Torch imports or host reads during tracing.
5. Register the internal semantic contract in `python/cudnn/frontend`. Every
   concrete target binding symbol must exactly equal the semantic operation
   name. The Torch binding is required; JAX must be a `TargetBinding` or explicit
   `TargetGap(reason, tracking_issue)`. Record parameter/output mappings,
   target-only arguments, exact `api_anchors`, and exact `kernel_anchors`.
6. Add common support-domain/numerical parity cases plus Torch and JAX lifecycle
   tests in `test/python/fe_api/`. JAX coverage should include `eval_shape`,
   `jit`, lowering, and execution on supported hardware.
7. Run `test_frontend_target_parity.py`. A new public class/wrapper or physical
   kernel without a semantic owner is a failure. Existing baselines are
   migration debt and must not grow for a new operation.
8. Keep the Torch exports in `cudnn` and expose JAX only from `cudnn.jax`.
   Register JAX dependencies only in the optional extra.

The catalog is for ownership, support reporting, and CI; it is not a public
backend-dispatch facade. Do not use tensor-type dispatch or a traced `target=`
argument. JAX does not emulate the mutable `APIBase.compile()` / `execute()`
lifecycle. JAX is optional for users to install, but declaring its support
status is mandatory for contributors; the normal policy rejects new JAX gaps.

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
