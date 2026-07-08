# JAX support for cuDNN Frontend OSS APIs

Status: 31 of 33 public FE-OSS entry points have source-level JAX bindings;
representative hardware tests currently cover 11 of those 31 entry points.

## Decision

JAX is an optional, explicit API surface under `cudnn.jax`. The existing
PyTorch operation API remains first class: unqualified `cudnn` and
operation-package symbols continue to mean PyTorch, regardless of which
frameworks are installed. Compatibility of lower-level public classes still
needs one correction before release, described below.

Both bindings use the same CuTe DSL kernels and share framework-neutral tensor
metadata and validation where the contracts genuinely match. They do not share
an execution lifecycle. PyTorch retains its explicit compile/execute flow; JAX
returns ordinary, un-jitted callables that lower through
`cutlass.jax.cutlass_call` when traced by JAX.

This is the right boundary because a JAX tracer describes an abstract value,
not a concrete device allocation. XLA must own output allocation, workspace
lifetime, stream ordering, and buffer aliasing.

## User and package model

JAX users install the optional dependency and opt into its namespace:

```bash
pip install 'nvidia-cudnn-frontend[jax]'
```

```python
import jax
import cudnn
import cudnn.jax

# Existing PyTorch API; its meaning never depends on whether JAX is installed.
torch_result = cudnn.rmsnorm_rht_amax_sm100(torch_x, torch_weight)

@jax.jit
def f(x, weight):
    return cudnn.jax.rmsnorm_rht_amax_sm100(x, weight)
```

Each operation co-locates its bindings:

```text
cudnn/<operation>/
  api.py       # PyTorch API
  jax.py       # JAX API
  kernel.py    # shared CuTe DSL implementation
```

Operation packages lazily expose `.jax`, while their unqualified exports come
from `api.py`. Importing `cudnn` does not import JAX, CUTLASS, or PyTorch
eagerly. Importing `cudnn.jax` is the JAX dependency boundary: it checks for
JAX and CUTLASS, verifies `cutlass.jax.is_available()`, and gives an install
hint when the integration is unavailable. Before release, an operation-local
`.jax` import must route through the same check; today it can bypass the
friendly compatibility diagnostic.

Comparable operation names and concepts are preferred, but exact signatures
are not required. Defaults, supported layouts, result containers, and
framework-specific controls may differ. A new PyTorch wrapper is not
automatically blocked on a JAX implementation, but review should identify the
gap and document whether it is intentional.

## Execution architecture

| Concern | PyTorch | JAX |
| --- | --- | --- |
| Inputs | Concrete tensors with observable strides and devices | Arrays or tracers with abstract shape and dtype |
| Compilation | Explicit `cute.compile()` lifecycle | CuTe compilation during XLA lowering |
| Execution | Python invokes the compiled callable | XLA invokes a typed custom call |
| Outputs | Allocated by the wrapper | Declared by shape/dtype; allocated by XLA |
| Stream | Torch/CUTLASS stream | Supplied by XLA/PJRT |
| Workspace | May be cached on the API object | XLA-owned, per-invocation hidden result |
| Mutation | Preallocated outputs are written | Public API is functional; aliases are declared |

`cutlass.jax.cutlass_call` is the integration seam. During lowering it builds
a CUTLASS function specification from abstract inputs, output metadata, tensor
layouts, aliases, compile options, and static arguments. It compiles the CuTe
launcher and embeds the resulting object in a typed XLA FFI custom call. At
runtime, the custom call receives XLA-owned buffers and XLA's CUDA stream;
Python is not on the execution path.

NSA sliding-window attention is the one non-CuTe exception. Its fixed-shape
inference binding delegates to JAX's public `dot_product_attention` cuDNN
implementation, which lowers to JAX's registered cuDNN custom call. The
frontend does not invoke its PyTorch/pygraph execution path while tracing.

`call_cutedsl` preserves CUTLASS's `allow_cuda_graph=True` default. This permits
the graph-capable FFI target but does not force capture; XLA decides whether to
use it. The frontend only needs to opt out if an operation introduces a known
non-capturable runtime effect outside the ordinary stream-ordered launch path.

The frontend's `call_cutedsl` adapter gives every kernel one canonical ABI:

```text
launcher(stream, *inputs, *outputs, *workspaces, **static_args)
```

Public outputs and hidden workspaces are described by `BufferSpec`. A
`fill_value` requests XLA-visible initialization and an input/output alias;
`None` means the kernel fully overwrites the buffer. Workspace shapes must be
derivable from abstract metadata. No tensor address, stream, output buffer, or
workspace is retained by a JAX API object.

## Shared metadata and validation

The base classes deliberately separate common contracts from framework policy:

- `TensorDesc` contains logical shape, dtype, packing, and kernel-visible
  layout metadata without importing a framework.
- `TorchTensorDesc` adds observed strides, device, view behavior, and packed
  storage rules.
- `JaxTensorDesc` adds the native `cutlass.jax.TensorSpec` used to declare
  compact physical layout and logical mode order to XLA. It never reads a
  physical JAX buffer stride.
- `ApiBase` contains framework-neutral validation helpers.
- `ApiBaseTorch` owns support checks, compilation, execution, allocation, and
  streams. The historical `APIBase` name remains its compatibility alias.
- `ApiBaseJax` owns sample-signature validation and is itself a stable,
  un-jitted callable. `get_jax_callable()` returns that same object.

JAX operation classes accept sample array-like values in their constructors,
immediately convert them to descriptors, and retain no arrays or tracers. Their
calls accept the real operands and verify that shape, dtype, and optional
presence match the validated sample signature. Applications retain control of
`jax.jit`, donation, sharding, and device placement.

Rank, logical shape and dtype, packing, divisibility, tile domains, and output
inference should be shared when they describe the same kernel path. Physical
Torch strides and devices, JAX `TensorSpec` declarations, framework streams,
allocation, and lowering remain adapter-specific. Validators should return
immutable, kernel-specific plans rather than branch on a framework flag.

For GEMM, a Torch tensor's observed strides determine M-major, N-major, or
K-major interpretation. JAX has no equivalent user-visible stride contract.
JAX GEMM APIs instead use compact row-major public axis-order strings with the
batch or expert mode `L` outermost: A uses `LMK` or `LKM`, B uses `LNK` or
`LKN`, and C/D use `LMN` or `LNM` where the selected kernel supports those
orders. `TensorSpec.mode` maps the public axes to the kernel's canonical
`(M,K,L)`, `(N,K,L)`, and `(M,N,L)` modes, while `TensorSpec.layout` declares
the physical XLA layout at the custom-call boundary. XLA may insert a layout
conversion to satisfy that constraint.

| Operand | Public order and shape | `TensorSpec.mode` | Contiguous kernel mode |
| --- | --- | --- | --- |
| A | `LMK`: `(L, M, K)` | `(1, 2, 0)` | K |
| A | `LKM`: `(L, K, M)` | `(2, 1, 0)` | M |
| B | `LNK`: `(L, N, K)` | `(1, 2, 0)` | K |
| B | `LKN`: `(L, K, N)` | `(2, 1, 0)` | N |
| C/D | `LMN`: `(L, M, N)` | `(1, 2, 0)` | N |
| C/D | `LNM`: `(L, N, M)` | `(2, 1, 0)` | M |

The current grouped-GEMM bindings fix A to `LMK` and matrix outputs to `LMN`.
Only quant, sReLU, dsReLU, and dGLU accept both `LNK` and `LKN` for B; the
SwiGLU, dSwiGLU, GLU, and GLU + Hadamard paths use `LNK`. Grouped wgrad keeps
its separate two-dimensional input contract. These layout strings apply to
matrix operands and results. Packed scale tensors, probabilities, reductions,
and other auxiliary buffers retain their operator-specific layouts.

## Coverage and limits

| Family | PyTorch wrappers | JAX wrappers |
| --- | ---: | ---: |
| RMSNorm + RHT + amax | 1 | 1 |
| Dense GEMM fusions | 4 | 4 |
| Grouped GEMM | 9 | 9 |
| Discrete grouped GEMM | 2 | 0 |
| SDPA | 2 | 2 |
| Native Sparse Attention | 4 | 4 |
| DeepSeek Sparse Attention | 11 | 11 |
| **Total** | **33** | **31** |

The two current gaps need separate ABI work, not another generic adapter:

1. Two discrete grouped-GEMM wrappers consume device-resident tables of raw
   addresses. JAX must receive the expert buffers as real operands and build
   pointer tables in XLA-owned device workspace.

The sliding-window binding deliberately covers fixed-shape BHSD inference with
`right_bound=0`. Training statistics and packed THD/ragged inputs still need a
native frontend XLA FFI adapter; depending on JAX's private cuDNN APIs would be
an unstable OSS contract.

Dense DSA indexer backward now uses a runtime `grad_loss` operand, an
XLA-owned score-gradient workspace, and a kernel-cleared FP32 accumulation
buffer. Its first JAX binding deliberately covers fixed-shape SM100 BSHD
inputs; the broader PyTorch SM90 and packed-THD paths remain unchanged.

Only RMSNorm, the four dense GEMMs, five DSA entry points, and NSA
sliding-window inference currently have real GPU execution tests. These tests
include numerical reference coverage for dense indexer backward across multiple
query and key tiles, runtime loss scaling, and the registered cuDNN
sliding-window lowering. The other families have source and CPU contract
coverage but have not been qualified through real CUTLASS lowering and SM100
execution.

Current JAX bindings require concrete shapes during tracing. They do not define
autodiff, `vmap`, or automatic SPMD partitioning rules. SDPA backward is an
explicit operation, not an autodiff rule. Operator-specific `shard_map` or
custom partitioning can be added only after local shapes, communication, and
workspace semantics are defined. Heterogeneous-GPU processes also remain to be
qualified.

## OSS readiness review

The architecture is appropriate for an open-source API, but four items should
block release:

1. **Publish and enforce a compatibility matrix.** The base package advertises
   Python 3.9+, but the new base descriptors use Python 3.10-only
   `dataclass(kw_only=True)`, and the JAX dependency requires Python 3.11+.
   The normal test requirements also pin NumPy below 2 while JAX 0.9.1 requires
   NumPy 2. Split the JAX environment, document Linux/CUDA 13 support, and test
   the minimum and current qualified JAX versions with CUTLASS 4.5.0.
   Installation tests must build a wheel, install only `[jax]` in a Torch-free
   environment, import `cudnn.jax`, and lower a representative operation.
2. **Add required SM100 GPU CI.** CPU contract tests and fake CUTLASS modules
   are useful but cannot validate generated code, workspace lifetime, layout
   conversion, stream use, or numerical correctness. CI should cover every
   family and representative initialized-output, hidden-workspace,
   multi-launch, and repeated-call paths. The checked-in wrapper count should
   be generated or tested rather than maintained only in prose.
3. **Make runtime metadata safe.** Grouped-GEMM offsets and sparse-attention
   counts/indices are documented as trusted values, but kernels use them for
   device indexing without complete bounds checks. Add device-side validation
   or guards, or define an explicit unsafe precondition with precise behavior
   and negative tests. Invalid public inputs must not silently permit
   out-of-bounds GPU access.
4. **Resolve the `TensorDesc` compatibility break.** This branch repurposes the
   previously Torch-specific public name as a keyword-only neutral descriptor
   and moves its old constructor and tensor-like behavior to
   `TorchTensorDesc`. Keep a deprecated Torch-compatible spelling or make the
   break explicit through versioning and migration documentation. Preserving
   only the historical `APIBase` alias is not sufficient.

Release and legal owners should also confirm the distribution terms for the
optional proprietary CuTe DSL dependency; the frontend source is MIT, but that
does not make the complete runtime stack MIT.

The following are important follow-ups, but need not block the first release if
their limits are explicit:

- Extract the remaining dual-purpose `_validate_only` functions into pure
  validators that return frozen plans. Revalidating in constructors and calls
  is a drift risk across 19 operation modules.
- Ensure every `JaxTensorDesc` or frozen plan contains the exact `TensorSpec`
  used for lowering. Some SDPA sample descriptors currently use defaults while
  the call path supplies a nontrivial layout.
- Define whether public `JaxTensorDesc.tensor_spec` is a stable API or an
  advanced escape hatch coupled to CUTLASS. Its compatibility policy should be
  tested.
- Coordinate with CUTLASS on its unbounded compilation cache and whether the
  target architecture participates in the key. Until then, document
  heterogeneous-architecture processes as unsupported.
- Add HLO and performance tests for declared GEMM layouts so hidden XLA
  transposes do not erase kernel gains.
- Publish a per-operation support matrix for shapes, dtypes, layouts,
  transforms, and architecture restrictions. Name parity alone does not imply
  domain parity.
- Make `TupleDict` immutable or keep its cached key order synchronized after
  dictionary mutation; its current tuple iteration can become stale.

## Recommendation

Keep the explicit `cudnn.jax` namespace, co-located `api.py`/`jax.py`
bindings, framework-neutral descriptors, and separate runtime adapters. Do not
unify PyTorch and JAX compile/execute behavior or dispatch implicitly on array
type. With the packaging matrix and GPU CI gates above, this design is a clear,
maintainable way to expose the same FE-OSS kernels to both frameworks.

## References

- [cuDNN Frontend OSS APIs](https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/fe-oss-apis.html)
- [CUTLASS `cutlass.jax.cutlass_call`](https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/CuTeDSL/cutlass/jax/primitive.py)
- [JAX FFI API](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html)
- [JAX package metadata](https://pypi.org/project/jax/)
