# CuTe DSL + JAX support for FE-OSS APIs

Status: design proposal with a proof of concept. The public API and internal
interfaces described here are experimental.

## Executive decision

Do not make the existing `APIBase.compile()` / `execute()` lifecycle accept JAX
arrays, and do not introduce a second replacement Torch API. The existing
top-level Torch classes and wrappers remain canonical and backward compatible.
Add JAX as an optional functional namespace under `cudnn.jax`, lowering CuTe DSL
launchers through `cutlass.jax.cutlass_call`.

An internal framework-neutral catalog connects those intentionally different
public APIs. It records semantic parameter/output mappings, exact API and kernel
ownership, and whether JAX is implemented or is an explicit tracked gap.

The distinction matters because a JAX value encountered during tracing is an
abstract value, not a device buffer:

| Concern | Current Torch FE-OSS binding | JAX binding |
| --- | --- | --- |
| Inputs | Concrete `torch.Tensor` objects | JAX arrays or tracers with abstract shape/dtype |
| Outputs | Allocated in Python with `torch.empty*` | Declared with `ShapeDtypeStruct`; XLA allocates them |
| Kernel compilation | Explicit `cute.compile()` before execution | CuTe compilation during XLA lowering inside `cutlass_call` |
| Execution | Python invokes a compiled callable | XLA invokes an embedded custom call without a Python callback |
| CUDA stream | Caller/default Torch or CUTLASS stream | Runtime stream supplied by XLA/PJRT |
| Workspace | Often a mutable tensor cached on the API object | Per-invocation XLA-owned temporary result |
| Mutation | Preallocated outputs are mutated | Public API remains functional; legal aliases are declared |
| Cache | Python object and compiled callable | JAX executable cache plus CUTLASS function/spec cache |

The proof of concept implements this split and ports one real FE-OSS operation,
RMSNorm + RHT + amax, without changing its existing Torch API. It also adds a
dependency-free ownership audit so a new public API or physical CuTe kernel
cannot silently omit its JAX support status.

## Scope

### Goals

- Run FE-OSS CuTe DSL kernels inside `jax.jit` without host callbacks.
- Preserve XLA's stream ordering, allocation, liveness, and functional
  semantics.
- Share operator configuration, shape inference, and kernel source between
  Torch and JAX.
- Model workspace, output initialization, and legal input/output aliases.
- Establish explicit policies for autodiff, batching, sharding, dynamic shapes,
  CUDA graphs, and caching.
- Avoid making Torch a dependency of the JAX-only path.

### Non-goals for the first production increment

- Making the existing class API traceable.
- Supporting every FE-OSS operator immediately.
- Arbitrary Torch-style strided JAX views.
- Data-dependent output or workspace sizes.
- Automatic gradients, `vmap`, or automatic SPMD partitioning for every
  operator.
- A JAX binding for the C++ cuDNN graph API. That is a related but separate
  adapter described later.

## Sources reviewed

The review started from the published
[Frontend OSS API guide](https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/fe-oss-apis.html)
and covered the local implementation under `python/cudnn`, its tests, and
representative operators with and without workspace. The main coupling points
are:

- [`api_base.py`](../../python/cudnn/api_base.py): `TensorDesc` stores
  `torch.dtype` and `torch.device`; dtype checks and tensor conversion are
  Torch-specific.
- [`datatypes.py`](../../python/cudnn/datatypes.py): framework mappings only
  cover Torch today.
- [`gemm_amax/api.py`](../../python/cudnn/gemm_amax/api.py) and
  [`sdpa/fwd/api.py`](../../python/cudnn/sdpa/fwd/api.py): compile fake CuTe
  tensors with a fake stream, then execute against preallocated Torch tensors.
- [`sdpa/bwd/api.py`](../../python/cudnn/sdpa/bwd/api.py) and grouped GEMM APIs:
  retain mutable workspaces on cached API objects.
- [`rmsnorm_rht_amax/api.py`](../../python/cudnn/rmsnorm_rht_amax/api.py): a
  small, single-launch operator suitable for the first vertical slice.

The CUTLASS package pinned by this repository is `nvidia-cutlass-dsl==4.5.0`.
The review used tag
[`v4.5.0`](https://github.com/NVIDIA/cutlass/tree/v4.5.0) at commit
`e406c186f510a15091cce01f782020ceb7ba8eb5`, plus current `main` at commit
[`e8ecfad`](https://github.com/NVIDIA/cutlass/commit/e8ecfad75b44d1ad56264f5001d877e9e47fe080)
to identify subsequent API changes. Relevant sources are:

- [`cutlass.jax.cutlass_call`](https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/CuTeDSL/cutlass/jax/primitive.py)
- [CuTe compilation and cache](https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/CuTeDSL/cutlass/jax/compile.py)
- [JAX tensor/layout adapter](https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/CuTeDSL/cutlass/jax/types.py)
- [basic, layout, and alias examples](https://github.com/NVIDIA/cutlass/blob/v4.5.0/examples/python/CuTeDSL/dsl_tutorials/jax/cutlass_call_basic.py)
- [symbolic export example](https://github.com/NVIDIA/cutlass/blob/v4.5.0/examples/python/CuTeDSL/dsl_tutorials/jax/cutlass_call_export.py)
- [`shard_map` and custom partitioning example](https://github.com/NVIDIA/cutlass/blob/v4.5.0/examples/python/CuTeDSL/dsl_tutorials/jax/cutlass_call_sharding.py)

The `python/CuTeDSL/cutlass/jax` implementation files carry NVIDIA's
`LicenseRef-NvidiaProprietary` header even though they are visible in the public
repository. They are architecture references only; no source is copied into
this OSS proof of concept. CUTLASS's example files use BSD-3-Clause.

The JAX-side behavior was checked against the official
[CuTe DSL + JAX guide](https://docs.jax.dev/en/latest/notebooks/cute_dsl_jax.html),
[FFI guide](https://docs.jax.dev/en/latest/ffi.html), and
[`jax.ffi.ffi_call` API](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html).

## What `cutlass_call` provides

`cutlass_call` is the correct CuTe DSL integration seam. It already supplies the
custom JAX primitive needed to compile Python CuTe code during lowering:

1. Normal JAX tracing flattens input pytrees and records output abstract values.
2. The CUDA lowering builds a CUTLASS function specification from abstract
   shapes, dtypes, layouts, aliases, compile flags, and static keyword arguments.
3. It calls `cute.compile` with fake JAX descriptors and a placeholder stream.
4. It emits a position-independent object containing the host launch wrapper and
   device code, then embeds those bytes in a typed XLA FFI custom call.
5. At execution, the CUTLASS runtime invokes that object with XLA-owned buffers
   and XLA's CUDA stream. Python is not on the execution path.

The placeholder stream used during compilation is not the runtime stream. FE
launchers must accept the stream as their first argument and pass it to every
helper and main-kernel launch.

The bridge supports explicit output metadata, compact physical layouts, logical
mode permutations, input/output aliases, static or runtime tensor dimensions,
CUDA-graph opt-out, and `jax.export`. It deliberately rejects autodiff and
`vmap`; sharding needs `shard_map` or an operator-specific partitioner.

## Proposed architecture

### Parity by construction

The unit of API parity is a **semantic operation variant**, not an individual
`@cute.kernel`. A forward operation may own a main kernel, initialization
helpers, and architecture-specific implementations while still presenting one
Torch/JAX contract. Conversely, a public helper that is independently callable
may deserve its own operation entry even when it shares a kernel module.

Each operation is registered once with:

```text
FrontendOperationSpec
  name                         stable semantic/variant identifier
  contract_signature           internal common parameters/defaults
  targets[TORCH]               required canonical wrapper binding
  targets[JAX]                 optional binding or explicit TargetGap
  parameter_map                semantic name -> target API name
  output_map                   semantic role -> target result name
  target_only_parameters       e.g. Torch current_stream
  api_anchors                  exact public class/wrapper symbols it owns
  kernel_anchors               exact @cute.kernel symbols it owns
  parity_case                  required when a JAX binding exists
```

The target APIs remain separate and explicit:

```python
# Canonical existing Torch API, unchanged.
from cudnn import rmsnorm_rht_amax_wrapper_sm100

torch_result = rmsnorm_rht_amax_wrapper_sm100(...)

# Optional JAX-native API.
from cudnn.jax import rmsnorm_rht_amax_sm100

jax_result = rmsnorm_rht_amax_sm100(...)
```

Implicit array-type dispatch is deliberately rejected. JAX tracers, Torch fake
tensors/proxies, mixed operands, and optional framework installations make it
ambiguous and brittle. A `target=` argument inside the call would also become
part of the traced call surface. Importing `cudnn.jax` is the explicit opt-in and
provides an ordinary stable function to `jax.jit`.

Semantic parity does not mean exact Python signatures or return containers.
Torch legitimately retains `current_stream`, singleton-padding compatibility,
eager allocation, `TupleDict`, and its explicit class lifecycle. JAX owns its
stream and buffers, consumes abstract values, and returns a standard pytree. The
catalog instead checks shared logical operands, common option names/defaults,
output roles, validation/inference, and registered numerical parity coverage.

The registry requires both target keys. A genuinely unavailable target must be
represented by `TargetGap(reason, tracking_issue)` rather than by omission. The
normal policy for a newly added operation is stricter: new target gaps do not
merge. `TargetGap` exists to inventory pre-existing debt and to make an
exception reviewable, not to make a Torch-only addition look complete.

Here, "optional JAX" means the dependency and namespace are opt-in for users;
it does not mean JAX support status is optional for contributors. Every
cataloged operation must say implemented or gap, and the no-growth gate expects
new operations to be implemented for JAX.

The POC's dependency-free AST audit currently finds 64 public Python API
anchors and 58 physical `@cute.kernel` symbols. RMSNorm owns two API anchors and
one kernel under a complete target contract. The remaining 62 API anchors and
57 kernels are exact-symbol migration baselines. Module-level ownership is not
sufficient: it would let a new kernel in an existing module pass unnoticed.
The operation-level JAX-gap baseline starts empty, so a newly cataloged
operation cannot declare JAX missing without an explicit baseline-policy
exception.

The current wrapper inventory makes the functional gap concrete:

| Family | Public Torch wrapper entry points | Available JAX bindings |
| --- | ---: | ---: |
| RMSNorm + RHT + amax | 1 | 1 |
| Dense GEMM fusions | 4 | 0 |
| Grouped GEMM | 9 | 0 |
| Discrete grouped GEMM | 2 | 0 |
| SDPA | 2 | 0 |
| Native sparse attention | 4 | 0 |
| DeepSeek sparse attention stages | 11 | 0 |
| **Total** | **33** | **1** |

This is an inventory, not yet a promise that every legacy helper wrapper should
remain an independent public operation. Phase 1 must group those anchors into
reviewed semantic family/variant rows before generating the long-term support
matrix.

CI should enforce all of the following:

1. Discovered public `APIBase` descendants and wrapper functions equal the
   union of registered API anchors and the checked-in migration baseline.
2. Every discovered `@cute.kernel` equals a registered kernel owner or a
   reviewed internal/migration exception.
3. On a pull request, each baseline set is a subset of its merge-base version.
   This is what makes debt non-growing; an equality test against an editable
   file alone cannot enforce that policy.
4. Every bound target resolves to a module-level callable whose declared
   semantic parameter/default and output-role mappings still match the catalog.
   Torch-only parameters are explicit rather than leaked into JAX.
5. Every complete operation supplies common support-domain and numerical parity
   cases. Torch and JAX target lanes also test their target-specific lifecycle.
6. JAX cases cover `eval_shape`, `jit`, custom-call lowering, execution, and
   documented transformation behavior.

The AST audit detects structural additions and mapped-default drift. It cannot
infer that an edit inside an already-owned kernel changed semantics. Shared
parity fixtures and target execution tests are therefore mandatory, not
optional follow-up coverage. The canonical Torch wrapper also needs a separate
compatibility snapshot for its signature, `TupleDict` behavior, singleton-rank
handling, and stream semantics.

The initial scan covers the Python FE-OSS surface. C++ OSS engine registrations
need a separate discovery adapter feeding the same semantic catalog; scanning
Python alone must not be presented as complete repository-wide coverage.

### 1. Framework-neutral operator core

Move compile-affecting logic out of framework bindings. The target shape is an
immutable operator definition with these responsibilities:

```text
OperatorDefinition
  config                         static algorithm/tile/mask attributes
  infer_outputs(input_specs)     logical output shapes and dtypes
  validate(input_specs, target)  rank, dtype, layout, divisibility, target limits
  make_launcher(config)          CuTe launcher or kernel factory
  workspace_specs(...)           size, alignment, initialization, lifetime
  aliasing                       legal in-place relationships
  transform_policy               AD, batching, and sharding behavior
```

The core must not allocate tensors, inspect device values, access data pointers,
or choose a framework stream.

The POC starts this extraction with
[`_rmsnorm_rht_amax_config.py`](../../python/cudnn/_rmsnorm_rht_amax_config.py),
which mirrors the canonical Torch launch rules for JAX without modifying the
Torch implementation. Moving both targets onto one neutral implementation is a
Phase 1 change gated by Torch compatibility snapshots.

### 2. Torch adapter

Keep existing class and wrapper APIs stable and first class:

- Convert Torch tensor metadata to canonical operator specs.
- Allocate outputs and eager workspace with Torch.
- Compile through the existing TVM-FFI path.
- Invoke on the explicit/current Torch stream.

As operator cores are extracted, `APIBase` remains the canonical Torch binding
rather than becoming a framework-generic abstraction. There is no second
`.torch` wrapper, shared result container, or forced signature migration. This
avoids a risky flag day across all existing APIs.

### 3. JAX adapter

Expose an optional JAX-native module-level function:

```python
from cudnn.jax import rmsnorm_rht_amax_sm100

@jax.jit
def f(x, weight):
    output, amax = rmsnorm_rht_amax_sm100(x, weight)
    return output, amax
```

Each wrapper:

1. Reads only abstract shape/dtype metadata.
2. Resolves static configuration.
3. Infers every public output and hidden workspace.
4. Binds a canonical launcher through `call_cutedsl`.
5. Returns a standard tuple, named tuple, or registered pytree.

Do not dispatch implicitly by checking whether an argument is a Torch tensor or
a JAX tracer. Namespace selection happens before tracing, and imports inside the
JAX implementation keep optional dependencies off the base/Torch path.

### 4. Future C++ graph adapter

The C++ cuDNN graph API should not be wrapped in another `cutlass_call`; it has
no Python CuTe compilation to perform. A future `cudnn.jax.graph` layer should
use public typed `jax.ffi.ffi_call` and a registered C++ XLA FFI handler around
the existing sorted-pointer graph execution seam.

That handler should use stateful FFI stages for immutable plan metadata and a
thread-safe per-device plan/handle cache, but execution must always use XLA's
stream. This work is independent of the FE-OSS CuTe DSL POC.

## Tensor metadata and layout

The current `TensorDesc` combines logical tensor metadata with Torch-specific
storage and device types. The production refactor should separate:

- canonical scalar type;
- logical shape;
- compact physical layout order;
- logical mode permutation presented to the kernel;
- storage packing, especially FP4;
- assumed alignment and dimension divisibility;
- target/device requirements.

`cutlass.jax.TensorSpec` models compact tensors. It derives runtime strides from
runtime dimensions and the declared layout order; it does not accept arbitrary
runtime stride vectors. The JAX binding must therefore:

- constrain supported compact layouts in the custom call;
- use `mode` to reinterpret logical modes without materializing a transpose;
- let XLA insert a layout conversion when an operand does not already satisfy
  the requested physical layout;
- reject overlapping, broadcast-stride, and non-compact layouts unless the
  kernel is explicitly adapted.

This differs from Torch, where wrappers can allocate an arbitrary
`empty_strided` result. Layout should be an operator contract, not inferred from
a fictitious JAX stride API.

FP4 also needs a target-specific storage policy. Existing Torch code sometimes
uses packed `uint8` and doubles the logical innermost dimension. JAX/CUTLASS has
a native FP4 dtype mapping. Canonical operator specs must record logical dtype
and packing separately so one target's physical representation does not leak
into another target's shape inference.

## Compile, cache, and run lifecycle

### Trace and lowering

- Shapes, dtypes, optional operand presence, layouts, algorithm choices, tile
  sizes, mask modes, and other compile-affecting configuration are static.
- Data-dependent scalars are JAX operands, not Python values. Values such as
  tile sizes or a rarely changed `eps` may remain static and cause a recompile.
- No `.item()`, `.cpu()`, `.tolist()`, `.data_ptr()`, DLPack conversion, Torch
  allocation, or stream lookup is allowed while tracing.
- Support checks based on shapes/dtypes/layouts run at trace time. Device-value
  validation must either become a device computation or an explicit static
  bound.

### Cache keys

Only immutable compilation state belongs in a cache key:

- launcher/kernel identity and source version;
- canonical input/output/workspace specs;
- static operator config;
- target architecture and compiler options;
- relevant CUTLASS/cuDNN/frontend ABI versions.

Do not cache output buffers, workspace buffers, tensor addresses, CUDA streams,
or mutable scheduler counters. CUTLASS currently keys its in-memory compile cache
partly by Python function identity, so the POC memoizes generated launcher
closures.

CUTLASS 4.5's outer JAX compile-cache key does not visibly include the target
architecture, and both that cache and the POC launcher cache retain entries
without a bound. Before supporting heterogeneous GPU processes or a large
unbounded set of generated launchers, verify device targeting with CUTLASS and
provide coordinated cache observability and clearing.

### Runtime

- XLA supplies inputs, outputs, workspaces, and the CUDA stream.
- All helper kernels and the main kernel enqueue on that stream.
- The call returns without synchronizing the host.
- Concurrent executions of one compiled executable must not share mutable
  per-call storage.

## Workspace and initialization design

Workspace is the main difference from the simple CUTLASS examples. The preferred
representation is an extra custom-call result that the Python wrapper drops:

```text
custom-call results = public outputs..., hidden workspace buffers...
user-visible result = public outputs...
```

OpenXLA documents unused tuple results as
[temporary custom-call buffers](https://openxla.org/xla/custom_call#tuple_outputs_as_temp_buffers).
This lets XLA account for the memory, reuse it according to liveness, preserve
asynchronous lifetime, and avoid runtime allocation during CUDA-graph capture.

The POC's [`call_cutedsl`](../../python/cudnn/jax/cutedsl.py) supports these
categories:

| Buffer requirement | JAX representation |
| --- | --- |
| Fully overwritten output/workspace | Uninitialized custom-call result |
| Must start at zero | `jnp.zeros` operand aliased to the result |
| Must start at a constant such as `-inf` | `jnp.full` operand aliased to the result |
| Public in-place result | Declared input/output alias; donation recommended |
| Forward residual needed by backward | Real JAX output/residual, not workspace |
| Data-dependent workspace size | Unsupported initially; use a static upper bound or later runtime allocator fallback |

An initializer is an aliased operand because custom-call result buffers are not
initialized. XLA retains functional semantics; `donate_argnums` is needed when a
caller wants guaranteed reuse of a user input rather than copy protection.

Workspace size must be derivable from abstract input metadata before XLA buffer
assignment. If a future plan can only determine its workspace at runtime, choose
one of:

1. a deterministic safe upper bound;
2. trace/lowering-time plan selection and a serialized, pointer-free recipe;
3. XLA FFI `ScratchAllocator` as a measured fallback.

Runtime scratch allocation is less visible to XLA's memory planner and needs
separate CUDA-graph, alignment, and asynchronous-lifetime validation. It should
not be the default.

For one packed byte workspace, overallocate and align internal slices when
operator-specific subregions require stronger alignment. A zero-byte workspace
must not assume a non-null pointer.

### Persistent descriptor and scheduler workspace

Several grouped/discrete GEMM APIs keep TMA descriptors or scheduler counters in
a cached Torch tensor. That storage cannot be shared by concurrent JAX calls.
Initially, allocate it per invocation and run descriptor/counter initialization
as helper kernels on XLA's stream before the main kernel. Persistent device state
is only worth adding after profiling proves this cost material, and it would need
a thread-safe stateful runtime design.

## Operator-specific audit

| Family | Main JAX work |
| --- | --- |
| RMSNorm + RHT + amax | Static output inference, BF16 validation, per-CTA amax shape. POC complete. |
| Dense GEMM fusions | Map M/N-major compact layouts, FP4/FP8 storage, multiple auxiliary outputs, and initialized amax buffers. |
| SDPA forward | Make max sequence bounds static; remove `.item()` inference; model optional varlen operands and output layouts. |
| SDPA backward | Per-call zeroed workspace, multi-kernel ordering, residual contract, and `custom_vjp`. |
| Grouped GEMM | Per-call descriptor/scheduler workspace, helper launches, counters, and metadata validation without host reads. |
| Discrete grouped GEMM | Current device-address tensors are not traceable. Pass expert buffers as real operands and build pointer tables in device workspace, or defer this mode. |
| DSA/NSA orchestration | Remove host value inspection, define fixed-capacity outputs for compaction, and audit every multi-kernel temporary. |
| Distributed kernels | Replace `torch.distributed` assumptions with explicit JAX collectives/partitioning; do not hide cross-replica state in a CuTe call. |

Optional tensors alter call arity and are therefore static graph variants.
Data-dependent compacted output lengths require a fixed-capacity result plus a
valid-count output.

Random operations must accept explicit JAX key/seed/counter operands and return
advanced state where needed. They must not consume hidden global RNG state.

## JAX transformations

### Autodiff

`cutlass_call` explicitly rejects JVP/transpose rules. Inference-only wrappers
should expose that limitation clearly. Operators with forward and backward
kernels should use `jax.custom_vjp`:

- forward returns the primal outputs plus explicit residual arrays;
- backward consumes residuals and cotangents and invokes a separate CuTe call;
- reserve/stat tensors needed later are residuals, not scratch;
- higher-order and forward-mode differentiation require separate decisions.

### Batching

`cutlass_call` explicitly rejects `vmap`, even though the lower-level public FFI
has generic batching modes. Add `custom_vmap` only when an operator defines how
the new batch axis changes shapes, layouts, workspace, and launch configuration.
Do not silently use sequential batching in the FE wrapper.

### Sharding

The initial contract is `shard_map`: the FE wrapper sees device-local shapes and
launches one local kernel. Automatic partitioning needs an operator-specific
`custom_partitioning` rule that states:

- which axes may be sharded without communication;
- whether output and workspace shapes are local or global;
- which layouts remain valid after partitioning;
- what collectives or resharding are required.

Do not claim transparent multi-device support solely because a custom call can
be replicated.

### Export and dynamic shapes

CUTLASS demonstrates `jax.export` with symbolic dimensions and divisibility
hints, but each FE kernel must be audited. The POC intentionally requires
concrete `M` and `N` and uses static tensors.

Exported programs also require the matching CUTLASS runtime and custom-call
registration in the consuming process. They are not runtime-independent
StableHLO artifacts.

## CUDA graphs and effects

Normal FE operations are pure: they read declared operands and write declared
results/aliases. Keep side effects false so XLA may optimize them normally.
Global state, logging, or hidden RNG mutation would violate this contract.

`cutlass_call` enables CUDA graphs by default. An operator should opt out when it
performs unsupported first-use initialization, runtime allocation, host
synchronization, or dynamic control. Tests must warm compilation and module
loading before measuring capture/replay.

## Public API and packaging

Recommended public shape:

```python
# Existing canonical Torch API and return type remain unchanged.
from cudnn import rmsnorm_rht_amax_wrapper_sm100

torch_result = rmsnorm_rht_amax_wrapper_sm100(
    x_tensor=x_torch,
    w_tensor=weight_torch,
    eps=1e-5,
    num_threads=128,
    rows_per_cta=2,
    current_stream=None,
)

# JAX is a separate optional install and namespace.
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

jax_result = run(x_jax, weight_jax)
```

- Torch remains the default, first-class API. Its class lifecycle, wrapper
  signature, `TupleDict`, singleton-padding behavior, and stream controls are
  compatibility contracts.
- JAX is explicitly selected by importing `cudnn.jax`; installing the base or
  Torch extras does not require JAX.
- JAX operations use functional inputs/outputs and standard named tuples or
  registered pytrees. They do not emulate Torch output buffers or streams.
- Common option names and defaults should be retained when meaningful, but
  exact target signatures and containers are not required.
- Compile-affecting JAX keyword arguments are Python-static: close them over as
  above or name them in `jax.jit(static_argnames=...)`. Dynamic tensor/scalar
  operands remain explicit inputs.
- Clear errors for unsupported transforms and shape modes.
- A flat array ABI at the generic CuTe call seam. Operator-level pytrees may be
  flattened and reconstructed by their wrappers, but nested inputs do not cross
  the low-level adapter.

The internal catalog is for ownership, support reporting, and CI; it is not a
replacement public dispatch facade.

The POC adds a `jax` optional dependency group. It intentionally does
not depend on Torch. Production CI needs a JAX lane whose test bootstrap also
does not unconditionally import Torch.

JAX follows a fast-moving pre-1.0 compatibility policy. The current official
CuTe DSL + JAX guide lists JAX 0.8.1 as its minimum, while CUTLASS 4.5's module
still declares a looser 0.5.0 minimum. The POC follows the current guide,
bounds the experimental range below JAX 0.10, and retains the repository's
CUTLASS 4.5.0 pin. JAX 0.8.1 and newer require Python 3.11, so this extra has a
narrower Python requirement than the base package. The repository's current
test requirements also pin NumPy below 2.0, while JAX 0.8.1 requires NumPy 2;
the JAX lane therefore needs a separate dependency bootstrap. Pin and test a
supported JAX/CUTLASS/Python matrix before promoting the integration. In
particular, CUTLASS 4.5 selects its original FFI implementation below JAX 0.9.1
and its stateful FFI implementation at JAX 0.9.1 or newer, so CI must cover both
sides of that boundary.

## Proof of concept

### Implemented pieces

- [`cudnn.frontend`](../../python/cudnn/frontend)
  - is an internal dependency-light support and ownership catalog, not a public
    execution facade;
  - requires a concrete canonical Torch wrapper plus a JAX binding or explicit
    tracked gap;
  - records semantic parameter/output mappings and intentional target-only
    parameters such as `current_stream`;
  - requires a parity-case identifier whenever JAX is available;
  - records exact public API and physical kernel ownership anchors;
  - includes a dependency-free audit of all current Python FE-OSS APIs and
    `@cute.kernel` definitions.
- [`cudnn.jax.cutedsl`](../../python/cudnn/jax/cutedsl.py)
  - remains an internal POC seam rather than a top-level public export until a
    real workspace-using operator validates the contract;
  - translates output/workspace metadata to `cutlass_call`;
  - appends hidden workspaces and drops them from public results;
  - supports uninitialized, zeroed, and constant-filled buffers;
  - reconstructs canonical launcher order when results alias inputs;
  - requires identical input/output metadata for an alias and rejects nested
    low-level input pytrees;
  - forwards compact layout/mode/divisibility/alignment metadata;
  - memoizes launcher adapters for stable CUTLASS cache identity.
- [`cudnn.jax.rmsnorm_rht_amax_sm100`](../../python/cudnn/jax/rmsnorm_rht_amax.py)
  - validates abstract rank, shape, and BF16 dtype;
  - matches the canonical Torch option names/defaults and launch heuristics;
  - declares `(M, N)` BF16 and `(M / rows_per_cta,)` FP32 outputs;
  - lowers the existing CuTe kernel through `cutlass_call`.
- CPU-only adapter contract tests cover hidden workspace, initialization,
  aliases, layouts, and invalid metadata.
- CPU-only parity-contract tests cover canonical Torch ownership, optional JAX
  status, mapped-default drift, parity-case presence, API discovery, and exact
  kernel ownership.
- A GPU test inspects StableHLO, executes one compiled program twice, and checks
  a numerical JAX reference when JAX, CuTe DSL, CUDA, and SM100 hardware are
  available.

### POC limitations

- Only one operation is exposed.
- Existing APIs and physical kernels remain in explicit migration baselines;
  the POC has not yet grouped all legacy anchors into semantic operation rows.
- The local AST test makes baseline edits visible but cannot by itself forbid a
  contributor from adding a new exception. Production CI must compare the
  baseline sets with the pull request's merge base and reject growth.
- `M` and `N` must be concrete during tracing.
- JAX currently accepts the rank-2/rank-1 core domain only; the canonical Torch
  wrapper's singleton-padded compatibility remains Torch-specific.
- No custom gradient, batching, or partitioning rule.
- No local SM100 GPU was available for execution in the development
  environment; the GPU test is present but must run in CI.
- The generic workspace path is contract-tested with a fake CUTLASS bridge; the
  first real operator does not need device workspace.
- The JAX optional dependency version range needs release-owner approval.
- Launcher and CUTLASS compile caches are currently unbounded.

## Rollout plan and acceptance gates

### Phase 0: vertical-slice POC

Status: implemented in this change.

- Add the JAX CuTe call adapter.
- Extract one framework-neutral launch config.
- Port RMSNorm + RHT + amax.
- Add the Torch-canonical support catalog and exact-symbol ownership audit.
- Add CPU contract tests and an SM100 numerical test.

Gate: run the GPU test on SM100, inspect lowered StableHLO for one custom call,
and verify a second invocation reuses the compiled executable.

### Phase 1: neutral metadata core

- Replace the flat migration baselines with catalog rows grouped by semantic
  family/variant, each carrying API/kernel ownership and gap issue metadata.
- Add merge-base non-growth enforcement and generate a checked support matrix
  from the catalog.
- Define canonical dtype plus explicit FP4 packing.
- Define logical shape, compact physical layout, mode mapping, alignment, and
  divisibility independent of Torch.
- Split shape/output inference and support validation from allocation.
- Preserve existing Torch constructors, wrappers, result containers, and dtype
  arguments as independently tested compatibility contracts.
- Add Torch/JAX cross-target metadata tests.

Gate: the RMSNorm operator has one validation/inference implementation, both
targets retain numerical behavior, and a new API or `@cute.kernel` fails CI
until it has a complete semantic owner.

### Phase 2: workspace-free dense kernels

- Port one dense GEMM fusion with multiple outputs.
- Validate M/N-major layout conversion and FP8.
- Exercise initialized auxiliary outputs and legal donation.
- Define compile-cache observability and limits.

Gate: JIT correctness, layout HLO inspection, concurrent calls, cache reuse, and
Torch parity on supported shapes.

### Phase 3: workspace and training

- Port SDPA forward/backward.
- Allocate per-call hidden workspace with explicit zero/fill semantics.
- Make sequence maxima static arguments or explicit bounds.
- Add `custom_vjp` and residual outputs.
- Validate CUDA graph replay and reentrancy on multiple streams.

Gate: forward/backward numerical parity, no shared mutable workspace, gradient
tests, concurrent execution, and memory/capture tests.

### Phase 4: grouped, discrete, and sharded operations

- Build descriptor and pointer tables from real operands on device.
- Remove host reads of tensor metadata values.
- Add operator-specific `shard_map` examples and partitioning rules.
- Integrate explicit JAX collectives where distributed semantics require them.

Gate: multi-device correctness, pointer lifetime safety, no host synchronization
in the runtime path, and deterministic workspace bounds.

### Separate track: C++ graph API

- Define a versioned deterministic graph/plan descriptor.
- Register a typed GPU XLA FFI target.
- Use a thread-safe per-device state/handle strategy.
- Declare logical outputs and hidden workspace through `jax.ffi.ffi_call`.

This track should reuse neutral tensor/operator metadata but not the CuTe
compilation adapter.

## Risks and decisions still needing owner approval

- Supported JAX/CUTLASS/CUDA version matrix and wheel packaging.
- Stable canonical dtype/packing representation, particularly FP4.
- Whether JAX operation naming and named-result conventions match broader cuDNN
  Python API plans.
- Which dynamic-shape/export guarantees are required per operation.
- Whether any operator's workspace is genuinely data-dependent after a safe
  plan policy is fixed.
- Performance threshold for reinitializing grouped-GEMM descriptor workspace on
  every invocation.
- Per-operator AD, batching, sharding, aliasing, and CUDA-graph policies.

These choices should be recorded per operator rather than hidden in a generic
adapter default.
