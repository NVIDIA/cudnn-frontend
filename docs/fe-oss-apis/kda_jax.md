# Kimi Delta Attention in JAX (draft)

`cudnn.jax.kimi_delta_attention` returns `(output, final_state_or_None)` and
supports `jax.jit` and first-order reverse-mode differentiation. It uses the
existing Frost KDA kernels, including their backward, checkpoint recompute,
split scheduler, gate gradients, and grouped-head reductions.

This draft is pinned to Frost. Other engines have different launch/runtime
contracts; accepting their graph plans would not make them executable through
this bridge. The existing torch API retains its engine routing.

## Inputs and outputs

`T` is the packed token count, `N` the sequence count, and `HO = max(HQ, HV)`.

| Tensor | Shape | Dtype |
|---|---|---|
| `q` | `[T, HQ, K]` | BF16 or FP16 |
| `k` | `[T, HK, K]` | Same as q |
| `v` | `[T, HV, V]` | Same as q |
| `g` | `[T, HO, K]` | FP32, BF16, or FP16 |
| `beta` | `[T, HO]` | FP32 or q dtype |
| `cu_seqlens` | `[N+1]` | INT32 |
| `initial_state` (optional) | `[N, HO, V, K]` | FP32 or BF16 |
| `a_log` (optional) | `[HO]` | FP32, BF16, or FP16 |
| `dt_bias` (optional) | `[HO, K]` | FP32, BF16, or FP16 |
| output | `[T, HO, V]` | Same as q |
| final state (optional) | `[N, HO, V, K]` | Initial state dtype; otherwise FP32 |

`cu_seqlens` must start at zero, be nondecreasing, and end at `T`. Values are
runtime device data: changing sequence lengths without changing tensor shapes
does not require recompilation. The wrapper validates metadata, not boundary
values; it never copies boundaries to the host. Callers must supply valid
boundaries. Empty individual sequences preserve their initial state (or zero).

The state is V-major. It is a differentiable array, not a mutable cache. Pass the
returned state explicitly to the next call, including inside `jax.lax.scan`.

## Usage

Install GPU-enabled JAX and CuTeDSL separately from the framework-neutral package:

```bash
pip install 'jax[cuda13]' 'nvidia-cutlass-dsl[cu13]>=4.7.0'
```

Select one visible supported GPU before starting Python. This draft requires
one local GPU to avoid guessing a target from tracers. The tested combination is
JAX 0.11.1, CuTeDSL 4.7.1, SM100; broader version/architecture qualification is
still required.

```python
import jax
import jax.numpy as jnp
from cudnn.jax import kimi_delta_attention as kda

q = jnp.full((32, 1, 64), 0.05, jnp.bfloat16)
k, v = q, jnp.ones_like(q)
g = jnp.full(q.shape, -0.1, jnp.float32)
beta = jnp.full((32, 1), 0.5, jnp.float32)
cu = jnp.array([0, 16, 32], jnp.int32)
state = jnp.zeros((2, 1, 64, 64), jnp.float32)

# Eager.
o, final = kda(q, k, v, g, beta, cu, initial_state=state,
               output_final_state=True)

# Options are static; arrays, boundaries and state remain runtime inputs.
def attend(q, k, v, g, beta, cu, state):
    return kda(q, k, v, g, beta, cu, initial_state=state,
               output_final_state=True)

o, final = jax.jit(attend)(q, k, v, g, beta, cu, state)

def loss(q, k, v, g, beta, state):
    o, final = attend(q, k, v, g, beta, cu, state)
    return o.astype(jnp.float32).sum() + final.sum()

grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5)))(
    q, k, v, g, beta, state)
```

Safe-gate parameters are ordinary differentiable arguments:

```python
a_log = jnp.zeros((1,), jnp.float32)
dt_bias = jnp.zeros((1, 64), jnp.float32)

def gate_loss(a_log, dt_bias):
    o, _ = kda(q, k, v, g, beta, cu, safe_gate=True,
               a_log=a_log, dt_bias=dt_bias,
               use_beta_sigmoid_in_kernel=True)
    return o.astype(jnp.float32).sum()

da_log, ddt_bias = jax.jit(jax.grad(gate_loss, argnums=(0, 1)))(a_log, dt_bias)
```

For explicit backward:

```python
from cudnn.jax import kimi_delta_attention_fwd, kimi_delta_attention_bwd

o, final, residual = kimi_delta_attention_fwd(
    q, k, v, g, beta, cu, initial_state=state,
    output_final_state=True, checkpoint_every_n_tokens=16)
grads = kimi_delta_attention_bwd(
    residual, jnp.ones_like(o), d_final_state=jnp.ones_like(final))
# grads.dq, dk, dv, dg, dbeta, d_initial_state, d_a_log, d_dt_bias
```

An omitted final-state cotangent means zero. Gradients for absent initial state
or parameters are `None`. Boundaries and static options are not differentiable.
The explicit APIs do not themselves provide autodiff rules; the high-level API
installs `custom_vjp` using the existing backward kernels. Higher derivatives and
forward-mode differentiation are unsupported.

## Bounded support

| Feature | Draft contract |
|---|---|
| Engine / GPU | Frost; SM100 or SM103; one visible GPU. SM103 not yet tested here. |
| Head dimensions | K and V independently 64 or 128 |
| Heads | HQ:HV ratios 1, 2, 4, or 8 in either direction; HK equals HQ or HV |
| Sequences | Positive static T/N; packed THD; empty individual sequences allowed |
| State | Optional initial/final state; gradients through both; recurrent scan |
| Gates | Natural-log decay by default; safe gate; optional a_log/dt_bias |
| Safe gate | `lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`; default bound -5, allowed [-5, 0); omitted parameters mean 0 |
| Beta | Direct write strength, or sigmoid of logits; `allow_neg_eigval=True` multiplies sigmoid by 2 and requires sigmoid enabled |
| Q/K normalization | Optional in-kernel L2 normalization |
| Scheduling | Default Frost split schedule; `batch_invariant=True` disables splitting |
| Checkpoints | Cadence 0 (backward recomputes) or 16 (saved for backward) |
| Transformations | Eager, jit, first-order grad/vjp, recurrent scan |

Checkpoint capacity is `[max(T // 16 + N, 1), HO, V, K]` in q dtype. The valid
prefix contains `sum(ceil(sequence_length / 16))` rows; padding is initialized to
zero. The residual holds primals, static configuration and optional checkpoints.
It owns no graph handle, workspace or executable. Do not fabricate or edit
residuals: checkpoints must correspond to their saved primals and configuration.

Deferred: cuTile/other engine routing, SM90/SM120/SM107, coarse checkpoints,
zero total tokens, arbitrary physical strides, `vmap`, JVP, higher derivatives,
sharding/multiple GPUs, portable executable export, and full FLA compatibility.
JAX supplies compact row-major operands to the custom call; upstream transposes
may require materialization.

## Execution and review notes

The shared graph builders and Frost workspace layouts are reused. The JAX
launcher composes existing CuTe host functions into one native XLA FFI call per
forward/backward. That call receives XLA's stream and launches the complete
sequence, including runtime TMA descriptors and PDL dependencies. No Python
callback, torch tensor, DLPack handoff, or KDA math implementation is introduced.

XLA allocates outputs and per-invocation byte workspace. Backward carves the
same scheduler, descriptor, recompute, gate-reduction and head-reduction regions
as torch. Workspace is never cached across calls. Graph/shape/launch metadata
is cached at trace time; executable caching belongs to the CuTeDSL/JAX bridge.
Checkpoints use the shared bridge's initialized-output support, with aliased
outputs restored to their declared argument positions.

Alternatives considered: a native C++ XLA FFI wrapper around the engine would
require exporting its Python-compiled launch artifacts and maintaining a new
ABI; Python callbacks or torch/DLPack would add host overhead and weaken stream
and capture guarantees. The composite CuTe path is the smallest working bridge.

Before production release, qualify customer shapes, CUDA/JAX/CuTeDSL versions,
SM103, true independent-stream stress and long-sequence numerical limits. The
10–20 us CPU target needs warm dispatch measurements separately from blocking
end-to-end latency and GPU kernel time. Compare against the same Frost schedule
and checkpoint policy. This draft does not assert that target for every shape.
BSA and quantized grouped GEMM are separate work.

Run coverage with:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest test/jax/test_call.py test/jax/test_kda.py
```

## Initial measurements and remaining blocker

On the local SM100, a BF16 `[T=1024, H=4, K=V=128]` stateless case passed
forward and explicit-backward parity against the torch Frost implementation.
Both used split scheduling and backward checkpoint recomputation.

The high-level jitted forward measured about 62 us median host dispatch without
command buffers and 46 us with command buffers (100 warmed calls; synchronization
outside the dispatch interval). This **does not meet the requested 10–20 us CPU
target**. Nsight shows graph-child updates as output/workspace addresses change.
Resolving that overhead, or choosing a different native bridge, remains a
production release blocker rather than a claimed result of this draft.

The 10-call Nsight sample measured the main prefill kernel at 25.8 us for JAX
versus 25.1 us for torch, and the main backward kernel at 78.0 us versus 76.8 us.
These are individual kernel durations, not full-call latency. The JAX run used
command buffers, torch used ordinary stream launches; this is a limited smoke
comparison, not full performance qualification.

For a standalone composite call, XLA may otherwise decline capture because it
counts only one custom-call operation. The validated capture configuration is:

```python
compiled = jax.jit(attend, compiler_options={
    "xla_gpu_enable_command_buffer": "CUSTOM_CALL",
    "xla_gpu_graph_min_graph_size": 1,
})
```

These are application compiler options; the library does not change global XLA
settings. Capture was confirmed in Nsight, including replay after pointer changes.

Reproduce warm forward and explicit-backward measurements with
`python test/jax/benchmark_kda.py --command-buffer`. Remove the flag for default
XLA settings. The benchmark also checks torch parity; it intentionally imports
both frameworks. GPU profiling can wrap the same command with `nsys profile
--trace=cuda,nvtx --capture-range=cudaProfilerApi --cuda-graph-trace=node`.

Memory-only Compute Sanitizer coverage passed five forward/backward and concurrent
replay cases with zero errors using `--tool memcheck --report-api-errors no`.
API reporting was disabled for that run because cuda-python probes API versions
newer than the local driver's 13.0 interface; the unfiltered run reported 68 such
probe errors, without invalid device-buffer accesses. Upgrade the driver before
claiming an unfiltered clean sanitizer run on the customer stack.
