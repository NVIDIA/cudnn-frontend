# Kimi Delta Attention in JAX

`cudnn.jax.kimi_delta_attention` returns `(output, final_state_or_None)` and
supports `jax.jit` and first-order reverse-mode differentiation. It uses the
existing Frost KDA kernels, including their backward, checkpoint recompute,
split scheduler, gate gradients, and grouped-head reductions.

This implementation is pinned to Frost. Other engines have different launch/runtime
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
| `cu_seqlens` | `[N+1]` | INT32 or INT64 |
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
Enable `jax_enable_x64` before constructing INT64 offsets; otherwise JAX may
downcast them to INT32. INT64 storage does not remove the signed-INT32 limits
on total tokens and scheduler counts.

The state is V-major. It is a differentiable array, not a mutable cache. Pass the
returned state explicitly to the next call, including inside `jax.lax.scan`.
The native engine's paged-state and in-place state options remain supported for
torch; this JAX API does not expose `state_indices` or `overwrite_initial_state`.

## Usage

Install GPU-enabled JAX and CuTeDSL separately from the framework-neutral package:

```bash
pip install 'jax[cuda13]' 'nvidia-cutlass-dsl[cu13]>=4.7.0'
```

Select one visible supported GPU before starting Python. This implementation requires
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

Pass `gate_domain="linear"` to either forward API when `g` contains decay
factors such as `0.9`, rather than their natural logarithms. Backward and
`jax.grad` then return gradients with respect to those factors. Linear gates
cannot be combined with `safe_gate=True`.

`checkpoint_every_n_tokens=64`, for example, saves fewer recurrent states than
cadence 16. Backward uses Frost's existing seeded recomputation to reconstruct
the intermediate states. This reduces saved residual storage, but backward
still needs workspace for the reconstructed states. Cadence 0 saves no
checkpoints; any positive multiple of 16 fitting signed INT32 is supported.

## Bounded support

| Feature | Supported behavior |
|---|---|
| Engine / GPU | Frost; SM100 or SM103; one visible GPU. SM103 not yet tested here. |
| Head dimensions | K and V independently 64 or 128 |
| Heads | HQ:HV ratios 1, 2, 4, or 8 in either direction; HK equals HQ or HV |
| Sequences | Positive static T/N; packed THD; empty individual sequences allowed |
| State | Optional initial/final state; gradients through both; recurrent scan |
| Gates | Natural-log decay by default; direct decay with `gate_domain="linear"`; safe gate with optional a_log/dt_bias |
| Safe gate | `lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`; default bound -5, allowed [-5, 0); omitted parameters mean 0 |
| Beta | Direct write strength, or sigmoid of logits; `allow_neg_eigval=True` multiplies sigmoid by 2 and requires sigmoid enabled |
| Q/K normalization | Optional in-kernel L2 normalization |
| Scheduling | Frost automatic piece-chain / decay-warmup / value-dimension split / uncut selection, shared with torch; `batch_invariant=True` uses Frost's batch-independent length rule |
| Checkpoints | Cadence 0 (backward recomputes), 16, or coarser positive multiples of 16 (seeded recomputation) |
| Transformations | Eager, jit, first-order grad/vjp, recurrent scan |

For a positive cadence `C`, checkpoint capacity is
`[max(T // C + N, 1), HO, V, K]` in q dtype. The valid
prefix contains `sum(ceil(sequence_length / C))` rows; padding is initialized to
zero. The residual holds primals, static configuration and optional checkpoints.
It owns no graph handle, workspace or executable. Do not fabricate or edit
residuals: checkpoints must correspond to their saved primals and configuration.

Deferred: cuTile/other engine routing, SM90/SM120/SM107, indexed state pools,
zero total tokens, arbitrary physical strides, `vmap`, JVP, higher derivatives,
sharding/multiple GPUs, portable executable export, and full FLA compatibility.
JAX supplies compact row-major operands to the custom call; upstream transposes
may require materialization.

## Execution

Frost chooses the schedule and workspace layout. JAX uses the same kernel
configuration builders and multi-kernel hosts as the native executor, with
XLA-owned outputs, per-call workspace and stream. Compilation metadata is
cached; workspace is never shared between calls. Execution needs no torch
import or Python callback.

SM103, additional CUDA/JAX/CuTeDSL versions, independent-stream stress and
customer workloads still need qualification. The 10–20 us CPU dispatch target
and JAX/native performance parity have not been established by numerical tests.

## Tests

Run JAX coverage without the torch-based parent fixtures:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest test/python/fe_api/jax/test_call_jax.py test/python/fe_api/jax/test_kda_jax.py \
  --confcutdir=test/python/fe_api/jax
```

The suite covers forward/backward parity, state and gate gradients, repeated
calls, packed boundaries, recurrent scans, command-buffer replay, concurrent
dispatch and torch-free execution. Tests skip when JAX, a supported GPU or
CuTeDSL >=4.7.0 is unavailable.

Run native cache and output-stride regressions from `test/python`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -s -q linear_attention/test_kda_execution.py
```
