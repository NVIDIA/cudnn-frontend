# Linear attention in JAX

The native JAX bindings run the existing CuTeDSL FROST kernels on SM100 and
SM103, without importing PyTorch. They support eager calls, `jax.jit`, and
first-order reverse-mode differentiation through output, initial/final state,
and optional gate parameters.

| Operation | Differentiable function | Explicit forward / backward |
|---|---|---|
| Gated DeltaNet | `cudnn.jax.gated_delta_net` | `cudnn.ops.gated_delta_net_forward_jax` / `gated_delta_net_backward_jax` |
| Gated DeltaNet v2 | `cudnn.jax.gated_delta_net_v2` | `cudnn.ops.gated_delta_net_v2_forward_jax` / `gated_delta_net_v2_backward_jax` |
| Gated DeltaProduct | `cudnn.jax.gated_delta_product` | `cudnn.ops.gated_delta_product_forward_jax` / `gated_delta_product_backward_jax` |
| Mamba-2 | `cudnn.jax.mamba2` | `cudnn.ops.mamba2_forward_jax` / `mamba2_backward_jax` |

The differentiable functions return `(output, final_state_or_None)`. Explicit
forward adds an opaque residual: `(output, final_state_or_None, residual)`.
Pass that residual unchanged to the matching backward function. XLA owns
outputs and scratch; no workspace, output buffer, or stream argument is needed.
[Kimi Delta Attention](/operations/LinearAttention/kda_jax) has its own JAX API.

## Packed sequences and state

Inputs use packed token-major tensors. `cu_seqlens[N+1]` is an INT32 or INT64
device array containing sequence boundaries: it starts at zero, is
nondecreasing, and ends at the packed token count `T`. Enable JAX x64 before
creating INT64 offsets; otherwise JAX may downcast them. Offsets remain runtime
data, and changing their values with the same shapes does not specialize the
call. The caller must provide valid boundary values; the bindings do not read
them on the host. Token and workspace indexing must fit signed INT32.

An empty sequence preserves its initial state, or zero when no state is
supplied. State is V-major `[N, HO, V, K]`. It is an ordinary differentiable
array: pass the returned state into the next call to continue a sequence.
`output_final_state=False` returns `None` for the state output.

## Delta-rule inputs

GDN and GDP use `(q, k, v, g, beta, cu_seqlens, **options)`; GDN2 adds `w`
between `beta` and `cu_seqlens`. GDP requires `num_householder=n` as a keyword.
`HO = max(HQ, HV)`; supported grouped heads expand Q/K/V onto the output heads.
The FROST delta-rule kernels support `K` and `V` of 64 or 128.

| Tensor | GDN | GDN2 | GDP |
|---|---|---|---|
| `q` | `[T,HQ,K]` | `[T,HQ,K]` | `[T,HQ,K]` |
| `k` | `[T,HK,K]` | `[T,HK,K]` | `[T*n,HK,K]` |
| `v` | `[T,HV,V]` | `[T,HV,V]` | `[T*n,HV,V]` |
| `g` | `[T,HO]` | `[T,HO,K]` | `[T,HO]` |
| `beta` | `[T,HO]` | `[T,HO,K]` | `[T*n,HO]` |
| `w` | — | `[T,HO,V]` | — |
| output | `[T,HO,V]` | `[T,HO,V]` | `[T,HO,V]` |

Q/K/V share BF16 or FP16 dtype. GDN2's beta and write gates share the I/O
dtype; scalar beta accepts FP32 or the I/O dtype. Decay gates and safe-gate
parameters accept FP32, BF16, or FP16. Initial state accepts FP32, BF16, or FP16;
final state follows its dtype, or uses FP32 when no initial state is supplied.
Gradients have their corresponding input dtype.

GDP boundaries, Q, decay, and output use real tokens. Each token's `n`
Householder updates occupy consecutive expanded rows of K/V/beta. The decay
acts before those updates, and the output is read after the last update.

All scalar options are static:

- `scale` multiplies Q and defaults to `1/sqrt(K)`.
- `use_qk_l2norm_in_kernel=True` normalizes Q and K with epsilon `1e-6`.
- `use_beta_sigmoid_in_kernel=True` treats beta as logits;
  `allow_neg_eigval=True` doubles that sigmoid and requires it to be enabled.
- `gate_domain="log"` interprets G as log decay. `"linear"` accepts decay
  factors and returns their gradients. It cannot be combined with `safe_gate`.
- `safe_gate=True` accepts raw G logits. GDN/GDP use
  `-exp(a_log) * softplus(g + dt_bias)`; both parameters have shape `[HO]`.
  GDN2 uses `gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`, with
  `a_log[HO]`, `dt_bias[HO,K]`, and default lower bound `-5`.
  Omitted parameters mean unit amplitude and zero bias. Supplying parameters
  requires safe-gate mode.
- GDN2 additionally exposes `beta_guard=True`, requiring in-kernel Q/K
  normalization, with the same straight-through gradient as the Torch API.
- `checkpoint_every_n_tokens=0` saves no checkpoints. Positive cadences must
  be multiples of 64 for GDN/GDP and 16 for GDN2. GDP counts expanded
  sub-tokens. Checkpoints are retained in the residual for backward reuse.

```python
import jax
import jax.numpy as jnp
from cudnn.jax import gated_delta_net

q = jnp.full((35, 1, 64), 0.05, jnp.bfloat16)
k, v = q, jnp.ones_like(q)
g = jnp.full((35, 1), -0.1, jnp.float32)
beta = jnp.full((35, 1), 0.5, jnp.float32)
cu = jnp.array([0, 17, 17, 35], jnp.int32)
state = jnp.zeros((3, 1, 64, 64), jnp.float32)

def loss(q, k, v, g, beta, state):
    out, final = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state,
        output_final_state=True, batch_invariant=True,
    )
    return out.astype(jnp.float32).sum() + final.sum()

grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5)))(
    q, k, v, g, beta, state
)
```

## Mamba-2 inputs

Mamba-2 follows the model-facing Torch signature:

```python
output, final = cudnn.jax.mamba2(
    x, dt, A, B, C, cu_seqlens,
    D=None, dt_bias=None, z=None, initial_state=None,
    output_final_state=True, checkpoint_every_n_tokens=0,
    dt_softplus=True, dt_min=0.0, dt_max=None, batch_invariant=True,
)
```

`x[T,H,64]`, `B[T,G,128]`, and `C[T,G,128]` share BF16 or FP16 dtype;
`H` is divisible by `G`. `dt[T,H]` holds raw step logits and `A[H]` holds
negative decay rates. Optional `D[H]` or `D[H,64]` supplies the skip weight,
`dt_bias[H]` supplies the step bias, and `z[T,H,64]` supplies the SiLU output
gate. `dt`, A, D, and step bias accept FP32/BF16/FP16; z matches x.
State is `[N,H,64,128]`.

The step is `clamp(softplus(dt + dt_bias), dt_min, dt_max)`; setting
`dt_softplus=False` omits softplus and `dt_max=None` removes the upper bound.
The gradient is zero where the clamp is active. Mamba-2 has no Q scaling
option. Checkpoint cadence must be zero or a positive multiple of 128 tokens.

## Explicit backward

```python
from cudnn.ops import gated_delta_net_forward_jax, gated_delta_net_backward_jax

out, final, residual = gated_delta_net_forward_jax(
    q, k, v, g, beta, cu, initial_state=state, output_final_state=True,
)
grads = gated_delta_net_backward_jax(
    residual, jnp.ones_like(out), d_final_state=jnp.ones_like(final),
)
```

Delta-rule backward returns the named tuple
`(dq, dk, dv, dg, dbeta, dw, d_initial_state, d_a_log, d_dt_bias)`.
`dw` is `None` for GDN/GDP. Mamba-2 returns
`(dx, ddt, da, db, dc, d_initial_state, dd, d_dt_bias, dz)` in model argument
order. Missing optional inputs have `None` gradients. An omitted
`d_final_state` contributes zero; supplying it requires a forward call with
`output_final_state=True`. Cotangents must match output/state shape and dtype.

Only the `cudnn.jax` convenience functions register custom VJPs. Explicit
forward/backward calls do not themselves support automatic differentiation.
Offsets and static options are not differentiable. Forward-mode AD, higher
derivatives, `vmap`, and distributed array partitioning are outside this API.

## Placement and qualification

All operands of a call must reside on one supported GPU. Other GPUs may remain
visible. For traced calls, select the intended supported device with
`jax.default_device`; the binding pins the CuTeDSL compiler architecture and
reads tuning properties from that device. Separate local calls do not require
a single-process job. The initial public target set is SM100/SM103.

`batch_invariant=True` requests the existing per-sequence scheduling mode.
The JAX tests check raw-byte replay and sequence isolation, including all
parameter and state gradients, separately from the FP64 numerical reference.
GPU numerical, determinism, and heterogeneous-device qualification for these
new bindings is pending; see the [determinism support page](/operations/determinism).

These bindings cover the four recurrent forward/backward operations. Summary
and context-parallel composition, paged/in-place state, and dedicated decode
or verify entry points are not exposed here. Install the frontend's JAX
dependency group and a CUDA-enabled JAX build; the bindings use
`cutlass.jax` and do not fall back to a Torch or cuTile implementation.
