# Cake mHC training

The mHC APIs launch the deterministic Miles Cake kernels for four-stream
manifold-constrained hyper-connections. They require PyTorch, TVM FFI, and CUDA
13.4 or newer on SM100/103/107. H is 4096, 5120, or 7168 on SM100/103 and
4096 on SM107. X is contiguous BF16 `[T,4,H]`, with positive T and int32 element
offsets. Mapping parameters must be FP32: `fn[24,4H]`, `scale[3]`, and `base[24]`.
All tensor pointers must be 32-byte aligned.

The external branch remains caller-owned. Explicit forward and backward stages
let callers place any branch between pre-mix and post-mix:

```python
from cudnn.ops.mhc import (pre_mix_forward, post_mix_forward,
                           post_mix_backward, pre_mix_backward)

u, stats = pre_mix_forward(x, fn, scale, base)
b = branch(u)
y = post_mix_forward(x, b, stats)["out"]

db, grad_stats = post_mix_backward(x, b, stats, dy)
du = branch_backward(db)
dx, dfn, dscale, dbase = pre_mix_backward(
    x, dy, du, fn, scale, base, stats, grad_stats)
```

Pre-mix forward normalizes the flattened residual streams, applies the mapping matrix,
forms sigmoid pre/post coefficients and a 4x4 Sinkhorn mixing matrix, and returns
the BF16 collapsed branch input. Post-mix forward returns the BF16 residual/branch
mixture.
`rms_eps=1e-5`, `hc_eps=1e-6`, and `sinkhorn_iters=20` are defaults; positive finite
epsilons and integer Sinkhorn counts 1..64 are supported. Pass the same constants
to `pre_mix_backward`. The default count uses the unrolled kernels; other counts use
the native runtime-count kernels.

Save the FP32 `stats[T,64]` record unchanged. `post_mix_backward` returns BF16
`grad_branch[T,H]` and FP32 `grad_stats[T,32]`. After the external branch backward,
`pre_mix_backward` returns BF16 `dx` and FP32 `dfn`, `dscale`, and `dbase`. The
parameter reductions use fixed orders and overwrite their output buffers.

`head_forward(x)` returns the BF16 mean over four streams, accumulated in
FP32. `head_backward(grad_out[T,H])` broadcasts BF16(grad_out/4) to `[T,4,H]`.
All APIs return `TupleDict` values in the order shown; single-output APIs use
`"out"` as their key.

Every API accepts explicit output buffers and `stream=`. `pre_mix_forward` and
`pre_mix_backward` also accept `workspace`, a contiguous CUDA uint8 vector aligned to 256
bytes. Query its size with `workspace_size(T,H,device=x.device)` or
`backward=True`. Each call resets its launch ticket in caller scratch. Scratch and
outputs must be disjoint from all inputs and from each other. Do not reuse one
scratch buffer for concurrent calls.

Warm the exact geometry/count before CUDA graph capture. With every output and
workspace supplied, execution makes no tensor allocations and reads no tensor
values on the host. Captured input values may change; keep all captured storage
alive. Native plans retain no caller tensor storage. These APIs use explicit
backward and have no autograd or cuDNN graph-engine registration. Nondeterministic
mHC variants and gradient accumulation are not exposed.

### JAX

`block_forward_jax` runs the same native Cake kernels under `jax.jit` and first-order
reverse-mode differentiation. Pass the external branch as a static callable and
its trainable arrays as explicit arguments:

```python
import jax
import jax.numpy as jnp
from cudnn.ops.mhc import block_forward_jax, head_forward_jax

def branch(u, weight):
    return jnp.tanh(u.astype(jnp.float32) * weight).astype(jnp.bfloat16)

@jax.jit
def loss(x, fn, scale, base, weight):
    y = block_forward_jax(x, fn, scale, base, branch, weight)
    head, = head_forward_jax(y)
    return jnp.sum(head.astype(jnp.float32))

grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(x, fn, scale, base, weight)
```

The branch must return BF16 `[T,H]`. Its arguments may be JAX pytrees. Backward
runs native post-mix backward, the branch VJP, then native pre-mix backward. This
keeps the single BF16 rounding of the combined input gradient. Pass trainable
branch parameters explicitly; differentiating arrays captured in a closure is
unsupported. The branch is recomputed for its VJP.

Explicit JIT-compatible stages are `pre_mix_forward_jax`, `post_mix_forward_jax`,
`post_mix_backward_jax`, and `pre_mix_backward_jax`. They take the same
input arrays and scalar options as their Torch peers and return plain tuples in
the same order. Use `block_forward_jax` for automatic differentiation across these stages.
`head_forward_jax` supports automatic differentiation;
`head_backward_jax` exposes its explicit gradient. Both return one-element
tuples.

JAX owns output and scratch buffers, and the native XLA FFI handler launches on
the XLA stream. Building the bridge requires a C++17 host compiler. Compiled
executables are process-local; cross-process export is unsupported. There are no Python execution callbacks or tensor conversions to
Torch. The pinned native package still uses Torch for build-time device queries.
Scalar options are static under JIT. Forward-mode AD, higher derivatives, vmap,
sharding, and multi-process execution are unsupported. Device support and input
geometry match the native APIs above.
