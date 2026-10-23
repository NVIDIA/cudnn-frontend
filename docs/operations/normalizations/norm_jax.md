# LayerNorm and RMSNorm in JAX

`cudnn.ops.layer_norm_forward_jax` and `cudnn.ops.rms_norm_forward_jax`
return `(y,)`, a one-element tuple containing the normalized JAX array. They
use the existing SM100 CuTeDSL normalization kernels and support eager calls,
`jax.jit`, and first-order reverse-mode
differentiation through `jax.grad` and `jax.vjp`. PyTorch is not required.

These APIs are experimental.

## Signatures and semantics

```python
layer_norm_forward_jax(x, weight=None, bias=None, *, normalized_shape=None, eps=1e-5)
rms_norm_forward_jax(x, weight=None, *, normalized_shape=None, eps=1e-5)
```

Both functions normalize over the trailing dimensions specified by
`normalized_shape`. The default is `(x.shape[-1],)`. Explicit dimensions must
be positive, static, and match the corresponding dimensions of `x` exactly.
Leading dimensions identify independent normalization rows.

LayerNorm computes
`y = (x - mean(x)) / sqrt(mean((x - mean(x))**2) + eps)`, then applies
`y * weight + bias` when those parameters are present. RMSNorm computes
`y = x / sqrt(mean(x**2) + eps)` and optionally multiplies by `weight`.
The means in these formulas are over the normalized dimensions.

`x` has dtype `bfloat16`, `float16`, or `float32`. Each supplied affine
parameter must have shape `normalized_shape` and the same dtype as `x`.
LayerNorm accepts weight and bias independently; omitted weight means one and
omitted bias means zero. RMSNorm has no bias argument.

The returned array `y` has the same shape and dtype as `x`. Statistics and
scratch remain internal; neither function returns or mutates them. The native backward
accumulates parameter gradients in FP32; the VJP casts each gradient to its
primal parameter's dtype. Input gradients also match the input dtype.
The backward implementation is private and is invoked by JAX's reverse-mode
transformations; these APIs do not expose a separate backward function.

`normalized_shape` and `eps` are static configuration: close over them in a
jitted function, or mark them as static arguments. Arrays remain runtime
inputs. Use a finite positive `eps` within the FP32 representable range.

## Usage

Install the frontend and a CUDA-enabled JAX build. The frontend already depends
on CuTeDSL; these kernels require a build at least version 4.7.1 that provides
`cutlass.experimental.primitives`. No additional frontend dependency is introduced.

Run in a single process with exactly one visible GPU. Select the SM100 device
with `CUDA_VISIBLE_DEVICES` before importing JAX, for example
`CUDA_VISIBLE_DEVICES=0 python train.py`. Traced arguments do not carry concrete
device placement, so this restriction keeps capability checks and compilation
on the execution device.

```python
import jax
import jax.numpy as jnp
from cudnn.ops import layer_norm_forward_jax, rms_norm_forward_jax

x = jnp.arange(8192, dtype=jnp.float32).reshape(8, 1024) / 8192
weight = jnp.ones((1024,), dtype=x.dtype)
bias = jnp.zeros_like(weight)

y, = jax.jit(layer_norm_forward_jax)(x, weight, bias)
rms_y, = jax.jit(rms_norm_forward_jax)(x, weight)

def loss(x, weight, bias):
    y, = layer_norm_forward_jax(x, weight, bias, eps=1e-5)
    return jnp.square(y.astype(jnp.float32)).mean()

dx, dweight, dbias = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(
    x, weight, bias
)
jax.block_until_ready((y, rms_y, dx, dweight, dbias))
```

For multidimensional normalization, pass the complete trailing shape:

```python
normalize = jax.jit(
    lambda x, weight: rms_norm_forward_jax(
        x, weight, normalized_shape=(16, 64)
    )
)
y, = normalize(x.reshape(8, 16, 64), weight.reshape(16, 64))
```

## Support matrix

| Feature | Supported contract |
|---|---|
| GPU | One visible SM100 GPU in a single process |
| Operations | LayerNorm and RMSNorm |
| Input and affine dtypes | BF16, FP16, FP32; matching dtypes |
| Normalization axes | Positive static trailing dimensions; exact shape match |
| Tensor size | Fewer than 2^31 elements (native signed 32-bit indexing) |
| Affine parameters | Optional LayerNorm weight/bias; optional RMSNorm weight |
| Outputs | `(y,)`; `y` has the input shape and dtype |
| Execution | Eager and `jax.jit`; XLA-managed outputs and execution stream |
| Differentiation | First-order reverse mode for `x` and supplied affine parameters |

GroupNorm, InstanceNorm, BatchNorm, fused normalization operations, JVP,
higher derivatives, `vmap`, and distributed sharding are outside this API's
qualified scope. The functions do not expose a cuDNN graph-planning interface.

## Validation

The focused suite is
[`test/python/norm/jax/test_norm_jax.py`](../../test/python/norm/jax/test_norm_jax.py).
It exercises the public entry points against numerical references and checks
the documented input contract. Passing collection or an environment skip does
not establish GPU support; qualify the suite on SM100 with CUDA-enabled JAX
and the required CuTeDSL runtime.

Qualified on B200 with Python 3.12, JAX/jaxlib 0.9.1, CUDA 13, and the internal
CuTeDSL build `4.9.0a0.dev0+internal.master.20261002010000.3617316`.
Run the focused suite without the parent Torch
test fixtures:

```bash
python -m pytest test/python/norm/jax/ --confcutdir=test/python/norm/jax -q
```
