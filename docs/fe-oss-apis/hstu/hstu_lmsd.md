# HSTU LayerNorm-Multiply-SiLU-Dropout (LMSD)

**This is an experimental API and subject to change.**

## Overview

HSTU LMSD is a fused training operation used by Hierarchical Sequential
Transduction Unit (HSTU) models. For each row of `x`, it computes LayerNorm,
applies the learned affine transform, multiplies the result by an optional
`SiLU(u)`, and optionally applies inverted dropout. With

$$
\hat{x} = (x - \mathrm{mean}(x))\mathrm{rstd}(x),
\qquad
\ell = \hat{x} \odot \mathrm{weight} + \mathrm{bias},
\qquad
a = \begin{cases}\mathrm{SiLU}(u), & \text{when activation is enabled} \\ u, & \text{otherwise}.\end{cases}
$$

the mandatory result is `dropout(ell * a)`. The output optionally prepends
`dropout(a)` and `dropout(x)` according to `concat_u` and `concat_x`:

$$
y = \left[
    \mathrm{dropout}(a)\ \text{if the activated-u segment is enabled},
    \mathrm{dropout}(x)\ \text{if the x segment is enabled},
    \mathrm{dropout}(\ell \odot a)
\right].
$$

When `dropout_ratio > 0`, the independent decisions are packed into one
`int8` mask per input element: bit 0 corresponds to the mandatory LMSD result,
bit 1 to the optional x segment, and bit 2 to the optional activated-u segment.
A set bit means that the element is kept. Disabled auxiliary segments leave
their bits clear. With `dropout_ratio=0`, Philox and mask traffic are compiled
out and forward returns `mask_tensor=None`. Forward also returns the row-wise
mean and reciprocal standard deviation needed by the explicit backward
operation.

This API does not register an autograd operator. Call
`hstu_lmsd_backward` explicitly with the tensors saved by
`hstu_lmsd_forward` and the matching four forward configuration parameters.

## Installation

Install cuDNN Frontend with the CuTe DSL optional dependencies and a supported
PyTorch installation:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
pip install torch torch-c-dlpack-ext
```

From a source checkout, the PyTorch dependencies can instead be installed with
`pip install --group torch`.

The forward and backward functions are available through lazy top-level
exports:

```python
from cudnn import hstu_lmsd_backward, hstu_lmsd_forward
```

## Supported configurations

| Property | Support |
| --- | --- |
| GPU architecture | SM10x |
| Input dtype | BF16 |
| Rows (`N`) | Positive, with `N * D <= 2^31` |
| Hidden dimension (`D`) | BF16 dimensions divisible by 8 in `8 <= D < 1024` |
| `eps` | Positive and finite |
| `dropout_ratio` | Finite and remains in `[0, 1)` after FP32 conversion |
| `apply_u_silu` | Apply SiLU to u when true; use u directly when false |
| `concat_u`, `concat_x` | Independently materialize the two auxiliary output segments |
| `compute_dweight` | Backward may compile out dWeight and its workspace |

All tensors must be CUDA tensors on the same device and have 16-byte-aligned
storage. Output tensors and backward workspaces must not overlap inputs or one
another.

### Tensor shapes and layouts

| Tensor | Shape | Dtype | Layout |
| --- | --- | --- | --- |
| `x`, `u` | `(N, D)` | BF16 | inner stride 1; each input independently supports a padded row stride |
| `weight`, `bias` | `(D,)` | BF16 | contiguous |
| `y`, `dy` | `(N, SD)` | BF16 | `S = 1 + concat_u + concat_x`; `y` is contiguous; `dy` may have a padded row stride |
| `mean`, `rstd` | `(N,)` | FP32 | contiguous |
| `mask` | `(N, D)` or `None` | `int8` | contiguous when dropout is enabled; `None` otherwise |
| `dx`, `du` | `(N, D)` | BF16 | contiguous |
| `dweight`, `dbias` | `(D,)` | BF16 | contiguous; dWeight may be `None` when disabled |

For BF16 matrices with a padded row stride, each row must remain 16-byte
aligned. A cached compiled implementation accepts any runtime `N` in the
supported range. The `x`, `u`, and `dy` row strides are runtime values and do
not participate in the compile-cache key; `D`, dtypes, devices, and feature
flags remain plan-time configuration.

The launch grid adapts to the runtime row count without changing the compiled
plan. Forward caps its persistent row blocks by the device SM count. Backward
uses at most one persistent tile per input row and caps large inputs at the
device SM count times 64 persistent CTAs per SM. The compiled vector width,
CTA shape, and backward row width are selected from `D`; they are not tied to
one model shape.

## Functions

The allocating functions cache compiled API objects for repeated calls with
the same tensor layout and configuration. The cache key excludes `N`, so one
compiled kernel is reused across supported row counts:

```python
import torch

from cudnn import hstu_lmsd_backward, hstu_lmsd_forward

n, d = 257, 128
x_storage = torch.randn((n, 3 * d), device="cuda", dtype=torch.bfloat16)
x = x_storage[:, :d]
u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
u = u_storage[:, :d]
weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

saved = hstu_lmsd_forward(
    x,
    u,
    weight,
    bias,
    eps=1e-6,
    dropout_ratio=0.1,
    seed=17,
)
y = saved["y_tensor"]

dy = torch.randn_like(y)
grads = hstu_lmsd_backward(
    dy,
    x,
    u,
    weight,
    bias,
    saved["mean_tensor"],
    saved["rstd_tensor"],
    saved["mask_tensor"],
    dropout_ratio=0.1,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
)
dx = grads["dx_tensor"]
du = grads["du_tensor"]
dweight = grads["dweight_tensor"]
dbias = grads["dbias_tensor"]
```

`hstu_lmsd_forward` returns a `TupleDict` in the order `y_tensor`,
`mean_tensor`, `rstd_tensor`, and `mask_tensor`. `hstu_lmsd_backward` returns
`dx_tensor`, `du_tensor`, `dweight_tensor`, and `dbias_tensor`. The backward
function accepts optional caller-owned gradient output tensors. Pass
`compute_dweight=False` for a non-trainable weight; the returned
`dweight_tensor` is then `None`, and neither its FP32 workspace nor its
reduction work is created.

Forward and backward must use the same `dropout_ratio`, `apply_u_silu`,
`concat_u`, `concat_x`, saved statistics, and optional packed mask. Pass all
four configuration values explicitly to `hstu_lmsd_backward`; output tensors
do not carry hidden Python metadata. The `seed` is a signed 64-bit integer.
Pass `stream=` to enqueue wrapper allocation and execution on a specific
`torch.cuda.Stream` or CUDA stream handle; `None` uses the current PyTorch CUDA
stream.

For example, this returns only the mandatory `LN(x) * u` segment, performs no
dropout work, and skips dWeight:

```python
saved = hstu_lmsd_forward(
    x,
    u,
    weight,
    bias,
    dropout_ratio=0.0,
    apply_u_silu=False,
    concat_u=False,
    concat_x=False,
)
grads = hstu_lmsd_backward(
    torch.randn_like(saved["y_tensor"]),
    x,
    u,
    weight,
    bias,
    saved["mean_tensor"],
    saved["rstd_tensor"],
    saved["mask_tensor"],  # None for this configuration
    dropout_ratio=0.0,
    apply_u_silu=False,
    concat_u=False,
    concat_x=False,
    compute_dweight=False,
)
assert grads["dweight_tensor"] is None
```

See the focused tests in [`test/python/fe_api/hstu/hstu_lmsd/`](../../../test/python/fe_api/hstu/hstu_lmsd/)
for complete function calls.
