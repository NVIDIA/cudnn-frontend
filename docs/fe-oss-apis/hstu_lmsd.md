# HSTU LayerNorm-Multiply-SiLU-Dropout (LMSD)

**This is an experimental API and subject to change.**

## Overview

HSTU LMSD is a fused training operation used by Hierarchical Sequential
Transduction Unit (HSTU) models. For each row of `x`, it computes LayerNorm,
applies the learned affine transform, multiplies the result by `SiLU(u)`, and
applies inverted dropout. With

$$
\hat{x} = (x - \operatorname{mean}(x))\operatorname{rstd}(x),
\qquad
\ell = \hat{x} \odot \operatorname{weight} + \operatorname{bias},
\qquad
s = \operatorname{SiLU}(u),
$$

the forward output is the concatenation

$$
y = \left[
    \operatorname{dropout}(s),
    \operatorname{dropout}(x),
    \operatorname{dropout}(\ell \odot s)
\right].
$$

The three dropout decisions are independent and packed into one `int8` mask
per input element: bit 0 corresponds to `dropout(ell * s)`, bit 1 to
`dropout(x)`, and bit 2 to `dropout(s)`. A set bit means that the element is
kept. Forward also returns the row-wise mean and reciprocal standard deviation
needed by the explicit backward operation.

This API does not register an autograd operator. Call
`hstu_lmsd_backward` explicitly with the tensors saved by
`hstu_lmsd_forward`.

## Installation

Install cuDNN Frontend with the CuTe DSL optional dependencies and a supported
PyTorch installation:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
pip install torch torch-c-dlpack-ext
```

From a source checkout, the PyTorch dependencies can instead be installed with
`pip install --group torch`.

The functions and explicit class APIs are available through lazy top-level
exports:

```python
from cudnn import (
    HSTULMSDBwdSm100,
    HSTULMSDFwdSm100,
    hstu_lmsd_backward,
    hstu_lmsd_forward,
)
```

## Supported configurations

| Property | Support |
| --- | --- |
| GPU architecture | SM10x |
| Input dtype | BF16 |
| Rows (`N`) | `1 <= N <= 4,194,304` |
| Hidden dimension (`D`) | `512` only |
| `eps` | Positive and finite |
| `dropout_ratio` | Finite and remains in `[0, 1)` after FP32 conversion |

All tensors must be CUDA tensors on the same device and have 16-byte-aligned
storage. Output tensors and backward workspaces must not overlap inputs or one
another.

### Tensor shapes and layouts

| Tensor | Shape | Dtype | Layout |
| --- | --- | --- | --- |
| `x` | `(N, 512)` | BF16 | row stride 512, inner stride 1 |
| `u` | `(N, 512)` | BF16 | inner stride 1; padded row strides are supported |
| `weight`, `bias` | `(512,)` | BF16 | contiguous |
| `y`, `dy` | `(N, 1536)` | BF16 | `y` is contiguous; `dy` may have a padded row stride |
| `mean`, `rstd` | `(N,)` | FP32 | contiguous |
| `mask` | `(N, 512)` | `int8` | contiguous |
| `dx`, `du` | `(N, 512)` | BF16 | contiguous |
| `dweight`, `dbias` | `(512,)` | BF16 | contiguous |

For BF16 matrices with a padded row stride, each row must remain 16-byte
aligned. A compiled class API accepts any runtime `N` in the supported range;
`D`, strides, dtypes, and devices must match the descriptors used to construct
it.

## High-level functions

The allocating functions cache compiled API objects for repeated calls with
the same tensor layout and configuration. The cache key excludes `N`, so one
compiled kernel is reused across supported row counts:

```python
import torch

from cudnn import hstu_lmsd_backward, hstu_lmsd_forward

n, d = 257, 512
x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
u = u_storage[:, :d]  # A padded row stride is supported for u.
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
)
dx = grads["dx_tensor"]
du = grads["du_tensor"]
dweight = grads["dweight_tensor"]
dbias = grads["dbias_tensor"]
```

`hstu_lmsd_forward` returns a `TupleDict` in the order `y_tensor`,
`mean_tensor`, `rstd_tensor`, and `mask_tensor`. `hstu_lmsd_backward` returns
`dx_tensor`, `du_tensor`, `dweight_tensor`, and `dbias_tensor`. The backward
function accepts optional caller-owned gradient output tensors; it always owns
and allocates its internal FP32 reduction workspaces.

Forward and backward must use the same `dropout_ratio`, saved statistics, and
packed mask. The convenience forward records its ratio on the returned mask;
the convenience backward uses that value by default and rejects an explicitly
supplied mismatch. If the mask was copied, serialized, or produced through the
explicit class API, pass the matching `dropout_ratio` explicitly. The `seed` is
a signed 64-bit integer. Pass `stream=` to enqueue wrapper allocation and
execution on a specific `torch.cuda.Stream` or CUDA stream handle; `None` uses
the current PyTorch CUDA stream.

## Explicit class APIs

`HSTULMSDFwdSm100` and `HSTULMSDBwdSm100` expose the explicit lifecycle for
callers that need precompiled, allocation-free execution:

1. Construct the API with sample inputs, outputs, and configuration.
2. Call `check_support()`.
3. Call `compile()` before a latency-sensitive region or CUDA Graph capture.
4. Call `execute()` with matching runtime tensors and optional
   `current_stream=`.

Forward requires caller-owned `y`, `mean`, `rstd`, and `mask` outputs. Backward
requires caller-owned `dx`, `du`, `dweight`, and `dbias` outputs, plus two FP32
reduction workspaces with shape `(13568, 512)`. Backward launches the main
gradient kernel followed by the dWeight/dBias reduction kernel on the selected
stream and does not recompute `y`.

See the focused tests in [`test/python/fe_api/hstu_lmsd/`](../../test/python/fe_api/hstu_lmsd/)
for complete class and function calls. Benchmark workloads and measurement
details are in the [HSTU LMSD benchmark README](../../benchmark/hstu_lmsd/README.md).
