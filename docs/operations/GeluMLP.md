# GELU MLP

`cudnn.gemm.ops.gelu_mlp` is a PyTorch-facing cuDNN implementation of the
dense feed-forward block used by DiT and ViT models:

```python
from cudnn.gemm.ops import gelu_mlp

y = gelu_mlp(x, w1, b1, w2, b2)
```

Its observable computation matches two ordinary `torch.nn.Linear` layers
separated by tanh-approximate GELU:

```python
h = torch.nn.functional.gelu(
    torch.nn.functional.linear(x, w1, b1), approximate="tanh"
)
y = torch.nn.functional.linear(h, w2, b2)
```

`x` has shape `[..., H]`; weights use the standard `nn.Linear` layouts
`w1[I, H]` and `w2[O, I]`; biases have shapes `b1[I]` and `b2[O]`. The output
has shape `[..., O]`.

## Fusion and numerical boundaries

Forward executes two cuDNN graphs. The first fuses the first matrix multiply,
column bias, and tanh-GELU. The second fuses the output matrix multiply and
bias. The post-bias first-layer value is rounded to BF16 before GELU, matching
the visible boundary of eager BF16 `Linear`; GEMMs accumulate in FP32.

First-order autograd is supported for all five inputs. Backward fuses
`dout @ w2` with GELU backward, uses cuDNN GEMMs for activation and weight
gradients, and currently uses PyTorch reductions for bias gradients. Higher
order gradients are not supported and fail explicitly.

## Current support

- SM100 GPUs
- contiguous BF16 CUDA tensors on one device
- rank-two-or-higher `x`
- tanh-approximate GELU, with biases and no dropout inside the operation
- eager PyTorch execution and first-order autograd

The first call for a new shape/device/stream builds and autotunes plans, so warm
the operation before timing. This release does not claim `torch.compile` or
CUDA Graph capture compatibility. Unsupported shapes, layouts, dtypes,
devices, or architectures fail rather than copying, converting, or falling
back silently.
