# Causal Conv1d

Depthwise causal 1-D convolution with optional fused activation:

$$ y = \text{activation}(\text{conv1d\_causal}(x, w) + b) $$

Causal padding: $(K - 1)$ zeros on the left, $0$ on the right, where $K$ is the kernel size. Each channel is convolved independently with its own 1-D filter (depthwise).

Supports forward and backward passes with `torch.autograd` and `torch.compile`.

## Support

- **Architectures**: Blackwell (SM100+)
- **Data types**: FP32, FP16, BF16
- **Activations**: `identity`, `silu`

## Python API

The high-level entry point is `cudnn.ops.causal_conv1d`:

```python
import cudnn

y = cudnn.ops.causal_conv1d(x, weight, bias=None, activation="identity")
```

**Args:**
- `x` (torch.Tensor): Input tensor of shape $(B, D, L)$. Must be contiguous and on CUDA.
- `weight` (torch.Tensor): Filter tensor of shape $(D, K)$. Same dtype as `x`.
- `bias` (torch.Tensor | None): Optional bias of shape $(D,)$. Same dtype as `x`. Defaults to zeros if `None`.
- `activation` (str): `"identity"` (default) or `"silu"`.

**Returns:**
- `y` (torch.Tensor): Output of shape $(B, D, L)$, same dtype as `x`.

Where:
- $B$ is the batch size
- $D$ is the number of channels (convolution is depthwise)
- $L$ is the sequence length
- $K$ is the kernel size

### Low-level primitives

The forward and backward C-level bindings are re-exported at the top level:

- `cudnn.causal_conv1d_forward(stream, x_ptr, weight_ptr, bias_ptr, out_ptr, batch, dim, seq_len, kernel_size, data_type, activation)`
- `cudnn.causal_conv1d_backward(stream, x_ptr, weight_ptr, bias_ptr, dy_ptr, dx_ptr, dweight_ptr, dbias_ptr, batch, dim, seq_len, kernel_size, data_type, dw_data_type, activation)`

In most cases you should use `cudnn.ops.causal_conv1d` instead, which handles autograd, `torch.compile`, and tensor management automatically.

### Tensors

#### Input Tensors (Forward)

| Tensor | Device | Data Type | Shape |
|--------|--------|-----------|-------|
| x | GPU | FP32, FP16, or BF16 | $(B, D, L)$ |
| weight | GPU | same as x | $(D, K)$ |
| bias | GPU | same as x | $(D,)$ |

#### Output Tensors (Forward)

| Tensor | Device | Data Type | Shape |
|--------|--------|-----------|-------|
| y | GPU | same as x | $(B, D, L)$ |

#### Additional Input Tensors (Backward)

| Tensor | Device | Data Type | Shape |
|--------|--------|-----------|-------|
| grad_out (dy) | GPU | same as x | $(B, D, L)$ |

#### Output Tensors (Backward)

| Tensor | Device | Data Type | Shape |
|--------|--------|-----------|-------|
| dx | GPU | same as x | $(B, D, L)$ |
| dweight | GPU | same as x | $(D, K)$ |
| dbias | GPU | same as x | $(D,)$ |

## Example

```python
import torch
import cudnn

B, D, L, K = 2, 768, 4096, 4

x = torch.randn(B, D, L, dtype=torch.bfloat16, device="cuda", requires_grad=True)
w = torch.randn(D, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
b = torch.randn(D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

y = cudnn.ops.causal_conv1d(x, w, bias=b, activation="silu")

loss = y.sum()
loss.backward()  # x.grad, w.grad, b.grad are populated
```

## Samples

- C++ sample: [samples/cpp/causal_conv1d](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/causal_conv1d)
