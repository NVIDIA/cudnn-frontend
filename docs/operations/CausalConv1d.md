# Causal Conv1d

The NHW and NWH APIs compute depthwise causal 1-D convolution with optional fused activation:

$$ y = \text{activation}(\text{conv1d\_causal}(x, w) + b) $$

Causal padding: $(K - 1)$ zeros on the left, $0$ on the right, where $K$ is the kernel size. Each channel is convolved independently with its own 1-D filter (depthwise). The B2B API fuses two causal convolutions with a fixed gating sequence described below.

Supports forward and backward passes with `torch.autograd` and `torch.compile`.

## Support

- **Architectures**: Turing (SM75) or later
- **Data types**: FP32, FP16, BF16
- **Activations**: `identity` and `silu` for NHW and NWH; B2B uses fixed gating

### Kernel sizes

The supported kernel-size range depends on the layout and, for the fused B2B operation, the role of the kernel:

| Frontend entry point | Kernel role | Supported kernel size |
|---|---|---:|
| `cudnn.ops.causal_conv1d` | NHW convolution | 2–256 |
| `cudnn.ops.causal_conv1d_nwh` | NWH convolution | 2–128 |
| `cudnn.ops.b2b_causal_conv1d` | Projection | 2–32 |
| `cudnn.ops.b2b_causal_conv1d` | Mixer | 2–256 |

The same limits apply to forward and backward execution.

## Python API

Three high-level Python APIs are available for NHW, NWH, and fused back-to-back (B2B) causal convolution.

### `cudnn.ops.causal_conv1d`

Runs depthwise causal convolution with NHW tensor layout:

```python
import cudnn

y = cudnn.ops.causal_conv1d(x, weight, bias=None, activation="identity")
```

**Minimum cuDNN version:** 9.22.0.

**Supported kernel size:** 2–256, inclusive.

**Args:**
- `x` (torch.Tensor): Input tensor of shape $(B, D, L)$. Must be on CUDA.
- `weight` (torch.Tensor): Filter tensor of shape $(D, K)$. Same dtype as `x`. $K$ must be between 2 and 256, inclusive.
- `bias` (torch.Tensor | None): Optional bias of shape $(D,)$. Same dtype as `x`. Defaults to zeros if `None`.
- `activation` (str): `"identity"` (default) or `"silu"`.

**Returns:**
- `y` (torch.Tensor): Output of shape $(B, D, L)$, same dtype as `x`.

The API supports `torch.autograd` and `torch.compile` for forward and backward execution. Backward computes `dx` with shape $(B, D, L)$, `dweight` with shape $(D, K)$, and `dbias` with shape $(D,)$. The supported kernel-size range remains 2–256 for both directions.

Where:
- $B$ is the batch size
- $D$ is the number of channels (convolution is depthwise)
- $L$ is the sequence length
- $K$ is the kernel size

See the [NHW forward notebook](../../samples/python/60_causal_conv1d_forward.ipynb) and [NHW backward notebook](../../samples/python/61_causal_conv1d_backward.ipynb) for PyTorch references and numerical comparisons.

### `cudnn.ops.causal_conv1d_nwh`

Runs the same depthwise causal convolution with NWH tensor layout. NWH stores the channel dimension last, so input and output use $(B, L, D)$ while the filter uses $(K, D)$. The operation is equivalent to:

```python
import torch

K, D = weight.shape
x_nhw = x.transpose(1, 2)                             # (B, D, L)
weight_nhw = weight.T.unsqueeze(1)                    # (D, 1, K)
x_padded = torch.nn.functional.pad(x_nhw, (K - 1, 0))
y_nhw = torch.nn.functional.conv1d(
    x_padded, weight_nhw, bias=bias, groups=D
)
if activation == "silu":
    y_nhw = torch.nn.functional.silu(y_nhw)
y = y_nhw.transpose(1, 2)                             # (B, L, D)
```

Each channel is convolved independently, and causal padding adds $K-1$ elements on the left and none on the right.

The public API call is:

```python
import cudnn

y = cudnn.ops.causal_conv1d_nwh(x, weight, bias=None, activation="identity")
```

**Minimum cuDNN version:** 9.24.0.

**Supported kernel size:** 2–128, inclusive.

**Args:**
- `x` (torch.Tensor): Input tensor of shape $(B, L, D)$. Must be on CUDA.
- `weight` (torch.Tensor): Filter tensor of shape $(K, D)$. Same dtype as `x`. $K$ must be between 2 and 128, inclusive.
- `bias` (torch.Tensor | None): Optional bias of shape $(D,)$. Same dtype as `x`. Defaults to zeros if `None`.
- `activation` (str): `"identity"` (default) or `"silu"`.

**Returns:**
- `y` (torch.Tensor): Output of shape $(B, L, D)$, same dtype as `x`.

The API supports `torch.autograd` and `torch.compile` for forward and backward execution. Backward computes `dx` with shape $(B, L, D)$, `dweight` with shape $(K, D)$, and `dbias` with shape $(D,)$. The supported kernel-size range remains 2–128 for both directions.

See the [NWH forward notebook](../../samples/python/62_causal_conv1d_nwh_forward.ipynb) and [NWH backward notebook](../../samples/python/63_causal_conv1d_nwh_backward.ipynb) for PyTorch references and numerical comparisons.

### `cudnn.ops.b2b_causal_conv1d`

Fuses projection convolution, gating, mixer convolution, a skip connection, and post-gating into a single kernel launch. The three projection channels for each output dimension are interleaved:

```python
proj = causal_conv1d(x, weights_proj)                       # (B, 3D, L)
gated = proj[:, 1::3, :] * proj[:, 2::3, :]                # (B, D, L)
y = causal_conv1d(gated, weights_mixer)                    # (B, D, L)
y = y + skip_bias[None, :, None] * gated
y_gated = y * proj[:, 0::3, :]                             # (B, D, L)
```

The API returns the final post-gated output `y_gated`:

```python
import cudnn

y_gated = cudnn.ops.b2b_causal_conv1d(x, weights_proj, weights_mixer, skip_bias)
```

**Minimum cuDNN version:** 9.24.0.

**Supported kernel sizes:**
- Projection kernel: 2–32, inclusive.
- Mixer kernel: 2–256, inclusive.

**Args:**
- `x` (torch.Tensor): Input tensor of shape $(B, 3D, L)$. Must be on CUDA.
- `weights_proj` (torch.Tensor): Projection filter of shape $(3D, K_{proj})$. Same dtype as `x`. $K_{proj}$ must be between 2 and 32, inclusive.
- `weights_mixer` (torch.Tensor): Mixer filter of shape $(D, K_{mixer})$. Same dtype as `x`. $K_{mixer}$ must be between 2 and 256, inclusive.
- `skip_bias` (torch.Tensor): Skip-connection bias of shape $(D,)$. Same dtype as `x`.

**Returns:**
- `y_gated` (torch.Tensor): Post-gated output of shape $(B, D, L)$, same dtype as `x`.

The API supports `torch.autograd` and `torch.compile` for forward and backward execution. Backward computes `dx` with shape $(B, 3D, L)$, `dweights_proj` with shape $(3D, K_{proj})$, `dweights_mixer` with shape $(D, K_{mixer})$, and `dskip_bias` with shape $(D,)$. The projection and mixer kernel-size ranges are unchanged for backward.

See the [B2B forward notebook](../../samples/python/64_b2b_causal_conv1d_forward.ipynb) and [B2B backward notebook](../../samples/python/65_b2b_causal_conv1d_backward.ipynb) for decomposed PyTorch references and numerical comparisons.

### Low-level bindings

The forward and backward C-level bindings are re-exported at the top level:

NHW layout, requiring cuDNN 9.22.0 or later and supporting kernel size 2–256:

- `cudnn.causal_conv1d_forward(stream, x_ptr, weight_ptr, bias_ptr, out_ptr, batch, dim, seq_len, kernel_size, data_type, activation)`
- `cudnn.causal_conv1d_backward(stream, x_ptr, weight_ptr, bias_ptr, dy_ptr, dx_ptr, dweight_ptr, dbias_ptr, batch, dim, seq_len, kernel_size, data_type, dw_data_type, activation)`

NWH layout, requiring cuDNN 9.24.0 or later and supporting kernel size 2–128:

- `cudnn.causal_conv1d_nwh_forward(stream, x_ptr, weight_ptr, bias_ptr, out_ptr, batch, dim, seq_len, kernel_size, data_type, activation)`
- `cudnn.causal_conv1d_nwh_backward(stream, x_ptr, weight_ptr, bias_ptr, dy_ptr, dx_ptr, dweight_ptr, dbias_ptr, batch, dim, seq_len, kernel_size, data_type, dw_data_type, activation)`

B2B, requiring cuDNN 9.24.0 or later, with projection kernel size 2–32 and mixer kernel size 2–256:

- `cudnn.b2b_causal_conv1d_forward(stream, x_ptr, weights_proj_ptr, weights_mixer_ptr, skip_bias_ptr, y_ptr, y_gated_ptr, batch, dim, seq_len, kernel_size_proj, kernel_size_mixer, data_type)`
- `cudnn.b2b_causal_conv1d_backward(stream, x_ptr, weights_proj_ptr, weights_mixer_ptr, skip_bias_ptr, y_ptr, dy_ptr, dx_ptr, dweights_proj_ptr, dweights_mixer_ptr, dskip_bias_ptr, batch, dim, seq_len, kernel_size_proj, kernel_size_mixer, data_type, dw_data_type)`

The low-level B2B forward binding writes both the mixer-plus-skip intermediate `y` and the final post-gated `y_gated`; the high-level API returns only `y_gated`.

In most cases, use the corresponding `cudnn.ops` API, which handles autograd, `torch.compile`, and tensor management automatically.
