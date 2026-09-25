# FFT Causal Conv1d

The cuDNN Frontend Python API exposes FFT-based depthwise causal convolution
through:

```python
y = cudnn.ops.fft_causal_conv1d(x, weight)
```

This API requires cuDNN 9.26.0 or newer. It supports FP16, BF16, FP32, and FP64
CUDA tensors, `torch.autograd`, and `torch.compile`.

## Operation

For `x` shaped `(batch, dim, seq_len)` and `weight` shaped
`(dim, kernel_size)`, the operation computes:

```text
y[b, c, t] = sum(x[b, c, t - j] * weight[c, j], j=0..kernel_size-1)
```

Here, `x[b, c, t - j]` is zero when `t - j` is outside the input sequence.
This is FIR-order weight storage:
`weight[0]` multiplies the current sample. It is reversed relative to
`cudnn.ops.causal_conv1d`, whose first stored weight multiplies the oldest
sample in the causal window.

## Path Selection

The convenience wrapper follows cuhyena's Python API:

- It rounds `kernel_size` up to a power of two, with a minimum of 128.
- It selects the medium FFT path when the filter fits that path's dtype and
  device limit.
- It right-pads the input to a multiple of the medium filter length.
- Otherwise it right-pads input and filter to one common power-of-two length
  and selects the long FFT path.
- It trims the output and autograd gradients back to the caller's shapes.

The medium path supports FP64 filters through 4096. Other dtypes support
filters through 8192 before SM90 and through 16384 on SM90 or newer.

The raw long backend path requires power-of-two `seq_len == kernel_size` in
`[4096, 16777216]`. FP64 supports lengths through 8388608; length 16777216 for
other dtypes requires SM90 or newer.

## Long FFT Buffers

Long FFT forward queries and allocates two opaque byte buffers:

- Workspace is temporary scratch for the current forward or backward call.
- Reserve space stores transformed signal and filter state from forward and is
  retained by the autograd context for the matching backward call.

Applications calling the C APIs directly must preserve this same reserve-space
lifetime. See
`samples/cpp/causal_conv1d/fft_causal_conv1d.cpp` for medium and long C API
examples.

## Example

```python
import cudnn
import torch

x = torch.randn(2, 16, 4096, device="cuda", requires_grad=True)
weight = torch.randn(16, 256, device="cuda", requires_grad=True)

y = cudnn.ops.fft_causal_conv1d(x, weight)
y.sum().backward()
```
