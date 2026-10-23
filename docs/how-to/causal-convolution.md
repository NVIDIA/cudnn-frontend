# Run Causal Conv1d operators

Use `cudnn.ops` Causal Conv1d operators when their fixed CUDA layouts and fused autograd registrations match the model. Use Graph convolution for general N-D padding, dilation, or layout requirements.

## Before you begin

You need CUDA-enabled PyTorch and a package built with the optional causal
convolution bindings. Check `hasattr(cudnn, "causal_conv1d_forward")` before
registering this path. Use the [convolution tutorial](/tutorials/convolution)
when the optional binding is absent or its fixed layouts do not fit the model.

## causal_conv1d

`cudnn.ops.causal_conv1d(x, weight, bias=None, activation="identity")` computes depthwise causal convolution with `K - 1` zeros on the left and no right padding.

| Value | Shape | Requirements |
| --- | --- | --- |
| `x` | `[batch, channels, sequence]` | CUDA, FP32/FP16/BF16; copied if noncontiguous |
| `weight` | `[channels, kernel_size]` | Same device and dtype as X |
| `bias` | `[channels]` | Optional; same device and dtype; zeros when omitted |
| result | `[batch, channels, sequence]` | Same dtype and device as X |

`activation` is `"identity"` or `"silu"`. Each channel has an independent filter. The registered backward returns gradients for X, weight, and bias; the wrapper participates in eager autograd, fake-tensor propagation, and `torch.compile`.

## causal_conv1d_nwh

`cudnn.ops.causal_conv1d_nwh(x, weight, bias=None, activation="identity")` performs the same operation in NWH storage:

| Value | Shape |
| --- | --- |
| `x` | `[batch, sequence, channels]` |
| `weight` | `[kernel_size, channels]` |
| `bias` | `[channels]` |
| result | `[batch, sequence, channels]` |

The device, dtype, activation, fake-tensor, autograd, and compilation rules are the same as for `causal_conv1d`. The two wrappers are layout-specific. They call `.contiguous()` before the raw binding, so a transposed view is accepted but may allocate and copy; use the native layout when that cost matters.

## b2b_causal_conv1d

`cudnn.ops.b2b_causal_conv1d(x, weights_proj, weights_mixer, skip_bias)` fuses a projection convolution, Q/K gate, mixer convolution with a skip term, and final V gate.

| Value | Shape |
| --- | --- |
| `x` | `[batch, 3 * channels, sequence]` |
| `weights_proj` | `[3 * channels, projection_kernel]` |
| `weights_mixer` | `[channels, mixer_kernel]` |
| `skip_bias` | `[channels]` |
| result | `[batch, channels, sequence]` |

All tensors must be CUDA tensors on one device with one dtype. The channel extent of X must be divisible by three. Backward computes gradients for all four inputs using the saved projected intermediate.

## Low-level pointer bindings

When the compiled extension was built with the optional kernels, top-level functions named `causal_conv1d_forward`, `causal_conv1d_backward`, `causal_conv1d_nwh_forward`, `causal_conv1d_nwh_backward`, `b2b_causal_conv1d_forward`, and `b2b_causal_conv1d_backward` accept a CUDA stream, raw device pointers, dimensions, dtype codes, and operation-specific options. Test availability with `hasattr(cudnn, "causal_conv1d_forward")` before registering a production path.

These bindings do not validate PyTorch ownership, allocate outputs, register fake behavior, or connect autograd. Use them only when the application already owns buffer lifetime and stream ordering. Keep every pointer alive until the supplied stream completes.

## Test eager, backward, and compile paths

The canonical sample runs the high-level channel-first operator, checks forward values and gradients against PyTorch convolution, then executes a compiled wrapper:

<Code
  src="../samples/python/causal_conv1d.py"
  language="python"
  maxLines={0}
/>

Current source: [PyTorch operators and registrations](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/ops/causal_conv1d.py#L35).

## Verification limits

The recorded Python extension did not export `causal_conv1d_forward` or the
other five optional bindings, so the high-level operator raised `AttributeError`
before launch. The wrapper delegates architecture checks to those compiled
bindings and does not expose a separate support query. The raw-pointer
functions also do not have individual API pages.

## Expected result

When the optional bindings are present, the Python sample checks eager output
and gradients against PyTorch and runs the operator through `torch.compile`.
There is no C++ counterpart because this guide covers a PyTorch dispatcher
operator; use C++ Graph convolution for a language-neutral implementation.
