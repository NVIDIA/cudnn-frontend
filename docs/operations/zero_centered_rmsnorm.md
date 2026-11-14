# Zero-centered RMSNorm training

Experimental explicit forward and backward APIs compute
`y = BF16(x * rsqrt(mean(x*x) + eps) * (1 + weight))`.
The zero-centered parameter is FP32. The effective weight `1 + weight`
is also computed in FP32, then consumed by the Cake kernel without conversion.

```python
from cudnn.ops import zero_centered_rmsnorm_forward, zero_centered_rmsnorm_backward

y, rstd = zero_centered_rmsnorm_forward(x, weight, eps=1e-6)
dx, dw = zero_centered_rmsnorm_backward(dy, x, weight, rstd)
```

Inputs `x` and `dy` are contiguous BF16 CUDA tensors with the same positive
shape `[..., H]` and at least two dimensions. `weight` is contiguous FP32
`[H]`. Data and parameter addresses must be 32-byte aligned. The product of
all dimensions must fit signed int32. No input is copied or converted.

Forward returns `TupleDict(output=..., rstd=...)`, where `rstd` is FP32 with
one element per flattened row. Backward uses that saved `rstd` and returns
`TupleDict(grad_x=..., grad_weight=...)`, with BF16 data gradients and FP32
parameter gradients. Pass the same x and weight values used by forward.
These functions do not install an autograd rule. The parameter gradient uses
a fixed reduction order; repeated calls with the same inputs, device and
build produce the same bits.

## Prepared execution

```python
from cudnn.norm.cake.zero_centered_rmsnorm import ZeroCenteredRMSNormPlan

plan = ZeroCenteredRMSNormPlan(tuple(x.shape), device=x.device)
y = torch.empty_like(x)
rstd = torch.empty(x.numel() // x.shape[-1], dtype=torch.float32, device=x.device)
dx = torch.empty_like(x)
dw = torch.empty_like(weight)
workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
plan.forward(x, weight, y, rstd, workspace)
plan.backward(dy, x, weight, rstd, dx, dw, workspace)
```

Construction selects and compiles the kernels. Execution uses caller-owned
outputs and a contiguous, 256-byte-aligned workspace. Each call writes the
FP32 effective weight into scratch. Backward clears its device counters before
the Cake launches. These stages allocate no tensors and perform no host reads
or synchronization. The prepared calls support CUDA Graph capture and an
optional `stream=`. Outputs and workspace must not overlap each other or any
input. Concurrent calls need separate scratch. Convenience wrappers allocate
outputs and scratch on the selected stream.

## Support

Requires Torch, TVM-FFI, CUDA toolkit 13.4+, and SM100a/SM103a/SM107a.
Widths are 256, 512, 1536, 2048, 4096, 5120 and 6144, subject to the vendored
architecture route table. SM107a width 4096 is rejected: this API does not yet
enable its oversized shared-memory mode. Empty inputs, BF16 weights, residual
fusion and noncontiguous inputs are rejected.

Sources are copied unchanged from
[miles-ext-kernels at f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_rmsnorm_det_train).
Build manifests, licenses and file hashes are included under
`cudnn/_cake/_vendor`; no Miles or FlashInfer installation is needed.
