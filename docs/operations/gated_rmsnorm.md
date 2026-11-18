# SiLU-gated RMSNorm training

Experimental forward and backward APIs compute
`y = BF16(x * rsqrt(mean(x*x) + eps) * weight * SiLU(z))`.
Each row has 128 elements. The gate `z` is separate from `x`.

```python
from cudnn.ops import gated_rmsnorm_forward, gated_rmsnorm_backward

y = gated_rmsnorm_forward(x, z, weight)["output"]
dx, dz, dw = gated_rmsnorm_backward(dy, x, z, weight)
```

`x`, `z`, and `dy` are BF16 CUDA tensors of one positive shape `[T, 128]`
or `[T, H, 128]`. The last stride must be one; other non-singleton strides
must be multiples of 16 elements. All addresses must be 32-byte aligned.
`weight` is contiguous BF16 `[128]`. Inputs are read in place.

Forward returns `TupleDict(output=...)`. Backward returns
`TupleDict(grad_x=..., grad_z=..., grad_weight=...)`: the data gradients are
BF16 and the weight gradient is FP32. These are explicit forward/backward
functions; forward does not install an autograd rule.

The weight reduction has a fixed order. Repeated calls with the same inputs,
shape, device and build produce the same bits. Empty inputs, bias, other
widths, and non-native layouts raise before launch.

## Reusing a prepared plan

```python
from cudnn.norm.cake.gated_rmsnorm import GatedRMSNormPlan

plan = GatedRMSNormPlan(tuple(x.shape), device=x.device)
y = torch.empty_like(x, memory_format=torch.contiguous_format)
dx = torch.empty_like(y)
dz = torch.empty_like(y)
dw = torch.empty(128, dtype=torch.float32, device=x.device)
scratch = torch.empty(plan.get_workspace_size() // 4,
                      dtype=torch.float32, device=x.device)
plan.forward(x, z, weight, y)
plan.backward(dy, x, z, weight, dx, dz, dw, scratch)
```

Plan construction compiles the three required kernels. The plan owns no
device memory. Calls validate every binding, use preallocated outputs and
scratch, and launch on the current Torch stream or the optional `stream=`.
Outputs and scratch must not overlap each other or any input. Use separate
buffers for concurrent calls. The convenience functions allocate their
outputs and backward scratch on the selected stream.

## Requirements and source

SM100a, SM103a or SM107a; CUDA toolkit 13.4+; Torch and TVM-FFI.
The package contains CUDA sources and does not need FlashInfer or Miles at
runtime. CuTeDSL is not used by this operation.

The deterministic Cake package is copied without changes from
[miles-ext-kernels, commit f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_rmsnorm_silu_gate_det_train).
Its file hashes are recorded in `_cake/_vendor/VENDORED_GATED_RMSNORM.json`.
