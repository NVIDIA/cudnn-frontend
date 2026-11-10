# Cake mHC training

The mHC APIs launch the deterministic Miles Cake kernels for four-stream
manifold-constrained hyper-connections. They require PyTorch, TVM FFI, and CUDA
13.4 or newer on SM100/103/107. H is 4096, 5120, or 7168 on SM100/103 and
4096 on SM107. X is contiguous BF16 `[T,4,H]`, with positive T and int32 element
offsets. Mapping parameters must be FP32: `fn[24,4H]`, `scale[3]`, and `base[24]`.
All tensor pointers must be 32-byte aligned.

The external branch remains caller-owned. Explicit forward and backward stages
let callers place any branch between prepare and finish:

```python
from cudnn.ops import (mhc_prepare, mhc_finish,
                       mhc_backward_prepare, mhc_backward_finish)

u, stats = mhc_prepare(x, fn, scale, base)
b = branch(u)
y = mhc_finish(x, b, stats)["out"]

db, grad_stats = mhc_backward_prepare(x, b, stats, dy)
du = branch_backward(db)
dx, dfn, dscale, dbase = mhc_backward_finish(
    x, dy, du, fn, scale, base, stats, grad_stats)
```

Prepare normalizes the flattened residual streams, applies the mapping matrix,
forms sigmoid pre/post coefficients and a 4x4 Sinkhorn mixing matrix, and returns
the BF16 collapsed branch input. Finish returns the BF16 residual/branch mixture.
`rms_eps=1e-5`, `hc_eps=1e-6`, and `sinkhorn_iters=20` are defaults; positive finite
epsilons and integer Sinkhorn counts 1..64 are supported. Pass the same constants
to backward finish. The default count uses the unrolled kernels; other counts use
the native runtime-count kernels.

Save the FP32 `stats[T,64]` record unchanged. Backward prepare returns BF16
`grad_branch[T,H]` and FP32 `grad_stats[T,32]`. After the external branch backward,
backward finish returns BF16 `dx` and FP32 `dfn`, `dscale`, and `dbase`. The
parameter reductions use fixed orders and overwrite their output buffers.

`mhc_head_forward(x)` returns the BF16 mean over four streams, accumulated in
FP32. `mhc_head_backward(grad_out[T,H])` broadcasts BF16(grad_out/4) to `[T,4,H]`.
All APIs return `TupleDict` values in the order shown; single-output APIs use
`"out"` as their key.

Every API accepts explicit output buffers and `stream=`. Prepare and backward
finish also accept `workspace`, a contiguous CUDA uint8 vector aligned to 256
bytes. Query its size with `mhc_workspace_size(T,H,device=x.device)` or
`backward=True`. Each call resets its launch ticket in caller scratch. Scratch and
outputs must be disjoint from all inputs and from each other. Do not reuse one
scratch buffer for concurrent calls.

Warm the exact geometry/count before CUDA graph capture. With every output and
workspace supplied, execution makes no tensor allocations and reads no tensor
values on the host. Captured input values may change; keep all captured storage
alive. Native plans retain no caller tensor storage. These APIs use explicit
backward and have no autograd or cuDNN graph-engine registration. Nondeterministic
mHC variants and gradient accumulation are not exposed.
