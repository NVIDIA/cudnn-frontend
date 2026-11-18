# Cake dense GEMM training

The experimental Cake adapter supports BF16 projection and FP32 router
GEMMs on SM100 with CUDA Toolkit 13.4+ and TVM-FFI.

```python
from cudnn.ops import dense_gemm_forward, dense_gemm_backward

y = dense_gemm_forward(x, weight)["out"]
grads = dense_gemm_backward(dy, x, weight)
dx, dw = grads["dx"], grads["dw"]
```

For BF16, `x[..., M, K]`, `weight[..., N, K]`, and `dy[..., M, N]`
have rank two or three, with equal batch sizes. Weight gradients are
per batch; no batch reduction is implicit. `weight_gradient_dtype=torch.float32`
selects FP32 weight gradients. FP32 router inputs are rank two and produce
FP32 outputs and gradients. The functions return explicit gradients without
an autograd registration.

Inputs must be contiguous matrices or transposes of contiguous matrices.
They must be 16-byte aligned, nonempty, and span fewer than 2^31 elements.
BF16 row and batch strides must be multiples of eight elements; the output
column count must be a multiple of eight. The upstream planner rejects a
layout or shape without a matching generated program. BF16 forward projection
with FP32 output is not supported. FP32 router rows must be 16-byte aligned,
and its effective output column count must be a multiple of eight.

For caller-owned outputs and scratch, use `DenseGemmPlan(a, b, out)` from
`cudnn.gemm.cake.dense`. It prepares `out = a @ b` and retains metadata and
compiled entries only. Allocate `plan.get_workspace_size()` bytes of CUDA
uint8 scratch, aligned to 128 bytes, then call
`plan.execute(a, b, out=out, workspace=scratch, stream=stream)`.
New buffers must match the plan's shapes, strides, dtypes, and device.
Output and scratch must not alias each other or the inputs. Concurrent calls
need separate outputs and scratch.

Execution does not allocate, compile, copy input layouts, or read device values
on the host. BF16 execution resets scheduler counters on the requested stream.
FP32 router execution explicitly splits its second operand into three BF16
parts in caller scratch, then runs the single-split deterministic GEMM.
This split is part of the FP32 arithmetic algorithm. CUDA graph replay
reads the current input values. Graph engine selection is unchanged.

Sources are pinned to
[miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_dense_gemm_det_train).
`VENDORED_DENSE_GEMM.json` records the source hashes.
