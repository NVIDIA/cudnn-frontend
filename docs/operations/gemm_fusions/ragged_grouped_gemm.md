# Ragged grouped GEMM training

This experimental Cake implementation supports deterministic forward, input
gradient, and weight gradient on SM100. It requires CUDA Toolkit 13.4+ and
Torch with TVM-FFI. The default Frontend engine selection is unchanged.

```python
from cudnn.ops import ragged_grouped_gemm_forward, ragged_grouped_gemm_backward

y = ragged_grouped_gemm_forward(x, weight, offsets)["out"]
grads = ragged_grouped_gemm_backward(dy, x, weight, offsets)
dx, dw = grads["dx"], grads["dw"]
```

`x[M,K]`, `weight[E,N,K]`, and `dy[M,N]` are contiguous BF16 tensors.
`offsets[E]` is contiguous CUDA int32 cumulative end offsets. Its values must
be nondecreasing, within `[0,M]`, and end at M. Empty groups are allowed.
The kernels read offsets on the device; no host copy or padding is needed.
Malformed offset values violate the device-data contract.

M and E must be positive; N and K must be positive multiples of 256.
All operand spans must be below 2^31 elements. Inputs and outputs must be
16-byte aligned. Forward and input gradients are BF16. Weight gradients can
be BF16 or FP32 (`weight_gradient_dtype=torch.float32`). The paired functions
return explicit gradients; they do not register an autograd operation.

For repeated execution, construct `RaggedGroupedGemmPlan` from
`cudnn.gemm.cake.grouped`, with operation `fwd`, `dgrad`, or `wgrad`.
For wgrad, the two operands are `dy` and `x`; otherwise they are the row
operand and the expert weights. Allocate `plan.output_shape` with
`plan.output_dtype`, and `plan.get_workspace_size()` bytes of CUDA uint8
scratch with 128-byte alignment. Pass both to
`plan.execute(a, b, offsets, out=out, workspace=scratch, stream=stream)`.
Plans retain no device storage. Each call can bind fresh buffers with the
same metadata. Concurrent calls require distinct outputs and scratch.

Execution initializes scheduler counters on the requested stream, then
launches the prepared kernels. It does not allocate, compile, convert inputs,
or read device values on the host. CUDA graph replay can change offsets and
operand values. Weight-gradient tail reductions use a fixed order.

This first adapter uses the 256-column deterministic programs. Fast variants,
weight-gradient accumulation, and graph-engine selection are not exposed.

Kernel sources and their manifests are pinned to
[radixark/miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_moe_grouped_gemm_det_train).
`VENDORED_MOE_GROUPED_GEMM.json` records each imported file's SHA256.
