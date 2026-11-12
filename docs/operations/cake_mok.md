# Cake MoK training

`cudnn.moe_ep.cake` exposes the experimental standalone MoK training API.
The initial contract is SM100, one NCCL rank, four routed experts, top-2,
and one shared expert. CUDA Toolkit 13.4+ and TVM-FFI are required.
This API is separate from `MoeEp` and does not select a graph engine.

```python
from cudnn.moe_ep.cake import prepare_mok_training, create_mok_workspace

adapter = prepare_mok_training(device=device)
config, workspace = create_mok_workspace(
    group=group, device=device, num_local_tokens=x.shape[0],
    hidden_size=x.shape[1],
)
schedule = adapter.build_schedule(workspace, config, ids, num_local_experts=4)
y, context = adapter.forward(config, workspace, schedule, x, scores, *weights)
grads = adapter.backward(config, workspace, schedule, context, dy, x, scores, *weights)
```

Initialize the NCCL group first. Set its CUDA device current for all adapter
calls. Calls use that device's current Torch stream. The workspace factory
allocates communication buffers and negotiates their capacity outside capture.
Retain the workspace, schedule, and forward context until backward finishes.
Use separate workspaces for concurrent iterations.

`x[T,H]` and `dy[T,H]` are BF16. `ids[T,2]` contains int64 expert indices
in `[0,4)`, with no duplicates in a row; `scores[T,2]` is FP32 and strictly
positive. Weights are ordered shared gate, up, down, then routed gate, up,
down. Shared gate/up have shape `[I,H]`, shared down `[H,I]`; routed weights
add an expert axis of size four. All inputs and weights are contiguous CUDA
tensors on the same device. H and I must be positive multiples of 256.
Outputs are BF16. Gradients are returned as `(dx, dscores, routed_dgate,
routed_dup, routed_ddown, shared_dgate, shared_dup, shared_ddown)`.
Score gradients are FP32; weight gradients default to BF16.

This is an allocating eager API: schedule, forward, recompute, and backward
allocate result and temporary tensors. It does not provide an allocation-free
`execute` contract. Warm each variant before capture with
`adapter.prepare(swiglu_limit, mxfp8)` and, for checkpointing,
`adapter.prepare_recompute(swiglu_limit, mxfp8)`. Capture retains the resulting
storage; keep the graph and its workspace alive for all replays.

For clamped SwiGLU, prepare with `clamped_swiglu=True` and pass a finite,
positive `swiglu_limit` to forward and backward. It computes
`silu(min(gate, limit)) * clamp(up, -limit, limit)`.
With `fp32_wgrad=True`, backward takes `weight_grad_accumulators`: six
caller-owned FP32 buffers in the same order as the input weights. Backward
adds into them; the caller controls when to clear them.
The six accumulators must not overlap each other or any live input,
forward-context tensor, schedule tensor, or workspace buffer.

`quantize_mok_weights(gate, up, down)` allocates native MXFP8 routed weights
in forward and backward orientations. H must then be a multiple of 512.
Requantize after each weight update, call `adapter.prepare(mxfp8=True)`, and pass the
matching orientations to forward and backward. Shared weights remain BF16.

Multi-rank expert parallelism is gated off in this first adapter. Its NVLink
multicast and collective behavior require separate multi-GPU qualification.
The upstream package includes those programs but the Frontend factory does
not admit them.

Sources are copied byte-for-byte from
[miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce/miles_ext_kernels/cake/_vendor/cake_mok_bf16).
`VENDORED_MOK.json` records their hashes. The vendored README describes native
schedule capacity and checkpoint contracts.
