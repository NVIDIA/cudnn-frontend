# DSv4.1 mHC projection/RMS backward

This experimental SM100 API computes the input and projection-weight gradients
of the projection/RMS stage in DSv4.1 mHC. It expects the upstream projection
and RMS gradients after the gate and normalization derivatives:

```text
P  = grad_proj[:, :24]
dX = P @ W + grad_r / (r * K) * X
dW = P.T @ X
```

The two matrix products convert operands to TF32 and accumulate in FP32.
The RMS contribution uses FP32 arithmetic. dX is stored in BF16 and dW in
FP32. Callers must explicitly pass `allow_tf32=True`; this API does not promise
full FP32 matrix-product precision. The reduction of partial dW is in a fixed
order, without atomics. Outputs are overwritten, not accumulated.

The supported training profile has 4 residual streams, hidden size 5120 and
4096 tokens per rank. This is one backward stage, not the complete mHC forward
or backward. Sinkhorn, gate derivatives and parameter-gradient accumulation
remain the caller's responsibility.

## Requirements and tensor contract

Use a Blackwell SM100 GPU, PyTorch, `cuda-tile>=1.5`, a compatible system
`tileiras` compiler and the Frontend `cutile,triton` optional dependencies:

```bash
pip install 'nvidia-cudnn-frontend[cutile,triton]' 'cuda-tile>=1.5'
```

Install the PyTorch CUDA build matching your environment separately.

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `x`, `dx` | `[4096, 20480]` | BF16 |
| `weight`, `dweight` | `[24, 20480]` | FP32 |
| `grad_proj` | `[4096, 32]` | FP32 |
| `grad_r`, `r` | `[4096, 1]` | FP32 |

All tensors use compact row-major strides, with stride `(1, 1)` for the column
vectors. The last 8 columns of `grad_proj` are ignored, including if nonzero.
`r` must contain the positive RMS values from forward. All operands and
workspace must be on one device, disjoint and 16-byte aligned. Other shapes,
strides, devices and precisions are rejected. No conversion or repacking is
performed.

## Allocating wrapper

```python
result = cudnn.mhc_projection_backward(
    x, weight, grad_proj, grad_r, r,
    allow_tf32=True, backend="frost",
)
dx, dweight = result  # TupleDict order: dx, dweight
```

The wrapper allocates outputs and workspace on the requested stream. Callers
must establish producer/consumer stream dependencies and keep tensors alive
until execution completes.

## Prepared class API

```python
plan = cudnn.MhcProjectionBackward(
    x, weight, grad_proj, grad_r, r, dx, dweight,
    allow_tf32=True, backend="frost",
)
plan.check_support()
plan.compile()
workspace = torch.empty(
    plan.scratch_workspace_bytes(), device=x.device, dtype=torch.uint8,
)
plan.execute(x, weight, grad_proj, grad_r, r, dx, dweight, workspace)
```

`compile()` compiles and loads both kernels without executing tensor work.
`execute()` enqueues exactly two kernels and allocates no tensor storage.
Workspace is 15,728,640 bytes (15 MiB). Use separate scratch and outputs for
overlapping executions. Prepared execution supports CUDA Graph capture; keep
its buffers alive for all replays. `current_stream` accepts a PyTorch stream,
a CUDA stream handle, or `None` for the current stream of the tensors' device.

## Source attribution

The projection kernel derives from Megatron-LM's
[`_ct_fused_grad_x_weight_kernel`](https://github.com/NVIDIA/Megatron-LM/blob/25dc53dbb0dbc4d9b884d749e4c432ed88c4b73b/megatron/core/fusions/fused_mhc_kernels.py)
under BSD-3-Clause. This implementation partitions the token reduction across
independent blocks, writes partial weight gradients, then reduces them in a
second kernel. The source notice is retained in `_kernels.py`.
