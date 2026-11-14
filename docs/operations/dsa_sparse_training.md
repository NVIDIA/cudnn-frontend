# Cake sparse-attention training

`cudnn.ops.dsa_sparse_forward` and `dsa_sparse_backward` launch the native
Miles Cake fast sparse-attention stages. They require SM100/103/107, CUDA
13.4 or newer, PyTorch, and TVM FFI. The first call builds its native stages.
They do not require CuTe DSL.

Inputs are BF16 `q[Tq,64,512]`, shared key/value `kv[Tk,512]`, and contiguous
int32 `indices[Tq,K]` with global key rows. Both token counts and K are positive.
Optional `q_rope[Tq,64,64]` and `k_rope[Tk,64]` add the RoPE score term.
The query, key, RoPE, and gradient-input tensors may use the native aligned
row/head strides. All pointers must meet the native 16-byte alignment rules.

No causal mask is added. The caller supplies the sparse pattern. Negative
indices and indices outside `[0,Tk)` are holes; repeated indices contribute
repeatedly. An all-hole row has zero output and negative-infinity LSE.
The default score scale is `1/sqrt(576)` with RoPE and `1/16` without RoPE.

```python
from cudnn.ops import dsa_sparse_forward, dsa_sparse_backward

out, lse, residual = dsa_sparse_forward(q, kv, indices,
                                      q_rope=q_rope, k_rope=k_rope)
grads = dsa_sparse_backward(q, kv, indices, out, lse, residual, grad_out,
                           q_rope=q_rope, k_rope=k_rope)
```

Forward returns a `TupleDict` with `out` (BF16), natural-log `lse` (FP32),
and the BF16 output `residual` needed by backward. Save all three unchanged.
Backward returns `dq`, `dkv`, and, with RoPE, `dq_rope` and `dk_rope`.
Query gradients are BF16. Without RoPE, combined dK+dV is FP32; with RoPE,
key gradients are BF16. Key reductions use FP32 atomics. Deterministic mode
raises before allocation or launch (warn-only mode warns and runs).

Supply every output and `workspace` to avoid allocations after warmup.
`dsa_sparse_workspace_size(Tq,Tk,K,rope=True,backward=False)` reports bytes;
the workspace is a contiguous CUDA uint8 vector aligned to 256 bytes.
A backward-size workspace also serves forward. Every writable buffer must
be disjoint from inputs and other writable buffers. The adapter uses one
key pass and a private FP32 key accumulator. It does not retain caller tensors.

`stream=` selects the launch stream. Warm the exact geometry before CUDA graph
capture. Calls then launch without host tensor reads or tensor allocation;
replay may change input and index values. Buffers and captured pointers must
remain alive for replay. There is no autograd or cuDNN graph-engine registration.
These fast kernels do not support attention sinks or head counts other than 64.
