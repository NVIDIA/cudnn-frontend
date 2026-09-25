# Aligned HCA Backward

**Experimental.** This API computes HCA backward for the aligned, single-sequence
DSV4 layout on GB300 and Rubin with CP4/8/16 and sequence lengths from 8K to 128K.
Each CP chunk must be a multiple of 128 tokens. It is separate from generic DSA backward: it does
not accept arbitrary sparse indices, multiple sequences per pack, rebalanced HCA layouts,
or unaligned CP chunks. Unsupported declarations are rejected without a layout copy.

## Installation

From this frontend source checkout (the experimental API is not in the
published wheel):

```bash
python -m pip install -e '.[cutedsl,triton]' torch
```

The implementation uses precompiled Triton kernels and cuDNN graph matmuls.
It is Torch-only and requires Triton 3.7 or later on GB300. Rubin requires
Triton 3.8.0 or later (validated with 3.8.0).
The implementation is validated with CUDA 13.2, cuDNN frontend 1.31, cuDNN 9.21 and GB300;
Rubin validation uses CUDA 13.4 and cuDNN 9.26.
Importing `cudnn` does not eagerly import this API or its optional dependencies.

## Computation

Each CP rank owns `L = S / cp_size` queries with 128 heads and head dimension 512. A query
at global position `t` attends to its causal 128-token window and compressed
groups `0 .. floor((t + 1) / 128) - 1`. The sequence has exactly `S` positions in the declared padded layout. K and V share one input tensor; their gradients are added into `dkv`.

`lse` is the natural-log normalization over ordinary KV keys **excluding** the
attention sink. `out` already includes the sink in its softmax denominator.
The sink has zero value. For valid pairs, the backward equations are:

```text
L = logaddexp(lse, attn_sink)
P = exp(softmax_scale * Q @ KV.T - L)
delta = sum(out * dout, dim=-1)
dS = P * (dout @ KV.T - delta) * softmax_scale
dQ = dS @ KV
dKV = dS.T @ Q + P.T @ dout
d_sink = sum_queries(-delta * exp(attn_sink - L))
```

For finite ordinary logits, a `+inf` sink has probability one. With the
corresponding `out = 0`, that head contributes zero gradients. A `-inf` sink
is disabled. These limits also apply when sink values change during graph replay.

Probability and score-gradient intermediates are BF16, GEMM accumulation and
gradient-reduction scratch are FP32. Q and dO are loaded separately by TMA;
the score kernel keeps two accumulators without materializing interleaved
inputs. Gradient outputs `dq`/`dkv` are BF16 and
`d_sink` is FP32. This is not bitwise equivalent to an all-FP32 reference.
Normalization uses four warps. The score pipeline uses three or four stages
selected from fixed shape and CP-rank metadata at plan time, without runtime autotuning.
For 32-token query groups, local KV-gradient reduction streams four output
rows per block. The 128-token groups retain the vector reduction. Both use
FP32 accumulation, with different addition orders before the BF16 output cast.

## Tensor Contract

All tensors are contiguous Torch CUDA tensors on one supported CUDA device. Input and
output pointers must be 16-byte aligned. Outputs and workspace must not share
storage with inputs or one another.

| Argument | Shape | Dtype |
|---|---|---|
| `q`, `out`, `dout`, `dq` | `(L, 128, 512)` | BF16 |
| `kv`, `dkv` | `(L + 128 + S / 128 + cp_size, 512)` | BF16 |
| `lse` | `(L, 128)` | FP32 |
| `attn_sink`, `d_sink` | `(128,)` | FP32 |
| `workspace` | At least `scratch_workspace_bytes()` bytes | uint8 |

`cp_size` is 4, 8 or 16 (default 16). `cp_rank` is a plan-time integer in
`[0, cp_size)`; the global query start is `L * cp_rank`. `softmax_scale` is a finite plan-time host scalar, default
`1 / sqrt(512)`.

`AlignedHCABackward.supports_configuration(L, cp_size, device)` checks the
architecture, Triton version and sequence/CP geometry; `check_support()` validates
the full tensor contract and reports an unsupported Triton version before compilation.

KV rows 0-127 hold the preceding boundary, the next `L` rows hold local
tokens, and the remaining rows hold rank-major compressed storage. Each rank
contributes `G + 1` physical slots for `G = L / 128` canonical groups:

```text
owner = c // G
physical_row = L + 128 + owner * (G + 1) + c % G + (owner > 0)
```

Slot `G` of rank zero and slot zero of later ranks are not canonical compressed
keys and receive zero gradients. The API addresses this layout directly.

KV rows unreachable by this CP rank, including rank zero's preceding boundary,
need not be initialized. Even nonfinite values in those rows do not affect
valid gradients, and their `dkv` entries are zero.

## Allocating Wrapper

```python
from cudnn import aligned_hca_backward_wrapper

result = aligned_hca_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, cp_rank=cp_rank, cp_size=cp_size,
)
dq, dkv, d_sink = result
assert dq is result["dq"]
```

Optional `dq`, `dkv`, `d_sink` and `workspace` arguments reuse caller buffers.
The wrapper caches compiled plans per host thread and tensor metadata. Warm it
once for each signature before graph capture. Missing outputs and scratch are
allocated on the launch stream; allocations made during capture belong to the
graph's private pool. Generic DSA backward is not redirected to this API.

## Preallocated Class API

```python
import torch
from cudnn import AlignedHCABackward

dq = torch.empty_like(q)
dkv = torch.empty_like(kv)
d_sink = torch.empty_like(attn_sink)
op = AlignedHCABackward(
    q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink, cp_rank=cp_rank, cp_size=cp_size,
)
op.check_support()
op.compile()
workspace = torch.empty(op.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
result = op.execute(
    q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink, workspace,
    current_stream=torch.cuda.current_stream(q.device),
)
```

Constructor arguments may instead be `TensorDesc` metadata with an indexed
Torch CUDA device. No input tensor storage is retained by the constructor.
Compilation builds and loads the Triton kernels and two cuDNN matmul plan shapes
without executing on sample data. Query workspace size after compilation.

`execute` allocates no tensors, reads no device data on the host, performs no
synchronization, and never invokes the Triton JIT dispatcher. It launches the
declared normalization/packing, score, three GEMMs and bounded reductions.
Algorithmic scratch is rebuilt from current inputs on every call. Workspace
must be contiguous and 256-byte aligned; its contents need not be initialized.
Scratch does not include a paired Q/dO buffer; always query the actual compiled
size, including cuDNN GEMM scratch.

`current_stream=None` uses Torch's current stream. A Torch CUDA stream, concrete
`cuda.CUstream`, or integer handle is accepted; zero means the default stream.
Legacy/PTDS sentinel handles 1 and 2 are rejected. All work uses the specified
stream. The caller orders producers and consumers, retains input/output/scratch
storage until work completes, and retains the operator for captured graphs.
Do not execute one class instance concurrently from multiple host threads, or
reuse the same scratch for overlapping calls on different streams.

## Testing

From `test/python` in a GB300 or Rubin environment:

```bash
pytest fe_api/dsa/test_DSA_aligned_hca_backward.py \
    fe_api/dsa/test_DSA_aligned_hca_backward_boundaries.py -m 'L0 or L1'
```

Coverage includes reference gradients, changed-input graph replay, explicit
streams, workspace ownership, support validation and allocation/JIT guards.
Boundary cases cover positive-infinite sinks, unused KV rows with nonfinite
values, and very negative LSE with large dO.
