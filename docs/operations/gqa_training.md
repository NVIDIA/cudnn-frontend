# Packed GQA training

These experimental APIs run the Cake BF16 grouped-query attention kernels.
They require SM100a, SM103a, or SM107a and CUDA Toolkit 13.4 or newer.
The runtime compiles sources on first use; warm both directions before capture.

```python
from cudnn.ops import gqa_attention_forward, gqa_attention_backward

out, lse = gqa_attention_forward(q, k, v, cu_q, cu_k)
dq, dk, dv = gqa_attention_backward(q, k, v, out, lse, grad_out, cu_q, cu_k)
```

Forward returns `TupleDict(out=..., lse=...)`; backward returns
`TupleDict(dq=..., dk=..., dv=...)`. The paired functions have no autograd
registration. Pass the forward output and LSE unchanged to backward.

| Operand | Shape | Type |
| --- | --- | --- |
| `q`, `out`, `grad_out`, `dq` | `[Tq, Hq, 256]` | BF16 |
| `k`, `v`, `dk`, `dv` | `[Tk, Hkv, 256]` | BF16 |
| `lse` | `[Hq, Tq]` | FP32, natural logarithm |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B + 1]` | INT32 |

All tensors must be contiguous CUDA tensors on one device. Data tensors need
16-byte alignment; prefixes need 4-byte alignment. `Tq`, `Tk`, and head counts
must be positive. `Hq / Hkv` must be an even integer. Operand element counts
must fit INT32, and `B` and `Hkv` must not exceed 65535.

The mask is causal, aligned at the bottom right within each segment. Query
position `i` sees key positions `j <= i + key_length - query_length`.
Rows with no visible keys return zero output and negative-infinity LSE.
The default scale is `1 / sqrt(256)`; `softmax_scale` accepts a finite host scalar.

Prefix values are a device-data contract: each starts at zero, is
nondecreasing, and ends at its operand's token count. The two prefixes must
describe the same number of segments. Empty segments are supported within
nonempty packed operands. Execution never copies prefixes to the host.

Backward accepts `max_seqlen_k`, an optional positive host integer that must
bound every key segment. A smaller value gives incomplete gradients. Omit
it to use `Tk`, an always-safe bound, with potentially more inactive CTAs.

Forward and dK/dV use fixed reduction orders. dQ uses FP32 atomic additions
and is not bitwise reproducible. Backward raises under
`torch.use_deterministic_algorithms(True)`, or warns and runs under `warn_only`.

## Outputs, workspace, and streams

Functions allocate omitted outputs and scratch. To avoid device allocation
after warmup, provide `out`, `lse`, and `workspace` for forward, or `dq`, `dk`,
`dv`, and `workspace` for backward. Outputs and workspace must not overlap
each other or any input.

`gqa_attention_workspace_size(Tq, Hq, B, backward=False)` returns the scratch
byte count. Allocate a contiguous, 256-byte-aligned CUDA UINT8 vector on the
operand device. Forward resets its tile counter on the launch stream;
backward initializes scratch in its preprocessing kernel. Use separate
workspace for overlapping calls.

`stream=None` uses the operand device's current PyTorch stream. A concrete
PyTorch stream or raw stream handle can be supplied. Caller-owned buffers
are recorded on the launch stream. After warmup, execution reads only host
metadata, launches kernels, and supports CUDA graph capture.

## Source and validation

The vendored `cake_gqa_nondet_train` package is copied unchanged from
[`radixark/miles-ext-kernels` at `f55fb996`](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce).
`python/cudnn/_cake/VENDORED_GQA.json` records its upstream revisions and file hashes.
Tests cover forward and gradient references, side streams, changed-input
capture replay, supplied outputs/workspace, and validation before launch.
