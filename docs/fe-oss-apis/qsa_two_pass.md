# QSA two-pass backward

`cudnn.ops.qsa_two_pass_backward` computes the gradients of the original
24Q/2KV, D256 QSA geometry on SM100/SM103. It requires CuTeDSL >= 4.7.0.
It supports one sequence, not packed documents or 12Q/1KV. For packed training
with 12Q/1KV, use [indexed QSA](qsa_indexed.md).

```python
from cudnn.ops import qsa_two_pass_backward

dq, dk, dv = qsa_two_pass_backward(q, k, v, out, dout, lse, topk_idxs)
```

The result is a `TupleDict` with keys `dq`, `dk`, `dv` in that order. There is
no autograd registration. All tensors must be contiguous on the same CUDA
device. The softmax scale is fixed at `1 / 16`.

| Tensor | Shape | Dtype |
|---|---|---|
| Q, saved out, dout, dQ | `[T, 24, 256]` | BF16 |
| K, V, dK, dV | `[T, 2, 256]` | BF16 |
| Natural-log LSE | `[T, 24]` | FP32 |
| topk_idxs | `[T, 2051]` | int32 |

For row `i`, the indices contain `min((i+1)//4, 512)` distinct complete
four-token blocks, followed by `(i+1)%4` causal tail tokens, then `-1` padding.
The four IDs within each complete block are consecutive and begin at a multiple
of four. Each selected block belongs to the causal prefix. Up to `T=2051`,
every row contains its entire causal prefix. The kernel derives the tail from
the query position; it does not validate these device-value relationships.
`1 <= T <= floor(INT32_MAX / (24 * 256))` is supported. This selection format and the saved BF16
output differ from the bitmap and indexed APIs; their saved tensors cannot be
substituted directly.

The query-major pass computes dQ and writes BF16 P/dS records. An inverted
block-to-query index lets the key-major pass accumulate dK/dV locally instead
of scattering every query's contribution through global floating-point atomics.
A separate dense kernel handles the early prefix, while long inputs are divided
into KV passes. The records trade memory for reduced contention and reuse:
the current 128K plan requires approximately 24.56 GiB of workspace, in addition
to inputs and outputs. This is not a bounded-workspace implementation.

The function accepts optional preallocated `dq`, `dk`, `dv`, `stream`, and
`workspace`. For explicit compilation and allocation ownership, construct
`cudnn.QSA.two_pass.QsaTwoPassBackward` with sample tensors, call
`check_support()` and `compile()`, allocate `scratch_workspace_bytes()` bytes,
then call `execute()` with current tensors and that workspace. Execution checks
current tensor layouts and storage; it does not allocate or read indices to the
host. The convenience function allocates omitted outputs and workspace. Warm up
before CUDA graph capture.

See [all QSA entry points](qsa.md#finding-qsa-attention-and-its-indexer) for
selection and the other attention contracts.
