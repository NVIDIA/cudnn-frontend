# QSA two-pass backend implementation notes

The internal backend in `cudnn.QSA.two_pass` computes the gradients of the original
12Q/1KV or 24Q/2KV, D256 QSA geometry on SM100/SM103. It requires CuTeDSL >= 4.7.0.
It supports one sequence or packed documents described by int32 `cu_seqlens`
and a host upper bound `max_seq_len`. For the public packed training interface,
use [indexed QSA](qsa_indexed.md). These are implementation notes,
not a public API. The backend is retained for further integration into the
common QSA functions after its contracts are aligned.

All tensors must be contiguous on the same CUDA
device. The softmax scale is fixed at `1 / 16`.

| Tensor | Shape | Dtype |
|---|---|---|
| Q, saved out, dout, dQ | `[T, 12 * Hkv, 256]` | BF16 |
| K, V, dK, dV | `[T, Hkv, 256]` | BF16 |
| Natural-log LSE | `[T, 12 * Hkv]` | FP32 |
| topk_idxs | `[T, 2051]` or `[T, 2052]` | int32 |

For row `i` at position `p` in its sequence, the indices contain `min((p+1)//4, 512)` distinct complete
four-token blocks, followed by `(p+1)%4` causal tail tokens, then `-1` padding.
The four IDs within each complete block are consecutive and begin at a multiple
of four relative to the sequence origin. Each selected block belongs to that
sequence's causal prefix. Token IDs address rows of the packed buffer. Up to `T=2051`,
every row contains its entire causal prefix. The kernel derives the tail from
the query position; it does not validate these device-value relationships.
`1 <= T <= floor(INT32_MAX / (12 * Hkv * 256))` is supported. This selection format and the saved BF16
output differ from the bitmap and indexed APIs; their saved tensors cannot be
substituted directly.

The query-major pass computes dQ and writes BF16 P/dS records. An inverted
block-to-query index lets the key-major pass accumulate dK/dV locally instead
of scattering every query's contribution through global floating-point atomics.
A separate dense kernel handles the early prefix, while long inputs are divided
into KV passes. The records trade memory for reduced contention and reuse:
sparse queries are processed in windows of at most 4096 compacted query IDs.
Each window rebuilds its inverted index and consumes its records before reusing
the same storage. dK/dV accumulate across windows; the dense prefix is computed
once. All sequence and window counts stay on the device.

P/dS records are bounded to approximately 0.376 GiB for 12Q/1KV and 0.752 GiB
for 24Q/2KV. At 128K, total plan scratch is approximately 2.49 GiB and 4.87 GiB,
respectively, for a single sequence or three packed sequences. Inputs and final
outputs are additional. Gradient accumulators and sequence tables still grow
with the total token count; the entire workspace is not constant-size.

The internal APIBase plan retains explicit compilation and caller-owned
workspace. Execution checks current tensor layouts and storage; it does not
allocate or read indices to the host. The backend's regressions exercise dense
and sparse crossovers, changed-input replay and explicit stream handling.

See [all QSA entry points](qsa.md#finding-qsa-attention-and-its-indexer) for
selection and the other attention contracts.
