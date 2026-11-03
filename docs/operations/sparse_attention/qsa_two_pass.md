# QSA two-pass backend implementation notes

The internal backend in `cudnn.QSA.two_pass` computes the gradients of the original
12Q/1KV or 24Q/2KV, D256 QSA geometry on SM100/SM103. It requires CuTeDSL >= 4.7.0.
The 12Q/1KV sparse path additionally requires the CUDA 13 compiler flavor
(`pip install 'nvidia-cutlass-dsl[cu13]>=4.7.0'` and a CUDA 13 compatible driver);
the CUDA 12 flavor is rejected before its known compiler crash. The 24Q/2KV
path also supports the CUDA 12 flavor.
It supports one sequence or packed documents described by int32 `cu_seqlens`
and a host upper bound `max_seq_len`. For the public packed training interface,
use indexed QSA ([docs](/operations/sparse_attention/qsa/qsa_indexed), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa_indexed.md)). These are implementation notes,
not a public API. The public `qsa_indexed_backward` selects this backend for
12Q/1KV when the caller explicitly declares `selection_mode="causal_topk"`.

All tensors must be contiguous on the same CUDA
device. The softmax scale is fixed at `1 / 16`.

| Tensor | Shape | Dtype |
|---|---|---|
| Q, dout, dQ | `[T, 12 * Hkv, 256]` | BF16 |
| K, V, dK, dV | `[T, Hkv, 256]` | BF16 |
| Saved output | `[T, 12 * Hkv, 256]` | FP32 (preferred) or BF16 |
| Natural-log LSE | `[12 * Hkv, T]` with FP32 saved output; `[T, 12 * Hkv]` with BF16 saved output | FP32 |
| topk_idxs | `[T, 2051]` or `[T, 2052]` | int32 |

For row `i` at position `p` in its sequence, the indices contain `min((p+1)//4, 512)` distinct complete
four-token blocks, followed by `(p+1)%4` causal tail tokens, then `-1` padding.
The four IDs within each complete block are consecutive and begin at a multiple
of four relative to the sequence origin. Each selected block belongs to that
sequence's causal prefix. Token IDs address rows of the packed buffer. Up to `T=2051`,
every row contains its entire causal prefix. The kernel derives the tail from
the query position; it does not validate these device-value relationships.
`1 <= T <= floor(INT32_MAX / (12 * Hkv * 256))` is supported. The FP32 saved-output mode reads the same output dtype and head-major LSE
layout as the public FE/Miles-oriented interface natively, without a cast or
repack. The BF16 mode retains the earlier internal saved-state contract.
Selection remains restricted to the canonical complete-block-plus-tail format
above, and packed execution also needs `cu_seqlens`. The public indexed mode
requires native FP32 saved state and index width 2052; the more general default
indexed and bitmap calls retain their existing implementation.

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

See all QSA entry points ([docs](/operations/sparse_attention/qsa#finding-qsa-attention-and-its-indexer), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa.md#finding-qsa-attention-and-its-indexer)) for
selection and the other attention contracts.
