# Sparse Attention (generic, index-driven)

Variant-neutral sparse attention parameterized by per-query index lists. One
signature covers the index-driven family: token-level top-k (DeepSeek DSA),
block-level top-k (MiniMax MSA, NSA selection), and micro-block selections
(Qwen QSA). Architecture-specific pipelines (indexers, top-k, compressors)
live in their own packages (`deepseek_sparse_attention`,
`native_sparse_attention`, `csa`); mask-driven block sparsity lives in
`block_sparse_attention`.

## Forward — `sparse_attention_forward_wrapper`

Softmax attention restricted to a per-query selected KV subset.
Returns `{'out', 'lse'}`.

Contract (normative — device kernels must match the reference exactly):

- **Storage-native indices.** `topk_idxs` entries index K/V as the kernel
  receives them: packed THD → global flat ids in `[0, T_kv)`; BSHD →
  within-sequence ids in `[0, S_kv)`. `-1` marks an invalid slot; ids are
  unique within a row. `topk_length` optionally gives per-row valid counts.
- **Index scope `G`.** `topk_idxs` is `(T_q, topk)` (one set shared by all
  query heads) or `(T_q, G, topk)` with `G ∈ {H_kv, H_q}` (per KV-head
  group / per query head).
- **`index_granularity`** = tokens covered per index entry (1 | 4 | 64 |
  128); entry `i` selects tokens `[i*g, i*g + g)`, tail clamped to the KV
  bound.
- **Separate K and V** (`(T_kv, H_kv, D_k)` / `(T_kv, H_kv, D_v)`,
  `D_v` may differ from `D_k`); `k` and `v` may alias the same storage
  (MLA-style latents).
- **LSE is KV-only, base-e, FP32** — `attn_sink (H_q,)` contributes to the
  softmax denominator but never to the LSE (the convention consumed by
  `deepseek_sparse_attention.sparse_attention_backward_wrapper`). Rows with
  no valid entry produce `lse = -inf`, `out = 0`.
- **Deterministic always** — identical inputs give bitwise-identical outputs.

Execution dispatches to registered device kernels (currently the SM100 DSA
sparse-prefill kernel for its envelope, when its module is present);
configurations no kernel serves raise `NotImplementedError`. The normative
reference implementation lives with the tests
(`test/python/sparse_attention/sparse_attention_reference.py`) and is the
oracle every kernel is validated against. The API surface is
framework-neutral: dlpack tensors (torch/JAX/numpy), no module-level torch,
torch-only kernels gate request-or-fail.

Frozen-but-unimplemented signature slots: `page_table`/`page_size` (paged
KV) and `max_seqlen_q`.

## Roadmap

1. SM100 device kernel for the DSA envelope (`G=1`, granularity 1, aliased
   K/V, `D_k ∈ {512, 576}`).
2. Decode: `(B, next_n, H_q, D_k)` queries, paged KV, split-KV with
   fixed-order merge.
3. GQA substrate kernels: `G = H_kv`, granularity 4/64/128 (QSA / MSA).
4. Precision: per-tensor FP8 scales, MXFP8 (`sf_vec_size=32`, ue8m0,
   scale-padded cu_seqlens — names as in `deepseek_sparse_attention`
   indexer forward), keyword-only additions.
5. Backward superset wrapper (deterministic two-pass dK/dV mode).

## Indexer + top-k — `indexer_topk_wrapper`

Fused indexer scoring + top-k selection: one pass, no dense `(T_q, T_e)`
logits tensor ever materialized. Returns `{'indices', 'topk_length'[, 'logits']}`.

Contract (normative):

- **Scoring**: `score[t,g,e] = sum_{h in group g} w[t,h] * f(q[t,h]·k[e,gk])`,
  `f ∈ {relu, identity}`, `weights=None` means 1, any softmax/temperature
  scale folds into `weights`. `k_index` has 1 (shared) or `head_groups` heads.
- **Causal bound**: entry `e` (covering tokens `[e·ratio, e·ratio+ratio)`)
  is a candidate iff fully past: per-row count `(p+1) // ratio`, with
  `q_causal_offsets` giving each segment's global start (chunked prefill).
- **`score_pool`**: max-pool scores over `score_pool` consecutive entries
  pre-selection; emitted ids are pooled ids
  (`index_granularity = ratio * score_pool` downstream).
- **`force_first` / `force_last`**: always-selected candidate entries, inside
  the `top_k` budget.
- **Output**: storage-native global entry ids, **compact + ascending**
  (`-1` beyond `topk_length`); `logits` slot-aligned FP32 opt-in
  (`return_logits=True`, off by default — serving discards them).
- **Deterministic always**: exact top-k, boundary ties → smallest id;
  ascending output order makes the downstream gather monotonic and the whole
  indexer→attention pipeline bitwise-reproducible.

Variant mapping: DSv4 (64 weighted ReLU heads, ratio 4, top-512/1024),
GLM-5.x (32 weighted heads, ratio 1, top-2048), QSA (4 unweighted heads,
ratio 4, top-512, `force_last=1`), MSA (`head_groups=H_kv`,
`activation="none"`, `score_pool=128`, top-16, `force_last=1`).

No device kernel is registered yet; the oracle lives at
`test/python/sparse_attention/indexer_topk_reference.py`.
