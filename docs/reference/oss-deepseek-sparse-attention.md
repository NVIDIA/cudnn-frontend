# OSS DeepSeek sparse-attention contract

Use `cudnn.DSA` for the score, selection, recomputation, and backward stages used by DeepSeek-style sparse attention. The namespace contains independent wrappers; it is not one end-to-end operator and does not infer routing metadata between stages.

## Before you begin

Install the `cutedsl` extra, select one DSA stage, and prepare every intermediate
that stage consumes. Record the deployed architecture, shape, dtype, layout,
top-k convention, and softmax scale in the specialization key.

## DSA namespace

`cudnn.DSA` is a lazy namespace object that exports the classes and wrappers listed below. Importing it loads optional CUTLASS DSL dependencies; it does not compile a kernel until a class or wrapper receives tensors.

## Choose a stage

| Public wrapper | Purpose | Principal outputs | Current architecture |
| --- | --- | --- | --- |
| `sparse_attention_backward_wrapper` | Backpropagate sparse attention | `dq`, `dkv`, `d_sink` | SM90 or SM100 |
| `indexer_forward_wrapper` | Compute dense indexer scores | `scores` | SM100+ |
| `indexer_top_k_wrapper` | Select local top-k entries | `indices`, optional `values` | SM90+ |
| `local_to_global_wrapper` | Convert local top-k indices | global `indices` | Same device as indices |
| `compactify_wrapper` | Pack valid indices | `indices`, `topk_length` | Same device as indices |
| `sparse_indexer_score_recompute_wrapper` | Recompute selected indexer scores | `predict` | SM90 or SM100 |
| `sparse_attn_score_recompute_wrapper` | Recompute selected attention scores | `target` | SM90 or SM100 |
| `dense_indexer_score_recompute_wrapper` | Recompute full-KV indexer scores | scores and denominator | SM90 or SM100 |
| `dense_attn_score_recompute_wrapper` | Recompute full-KV attention scores | scores and denominator | SM90 or SM100 |
| `indexer_backward_wrapper` | Backpropagate selected indexer scores | index-Q, weights, and index-K gradients | SM90 or SM100 |
| `dense_indexer_backward_wrapper` | Backpropagate the dense indexer path | index-Q, weights, and index-K gradients | SM90 or SM100 |

## indexer_forward_wrapper

Pass CUDA `q`, `k`, and `w` tensors plus `ratio`, optional `qhead_per_kv_head`, block sizes, stages, and `sm_scale`. The fixed-shape path requires head dimension 128 and supports `qhead_per_kv_head` values 32 or 64. The wrapper also accepts THD metadata through `cu_seqlens_q`, `cu_seqlens_k`, maximum sequence lengths, and optional query causal offsets.

The returned score tensor includes TMA padding on the physical KV extent. Use the documented logical sequence length when feeding top-k; do not interpret padded columns as valid keys.

## indexer_top_k_wrapper

Pass a two-dimensional score view, per-row valid sequence lengths, `top_k`, and optional `next_n`. `top_k` must be positive and no greater than 2048. `return_val=False` omits values. The indices are local to the row's current KV range unless converted with `local_to_global_wrapper`.

## Score recomputation

Sparse recomputation consumes top-k indices and optionally `topk_length`. `topk_indices_global` must match the index representation produced by the preceding stage. Dense recomputation accepts optional THD sequence metadata. Indexer recomputation also consumes learned weights; attention recomputation consumes LSE and the same softmax scale used by attention forward.

Output tensors are FP32 score data. Caller-provided `out` and denominator buffers must have the exact shape, dtype, device, and contiguous dimension required by the selected implementation.

## Backward stages

`sparse_attention_backward_wrapper` consumes Q, combined KV, forward output, upstream gradient, natural-log LSE, sink values, and selected indices. Its FlashMLA-shaped SM90 path supports head dimensions 512 and 576; other shapes are checked by the current implementation. Preserve `softmax_scale` and `topk_length` from forward.

`indexer_backward_wrapper` runs three stages and accumulates into caller-provided or allocated gradient tensors. `dense_indexer_backward_wrapper` additionally accepts dense-attention L1 norm, indexer LSE, THD metadata, and causal offsets. Both accept a scalar FP32 `grad_loss` and optional `loss_coeff`.

## Class APIs

Kernel-backed stages have corresponding classes such as `IndexerForward`, `IndexerTopK`, `SparseAttentionBackward`, or `DenseIndexerBackward`. Construct the class with representative sample tensors to compile a specialization, then call `execute` with tensors of the same metadata. `local_to_global_wrapper` and `compactify_wrapper` are utility wrappers without public class counterparts. Prefer wrappers for one-off use and classes when the caller owns cache lifetime and repeated execution.

Current exports: [DSA namespace](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/deepseek_sparse_attention/__init__.py#L52).

## Expected result

The selected stage writes outputs with the documented shape and precision and
matches an independently composed dense or sparse reference. No end-to-end DSA
sample is published here; source-level wrapper inventory does not establish
runtime support for an untested combination.
