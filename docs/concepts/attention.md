# Attention layouts, masks, and state

SDPA computes scores from query and key, modifies and masks those scores, applies softmax and optional dropout, then multiplies by value.

## Tensor layouts

The layout letters name the tensor axes and the order in which they appear:

| Layout | Shape | Meaning |
| --- | --- | --- |
| BSHD / BMHK | `[B, S, H, D]` / `[B, M, H, K]` | `B` is batch size, `S` or `M` is sequence length, `H` is the number of heads, and `D` or `K` is the embedding dimension per head. BSHD is the name used most often in this project; BMHK is equivalent notation with different letters for the same axes. |
| BHSD | `[B, H, S, D]` | `B` is batch size, `H` is the number of heads, `S` is sequence length, and `D` is the embedding dimension per head. Compared with BSHD, the head axis appears before the sequence axis. |
| THD | `[T, H, D]` | `T` is the total number of valid tokens across the batch, `H` is the number of heads, and `D` is the embedding dimension per head. Tokens from variable-length sequences are packed together without padding, and cumulative sequence lengths or ragged offsets identify each sequence. For sequence lengths $S_i$, $T = \sum_i S_i$. |

These names describe an axis convention, but explicit dimensions and strides remain authoritative. Two tensors with the same logical axes can use different physical storage when their strides differ.

[GQA](https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29#Multi-Query_Attention) (grouped-query attention) uses more query heads than key/value heads. Each key/value head is shared by a group of query heads; for example, 32 query heads and 8 key/value heads form groups of four query heads. Q, K, and V can use the same axis convention and compatible physical layouts; the defining difference is that Q has more heads than K and V. The query-head count must be divisible by the key/value-head count for equal-sized groups.

## Masks

Diagonal bands express causal and sliding windows. Alignment determines whether the diagonal is top-left or bottom-right. Padding masks use per-batch sequence lengths. Block masks pack tile visibility. Score modifier functions insert custom score dataflow and can conflict with built-in mask implementations.

## Statistics and dropout

Training forward can emit log-sum-exp statistics for backward. Dropout applies after softmax and uses either Philox seed/offset state or an explicit mask and scale. RNG state must advance according to the application's reproducibility policy.

## KV caches

Paged attention stores K/V in block containers and uses page tables of integer block indices. A logical maximum KV length controls score and RNG shapes. Sink tokens and cache updates are persistent application state, not virtual intermediates.

## Precision

FP8 and MXFP8 attention add input/output scales, descales, amax values, layout reorders, dimension limits, and architecture gates. Their contracts are not aliases for BF16 SDPA.

For release milestones, logical-versus-physical layout guidance, determinism, Flex Attention controls, and benchmark provenance, see [Attention support history and benchmarks](/concepts/attention-support-history).
