# Sparse Attention

These pages describe attention APIs with selected tokens, blocks, or masks,
and related compression operations. The APIs are experimental and subject to change.

| API | Use |
| --- | --- |
| [Native Sparse Attention (NSA)](/operations/sparse_attention/nsa) | Selection, compression, top-k, and sliding-window attention. |
| [DeepSeek Sparse Attention (DSA)](/operations/sparse_attention/dsa) | Sparse attention and its indexer, score, and top-k operations. |
| [Query Sparse Attention (QSA)](/operations/sparse_attention/qsa) | Attention over selected four-token blocks, with bitmap and indexed training APIs. |
| [HSTU Attention](/operations/sparse_attention/hstu_attention) | Masked attention for recommender models, with SiLU scores instead of softmax. |
| [HSTU LMSD](/operations/sparse_attention/hstu_lmsd) | Fused layer normalization, multiplication, SiLU, and dropout. |
| [Block Sparse Attention (BSA)](/operations/sparse_attention/bsa) | Attention over a caller-supplied block-sparse mask. |
| [CSA Fused Compressor](/operations/sparse_attention/csa) | Compression for compressed sparse attention, with aligned HCA backward. |
| [Flex Attention](/operations/sparse_attention/flex_attention) | Attention with reusable static-mask plans. |

See [Attention](/operations/Attention) for scaled dot-product attention and
[Linear Attention](/operations/LinearAttention) for recurrent attention APIs.
