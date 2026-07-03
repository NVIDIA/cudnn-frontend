## DSA module

- **Indexer Forward**: `indexer_forward_wrapper` materializes dense scores on SM90/SM100; `indexer_forward_top_k_wrapper` is the separate SM100 compact-logits path that runs radix Top-K and its fused softmax without materializing the dense score tensor (BSHD/THD BF16 and MXFP8, with optional LSE).
- **Indexer Top-K**: SM90+ CuTe-DSL radix top-K kernel with per-row ``seq_lens``.
- **Sparse Attention Backward**: DSA backward (FlashMLA-shape, SM90/SM100).
- **Sparse Indexer / Attention Score Recompute**: Sparse (top-K) recomputation of indexer and attention scores for training loss.
- **Dense Indexer / Attention Score Recompute**: Dense (full-KV) analogues of the above.
- **Indexer Backward**: Three-stage pipeline (score-grad, three GEMMs, dtype cast) for sparse top-K score tensors.
- **Dense Indexer Backward**: Full-KV counterpart of Indexer Backward.

## Acknowledgements

The DSA/CSA kernels were a collaborative effort, jointly developed by: Hongxiao Bai, Jiayu Sun and Jie Fang
