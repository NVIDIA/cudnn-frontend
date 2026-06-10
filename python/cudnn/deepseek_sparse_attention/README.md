## DSA module

- **Indexer Forward**: CuTe-DSL score kernel (Q @ Kᵗ, ReLU, head reduce, ratio causal mask). Supports the SM90 MQA path through `indexer_forward_wrapper` and SM100+ through `IndexerForward`; non-fused, so pair with **Indexer Top-K** for the top-K stage.
- **Indexer Top-K**: SM90+ CuTe-DSL radix top-K kernel with per-row ``seq_lens``.
- **Sparse Attention Backward**: DSA backward (FlashMLA-shape, SM90/SM100).
- **Sparse Indexer / Attention Score Recompute**: Sparse (top-K) recomputation of indexer and attention scores for training loss.
- **Dense Indexer / Attention Score Recompute**: Dense (full-KV) analogues of the above.
- **Indexer Backward**: Three-stage pipeline (score-grad, three GEMMs, dtype cast) for sparse top-K score tensors.
- **Dense Indexer Backward**: Full-KV counterpart of Indexer Backward.

## Acknowledgements

The DSA/CSA kernels were a collaborative effort, jointly developed by: Hongxiao Bai, Jiayu Sun and Jie Fang
