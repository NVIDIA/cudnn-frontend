## DSA module

Install the JAX integration with
`pip install 'nvidia-cudnn-frontend[jax]'`. JAX classes and JIT-compiled
wrappers are available as direct `cudnn.jax` exports and through
`cudnn.jax.DSA`.

- **Indexer Forward**: CuTe-DSL score kernel (Q @ Kᵗ, ReLU, head reduce, ratio causal mask). Supports SM90 MQA and SM100+ with fixed BSHD/SBHD or packed THD JAX inputs; non-fused, so pair with **Indexer Top-K** for the top-K stage.
- **Indexer Top-K**: SM90+ CuTe-DSL radix top-K kernel with per-row ``seq_lens``.
- **Sparse Attention Backward**: DSA backward (FlashMLA-shape, SM90/SM100).
- **Sparse Indexer / Attention Score Recompute**: Sparse (top-K) recomputation of indexer and attention scores for training loss.
- **Dense Indexer / Attention Score Recompute**: Dense (full-KV) analogues of the above.
- **Indexer Backward**: Three-stage pipeline (score-grad, three GEMMs, dtype cast) for sparse top-K score tensors. JAX additionally supports packed THD tensors when `topk_indices_global=True`; the PyTorch sparse wrapper remains fixed BSHD.
- **Dense Indexer Backward**: Full-KV counterpart of Indexer Backward.

## Acknowledgements

The DSA/CSA kernels were a collaborative effort, jointly developed by: Hongxiao Bai, Jiayu Sun and Jie Fang
