## DSA module

- **Sparse Attention Forward**: SM90/SM100 sparse-prefill CuTe-DSL kernels. SM90 serves H16/H32/H64/H96/H128 with D512/D576; SM100 serves H64/D512 or D576, H96/D512 or D576 (two H64 launches on overlapping head windows), H128/D512 small-top-k Prefill, and H128/D576 or H128/D512 with `indexer_topk=2048` (two H64 launches on disjoint head windows) MQA shapes. Supports arbitrary logical K, invalid/OOB and duplicate indices, per-query lengths, attention sinks, and indexer LSE.
- **Indexer Forward**: `indexer_forward_wrapper` materializes dense scores on SM90/SM100. `indexer_forward_top_k_wrapper` runs compact score generation, radix Top-K, and optional fused softmax without materializing dense scores: SM90 serves the packed-THD ratio-4, D128, 64Q/1KV BF16 or E4M3-FP8 row; SM100 retains its BSHD/THD BF16, MXFP8, and MXFP4 paths (optional LSE on both architectures).
- **Indexer Top-K**: SM90+ CuTe-DSL radix top-K kernel with per-row ``seq_lens``.
- **Sparse Attention Backward**: DSA backward for flat MQA tensors on SM90/SM100; consumes the forward wrapper's `out` and KV-only `lse`.
- **Sparse Indexer / Attention Score Recompute**: Sparse (top-K) recomputation of indexer and attention scores for training loss.
- **Dense Indexer / Attention Score Recompute**: Dense (full-KV) analogues of the above (BSHD on SM90/SM100; the THD `cu_seqlens` layout is SM100+ only).
- **Indexer Backward**: Three-stage pipeline (score-grad, three GEMMs, dtype cast) for sparse top-K score tensors.
- **Dense Indexer Backward**: Full-KV counterpart of Indexer Backward.

## Acknowledgements

The DSA/CSA kernels were a collaborative effort, jointly developed by: Hongxiao Bai, Jiayu Sun, Jerry Chen, Jie Fang, Qi Zhang, Bin Chai and Yue Weng.

The SM90 sparse-attention forward schedule and pipeline are adapted primarily
from [FlashMLA's MIT-licensed Hopper implementation](https://github.com/deepseek-ai/FlashMLA/tree/4f38f29ef6793c228363e4af5be66d44e81167ba/csrc/kernels/sm90/prefill/sparse).
The cuDNN Frontend port adds wider H coverage, arbitrary logical K boundary
handling, bounded symbolic compilation, and an allocation-free CUDA Graph
prepared-execution contract.
