# CSA module

Fused CuTe-DSL kernels for the CSA/HCA experimental attention variants (the components
that are not shared with the DSA module, which lives in
`python/cudnn/deepseek_sparse_attention/`).

- **Compressor**: fused forward+backward kernels for the `Compressor` gated-softmax
  pooling (THD packed layout): gather -> `+ APE` -> optional overlap-window transform
  (`coff == 2`) -> fp32 softmax -> gated weighted sum -> bf16 cast, as one kernel per
  direction. Ported from
  Megatron-LM ([PR #5984](https://github.com/NVIDIA/Megatron-LM/pull/5984), measurements
  in [issue #5968](https://github.com/NVIDIA/Megatron-LM/issues/5968)). See
  [docs/fe-oss-apis/csa.md](../../../docs/fe-oss-apis/csa.md).

## Acknowledgements

The fused Compressor kernels were contributed by the GLM training-performance team
(Zhipu AI). The CSA/HCA attention variants and the surrounding DSA/CSA kernel family are
by Hongxiao Bai, Jiayu Sun and Jie Fang.
