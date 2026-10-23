# Attention support history and benchmarks

Use this page to understand when major SDPA capabilities entered cuDNN and where repository benchmark artifacts are produced. For deployment decisions, the individual SDPA API page and `check_support()` on the installed backend remain authoritative.

## Historical feature milestones

| cuDNN | SDPA milestone recorded by the repository's former guide |
| --- | --- |
| 8.9.3 | Initial FP16/BF16 SDPA on SM80+ |
| 8.9.6 | Padding, ALiBi, and bias masks |
| 9.0 | Sequence lengths no longer required to be multiples of 64 |
| 9.1 | FP8 SDPA on Hopper and newer |
| 9.2 | Sliding-window attention |
| 9.3 | Bottom-right causal masking for FP16/BF16 |
| 9.5 | Paged attention and dBias with variable sequence lengths |
| 9.6 | GQA with ragged offsets and broader bottom-right causal lengths |
| 9.7 | FP8 bottom-right causal masking on SM100+ and paged plus ragged composition |
| 9.9 | Broader decode head dimensions |
| 9.10.2 | Packed page tables; 9.10.0 and 9.10.1 were historically marked problematic |
| 9.11 | DeepSeek asymmetric head dimensions in selected Hopper/Blackwell backward paths |
| 9.13 | Unified SDPA, FP8-to-FP16/BF16 output on Blackwell, and sink tokens |
| 9.14 | Unified block masks; the former guide recorded a noncausal sliding-window issue for large KV sequences |
| 9.15 | Unified padding masks and paged inputs |
| 9.18 | Ragged support on Ampere/Ada and deterministic FP16/BF16 on Blackwell |
| 9.19 | Deterministic FP8 on Blackwell and FP8 QK=192/V=128 |
| 9.21 | Unified SM100 `unfuse_fma` control |

This table is historical context, not a complete compatibility matrix. Minor releases can contain fixes, and one feature's first appearance does not imply compatibility with every mask, layout, dtype, backward mode, or architecture.

## Flex attention controls

Current SDPA forward accepts `score_mod`, which builds a pointwise subgraph over the score tensor. Backward accepts the forward modifier and a corresponding `score_mod_bprop` callback. Inputs used by the modifier must first be registered in the parent graph, then passed into the callback's subgraph.

Custom score modifiers are not freely composable with every built-in ALiBi, padding, causal, sliding-window, or block-mask implementation. The current support checks reject unsupported combinations. Block masks use the implementation's packed tile representation; do not assume a framework boolean mask can be passed directly.

## Logical and physical layouts

BHSD and BSHD describe logical axis orders. The frontend ultimately uses explicit dimensions and strides, so two tensors with the same logical labels can have different physical storage. Dense SDPA samples commonly use BSHD or BHSD dimensions and encode physical storage through strides. BMHK is equivalent historical notation for BSHD and appears only where an interface or older source uses those letters.

When converting from a framework tensor, preserve its actual stride. Do not relabel a view as another layout unless the stride describes that storage and the selected backend supports it.

## Determinism

Forward SDPA is deterministic for a fixed supported plan and inputs unless the chosen feature introduces stochastic state such as dropout. Backward determinism is architecture, dtype, feature, and version dependent. Request `use_deterministic_algorithm=True`, then require `check_support()` to succeed; do not infer determinism from architecture alone.

## Repository benchmark suites

The current benchmark runner covers Llama 3.1, DeepSeek V3, Kimi-K2.6, GPT-OSS, Qwen 3.5, Wan 2.2, LTX-2, and autoregressive video-DiT configurations. It can compare cuDNN with available FlashAttention backends for BF16, FP8, and MXFP8 paths, with forward, backward, mask, and determinism controls.

The checked-in GB200 and GB300 result directories include CSV data and generated charts. The former guide highlighted these representative cases:

| Workload | Q heads | KV heads | Head dimensions | Masks |
| --- | ---: | ---: | --- | --- |
| Llama 3.1 | 64 | 8 | QK=V=128 | top-left causal and no mask |
| DeepSeek V3 | 128 | 128 | QK=192, V/O=128 | top-left causal and no mask |

See the [benchmark instructions](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/benchmark/sdpa_benchmark_training/README.md) and [checked-in result artifacts](https://github.com/NVIDIA/cudnn-frontend/tree/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/benchmark/sdpa_benchmark_training/results). Compare CSV metadata, backend versions, dtype, mask, pass, sequence lengths, warmup, and timing method before comparing two charts.
