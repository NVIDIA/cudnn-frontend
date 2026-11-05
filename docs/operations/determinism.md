# Determinism support

This page tracks bitwise determinism and batch invariance by API, datatype,
and GPU. Status applies to the documented configurations on a fixed GPU and
software runtime. It does not imply equal bytes across GPU architectures or
software versions.

## Status key

| Status | Bitwise repeatability | Batch invariance |
| --- | --- | --- |
| Supported | Supported | Supported |
| Replay only | Supported | Not guaranteed |
| Not guaranteed | Not guaranteed | Not guaranteed |
| Unverified | Not verified on this GPU | Not verified on this GPU |
| Unsupported | This API precision path is unavailable | This API precision path is unavailable |

Repeatability means identical result bytes for repeated calls with fixed
inputs and shapes. Batch invariance means a sequence's results stay identical
when its batch size, order, or neighbors change, with its inputs and upstream
gradients held fixed. Support is limited to the documented configurations;
unverified status does not imply an unavailable API.

## API support by GPU

The four linear-attention APIs are in `cudnn.linear_attention.ops`; bprop uses
PyTorch autograd through the same API. QSA APIs are in `cudnn.ops`. `DSA` is
imported from `cudnn`. The required settings below apply to this table.

| API | Direction | Datatype | SM90 (Hopper) | SM100/103 | SM107 (Rubin) |
| --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` (KDA) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_product` (GDP) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net` (GDN) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net_v2` (GDN-2) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `qsa_forward` | fprop | BF16 | Unverified | Replay only | Replay only |
| `qsa_backward` | bprop | BF16 | Unverified | Not guaranteed | Not guaranteed |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16, BF16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | Unverified | Replay only | Replay only |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 or FP32 weights | Unverified | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | Unsupported | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 dK | Unverified | Supported | Supported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; FP32 dK | Unverified | Replay only | Replay only |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 or FP32 dK | Unverified | Not guaranteed | Not guaranteed |

## Required settings and precision limits

- Linear attention: select the FROST plan with `batch_invariant=True`.
  Initial/final state and state gradients use FP32.
- DSA sparse attention backward and sparse indexer backward: set
  `deterministic=True`. Sparse attention backward requires unique active
  indices per query.
- QSA and dense indexer backward expose no determinism switch.
- MXFP8 indexer forward uses E8M0 block scales, one scale per 32 values.
  The Hopper FP8 path uses FP32 token/head scales. Indexer forward returns
  FP32 scores; the Hopper FP8 path also returns FP32 LSE.
- DSA sparse attention forward/backward accepts FP16/BF16 inputs. FP8/MXFP8
  inputs are unsupported on all listed GPUs.
- DSA sparse/dense indexer backward requires BF16 Q/K/weights and supports
  BF16 or FP32 dK. FP8/MXFP8 inputs are unsupported on all listed GPUs.
- Use a compatible CuTe DSL runtime, including `sm_107a` support for SM107.

Other linear-attention backends, FP16 indexer inputs, MXFP4, paged indexer,
fused/separate Top-K generation, THD dense indexer backward, score recompute,
pooled DSA, and distributed execution are outside this status page's scope.

See the [test report](https://github.com/Anerudhan/cudnn-frontend/blob/feat/attention-determinism-tests/test/python/determinism_goldens/README.md)
for the covered configurations, validation runtimes, and supporting measurements.
