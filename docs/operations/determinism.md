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
PyTorch autograd through the same API. The two fused GDN step entries
(`GdnDecodeFused`, `GdnVerifyFused`) are imported from `cudnn` (lazily; also
under `cudnn.linear_attention.frost`): inference-only direct-call step kernels
with no backward. QSA APIs are in `cudnn.ops`. `DSA` is imported from `cudnn`.
The required settings below apply to this table.

| API | Direction | Datatype | SM90 (Hopper) | SM100/103 | SM107 (Rubin) |
| --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` (KDA) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_product` (GDP) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net` (GDN) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net_v2` (GDN-2) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `GdnDecodeFused` (fused GDN decode step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Supported | Unverified |
| `GdnVerifyFused` (fused GDN MTP-verify step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Replay only | Unverified |
| `qsa_forward` | fprop | BF16 | Unverified | Replay only | Replay only |
| `qsa_backward` | bprop | BF16 | Unverified | Not guaranteed | Not guaranteed |
| `CSA.csa_compressor_forward_wrapper` | fprop | BF16 output; FP32 APE | Replay only | Replay only (SM100); Unverified (SM103) | Unverified |
| `CSA.csa_compressor_backward_wrapper` (`deterministic=True`) | bprop | BF16 dKV/dScore; FP32 dAPE | Replay only | Replay only (SM100); Unverified (SM103) | Unverified |
| `CSA.csa_compressor_backward_wrapper` (default atomic dAPE) | bprop | BF16 dKV/dScore; FP32 dAPE | Not guaranteed | Not guaranteed | Not guaranteed |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16, BF16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | Unverified | Replay only | Replay only |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 or FP32 weights | Unverified | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | Unsupported | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 dK | Unverified | Supported | Supported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; FP32 dK | Unverified | Replay only | Replay only |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 or FP32 dK | Unverified | Not guaranteed | Not guaranteed |

`GdnVerifyFused` is batch-invariant under one gated-norm association only: the
one-request pick takes the two-CTAs-per-head association (`ctas2`, 4-row
chains), two and more requests one CTA per head (`ctas1`, 8-row chains), so a
request's normed BF16 output alone differs in its last bits from the same
request inside a batch by design (`norm_form_for(requests)`); under one
association its bytes are identical. `GdnDecodeFused` writes the same bytes for
a row alone and inside a batch under the entry's own picks.

## Required settings and precision limits

- Linear attention: select the FROST plan with `batch_invariant=True`.
  Initial/final state and state gradients use FP32.
- Fused GDN step entries (`GdnDecodeFused`, `GdnVerifyFused`): no determinism
  switch; every form the entry picks or a knob selects writes the one-CTA
  form's bytes, so the status holds under any knob and at every served
  geometry (`Geometry(K_HEADS, V_HEADS, HIDDEN)`; verified at the default
  27B-class geometry and four per-rank shards). SM100/103 is verified on
  cc 10.0 and cc 10.3; SM90 is declined by type. The entries also serve
  SM120-class parts (compute capability 12.x; verified on cc 12.0 with the
  forms a 99 KB per-CTA opt-in holds) with the same statuses; this page has
  no SM120 column.
- DSA sparse attention backward and sparse indexer backward: set
  `deterministic=True`. Sparse attention backward requires unique active
  indices per query.
- QSA and dense indexer backward expose no determinism switch.
- CSA compressor forward and dKV/dScore use fixed orders. Complete backward
  determinism requires `deterministic=True`, caller-owned workspace on the class API,
  and a prepared plan before CUDA Graph capture; the default dAPE path uses FP32
  atomics. The current replay tests cover fixed inputs, poisoned outputs/workspace and
  changed packed prefixes, but the determinism suite has not yet established batch
  invariance on the listed GPUs.
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
