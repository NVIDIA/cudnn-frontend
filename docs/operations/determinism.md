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

Forward SDPA standalone loaders and fixed-body engines share the isolated
main implementations. SM120 padded head envelopes now use their covering
body's explicit engine identity; this ownership change does not extend the
repeatability or batch-invariance claims below. Replayed records still fix
split count, packing and tile choices. See the [retirement validation
record](../../test/python/determinism_goldens/README.md#forward-sdpa-body-retirement-2026-11-15)
for current-runtime checks and retained verification gaps.

## API support by GPU

The shared SDPA backward delta preprocess retains per-row arithmetic when
batch or head counts require a flat launch grid. Direct-helper replay,
batch-isolation and changed-input capture checks cover FP16/BF16 on SM80 and
FP16/BF16/E4M3 on SM89 and SM100, including half THD with independent physical origins.
This helper evidence does not upgrade any full SDPA API status below, establish
SDPA support on SM89, or qualify the new grid boundary on SM103, SM107 or
SM120. SM100 also retains the tested D512 THD graph/standalone integration;
full attention chains at the new boundary remain unverified. See the
[delta grid qualification record](../../test/python/determinism_goldens/README.md#shared-backward-delta-grid-boundaries-2026-11-16).

The four linear-attention APIs are in `cudnn.linear_attention.ops`; bprop uses
PyTorch autograd through the same API. The two fused GDN step entries
(`GdnDecodeFused`, `GdnVerifyFused`) are imported from `cudnn` (lazily; also
under `cudnn.linear_attention.frost`): inference-only direct-call step kernels
with no backward. `SharedExpertDecode` (the shared-expert decode pair, also
under `cudnn.moe.frost`), `AttentionGateQuantDecode` (the attention gate +
quantize decode step, also under `cudnn.attention_gate.frost`) and
`Fp8ProjectionDecode` (the FP8 projection for decode rows, also under
`cudnn.gemm`) are the same kind of entry. QSA APIs are in `cudnn.ops`. `DSA` is imported from `cudnn`.
`AddNormQuantDecode` (the add + RMSNorm + quantize decode step, also under
`cudnn.norm.frost`) is a direct-call step entry of the same kind.
`SwigluFp4Decode` (the FP4 gate / up product with SwiGLU for decode rows, also
under `cudnn.gemm`) is a direct-call step entry of the same kind.
`AttentionPrepareDecode` (the attention preparation decode step, also under
`cudnn.attention_prepare.frost`) is a direct-call step entry of the same kind.
`RowQuantizeFp4Decode` (the FP4 row quantization decode step, also under
`cudnn.row_quantize.frost`) is a direct-call step entry of the same kind.
The block-sparse attention (BSA) APIs `block_sparse_attention_forward`,
`block_sparse_attention_backward` and `block_sparse_attention_backward_metadata`
are in `cudnn.ops`; their rows cover the blk64 kernels (`sparse_block_size=64`).
The required settings below apply to this table. The `pygraph.custom_reduction`
rows describe only the named reducer bodies and consumed graph outputs, not
arbitrary user callbacks.

| API | Direction | Datatype | SM90 (Hopper) | SM100/103 | SM107 (Rubin) |
| --- | --- | --- | --- | --- | --- |
| `Conv3dRawSm100`, `Conv3dRmsNormSiluSm100`, `Conv3dRmsNormSiluPadSm100`, `Conv3dBiasResidualPadSm100`, `CausalConv3dWithCacheSm100` and their wrappers | fprop | BF16; FP32 with `allow_tf32=True` | Unsupported | Supported on SM100 | Unsupported |
| `RmsNormSiluPadSm100` and its wrapper | fprop | BF16, FP32 | Unsupported | Supported on SM100 | Unsupported |
| Cake `dsa_det_forward` / `dsa_det_backward` | fprop, bprop | BF16 | Unsupported | Unverified (SM100 repeatability and isolated-row bounds cases tested; general batch invariance and SM103 unverified) | Unverified |
| `grouped_gemm_glu_hadamard_quant_wrapper_sm100` (Torch); colwise RHT | fprop | NVFP4 E2M1 / E4M3; standard GEMM-blocked column scales | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Unverified |
| `grouped_gemm_glu_hadamard_quant_jax_forward`, JAX shared wrapper; colwise RHT | fprop | NVFP4 E2M1 / E4M3; standard GEMM-blocked column scales | Unsupported | SM100 replay verified; batch invariance and SM103 unverified | Unverified |
| `kimi_delta_attention` (KDA) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_product` (GDP) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net` (GDN) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net_v2` (GDN-2) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `GdnDecodeFused` (fused GDN decode step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Supported | Supported |
| `GdnVerifyFused` (fused GDN MTP-verify step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Replay only | Replay only |
| `cudnn.ops.causal_conv1d_update`, width four | fprop (inference step) | FP16 inputs, output, and state with FP32 accumulation | Supported (GH200, tested configurations) | Unverified | Unsupported |
| `SharedExpertDecode` (shared-expert decode pair) | fprop (inference step) | BF16 in / out; FP32 accumulate | Unsupported | Supported | Supported |
| `AttentionGateQuantDecode` (attention gate + quantize decode step) | fprop (inference step) | BF16 in; FP8 E4M3 out | Unsupported | SM103: Supported; SM100: Unverified | Supported |
| `Fp8ProjectionDecode` (FP8 split-K projection for decode rows) | fprop (inference step) | FP8 E4M3 in, one FP32 scale; BF16 out; FP32 accumulate | Unsupported | SM100: Supported; SM103: Supported at `ProjectionGeometry(4608, 8192)` and `(8192, 2048)`, Unverified at the other served geometries | Supported at `ProjectionGeometry(4608, 8192)` and `(8192, 2048)`; Unverified at the other served geometries |
| `AddNormQuantDecode` (add + RMSNorm + quantize decode step) | fprop (inference step) | BF16 in; BF16, FP8 E4M3 or NVFP4 out; FP32 accumulate | Unsupported | SM100: Supported; SM103: Unverified | Unverified |
| `SwigluFp4Decode` (block-scaled FP4 gate / up product with SwiGLU for decode rows) | fprop (inference step) | FP4 E2M1 in / out with E4M3 block scales; FP32 accumulate; BF16 intermediate | Unsupported | SM100: Supported; SM103: Unverified | Unverified |
| `AttentionPrepareDecode` (attention preparation decode step) | fprop (inference step) | BF16 in and out; FP32 accumulate | Unsupported | SM100: Supported; SM103: Unverified | Unverified |
| `RowQuantizeFp4Decode` (4-bit block-scaled quantization of decode rows) | fprop (inference step) | BF16 in; FP4 E2M1 codes with E4M3 block scales out | Unsupported | SM100: Supported; SM103: Unverified | Unverified |
| `simplified_sparse_attention_forward` | fprop | BF16 output; FP32 indexer loss | Unsupported | SM100: Replay only; SM103: Unverified | Unsupported |
| `simplified_sparse_attention_forward` | bprop | BF16 Q/K/V and indexer gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `simplified_sparse_attention_forward_jax` | fprop | BF16 output; FP32 indexer loss | Unsupported | SM100: Replay only; SM103: Unverified | Unsupported |
| `simplified_sparse_attention_forward_jax` | bprop | BF16 Q/K/V and indexer gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `dense_indexer_kl_forward`, `use_kf=False` | fprop / bprop | BF16 inputs/gradients; FP32 loss and saved gradients | Unverified | SM100: Replay only; SM103: Unverified | Unverified |
| `dense_indexer_kl_forward`, `use_kf=True` | fprop / bprop | BF16 inputs/gradients; FP32 loss and saved gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `qsa_forward` | fprop | BF16 | Unverified | Replay only | Replay only |
| `qsa_backward`, `qsa_indexed_backward` | bprop | BF16 | Unverified | Replay only | Replay only |
| `CSA.csa_compressor_forward_wrapper` | fprop | BF16 output; FP32 APE | Unverified | Replay only | Unverified |
| `CSA.csa_compressor_backward_wrapper` (`deterministic=True`) | bprop | BF16 dKV/dScore; FP32 dAPE | Unverified | Supported | Unverified |
| `CSA.csa_compressor_backward_wrapper` (default atomic dAPE) | bprop | BF16 dKV/dScore; FP32 dAPE | Not guaranteed | Not guaranteed | Not guaranteed |
| `DSA.sparse_attention_backward_wrapper(return_cli_target=True)` teacher output | bprop auxiliary | FP32 target; FP16/BF16 inputs | Unverified (tests added) | Unsupported | Unsupported |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16, BF16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | Unverified | Replay only | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | Unverified | Replay only | Replay only |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 or FP32 weights | Unverified | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | Unsupported | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | NVFP4 packed E2M1 Q/K; E4M3 scales; BF16 weights | Unsupported | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_top_k_wrapper` with `starts` and `deterministic=True` | fprop | FP32, FP16, BF16 | Supported | Supported on SM103; SM100 unverified | Unverified |
| `DSA.indexer_forward_top_k_wrapper` (`deterministic=True`) | fprop | BF16 Q/K; BF16 or FP32 weights | Supported | Supported (cc 10.0 verified; cc 10.3 unverified) | Unverified |
| `DSA.indexer_forward_top_k_wrapper` (`deterministic=True`) | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` (`deterministic=True`) | fprop | BF16 Q/K; BF16 or FP32 weights | Supported | Unsupported | Unsupported |
| `DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` (`deterministic=True`) | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 dK | Replay only (H32); H64 unverified | Supported | Supported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; FP32 dK | Replay only (H32); H64 unverified | Replay only | Replay only |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 or FP32 dK | Unverified | Not guaranteed | Not guaranteed |
| `block_sparse_attention_forward` (blk64, `sparse_block_size=64`) | fprop | BF16 | Supported | Supported (cc 10.3 verified; cc 10.0 unverified) | Unverified |
| `block_sparse_attention_forward` (blk64, `sparse_block_size=64`) | fprop | FP16 | Supported | Unsupported (the SM100/SM103 blk64 forward serves BF16 only) | Unverified |
| `block_sparse_attention_backward` (blk64, `sparse_block_size=64`; the default route and, on SM90, the `metadata=` route of `block_sparse_attention_backward_metadata`) | bprop | BF16 | Not guaranteed | Not guaranteed (cc 10.3 measured; cc 10.0 unverified; the prepared route is SM90-only) | Unverified |
| `pygraph.custom_reduction` with `reduction_modifiers.argmax/argmin` (Torch finite-value rule; index output) | fprop | FP16, BF16, FP32 values; INT32 indices | Unverified | Supported (SM100, tested configurations); SM103 unverified | Unverified |
| `pygraph.custom_reduction` with a direct FLOAT ADD body (sum output) | fprop | FP32 | Unverified | Supported (SM100, tested configurations); SM103 unverified | Unverified |
| FROST SM90 three-WG D512 dense-Q split forward (standalone and fixed-body qualification; fixed split count and packing) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested configurations) | Unsupported for this owner | Unsupported for this owner |
| FROST native D128/D256 dense-Q forward on the D512 body (64 x 64 tile; standalone qualification, fixed split count and packing) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested configurations) | Unverified | Unverified |
| FROST SM107 D256 swap-AB decode, 64-column tile (fixed packing and split count) | fprop only | matching FP16/BF16 Q/K/V/O; FP32 Stats | Unsupported for this tile | Unsupported for this tile | Unverified after output-store register fix; replay and batch checks added |
| `pygraph.sdpa`, FROST SM90 native D64 owner (dense, unpadded, unsplit, unpacked; fixed scheduler; O and FP32 Stats) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested noncausal and top-left causal configurations) | Unsupported for this owner | Unsupported for this owner |
| FROST SM90 dedicated D128 owner (exact dense D128, 128 x 128 tile; unpadded, unsplit, unpacked, no sink or window) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested configurations) | Unsupported | Unsupported |
| FROST grid / lean-walk split-combine component (fixed reducer form and split count) | fprop combine only | FP32 partial O/LSE, FP16/BF16 O | Unverified | Supported (tested configurations) | Supported (tested configurations) |
| FROST native D576/V512 dense-Q paged forward (standalone qualification) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Unverified | Supported (tested configurations) |
| FROST native D512 dense-Q paged forward (standalone qualification) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Unverified | Supported (tested configurations) |
| FROST native D512 dense-Q mixed-KV forward (CGA2, dense/paged KV) | fprop | FP16/BF16 Q/O, E4M3/E5M2 KV | Unsupported | Unsupported | Supported (tested configurations) |
| FROST native D256 dense-Q mixed-KV forward (CGA2, unpacked, dense/paged KV) | fprop | FP16/BF16 Q/O, E4M3/E5M2 KV | Unsupported | Unsupported | Supported (tested configurations) |
| FROST native D64 dense/THD paged forward (standalone qualification; fixed split count and packing) | fprop | FP16/BF16 Q/O; matching half or E4M3/E5M2 KV | Unsupported | Unverified | Supported (tested configurations, including matching-half THD splits) |
| FROST native D128 dense-Q paged forward (standalone qualification) | fprop | FP16/BF16 Q/O; matching half or E4M3/E5M2 KV | Unsupported | Unverified | Supported (tested configurations) |
| FROST D128 paged forward with at most eight live query rows per packed unit (fixed split count and packing) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported for this lowering | Supported (tested configurations) | Unsupported for this lowering |
| FROST SM100 D256 swap-AB paged decode with 2 to 8 KV chunks, reduced by the separate pass or inside the launch (`CLUSTER_K_REDUCE=1`; fixed packing and split count) | fprop | matching FP16/BF16 Q/K/V/O; FP32 Stats | Unsupported for this engine | Supported (SM100, tested configurations); SM103 unverified, and `CLUSTER_K_REDUCE=1` is declined there | Unsupported for this engine |
| `pygraph.sdpa_fp8`, FROST D512 paged KV with dense Q | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; FP16/BF16 or FP8 O | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |
| `pygraph.sdpa_fp8`, FROST D128 paged decode with at most eight live query rows per packed unit (fixed packing and split count) | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; FP16/BF16 or FP8 O | Unsupported for this lowering | SM100: Supported (tested configurations); SM103: Supported (tested D128 paged configurations, existing lowering) | Unsupported for this lowering |
| `pygraph.sdpa_fp8`, FROST D64/D128 paged KV with dense Q (fixed CTA width, packing and split count) | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; FP16/BF16 or matching FP8 O | Unsupported | Supported (tested configurations) | Unverified by this suite |
| `pygraph.sdpa_fp8`, FROST D128 nonpaged THD with independent physical origins (CGA2, split1, unpacked) | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; BF16 O | Unsupported | Supported (SM100, tested configurations); SM103 execution unverified | Supported (tested configurations) |
| `pygraph.sdpa_fp8`, FROST native D128 dense single-CTA splits (fixed split count and packing) | fprop | E4M3 to BF16; E5M2 to FP16; FP32 tensor scales | Unsupported | Unverified by this suite | Supported (tested configurations) |
| `pygraph.sdpa_fp8`, FROST D128 dense `SWAP_AB=1` (at most eight live query rows per packed unit; fixed packing and split count) | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; FP16/BF16 or FP8 O | Unsupported for this lowering | Supported (tested configurations) | Supported (tested configurations) |
| `pygraph.sdpa` / `sdpa_fp8`, FROST D512 nonpaged THD with independent physical origins (unsplit, unpacked) | fprop | FP16/BF16; E4M3/E5M2 with FP32 tensor scales and BF16 O | Unverified by this suite | Supported (SM100, tested configurations); full SM103 suite unverified | Supported (tested configurations) |
| `pygraph.sdpa_fp8`, FROST native D192/V128 and D256 nonpaged THD with independent physical origins (unsplit, unpacked) | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; BF16 O | Unsupported | Supported (SM100, tested configurations); SM103 execution unverified | Supported (tested configurations) |
| `pygraph.sdpa_mxfp8`, explicit SM100 D64/D128, D192/V128 and D256 implementation records | fprop | E4M3/E5M2 Q/K/V, E8M0 block scales; BF16 or FP8 O | Unsupported for these engines | Unverified for general batch invariance; fixed-plan replay verified on SM100 and SM103 in the tested configurations | Unsupported for these engines |
| `pygraph.sdpa_mxfp8`, FROST D128/D512 nonpaged THD with independent physical origins (unsplit, unpacked) | fprop | E4M3/E5M2 Q/K/V, E8M0 block scales; BF16 O | Unsupported | Supported (SM100, tested configurations); SM103 execution unverified | Unverified |
| `pygraph.sdpa_mxfp8`, FROST D256 nonpaged THD with independent physical origins (CGA1, unsplit, unpacked) | fprop | E4M3/E5M2 Q/K/V, E8M0 block scales; BF16 O | Unsupported | Supported (SM100, tested configurations); SM103 execution unverified | Supported (tested configurations) |
| `pygraph.sdpa` / standalone FROST D256 paged THD (CGA2, fixed PackGQA and split count, independent physical origins) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Supported (SM100, tested configurations); SM103 execution unverified | Unsupported for this combined path |
| `pygraph.sdpa`, FROST D256 nonpaged THD split with independent physical origins (CGA2, unpacked, fixed splits 2/5, B=1/4) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported for this lowering | Supported (tested configurations on both SM100 and SM103) | Supported (tested configurations) |
| `pygraph.sdpa`, FROST single-sequence THD setup (dense D128 or paged D64/D128, unsplit) | fprop | matching FP16/BF16 Q/K/V/O | Unverified | Supported (SM100, tested configurations); SM103 execution unverified | See native single-sequence limits below |
| `pygraph.sdpa`, FROST D64/D128 split THD setup (CGA1, fixed packing and split count) | fprop | matching FP16/BF16 Q/K/V/O | Unverified | Supported (SM100, tested configurations); SM103 execution unverified | Unverified by this suite |
| `pygraph.sdpa` / standalone FROST D64 packed THD prefill (CGA1/2, N128, unsplit, independent origins) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Supported (tested configurations) | Unverified by this suite |
| `pygraph.sdpa_backward`, explicit FROST SM100 D512 role-split and 2x2 owners (batch <= 65535; each GEMM head chunk respects the same batch/head product limit) | bprop | matching FP16/BF16 inputs and gradients | Unsupported for these engines | Unverified for general batch invariance; fixed-plan repeatability verified on SM100 in tested configurations; SM103 unverified | Unsupported for these engines |
| `pygraph.sdpa_backward`, fixed D256 and D192/V128 half implementations (20607, 20610, 20632, 20633) | bprop | matching FP16/BF16 inputs and gradients | Unsupported for these engines | SM100: repeatability and batch isolation verified at causal S=257, MHA/GQA; general batch invariance and SM103 unverified | Repeatability and batch isolation verified at causal S=257, MHA/GQA for the three fixed owners; general batch invariance unverified; deterministic-mode declarations unchanged |

| `pygraph.sdpa_backward`, explicit FROST SM100 D512 role-split and 2x2 owners | bprop | matching FP16/BF16 inputs and gradients | Unsupported for these engines | Unverified for general batch invariance; fixed-plan repeatability verified on SM100 in tested configurations; SM103 unverified | Unsupported for these engines |
| `pygraph.sdpa_backward`, fixed D256 and D192/V128 half implementations (20607, 20610, 20632, 20633), within final GEMM chunk-grid bounds | bprop | matching FP16/BF16 inputs and gradients | Unsupported for these engines | SM100: repeatability and batch isolation verified at causal S=257, MHA/GQA; general batch invariance and SM103 unverified | Repeatability and batch isolation verified at causal S=257, MHA/GQA for the three fixed owners; general batch invariance unverified; deterministic-mode declarations unchanged |
| `pygraph.sdpa_backward`, FROST SM107 D512 fixed implementation (20608) | bprop | matching FP16/BF16 inputs and gradients | Unsupported for this engine | Unsupported for this engine | Repeatability and batch isolation verified for dense causal D384 GQA / D512 MHA in FP16/BF16; general batch invariance unverified; the graph does not advertise deterministic mode |
| `pygraph.sdpa_fp8_backward` / `sdpa_mxfp8_backward`, fixed SM107 implementations (20605, 20606), within full-batch/whole-group GEMM grid bounds | bprop | E4M3 inputs with tensor/block scales; BF16 gradients in these checks | Unsupported for these engines | Unsupported for these engines | Fixed-plan repeatability, record rebuild and changed-input capture verified at causal D256, S=257, MHA/GQA; general batch invariance unverified; deterministic-mode declarations unchanged |
| `pygraph.sdpa_fp8_backward`, FROST exact D512 dense pair | bprop | E4M3 payloads and FP32 tensor scales; E4M3/FP16/BF16 gradients | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |
| `pygraph.sdpa_mxfp8_backward`, FROST exact D512 dense pair | bprop | E4M3 payloads and canonical E8M0 block scales; FP16/BF16 gradients | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |
| `pygraph.sdpa_mxfp8_backward`, FROST `sdpa_bwd_sm100_d128_mxfp8` (opt-in; D128, BHSD-compact, S_q == S_kv, unmasked or causal, either anchor) | bprop | E4M3 payloads and E8M0 block scales; BF16 `o_f16` / `dO_f16` / gradients; optional FP32 amax | Unsupported | Supported with `use_deterministic_algorithm`, Replay only without it (SM100 and SM103, tested configurations) | Unsupported |
| `pygraph.sdpa_mxfp8` with `causal_exclude_own_block=32` | fprop | MXFP8 E4M3 or E5M2 Q/K/V; E8M0 scales; BF16 O, FP32 Stats | Unsupported | Supported (cc 10.0 verified; cc 10.3 unverified) | Unsupported |
| `grouped_gemm_glu_jax_sm100`, `situglu` | fprop | MXFP8 | Unsupported | Unverified | Unsupported |
| `grouped_gemm_dglu_jax_sm100`, `dsituglu` | bprop | MXFP8 | Unsupported | Unverified; atomic dprob is not guaranteed | Unsupported |
| `grouped_gemm_glu_hadamard_wrapper_sm100`, `srelu` with `tanh_clamp_scale` and `rht_sign_mask` | fprop | FP4 inputs with E8M0 block scales; BF16 C/D; FP32 Amax and PostRhtAmax | Unsupported | Replay only (tested configurations) | Unverified |
| `cutemok_forward` / `cutemok_backward` | fprop, bprop | BF16; MXFP8 routed weights | Unsupported | Unverified | Unsupported |

On SM89 (GeForce RTX 4090), the FP16 width-four `cudnn.ops.causal_conv1d_update`
passed replay, batch-isolation, and golden checks for both output and state.
On SM90 (GH200) the same checks pass, and the golden records equal the SM89
ones byte for byte.

Exact D256 FP16/BF16 on SM100/SM103 has separate lookahead and swap-AB
engine entries. The public engine/knob record fixes the main body and Q tile;
this alone does not guarantee batch invariance across all legal configurations.
The refactor retains the existing arithmetic and support conclusions above.
Its fresh-graph replay and parent-output comparisons are bounded correctness
checks, not a new general determinism qualification. Backward is unchanged.
See the [coverage record](../../test/python/determinism_goldens/README.md#sm100sm103-d256-engine-isolation-pilot--2026-11-14).
| `pygraph.sdpa_backward`, FROST dense D64 / D128 key-owner kernel with the ordered dQ relay (`sdpa_bwd_sm90`, opt-in), including bottom-right causal when `S_q <= S_kv` | bprop | FP16/BF16 Q/K/V/O/dO and gradients; FP32 Stats | Supported (tested configurations) | Unsupported | Unsupported |

The SM100 D256 swap-AB split decode tile reduces its per-chunk partial results
in the separate reduction pass or, with `CLUSTER_K_REDUCE=1`, inside its own
launch. Both forms add the partials in split order with the same operations.
In the tested configurations (2 to 8 chunks; FP16/BF16; head ratios 24/4, 8/1,
16/4, 32/8 and 32/2; head widths 256, 224 and 192; packed units of one to four
tokens and unpacked units of one to eight; pages of 16, 32 and 64 with
per-batch lengths and a dense cache without them; padding, bottom-right and
top-left causal masks and a sliding window) they store identical O and Stats
bytes, so
the knob does not change a plan's result at a fixed split count. The result
still follows the split count, which the heuristics choose from the declared
cache length and the batch: a caller that needs a sequence's bytes to stay
equal across batch sizes fixes `SPLIT_KV`. At a fixed split count the tested
sequences keep their bytes alone, among 8 and among 64, reordered, duplicated
and with changed neighbours, in both forms. Bytes are not equal across split
counts, to the unsplit plan, or to another engine. The configurations were
run on a 148-SM and on a 152-SM compute capability 10.0 device (x86-64 and
aarch64 hosts). On the 152-SM device the heuristics' default plan was also run
in each form -- inside the cluster counts recorded for that part and, at 4 and
8 chunks, one cluster past them -- and rebuilt from public records with equal
bytes. At 76 and 77 clusters of 2 chunks it ran in the form the count gives
and matched the reference, and explicit records of both forms stored equal
bytes there. The two golden cases are recorded for one runtime of each device.
SM103 execution is unverified.
See the coverage records of the
[suite](../../test/python/determinism_goldens/README.md#sm100-d256-swap-ab-in-launch-reduction-2026-11-18),
of its
[152-SM run](../../test/python/determinism_goldens/README.md#sm100-d256-swap-ab-in-launch-reduction-on-a-152-sm-part-2026-11-19)
and of that part's
[goldens](../../test/python/determinism_goldens/README.md#sm100-d256-swap-ab-in-launch-reduction-goldens-for-a-152-sm-part-2026-11-20).

The D64/D128 per-tensor FP8 paging checks preserve quantized payloads and
scales across reordered, duplicated and changed-neighbor batches. O and Stats
are batch invariant for the tested fixed plans; Amax is a whole-batch reduction,
checked numerically after rebatching and bitwise during repeated execution.
The coalesced page-table loader preserves strict parent-output goldens on
SM100 and SM103. The interleaved SM100 two-KV-stream schedule retains these
checks, including refilled-ring stress and delayed-MMA odd tails; SM103 keeps
its existing TMEM-P schedule. This adds no paged-KV backward path.
The separate SM100 paged D128 Q/K and V producers preserve the tested fixed-plan
O/Stats batch invariance and parent output bytes. HND/NHD capture checks use
independent K/V page tables and restore exact O/Amax bytes after switching live
KV lengths through shorter, empty and original batches. SM103 retains its
existing producer path and passes the same contract checks.

The SM100 eight-live-row FP8 decode lowering preserves quantized payloads and
per-tensor scales during batch changes. Its fixed-plan checks cover E4M3/E5M2,
FP8 and half outputs, optional Amax/Stats, natural-log and log2 final Stats,
and padded/strided storage. O and Stats are batch invariant; Amax is compared
bitwise for a repeated batch and numerically after rebatching. Split partial
Stats always use natural logarithms. This lowering changes the reduction order
and FP8 probability conversion relative to the older implementation, so it does
not preserve that implementation's output bytes. SM103 retains its existing
FP8 lowering. No backward entry point is added.

Dense D128 FP8 graphs can explicitly select the eight-live-row body with
`SWAP_AB=1` on SM100, SM103 and SM107. The CI-selected
`TestDenseFp8SwapAB` checks preserve quantized input payloads and per-tensor
scales under reordering, duplication and changed neighboring sequences.
O and Stats are compared across batches; whole-batch Amax is compared across
replays and checked numerically after rebatching. The existing D128 admission
rules still apply, including the decline for padded dense splits. Unsplit
checks cover empty actual Q/KV lengths; split partial Stats use natural logs
regardless of the final Stats base. `SWAP_AB=0` and an unset knob keep the
prefill body, whose output bytes need not equal this body's. This is forward
only and adds no backward implementation.
The SM100/SM103 dense body also preserves O, Stats and Amax bytes under
retained replay with output batch strides or address products beyond 32 bits.
These wide-address checks cover both FP8 input types and all four scalar
output types; batch invariance retains the fixed-plan scope described above.

SM100 D64 per-tensor FP8 paging also offers an explicit `SWAP_AB=1`
implementation. Each token unit computes eight packed query columns using a
D128 physical tile with TMA zero fill; up to four units serve a short query.
`SWAP_AB=0` and an unset knob retain the existing D64 prefill implementation.
The new implementation has the same fixed-plan replay and O/Stats batch
invariance contract, including E4M3/E5M2 inputs, mixed logical QK/V widths
inside the D64 envelope, split KV and retained capture. Amax remains a
whole-batch reduction. Equality is not promised between the two implementations.
The knob's candidate selection belongs to the graph heuristics; these results
qualify the implementation on a local SM100 development GPU, not a new default
ranking or an SM103 path.

The native D128 paging row fixes the CTA width, tile width and split count per
plan. Its standalone suite checks O/Stats bytes across repeated launches,
reordered/duplicated batches and fresh buffers, including mixed FP8 KV storage.
Graph qualification separately checks numerical outputs and changed-input replay.
The D64 mixed-storage checks also cover fixed-plan dense splits and unsplit
THD, preserving each quantized KV payload across batch changes. D64 THD
split with FP8 KV remains unsupported. These paging changes add no backward path.
Matching-half D64 THD split checks fix CGA1/N128 and the split count; they
cover fresh allocations, concurrent GPU work, ragged/empty sequences,
reordered and duplicated batches, and exact O/Stats bytes. This does not
guarantee identical bytes between split counts or between split and unsplit plans.

The SM107 D512/D576 split partial-store optimization preserves the existing
O/Stats golden bytes for matching half storage and D512 mixed FP8 KV storage.
Its checks retain a fixed split count and packing, and include partial value
widths and empty rows. Vector stores do not change the reduction order or
extend determinism across different plans or runtimes.

The D512 FP8 paged row covers the native role-split forward path with whole-group
PackGQA and a fixed split count (1 or 4 in its determinism suite). Its batch
checks preserve quantized payloads and scalar scales, and compare per-sequence
O and Stats. Amax is a whole-batch reduction: repeated calls compare its bytes,
while changed batches check its numerical value against the full batch reference.
The paging addition does not add an FP8 paged backward path.
The SM107 exact D512 public record names `sdpa_fwd_sm107_d512_fp8_role_split`
(engine 20549). Its fixed-plan checks include independent THD origins, fresh
bindings and retained CUDA Graph execution. Packing and split count remain
fixed during byte comparisons; the implementation identity does not extend
the numerical or cross-runtime guarantees of the table.

Exact SM107 D192/V128 and D256 per-tensor FP8 use engines 20561 and 20562.
Their fixed-record checks cover fresh bindings, retained captures, independent
THD origins and dense packing/splits, including the bounded D192 single-CTA
causal case and D256 sigmoid gate. Reference checks and replay/batch-byte
checks pass on the qualified runtime; strict historical THD goldens are
unavailable for that runtime. These identities preserve the existing math
and do not promise equal bytes across split counts or softmax precisions.

Exact SM107 D64/D128 per-tensor FP8 uses native multi-Q engine 20563;
explicit dense D128 swap-AB uses 20564 with the existing shared device body.
Fixed-record checks preserve rebuild, fresh bindings and retained capture
across supported splits, while independent D128 THD origins and dense batch
transformations keep their existing byte guarantees. Rebinding block-scaled
D128 output includes its caller-owned scale-factor storage. Historical
goldens remain unverified where the runtime has no matching baseline.

The native D128 FP8 single-CTA split checks fix the plan's split count and
packing, preserve quantized payloads, and compare O/Stats across fresh
buffers, reordered and duplicated sequences, and changed neighbors. Amax_O
is compared bitwise on replay and numerically for each changed batch.
The dense forward addition does not extend the THD or backward contract.

`GdnVerifyFused` is batch-invariant under one gated-norm association only: the
one-request pick takes the two-CTAs-per-head association (`ctas2`, 4-row
chains), two and more requests one CTA per head (`ctas1`, 8-row chains), so a
request's normed BF16 output alone differs in its last bits from the same
request inside a batch by design (`norm_form_for(requests)`); under one
association its bytes are identical. `GdnDecodeFused` writes the same bytes for
a row alone and inside a batch under the entry's own picks.

## Required settings and precision limits

The Conv3D/post-op APIs provide deterministic inference by default. FP32
convolution requires `allow_tf32=True`; standalone RMSNorm/SiLU preparation
supports strict FP32. Callers populate the history interiors of fused prepared
outputs. See the [Conv3D coverage report](../../test/python/determinism_goldens/README.md#conv3d-post-operations)
for replay, batch-invariance, and stress-test results.

- `grouped_gemm_glu_hadamard_wrapper_sm100` with `act_func="srelu"`: no deterministic
  mode is needed for replay. The qualified case sets `tanh_clamp_scale` and
  `rht_sign_mask`, covers both `use_tmem_post_rht_amax` paths, and uses unequal
  expert sizes with one empty expert; C, D, Amax and PostRhtAmax keep identical
  bytes across repeated calls under concurrent work on another stream. Batch
  invariance is not established: Amax and PostRhtAmax are whole-expert
  reductions and PostRhtAmax depends on 16-token block membership. Unclamped
  SReLU D is additionally bit-identical to the activation recomputed from the
  saved C for BF16 and FP16 C/D. The clamped path's approximate `tanh` is not
  bitwise equal to a host `tanh`. Other activations and SM107 are not covered
  by this row.

- Phase 1 dense indexer KL: no deterministic mode. The chunked PyTorch path
  repeats on SM100 but changes loss or saved FP32 gradient bytes when packed
  rows move. The fused CuTe path also changes scalar loss and saved gradient
  bytes across runs through atomic reductions. Only indexer Q/K receive
  gradients; the teacher stays frozen. Tiny single-row goldens do not qualify
  larger reductions. CPU and other GPUs are not qualified for determinism.

- Simplified sparse attention (PyTorch and JAX): no deterministic mode. The SM100 forward
  result covers short packed sequences with Hq 16/32/96 and top-k
  512/1024/2048, plus an Hq16 scheduling-boundary case. Long filtered-indexer
  batch invariance and saved internal statistics are unverified. PyTorch host-thread
  isolation is checked for dense and filtered repair calls on shared and separate streams.
  Triangular two-tile forward replay is checked at all three head counts.
  Moving packed rows changes the scalar loss bytes for Hq32/top-k1024 and
  Hq96/top-k2048. CI also found this for PyTorch Hq16/top-k512.
  Its approved exception covers only the observed one-byte packed forward loss mismatch on SM100.
  Backward uses atomic reductions; dK/dV fail raw-byte replay and batch invariance.
  Numerical agreement does not establish bitwise determinism.

- Custom reduction: these rows have no determinism switch and pin the native
  cuDNN backend. SM100 support applies only to the suite's dense, 257-element
  row reductions and consumed outputs. Argmax/argmin use bounded finite values
  with ties and row-local INT32 indices; NaN/infinity behavior is unverified.
  The direct FLOAT ADD body uses the suite's bounded multiples of one eighth,
  whose partial sums are all exactly representable. Arbitrary floating-point
  sum inputs, other reducer bodies, and unused tuple outputs are unverified.
  No custom-reduction backward entry point exists. SM103, SM90, SM107, and
  other unmeasured GPU paths remain unverified.
- Linear attention: select the FROST plan with `batch_invariant=True`.
  Initial/final state and state gradients use FP32.
  KDA's packed workspace uses a shape-only bound on the number of piece slots;
  it preserves the per-sequence split rule and arithmetic. Compact-workspace
  output, gradient and CP-summary byte parity is additionally checked on SM103
  for BF16/FP16, including changed-length CUDA Graph replay. Large KDA backward
  plans can also reuse checkpoint and owned-state storage across independent
  head waves. This preserves each head's recurrence, FP32 state and reduction
  order; BF16/FP16 grouped heads, rectangular states and external contexts have
  additional SM103 byte-parity coverage. Runtime-head reuse of compiled code
  and independent CUDA Graph replays on two streams with separate workspaces
  are also checked against the unsplit backward path on SM103, including H96
  with BF16/log gates and FP32 states (two/four-wave backward). Forward state
  waves and eight-wave backward have additional SM103 byte-parity coverage
  for grouped heads, rectangular forward states, public checkpoints and
  changed-prefix graph replay. Exact FP32 forward H/X storage reuse has
  additional SM100 byte-parity coverage, including rectangular states,
  forward/reverse chains, changed-input graph replay and public checkpoints.
  Four/eight-wave forward and sixteen-wave backward also have SM100 coverage
  for grouped heads, log/safe gates and runtime-head reuse through H96.
  Their automatic policy may retain more storage to protect parallelism;
  it is not a hard workspace cap. The new H/X alias and sixteen-wave paths
  have not been requalified on SM103 or SM107; earlier architecture coverage
  remains recorded separately in the test report.
- Linear attention cuTile engine (GDN, KDA): the launch autotune times
  occupancy and worker-warp candidates, and the worker-warp count changes the
  cross-warp summation trees of the kernels that reduce with `ct.sum`, so two
  processes that time their own candidates can write different bytes. Every
  candidate replays its own bytes, and a run whose tuning fails or times out
  launches the first candidate's hints rather than the unhinted kernel. A host
  that needs the same bytes on every rank and in every run pins the pick
  through the hooks in `cutile/kernels/common.py`: `record_hint` is told each
  timed winner, `select_hints` returns the candidate to run and skips the
  timing (Megatron-LM's `megatron.core.tuning` records a table and pins from
  it). The hooks are read only when a launch misses the in-process hint
  cache, so a host installs them before the first launch or clears
  `launch_hint_cache` when its policy changes; a failed or timed-out search
  launches the first candidate and records nothing. The cuTile engine is not on this page's table: per-candidate replay and
  the hooks are verified on cc 10.3 by `test_la.py`'s determinism tests, batch
  invariance is not claimed.
- Fused GDN step entries (`GdnDecodeFused`, `GdnVerifyFused`): no determinism
  switch; every form the entry picks or a knob selects writes the one-CTA
  form's bytes, so the status holds under any knob and at every served
  geometry (`Geometry(K_HEADS, V_HEADS, HIDDEN)`; verified at the default
  27B-class geometry and five per-rank shards). SM100/103 is verified on
  cc 10.0 and cc 10.3, SM107 on cc 10.7 (a VR200-class part, 212 SMs);
  SM90 is declined by type. The entries also serve
  SM120-class parts (compute capability 12.x; verified on cc 12.0 with the
  forms a 99 KB per-CTA opt-in holds, at the default geometry and four of
  the five shards -- `Geometry(2, 8, 4096)` is unverified there) with the
  same statuses; this page has no SM120 column.
- Shared-expert decode pair (`SharedExpertDecode`): no determinism switch; the
  accumulation order of every output element is a function of the geometry
  alone (`K_SPLIT` pinned per geometry -- never per part: a GPU too small for
  the pinned form is declined by type -- the gate tree's shape by
  `GATE_CLUSTER_FOLD`), so a token's bytes alone equal its bytes inside a batch
  of up to 24 tokens, across the one- and two-m-tile forms and the one- and
  two-group forms; every other knob writes the default's bytes, and the first
  release's form (`v1_cfg`) keeps its own goldens as an A/B arm. SM100/103 is
  verified on cc 10.0 (a GB200, 152 SMs) and cc 10.3 (a GB300, 152 SMs),
  SM107 on cc 10.7 (a VR200-class part, 212 SMs). The committed goldens
  belong to one cc 10.3 runtime and one cc 10.7 runtime: on cc 10.0 the
  status rests on the replay and batch-invariance cells, and the golden
  comparison is not covered there. SM90 and SM120 are declined by type.
- Attention gate + quantize decode step (`AttentionGateQuantDecode`): no
  determinism switch and no numerics knob. The kernel has no reduction, so a
  row's bytes are a function of its own operand values and of the scale: a
  row alone equals the same row inside a call of up to 160 rows, at any
  position and among any neighbours. Both knobs (`THREADS`, `PDL_RELEASE`)
  and the thread-count pick write the default's bytes; the release is
  checked with no dependent grid on the stream. SM103 is verified on cc 10.3
  (a GB300, 152 SMs) and SM107 on cc 10.7 (a VR200-class part, 212 SMs),
  each with its own committed goldens; cc 10.0 has not been run, so SM100 is
  unverified (the entry admits compute capability 10.x, and the kernel's
  rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets).
  SM90 and SM120 are declined by type. The bytes follow the
  installed CuTe DSL's lowering of the exponential: the status holds on a
  fixed runtime, and a golden recorded on one runtime does not qualify
  another.
- FP8 projection for decode rows (`Fp8ProjectionDecode`): no determinism
  switch. The result bytes are a function of `(geometry, split_k)` on one
  runtime: the number of K slices is fixed when the entry is built (given, or
  a function of the geometry alone -- never of the row count or of the part),
  and every other knob, including the pick per row count and part and the
  order in which the epilogue releases the tensor memory, is a performance
  form that writes the same bytes. A row's bytes alone equal its bytes among
  up to 32 rows at any position, across the 8- / 16- / 32-row token tiles; the
  kernel is specialised on the row count, so replay and batch invariance are
  verified at every row count 1 .. 32 of each served class
  (`ProjectionGeometry(4608, 8192)` at split 4 and at split 2,
  `ProjectionGeometry(8192, 2048)` at split 2, `ProjectionGeometry(5120, 6144)`
  at split 4 and at split 2, `ProjectionGeometry(16384, 5120)` and
  `ProjectionGeometry(14336, 5120)` at split 1). Another split is another byte
  class and another entry; the status holds per entry. The evidence differs by
  class. `ProjectionGeometry(4608, 8192)` and `ProjectionGeometry(8192, 2048)`
  are verified on cc 10.0 (148 SMs), cc 10.3 (a GB300, 152 SMs) and cc 10.7 (a
  VR200-class part, 212 SMs). The other three geometries are verified on
  cc 10.0 only: the entry admits them on cc 10.3 and cc 10.7, where they have
  not been run. Goldens are committed for one cc 10.3 runtime (the first three
  classes) and for one cc 10.0 runtime (all seven); on any other runtime the
  status rests on the replay and batch-invariance cells and the golden
  comparison is not covered. No backward entry point exists. SM90 and SM120
  are declined by type.
- Add + RMSNorm + quantize decode step (`AddNormQuantDecode`): no
  determinism switch and no numerics knob. A row's sum of squares is one
  fixed tree whose shape is a function of `hidden` alone (a fused chain over
  a thread's eight elements, a butterfly over a warp's lanes, a butterfly
  over the warps) -- never of the row count, of the part or of a knob -- and
  nothing of another row enters a row's thread block. A row alone writes the
  bytes it writes inside a call of up to 1024 rows, at any position and among
  any neighbours: the residual update, the bf16 output, the codes and the
  row's NVFP4 block scales (a block scale's position in the 128 x 4 scale
  layout follows the row index; its value does not). The output set (`bf16`,
  `store_residual`), the launch knobs (`pdl`, `pdl_release`) and the owner of
  an output write the same bytes. SM100 is verified on cc 10.0 (148 SMs)
  with the committed goldens of that one runtime, under both compiler
  flavours of CuTe DSL 4.8.0. cc 10.3 and cc 10.7 have not been run, so
  SM103 and SM107 are unverified: the entry admits compute capability 10.x,
  the kernel's rendering is pinned for the `sm_100a`, `sm_103a` and
  `sm_107a` targets, and cc 10.7 needs the DSL's CUDA 13 compiler flavour.
  SM90 and SM120 are declined by type. The bytes follow the part's
  approximate reciprocal square root and approximate reciprocal: the status
  holds on a fixed runtime, equal bytes across GPU architectures are not
  claimed (an NVFP4 code at an exact rounding tie under a block scale that is
  not a power of two follows the approximate reciprocal), and a golden
  recorded on one runtime does not qualify another. No backward entry point
  exists.
- SwiGLU FP4 gate / up product for decode rows (`SwigluFp4Decode`): no
  determinism switch and no numerics knob. The kernel has no split and no
  reduction across rows: an output element is one fp32 chain in ascending K,
  and a block scale is a function of its own row's 16 channels (the two lanes
  that own a block exchange under a mask of one token's lanes). A row's packed
  codes and block scales alone equal its bytes among up to 64 rows at any
  position, across the 8- / 16- / 32- / 64-token tiles; one compiled kernel
  per token tile serves every row count the tile holds, and replay and batch
  invariance are verified at every row count 1 .. 64 of both served
  geometries (`SwigluGeometry(5120, 17408)`, `SwigluGeometry(5120, 128)`).
  Every config field (the token tile, the pre-wait stages, the L2 policy, the
  grid cap, the launch attribute) writes the same bytes. SM100 is verified on
  cc 10.0 (148 SMs) with the committed goldens of that one runtime, under the
  CUDA 13.4 compiler of CuTe DSL 4.8.0; an older DSL compiler is declined by
  name. cc 10.3 and cc 10.7 have not been run, so SM103 and SM107 are
  unverified: the entry admits compute capability 10.x and the kernel's
  rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets.
  SM90 and SM120 are declined by type. The bytes follow the part's
  approximate exponential, division and reciprocal: the status holds on a
  fixed runtime, equal bytes across GPU architectures are not claimed, and a
  golden recorded on one runtime does not qualify another. No backward entry
  point exists.
- Attention preparation decode step (`AttentionPrepareDecode`): no
  determinism switch and no numerics knob. A head row's sum of squares is one
  fixed tree whose shape is a function of the head width alone (a fused step
  over a thread's two elements, a butterfly over a warp's lanes with the
  offsets 16, 8, 4, 2, 1, a butterfly over the four warp sums) -- never of
  the token count, of the part or of a knob -- and nothing of another token
  or head enters a head row's thread block. A token alone writes the bytes it
  writes inside a call of up to 1024 tokens, at any place and among any
  neighbours: its `q_out` and `gate_out` rows and the K and V pool rows of its
  slot. Which pool rows a call writes follows the slot vector (a negative
  slot, a slot at or past the pools' capacity and a token past the vector's
  length write none) and never changes `q_out` or `gate_out`. The page size,
  the integer dtypes of the positions and the slots, the launch knobs (`pdl`,
  `pdl_release`) and the owner of an output write the same bytes. By the
  caller's contract two tokens of one call do not share a non-negative slot;
  a call that breaks it has two thread blocks writing one pool row, and the
  status does not cover it. SM100 is verified on cc 10.0 (148 SMs) with the
  committed goldens of that one runtime, under both compiler flavours of
  CuTe DSL 4.8.0. cc 10.3 and cc 10.7 have not been run, so SM103 and SM107
  are unverified: the entry admits compute capability 10.x, the kernel's
  rendering is pinned for the `sm_100a`, `sm_103a` and `sm_107a` targets,
  and cc 10.7 needs the DSL's CUDA 13 compiler flavour. SM90 and SM120 are
  declined by type. The bytes follow the part's approximate division and
  approximate reciprocal square root: the status holds on a fixed runtime,
  equal bytes across GPU architectures are not claimed, and a golden recorded
  on one runtime does not qualify another. No backward entry point exists.
- FP4 row quantization decode step (`RowQuantizeFp4Decode`): no
  determinism switch and no numerics knob. A block's bytes -- its eight code
  bytes and its one scale byte -- are a function of its sixteen values and of
  the global scale: one thread owns the whole block, and the kernel has no
  reduction across threads, no barrier and no shared memory. A row alone
  writes the bytes it writes inside a call of up to 1024 rows, at any position
  and among any neighbours: its codes and its block scales (a block scale's
  position in the 128 x 4 scale layout follows the row index; its value does
  not). The launch knobs (`pdl`, `pdl_release`) and the owner of an output
  write the same bytes. SM100 is verified on cc 10.0 (148 SMs) with the
  committed goldens of that one runtime, recorded under CuTe DSL 4.8.0 and
  checked under both of its compiler flavours; the suites also ran under both
  compiler flavours of CuTe DSL 4.8.1 with the same bytes. cc 10.3 and cc 10.7
  have not been run, so SM103 and SM107 are unverified: the entry admits
  compute capability 10.x, the kernel's rendering is pinned for the
  `sm_100a`, `sm_103a` and `sm_107a` targets, and cc 10.7 needs the DSL's CUDA
  13 compiler flavour. SM90 and SM120 are declined by type. The 1024-row call
  crosses a scheduling wave on a part of at most 170 SMs and no served call
  does on a 212-SM part; the kernel holds no state between thread blocks. The
  codes follow the part's approximate reciprocal: the status holds on a fixed
  runtime, equal bytes across GPU architectures are not claimed (a code at an
  exact rounding tie under a block scale that is not a power of two follows
  the approximate reciprocal), and a golden recorded on one runtime does not
  qualify another. No backward entry point exists.
- DSA sparse attention backward and sparse indexer backward: set
  `deterministic=True`. Sparse attention backward requires unique active
  indices per query, except in SM100/SM103 plans that store dKV in
  conflict-rank storage, which accept duplicate indices: BF16 H128 two-CTA
  plans and generic plans for which `deterministic_dkv_mode` selects records
  or the chain relay.
- On SM90, the default `deterministic_dkv_mode='records'` uses four
  conflict-rank planes and fixed-order plane-zero overflow tickets. A resident
  128-CTA wave pipelines independent dKV column groups; the two-launch schedule
  uses one ticket per row. An explicit `deterministic_relay_cluster_size`
  selects the shard relay instead. H20-3e BF16 H64/D512 and D576 high-conflict
  replay/reference and H96/D576 changed-input CUDA Graph replay are verified.
  The AIStudio H200 allocation additionally verifies the one-wave schedule with
  the same high-conflict and changed-input CUDA Graph cases. Every phase is a
  cooperative launch with at most one CTA per SM; smaller SM90 devices use
  additional ordered power-of-two phases. Cross-wave dKV remains batch
  dependent, so the SM90 BF16 status stays **Replay only**.
  Duplicate active indices within one query remain outside the SM90
  deterministic contract. FP16 and other SM90 head-count/dimension combinations
  are unverified for this new records path.
- On SM100/SM103, `deterministic_dkv_mode` selects the sparse attention
  backward dKV storage: `'records'` (default; conflict-rank storage where it
  fits the shard, relay or private-plane layout, that layout elsewhere),
  `'chain'`, or `'shards'`. Each mode is repeatable. The modes round dKV
  differently; generic M64 plans give the same dSink bytes under every mode.
  With any mode, as with the shard layouts alone, dKV is not batch invariant
  once a batch spans several waves: the layout, its waves and its plane or
  shard count follow the batch's query and KV extents. Forward results, dQ
  and, for generic M64 plans, dSink keep their bytes while the batch keeps the
  plan's layout. H128 two-CTA plans sum dSink in an order that follows query
  positions, and H128 D576 plans with top-k 1024 or wider sort each query's
  rows by ranges of the batch's KV extent, which also changes dQ. SM107 plans
  keep the shard and relay layouts: `'records'` runs as `'shards'` and
  `'chain'` is rejected.
- QSA backward (`qsa_backward`, `qsa_indexed_backward`): set
  `deterministic=True`. `dq`, `dk` and `dv` then repeat for fixed inputs and a
  fixed packed layout: every FP32 accumulation adds in an order the layout
  fixes. The bytes change with the packed layout (key tiles, query groups,
  query ranges and KV passes are positions in the packed buffer), with
  `max_seq_len` and with the GPU's SM count. Every
  document must use one four-token block grid, and a row must not repeat a
  block.
- Dense indexer backward exposes no determinism switch.
- CSA compressor forward and dKV/dScore use fixed orders. Complete backward
  determinism requires `deterministic=True`, caller-owned workspace on the class API,
  and a prepared plan before CUDA Graph capture; the default dAPE path uses FP32
  atomics. On SM100 and SM103, deterministic backward fixed-sequence batch comparisons
  cover ratios 2, 4, and 128, both overlap forms, scalar and wide generic layouts, both
  ratio-128 head dimensions, reordered and duplicated sequences, changed neighbors,
  and a ratio-128 schedule-boundary crossing. Forward passes the same comparisons except
  for the ratio-128 `coff=2`, head-dimension-512 small/default schedule boundary: SM100 CI
  observed a one-byte output difference when the fixed sequence moved from the small
  batch schedule to the default schedule. Forward is therefore Replay only. Replay and
  committed goldens cover the generic ratio-2, generic ratio-4 overlap, and dedicated
  ratio-128 numerical paths. SM90 and SM107 remain unverified for batch invariance.
- `sdpa_mxfp8(causal_exclude_own_block=32)`: no determinism switch. Every plan is repeatable. PackGQA does not change a row's bytes. A split-KV plan adds its per-split partials in a fixed order, so its bytes follow the split count; the heuristics choose that count from the launch grid (on B200 only a single sequence at one or two KV heads splits), so a row keeps its bytes across batches while the plan keeps its split count. The option is forward-only: `sdpa_mxfp8_backward` refuses it, so there is no bprop row. It is served by the SM100/SM103 MXFP8 D128 kernel only; SM90 and SM107 decline it. Rows 0-31 keep no key and return O = 0 and Stats = -inf, which the tests accept as the declared sentinel.
- MXFP8 indexer forward uses E8M0 block scales, one scale per 32 values.
  NVFP4 indexer forward reads packed E2M1 Q/K with non-negative E4M3 block
  scales, one per 16 values; the per-tensor global decode scales are folded
  into `sm_scale`, a launch argument, so they do not change the plan.
  The Hopper FP8 path uses FP32 token/head scales. Indexer forward returns
  FP32 scores; the Hopper FP8 path also returns FP32 LSE.
- SM90 combined Indexer Forward + Top-K requires packed THD, ratio 4, head
  dimension 128, 64 query heads per single KV head, and `deterministic=True`
  for the documented tie policy. Its indices, logits, optional selected-logit
  softmax, and optional LSE passed SM90 replay, batch reorder/duplication,
  changed-neighbor isolation, and exact-tie tests.
- The prepared SM90 Indexer Forward + Top-K plan has the same numerical and
  shape limits as that combined path and also requires caller-owned outputs
  and candidate workspace. Its retained scorer and radix Top-K artifacts pass
  direct fixed-input replay, packed batch reorder/duplication, changed-neighbor
  isolation, independent reference checks, and changed-metadata CUDA Graph
  replay. It is a forward-only execution surface; there is no corresponding
  prepared backward API. Other architectures are rejected by contract.
- DSA sparse attention forward/backward accepts FP16/BF16 inputs. FP8/MXFP8
  inputs are unsupported on all listed GPUs.
- DSA sparse/dense indexer backward requires BF16 Q/K/weights and supports
  BF16 or FP32 dK. FP8/MXFP8 inputs are unsupported on all listed GPUs.
- Block-sparse attention blk64: no determinism switch. The rows cover
  `sparse_block_size=64` with head dimension 128, MHA and BHSD tensors: fixed
  and variable block counts (`q2k_block_nums`; empty query blocks with
  `allow_empty_block_nums=True` give O = 0 and LSE = -inf), partial KV blocks
  (`block_sizes`), a KV length that is not a multiple of 64 (`block_sizes`
  marking the final block, as the API requires) and `kv_splits=2`; every
  forward configuration is measured for repeatability and for batch
  invariance in each datatype its row supports. SM90 requires `S_q` to be a
  multiple of 64. The forward computes each query tile in one CTA with no
  cross-CTA reduction (split KV combines FP32 partials in a fixed order). The
  backward (BF16 only) reduce-adds dQ in FP32 across the CTAs of a query
  block's KV blocks; dK/dV are reduce-added the same way through the FP32
  workspace (several query groups or a KV tail) and by the
  paired-KV kernel's later arrivers; and the K-to-Q CSR build orders each KV
  block's query list with an atomic cursor, so even the direct dK/dV stores of
  the single-group route accumulate in a call-dependent order. dQ, dK and dV
  are therefore not repeatable on any route, as `bsa.md` documents for its FP32
  atomics; an empty query block's dQ is zero. One narrow, verified exception on
  SM90: with prepared metadata (`block_sparse_attention_backward_metadata`,
  SM90 only) the single-group, whole-block route builds its CSR once and its dK
  and dV replay bitwise while dQ still differs; this does not raise the row's
  status. GQA/MQA, head dimensions 64 and 96, BSHD, the blk128 paths, KPage and
  FP8 are outside these rows.
- Use a compatible CuTe DSL runtime, including `sm_107a` support for SM107.

Other linear-attention backends, FP16 indexer inputs, MXFP4, paged indexer,
separate Top-K generation, THD dense indexer backward, score recompute,
pooled DSA, distributed execution, and the block-sparse attention blk128,
KPage and FP8 paths are outside this status page's scope.

See the [test report](https://github.com/NVIDIA/cudnn-frontend/blob/develop/test/python/determinism_goldens/README.md)
for the covered configurations, validation runtimes, and supporting measurements.


The D512 per-tensor FP8 backward pair uses fixed query/key owners and reduces
GQA heads in ascending order. Its bounded dense BSHD suite preserves all FP8
payloads, scalar scales, O, Stats and upstream gradients across batch changes;
it compares dQ/dK/dV bytes and whole-batch Amax reductions. Cases include
left windows, repeated/reordered/duplicated batches through 129 requests and
changed-input CUDA Graph replay. The deterministic flag selects the same
arithmetic. This qualification does not add MXFP8 or packed backward support.

The D512 MXFP8 backward pair uses the same fixed owner order, consuming
canonical E8M0 scale bytes and independently quantized columnwise payloads.
Its FP16/BF16 checks preserve these exact bytes across singleton, reordered,
duplicated and129-request batches, then mutate dO and a canonical scale
atom after CUDA Graph capture. Four runtime goldens cover SM100 and SM107;
SM103 execution remains unverified. The deterministic flag does not select
a different reduction. THD and other unsupported graph forms are unchanged.

The SM90 D64 / D128 half-precision backward chain (`sdpa_bwd_sm90`) owns dK and
dV per key tile and reduces every query tile's dQ over its key tiles in index
order through per-tile turn counters, so the deterministic flag and the default
request run the same arithmetic. Repeatability: dQ / dK / dV bytes are compared
across repeated executions with a poisoned workspace on causal and unmasked MHA
and GQA graphs at D64 and D128, including grids beyond one CTA wave, and across
CUDA Graph replays for causal GQA at D128. Batch invariance: the same bytes are
compared across reordered and duplicated batches and against the sequence
declared alone, in four cells: BF16 D128 causal GQA, FP16 D64 unmasked MHA,
BF16 D64 causal MHA and FP16 D128 unmasked GQA. Rectangular bottom-right causal
GQA also preserves all three gradients across the same batch changes at FP16 D64
and BF16 D128. The D128 case passes changed-gradient CUDA Graph replay under
both deterministic and default requests.

## Split-combine zero-weight qualification

The grid and lean-walk split-combine components preserve repeatability and
batch invariance for the tested FP32 partials with FP16/BF16 outputs on
SM100, SM103 and SM107, at fixed reducer form and split count. Coverage
includes discarded NaN/infinite payloads whose finite partial LSE produces
a zero weight, empty rows, cross-chain/round rescaling, multiple CTA waves
and changed-input CUDA Graph replay. Runtime-specific goldens check O and
LSE. This component qualification does not establish an attention producer's
or backward kernel's determinism, or identity across forms or split counts.

## SM80 SDPA sink qualification

Native FP16/BF16 THD forward with FP32 attention sinks supports repeatability
and batch invariance on SM80 for the tested D64 causal/window configurations,
with dense and paged K/V. This qualification uses a fixed GPU and runtime;
it does not imply identical bytes across GPU or software versions. Detailed
coverage is recorded in the determinism test report.

Native SM80 dense-query split-KV forward is also repeatable and batch invariant
for the tested D128 and D512 FP16/BF16 configurations with fixed split counts.
The checks include padded lengths, sinks, causal windows, reordered/duplicated
batches and changed neighbors under CUDA Graph replay. O and FP32 Stats are
compared byte for byte. This is forward-only qualification of this fixed
GPU/runtime; THD split-KV remains unsupported and cross-runtime identity is
not claimed.

The measured dense split coverage also includes small native tiles:
D64 at M16/N16, D128 at M16/N32 and M32/N16, and D256 at M16/N16,
in FP16/BF16 with fixed split3. These configurations preserve O/Stats bytes
under replay, batch reordering/duplication and changed neighbors. This
extends the tested forward configuration set; it does not require equality
between different tile sizes or split counts.

Dense-query native PackGQA forward is repeatable and batch invariant in the
tested FP16/BF16 D64, D128, D256 and D512 configurations, including head groups
that cross tile boundaries, unsplit and fixed-split plans. O and FP32 Stats
retain bytewise identity under replay, reordered/duplicated batches and changed
neighbors on the qualified A100 runtime. THD PackGQA remains unsupported;
this forward result does not establish a backward or cross-runtime guarantee.

## SM120 D128 additive-bias forward

FP16/BF16 dense D128 forward with FP32 or matching-input bias is repeatable
and batch invariant for the tested fixed M64 dense and M128 causal plans.
O and FP32 Stats preserve their bytes across fresh allocations, concurrent
work, isolated/reordered/duplicated batches, changed neighbors and CUDA Graph
replay on SM120 (cc 12.0). The M128 path also passes sustained K/V stage-refill
stress with its result-dependent stores before stage reuse.

This covers unpaged, unpacked, unsplit forward with no physical origins;
bias backward is unsupported. SM121 execution and cross-GPU, cross-plan or
cross-runtime bitwise identity remain unverified. The coverage report records
both tested DSL compiler flavors and the original stress failure.

## SM120 paged SDPA

Native FP16/BF16 dense-query forward over paged K/V is repeatable and batch
invariant for the tested D64 and D120/V88 causal-window configurations, with
and without packed GQA. RTX5080 checks compare O and Stats bytes across
replays, reordered/duplicated batches and changed neighbors under capture.
This is a fixed-GPU/runtime result; SM121 execution and cross-GPU bitwise
identity are unverified. The determinism report records the exact coverage.

The D128 M128/N128 paged prefill path also preserves O and Stats bytes on
RTX PRO 6000 for FP16/BF16 with fixed packed or unpacked plans. The rolled
page-copy loop passes replay, fresh-buffer, reordered/duplicated/changed-neighbor
batch and captured-replay checks, plus sustained stage-refill stress. This
extends the tested forward configurations; SM121 and cross-runtime identity
remain unverified, and no paged backward path is qualified by these checks.

The warp-cached page-table loader also preserves strict parent O/Stats goldens
for D128 M128/N128 with eight-token pages, including half and FP8 K/V,
sixteen-page full tiles, a one-row final tile and empty sequences. These
RTX PRO 6000 checks retain fixed packing and split choices across batch
changes and CUDA Graph replay. SM121 remains unverified.

Using all four service warps for the paged prefill loader preserves those
fixed-plan O/Stats bytes for half and FP8 K/V on RTX PRO 6000. The producer
warps retain their copy completion waits, and publication/reuse barriers
count every participating thread. The coverage report records refill stress
and sanitizer results; backward and SM121 remain unverified by these checks.

The SM120 paged16/32-row decode tiles additionally have eight fixed-runtime
replay/batch goldens for FP16/BF16 D64/V64 and D120/V88 with split1/3.
The split count is held fixed across batch changes; the common fixed-order
combine folds the sink once. This does not claim equality across different
split counts or between the prefill and decode arithmetic orders.

The native SM120 paged decode contract also covers independent eight-aligned
QK/V widths through256 on M16/M32 tiles. The D256 and D192/V128 cases retain
CPU-seeded inputs, eager/captured byte comparisons, batch invariance and
runtime-keyed golden outputs in `test_sm120_frost_paged_wide_decode_determinism`.

The D512 native paged envelope additionally has fixed-runtime FP16/BF16
replay and batch-invariance checks on RTX5080, with and without PackGQA,
and with split counts one and three held fixed. The M64/N32 path preserves
O and Stats bytes through fresh buffers, batch reordering/duplication and
changed-neighbor CUDA Graph replay. The independent numerical checks include
empty Q/KV, causal windows and attention sinks. SM121 and other runtimes
remain unverified; this is forward-only qualification.

SM120 wide-half packing also preserves O and Stats bytes for the tested
D512/V512 and D576/V512 groups of 96 and 128 query heads per KV head on
RTX PRO 6000. Dense and paged FP16/BF16 queries cover split counts 1 and 3;
paged E4M3/E5M2 KV storage covers split count 3. These explicit-plan checks
include repeat, reordered/duplicated/changed-neighbor batches and CUDA Graph
replay. SM121 and other runtimes remain unverified; this is forward-only
qualification and does not establish automatic candidate selection.

The native paged path with half Q/O and FP8 E4M3/E5M2 K/V also preserves
O and Stats bytes in the tested D64/D128/D256/D512 configurations on RTX5080.
Quantized cache payloads and split/packing choices stay fixed across batch
reordering, duplication and changed-neighbor capture replay. Eight new
runtime goldens accompany the independent numerical checks. This is
forward-only evidence; SM121 and other runtimes remain unverified.

SM90 D512 dense split forward preserves per-sequence O/Stats bytes across replay, fresh storage, duplicated/reordered batches and changed neighbors under a fixed split count and PackGQA choice. The qualification covers FP16 and BF16 with masked, padded inputs and sinks, including partial PackGQA groups G12 and G128. Different split counts or packing configurations may round differently. THD split and backward are outside this addition.

SM90 native D128/D256 forward on the D512 body (its head_tile 128/256 path at the
64 x 64 tile) also preserves O/Stats bytes under those batch changes for FP16/BF16,
with unsplit unpacked and split3 PackGQA plans. The tested declarations include
padding, a bottom-right causal window and sinks. This is fixed-plan, fixed-runtime
forward qualification; THD batch invariance and backward remain outside this
addition. An exact dense (128, 128) declaration with no tile pinned runs the
dedicated D128 template instead (next paragraph); these results describe the
64 x 64 tile only.

The dedicated SM90 D128 implementation (its fixed owner uses a 128 x 128 tile) has its own
forward suite: replay with fresh storage under concurrent work; a sequence alone
against first, reordered, duplicated and changed-neighbor batches whose plans run
different records of the template (the persistent loop, the paired causal walk,
the NATURAL stride and the claim counter); an FP64 reference; goldens; and warm
stress on the K/V stages (the claim counter's ring has its own in the main suite). Its H100 run (2026-11-13) passed replay and batch invariance
on every arrangement, so its status is supported for the tested configurations. It serves only uniform-length dense declarations
(no THD, paged KV, split-KV, PackGQA, sinks, windows or per-batch lengths) and
has no backward.


The fixed SM90 D128 owner and three-WG D512 owner retain these observed byte
properties in FP16 and BF16 on H100. The owned-body checks assert the loaded
implementation and its local parameter type before comparing replay, fresh
storage, reordered/duplicated batches and changed neighbors. D128 covers the
persistent, causal-walk, stride and claim configurations; three-WG covers
split3 unpacked and split17 packed plans with masking, sinks and empty rows.
Explicit graph-record checks also retain the standalone parameter values and
O/Stats bytes, including changed-input capture replay. The current runtime's
strict golden baselines are missing; none were generated for this migration.
These results do not extend THD batch invariance or backward support. Detailed
counts, runtime and stress limits are in the
[coverage report](../../test/python/determinism_goldens/README.md).

## SM100/SM107 forward register redistribution

The native half, FP8 and MXFP8 forward templates release the service
warpgroup's register budget at one common instruction before the MMA, load,
store and scheduler roles diverge. This preserves their arithmetic and
register budgets. On SM100, the existing D64/D128/D512 paged FP8 cases retain
their documented repeatability and batch invariance against unchanged-parent
goldens; dense half, FP8 and MXFP8 D128/D192/D256/D512 also retain numerical
and changed-buffer replay coverage. Dense replay alone does not establish
batch invariance. Native SM107 also retains strict parent goldens and
repeatability/batch invariance for paged D512 FP8 and D576/V512 half,
alongside dense half/FP8/MXFP8 numerical and changed-buffer replay coverage.
Qualification of this update on SM103 remains unverified; compiling an
architecture's instructions is not an execution result.
Backward kernels and their support statuses are unchanged. See the
[coverage report](../../test/python/determinism_goldens/README.md).

## SM120 FP8 and MXFP8 forward tails

Native per-tensor FP8 and block-scaled MXFP8 forward preserve per-sequence
O/Stats bytes across repeated execution, retained CUDA Graph replay,
reordered/duplicated batches and changed neighbors in the tested SM120
D128/D512 configurations. The checks use E4M3/E5M2 inputs, BF16 output,
runtime KV lengths with poisoned padding, and fixed split1/unpacked plans.
Quantized payloads and logical scale bytes remain fixed across batch changes;
columnwise MXFP8 scales are repacked into each batch's required swizzle.
The global output Amax is repeatable for identical inputs and checked against
the whole-batch reference when batch membership changes.

Eight unchanged-parent runtime goldens accompany the independent numerical
checks. SM120 RTX PRO 6000 execution passed functional, memcheck, racecheck
and synccheck coverage, including separate multi-tile THD cases. This evidence
does not establish SM121 or cross-runtime bit identity, independent physical
THD origins for quantized inputs, or backward behavior.

## Shared THD backward fold

The shared physical-origin dK/dV fold preserves fixed-order FP32 group sums
when a large batch or KV-head count requires a flat launch grid. Direct
helper checks cover FP16/BF16 outputs from FP32 or matching half partials on
SM80, SM89, SM90 and SM100, including empty sequences, unequal dK/dV widths,
independent output origins, fresh bindings and changed-input capture replay.
Bounded physical grids and Int64 live-work iteration also cover logical work
above the CUDA grid.x limit and 2^32, including the final live sequence/head.
These checks exercise runtime maximum sequence lengths and retain fixed-order
FP32 accumulation.
SM100 also has ordinary-shape D128 single/two-CTA and D512 THD caller checks.
This evidence does not upgrade the full SDPA API rows, establish an SM89
attention API, or qualify complete attention chains at the new boundary.
Native SM103, SM107 and SM120 execution remains unverified for these
new helper cases. See the [qualification record](../../test/python/determinism_goldens/README.md#shared-thd-backward-fold-grid-boundaries-2026-11-16).
## SM89 half backward

On SM89 (Ada), `sdpa_bwd_sm89` runs the SM80 FP16/BF16 backward chain and
accepts `use_deterministic_algorithm=True` with the same scope as the SM80 row:
eight-aligned `D_V <= D_QK <= 256`, dense and THD layouts, MHA/GQA,
causal/bottom-right/window masks, padding, sinks and bias/dBias (a bias
broadcast over B > 1 is declined, since its dBias reduces through unordered
atomics). On an L40S,
repeat execution preserves dQ/dK/dV raw bytes for dense and THD-with-origins
graphs, and CUDA Graph replay after changing the input payload in place matches
the reference. Batch reordering/duplication and cross-GPU identity are not
covered by these checks.

## SM100 / SM103 D128 MXFP8 backward

The opt-in `sdpa_bwd_sm100_d128_mxfp8` row runs three CuTeDSL kernels: prep,
a key-stationary dK/dV kernel and a persistent dQ kernel. Every output tile has
one writer, and there are no gradient atomics. When two CTAs split a key
block's query walk, their FP32 partials are added in a fixed order through
cluster shared memory. The optional `amax_dQ`, `amax_dK` and `amax_dV` are
order-free maxima of the FP32 values before the BF16 cast. Every plan is
bitwise repeatable. The walk split is the plan's only rounding-relevant choice.
Under `use_deterministic_algorithm` it is decided per batch element, so results
are batch invariant. Without the request it is decided over the whole batch,
which is faster at large B but is Replay only.

`test_sdpa_bwd_mxfp8_sm100_d128.py` compares raw bytes across:

- eager calls and CUDA-graph replays into poisoned outputs, for aligned,
  tail and causal shapes, including the amax outputs;
- fresh allocations under concurrent GPU work;
- a B = 2 problem against two B = 1 problems;
- a B = 8 problem (4/2 heads, S = 1024) against eight B = 1 problems, where
  the default plan would differ.

Payloads and scale bytes are fixed across batch comparisons. amax is a
whole-batch maximum, so it is compared bitwise across replays, not across batch
changes. Measured on B200 (cc 10.0) and B300 (cc 10.3). The four golden cases write the same
bytes on both; other cross-runtime bit identity is unverified. Other GPUs are not served.

## SM120 packed backward

The native FP16/BF16 THD backward path through D256/GQA retains unordered
FP32 dQ atomic accumulation. General deterministic backward and batch
invariance are unsupported by this path; deterministic-mode graphs decline.
The bounded single-KV-tile test compares dQ/dK/dV bytes across replays,
fresh storage, duplicated/reordered batches and changed-neighbor capture.
That limited coverage passes on SM121 (GB10); SM120 remains unverified.
It does not extend the deterministic contract to multiple KV tiles. This change adds
no forward kernel.

## SM120 per-tensor FP8 backward

The native E4M3 backward engine assigns each gradient row and column partition
to one fixed query/key owner. Key owners visit GQA heads in ascending order;
there are no gradient atomic reductions. The bounded SM121 (GB10) suite
preserves dQ/dK/dV and all four amax outputs across repeated execution, fresh
storage, duplicated/reordered batches of 1, 3, 4 and 129 members, and changed
input capture replay. Six runtime-keyed goldens cover D192 and D512 with
E4M3, FP16 and BF16 gradients. Scalar scales and input payloads remain fixed
for batch comparisons. This qualifies the tested runtime and dense BSHD
contract; SM120 hardware, MXFP8, THD and cross-runtime bit identity remain
unverified.


## SM120 MXFP8 backward

The native block-scaled backward engine uses fixed query/key owners and
fixed-order GQA accumulation. Canonical rowwise and columnwise scale atoms
are read from current device storage, without a global scale repack.
The bounded GB10 SM121 checks preserve dQ/dK/dV bytes across replays,
fresh storage, duplicated/reordered batches of 1, 3, 4 and 129 members,
and changed-payload capture. Captured execution also observes changed
scale bytes. Four runtime goldens cover D128/D512 with FP16/BF16 gradients.
The graph may independently request FP32 `amax_dQ`, `amax_dK` and `amax_dV`.
Each is the global maximum absolute FP32 gradient accumulator before conversion
to FP16/BF16. Initialization shares the delta launch; the gradient epilogue
reduces only requested outputs. This adds no gradient atomic reduction.
For fixed inputs, optional amax outputs preserve gradient bytes. Across batch
reordering and duplication, global amax equals the maximum of the corresponding
singleton values; adding different sequences can change that global value.
This evidence keeps quantized payloads and scale bytes fixed across batch
comparisons; it does not qualify cross-runtime identity, SM120 execution,
THD or unsupported masks.

## SM120 wide half backward

For eight-aligned `256 < D_QK <= 512` and `0 < D_V <= D_QK`, the native
FP16/BF16 backward path supports `use_deterministic_algorithm=True` on the
FE graph (`deterministic=True` on the standalone adapter). A fixed-order
relay orders dQ accumulation independently for each gradient-column partition;
dK/dV have fixed output owners. Dense and THD layouts, MHA/GQA/MQA,
padding, top-left/bottom-right causal masks and left/right windows are
supported; bias and sink are outside this wide path.

The measured D512 configurations preserve dQ/dK/dV raw bytes across repeat
execution, fresh pointers, batch reordering/duplication and changed-neighbor
CUDA Graph replay on SM121 (GB10). Independent native SM120 (RTX 5080)
validation also covers the wide deterministic path. These are runtime-specific
results; they do not guarantee bitwise identity across GPUs or software
versions, or extend the generally unordered through-D256 THD path above.

## SM120 D576 forward

D576/V512 dense, compact THD and paged forward preserves O/Stats bytes in
the tested FP16/BF16 configurations on GB10 (SM121), including half-Q with
FP8 E4M3/E5M2 paged KV. Split count and packing stay fixed across batch
reordering, duplication and changed-neighbor capture replay. Compact THD
repacking follows changed lengths; this earlier qualification covered compact
origins only. Independent physical gaps on SM120 are covered below; their
execution on SM121 and other runtimes remains unverified.
This qualification adds no backward support.

SM100 half single-sequence THD setup retains O/Stats replay and batch
invariance in the tested unsplit dense D128 and paged D64/D128 configurations.
One capacity-4096 plan alternates between runtime B1 and multiple sequences;
the fast setup applies only to runtime B1. Independent physical origins,
changed bindings, empty/keyless replay and fixed-plan packing are covered.
SM103 compiles but has no execution evidence from this change. The optimization
adds no backward or cross-runtime byte guarantee.

SM100 split THD setup also retains O/Stats bytes when one plan alternates
between runtime B1 and B3/B5/B129, then returns to B1. The tested split5
plans use packed paged D64/D128 or unpacked nonpaged D128, matching
FP16/BF16 storage and independent Q/O/Stats origins. Each plan retains its
split count and packing. Mixed FP8 KV has separate numerical and replay
coverage, without a new batch-invariance claim. SM103 execution remains
unverified. This setup optimization adds no backward or cross-plan guarantee.

Native SM107 half D64/D128 single-sequence THD origin setup retains forward
O/Stats byte and batch invariance on the qualified runtime. Independent
physical origins, changed strides/addresses, empty and keyless replay and the
runtime multi-sequence fallback, including one capacity-4096 plan alternating
between single and multiple sequences through shape overrides, are covered by
`test_sdpa_single_sequence_setup_sm107.py`. The setup optimization is limited
to nonpaged, unsplit uniform FP16/BF16; other supported paths retain their
existing setup. Measurement details and runtime goldens are in the coverage
report. This adds no backward or cross-runtime byte guarantee.


Native SM107 D512 split forward also retains exact O/Stats bytes with the
coalesced partial-store path for packing groups32/64. FP16/BF16 tests hold the
split count fixed while comparing replay, standalone requests, reordered and
duplicated batches, changed neighbors, and changed-input CUDA Graph replay.
Sixteen unchanged-parent runtime goldens cover the new cases and existing
D512/D576 half/mixed controls; other runtimes remain unverified by this check.

## SM120 half THD physical origins

Native FP16/BF16 THD forward preserves O/Stats bytes across repeat execution,
batch reordering, duplicated sequences, a sequence alone, changed neighbors,
fresh pointers and retained CUDA Graph replay in the tested RTX PRO 6000
configurations. Q/K/V/O/Stats each retain their own physical offset, pitch
and capacity; token-major and head-major Stats are both covered. Tested
head shapes are D64, D128, D192/V128, D256, D512 and D576/V512.

These are fixed-runtime forward results with the selected plan held fixed.
SM121 physical-gap execution, FP8/MXFP8 origins and backward are not qualified
by this suite. See the coverage report for numerical, golden and sanitizer
results; passing byte comparisons alone do not establish race freedom.

## SM120 per-tensor FP8 THD physical origins

Native E4M3/E5M2 THD forward preserves per-sequence O/Stats bytes across
repeat execution, fresh storage, batch reordering/duplication, a sequence
alone, changed neighbors and retained CUDA Graph replay in the tested
RTX 5090 configurations. Q/K/V/O/Stats have independent physical origins
and pitches; logical lengths continue to determine attention work. Coverage
includes BF16 output, token/head-major Stats, D64, D128, D192/V128, D256,
D512 and D384/V512 with fixed split1/unpacked plans. Quantized payloads and
scalar scales stay fixed across batch comparisons; replay also checks changed
lengths, origins and scalar scales against an independent reference.

This is fixed-runtime forward evidence. SM121 physical-gap execution,
MXFP8 physical origins and backward are not qualified by this suite. The
coverage report records the numerical, strict-golden and sanitizer results.

## SM120 MXFP8 THD physical origins

Native E4M3/E5M2 MXFP8 THD forward preserves per-sequence O/Stats bytes
across repeat execution, batch reordering/duplication, a sequence alone,
changed neighbors, fresh storage and retained CUDA Graph replay in the
tested RTX 5090 configurations. Q/K/V/O/Stats retain independent physical
origins, pitches and capacities; scale tiles remain packed by logical
sequence lengths. Coverage includes BF16 output, token/head-major Stats,
D64, D128, D192/V128, D256, D512 and D384/V512 with fixed split1/unpacked
plans. Batch comparisons preserve quantized payloads and logical scale
bytes. Replay checks changed lengths and origins; doubling V scale factors
also doubles live O exactly while leaving Stats and padding unchanged.

This is fixed-runtime forward evidence. SM121 physical-gap execution,
other runtime combinations and backward are unverified by this suite.
See the coverage report for numerical, golden and sanitizer results.

## SM100 per-tensor FP8 THD physical origins

Native D128 E4M3/E5M2 forward preserves O/Stats bytes across repeated
execution, batch reordering/duplication, an isolated sequence, changed
neighbors and retained CUDA Graph replay on the tested SM100 runtime.
The unsplit, unpacked CGA2 cases bind independent Q/K/V/O/Stats origins,
including mixed offset widths and multipliers, physical padding, changed
strides, and empty/keyless transitions. Token/head-major Stats and BF16
output are covered. The prepared and standalone graph routes agree.

SM103 lowering compiles, but its GPU execution remains unverified. This
suite does not qualify wider heads, MXFP8, SM107 or backward. See the
coverage report for exact configurations and validation receipts.

## SM100 D512 THD physical origins

D512 forward preserves O/Stats bytes across repeat execution, batch
reordering/duplication, an isolated sequence, changed neighbors and retained
CUDA Graph replay on the tested SM100 runtime. Coverage pins both the
role-split and 2x2 half kernels with FP16/BF16, and the per-tensor FP8
kernel with E4M3/E5M2 and BF16 output. Each tested plan is unsplit and
unpacked, with token/head-major Stats. The prepared and standalone graph
routes agree; origins and strides can change between invocations while
logical lengths independently determine live work. Both half implementations
also cover the already-admitted D384/V448 and D264/V504 head envelopes.

The role-split half and FP8 kernels join all output-reader warps in both CTAs
before either MMA warp releases TMEM. Symmetric and single-CTA delayed-reader
lifetime checks complement numerical and
byte comparisons: readable bytes after deallocation do not establish safe
ownership. This check is measured on SM100 and does not extend the GPU
qualification below.

All six dtype/implementation lowerings compile for SM103. The explicit-engine
migration additionally checks FP16/BF16 role-split and 2x2 plans on SM103 with
token/head-major Stats, raw-byte replay, batch reordering/duplication, an
isolated sequence and changed neighbors. The full SM103 suite, including FP8
execution and runtime goldens, remains unverified. These results do not
qualify MXFP8, SM107 or backward.

Exact SM100/SM103 half D512 graph plans now name the role-split or 2x2 main
implementation explicitly. Their public cluster width is four; D576/V512 uses
the 2x2 implementation with width two. Replaying an engine/knob record fixes
that implementation. This identity contract does not make results invariant
across different engines, packing choices or split counts, and does not add a
general determinism or batch-invariance note to these engines.
See the coverage report for the exact cases and validation receipts.

## SM100 MXFP8 THD physical origins

Native D128 (CGA2) and D512 (CGA1) E4M3/E5M2 forward preserve O/Stats
bytes across repeat execution, batch reorder/duplication, an isolated sequence,
changed neighbors and retained CUDA Graph replay on the tested SM100 runtime.
The unsplit, unpacked cases use BF16 output and token/head-major Stats.
Q/K/V/O/Stats origins are independent; quantized payloads and logical scale
bytes are retained across batch changes. Physical data gaps do not move the
packed scale-tile prefixes. Prepared and standalone graph execution agree.

All four admitted SM103 dtype/width lowerings compile offline; GPU execution
there remains unverified. D192/D256 evidence is documented below; SM107 and
backward remain unverified by these suites. See the coverage report for exact
cases and validation receipts.


## SM100 D192/D256 quantized THD physical origins

Native D192/V128 (CGA1 and CGA2) and D256 (CGA1) per-tensor FP8 and
MXFP8 forward preserve O/Stats bytes across repeat execution, concurrent
GPU work, reordered/duplicated batches, an isolated sequence, changed neighbors
and retained CUDA Graph replay on the tested SM100 runtime. The unsplit,
unpacked cases cover E4M3/E5M2 inputs, BF16 output and token/head-major Stats.
Independent physical Q/K/V/O/Stats origins and changed pitches retain the
logical length and MXFP8 scale-tile coordinate systems. Prepared and standalone
graph routes agree. A two-cluster grid exercises persistent sequence changes.

SM103 execution, native SM107 and backward are not qualified by this suite.
See the coverage report for exact configurations and measured validation.

## Native SM107 D128 FP8 THD physical origins

The native D128 per-tensor FP8 origin suite covers E4M3/E5M2 inputs, BF16 O
and independent token/head-major Stats in unsplit, unpacked CGA2 plans.
It checks independent references, fresh bindings, retained capture, concurrent
work and exact-byte batch transformations. Native SM107 execution preserves
O/Stats bytes for the tested configurations, including changed physical
origins and persistent sequence transitions. The qualification uses CUDA 13.4;
backward and other native SM107 quantized widths remain outside this suite.
These O/Stats guarantees also hold after moving the output-scale fold into
the correction role: the existing origin tests retain their parent output
bytes. Dense D64/D128 mask and output-dtype checks provide numerical coverage;
they do not extend the THD suite's batch-invariance claim to dense plans.

## Native SM107 wider FP8 THD physical origins

The D192/V128, D256 and D512 per-tensor FP8 suites exercise independent
Q/K/V/O/Stats origins under unsplit, unpacked, nonpaged plans. They preserve
logical mask lengths while changing physical addresses, and test numerical
references separately from raw-byte replay and batch transformations.
Native SM107 preserves O/Stats bytes for the tested configurations with
CGA2 D192/V128, CGA1 D256 and the D512 role-split body. The qualification
uses CUDA13.4. Backward and MXFP8 are outside these cases; see the test
coverage report for exact validation scope.

## Native SM107 D256 MXFP8 THD physical origins

Native D256 nonpaged, unsplit, unpacked CGA1 preserves O/Stats bytes across
replay, concurrent work and fixed-input batch transformations for the tested
E4M3/E5M2, E8M0-scale and BF16-output cases. Independent data origins do
not move the logical scale-tile prefixes. The CUDA13.4 qualification also
covers fresh strides and retained live/empty captures. Other native MXFP8
THD widths and backward are not qualified by this suite.

## SM100/SM103 D64 packed THD prefill

The direct-output D64 route supports whole-group PackGQA with dense or paged
K/V, N128 and fixed CGA1/2. Its O and FP32 Stats are final outputs; it uses
neither split partials nor a combine launch. The L0 suite exercises independent
origins, current strides, retained captures, changed lengths and tables, fixed
sequence payloads across batch changes, and a KV-ring refill workload. B200 and
B300 preserve O/Stats bytes for the tested configurations under CUDA13.4.
The refill checks cover the native one-CTA-per-SM persistent grid; their
stress bound is limited to that geometry.
Backward uses separate kernels and is outside this forward-path change.


## SM107 half explicit implementation records

The small and medium half forward engines name their main implementation
before compilation. Their public engine/knob records retain that implementation
on replay; this is an identity contract, not a numerical determinism guarantee.
The explicit-record tests cover native output, rebuilding, fresh bindings and
retained CUDA Graph replay on SM107 with FP16 and BF16. They retain two strict
expected failures for the existing shared D256 nonpaged split independent-origin
gap; compact nonpaged split is covered positively. General batch invariance of
the newly registered owners remains **Unverified** by this qualification.
Existing SDPA status rows retain their documented scopes. In particular, neither
registration nor host-only configuration equality establishes a general
Stats-stride-override or batch-invariance claim.

See the [SM107 support reference](../../python/cudnn/sdpa/frost/support/sm107.md)
for the implementation boundaries and the
[coverage report](../../test/python/determinism_goldens/README.md) for measured scope.

### Experimental JAX SDPA route

`cudnn.ops.sdpa_backward_jax(..., deterministic=True)` selects deterministic
backend backward plans. The BF16 `(B=2, S=128, H=4, D=64)` case checks output,
FP32 stats and dQ/dK/dV replay and isolated-batch equality for the explicit
forward/backward calls on SM90 and SM100. This case has no bias. General batch
invariance, dBias determinism, FP16 byte determinism, concurrency, golden hashes,
and other GPUs (including SM103) remain unverified.
This does not establish determinism for frontend kernel plans.
