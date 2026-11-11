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
with no backward. `SharedExpertDecode` (the shared-expert decode pair, also
under `cudnn.moe.frost`), `AttentionGateQuantDecode` (the attention gate +
quantize decode step, also under `cudnn.attention_gate.frost`) and
`Fp8ProjectionDecode` (the FP8 projection for decode rows, also under
`cudnn.gemm`) are the same kind of entry. QSA APIs are in `cudnn.ops`. `DSA` is imported from `cudnn`.
The required settings below apply to this table. The `pygraph.custom_reduction`
rows describe only the named reducer bodies and consumed graph outputs, not
arbitrary user callbacks.

| API | Direction | Datatype | SM90 (Hopper) | SM100/103 | SM107 (Rubin) |
| --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` (KDA) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `kimi_delta_attention`, `plan_name="kda_hopper_cuda"` | fprop, bprop | BF16 Q/K/V; FP32 gates and state | Supported for the qualified configurations below | Unsupported | Unsupported |
| `gated_delta_product` (GDP) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net` (GDN) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `gated_delta_net_v2` (GDN-2) | fprop, bprop | FP16, BF16 | Unverified | Supported | Supported |
| `GdnDecodeFused` (fused GDN decode step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Supported | Supported |
| `GdnVerifyFused` (fused GDN MTP-verify step) | fprop (inference step) | BF16 in; FP8 E4M3 and BF16 out; FP32 state | Unsupported | Replay only | Replay only |
| `SharedExpertDecode` (shared-expert decode pair) | fprop (inference step) | BF16 in / out; FP32 accumulate | Unsupported | Supported | Supported |
| `AttentionGateQuantDecode` (attention gate + quantize decode step) | fprop (inference step) | BF16 in; FP8 E4M3 out | Unsupported | SM103: Supported; SM100: Unverified | Supported |
| `Fp8ProjectionDecode` (FP8 split-K projection for decode rows) | fprop (inference step) | FP8 E4M3 in; BF16 out; FP32 accumulate | Unsupported | SM100: Unverified; SM103: Supported | Supported |
| `simplified_sparse_attention` | fprop | BF16 output; FP32 indexer loss | Unsupported | SM100: Replay only; SM103: Unverified | Unsupported |
| `simplified_sparse_attention` | bprop | BF16 Q/K/V and indexer gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `simplified_sparse_attention_jax` | fprop | BF16 output; FP32 indexer loss | Unsupported | SM100: Replay only; SM103: Unverified | Unsupported |
| `simplified_sparse_attention_jax` | bprop | BF16 Q/K/V and indexer gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `dense_indexer_kl`, `use_kf=False` | fprop / bprop | BF16 inputs/gradients; FP32 loss and saved gradients | Unverified | SM100: Replay only; SM103: Unverified | Unverified |
| `dense_indexer_kl`, `use_kf=True` | fprop / bprop | BF16 inputs/gradients; FP32 loss and saved gradients | Unsupported | SM100: Not guaranteed; SM103: Unverified | Unsupported |
| `qsa_forward` | fprop | BF16 | Unverified | Replay only | Replay only |
| `qsa_backward`, `qsa_indexed_backward` | bprop | BF16 | Unverified | Replay only | Replay only |
| `CSA.csa_compressor_forward_wrapper` | fprop | BF16 output; FP32 APE | Unverified | Replay only | Unverified |
| `CSA.csa_compressor_backward_wrapper` (`deterministic=True`) | bprop | BF16 dKV/dScore; FP32 dAPE | Unverified | Supported | Unverified |
| `CSA.csa_compressor_backward_wrapper` (default atomic dAPE) | bprop | BF16 dKV/dScore; FP32 dAPE | Not guaranteed | Not guaranteed | Not guaranteed |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16, BF16 | Unverified | Supported | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | Unverified | Replay only | Supported |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | Unverified | Replay only | Replay only |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 or FP32 weights | Unverified | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | Unsupported | Supported | Supported |
| `DSA.indexer_forward_wrapper` | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_forward_top_k_wrapper` (`deterministic=True`) | fprop | BF16 Q/K; BF16 or FP32 weights | Supported | Supported (cc 10.0 verified; cc 10.3 unverified) | Unverified |
| `DSA.indexer_forward_top_k_wrapper` (`deterministic=True`) | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` (`deterministic=True`) | fprop | BF16 Q/K; BF16 or FP32 weights | Supported | Unsupported | Unsupported |
| `DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` (`deterministic=True`) | fprop | FP8 E4M3 Q/K; FP32 token/head scales; BF16 weights | Supported | Unsupported | Unsupported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 dK | Replay only (H32); H64 unverified | Supported | Supported |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/weights; FP32 dK | Replay only (H32); H64 unverified | Replay only | Replay only |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/weights; BF16 or FP32 dK | Unverified | Not guaranteed | Not guaranteed |
| `pygraph.custom_reduction` with `reduction_modifiers.argmax/argmin` (Torch finite-value rule; index output) | fprop | FP16, BF16, FP32 values; INT32 indices | Unverified | Supported (SM100, tested configurations); SM103 unverified | Unverified |
| `pygraph.custom_reduction` with a direct FLOAT ADD body (sum output) | fprop | FP32 | Unverified | Supported (SM100, tested configurations); SM103 unverified | Unverified |
| FROST native D512 dense-Q split forward (standalone qualification, fixed split count and packing) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested configurations) | Unverified | Unverified |
| FROST native D128/D256 dense-Q forward (standalone qualification, fixed split count and packing) | fprop | matching FP16/BF16 Q/K/V/O | Supported (tested configurations) | Unverified | Unverified |
| FROST native D576/V512 dense-Q paged forward (standalone qualification) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Unverified | Supported (tested configurations) |
| FROST native D512 dense-Q paged forward (standalone qualification) | fprop | matching FP16/BF16 Q/K/V/O | Unsupported | Unverified | Supported (tested configurations) |
| FROST native D512 dense-Q mixed-KV forward (CGA2, dense/paged KV) | fprop | FP16/BF16 Q/O, E4M3/E5M2 KV | Unsupported | Unsupported | Supported (tested configurations) |
| FROST native D256 dense-Q mixed-KV forward (CGA2, unpacked, dense/paged KV) | fprop | FP16/BF16 Q/O, E4M3/E5M2 KV | Unsupported | Unsupported | Supported (tested configurations) |
| FROST native D64 dense/THD paged forward (standalone qualification) | fprop | FP16/BF16 Q/O; matching half or E4M3/E5M2 KV | Unsupported | Unverified | Supported (tested configurations) |
| FROST native D128 dense-Q paged forward (standalone qualification) | fprop | FP16/BF16 Q/O; matching half or E4M3/E5M2 KV | Unsupported | Unverified | Supported (tested configurations) |
| `pygraph.sdpa_fp8`, FROST D512 paged KV with dense Q | fprop | E4M3/E5M2 Q/K/V, FP32 tensor scales; FP16/BF16 or FP8 O | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |
| `pygraph.sdpa_fp8_backward`, FROST exact D512 dense pair | bprop | E4M3 payloads and FP32 tensor scales; E4M3/FP16/BF16 gradients | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |
| `pygraph.sdpa_mxfp8_backward`, FROST exact D512 dense pair | bprop | E4M3 payloads and canonical E8M0 block scales; FP16/BF16 gradients | Unsupported | Supported (SM100, tested configurations); SM103 unverified | Supported (tested configurations) |

The native D128 paging row fixes the CTA width, tile width and split count per
plan. Its standalone suite checks O/Stats bytes across repeated launches,
reordered/duplicated batches and fresh buffers, including mixed FP8 KV storage.
Graph qualification separately checks numerical outputs and changed-input replay.
The D64 mixed-storage checks also cover fixed-plan dense splits and unsplit
THD, preserving each quantized KV payload across batch changes. D64 THD
split with FP8 KV remains unsupported. These paging changes add no backward path.

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

`GdnVerifyFused` is batch-invariant under one gated-norm association only: the
one-request pick takes the two-CTAs-per-head association (`ctas2`, 4-row
chains), two and more requests one CTA per head (`ctas1`, 8-row chains), so a
request's normed BF16 output alone differs in its last bits from the same
request inside a batch by design (`norm_form_for(requests)`); under one
association its bytes are identical. `GdnDecodeFused` writes the same bytes for
a row alone and inside a batch under the entry's own picks.

## Required settings and precision limits

The SM90 CUDA KDA qualification uses `batch_invariant=True`, K=V=128, equal
head counts, FP32 log gates/beta/state, and fixed upstream gradients. It covers
full and partial 64-token chunks, empty documents inside nonempty packs,
sequence isolation/reordering/duplication, changed neighbors, fresh-buffer
replay, and concurrent work. This row applies only to the pinned CUDA provider;
FP16 and grouped-head backward are outside its contract. BF16 P was replaced
by FP32 P in the backward state-gradient addition, so its output hashes change.
Packing P into MMA tiles and prefetching it asynchronously preserve those
FP32-P hashes. The other BF16 intermediates retain their existing precision. See the
[H20 precision and cost report](https://github.com/NVIDIA/cudnn-frontend/blob/develop/benchmark/linear_attention/sm90_kda_fp32_p_results.md)
for the tested shapes and limitations.

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
  and every other knob, including the measured pick per row count and part, is
  a performance form that writes the same bytes. A row's bytes alone equal its
  bytes among up to 32 rows at any position, across the 8- / 16- / 32-row
  token tiles; the kernel is specialised on the row count, so replay and batch
  invariance are verified at every row count 1 .. 32 of each served class
  (`ProjectionGeometry(4608, 8192)` at split 4 and at split 2,
  `ProjectionGeometry(8192, 2048)` at split 2). Another split is another byte
  class and another entry; the status holds per entry. SM100/103 is verified
  on cc 10.3 (a GB300, 152 SMs) only -- cc 10.0 is admitted by the entry and
  unverified; SM107 on cc 10.7 (a VR200-class part, 212 SMs). The committed
  goldens belong to one cc 10.3 runtime; on any other runtime the status rests
  on the replay and batch-invariance cells and the golden comparison is not
  covered. No backward entry point exists. SM90 and SM120 are declined by type.
- DSA sparse attention backward and sparse indexer backward: set
  `deterministic=True`. Sparse attention backward requires unique active
  indices per query, except in SM100/SM103 plans that store dKV in
  conflict-rank storage, which accept duplicate indices: BF16 H128 two-CTA
  plans and generic plans for which `deterministic_dkv_mode` selects records
  or the chain relay.
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
  `max_seq_len`, with the GPU's SM count and with `ds_precision`. Every
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
- MXFP8 indexer forward uses E8M0 block scales, one scale per 32 values.
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
- Use a compatible CuTe DSL runtime, including `sm_107a` support for SM107.

Other linear-attention backends, FP16 indexer inputs, MXFP4, paged indexer,
separate Top-K generation, THD dense indexer backward, score recompute,
pooled DSA, and distributed execution are outside this status page's scope.

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

## SM120 paged SDPA

Native FP16/BF16 dense-query forward over paged K/V is repeatable and batch
invariant for the tested D64 and D120/V88 causal-window configurations, with
and without packed GQA. RTX5080 checks compare O and Stats bytes across
replays, reordered/duplicated batches and changed neighbors under capture.
This is a fixed-GPU/runtime result; SM121 execution and cross-GPU bitwise
identity are unverified. The determinism report records the exact coverage.

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

The native paged path with half Q/O and FP8 E4M3/E5M2 K/V also preserves
O and Stats bytes in the tested D64/D128/D256/D512 configurations on RTX5080.
Quantized cache payloads and split/packing choices stay fixed across batch
reordering, duplication and changed-neighbor capture replay. Eight new
runtime goldens accompany the independent numerical checks. This is
forward-only evidence; SM121 and other runtimes remain unverified.

SM90 D512 dense split forward preserves per-sequence O/Stats bytes across replay, fresh storage, duplicated/reordered batches and changed neighbors under a fixed split count and PackGQA choice. The qualification covers FP16 and BF16 with masked, padded inputs and sinks, including partial PackGQA groups G12 and G128. Different split counts or packing configurations may round differently. THD split and backward are outside this addition.

SM90 native D128/D256 forward also preserves O/Stats bytes under those batch changes
for FP16/BF16, with unsplit unpacked and split3 PackGQA plans. The tested declarations
include padding, a bottom-right causal window and sinks. This is fixed-plan,
fixed-runtime forward qualification; THD batch invariance and backward remain
outside this addition.

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
