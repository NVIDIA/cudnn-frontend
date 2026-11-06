# Attention determinism

For API support by GPU and datatype, see the
[determinism status page](../../../docs/operations/determinism.md).
This report holds test coverage, measured results, and failure diagnostics.

These suites check six attention families and the DSA indexer APIs. Run from `test/python`:

```bash
pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
pytest core/test_determinism_utils.py
```

| Suite | API | Forward results | Backward results |
| --- | --- | --- | --- |
| `TestKDA` | `kimi_delta_attention` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDP` | `gated_delta_product`, 1/2/3 Householder steps | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN` | `gated_delta_net` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN2` | `gated_delta_net_v2` | output, final state | dQ, dK, dV, dG, dBeta, dW, dInitialState |
| `TestQSA` | `qsa_forward`, `qsa_backward` | BF16 output, FP32 output, LSE | dQ, dK, dV |
| `TestDSA` | `sparse_attention_forward_wrapper`, `sparse_attention_backward_wrapper` | output, max logits, LSE, optional indexer LSE | dQ, dKV, dSink |
| `TestDSAIndexerForward` | `indexer_forward_wrapper` | dense scores | — |
| `TestDSAIndexerBackward` | `indexer_backward_wrapper`, `dense_indexer_backward_wrapper` | — | dIndexerQ, dWeights, dIndexerK, mutated score buffers |
| `TestDSAIndexerFP8` | `indexer_forward_wrapper` | FP32 scores; SM90 FP32 LSE | Unsupported for FP8/MXFP8 inputs |
| `moe/frost/test_shared_expert_decode_determinism.py` | `SharedExpertDecode` (the shared-expert decode kernel pair, `Geometry(HIDDEN, INTERMEDIATE)`) | BF16 output `y`, the inter-launch BF16 `act` and FP32 `gate_part` scratch rows | — (inference-only step kernels; no backward) |
| `test_gdn_decode_fused.py` / `test_gdn_verify_fused.py` (the `rank_geometry` cells) | `GdnDecodeFused`, `GdnVerifyFused` (the fused GDN decode / MTP-verify step kernels, per-rank `Geometry`) | FP8 output, BF16 output, SSM state slots, conv-state pool, reader tickets | — (inference-only step kernels; no backward) |

The four linear-attention suites live in `linear_attention/frost/test_*_determinism.py`.
They pin FROST and set `batch_invariant=True`. They cover FP16 and BF16 with
nonzero initial state and fixed output/final-state gradients. QSA lives in
`QSA/cutedsl/test_qsa_determinism.py`; its public API has no determinism switch.
It covers BF16 with 12:1, 24:2, and 6:1 query/KV head counts.
DSA uses FP16/BF16 and enables `deterministic=True` on backward.

Replay checks use fresh inputs and side-stream contention. Batch
checks reuse exact sequence data and upstream gradients while removing,
reordering, and inserting neighbors. They include nonaligned lengths, partial
tiles, and empty linear-attention sequences. QSA retains each sequence's mask
while shifting its packed token and block offsets. Stress cases cross the
8192-token linear-attention span boundary and QSA's dense/sparse forward
boundary, or create more sequence/head work items than GPU SMs.

Comparisons check tensor names, shapes, dtypes, and raw bytes. Signed zero
differences fail. Non-finite results fail unless the operation declares an
infinity sentinel: QSA uses negative-infinity LSE; DSA uses positive-infinity
LSE and negative-infinity max logits for empty rows. DSA also checks the exact
empty-row locations. Snapshots own their storage. Helper controls reject
changed signed zeros, NaN payloads, missing gradients, wrong infinity signs,
and golden mutations.

## How to read the test cases

A **suite** is one operation's test class. A **case** is one collected pytest
node after parameter expansion. For example,
`TestKDA::test_replay[bf16-bprop]` is one case: BF16 backward, eight executions.
Those eight executions are not eight reported tests. Backward cases also check
the forward outputs and saved state used to compute gradients.

The expanded set has **285 cases**: 279 operation cases and six harness
controls. All 285 cases are L0,
including stress and boundary cases, so the default marker selects the full set.
List the exact node IDs without executing the kernels:

```bash
pytest core/test_determinism_utils.py linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py deepseek_sparse_attention/cutedsl/test_DSA*determinism.py --collect-only -q
```

| Check | What stays fixed | What changes | Pass condition |
| --- | --- | --- | --- |
| Replay | Input bytes, state, masks, upstream gradients | Fresh buffers and concurrent GPU work | Every result byte matches the first execution |
| Batch invariance | Each selected sequence and its upstream gradients | Batch size, neighbors, order, packed offsets | Its results match the corresponding baseline slice |
| Neighbor isolation | Target sequence, mask, state, upstream gradients | Other sequences' data, state, gradients, and QSA masks | Target results do not change |
| Golden regression | Input hashes and complete runtime key | Frontend source revision | All result hashes match the committed baseline |
| Numerical reference | Mathematical inputs and gradient sources | GPU kernel versus independent reference | Existing operation tolerances pass |

The golden cases check replay and the numerical reference before reading the
saved baseline. The new head-layout, GDP step-count, isolated-loss, and QSA
mask-replay cases also check an independent reference. A reference pass alone
does not establish bitwise determinism.

### Linear attention: 28 cases per family, plus four GDP cases

The shared implementation is `linear_attention/determinism.py`. Each FROST
wrapper pins one operation. Except where listed, inputs use BF16, two Q/K/V
heads, K=V=64, and a nonzero FP32 initial state.

| Test method | Level | Parameters and purpose | Cases per family |
| --- | --- | --- | ---: |
| `test_replay` | L0 | FP16/BF16 × fprop/bprop; lengths 65, 0, 17, 1; eight executions; input mutation check | 4 |
| `test_batch_invariance` | L0 | FP16/BF16 × fprop/bprop; removal/reorder; empty state and state-gradient passthrough | 4 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes | 2 |
| `test_long_batch_invariance` | L0 | fprop/bprop; lengths 17001, 17; four executions then standalone comparison | 2 |
| `test_many_work_items_replay` | L0 | fprop/bprop; four sequences × enough heads to exceed the SM count; eight executions | 2 |
| `test_tile_boundaries` | L0 | FP16/BF16 × fprop/bprop; lengths 1, 15, 16, 17, 63, 64, 65 | 4 |
| `test_head_layouts` | L0 | fprop/bprop × three head/dimension layouts below; checks shared-head gradient reductions | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; fixed 17-token target, changed 65- and 3-token neighbors | 2 |
| `test_gradient_source` | L0 | bprop with output loss only or final-state loss only | 2 |
| GDP `test_householder_steps` | L0 | fprop/bprop × 1/2 steps; the other GDP tests use 3 steps | +4 for GDP |

The three head layouts are `(Q/K heads, V heads, K dimension, V dimension)`:
`(1, 4, 64, 128)`, `(4, 1, 128, 64)`, and `(2, 2, 128, 128)`.
Tile, head-layout, isolated-loss, and GDP step-count cases each run four
executions, then compare every sequence alone, the reversed batch, and a
duplicated first sequence. GDP slices K/V/Beta gradients on the expanded
Householder axis. State tensors use `[sequence, output heads, V, K]`.

### Fused GDN step kernels: per-geometry replay and batch invariance, no goldens

The fused decode / MTP-verify step entries (`cudnn.linear_attention.frost.gdn_decode_fused` / `gdn_verify_fused`) carry their determinism
cells inside their own modules (`linear_attention/frost/test_gdn_decode_fused.py`, `test_gdn_verify_fused.py`; the FlashInfer plan-API twin in
`test_flashinfer_gdn_backend.py`), at the default geometry (16 K / 48 V heads, hidden 5120) and at five per-rank geometries ((1, 8, 8192),
(2, 16, 8192), (4, 32, 8192), (4, 16, 4096) and, since 2026-11-06, (2, 8, 4096) -- 8-way tensor parallelism of a hidden-4096 model; the FlashInfer
plan-API twin covers the first four).  Every comparison is raw bytes: the FP8 and BF16 outputs, every FP32 state slot, the conv-state
pool and the reader tickets.

| Check | Cells | What it holds |
|---|---|---|
| Replay | `*_matches_the_torch_model_and_replays_bitwise` / `*_matches_the_torch_reference_and_replays_bitwise`, `test_deterministic_*`, the graph-replay cells | Two fresh identical operand sets (and a CUDA-graph replay over distinct layers) are byte-identical; the independent fp64 reference is a separate check at the operations' tolerances |
| Batch invariance | `test_rank_geometry_rows_are_batch_invariant` (decode), `test_rank_geometry_requests_are_batch_invariant` (verify) | A row / request alone writes the bytes it writes inside a 4-row / 4-request step (its outputs, its state slot(s), its conv slot).  The verify step holds this under ONE gated-norm association: the one-request pick takes the reference kernel's two-CTAs-per-head association (`ctas2`, 4-row chains), several requests `ctas1` (8-row chains) -- `norm_form_for(requests)`, a by-design dependence of the normed output's last bits on the request count, so the lone request runs under `ctas1` in the cell |
| Cross-form | `*_knob_arms_are_bitwise_*`, `test_pair_and_one_cta_forms_are_bitwise_at_one_request` | Every form family (cluster pair, register-capped, streamed, thread mappings, issue orders, ring depths) writes the bytes of the one-CTA form |
| Reference-kernel bytes | `test_bitwise_with_the_reference_verify_kernel_outputs[*]` (with `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR`), the FlashInfer `frost` bitwise cells | The default geometry replays the saved reference-kernel output sets byte for byte; no committed goldens (the saved sets are the oracle) |

GPU coverage: cc 10.0 (a GB200; the per-rank cells 2026-11-05), cc 10.3 (verified on a GB300, 2026-11-05: the same cells at the default and the four per-rank geometries, the saved reference-kernel sets byte for byte, the three register-capped default-geometry forms rendering the same `sm_103a` cubin as before the geometry change) and cc 12.0 (an RTX 5090, 99 KB per-CTA opt-in, 2026-11-05: the default geometry's cells and the per-rank cells -- the replay, batch-invariance and cross-form cells there compare against the landing-slab one-CTA form, the one this part holds; the arms the part does not hold skip by their typed decline); cc 10.7 unverified.  The fifth geometry's cells, (2, 8, 4096), were verified
on cc 10.3 (a GB300, 152 SMs, 2026-11-06, inside a vLLM container image: torch 2.14.0+cu130, nvidia-cutlass-dsl 4.8.0, cuDNN backend 9.24.0;
`pytest linear_attention/frost/test_gdn_decode_fused.py -k k2v8h4096` 30 passed / 244 deselected, `pytest linear_attention/frost/test_gdn_verify_fused.py
-k k2v8h4096` 37 passed / 306 deselected); on cc 10.0 and cc 12.0 those cells are unverified.

### Shared-expert decode pair: replay, batch invariance and goldens per geometry and form

`moe/frost/test_shared_expert_decode_determinism.py` pins `cudnn.SharedExpertDecode` (the bf16 shared expert next to a routed MoE,
1 .. 24 tokens, two launches) at both validated geometries (hidden 8192 with intermediate 256 and 128).  Every comparison is raw bytes
over the pair's output `y` and the live rows of its inter-launch scratch (`act` bf16, `gate_part` fp32); the independent fp64 reference of
the bf16 operands is a separate check at the harness band (rel-L2 <= 2e-2; `test_shared_expert_decode.py`).

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | geometry x T 1 / 6 / 24; eight executions on fresh buffers under side-stream contention; reference check | 6 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two fresh identical operand sets | 2 |
| `test_batch_invariance` | L0 | geometry x inside T 6 / 24: every token alone (T 1, one m-tile, row 0) vs inside, and inside the reversed batch -- the T 24 cell crosses the two-m-tile form boundary and the fragment row position | 4 |
| `test_duplicated_token_and_neighbour_isolation` | L0 | a token duplicated (both copies = alone), neighbours redrawn around it in the two-tile form, a row-strided x view vs its contiguous copy | 2 |
| `test_twenty_four_to_twenty_five_is_the_typed_decline` | L0 | the T ceiling is a `NotImplementedError`, never a provider switch | 1 |
| `test_golden` | L0 | geometry x form (T 3 / 19); three executions, the reference, the saved hashes (`shared-expert-decode-<geometry>-tiles<n>-ksplit8`: the default form -- eight K slices, x resident by bulk row copies, the gate leaves folded on the cluster leader) | 4 |
| `test_v1_arm_reproduces_the_v1_goldens` | L0 | geometry x form on the SAME operands, the first release's form (`v1_cfg`: K_SPLIT 2 / 4, the x ring, the launch-2 gate fold) against the `ksplit2` / `ksplit4` sets recorded with that release -- the shared code paths moved no first-release addition | 4 |

Golden sets: the `ksplit2` (I 256) / `ksplit4` (I 128) sets were recorded with the first release (2026-11-06, cc 10.3) and are kept as the
v1 arm's; the `ksplit8` sets are the one-wave default's -- a NEW rounding class (reason: the pinned `K_SPLIT` moved from 2 / 4 to 8, the eight
slices are added in cluster-rank order, and the gate logit's leaves are added per row-pair group on the cluster leader before launch 2's
lane fold), recorded on the same part after the replay, batch-invariance and reference cells passed.  The knob arms (`PDL`, `PDL_PAIR`,
`PDL_RELEASE`, `N_GROUPS`, `X_BULK_ROWS`, `X_STAGES`, `W_PHASES`, `ROWS_PER_CTA`, `KRED_SPLIT`) are bitwise with the default in
`test_shared_expert_decode.py` (`test_bit_neutral_knob_arms_write_the_default_bytes`, `test_kred_split_arm_is_bitwise_at_the_one_group_geometry`);
the `K_SPLIT` / `GATE_CLUSTER_FOLD` / v1 rounding-order arms are within the band there and each is pinned bitwise-replay (three executions
under side-stream contention) and batch-invariant (every token of a T 6 step alone vs inside) in
`test_order_arms_are_rounding_order_arms_within_the_band`; goldens exist for the default and the first release's form only; the
graph-capture replay is bitwise with eager there too.

GPU coverage: cc 10.3 only.  Run of record (2026-11-06): a GB300 (cc 10.3, 152 SMs) with torch 2.14.0+cu130, CUDA 13.0, driver 13030,
cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (the goldens' `runtime` block); `cd test/python && pytest
moe/frost/test_shared_expert_decode_determinism.py` -> 27 passed, 0 skipped, and `pytest moe/frost/test_shared_expert_decode.py` ->
191 passed, both from a clean tree at the commit that carries the `ksplit8` goldens.  Another runtime stack (torch / CUDA / cuDNN / DSL
version) hashes to another goldens directory, so the golden cells SKIP there -- a coverage gap, not a pass -- until a set is recorded for
it with a stated reason.  cc 10.0 and cc 10.7 unverified; cc 12.x declined by type.

### QSA: 35 cases

QSA uses BF16 Q/K/V with head dimension 256. The saved FP32 output and LSE
must match even when the BF16 output rounds to the same value.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × Q:KV heads 12:1, 24:2, 6:1; eight executions; input mutation check | 6 |
| `test_batch_invariance` | L0 | Same six combinations; lengths 65, 3, 17; removal/reorder with masks preserved | 6 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes | 2 |
| `test_sparse_batch_invariance` | L0 | fprop/bprop; lengths 2057, 3; four executions then standalone comparison | 2 |
| `test_dispatch_boundary` | L0 | fprop/bprop × total tokens 512/513; a fixed 3-token target before/after a neighbor | 4 |
| `test_mask_replay` | L0 | fprop/bprop × all selected, every seventh row empty, all masked; reference and eight executions | 6 |
| `test_mask_batch_invariance` | L0 | Same six combinations; each sequence alone, reversed batch, duplicate sequence | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring values, masks, and upstream gradients | 2 |
| `test_zero_upstream_gradient_bprop` | L0 | Eight executions with zero dO; all gradients must be zero | 1 |

Head parameters have explicit IDs such as `q24-kv2`. A failure such as
`test_dispatch_boundary[total513-fprop]` identifies the routing boundary and
direction without a separate lookup table. A test stops at its first mismatch;
later checks in that case are not claimed as passes.

### DSA: 48 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_determinism.py`.
Each case checks all returned forward tensors. Backward uses deterministic
execution and checks dQ, the shared K/V latent gradient dKV, and dSink.
The three `(heads, QK dimension, Top-K)` layouts are `(64,512,64)`,
`(64,576,64)`, and `(128,512,128)`. The 576-wide path uses all QK dimensions
but only the first 512 KV dimensions for values.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × FP16/BF16 × three layouts; eight executions, input mutation check, numerical reference | 12 |
| `test_batch_invariance` | L0 | Same twelve combinations; each sequence alone, packed, reversed, and duplicated | 12 |
| `test_selection` | L0 | fprop/bprop × ragged lengths, invalid indices, empty rows, Top-K 65 tail, indexer prefix 512; replay, reference, batch checks | 10 |
| `test_query_boundary` | L0 | fprop/bprop × H64 with 128/129/257 queries and H128 with 74/75 queries; four replays then batch checks | 10 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring Q/KV values and Top-K selection while keeping the target fixed | 2 |
| `test_golden` | L0 | fprop/bprop; query lengths 3, 2; Top-K 16; three executions, reference, saved hashes | 2 |

Packing shifts valid Top-K indices by the sequence's KV offset. Negative and
out-of-bounds indices stay invalid; they must not name a neighboring KV row.
Every active row has unique valid indices, as required by generic deterministic
DSA backward. Duplicating a sequence creates a separate KV region.

dSink is one shared per-head vector, so it cannot be sliced by sequence.
Backward batch checks isolate one target's loss: its dO stays fixed and all
other sequences receive zero dO. The complete dSink vector is then comparable
to the target alone. Replay/reference cases also cover nonzero dO for every
sequence. Empty-row sentinels retain their actual float dtype and raw bits;
NaNs and non-finite outputs/gradients still fail.

### DSA BF16 indexer: 46 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py`.
Q/K are BF16. Forward weights use BF16 or FP32; scores are FP32. Backward
weights, dQ and dWeights are BF16; dK uses BF16 or FP32. Score operands,
normalizers and the loss gradient are FP32. This BF16-input suite is separate from the FP8/MXFP8 cases below.

| Test method | API path | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | Forward | H32/H64 × BF16/FP32 weights × ratio 1/4; eight executions and FP64 reference | 8 |
| `test_batch_invariance` | Forward | Same eight combinations; standalone, packed, reversed, duplicate | 8 |
| `test_boundary` | Forward | Q/K lengths 127/128/129; all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | Forward | BF16/FP32 weights; packed query/key lengths (3,7)/(17,129), reorder/duplicate, mask padding | 2 |
| `test_neighbor_isolation` | Forward | Fixed target with changed neighboring Q/K/weights | 1 |
| `test_golden` | Forward | BF16 weights, three queries; reference, three executions, saved hash | 1 |
| `test_replay` | Backward | Sparse H32/H64 and dense H64 × BF16/FP32 dK; reference then eight executions | 6 |
| `test_batch_invariance` | Backward | Same six combinations; standalone, packed, reversed, duplicate | 6 |
| `test_sparse_invalid_slots` | Sparse backward | Local/global indices with negative padding; reference, replay, batch checks | 2 |
| `test_sparse_persistent_rows` | Sparse backward | Top-K 896/2048, 193 queries over 96 producer planes; reference, four executions, batch checks | 2 |
| `test_zero_loss_gradient` | Both backward APIs | Zero loss gradient; eight executions and exact zero gradients | 2 |
| `test_neighbor_isolation` | Both backward APIs | Fixed target and loss scale with changed neighbors | 2 |
| `test_golden` | Both backward APIs | One query, FP32 dK; reference, three executions, saved hashes | 2 |

All 46 cases are L0. Sparse backward sets `deterministic=True`; dense backward
has no determinism switch. Each call restores mutable score inputs and checks
immutable inputs. Gradient buffers start at NaN; sparse deterministic workspace
starts at `0xA5`. Results include all three gradients and each mutated score
buffer. Numerical references use the existing autograd checks with cosine
similarity at least 0.99 and RMS relative error at most 0.02; bitwise checks
still allow no difference.

Backward sets `loss_coeff = B * S_q` and fixes `grad_loss = 0.75`, so adding or
removing a batch member does not change its gradient scale. Local Top-K IDs
stay local. Global IDs are rebuilt for each batch order; negative padding stays
invalid. The 193-row cases force a persistent producer to process a third row. Golden backward cases have one query; their passes do not
establish determinism for multi-query reductions.

See the [API-by-dtype results](#api-and-datatype-results) and
[SM100 indexer failures](#indexer-failures-sm100) below.

### DSA FP8/MXFP8 indexer: 34 cases

`deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py` adds
17 L0 cases for SM100/SM103/SM107 MXFP8 and the same 17 for SM90 FP8.
Each parameter ID names the architecture group and precision. The fixture
skips other architectures before launching a kernel and checks the DSL target.

| Method | Parameters | Cases per precision |
| --- | --- | ---: |
| `test_replay` | H32/H64 × ratio 1/4; eight executions, numerical reference | 4 |
| `test_batch_invariance` | Same combinations; standalone, packed, reordered, each sequence duplicated | 4 |
| `test_boundary` | Q/K lengths 127/128/129, all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | H32/H64; lengths (3,7)/(17,129); replay, reference, reorder/duplicate with repacked scales | 2 |
| `test_neighbor_isolation` | Fixed target payload/scales with changed neighbors | 1 |
| `test_scale_change` | Fixed FP8 payload; doubled Q scales must change scores and still match the reference/replay | 1 |
| `test_golden` | Three queries; reference, three executions, payload/physical-scale/output hashes | 1 |

Both paths use E4M3 Q/K and BF16 weights. MXFP8 uses nonuniform E8M0 scales
per 32 values, packed with the independent scale-layout reference. SM90 uses
FP32 token/head descales. Quantization happens once per sequence using CPU-seeded
inputs; repacking never requantizes. Input mutation checks compare raw payload
and scale bytes. Both paths compare FP32 scores; SM90 also compares FP32 LSE.
The score reference uses dequantized inputs and FP64 computation. Existing
correctness tolerances apply only to references; replay and batch comparisons
remain byte-exact.

SM100/SM103/SM107 MXFP8 and SM90 FP8 each had **17 passed, 0 failed**,
including their own saved golden. Each device skips the other precision group.
The [API-by-dtype results](#api-and-datatype-results) keep these gaps separate
from measured results. FP8/MXFP8 backward is not
supported by these DSA attention/indexer APIs and has no positive test cases.

## Golden checks

Each golden records SHA-256 hashes of all input and result bytes, with their
shapes and dtypes. This retains full precision without large tensor files.
The runtime key includes the GPU, SM count, Torch, CUDA, driver API version,
cuDNN, CuTe DSL, and TVM FFI.
The frontend revision is deliberately not part of the key: the check must
detect changes across frontend revisions on the same runtime.

Golden cases first pass replay and an independent numerical reference.
Linear attention uses the existing FP64 recurrent references on the CPU.
QSA uses its existing FP64 attention and gradient reference. DSA uses the
existing FP32 gather-based forward and autograd gradient references. Reference
tolerances test mathematical correctness; golden and replay checks use no
tolerance. A first execution is only a replay baseline, never a saved golden.

A normal run reads this directory. An unrecorded runtime skips only the golden
comparison; replay and batch checks still run. For CI, require a complete
runtime baseline with an explicit directory. Missing goldens then fail:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
```

Record candidates into a **new** directory on a known-good revision:

```bash
CUDNN_DETERMINISM_RECORD=/tmp/attention-goldens-candidate \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0 -k golden
```

Recording refuses to overwrite an existing file. Check every golden supported on the target architecture, inspect
the input/output hashes, and explain any numerical change before replacing
committed files. Never enable recording in a PR check. A new runtime needs its
own reviewed baseline; matching results across architectures is not assumed.

## Agent and review rules

Follow [the root agent rules](../../../AGENTS.md#kernel-determinism-and-batch-invariance)
and [the test guide](../../AGENTS.md). The PR template requires an accompanying
determinism test change for kernel changes, or an explicit unsupported-path
explanation. Extend this table and the suites when adding an operation or a
new numerical path. Keep hardware/dependency skips specific; a missing FROST
plan on a supported runtime is a failure. Explicitly requested temporary skips
must name the defect, retain the recorded failures, and be removed with its fix.

The earlier suites cover FROST and public QSA/DSA APIs on SM100/SM103/SM107.
The quantized indexer suite adds SM90 FP8 and SM100/SM103/SM107 MXFP8.
The set excludes cuTile/CAKE, other Hopper paths, QSA indexed/two-pass APIs,
checkpoint/summary/decode paths, shared gate parameters, fused/separate Top-K,
paged/MXFP4 indexer forward, THD dense indexer backward, score recompute,
pooled DSA, and distributed recipes. Existing tests cover some paths separately.
Recipe CI and an automatic source-to-test gate remain integration work;
the checklist is a review safeguard.

## Temporary skips

The 26 cases below use `skip until fix` reasons, as requested. They remain L0.
The set is the union of failures recorded on SM100, SM103, and SM107. A pass
on one run does not clear a nondeterministic case that failed on another GPU.
Passing parameters, golden checks, and all FP8/MXFP8 cases remain active.

| Suite | Temporarily skipped parameters | Cases |
| --- | --- | ---: |
| QSA | Backward replay and basic batch checks, all three head layouts | 6 |
| QSA | Sparse batch checks, both directions | 2 |
| QSA | Dispatch boundary at total513, both directions | 2 |
| QSA | Backward mask replay/batch checks, all-selected and empty-rows | 4 |
| DSA attention | H128/D512/Top-K128 BF16 backward batch check | 1 |
| DSA attention | Backward query boundaries H64 Q129/Q257 and H128 Q74/Q75 | 4 |
| DSA indexer | Dense H64 replay/batch checks, BF16 and FP32 dK | 4 |
| DSA indexer | Sparse persistent rows, Top-K896/2048 | 2 |
| DSA indexer | Dense backward neighbor isolation | 1 |

These skips do not change API support status. The baseline measurements and
failure diagnostics below were recorded before these skips. Remove each skip
with the kernel fix and verify the affected cases on the supported GPUs.

### Skip validation: 2026-11-05 (UTC)

The three changed suites ran on SM100: **103 passed, 26 skipped, 0 failed**
in 131.55 seconds. Runtime key: `5f8a825586430e85`, as in the SM100 baseline
below. Imports resolved to this checkout. Golden recording was disabled and
all seven golden cases in these suites passed.

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| QSA | 21 | 14 |
| DSA sparse attention | 43 | 5 |
| DSA indexer forward | 24 | 0 |
| DSA indexer backward | 15 | 7 |

The skipped case IDs exactly match the union of the recorded SM100/SM103/SM107
failures. All 285 original case IDs still collect with `-m L0`. Pre-commit and
`git diff --check` passed. Other suites and GPUs were not rerun for this change.

Run from `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -m L0 -q -rs
```

## Baseline validation: 2026-11-04

Kernel base: `da88207aa5b9197b0c84514a7321d5bc3d2683b4` (`develop`).
GPU: SM100, 148 SMs. Runtime: Torch `2.15.0.dev20260820+cu134`, CUDA `13.4`,
cuDNN `92500`, CuTe DSL `4.8.0.dev0`, TVM FFI `0.1.12`.
The complete runtime record is in [the goldens](5f8a825586430e85/dsa-fprop.json).
The test banners confirmed imports from this checkout.

Three disjoint default-L0 runs on SM100 covered all 285 cases. Run from `test/python`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -q
python -m pytest linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py -q
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py -q
```

- DSA attention, indexer, and harness: **89 passed, 11 failed, 0 skipped** in 128.77 seconds.
- Existing five operation suites: **137 passed, 14 failed, 0 skipped** in 489.44 seconds.
- New FP8/MXFP8 module: **17 passed, 0 failed, 17 skipped** in 20.17 seconds.
- Combined: **243 passed, 25 failed, 17 skipped**. Node IDs match the
  285 cases selected by default L0 collection (0.29 seconds). The five existing
  operation suites retain their earlier results; their code is unchanged by
  the indexer addition.

| Suite | fprop | bprop |
| --- | --- | --- |
| KDA | 13 passed | 15 passed |
| GDP | 15 passed | 17 passed |
| GDN | 13 passed | 15 passed |
| GDN-2 | 13 passed | 15 passed |
| QSA | 15 passed, 2 failed | 6 passed, 12 failed |
| DSA sparse attention | 24 passed | 19 passed, 5 failed |
| DSA BF16 indexer forward | 24 passed | — |
| DSA sparse indexer backward | — | 13 passed, 2 failed |
| DSA dense indexer backward | — | 3 passed, 4 failed |
| DSA MXFP8 indexer (SM100) | 17 passed | Unsupported |
| DSA FP8 indexer (SM90) | 17 passed on SM90; 17 skipped on SM100 | Unsupported |

All six harness controls and all sixteen goldens runnable on SM100 passed,
including replay and independent references. SM90 also passed its FP8 golden
on Hopper. MXFP8 and Hopper FP8 each add one record; the fifteen earlier
records are unchanged. Golden recording was disabled in validation. Sparse indexer large-Top-K cases pass replay and
reference, then fail dK after reorder. Dense indexer backward fails dK replay,
batch, or neighbor checks. The [API-by-dtype results](#api-and-datatype-results)
and [SM100 indexer failure list](#indexer-failures-sm100) record the measurements.

Pre-commit (Black, Black Jupyter, SPDX), Python syntax, and `git diff --check`
passed. Runtime warnings were Torch context/debug notices and existing CuTe
DSL scheduler/allocator deprecations. `fern check --warnings` found zero
errors; the remote redirect check was skipped after an FDR HTTP 403 response.

### API and datatype results

Each cell is **passed / failed / skipped** for the 2026-11-04 runs.
Counts are expanded pytest cases, each assigned once to its direction.
Backward cases also compare forward results where applicable. A case stops at
its first mismatch; later comparisons do not count as passes.

| API | Direction | Datatype | SM90 Hopper | SM100 | SM103 | SM107 Rubin |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `kimi_delta_attention` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `kimi_delta_attention` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `kimi_delta_attention` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | fprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_product` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | bprop | BF16 | 0 / 0 / 14 | 14 / 0 / 0 | 14 / 0 / 0 | 14 / 0 / 0 |
| `gated_delta_product` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net_v2` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net_v2` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `qsa_forward` | fprop | BF16 | 0 / 0 / 17 | 15 / 2 / 0 | 15 / 2 / 0 | 15 / 2 / 0 |
| `qsa_backward` | bprop | BF16 | 0 / 0 / 18 | 6 / 12 / 0 | 7 / 11 / 0 | 6 / 12 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | 0 / 0 / 18 | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | 0 / 0 / 18 | 13 / 5 / 0 | 13 / 5 / 0 | 14 / 4 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | 0 / 0 / 15 | 15 / 0 / 0 | 15 / 0 / 0 | 15 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | 0 / 0 / 9 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 4 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 11 | 9 / 2 / 0 | 9 / 2 / 0 | 9 / 2 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 2 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 5 | 2 / 3 / 0 | 2 / 3 / 0 | 2 / 3 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | 0 / 0 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 17 / 0 / 0 | 0 / 0 / 17 | 0 / 0 / 17 | 0 / 0 / 17 |

The table covers 279 operation cases per GPU. Six harness controls also passed
on each GPU. Hopper attention and BF16 indexer skips are coverage gaps; they
do not establish that the APIs are unsupported. The other precision group's
FP8/MXFP8 cases skip on each architecture. Runtime details follow below.

Run the full selection from `test/python`, with imports from this checkout:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -q
```

Add `--collect-only` to list the 285 L0 cases. The explicit golden directory
makes missing runtime baselines fail. Keep golden recording disabled.

### SM90 Hopper validation

The same 285-case default-L0 selection had **23 passed, 0 failed, 262 skipped**
in 13.71 seconds. All 17 FP8 indexer cases and six harness controls passed.
The other cases skip at their architecture gates. This does not measure
attention or BF16 indexer behavior on Hopper.

Runtime: SM90, 124 SMs; Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`,
driver API `13040`, cuDNN `92700`, CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`.
The [Hopper golden](ffe6ff35e0022211/dsa-indexer-fp8-fprop.json) was recorded
only after replay and the FP64 reference passed, then checked again with
recording disabled. It includes E4M3 input bytes, FP32 scales, weights, offsets,
scores, and LSE. Test source came from commit
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.

### SM103 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 482.93 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 13 QSA, five DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA B300 SXM6 AC`, SM103, 148 SMs. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13040`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`8671c79c2a1e7073bf1efe0cce1939f093a0d840`; imports resolved to that snapshot.
The [SM103 goldens](7a727a3958771fbc) passed replay and independent references
before a separate run with recording disabled. The same full-suite command
listed under [API and datatype results](#api-and-datatype-results) selected all 285 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Compared with SM100, linear-attention gate input hashes differ; the other input
hashes match. The numerical reference checks use each run's actual inputs.
Software versions differ between SM100 and SM103, so these are separate
measurements, not a hardware-only comparison.

| SM103 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 2/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 3/87040 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (1, 0), sequence 0 dq: 1/399360 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 2/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (1, 0), sequence 0 dq: 1/199680 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 142/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 3 dk: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 2/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 5 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2038/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 989/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 513/131072 bytes differ |

### SM107 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 399.55 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 14 QSA, four DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA VR NVL72`, SM107, 212 SMs, on ARM64. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13050`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.
Test and kernel source match the SM103 run. The [SM107 goldens](72e27159dc117014)
passed replay and independent references before a separate full-suite run
with recording disabled. The same command selected all 285 L0 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Some CPU-generated input hashes differ from the x86 runs; references use the
actual inputs. No cross-device byte equality is inferred from separate passes.

| SM107 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `TestQSA::test_replay[bprop-q6-kv1]` | replay 1 dv: 2/43520 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 2/33280 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 140/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 4 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 9/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 12/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 1 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 1843/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1083/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132358/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202644/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1035/131072 bytes differ |

### Indexer failures (SM100)

**DSA BF16 indexer: 40 passed, 6 failed, 0 skipped.** The first mismatch in each failed case is dK.
Sparse large-Top-K cases pass replay and reference, then fail after batch
reorder. Dense backward also has replay and neighbor differences. Sparse dK
reduction assigns query rows to fixed producers; changing batch position can
change the grouping even when replay is stable. Dense backward uses atomic
dK accumulation. These facts explain why a replay pass alone cannot establish
batch invariance. Exact mismatch counts may vary between runs.

| Indexer case | First mismatch |
| --- | --- |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2166/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-bf16-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1/65536 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 793/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1095/131072 bytes differ |

The backward golden cases use one query. Their passes do not establish
bitwise determinism for the multi-query reductions that fail above.

### DSA failures (SM100)

All 24 fprop cases passed. Bprop had 19 passes and five failures, all in batch
invariance. Every replay and numerical-reference check reached in this suite
passed. The first mismatch in each failed case was dKV; forward results and
dQ matched. dSink comparison was not reached after those failures.

| Test | First mismatch |
| --- | --- |
| `test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |

The H64 failures cross the documented 32/64/128-shard query-count thresholds.
The H128 BF16 failures move the target within the two-CTA path. Same-shape
replay passed: `deterministic=True` did not establish batch invariance for
these cases. No kernel changes, xfails, or relaxed comparisons were added.

### QSA failures (SM100)

The same 14 case IDs failed in the earlier run. First mismatch counts and
tensors can vary with the order of atomic reductions.

| Test | First mismatch |
| --- | --- |
| `test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `test_replay[bprop-q6-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 3/33280 bytes differ |
| `test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `test_sparse_batch_invariance[bprop]` | replay 1 dk: 136/1054720 bytes differ |
| `test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_mask_replay[all-selected-bprop]` | replay 1 dv: 3/43520 bytes differ |
| `test_mask_replay[empty-rows-bprop]` | replay 1 dk: 1/43520 bytes differ |
| `test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 3/33280 bytes differ |
| `test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |

QSA backward [documents floating-point atomic reductions](../../../docs/operations/sparse_attention/qsa.md).
The dK/dV replay changes are consistent with changing reduction order. Sparse
bprop fails replay before the batch comparison. Both 513-token cases first
fail on saved FP32 output, before gradient comparison; BF16 output matches.
Both 512-token controls pass. All-masked rows, changed-neighbor isolation,
and zero-upstream-gradient QSA cases pass.

## Method

- [Megatron module tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/unit_tests/determinism): fixed inputs, restored state, outputs and gradients.
- [Megatron kernel replay PR](https://github.com/NVIDIA/Megatron-LM/pull/7148): raw-byte comparison, side-stream contention, negative controls, and coverage review.
- [Nemotron functional tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/functional_tests/test_cases/nemotron): deterministic execution and saved regression values.
- Internal recipe CI (not linked from this public repository): follow-up integration target; its contents were not accessible during this change.

### Hopper compact H32 sparse indexer backward

Verified on 2026-11-07 with NVIDIA L20X SM90, PyTorch 2.12.1+cu130,
cuDNN 9.20.0, and CuTe DSL 4.8.0 compiling with the CUDA 13.4 flavor.
The source under test was develop `938f1db4` plus this change at pre-rebase
commit `a5252d7e`; the DSA modules resolved from that source overlay. The L0
`test_sm90_deterministic_h32_topk512` cases cover BF16 and FP32 dK,
sparse negative IDs, replay, batch reordering with global IDs, and an
independent gradient reference. Both the golden-recording run and the strict
golden replay passed 2/2 cases. The committed runtime key is
`70c200b13182bbc2`.

The L0 support-envelope cells passed 24/24. Four H32 stress cases passed in
15.13 seconds under their then-current L1 marker, covering Top-K 384/512/2048,
both deterministic dW folds, BF16 and FP32 dK, local and global IDs, and 194
query rows over 96 producer planes. This change promotes those four unchanged
cells to L0. Before that marker promotion, the complete indexer-backward L0
file finished with 39 passed, 20 expected SM100-only skips, and 67 deselected
in 21.60 seconds. Commands run:

```bash
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k sm90_deterministic_h32_topk512 -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k check_support_envelope -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L1 -k 'deterministic_reproducible_and_correct and h32' -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -q -rs
```

The CUDA 12 compiler flavor was unavailable and remains unverified. On
2026-11-09 the AIStudio workflow had expired before the rebased branch and the
promoted L0 selection could be rerun. The 42 intervening develop commits did
not change the SM90 indexer backward kernel, its API, or its tests.
