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
| `TestDSADeterministicDKV` | `sparse_attention_backward_wrapper` dKV layouts (`deterministic_dkv_mode`, SM100/SM103) | output, max logits, LSE (H128 D576: reference inputs) | dQ, dKV, dSink |
| `TestDSAIndexerForward` | `indexer_forward_wrapper` | dense scores | — |
| `TestDSAIndexerBackward` | `indexer_backward_wrapper`, `dense_indexer_backward_wrapper` | — | dIndexerQ, dWeights, dIndexerK, mutated score buffers |
| `TestDSAIndexerFP8` | `indexer_forward_wrapper` | FP32 scores; SM90 FP32 LSE | Unsupported for FP8/MXFP8 inputs |
| `moe/frost/test_shared_expert_decode_determinism.py` | `SharedExpertDecode` (the shared-expert decode kernel pair, `Geometry(HIDDEN, INTERMEDIATE)`) | BF16 output `y`, the inter-launch BF16 `act` and FP32 `gate_part` scratch rows | — (inference-only step kernels; no backward) |
| `test_gdn_decode_fused.py` / `test_gdn_verify_fused.py` (the `rank_geometry` cells) | `GdnDecodeFused`, `GdnVerifyFused` (the fused GDN decode / MTP-verify step kernels, per-rank `Geometry`) | FP8 output, BF16 output, SSM state slots, conv-state pool, reader tickets | — (inference-only step kernels; no backward) |
| `test_custom_reduction_determinism.py` | `pygraph.custom_reduction`: finite Torch-rule argmax/argmin and direct ADD body | INT32 index projection for FP16/BF16/FP32 values; FP32 sum output | — (no custom-reduction backward entry point) |

### Custom reduction — SM100 validation, 2026-11-09 (UTC)

The new `test_custom_reduction_determinism.py` L0 suite has **seven cases**:
argmax and argmin index projections for FP16, BF16 and FP32, plus a direct
FP32 ADD body. These measurements are new; the earlier attention-suite
measurements and 330-case collection record below are retained separately
and were not rerun for this addition.

Each case checks its independent Torch CPU reference, device input bytes,
fresh-buffer replay with queued side-stream work, and a target row alone,
packed among 32 neighbors, reordered, duplicated, and with changed neighbor
data. Each also replays a large batch of `16 * SM_count + 1` rows: **2369 rows**
on the measured 148-SM device, with the target at row 1184. The reduction
extent is 257, exercising the nominal 256-element boundary. These are workload
shapes, not measurements of the backend's tile size or physical CTA waves.
There are 10 graph calls per case (70 per complete suite run), not 70 pytest
cases. Actual kernel overlap is not measured.

The suite orders the producing Torch stream into the actual cuDNN handle
stream, then orders outputs back before comparisons. It preserves the handle
stream and uses no device-wide prelaunch barrier to drain the queued noise.
Snapshots own their storage; result comparisons check names, shapes, dtypes,
finite values, and raw bytes without tolerance. Index inputs remain local to
each row. No packed offsets, upstream gradients, mutable state, or custom-
reduction backward entry point exist in these tested graphs.

The index-projection graphs consume only the INT32 index. The helper's value
stream remains internal and is not claimed as a tested output. The finite
integer-valued inputs include repeated extrema: the target argmax is **5**
and argmin is **7** under the Torch tie rule. The direct ADD body uses the
same bounded data divided by eight, making all partial sums exactly
representable in FP32; the target sum is **-41/8 (-5.125)**. This establishes
no blanket guarantee for arbitrary user callbacks, arbitrary floating-point
sum data, NaN/infinity policies, or unused tuple outputs. Each graph strictly
selects a publicly enumerated native cuDNN backend plan; absence fails rather
than falling back to another provider. No OSS CuTe DSL kernel path is qualified
by this run.

Source qualification: public parent `8202053a0eac9f224a9961c3f3150765ab3be98f`,
frozen test SHA256
`5e4d7d169a55829a7a4ba43e24c0c34c5f9aa991bc72c2b8be3ff930932db644`.
The test was overlaid onto an isolated validation artifact freshly built at
`a49f23d07c068ec8442da6241bbd755c51460624`; the compiled sources are equivalent
to the public parent. Import resolution was asserted against that artifact's
`/artifacts/python/cudnn`, and both the compiled-header and loaded-backend
versions were asserted to be **92900** before execution. This is native
artifact validation with the frozen source test, not a rebuilt module from
the later documentation/golden addition.

Runtime: GPU name reported as `NVIDIA Graphics Device`, SM100 (cc 10.0),
148 SMs; Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA 13.4, driver API 13030,
cuDNN 92900, nvidia-cutlass-dsl 4.7.1 and apache-tvm-ffi 0.1.13.post3.
CuTe DSL is runtime-key metadata here; the selected numerical path is native
cuDNN. The golden runtime key is `5592daabbefc7ee5`.

| Run | Passed | Failed | Errors | Skipped | Deselected | Pytest time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New baseline recording | 7 | 0 | 0 | 0 | 0 | 11.64 s |
| Strict check, goldens mounted read-only and recording unset | 7 | 0 | 0 | 0 | 0 | 11.89 s |
| Negative control: swap valid argmax/argmin factories | 0 | 6 intended reference failures | 0 | 0 | 1 | 2.38 s |
| Negative control: replace direct ADD with legal MUL, identity 1, and a distinct body key | 0 | 1 intended reference failure | 0 | 0 | 6 | 0.81 s |

Both normal runs collected all seven L0 cases, with two Torch JIT deprecation
warnings each. The negative controls reached native execution and failed at
the independent reference assertions, rather than compilation, support checks,
missing goldens, or harness exceptions. The mutations were confined to isolated
validation copies; the frozen source above was unchanged.

Executed pytest commands inside the validation container, from `/tmp/tests`
with the asserted artifact import active:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_TEST_NO_ISOLATION=1
export CUDNN_DETERMINISM_RECORD=/results/goldens
unset CUDNN_DETERMINISM_GOLDENS
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py

# Separate strict run; /goldens is the reviewed baseline, mounted read-only.
unset CUDNN_DETERMINISM_RECORD
export CUDNN_DETERMINISM_GOLDENS=/goldens
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py
```

The swapped-factory control used the same strict environment and pytest command
with `-p det_mutation -k arg_index`; its plugin interchanged the two public
modifier factories. The legal-MUL control used `-k direct_add` after replacing
only the copied body's ADD with MUL, its identity with 1, and its body key with
`test.python.custom_reduction.determinism.direct_multiply.v1`.

Seven JSON files under `5592daabbefc7ee5/` are new baselines for this previously
uncovered numerical API; no prior golden was replaced and no numerical change
is intended. Recording followed replay and references. Independent integer
arithmetic and IEEE byte encoding reconstructed all input, iota, and expected
output hashes (5, 7, and -41/8), and the seven complete file SHA256 values
remained unchanged after the strict run and when copied into this directory.
A normal checkout check uses:

```bash
unset CUDNN_DETERMINISM_RECORD
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest test_custom_reduction_determinism.py -m L0 -q -rs
```

The two custom-reduction support rows now cover the measured SM100
configurations only. SM103 remains explicitly unverified in the existing
combined SM100/103 column; SM90, SM107, other hardware, other data profiles,
and arbitrary reducers remain unverified.

The four linear-attention suites live in `linear_attention/frost/test_*_determinism.py`.
They pin FROST and set `batch_invariant=True`. They cover FP16 and BF16 with
nonzero initial state and fixed output/final-state gradients. QSA lives in
`QSA/cutedsl/test_qsa_determinism.py`; its public API has no determinism switch.
It covers BF16 with 12:1, 24:2, and 6:1 query/KV head counts.
DSA uses FP16/BF16 and enables `deterministic=True` on backward;
`TestDSADeterministicDKV` also pins each SM100/SM103 dKV layout.

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

The expanded set has **330 cases**: 324 operation cases and six harness
controls. All 330 cases are L0,
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

### DSA: 93 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_determinism.py`:
48 `TestDSA` cases and 45 `TestDSADeterministicDKV` cases.
Each `TestDSA` case checks all returned forward tensors. Backward uses deterministic
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
`TestDSA` rows have unique valid indices, as the shard and relay layouts of
deterministic DSA backward require. Duplicating a sequence creates a separate
KV region.

dSink is one shared per-head vector, so it cannot be sliced by sequence.
Backward batch checks isolate one target's loss: its dO stays fixed and all
other sequences receive zero dO. The complete dSink vector is then comparable
to the target alone. Replay/reference cases also cover nonzero dO for every
sequence. Empty-row sentinels retain their actual float dtype and raw bits;
NaNs and non-finite outputs/gradients still fail.

### DSA deterministic dKV layouts: 45 cases

`TestDSADeterministicDKV` runs on SM100/SM103, where deterministic backward
stores dKV in conflict-rank storage, and skips elsewhere: SM107 plans keep the
shard layouts that `TestDSA` covers. Every case runs the public wrappers with
`deterministic=True` and `deterministic_dkv_mode`, and asserts the backend and
dKV layout each batch selects. Each layout packs two sequences; the packed
batch spans several waves, and the KV extents keep fewer dense planes than a
wave has queries, so shared rows also take overflow records.

| Layout ID | Heads, QK dim, Top-K, dtype | Query lengths | KV lengths | Mode | Backend/layout |
| --- | --- | --- | --- | --- | --- |
| `records-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d512-k64-fp16` | 64, 512, 64, FP16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d576-k64-bf16` | 64, 576, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-strided-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 1183 | 1000, 1000 | `records` | generic M64, records, strided waves |
| `chain-h64-d512-k128-bf16` | 64, 512, 128, BF16 | 17, 300 | 2048, 2048 | `chain` | generic M64, chain relay |
| `shards-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `shards` | generic M64, shards |
| `rank-fused-h128-d512-k128-bf16` | 128, 512, 128, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, fused conflict-rank |
| `rank-setup-h128-d512-k1024-bf16` | 128, 512, 1024, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, per-wave-setup conflict-rank |
| `rank-h128-d576-k128-bf16` | 128, 576, 128, BF16 | 17, 60 | 8192, 16384 | `records` | H128 D576 two-CTA, one-round conflict-rank |
| `rank-sorted-h128-d576-k1024-bf16` | 128, 576, 1024, BF16 | 17, 60 | 2200, 2201 | `records` | H128 D576 two-CTA, one-round conflict-rank, sorted rows |

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_dkv_replay` | L0 | Every layout with full selections; the records BF16, chain, both H128 D512 and the D576 Top-K 128 layouts also with ragged and duplicate selections; shards also with ragged selections. Eight executions with fresh outputs, caller workspaces filled with 0x00, 0xA5 and 0xFF bytes, and side-stream work; input mutation check; numerical reference | 21 |
| `test_dkv_neighbor_isolation` | L0 | Every layout, ragged selections; the other sequence's values, Top-K selection and lengths change while the shapes stay; every result | 10 |
| `test_batch_invariance` | L0 | Every layout except the sorted D576 one, ragged selections; each sequence alone, packed, reversed and duplicated; forward results, dQ and, for the generic M64 layouts, dSink | 9 |
| `test_independent_of_dkv_storage` | L0 | Same inputs under two modes: records and shards (H64; forward results, dQ, dSink), chain and shards (forward results, dSink), conflict-rank and private planes (H128 D512 Top-K 128 and 1024, D576 Top-K 128; forward results and dQ) | 5 |

Ragged selections add per-query lengths with empty rows and negative and
out-of-range indices. Duplicate selections repeat a quarter of every row's slots
and make every query select its sequence's KV row 0 Top-K / 32 times; they
replay the larger sequence alone, so that row is shared by every query of a
wave. Duplicates are inside the contract of every conflict-rank layout; no
shard case uses them. H128 D576 has no public forward kernel: its sequences
carry reference forward results as fixed backward inputs, so these cases
compare no forward results.

Known gaps, stated rather than tolerated:

- dKV is not batch invariant once a batch spans several waves, on any layout.
  A conflict-rank layout ranks a KV row's writers by their wave position and
  sizes its dense planes from the batch's KV extent; the shard layouts take
  their shard count from the batch's query extent and fold shards in groups;
  the default mode picks the layout from the batch's extents. `develop`
  changes dKV in the same multi-wave comparisons (validation record below).
  `test_batch_invariance` therefore does not compare dKV.
- Generic M64 plans reduce dSink with one warp per head (lane `t` sums queries
  `t, t + 32, ...`, then an xor butterfly); a uniform query shift only rotates
  the lanes, so dSink is batch invariant under the measured packings and
  records and chain plans give the shard plans' bits. H128 two-CTA plans sum
  dSink in an order that follows query positions, in both layouts, as on
  `develop`; their batch cases do not compare dSink.
- H128 D576 plans with Top-K 1024 and wider sort each query's rows by ranges of
  the batch's KV extent and alternate the walk direction between waves, so
  their dQ is not batch invariant either and differs in bits from the
  private-plane plans; the sorted layout has replay and neighbor cases only.
- Not exercised: records plans with split pairs (their dQ adds two partials
  once the batch's KV extent selects them), chain plans whose rows are shared
  by every active query of a wave (their tile order follows those rows),
  H128 D576 one-round plans wider than Top-K 128 that stride their waves (the
  wave order follows the batch's indices), the few-plane H128 D512 variant,
  H96 D576 plans and the cluster relay. The L1 bitwise tests in
  `test_DSA_sparse_attention_backward.py` cover several of these paths.
- These layouts have no golden records; their bits depend on the plan, and
  for D576 on the SM count.

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

## Deterministic dKV storage validation: 2026-11-06 (UTC)

Change under test: conflict-rank dKV storage for deterministic SM100/SM103 DSA
backward, the `deterministic_dkv_mode` option, conflict-rank plans that reduce
dSink like the shard plans, and `TestDSADeterministicDKV`. Source revision:
<RESULT revision>. The DSA suite was rerun; the other suites were not rerun
for this change and keep their earlier results.

The default plans of these `TestDSA` cases now select conflict-rank storage,
so their dKV bits change: `test_replay[h128-d512-k128-bf16-bprop]` and the
skipped H128 BF16 batch and query-boundary cases (fused H128 D512 layout), and
the packed batch of the skipped `test_query_boundary[h64-q257-bprop]`
(records). The golden cases (H64, D512, Top-K 16, query lengths 3 and 2) keep
the 32-shard layout; no golden record changes.

Run from `test/python`:

```bash
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 --collect-only -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

The B200 runs below omitted `CUDNN_DETERMINISM_GOLDENS` because their runtime
has no recorded baseline; the two golden cases then skip.

| GPU | Runtime | Result |
| --- | --- | --- |
| B200, SM100, 148 SMs | Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`, driver API `13040`, cuDNN `92500`, CuTe DSL `4.8.0` (CUDA 13.4 compiler), TVM FFI `0.1.13.post3`; key `3c93e700c060c612` | **86 passed, 0 failed, 7 skipped** in 192 s: 45 of 45 `TestDSADeterministicDKV`, 41 `TestDSA`; five temporary skips, two goldens without a baseline |
| B200, SM100, 148 SMs | Same, CuTe DSL `4.8.0` with the CUDA 12.9 compiler | Earlier revision of this change, before the dSink reduction change: **85 passed, 0 failed, 7 skipped** in 186 s |
| SM100, recorded runtime `5f8a825586430e85` | <RESULT runtime> | <RESULT pytest summary, including both goldens> |
| B300, SM103, 148 SMs | <RESULT runtime> | <RESULT pytest summary> |

The new cases take about 140 s of a B200 run with empty compile caches,
mostly first-use compilation of their plans.

With the temporary skips removed locally (not committed), four of the five
`TestDSA` cases fail on B200 on dKV, as on `develop`; 42 passed and the two
goldens skipped. The H128 cases now fail at the first packed batch, because the
conflict-rank plane count follows the batch's KV extent.
`test_query_boundary[h64-q257-bprop]` passes on B200: the target alone selects
shards and the packed batch selects records, whose dQ and dSink match the shard
plans and whose dKV for this target adds the same contributions in the same
order. The case keeps its skip: SM107 plans keep the shard layouts, where it
failed on dKV ([SM107 validation](#sm107-validation)), and the skip set is the
union of the recorded failures.

| Skipped case | First mismatch with conflict-rank storage (B200) |
| --- | --- |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | passes |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (0, 1), sequence 0 dkv: 9/164864 bytes differ |

#### Batch layouts on `develop` and with this change (B200)

The same comparisons ran on `develop` (no `deterministic_dkv_mode`) and with
this change under `'shards'`, `'records'` and `'chain'`: each sequence alone
against its slice of the packed, reversed and duplicated batches (six
comparisons per configuration, other sequences' dO zero), neighbor isolation,
and replay, on FP16 and BF16. Configurations: the `TestDSA` sequences (lengths
17 and 3) for H64 D512, H64 D576 and H128 D512; H64 D512 Top-K 64 with lengths
17 and 300 and KV 1000 + 1000 (three records waves), and lengths 17 and 1183
(strided records waves), full and ragged selections; the same with KV
2048 + 2048 (Top-K 128 and 64) for the chain relay. Counts are comparisons out
of six with any differing byte.

| Configuration | `develop` | `'shards'` | `'records'` | `'chain'` |
| --- | --- | --- | --- | --- |
| `TestDSA` sequences, H64 D512/D576 and FP16 H128 | none differ | none differ | none differ (shard layout selected) | none differ (shard layout selected) |
| `TestDSA` sequences, BF16 H128 D512 (two-CTA) | dKV 1, dSink 2 | dKV 1, dSink 2 | dKV 3, dSink 2 | rejected (two-CTA backend) |
| Three waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 3 / 3 | dKV 5 / 2-3 |
| Strided waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 4-5 / 3 | dKV 4-5 / 3 |
| KV 2048 + 2048, three and strided waves, FP16 and BF16, full / ragged | not run | dKV 1-3 / 1-2 | dKV 3 / 3 | dKV 3-5 / 2-3 |

Forward results, dQ, replay and neighbor isolation matched in every
configuration. Every generic configuration kept dSink under all three modes;
records and chain dSink is bitwise the shard plans' dSink for the same inputs.
`develop` and `'shards'` gave the same per-comparison counts. FP16 DSA backward
on SM100/SM103 is listed as Replay only because `develop`'s shard layouts
already change dKV at these multi-wave shapes; the earlier Supported status
reflected the single-wave `TestDSA` shapes only.

SM107 was not run for this change: its plans keep the shard layouts that
`TestDSA` measured on SM107, and `TestDSADeterministicDKV` skips there.

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

Verified after the rebase on 2026-11-09 with NVIDIA L20X SM90 (132 SMs),
PyTorch 2.12.1+cu130, cuDNN 9.20.0, and CuTe DSL 4.8.0 compiling with the
CUDA 13.4 flavor. The checkout was the exact pushed commit `f4540d88` on
develop parent `7a68feff`. `cudnn`, the API module, and the SM90 kernel all
resolved from `/tmp/cudnn-fe-h32-f4540d88/python`, reporting frontend version
1.33.0; the host's installed frontend 1.31.0 was not used.

The L0 `test_sm90_deterministic_h32_topk512` cases cover BF16 and FP32 dK,
sparse negative IDs, replay, short five-query batch reordering with global
IDs, and an independent gradient reference. Strict replay against the
committed runtime key `70c200b13182bbc2` passed 2/2 cases in 7.63 seconds. The
L0 support-envelope cells passed 24/24 in 0.38 seconds. Four H32 L0 stress
cases passed in 15.25 seconds, covering Top-K 384/512/2048, both deterministic
dW folds, BF16 and FP32 dK, local and global IDs, and 194 query rows over 96
producer planes. Those stress cases check fixed-layout replay and the
independent numerical reference; they do not compare batch reordering across
the 96-plane producer boundary. Because the producer-plane assignment depends
on the flattened batch row after that boundary, SM90 H32 is documented as
Replay only rather than batch-invariant. The complete indexer-backward L0 file
finished with 43 passed, 20 expected SM100-only skips, and 63 deselected in
32.64 seconds.

The RED control used parent `7a68feff` with the new test file but without the
implementation. Both H32 core cases failed at admission with
`SM90 IndexerBackward requires H=64, got H=32`, confirming that the green run
exercised this change. Commands run with `PYTHONPATH` pinned to the checkout:

```bash
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k sm90_deterministic_h32_topk512 -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k check_support_envelope -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k 'deterministic_reproducible_and_correct and h32' -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -q -rs
```

The CUDA 12 compiler flavor was unavailable and remains unverified.

## Simplified sparse attention — 2026-11-09

The new suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py`.
It tests the public `cudnn.ops.simplified_sparse_attention` call. There is no
mode that guarantees determinism. All cases are L0 and use BF16 input,
BF16 output/gradients, and FP32 loss. Tests check output, indexer loss,
dQ, dK, dV, dIndexerQ, and dIndexerK. They do not expose or compare saved
internal statistics. There are no gate or recurrent-state gradients.

The initial SM100 run expanded to 16 cases: **9 passed, 7 failed, 0 skipped**.
Forward passed all eight cases. Backward passed one small golden case and
failed the other seven cases. Each failure stopped at dK, so it does not
establish the status of later gradient comparisons. No backward golden is
approved from that run. The three forward baselines passed the independent
FP64 reference and eight fresh-allocation replays before recording. Golden
names include the loaded DSL compiler's CUDA flavor.

| Check | Shapes / variation | Result |
| --- | --- | --- |
| Replay | Packed lengths 65/3/17; Hq16/top-k512, Hq32/top-k1024, Hq96/top-k2048 | Forward passes; backward dK differs by 3, 1, and 1 bytes in 43,520 bytes on replay 1. |
| Scheduling boundary | 297 queries, Hq16, top-k512; 148 SMs | Forward passes; backward dK differs by 13 of 152,064 bytes. |
| Batch changes | Hq16/top-k512; target length65; neighbor length17; insertion, removal, reordering, duplication, changed neighbor | Forward passes; backward stops at first insertion, with 3 of 33,280 dK bytes changed. Later layouts are unmeasured for backward. |
| Golden | Length17; same three Hq/top-k pairs | All forward cases pass. Hq16 backward passes locally; Hq32/Hq96 dK each differ by 1 of 8,704 bytes during replay. |

Inputs and upstream gradients are generated once on CPU, then packed without
new draws. Invalid rows include empty key intervals. The scalar loss uses a
fixed denominator of 257, and batch checks isolate the target's loss and
upstream gradient. Full empty Q/K streams are unsupported. Concurrent GPU
work and fresh allocations vary scheduling without changing the inputs.

Runtime: SM100, 148 SMs, driver 610.11 (CUDA driver API 13.3), PyTorch
2.13.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0 with the CUDA 13.4 native compiler,
TVM FFI 0.1.13.post2. The integration starts from Frontend revision
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`. The printed import path was the
integration checkout, verified before testing. SM103 and long filtered-path
determinism are unverified.

Initial measurement, from `test/python`, with the runtime library path set:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q -s
```

Use a new recording directory. Inspect hashes and reference results before
copying baselines. Normal checks leave `CUDNN_DETERMINISM_RECORD` unset:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q
```

The expanded suite has 20 cases. Adding Hq32/top-k1024 and Hq96/top-k2048
batch checks exposed two forward failures: each changes 1 of 4 scalar-loss
bytes when the target moves to packed slot 1. The output bytes match before
the loss comparison fails. Loss accumulation groups physical rows into warps,
so moving a sequence can change FP32 rounding. The expanded strict forward
run was **8 passed, 2 failed, 10 deselected**.

At the requester's direction, retain kernel behavior and scope expected
failures on SM100 to the measured dK byte mismatches and those two packed-loss
cases. Each replay passes its numerical reference before byte comparison. Shape/dtype changes,
non-finite results, other tensors, and other loss mismatches still fail.
Backward golden cases remain expected failures until replay is stable and
a baseline can be approved; an intermittent replay pass is insufficient.
Remove these exceptions with the corresponding reduction fixes.

The final installed-wheel L0 run on the same SM100 runtime reported
**106 passed, 12 xfailed, 4 deselected, 0 failed, 0 skipped**. Of these,
the determinism suite accounts for **8 passed and 12 xfailed**. Nine xfails
stop at a dK byte mismatch, two at the packed scalar loss, and one at the
unapproved backward golden after its small replay passes. Later tensor and
layout comparisons after each mismatch remain unmeasured. Forward status
is **Replay only**; backward is **Not guaranteed**. Expected failures do
not improve those support claims.

A final 20-case rerun with the FP64 reference checked on every replay also
reported **8 passed, 12 xfailed**. Eight cases stopped at dK, two at packed
loss, and two at unapproved backward goldens. Hq32's small backward replay
passed in this run despite failing earlier; no new baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

The installed wheel contains all 32 Python operation modules and its source
notice. Separate CuTe DSL 4.8.0 CUDA 12.9 compiler testing passed 17 public API
cases, including all nine Hq/top-k forward/backward reference combinations,
selected-support numerics, tails, invalid configurations, and capture rejection.
This used PyTorch/cuDNN built for CUDA 13; it qualifies the DSL compiler
flavor, not a deployment using CUDA 12 throughout. The loaded native libraries
were `cutlass/_mlir/_mlir_libs/_cutlass_ir.cu12.cpython-312-x86_64-linux-gnu.so`
and `cutlass/_mlir/_mlir_libs/_cutlass_ir.cu13.cpython-312-x86_64-linux-gnu.so`
in separate environments. SM103, long-path CUDA 12 compilation, and newer DSL
versions remain unverified. The actual DSL 4.7.1 runtime rejects this API before
kernel import. A missing PyTorch conditional-graph entry also rejects before
filtered-state allocation; bypassing that guard makes its regression test fail.

The installed-wheel L1 run passed **4 cases, 118 deselected, 0 skipped**:
cold exact repair through the public API, changed addresses/bounds/extents,
reserved capacity at its maxima, sampler diagnostics, and a 16,384-query
structured-score stress case. These are numerical and storage-reuse checks,
not bytewise determinism qualification of the long path.

```sh
python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

Compute Sanitizer `synccheck` passed all three Hq16/top-k512,
Hq32/top-k1024, and Hq96/top-k2048 public forward/backward reference cases:
**3 passed, 15 deselected; 0 sanitizer errors**. This covers those short
packed numerical paths, not the long filtered repair graph.

```sh
compute-sanitizer --tool synccheck --error-exitcode 86 python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py -m L0 -q -s \
  -k 'public_forward_backward_reference and ((k512 and bf16-h16) or (k1024 and bf16-h32) or (k2048 and bf16-h96))'
```

## Simplified sparse attention JAX — 2026-11-09

The JAX suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py`.
It exercises the public native-FFI API under `jax.jit` and `jax.vjp`, using
JAX/JAXlib 0.11.0 on the SM100 runtime above. The source base remains
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`, with this integration applied.

| Suite | Forward results | Backward results | Cases |
| --- | --- | --- | --- |
| PyTorch determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |
| JAX determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |

Both suites check the independent FP64 reference on every replay, then compare
raw bytes with fresh result buffers and concurrent GPU work. The JAX suite
uses the same fixed packed data, ragged tails, insertion/removal/reordering,
duplication, changed neighbors, and 297-query scheduling boundary. Internal
LSE, teacher distributions, and long filtered-path determinism remain unverified.

The JAX recording run reported **8 passed, 12 xfailed, 0 skipped**. Nine
cases stopped at atomic dK, two at the known packed loss, and one at the
unapproved backward golden. The three new forward JSON records passed their
references and replay checks. All input and output hashes match the existing
PyTorch goldens. Their filenames also pin JAX 0.11.0 and compiler CUDA 13.4.
No backward baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-jax-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py -m L0 -q -s -rxX
```

A combined installed-wheel run reached three additional byte mismatches after
dK happened to match: PyTorch H96/top-k2048 packed backward dV (1/33280 bytes),
its backward golden replay dV (1/8704 bytes), and JAX H32/top-k1024 replay dV
(2/43520 bytes). Each passed its independent numerical reference first. These
are the same already-excepted backward cases; both dK and dV use FP32 atomic
reductions. The scoped SM100 backward exception now names both gradients.
Other tensors, non-finite values, shapes, dtypes, and reference failures still
fail. The interrupted diagnostic run reported **104 passed, 3 failed,
13 xfailed, 6 deselected**; it is not a qualification pass.

The two JAX L1 numerical cases passed at top-k512 and top-k2048 with 129
queries and 131,329 keys. All 129 tied rows force two exact-repair chunks;
a second call changes indexer inputs and key bounds and includes invalid
rows. The tests check selected scores against full FP32 top-k and compare
output, loss, and all five gradients with an independent FP64 reference
using the selection from that same public call. Tied selections may differ
between calls, so a separate selector execution cannot supply its reference.

Near-uniform KL is about 2.5e-6 in the tied top-k2048 case. A pure 1% relative
loss check exposed about 2.6e-8 absolute error. A shared native attention
probe also showed FP32 teacher-mass rounding (up to 5.3e-7), contributing
about 1.5e-8 of loss error. The new long-path loss check uses 1% relative
error plus an absolute budget of `16 * FP32_epsilon * loss_coeff * valid_rows /
loss_denominator`; output and gradient tolerances remain unchanged. This
budget applies equally to both compiler flavors. No kernel arithmetic changed.

The final combined installed-wheel L0 run reported **135 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. Each framework's determinism suite
contributes 8 passes and 12 expected failures. Across both suites, 16 cases
stop at atomic gradient differences, four at packed scalar loss, and four at
unapproved backward goldens after small replays pass. Later comparisons after
a mismatch remain unmeasured. The 21 JAX public API cases cover all nine
head/top-k forward/backward combinations, selected support, tails, triangular
metadata, zero loss coefficient, input ownership, invalid inputs, captured-input
JIT, rejection of mixed GPU architectures, and the dense workspace index limit.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu13-cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

In a separate installed-wheel environment, CuTe DSL 4.8.0's CUDA 12.9 compiler
passed **23 JAX cases, 0 failed, 0 skipped**: all 21 public cases plus both long
Hq16 filtered-repair cases. The loaded native library was
`_cutlass_ir.cu12.cpython-312-x86_64-linux-gnu.so`; the CUDA 13.4 run loaded
`_cutlass_ir.cu13.cpython-312-x86_64-linux-gnu.so`. Each flavor used a fresh
process and its own initially empty DSL cache. PyTorch, JAX, and cuDNN still
use CUDA 13 packages, so this is compiler compatibility, not a full CUDA 12
deployment. CUDA 12 bytewise determinism and SM103 execution remain unverified.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu12-cache python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_jax.py -m 'L0 or L1' -q -s
```

The final wheel includes 34 Python operation modules, its Apache-2.0 notice,
and `contributors.md`. The inventory was compared byte for byte with the
checkout. The final rebuild changes only the JAX docstring's description of
the XLA repair loop relative to the full L0 run; the CUDA 12 run uses that
rebuild. Source copyright notices remain in the released files.

Compute Sanitizer **memcheck and synccheck each passed 1 JAX L1 case,
1 deselected, with 0 reported errors** on the final rebuilt wheel. The case
covers top-k512 cold repair in two chunks, changed inputs/bounds, selected
scores, and all five gradients. This does not qualify every head/top-k route.

```sh
compute-sanitizer --tool memcheck --report-api-errors explicit \
  --suppressions /tmp/cuda-binding-probe-suppression.xml --error-exitcode 86 \
  python -m pytest simplified_sparse_attention/cutedsl/test_filtered_jax.py -m L1 -q -s -k 512
# Repeat with --tool synccheck.
```

The initial memcheck run completed its numerical test and reported only seven
CUDA-binding feature-probe failures, plus seven extended diagnostics: the
bindings request CUDA API version 13041 against driver API 13030. A separate
lazy-import test reproduced the API errors without running these kernels.
The final run uses explicit API-error reporting and suppresses only
`cuGetProcAddress_v2` returning `CUDA_ERROR_INVALID_VALUE` (result 1) from
`cuda/bindings/_internal/driver*.so`. The XML record came from that import
control and retains its binding-module stack constraint. All memory-access
checks and other explicit API errors remain enabled; no device-access or
barrier error was suppressed.

JAX installed-wheel benchmarks (three warmups, 20 timed calls, synchronized
wall time) measured 2.951 ms forward and 3.441 ms training at
Tq1024/Tk4096/H16/top-k512, and 6.914 ms filtered forward at
Tq1024/Tk131072/H32/top-k2048. These shared-GPU samples include host dispatch
and completion waits; they are not direct speed comparisons with the earlier
PyTorch CUDA-event measurements.

### CodeQL follow-up — 2026-11-09

The follow-up to `40d2377` adds explicit lazy-export declarations, rejects
invalid GEMM layouts before allocation, and initializes top-k settings in
the base constructor. Removing unused names preserves the Hq16 kernel's
shared-memory reservation and arithmetic. No golden or expected-failure
rule changed.

The source-checkout L0 run on SM100 reported **148 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. This includes 13 new host cases for
invalid layouts and explicit/automatic top-k settings. The lazy-import test
also checks all four package exports against `cudnn.ops`. Both frameworks'
forward/backward reference, replay, batch-invariance, and golden checks ran
with the existing scoped exceptions. Support classifications remain unchanged;
SM103 and long-path bytewise determinism remain unverified.

Runtime: Python 3.12, PyTorch 2.13.0+cu130, JAX 0.11.0, cuDNN 9.27,
and CuTe DSL 4.8.0 with its CUDA 13.4 compiler. Imports resolved to the
edited checkout. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

CodeQL 2.27.4 with `codeql/python-queries@1.9.0` scanned all 34 operation
modules using `python-security-and-quality.qls`. None of the 14 inline
findings on PR #2370 remains. The scan still reports 239 other package
diagnostics; no query or path was excluded by this fix. Pre-commit passed.
Fern reported zero errors and one existing warning.

The subsequent L1 run reported **6 passed, 172 deselected, 0 failed,
0 skipped**. It covers PyTorch repair-state reuse, reservation, sampling,
and diagnostics, plus both JAX long-key repair cases and their gradients.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

### Host concurrency and export follow-up — 2026-11-09

The follow-up to `80de1aa` serializes each stream's shared-scratch enqueue
sequence for dense indexing, filtered indexing/reservation, and H16 backward.
Filtered-state retirement also protects its shared host list. No kernel
arithmetic, golden, or expected-failure rule changed.

The six new public-API cases pause a host thread after writing shared scores
or filtered inputs. Another thread then enters with opposite indexer queries.
They cover default, shared side, and separate streams at Tq8/Hq16/top-k512,
with Tk1024 and Tk131072. The independent expected attention outputs are all
ones and all zeros. Concurrent output and scalar-loss bytes must match the
sequential results; separate-stream calls must complete while the first is
paused.

With the original `80de1aa` kernels, these cases reported **4 failed,
2 passed, 31 deselected**: both shared-stream modes corrupted all 32,768
output elements in the first call, on both indexer paths. With the fix,
the six cases and the Torch-only import check reported **7 passed,
30 deselected, 0 failed, 0 skipped**. The import check blocks JAX in a fresh
interpreter; the JAX export assertion now lives in the optional JAX suite.

```sh
python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py \
  -m L0 -q -s -k 'host_threads_share_stream or lazy_imports'
```

The rebuilt `nvidia_cudnn_frontend-1.33.0-cp312-cp312-linux_x86_64.whl`
has SHA256 `55db69c2370f8bdc0fdd294c7a8352790011c0d54cd8ee1c71c6ada153a8eade`.
All 35 operation Python modules, `NOTICE`, and `contributors.md` match the
edited checkout byte for byte. The installed-wheel API-index check reported
**11 passed**, including the runtime baseline with six intentional additions
and no removals.

The installed-wheel L0 and L1 run reported **161 passed, 24 xfailed,
0 failed, 0 skipped, 0 deselected** in 750.39 seconds. It includes all six
concurrency cases, both frameworks' numerical and determinism suites, and
all six long-sequence cases. Existing expected failures and goldens are
unchanged. SM100 forward remains replay-only and backward remains not
guaranteed; SM103, long-path batch invariance, and saved internal statistics
remain unverified.

Runtime: SM100 (148 SMs), driver 610.11, Python 3.12.12,
PyTorch 2.13.0+cu130, JAX/JAXlib 0.11.0, cuDNN 9.27.0, CuTe DSL 4.8.0
with the CUDA 13.4 compiler, and TVM FFI 0.1.13.post2. Imports resolved to
the installed wheel. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m 'L0 or L1' -q -s -rxX
```
