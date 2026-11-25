# python/cudnn/sdpa — Agent Guide

SDPA-specific hard rules, on top of the package-wide Rules 1-11 in [../AGENTS.md](../AGENTS.md). Numbered `S1, S2, ...`
so reviews can cite them; append, never renumber. Each rule names its detector; run or cite it in review.

## Hard rules

General (unnumbered):

- **Mask attributes.** Check contradictory mask attributes before normalizing legacy spellings: a causal flag can
  overwrite explicit diagonal alignment and change non-square attention. Keep Python validation, analyzer facts and all
  six compiled SDPA builders consistent. Detector: `test_sdpa_causal_alignment.py`; architecture skips must not hide a
  host-side regression. In synthetic device fixtures replace both `_device_cc` and `_device_sm_count`, or CUDA is still
  probed.
- **Import boundaries.** An implementation's thin host entry must not import the engine registry that imports it;
  inject lowering and knob construction from the registry. Keep historical-domain metadata with engine definitions so
  placement need not import the candidate generator. Shared body-selection functions live in
  `fwd/implementation_selection.py`; SM107 half selection lives in `fwd/sm107_half_selection.py` and receives scalar
  inputs from the registry (no reverse import, no copy into placement). Keep performance policy out of fixed kernel
  contracts. Detectors: cold host-import and explicit-lowering tests in
  `test_sdpa_fwd_d512_fp8_implementation_contract.py`, the wide-placement and selection-module import guards.
- **CTA cluster vs MMA group.** In explicit implementation records `TILE_CGA_M/N` is the physical cluster and
  `CTA_GROUP` the MMA group (D512: four-CTA cluster of two-CTA groups; D576: one pair). Translate only at an adapter
  boundary that uses the historical MMA-group encoding, including its second support check. Detectors:
  `test_sdpa_fwd_d512_explicit_engines.py`, `test_public_cluster_width_is_honored_or_rejected`.
- **Specialized body grid bounds.** Check a specialized launch body's grid bounds independently of the generic
  scheduler. SM100 FP8 D128 decode puts batch in `gridDim.y`, so it replaces the existing body only for batch <= 65535.
- **Swap-AB decode** counts token units by live softmax columns, not padded MMA width (SM100 FP8: 16 physical, 8 live).
  Detector: D64 cases in `TestNativeFp8Decode` (Q3/G8, Q4/G8, explicit `SWAP_AB=1`, unset/zero controls).
- **Paged live-page counts** stay scalar: keep `(length + page_size - 1) // page_size` in persistent loaders;
  `cute.ceil_div(Int32, Int32)` can fail with an illegal `cute.derefine` on CuTe DSL 4.8. Detector: page8 case in
  `TestNativeSingleQPaged`. For an opaque compiler failure, retain the module after the failing pass with line info.
- **Backward band narrowing.** When narrowing a backward owner's mask flags with `dataclasses.replace`, pass
  `band=None` so `BandSupport` is recomputed. The band matrix samples each owner's real non-mask domain. Detectors:
  `test_every_row_carries_a_normalized_band_support`, `test_unmasked_baseline_is_accepted_by_every_row`.

**Rule S1 — Respect the declared Stats (LSE) packing: packed and explicitly padded Stats are different contracts.**

- Packed Stats use the same `cu_seqlen` packing as Q/O: token-major `(T, H)` (TE/Megatron, cuDNN TH1) or head-major
  `(H, head_stride)` (FA-style). Reject a dense-padded declaration for packed Stats.
- The requirement applies to the **graph declaration**, not the variant-pack buffer. A flat or capacity-shaped carrier
  is raw storage: check its storage span, then bind the declared strides. Never infer packing from carrier strides.
  Standalone APIs without graph storage semantics keep their layout checks.
- Validate by stride: token-major is `stride_h == 1 and stride_s == H`; head-major is `stride_s == 1` with `stride_h`
  covering the packed token count `T` (`stride_h >= H` is not the bound). `T` is a device value (Rule 3), so
  `stride_h >= T` is caller contract; never add a host length read, only an in-kernel assert. Classify only with
  `graph_analyzer.thd_stats_packing(stride_h, stride_s, h_q)`.
- Detectors: `test_fwd_probe_rejects_invalid_stats_metadata`, `stats_layout`-parametrized THD tests
  (`test_dsl_sm100_thd_stats` and siblings) in `test/python/sdpa/frost/`.

Related Stats and packed-binding contracts:

- A `(T, H, 1)` carrier is valid for either packing: validate strides/capacity, bind without copying
  (`test_graph_thd_independent_pitches_rebind_replay`).
- **Explicitly padded Stats** (no ragged offset, `thd_padded_stats=True`) address declared `(B, H, S_max)` strides even
  when Q/O are THD (FlashInfer prefill LSE). Never infer Stats packing from Q/O. Reject bad bindings before the
  stream-ordered tail initialization. It is not a backward contract. Contiguous rank-3/4 carriers with a different axis
  order are valid. Detectors: `test_sdpa_native_padded_stats_binding.py`,
  `test_padded_graph_contiguous_storage_uses_declared_strides`.
- Zero packed-Q capacity still runs the padded Stats seed; validate sinks, workspace and paged tables before it in both
  binders, then skip the attention entry. Quantized empty-Q calls also initialize Amax and scalar scratch; only
  nonquantized calls without padded Stats return early with no writes. Detector:
  `test_empty_quantized_validates_sink_before_amax`.
- Head padding is storage, not tokens: check the HN storage span separately from logical packed-Q coverage
  (`test_head_major_padding_separates_logical_rows_from_storage`).
- THD PackGQA: setup and decode count token tiles (`CGA_TILE_M / PACK_G`); Stats stores use the unpacked query head.
  The D128 BF16 top-left causal bounded decoder also serves paged THD (128x128, two CTAs, `SPLIT_KV=1`, shared SM107
  paged body): `test_thd_work_lookup_paged_replay`. Count packed token tiles for the initial-wave limit.
- **Physical origins.** Split partials use compact prefixes, but Q/K/V and final O/Stats each keep independent physical
  origins; never reuse partial indexing for inputs. Enable origin binding on every architecture, layout and admitted
  CGA (assert `origin_specs` in the prepared THD spec). SM120 hosts retain each port's physical capacity
  (`max_total_seq_len_*` bounds live work only) in both prepared and standalone-fallback routes. Quantized origin
  operands follow the quantization-role tuple (MXFP8: full tuple including scale buffers); data origins do not move
  scale-tile prefixes. Pass each tensor map's own token dimension to `build_thd_origins` (SM100 D512 MXFP8 Q/K/V use
  dimension 1, O dimension 2). Detectors: `test_sdpa_fwd_thd_origins.py`, `test_sdpa_fwd_mixed_kv_sm100.py`, SM107
  `TestMixedFp8Kv` (incl. `test_thd_cga1_origins`), `test_sdpa_fwd_thd_origins_fp8_sm120.py`,
  `test_sdpa_fwd_thd_origins_mxfp8_sm120.py`, `TestPhysicalOrigins` in `test_sdpa_fwd_mxfp8_sm100.py`.
- MXFP8 V-scale tail sanitization can touch the peer CTA's slab: keep the peer alive until cluster-memory accesses
  finish (D128 CGA2 `TestPhysicalOrigins` under racecheck).
- A test that changes a port's token pitch also updates that port's ragged offsets, bound through the tensor's
  `ragged_offset` reference (output names vary). Detectors: `test_graph_thd_independent_pitches_rebind_replay`,
  `test_independent_origins_rebind_replay`.
- Single-Q ragged split decode: the combine uses declared device Q lengths, not output address differences. Standalone
  THD requires Q lengths regardless of `seq_q_lens_present`. Detector: `test_half_q_single_q_ragged_decode`.

**Rule S2 — A change to FROST SDPA graph eligibility updates the affected architecture section under
[frost/support/](frost/support/README.md) in the same commit.**

- Pages describe current support by pass, precision and head domain. Use [the template](frost/support/TEMPLATE.md) and
  [SM90 example](frost/support/sm90.md); mark unchecked sections **Pending migration** (not "unsupported"); update the
  index coverage; never copy unverified legacy rows. `SUPPORT_MATRIX_TRACKER.md` is legacy reference only.
- Applies to every eligibility field: `dtypes`, `out_dtypes`, `d_shapes` / `d_pad_multiple` / `d_envelope_floors` /
  `thd_d_shapes`, mask and feature booleans (`causal`, `bottom_right`, `right_band_widening`, `swa`, `padded`,
  `padded_stats`, `sink`, `thd`, `cu_seq_len`, `bias`, `decode`, `layouts`, ...), and adding/retiring an `EngineSpec`.
  Knob-domain-only changes (`tile_ms`, `sched_policies`, ...) are exempt. A support-preserving refactor still updates
  moved source links and engine identities.
- Distinguish **native** from **envelope-served** (TMA zero-padding at a larger flavor's cost) for each head dim.
  `d_pad_multiple=0` means exact-only.
- Link enforcing capabilities, declaration checks and accept/reject tests; state feature-combination restrictions and
  opt-in vs ordinary discovery. Keep logs, test counts and perf results out of the pages.
- Review: a diff touching `fwd/engines.py`, `bwd/engines.py` or `cudnn/engines/manifest.py` without a support-page edit
  must be shown exempt.
- Prefill-only backward rows set `decode=False` explicitly (default `True`):
  `test_sdpa_bwd_d192_sm107.py::test_decode_declines_before_lowering`.
- Partial PackGQA: test ordinary candidate generation (`_pack_gqa_points`) and explicit plans at groups `tile_q`,
  `tile_q + 2`, `2 * tile_q`. Use the effective subgroup width for the packed span; keep the full Q/K ratio for KV-head
  mapping.

**Rule S3 — When P is aliased into a score slot's TMEM tail, the warpgroup that stores into columns another warpgroup
still has to read must be ordered after that read.**

- Kernels without spare TMEM pack P into the S slot tail (`P_EVEN_OFF`/`P_ODD_OFF` inside `S_ACC_*`, e.g.
  `sm100/d256_fp8_single_q/kernel.py`). Safe with one softmax owner; a race once a row's keys are split across two
  warpgroups (#981).
- Fix with an explicit release: the reader arrives an mbarrier after `tcgen05.wait::ld`; the storer waits on it right
  before the aliased store (`mb_softmax_hi_loaded`). Do not swap key ownership: alpha/stats alias S column 0, so the
  original ownership is forced.
- The split can hide behind another role: `sm100/d256_mxfp8_single_q/kernel.py` `FUSED_CORR_SPLIT_P` gives keys 64-127
  to the correction warpgroup; wait on `mb_stat_empty` before the aliased store. Its dense split-P schedule is safe only
  while both halves meet at named barrier `barrier_id=8` before the P store.
- Detector: `grep -l "P_EVEN_OFF: int = " python/cudnn/sdpa/fwd/kernels/*/*.py` crossed with
  `grep -n "SOFTMAX_WARPGROUPS=2\|split_p\|fused_corr_split_p" python/cudnn/sdpa/fwd/config_*.py`; for each P store name
  the reading warpgroup and the ordering barrier. The lagging half lags mainly when it rescales O, which `RESCALE_THRESHOLD` makes rare: force
  rescaling with `k_tile_growth` in
  `test_sdpa_fwd_mxfp8_sm100.py`. Fuzz symptom: `ref == 0, gpu != 0` on masked configs at aliased `key mod TILE_N`.
- Prove completion and inter-thread ordering separately for every TMEM reuse edge (`tcgen05.wait::ld/st`, producer
  `before_thread_sync`, sync, consumer `after_thread_sync`; `tcgen05.commit` fences implicitly). Trace each dependency in
  the emitted PTX and cubin; a numerical pass or STORE-completion fence is not proof. See PTX ISA 9.7.18.6.3-6.4 and
  9.7.18.11.1.
- A DSL version gate does not prove every PTX modifier assembles: cross-compile with the public package's default CUDA
  flavor. A CTA-group-1 kernel using all 512 TMEM columns uses ordinary alloc/dealloc (`.exclusive` can fail).
- A synthetic final ownership event keeps the consumer-completion dependency of the real lookahead event (D192 FP8
  rotated-P: final S0 waits for PV1[j-1]). Detector: `test_fp8_d192_final_p1_reuse_under_mma_skew`.

**Rule S4 — Split-K partial Stats keep the combiner's log base.**

- `stats_use_log2` changes final Stats only; partial LSE stays natural-log, including direct kernel entry points.
  Apply `log2(e)` once at the final store.
- Check O and Stats against an independent reference with nonzero logits. Detectors:
  `test_fp8_graph_stats_use_log2`, split-KV Stats tests, `test_sm120_direct_template_stats_base` (bypasses the adapter).

**Rule S5 — Strided outputs must retain their layout through the final store.**

- `make_array_view(t)[b, s, h, :]` returns a row pointer assuming unit D stride. Use full indexing or require
  D-contiguous storage; Stats stores too. Detectors: `test_pointer_combine_strided_outputs_and_dead_splits`,
  `test_sm107_fp8_stats_nonunit_row_stride`.
- TMA alignment uses each operand's element width (FP8 Q/K/V may produce one-byte O):
  `test_prepared_fp8_output_stride_uses_output_element_width`.
- Vector-load alignment follows declared base and strides; inspect the tensor iterator before `raw_ptr()`. Compact
  strides do not prove base alignment. Detector: `test_sdpa_bwd_dot_strides.py` with PTX controls.
- Block-scaled SF_O byte addressing is Int64 end to end; widen before multiplying:
  `test_block_scaled_sf_plane_stride_above_int32`.
- Padding helpers preserve strides when reinterpreting as 32-bit words:
  `test_sdpa_bwd_d256_sm100.py::test_batch_pitch_rebind_and_replay`.
- A TMA view split into head chunks keeps the logical head bound; keep bounded rank-4 loads for partial chunks
  (`0 * NaN` poisons BMM1). Paged head chunks keep the full tile-height pitch between dimension slabs and step
  dimensions in exactly one layer. Detectors: `TestD192Fp8HeadPadding`, `TestNativeD192Fp8HeadPadding`,
  `TestD256Fp8HeadPadding::test_paged_rebind_replay`, `test_paged_graph_fp8_d256_page_boundaries`.
- **Native backward gradient outputs must not overwrite each other.** Reject overlapping gradient address sets before
  any initialization or launch, on graph and standalone calls, rechecking pointers on every binding. Overlapping spans
  are not overlapping elements (interleaved Q/K/V gradients are legal); use declared strides. Cache disjointness
  geometry at plan time without per-row walks. Detectors: `test_native_gradient_overlap_rejected_before_entry`,
  `test_native_shared_gradient_storage_matches_address_oracle`, `test_native_overlapping_gradients_rejected_before_writes`.

**Rule S6 — A kernel feature lands on every arch line's test file, and its other-arch lowerings are smoke-compiled
from whatever GPU you have.**

- Each fwd arch line has its own file and marker: `test_sdpa_fwd_fp8_sm100.py` under `requires_blackwell` (SM 100..119,
  Rubin included; `_D128_ARCH` picks the kernel), `test_sdpa_fwd_fp8_sm120.py` under `requires_blackwell_geforce`
  (120..129), SM80 in its own file. `_skip_on_rubin` is d192/d256-specific, not a default. Detector:
  `pytest --collect-only -q -k <feature>` per file.
- When a CI lane selects an explicit file list, check collection through those entries. Helpers exposed via
  `staticmethod` in a guarded class keep the class's arch guard (`test_sdpa_fp8_sm107.py::TestNativeFp8SinkSplit`).
  Put shared implementations in modules without the `test_` prefix.
- Smoke-compile other-arch lowerings locally, e.g. `_load_sm120_kernel_module(None, TemplateParams(dtype_qkv=0,
  dtype_o=5), fp8=True, pertensor=True).compile(compute_capability=(12, 0), b=1, qh=2, kh=2, sq=256, skv=256, d_qk=128,
  d_v=128)`, for every template variant touched.
- In a `def` nested in a kernel body, do not touch a free variable inside a dynamic `if` (it becomes an unbound
  closure-local, `None` at trace). Hoist into a local before the `def` (`o_ptr = o.iterator.raw_ptr()`).
- Long flat `and`/`or` constexpr chains expand exponentially in CuTeDSL's AST preprocessing and stall cold compiles;
  group into short parenthesized clauses preserving order. Do not remove checks or raise timeouts. Detector: a child
  stack in `ast.fix_missing_locations`; retain fresh-process compile and artifact reload tests.

**Rule S7 — Promote indices before stride multiplication when the addressed span needs Int64.**

- Promote the index **before** multiplying; casting the product keeps the overflow. Test a stride above `2**32` and a
  smaller stride whose last batch offset exceeds `2**32` on input, Stats and gradient ports, with wrapped addresses
  seeded inside guard storage. Detector: `TestPreparedSm120Bwd.test_physical_batch_stride_above_int32`.
- The COMPACT path (no `wide_index`) is the production path: a BSHD tensor past 2^31 elements wraps in Int32
  `block_idx()` product chains (`dot_do_o`, `dkv_reduce`, `fold_quant`). Start every chain from `wide_index(...)` or
  `Int64(...)` (`_span_exceeds_int32` for vector indices); e.g. `((batch * S_Q + q_block * Q_TILE) * H + head) * D_V`
  must start from `wide_index(batch, o)`. Detectors:
  `test_sdpa_bwd_prepared_sm100_wide.py::test_prepared_sm100_compact_io_past_int32_elements` (L1, ~77 GiB) and
  `grep -nE "^\s+(base|pos|ws_base|dq_accum_base) = " python/cudnn/sdpa/bwd/kernels/bprop_chain_common.py`. Host-only:
  render for a `>= 2^31` geometry and grep the PTX address path for `mul.wide.s32` / `cvt.s64.s32`.
- Large-span reference checks are RELATIVE (`max_err <= rel * max|ref|`) and include the LAST batch.
- To name a faulting kernel without a sanitizer, use `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` and cuda-gdb.
- Paired dQ and dK/dV schedulers bound each Int32 divisor with their own tile count and footprint:
  `test_fp8_accumulation_swizzle_accounts_for_byte_payload`.

**Rule S8 — Hoist tensor-map acquire only over an immutable descriptor lifetime.**

- A setup kernel may publish K/V maps once; acquire each in every consuming loader warp (both CTAs of a pair) before the
  persistent loop, on every launch and replay. Maps rewritten or selected in the loop are reacquired at that boundary.
  Keep the shared TMA helpers' safe default for other callers.
- A caller that acquires explicitly disables its tile helper's default acquire. O slabs of one work item share a map;
  reacquire when the batch/map changes. 2x2 backward: loaders acquire compact maps at entry and physical maps per work
  item; the fused Qdk map needs its own acquire in the store warp on both CTAs.
- Detectors: `test_thd_tensormaps_rebind_and_replay`, `test_quantized_thd_tensormaps_rebind_and_replay` (NaN K/V tails),
  `thd_over_launched_units_are_dead`, `thd_multi_unit_per_cta`, D192/D256 THD origin/rebind suites.

**Rule S9 — Every scheduler reader needs a reuse back-edge; match output-phase cadence.**

- Every scheduler payload reader returns a reuse credit or gates the next publication. Remove an idle role's loop; if
  it was the only waiter on a credit barrier, stop arriving it. Check scheduler and credit-helper selection together
  (credit helpers need converged warps). Detector:
  `test_sdpa_fwd_o_empty_backedge.py::test_scheduler_and_credit_target_select_the_same_layouts`.
- A completed mbarrier phase needs a successful wait before next-phase arrivals. D128 backward store warp observes all
  four dQ source-credit terminal phases (zero-trip owners too). Detector: `test_sdpa_bwd_d128_terminal_credits.py` under
  synccheck.
- Match barrier cadence to the consumer loop (SM100 D512 MXFP8 SWA=128: use the once-per-tile Q/O alias signal).
  Detector: `test_d512_swa_waits_for_both_output_slices`; delays live in a private test template copy, never a
  production knob.

**Rule S10 — Every warp role decodes a work item with the same scheduler inputs.**

- LPT-L2 needs the runtime GQA ratio and KV length at every decode call, including the output-store warp.
- Packed tile heads and per-row query heads are different coordinates; derive each output head from the tile head.
- A THD body using the device-bounded persistent claim counter declares `THD_PERSISTENT = True`; inspect the launch
  grid when comparing persistent implementations.
- Preserve the initial-wave policy through binding: broad declarations must not freeze a one-wave cap; packed bounds
  use token rows per tile and packed head groups. Detectors: `test_native_initial_waves_bind_current_work`,
  `test_native_thd_initial_wave_geometry_rebinds`, `test_native_d128_packing_dense_replay`.

**Rule S11 — THD dead units must not access uninitialized per-sequence maps.**

- `B+1` map slots make `batch_idx == n_batch` in bounds but uninitialized. Exit dead clusters before acquiring maps,
  allocating TMEM or entering barrier protocols, using cluster-uniform `exit_if_dead_thd_cluster`.
- Under PDL, wait for setup before the dead-unit check reads metadata.
- Test overlaunch without total-length hints and replay with all lengths zero, under memcheck with O/Stats canaries.
- Physical-origin Stats and raw per-row scalar loads need their own live-unit guard; clamping to the last batch is
  insufficient. Detector: `test_empty_final_origin_stats_guard` under initcheck.

**Rule S12 — Placement grows by measured table entries, not by new hand-written boxes.**

- `cudnn/sdpa/placement_table.py` answers first in `fwd/placement.place` and `bwd/heuristics._leads`. Generate entries
  with `benchmark/placement_gen/gen.py` from a frozen sweep, qualify on a frozen holdout used once, cite the study.
  Never edit `TABLE` by hand.
- A new boundary is a study: class key from graph facts, one monotone threshold per class on a
  `placement_table.FEATURES` scalar. Hand rules in `placement.py` only for what the table cannot express, with a reason.
  A second condition (`Entry.also`) is allowed when no single feature orders a class; pre-register the candidate set
  (`gen.py --constrained [--two-feature]`, `--score`).
- A twin side with no changed holdout cell is untested: it keeps its entry and ships when neither side fails.
- When a study fails on a few cells, pool train and holdout and score once on a fresh holdout at new head pairs.
- Time on a full-power board; record `power.limit` per shard.
- A NON-MONOTONE class needs a finer key.
- Deterministic graphs are unmeasured except `placement_table.DETERMINISTIC_KEYED_ROWS` (phase `<phase>_det`); add a row
  only with a study against the backend's deterministic engine.
- The key omits the GQA group: the train grid spans small (2/3), non-power-of-two (5/6/7), served groups and, for tiled
  kernels, groups that do not divide the tile.
- The off-tile twin needs its own train lattice.
- Fit under the criterion: print changed-train worst before reading the holdout; prefer constrained fits (changed train
  worst <= 1.05) and one-parameter changes. Holdout gate is worst <= 1.10.
- A split-chooser fit times every split `split_kv_candidates` can return, then runs the real-path check (new vs
  committed chooser on fresh cases) before commit.
- Keep refinements local; an unqualified new class gets no entry. Check both sides of the first tile boundary
  (`test_sm120_short_square_class_keeps_default_and_long_square_leads`).
- Test ownership via `class_key` and the matched entry's `study`. Square causal canonicalizes to bottom-right; use
  unequal Q/KV for top-left. Re-run placement tests on the develop integration when another study lands.
- Default-placement tests under `test/python/sdpa/frost` unset `CUDNN_FRONTEND_ENABLE_FROST_ENGINES` in the test body
  (the autouse fixture sets it); record default and candidate plan identities (`test_sdpa_fwd_placement.py`).
- The grid reaches the class's extremes and straddles every boundary; draw holdouts off the training lattice; add a key
  field only when it follows the kernel (`_unpacked_decode`).
- A new capability that changes a class's plan makes its entries stale: key the new regime from the row's own decision
  (`gqa_split`); keep older entries for the unchanged regime.
- With explicit implementation rows, look up their entries before legacy-name fallback, keeping inherited pack/split
  classification (`test_sdpa_fwd_d256_implementation_contract.py`).

**Rule S13 — Register redistribution must converge at warpgroup scope.**

- All four warps of a warpgroup execute the same `setmaxnreg`. Put the shared service-role release before branching
  into role warps; keep the original register budget reachable by every releasing warp.
- Detector: in PTX/SASS, one common `setmaxnreg.dec` / `USETMAXREG.DEALLOC` site before role dispatch; trace the release
  guard to `tid.x` (counts alone are insufficient). Identify entries by the `cudnn_kernel__kernel_` symbol. Tests:
  `test_sdpa_fp8_sm107.py::TestNativeHalfRegisterRelease`, `TestServiceRegisterRelease` (SM100/SM107 entries, both D512
  MXFP8 V-half entries with negative controls). Also run replay and sanitizer checks.

**Rule S14 — Empty work must retain the buffer-acquisition acknowledgment.**

- An empty tile still waits until the loader has observed the buffer's empty phase. D192 accumulation publishes
  zero-byte K/V full arrivals for empty tiles. Detector: `test_d192_causal_empty_kv_tiles` (repeated replay); packed
  kernels also need envelope tails and live/empty transitions.

**Rule S15 — A scheduler payload, a statistics word and a mailbox slot are consumed before their slot credit (package
Rule 11).**

- Applies package [Rule 11](../AGENTS.md) (hazard, helper, scope, detectors) to scheduler responses and predecoded
  fields, softmax statistics and alpha slots, and cross-CTA mailboxes. Release through `tile_dsl.consume` or name the
  Rule 11 alternative in the kernel's barrier table.
- `read_clc_payload` keeps its cross-proxy fence; it does not order slots refilled by ordinary stores. Lane-zero
  read-and-broadcast does not order lane zero's load before the credit.
- An elected slot release joins every reader warp first (fence in each reader, then group barrier). Detector:
  `TestD512PhysicalOrigins.test_stats_slot_waits_for_all_warps`; racecheck can pass while it fails.

**Rule S16 — TMEM deallocation joins every reader warp.**

- Each reader completes `tcgen05.wait::ld`; the release joins all readers before the MMA pair deallocates (D512
  role-split: 128-thread named barrier before local and peer credits). Keep the join outside the persistent loop.
- Detector: `TestD512PhysicalOrigins.test_tmem_release_waits_for_all_readers` (event order, not duration). Delay the even
  CTA, the odd CTA and both.

**Rule S17 — Keep additive logits in range before softmax.**

- Form `scale * QK + bias` in natural units, subtract the row max before base-2 conversion, keep the natural-unit max
  for LSE; preserve the bias-free specialization. All-`-inf` row with `-inf` sink: O = 0, LSE = -inf; `+inf` sink:
  O = 0, LSE = +inf. Detector: `test_frost_sdpa_bias_sm120.py`.
- Keep result-dependent shared stores ahead of named stage credits (`store_consumed_result`, Rule 11), dense bias
  included. Detector: `test_frost_sdpa_bias_sm120_stress.py`.

## 2x2 datapath (cta_group::2, M=128 collective = 64 rows per CTA) lessons

Kernels: forward `fwd/kernels/sm100/d512_f16_2x2/kernel.py`, `fwd/kernels/sm107/d512_f16_2x2/kernel.py`
(`TemplateParams.mma_2x2`); backward stage 2 `bwd/kernels/sm100/d512_f16_2x2/kernel.py` (`config_sm100.CfgBwdD512x2`,
`TemplateParams2x2`); backward d256 `bwd/kernels/sm100/d256_f16_2x2/kernel.py`. Detector suites:
`test_sdpa_fwd_d512_2x2_sm100.py`, `test_sdpa_bwd_config_sm100_2x2.py`, `test_sdpa_bwd_dsl_sm100.py`,
`test_sdpa_bwd_thd_sm100.py`. Add lessons; do not restate.

- D256 packed backward schedules the 256-row write pair: run both halves of a tail pair; keep physical origins separate
  from compact prefixes; keep barrier arrivals for dead work (`test_sdpa_bwd_thd_d256_sm100.py::test_origins_rebind_and_replay`).
- The loader follows a lookahead consumer's operand order (D192 backward: preload Q0, then dO(i), Q(i+1), dO_dV(i)):
  `test_mma_order_arms_match_the_fp64_reference`.

Shared protocol:

- **An end-of-kernel ring drain walks the WHOLE ring** (`range_constexpr(STAGES)` from the current state), never
  `min(total, stages)` steps. Detector: `test_ring_drain_walks_every_used_slot` plus single-kv-tile GPU cells.
- **Never pin a nondeterministic compiler output by byte hash** (SM100 D512 THD ACK body). Use
  `test_stage2_2x2_sass_pins[sm_100a-thd]` structural checks.
- **Every local barrier object needs a successful phase observation before it advances again** (ABA in the D512 shared
  backward ring). Gate both consumers on all four observations via remote ACKs sent after the local empty wait.
  Detectors: `test_stage2_2x2_waits_for_delayed_empty_observer`,
  `test_sdpa_bwd_d512_sm107.py::test_chain_waits_for_delayed_empty_observer`.
- **Keep cross-pair polling separate from the phase-lifetime proof.** Polling stays at the ring full/empty and scheduler
  payload sites (`test_kernel_source_pins`). Poll shape per kernel: backward 128 tight tests / 128 ns timer sleep,
  forward 32 / 128. Do not change polling in a lifecycle fix without separate attribution. Keep the diagnostic levers
  `debug_heartbeat` + `debug_dump_addr` (and `debug_wait_ms`, which changes the wait shape). Forward detectors:
  `test_two_by_two_cross_pair_waits_poll`, `test_two_by_two_survives_gpu_time_slicing`,
  `test_two_by_two_parking_wait_form_under_time_slicing`; also `test_stage2_2x2_survives_gpu_time_slicing`.
- **Time-slicing: a watchdogged detector and a negative control that WEDGES the GPU both carry
  `@pytest.mark.xdist_group(name="gpu_exclusive")`** (with the `gpu_exclusive` marker; CI uses `--dist loadgroup`).
  Grouping does not reserve a GPU against other workers; a grouped timeout still needs diagnosis (junit durations are
  the tell). On HANG, report GPU UUIDs with PIDs and the child's `CUDA_VISIBLE_DEVICES`; report a failed `nvidia-smi`
  query as unavailable.
- **Pairs sharing an operand ring by cross-pair TMA multicast need an `empty` barrier with init = number of PAIRS**,
  released by every pair leader's `tcgen05.commit` with mask 0xF. Under `group = cta_2` only pair leaders arm
  `expect_tx` and wait. Detectors: `test_barrier_counts_and_cluster_span` (`RING_EMPTY_ARRIVERS == CGA_M // CTA_MMA`),
  `test_stage2_2x2_is_bitwise_the_role_split`.
- **Chunked K accumulation into one TMEM accumulator is bitwise equal to one long chain**: twin tests use `torch.equal`
  on int16 views.

Forward (d512 prefill):

- **Per-row reductions are HALF-row values until combined with lane r + 64**: exchange through SMEM + named barrier and
  run identical scalar chains (`test_two_by_two_directed_numerics[skewed_halves]`).
- **A slab a twin CTA multicasts into is gated by every DESTINATION's consumer** (`mb_o_empty` init `32 x KV_SHARE`,
  `arrive()` + `arrive_on_peer(cta ^ 2)`, drain last phase before exit). Detector:
  `test_two_by_two_twin_alias_gate_under_pair_skew` (`DEBUG_STG_DELAY_US`); a multi-tile cell alone
  (`test_two_by_two_persistent_multi_tile`) is not a detector.
- **`mb_o_full` init = 64** (one column half): `test_config_2x2_pins_and_ledger_formulas`
  (`O_CHUNK_ARRIVERS == CORR_LANES // 2`),
  `test_kernel_source_arrive_sites_match_the_ledger`.
- **A fixed TMEM column is not protected by a ring it rides**: per-slot columns (alpha `384 + s`, stats `386 + 2s`).
  Detector: `test_two_by_two_stats_ring_slot_race_empty_after_live`.
- **Dead/trimmed rows are zeroed by a SELECT, never `o * beta`** (`NaN * 0`):
  `test_two_by_two_trimmed_rows_with_nan_inputs_store_zero`.
- **Cluster-shared K/V needs cluster-UNION tile bounds** (`make_sdpa_helpers(kv_shared_cluster=True)`):
  `test_two_by_two_causal_512_cluster_union_bounds`.
- **Forward prepared hosts always use native binding**, including new template names; a new host ABI updates the native
  binder and frame-parity tests together. Detectors: `test_two_by_two_host_slots_match_role_split`,
  `test_two_by_two_default_plan_binds_natively`, `test_sdpa_native_prefill_binding.py`, `test_sdpa_binder_ownership.py`.

Backward (d512 stage 2):

- **The 2x2 D image puts the same rows in two warps** (warp w: rows `32 * (w % 2) ..+31` of kv-column half `w // 2`).
  Dense-only passes do not validate a lane map. Detector:
  `test_graph_thd_kv_len_64_masks_the_upper_column_half` (THD compiles `MASK_PADDED`) plus masked accepts under
  `stage2_datapath`. `test_dense_skv_96_zero_filled_tail` is not a mask detector: dense compiles `MASK_NONE`.
- **A tcgen05 SMEM descriptor root at or past 256 KiB needs `desc_version=1` on every `SmemTile`**; compute it from the
  slab list (`desc_version_2x2`), bind it once as module constant `DESC_VERSION`, and pin slab order. Detectors:
  `test_descriptor_roots_and_version_per_arm`, `test_kernel_source_pins` (every `SmemTile(` takes
  `desc_version=DESC_VERSION`).
- **A per-arch sibling fork is pinned to its parent by a code-diff allowlist, import-time refusal and rendering
  identity** (`bwd/kernels/sm107/d512_f16_2x2/kernel.py` vs `sm100/`; listed deltas include the ring-arm default,
  hard-coded `DESC_VERSION`, `SPIN_RING_WAITS`). Detectors:
  `test_fork_code_diff_is_only_the_listed_deltas`, `test_fork_refuses_the_sm100_arm_and_a_ninth_stage`,
  `renderings/md5_stage2_2x2_sm107a.txt`, `test_fork_is_bitwise_the_sm100_body_at_rubin_params`. Fork only after the
  parent has run at the new arm.
- **A directly constructed adapter spells the analyzer's conventions**: `window_size_left =
  diagonal_band_left_bound - 1`. A graph pass with a direct-adapter fail is the convention (`test_sliding_window`).
- **An omitted graph `attn_scale` is 1.0; an adapter's `scale_softmax=None` is 1/sqrt(d).** Detectors:
  `test_omitted_attn_scale_is_no_scaling`, `test_default_attn_scale`.
- **The `LDTM` SASS count of the masked compute body is a toolchain fact.** Pin per-body counts against the set a
  toolchain can produce (`test_fork_sass_pins_sm_107a`: `LDTM in (4, 6)`) and record the toolchain next to the pin.
- **cuDNN 9.26 builds no d > 256 backward plan on cc 10.7**, so that backward has no FROST-vs-backend A/B and is gated
  against its MMA floor. Probe first: `g.create_execution_plans` raises `cudnnGraphNotSupportedError` with reasons.
- **Attribute a floor gap with in-kernel clocks before touching a wait; quote cycles.** Levers:
  `TemplateParams2x2.debug_clk` / `TemplateParamsDbg.debug_clk` bracket each wait with `%clock64` and dump 32 x Int64 per
  warp (`DBG_CLK_*` constants; slot 0 / slot `DBG_CLK_BODY_NS` = actual SM clock in GHz). Assert
  `grid * 8 * 32 <= DBG_CLK_MAX_WORDS` before arming. Never compare wall times without a clock control (in-kernel clock
  or `ncu --clock-control base`), nor fixed-order arms on a power-capped board without an in-run thermal control.
  Detectors: `test_stage2_default_rendering_ptx_md5_is_unchanged`, `test_stage2_2x2_default_rendering_ptx_md5_is_unchanged`,
  `test_stage2_{2x2,4x1}_debug_clk_lever_renders_code`, `test_stage2_{2x2,4x1}_debug_clk_dump_accounts_the_waits`.

## Stage-3 GEMM (`bwd/kernels/bprop_matmul_blackwell.py`) perf-method lessons

- **A workspace budget does not bound the launch grid**: check the final GEMM's batch/head product (THD uses the real
  sequence count). Detectors: `test_sdpa_bwd_d256_chunk_grid.py`, `test_sm100_head_envelope_and_grid_compose`.
- **A head subgroup is legal only when the final reduction covers all global per-query-head partials**: chunks divide
  the group, the grouped dQ map keeps one K head (ceiling division), the global dK/dV fold runs once. Gate inherited
  chooser rules by dtype, layout and cc. Detectors in `test_sdpa_bwd_d256_sm100.py`:
  `test_head_subgroups_match_full_group_gradients`, `test_head_subgroups_preserve_mask_and_tail_gradients`,
  `test_head_subgroups_rebind_stream_replay_and_launch_count`, `test_head_subgroup_rule_is_scoped_and_workspace_is_honest`;
  SM107 `test_stage3_b_descriptor_covers_partial_head_groups`.
- **Read "residual vs the MMA floor" off the clock ratio.** `ncu --clock-control base --metrics
  sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__ctas_launched.sum -k regex:_bprop_matmul_bh_sm100_kernel` on the
  chain; compare with `CUPTI median us x NVML MHz`. Cycles growing at full clock = memory-latency bound; clock-invariant
  = issue/occupancy bound. Byte counters alone miss this.
- **Residency under CLC**: `launch__cluster_max_active`, `sm__ctas_launched.sum`, `sm__ctas_launched.min = 0` (idle SMs);
  compute rounds as `tiles / cluster_max_active` (`launch__waves_per_multiprocessor` ignores clusters).
- **A gain at base clock need not survive full clock**: close with a full-clock round-robin A/B, every arm built once and
  bitwise-checked, over the whole chain. Rule: `api_dsl._sm100_stage3_cgrp_tile_mn`. Detectors:
  `test_stage3_cluster_tile_rule_by_sequence_length`, `test_stage3_small_s_tile_is_bitwise_the_wide_row`,
  `test_stage3_tile_rule_reads_the_device_cc`.
- **A plan-time rule measured on one board needs a cc term, because adapters inherit `compile`.** Resolve cc through
  `SdpaBwdDslSm100._device_cc` and gate on an inclusive range (`_SM100_STAGE3_SMALL_S_CC`). Detectors: cc 10.7 suite's
  `test_stage3_tile_rule_keeps_the_wide_row_off_the_sm100_line`, `test_stage3_tile_rule_reads_the_device_cc`. Tripwire:
  `grep -n "compute_capability\|_device_cc" python/cudnn/sdpa/bwd/api_dsl.py`.
- **Refuted levers**: a non-power-of-two S/dS workspace row stride (`_skv_pad + 64`) and TMA L2 promotion on A/B
  descriptors are slower. Do not retry without new evidence.
- **Tile-row twins are `torch.equal`**; re-seed before every build compared (`bench_baselines.build_bwd` draws a fresh dO).
- **A pin whose record is local-only has never run.** Commit records under `test/python/sdpa/frost/renderings/` with a
  `dsl=` line; derive the expected set from the rule (`_stage3_default_tiles`). Tripwire:
  `test_stage3_md5_record_is_committed_and_complete`.
- **A hand-set `FROST_SOURCE_DIGEST` must name the record**: `compiled_cache.template_key` keys only the digest and plain
  call arguments, so a test that execs a template (`spec_from_file_location` + `FROST_TEMPLATE_PARAMS`) with a per-tag
  digest reuses the first load's artifact. Fold `sha1(repr(params))` into the digest (`_params_digest`) or leave it unset. Detector: dense and
  causal cell of one tag in one process with an empty `XDG_CACHE_HOME`.

## Output initialization regressions

- D256 chunk-local GQA partials are reused only after a complete group is folded; input origins stay global, output
  origins local. Detectors: `test_chunk_partials_workspace_and_subgroup_boundary`,
  `test_chunk_partials_match_full_partials`, `test_chunk_partials_preserve_per_member_dq`,
  `test_chunk_partials_rebind_stream_and_replay`. When monkeypatching the scratch policy, capture baseline shapes inside
  that context.
- Removing wrapper-side output clears requires the chain to overwrite every element (NaN-poisoned aux outputs, forbidden
  torch clears): `test_wrapper_aux_outputs_need_no_torch_clear`.

## Adapter declines that other suites pin

- SM80 native admission: THD never reads the declared batch stride; do not apply dense-layout predicates to THD
  (`test_sdpa_native_admission_sm80.py`). Keep declaration-only validators in `native_sm80.py`, outside prepared launch
  modules (import cycles). Remove unreachable test branches with a removed route.
- `gated_attention_block` tests pin SDPA adapter declines by message and run only on `oss` lanes. A change that serves a
  previously declined shape needs `@cudnn-ci-bot run ...,oss`. Detector: `git grep -n "match=" test/python/gated_attention_block`.

## Prepared THD launch bounds and setup

- Single-sequence setup specialization checks the invocation's runtime batch, not the declaration (FlashInfer declares
  4096). Each descriptor writer reads its own lengths and origin. Detectors: `test_single_sequence_runtime_batch_guard`,
  `test_single_sequence_batch_bytes`.
- Optional TMA descriptors keep the grid-constant ABI: `Optional[cutlass.GridConstant[TensorMap]]`. Run
  `test_sm120_prepared_rebind_strides_and_storage` dense control with `test_sdpa_fwd_paged_sm120.py`.
- Compiled host defaults must be serializable by `compiled_cache._json_plain` (use `None`, not tuples). Detectors:
  `test_f16_host_entries_carry_only_record_reproducible_defaults`, `TestNativeHalfStrides::test_artifact_reload`,
  D256 `test_prefill_artifact_reload`.
- Ragged-offset integer widths are pointer-ABI facts: include the per-port dtype tuple in the artifact key
  (`test_origin_integer_types_separate_compiled_artifacts`).
- One production binder per compiled host contract. Forward and half backward bind natively; quantized backward and
  standalone compact-GQA backward stay Python until migrated whole. Backward builders choose ownership explicitly. Never
  retry in Python after a native validation error. Legacy forward framing lives only in test reference modules.
- Staged forward adapters bind their core through the native binder; validate the whole core before any gather, seed
  or write; keep alias checks. Detector: `test_half_staged_has_no_execute_allocations_and_replays_current_storage`.
- Bound a cached envelope's launch with host-known token capacity and batch count; test old captures after replanning.
- Parallel descriptor setup fences every tensor-map writer
  (`test_parallel_thd_metadata_matches_lengths_and_normalized_cu`; run racecheck/memcheck).
- **Shared THD unit decode** (`_common_blackwell._thd_decode`, `tile_dsl.thd.thd_decode_unit_prefix`): setup
  (`write_thd_work_prefix`), host reservation (`THD_WORK_META_WORDS` / `THD_BWD_META_WORDS`) and decode `cu0` /
  `prefix0` must agree (`prefix0 = THD_BWD_WORK_PREFIX_OFF` in the backward layout). Derive batch count via
  `THD_BATCH_OF_WORK_META_WORDS`. Read family-specific config fields (`PAGED_KV`, `THD_VARLEN`) through defaults. Detectors: `test_thd_decode_unit_prefix_matches_walk`, `test_parallel_bwd_setup_prefixes`,
  `test_thd_work_prefix_replay_matches_live_tiles`, `test_dsl_sm120_thd_multi_unit_per_cta`, D512 `test_two_by_two_thd`.
- DSL traps: pass a `@cute.jit` helper's `Constexpr` through a kernel `Constexpr` parameter, not a literal. In a
  `@cute.jit` helper never rebind a None-defaulted parameter in a dynamic region (`TYPE_UNSTABLE_JOIN`); assign a fresh
  local and re-run callers that use the default.
- Standalone scratch queries after `check_support()` must not depend on a compiled module
  (`test_thd_split_workspace_query_before_compile`).
- **THD claim-counter re-seed: with several persistent-scheduler launches per execute, the scheduler words are per
  launch.** `live` (`meta[4B+2]`) counts one launch's units; re-seed the claim counter (`meta[4B+3]`) before every launch
  (done in `_clamp_thd_input_descs`). Detectors: `test_graph_thd_forced_head_chunks_match_reference_and_unchunked`,
  `test_graph_thd_forced_head_chunks_publish_live_and_reseed_the_claim_counter`. A new multi-launch consumer of
  `write_thd_live_and_ctr` owes both pins.
- THD head-chunk budget charges token rows; padded slab may overshoot by `budget // _SM100_WS_THD_PAD_SLACK`
  (`api_dsl._sm100_head_chunk_thd(..., t_rows=)`). Detectors:
  `test_thd_head_chunk_matches_dense_at_equal_tile_multiple_lengths`, `test_graph_thd_launch_count_is_the_promised_chain`.
- `prepared_sm100.compile_plan` keys on template digests, `Params`, geometry, regions and `sm`. Edits to
  `bwd/kernels/sm100/prepared_host.py`, `bwd/kernels/thd_helpers.py` or `bwd/kernels/bprop_chain_common.py` do not move
  the key, so a warm `CUDNN_FRONTEND_COMPILED_CACHE` serves the old host: test with
  `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`.
- **A host lever that changes launch shape rides in `Params`** (e.g. `Params.dq_b_head_group`, appended;
  `api_dsl.DQ_SINGLE_LAUNCH`). Detectors:
  `test_stage3_dq_single_launch_per_chunk_is_bitwise_the_per_member_launches`,
  `test_stage3_dq_launches_pair_every_q_head_with_its_k_head`. Capture adapter facts by monkeypatching
  `SdpaBwdDslSm100.compile`; spell measured and expected counts in launch-count asserts.
- **THD causal K-trim** (`_thd_causal_k_range`, `api_dsl.THD_STAGE3_TRIM`): the range may be empty and the kernel stores
  zeros; keep the host twin. Detectors: `test_thd_trim_covers_every_cell_stage2_writes_per_sequence`
  (`_thd_k_range_twin`), `test_graph_thd_causal_stage3_is_trimmed_per_sequence`,
  `test_graph_thd_causal_trim_is_bitwise_the_untrimmed_rendering`,
  `test_graph_thd_causal_matches_dense_bits_at_equal_tile_multiple_lengths`. Prove byte-identical renderings with
  `renderings/md5_stage3_sm100a.txt` (needs a GPU slot), run twice from independent processes before pinning.

## Heuristic geometry regressions

- When changing tile, packing, CGA or split candidates, spy on chooser inputs for split and unsplit legs; compare masked
  bounds with a visible-key oracle; check rescoring, dedup and the candidate cap. A winning-rank golden alone is not a
  detector.
- SM107 sweep seams (`config_sm107._N_CHUNK_OVERRIDE_FOR_SWEEPS`, `_BIAS_MODE_OVERRIDE_FOR_SWEEPS`,
  `_BIAS_STAGES_OVERRIDE_FOR_SWEEPS`) change plans and eligibility. Set them only via `monkeypatch.setattr` or a sweep
  driver that resets to `None`. Pin behaviour on the pick output (`test_n_chunk_override_seam_is_read_by_the_pick`,
  `test_seams_pin_the_form_and_the_depth`).
- SM107 small-tile bias table is code, one function per axis: `sm107_tile_n_pick` (used by
  `heuristics._resident_arm_tile_n`, `engines._bias_arm_decline`, adapter `_effective_tile_n`), `sm107_n_chunk_pick`,
  `sm107_n_chunk_refine`, `sm107_bias_stages_pick`, `sm107_bias_mode_pick`, `sm107_softmax_wgs_pick`
  (`SM107_THREE_WG_S_KV_MIN` / `SM107_THREE_WG_B_FLAT_MAX`), `sm107_bias_preload_chunks`
  (`BIAS_PRELOAD_CHUNKS` via `SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N`: 0 at TILE_N 128, 1 at TILE_N 64; a per-rendering
  constant read from `CFG.TILE_N`, never a module default).
- Bias plus mask is superadditive (instruction-cache cliff): never score them as independent costs. Check the
  `smsp__average_warps_issue_stalled_no_instruction_per_issue_active` ratio on any rendering that adds body.
- `test/python/sdpa/frost/test_sdpa_heuristics_small_tile_sm107.py` pins every cell with its margin. A new cell goes there
  with margin, node class, clock and control pair; never flip a default from one cell. 3-WG renderings are bitwise the
  two-group kernel (`test_sdpa_softmax_wgs_sm107.py`). Per-rendering constants carry their A/B/A in comments in
  `fwd/kernels/sm107/d128_f16_multi_q/kernel.py` and `fwd/kernels/sm107/small_f16_batch_pair/kernel.py`.

## Single-CTA packed split scheduler

- A one-CTA persistent scheduler publishes locally and never DSMEM-stores to a nonexistent peer. Consumers broadcast the
  response and predecoded fields (Q tile, head, batch, bounds, split) through lane zero before the slot credit. Applies
  to D256 FP8 predecode and D256/D512 MXFP8 schedulers (D512 keeps per-lane credits, 32x arrival count). Detectors:
  `test_sdpa_fwd_paged_thd_fp8.py` multiwave under memcheck/racecheck, `TestPhysicalOrigins`.
- D256 FP8 shared-alpha handoff uses per-lane release arrivals; elected credits only for TMEM-only handoffs (top-left
  mask/sink case under racecheck).
- Packed split workspace is bounded by packed-Q capacity; stores and combine reads use live token coordinates
  (`test_paged_thd_split_capture_lengths_and_stats`). Oversized Q/O allocations do not enlarge the live-Q bound
  (`B * S_q` without a total hint).
- A zero combine weight masks its partial O before arithmetic (`0 * NaN`), including underflowed chain/round
  accumulators. Detectors: `test_packed_prefetch_excludes_zero_weight_partials`,
  `test_quantized_sink_combine_excludes_zero_weight_payload`, `TestSplitCombineZeroWeight`. After changing a weight
  predicate, inspect load placement and run finite-input A/B/A.
- Sink-aware split partials exclude the sink; combine adds sink mass once to the denominator. Detectors:
  `test_packed_sink_combine_counts_virtual_key_once`, sink cells of `test_paged_thd_split_capture_lengths_and_stats`.

## Other kernel and adapter lessons

- Standalone prepared calls enter Q's CUDA device before resolving streams and restore the caller's device
  (`test_sm80_direct_standalone_with_another_device_current`, two GPUs).
- SM80 THD compiles explicitly for `sm_80` (`test_graph_thd_targets_nonzero_ordinal_a100`). After compiling
  `score_sign`, launch with `config_sm80.kernel_scale`; zero is a logical zero scale
  (`test_graph_thd_nonpositive_scale_and_replay`).
- **SM120 shared-memory ownership.** A paged producer that zero-fills invalid rows does not also run compute-warp tail
  sanitation; one warp owns zero fill, then sync before PV. Per-thread `cp.async` publishes with
  `cp_async_mbarrier_arrive(noinc=True)`. Prefill keeps producer commit/wait-group completion (the K-only exception in
  `sm120/f16_general/kernel.py`; never remove the V drain). Paged half decode (`decode_f16.py`) has its own qualified
  exception (`test_dsl_sm120_paged_decode_async_warm`). Persistent CTAs keep the same arrival count at every rearm
  (`test_sm120_frost_paged_d512`). Async-copy completion does not release other shared buffers (wide-head score
  exchange). Quantized siblings follow the same rules (`test_sdpa_fp8_v_tail_sync_sm120.py`). Mixed-FP8 page loads
  publish with release arrivals: choose publication by the producer's storage operation, not the MMA type
  (`test_sm120_mixed_paged_conversion_all_encodings`). Detectors under racecheck: `test_sdpa_fwd_d576_sm120.py`,
  `test_sdpa_fwd_thd_origins_sm120.py`, `test_sm120_frost_paged_d512[split_pack-dtype1]`,
  `test_d576_dense_features[dims0-plain-dtype1]`. Confirm active instrumentation with a current sanitizer version.
- Paged pool layout is per operand; validate each pool against its declared kind (`independent_pool_layouts` tests;
  backend fallback is not kernel support).
- An adapter subclass explicitly retains unsupported layout flags when its parent gains a mode
  (`test_standalone_adapter_rejects_thd`). Protect input, output and gradient padding with canaries.
- An admission test keeps other knobs legal so it reaches the rule it protects
  (`test_thd_pack_gqa_unsupported_flavors_decline`).
- A sink-only row publishes O=0, LSE=sink, selected explicitly (`test_paged_graph_fp8_sink_rebind_replay`,
  `test_prepared_fp8_d256_dense_keyless_sink`).
- A shared GMEM operand can need different SMEM swizzles per MMA consumer (FP8 D192x128 backward);
  `test_fp8_tma_layout_matches_each_mma_consumer` plus numerical tests.
- An explicit tile-domain extension must not enroll another dtype in a default picker; explicit-only widths go in
  `effective_tile_ns` (`test_fp8_d64_narrow_tile_keeps_graph_defaults`).
- Shared kernel geometry constants live below the heuristics/adapter layer
  (`test_split_combine_import_without_heuristics`, cold interpreter).
- Test launch recorders retained by a native binder capture only immutable metadata, never the API or prepared spec
  (`test_dsl_sm90_graph_singleton_strides`).
- A smaller backward KV tile narrows the transposed dS row: choose swizzle from the row byte width
  (`test_sdpa_bwd_thd_sm120_gqa.py` D256).
- A backward gradient-column partition multiplies the CTA grid; divide it out in reverse scheduling and the
  deterministic relay (`test_sdpa_bwd_sm120_wide.py`).
- FP8 `mma.sync` operands must be registers; keep zeros opaque via a register `mov` (`_zero_word`;
  `test_sdpa_bwd_fp8_sm120.py` SQ19/SKV37/D64).
- Head-dimension halves split by two warps contain whole swizzle chunks; check aliased storage phases; keep binder
  dimension admission in sync (`test_sdpa_fwd_d576_sm120.py`).
- Padded output rows get an explicit zero store after conversion (`test_sdpa_sm80_padding_nan.py`).
- Split-KV slab offsets use the same 128-byte alignment as `scratch_workspace_bytes`
  (`test_split_workspace_128_byte_alignment`).
- Splitting a broad engine into explicit implementations preserves each capability subdomain
  (`test_d256_thd_split_explicit_contract`, `test_explicit_d256_body_does_not_reselect_for_query_extent`).
- Backward rows delegate by selected body: declare `select_implementation` and `delegated`; owners come from
  `explicit_engine`; heuristics read `policy_name`; return a tuple of legal bodies, default first
  (`test_sdpa_bwd_explicit_engines.py`, `test_sm80_selector_is_the_adapters_d64_rule`). A row derived with
  `replace(parent.capabilities)` rebinds its selector and delegation (`test_default_plan_order`,
  `test_fixed_d64_owner_rebind_replay`).
- Moving a body changes relative-import resolution: `importlib.import_module` every moved module before the GPU run;
  diff failing test ids across `test/python/sdpa` against the parent. Search the retired path and basename
  (`git grep -n retired_kernel.py -- docs python test`); preserve frozen tracker entries and diagnostic labels.
