# python/cudnn/sdpa — Agent Guide

SDPA-specific hard rules, on top of the package-wide Rules 1-5 in
[../AGENTS.md](../AGENTS.md). Numbered `S1, S2, ...` so reviews can cite them
without colliding with the package-wide numbering; the list grows — append,
never renumber.

## Hard rules

**Rule S1 — Respect the declared Stats (LSE) packing: packed and explicitly
padded Stats are different contracts.**

- Consumers of packed Stats read through the same `cu_seqlen` packing as Q/O: TE and
  Megatron take token-major `(T, H)` (cuDNN's TH1 recipe) natively, FA-style
  callers take head-major `(H, head_stride)`. A dense-padded declaration
  (per-sequence stride) mis-addresses under that packing — it must be
  rejected for a packed Stats declaration, not silently accepted and mis-read.
- Validate by stride, not by a `thd`/`packed` flag: token-major is
  `stride_h == 1 and stride_s == H`; head-major is `stride_s == 1` with
  `stride_h` the declared head stride, which must cover the **packed token
  count `T`** — `stride_h >= H` is not the bound; at `T > stride_h` the
  per-head slices alias. `T` is a runtime total, so plan-time
  classification can only check `stride_s == 1 and stride_h >= 1`. In the
  THD path the packed total is a *device* value — Rule 3 bans reading it
  back, so `stride_h >= T` is **caller contract** (stated in
  the prepared THD binding contract), not something the adapter verifies:
  `as_strided` bounds-checks storage capacity, never overlap. Do not "fix"
  this with a host-side length read; an in-kernel assert is the only
  legal detector. Classify with `graph_analyzer.thd_stats_packing(stride_h,
  stride_s, h_q)` — the one classifier the fwd adapters, the bwd probe and the
  bwd lowering share; never re-implement the stride test inline (Rule 3's
  "suspect duplicated logic first").
- Covered by `test_fwd_probe_rejects_invalid_stats_metadata` and the
  `stats_layout`-parametrized THD tests (`test_dsl_sm100_thd_stats` and
  siblings) in `test/python/sdpa/frost/`.

An explicitly padded Stats output is a separate existing forward contract:
when Stats has no ragged offset, eligible rows with `thd_padded_stats=True` address
its declared `(B, H, S_max)` strides even when Q/O are THD. FlashInfer prefill's
padded LSE uses this contract. Do not infer Stats packing from Q/O alone, or
reinterpret a packed Stats declaration as padded. Preserve the declared
stream-ordered tail initialization and reject bad bindings before it writes.
`test_sdpa_native_padded_stats_binding.py` checks this path; native binding
changes no capability row and does not make padded Stats a backward contract.

Zero packed-Q capacity does not make this a no-write call: the declared padded
Stats seed still runs. Validate sinks, workspace and paged tables before that
seed, in both binders. After binding succeeds, execution seeds Stats and skips
the attention entrypoint when Q is empty. Binding itself remains metadata-only.
The empty padded Stats regressions in the same test file
exercise rejection after warmup and verify that output/workspace sentinels survive.
The padded declaration addresses caller storage: contiguous rank-3/4 carriers
are valid even when their own axis order differs from the graph's declared
`(B, H, S_max)` strides. `test_padded_graph_contiguous_storage_uses_declared_strides`
checks their actual stores, rebinding and replay; do not restrict them to flat
carriers or confuse them with a mismatched noncontiguous view.

Quantized empty-Q calls also initialize Amax and omitted scalar scratch, even
without padded Stats. Complete the same checks before those writes; only
nonquantized calls without padded Stats retain the early no-write return.
`test_empty_quantized_validates_sink_before_amax` covers FP8/MXFP8, and the
standalone regression checks real Amax/workspace sentinels and recovery.

Head padding occupies storage, not logical tokens. Check the full observed
HN storage span separately from the logical descriptor's bounded packed-Q
coverage; logical `numel` need not count inter-head padding. The detector is
`test_head_major_padding_separates_logical_rows_from_storage` for both binders.

Under THD PackGQA, setup and decoding count **token** tiles
(`CGA_TILE_M / PACK_G`), while Stats stores use the unpacked query head.
Changing only one side misses or aliases rows. The packing/capture tests
exercise partial groups and protect untouched tails with sentinels.
The D128 BF16 top-left causal bounded work decoder also serves paged THD
at 128x128, two CTAs and SPLIT_KV=1, including the shared SM107 paged body.
Packed-KV tests alone do not cover it: `test_thd_work_lookup_paged_replay`
asserts the paged bounded route and checks unequal Q/KV lengths, partial
PackGQA, changing K/V tables and live/empty transitions under one capture.
FP16 and sliding-window paged cases do not exercise this opt-in.
The initial-wave limit is a plan-time choice evaluated against current host-known
capacity at binding. Where an exact packed workload is available, count packed
token tiles rather than unpacked tiles times all query heads.

Packed split partials use compact work prefixes, but Q/K/V and final O/Stats
may each have independent physical origins. Do not reuse partial indexing for
input addressing or drop the origin declarations while lowering a split plan.
The D128 nonpaged split cases in `test_sdpa_fwd_thd_origins.py` poison the gaps
and rebind all pitches; mixed FP8 KV repeats them in
`test_sdpa_fwd_mixed_kv_sm100.py`. Sequence-local K/V maps clip each live segment,
so interior NaN padding cannot enter a tile tail.
The physical-origin binding must be enabled for every supported architecture
and layout, not just its paged path. Assert the prepared THD spec contains
`origin_specs` when the graph declares offsets. The SM107 `TestMixedFp8Kv`
THD cases cover mixed KV and uniform half, packed and unpacked, with poisoned
gaps, independent port origins, rebinding, and changed-length replay.

For single-Q ragged split decode, physical Q-offset differences can include
storage gaps. The combine must use the declared device Q lengths (ordinary or
cumulative), independently of output addresses. Exercise an empty sequence with
nonzero offset distance and assert that both O and Stats canaries remain intact;
`test_half_q_single_q_ragged_decode` covers graph and standalone execution,
rebinding and retained replay. Standalone THD requires Q lengths independently
of the dense-only `seq_q_lens_present` flag: bind against the compiled spec,
validate ordinary B or cumulative B+1 storage, and reject uncompiled operands.
A graph-only test misses a direct-API operand silently dropped by that flag.

**Rule S2 — A change to any FROST SDPA `Capabilities` row updates
`python/cudnn/sdpa/frost/SUPPORT_MATRIX_TRACKER.md` in the same commit.**

- The matrix is the only place a reader can see what FROST serves across
  arch x pass x head-dim x dtype without reading five `engines.py` rows. It
  is generated by hand from those rows, so it goes stale the moment one
  changes and nothing else notices.
- Applies to every field that changes what a graph gets: `dtypes`,
  `out_dtypes`, `d_shapes` / `d_pad_multiple` / `d_envelope_floors` /
  `thd_d_shapes`, the mask and feature booleans (`causal`, `bottom_right`,
  `right_band_widening`, `swa`, `padded`, `padded_stats`,
  `sink`, `thd`, `cu_seq_len`, `bias`, `decode`,
  `layouts`, ...), and adding or retiring an `EngineSpec` row. A change
  confined to knob domains (`tile_ms`, `sched_policies`, ...) does not
  need a matrix edit — the matrix deliberately does not track knobs.
- The matrix distinguishes **native** (a kernel for this geometry exists)
  from **envelope-served** (the graph rides a larger flavor with TMA
  zero-padding, at that flavor's MMA cost). When adding a head dim, say
  which one it is — an envelope hit is a perf cliff a user cannot see from
  a `Capabilities` row. `d_pad_multiple=0` means exact-only, so no
  envelope at all.
- Reviewing: if the diff touches `fwd/engines.py`, `bwd/engines.py` or
  `cudnn/engines/manifest.py` and `python/cudnn/sdpa/frost/SUPPORT_MATRIX_TRACKER.md` is
  untouched, ask why before approving.
- Prefill-only backward rows explicitly set `decode=False`: the capability
  default is `True`, which otherwise admits `S_q == 1` before the adapter
  rejects it. `test_sdpa_bwd_d192_sm107.py::test_decode_declines_before_lowering`
  checks real dense/THD graph declarations at that boundary.

**Rule S3 — When P is aliased into a score slot's TMEM tail, the warpgroup
that stores into columns another warpgroup still has to read must be
ordered after that read.**

- Some kernels have no spare TMEM and pack P into the tail of the S slot it
  was computed from (`P_EVEN_OFF`/`P_ODD_OFF` inside the `S_ACC_*` column
  range — e.g. `sm100/prefill_d256_fp8.py`, `P_EVEN_OFF = 96` in a 128-col
  slot). With a single softmax owner that is safe: all of S is in registers
  before any P store. It is a race the moment a row's keys are split across
  two softmax warpgroups: the half storing into the aliased columns
  overwrites the *other* half's unread S whenever that half lags a tile.
  GitHub #981 was exactly this — garbage weights on keys 96–111, masked
  paths only, sporadic and data-dependent (the no-mask fast path is
  single-owner).
- The fix is an explicit release: the reading half arrives an mbarrier
  after its `tcgen05.ld` has landed (`tcgen05.wait::ld` first — the
  softmax paths otherwise never wait on loads explicitly), and the storing
  half waits on it right before the aliased store (`mb_softmax_hi_loaded`
  in `sm100/prefill_d256_fp8.py`, mirroring `mb_softmax_max`). Do NOT try
  to fix it by swapping which half owns which keys: the alpha/stats value
  is aliased too (`STATS_*_OFF == S_ACC_*_OFF`, i.e. S column 0), so the
  half that stores alpha must also be the one that owns key 0 — the
  original ownership is forced, only the ordering was missing.
- The same kernel family can hide the split behind a different role name.
  `sm100/prefill_d256_mxfp8.py` has no `mb_softmax_max`: its
  `FUSED_CORR_SPLIT_P` schedule (mxfp8 + strict top-left causal) gives
  keys 64–127 to the *correction* warpgroup (`_fused_p1_step`), and half 0
  stored P into cols 96–111 with nothing ordering it after that
  warpgroup's S-hi load. There the release already existed —
  `mb_stat_empty` is arrived right after that load — so the fix is to wait
  on it *before* the aliased store instead of at the end of the iteration.
  The dense split-P schedule of the same kernel is safe only because both
  halves meet at a 256-thread named barrier (`barrier_id=8`) after their
  loads; if that barrier ever moves below the P store, this comes back.
- Detector: every kernel with `P_EVEN_OFF: int = ` inside the `S_ACC_*`
  range (`grep -l "P_EVEN_OFF: int = " python/cudnn/sdpa/fwd/kernels/*/*.py`,
  today the six d256 kernels) whose config can select
  `SOFTMAX_WARPGROUPS == 2` (`grep -n "SOFTMAX_WARPGROUPS=2\|split_p\|fused_corr_split_p" python/cudnn/sdpa/fwd/config_*.py`)
  — regardless of which role runs the second half. For each P store, name
  the warpgroup that still reads the target columns and the barrier that
  orders the store after that read; if there is none, it is a bug. Random
  data can hide it: the lagging half usually lags only when it rescales O
  (alpha ≠ 1), which `RESCALE_THRESHOLD` makes rare — force it with K
  scaled by `growth ** (key // 128)` (`k_tile_growth` in
  `test/python/sdpa/frost/test_sdpa_fwd_mxfp8_sm100.py`; the pre-fix mxfp8
  kernel then fails every run with inf/NaN O). The symptom in a fuzz suite
  is `ref == 0, gpu != 0` elements on masked configs with `key mod TILE_N`
  in the aliased range — recover the leaked keys by matching
  `O_gpu - O_ref` against `V` rows.

- Prove completion and inter-thread ordering separately for every TMEM reuse
  edge. The specialized synchronization route uses the applicable
  `tcgen05.wait::ld/st`, producer `before_thread_sync`, synchronization, and
  consumer `after_thread_sync`; `tcgen05.commit` supplies its implicit
  producer fence. A standard memory-model alternative needs a complete
  ordering proof. Detector: enumerate the `tcgen05` load/store/MMA,
  completion, fence, and mbarrier sites in each emitted PTX flavor, then
  trace each dependency in its actual cubin. A STORE-completion
  `FENCE.VIEW.ASYNC.T` or numerical pass alone does not establish inter-thread
  ordering. Specialized fences may constrain code motion without a distinct
  SASS opcode. See [PTX ISA, specialized synchronization and canonical
  patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/),
  sections 9.7.18.6.3–6.4 and 9.7.18.11.1 in PTX 9.4.
- A DSL version gate does not establish support for every PTX modifier in
  the installed compiler. Cross-compile with the advertised public package's
  default CUDA flavor. A CTA-group-1 kernel allocating all 512 TMEM columns
  can use ordinary allocation/deallocation; the `.exclusive` modifier can
  fail assembly even with public CuTeDSL 4.8. The d128 dense, THD and grouped
  execution tests must cover all four accumulation routes, plus recompute
  as an unchanged control.


- A synthetic final ownership event must preserve the consumer-completion
  dependency supplied by a real lookahead event. In the D192 FP8 rotated-P
  schedule, S0[j+1] also orders reuse after PV1[j-1]; the synthetic final S0
  event must therefore wait for that PV completion. A thread arrive alone
  can overwrite P1 and even lap the chunk-ready parity. Detector:
  `test_fp8_d192_final_p1_reuse_under_mma_skew` delays the previous P1 reader.

**Rule S4 — Split-K partial Stats keep the combiner's log base.**

- `stats_use_log2` changes final Stats only. Partial LSE consumed by a
  natural-log combine stays natural-log, including direct kernel entry points;
  apply `log2(e)` exactly once at the final store.
- Check both O and Stats against an independent reference with nonzero logits.
  O-only checks miss a wrong Stats base; zero logits miss an unscaled row maximum.
  See `test_fp8_graph_stats_use_log2` and the split-KV Stats tests.
  `test_sm120_direct_template_stats_base` bypasses the adapter: the adapter
  clears the partial-log2 flag itself, so adapter-only tests cannot detect
  a missing guard in a directly called template.

**Rule S5 — Strided outputs must retain their layout through the final store.**

- `make_array_view(t)[b, s, h, :]` returns a row pointer; indexing that pointer
  by `d` assumes a unit D stride. Use full indexing (`view[b, s, h, d]`) when
  accepting an arbitrary declared D stride, or explicitly require D-contiguous
  storage. Test padding canaries as well as numerical output; the detector is
  `test_pointer_combine_strided_outputs_and_dead_splits`.
- TMA alignment checks use each operand's actual element width. FP8 Q/K/V
  can produce half or FP8 O; treating O as always two bytes admits strides
  aligned to eight elements that are illegal for one-byte O. Keep engine and
  adapter admission in agreement; the detector is
  `test_prepared_fp8_output_stride_uses_output_element_width`.

Stats stores obey the same full-indexing rule as O. Nonunit sequence stride
can otherwise leave half the rows unwritten while corrupting padding. The
SM107 detector is `test_sm107_fp8_stats_nonunit_row_stride`; MXFP8 also checks
rebound padded and batch-inner layouts under CUDA Graph replay.

Vector-load alignment must follow each input's declared base and physical
strides. Inspect the original tensor iterator before `raw_ptr()` conversion,
which can discard its alignment declaration; checking the converted pointer
can leave an intended vector path scalar. Compile both aligned and misaligned
controls and inspect the emitted loads, then check physical pitches, pointer
rebinding and replay. `test_sdpa_bwd_dot_strides.py` covers the shared dot helper.
Compact strides do not establish base-pointer alignment. Keep the declaration
check for both inputs in the compact path too; the compact alignment-2 cases
exercise O alone, dO alone, and both inputs, with matching PTX controls.

Block-scaled SF_O uses byte addressing: its fake tensor extent, host geometry
arguments, device parameters, and every intermediate offset product must all
stay Int64. Widen operands before multiplication. The physical detector is
`test_block_scaled_sf_plane_stride_above_int32`: it writes two live SF planes
separated above 2**32 and checks capture/replay. A wide fake extent alone only
fixes binding; deliberately narrowing the plane stride must fail numerically.

Padding helpers must preserve strides too: reinterpreting data as 32-bit words
changes stride units, not physical layout. Both the input address and output store
must use those strides. Pitched BSHD can look correct at batch zero and silently
read the wrong K/V at batch one. The detector is
`test_sdpa_bwd_d256_sm100.py::test_batch_pitch_rebind_and_replay`, including
both source padding and output copy-out with canaries in unused physical rows.

A TMA view split into fixed head chunks must retain the logical head bound.
A rank-5 `(chunk, row, inner)` view can treat storage slack as live columns even
when Q is correctly zero-padded; `0 * NaN` still poisons BMM1. Keep the bounded
rank-4 loads for partial head chunks, and preserve the exact-head lowering.
`test_sdpa_fwd_fp8_sm100.py::TestD192Fp8HeadPadding` and the native SM107
`TestNativeD192Fp8HeadPadding` bridge poison Q/K/V head slack and check full
O/Stats/Amax, output canaries, rebinding and retained replay on both arch lines.
For paged head chunks, the descriptor box may cover only one page slice while
the MMA operand still uses the full tile-height pitch between dimension slabs.
Preserve that pitch and step dimensions in exactly one layer: a paged helper
that loops dimension subtiles must not also request multiple loads per subtile.
`TestD256Fp8HeadPadding::test_paged_rebind_replay` covers partial K/V heads,
short and wide pages, independent pools/tables, sinks and changed-length replay.

**Rule S6 — A kernel feature lands on every arch line's test file, and its
other-arch lowerings are smoke-compiled from whatever GPU you have.**

- Each FROST fwd arch line has its own test file and marker:
  `test_sdpa_fwd_fp8_sm100.py` runs under `requires_blackwell` (SM 100..119,
  the Rubin lane included — `_D128_ARCH` picks the kernel), the sm120 line
  under `requires_blackwell_geforce` (120..129) in `test_sdpa_fwd_fp8_sm120.py`,
  sm80 in its own file. A test added to one file never runs on the other lanes,
  and `_skip_on_rubin` is a d192/d256-flavor statement, not a default decorator
  to copy. Detector: `pytest --collect-only -q -k <feature>` per file must list
  the cases (the block-scaled O review found SM120 FP4 declining itself
  and the Rubin lane skipping the epilogue entirely).
- When an architecture CI lane selects an explicit file list, check collection
  through those entries. A directory-wide local sweep does not prove CI coverage.
  A guarded class in the selected entry can expose parametrized helper tests via
  `staticmethod`; retain the architecture guard because the helper module's
  `pytestmark` does not transfer with its functions. The SM107 sink/split
  detector is `test_sdpa_fp8_sm107.py::TestNativeFp8SinkSplit`.
  Put shared implementations in modules without the `test_` prefix so directory
  sweeps collect each architecture entry once, rather than also collecting helpers.
- The DSL traces the kernel in Python before any arch-specific codegen, so a
  lowering for an arch you do not have still fails or passes its trace here:
  `_load_sm120_kernel_module(None, TemplateParams(dtype_qkv=0, dtype_o=5),
  fp8=True, pertensor=True).compile(compute_capability=(12, 0), b=1, qh=2, kh=2, sq=256,
  skv=256, d_qk=128, d_v=128)` on an SM100 box reproduced the SM120 lane's
  `'NoneType' object has no attribute 'iterator'` exactly. Run it for every
  template variant you touched before pushing.
- Inside a `def` nested in a kernel body, do not touch a free variable
  (attribute access, store through it) inside a dynamic `if`: the DSL's region
  rewrite yields and rebinds the names it sees written there, which makes the
  free variable an unbound closure-local — it reads as `None` at trace time,
  or `UnboundLocalError` if you print it at the closure's top. Hoist what the
  closure needs into a local before the `def` (`o_ptr = o.iterator.raw_ptr()`,
  `sfo_base_ptr`) and let the closure add offsets only.

A cold compile can spend its deadline in Python AST preprocessing before any
GPU launch. CuTeDSL's Boolean rewrite can reuse the preceding expression
multiple times per operand; a long flat `and`/`or` chain then expands
exponentially for recursive AST visitors. Group a long constexpr predicate
into short parenthesized clauses while preserving operand and short-circuit
order. Do not remove alignment checks or raise the timeout to hide this cost.
Detector: capture the child's stack during cold artifact compilation; if it
lands in `ast.fix_missing_locations`, inspect the transformed AST's recursive
visit count, not just its unique-node count. Retain fresh-process compile,
artifact reload, and the affected kernel's numerical/layout checks; a cache
hit alone bypasses the problematic preprocessing.

**Rule S7 — Promote indices before stride multiplication when the addressed
span needs Int64.**

A stride can fit in Int32 while `index * stride` does not. Promote the index **before** multiplication when the addressed span
requires Int64; casting the completed product preserves an overflow. Exercise
both a physical stride above `2**32` and a smaller stride whose last batch
offset exceeds `2**32`, including input, Stats and gradient ports. Seed the
wrapped addresses inside allocated guard storage, so a deliberately narrowed
control fails numerically without an out-of-bounds access. See
`TestPreparedSm120Bwd.test_physical_batch_stride_above_int32`.

The COMPACT path is the one without `wide_index`, and it is the one production
hits. A compact BSHD io tensor past 2^31 ELEMENTS (`B x S x H x D > 2^31`:
d512 at B2 S32K H128, B1 S64K H72+, B32 S2K H128 -- B8 S8K H128 is the same
count) is addressed by the chain's `dot_do_o` through
`((batch * S_Q + q_block * Q_TILE) * H + head) * D_V`, an Int32 product chain
of `block_idx()` values and static extents that the DSL widens only at the
byte multiply (`mul.wide.s32 %rd, %r, 2` in the PTX), so every row past 2^31
elements read 8 GiB below the tensor. Measured 2026-10-01 on the B200: B1 S64K
H72 / H96 completed with a silently wrong delta (dq error 2-3x the bf16 norm,
inside a vacuous absolute tolerance), B1 S64K H128 / B2 S32K H128 / B32 S2K
H128 died with an illegal address -- cuda-gdb on the GPU coredump
(`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`; the only sanitizer-free way to name the
kernel when compute-sanitizer and the driver disagree): `Warp MMU Fault` in
`cudnn_kernel_dot_do_o_kernel`, grid (16, 128, 32), block (15, 6, 29) = batch
29 (the first batch whose base passes 2^31 is 16). `dkv_reduce` and
`fold_quant` carried the same `idx * 8` chain over the GQA partials
(`B x S x H_q x D`). The fix promotes the block indices BEFORE the multiply
(`wide_index(batch, o) * S_Q + ...`; `_span_exceeds_int32` for the vector
indices), the identity below 2^31 -- the small-shape PTX is byte-identical
(md5 8ad680c0 before and after at B2 S1K H8). Detectors:
`test_sdpa_bwd_prepared_sm100_wide.py::test_prepared_sm100_compact_io_past_int32_elements`
(L1, ~77 GiB: the chain's delta region against torch's rowsum on every batch --
a wrapped read is O(100 %) off -- and dQ/dK/dV of the FIRST and LAST batch
against fp32; it skips where the memory is not free), and the grep
`grep -nE "^\s+(base|pos|ws_base|dq_accum_base) = " python/cudnn/sdpa/bwd/kernels/bprop_chain_common.py`:
every product chain it lists must start from a `wide_index(...)` or an
`Int64(...)` index. Reference checks that gate a large-span run must be
RELATIVE to the reference magnitude (`max_err <= rel * max|ref|`; with |dq|
~3e-5 an absolute 5e-2 passed an all-zero tensor) and must include the LAST
batch, where the linear offsets are largest. Rendering a kernel host-only for
a `>= 2^31` geometry and grepping its PTX for `mul.wide.s32` / `cvt.s64.s32`
on the address path shows the wrap before any GPU run.

For paired dQ and dK/dV schedulers, bound each decoder's Int32 divisor using
its own tile count and operand footprint. FP8 dQ derives its swizzle from
K/V while dK/dV derives it from Q/dO; checking just one misses asymmetric
sequence lengths. The host-only detector
`test_fp8_accumulation_swizzle_accounts_for_byte_payload` exercises both
orientations, the last safe tile and the first overflowing row.

**Rule S8 — Hoist tensor-map acquire only over an immutable descriptor lifetime.**

- A setup kernel may publish K/V maps once before attention. Acquire each map
  in every consuming loader warp before its persistent loop, including both
  CTAs of a pair, before disabling the per-load acquire. A fence in another
  CTA is insufficient; cluster or stream ordering does not replace it.
- Repeat the acquire on every launch and graph replay. A map rewritten or
  selected inside the loop needs acquisition at the corresponding boundary.
  Preserve the shared TMA helpers' safe default for other callers.
- Check fresh bindings and changed device-side lengths after capture, with
  NaN-filled K/V capacity tails and independent O/LSE references.
  `test_thd_tensormaps_rebind_and_replay` covers D128, D256 and D512 half with
  two CTAs and D192/V128 half with both one and two CTAs.
  `test_quantized_thd_tensormaps_rebind_and_replay` covers D128/D192/D512 FP8
  and D128/D192 MXFP8, including both E4M3 and E5M2 inputs. The same probe
  covers all four half/FP8 widths on SM107 (including FP8 D256); unsupported
  SM107 MXFP8 THD and D192 half single-CTA configurations are skipped.
- O slabs within one work item share a map. Acquire before the first slab
  inside the existing live-work guard; retain store commit/wait and pipeline
  synchronization for every slab. Reuse across work items requires reacquiring
  whenever the selected batch/map changes. Exercise empty and repeated work items
  with the existing `thd_over_launched_units_are_dead` and
  `thd_multi_unit_per_cta` regressions.

A caller that explicitly acquires an immutable runtime map must also disable
its tile helper's default acquire for that lifetime. Otherwise every q tile
repeats the tensor-map invalidation and flush despite the outer acquire.
In 2x2 backward, loaders acquire compact maps at entry and physical maps per
work item; the fused Qdk map belongs to the store warp and needs its own acquire
on both CTAs. Preserve the initialized dead-unit fallback. The D192/D256 THD
origin/rebind suites and their empty replay cases exercise these boundaries.

**Rule S9 — Every scheduler reader needs a reuse back-edge; match output-phase cadence.**

Every scheduler payload reader must either return a reuse credit or have a
back-edge that gates the next publication on its observation. An idle role
that only reads a CLC response can miss a complete parity cycle, including
the terminal response. Remove that role's loop when it has no work. If this
also removes the only waiter on a credit barrier, stop arriving that object:
common dense CLC routes credits only to its leader, while persistent THD and
local predecode retain broadcast credits and their local scheduler roles.
Peer credit barriers then receive zero arrives and have no waiters. Credit
helpers require converged warps; retain their per-specialization lowering
choice when changing the target. Check scheduler and credit-helper selection
together at each caller; THD does not imply persistent scheduling in every
kernel family. Detector:
`test_sdpa_fwd_o_empty_backedge.py::test_scheduler_and_credit_target_select_the_same_layouts`.

A completed mbarrier phase needs a successful wait before arrivals in the
next phase, even when output ordering already protects the shared storage.
The D128 backward store warp bootstraps four dQ source-credit barriers per
CLC owner. Its native dV/dK epilogue waits on only slices 1/3, so explicitly
observe all four terminal phases before publishing those outputs, including
zero-trip owners. Keep dense and THD loops consistent; grouped kernels reuse
them. Run `test_sdpa_bwd_d128_terminal_credits.py` under Compute Sanitizer
synccheck as well as its ordinary FP64/canary and capture/replay checks:
correct numerical output alone does not detect this protocol violation.

Match barrier completion cadence to the consumer's loop. SM100 D512 MXFP8
SWA=128 drains two Dv slices per tile, so a once-per-tile softmax wait must use
the once-per-tile Q/O alias signal, including its explicit bootstrap phase,
not the twice-per-tile O-empty signal. Detector:
`test_sdpa_fwd_o_empty_backedge.py::test_d512_swa_waits_for_both_output_slices`
delays softmax after publishing retained P, then checks dense/THD O, LSE and
Amax after capture/replay. The delay lives in a private test template copy;
production kernels gain no debug knob or timing-dependent assertion.

**Rule S10 — Every warp role decodes a work item with the same scheduler inputs.**

- LPT-L2 requires the runtime GQA ratio and KV length at both initial and
  persistent decode calls, including the output-store warp. NATURAL/LPT and
  THD can hide missing arguments because they do not read those inputs.
- Packed tile heads and per-row query heads are different coordinates. Keep
  the next decoded tile head in the scheduler variable; derive each output
  head from it again for sinks, Stats and partial stores.
- A THD body using the device-bounded persistent claim counter must declare
  `THD_PERSISTENT = True` to the adapter. Otherwise the launch cap retains the
  declaration envelope and only the binder's runtime capacity tightens it,
  which can still add unnecessary worker waves. Assert the module route
  and declaration in changed-length replay tests; inspect the actual launch
  grid when comparing persistent implementations.
- Preserve the body's initial-wave policy through runtime binding. A broad
  shape-override declaration must not freeze a one-wave cap before the binder
  sees the current host-known Q capacity and batch. Larger runtime bounds still
  use the resident wave; no device-length read is needed. Check native/reference
  frames and small-to-large-to-small rebinding, then changed-length graph replay
  (`test_native_initial_waves_bind_current_work`).
- Packed runtime bounds use token rows per tile and packed head groups. Keep
  the body's initial-work budget with that geometry in immutable plan metadata;
  counting unpacked rows and all query heads can select an extra worker wave.
  Check both packing modes and partial groups against explicit native frames
  (`test_native_thd_initial_wave_geometry_rebinds`), including a broad declaration.
- `test_native_d128_packing_dense_replay` compiles all three schedulers and
  checks changed-input replay, split partials and MHA's bitwise packing fold.
  The ragged changed-length tests cover partial packing and retained captures.

**Rule S11 — THD dead units must not access uninitialized per-sequence maps.**

- A launch uses a declared envelope; live lengths can leave initial clusters
  with `batch_idx == n_batch`. Allocating `B+1` map slots makes that address
  in bounds but does not initialize a descriptor there.
- When setup writes maps only for the `B` real sequences, exit dead initial
  clusters before acquiring maps, allocating TMEM, or entering role-specific
  barrier protocols. Use the cluster-uniform
  `exit_if_dead_thd_cluster` helper; a role-local exit can strand barrier peers.
- Under programmatic dependent launch, wait for setup before the dead-unit
  check reads its metadata. Uniform CTA/cluster-local initialization may
  overlap setup if all roles finish it before the exit and it owns no TMEM
  or tensor maps. The SM107 paged physical-origin replay tests cover this order.
- Exercise overlaunch without total-length hints and replay after changing
  all query lengths to zero. Check O/Stats canaries and run memcheck; ordinary
  live-row numerical comparisons do not cover the dead-unit path.

**Rule S12 — Placement grows by measured table entries, not by new hand-written boxes.**

- `cudnn/sdpa/placement_table.py` answers first in both `fwd/placement.place` and `bwd/heuristics._leads`. Its entries are
  generated by `benchmark/placement_gen/gen.py` from a frozen sweep and qualified on a frozen holdout. Never edit
  `TABLE` by hand; regenerate it and cite the study in the entry.
- A new FROST-vs-backend boundary first tries a study: a class key read off graph facts and, per class, one monotone
  threshold on whichever `placement_table.FEATURES` scalar (cost proxy, parallel tiles) fits the training grid best,
  qualified on a frozen holdout that exercises every class it covers and is used once. Write a hand rule in `placement.py` only for what the table cannot express (a
  non-monotone class the generator reports, or a feature the class key excludes), and say why in the rule.
- A class the generator reports as NON-MONOTONE needs a finer key, not a hand-tuned region around its points.


**Rule S13 — Register redistribution must converge at warpgroup scope.**

All four warps in a warpgroup must execute the same `setmaxnreg` instruction.
Put a shared service-role register release before branching into individual
MMA, load, store and scheduler warps; calling the same helper separately from
those branches emits distinct instructions and violates the PTX contract.
An increase in one warpgroup may wait for registers released by another, so
keep the original register budget and ensure every releasing warp can reach it.

Detector: inspect generated PTX/SASS around the kernel's role dispatch. The
service warpgroup must reach one common `setmaxnreg.dec` / `USETMAXREG.DEALLOC`
site before splitting into roles. Separate sites reached by individual warps
are invalid even when their immediate register counts match. Repeated CUDA
Graph replay and sanitizer checks are also required; a single launch cannot
establish liveness. See the [PTX setmaxnreg contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg).


**Rule S14 — Empty work must retain the buffer-acquisition acknowledgment.**

An empty tile can skip memory transfers and arithmetic, but its output path
must still wait until the loader has observed the buffer's empty phase.
Otherwise zero-output completion can release the buffer before the loader's
initial wait, making that parity wait target the next cycle forever.
For D192 accumulation, publish zero-byte K/V full arrivals for empty tiles and
consume those phases on every tile before signaling output completion.

Detector: `test_d192_causal_empty_kv_tiles` repeatedly replays a causal graph
whose KV extent exceeds Q. Packed kernels also need declared-envelope tails
and live/empty transitions; a sanitizer pass or a single launch does not
establish liveness.


## 2x2 datapath (cta_group::2, M=128 collective = 64 rows per CTA) lessons

Kernels on this atom: forward `fwd/kernels/sm100/prefill_d512_f16_2x2.py` + `fwd/kernels/sm107/prefill_d512_f16_2x2.py`
(`TemplateParams.mma_2x2`), backward stage 2 `bwd/kernels/sm100/bprop_d512_f16_2x2.py` (`config_sm100.CfgBwdD512x2`,
`TemplateParams2x2`), backward d256 `bwd/kernels/bprop_d256_2x2_f16.py` (engine `sdpa_bwd_sm100_d256`). Each lesson names
its runnable detector (forward: `test/python/sdpa/frost/test_sdpa_fwd_d512_2x2_sm100.py`; backward:
`test_sdpa_bwd_config_sm100_2x2.py`, `test_sdpa_bwd_dsl_sm100.py`, `test_sdpa_bwd_thd_sm100.py`); add to the list, do not restate.

D256 packed backward schedules the **256-row write pair**, even though each
2x2 compute cluster covers only 128 KV rows. Run both halves of a tail pair:
stage 3 can read the masked half, so omitting it consumes stale dS after replay.
Keep physical per-port origins separate from compact workspace prefixes and
retain barrier arrivals for dead work while suppressing its output stores.
`test_sdpa_bwd_thd_d256_sm100.py::test_origins_rebind_and_replay` checks this
with poisoned scratch, one-sided empties and changed device metadata.

The loader must follow a lookahead consumer's operand order. In the D192
backward body, waiting for dO_dV(i)'s slot before loading Q(i+1) delays the
MMA's next score even with extra Q buffers. Preload Q0, then issue dO(i),
Q(i+1), dO_dV(i); guard the empty prologue and final prefetch and advance the
Q ring once per actual tile. The D192 dense/THD replay suites and D256
`test_mma_order_arms_match_the_fp64_reference` exercise both issue orders.

Shared protocol (both passes):

- **An end-of-kernel ring drain walks the WHOLE ring, never `min(total, stages)` steps.**  A consumer started at
  `PipelineState.start(phase=1)` passes its first `stages` waits free; at kernel end the releases still pending are those
  of the slots used LAST, but a `min(total, stages)`-step drain starts at the current index, i.e. at the slots used least
  recently -- with fewer issues than stages it waits untouched slots (free) and skips the used ones.  On the d512 2x2
  backward that was the TMEM lifetime: one kv tile at `STAGES_ACC = 2` left slot 0's compute release un-awaited and the
  MMA warp could `tcgen05.dealloc` while compute warps 1-3 still read (review P1 on #1323).  Walk `range_constexpr(STAGES)`
  from the current state: unused slots pass at their pre-armed parity, used slots wait their last release.  Detector:
  `test_sdpa_bwd_dsl_sm100.py::test_ring_drain_walks_every_used_slot` (a pure-Python twin of the protocol over every issue
  count; the old rule misses every `0 < total < stages`) plus the single-kv-tile GPU cells on both arch lines.
- **Do not pin a nondeterministic compiler output by its byte hash.** The SM100 D512 THD backward ACK body can
  produce different PTX from identical raw/clean MLIR under the same DSL build, even with a fixed Python hash seed.
  Use `test_stage2_2x2_sass_pins[sm_100a-thd]` for its structural lowering checks and retain THD numerical/capture
  tests. Archive both outputs when diagnosing this; do not retry until a chosen golden happens to match.
- **Every local barrier object needs a successful phase observation before it can be advanced again.**
  Matching arrival counts and a different wait instruction do not establish this ordering. In the D512 shared backward
  ring, multicast commits advance all four local empty objects. A passive follower could lag while the other CTAs recycle
  its slot, losing the expected parity (ABA). Keep its wait: deleting it leaves an unobserved object in the multicast.
  Gate both consumers on all four observations instead: each leader's full barrier counts its own expect-tx arrival and
  three remote ACKs, sent only after the sending CTA's successful local empty wait. Then both MMAs and the next empty
  release follow all four observations. Keep whole-ring drains and the KV_SHARE=1 protocol unchanged. Detectors:
  `test_sdpa_bwd_dsl_sm100.py::test_stage2_2x2_waits_for_delayed_empty_observer` and
  `test_sdpa_bwd_d512_sm107.py::test_chain_waits_for_delayed_empty_observer` delay a passive CTA at chunk 0 of the SECOND
  KV tile, so both four- and eight-stage rings reuse a slot. They check reference gradients and capture/replay in a
  watchdogged child. Check generated wait -> ACK ordering as well as source when changing this protocol.
- **Keep cross-pair polling separate from the phase-lifetime proof.** The historical October 1 backward experiment
  hung within 2-74 launches with sleeping/hint-less try-wait, while polling completed 200/200 and 300/300. Those finite
  observations did not prove a lost wake-up or a correct ring; polling alone does not fix the observer ordering above.
  Existing cross-pair polling remains in place (ring full/empty and scheduler payload); pair-local waits retain their
  current form. `test_sdpa_bwd_config_sm100_2x2.py::test_kernel_source_pins` pins the three ring poll sites.
  Poll shape is per kernel: backward 128 tight tests / 128 ns timer sleep, forward 32 / 128. A timer sleep is not
  `NANOSLEEP.SYNCS`; a tight poll can starve compute warps sharing the SMSP. The historical backward tight-poll experiment
  was 2.2x slower. Do not change polling as part of a lifecycle fix without separate attribution.
  Keep `test_stage2_2x2_survives_gpu_time_slicing` as a natural positive stress test; the old probabilistic must-hang
  `test_stage2_2x2_prefix_wait_form_hangs_under_time_slicing` is replaced by the delayed-observer regression above.
  Keep all diagnostic levers: `debug_heartbeat` + `debug_dump_addr` record barrier, stage, phase, kv_loop and chunk
  before each wait (`lane_d512_bprop/fix/hang_dbg.py` decodes them). `debug_wait_ms` changes the wait's shape, so failure
  to reproduce with it is not a waiver. Forward detectors remain `test_two_by_two_cross_pair_waits_poll`,
  `test_two_by_two_survives_gpu_time_slicing`, and `test_two_by_two_parking_wait_form_under_time_slicing`; forward hangs
  require their own diagnosis, not an inference from this backward fix.
- **A watchdogged detector and the negative control that deliberately WEDGES the GPU must never share a device at
  the same time: put BOTH in `@pytest.mark.xdist_group(name="gpu_exclusive")`, the repo's convention next to the
  `gpu_exclusive` marker (which xdist does not enforce by itself).** The CI FROST lane runs 16 workers over 4 GPUs
  with `--dist loadgroup`; items outside a group are dealt to idle workers in collection order, so two adjacent
  tests start together and share a GPU one time in four. The d512 backward detector then read its own control's
  45 s wedge as a hang: pipelines 71863093 (launch 22) and 71991279 (launch 35), the detector at ~63 s = compile
  + its 45 s budget with the control passing at 60-62 s, against ~25 s in the two green pipelines. Junit durations
  are the tell. The children print the launch-time history and compute processes on HANG, so
  a starved launch (ms ... ms, one 45 s wall, neighbours listed) reads differently from a wedged one. The coupling
  mechanism is NOT the obvious one: on a time-sliced B200 a kernel wedged for 180 s beside the shipped twin did not
  slow it at all (six 100-launch runs at 0.24 s median, `jobs/c9d07061/tmp/rebase/wedge/run_wedge.sh`, 2026-10-06), so
  the CI node shares its GPUs differently (concurrent contexts with the wedged clusters holding 136 of 148 SMs is
  the candidate). Grouping prevents these controls from overlapping on one worker; it does not reserve a GPU against
  unrelated workers or other jobs. A timeout with grouping enabled still needs diagnosis. The process query enumerates
  all GPUs: retain GPU UUIDs with the PIDs and the child's `CUDA_VISIBLE_DEVICES`, and correlate them with the runner's
  device map before attributing a neighbour to the test GPU. The list may include the reporting child; not every row is
  another process. A bare PID list is not a per-GPU process count, and a failed `nvidia-smi` query must be reported as
  unavailable rather than an empty device.
- **Two pairs sharing an operand ring by cross-pair TMA multicast need an
  `empty` barrier with init = number of PAIRS, released by EVERY pair leader's
  `tcgen05.commit` with the whole-cluster mask (0xF).** CTA c's multicast lands
  in CTA c ^ 2's slot too, so a stage is free only when both pairs' MMAs have
  read it; init 1 lets one pair's producer overwrite bytes the other pair's MMA
  is still reading -- silent corruption, not a hang. With `group = cta_2` the
  multicast's `complete_tx` is signalled on the DESTINATION's pair leader
  (probe `mcast_twin`, 2026-10-01), so ONLY pair leaders arm `expect_tx` (the
  bytes landing in the pair, from either issuer) and only the leader's MMA warp
  waits; a follower that arms or waits its own copy under `group = cta_2` hangs.
  Detectors: `test_sdpa_bwd_config_sm100_2x2.py::test_barrier_counts_and_cluster_span`
  (`RING_EMPTY_ARRIVERS == CGA_M // CTA_MMA`) and the bitwise twin
  `test_sdpa_bwd_dsl_sm100.py::test_stage2_2x2_is_bitwise_the_role_split`
  (a stale slot is a non-bitwise S / dS workspace).
- **Chunked K accumulation into one TMEM accumulator is bitwise equal to one
  long chain** (`accumulate=(c > 0)` over 8 x K=64 == one K=512; probe
  `ss_slabs` S3, two seeds). Make the twin test `torch.equal` on int16 views,
  not a tolerance: any difference is a real bug.

Forward (d512 prefill):

- **Every per-row reduction is a HALF-row value until combined with lane r + 64's.** The fp32 D of a 2SM M=128 MMA
  lands on TMEM lane `(m % 64) + 64 * (n // (N/2))`: warps w and w + 2 hold the same 64 rows, one column half each, so
  `row_max_reduction` / `ld.red.max` / a row sum on one lane is half the row. Exchange through SMEM + a named barrier
  before the online-softmax chain, and run the identical scalar chain on both halves so alpha / beta agree bitwise.
  Detector: `test_two_by_two_directed_numerics[skewed_halves]` (every odd 64-key half scaled by 2^10: the half with
  the smaller max exponentiates unnormalised without the exchange).
- **An SMEM slab that a TWIN CTA multicasts into is gated by every DESTINATION's consumer, not the issuer's.** Under
  `KV_SHARE=2` CTA c's TMA lands bytes in CTA c ^ 2 too; if that slab aliases something the destination still reads
  (here `sO` over the V ring), the issuer's own `*_empty` wait proves nothing about the destination. Make the gate
  pair-wide (`mb_o_empty` init `32 x KV_SHARE`, `arrive()` + `arrive_on_peer(cta ^ 2)` from the TMA-STG warp, and
  drain the last phase before exit so no remote arrive targets an exited CTA). Detector:
  `test_two_by_two_twin_alias_gate_under_pair_skew` (the test-only `DEBUG_STG_DELAY_US` lever holds pair 0's O store;
  RED on a per-CTA gate with the twin's V subtile over O subtiles 2, 3, 6, 7; GREEN on the pair-wide one). The natural
  race is frequency-bounded -- `test_two_by_two_persistent_multi_tile` passed on the racy protocol -- so a
  multi-tile cell alone is not a detector; the skew lever is.
- **`mb_o_full` init = the lanes of ONE column half (64), not the warpgroup (128).** Lane (r, h) drains d_v
  `[256h, +256)`, so an 8 KiB O subtile is published by the 64 lanes of half h only. Detector:
  `test_config_2x2_pins_and_ledger_formulas` (`O_CHUNK_ARRIVERS == CORR_LANES // 2`, raises on 128) +
  `test_kernel_source_arrive_sites_match_the_ledger` (arrive-site counts per barrier).
- **A fixed TMEM column is NOT protected by a ring it rides.** Per-ring-step payload goes in per-SLOT columns (alpha
  at `384 + s`, tile stats at `386 + 2s`): a fixed stats pair is protected only if the next writer waits the SAME
  slot's empty, and an EMPTY tile (one ring step) waits the other slot's -- the slow-arm correction then read
  `final_sum = 0` and published live rows DEAD (sm107 lane, 2026-10-01; reproduced on the SM100 body:
  `lane_d512_fprop/fix/stair_red_sm100.log`). Detector: `test_two_by_two_stats_ring_slot_race_empty_after_live`.
- **Dead / trimmed rows are zeroed by a SELECT, never by `o * beta` with beta = 0.** A trimmed row whose Q memory is
  NaN (poisoned padded tail) carries NaN through S, P and O; `NaN * 0 = NaN` reaches the output. Detector:
  `test_two_by_two_trimmed_rows_with_nan_inputs_store_zero`.
- **Cluster-shared K/V needs cluster-UNION tile bounds.** Two pairs sharing one K/V ring must iterate the identical
  KV range or the shared `k/v_empty` ring deadlocks; `make_sdpa_helpers(kv_shared_cluster=True)` derives the bounds
  over the cluster's 256 rows and the per-cell mask trims. Detector:
  `test_two_by_two_causal_512_cluster_union_bounds`.
- **Forward prepared hosts always use native binding, including a new kernel-template name.**
  The old name allowlist silently demoted a new d512 twin to Python while numerics still passed.
  That fallback and allowlist are removed: dense and THD builders construct their native binder
  unconditionally. A new host ABI must update the native binder and frame-parity tests in the
  same change. `test_two_by_two_host_slots_match_role_split` checks the d512 ABI;
  `test_two_by_two_default_plan_binds_natively` and `test_sdpa_native_prefill_binding.py` check
  default and alternate plans on device. `test_sdpa_binder_ownership.py` prevents restoring
  production Python forward framing or implicit backward ownership.

Backward (d512 stage 2):

- **The 2x2 D image puts the SAME rows in two warps: warp w holds rows
  `32 * (w % 2) ..+31` of kv-column half `w // 2`.** A map that swaps the two
  at the MASK site (`kv_col_base` picks the half) gives correct DENSE output --
  every random dense accept passes, no mask code is even traced -- and wrong
  MASKED columns. Detector: a COMPILED mask whose band edge is exactly column
  64 of a tile for every row, `test_sdpa_bwd_thd_sm100.py::test_graph_thd_kv_len_64_masks_the_upper_column_half`
  (THD -> `MASK_PADDED`; kv lengths 64 and 192 mask [64, 128) of a tile for
  both sequences), plus the causal / SWA / bottom-right accepts under the
  `stage2_datapath` fixture. Proven RED by swapping `col_half` at the
  `apply_mask_chunk` call (`lane_d512_bprop/fix/RED_lane_map_swap.log`). A
  dense `S_kv = 96` shape is NOT a detector: dense padding is not served, so
  it compiles `MASK_NONE` and its 32 tail columns are TMA zero-filled, not
  masked (that case stays as `test_dense_skv_96_zero_filled_tail`). Never
  trust a dense-only pass for a lane map.
- **A tcgen05 SMEM descriptor root AT or past 256 KiB needs `desc_version=1`
  on EVERY `SmemTile`, and the margin can be exactly zero.** The Rubin arm's
  last root (`sRingV` stage 7 at 253952) ends at byte 262143 only because the
  TMA-store-only cast slabs are declared after the rings; one more stage flips
  the whole module to version 1. Compute the version from the slab list
  (`desc_version_2x2`), bind it once as a module constant, and pin the slab
  ORDER -- a version-0 descriptor past the window silently wraps to an
  untouched buffer (an exactly-zero accumulator). Detectors:
  `test_descriptor_roots_and_version_per_arm`, `test_kernel_source_pins`
  (every `SmemTile(` takes `desc_version=DESC_VERSION`).
- **A per-arch sibling fork of a 2x2 body is pinned to its parent by a
  CODE-DIFF ALLOWLIST, by import-time refusal of the parent's arm, and by
  rendering identity -- not by prose.** The cc 10.7 d512 stage-2 fork
  (`bwd/kernels/sm107/bprop_d512_f16_2x2.py`, parent `sm100/`) differs from its
  parent in a handful of code lines (the ring-arm default record, the
  hard-coded `DESC_VERSION` with its `_require`d derivation and zero-margin
  fact, `SPIN_RING_WAITS` threaded into the one pair-local wait branch);
  `test_sdpa_bwd_d512_sm107.py::test_fork_code_diff_is_only_the_listed_deltas`
  blanks strings and comments and asserts every `+` / `-` line matches a listed
  regex, so a fix landed in ONE sibling fails the test. The fork refuses the
  parent's arm (and a 9th ring stage) at import
  (`test_fork_refuses_the_sm100_arm_and_a_ninth_stage`, RED-first: both raise),
  and at the same parameters its sm_107a PTX md5 EQUALS the parent's
  (`renderings/md5_stage2_2x2_sm107a.txt`; `wait(spin=False)` and a literal
  `DESC_VERSION = 0` render identically), which is what makes the on-board
  `test_fork_is_bitwise_the_sm100_body_at_rubin_params` (`torch.equal` on int16
  views of dQ / dK / dV and the S / dS workspace) a guarantee rather than a hope.
  Fork AFTER the parent has run at the new arm (the d512 case: 14 / 14 direct
  adapter cases on the board first), so every delta is deliberate.
- **A directly constructed adapter must spell the analyzer's mask
  conventions; one of them is `window_size_left = diagonal_band_left_bound - 1`**
  (`graph_analyzer.py`, the `left_bound - 1` line). A hand-built
  `SdpaBwdDsl*(window_size_left=L)` checked against `_causal_keep(left=L)` is
  one column off: it reads as cos 0.9996 / max-rel 2-3e-2 on dQ / dK -- inside a
  5e-2 / 3e-2 `allclose`, outside the suites' 0.9999 / 2e-2 gate -- and looks like
  a kernel bug on a new arch (the cc 10.7 d512 bring-up's only "failure":
  9 / 10 direct-adapter cases passed on the board on 2026-10-01 and the SWA case
  read cos 0.9996 until the convention was applied). Detector: run the same case
  through the graph API with the engine pinned (`test_sliding_window`); a graph
  pass with a direct-adapter fail is the convention, not the kernel.
- **An omitted graph `attn_scale` is 1.0; an adapter's `scale_softmax=None` is 1/sqrt(d).** The backend inserts no
  scale multiply when `attn_scale` is absent, and the analyzer turns absence into `facts.scale = 1.0`, so None never
  reaches an adapter from a graph. A graph test that omits the scale needs a 1.0 oracle. When it passed None through,
  FROST computed 1/sqrt(d) on the same graph: O differed by ~4 and Stats by 48-100 from backend engines 8/10 on B200.
  Detector: `test_omitted_attn_scale_is_no_scaling` (analyzer) and the `test_default_attn_scale` cases.
- **The `LDTM` SASS count of the masked (three-range) compute body is a
  TOOLCHAIN fact, not a kernel fact.** The public DSL 4.7.0 + CUDA 13.3 ptxas
  keeps three traced range bodies (6 `LDTM`, the SM100 twin's pin); the board's
  internal DSL 0.3.0 + cuda-39029786 emits two for the same source (4). Pin the
  per-body count the kernel owes (2 per traced body) against the set a toolchain
  can produce (`test_fork_sass_pins_sm_107a`: `LDTM in (4, 6)` on the causal arm),
  never `2 * n_bodies` alone -- and record the toolchain next to the pin.
- **cuDNN 9.26 builds NO d > 256 backward plan on cc 10.7** (engines 17 and 7
  decline `d_qk > 128` outside the 192x128 and 256x256 cases; probed 2026-10-01
  with the FROST opt-in OFF on cuDNN 9.26.0.51 at B1 H8 S2048 dense and causal:
  `create_execution_plans` raised `cudnnGraphNotSupportedError` both times), so a
  Rubin-line d512 backward has no FROST-vs-backend A/B -- its perf gate is
  floor-relative (stage-2 MMA floor 2502 clk per CTA-kv-tile at the board's
  39.1-clk 2SM M=128 N=128 K=16 rate, stage 3 at 4096 MAC/clk/SM). Probe before
  planning an A/B: `g.create_execution_plans` raises `cudnnGraphNotSupportedError`
  with the backend's decline reasons when no engine proposes a plan.
- **Attribute a floor gap with in-kernel clocks before touching a wait, and
  quote cycles, not nominal-clock wall time.** Both SM100 d512 stage-2
  kernels carry a default-off attribution lever (`TemplateParams2x2.debug_clk`
  on the twin, the sibling record `TemplateParamsDbg.debug_clk` on the 4x1):
  every mbarrier wait keeps its production form and is bracketed by
  `%clock64`, the elected lane accumulates per barrier id in the warp's SMEM
  slice, and each warp dumps 32 x Int64 (waits, blocked counts, issue
  segments, tiles, and the body's `%globaltimer` span) to a host-pinned buffer
  at exit. Measured 2026-10-01 (B200, CLEAN slots): the two datapaths wait on
  DIFFERENT things -- the 4x1's sg0 compute warps are busy ~2850 clk/tile and
  wait ~1200 on `smem_empty` (a 32 KiB TMA store drains in ~2600 clk), its
  MMA idles 3100 on `acc_empty`; the twin's MMA waits 2100-2650 on
  `ring_full` (K/V chunk latency, ~800 of it the cross-pair lock-step:
  `kv_share 1` measured -6.9 / -10.0 / -9.6 % of stage-2 wall time on dense
  8K / dense 2K / causal 8K, bitwise -- ONCE per cell as the B arms of an
  A/B/A series in one slot whose four control arms spread +-1.7 / 0.2 / 0.8 %,
  so n = 1 per cell, three cells agreeing) while `acc_empty` and
  `smem_empty` never block (`stages_acc 4` and a two-stage cast are no-ops
  there) -- a lever that is right for one kernel is a no-op on the other.
  The body's clk / ns ratio is the SM clock the kernel ACTUALLY ran at:
  705-820 MHz on sustained S >= 8K backwards while NVML sampled 1155 MHz (the
  600 W power cap gates cycles; the PLL readout does not move), so a CUPTI
  "clk per tile at 1155 MHz" overstates cycles by up to 1.6x and the
  S_kv-scaling of such numbers (5400 -> 6900 -> 7600 clk/tile at 2K / 8K /
  16K) is the throttle, not kernel work (the bodies are 4860 / 4700 / 4660
  cycles). Detectors: the default renderings are PTX-identical with the lever
  off (`test_stage2_default_rendering_ptx_md5_is_unchanged`,
  `test_stage2_2x2_default_rendering_ptx_md5_is_unchanged`), the armed lever
  renders different PTX (`test_stage2_{2x2,4x1}_debug_clk_lever_renders_code`),
  and `test_stage2_{2x2,4x1}_debug_clk_dump_accounts_the_waits` checks every
  role accumulates exactly the barriers it waits on with bitwise outputs and
  reads the record layout (the module's `DBG_CLK_*` constants: slot 0 is the
  body's clk, slot `DBG_CLK_BODY_NS` -- 28 on the twin, 31 on the 4x1 -- its
  %globaltimer span, so `slot0 / slot28|31` is the SM clock in GHz the body
  ran at; a decoder is a few lines over those constants and the per-warp
  `(linear block * 8 + warp) * 32` record offset). The record index is not
  bounds-checked: the 16 MiB dump holds 8192 CTAs per launch, so assert
  `grid * 8 * 32 <= DBG_CLK_MAX_WORDS` before arming (both accounting tests
  do). NCU `--clock-control base` (688 MHz on that box) is the other way to
  hold the clock -- never compare two wall times without one of the two, and
  never compare two ARMS run in a fixed order on a power-capped board without
  an in-run thermal control: the matrix's `4x1, twin` order timed the
  IDENTICAL stage-3 GEMMs 10-25 % slower in the second arm at S >= 8K.

## Stage-3 GEMM (`bwd/kernels/bprop_matmul_blackwell.py`) perf-method lessons

- **A head subgroup is legal only when the final reduction covers all global
  per-query-head partials.** The dense BF16 SM100 d256 chain can divide one GQA
  group after whole-group choices exceed its workspace target. Every chunk must
  divide the group, its grouped dQ tensor map must retain one K head (ceiling
  division, never a zero-sized floor), and the global dK/dV fold must run once
  after all chunks. Keep other chains' validation strict and gate inherited
  chooser rules by dtype, layout and compute capability. Detectors in
  `test_sdpa_bwd_d256_sm100.py`: `test_head_subgroups_match_full_group_gradients`,
  `test_head_subgroups_preserve_mask_and_tail_gradients`,
  `test_head_subgroups_rebind_stream_replay_and_launch_count` and
  `test_head_subgroup_rule_is_scoped_and_workspace_is_honest`. They check
  old/new bitwise gradients against an independent oracle, poisoned scratch,
  stream/storage replay, exactly one fold and unchanged fitting choices.
  Keep the stage-3 source pins consistent with ceiling division while retaining
  the `b_head_group == 1` constant-fold checks. The SM107 suite's host-only
  `test_stage3_b_descriptor_covers_partial_head_groups` evaluates the actual
  template expression against the K heads reached by every Q head; restoring
  floor division must fail for a subgroup smaller than one complete GQA group.

Measured on the SM100 d512 backward's dV / dK / dQ GEMMs (B200, 148 SMs, 1155 MHz SW power cap, cuDNN 9.26.0.51, DSL 4.7.0,
2026-10-01; the numbers quoted here are the per-GEMM CUPTI medians of CLEAN in-process round-robin A/B slots, and the constant's
comment in `bwd/api_dsl.py` carries the full set).  Each lesson names its runnable detector; add to the list, do not restate.

- **Read "residual vs the MMA floor" off the CLOCK RATIO, not off byte counters.**  `ncu --clock-control base` locks the SM
  clock to base (684 MHz here) and leaves DRAM / L2 at speed, so a kernel that sits at its MMA floor under NCU but takes 1.3x the
  cycles at the full clock is memory-LATENCY bound (the memory side is 1.69x slower relative to the SMs at 1155 MHz); one whose
  cycle count is clock-invariant is issue / occupancy bound.  The (512,512) row: 1.09 M cycles per S8K launch under NCU = 96.5 %
  tensor duty on its 136 active SMs, 1.41 M cycles at 1155 MHz (CUPTI us x NVML MHz) = the whole "1.35x unattributed residual";
  S2K: 0.595 M vs 0.609 M (1.02x) -- the "S-dependent 14 %" IS this term.  Bytes per FLOP and DRAM % of peak are flat across S,
  so `dram__bytes_read` alone says "nothing to see".  Detector: `ncu --clock-control base --metrics
  sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__ctas_launched.sum -k regex:_bprop_matmul_bh_sm100_kernel --launch-skip <two
  backwards' worth of GEMM launches> --launch-count 3` on the chain, then `elapsed(full) / elapsed(base)` with
  `elapsed(full) = CUPTI median us x NVML MHz`.
- **Residency and idle SMs under CLC are direct metrics.**  `launch__cluster_max_active` (Occupancy section) is the co-resident
  cluster count at the launch's SMEM (34 four-CTA clusters at 231 KiB/CTA, 74 two-CTA), `sm__ctas_launched.sum` is how many CTAs
  CLC actually launched to serve the grid (136 = 34 x 4), and `sm__ctas_launched.min = 0` means SMs sat idle for the whole launch
  (12 of 148 on the 2x2 row; 0 on the 2x1 row).  `launch__waves_per_multiprocessor` does not account for clusters (6.92 for a
  7.53 -> 8-round launch); compute rounds as `tiles / cluster_max_active`.
- **A structural gain measured at base clock need not survive the full clock.**  The (512,256) row (cluster 2x1, no A multicast,
  A read twice from DRAM) is -12 % elapsed vs (512,512) under NCU at 684 MHz and +0.1..+0.8 % at 1155 MHz at dense S8K (three
  CLEAN in-process A/Bs; +7.5 % at causal S8K), while at S2K / S4K -- where the launch is MMA-bound at full clock -- the same
  -11.5 / -11.6 % (dense) and -6.4 % (causal S2K) appear in every round.  Always close with the full-clock A/B (`bwd_bench.py --ab`
  in the lane: every arm built once, timed round-robin, bitwise-checked).  Outcome: the SM100 d512 chain keys the row on the
  padded max(S_q, S_kv) and the device cc (`api_dsl._sm100_stage3_cgrp_tile_mn`, (512,256) for BSHD at <= 4096 on cc 10.0..10.6);
  detectors `test_sdpa_bwd_dsl_sm100.py::test_stage3_cluster_tile_rule_by_sequence_length`,
  `::test_stage3_small_s_tile_is_bitwise_the_wide_row` (a spy pins which row each arm loaded; the whole causal family and the
  4096 / 4224 boundary as served) and `::test_stage3_tile_rule_reads_the_device_cc`.  Behind a (512,256) GEMM at S8K the
  UNCHANGED stage-2 kernel ran +13-15 % slower in three CLEAN slots (wall == CUPTI sum; per-arm clock mean within 1 % on the
  dense telemetry slot) -- a chain-level effect no per-kernel metric predicted; measure the whole chain, not the kernel.
- **A plan-time rule measured on one board needs a compute-capability term, because adapters INHERIT `compile`.**  The cc 10.7
  d512 row's adapter subclasses `SdpaBwdDslSm100` and overrides `check_support` and the stage-2 file / record only, so a tile
  rule keyed on S alone in `SdpaBwdDslSm100.compile` would have flipped that row's default on a board where it was never rendered
  or run (review of the (512,256) rule).  Resolve the cc once per compile through one seam (`SdpaBwdDslSm100._device_cc`, the
  prepared host's own `compute_capability(resolve_device(q.device))`) and gate on an explicit inclusive range
  (`_SM100_STAGE3_SMALL_S_CC = (100, 106)`).  Detectors: the faked-cc host pin in the UNGATED suite
  (the cc 10.7 backward suite's `::test_stage3_tile_rule_keeps_the_wide_row_off_the_sm100_line` -- the SM100 suite is module-gated
  to SM 10.0..10.6 and never runs on another lane) and the GPU pin that fakes the seam on the board it does run on
  (`test_sdpa_bwd_dsl_sm100.py::test_stage3_tile_rule_reads_the_device_cc`: `_device_cc` -> (10, 7) must load (512,512)).
  Grep tripwire for the next rule: `grep -n "compute_capability\|_device_cc" python/cudnn/sdpa/bwd/api_dsl.py` --
  every plan-time branch on the device must go through the seam.
- **Two refuted memory-side levers, so nobody re-tries them blind:** (1) a non-power-of-two S / dS workspace row stride
  (`_skv_pad + 64`, 16 KiB -> 16.1 KiB) is +5 % SLOWER on stage 3 at dense S8K and neutral elsewhere; (2) TMA L2 promotion
  (`l2_128b` / `l2_256b`) on the A / B operand descriptors is +1-2 % slower.  Both diffs are kept as patches in the lane dir
  (`kv_stride_pad_lever.patch`, `tma_l2_promotion_lever.patch`) with their detectors; a descriptor hint does not change the kernel
  PTX (the md5 record stays valid), a carve change does not change the templates either.
- **Tile-row twins are `torch.equal`, and a bench that draws its own dO per build is not a twin test.**  The (512,256) row is
  bitwise the (512,512) row (same per-pair work and k walk; the 2x2 row only multicasts A to a second pair), but
  `bench_baselines.build_bwd` draws a fresh random dO per call, so an unseeded second build "proved" the rows differ by 1e-4.
  Re-seed before every build you compare.
- **A pin whose record is local-only has never run.**  The stage-3 PTX md5 pin read `frost_dev/.../md5_develop_sm100a.txt`, absent
  in every checkout -> it skipped everywhere until the record was committed under `test/python/sdpa/frost/renderings/` with a
  `dsl=` line (`md5_stage2_4x1_sm100a.txt`, `md5_stage3_sm100a.txt`); `test_stage3_md5_record_is_committed_and_complete` is the
  tripwire, `ab_stages` 4 -> 3 on the (512,512) row the RED proof.  And a completeness test that compares the record against a
  HAND-KEPT dict pins whatever the hand remembered: the tile rule made eight renderings default renderings and two were pinned.
  Derive the expected set from the rule (`_stage3_default_tiles` walks `_sm100_stage3_cgrp_tile_mn` over S x cc per record) so
  a rule that grows a row demands its renderings.
- **A hand-set `FROST_SOURCE_DIGEST` must name the record, not just the tag.**  `compiled_cache.template_key` keys a
  `compile()` on the module digest + the call's plain arguments ONLY; the mask, dtype and every other `TemplateParams` field
  live in the module.  A test that execs a template itself (`spec_from_file_location` + `FROST_TEMPLATE_PARAMS`) and sets a
  per-tag digest hands the FIRST load's artifact to every later load of that tag: the causal cells of
  `test_sdpa_fwd_d512_2x2_sm100.py` (multi-tile, pair-skew) ran the DENSE kernel under a causal reference -- 4096 / 4096
  cells wrong, LSE ~= log(S_kv), deterministic, and order-dependent under xdist (green when the causal cell compiled first).
  Fold `sha1(repr(params))` into the digest (`_params_digest`) or leave it unset (uncacheable).  Detector: run the dense and
  the causal cell of one tag in one process with an empty `XDG_CACHE_HOME`; the second must not inherit the first's LSE.

## Output initialization regressions

D256 chunk-local GQA partial buffers may be reused only after a complete
GQA group has been folded. Input batch/head origins remain global while the
dV store and dK GEMM use independent local output origins. A head subgroup
smaller than the GQA group still needs global partials until the final fold.
`test_chunk_partials_workspace_and_subgroup_boundary` checks the allocation
boundary; `test_chunk_partials_match_full_partials` checks batch, head and
combined chunks against FP64 and the original fold, with poisoned workspace
under dense, causal, bottom-right and sliding-window masks.
`test_chunk_partials_preserve_per_member_dq` also retains the per-member dQ path.
`test_chunk_partials_rebind_stream_and_replay` checks changed inputs and the
caller's stream under capture.
When a test monkeypatches the scratch policy, capture its shapes and byte
count inside that context: these queries call the current policy dynamically,
so restoring the class before reading the baseline silently changes the control.

When removing wrapper-side output clears, verify that the prepared chain
overwrites every element, including masked rows and partial tiles. Poison
fresh auxiliary outputs with NaNs, forbid the removed Torch clear calls, and
replay after previously active rows become fully masked. The detector is
`test_wrapper_aux_outputs_need_no_torch_clear` for SM80 backward dBias/dSink.


## Adapter declines that other suites pin

`gated_attention_block` runs the SDPA forward adapter as its attention stage, and
its tests pin the adapter's declines by message (`match="multiple of 128"`).
Those tests run only on the `oss` CI lanes (`python/*/cutedsl`, SM107). A change
that serves a previously declined shape needs `@cudnn-ci-bot run ...,oss`, not
only `frost`. #1520 (the KV-tail mask) turned 15 GAB decline pins red because its
CI ran `python_tests,frost` (#1522). Detector: `git grep -n "match=" test/python/gated_attention_block`
for the old message.


## Prepared THD launch bounds and setup

Ragged-offset integer widths are compiled pointer-ABI facts. Include the whole
per-port dtype tuple in the prepared artifact key, alongside origin strides and
multipliers. A valid second plan can otherwise read INT64 offsets as INT32 (or
the reverse) from an earlier cached host. The D256 detector
`test_origin_integer_types_separate_compiled_artifacts` checks both transitions
for Q and Stats, distinct artifacts, disk reload with compilation forbidden,
and numerical output after replay. Check artifact identity before launching a
known-colliding negative control.

Each compiled host contract has one production binder across graph, standalone,
and staged entry points. Forward and half backward use native binding. Quantized
backward and standalone compact-GQA backward remain Python until migrated as a
whole. Backward builders must explicitly choose ownership; omission fails at plan
construction. A native validation error is never a reason to retry in Python.
Legacy forward framing exists only in test-only reference modules for differential
checks; production must not import them. Pure geometry helpers still used by native
binding stay in production.

Staged forward adapters bind their compact core through the same native binder
as direct calls. Validate the complete core before any gather, scalar seed, or
output write; keep alias checks against original carriers as well as workspace
replacements. The staged half/FP8/MXFP8/SM120 tests forbid Python core binding
while checking current storage, split outputs and replay. Include legal head-dim
envelopes: native template ownership must not silently disappear at non-exact
head dims. `test_half_staged_has_no_execute_allocations_and_replays_current_storage`
covers those dimensions and the split path.

A cached graph envelope does not describe the current packed allocation.
Bound its launch using host-known token capacity and effective batch count,
without reading device lengths or changing the compiled artifact. Replay may
change the device lengths within that capacity; test the old capture after
replanning as well as freshly bound calls.

Parallel descriptor setup must fence on every writer that publishes a
tensor map. Keep prefix construction, remapping, and live-count publication
ordered by CTA barriers. Check prefix lengths around warp boundaries and
zero-length sequences (`test_parallel_thd_metadata_matches_lengths_and_normalized_cu`),
and run racecheck/memcheck before changing this shared setup again.

The shared THD unit decode (`_common_blackwell._thd_decode`,
`tile_dsl.thd.thd_decode_unit_prefix`) binary-searches the remap-ordered work
prefix for EVERY THD flavor, forward and backward, so three sites must agree
or units decode garbage from an unwritten table: the setup launch publishes
the prefix along the blocked axis (`write_thd_work_prefix`: Q rows by default,
`cu0 = 2B+1` for kv-block units, `prefix0 = THD_BWD_WORK_PREFIX_OFF` in the
backward layout, whose forward slot holds `row_off`), the host reserves
`THD_WORK_META_WORDS` / `THD_BWD_META_WORDS` words, and the decode is handed
the same `cu0` / `prefix0`. A kernel that derives its batch count from the
metadata tensor's length (the SM120 prefill family) must do it through
`THD_BATCH_OF_WORK_META_WORDS`: the old `(n - 4) // 4` read B = 5 for B = 4 on
the 5B+5-word tensor and faulted only once units outnumbered CTAs
(`test_dsl_sm120_thd_multi_unit_per_cta`). Shared decoders still see configs
without family-specific fields (the D512 two-by-two config has no `PAGED_KV`;
D256 decode has no `THD_VARLEN`), so read those through defaults, and keep the
D512 `test_two_by_two_thd` compile-and-execute test green. Detectors:
`test_thd_decode_unit_prefix_matches_walk` (prefix == walk unit for unit, B 1
to 257, zero-length sequences, reversed rows), `test_parallel_bwd_setup_prefixes`
(asserts the whole backward buffer, prefix words included) and
`test_thd_work_prefix_replay_matches_live_tiles`. Two DSL traps met on the way:
inside a `@cute.kernel` that already has `Constexpr` parameters, hand a
`@cute.jit` helper's `Constexpr` argument through a kernel `Constexpr` parameter
rather than a literal (the fp8/mxfp8 setup kernel failed with `requires a code
object with 0 free vars, not 2` while its f16 twin passing `splits` compiled);
and inside a `@cute.jit` helper never rebind a None-defaulted parameter to a
staged value within a dynamic region (`TYPE_UNSTABLE_JOIN` at the region join)
-- assign a fresh local (`cuq0 = n_batch if const_expr(cu0 is None) else cu0`)
and re-run the callers that use the default whenever a shared helper changes,
since callers passing the value explicitly compile fine.

Standalone scratch queries after `check_support()` must not depend on a
compiled module. Reserve metadata from checked plan geometry and keep the
size stable through compilation. `test_thd_split_workspace_query_before_compile`
allocates first, then compiles and executes with that buffer, including a
batch size that crosses the metadata alignment boundary.

**One metadata buffer, several persistent-scheduler launches per execute (the
SM100 d512 backward's `heads // chunk` stage-2 launches): the two scheduler
words are per LAUNCH.** `live` (`meta[4B+2]`) counts the units ONE launch
decodes -- `sum_b ceil(s_b / 256) * chunk`, the head extent that launch's kernel
receives as `n_qh` -- never the plan's head count; and the claim counter
(`meta[4B+3]`) must be re-seeded to the cluster count before EVERY launch,
because a launch leaves it at `live + clusters` and an unreset second launch
finds every claim past the bound, computes only the units its clusters were
pre-assigned by blockIdx, and stage 3 reads the previous chunk's S/dS for the
rest (wrong, finite gradients -- cos 0.50 on dQ/dK of 7 of 9 sequences at
d4b024671, 2026-10-01). The reset rides in the clamp kernel that already
precedes each stage-2 launch (`_clamp_thd_input_descs`, both stage-2 files),
so the fix added no launch (Rule 2). The miss is observable only when a launch
has MORE live units than clusters (SMs // 4 = 37 on a 148-SM part), which is why
every one-chunk THD test (h <= 4) passed. Detectors
(`test_sdpa_bwd_thd_sm100.py`):
`test_graph_thd_forced_head_chunks_match_reference_and_unchunked` (forces
`_sm100_head_chunk_thd` to 8 / 4 over 12 q-units; per-sequence fp64 and
bitwise the one-chunk plan) and
`test_graph_thd_forced_head_chunks_publish_live_and_reseed_the_claim_counter`
(reads `meta[4B+2] == q_units * chunk` and `meta[4B+3] == live + clusters`
back out of the executed workspace). A new multi-launch consumer of
`write_thd_live_and_ctr` owes the same two pins.

**The blocked workspace's padding tips the head-chunk divisor at the budget
edge.**  Each sequence's block is padded to 128 rows, so an equal-length packed
plan carries `B * 128` more rows per head than the dense plan of the same
lengths; charged in full that halved the chunk whenever the dense chunk sat
exactly on the 4 GiB budget (B=1 / B=4 S=8192: 16 -> 8 / 4 -> 2 heads per
launch, twice the stage-2 and stage-3 launches for the same work).  The rule
(`api_dsl._sm100_head_chunk_thd(..., t_rows=)`) charges the budget on the TOKEN
rows and lets the padded slab overshoot by at most `budget //
_SM100_WS_THD_PAD_SLACK` (1/8); `scratch_workspace_bytes` is still computed
from the chosen chunk (Rule 8).  Detectors (`test_sdpa_bwd_thd_sm100.py`):
`test_thd_head_chunk_matches_dense_at_equal_tile_multiple_lengths` (THD chunk
== dense chunk at six equal-length shapes, the original rule's halved answers
pinned, the many-short-sequences cap) and
`test_graph_thd_launch_count_is_the_promised_chain` (launches per execute from
a CUPTI trace == `setup + dot [+ zero] + chunks * (clamp + stage 2 + (2 +
group) * (patch + GEMM)) [+ dkv_reduce]`, per-chunk helpers counted by name --
the Rule 1-2 pin for this chain).

Compiled-host cache key: `prepared_sm100.compile_plan` keys the host artifact
on the stage-2 / stage-3 TEMPLATE digests plus `Params`, geometry, regions and
`sm` -- an edit confined to `kernels/sm100/prepared_host.py`,
`kernels/thd_helpers.py` or `kernels/bprop_chain_common.py` does not move the
key and a warm `CUDNN_FRONTEND_COMPILED_CACHE` serves the OLD host. When
testing such an edit, run with `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` or
confirm the key moved (the detector pair above fails on the stale host).

**A host lever that changes the launch shape rides in `Params` so the key
moves with it.** The SM100 d512 dQ GEMM under GQA is ONE launch per head chunk
(`api_dsl.DQ_SINGLE_LAUNCH`; the stage-3 record's `b_head_group = group`, the
cc 10.7 d256 chain's #1318 arm) where it used to be one per group MEMBER;
`prepared_host.Params.dq_b_head_group` (14th, appended) carries the rendered
record's value into both the compile key and the traced host, so the per-member
twin (`DQ_SINGLE_LAUNCH = False`) can never be served the shipped artifact, and
`_dq_launches` refuses a value that is neither 1 nor the group. Detectors
(`test_sdpa_bwd_dsl_sm100.py`):
`test_stage3_dq_single_launch_per_chunk_is_bitwise_the_per_member_launches`
(record spy + CUPTI launch count `3 * chunks` vs `(2 + group) * chunks` + int16
bitwise dQ / dK / dV vs the twin + fp32 oracle) and
`test_stage3_dq_launches_pair_every_q_head_with_its_k_head` (the host's
coordinate arithmetic in plain Python). Two traps the detector hit: the graph's
`_compiled_plans[i]` is the `_FrostSdpaBwdPlan` wrapper and never exposes the
adapter, so a test that needs the adapter's facts (`_qh_chunk`,
`_dq_b_head_group`) captures `self` off a monkeypatched
`SdpaBwdDslSm100.compile`; and a bare `assert ..., facts_dict` is truncated by
pytest, so a launch-count pin must spell the measured and expected counts.

**The THD causal K-trim is the dense arithmetic per sequence; keep a host
twin of the range arithmetic, because its empty-range clamps hide off-by-ones
no kernel test localizes.** `_thd_causal_k_range` works in sequence-relative
rows with the sequence's own k count, so `api_dsl.THD_STAGE3_TRIM` renders the
SM100 chain's packed stage 3 trimmed with the diagonal edge
(`thd_causal_bottom_right` makes the kernel read the bottom-right
`S_kv[b] - S_q[b]` per group); the window edge and the right-band widening stay
untrimmed on that chain (the THD arm takes no constant shift).
The range may be EMPTY (`nkt == 0`, a one-sided empty sequence; a tile with no
kept cell) and the kernel stores zeros for it -- a never-empty floor there is
wrong, and `k_lo = min(k_lo, k_hi)` after a `nkt - 1` floor is not enough: the
first port of this trim produced `range(-1, 0)`, one iteration at a negative
k tile, which
`test_sdpa_bwd_thd_sm100.py::test_thd_trim_covers_every_cell_stage2_writes_per_sequence`
(`_thd_k_range_twin`, pure Python over random packed lengths, no GPU) caught
before any kernel ran. Runtime detectors: `test_graph_thd_causal_stage3_is_trimmed_per_sequence`
(record spy; RED on the untrimmed tree),
`test_graph_thd_causal_trim_is_bitwise_the_untrimmed_rendering`
(`THD_STAGE3_TRIM = False` twin, int16 views, both arms on the fp64 reference)
and `test_graph_thd_causal_matches_dense_bits_at_equal_tile_multiple_lengths`
(THD vs the dense BSHD graph over the same memory). A template field that must
leave every existing rendering byte-identical is proven by the PTX md5 list
`test/python/sdpa/frost/renderings/md5_stage3_sm100a.txt` through its consuming
test (the probe is a host trace-compile for sm_100a, but the test harness
still needs a CUDA device: the conftest asserts one and opens a cuDNN handle,
and the env wrapper re-fills an empty `CUDA_VISIBLE_DEVICES` -- so run it through
the GPU slot like any other test) -- run it twice from independent processes
before pinning a new line.

## Heuristic geometry regressions

When changing tile, packing, CGA or split candidates, spy on the chooser's
inputs for both split and unsplit legs: physical CTA count can differ from
public MMA width, and masked KV work depends on the candidate Q span and tile
alignment. Compare masked bounds with an independent visible-key oracle and
verify every alternative is rescored, deduplicated and within the candidate
cap. An exact winning-rank golden alone does not detect stale model inputs.

The SM107 resident-bias arm's (n, n+1) pair item has no public chunk-length
knob until the frontend-band enum member lands in both pybind extensions;
`config_sm107._N_CHUNK_OVERRIDE_FOR_SWEEPS` is a private, process-local TEST
seam read by `sm107_n_chunk_pick` and, through `sm107_pair_item_active`, by
the engine row's mismatch, the heuristics and the adapter -- a set value
changes the plan the graph builds and its eligibility. Set it only through
`monkeypatch.setattr` in a test or from a sweep driver that resets it to
`None`; never from a plan path, a module import or a config. Pin behaviour on
the pick's output (`test_n_chunk_override_seam_is_read_by_the_pick`), not on
the seam's name.  The bias FORM and the streamed ring's depth have the same
shape of seam: `config_sm107._BIAS_MODE_OVERRIDE_FOR_SWEEPS` (1 = the resident
block, declining above its budget; 2 = the streamed ring wherever it fits,
including where the block would also fit -- the A/B's two arms) and
`_BIAS_STAGES_OVERRIDE_FOR_SWEEPS` (2 / 3), read by `sm107_bias_mode_pick` /
`sm107_bias_stages_pick` and so by the engine row, the heuristics' width pick
and the adapter; the same rules apply (pinned by
`test_seams_pin_the_form_and_the_depth`).

Measured facts the small-tile bias heuristics must honour (bf16, D = 32,
S_q = S_kv = 384, H = 4, 384 flattened batch rows; a 212-SM cc 10.7 node at the
2376 MHz lock, 100 x 3 graph-replay rounds, control pair within 0.2 %):

- A graph with BOTH a bias and a bool `kv_mask` is superadditive on the
  128-wide tile: plain 46.3 us, bias +10.0, `kv_mask` +20.0, both +34.7 (the
  sum + 15 %; the one-batch-entry item + 21 %). The mechanism is the
  instruction cache: the combined softmax body is ~3480 hot SASS instructions
  against ~3050 for either feature alone, and the warp-stall ratio
  `smsp__average_warps_issue_stalled_no_instruction_per_issue_active` jumps
  0.5 -> 2.0 while every other reason stays put and the instruction count
  stays sub-additive. `TILE_N = 64` halves the unrolled body and restores
  additivity: both 81.0 -> 70.9 us (+14 %; `no_instruction` 2.0 -> 0.25) while
  every single-feature arm LOSES at the half tile (plain -12 %, bias -4 %,
  `kv_mask` -10 %). A chooser that scores the bias and the mask as independent
  costs is wrong by that excess. This is ONE measured cell: at this shape the
  combined graph's best measured width is `TILE_N 64` (+14 % same-process; the
  one-batch-entry form +12 %, the additive-vector form +22 %) -- a finding the
  chooser scores per S / n_chunk / form from its own sweep before any default
  flips, not a pick to apply elsewhere. Check the ratio on any new rendering
  that adds body (a streamed ring, a third operand) before shipping it.
- The default `n_chunk` rule's pick (`sm107_n_chunk_rule`) is one step short at S = 128 (2 vs the measured
  4: 12.8 vs 11.2 us, +15 %) and one past at S = 768 (16 vs 8: 328.9 vs 316.7
  us, +4 %); it is the measured best at 384 and 512. Past ~2.5 items per SM
  every chunk length loses to the wave tail (`n_chunk = N` 13-16x).
- The item-boundary bubble of the resident block is 0.13 / 0.23-0.33 /
  0.36-0.56 us per item at S = 384 / 512 / 768 (it grows with the block bytes:
  the exposed tail of the per-item fill), 0.8-1.3 % of the cell at the rule's
  pick -- not a lever. The pair item's gain over the one-batch-entry item is
  the 256-row super-tile's q-row waste (25 % at S = 384), not the bubble: per
  computed 128 x 128 sub-tile step the two forms cost the same.
- `BIAS_PRELOAD_CHUNKS` 1 -> 0 (no bias chunk loaded before the `bmm1_done`
  wait) is +1.7..+1.9 % on the pair item and +3 % on the one-batch-entry item;
  2 is -1 %. The row read hides in place on the issue-bound small tile.
- The ADDITIVE bf16 `kv_mask` vector costs twice the bool word's increment
  (+85 % vs +43 % over plain): the per-pair FFMA2 + unpack is the cost, the
  ballots are not (0.8 % of the bool arm's stall samples). Keep the bool word
  the default where the caller can give one.

Measured facts from the knob sweep of the same arms over the grid (bf16, D = 32
and 64, S_q = S_kv = S in {128, 256, 384, 512, 768, 1024} plus the padding leg
{200, 217, 300, 450, 600}, H = 4, B x S flattened rows with B = 1 and 4, the bias
batch-broadcast; a 212-SM cc 10.7 node at the 2376 MHz lock -- the sampled busy
median 2364 MHz on 32 of 35 processes, 2316-2352 MHz on three d64 processes
whose absolute times are lower bounds; 100 x 3 graph-replay rounds, a control
twin per process within 0.25 % on 31 of 35 processes and up to 1.0 % on the
launch-bound small-S processes, 648 cells, EVERY ratio same-process; at S <= 256
the graph replay is launch-bound on the host (replay / CUPTI 1.2-1.6), so those
ratios read the kernel-only CUPTI time):

- `TILE_N 64` for every graph with BOTH a bias and a bool `kv_mask` -- 26 of 27
  (D, B, S) sets, +9..+55 %, one tie at d32 S 1024 B = 1 (d32 B = 1: +21 / +30 /
  +13 / +18 / +13 % at S 128 / 256 / 384 / 512 / 768, 0 % at 1024; B = 4
  +9..+55 %, the top end the kernel-only time at S 128; d64 +22..+44 % at every
  S).
  Where the knob-less pick is the one-batch-entry streamed ring (d32 S >= 512)
  the best form is the PAIR item's resident block at TILE_N 64 (`n_chunk 8`:
  169 -> 144 us at 512, 487 -> 429 at 768) -- the form pick is coupled to the
  width pick (the same pair form LOSES to the ring at TILE_N 128).
- The padded `plain` rendering (S_kv % 128 != 0, no bias, no mask) at TILE_N 128
  is pathologically slow at D = 32: kernel-only 25.9 us at S 217 against 15.4 for
  the DENSE S 256 cell and 19.6 for the same padded shape WITH a kv_mask; 58.7 at
  S 300 against 43.5 for the dense 384. TILE_N 64 reads +51 / +52 / +57 / +19 /
  +4 % at S 200 / 217 / 300 / 450 / 600 (B = 4: +68 / +64 % at 217 / 300; D = 64:
  +36 % at 200, -4 % at 600). A code-generation effect of the padded dense arm,
  not a work difference (the same kv tiles) -- an open kernel item; until it is
  fixed the chooser takes TILE_N 64 on the padding leg for plain and bias graphs
  (not for the kv_mask-only graph: a wash or a loss there except at one kv tile).
- `TILE_N 64` for the bias-only graph at S_kv <= 256 (+7..+37 % at every D and
  B; with `n_chunk 4` at one kv tile) and on the padding leg (+17..+38 %);
  TILE_N 128 at S_kv % 128 == 0 >= 384 (TILE_N 64 is -3..-13 % there). Plain and
  kv_mask-only graphs at S % 128 == 0 keep TILE_N 128 (64 loses -3..-20 %).
- `n_chunk`: a floor of 4 at ONE kv tile (+8..+16 % over the clamp-2 pick; 8 is
  worse); above ~1500 flattened rows the rule's pick is one power of two too
  large (8 at 1536 rows: +3.1 / +3.7 % at d32 / d64; 4 for the d64 pair ring at
  S_kv 640..896: +2..+5 %) -- a 6-wave reading of the trade with the one-tile
  floor; the `S_q % 256 == 0` one-batch-entry guard holds at TILE_N 128 and
  <= 768 rows only (at TILE_N 64 the pair n2 wins S 256 by +3 %; at 1024 rows the
  pair n8 wins +3.7 % and the one-batch-entry 2-stage ring beats its own resident
  block by +9.6 %, one process).
- The one-batch-entry streamed ring wants 2 stages, not the derived 3:
  +1.5..+3.6 % on 6 of 6 processes (d32 S 512 / 768 / 1024, B = 1 and 4, with and
  without a kv_mask); the pair ring is a tie at d64 and -0.3 / -1.7 % at 2 stages
  in the earlier d32 measurement.
- The d64 one-batch-entry item at cga 2 with the 2-stage ring beats the pair
  ring at S_kv 1024 only (+4.0 % three times at B = 1, +4.6 % at B = 4); at 512 /
  640 / 768 / 896 it is -1 / -33 / -19 / -5 % (two GPUs, the gap closing
  monotonically towards 1024) -- not a rule; the pair ring with `n_chunk 4` is
  within 0..3 % of the best form everywhere >= 640.
- `BIAS_PRELOAD_CHUNKS` 1 -> 0 flips sign with the rendering's FORM (a base / 0 /
  base cross-process A/B/A over 27 cells, base drift <= 0.23 %, then a 12-cell
  confirm of the applied constant, control floors <= 0.18 %): at TILE_N 128
  +1.8..+3.2 % on the RESIDENT bias forms at S_kv >= 384 (+0.9..+2.1 % with a
  kv_mask on the pair forms, -0.7 % on the one-batch-entry + kv_mask; a second node
  of the class read the same S 384 d32 forms +1.7..+1.9 / +3.1 / +1.2 %), +0.0..+0.6 %
  on the streamed ring, -1.2..+0.7 % at one kv tile (S 128, where the chooser never
  picks 128 for a bias graph); the confirm on the applied tree +2.1 % (resident
  pair) and +0.3 / +0.5 / +0.3 % on the three 128-wide ring renderings (above the
  replay control floors, inside the CUPTI control spread); at TILE_N 64 the signs
  are MIXED -- the resident bias + kv_mask body (the body the chooser picks for
  that graph) -2.5..-2.7 %, the resident bias-only forms +0.3..+1.4 %, the ring
  -2.6 % plain / +1.4 % with a kv_mask (one cell each) -- so it is a per-RENDERING
  constant, never a module default: `config_sm107.sm107_bias_preload_chunks`
  (`SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N`: 0 at TILE_N 128, 1 at TILE_N 64), read
  once by the kernel module from `CFG.TILE_N`; numerics are unchanged at every
  value, only the instruction stream around the `bmm1_done` wait moves.
- The THIRD softmax warp group (`TemplateParams.softmax_wgs` 3: a 20-warp body
  whose pair work item takes three rows of the chunk per step, the split 12 x 128
  + 4 x 56 + 4 x 40 = the 20-warp entry pool) is the default of exactly ONE form
  -- a bias TOGETHER WITH a kv_mask on the pair item at TILE_N 64, cga 1, dense KV
  lengths, S_kv >= 640 and at most 3072 flattened batch rows
  (`config_sm107.SM107_THREE_WG_S_KV_MIN` / `SM107_THREE_WG_B_FLAT_MAX`,
  `sm107_softmax_wgs_pick`): +8.1 / +7.9 / +3.3 / +2.9 % over the pair's pick at
  d32 S 640 / 768 / 896 / 1024 and +3.5 / +8.7 % at d64 S 768 / 1024 (the arm and
  its base in one process, 100 x 3 replays, control twins <= 0.14 %); the applied
  default confirmed knob-less +6.6 / +7.3 / +3.3 % at d32 S 640 / 768 / 1024 and
  +5.4 / +10.7 % at d64 S 768 / 1024 at B 1, +7.2 / +7.4 % at d32 S 640 / 768 and
  +0.8 % at d64 S 768 with four batch entries (2560 / 3072 rows), but -2.4 % at
  d32 S 1024 with four (4096 rows) -- hence the row bound.  Inside the resolution
  at S 384 / 512 (+0.5..+0.8 %, d64 S 512 -2.5 %), a LOSS on the bias alone at
  TILE_N 64 (d64 -10..-17 %), and -2.5..-28.5 % at ANY chunk that is not a
  multiple of three (the 3-WG body recomputes one or two DEAD rows per chunk:
  `rows_per_step` in the chunk rule, 6 under the pair's cap of 8).  The P-publish
  hand-off chain is not the binder (a no-chain variant +1.6 / -0.3 %).  Every
  two-group rendering is byte-identical to its md5 record; O / Stats are BITWISE
  the two-group kernel's (`test_sdpa_softmax_wgs_sm107.py`).

The table those facts imply is CODE, in one place per axis, and the sweep's
second half (S = 2048 at D 32 / 64 and B 1 / 4; cluster 1 vs 2 at D 64 S 128..2048;
the padding-mask leg with per-batch lengths of 0.75 S; the one-batch-entry
resident block vs its 2- / 3-stage ring at two kv tiles; the kv_mask-only body at
one kv tile) is in it:

- `config_sm107.sm107_tile_n_pick` is the width (a bias with ANY mask -- the
  kv_mask or the kernel's padding-mask arm -- takes TILE_N 64: the combined body
  is the one over the instruction-cache cliff: with per-batch lengths of 0.75 S
  +11..+73 % at S 256..2048, both D, B 1 / 4, 13 of 13 cells with the table's form
  measured; a bias alone at S_kv <= 256; a kv_mask alone at one kv tile,
  +7..+14 %; the plain body at S_kv % 128 != 0 at d32 and up to 256 at d64; 128
  everywhere else -- per-batch lengths at a tile multiple included, where 64 is
  -11..-18 % on the plain body at S 512 / 1024 and -19 / -20 % at d32 / d64 S 2048;
  the pick sees S_kv, not the lengths, so where the lengths themselves end
  mid-tile (0.75 S at S 256 = 192 keys) the half tile's measured +58..+119 % is
  out of its reach -- the padded dense body's open kernel item). `heuristics._resident_arm_tile_n`, the engine
  row's form evaluation in `engines._bias_arm_decline` and the adapter's
  `_effective_tile_n` call it (rule 8b').
- The one-batch-entry guard of `sm107_n_chunk_pick` holds at TILE_N 128 only: on
  the 64-wide tile the pair's resident block wins (+2.8 % at S 256, +12 / +13 %
  at S 512 / 768 with a kv_mask), so the pick takes the plan's `tile_n`.
- `sm107_n_chunk_refine` caps the pair chunk at 8 (every larger rule pick lost
  where it was measured against n8: 1536 / 2048 / 8192 rows, the d64 ring at
  640..2048) and floors it at 4 at one 128-wide kv tile while the 2-row items
  stay under 2 waves (128 rows: +8..+16 %; 512 rows: n2 wins back +8 %).
- `sm107_bias_stages_pick` gives the one-batch-entry ring 2 stages (6 of 6 d32
  processes +1.5..+3.6 %, +2.0..+2.1 % at S 2048), the pair ring the deepest.
- `sm107_bias_mode_pick` STREAMS the one-batch-entry item at the 32-wide tile
  wherever its ring fits and the range is two kv tiles or more (+6.4 / +10.0 % over
  its resident block at S 256 with 256 / 1024 rows, +9.9 / +12.9 % at S 384; the
  256-row item's block fill is exposed per item, the ring overlaps it); the pair
  item keeps "resident where it fits" (its 2-stage ring is -0.3 / -1.7 % at d32).
- Cluster 2 is never the pick on the small tiles: -32 / -31 % at d64 S 128 / 256,
  a tie at 384 / 512, -2.7 % at 2048 (graph replay; the CUPTI kernel time -2.4 %); a cga2 request at D <= 32 rides the 64-wide
  tile (-45 % at S 256); the d64 one-batch-entry cga2 ring stays an isolated
  point (+4 % at S_kv 1024 only).

`test/python/sdpa/frost/test_sdpa_heuristics_small_tile_sm107.py` pins every
cell of the table with the margin of the best measured arm over the pick the
heuristics made before it, and the public knob record's round trip through
`knobs_to_public` / `knobs_from_public`. A new measured cell goes into that table
with its margin and its process's node class, clock and control pair -- never a
default flip from one cell. The per-rendering constants were re-measured on the
new renderings: see the module constants' comments in
`kernels/sm107/prefill_d128_f16.py` for the A/B/A each carries.

## Single-CTA packed split scheduler

A one-CTA persistent scheduler publishes locally and releases its local
barrier; it must not issue a DSMEM async store to a nonexistent peer. A
consumer must finish reading the whole response before returning its slot
credit. Source-level vector loads can be scalarized, so the single-CTA THD
path reads through lane zero and broadcasts before the release. Validate
repeated waves with Compute Sanitizer racecheck as well as O/Stats tests.

This also applies to the custom D256 FP8 predecode scheduler: use local
stores and a release arrive for its single-CTA THD payload, preserving the
transaction-counted DSMEM arm for multi-CTA renderings. The paged physical
origin and multiwave cases in `test_sdpa_fwd_paged_thd_fp8.py` exercise this
composition; numerics alone can pass while memcheck rejects the DSMEM write.
Broadcast predecoded fields (Q tile, head, batch, bounds and split) as well as
the original response before returning a slot credit. A vector response read
alone leaves scalar field loads racing with the next local publication; the
D256 multiwave racecheck catches the output-store warp reading a reused slot.

The D256 FP8 shared-alpha handoff uses per-lane release arrivals for both
publication and reuse, with matching barrier counts. Keep elected warp
credits confined to the TMEM-only handoff: a TMEM wait does not publish the
other lanes' shared alpha stores. Run the top-left mask/sink case in
`test_sdpa_fwd_paged_thd_fp8.py` under racecheck, including its dense and
compact controls, as well as the persistent scheduler case.

Packed split workspace is bounded by declared packed-Q capacity. Every
partial store and combine read must use live token coordinates, including
empty sequences and nonaligned tails. `test_paged_thd_split_capture_lengths_and_stats`
checks changed device lengths under retained captures and protects tails
with sentinels; the combine tests poison dead partials with NaNs.

A finite partial LSE can still produce a zero combine weight after exponentiation,
including when a sink dominates the row. Mask its partial O by the computed
weight before arithmetic; testing only `LSE == -inf` misses `0 * NaN` contamination.
`test_packed_prefetch_excludes_zero_weight_partials` compares explicit reducer
specializations with FP64, including sink-dominated rows and changed-total replay.
`test_quantized_sink_combine_excludes_zero_weight_payload` covers the dense
quantized small/serial reductions with NaN and both infinite payloads.

Oversized Q/O backing allocations do not enlarge a split plan's live-Q bound.
Without an explicit packed-total hint, the split workspace is still bounded by
`B * S_q`. Clamp the observed extent to that bound before checking logical HN
coverage and binding descriptors, partial strides or combine arguments; retain
physical storage checks. Cover omitted total hints with oversized Q/O/Stats,
changed device lengths and untouched tail canaries in both binding paths.


Standalone prepared calls must enter Q's CUDA device before resolving an
implicit stream or invoking a compiled host, and restore the caller's device on
success and validation failure. A matching device in buffer metadata alone does
not set the calling thread's CUDA context. The SM80 regression is
`test_sm80_direct_standalone_with_another_device_current`: run with two GPUs,
check the launch context and stream, then verify changed-input graph replay.


Sink-aware packed split partials must exclude the sink. Combine adds its
per-query-head virtual-key mass once to the stable denominator and never to
the value numerator. Partial LSE remains natural-log; convert only the final
Stats. Test keyless rows (O=0, LSE=sink), negative-infinite sinks, poisoned dead
partials, both runtime reduction branches, and changed sink contents/pointers
under retained captures. `test_packed_sink_combine_counts_virtual_key_once`
and the sink cells of `test_paged_thd_split_capture_lengths_and_stats` are the
detectors; a sink in each partial silently counts it once per partition.

SM80 THD graph compilation explicitly targets `sm_80` for both metadata and
attention artifacts. Making an A100 current does not suffice when a different
architecture is visible first; `test_graph_thd_targets_nonzero_ordinal_a100`
checks both compile options and numerical replay on that configuration.
After compiling `score_sign`, launch with `config_sm80.kernel_scale` of the
logical scale and reject a runtime sign change. Zero is a logical zero scale,
not a request to reuse the plan's scale. `test_graph_thd_nonpositive_scale_and_replay`
checks zero/negative scales, keyless rows and changed-V replay.


Paged pool layout is per operand. A logical `(pages, heads, page_size, dim)`
shape does not imply physical HND storage; singleton head axes can make two
layout kinds both report `is_contiguous()`. Validate each runtime pool against
its own declared kind. TMA descriptors may share a coordinate order while
carrying different strides; keep descriptor axes and loader coordinates
consistent rather than changing one alone. The `independent_pool_layouts`
tests pin FROST execution and compare O/Stats in both mixed directions,
including one KV head; backend fallback must not count as kernel support.

An adapter subclass must explicitly retain unsupported layout flags when its
parent gains a new mode. Graph capability filtering does not protect direct
adapter callers. `test_standalone_adapter_rejects_thd` checks the SM100 D256
decline with otherwise valid descriptors and a positive parent-row control;
it must fail if the subclass inherits the parent's THD admission. For pitched
views, protect input and output padding with canaries as well as gradient
padding, and check all of them after eager execution and graph replay.


An admission test must reach the rule it claims to protect. Keep other knob
values inside the engine's legal domain; an earlier CGA or dtype decline can
otherwise hide a missing attribute restriction. For example, the pre-folded
scale case in `test_thd_pack_gqa_unsupported_flavors_decline` includes legal
CGA2 as well as CGA1. Removing only the pre-folded-scale decline must make the
CGA2 case fail while the unrelated-domain cases still pass.


Paged FP8 D256 uses rank-5 TMA maps with more than one dimension subtile.
When a page is shorter than the KV tile, keep the shared-memory pitch of each
dimension slab at the full tile height; a short TMA box must not pack the
next dimension slab directly after the page rows. Use each pool's own strides
and table. `test_paged_graph_fp8_d256_page_boundaries` exercises both sides
of the tile/page boundary and independently laid-out K/V pools.

A sink-only row must publish O=0 and LSE=sink even for a very negative finite
sink. An empty softmax's reset max is a sentinel, not a real score: folding
`exp(sink - sentinel)` can underflow and turn LSE into -inf. Select the
sink-only result explicitly and retain the later padded-Q trim override.
Detectors: `test_paged_graph_fp8_sink_rebind_replay` and
`test_prepared_fp8_d256_dense_keyless_sink` include sink logits -120/+120.


A shared GMEM operand can need different SMEM swizzles for its MMA consumers.
In native FP8 D192x128 backward, the nontransposed dP contraction reads dO/V
with 128-byte atoms, while transposed dV reads 64 value columns per CTA with
64-byte atoms. Keep each TMA map, SMEM descriptor and MMA instruction walker
consistent; V's descriptor stride follows V, not Q/K. The host detector
`test_fp8_tma_layout_matches_each_mma_consumer` compares config values against
`MmaDesc`'s independently derived contraction/transpose geometry. Whole-backward
native numerical and replay tests remain necessary; compilation alone misses
these wrong-address walks.

Physical THD origin forwarding must cover every admitted CGA, including unsplit
nonpaged paths. A CGA2 replay does not validate CGA1 lowering: the kernel can
accept the plan while missing origin specs silently select compact prefixes.
`TestMixedFp8Kv.test_thd_cga1_origins` pins CGA1 for uniform and mixed KV, asserts
prepared origin specs before launch, and checks gapped inputs/output canaries
under changed bindings and retained replay.

An explicit tile-domain extension must not silently enroll another dtype in
a measured default picker. Rubin's `tile_ns_by_d_shape` also feeds the
half-precision resident-arm selector; explicit-only per-tensor widths belong
in `effective_tile_ns`, like the native D128 explicit narrow tile. The detector
`test_fp8_d64_narrow_tile_keeps_graph_defaults` checks both A and FALLBACK at
first-half tails as well as tile multiples, while admitting an explicit N64.
