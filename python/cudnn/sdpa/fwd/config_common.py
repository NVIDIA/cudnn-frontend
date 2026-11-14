# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight immutable SDPA template parameters and shared geometry checks.

This module owns no kernel selection, target-specific pipeline, or mutable
sweep state. Implementation config modules depend on it, never on another
implementation's config. The legacy config_sm100 imports remain re-exports.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_O_MXFP8,
    DTYPE_O_NVFP4,
    O_BLOCK_SCALE_BY_DTYPE,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph compile-time parameters threaded into a kernel template.

    Design notes (see also python/cudnn/frost/README.md):

    - Contents: graph-derived semantics (dtype, masks, sink/padded/THD) plus
      the chosen values of true tuning knobs (sched_policy). Shapes are
      deliberately NOT here — they are handled by the template's per-shape
      ``compile()`` cache; this record holds only what changes the *traced
      code* of the kernel.
    - Direction: this is the OUTPUT of a successful eligibility match, never
      an input to it. The probe compares facts and requested knobs against
      an engine's Capabilities; only after that passes does the engine's
      ``lower`` hook assemble this record.
    - Frozen + hashable on purpose: it doubles as the kernel-module cache key
      in frost.template_loader — one distinct TemplateParams == one distinct
      compiled specialization, and identical params reuse the compiled one.
    - Validation here (``_validate_params`` and the ``make_cfg_*`` checks)
      raises ValueError but is a BACKSTOP: every reachable violation must be
      rejected earlier by a Capabilities row; tripping it means that row is
      dishonest, not that a user erred.
    """

    dtype_qkv: int = DTYPE_FP16  # E4M3/E5M2 (0/1, the quantized kernels) or BF16/FP16 (2/3)
    dtype_o: int = -1  # output dtype (0..3); -1 = inherit dtype_qkv. MXFP8 writes BF16/FP16.
    # The mask is ONE diagonal band (the model FlashAttention / CUTLASS FMHA /
    # the analyzer facts all share): per-side OFFSETS from the diagonal, None =
    # unbounded on that side. Row q attends kv in [q - window_left, q +
    # window_right] (shifted by S_kv - S_q when bottom_right):
    #   window_right = None -> no upper bound;  0 -> plain causal;  R > 0 ->
    #   causal widened by R future tokens (diagonal_band_right_bound).
    #   window_left  = None -> no lower bound;  W -> sliding window of W past
    #   tokens (cuDNN diagonal_band_left_bound - 1).
    #   bottom_right anchors the band's diagonal at the bottom-right corner.
    # make_cfg_* derives the kernels' MASK_FLAGS bits from these.
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    bottom_right: bool = False
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the
    # epilogue scales the natural-log LSE by log2(e) right before the store.
    stats_log2: bool = False
    seq_kv_lens_present: bool = False
    # Dense padded-Q trim: per-batch seq_len_q is a SEPARATE (B,)-int32
    # kernel parameter (seq_q_lens_addr — mirrors cuDNN's distinct SEQLEN_Q
    # pointer / FA's seqused_q; its reads compile out unless this flag is
    # set); q rows >= seq_len_q[b] write O := 0 / LSE := -inf (cuDNN >= 9.14
    # convention). Dense-only — THD carries per-sequence Q lengths via
    # cu_seqlens instead.
    seq_q_lens_present: bool = False
    sched_policy: int = SCHED_NATURAL
    # Compile-time LPT head/batch grouping. Keep 1 unless the selected kernel
    # and concrete graph shape opt into a divisor of B*Hq.
    lpt_head_group: int = 1
    # Dense FP8 kernels may specialize scheduler selection/decoding to the
    # graph's compile-time number of query tiles. Zero keeps runtime derivation.
    lpt_q_tiles: int = 0
    # Optional L2 working-set budget for SCHED_LPT_L2. Zero keeps the flavor's
    # default budget.
    lpt_l2_size_mib: int = 0
    thd_varlen: bool = False
    # Ragged Q/O/Stats over PAGED K/V on the d128 DECODE tile (S_q(max) == 1):
    # the Q row coordinate is the batch's ragged offset over the packed Q view
    # and the split combine places the final O / Stats rows at their ragged
    # offsets; the dense grid, PackGQA and the split path are all kept.  Not
    # the prefill tile's THD leg (thd_varlen), which is mutually exclusive.
    ragged_q: bool = False
    # PackGQA: pack Q rows from the G query heads sharing one KV head into a
    # single TILE_M tile, token-major (row r ↔ token r // G, head r % G), so
    # tiles stay full for GQA/MQA.  When G does not divide TILE_M the d128 and
    # d256 f16 kernels pack its largest divisor that does (Cfg.PACK_G, from
    # pack_gqa_group_size); ``qh_per_kh`` is always the graph's GQA ratio.
    pack_gqa: bool = False
    qh_per_kh: int = 1
    # KV split: each Q tile's KV loop range is cut into ``split_kv`` contiguous
    # chunks, each run as its own persistent tile writing a partial (O, LSE)
    # that kernels/sm100/split_combine.py reduces.  1 = off (byte-identical
    # codegen to the single-pass kernel).
    #
    # A split writes its partials as fp32, straight from the epilogue's
    # accumulator registers and bypassing the SMEM O tile and its TMA store, so
    # the combine reduces an unrounded input and performs the only rounding to
    # O's dtype.  Partial-only: the caller-visible O is unchanged.
    split_kv: int = 1
    # MMA cluster width: 2 = cga2 collective tcgen05.mma.cta_group::2 (a CTA
    # pair share one MMA, each holding half of every K/V tile); 1 = cga1, one
    # independent CTA per tile.
    #
    # cga1 has no collective MMA to halve per-CTA K/V, so at a fixed STAGES_KV
    # it doubles that footprint.  d128 buys the 64 KiB back by aliasing Q and O
    # into one slab (make_cfg_d128 turns QO_ALIAS on for cga1; see
    # _validate_cfg_d128's SMEM check); the fp8 family instead scales the stage
    # count with the width so stages x per-CTA-buffer stays constant.
    cta_mma: int = 2
    # Native head-dim geometry of the flavor this template compiles to (TILE_K =
    # TILE_O). The SM100 f16 prefill file serves both the llama d128 geometry and
    # the narrow d64 one (gpt-oss class, d_qk = d_v = 64): at 64 the MMA runs a
    # K=64 tile instead of zero-filling a 128-wide one, which is half the BMM1
    # work and half the Q/K/V/O SMEM. Shapes are deliberately absent from this
    # record, but the flavor changes the TRACED code (tile extents, SMEM slabs,
    # TMA boxes), so it is a template-key field rather than a compile() arg.
    d_flavor: int = 128
    # cc10.3+ fuses the S_acc row-max into the LDTM (tcgen05.ld.red.f32.max); cc10.0
    # lacks it and uses the manual load + software reduction. Auto-set from the device
    # capability at compile time (MXFP8 + the per-tensor FP8 d192x128 kernel; the f16
    # kernels do not read it, and the SM107 siblings carry the instruction unconditionally).
    fused_ldtm_stat: bool = False
    # exp2 MUFU / FMA split of the sm100 softmax (the _E2E_* block of sm100/prefill_d128_mxfp8.py,
    # prefill_d128_fp8.py and prefill_d192_d128_f16.py; the _exp2_* helper mix of
    # prefill_d192_d128_fp8.py): a slice of the exp2 per row on the FMA pipe instead of MUFU.EX2.
    # Claimed per KERNEL and per ARCH, because its sign follows the part's MUFU.EX2 rate: MEASURED
    # 16 elements/clk/SM on cc 10.0 (B200, where the split is +7.8 % on d128 MXFP8, +4.5 % on d128
    # FP8, +1.9 % on d192x128 bf16 at the chart layers, +7.0 % causal / +5.7 % dense on d192x128 FP8
    # at the DSv3 layer) and 32 on cc 10.7 (Rubin, where the same split is -9..-10 %: an emulated
    # exp2 costs 1.99x the MUFU time it frees); cc 10.3 (B300) has the doubled rate too -- MEASURED
    # -10 % causal / -6 % dense on d192x128 FP8 with the split on.  Auto-set by the adapter from the
    # BUILD device (api_dsl._exp2_fma_split_for: cc == (10, 0) x the four kernels above, and the d64
    # decode tile, sm100/decode_d128_f16.py, through its own set) -- widen only after an A/B on the new
    # cc / kernel.  Off, the kernel traces the plain MUFU exp2 (the develop spelling).  Only those
    # kernels read it; make_cfg_d192 also derives CfgD192.EXP2_TAIL_MIX, the d192x128 f16 kernel's dense-causal arm
    # spelling, from it.
    exp2_fma_split: bool = False
    # The L2-prefetching loader of the half-input single-CTA d192x128 tile (sm100/prefill_d192_d128_f16.py at cta_mma=1,
    # half precision): the TMA-LDG warp prefetches the K/V tiles one kv step ahead and the next tile's Q / first K/V
    # into L2, runs its kv loop unrolled 1x, and the dense-causal arm keeps the plain MUFU exp2.  Claimed per arch like
    # exp2_fma_split: set by the adapter (api_dsl._cga1_l2_prefetch_for) on the cc it is measured on; off, the tile
    # keeps the develop loader.  Only that kernel reads it.
    cga1_l2_prefetch: bool = False
    # sdpa(softmax_precision=cudnn.data_type.HALF) op attribute: exponent + P-cast run as
    # f16x2 pairs (MUFU EX2.F16x2 + cvt.rn.satfinite.*x2.f16x2) instead of
    # scalar f32 ex2. The cc 10.7 FP8 / MXFP8 kernels (every flavor; see
    # config_sm107.SM107_SOFTMAX_F16_FLAVORS) — the exp arguments are bounded
    # (<= RESCALE_THRESHOLD [+ P_CAST_LOG2_SCALE on per-tensor FP8]), so f16
    # range is exact where it matters and P quantizes to FP8 either way.
    softmax_f16: bool = False
    # The caller has already multiplied Q by attn_scale * log2(e): the kernel runs exp2(S - m) on the
    # raw QK^T (no per-score FFMA2 by the scale) and, together with softmax_f16, fuses the shift and the
    # f32->f16 convert into one instruction per pair.  The published Stats are unchanged -- the running
    # max and the scores are in the same log2 domain as when the kernel applies the scale itself.
    # Served by the cc 10.7 MXFP8 and f16/bf16 kernels (config_sm107.SM107_SCALE_PREFOLDED_FLAVORS); per-tensor
    # FP8 declines it (the kernel folds descale_q * descale_k into the softmax scale), as do the cc 10.0 / 10.3
    # line at config time and the adapters of the other architectures at check_support.
    softmax_scale_prefolded: bool = False
    # Paged KV cache (FlashInfer / vLLM decode contract): K/V are page pools
    # indexed through a per-batch ``block_table`` [B, max_pages] int32, and
    # the per-batch KV length is the (B,) ``seq_kv_lens`` device tensor
    # (``seq_kv_lens_present`` is therefore mandatory). ``page_size`` tokens
    # per page. The in-page layout (HND ``[num_pages, H_kv, page_size, D]`` vs
    # NHD ``[num_pages, page_size, H_kv, D]``) is NOT a parameter: it is the
    # pool's strides, which the per-shape compile() already pins. Dense-only.
    paged_kv: bool = False
    page_size: int = 0
    # Experimental D128 MXFP8 specialization: Q/K remain block-scaled FP8,
    # while the complete softmax(P) @ V leg uses BF16 operands. This is a
    # template axis because it changes the TMA maps, shared-memory layout and
    # BMM2 instruction kind. It is intentionally not wired into graph routing.
    pv_bf16: bool = False
    # Output Amax is an independently optional graph output. It is normally
    # enabled for the existing MXFP8 kernels; the direct QK-MXFP8/BF16-PV
    # experiment enables it only when its plan declares an Amax_O buffer.
    # This is plan-time state: it must never be inferred from execute() args.
    emit_amax_o: bool = True
    # Epilogue gate: O := O * sigmoid(GATE). GATE is TMA-staged by the load warp after the KV loop into a
    # 64 KiB SMEM tile and consumed in the correction epilogue. Compile-time: sizes the tile + 2 mbarriers.
    # Served by the SM107 d256 f16/bf16 and per-tensor FP8 kernels only (config_sm107._EPILOGUE_GATE_FLAVORS).
    # Being a TemplateParams field it is part of the module-cache key: gate on/off are two coexisting
    # specializations of one template, and an ungated module traces byte-identically to before the field existed.
    epilogue_gate: bool = False
    # Decode-shaped d256 f16/bf16 graphs (S_q * pack_g within the arch line's routed
    # rows: D256_DECODE_ROUTED_MAX_Q_ROWS here, config_sm107's table (16, 32, 64) on
    # cc 10.7, where the 32- and 64-column tiles serve the MTP step in token units) lower onto
    # sm100/d256_f16_swap_ab/kernel.py, the swap-AB tile: the KV tokens ride the MMA M axis
    # and the packed Q rows ride N, so the MMA and exp work scale with the LIVE
    # rows instead of a 128-row Q tile.  The value is the N extent the kernel is
    # compiled for -- the rows of ONE token unit (16; 32 at the template level, 32
    # and 64 on the Rubin route, from decode_d256_q_tile); 0 = the prefill tile.
    # Plan-time only (S_q is a declared shape).
    decode_q_tile: int = 0
    # The 128-row DECODE tile (sm100/decode_d128_f16.py) for flavors that do not
    # key it off cta_mma.  d128 routes that tile by cga1, because its prefill
    # pipeline wants cga2 anyway; d64's prefill is FASTEST at cga1 (a 512-row
    # cga2 cluster wastes most of a narrow diagonal band), so the two legs would
    # be indistinguishable by cta_mma alone -- and TemplateParams is the template
    # cache key, so one record must never name two kernels.  Set by
    # SdpaFwdDslSm100._d64_decode_tile, the twin of _decode_q_tile.
    decode_tile: bool = False
    # The quadrant-cooperative softmax of the d64 decode tile (sm100/decode_d128_f16.py QUAD): the live Q rows (S_q x packed heads <= 32) are loaded into
    # all four TMEM lane quadrants and each softmax warp owns a quarter of the key columns, the correction warps merging the four partials per unit.
    # Needs decode_tile; set by SdpaFwdDslSm100._d64_decode_quad.
    decode_quad: bool = False
    # The decode tile's early-Q form (sm100/decode_d128_f16.py EARLY_Q): the Q slab is released by the last BMM1 of a unit instead of the O store's
    # return edge, and the correction warps store the half output straight from TMEM.  Claimed per arch, like exp2_fma_split: set by
    # SdpaFwdDslSm100._decode_early_q on the cc it was measured on (10.0, 10.3); an unmeasured part (cc 10.7 loads the same tile) keeps the staged
    # TMA store.  The THD / ragged legs keep it whatever the field says.
    early_q: bool = False
    # Per-tensor FP8 d128 decode: the two Q sub-tiles of sm100/prefill_d128_fp8.py
    # become two KV STREAMS over one 128-row Q tile (even KV tiles on sub-tile 0,
    # odd on sub-tile 1, each with its own running max / sum, merged once in
    # the epilogue).  Doubles the K/V bytes per pipeline step, which is what the
    # decode-shaped step -- chain-latency-bound at ~1.1 us a 128-key tile -- is
    # short of.  cga1, dense or paged, S_q x packed heads <= 128 rows; set by
    # SdpaFwdDslSm100._fp8_dual_kv_stream.  Consumed by that template only.
    dual_kv_stream: bool = False
    # Internal specialization of the shared one-Q-tile half pipeline.
    single_q_head_dim: int = 128
    # The graph declares capacity for exactly one sequence. Runtime lengths
    # remain device data; the prepared binder rejects a larger batch.
    thd_batch_one: bool = False
    # d512 f16/bf16 forward on the "2x2 datapath": ONE pipeline per CTA on the
    # cta_group::2 M=128 atom (64 Q rows per CTA, 12 warps), a (4,1,1) cluster of
    # two pairs sharing K/V by TMA multicast -- sm100/prefill_d512_f16_2x2.py
    # (make_cfg_d512_2x2 / CfgD512X2) instead of the cga4x1 role-split kernel.
    # APPEND-ONLY and default False: every existing record renders byte-identically
    # (make_cfg_d512 and the loader read it through getattr).  Plan-time only;
    # api_dsl.D512_2X2 is the call-time switch that sets it (default True since 2026-10-06; False = the
    # role-split A/B arm).
    mma_2x2: bool = False
    # attn_scale < 0: BMM1 negates Q (tcgen05 a_negate), so the kernel's raw-score max, masks and exp2 run on -S at
    # |attn_scale| (#1435). APPEND-ONLY, default False.
    negate_scores: bool = False
    # A dense S_kv off the KV tile with no mask covering the tail: compile the padded mask against the scalar
    # S_kv, with no per-batch lengths buffer (MASK_PADDED without SEQ_KV_LENS_PRESENT; #1425).
    kv_tail_mask: bool = False
    # Index-list (block-sparse) attention on the cc 10.7 d256 f16/bf16 line (sm107/sparse_d256_f16.py,
    # config_sm107.make_cfg_d256_sparse): the query attends the keys of its caller-selected 4-token blocks plus the open
    # tail block of its visible range, under the causal mask.  These four are DECLARATION attributes (the selection IS the
    # function), never knobs, and they are the template's compile-time geometry:
    #   qsa_block_topk       0 = not an index-list record (every dense template declines a non-zero value: it would read
    #                        K/V densely and silently ignore the list).  Otherwise the list width: the block kernel's
    #                        CFG.BLOCK_TOPK, 4 <= top_k <= 512, top_k % 4 == 0 -- every ids byte count in the kernel is
    #                        top_k x 4 (the staging slot, the bulk-copy length, the expect_tx), one compiled module per value.
    #   qsa_block_size       tokens per selectable block; 4 = one tma_gather4 (four rows) per block per 128-B column box.
    #   qsa_include_open_block  True = the kernel appends the open tail block from the query's position (the model's
    #                        semantics).  False is a KERNEL-ONLY test arm (a pure caller list, for A/B against references
    #                        that materialise the tail); it is not a public field of the block's spec.
    #   qsa_list_per_sequence  the decode / MTP form: block_ids is [B, top_k], one list per sequence, and the tail runs
    #                        from the first decode token's position.  False = one list per query token ([T_q, top_k]).
    # APPEND-ONLY, defaults inert: every existing record renders byte-identically (the make_cfg_* factories read these
    # through the record, and 0 / 0 / True / False is "no index list").
    qsa_block_topk: int = 0
    qsa_block_size: int = 0
    qsa_include_open_block: bool = True
    qsa_list_per_sequence: bool = False
    # The sparse kernel's softmax SPLIT -- a PERFORMANCE knob of its softmax role, never numerics (the same function, bitwise, at
    # either value): 0 = the flavor default (TWO 4-warp column groups of 8 columns each, 20 warps, 96 registers flat), 1 = ONE 4-warp
    # column group over the 16 columns (16 warps, the pre-split body).  A template-cache axis like mma_2x2 (two coexisting renderings of one file);
    # APPEND-ONLY, default inert; every dense template declines a non-zero value together with the other qsa_* fields.
    qsa_softmax_groups: int = 0
    # Additive attention bias (the graph's ``bias`` operand, ``[1|B, 1|H_q, S_q, S_kv]`` in the io dtype) on the Rubin
    # f16/bf16 d128-file tiles: 0 = none (every existing record renders byte-identically), 2 = the STREAMED ring (one bias
    # tile per kv tile through a ``bias_stages``-deep SMEM ring, any S_kv -- the fallback above the resident budget), 1 = the RESIDENT row-block --
    # the TMA-LDG warp loads the CTA's ``TILES_Q * TILE_M`` bias rows for its whole KV range into SMEM once per work
    # item and the softmax warps read their row slice with swizzled lane loads, adding it pre-scale as one FFMA per
    # score (``s + bias * inv_scale``) before the band mask and the software row max.  A compile-time axis (the SMEM
    # block, two mbarriers and the masked-path softmax arm fold on it) and therefore part of the module-cache key.
    # Served by ``sm107/prefill_d128_f16.py`` at the widths whose SMEM holds at least one 128-column block
    # (``config_sm107.sm107_bias_resident_s_kv_max``); the SM100 line and every other flavor decline it.
    bias_mode: int = 0
    # Per-key mask (the python-only ``kv_mask`` input of ``sdpa``, ``[1|B, 1, 1, S_kv]``): 0 = none; 1 = the BOOLEAN
    # mask in the bit-word form -- each softmax warp assembles one 32-bit keep word per 32 kv columns of the tile by a
    # warp ballot over byte loads and folds it into the bit-word mask (``tile_dsl.mask.apply_mask_chunk``); a key whose
    # byte is 0 is masked like a padded one, a row whose every key is masked reads O = 0 / Stats = -inf; 2 / 3 = the
    # ADDITIVE per-key vector in the io dtype (bf16 / fp16 pairs) / in fp32, loaded as warp-uniform 16-byte granules and
    # folded into the raw scores pre-scale exactly like the bias (``s + v * inv_scale``, one packed FFMA per pair) --
    # the models' finite -1e9 fold preserved as a VALUE (a fully -1e9 row is their uniform average, not the empty-row
    # contract); nothing staged in SMEM, no barrier row.  Same kernel and same serving set as ``bias_mode``; composes
    # with ``seq_kv_lens_present`` and with the bias.
    kv_mask_form: int = 0
    # KV tile width of the d128-file softmax (the S-accumulator columns per sub-tile, the K/V ring tile, the P packing
    # and the softmax chunk count all derive from it).  128 is the value every rendering has run at; 64 is the
    # one-chunk arm of the f16/bf16 small tiles (trace-compiled and validated on cc 10.7; served through the TILE_N
    # knob, ``config_sm107.SM107_TILE_NS_BY_D_SHAPE``) and 256 is refused by the TMEM predicate at two sub-tiles -- the
    # config factory pins ``N_BMM2_CHUNKS x BMM2_CHUNK_SIZE == TILE_N``.
    tile_n: int = 128
    # The (n, n+1) PAIR work item of the resident-bias arm (``bias_mode = 1`` on the Rubin f16/bf16 d128-file tiles): 0 = the
    # item is a 256-row Q super-tile of ONE batch entry (every existing record renders byte-identically); n >= 1 = a work item
    # is (a 128-row q tile, a head, a CHUNK of up to n consecutive entries of the flattened batch axis), the two softmax
    # sub-tiles are rows (nb, nb + 1) of the SAME q tile, and the bias row-block (now 128 rows) is loaded ONCE per item and
    # read by every step of the chunk -- the per-item fixed cost and the bias feed divided by n.  Needs a batch-broadcast
    # bias ([1, 1|H, S_q, S_kv]: two rows share one block), cta_mma = 1 and a top-left band (config_sm107._validate_bias).
    # A compile-time axis (the block rows, the grid decode, the pair's ring protocol fold on it), hence a cache-key field.
    n_chunk: int = 0
    # The STREAMED bias ring's depth (``bias_mode = 2`` on the Rubin f16/bf16 d128-file tiles: the bias row-block is no longer
    # resident for the item's whole KV range -- the TMA-LDG warp streams ONE ``BIAS_BLOCK_ROWS x TILE_N`` bias tile per kv tile
    # through a ``bias_stages``-deep SMEM ring the softmax warps release per kv tile, so S_kv is bounded by nothing but the
    # payload / stride rules).  0 = the factory's derived default (the deepest of 3 / 2 the carveout holds beside the operands;
    # ``config_sm107.sm107_bias_stages_pick``); 2 / 3 = a request.  Meaningless (and must be 0) on ``bias_mode`` 0 / 1.  A
    # compile-time axis (the ring's SMEM extent and its two mbarrier rings fold on it), hence a cache-key field; APPENDED.
    bias_stages: int = 0
    # The PADDED tail form of the PLAIN padded kv loop on the Rubin f16/bf16 32- / 64-wide tiles (MASK_FLAGS == MASK_PADDED alone: per-batch
    # KV lengths or an S_kv that is not a tile multiple, no bias / kv_mask): 0 = the tail kv tile runs the SOFTWARE-max masked body as a
    # second loop beside the HW-max body (a second full softmax body per warp group -- the 128-wide rendering outgrows the per-SM
    # instruction cache, the padding leg's measured pathology); 1 = the PRE-MASKED tail: one loop, the HW-max body on every tile, and on
    # the tail tile the softmax warp first rewrites its sub-tile's out-of-range S columns as -inf in TMEM
    # (``config_sm107.sm107_padded_tail_form_pick``).  A compile-time axis of the kv loop, hence a cache-key field; 0 on every other
    # rendering (the factory refuses 1 off the eligible set).  APPENDED.
    padded_tail_form: int = 0
    # The DEAD-SUB-TILE QUIET body of the one-batch-entry item on the Rubin f16/bf16 32- / 64-wide tiles (no bias): a softmax sub-tile
    # whose FIRST row is at or past its batch entry's S_q has no live row -- sub-tile 1 of the last 256-row super-tile when S_q % 256 is
    # in 1..128 (S 384: a quarter of the item's softmax work), and under cga2 the non-leader CTA's sub-tiles past S_q.  0 = that warp
    # group runs the FULL softmax body on rows nobody stores (the correction trims them, the TMA store clips them); 1 = it runs the QUIET
    # body: every barrier interaction of the live body (the bmm1_done wait, alpha = 1 and the stat_full arrive, the bmm2_ready arrives,
    # the warp-group ping-pong, the stat_empty wait), no TMEM load and no exp2 / max / sum, so the live sibling warp group runs the
    # ping-pong alone (``config_sm107.sm107_dead_subtile_quiet_pick``).  A compile-time axis of the kv loop, hence a cache-key field;
    # 0 on every other rendering (the factory refuses 1 with a bias, on the pair item, on PackGQA / THD / the gated epilogue).  APPENDED.
    dead_subtile_quiet: int = 0
    # The ROW-MAX FORM of the bias / kv_mask softmax body (the -inf masked body of the Rubin f16/bf16 d128-file tiles): 0 = the
    # SOFTWARE row max over the biased, masked scores (every existing record renders byte-identically); 1 = the HARDWARE UPPER
    # BOUND -- the fused ``tcgen05.ld.red...max`` over the RAW scores (the plain body's own load) plus the thread's row max of the
    # bias tile in the packed 16-bit domain: ``max_k S + max_k bias >= max_k (S + bias)``, so the online softmax runs on an upper
    # bound of the true row max (valid for stability: P <= 2^RESCALE_THRESHOLD as before; the LSE and O are shift-invariant), the
    # exps shrink by at most the row's RAW score range inside the tile (tens of log2 units against the fp32 flush at -126), and the
    # software max tree leaves the post-load critical path.  Served with ``bias_mode`` / ``kv_mask_form`` 1 only (an additive
    # kv_mask vector has no cheap per-tile max); the score side outputs stay on the software arm (the bias arms decline them).  The record
    # field defaults to 0; the ADAPTER picks the measured per-form default for a plan that requests none (``config_sm107.
    # sm107_bias_max_form_default``: 1 on the kv_mask-only form, 0 with a bias).  A compile-time axis (the body folds on it), hence a
    # cache-key field; APPENDED.
    bias_max_form: int = 0
    # The number of SOFTMAX WARP GROUPS of the Rubin f16/bf16 d128-file body = its Q sub-tiles per work item (``TILES_Q``): 2 = the
    # classic 16-warp body (every existing record renders byte-identically); 3 = the 20-warp body of the (n, n+1) pair item at TILE_N 64
    # (three rows of the chunk per step, the split 12 x 128 + 4 x 56 + 4 x 40 = the 20-warp entry pool at the 32-wide tile) -- the
    # MEASURED default of ONE form, a bias together with a kv_mask on that item at cga1 with dense KV lengths from S_kv 640 up to 3072
    # flattened batch rows (``config_sm107.sm107_softmax_wgs_pick``), selectable through the adapter's ``softmax_wgs=`` / the sweep seam
    # on the rest of its serving set.  A compile-time axis (the warp table, the per-sub-tile ring depths and the TMEM map fold on it),
    # hence a cache-key field -- the kernel-module cache is keyed by this record, so the 3-WG build is its own module; APPENDED.
    # ``config_sm107._validate_softmax_wgs`` pins its serving set (TILE_N 64, the pair item, cga1, the f16/bf16 32- / 64-wide tiles).
    softmax_wgs: int = 2
    # The S / P TMEM slot count per softmax sub-tile of the Rubin 32- / 64-wide f16/bf16 tiles at TILE_N 64 (APPENDED after ``softmax_wgs``; the sm107
    # d128 file reads it through getattr, every other template ignores it): 1 = one slot per sub-tile (every record so far renders
    # byte-identically), 2 = the double-buffered S -- the MMA warp issues BMM1(k + 1) into the free slot ahead of BMM2(k) so the
    # tensor core runs it while the softmax is still on S(k).  A template-key field (not a compile() arg, not a config seam): the
    # slot count changes the TRACED code (the TMEM map, two barrier rings' extents, three roles' cursors) and the template-module
    # cache keys on this record alone, so a process that builds both forms must see two records.  Picked by
    # config_sm107.sm107_s_slots_pick; validated by the factory (TILE_N 64, the small f16/bf16 tiles, the TMEM fit).
    s_slots: int = 1
    # Q resident in TMEM (the Rubin f16/bf16 32- / 64-wide d128-file tiles; every other template ignores it): 0 = BMM1 reads Q from
    # SMEM through its descriptor on every kv tile (every record so far renders byte-identically), 1 = the MMA warp copies each
    # sub-tile's Q tile SMEM -> TMEM once per n-step (``tcgen05.cp 128x128b`` through the same descriptor, one elect block) and issues
    # every BMM1 of the step as an A-from-TMEM ``tcgen05.mma``; the UTCCP's commit releases the Q SMEM slab (``mb_q_empty``) at the START of
    # the step instead of after its last BMM1, so the TMA-LDG warp prefetches the next step's Q / K / V during the current kv loop.  The
    # same MMAs over the same operands in the same k-step order: O and Stats are BITWISE the SMEM-Q kernel's.  A template-key field (the
    # TMEM map gains the Q slots, the MMA form and the commit site move): the template-module cache keys on this record alone.  Picked by
    # config_sm107.sm107_q_in_tmem_pick (0 until measured; the sweep seam pins 1); validated by the factory (config_sm107._validate_q_in_tmem).
    q_in_tmem: int = 0
    # Optional native FP8 K/V storage for the half-input d128 pipeline.
    # Appended: -1 keeps the existing uniform Q/K/V behavior.
    dtype_kv: int = -1


_PAGED_KV_FLAVORS = frozenset({"d64", "d128", "d192", "d256", "d512", "d576"})


_EPILOGUE_GATE_FLAVORS = frozenset({"d256"})


_SPLIT_KV_FLAVORS = frozenset({"d64", "d128", "d192", "d256", "d512"})


_CTA_MMA_FLAVORS = frozenset({"d32", "d64", "d128", "d192"})


_D512_2X2_TILE_M = 64


def _validate_params(
    flavor: str, k: TemplateParams, *, scale_prefolded_wired: bool = False, d512_partial_pack_wired: bool = False, d512_paged_2x2_wired: bool = False
) -> None:
    if k.qsa_block_topk or k.qsa_block_size or k.qsa_softmax_groups:
        # The loader routes an index-list record (qsa_block_topk != 0) to the cc 10.7 sparse kernel
        # (config_sm107.make_cfg_d256_sparse); a dense template must never consume one -- it would read K/V densely and
        # silently ignore the block list.  The selection is the FUNCTION, so this is a decline, not a knob fallback (the
        # sparse kernel's own softmax-split knob qsa_softmax_groups has no meaning on a dense template either).
        raise ValueError(
            f"{flavor}: qsa_block_topk={k.qsa_block_topk} / qsa_block_size={k.qsa_block_size} / qsa_softmax_groups={k.qsa_softmax_groups} select the "
            f"index-list sparse kernel; the dense templates do not consume them"
        )
    if k.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_QKV must be E4M3/E5M2/BF16/FP16 (0..3); got {k.dtype_qkv}")
    fp8 = k.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if k.dtype_kv != -1 and not (flavor == "d128" and k.dtype_qkv in (DTYPE_BF16, DTYPE_FP16) and k.dtype_kv in (DTYPE_E4M3, DTYPE_E5M2)):
        raise ValueError(f"{flavor}: independent FP8 KV storage is wired only on the half-input d128 pipeline")
    if fp8 and flavor not in ("d64", "d128", "d192", "d256", "d512"):
        raise ValueError(f"{flavor}: FP8/MXFP8 inputs (DTYPE_QKV 0/1) are only supported on d64, d128, d192, d256, and d512")
    if k.softmax_f16 and not fp8:
        raise ValueError(f"{flavor}: softmax_f16 is a quantized-kernel (FP8 / MXFP8) specialization (f16/bf16 softmax already runs the f32 pipeline)")
    if k.softmax_scale_prefolded and not scale_prefolded_wired:
        # The cc 10.7 kernels carry the arm (config_sm107.SM107_SCALE_PREFOLDED_FLAVORS; the d512 2x2 twin
        # validates here with the flag); this line's bodies apply the scale in-kernel.
        raise ValueError(f"{flavor}: softmax_scale_prefolded is served by the cc 10.7 kernels only (this line applies the scale in-kernel)")
    if k.pv_bf16 and (not fp8 or flavor not in ("d128", "d192")):
        raise ValueError(f"{flavor}: pv_bf16 is an experimental MXFP8 D128/D192 specialization")
    dtype_o = k.dtype_qkv if k.dtype_o < 0 else k.dtype_o
    if dtype_o not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16, DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        raise ValueError(f"{flavor}: DTYPE_O must be 0..5; got {dtype_o}")
    if dtype_o in (DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        if not fp8:
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) requires FP8 inputs")
        if flavor != "d128":
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) is only supported on d128")
        if k.thd_varlen or k.seq_q_lens_present or k.split_kv > 1 or k.pack_gqa or k.paged_kv:
            raise ValueError(f"{flavor}: block-scaled O (DTYPE_O {dtype_o}) serves dense (unpaged), unsplit, unpacked graphs only")
    if not fp8 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires DTYPE_O == DTYPE_QKV; got dtype_o={dtype_o}")
    if k.window_left is not None and k.window_left < 0:
        raise ValueError(f"{flavor}: window_left must be >= 0 (or None for unbounded); got {k.window_left}")
    if k.window_right is not None and k.window_right < 0:
        raise ValueError(f"{flavor}: window_right must be >= 0 (or None for unbounded); got {k.window_right}")
    if k.bottom_right:
        if k.window_right is None:
            raise ValueError(f"{flavor}: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if k.thd_varlen and not k.seq_kv_lens_present:
        raise ValueError(f"{flavor}: THD/varlen requires SEQ_KV_LENS_PRESENT (per-sequence padded masking)")
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    if k.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: only SCHED_NATURAL (0) / SCHED_LPT (1) / SCHED_LPT_L2 (2) are wired up; got {k.sched_policy}")
    if k.cta_mma not in (1, 2):
        raise ValueError(f"{flavor}: cta_mma must be 1 (cga1) or 2 (cga2); got {k.cta_mma}")
    if k.decode_tile and flavor != "d64":
        # The loader routes a decode_tile record to sm100/decode_d128_f16.py at
        # d_flavor=64 (make_cfg_d64_decode); no other template consumes it.
        raise ValueError(f"{flavor}: decode_tile selects the d64 decode tile; this template does not consume it")
    if k.decode_quad and not (k.decode_tile and k.early_q):
        raise ValueError(f"{flavor}: decode_quad is the d64 decode tile's quadrant softmax; it needs decode_tile and early_q")
    if k.dual_kv_stream and (flavor != "d128" or k.dtype_qkv > 1 or k.cta_mma != 1 or k.thd_varlen or k.pv_bf16):
        # Two KV streams over one Q tile: the per-tensor FP8 d128 pipeline at cga1
        # on a dense or paged graph (sm100/prefill_d128_fp8.py); nothing else
        # consumes the record (the sibling templates raise on it at import).
        raise ValueError(
            f"{flavor}: dual_kv_stream is the per-tensor FP8 d128 cga1 dense/paged decode specialization; got dtype_qkv={k.dtype_qkv} cta_mma={k.cta_mma} thd={k.thd_varlen}"
        )
    if k.decode_q_tile:
        # The loader routes a decode_q_tile record to sm100/d256_f16_swap_ab/kernel.py
        # (make_cfg_d256_decode); a prefill template must never consume one.
        raise ValueError(f"{flavor}: decode_q_tile={k.decode_q_tile} selects the d256 decode tile; the prefill templates do not consume it")
    # A flavor must explicitly consume split_kv / cta_mma. Accepting either
    # elsewhere would silently ignore it; for split_kv that is also WRONG: the
    # caller sizes an (S*B)-batch partial workspace and runs the combine, while the kernel keeps
    # writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
    # than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
    # corrupt the result instead of dropping out.  Reject at the door.
    if k.split_kv != 1 and flavor not in _SPLIT_KV_FLAVORS:
        raise ValueError(f"{flavor}: split_kv is not implemented on this flavor (got {k.split_kv}); supported: {sorted(_SPLIT_KV_FLAVORS)}")
    # D256 selects its topology from the input family rather than exposing a
    # free CTA-MMA knob: half inputs stay CTA2, while FP8/MXFP8 use CTA1.
    d256_quantized_cta1 = flavor == "d256" and fp8 and k.cta_mma == 1
    if k.cta_mma != 2 and flavor not in _CTA_MMA_FLAVORS and not d256_quantized_cta1:
        raise ValueError(f"{flavor}: cta_mma is not selectable on this flavor (got {k.cta_mma}); supported: {sorted(_CTA_MMA_FLAVORS)}")
    if flavor == "d192" and k.split_kv > 1 and k.cta_mma != 2:
        raise ValueError("d192: split_kv > 1 is validated only with cta_mma=2")
    if k.split_kv < 1:
        raise ValueError(f"{flavor}: split_kv must be >= 1 (1 = KV-split off); got {k.split_kv}")
    if k.split_kv > 1:
        # Each of these would need extra machinery in the combine pass, so the
        # backstop rejects them rather than silently producing a wrong answer.
        if k.thd_varlen and not (
            (flavor == "d64" and k.decode_tile and k.cta_mma == 1 and not fp8)
            or (flavor == "d256" and k.cta_mma == 2 and not fp8 and (not k.pack_gqa or k.paged_kv))
            or (flavor == "d512" and k.d_flavor == 576 and k.cta_mma == 2 and not fp8)
            or (
                flavor == "d128"
                and k.cta_mma == 1
                and not fp8
                and (k.single_q_head_dim == 128 or (k.single_q_head_dim == 192 and not k.paged_kv and not k.pack_gqa))
            )
        ):
            raise ValueError(f"{flavor}: split_kv > 1 is dense-only (THD packs its own flat grid)")
        if k.has_sink and fp8:
            # The half kernels write sink-free partials under a split and the
            # combine folds the sink once; the FP8 epilogues still fold it per
            # tile, so a split would count it once per split.
            raise ValueError(f"{flavor}: split_kv > 1 with attention sink is not supported for FP8 (the sink would be counted once per split)")
    lpt_head_groups = (1, 8, 16, 32) if flavor == "d256" else (1, 8, 16)
    if k.lpt_head_group not in lpt_head_groups:
        raise ValueError(f"{flavor}: LPT_HEAD_GROUP must be one of {lpt_head_groups}; got {k.lpt_head_group}")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.pack_gqa:
        if k.thd_varlen and not (
            not fp8
            and (
                (flavor == "d64" and ((not k.decode_tile and k.cta_mma in (1, 2) and k.split_kv == 1) or (k.decode_tile and k.cta_mma == 1 and k.split_kv > 1)))
                or (
                    flavor == "d128"
                    and (
                        ((k.cta_mma == 2 or (k.cta_mma == 1 and k.paged_kv)) and k.split_kv == 1)
                        or (k.cta_mma == 1 and k.split_kv > 1 and k.single_q_head_dim == 128)
                    )
                )
                or (flavor == "d256" and k.paged_kv and k.cta_mma == 2)
                or (flavor == "d512" and k.d_flavor == 576 and k.cta_mma == 2)
            )
        ):
            raise ValueError(
                f"{flavor}: THD PackGQA requires half d128 cga2 unsplit/cga1 split, half d64 cga1/2 unsplit or cga1 split, "
                "paged half d256 cga2, or native half d576 cga2"
            )
    if k.ragged_q:
        # The decode tile's ragged-Q leg (sm100/decode_d128_f16.py): dense grid
        # over the declared batch, Q rows at the ragged offsets, final O / Stats
        # placed by the split combine -- so the split path is mandatory and the
        # K/V side must be the page pools (a ragged K/V needs the THD leg).
        if flavor != "d128" or k.cta_mma != 1:
            raise ValueError(f"{flavor}: ragged_q is wired on the d128 decode tile only (cta_mma=1)")
        if k.thd_varlen:
            raise ValueError("d128 decode: ragged_q and thd_varlen are mutually exclusive")
        if k.split_kv < 2:
            raise ValueError("d128 decode: ragged_q rides the split path (the combine places the ragged O / Stats rows); split_kv must be >= 2")
        if not k.paged_kv:
            raise ValueError("d128 decode: ragged_q serves ragged Q/O/Stats over PAGED K/V only")
        if k.seq_q_lens_present:
            raise ValueError("d128 decode: ragged_q derives per-sequence Q lengths from the ragged offsets; seq_q_lens_present is dense-only")
    if k.paged_kv:
        if flavor not in _PAGED_KV_FLAVORS:
            raise ValueError(f"{flavor}: paged_kv is not implemented on this flavor; supported: {sorted(_PAGED_KV_FLAVORS)}")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: paged_kv requires seq_kv_lens_present (the per-batch KV length bounds the block-table walk)")
        # dtype_qkv alone cannot tell per-tensor FP8 (d128 wired) from MXFP8
        # (wired on every native flavor), so
        # the dtype family is NOT gated here: every kernel file WITHOUT the
        # PAGED_KV specialization raises at module scope on paged_kv=True
        # (next to its softmax_f16 guard), which is the backstop that cannot
        # silently read a page pool as dense K/V.
        # A K/V tile is loaded as a stack of page-sized row boxes (or one box
        # inside a page when the page is taller than the tile). Either way a
        # box must never straddle a page, and the 128 B swizzle atom is 8 rows.
        p = k.page_size
        if p < 8 or p % 8 != 0:
            raise ValueError(f"{flavor}: page_size must be a positive multiple of 8; got {p}")
        if (p < 128 and 128 % p != 0) or (p > 128 and p % 128 != 0):
            raise ValueError(f"{flavor}: page_size must divide the 128-row KV tile or be a multiple of it; got {p}")
    elif k.page_size:
        raise ValueError(f"{flavor}: page_size requires paged_kv=True")
    if k.epilogue_gate:
        # Only the HALF d256 body of this line carries the gate seams (_EPILOGUE_GATE_FLAVORS above); the quantized d256
        # bodies and every other flavor would trace an ungated epilogue and silently return O instead of O * sigmoid(G).
        if flavor not in _EPILOGUE_GATE_FLAVORS or fp8:
            raise ValueError(
                f"{flavor}: epilogue_gate is wired on the f16/bf16 d256 body of this line only (the paged / THD / PackGQA prefill compiled "
                f"for cc 10.7); the quantized d256 bodies and the other flavors are served by the SM107 d256 kernels (config_sm107)"
            )
        if k.split_kv > 1:
            # A split's fp32 partials bypass the gated epilogue and the combine would write the un-gated O (the d256 decode
            # tile's split applies the gate in ITS combine; this prefill body has no such arm).
            raise ValueError(f"{flavor}: epilogue_gate with split_kv > 1 is not served on this body (the split partials bypass the gated epilogue)")
    if getattr(k, "mma_2x2", False):
        # The 2x2-datapath kernel exists for the half d512 flavor only (sm100/prefill_d512_f16_2x2.py).
        if flavor != "d512":
            raise ValueError(f"{flavor}: mma_2x2 selects the d512 2x2-datapath kernel; the other flavors do not consume it")
        if fp8:
            raise ValueError("d512: mma_2x2 is wired for BF16/FP16 inputs only (the fp8 / mxfp8 d512 kernels stay role-split)")
        if k.paged_kv and k.d_flavor != 576 and not d512_paged_2x2_wired:
            raise ValueError("d512: mma_2x2 does not serve paged KV; the three-chunk d576 specialization is required")
        if k.pack_gqa and not d512_partial_pack_wired and (k.qh_per_kh <= 0 or _D512_2X2_TILE_M % k.qh_per_kh != 0):
            raise ValueError(f"d512: mma_2x2 packs whole GQA groups into its {_D512_2X2_TILE_M}-row tile; qh_per_kh ({k.qh_per_kh}) must divide it")


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding: the kernels
    fold trace-time branches on these bits; the band VALUES ride the CFG's
    WINDOW_LEFT / WINDOW_RIGHT fields (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present or params.kv_tail_mask:
        flags |= MASK_PADDED
    return flags


def bpe(dtype: int) -> int:
    """Bytes per STORAGE element: the block-scaled O codes are byte containers
    (E2M1 packs two per byte -- see ``o_pack_div``)."""
    if dtype in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_O_NVFP4, DTYPE_O_MXFP8):
        return 1
    return 2


def o_pack_div(dtype_o: int) -> int:
    """Logical O elements per storage byte-container: 2 for E2M1, else 1."""
    return 2 if dtype_o == DTYPE_O_NVFP4 else 1


def o_row_bytes(tile_o: int, dtype_o: int) -> int:
    return tile_o * bpe(dtype_o) // o_pack_div(dtype_o)


def tile_k_hw(dtype_qkv: int) -> int:
    if dtype_qkv <= 1:
        return 64
    return 16


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} not multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int, pack_div: int = 1) -> int:
    return 128 if (tile_o * bpe_o // pack_div) % 128 == 0 else 64


def rescale_threshold(dtype_qkv: int) -> float:
    return 4.0 if dtype_qkv <= 1 else 8.0


def pack_gqa_group_size(qh_per_kh: int, tile_m: int = 128, *, partial: bool = False, requires_divisible: bool = True) -> int:
    """Heads packed into one Q tile row-group for the GQA ratio ``G = qh_per_kh``.

    Full contract (``partial=False``): ``G`` when it divides ``tile_m``, else 0
    (nothing can be packed).  Partial contract (``partial=True``, the d128 and
    d256 f16 kernels): the largest divisor of ``G`` that divides ``tile_m`` --
    ``gcd(G, tile_m)`` -- so a 12-head group packs 4 heads per token row-group
    (three packed heads per KV head), a 6-head group packs 2, and a group with
    no factor in common with the tile (3, 5, ...) yields 1 = unpacked.
    Without the divisibility requirement (``requires_divisible=False``, the
    SM120 kernels): ``G`` whenever it fits the tile, ``tile_m % G != 0`` included --
    ``tile_m // G`` tokens per tile, the rows past ``(tile_m // G) * G`` dead
    (G = 6 at 64 rows: ten tokens in 60 live rows) -- else 0.  MHA (``G == 1``)
    is the identity, 1.
    """
    if qh_per_kh <= 0:
        return 0
    if tile_m % qh_per_kh == 0 or (not requires_divisible and qh_per_kh <= tile_m):
        return qh_per_kh
    return math.gcd(qh_per_kh, tile_m) if partial else 0


def pack_gqa_supported(h_q: int, h_kv: int, tile_m: int = 128, *, partial: bool = False, requires_divisible: bool = True) -> bool:
    """Whether a PACK_GQA=1 plan is HONORABLE for this head count: the group
    divides the tile (full packing), -- ``partial`` -- shares a factor > 1
    with it, or -- without ``requires_divisible`` -- fits it.  MHA (G == 1) is the bit-exact
    unpacked fold, so it is honorable too; a group that cannot pack at all
    (G=3 at tile_m=128) is declined rather than silently run unpacked
    (frost/README.md rule 5)."""
    if h_q <= 0 or h_kv <= 0 or h_q % h_kv != 0:
        return False
    g = h_q // h_kv
    p = pack_gqa_group_size(g, tile_m, partial=partial, requires_divisible=requires_divisible)
    return p == g or p > 1


def _pack_g(params: TemplateParams, tile_m: int, *, partial: bool) -> int:
    """``Cfg.PACK_G`` for a flavor: the packed group size, validated the way
    ``engines.mismatch`` admits it (full contract, or partial where the kernel
    carries it), so a knob that passed the gate can never reach a kernel that
    would silently run it unpacked."""
    if not params.pack_gqa:
        return 1
    g = int(params.qh_per_kh)
    p = pack_gqa_group_size(g, tile_m, partial=partial)
    if p == 0 or (p == 1 and g != 1):
        how = "share a factor with" if partial else "divide"
        raise ValueError(f"qh_per_kh ({g}) must {how} TILE_M ({tile_m}) when PACK_GQA is enabled")
    return p


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"inner {inner_bytes} not multiple of swz {swz_b}")
    return inner_bytes // swz_b


@dataclass(frozen=True)
class TmaIters:
    """TMA iteration granularity derived from a flavor's tile geometry."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, getattr(cfg, "BPE_V", cfg.BPE), cfg.V_SWZ_BYTES)
    return TmaIters(
        QK_ITERS=qk,
        VO_ITERS=vo,
        QK_GRANU_ELEMS=cfg.TILE_K // qk,
        VO_GRANU_ELEMS=cfg.TILE_O // vo,
    )
