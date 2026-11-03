# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped THD d128 accumulation geometry on the cta_group::2 datapath: the dQ image in S^T's dead tail with FAv4's rotated MMA order.

Layout-only record for the pair body of ``kernels/sm100/bprop_d128_f16.py`` (``bprop_d128_accum_grouped_2cta_stail_kernel``, the grouped THD route's cta_group::2 implementation) on top of the
base pair geometry ``config_sm100_d128_accum_2cta.Config``.  Every byte offset, TMEM column and ring depth derives from the geometry below
and is re-derived by :func:`validate_config` at import; the barrier tables (``BAR_LAYOUT``, ``BAR_LAYOUT_XFA4``) are the barrier audit's
tables, pasted, never re-derived.

What the body does against the base geometry:

* TMEM: the dQ half-row image (64 fp32 columns, the M = 128 pair MMA's D image) sits in S^T's dead tail [64, 128) (the 64 columns the
  packed P^T does not reuse): ``DQ_COL = S_COL + PACK_COLS``; [192, 256) is DEAD.  Two alias edges on the S tail: the WRITE edge dQ(t-1) <-
  the softmax's read of S^T(t) (barrier ``s_read``) and the REUSE edge S^T(t+1) <- the epilogue's readout of dQ(t-1) (barrier
  ``s_tail_empty``); ``dp_empty`` keeps its count and waiter, its arrive sits at the dS^T readout.
* TMEM MAP (the typed op-level field ``tmem_map``, PART-KEYED; kernel header sections 9 / 11): ``"s_tail"`` = the map above (THE default on the
  512-column parts cc 10.0 / 10.3: ``TMEM_COLS`` 512, the dQ image in S^T's dead tail behind the two LEADER rows) | ``"dq_576"`` = the INDEPENDENT dQ
  image at [DK_COL + DK_COLS, +DQ_COLS) = [512, 576) of the 576-column carve (``tmem_alloc(TMEM_COLS = 576, is_exclusive=True)`` on BOTH CTAs, the
  SAME constant at the dealloc; selectable BY NAME on cc 10.7, declined BY NAME on the 512-column parts; NOT a default anywhere: a NEUTRAL structural
  map -- the ceiling probe of its two rows (s_tail_wait_off) read +0.0 / +0.55 % at 16K and -0.3 / +0.27 % at 32K (step ms / main-kernel cycles), the combined s_tail + s_read probe -0.4 / -0.2 % (benchmarker, cc 10.7, A/B/A x 3 x 100 both build orders, 2026-11-03): inside noise, below the +2.0 % threshold -- kept as the base of the two-stage staging arm, whose own row decides a flip): the S tail has no
  second tenant, so the rows ``s_read`` and ``s_tail_empty`` are DELETED with their five leader wait sites, the leader's state, the drain and the
  two arrives (``BAR_LAYOUT_DQ576`` / ``BAR_LAYOUT_RING_DQ576`` = the audited table minus ``DELETED_BARRIERS_DQ576``, re-packed: 33 / 35 names);
  the new REUSE edge dQ(t) readout -> dQ(t+1) write rides the epilogue's PROGRAM ORDER (its dq_done(t) wait + readout precede its next tile's dS
  STS + ``ds_smem.arrive()``, the dQ issue's gate) -- pinned as a source assertion by the test module, voided by any staging-before-readout reorder
  and by ``16w_fa4`` (which declines the map by name); SMEM, every slab and every other TMEM tenant unchanged; the three S-tail WAIT probes decline
  by name under the map (their sites are the deleted rows).  Records: ``CFG_DQ576`` / ``CFG_RING_DQ576`` (``cfg_for_part`` resolves per cc).
* ORDER: the leader issues FAv4's rotated stream per q tile: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); the tail issues dQ(hi-1).
* SMEM: Q1 one stage (FAv4's ``Q_stage = 1`` under 2-CTA; -16 KiB) funds a DEDICATED dQ fp32 staging ring ``sDQ`` of ``DQ_SLOTS`` = 2 x
  [64 q x 32 d] SW128 slices (the reduce-add box), two rounds per tile; no dQ staging alias on ds_own + XCHG (under the rotated order
  dq_done(t) lands ~150 clk before the dS(t+1) STS, so an alias would gate the dS exchange by the reduce-add drain).  Total 231424 B
  (the SMEM audit PASS, bank check PASS).
* dQ ROUTE: the sDQ ring STS + the store warp's four ``tma_reduce_add`` per tile into the fp32 ``dq_accum`` workspace (the one-CTA chain's
  route); the fp32 partial is summed in a non-deterministic order across the pairs and folded by ``cast_dq_thd`` in the next launch.
* DIAGNOSTIC PROBES + STAMPS: the ``diag_2cta`` values this body has code for (the code space ``config_sm100_d128_accum_2cta.CFG.DIAG_2CTA_NAMES``)
  -- ``"dq_reduce_add_off"`` (the store warp's reduce-add issues skipped), ``"exchange_off"`` (the XCHG ship skipped and ds_recv_full armed
  with 0 tx bytes, EVERY wait kept = the SAFE exchange-family ceiling), ``"stats_one_chunk"`` (one LDS.128 stats / delta chunk per quarter
  reused for all eight), ``"sdq_sts_off"`` (the sDQ slice STS + its proxy fence skipped), ``"s_tail_wait_off"`` (the leader's four
  s_tail_empty WAIT sites skipped, every arrive kept), ``"s_read_wait_off"`` (the leader's s_read WAIT skipped, the arrive kept) and
  ``"s_tail_and_s_read_wait_off"`` (both; the ceiling of the whole S-tail alias) -- and the typed bool field ``debug_clk`` (the
  STAMPS lever, the d512 bodies' ``debug_clk`` pattern): every mbarrier wait of ONE representative warp per role is bracketed by %clock64
  and accumulated per (role, barrier) into the Int32 words of the bar block's spare (``DBG_CLK``, below), with the epilogue's per-tile
  hand-off segments, dumped per CTA at exit into a workspace region the adapter carves only under the lever; output CORRECT, timing
  diagnostic; default False = no traced code (the default rendering stays byte-identical).
* MASK FORM: the typed field ``mask_form`` = ``"fsel"`` (THE default: the keep-word mask selected on every tile) | ``"predicated"`` (the same
  mask body only on a tile that can hold a masked cell; measured neutral) | ``"diag_mask_off"`` (DIAGNOSTIC: the mask body skipped everywhere).
* ROLE SHAPE: the typed field ``warp_roles`` = ``"16w"`` (THE default: the shipped rendering, byte-identical) | ``"16w_fa4"`` (an alternate
  shape kept for follow-up work: the SOFTMAX writes sDS own half + the XCHG half straight from its 32 packed dS^T words -- the LDTM image
  word for word, the same SW128 ``(row % 8) ^ vec`` STS form -- fences ONCE, arrives ``ds_smem`` (256 = SOFTMAX_LANES), passes the 256-lane
  XCHG named barrier and its lead warp's elected lane arms ``ds_recv_full`` + ships; a softmax WAR wait on ``dq_war_done`` (``dq_war`` st(1),
  one wait per live tile) precedes the first dS STS of every tile; the EPILOGUE warps 8-11 become the REDUCE group (dq_done -> x64 dQ ld ->
  s_tail_empty -> rounds A, B; the item-end dV / dK staging kept; no dS work); ``ds_tmem`` AND ``dp_empty`` are DELETED and ``dq_war_done`` is
  ADDED (``BAR_LAYOUT_XFA4``: 33 names / 38 slots, END 230928, re-packed); the dQ ring keeps form A (the store warp issues); s_read kept;
  TMEM / SMEM / the 144-144-80 split UNCHANGED; measured slower than 16w on GB200 at 8K-64K).  The kernel's ``warp_roles_code``
  Constexpr[int] (``WARP_ROLES_CODES``); the compile key carries the NAME.  Under ``16w_fa4`` the STAMPS lever and ``dkv_accum =
  "fp32_reduce"`` are not built: both decline BY NAME.
"""

from dataclasses import dataclass, replace

from .config_sm100_d128_accum_2cta import CFG as CFG_BASE
from .config_sm100_d128_accum_2cta import Config, resolve_diag_2cta

# chain-level truth: the admitted parts (the TMEM map's part keys), the tma_ring stats region
from .d128_grouped_2cta_contract import CTA_PAIR_CCS, STATS_KINDS, stats2_pad_total, stats2_shape_strides  # noqa: F401


@dataclass(frozen=True)
class ConfigSTail(Config):
    """The first body's geometry with the arm-(ii) deltas; every inherited property re-derives from the fields below."""

    # Q1 single stage (FAv4 2-CTA form) funds sDQ 16 KiB; Q1(t+1) window ~2000 clk
    Q1_STAGES: int = 1
    # --- the dedicated dQ staging ring: DQ_SLOTS slices of DQ_SLICE_BYTES, DQ_ROUNDS rounds per tile (DQ_SLOTS * DQ_ROUNDS == DQ_SLICES) ---
    # sDQ 16 KiB dedicated ring, 2 x [64 q x 32 d] fp32 SW128; dS STS waits no dq_empty
    DQ_SLOTS: int = 2
    DQ_ROUNDS: int = 2
    # --- the softmax's MASK FORM: fsel (default; measured neutral against predicated) | predicated | diag_mask_off (diagnostic) ---
    MASK_FORM_DEFAULT: str = "fsel"
    MASK_WORD_COLS: int = 32  # columns per keep-word = tile_dsl.mask.MASK_WORD_COLS (the body asserts the two agree at import)
    # --- the TMEM MAP (the typed op-level field ``tmem_map``, PART-KEYED; the module docstring): "s_tail" = the dQ image in S^T's dead tail (this record
    # and CFG_RING: the 512-column parts' rendering) | "dq_576" = the independent image past dK of the 576-column carve (CFG_DQ576 / CFG_RING_DQ576:
    # TMEM_COLS 576, TMEM_IS_EXCLUSIVE; cc 10.7 only, selectable by name).  The record's own map is its identity (the resolution token, the record names). ---
    TMEM_MAP: str = "s_tail"
    TMEM_IS_EXCLUSIVE: bool = False  # the SKU's alloc form for a request past 512 columns (`tmem_alloc(..., is_exclusive=True)`); True iff TMEM_COLS > 512
    # --- the SMEM MAP (the typed op-level field ``smem_map``, PART-KEYED; kernel header sections 5 / 8 / 11): "sm100" = this record's slab chain (the
    # 232448-B opt-in cap of every part: THE default on every part, byte-identical) | "rubin_b2" = the 327 KiB oversized carveout of cc 10.7 ON the
    # dq_576 TMEM map (CFG_RING_DQ576_B2; selectable BY NAME on cc 10.7, not a default anywhere until its A/B/A row): sQ1 x2, sDO1 x2, sDS x2 + sDS_xchg
    # x2 (the TWO-STAGE dS / exchange staging ring: the epilogue stages dS(t) before it waits dQ(t-1), the leader gates every dQ issue on the NEW LEADER
    # row dq_read_done) = 314368 B.  Every slab offset derives from the ring depths below; the SMEM cap is the part's (SMEM_CAP).  The record's own
    # map is its identity (the resolution token). ---
    SMEM_MAP: str = "sm100"
    DO1_STAGES: int = 1  # the sDO1 ring's depth (the base chain's DO_STAGES keeps sDO2's single stage); 2 under rubin_b2
    DS_STAGES: int = 1  # the sDS ring's depth (the dQ A operand: own half by STS, the peer's half by its ship); 2 under rubin_b2
    XCHG_STAGES: int = 1  # the sDS_xchg ring's depth (the ship's source) == DS_STAGES (an sDS-only ring decouples nothing: the XCHG WAR has no local signal)
    # dead bytes declared ABOVE the scaffold to the launcher only (the DIAGNOSTIC map sm100_oversized: the sm100 map itself, launched in the oversized
    # carveout); every address of the map (slabs, bar block, TMEM handle, CLC payload) is the unpadded map's -- SMEM_BYTES alone grows
    SMEM_PAD_BYTES: int = 0
    SMEM_OVERSIZED_CAP: int = 334848  # cc 10.7's per-CTA oversized SMEM cap (the SKU card's attr 150 figure); the DSL launcher sets the mode from the slab size
    # --- the STAMPS lever (typed adapter field ``debug_clk``; header section 11): default OFF = the shipped rendering, byte-identical ---
    DEBUG_CLK_DEFAULT: bool = False
    # (the record's word count is DERIVED per (roles, stats_form) as the WHOLE bar-block spare (TMEM_HANDLE - BAR_LAYOUT_END) / 4: 96 under ldg
    # (82 used), 88 under tma_ring (84 used: + the two ring wait words) -- never a literal; ``DBG_CLK`` / ``DBG_CLK_RING`` below)
    DBG_CLK_BLOCKED_THRESH: int = 128  # a wait above this many clk really waited (a first-test pass costs ~30-60 clk of issue): the d512 bodies' value
    DBG_CLK_HIST_EDGES: tuple = (100, 400, 1600)  # clk bins of the per-tile "ship issue - ds_tmem pass" segment (bin 0 = under 100 clk)
    DBG_CLK_MAGIC: int = 0x5354414D  # "STAM": the record's first word (a decoder refuses a record without it)
    DBG_CLK_RECORD_ALIGN: int = 256  # the stamps region starts at the first 256-B boundary past the chain's workspace regions
    # --- the DIAGNOSTIC probes' constants (header section 11; OUTPUT WRONG BY DESIGN under a probe) ---
    DIAG_EXCHANGE_OFF_TX: int = 0  # exchange_off: ds_recv_full armed with ZERO tx bytes -> every phase completes at its own arm (no copy is ever in flight)
    DIAG_STATS_CHUNK: int = 0  # stats_one_chunk: the ONE 4-wide stats / delta chunk of each quarter that is loaded and reused for all GROUP_QUARTER // 4 chunks
    # s_tail_wait_off / s_read_wait_off: the leader's WAIT sites the probe skips (header section 5 / 11; the barrier table is unchanged: every
    # arrive and every PipelineState advance stays).  s_tail_empty: the prologue S^T(lo), the lookahead S^T(t+1), the tail dQ(hi-1), the P15
    # drain = 4 static `_wait_tcgen` sites; s_read: slot 2 of every tile = 1 site.  The render census drops exactly these many try_wait /
    # bar.warp.sync / tcgen05.fence::after_thread_sync.
    DIAG_S_TAIL_WAIT_SITES: int = 4
    DIAG_S_READ_WAIT_SITES: int = 1
    # --- the ROLE SHAPE: 16w (default = the shipped rendering) | 16w_fa4 (the alternate shape; module docstring) ---
    WARP_ROLES_DEFAULT: str = "16w"
    # XCHG_BARRIER lanes: SOFTMAX_LANES 256 under 16w_fa4, EPILOGUE_LANES 128 under 16w
    SOFTMAX_LEAD_WARP: int = 0  # 16w_fa4: the softmax warp whose elected lane arms ds_recv_full and ships XCHG (FAv4's tidx 0; inspector X4)
    # --- the STATS FORM (typed adapter field ``stats_form``; kernel header section 11): how the softmax gets the q tile's LSE / delta rows ---
    #   "ldg" (the previous default, selectable; the form the "16w_fa4" role shape resolves to when the field is unset, the ring rows being the
    #         16w bundle's): the softmax's per-tile `_stats` -- one LDG per lane (LSE for group 0, delta for group 1), the FMUL scale, one STS
    #         into the single sSTATS / sDELTA slab, inside a `bar.sync SOFTMAX_BARRIER, 256` pair.
    #   "tma_ring" (THE default since the day-4 measurement -- +1.8 / +1.9 % at the step at 16K / 32K, +1.4..+1.9 % at every served cell, +3.3 %
    #         on the graph replay against "ldg", bitwise the same outputs; FAv4's form: the LSElog2 preprocess + a bulk copy per tile): the
    #         chain's delta launch writes the two stat rows PRE-SCALED
    #         (-log2(e) * LSE with the -inf -> 0 guard, -scale * delta: the SAME two fp32 multiplies the ldg softmax performs, once per q row) into a
    #         q-tile-PADDED workspace region (pad rows 0.0 = the ldg form's `stat = 0` for a row past the sequence, bit for bit); the LOADER warp
    #         bulk-copies ONE 1 KiB (head, q tile) record per tile into stage s of a STATS_STAGES-deep SMEM ring one tile ahead; `stats_full[s]`
    #         (TMA_LOAD, expect_tx STATS_TX) / `stats_empty[s]` (THREAD, the 8 softmax warps) replace the LDG chain, the STS and both named-barrier
    #         fences; `_p_values` / `_ds_values` read the stage by the same LDS.128 broadcast form.  The ring's second stage is the 1 KiB of
    #         cap slack past today's SMEM_BYTES: the scaffold follows it (+1024, every offset derived from SLABS_END) and SMEM_BYTES == the cap.
    STATS_FORM_DEFAULT: str = "tma_ring"  # the unset field under the default role shape (the day-4 decision; "ldg" stays selectable)
    STATS_FORM_FA4: str = "ldg"  # the unset field under "16w_fa4": the only form sited in that bundle (a NAMED "tma_ring" there declines by name)
    # RESOURCE-NOTE: stats ring 2 x 1 KiB @229376: stage 1 = the cap slack; SMEM_BYTES 232448 == cap, slack 0 (CFG_RING)
    STATS_STAGES: int = 1  # ldg: ONE slab (today's sSTATS + sDELTA); the tma_ring geometry CFG_RING = replace(CFG, STATS_STAGES=2) (the ring depth)
    SMEM_OPT_IN_CAP: int = (
        232448  # the part's per-CTA dynamic SMEM opt-in cap (the SKU card's 232448-B figure): the tma_ring footprint fills it exactly (validate_config)
    )

    @property
    def DIAG_2CTA_NAMES_ON_MAP(self):
        # The probes this body has code for (== the code space config_sm100_d128_accum_2cta.CFG.DIAG_2CTA_NAMES; every other name declines).
        # Two probes of an earlier pair body are NOT defined here: a skipped dp_empty wait measures nothing on this map (dp_empty is arrived at
        # the dS^T readout, four GEMMs before dP^T(t+1) needs it) and skipping the three dS gates removes the ONLY back-edge of the ds_recv_full
        # expect_tx arm (the arm of tile t+1 lands on a tx-pending phase = an over-arrival fault); the safe exchange ceiling is exchange_off.
        #   dq_reduce_add_off (dq_reduce_add_off): the store warp's four reduce-add ISSUES skipped (ring protocol kept; dQ zero)
        #   exchange_off      (exchange_off):     the epilogue's XCHG ship skipped + ds_recv_full armed with DIAG_EXCHANGE_OFF_TX = 0 bytes; the three dS
        #                                gates, the relay, the XCHG STS / proxy fence / named barrier and every other wait KEPT (the recv half
        #                                is never written: dQ wrong) = the exchange family's ceiling {lever 1, lever 2's exchange part, #4}
        #   stats_one_chunk   (stats_one_chunk):     _p_values / _ds_values load ONE stats / delta chunk per quarter (DIAG_STATS_CHUNK) and reuse it for all
        #                                eight: -DIAG_STATS_LDS_REMOVED_PER_LANE of DIAG_STATS_LDS_PER_LANE LDS.128 per lane per tile, registers /
        #                                compute / waits unchanged (the byte-removal-only residual test of the SMEM-pipe reading; I1's ceiling)
        #   sdq_sts_off       (sdq_sts_off):      _dq_round skips the sDQ slice STS + its proxy fence (the UTMAREDG reads stale staging; -256 wavefronts;
        #                                the dq_empty waits and the dq_full arrive stay) = the second point of the wavefront -> time slope
        #   s_tail_wait_off   (s_tail_wait_off):  the LEADER skips its DIAG_S_TAIL_WAIT_SITES s_tail_empty waits (D1 i / ii / iii + the P15 drain);
        #                                the epilogue's pair-wide arrive and the leader's s_tail_state advance stay.  s_tail_empty is a LEADER
        #                                relaxed thread-count row (256 bare arrives, no tx bytes): the phases flip unobserved, an arrive on a
        #                                phase nobody waits is not an over-arrival.  S^T(t+1) / the tail dQ may overwrite an unread dQ(t-1):
        #                                dQ WRONG BY DESIGN = the ceiling of the s_tail_empty edge (what an independent dQ image removes)
        #   s_read_wait_off   (s_read_wait_off):  the LEADER skips its DIAG_S_READ_WAIT_SITES s_read wait (slot 2, every tile); the softmax's
        #                                512-lane arrive and ready_state stay (the same relaxed-row argument).  dQ(t-1) may land under the
        #                                softmax's S^T(t) read: S / dQ WRONG BY DESIGN = the ceiling of the s_read edge
        #   s_tail_and_s_read_wait_off:  both guards at once in ONE rendering (no third code path) = the ceiling of the whole S-tail alias
        return (
            "dq_reduce_add_off",
            "exchange_off",
            "stats_one_chunk",
            "sdq_sts_off",
            "s_tail_wait_off",
            "s_read_wait_off",
            "s_tail_and_s_read_wait_off",
        )

    # --- derived: the stats_one_chunk probe's counts (the softmax's stats / delta reads of header section 4: 4-wide chunks per q quarter, two quarters per group) ---
    @property
    def DIAG_STATS_CHUNKS_PER_QUARTER(self):
        return self.GROUP_QUARTER // 4  # 8 LDS.128 per quarter per lane: one 16-B chunk per 4 q columns

    @property
    def DIAG_STATS_LDS_PER_LANE(self):
        # per lane per tile: _p_values (LSE) + _ds_values (delta), QUARTERS_PER_GROUP quarters each
        return 2 * self.QUARTERS_PER_GROUP * self.DIAG_STATS_CHUNKS_PER_QUARTER  # 32

    @property
    def DIAG_STATS_LDS_REMOVED_PER_LANE(self):
        return self.DIAG_STATS_LDS_PER_LANE - 2 * self.QUARTERS_PER_GROUP  # 28 of 32 (-448 of 512 wavefronts per tile per CTA at 16 warps)

    @property
    def MASK_FORMS(self):
        # fsel: today's rendering -- band_mask_words + apply_mask_words (one arith.select per cell: R2P + FSEL) on EVERY q tile of every item.
        # predicated: the SAME mask body under a CTA-uniform runtime branch that runs it only when the tile can hold a masked cell
        # (`tile_may_mask`: the causal diagonal straddle, the last partial q tile, kv rows at or past seq_k); an interior tile packs the exp2
        # values straight (0 FSEL / R2P / keep-word per lane).  diag_mask_off: the mask body skipped on EVERY tile = the ceiling of the lever,
        # OUTPUT WRONG BY DESIGN (a DIAGNOSTIC value, never shipped armed; the driver's --validate refuses its arm).  P and dS are BITWISE
        # between fsel and predicated by construction (a masked cell is exactly +0.0 in both; an interior tile has no masked cell), hence dQ
        # (per reduce-add order) / dK / dV.  No barrier, SMEM or TMEM change under any value (kernel header section 11).
        return ("fsel", "predicated", "diag_mask_off")

    @property
    def MASK_FORM_DIAGNOSTIC(self):
        return "diag_mask_off"  # the one OUTPUT-WRONG-BY-DESIGN value of MASK_FORMS (the deletion probe of the mask-form lever)

    # --- derived: the softmax lane's per-tile mask cost (header section 11; the FSEL / R2P census rows and the PTX selp delta of the forms) ---
    @property
    def MASK_SELECTS_PER_LANE_PER_TILE(self):
        return self.QUARTERS_PER_GROUP * self.GROUP_QUARTER  # 64: one select per cell of the lane's two q quarters

    @property
    def MASK_WORDS_PER_LANE_PER_TILE(self):
        return self.QUARTERS_PER_GROUP * ((self.GROUP_QUARTER + self.MASK_WORD_COLS - 1) // self.MASK_WORD_COLS)  # 2: one keep-word per quarter

    @property
    def WARP_ROLES(self):
        # 16w: softmax 0-7 / epilogue 8-11 (dS hop + dQ) / 4 service warps = today.  16w_fa4: softmax 0-7 (dS^T -> TMEM AND sDS own + XCHG from the
        # same registers, the ship) / REDUCE 8-11 (dQ readout + rounds + the item-end dV / dK staging) / the same 4 service warps.  The 20-warp
        # lever-2 shape (rank 2) is NOT built here (its split 144/64/64/64 at entry 96: the resource plan).
        return ("16w", "16w_fa4")

    # --- derived: the 16w_fa4 role names and lane counts (the SAME warps 8-11 and lanes; renamed: the barrier audit "RENAMED" rows) ---
    @property
    def REDUCE_WARPS(self):
        return self.EPILOGUE_WARPS  # 4: warps 8-11 = the reduce group under 16w_fa4

    @property
    def REDUCE_LANES(self):
        return self.EPILOGUE_LANES  # 128

    @property
    def PAIR_REDUCE_LANES(self):
        return self.PAIR_EPILOGUE_LANES  # 256 = s_tail_empty's init (the same lanes as the 16w epilogue)

    @property
    def REDUCE_LEAD_WARP(self):
        return self.SOFTMAX_WARPS  # 8: the former epilogue lead warp; unused under the dQ ring's form A (the store warp issues) except as the role name

    def xchg_barrier_lanes(self, roles):
        """``bar.sync CFG.XCHG_BARRIER, n``: the threads whose XCHG STS + proxy fence the ONE elected DSMEM ship reads (P11 KEEP) -- the 4 epilogue
        warps under ``16w``, the 8 softmax warps under ``16w_fa4`` (inspector X3; never a literal)."""
        return {"16w": self.EPILOGUE_LANES, "16w_fa4": self.SOFTMAX_LANES}[roles]

    def ds_smem_lanes(self, roles):
        """``ds_smem``'s init = its bare arrivers: the 4 epilogue warps (``16w``) or the 8 softmax warps (``16w_fa4``; inspector X2)."""
        return {"16w": self.EPILOGUE_LANES, "16w_fa4": self.SOFTMAX_LANES}[roles]

    # --- derived: the softmax's dS STS image under 16w_fa4 (the resource plan: group g writes vec 4g .. 4g+3 of BOTH 64-q halves) ---
    @property
    def DS_STS_VECS_PER_QUARTER(self):
        return self.PACK_QUARTER_COLS // 4  # 4: a 32-q quarter = 16 packed Int32 words = four 16-B vectors of the lane's 128-B SW128 row

    @property
    def DS_STS_PER_LANE_PER_TILE(self):
        # dS STS from the 32 packed words = today's LDTM image; 4 own + 4 XCHG STS.128/lane
        return self.QUARTERS_PER_GROUP * self.DS_STS_VECS_PER_QUARTER  # 8 per softmax lane per tile = 64 warp-STS per CTA per tile (today's 4 x 16)

    # --- derived: the slab chain gains the sDQ ring between XCHG and STATS (Q -16384, +DQ 16384: SLABS_END unchanged); the sDO1 / sDS / sDS_xchg
    # rings have their OWN depths (the SMEM map): sm100 = one stage each (today's offsets), rubin_b2 = two stages each ---
    @property
    def DO2(self):
        return self.DO1 + self.DO1_STAGES * self.DO1_BYTES  # the sDO1 ring's own depth (the base chain's DO_STAGES keeps sDO2's single stage)

    @property
    def XCHG(self):
        return self.DS + self.DS_STAGES * self.DS_BYTES  # 196608 (sm100) / 262144 (rubin_b2: past the version-0 descriptor line; XCHG is no tcgen05 root)

    @property
    def ALIAS_BYTES(self):
        return self.DQ - self.DS  # the dS / exchange region [DS, DQ): 48 KiB (sm100) / 96 KiB (rubin_b2); the dK staging tenants are its STAGE-0 bytes

    @property
    def DQ(self):
        return self.XCHG + self.XCHG_STAGES * self.XCHG_BYTES  # 212992 (sm100) / 294912 (rubin_b2)

    # --- derived: the ring stages' byte bases (stage s of a ring; stage 0 = the base chain's single slab) ---
    def q1_stage(self, stage):
        return self.Q1 + stage * self.Q1_BYTES

    def do1_stage(self, stage):
        return self.DO1 + stage * self.DO1_BYTES

    def ds_stage(self, stage):
        return self.DS + stage * self.DS_BYTES

    def xchg_stage(self, stage):
        return self.XCHG + stage * self.XCHG_BYTES

    def ds_own_stage(self, rank, stage):
        """Own kv half of sDS stage ``stage`` (kv-global row order on both CTAs); ``ds_own(rank)`` = stage 0 (the dK staging's alias half)."""
        return self.ds_stage(stage) + rank * self.DS_HALF_BYTES

    def ds_recv_stage(self, rank, stage):
        """The peer's kv half of sDS stage ``stage``: landed by the peer's ship of ITS tile on that stage, never written locally."""
        return self.ds_stage(stage) + (1 - rank) * self.DS_HALF_BYTES

    @property
    def SMEM_MAPS(self):
        return ("sm100", "rubin_b2", "sm100_oversized")  # the typed domain of ``smem_map`` (SMEM_MAP_CODES below is its kernel code space, in this order)

    @property
    def SMEM_CAP(self):
        """The per-CTA SMEM cap the map's slab must fit (validate_config: SMEM_BYTES <= SMEM_CAP): the opt-in cap (sm100) | the oversized cap (rubin_b2)."""
        if self.SMEM_MAP == "sm100":
            return self.SMEM_OPT_IN_CAP
        if self.SMEM_MAP in ("rubin_b2", "sm100_oversized"):
            return self.SMEM_OVERSIZED_CAP
        raise ValueError(f"SMEM_MAP must be one of {self.SMEM_MAPS}; got {self.SMEM_MAP!r}")

    @property
    def SMEM_BYTES(self):
        """The dynamic SMEM the kernel declares (the launcher's size): the slab chain + the scaffold (+ SMEM_PAD_BYTES of dead space under the
        DIAGNOSTIC map sm100_oversized: the launcher then takes the oversized carveout for the sm100 map itself)."""
        return self.SLABS_END + self.SCAFFOLD_BYTES + self.SMEM_PAD_BYTES

    @property
    def SCHED_PAYLOAD(self):
        # the CLC payload ends the SCAFFOLD, below the pad: every address of a padded map is the unpadded map's (the body renders byte-identical)
        return self.SLABS_END + self.SCAFFOLD_BYTES - self.SCHED_PAYLOAD_BYTES

    @property
    def TCGEN05_V0_ADDR_LIMIT(self):
        return (1 << 14) * 16  # 262144: the version-0 tcgen05 SMEM descriptor's 14-bit start-address field in 16-B units (a root at or past it needs v1)

    def desc_roots(self):
        """Every tcgen05 descriptor ROOT of the map in declaration order (the Q1 stages, Q2, the dO1 stages, dO2, K1, K2's two 128-row boxes, V, the dS
        stages): validate_config keeps each below TCGEN05_V0_ADDR_LIMIT (the inherited DESC_VERSION 1 would cover more; sDS_xchg, sDQ and the stats ring
        past the line are bulk-copy / UTMAREDG / LDS tenants, never roots)."""
        return tuple(
            [self.q1_stage(s) for s in range(self.Q1_STAGES)]
            + [self.Q2]
            + [self.do1_stage(s) for s in range(self.DO1_STAGES)]
            + [self.DO2, self.K1, self.K2, self.K2 + self.K2_HALF_BYTES, self.V]
            + [self.ds_stage(s) for s in range(self.DS_STAGES)]
        )

    @property
    def DQ_BYTES(self):
        return self.DQ_SLOTS * self.DQ_SLICE_BYTES  # 16384

    @property
    def DQ_STAGING_BYTES(self):
        return self.DQ_BYTES  # the dedicated ring IS the staging (the first body: 32 KiB aliased)

    @property
    def STATS(self):
        return self.DQ + self.DQ_BYTES  # 229376 (unchanged)

    def dq_slot(self, slot):
        """Byte base of staging slot ``slot`` (one [Q_HALF x DQ_BOX] fp32 SW128 slice); slot s is written by the epilogue warp pair of d half s."""
        return self.DQ + slot * self.DQ_SLICE_BYTES

    def dq_use(self, u):
        """Store-warp use ``u`` of a tile (0 .. DQ_SLICES - 1) -> (slot, slice): round r = u // DQ_SLOTS, slot = u % DQ_SLOTS, slice = DQ_ROUNDS * slot + r.

        Uses per tile: (0, 0), (1, 2), (0, 1), (1, 3) -- round A reduces slices 0 and 2 (the first DQ_BOX d of each half), round B slices 1 and 3."""
        slot, r = u % self.DQ_SLOTS, u // self.DQ_SLOTS
        return slot, self.DQ_ROUNDS * slot + r

    @property
    def DQ_SLICE_ORDER(self):
        return tuple(self.dq_use(u)[1] for u in range(self.DQ_SLICES))  # (0, 2, 1, 3): the store warp's reduce-add issue order as SLICES

    def dq_slice(self, rank, index):
        raise ValueError("the s_tail map has no dQ staging alias on ds_own + XCHG (the dedicated sDQ ring: dq_slot(slot)); no other map has one")

    # --- dkv_accum = fp32_reduce (fp32_reduce; the resource plan): fp32 [M x DQ_BOX] staging slots over the dV / dK bf16 staging
    # bytes + the dS slab, two rounds per item: R1 = dV's four slices + dK's slice 0 (80 KiB) -> the store warp's drain -> dvk_stage_empty -> R2 = dK's
    # slices 1..3 (48 KiB).  The dq_empty ring is NOT touched (its pre-armed wait k <-> release k-1 accounting cannot carry an item-end credit). ---
    @property
    def DKV_ACC_SLICES(self):
        return self.D // self.DQ_BOX  # 4 reduce-add slices per fp32 [M x D] tile (dV and dK)

    @property
    def DKV_ACC_R1_SLICES(self):
        return self.DKV_ACC_SLICES + 1  # round 1: dV's four slices + dK's slice 0

    @property
    def DKV_ACC_R2_SLICES(self):
        return self.DKV_ACC_SLICES - 1  # round 2: dK's slices 1..3, after dvk_stage_empty

    @property
    def DKV_ACC_SLICE_BYTES(self):
        return self.M * self.DQ_BOX * self.BPE_ACC  # 16 KiB: fp32 [128 kv x 32 d] = one bf16 [128 kv x 64 d] staging box

    def dkv_acc_r1_slot(self, index):
        """Round-1 slot ``index``: K2's two halves (the bf16 dV staging), the dS slab's two halves (own + recv: both dead after dq_done(hi-1) until
        the next item's dS writes, which the dvk_empty chain gates), XCHG (the bf16 dK staging) -- rank-independent byte bases."""
        return (self.dv_store(0), self.dv_store(1), self.DS, self.DS + self.DS_HALF_BYTES, self.XCHG)[index]

    def dkv_acc_r2_slot(self, index):
        """Round-2 slot ``index``: K2's two halves and the dS slab's first half (dK's slices 1..3)."""
        return (self.dv_store(0), self.dv_store(1), self.DS)[index]

    # --- derived: TMEM column map PER MAP (kernel header section 9) -- s_tail: the dQ image in S^T's dead tail; dq_576: the independent image past dK ---
    @property
    def TMEM_MAPS(self):
        return ("s_tail", "dq_576")  # the typed domain of ``tmem_map`` (TMEM_MAP_CODES below is its kernel code space, in this order)

    @property
    def DQ_COL(self):
        if self.TMEM_MAP == "s_tail":
            # dQ @64..128 = S^T dead tail: s_read gates dQ(t-1), s_tail_empty gates S^T(t+1)
            return self.S_COL + self.PACK_COLS  # 64
        if self.TMEM_MAP == "dq_576":
            # dQ @512..576 = the 64 columns past dK of the 576-column carve: no second tenant, no alias edge, no row (the epilogue's program order)
            return self.DK_COL + self.DK_COLS  # 512
        raise ValueError(f"TMEM_MAP must be one of {self.TMEM_MAPS}; got {self.TMEM_MAP!r}")

    @property
    def TMEM_USED(self):
        return max(self.DK_COL + self.DK_COLS, self.DQ_COL + self.DQ_COLS)  # 512 (s_tail: dK ends the map) / 576 (dq_576: the dQ image ends it)

    @property
    def TMEM_DEAD_COL(self):
        return self.DS_COL + self.PACK_COLS  # 192: [TMEM_DEAD_COL, DV_COL) is DEAD under BOTH maps (was the first body's dQ image)

    @property
    def DIAG_2CTA_S_TAIL_PROBES(self):
        # The probes whose wait sites ARE the two S-tail rows: defined on the s_tail map only (under dq_576 the rows do not exist, so the probe would
        # render the default body = an identity arm; it declines BY NAME in validate_fields_for_roles and at the trace instead)
        return ("s_tail_wait_off", "s_read_wait_off", "s_tail_and_s_read_wait_off")

    # --- the STATS FORM (header section 11): the typed domain, the stats ring's bytes and roots (the slab chain), the padded stats region ---
    @property
    def STATS_FORMS(self):
        return ("ldg", "tma_ring")  # ldg = the previous rendering (selectable); tma_ring = the loader-fed SMEM stats ring, the default (bitwise the same stats)

    @property
    def STATS_FORM_DEFAULT_BY_ROLES(self):
        """The form an UNSET ``stats_form`` resolves to per role shape: ``STATS_FORM_DEFAULT`` under the default roles, ``STATS_FORM_FA4`` under
        "16w_fa4" (the ring's two barrier rows are sited in the 16w bundle only)."""
        return {self.WARP_ROLES_DEFAULT: self.STATS_FORM_DEFAULT, "16w_fa4": self.STATS_FORM_FA4}

    @property
    def STATS_STAGE_BYTES(self):
        return self.STATS_BYTES + self.DELTA_BYTES  # 1024: one ring stage = the LSE row (512 B) then the delta row (512 B) of one q tile

    @property
    def STATS_DELTA_OFF(self):
        return self.STATS_BYTES  # 512: the delta row's offset inside a stage (= today's DELTA - STATS)

    @property
    def STATS_TX(self):
        # RESOURCE-NOTE: STATS_TX 1024 = ONE expect_tx per stage; 1 (or 2) cp.async.bulk name the same mbar
        return self.STATS_STAGE_BYTES  # 1024: the ONE bulk copy of a (head, q tile) record lands on THIS CTA's stats_full[s] (x 1, never routed)

    @property
    def STATS_RING(self):
        return self.STATS  # 229376: stage 0 = today's sSTATS + sDELTA bytes; stage s = STATS_RING + s * STATS_STAGE_BYTES

    @property
    def STATS_RING_BYTES(self):
        return self.STATS_STAGES * self.STATS_STAGE_BYTES  # 1024 (ldg: the one slab) / 2048 (tma_ring)

    @property
    def SLABS_END(self):
        # RESOURCE-NOTE: BARS = SLABS_END 231424 under tma_ring: bar rows re-based +1024; layout = f(roles, form)
        return self.STATS_RING + self.STATS_RING_BYTES  # 230400 (ldg, == the base chain) / 231424 (tma_ring: stage 1 took the cap slack)

    def stats_stage(self, stage):
        """Byte base of ring stage ``stage`` (0 .. STATS_STAGES - 1): the LSE row at +0, the delta row at +STATS_DELTA_OFF."""
        return self.STATS_RING + stage * self.STATS_STAGE_BYTES

    def stats_lse(self, stage):
        return self.stats_stage(stage)

    def stats_delta(self, stage):
        return self.stats_stage(stage) + self.STATS_DELTA_OFF

    @property
    def STATS_KINDS(self):
        return STATS_KINDS  # 2: the LSE row and the delta row of one (head, q tile) record of the padded stats region (the contract's truth)

    @property
    def STATS_RECORD_BYTES(self):
        return self.STATS_KINDS * self.N * self.BPE_ACC  # 1024: one (head, q tile) record of the padded region = one SMEM ring stage image

    @property
    def STATS_PAD_GRAN(self):
        return self.N  # 128: every sequence's stat rows are padded to a whole q tile (write_thd_row_offsets gran); pad rows hold 0.0


CFG = ConfigSTail()
# The tma_ring geometry: the SAME record with the ring depth 2 -> SLABS_END 231424, SMEM_BYTES 232448 (== the cap), BARS / TMEM_HANDLE /
# SCHED_PAYLOAD re-based +1024 by derivation; every other property equal (validate_config asserts the equality field by field).
CFG_RING = replace(CFG, STATS_STAGES=2)
# The Rubin TMEM map (kernel header sections 9 / 11; the resource plan's option A, the barrier skeleton R1 of 2026-11-03): the SAME geometry with the
# dQ image moved out of S^T's tail into the 64 columns past dK of the 576-column carve -- one record per stats form; SMEM, every slab and every
# other TMEM tenant unchanged (validate_config asserts the equality field by field).  TMEM_COLS_SM107 = the part's whole carve (the SKU card: 576
# columns, allocated exclusive; == DK_COL + DK_COLS + DQ_COLS = exactly what the map uses), the ONE constant of the alloc AND the dealloc.
TMEM_COLS_SM107 = 576
CFG_DQ576 = replace(CFG, TMEM_COLS=TMEM_COLS_SM107, TMEM_IS_EXCLUSIVE=True, TMEM_MAP="dq_576")
CFG_RING_DQ576 = replace(CFG_RING, TMEM_COLS=TMEM_COLS_SM107, TMEM_IS_EXCLUSIVE=True, TMEM_MAP="dq_576")
# The Rubin SMEM map (kernel header sections 5 / 8 / 11; the resource plan's option B2, the barrier skeleton R2.c of 2026-11-03): the dq_576 ring
# geometry with sQ1 / sDO1 / sDS / sDS_xchg at TWO stages each on the 327 KiB oversized carveout of cc 10.7 -- 314368 B (slack 20480 of 334848), every
# tcgen05 root below 262144 (the last, sDS stage 1 @ 229376, ends exactly at the line), the bar block + the STAMPS record re-based by derivation.
# Served ON the dq_576 map under the default role shape and stats form only (validate_fields_for_roles): the lagging epilogue order the ring pays for
# voids the dq_576 map's program-order carrier, so the row dq_read_done (the B2 barrier table) carries the dQ image's reuse edge instead.
CFG_RING_DQ576_B2 = replace(CFG_RING_DQ576, SMEM_MAP="rubin_b2", Q1_STAGES=2, DO1_STAGES=2, DS_STAGES=2, XCHG_STAGES=2)
# The DIAGNOSTIC map "sm100_oversized" (2026-11-03; the follow-up question of the R2.c STAMPS -- the arm's loop is faster, the kernel slower; the SKU
# card says the oversized carveout shrinks L1, so tensor-map / descriptor / CLC payload fetches may miss it): the sm100 map ITSELF (every slab base,
# ring depth, barrier row and address identical: the kernel body renders byte-identical) with the DECLARED dynamic SMEM padded above the opt-in cap,
# so the DSL launcher takes the oversized carveout for it.  Output CORRECT (it is the shipping map; --validate may run it); never a default; cc 10.7
# only; the benchmarker's A/B/A against the part's default at 16K / 32K is its number (whether ANY oversized-SMEM arm can pay on this kernel).
SMEM_OVERSIZED_PROBE_BYTES = 262144  # the padded declaration: above the 232448-B opt-in cap, below the 334848-B carveout; no tcgen05 root moves
CFG_RING_OVS = replace(CFG_RING, SMEM_MAP="sm100_oversized", SMEM_PAD_BYTES=SMEM_OVERSIZED_PROBE_BYTES - CFG_RING.SMEM_BYTES)
CFG_RING_DQ576_OVS = replace(CFG_RING_DQ576, SMEM_MAP="sm100_oversized", SMEM_PAD_BYTES=SMEM_OVERSIZED_PROBE_BYTES - CFG_RING_DQ576.SMEM_BYTES)
SMEM_MAP_DIAGNOSTIC = ("sm100_oversized",)  # DIAGNOSTIC maps: selectable by name, never a part default (validate_config); output correct


# The stats form's kernel code space (``Constexpr[int]`` ``stats_form_code``; the body names STATS_FORM_LDG / STATS_FORM_TMA_RING from these).
STATS_FORM_CODES = {"ldg": 0, "tma_ring": 1}


def resolve_stats_form(name=None, roles=None):
    """The softmax's STATS FORM, typed: ``None`` -> the form the (resolved) role shape renders when the field is unset
    (``CFG.STATS_FORM_DEFAULT_BY_ROLES``: "tma_ring" under the default roles, "ldg" under "16w_fa4" -- the only form sited in that bundle;
    ``roles`` None = the default role shape); a name outside ``CFG.STATS_FORMS`` is a ``ValueError`` BY NAME.  Both values compute the same
    function (bitwise LSE / delta words: header section 2); a NAMED "tma_ring" under "16w_fa4" declines by name in validate_fields_for_roles."""
    if name is None:
        shape = CFG.WARP_ROLES_DEFAULT if roles is None else roles
        if shape not in CFG.STATS_FORM_DEFAULT_BY_ROLES:
            raise ValueError(f"warp_roles must be one of {CFG.WARP_ROLES}; got {roles!r}")
        form = CFG.STATS_FORM_DEFAULT_BY_ROLES[shape]
    else:
        form = name
    if type(form) is not str or form not in CFG.STATS_FORMS:
        raise ValueError(f"stats_form must be one of {CFG.STATS_FORMS}; got {form!r}")
    return form


def stats_form_code(name=None, roles=None):
    """The body's ``stats_form_code`` Constexpr[int] for a (resolved) form name (``STATS_FORM_CODES``; 0 = ldg, 1 = tma_ring); ``None`` resolves
    per role shape; declines exactly as :func:`resolve_stats_form`."""
    return STATS_FORM_CODES[resolve_stats_form(name, roles)]


def cfg_for_stats_form(name=None, roles=None):
    """The geometry record a (resolved or ``None``) stats form renders with ON THE s_tail MAP: ``CFG`` (ldg: one stats slab, the merged scaffold) or
    ``CFG_RING`` (tma_ring: the two-stage ring, the scaffold re-based +1024, SMEM_BYTES == the cap).  Every property the two records share is equal
    (validate_config); the body reads the form-dependent ones (SMEM_BYTES, BARS, TMEM_HANDLE, SCHED_PAYLOAD, the ring bases) through this.  The
    per-PART record (the TMEM map keyed by cc) is :func:`cfg_for_part`."""
    return CFG_RING if resolve_stats_form(name, roles) == "tma_ring" else CFG


# The TMEM map's kernel code space (``Constexpr[int]`` ``tmem_map_code``; the body names TMEM_MAP_S_TAIL / TMEM_MAP_DQ576 from these).
TMEM_MAP_CODES = {"s_tail": 0, "dq_576": 1}
# PER PART (cc -> the maps SERVED and the map an UNSET field resolves to under the default role shape; the keys = the pair body's admitted parts,
# ``CTA_PAIR_CCS``): "s_tail" is served everywhere (512 columns) and is THE default on every part; "dq_576" needs the 576-column carve = cc 10.7
# only, where it is SELECTABLE BY NAME and not the default: the ceiling probe of its two rows (s_tail_wait_off) read +0.0 / +0.55 % at 16K and -0.3 / +0.27 % at 32K (step ms / main-kernel cycles), the combined s_tail + s_read probe -0.4 / -0.2 % (benchmarker, cc 10.7, A/B/A x 3 x 100 both build orders, 2026-11-03): inside noise, below the +2.0 % threshold -- a
# NEUTRAL structural map (the SOL card's arm R1), kept as the base of the two-stage sDS + XCHG staging arm; the default flips only when that arm
# lands on it with its own A/B/A row (recorded beside the entry below with its date).  Under "16w_fa4" the unset field resolves to "s_tail" on
# every part (the shape has no program-order carrier for the independent image's reuse edge; a NAMED "dq_576" declines there).
TMEM_MAP_PARTS = {"s_tail": CTA_PAIR_CCS, "dq_576": ((10, 7),)}
TMEM_MAP_DEFAULT_BY_PART = {
    # GB200 / GB300: the 512-column carve; the S-tail map is the only one served
    (10, 0): "s_tail",
    (10, 3): "s_tail",
    # Rubin: the 576-column carve; "dq_576" selectable by name, NOT the default (its probe read inside noise, 2026-11-03: the comment above);
    # flips to "dq_576" only with the two-stage staging arm's own A/B/A row on this part
    (10, 7): "s_tail",
}


class TmemMapNotServed(ValueError):
    """A map of ``CFG.TMEM_MAPS`` NAMED on a part that does not serve it (``TMEM_MAP_PARTS``): the engine's typed decline -- the prepared module
    raises ``NotImplementedError`` with this text.  A name outside the domain is a plain ``ValueError`` (a bug of the caller, never a decline)."""


def resolve_tmem_map(name=None, cc=None, roles=None):
    """The pair body's TMEM MAP, typed and PART-KEYED: ``None`` -> the map the part renders when the field is unset (``TMEM_MAP_DEFAULT_BY_PART[cc]``
    under the default role shape, "s_tail" under "16w_fa4"); a name outside ``CFG.TMEM_MAPS`` is a ``ValueError`` BY NAME; a map the part does not
    serve is :class:`TmemMapNotServed` (a ``ValueError``) BY NAME; ``cc`` (the device's compute capability as a tuple -- the prepared module's read,
    or ``compile_host``'s ``sm``) outside the admitted parts has NO default and declines by name.  Both maps compute the same function (bitwise dK /
    dV: the same accumulation chain per GEMM; dQ by the fp64 gate)."""
    cc = None if cc is None else tuple(cc)
    if cc not in TMEM_MAP_DEFAULT_BY_PART:
        raise ValueError(f"tmem_map: no TMEM map is defined for cc {cc}; the pair body is admitted on {tuple(TMEM_MAP_DEFAULT_BY_PART)}")
    shape = resolve_warp_roles(roles)
    if name is None:
        form = TMEM_MAP_DEFAULT_BY_PART[cc] if shape == CFG.WARP_ROLES_DEFAULT else CFG.TMEM_MAP  # 16w_fa4: the s_tail map (the only one it serves)
    else:
        form = name
    if type(form) is not str or form not in CFG.TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {CFG.TMEM_MAPS}; got {form!r}")
    if cc not in TMEM_MAP_PARTS[form]:
        raise TmemMapNotServed(
            f"tmem_map {form!r} is not served on cc {cc} (the {TMEM_COLS_SM107}-column TMEM carve it allocates exists on {TMEM_MAP_PARTS[form]}); "
            f"choose tmem_map {TMEM_MAP_DEFAULT_BY_PART[cc]!r} or leave the field unset"
        )
    return form


def tmem_map_code(name=None, cc=None, roles=None):
    """The body's ``tmem_map_code`` Constexpr[int] for a (resolved) map name on part ``cc`` (``TMEM_MAP_CODES``; 0 = s_tail, 1 = dq_576); declines
    exactly as :func:`resolve_tmem_map`."""
    return TMEM_MAP_CODES[resolve_tmem_map(name, cc, roles)]


# The SMEM map's kernel code space (``Constexpr[int]`` ``smem_map_code``; the body names SMEM_MAP_SM100 / SMEM_MAP_RUBIN_B2 from these).
SMEM_MAP_CODES = {"sm100": 0, "rubin_b2": 1, "sm100_oversized": 2}
# PER PART (cc -> the maps SERVED and the map an UNSET field resolves to): "sm100" is served everywhere (the opt-in cap) and is THE default on every
# part; "rubin_b2" needs the oversized carveout = cc 10.7 only, where it is SELECTABLE BY NAME ON tmem_map "dq_576" (x 16w x tma_ring: the B2 tables'
# seat; a NAMED "rubin_b2" under "s_tail", "16w_fa4" or "ldg" declines by name in validate_fields_for_roles) and NOT the default: the SOL card's arm
# R2.c (2026-11-03) -- its ceiling probe exchange_off read +5.4 / +6.7 % step at 16K / 32K on rubin-perf-212, above the +2.0 % MAIN-cycle threshold, so the arm was
# BUILT; the part default (together with tmem_map "dq_576") flips only with the built map's own A/B/A row against "sm100" on cc 10.7, recorded beside
# the entry below with its date.
SMEM_MAP_PARTS = {"sm100": CTA_PAIR_CCS, "rubin_b2": ((10, 7),), "sm100_oversized": ((10, 7),)}  # the DIAGNOSTIC map needs the carveout too
SMEM_MAP_SLAB_BYTES = {"rubin_b2": CFG_RING_DQ576_B2.SMEM_BYTES, "sm100_oversized": SMEM_OVERSIZED_PROBE_BYTES}  # the declared size per oversized map
SMEM_MAP_DEFAULT_BY_PART = {
    (10, 0): "sm100",
    (10, 3): "sm100",
    # Rubin: the 327 KiB carveout; "rubin_b2" selectable by name ON dq_576, NOT the default: its A/B/A vs "sm100" on this part measured a LOSS
    # (2026-11-03, a Rubin class node, both build orders: -2.5..-5.0 % step at 8K / 16K / 32K) -> below the +2.0 % MAIN-cycle threshold; no default flip
    (10, 7): "sm100",
}


class SmemMapNotServed(ValueError):
    """A map of ``CFG.SMEM_MAPS`` NAMED on a part that does not serve it (``SMEM_MAP_PARTS``): the engine's typed decline -- the prepared module
    raises ``NotImplementedError`` with this text.  A name outside the domain is a plain ``ValueError`` (a bug of the caller, never a decline)."""


def resolve_smem_map(name=None, cc=None, roles=None, stats_form=None, tmem_map=None):
    """The pair body's SMEM MAP, typed and PART-KEYED: ``None`` -> the map the part renders when the field is unset (``SMEM_MAP_DEFAULT_BY_PART[cc]``:
    "sm100" on every admitted part today); a name outside ``CFG.SMEM_MAPS`` is a ``ValueError`` BY NAME; a map the part does not serve is
    :class:`SmemMapNotServed` (a ``ValueError``) BY NAME; ``cc`` outside the admitted parts has NO default and declines by name.  ``roles`` /
    ``stats_form`` / ``tmem_map`` are resolved here only so a value outside its domain declines by name first; the cross-field rules of a NAMED
    "rubin_b2" (built ON dq_576 x 16w x tma_ring only) are ``validate_fields_for_roles``'.  Both maps compute the same function (bitwise dK / dV: the
    same accumulation chain per GEMM; dQ by the fp64 gate)."""
    cc = None if cc is None else tuple(cc)
    if cc not in SMEM_MAP_DEFAULT_BY_PART:
        raise ValueError(f"smem_map: no SMEM map is defined for cc {cc}; the pair body is admitted on {tuple(SMEM_MAP_DEFAULT_BY_PART)}")
    shape = resolve_warp_roles(roles)
    resolve_stats_form(stats_form, shape)
    resolve_tmem_map(tmem_map, cc, shape)
    smap = SMEM_MAP_DEFAULT_BY_PART[cc] if name is None else name
    if type(smap) is not str or smap not in CFG.SMEM_MAPS:
        raise ValueError(f"smem_map must be one of {CFG.SMEM_MAPS}; got {smap!r}")
    if cc not in SMEM_MAP_PARTS[smap]:
        raise SmemMapNotServed(
            f"smem_map {smap!r} is not served on cc {cc} (its {SMEM_MAP_SLAB_BYTES[smap]}-B slab needs the oversized SMEM carveout, which exists on "
            f"{SMEM_MAP_PARTS[smap]}); choose smem_map {CFG.SMEM_MAP!r} or leave the field unset"
        )
    return smap


def smem_map_code(name=None, cc=None, roles=None, stats_form=None, tmem_map=None):
    """The body's ``smem_map_code`` Constexpr[int] for a (resolved) map name on part ``cc`` (``SMEM_MAP_CODES``; 0 = sm100, 1 = rubin_b2); declines
    exactly as :func:`resolve_smem_map`."""
    return SMEM_MAP_CODES[resolve_smem_map(name, cc, roles, stats_form, tmem_map)]


# The five geometry records by (stats form, TMEM map, SMEM map): the body reads every form- / map-dependent number through the one it was handed
# (SMEM_BYTES, BARS, TMEM_HANDLE, SCHED_PAYLOAD, the slab and ring bases, the ring depths; DQ_COL, TMEM_USED, TMEM_COLS, TMEM_IS_EXCLUSIVE).
GEOMETRIES = {
    ("ldg", "s_tail", "sm100"): CFG,
    ("tma_ring", "s_tail", "sm100"): CFG_RING,
    ("ldg", "dq_576", "sm100"): CFG_DQ576,
    ("tma_ring", "dq_576", "sm100"): CFG_RING_DQ576,
    ("tma_ring", "dq_576", "rubin_b2"): CFG_RING_DQ576_B2,
    # the DIAGNOSTIC map: the two tma_ring records, padded (byte-identical bodies; the declared SMEM 262144)
    ("tma_ring", "s_tail", "sm100_oversized"): CFG_RING_OVS,
    ("tma_ring", "dq_576", "sm100_oversized"): CFG_RING_DQ576_OVS,
}


def cfg_for_geometry(stats_form, tmem_map, smem_map="sm100"):
    """The geometry record of a RESOLVED (stats form, TMEM map, SMEM map) triple -- one of the five frozen records of ``GEOMETRIES``; a triple outside
    the domains (or a combination the body does not build: "rubin_b2" off dq_576 x tma_ring) is a ``ValueError`` by name (the resolvers above are the
    typed entry)."""
    if (stats_form, tmem_map, smem_map) not in GEOMETRIES:
        raise ValueError(
            f"no geometry for stats_form {stats_form!r} x tmem_map {tmem_map!r} x smem_map {smem_map!r}: stats_form in {CFG.STATS_FORMS}, tmem_map in "
            f"{CFG.TMEM_MAPS}, smem_map in {CFG.SMEM_MAPS} (rubin_b2 is built on tma_ring x dq_576 only; sm100_oversized on tma_ring)"
        )
    return GEOMETRIES[(stats_form, tmem_map, smem_map)]


def cfg_for_part(cc, stats_form=None, roles=None, tmem_map=None, smem_map=None):
    """The geometry record PART ``cc`` renders under the (resolved or ``None``) stats form, role shape, TMEM map and SMEM map: cc 10.0 / 10.3 -> the
    s_tail records (the SAME objects :func:`cfg_for_stats_form` returns: the sm_100a / sm_103a renderings and their md5 records do not move); cc 10.7
    -> the s_tail records too when the fields are unset (the part's defaults), the dq_576 records when "dq_576" is NAMED under the default role shape,
    the dq_576 x rubin_b2 record when BOTH "dq_576" and "rubin_b2" are NAMED (x tma_ring x 16w).  Declines exactly as :func:`resolve_stats_form` /
    :func:`resolve_tmem_map` / :func:`resolve_smem_map` / :func:`cfg_for_geometry` (``compile_host`` keys the artifact on every resolved name and
    runs ``validate_fields_for_roles`` first, so an unbuilt combination declines by name before any trace)."""
    shape = resolve_warp_roles(roles)
    form = resolve_stats_form(stats_form, shape)
    tmap = resolve_tmem_map(tmem_map, cc, shape)
    return cfg_for_geometry(form, tmap, resolve_smem_map(smem_map, cc, shape, form, tmap))


# The mask form's kernel code space (``Constexpr[int]`` ``mask_form_code``; the body names MASK_FORM_FSEL / _PREDICATED / _DIAG_OFF from these).
MASK_FORM_CODES = {"fsel": 0, "predicated": 1, "diag_mask_off": 2}


def resolve_mask_form(name=None):
    """The softmax's MASK FORM, typed: ``None`` -> ``CFG.MASK_FORM_DEFAULT`` (``"fsel"``, the shipped rendering); a name outside ``CFG.MASK_FORMS``
    is a ``ValueError`` BY NAME.  ``CFG.MASK_FORM_DIAGNOSTIC`` (``"diag_mask_off"``) is the deletion probe: output WRONG BY DESIGN, resolved here
    like the ``diag_2cta`` probes (set on the adapter INSTANCE by a probe driver, keyed apart, refused by the driver's --validate)."""
    form = CFG.MASK_FORM_DEFAULT if name is None else name
    if type(form) is not str or form not in CFG.MASK_FORMS:
        raise ValueError(f"mask_form must be one of {CFG.MASK_FORMS}; got {form!r}")
    return form


def mask_form_code(name=None):
    """The body's ``mask_form_code`` Constexpr[int] for a (resolved) form name (``MASK_FORM_CODES``; 0 = fsel); declines exactly as
    :func:`resolve_mask_form`."""
    return MASK_FORM_CODES[resolve_mask_form(name)]


def resolve_debug_clk(value=None):
    """The STAMPS lever, typed: ``None`` -> ``CFG.DEBUG_CLK_DEFAULT`` (False: the shipped rendering); a bool; anything else (an int, a string)
    is a ``ValueError`` BY NAME."""
    flag = CFG.DEBUG_CLK_DEFAULT if value is None else value
    if type(flag) is not bool:
        raise ValueError(f"debug_clk must be a bool (None = {CFG.DEBUG_CLK_DEFAULT}); got {flag!r} ({type(flag).__name__})")
    return flag


# The role shape's kernel code space (``Constexpr[int]`` ``warp_roles_code``; the body names WARP_ROLES_16W / WARP_ROLES_16W_FA4 from these).
WARP_ROLES_CODES = {"16w": 0, "16w_fa4": 1}
# The barrier rows that have neither producer nor consumer under 16w_fa4 (the barrier audit / section 6: ds_tmem -- no dS^T readout; dp_empty --
# its ONLY arrive site was that readout; dP^T(t+1)'s overwrite of the packed dS^T(t) is ordered by the in-order tensor pipe after dK(t), of dP^T(t)'s
# fp32 by ds_ready(t) -> dK(t)).  Deleted from BAR_LAYOUT_XFA4 with their states, waits and drains.
DELETED_BARRIERS_XFA4 = ("ds_tmem", "dp_empty")
# The barrier row ADDED under 16w_fa4 (the barrier audit, 2026-10-27 10:55 UTC, after the the pre-fix tree hang): `dq_war_done` = the softmax's sDS
# WAR signal, an MMA_COMMIT committed INSIDE `_dq_issue` at the two dQ issue sites only (BEFORE dq_done; never at the zero-trip empty-commit site),
# waited ONLY by the softmax (st(1) pre-armed, one wait per live tile, no zero-trip wait or advance, one unconditional post-loop drain): every flip
# depends on the softmax's own preceding STS, so the pre-armed waiter can never find a flip landed early (F in {w-1, w}) -- the shared `dq_done`
# waiter set of the first build raced the leader's zero-trip EMPTY commit (a zero-trip item as a cluster's first item: a HANG; consecutive zero-trips:
# a lap).  `dq_done` reverts to the reduce group alone (the 04:17 AUDIT row).
ADDED_BARRIERS_XFA4 = ("dq_war_done",)
# The barrier row that has NO USER under 16w_fa4 although it is in the 16w table: `dvk_stage_empty` serves `dkv_accum = "fp32_reduce"`, whose
# fp32 dK / dV staging lives in the 16w EPILOGUE loop; under 16w_fa4 the field DECLINES BY NAME (`validate_fields_for_roles`), so the row is
# absent from BAR_LAYOUT_XFA4 / BarsXFA4 (no init, no arrive, no wait) and the 33-row table, its END and its PTX record stand unchanged.
UNSERVED_BARRIERS_XFA4 = ("dvk_stage_empty",)


def resolve_warp_roles(name=None):
    """The ROLE SHAPE, typed: ``None`` -> ``CFG.WARP_ROLES_DEFAULT`` ("16w", the shipped rendering); a name outside ``CFG.WARP_ROLES`` is a
    ``ValueError`` BY NAME."""
    roles = CFG.WARP_ROLES_DEFAULT if name is None else name
    if type(roles) is not str or roles not in CFG.WARP_ROLES:
        raise ValueError(f"warp_roles must be one of {CFG.WARP_ROLES}; got {roles!r}")
    return roles


def warp_roles_code(name=None):
    """The body's ``warp_roles_code`` Constexpr[int] for a (resolved) role-shape name (``WARP_ROLES_CODES``; 0 = 16w); declines exactly as
    :func:`resolve_warp_roles`."""
    return WARP_ROLES_CODES[resolve_warp_roles(name)]


def validate_fields_for_roles(roles, diag_2cta=None, debug_clk=False, dkv_accum=None, stats_form=None, tmem_map=None, smem_map=None):
    """The field bundle a (resolved) role shape SERVES, typed BY NAME: under ``16w_fa4`` the STAMPS lever ``debug_clk`` (its record layout names
    the 16w roles' waits), ``dkv_accum = "fp32_reduce"`` (its fp32 dK / dV staging rounds live in the 16w epilogue loop), ``stats_form =
    "tma_ring"`` NAMED (the stats ring's two rows are the 16w bundle's; the first build of the ring is sited under the 16w roles only -- an UNSET
    ``stats_form`` resolves to "ldg" there) and ``tmem_map = "dq_576"`` NAMED (the independent dQ image's reuse edge rides the 16w EPILOGUE's
    program order into ds_smem; the reduce group of 16w_fa4 has no such carrier -- an UNSET ``tmem_map`` resolves to "s_tail" there) are not built
    and decline; the three S-tail WAIT probes (``CFG.DIAG_2CTA_S_TAIL_PROBES``) decline BY NAME under ``tmem_map = "dq_576"`` on every shape (their
    wait sites are the two rows the map deletes: the probe would render the default body); every other ``diag_2cta`` probe of
    ``CFG.DIAG_2CTA_NAMES`` serves both shapes and both maps; ``stats_form = "tma_ring"`` x ``debug_clk`` is SERVED (the record derives its words
    from the re-based spare and carries the two ring wait words; under dq_576 the four S-tail words are dead = 0).  ``tmem_map`` is the RESOLVED
    map name (``resolve_tmem_map``: the caller knows the part); ``None`` = the s_tail map.  ``smem_map`` is the RESOLVED SMEM map name
    (``resolve_smem_map``); ``None`` = the sm100 map: "rubin_b2" is built ON "dq_576" under the default role shape and stats form only -- NAMED under
    "s_tail" (the ring's lagging epilogue order voids the S-tail map's carrier and its two rows have no seat in the B2 table), under "16w_fa4" (no
    B2 table for the reduce group) or under "ldg" (the B2 tables and their rendering record are sited under tma_ring; the check that would admit ldg
    is ONE host render + its record, not done 2026-11-03) it declines by name.  Called by ``compile_host`` before any trace; returns ``roles``."""
    if roles not in CFG.WARP_ROLES:
        raise ValueError(f"warp_roles must be one of {CFG.WARP_ROLES}; got {roles!r}")
    diag = resolve_diag_2cta(diag_2cta)
    form = resolve_stats_form(stats_form, roles)
    tmap = CFG.TMEM_MAP if tmem_map is None else tmem_map
    if type(tmap) is not str or tmap not in CFG.TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {CFG.TMEM_MAPS}; got {tmap!r}")
    smap = CFG.SMEM_MAP if smem_map is None else smem_map
    if type(smap) is not str or smap not in CFG.SMEM_MAPS:
        raise ValueError(f"smem_map must be one of {CFG.SMEM_MAPS}; got {smap!r}")
    if smap != CFG.SMEM_MAP:
        if smap == "rubin_b2" and tmap == CFG.TMEM_MAP:  # the lagging order of the two-stage ring needs the independent image's dq_read_done row
            raise ValueError(
                f"smem_map {smap!r} is not built under tmem_map {tmap!r} (the two-stage dS / exchange staging ring stages dS(t) before the dQ(t-1) readout, "
                f"which voids the S-tail map's program-order carrier; the ring's barrier table is sited on the independent dQ image with the dq_read_done "
                f"row); choose tmem_map {CFG.TMEM_MAPS[1]!r} or smem_map {CFG.SMEM_MAP!r}"
            )
        if roles != CFG.WARP_ROLES_DEFAULT:
            raise ValueError(
                f"smem_map {smap!r} is not built under warp_roles {roles!r} (the oversized-carveout maps render the 16w body: the two-stage dS / exchange "
                f"ring and its dq_read_done row are sited in the 16w epilogue, the diagnostic sm100_oversized map has no 16w_fa4 record); choose warp_roles "
                f"{CFG.WARP_ROLES_DEFAULT!r} or smem_map {CFG.SMEM_MAP!r}"
            )
        if form != CFG.STATS_FORM_DEFAULT:
            raise ValueError(
                f"smem_map {smap!r} is not built under stats_form {form!r} (the oversized-carveout maps' tables and rendering records are sited under stats_form "
                f"{CFG.STATS_FORM_DEFAULT!r}; the ldg form under the map has no pinned record -- one host render + record admits it, not done 2026-11-03); "
                f"choose stats_form {CFG.STATS_FORM_DEFAULT!r} or smem_map {CFG.SMEM_MAP!r}"
            )
    if tmap != CFG.TMEM_MAP and diag in CFG.DIAG_2CTA_S_TAIL_PROBES:
        raise ValueError(
            f"diag_2cta {diag!r} is not defined under tmem_map {tmap!r} (its wait sites are the s_read / s_tail_empty rows, which the independent dQ "
            f"image deletes: the probe would render the default body); choose tmem_map {CFG.TMEM_MAP!r} or a probe outside {CFG.DIAG_2CTA_S_TAIL_PROBES}"
        )
    if roles == CFG.WARP_ROLES_DEFAULT:
        return roles
    if tmap != CFG.TMEM_MAP:  # the reduce group reads dQ out, the softmax stages dS: no program-order carrier between the readout and ds_smem(t+1)
        raise ValueError(
            f"tmem_map {tmap!r} is not built under warp_roles {roles!r} (the independent dQ image's reuse edge dQ(t) readout -> dQ(t+1) write rides "
            f"the 16w epilogue's program order into ds_smem(t+1); the reduce group of {roles} has no such carrier and would need an explicit "
            f"dq_read_done row); choose warp_roles {CFG.WARP_ROLES_DEFAULT!r} or tmem_map {CFG.TMEM_MAP!r}"
        )
    if debug_clk:  # the STAMPS record layout (DBG_CLK.ROLE_WAITS / WORD) is the 16w roles'
        raise ValueError(
            f"debug_clk is not defined under warp_roles {roles!r} (the STAMPS record layout names the 16w roles' waits: no words for the softmax's "
            f"dq_war_done wait or the reduce group); choose warp_roles {CFG.WARP_ROLES_DEFAULT!r}"
        )
    if form != CFG.STATS_FORM_FA4:  # the stats ring rows live in the 16w bundle only (the first build; the inspector's table rules a 16w_fa4 merge)
        raise ValueError(
            f"stats_form {form!r} is not built under warp_roles {roles!r} (the stats ring's stats_full / stats_empty rows and the softmax's ring "
            f"waits are sited in the 16w bundle; the 16w_fa4 softmax keeps the ldg form); choose warp_roles {CFG.WARP_ROLES_DEFAULT!r} or "
            f"stats_form {CFG.STATS_FORM_FA4!r}"
        )
    from .d128_grouped_contract import DKV_ACCUM_DEFAULT, resolve_dkv_accum  # local: the contract module is the field's truth, the config stays light

    if resolve_dkv_accum(dkv_accum) != DKV_ACCUM_DEFAULT:
        raise ValueError(
            f"dkv_accum {dkv_accum!r} is not built under warp_roles {roles!r} (the fp32 dK / dV staging rounds live in the 16w epilogue loop; the "
            f"reduce group of 16w_fa4 stages bf16 halves only); choose warp_roles {CFG.WARP_ROLES_DEFAULT!r} or dkv_accum {DKV_ACCUM_DEFAULT!r}"
        )
    return roles


# (name, byte offset, active slots, init count, producer kind, scope) -- the barrier audit 2026-10-26 (SKELETON) + 2026-10-27 (ARM (ii) rows).
# Pasted from the audit record section 10 (34 rows / 39 slots, re-packed in declaration order, 16 B per row; END 230944 = 544 B
# of the 944-B block [230400, 231344), 400 B spare -- FITS).  The lane ledger (SUM(issuing lanes) == init per row, both CTAs), the states
# and the per-item flips are the kernel header's section 5 / 6; ``Bars`` is built from this tuple (one MBarrier per row, same init on BOTH
# CTAs: a leader-waited barrier's follower copy is initialised with the same count and never armed, never waited).  Kinds as the first body's.
BAR_LAYOUT = (
    # per q tile: operand rings (loader -> leader MMA; leader MMA commit -> both loaders)
    ("q1_full", 230400, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # Q1_TX; ONE stage (FAv4 Q_stage = 1 under 2-CTA; funds the sDQ ring)
    ("q1_empty", 230416, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after each S^T issue (same site as s_done); DRAIN x Q1_STAGES = 1
    ("q2_full", 230432, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # Q2_TX
    ("q2_empty", 230448, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dK(t)
    ("do1_full", 230464, 2, CFG.ONE_LANE, "tma_leader", "LEADER"),  # 2 half slots x DO1_HALF_TX (dP^T in 2 k-halves), ONE do1_state
    ("do1_empty", 230480, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dP^T(t) (same site as dp_done)
    ("do2_full", 230496, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # DO2_TX
    ("do2_empty", 230512, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dV(t)
    # per kv block
    ("k1_full", 230528, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # K1_TX
    ("k1_empty", 230544, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit at block end (after the tail dQ)
    ("v_full", 230560, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # V_TX
    ("v_empty", 230576, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit at block end
    ("k2_full", 230592, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # K2_TX
    ("k2_empty", 230608, 1, CFG.STORE_LANES, "thread", "LOCAL"),  # store warp after the dV store drain (dV staging aliases K2); fix A back-edge via dv_full
    # MMA -> compute hand-offs (leader commit, multicast; each CTA's consumers wait their own copy)
    ("s_done", 230624, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # softmax (both CTAs)
    ("dp_done", 230640, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # softmax (both CTAs)
    ("dq_done", 230656, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # epilogue (both CTAs): dQ(t-1) in tile t, the tail dQ(hi-1), + one empty commit per zero-trip item
    # compute -> MMA hand-offs (pair-wide fan-ins on the LEADER: the leader's MMA reads / overwrites both CTAs' TMEM)
    ("p_ready", 230672, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # 256 softmax lanes x 2 CTAs; waiter MMA(L) before dV(t)
    ("ds_ready", 230688, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # same lanes; waiter MMA(L) before dK(t)
    # s_read (ARM (ii)): the same softmax lanes after the S^T ld pair + wait(LOAD), EVERY tile; waiter MMA(L) at slot 2: dQ(t-1) lands in S^T's tail
    ("s_read", 230704, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),
    ("ds_tmem", 230720, 1, CFG.SOFTMAX_LANES, "thread", "LOCAL"),  # softmax -> local epilogue: dS^T packed in OUR TMEM
    # dp_empty: 128 epilogue lanes x 2 after the dS^T READOUT (ARM (ii): moved off the dQ readout); MMA(L) pre-armed
    ("dp_empty", 230736, 1, CFG.PAIR_EPILOGUE_LANES, "leader", "LEADER"),
    # s_tail_empty (ARM (ii)): the same epilogue lanes after the dQ readout (64 cols @ DQ_COL); MMA(L) before the next writer of the S tail; pre-armed
    ("s_tail_empty", 230752, 1, CFG.PAIR_EPILOGUE_LANES, "leader", "LEADER"),
    ("ds_smem", 230768, 1, CFG.EPILOGUE_LANES, "thread", "LOCAL"),  # own dS half in sDS: waiter = local MMA warp (L: dQ issue on xchg_state, F: relay)
    # ds_recv_full: we arm XCHG_TX (lead warp elect); the PEER's bulk copy lands our recv half; L waits on xchg_state, F relays
    ("ds_recv_full", 230784, 1, CFG.ONE_LANE, "tma_dsmem", "LOCAL"),
    ("ds_pair_ready", 230800, 1, CFG.ONE_LANE, "peer", "LEADER"),  # follower MMA warp (ds_smem + ds_recv_full observed) -> leader; waited on xchg_state
    # dQ staging ring (epilogue -> store -> epilogue): the dedicated sDQ slab, 2 slots, 2 rounds per tile
    # dq_empty: store warp after each slot's reduce-add source read per round (the closed per-round release); epilogue pre-armed, NO bootstrap (D3)
    ("dq_empty", 230816, 2, CFG.STORE_LANES, "thread", "LOCAL"),
    ("dq_full", 230832, 2, CFG.DQ_SLICE_LANES, "thread", "LOCAL"),  # 64 = the two epilogue warps of ONE d half per slot per round
    # native dV / dK stores
    ("dv_full", 230848, 1, CFG.DV_FULL_LANES, "thread", "LOCAL"),  # epilogue 128 + loader 32 after its k2_empty wait (AUDIT fix A)
    ("dk_full", 230864, 1, CFG.EPILOGUE_LANES, "thread", "LOCAL"),
    ("dvk_empty", 230880, 1, CFG.PAIR_STORE_LANES, "leader", "LEADER"),  # both store warps -> MMA(L): dV / dK TMEM reusable; pre-armed
    # scheduler ring (CLC try_cancel multicast from the LEADER's scheduler; credits land on the leader only)
    ("sched_full", 230896, 2, CFG.ONE_LANE, "tma", "LOCAL"),  # CLC_TX on each CTA's copy; every consumer warp waits its own
    ("sched_empty", 230912, 2, CFG.PAIR_SCHED_CREDIT_WARPS, "peer", "LEADER"),  # 15 warps x 2 CTAs, one release arrive per warp
    # teardown
    ("tmem_dealloc", 230928, 1, CFG.TMEM_DEALLOC_ARRIVERS, "thread", "LOCAL"),  # local elect arrive + the peer's arrive_on_peer
    # fp32_reduce (dkv_accum = fp32_reduce; , the resource plan): the store warp after round 1's fp32 reduce-add
    # source reads drained -> the epilogue's round-2 staging may overwrite the five slots; 1 per item incl. zero-trip; INIT ONLY under dkv_fp32 (no
    # user under the default rendering, which stays byte-identical); AUDIT owed with the the alternate role shape AUDIT
    ("dvk_stage_empty", 230944, 1, CFG.STORE_LANES, "thread", "LOCAL"),
)
BAR_LAYOUT_END = 230960  # next free byte of the bar block (384 B of the 944 spare); 35 names / 40 slots, 560 B (ldg: the audited table verbatim)
# --- stats_form = tma_ring (the barrier audit's S1 skeleton 2026-10-27 17:30 + its 17:46 reconciliation, resources/s1-barrier-table.md sections 1 and 14;
# the resource-manager's slab plan s1-stats-ring.md section 1): the 35 rows above RE-BASED onto CFG_RING.BARS (= SLABS_END 231424: stage 1 of the ring
# took the 1 KiB the scaffold sat on), then the ring's two rows appended -- 37 names / 44 slots / 592 B, END 232016; the STAMPS record = the spare
# [232016, 232368) = 88 words.  Derived, never pasted twice: the ldg table is the truth, the re-base is arithmetic (validate_config pins both).
#   stats_full[2]   TMA_LOAD / LOCAL (kind "tma" = the sched_full kind), init ONE_LANE: the loader's `arrive(n_bytes=STATS_TX, pred=elect_sync())`
#                   on BOTH CTAs (never is_leader: a plain bulk copy lands on the issuing CTA) + ONE `bulk_copy` of the (head, q tile) record into
#                   stage s; waited by the 8 softmax warps (stats_state st(0), +1 per tile) after the s_read arrive, before _p_values
#   stats_empty[2]  THREAD / LOCAL, init SOFTMAX_LANES: the 8 softmax warps bare `.arrive()` right after _ds_values (the tile's last stats read);
#                   waited by the loader (stats_ld st(1) PRE-ARMED, +1 per tile: wait k <-> release k - 2) before the arm; no bootstrap, no drain
STATS_RING_ROWS = (
    ("stats_full", 2, CFG.ONE_LANE, "tma", "LOCAL"),
    ("stats_empty", 2, CFG.SOFTMAX_LANES, "thread", "LOCAL"),
)


def _rebase_layout(layout, base):
    """The audited table's rows at ``base + 16 i`` (declaration order, 16 B per row): the ldg table at CFG.BARS is the identity."""
    return tuple((name, base + 16 * i, slots, count, kind, scope) for i, (name, _off, slots, count, kind, scope) in enumerate(layout))


def _ring_layout(layout, base):
    rows = _rebase_layout(layout, base)
    return rows + tuple((name, base + 16 * (len(rows) + j), slots, count, kind, scope) for j, (name, slots, count, kind, scope) in enumerate(STATS_RING_ROWS))


BAR_LAYOUT_RING = _ring_layout(BAR_LAYOUT, CFG_RING.BARS)
BAR_LAYOUT_RING_END = CFG_RING.BARS + 16 * len(BAR_LAYOUT_RING)  # 232016
BAR_KIND_COUNTS_RING = {"tma_leader": 7, "mma": 9, "thread": 10, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 2}  # 37
BAR_LAYOUT_RING_STATUS = (
    "SKELETON by the barrier audit (S1 2026-10-27 17:30 + the 17:46 reconciliation with the bulk-copy form): the 35 ldg rows re-based onto "
    "CFG_RING.BARS + stats_full[2] (TMA_LOAD / LOCAL, ONE_LANE, expect_tx STATS_TX) + stats_empty[2] (THREAD / LOCAL, SOFTMAX_LANES); AUDIT of the "
    "committed body owed"
)
BAR_LAYOUT_STATUS = (
    "FILLED by the barrier audit (SKELETON, ARM (ii), the audit record sections 1-10): 34 names / 39 slots; "
    "NEW s_read (512) + s_tail_empty (256), dp_empty arrive MOVED to the dS^T readout, xchg_state split (P2), q1 ring 1 stage, dq rings 2 slots "
    "x 2 rounds (D4 per-round release), dq_empty pre-armed without bootstrap (D3), dv_full 160 (fix A) carried; AUDIT from the committed source owed; "
    "+ dvk_stage_empty (THREAD 32 store -> epilogue, fp32_reduce rounds only, init under dkv_fp32; audited with the role shape)"
)
BAR_KIND_COUNTS = {"tma_leader": 7, "mma": 9, "thread": 9, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 1}
# --- tmem_map = dq_576 (the barrier skeleton R1 2026-11-03 sections 1.1 / 1.2; PART-KEYED: cc 10.7): the rows the independent dQ image DELETES --
# the WRITE edge s_read and the REUSE edge s_tail_empty of the S-tail alias have nothing left to guard once the dQ image is [512, 576); the new
# reuse edge dQ(t) readout -> dQ(t+1) write rides ds_smem(t+1) / ds_pair_ready(t+1) through the epilogue's program order (the readout trio of
# tile t precedes the dS STS + ds_smem.arrive() of tile t+1 in the SAME warps; pinned by the test module).  The tables = the audited rows minus the
# two, re-packed from the geometry's BARS by the _rebase_layout form (16 B per row); under tma_ring the ring's two rows follow.  Every kept row keeps
# its count, kind, waiter and cadence (validate_config asserts it row by row); no row is added under 16w.
DELETED_BARRIERS_DQ576 = ("s_read", "s_tail_empty")


def _filter_layout(layout, deleted):
    return tuple(row for row in layout if row[0] not in deleted)


BAR_LAYOUT_DQ576 = _rebase_layout(_filter_layout(BAR_LAYOUT, DELETED_BARRIERS_DQ576), CFG_DQ576.BARS)  # 16w x ldg: 33 names / 38 slots, END 230928
BAR_LAYOUT_DQ576_END = CFG_DQ576.BARS + 16 * len(BAR_LAYOUT_DQ576)
BAR_LAYOUT_RING_DQ576 = _ring_layout(_filter_layout(BAR_LAYOUT, DELETED_BARRIERS_DQ576), CFG_RING_DQ576.BARS)  # 16w x tma_ring: 35 names / 42 slots, END 231984
BAR_LAYOUT_RING_DQ576_END = CFG_RING_DQ576.BARS + 16 * len(BAR_LAYOUT_RING_DQ576)
BAR_LAYOUT_DQ576_STATUS = (
    "SKELETON by the barrier audit (R1 2026-11-03, BARRIER-SKELETON-RUBIN sections 1.1-1.6): the audited table minus s_read (LEADER 512) and "
    "s_tail_empty (LEADER 256), re-packed; 0 rows added under 16w (the reuse edge of the independent image rides the epilogue's program order into "
    "ds_smem / ds_pair_ready); AUDIT of the committed body owed"
)
# --- smem_map = rubin_b2 (the barrier skeleton R2.c 2026-11-03 sections 2.2 / 2.3 / 2.5 b; PART-KEYED: cc 10.7 ON the dq_576 map): the table a geometry
# renders is DERIVED from the audited rows -- the rows whose SLOT count follows a ring depth (RING_SLOT_ROWS: q1_full / q1_empty x Q1_STAGES, do1_full
# x D_HALVES x DO1_STAGES, do1_empty x DO1_STAGES, ds_smem / ds_recv_full / ds_pair_ready x DS_STAGES; the init count per slot, the kind, the scope and
# the waiter are the audited row's), the ONE row the two-stage ring ADDS (dq_read_done: LEADER relaxed, PAIR_EPILOGUE_LANES 256 = the four epilogue
# warps of BOTH CTAs bare after each dQ readout's LDTM + tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC); waited by MMA(L) inside _dq_issue
# before EVERY dQ issue on `dq_read_state` st(1) pre-armed, +1 per wait, ONE drain after the loop: the dQ TMEM image's REUSE edge under the LAGGING
# epilogue order -- the epilogue stages dS(t) and arrives ds_smem(t) BEFORE it reads dQ(t-1) out, so the dq_576 map's program-order carrier is gone;
# it takes the deleted s_tail_empty's seat after dp_empty: the same lanes, the same kind), and the re-pack with a CUMULATIVE walk (a row of up to two
# 8-B slots takes 16 B, the 4-slot do1_full row 32 B -- `_rebase_layout`'s 16 B per row would overlap it).  validate_config asserts the derivation
# reproduces every pasted table above (the ldg / ring / dq_576 / 16w_fa4 ones) and pins the B2 table's names, slots, counts, kinds and addresses.
RING_SLOT_ROWS = {
    "q1_full": ("Q1_STAGES",),
    "q1_empty": ("Q1_STAGES",),
    "do1_full": ("D_HALVES", "DO1_STAGES"),
    "do1_empty": ("DO1_STAGES",),
    "ds_smem": ("DS_STAGES",),
    "ds_recv_full": ("DS_STAGES",),
    "ds_pair_ready": ("DS_STAGES",),
}
ADDED_BARRIERS_B2 = ("dq_read_done",)
DQ_READ_DONE_ROW = ("dq_read_done", 1, CFG.PAIR_EPILOGUE_LANES, "leader", "LEADER")
DQ_READ_DONE_AFTER = "dp_empty"  # the row it follows in declaration order (the deleted s_tail_empty's seat)


def _row_bytes(slots):
    return 16 * ((slots * 8 + 15) // 16)  # the bar block's row granularity: 16 B per row of up to two 8-B slots, 32 B for three or four


def _pack_layout(rows, base):
    """``rows`` (name, _, slots, count, kind, scope) re-packed from ``base`` in order with the cumulative walk (16 B per <= 2 slots, 32 B per 4)."""
    packed, off = [], base
    for name, _off, slots, count, kind, scope in rows:
        packed.append((name, off, slots, count, kind, scope))
        off += _row_bytes(slots)
    return tuple(packed)


def layout_for_geometry(geo, roles="16w"):
    """The barrier table geometry ``geo`` renders under role shape ``roles``: the audited base table (BAR_LAYOUT under 16w, BAR_LAYOUT_XFA4 under
    16w_fa4) minus the rows the TMEM map deletes (dq_576: DELETED_BARRIERS_DQ576), the ring rows' slots from the geometry's depths (RING_SLOT_ROWS),
    the dq_read_done row after DQ_READ_DONE_AFTER under a two-stage dS ring, re-packed from ``geo.BARS`` with the cumulative walk, then the stats ring's
    rows under a two-stage stats ring (tma_ring).  Pure arithmetic over the pasted rows: the truth stays the audited tables."""
    base = {"16w": BAR_LAYOUT, "16w_fa4": BAR_LAYOUT_XFA4}[roles]
    rows = []
    for name, off, slots, count, kind, scope in base:
        if geo.TMEM_MAP != CFG.TMEM_MAP and name in DELETED_BARRIERS_DQ576:
            continue
        if name in RING_SLOT_ROWS:
            slots = 1
            for field_name in RING_SLOT_ROWS[name]:
                slots *= getattr(geo, field_name)
        rows.append((name, off, slots, count, kind, scope))
        if name == DQ_READ_DONE_AFTER and geo.DS_STAGES > 1:
            rows.append(DQ_READ_DONE_ROW[:1] + (0,) + DQ_READ_DONE_ROW[1:])  # (name, offset placeholder, slots, count, kind, scope): packed below
    packed = _pack_layout(rows, geo.BARS)
    if geo.STATS_STAGES > 1:
        ring_rows = [(name, 0, slots, count, kind, scope) for name, slots, count, kind, scope in STATS_RING_ROWS]
        packed = packed + _pack_layout(ring_rows, packed[-1][1] + _row_bytes(packed[-1][2]))
    return packed


def layout_end(layout):
    return layout[-1][1] + _row_bytes(layout[-1][2])  # the next free byte past the table's last row


# (BAR_LAYOUT_RING_DQ576_B2 and its STAMPS record are built below BAR_LAYOUT_XFA4: the derivation reads both base tables)

# --- the STAMPS record (typed field ``debug_clk``; kernel header section 11): CFG.DBG_CLK_WORDS Int32 words per CTA in the bar block's spare
# [BAR_LAYOUT_END, TMEM_HANDLE), written by ONE representative warp per role (softmax warp 0, the epilogue lead warp, the single service warps;
# the scheduler warp's library loop is not instrumented), dumped per CTA at kernel exit into the workspace region the adapter carves under the
# lever.  Sums are u32 (wrap at 2^32 clk = 2.4 s at 1.8 GHz per CTA per launch; the decoder flags a saturating word).  Layout: the header, the
# leader's wait sums, the leader's blocked counts (waits above DBG_CLK_BLOCKED_THRESH; stride DBG_CLK.LEADER_BLOCKED_STRIDE from the wait word),
# the other roles' wait sums, the epilogue's hand-off segments, the ship -> recv handshake words, the store warp's bulk-group drains.
DBG_CLK_HEADER = ("magic", "rank", "tiles", "tiles_leader", "tiles_follower", "items", "body_clk", "body_ns")
DBG_CLK_ROLE_WAITS = {
    # the leader MMA warp (rank 0): every barrier it waits, in issue order (header section 5 MMA ISSUE ORDER + the P15 drains)
    "leader": (
        "k1_full",
        "q1_full",
        "s_tail_empty",
        "v_full",
        "k2_full",
        "dvk_empty",
        "dp_empty",
        "do1_full",
        "s_read",
        "ds_smem",
        "ds_recv_full",
        "ds_pair_ready",
        "dq_read_done",  # smem_map rubin_b2 only (the dQ image's reuse edge inside _dq_issue, after the three dS gates); dead = 0 under sm100
        "p_ready",
        "do2_full",
        "q2_full",
        "ds_ready",
        "sched_full",
    ),
    "softmax": ("s_done", "dp_done", "sched_full"),  # warp 0 of the 8
    "epilogue": ("ds_tmem", "dq_empty", "dq_done", "sched_full"),  # the leading warp (dq_empty = both slots of a round, one word)
    "follower": ("ds_smem", "ds_recv_full", "sched_full"),  # the follower's MMA warp (rank 1): the relay
    "loader": ("k1_empty", "v_empty", "k2_empty", "q1_empty", "do1_empty", "do2_empty", "q2_empty", "sched_full"),
    "store": ("dq_full", "dv_full", "dk_full", "sched_full"),
}
DBG_CLK_SEGMENTS = (
    # the epilogue lead warp, per tile: ds_tmem pass -> ship issue (sum, the DBG_CLK_HIST_EDGES histogram, the max), ds_tmem pass -> dq_done pass,
    # dq_done pass -> s_tail_empty arrive (the readout)
    "seg_ship_sum",
    "seg_ship_hist0",
    "seg_ship_hist1",
    "seg_ship_hist2",
    "seg_ship_hist3",
    "seg_ship_max",
    "seg_dq_done_sum",
    "seg_readout_sum",
    # the ship -> recv handshake (SAME-CTA proxy of the exchange flight: our ship(t) stamp vs our MMA warp's ds_recv_full(t) pass, which the PEER's
    # copy completes; both CTAs are symmetric): the epilogue lead writes (clk lo32, tile tag); the leader's _dq_issue / the follower's relay read it
    # after their ds_recv_full wait and accumulate when the tag matches their tile (else count a miss)
    "ship_clk",
    "ship_tag",
    "recv_leader_sum",
    "recv_leader_matched",
    "recv_leader_missed",
    "recv_follower_sum",
    "recv_follower_matched",
    "recv_follower_missed",
    # the store warp: clk inside the per-round `tma_store_wait` (the reduce-add source drain) and the item-end `tma_store_wait(0)`
    "store_wait_group_rounds",
    "store_wait_group_item",
)


# The stats ring's two wait words (stats_form = tma_ring; the barrier audit S1-12): appended AFTER the segments so every ldg word keeps its index
# (the ldg x debug_clk rendering is unchanged); unused (never written) under ldg, where no site waits either barrier.
DBG_CLK_RING_WAITS = (("softmax", "stats_full"), ("loader", "stats_empty"))


def _dbg_clk_layout():
    names = list(DBG_CLK_HEADER)
    names += [f"wait:leader:{b}" for b in DBG_CLK_ROLE_WAITS["leader"]]
    names += [f"blocked:leader:{b}" for b in DBG_CLK_ROLE_WAITS["leader"]]
    for role in ("softmax", "epilogue", "follower", "loader", "store"):
        names += [f"wait:{role}:{b}" for b in DBG_CLK_ROLE_WAITS[role]]
    names += list(DBG_CLK_SEGMENTS)
    names += [f"wait:{role}:{b}" for role, b in DBG_CLK_RING_WAITS]
    return {name: index for index, name in enumerate(names)}


@dataclass(frozen=True)
class DbgClk:
    """The stamps record as the kernel / the adapter / the decoder read it (config = truth; every number derives from CFG)."""

    SMEM: int  # byte offset of the record in SMEM (the bar block's spare)
    WORDS: int
    BYTES: int
    MAGIC: int
    BLOCKED_THRESH: int
    HIST_EDGES: tuple
    LEADER_BLOCKED_STRIDE: int  # word(blocked:leader:X) - word(wait:leader:X)
    RECORD_ALIGN: int
    # the ship -> recv handshake's tag arithmetic (header section 11): the epilogue lead tags its ship(t) with its tile count BEFORE the tile's
    # increment (= t within the CTA's running count); an MMA warp increments its own count at the tile top, so the wait that consumes ship(t)
    # finds expected = count - TAG_BACK: the leader's in-tile issue in tile t+1 consumes ship(t) (count t+2 -> back 2), its tail issue after
    # the loop consumes ship(hi-1) (count = tiles -> back 1), the follower's relay in tile t consumes ship(t) (count t+1 -> back 1)
    TAG_BACK_IN_TILE: int
    TAG_BACK_TAIL: int
    TAG_BACK_RELAY: int
    WORD: dict  # name -> word index (DBG_CLK_HEADER, wait:<role>:<barrier>, blocked:leader:<barrier>, DBG_CLK_SEGMENTS)
    ROLE_WAITS: dict
    HEADER: tuple
    SEGMENTS: tuple


def _dbg_clk(layout_end, tmem_handle):
    """The STAMPS record of one (roles, stats_form) geometry: it IS the bar block's spare [layout_end, tmem_handle) -- SMEM = the layout's END,
    WORDS = the spare / 4 (DERIVED: 96 under ldg, 88 under tma_ring), the same word map for every geometry."""
    words = (tmem_handle - layout_end) // 4
    return DbgClk(
        SMEM=layout_end,
        WORDS=words,
        BYTES=words * 4,
        MAGIC=CFG.DBG_CLK_MAGIC,
        BLOCKED_THRESH=CFG.DBG_CLK_BLOCKED_THRESH,
        HIST_EDGES=CFG.DBG_CLK_HIST_EDGES,
        LEADER_BLOCKED_STRIDE=len(DBG_CLK_ROLE_WAITS["leader"]),
        RECORD_ALIGN=CFG.DBG_CLK_RECORD_ALIGN,
        TAG_BACK_IN_TILE=2,
        TAG_BACK_TAIL=1,
        TAG_BACK_RELAY=1,
        WORD=_dbg_clk_layout(),
        ROLE_WAITS=DBG_CLK_ROLE_WAITS,
        HEADER=DBG_CLK_HEADER,
        SEGMENTS=DBG_CLK_SEGMENTS,
    )


DBG_CLK = _dbg_clk(BAR_LAYOUT_END, CFG.TMEM_HANDLE)  # ldg (16w): [230960, 231344) = 96 words, 82 used
# RESOURCE-NOTE: STAMPS record = (TMEM_HANDLE - BAR_LAYOUT_END) / 4 = 88 words under tma_ring: derive it
DBG_CLK_RING = _dbg_clk(BAR_LAYOUT_RING_END, CFG_RING.TMEM_HANDLE)  # tma_ring (16w): [232016, 232368) = 88 words, 84 used
# tmem_map = dq_576: the SAME word map (every index unchanged: the kernel's `_W[...]` sites are one module-level map) over the spare the shorter bar
# block leaves -- the four S-tail words (`dbg_clk_dead_words`) have no site under the map and read 0 (the record is still valid: magic, tiles, every
# other wait); WORDS derived from the layout's END, never a literal.
DBG_CLK_DQ576 = _dbg_clk(BAR_LAYOUT_DQ576_END, CFG_DQ576.TMEM_HANDLE)  # ldg x dq_576: [230928, 231344) = 104 words, 84 named (4 dead)
DBG_CLK_RING_DQ576 = _dbg_clk(BAR_LAYOUT_RING_DQ576_END, CFG_RING_DQ576.TMEM_HANDLE)  # tma_ring x dq_576: [231984, 232368) = 96 words, 84 named (4 dead)
# (DBG_CLK_RING_DQ576_B2 / DBG_CLK_BY_GEOMETRY are built below BAR_LAYOUT_XFA4, with the B2 table)


def dbg_clk_dead_words(tmem_map, smem_map="sm100"):
    """The STAMPS words with NO site under (``tmem_map``, ``smem_map``) (they read 0 in every record): the wait / blocked words of the rows the TMEM
    map deletes (``DELETED_BARRIERS_DQ576`` under dq_576; none under s_tail) and of the row the SMEM map ADDS (``ADDED_BARRIERS_B2``: dq_read_done has
    a site under rubin_b2 only).  The decoder and the dump test skip exactly these."""
    if tmem_map not in CFG.TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {CFG.TMEM_MAPS}; got {tmem_map!r}")
    if smem_map not in CFG.SMEM_MAPS:
        raise ValueError(f"smem_map must be one of {CFG.SMEM_MAPS}; got {smem_map!r}")
    dead = (DELETED_BARRIERS_DQ576 if tmem_map == "dq_576" else ()) + (() if smem_map == "rubin_b2" else ADDED_BARRIERS_B2)
    return tuple(f"{kind}:leader:{b}" for b in dead for kind in ("wait", "blocked"))


def dbg_clk_for_stats_form(name=None, roles=None, tmem_map=None, smem_map=None):
    """The STAMPS record a (resolved or ``None``) stats form renders with on a (resolved) TMEM map and SMEM map: ``DBG_CLK_BY_GEOMETRY`` (the same word
    map everywhere); ``None`` stats form resolves per role shape; ``tmem_map`` ``None`` = the s_tail map's record, ``smem_map`` ``None`` = the sm100
    map's (the per-part maps are the resolvers': the prepared module passes the resolved names)."""
    tmap = CFG.TMEM_MAP if tmem_map is None else tmem_map
    if tmap not in CFG.TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {CFG.TMEM_MAPS}; got {tmap!r}")
    smap = CFG.SMEM_MAP if smem_map is None else smem_map
    if smap not in CFG.SMEM_MAPS:
        raise ValueError(f"smem_map must be one of {CFG.SMEM_MAPS}; got {smap!r}")
    return DBG_CLK_BY_GEOMETRY[(resolve_stats_form(name, roles), tmap, smap)]


def dbg_clk_records(n_batch, nheads, max_skv):
    """CTA records per launch = the s_tail MAIN kernel's DECLARED grid (header section 14: ceil(max_skv / THD_UNIT_ROWS) pairs, at least one, x
    CGA_M CTAs x nheads x n_batch).  Every CTA that RUNS writes its record at exit (a zero-trip CTA writes zeros + the magic); a cluster the CLC
    scheduler cancelled never launches and leaves its two records unwritten (the region is not zeroed: the execute path allocates / converts
    nothing), so a reader keeps the records whose first word is DBG_CLK.MAGIC (the decoder reports the never-launched count)."""
    if any(type(n) is not int or n <= 0 for n in (n_batch, nheads, max_skv)):
        raise ValueError("dbg_clk_records wants positive ints (n_batch, nheads, max_skv)")
    return max((max_skv + CFG.THD_UNIT_ROWS - 1) // CFG.THD_UNIT_ROWS, 1) * CFG.CGA_M * nheads * n_batch


def dbg_clk_bytes(n_batch, nheads, max_skv, stats_form=None, tmem_map=None, smem_map=None):
    return dbg_clk_records(n_batch, nheads, max_skv) * dbg_clk_for_stats_form(stats_form, None, tmem_map, smem_map).BYTES


def dbg_clk_region_offset(regions_end):
    """Byte offset of the stamps region inside the chain's workspace payload: the first RECORD_ALIGN boundary at or past the end of the chain's own
    regions (``thd_workspace_offsets``' total).  Pure arithmetic: the prepared host calls it on its runtime Int64, the adapter on Python ints."""
    return (regions_end + (DBG_CLK.RECORD_ALIGN - 1)) & -DBG_CLK.RECORD_ALIGN


# Alias tenants of the time-shared bytes: (tenant, bytes it occupies, the seam the inspector owns).  The first body's dq_staging row is DELETED:
# dq_staging alias on ds_own+XCHG dropped: dq_done(t) lands ~150 clk before dS(t+1)
ALIASES = (
    ("dk_store", "ds_own(rank) + XCHG", "dq_done(t_last) (the dQ MMA read sDS; the ship read XCHG) + dvk_empty before the next item's first dS write"),
    ("dv_store", "K2", "last tile's dq_done -> k2_empty before the next block's K2 load"),
    ("ds_recv(rank)", "DS, the peer-rank half", "written ONLY by the peer's bulk copy; its next write waits the peer's dq_done(t)"),
)

# (name, byte offset, active slots, init count, producer kind, scope) -- the barrier audit (the alternate role shape 16w, the audit record
# section 8, PASTED): the S-tail table minus ds_tmem / dp_empty, ds_smem counted from the softmax; re-packed in declaration order, 16 B per row;
# 32 names / 37 slots / 512 B; END 230912 (432 B spare of [230400, 231344)).  `validate_config` re-derives it from BAR_LAYOUT (filter + count + re-pack)
# and asserts equality, so the delta against the audited 34-row table is exactly the two deletions and ds_smem's count.  Selected by ``warp_roles``.
BAR_LAYOUT_XFA4 = (
    ("q1_full", 230400, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # Q1_TX; ONE stage
    ("q1_empty", 230416, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after each S^T issue; DRAIN x Q1_STAGES = 1
    ("q2_full", 230432, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),
    ("q2_empty", 230448, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("do1_full", 230464, 2, CFG.ONE_LANE, "tma_leader", "LEADER"),  # 2 half slots x DO1_HALF_TX, ONE do1_state
    ("do1_empty", 230480, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("do2_full", 230496, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),
    ("do2_empty", 230512, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("k1_full", 230528, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),
    ("k1_empty", 230544, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("v_full", 230560, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),
    ("v_empty", 230576, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("k2_full", 230592, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),
    ("k2_empty", 230608, 1, CFG.STORE_LANES, "thread", "LOCAL"),  # store warp after the dV store drain; fix-A back-edge via dv_full
    ("s_done", 230624, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    ("dp_done", 230640, 1, CFG.ONE_LANE, "mma", "LOCAL"),
    (
        "dq_done",
        230656,
        1,
        CFG.ONE_LANE,
        "mma",
        "LOCAL",
    ),  # ONE waiter set: the reduce group's `dq_state` st(0) (the readout); the softmax's sDS WAR wait is the `dq_war_done` row below
    ("p_ready", 230672, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),
    ("ds_ready", 230688, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # arrived BEFORE the softmax's dq_war wait: dK(t) never waits dQ(t-1)
    ("s_read", 230704, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # KEPT (X1): every tile after the S^T ld + wait(LOAD); MMA(L) slot 2
    (
        "s_tail_empty",
        230720,
        1,
        CFG.PAIR_EPILOGUE_LANES,
        "leader",
        "LEADER",
    ),  # the REDUCE group (the same lanes) after the x64 dQ ld; MMA(L) D1 sites; pre-armed
    (
        "ds_smem",
        230736,
        1,
        CFG.SOFTMAX_LANES,
        "thread",
        "LOCAL",
    ),  # X2: SOFTMAX 256 bare after 4 own + 4 XCHG STS + fence.proxy.async; L dQ gate (xchg_state), F relay
    (
        "ds_recv_full",
        230752,
        1,
        CFG.ONE_LANE,
        "tma_dsmem",
        "LOCAL",
    ),  # X4: softmax warp 0 elect arms XCHG_TX AFTER the XCHG bar.sync, then ships; the PEER's copy lands here
    ("ds_pair_ready", 230768, 1, CFG.ONE_LANE, "peer", "LEADER"),  # follower MMA warp (ds_smem + ds_recv_full observed) -> leader; waited on xchg_state
    ("dq_empty", 230784, 2, CFG.STORE_LANES, "thread", "LOCAL"),  # store warp per slot per round (closed release); the REDUCE group waits both slots (D4 pair)
    ("dq_full", 230800, 2, CFG.DQ_SLICE_LANES, "thread", "LOCAL"),  # the REDUCE warp pair of d half per slot per round
    ("dv_full", 230816, 1, CFG.DV_FULL_LANES, "thread", "LOCAL"),  # reduce 128 + loader 32 (fix A)
    ("dk_full", 230832, 1, CFG.EPILOGUE_LANES, "thread", "LOCAL"),
    ("dvk_empty", 230848, 1, CFG.PAIR_STORE_LANES, "leader", "LEADER"),  # both store warps -> MMA(L); pre-armed; TWO seams (TMEM reuse + the dK staging WAR)
    ("sched_full", 230864, 2, CFG.ONE_LANE, "tma", "LOCAL"),
    ("sched_empty", 230880, 2, CFG.PAIR_SCHED_CREDIT_WARPS, "peer", "LEADER"),  # 15 warps x 2 CTAs
    ("tmem_dealloc", 230896, 1, CFG.TMEM_DEALLOC_ARRIVERS, "thread", "LOCAL"),
    # section 15 (10:55 UTC): the softmax's sDS WAR signal -- MMA(L) `_commit_pair` inside `_dq_issue` at the two dQ sites only, BEFORE dq_done, never at
    # the zero-trip site; waited ONLY by the softmax (`dq_war` st(1), one wait per live tile, no zero-trip wait / advance, one post-loop drain)
    ("dq_war_done", 230912, 1, CFG.ONE_LANE, "mma", "LOCAL"),
)
BAR_LAYOUT_XFA4_END = 230928  # 33 names / 38 slots, 528 B (416 B of the 944 spare)
BAR_KIND_COUNTS_XFA4 = {"tma_leader": 7, "mma": 10, "thread": 7, "leader": 5, "peer": 2, "tma_dsmem": 1, "tma": 1}  # 33
BAR_LAYOUT_XFA4_STATUS = (
    "FILLED by the barrier audit (SKELETON, the alternate role shape 16w, the audit record sections 0-14) + the section-15 ZERO-TRIP "
    "CORRECTION 10:55 UTC: 33 names / 38 slots; ds_tmem + dp_empty DELETED (X5 / section 6), ds_smem 128 -> 256 from the SOFTMAX (X2), ds_recv_full armed "
    "by softmax warp 0 after the 256-lane XCHG bar.sync (X3 / X4), dq_war_done NEW = the softmax's WAR signal (section 15.3; dq_done back to the reduce "
    "group alone), s_read kept (X1), dQ ring form A (X6); AUDIT from the committed source owed (section 12 + 15.5 punch lists)"
)


BAR_LAYOUT_RING_DQ576_B2 = layout_for_geometry(CFG_RING_DQ576_B2)  # 16w x tma_ring x dq_576 x rubin_b2: 36 names / 51 slots / 592 B, END 313936
BAR_LAYOUT_RING_DQ576_B2_END = layout_end(BAR_LAYOUT_RING_DQ576_B2)
BAR_KIND_COUNTS_RING_DQ576_B2 = {"tma_leader": 7, "mma": 9, "thread": 10, "leader": 5, "peer": 2, "tma_dsmem": 1, "tma": 2}  # 36: dq_576's + dq_read_done
BAR_LAYOUT_RING_DQ576_B2_STATUS = (
    "SKELETON by the barrier audit (R2.c 2026-11-03, BARRIER-SKELETON-RUBIN sections 2.1-2.7 + 1.7): the dq_576 ring table with q1_full / q1_empty / "
    "ds_smem / ds_recv_full / ds_pair_ready at 2 slots, do1_full at 4 and do1_empty at 2 (the same init per slot), + dq_read_done (LEADER 256, "
    "pre-armed waiter inside _dq_issue, one drain), re-packed with the 32-B do1_full row; AUDIT of the committed body owed"
)

# smem_map = rubin_b2: the SAME word map over the spare its re-based bar block leaves (the leader's dq_read_done wait / blocked words are LIVE here only)
DBG_CLK_RING_DQ576_B2 = _dbg_clk(
    BAR_LAYOUT_RING_DQ576_B2_END, CFG_RING_DQ576_B2.TMEM_HANDLE
)  # tma_ring x dq_576 x rubin_b2: [313936, 314288) = 88 words, 86 named
DBG_CLK_BY_GEOMETRY = {
    ("ldg", "s_tail", "sm100"): DBG_CLK,
    ("tma_ring", "s_tail", "sm100"): DBG_CLK_RING,
    ("ldg", "dq_576", "sm100"): DBG_CLK_DQ576,
    ("tma_ring", "dq_576", "sm100"): DBG_CLK_RING_DQ576,
    ("tma_ring", "dq_576", "rubin_b2"): DBG_CLK_RING_DQ576_B2,
    ("tma_ring", "s_tail", "sm100_oversized"): DBG_CLK_RING,  # the DIAGNOSTIC map: the sm100 records (the same body)
    ("tma_ring", "dq_576", "sm100_oversized"): DBG_CLK_RING_DQ576,
}


def bar_layout_for_roles(roles, stats_form=None, tmem_map=None, smem_map=None):
    """The barrier table a (resolved) role shape renders with under a stats form (``None`` = the form the shape renders when the field is unset:
    tma_ring under 16w, ldg under 16w_fa4) and a (resolved) TMEM map (``None`` = the s_tail map; the per-part map is resolve_tmem_map's): the
    audited 35-row ``BAR_LAYOUT`` (``16w`` x ldg) or the audit's 33-row ``BAR_LAYOUT_XFA4``; under ``tma_ring`` the rows re-based onto CFG_RING.BARS
    + the stats ring's two rows (the 16w_fa4 x tma_ring table is derived for completeness: the combination declines by name in
    validate_fields_for_roles until the lift); under ``dq_576`` (16w only: the shape rule of validate_fields_for_roles) the s_read / s_tail_empty
    rows are deleted and the rest re-packed (``BAR_LAYOUT_DQ576`` / ``BAR_LAYOUT_RING_DQ576``); under ``smem_map = "rubin_b2"`` (16w x tma_ring x
    dq_576 only: the rule of validate_fields_for_roles) the ring rows gain their slots and dq_read_done joins (``BAR_LAYOUT_RING_DQ576_B2``)."""
    tmap = CFG.TMEM_MAP if tmem_map is None else tmem_map
    if tmap not in CFG.TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {CFG.TMEM_MAPS}; got {tmap!r}")
    smap = CFG.SMEM_MAP if smem_map is None else smem_map
    if smap not in CFG.SMEM_MAPS:
        raise ValueError(f"smem_map must be one of {CFG.SMEM_MAPS}; got {smap!r}")
    base = {"16w": BAR_LAYOUT, "16w_fa4": BAR_LAYOUT_XFA4}[roles]
    ldg = resolve_stats_form(stats_form, roles) == "ldg"  # the base tables are the ldg layouts (the merged rendering's)
    if smap == "rubin_b2":
        if tmap == CFG.TMEM_MAP or roles != CFG.WARP_ROLES_DEFAULT or ldg:
            raise ValueError(
                f"smem_map {smap!r} is built under warp_roles 16w x stats_form tma_ring x tmem_map dq_576 only (validate_fields_for_roles); no table exists for it"
            )
        return BAR_LAYOUT_RING_DQ576_B2
    # smem_map "sm100_oversized" (DIAGNOSTIC) = the sm100 tables: the same rows at the same addresses
    if tmap == CFG.TMEM_MAP:
        if ldg:
            return base
        return BAR_LAYOUT_RING if roles == "16w" else _ring_layout(base, CFG_RING.BARS)
    if roles != CFG.WARP_ROLES_DEFAULT:
        raise ValueError(f"tmem_map {tmap!r} is not built under warp_roles {roles!r} (validate_fields_for_roles); no table exists for it")
    return BAR_LAYOUT_DQ576 if ldg else BAR_LAYOUT_RING_DQ576


def bar_layout_end_for_roles(roles, stats_form=None, tmem_map=None, smem_map=None):
    return layout_end(bar_layout_for_roles(roles, stats_form, tmem_map, smem_map))


def bar_kind_counts_for_roles(roles, stats_form=None, tmem_map=None, smem_map=None):
    counts = dict({"16w": BAR_KIND_COUNTS, "16w_fa4": BAR_KIND_COUNTS_XFA4}[roles])
    if resolve_stats_form(stats_form, roles) != "ldg":
        for _name, _slots, _count, kind, _scope in STATS_RING_ROWS:
            counts[kind] = counts.get(kind, 0) + 1
    if (CFG.TMEM_MAP if tmem_map is None else tmem_map) != CFG.TMEM_MAP:
        kinds = {row[0]: row[4] for row in BAR_LAYOUT}
        for name in DELETED_BARRIERS_DQ576:  # the two LEADER rows
            counts[kinds[name]] -= 1
    if (CFG.SMEM_MAP if smem_map is None else smem_map) == "rubin_b2":
        counts[DQ_READ_DONE_ROW[3]] += 1  # the dq_read_done row (LEADER); sm100_oversized = the sm100 counts
    return counts


def validate_config():
    """The arm-(ii) deltas on top of the first body's geometry; CPU only (raises at import).  Every number the kernel header quotes is here."""
    c = CFG
    base = CFG_BASE
    assert c.TMEM_MAP == "s_tail"
    # unchanged geometry (inherited): MMA shapes, pair extents, boxes, warps, registers, the per-stage slab bytes, the lane constants
    assert (c.M, c.N, c.D, c.K_HW, c.BPE, c.BPE_ACC, c.CTA_MMA, c.CGA_M, c.CGA_N) == (base.M, base.N, base.D, base.K_HW, base.BPE, base.BPE_ACC, 2, 2, 1)
    assert (c.PAIR_KV_ROWS, c.Q_HALF, c.D_HALF, c.COLLECTIVE_M_KV, c.COLLECTIVE_M_Q) == (256, 64, 64, 256, 128)
    assert (c.Q1_BYTES, c.Q2_BYTES, c.DO1_BYTES, c.DO2_BYTES) == (16384,) * 4 and (c.K1_BYTES, c.K2_BYTES, c.V_BYTES, c.DS_BYTES) == (32768,) * 4
    assert (c.DS_HALF_BYTES, c.XCHG_BYTES, c.DQ_SLICES, c.DQ_SLICE_BYTES) == (16384, 16384, 4, 8192)
    assert (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS, c.ENTRY_REGS) == (144, 144, 80, 128)
    assert (c.MAPS_PER_SEQUENCE, c.MAP_BOX_ROWS, c.THD_UNIT_ROWS, c.Q_TILES_PER_PAIR) == (9, base.MAP_BOX_ROWS, 256, 2)
    assert (c.PACK_QUARTER_COLS, c.PACK_GROUP_COLS, c.K_STEPS_PER_BOX, c.K2_HALF_BYTES, c.DS_A_LBO) == (16, 32, 4, 16384, 8192)
    assert [c.ts_k_perm(k) for k in range(c.N // c.K_HW)] == [base.ts_k_perm(k) for k in range(base.N // base.K_HW)]
    assert c.DIAG_2CTA_NAMES == base.DIAG_2CTA_NAMES
    # ring depths: Q1 one stage; the dQ staging ring 2 slots x 2 rounds = the 4 reduce-add slices per tile
    assert (c.Q1_STAGES, c.Q2_STAGES, c.DO_STAGES, c.KV_STAGES, c.SCHED_STAGES) == (1, 1, 1, 1, 2)
    assert (c.DQ_SLOTS, c.DQ_ROUNDS) == (2, 2) and c.DQ_SLOTS * c.DQ_ROUNDS == c.DQ_SLICES
    assert c.DQ_BYTES == c.DQ_STAGING_BYTES == c.DQ_SLOTS * c.DQ_SLICE_BYTES == 16384
    # the slab chain (declaration order; every slab 1024-B aligned; STATS + DELTA share one 1 KiB): Q -16384, +DQ 16384, SLABS_END unchanged
    order = (c.Q1, c.Q2, c.DO1, c.DO2, c.K1, c.K2, c.V, c.DS, c.XCHG, c.DQ, c.STATS, c.DELTA, c.SLABS_END)
    assert order == (0, 16384, 32768, 49152, 65536, 98304, 131072, 163840, 196608, 212992, 229376, 229888, 230400), order
    assert all(off % 1024 == 0 for off in order[:-2]) and c.DELTA % 512 == 0
    assert c.Q1_STAGES * c.Q1_BYTES + c.Q2_STAGES * c.Q2_BYTES == 32768 and c.DQ == c.XCHG + c.XCHG_BYTES
    assert (c.ALIAS, c.ALIAS_BYTES) == (c.DS, 49152) and c.ALIAS + c.ALIAS_BYTES == c.DQ  # the alias region ends where the ring begins
    assert c.DVK_STORE_BYTES <= c.K2_BYTES and c.DVK_STORE_BYTES <= c.ALIAS_BYTES - c.DS_HALF_BYTES
    assert (c.SMEM_MAP, c.DO1_STAGES, c.DS_STAGES, c.XCHG_STAGES, c.SMEM_CAP, c.SMEM_OPT_IN_CAP, c.SMEM_OVERSIZED_CAP) == (
        "sm100",
        1,
        1,
        1,
        232448,
        232448,
        334848,
    )
    assert c.DO_STAGES == c.DO1_STAGES == 1 and c.DO2 == c.DO1 + c.DO1_BYTES and c.XCHG == c.DS + c.DS_BYTES and c.DQ == c.XCHG + c.XCHG_BYTES  # one stage each
    assert [c.q1_stage(0), c.do1_stage(0), c.ds_stage(0), c.xchg_stage(0)] == [c.Q1, c.DO1, c.DS, c.XCHG]
    assert all(c.ds_own_stage(rk, 0) == c.ds_own(rk) and c.ds_recv_stage(rk, 0) == c.ds_recv(rk) for rk in (0, 1))
    assert c.desc_roots() == (0, 16384, 32768, 49152, 65536, 98304, 114688, 131072, 163840) and max(c.desc_roots()) < c.TCGEN05_V0_ADDR_LIMIT == 262144
    assert (c.SCAFFOLD_BYTES, c.SMEM_BYTES, c.BARS, c.TMEM_HANDLE, c.SCHED_PAYLOAD, c.BAR_BLOCK_BYTES) == (1024, 231424, 230400, 231344, 231360, 944)
    assert c.SMEM_BYTES == base.SMEM_BYTES and c.SLABS_END == base.SLABS_END  # the same bytes, re-partitioned
    # the dedicated ring: disjoint from every alias tenant, inside [DQ, STATS); slot s <-> the epilogue warp pair of d half s
    slots = [c.dq_slot(s) for s in range(c.DQ_SLOTS)]
    assert slots == [212992, 221184] and all(s % 1024 == 0 and c.DQ <= s and s + c.DQ_SLICE_BYTES <= c.STATS for s in slots)
    uses = [c.dq_use(u) for u in range(c.DQ_SLICES)]
    assert uses == [(0, 0), (1, 2), (0, 1), (1, 3)], uses  # round A: slices 0, 2; round B: slices 1, 3
    assert all(slot == u % c.DQ_SLOTS and sl // c.DQ_ROUNDS == slot and sl % c.DQ_ROUNDS == u // c.DQ_SLOTS for u, (slot, sl) in enumerate(uses))
    assert sorted(sl for _, sl in uses) == list(range(c.DQ_SLICES)) and c.DQ_SLICE_ORDER == (0, 2, 1, 3)
    for rank in (0, 1):
        own, recv = c.ds_own(rank), c.ds_recv(rank)
        assert {own, recv} == {c.DS, c.DS + c.DS_HALF_BYTES} and own != recv
        assert {c.dk_store(rank, 0), c.dk_store(rank, 1)} == {own, c.XCHG}  # the dK staging keeps its alias (item end; dq_done(t_last) + dvk_empty)
        for s in slots:
            assert s >= c.XCHG + c.XCHG_BYTES  # never inside the dS / XCHG bytes
    assert (c.dv_store(0), c.dv_store(1)) == (c.K2, c.K2 + 16384)
    # dkv_accum = fp32_reduce (form (i)): 5 + 3 fp32 16 KiB slots over [K2, DQ) = the dV / dK bf16 staging bytes + the dS slab; 1024-B aligned,
    # disjoint, never the sDQ ring; round 2 reuses three of round 1's slots after dvk_stage_empty
    assert c.DKV_ACC_SLICES == c.D // c.DQ_BOX == 4 and (c.DKV_ACC_R1_SLICES, c.DKV_ACC_R2_SLICES) == (5, 3)
    assert c.DKV_ACC_R1_SLICES + c.DKV_ACC_R2_SLICES == 2 * c.DKV_ACC_SLICES  # dV's four + dK's four slices, once each
    assert c.DKV_ACC_SLICE_BYTES == c.M * c.DQ_BOX * c.BPE_ACC == 16384 == c.DVK_STORE_HALF_BYTES == c.DS_HALF_BYTES == c.XCHG_BYTES
    acc_r1 = [c.dkv_acc_r1_slot(i) for i in range(c.DKV_ACC_R1_SLICES)]
    acc_r2 = [c.dkv_acc_r2_slot(i) for i in range(c.DKV_ACC_R2_SLICES)]
    assert acc_r1 == [c.K2, c.K2 + 16384, c.DS, c.DS + 16384, c.XCHG] and acc_r2 == acc_r1[:3], (acc_r1, acc_r2)
    assert all(s % 1024 == 0 and c.K2 <= s and s + c.DKV_ACC_SLICE_BYTES <= c.DQ for s in acc_r1)  # inside [K2, DQ): never the sDQ ring
    acc_spans = sorted((s, s + c.DKV_ACC_SLICE_BYTES) for s in acc_r1)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(acc_spans, acc_spans[1:]))
    assert {acc_r1[2], acc_r1[3]} == {c.ds_own(0), c.ds_recv(0)} == {c.ds_own(1), c.ds_recv(1)}  # both dS halves: dead after dq_done(hi-1) (table 9)
    assert {acc_r1[0], acc_r1[1]} == {c.dv_store(0), c.dv_store(1)} and acc_r1[4] == c.XCHG == c.dk_store(0, 1) == c.dk_store(1, 1)
    try:
        c.dq_slice(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("dq_slice must decline on the s_tail map (no dQ staging alias)")
    # TMEM: the dQ image in S^T's dead tail; [TMEM_DEAD_COL, DV_COL) dead; one alloc / dealloc constant
    assert (c.S_COL, c.S_COLS, c.P_COL, c.PACK_COLS) == (0, 128, 0, 64)
    assert (c.DP_COL, c.DP_COLS, c.DS_COL, c.DQ_COL, c.DQ_COLS) == (128, 128, 128, 64, 64)
    assert c.P_COL + c.PACK_COLS == c.DQ_COL and c.DQ_COL + c.DQ_COLS == c.S_COL + c.S_COLS  # the image fills exactly S^T's dead tail
    assert c.TMEM_DEAD_COL == c.DS_COL + c.PACK_COLS == 192 and c.TMEM_DEAD_COL <= c.DV_COL  # the first body's dQ columns are DEAD
    assert (c.DV_COL, c.DV_COLS, c.DK_COL, c.DK_COLS, c.TMEM_USED, c.TMEM_COLS) == (256, 128, 384, 128, 512, 512)
    assert c.TMEM_USED <= c.TMEM_COLS and c.DQ_LANE_SPLIT == 64 and c.TMEM_COLS == base.TMEM_COLS and c.TMEM_IS_EXCLUSIVE is False
    # tmem_map = dq_576 (the two Rubin records): the SAME map but the dQ image = the 64 columns past dK, TMEM_COLS = the whole 576-column carve
    # (alloc == dealloc constant), exclusive; [192, 256) stays dead; the lane split and every other column unchanged; SMEM untouched
    for g, s_tail_twin in ((CFG_DQ576, c), (CFG_RING_DQ576, CFG_RING)):
        assert (
            g.TMEM_MAP == "dq_576"
            and g.TMEM_IS_EXCLUSIVE is True
            and g.TMEM_COLS == TMEM_COLS_SM107 == 576
            and g.TMEM_IS_EXCLUSIVE == (g.TMEM_COLS > base.TMEM_COLS)
        )
        assert g.DQ_COL == g.DK_COL + g.DK_COLS == 512 and g.DQ_COL + g.DQ_COLS == g.TMEM_USED == g.TMEM_COLS  # the image fills the carve past dK exactly
        assert (g.S_COL, g.S_COLS, g.P_COL, g.PACK_COLS, g.DP_COL, g.DP_COLS, g.DS_COL, g.DQ_COLS) == (0, 128, 0, 64, 128, 128, 128, 64)
        assert (g.DV_COL, g.DV_COLS, g.DK_COL, g.DK_COLS, g.TMEM_DEAD_COL, g.DQ_LANE_SPLIT) == (256, 128, 384, 128, 192, 64)
        assert g.DQ_COL >= g.DK_COL + g.DK_COLS and g.TMEM_DEAD_COL + g.PACK_COLS <= g.DV_COL  # no 512-column tenant under the image; the dead half stays dead
        assert (g.SMEM_BYTES, g.BARS, g.TMEM_HANDLE, g.SCHED_PAYLOAD, g.SLABS_END) == (
            s_tail_twin.SMEM_BYTES,
            s_tail_twin.BARS,
            s_tail_twin.TMEM_HANDLE,
            s_tail_twin.SCHED_PAYLOAD,
            s_tail_twin.SLABS_END,
        )  # SMEM is not the map's business
    assert c.TMEM_MAPS == ("s_tail", "dq_576") == tuple(TMEM_MAP_CODES) and TMEM_MAP_CODES == {"s_tail": 0, "dq_576": 1} and c.TMEM_MAP == c.TMEM_MAPS[0]
    assert c.DIAG_2CTA_S_TAIL_PROBES == ("s_tail_wait_off", "s_read_wait_off", "s_tail_and_s_read_wait_off") == c.DIAG_2CTA_NAMES_ON_MAP[4:]
    try:
        replace(c, TMEM_MAP="bogus").DQ_COL
    except ValueError:
        pass
    else:
        raise AssertionError("a record outside the map domain must decline at its first map-dependent property")
    # the probes (typed; the body has code for every name of the code space; the two probes of an earlier pair body are outside the domain)
    assert (
        c.DIAG_2CTA_NAMES_ON_MAP
        == (
            "dq_reduce_add_off",
            "exchange_off",
            "stats_one_chunk",
            "sdq_sts_off",
            "s_tail_wait_off",
            "s_read_wait_off",
            "s_tail_and_s_read_wait_off",
        )
        == c.DIAG_2CTA_NAMES
    )
    assert resolve_diag_2cta(None) is None and [resolve_diag_2cta(n) for n in c.DIAG_2CTA_NAMES_ON_MAP] == list(c.DIAG_2CTA_NAMES_ON_MAP)
    for bad_diag in ("dp_empty_wait_off", "ds_gates_off", "bogus", "", 1):
        try:
            resolve_diag_2cta(bad_diag)
        except ValueError as error:
            assert "diag_2cta must be None or one of" in str(error), str(error)
        else:
            raise AssertionError((bad_diag, "must decline by name"))
    assert (c.DIAG_EXCHANGE_OFF_TX, c.DIAG_STATS_CHUNK) == (0, 0) and 0 <= c.DIAG_STATS_CHUNK < c.DIAG_STATS_CHUNKS_PER_QUARTER == 8
    # the S-tail wait probes: the leader's static wait sites per row (header section 5: prologue + lookahead + tail + drain; slot 2 every tile)
    assert (c.DIAG_S_TAIL_WAIT_SITES, c.DIAG_S_READ_WAIT_SITES) == (4, 1)
    assert (c.DIAG_STATS_LDS_PER_LANE, c.DIAG_STATS_LDS_REMOVED_PER_LANE) == (32, 28) and c.DIAG_STATS_LDS_REMOVED_PER_LANE * c.SOFTMAX_WARPS == 224
    # the STAMPS lever: typed bool; the record fits the bar block's spare; the layout is unique, contiguous, and names real barriers
    assert c.DEBUG_CLK_DEFAULT is False and resolve_debug_clk(None) is False and resolve_debug_clk(True) is True and resolve_debug_clk(False) is False
    for bad_flag in (1, "True", 0, ""):
        try:
            resolve_debug_clk(bad_flag)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_flag, "debug_clk must decline by name"))
    d = DBG_CLK
    assert d.SMEM == BAR_LAYOUT_END == 230960 and d.WORDS == 96 and d.BYTES == 384 and d.SMEM % 16 == 0  # derived: the ldg spare (82 words used)
    assert d.SMEM + d.BYTES == c.TMEM_HANDLE and d.BYTES == c.TMEM_HANDLE - BAR_LAYOUT_END  # the record IS the spare: no byte past it, none left
    assert len(d.WORD) <= d.WORDS and sorted(d.WORD.values()) == list(range(len(d.WORD))) and d.MAGIC == c.DBG_CLK_MAGIC != 0
    assert d.BLOCKED_THRESH == 128 and d.HIST_EDGES == (100, 400, 1600) == tuple(sorted(d.HIST_EDGES)) and d.RECORD_ALIGN == 256
    bar_names = {row[0] for row in BAR_LAYOUT} | set(ADDED_BARRIERS_B2)  # + the row the rubin_b2 table adds (its words are dead elsewhere)
    for role, waits in d.ROLE_WAITS.items():
        assert waits and len(set(waits)) == len(waits) and set(waits) <= bar_names, role
        assert all(f"wait:{role}:{b}" in d.WORD for b in waits), role
    for b in d.ROLE_WAITS["leader"]:
        assert d.WORD[f"blocked:leader:{b}"] - d.WORD[f"wait:leader:{b}"] == d.LEADER_BLOCKED_STRIDE == 18, b
    assert [d.WORD[n] for n in d.HEADER] == list(range(len(d.HEADER))) and d.HEADER[0] == "magic" and len(d.SEGMENTS) == 18
    # 82 ldg words + the leader's dq_read_done wait / blocked pair (rubin_b2; dead = 0 elsewhere) + the 2 ring words
    assert set(d.ROLE_WAITS) == {"leader", "softmax", "epilogue", "follower", "loader", "store"} and len(d.WORD) == 86
    assert "dq_read_done" in d.ROLE_WAITS["leader"] and d.ROLE_WAITS["leader"].index("dq_read_done") == d.ROLE_WAITS["leader"].index("ds_pair_ready") + 1
    assert DBG_CLK_RING_WAITS == (("softmax", "stats_full"), ("loader", "stats_empty"))
    assert [d.WORD[f"wait:{r}:{b}"] for r, b in DBG_CLK_RING_WAITS] == [84, 85]  # appended AFTER every leader / role word (the ring words close the record)
    assert (d.TAG_BACK_IN_TILE, d.TAG_BACK_TAIL, d.TAG_BACK_RELAY) == (2, 1, 1)  # the handshake's tag arithmetic (DbgClk's comment)
    assert (
        dbg_clk_records(1, 1, 1) == c.CGA_M == 2
        and dbg_clk_records(4, 32, 16384) == 64 * 2 * 32 * 4
        and dbg_clk_bytes(1, 1, 256, "ldg") == 2 * DBG_CLK.BYTES == 2 * 384
        and dbg_clk_bytes(1, 1, 256) == dbg_clk_bytes(1, 1, 256, "tma_ring") == 2 * DBG_CLK_RING.BYTES == 2 * 352  # the default form's record
    )
    assert dbg_clk_region_offset(0) == 0 and dbg_clk_region_offset(1) == 256 and dbg_clk_region_offset(256) == 256 and dbg_clk_region_offset(257) == 512
    for bad in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (1.0, 1, 1)):
        try:
            dbg_clk_records(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    # the softmax's MASK FORM (typed; the default is the FSEL rendering: measured neutral against the predicated form)
    assert c.MASK_FORMS == ("fsel", "predicated", "diag_mask_off") and c.MASK_FORM_DEFAULT == "fsel" == c.MASK_FORMS[0]
    assert c.MASK_FORM_DIAGNOSTIC == "diag_mask_off" and c.MASK_FORM_DIAGNOSTIC in c.MASK_FORMS and c.MASK_FORM_DIAGNOSTIC != c.MASK_FORM_DEFAULT
    assert MASK_FORM_CODES == {"fsel": 0, "predicated": 1, "diag_mask_off": 2} and tuple(MASK_FORM_CODES) == c.MASK_FORMS
    assert resolve_mask_form(None) == "fsel" == resolve_mask_form("fsel") and resolve_mask_form("predicated") == "predicated"
    assert resolve_mask_form("diag_mask_off") == "diag_mask_off"
    assert (mask_form_code(None), mask_form_code("fsel"), mask_form_code("predicated"), mask_form_code("diag_mask_off")) == (0, 0, 1, 2)
    for bad_form in ("bogus", "", 1, "FSEL"):
        try:
            resolve_mask_form(bad_form)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_form)
    # the softmax lane's mask cost per tile: one select per cell of its two 32-column q quarters over one keep-word per quarter; the two groups'
    # quarters tile the 128 q columns, so the CTA's 256 softmax lanes select 128 x 128 cells per tile (the sass census' 64 FSEL + 8 R2P per lane)
    assert c.MASK_WORD_COLS == 32 == c.GROUP_QUARTER and c.QUARTERS_PER_GROUP == 2 and c.SOFTMAX_GROUPS == 2
    assert c.MASK_SELECTS_PER_LANE_PER_TILE == 64 == c.QUARTERS_PER_GROUP * c.GROUP_QUARTER and c.MASK_WORDS_PER_LANE_PER_TILE == 2
    assert c.MASK_SELECTS_PER_LANE_PER_TILE * c.SOFTMAX_GROUPS == c.N and c.MASK_SELECTS_PER_LANE_PER_TILE * c.SOFTMAX_LANES == c.M * c.N
    # --- the STATS FORM (typed; the default is the tma_ring rendering under the default roles, ldg under 16w_fa4): the domain, the ring's bytes /
    # roots / depth per geometry, the padded stats region, the two geometries' equality on everything else, the declines ---
    assert c.STATS_FORMS == ("ldg", "tma_ring") and c.STATS_FORM_DEFAULT == "tma_ring" == c.STATS_FORMS[1] and tuple(STATS_FORM_CODES) == c.STATS_FORMS
    assert c.STATS_FORM_FA4 == "ldg" == c.STATS_FORMS[0] and c.STATS_FORM_DEFAULT_BY_ROLES == {"16w": "tma_ring", "16w_fa4": "ldg"}
    assert set(c.STATS_FORM_DEFAULT_BY_ROLES) == set(c.WARP_ROLES) and STATS_FORM_CODES == {"ldg": 0, "tma_ring": 1}
    assert resolve_stats_form(None) == "tma_ring" == resolve_stats_form("tma_ring") == resolve_stats_form(None, "16w")
    assert resolve_stats_form("ldg") == "ldg" == resolve_stats_form(None, "16w_fa4") == resolve_stats_form("ldg", "16w")
    assert (stats_form_code(None), stats_form_code("ldg"), stats_form_code("tma_ring"), stats_form_code(None, "16w_fa4")) == (1, 0, 1, 0)
    try:
        resolve_stats_form(None, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown role shape must decline the unset stats form by name")
    for bad_stats in ("bogus", "", 1, "LDG", "tma"):
        try:
            resolve_stats_form(bad_stats)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_stats)
    assert cfg_for_stats_form(None) is cfg_for_stats_form("tma_ring") is CFG_RING and cfg_for_stats_form("ldg") is cfg_for_stats_form(None, "16w_fa4") is CFG
    r = CFG_RING
    assert (c.STATS_STAGES, r.STATS_STAGES) == (
        1,
        2,
    ) and c.STATS_STAGE_BYTES == r.STATS_STAGE_BYTES == c.STATS_BYTES + c.DELTA_BYTES == 1024 == c.STATS_TX == r.STATS_TX
    assert c.STATS_DELTA_OFF == r.STATS_DELTA_OFF == c.STATS_BYTES == 512 == c.DELTA - c.STATS and c.STATS_RING == r.STATS_RING == c.STATS == 229376
    assert (c.STATS_RING_BYTES, r.STATS_RING_BYTES) == (1024, 2048) and c.STATS_RING_BYTES == c.STATS_BYTES + c.DELTA_BYTES  # ldg: the one slab as today
    # the ldg geometry IS the base chain's (every scaffold offset unchanged: the ldg rendering stays byte-identical to the merged one)
    assert (c.SLABS_END, c.SMEM_BYTES, c.BARS, c.TMEM_HANDLE, c.SCHED_PAYLOAD) == (230400, 231424, 230400, 231344, 231360)
    # the tma_ring geometry: stage 1 = the 1 KiB the scaffold sat on; the scaffold follows by derivation; the footprint is exactly the cap
    assert (r.SLABS_END, r.SMEM_BYTES, r.BARS, r.TMEM_HANDLE, r.SCHED_PAYLOAD, r.BAR_BLOCK_BYTES) == (231424, 232448, 231424, 232368, 232384, 944)
    assert r.SMEM_BYTES == c.SMEM_OPT_IN_CAP == 232448 and r.SMEM_BYTES == c.SMEM_BYTES + c.STATS_STAGE_BYTES and r.SLABS_END == c.SMEM_BYTES
    stages = [r.stats_stage(st) for st in range(r.STATS_STAGES)]
    assert stages == [229376, 230400] and all(st % 1024 == 0 for st in stages) and stages[1] == c.SLABS_END  # stage 1 begins where the ldg slabs end
    assert [r.stats_lse(st) for st in range(r.STATS_STAGES)] == stages and [r.stats_delta(st) for st in range(r.STATS_STAGES)] == [st + 512 for st in stages]
    assert (r.stats_lse(0), r.stats_delta(0), c.stats_lse(0), c.stats_delta(0)) == (c.STATS, c.DELTA, c.STATS, c.DELTA)  # stage 0 = today's slab, both forms
    assert stages[-1] + r.STATS_STAGE_BYTES == r.SLABS_END == r.BARS  # the ring ends where the re-based bar block begins: no gap, no overlap
    # every property but the ring depth and the scaffold chain is equal between the two stats geometries, and every property but the TMEM map's
    # five between the two maps' geometries (the body reads them through the record it was handed)
    moved = {"STATS_STAGES", "STATS_RING_BYTES", "SLABS_END", "SMEM_BYTES", "BARS", "TMEM_HANDLE", "SCHED_PAYLOAD"}
    moved_map = {"TMEM_MAP", "TMEM_COLS", "TMEM_IS_EXCLUSIVE", "DQ_COL", "TMEM_USED"}
    props = [n for n in dir(type(c)) if not n.startswith("_") and isinstance(getattr(type(c), n), property)]
    names_all = set(props) | set(type(c).__dataclass_fields__)
    assert moved <= names_all and moved_map <= names_all, (moved | moved_map) - names_all
    for a, b, skip in ((c, r, moved), (CFG_DQ576, CFG_RING_DQ576, moved), (c, CFG_DQ576, moved_map), (r, CFG_RING_DQ576, moved_map)):
        for n in props:
            if n not in skip:
                assert getattr(a, n) == getattr(b, n), n
        for n in type(c).__dataclass_fields__:
            if n not in skip:
                assert getattr(a, n) == getattr(b, n), n
    # the padded pre-scaled stats region (the contract's truth; one (head, q tile) record = one ring stage image)
    assert c.STATS_KINDS == STATS_KINDS == 2 and c.STATS_RECORD_BYTES == 2 * c.N * c.BPE_ACC == 1024 == c.STATS_STAGE_BYTES and c.STATS_PAD_GRAN == c.N == 128
    assert stats2_pad_total(4, 16384) == 16384 + 4 * 128 and stats2_pad_total(1, 128) == 256
    shape, strides = stats2_shape_strides(32, stats2_pad_total(4, 16384))
    assert shape == (32, (16384 + 512) // 128, 2, 128) and strides == (shape[1] * 256, 256, 128, 1)
    for b, lens in ((3, (129, 0, 127)), (4, (1, 1, 1, 1)), (2, (16384, 16384)), (5, (300, 128, 1000, 0, 257))):
        tq = sum(lens)
        rounded_q = (tq + 127) // 128 * 128
        assert sum((n + 127) // 128 * 128 for n in lens) <= stats2_pad_total(b, rounded_q), (b, lens)  # PAD_TOTAL bounds every sequence's padded block
    # the declines: tma_ring x 16w_fa4 by name; tma_ring passes under 16w with every other field at its default, and WITH debug_clk (the record re-plan)
    assert validate_fields_for_roles("16w", None, stats_form="tma_ring") == "16w" and validate_fields_for_roles("16w", None, stats_form="ldg") == "16w"
    assert validate_fields_for_roles("16w", None, debug_clk=True, stats_form="tma_ring") == "16w"
    assert validate_fields_for_roles("16w_fa4", None, stats_form="ldg") == "16w_fa4"
    assert validate_fields_for_roles("16w_fa4") == "16w_fa4" and validate_fields_for_roles("16w") == "16w"  # unset: ldg under 16w_fa4, tma_ring under 16w
    try:
        validate_fields_for_roles("16w_fa4", None, stats_form="tma_ring")
    except ValueError as error:
        assert "stats_form 'tma_ring' is not built under warp_roles '16w_fa4'" in str(error), str(error)
    else:
        raise AssertionError("tma_ring must decline by name under 16w_fa4 (the first build)")
    # the ring barrier table: the ldg rows re-based onto CFG_RING.BARS + the two ring rows; END 232016; the record = the spare (88 words, 84 used)
    assert STATS_RING_ROWS == (("stats_full", 2, c.ONE_LANE, "tma", "LOCAL"), ("stats_empty", 2, c.SOFTMAX_LANES, "thread", "LOCAL"))
    assert len(BAR_LAYOUT_RING) == 37 and sum(row[2] for row in BAR_LAYOUT_RING) == 44 and BAR_LAYOUT_RING_END == 232016 == r.BARS + 16 * 37
    assert BAR_LAYOUT_RING[:35] == tuple((n, off + r.BARS - c.BARS, sl, cnt, k, sc) for n, off, sl, cnt, k, sc in BAR_LAYOUT)  # the re-base = +1024 per row
    assert [row[:1] + row[2:] for row in BAR_LAYOUT_RING[35:]] == [(n, sl, cnt, k, sc) for n, sl, cnt, k, sc in STATS_RING_ROWS]
    assert (BAR_LAYOUT_RING[35][1], BAR_LAYOUT_RING[36][1]) == (231984, 232000) and BAR_LAYOUT_RING_STATUS.startswith("SKELETON")
    assert (
        BAR_KIND_COUNTS_RING == {"tma_leader": 7, "mma": 9, "thread": 10, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 2}
        and sum(BAR_KIND_COUNTS_RING.values()) == 37
    )
    ring_addresses = set()
    for name, offset, slots_n, count, kind, scope in BAR_LAYOUT_RING:
        for i in range(slots_n):
            address = offset + i * 8
            assert r.BARS <= address and address + 8 <= r.TMEM_HANDLE and address not in ring_addresses, name
            ring_addresses.add(address)
    assert (
        bar_layout_for_roles("16w", "tma_ring") is BAR_LAYOUT_RING
        and bar_layout_for_roles("16w", None) is BAR_LAYOUT_RING  # unset under 16w = the default form's table
        and bar_layout_for_roles("16w", "ldg") is BAR_LAYOUT
        and bar_layout_for_roles("16w_fa4", None) is BAR_LAYOUT_XFA4  # unset under 16w_fa4 = the ldg table
    )
    assert (
        bar_layout_end_for_roles("16w", "tma_ring") == BAR_LAYOUT_RING_END == bar_layout_end_for_roles("16w")
        and bar_layout_end_for_roles("16w", "ldg") == BAR_LAYOUT_END
    )
    assert (
        bar_kind_counts_for_roles("16w", "tma_ring") == BAR_KIND_COUNTS_RING == bar_kind_counts_for_roles("16w")
        and bar_kind_counts_for_roles("16w", "ldg") == BAR_KIND_COUNTS
    )
    fa4_ring = bar_layout_for_roles("16w_fa4", "tma_ring")  # derived for completeness (the combination declines by name): 35 names / 42 slots, END 231984
    assert (
        len(fa4_ring) == 35 and sum(row[2] for row in fa4_ring) == 42 and bar_layout_end_for_roles("16w_fa4", "tma_ring") == 231984 and fa4_ring[0][1] == r.BARS
    )
    dr = DBG_CLK_RING
    assert dr.SMEM == BAR_LAYOUT_RING_END == 232016 and dr.WORDS == 88 and dr.BYTES == 352 and dr.SMEM + dr.BYTES == r.TMEM_HANDLE and dr.SMEM % 16 == 0
    assert dr.WORD == DBG_CLK.WORD and len(dr.WORD) == 86 <= dr.WORDS and dr.MAGIC == DBG_CLK.MAGIC and dbg_clk_for_stats_form("tma_ring") is dr
    assert dbg_clk_for_stats_form(None) is DBG_CLK_RING and dbg_clk_for_stats_form("ldg") is DBG_CLK is dbg_clk_for_stats_form(None, "16w_fa4")
    # BAR_LAYOUT: 34 names / 39 slots, 16-B rows, contiguous from BARS, disjoint, inside [BARS, TMEM_HANDLE); counts by kind; scope by kind
    assert BAR_LAYOUT_STATUS.startswith("FILLED") and len(BAR_LAYOUT) == 35 and sum(row[2] for row in BAR_LAYOUT) == 40
    expected_counts = dict(
        k2_empty=c.STORE_LANES,
        p_ready=c.PAIR_SOFTMAX_LANES,
        ds_ready=c.PAIR_SOFTMAX_LANES,
        s_read=c.PAIR_SOFTMAX_LANES,
        ds_tmem=c.SOFTMAX_LANES,
        dp_empty=c.PAIR_EPILOGUE_LANES,
        s_tail_empty=c.PAIR_EPILOGUE_LANES,
        ds_smem=c.EPILOGUE_LANES,
        ds_pair_ready=c.ONE_LANE,
        dq_empty=c.STORE_LANES,
        dq_full=c.DQ_SLICE_LANES,
        dv_full=c.DV_FULL_LANES,
        dk_full=c.EPILOGUE_LANES,
        dvk_empty=c.PAIR_STORE_LANES,
        sched_empty=c.PAIR_SCHED_CREDIT_WARPS,
        tmem_dealloc=c.TMEM_DEALLOC_ARRIVERS,
        dvk_stage_empty=c.STORE_LANES,
    )
    kind_scope = {"tma_leader": "LEADER", "tma": "LOCAL", "tma_dsmem": "LOCAL", "mma": "LOCAL", "thread": "LOCAL", "leader": "LEADER", "peer": "LEADER"}
    addresses, names, kinds_seen = set(), [], {}
    prev_end = c.BARS
    for name, offset, slots_n, count, kind, scope in BAR_LAYOUT:
        assert name not in names and slots_n in (1, 2) and offset % 16 == 0 and offset == prev_end, name
        names.append(name)
        prev_end = offset + 16
        assert kind in kind_scope and scope == kind_scope[kind], name
        kinds_seen[kind] = kinds_seen.get(kind, 0) + 1
        if kind in ("tma_leader", "tma", "tma_dsmem", "mma"):
            assert count == c.ONE_LANE, name  # one elected arm / one commit per phase (a commit lands ONE arrive per target CTA)
        else:
            assert count == expected_counts[name], name
        for i in range(slots_n):
            address = offset + i * 8
            assert c.BARS <= address and address + 8 <= c.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert prev_end == BAR_LAYOUT_END == c.BARS + 16 * len(BAR_LAYOUT) == 230960 <= c.TMEM_HANDLE and kinds_seen == BAR_KIND_COUNTS
    # D5: the new rows sit next to their siblings (re-packed in declaration order, no 4-slot rows left)
    assert names[names.index("ds_ready") + 1] == "s_read" and names[names.index("dp_empty") + 1] == "s_tail_empty"
    assert {n for n in names if n.endswith("_full") and n not in ("sched_full", "ds_recv_full", "dq_full", "dv_full", "dk_full")} == {
        "q1_full",
        "q2_full",
        "do1_full",
        "do2_full",
        "k1_full",
        "v_full",
        "k2_full",
    }  # the seven cga2 tensor-load barriers: LEADER-armed, LEADER-waited (P9)
    first_body = [row[0] for row in __import__("cudnn.sdpa.bwd.config_sm100_d128_accum_2cta", fromlist=["BAR_LAYOUT"]).BAR_LAYOUT]
    assert [n for n in names if n not in ("s_read", "s_tail_empty", "dvk_stage_empty")] == first_body  # every other row keeps the base table's name and order
    assert names[-1] == "dvk_stage_empty"  # the fp32_reduce row is appended after the base table's last row (tmem_dealloc)
    assert len(ALIASES) == 3 and [a[0] for a in ALIASES] == ["dk_store", "dv_store", "ds_recv(rank)"]  # the dq_staging alias is deleted
    assert (c.PAIR_EPILOGUE_LANES, c.PAIR_SOFTMAX_LANES, c.DQ_SLICE_LANES, c.STORE_LANES) == (256, 512, 64, 32)
    # --- the alternate role shape (warp_roles 16w_fa4): typed, and its 33-row table equals the audited 35-row table minus the two deleted rows
    # and the unserved fp32 row, with ds_smem counted from the softmax, re-packed, plus the dq_war_done row (config = truth) ---
    assert c.WARP_ROLES == ("16w", "16w_fa4") and c.WARP_ROLES_DEFAULT == "16w" == c.WARP_ROLES[0] and tuple(WARP_ROLES_CODES) == c.WARP_ROLES
    assert WARP_ROLES_CODES == {"16w": 0, "16w_fa4": 1} and DELETED_BARRIERS_XFA4 == ("ds_tmem", "dp_empty") and ADDED_BARRIERS_XFA4 == ("dq_war_done",)
    assert resolve_warp_roles(None) == "16w" == resolve_warp_roles("16w") and resolve_warp_roles("16w_fa4") == "16w_fa4"
    assert (warp_roles_code(None), warp_roles_code("16w"), warp_roles_code("16w_fa4")) == (0, 0, 1)
    for bad_roles in ("bogus", "", 1, "16W", "20w"):
        try:
            resolve_warp_roles(bad_roles)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_roles)
    assert validate_fields_for_roles("16w", "dq_reduce_add_off") == "16w" and validate_fields_for_roles("16w_fa4", "dq_reduce_add_off") == "16w_fa4"
    assert validate_fields_for_roles("16w_fa4", None) == "16w_fa4" and validate_fields_for_roles("16w_fa4", "exchange_off") == "16w_fa4"
    # debug_clk and dkv_accum = fp32_reduce decline BY NAME under 16w_fa4 and pass under 16w; the unserved row is named
    assert validate_fields_for_roles("16w_fa4", None, debug_clk=False, dkv_accum="bf16_partials") == "16w_fa4"
    assert validate_fields_for_roles("16w", None, debug_clk=True, dkv_accum="fp32_reduce") == "16w"
    for bad_kw in ({"debug_clk": True}, {"dkv_accum": "fp32_reduce"}):
        try:
            validate_fields_for_roles("16w_fa4", None, **bad_kw)
        except ValueError as error:
            assert "is not defined under warp_roles '16w_fa4'" in str(error) or "is not built under warp_roles '16w_fa4'" in str(error), str(error)
        else:
            raise AssertionError(bad_kw)
    assert UNSERVED_BARRIERS_XFA4 == ("dvk_stage_empty",) and all(n in {row[0] for row in BAR_LAYOUT} for n in UNSERVED_BARRIERS_XFA4)
    assert not any(row[0] in UNSERVED_BARRIERS_XFA4 for row in BAR_LAYOUT_XFA4)
    for bad_roles, bad_diag in (("20w", None), ("16w_fa4", "ds_gates_off"), ("16w", "dp_empty_wait_off")):
        try:
            validate_fields_for_roles(bad_roles, bad_diag)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_roles, bad_diag))
    # the role-derived lane counts (never a literal in the body): the XCHG named barrier and ds_smem follow the warps that write XCHG / sDS own
    assert c.xchg_barrier_lanes("16w") == c.ds_smem_lanes("16w") == c.EPILOGUE_LANES == 128
    assert c.xchg_barrier_lanes("16w_fa4") == c.ds_smem_lanes("16w_fa4") == c.SOFTMAX_LANES == 256
    assert (c.SOFTMAX_LEAD_WARP, c.REDUCE_LEAD_WARP, c.REDUCE_WARPS, c.REDUCE_LANES, c.PAIR_REDUCE_LANES) == (0, 8, 4, 128, 256)
    assert 0 <= c.SOFTMAX_LEAD_WARP < c.SOFTMAX_WARPS and c.REDUCE_LEAD_WARP == c.SOFTMAX_WARPS and c.PAIR_REDUCE_LANES == c.PAIR_EPILOGUE_LANES
    # the softmax's dS STS image (the resource plan / 2): 4 vecs per 32-q quarter, 2 quarters per group = 8 STS.128 per lane per tile; the two
    # groups' 4-vec slots (vec 4g .. 4g+3) tile the 128-B SW128 row; 128 lanes x 2 groups x 8 x 16 B = the 32 KiB of sDS own + XCHG per CTA per tile
    assert (c.DS_STS_VECS_PER_QUARTER, c.DS_STS_PER_LANE_PER_TILE) == (4, 8) and c.DS_STS_VECS_PER_QUARTER * c.SOFTMAX_GROUPS * 16 == c.SWIZZLE_BYTES
    assert c.M * c.SOFTMAX_GROUPS * c.DS_STS_PER_LANE_PER_TILE * 16 == c.DS_HALF_BYTES + c.XCHG_BYTES == 32768
    assert c.DS_STS_VECS_PER_QUARTER * 4 == c.PACK_QUARTER_COLS == c.GROUP_QUARTER // 2  # 16 packed words = 32 q per quarter
    # BAR_LAYOUT_XFA4: pasted == derived (filter the two deleted rows, ds_smem from the softmax, re-pack at 16 B per row from BARS, then the section-15
    # row dq_war_done appended: MMA_COMMIT, LOCAL, ONE_LANE)
    derived, off = [], c.BARS
    for name, _offset, slots_n, count, kind, scope in BAR_LAYOUT:
        if name in DELETED_BARRIERS_XFA4 or name in UNSERVED_BARRIERS_XFA4:  # deleted by the role shape / unserved (fp32_reduce declines)
            continue
        derived.append((name, off, slots_n, c.ds_smem_lanes("16w_fa4") if name == "ds_smem" else count, kind, scope))
        off += 16
    derived.append(("dq_war_done", off, 1, c.ONE_LANE, "mma", "LOCAL"))
    off += 16
    assert (
        tuple(derived) == BAR_LAYOUT_XFA4
    ), "BAR_LAYOUT_XFA4 must be the audited table minus ds_tmem / dp_empty, ds_smem from the softmax, re-packed, + dq_war_done"
    assert BAR_LAYOUT_XFA4_STATUS.startswith("FILLED") and len(BAR_LAYOUT_XFA4) == 33 and sum(row[2] for row in BAR_LAYOUT_XFA4) == 38
    assert off == BAR_LAYOUT_XFA4_END == c.BARS + 16 * len(BAR_LAYOUT_XFA4) == 230928 <= c.TMEM_HANDLE
    assert BAR_LAYOUT_XFA4[-1][0] == "dq_war_done" and BAR_LAYOUT_XFA4[-1][3] == c.ONE_LANE and BAR_LAYOUT_XFA4[-1][4] == "mma"
    kinds_xfa4, addresses_xfa4 = {}, set()
    for name, offset, slots_n, count, kind, scope in BAR_LAYOUT_XFA4:
        assert scope == kind_scope[kind] and (count == c.ONE_LANE if kind in ("tma_leader", "tma", "tma_dsmem", "mma") else True), name
        kinds_xfa4[kind] = kinds_xfa4.get(kind, 0) + 1
        for i in range(slots_n):
            address = offset + i * 8
            assert c.BARS <= address and address + 8 <= c.TMEM_HANDLE and address not in addresses_xfa4
            addresses_xfa4.add(address)
    assert kinds_xfa4 == BAR_KIND_COUNTS_XFA4 == {"tma_leader": 7, "mma": 10, "thread": 7, "leader": 5, "peer": 2, "tma_dsmem": 1, "tma": 1}
    counts_xfa4 = {row[0]: row[3] for row in BAR_LAYOUT_XFA4}
    assert counts_xfa4["ds_smem"] == c.SOFTMAX_LANES == 256 and "ds_tmem" not in counts_xfa4 and "dp_empty" not in counts_xfa4
    assert all(counts_xfa4[n] == expected_counts[n] for n in expected_counts if n in counts_xfa4 and n != "ds_smem")
    assert [n for n in names if n not in DELETED_BARRIERS_XFA4 and n not in UNSERVED_BARRIERS_XFA4] + list(ADDED_BARRIERS_XFA4) == [
        row[0] for row in BAR_LAYOUT_XFA4
    ]  # 32 kept in order + the new row
    assert (
        bar_layout_for_roles("16w", "ldg") is BAR_LAYOUT
        and bar_layout_for_roles("16w_fa4") is BAR_LAYOUT_XFA4
        and bar_layout_for_roles("16w_fa4", "ldg") is BAR_LAYOUT_XFA4
    )
    assert (bar_layout_end_for_roles("16w", "ldg"), bar_layout_end_for_roles("16w_fa4")) == (BAR_LAYOUT_END, BAR_LAYOUT_XFA4_END)
    assert (bar_kind_counts_for_roles("16w", "ldg"), bar_kind_counts_for_roles("16w_fa4")) == (BAR_KIND_COUNTS, BAR_KIND_COUNTS_XFA4)
    # --- the TMEM MAP (arm R1, PART-KEYED): the code space, the parts, the defaults, the resolver's declines, the four records, the filtered tables,
    # the STAMPS records per map, the shape / probe rules ---
    assert set(TMEM_MAP_DEFAULT_BY_PART) == set(CTA_PAIR_CCS) == set(TMEM_MAP_PARTS["s_tail"]) and TMEM_MAP_PARTS["dq_576"] == ((10, 7),)
    assert TMEM_MAP_DEFAULT_BY_PART == {(10, 0): "s_tail", (10, 3): "s_tail", (10, 7): "s_tail"} and set(TMEM_MAP_PARTS) == set(c.TMEM_MAPS)
    assert all(m == c.TMEM_MAP for m in TMEM_MAP_DEFAULT_BY_PART.values())  # dq_576 is selectable by name only (a neutral structural map: the comment)
    assert all(cc in TMEM_MAP_PARTS[m] for cc, m in TMEM_MAP_DEFAULT_BY_PART.items())  # a part's default is a map it serves
    assert (
        resolve_tmem_map(None, (10, 0))
        == "s_tail"
        == resolve_tmem_map(None, (10, 3))
        == resolve_tmem_map("s_tail", (10, 7))
        == resolve_tmem_map(None, (10, 7), "16w_fa4")
    )
    assert resolve_tmem_map(None, (10, 7)) == "s_tail" == resolve_tmem_map(None, [10, 7]) == resolve_tmem_map(None, (10, 7), None)  # the part's default
    assert resolve_tmem_map("dq_576", (10, 7)) == "dq_576" == resolve_tmem_map("dq_576", (10, 7), "16w")  # selectable by name on the 576-column part
    assert (tmem_map_code(None, (10, 0)), tmem_map_code(None, (10, 7)), tmem_map_code("s_tail", (10, 7)), tmem_map_code("dq_576", (10, 7))) == (0, 0, 0, 1)
    for bad_name, bad_cc, bad_roles in (("bogus", (10, 7), None), ("", (10, 7), None), (1, (10, 7), None), ("S_TAIL", (10, 0), None), (None, (10, 7), "20w")):
        try:
            resolve_tmem_map(bad_name, bad_cc, bad_roles)
        except TmemMapNotServed:
            raise AssertionError((bad_name, bad_cc, bad_roles, "a name outside the domain is a plain ValueError, never a decline"))
        except ValueError:
            pass
        else:
            raise AssertionError((bad_name, bad_cc, bad_roles))
    for bad_cc in (None, (9, 0), (12, 0), (10, 9), (10,)):
        try:
            resolve_tmem_map(None, bad_cc)
        except TmemMapNotServed:
            raise AssertionError((bad_cc, "an unadmitted part has no default: a plain ValueError"))
        except ValueError:
            pass
        else:
            raise AssertionError(bad_cc)
    for bad_cc in ((10, 0), (10, 3)):  # the 576-column carve does not exist there: the typed decline (the prepared module's NotImplementedError)
        try:
            resolve_tmem_map("dq_576", bad_cc)
        except TmemMapNotServed as error:
            assert "tmem_map 'dq_576' is not served on cc" in str(error) and "576-column" in str(error), str(error)
        else:
            raise AssertionError(bad_cc)
    assert cfg_for_part((10, 0)) is CFG_RING is cfg_for_part((10, 3)) is cfg_for_part((10, 7)) is cfg_for_part((10, 7), tmem_map="s_tail")
    assert cfg_for_part((10, 0), "ldg") is CFG is cfg_for_part((10, 7), "ldg")
    # cc 10.7: the dq_576 ring record when the TMEM map is NAMED (the SMEM map unset or NAMED "sm100": the SMEM-map block below)
    assert cfg_for_part((10, 7), tmem_map="dq_576") is CFG_RING_DQ576 is cfg_for_part((10, 7), "tma_ring", "16w", "dq_576")
    assert cfg_for_part((10, 7), tmem_map="dq_576", smem_map="sm100") is CFG_RING_DQ576 and cfg_for_part((10, 7), "ldg", tmem_map="dq_576") is CFG_DQ576
    assert cfg_for_part((10, 7), roles="16w_fa4") is CFG and cfg_for_part((10, 7), roles="16w_fa4", stats_form="tma_ring") is CFG_RING  # the shape's unset map
    assert cfg_for_geometry("tma_ring", "s_tail") is CFG_RING and {(f, m) for f, m, _s in GEOMETRIES} == {(f, m) for f in c.STATS_FORMS for m in c.TMEM_MAPS}
    assert all(cfg_for_geometry(f, m, s).TMEM_MAP == m and cfg_for_geometry(f, m, s).STATS_STAGES == (2 if f == "tma_ring" else 1) for f, m, s in GEOMETRIES)
    try:
        cfg_for_geometry("tma_ring", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown map has no geometry")
    # the shape / probe rules under the map
    assert validate_fields_for_roles("16w", None, tmem_map="dq_576") == "16w" and validate_fields_for_roles("16w", None, tmem_map="s_tail") == "16w"
    assert validate_fields_for_roles("16w", "exchange_off", debug_clk=True, dkv_accum="fp32_reduce", stats_form="tma_ring", tmem_map="dq_576") == "16w"
    assert all(validate_fields_for_roles("16w", p, tmem_map="dq_576") == "16w" for p in c.DIAG_2CTA_NAMES_ON_MAP if p not in c.DIAG_2CTA_S_TAIL_PROBES)
    for probe in c.DIAG_2CTA_S_TAIL_PROBES:
        assert validate_fields_for_roles("16w", probe, tmem_map="s_tail") == "16w" and validate_fields_for_roles("16w", probe) == "16w"
        try:
            validate_fields_for_roles("16w", probe, tmem_map="dq_576")
        except ValueError as error:
            assert f"diag_2cta {probe!r} is not defined under tmem_map 'dq_576'" in str(error), str(error)
        else:
            raise AssertionError(probe)
    try:
        validate_fields_for_roles("16w_fa4", None, tmem_map="dq_576")
    except ValueError as error:
        assert "tmem_map 'dq_576' is not built under warp_roles '16w_fa4'" in str(error), str(error)
    else:
        raise AssertionError("dq_576 must decline by name under 16w_fa4")
    assert validate_fields_for_roles("16w_fa4", None, tmem_map="s_tail") == "16w_fa4"
    for bad_map in ("bogus", "", 1):
        try:
            validate_fields_for_roles("16w", None, tmem_map=bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_map)
    # the filtered tables: the audited rows minus the two, re-packed from the geometry's BARS; every kept row identical in (slots, count, kind, scope)
    assert DELETED_BARRIERS_DQ576 == ("s_read", "s_tail_empty") and all(n in {row[0] for row in BAR_LAYOUT} for n in DELETED_BARRIERS_DQ576)
    assert len(BAR_LAYOUT_DQ576) == 33 and sum(row[2] for row in BAR_LAYOUT_DQ576) == 38 and BAR_LAYOUT_DQ576_END == c.BARS + 16 * 33 == 230928
    assert len(BAR_LAYOUT_RING_DQ576) == 35 and sum(row[2] for row in BAR_LAYOUT_RING_DQ576) == 42 and BAR_LAYOUT_RING_DQ576_END == r.BARS + 16 * 35 == 231984
    assert [row[0] for row in BAR_LAYOUT_DQ576] == [n for n in names if n not in DELETED_BARRIERS_DQ576]
    assert [row[0] for row in BAR_LAYOUT_RING_DQ576] == [row[0] for row in BAR_LAYOUT_DQ576] + [row[0] for row in STATS_RING_ROWS]
    audited = {row[0]: row[2:] for row in BAR_LAYOUT_RING}  # the s_tail ring table carries every row the two filtered tables keep
    for layout, g in ((BAR_LAYOUT_DQ576, CFG_DQ576), (BAR_LAYOUT_RING_DQ576, CFG_RING_DQ576)):
        prev_end, seen = g.BARS, set()
        for name, offset, slots_n, count, kind, scope in layout:
            assert (slots_n, count, kind, scope) == audited[name] and name not in DELETED_BARRIERS_DQ576, name
            assert offset == prev_end and offset % 16 == 0, name
            prev_end = offset + 16
            for i in range(slots_n):
                address = offset + i * 8
                assert g.BARS <= address and address + 8 <= g.TMEM_HANDLE and address not in seen, name
                seen.add(address)
        assert prev_end == layout[-1][1] + 16 <= g.TMEM_HANDLE
    assert BAR_LAYOUT_DQ576_STATUS.startswith("SKELETON")
    assert bar_layout_for_roles("16w", "tma_ring", "dq_576") is BAR_LAYOUT_RING_DQ576 is bar_layout_for_roles("16w", None, "dq_576")
    assert bar_layout_for_roles("16w", "ldg", "dq_576") is BAR_LAYOUT_DQ576 and bar_layout_for_roles("16w", "tma_ring", "s_tail") is BAR_LAYOUT_RING
    assert bar_layout_for_roles("16w", "tma_ring", None) is BAR_LAYOUT_RING and bar_layout_for_roles("16w_fa4", None, "s_tail") is BAR_LAYOUT_XFA4
    assert (bar_layout_end_for_roles("16w", "tma_ring", "dq_576"), bar_layout_end_for_roles("16w", "ldg", "dq_576")) == (231984, 230928)
    counts_dq576 = bar_kind_counts_for_roles("16w", "tma_ring", "dq_576")
    assert counts_dq576 == {"tma_leader": 7, "mma": 9, "thread": 10, "leader": 4, "peer": 2, "tma_dsmem": 1, "tma": 2} and sum(counts_dq576.values()) == 35
    assert (
        bar_kind_counts_for_roles("16w", "ldg", "dq_576") == dict(BAR_KIND_COUNTS, leader=4)
        and bar_kind_counts_for_roles("16w", "tma_ring", "s_tail") == BAR_KIND_COUNTS_RING
    )
    for bad_call in (("16w_fa4", "ldg", "dq_576"), ("16w_fa4", None, "dq_576"), ("16w", None, "bogus")):
        try:
            bar_layout_for_roles(*bad_call)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_call)
    # the STAMPS records per map: the same word map, the spare re-derived; the four S-tail words dead under dq_576
    dd, drd = DBG_CLK_DQ576, DBG_CLK_RING_DQ576
    assert (dd.SMEM, dd.WORDS, dd.BYTES) == (230928, 104, 416) and dd.SMEM + dd.BYTES == CFG_DQ576.TMEM_HANDLE == c.TMEM_HANDLE and dd.SMEM % 16 == 0
    assert (drd.SMEM, drd.WORDS, drd.BYTES) == (231984, 96, 384) and drd.SMEM + drd.BYTES == CFG_RING_DQ576.TMEM_HANDLE == r.TMEM_HANDLE and drd.SMEM % 16 == 0
    assert dd.WORD == drd.WORD == DBG_CLK.WORD and len(dd.WORD) == 86 <= drd.WORDS <= dd.WORDS and dd.MAGIC == drd.MAGIC == DBG_CLK.MAGIC
    dead_b2 = ("wait:leader:dq_read_done", "blocked:leader:dq_read_done")  # the row has a site under rubin_b2 only
    assert dbg_clk_dead_words("dq_576") == ("wait:leader:s_read", "blocked:leader:s_read", "wait:leader:s_tail_empty", "blocked:leader:s_tail_empty") + dead_b2
    assert set(dbg_clk_dead_words("dq_576")) <= set(dd.WORD) and dbg_clk_dead_words("s_tail") == dead_b2 == dbg_clk_dead_words("s_tail", "sm100")
    assert dbg_clk_dead_words("dq_576", "rubin_b2") == dbg_clk_dead_words("dq_576")[:4] and dbg_clk_dead_words("dq_576", "sm100") == dbg_clk_dead_words(
        "dq_576"
    )
    assert (
        dbg_clk_for_stats_form("tma_ring", None, "dq_576") is drd is dbg_clk_for_stats_form(None, None, "dq_576")
        and dbg_clk_for_stats_form("ldg", None, "dq_576") is dd
    )
    assert (
        dbg_clk_for_stats_form(None, None, "s_tail") is DBG_CLK_RING is dbg_clk_for_stats_form(None)
        and dbg_clk_for_stats_form("ldg", "16w_fa4", "s_tail") is DBG_CLK
    )
    assert set(DBG_CLK_BY_GEOMETRY) == set(GEOMETRIES) and all(
        DBG_CLK_BY_GEOMETRY[k].SMEM + DBG_CLK_BY_GEOMETRY[k].BYTES == GEOMETRIES[k].TMEM_HANDLE for k in GEOMETRIES
    )
    assert (
        dbg_clk_bytes(1, 1, 256, "tma_ring", "dq_576") == 2 * 384
        and dbg_clk_bytes(1, 1, 256, "ldg", "dq_576") == 2 * 416
        and dbg_clk_bytes(1, 1, 256) == 2 * 352
    )
    for bad_map in ("bogus", None):
        try:
            dbg_clk_dead_words(bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_map)
    # --- the SMEM MAP (arm R2.c, PART-KEYED; kernel header sections 5 / 8 / 11): the code space, the parts, the defaults, the resolver's declines, the
    # B2 record's slab chain (every root below the version-0 line; the slab under the oversized cap), its barrier table (derived; names / slots / counts
    # / kinds / addresses; SUM(issuing lanes) == init per row is the kernel header's ledger), its STAMPS record, the cross-field rules ---
    b2 = CFG_RING_DQ576_B2
    assert c.SMEM_MAPS == ("sm100", "rubin_b2", "sm100_oversized") == tuple(SMEM_MAP_CODES) and c.SMEM_MAP == c.SMEM_MAPS[0]
    assert SMEM_MAP_CODES == {"sm100": 0, "rubin_b2": 1, "sm100_oversized": 2}
    assert (b2.SMEM_MAP, b2.TMEM_MAP, b2.STATS_STAGES, b2.Q1_STAGES, b2.DO1_STAGES, b2.DS_STAGES, b2.XCHG_STAGES) == ("rubin_b2", "dq_576", 2, 2, 2, 2, 2)
    assert b2.DS_STAGES == b2.XCHG_STAGES and b2.DO_STAGES == 1 and b2.DO1_STAGES == 2  # the base chain's DO_STAGES keeps sDO2 at one stage
    order_b2 = (b2.Q1, b2.Q2, b2.DO1, b2.DO2, b2.K1, b2.K2, b2.V, b2.DS, b2.XCHG, b2.DQ, b2.STATS, b2.SLABS_END)
    assert order_b2 == (0, 32768, 49152, 81920, 98304, 131072, 163840, 196608, 262144, 294912, 311296, 313344), order_b2
    assert all(off % 1024 == 0 for off in order_b2)
    assert (b2.SMEM_BYTES, b2.BARS, b2.TMEM_HANDLE, b2.SCHED_PAYLOAD, b2.BAR_BLOCK_BYTES, b2.SMEM_CAP) == (314368, 313344, 314288, 314304, 944, 334848)
    assert c.SMEM_OPT_IN_CAP < b2.SMEM_BYTES <= b2.SMEM_CAP == c.SMEM_OVERSIZED_CAP  # the oversized mode is what the map buys; it fits the carveout
    assert b2.SMEM_BYTES - r.SMEM_BYTES == c.Q1_BYTES + c.DO1_BYTES + c.DS_BYTES + c.XCHG_BYTES == 81920  # +16 +16 +32 +16 KiB: the four second stages
    roots_b2 = b2.desc_roots()
    assert roots_b2 == (0, 16384, 32768, 49152, 65536, 81920, 98304, 131072, 147456, 163840, 196608, 229376), roots_b2
    assert max(roots_b2) < b2.TCGEN05_V0_ADDR_LIMIT and all(rt % 1024 == 0 for rt in roots_b2) and b2.DESC_VERSION == c.DESC_VERSION == 1
    assert (
        b2.ds_stage(1) + b2.DS_BYTES == b2.TCGEN05_V0_ADDR_LIMIT == b2.XCHG
    )  # the last root ends exactly at the line; XCHG / sDQ / the stats ring are no roots
    assert [b2.q1_stage(s) for s in range(2)] == [0, 16384] and [b2.do1_stage(s) for s in range(2)] == [49152, 65536]
    assert [b2.ds_stage(s) for s in range(2)] == [196608, 229376] and [b2.xchg_stage(s) for s in range(2)] == [262144, 278528]
    assert [b2.dq_slot(s) for s in range(b2.DQ_SLOTS)] == [294912, 303104] and [b2.stats_stage(s) for s in range(2)] == [311296, 312320]
    assert (
        b2.stats_stage(1) + b2.STATS_STAGE_BYTES == b2.SLABS_END == b2.BARS
        and (b2.ALIAS, b2.ALIAS_BYTES) == (b2.DS, 98304)
        and b2.ALIAS + b2.ALIAS_BYTES == b2.DQ
    )
    for rank in (0, 1):
        assert {b2.dk_store(rank, 0), b2.dk_store(rank, 1)} == {b2.ds_own_stage(rank, 0), b2.xchg_stage(0)}  # the dK staging over the STAGE-0 bytes
        assert b2.ds_own(rank) == b2.ds_own_stage(rank, 0) and b2.ds_recv(rank) == b2.ds_recv_stage(rank, 0)
        for s in range(b2.DS_STAGES):
            assert {b2.ds_own_stage(rank, s), b2.ds_recv_stage(rank, s)} == {b2.ds_stage(s), b2.ds_stage(s) + b2.DS_HALF_BYTES}
            assert b2.ds_own_stage(rank, s) == b2.ds_recv_stage(1 - rank, s)  # our own half IS the peer's recv half of the same stage (kv-global order)
    assert [b2.dkv_acc_r1_slot(i) for i in range(5)] == [b2.K2, b2.K2 + 16384, b2.DS, b2.DS + 16384, b2.XCHG]  # stage-0 bytes (dead from dq_done(hi-1))
    assert (b2.dv_store(0), b2.dv_store(1)) == (b2.K2, b2.K2 + 16384) and all(
        s >= b2.XCHG + b2.XCHG_STAGES * b2.XCHG_BYTES for s in [b2.dq_slot(0), b2.dq_slot(1)]
    )
    moved_smem = {  # the SMEM map's business and nothing else (the TMEM map's five, the lane counts, the boxes, the probes: equal)
        "SMEM_MAP",
        "Q1_STAGES",
        "DO1_STAGES",
        "DS_STAGES",
        "XCHG_STAGES",
        "Q2",
        "DO1",
        "DO2",
        "K1",
        "K2",
        "V",
        "DS",
        "XCHG",
        "ALIAS",
        "ALIAS_BYTES",
        "DQ",
        "STATS",
        "DELTA",
        "STATS_RING",
        "SLABS_END",
        "SMEM_BYTES",
        "BARS",
        "TMEM_HANDLE",
        "SCHED_PAYLOAD",
        "SMEM_CAP",
    }
    assert moved_smem <= names_all, moved_smem - names_all
    for n in props:
        if n not in moved_smem:
            assert getattr(CFG_RING_DQ576, n) == getattr(b2, n), n
    for n in type(c).__dataclass_fields__:
        if n not in moved_smem:
            assert getattr(CFG_RING_DQ576, n) == getattr(b2, n), n
    # the B2 barrier table (derived) and the derivation's fidelity to every pasted table
    assert layout_for_geometry(c) == BAR_LAYOUT and layout_for_geometry(r) == BAR_LAYOUT_RING and layout_for_geometry(c, "16w_fa4") == BAR_LAYOUT_XFA4
    assert layout_for_geometry(CFG_DQ576) == BAR_LAYOUT_DQ576 and layout_for_geometry(CFG_RING_DQ576) == BAR_LAYOUT_RING_DQ576
    assert layout_for_geometry(r, "16w_fa4") == fa4_ring and [_row_bytes(s) for s in (1, 2, 3, 4)] == [16, 16, 32, 32]
    t_b2 = BAR_LAYOUT_RING_DQ576_B2
    names_b2 = [row[0] for row in t_b2]
    kept = [n for n in names if n not in DELETED_BARRIERS_DQ576]
    assert names_b2 == kept[: kept.index(DQ_READ_DONE_AFTER) + 1] + list(ADDED_BARRIERS_B2) + kept[kept.index(DQ_READ_DONE_AFTER) + 1 :] + [
        row[0] for row in STATS_RING_ROWS
    ]
    assert len(t_b2) == 36 and sum(row[2] for row in t_b2) == 51 and BAR_LAYOUT_RING_DQ576_B2_END == 313936 == b2.BARS + 35 * 16 + 32
    assert BAR_LAYOUT_RING_DQ576_B2_END == layout_end(t_b2) and layout_end(BAR_LAYOUT_RING) == BAR_LAYOUT_RING_END and layout_end(BAR_LAYOUT) == BAR_LAYOUT_END
    slots_b2 = {row[0]: row[2] for row in t_b2}
    grown = dict(q1_full=2, q1_empty=2, do1_full=4, do1_empty=2, ds_smem=2, ds_recv_full=2, ds_pair_ready=2)
    assert slots_b2 == dict({row[0]: row[2] for row in BAR_LAYOUT_RING_DQ576}, dq_read_done=1, **grown), slots_b2
    assert (
        set(RING_SLOT_ROWS) == set(grown)
        and DQ_READ_DONE_ROW == ("dq_read_done", 1, c.PAIR_EPILOGUE_LANES, "leader", "LEADER")
        and c.PAIR_EPILOGUE_LANES == 256
    )
    row_b2 = {row[0]: row for row in t_b2}
    for name, (slots_n, count, kind, scope) in audited.items():
        if name not in DELETED_BARRIERS_DQ576:
            assert row_b2[name][3:] == (count, kind, scope), name  # every kept row keeps its init per slot, kind and scope
    assert row_b2["dq_read_done"][3:] == (c.PAIR_EPILOGUE_LANES, "leader", "LEADER") and names_b2.index("dq_read_done") == names_b2.index("dp_empty") + 1
    prev_end, seen_b2, kinds_b2 = b2.BARS, set(), {}
    for name, offset, slots_n, count, kind, scope in t_b2:
        assert offset == prev_end and offset % 16 == 0 and scope == kind_scope[kind], name
        prev_end = offset + _row_bytes(slots_n)
        kinds_b2[kind] = kinds_b2.get(kind, 0) + 1
        for i in range(slots_n):
            address = offset + i * 8
            assert b2.BARS <= address and address + 8 <= b2.TMEM_HANDLE and address not in seen_b2, name
            seen_b2.add(address)
    assert prev_end == BAR_LAYOUT_RING_DQ576_B2_END <= b2.TMEM_HANDLE and kinds_b2 == BAR_KIND_COUNTS_RING_DQ576_B2 and sum(kinds_b2.values()) == 36
    assert row_b2["do1_full"][1] + 32 == row_b2["do1_empty"][1]  # the 4-slot row takes 32 B: the cumulative walk
    assert BAR_LAYOUT_RING_DQ576_B2_STATUS.startswith("SKELETON")
    assert bar_layout_for_roles("16w", "tma_ring", "dq_576", "rubin_b2") is BAR_LAYOUT_RING_DQ576_B2 is bar_layout_for_roles("16w", None, "dq_576", "rubin_b2")
    assert bar_layout_for_roles("16w", "tma_ring", "dq_576", "sm100") is BAR_LAYOUT_RING_DQ576 is bar_layout_for_roles("16w", "tma_ring", "dq_576", None)
    assert (
        bar_layout_end_for_roles("16w", "tma_ring", "dq_576", "rubin_b2") == 313936
        and bar_kind_counts_for_roles("16w", "tma_ring", "dq_576", "rubin_b2") == kinds_b2
    )
    for bad_call in (
        ("16w", "tma_ring", "s_tail", "rubin_b2"),
        ("16w_fa4", "ldg", "s_tail", "rubin_b2"),
        ("16w", "ldg", "dq_576", "rubin_b2"),
        ("16w", None, "dq_576", "bogus"),
    ):
        try:
            bar_layout_for_roles(*bad_call)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_call)
    # the resolvers, the defaults and the declines
    assert set(SMEM_MAP_DEFAULT_BY_PART) == set(CTA_PAIR_CCS) == set(SMEM_MAP_PARTS["sm100"]) and SMEM_MAP_PARTS["rubin_b2"] == ((10, 7),)
    assert SMEM_MAP_PARTS["sm100_oversized"] == ((10, 7),) and SMEM_MAP_SLAB_BYTES == {"rubin_b2": 314368, "sm100_oversized": 262144}
    # --- the DIAGNOSTIC map sm100_oversized: the sm100 map itself (every address identical) with the padded declaration; never a default; cc 10.7 only ---
    for ovs, ref, rec in ((CFG_RING_OVS, CFG_RING, DBG_CLK_RING), (CFG_RING_DQ576_OVS, CFG_RING_DQ576, DBG_CLK_RING_DQ576)):
        assert ovs.SMEM_MAP == "sm100_oversized" and ref.SMEM_MAP == "sm100" and ovs.SMEM_PAD_BYTES == 29696 and ref.SMEM_PAD_BYTES == 0
        assert ovs.SMEM_BYTES == SMEM_OVERSIZED_PROBE_BYTES == 262144 and c.SMEM_OPT_IN_CAP < ovs.SMEM_BYTES <= ovs.SMEM_CAP == c.SMEM_OVERSIZED_CAP
        assert (ovs.BARS, ovs.TMEM_HANDLE, ovs.SCHED_PAYLOAD, ovs.SLABS_END) == (ref.BARS, ref.TMEM_HANDLE, ref.SCHED_PAYLOAD, ref.SLABS_END)
        assert ovs.SCHED_PAYLOAD + c.SCHED_PAYLOAD_BYTES == ref.SMEM_BYTES == 232448  # the pad is dead space ABOVE every address of the map
        assert ovs.desc_roots() == ref.desc_roots() and ovs.TMEM_MAP == ref.TMEM_MAP and ovs.STATS_STAGES == ref.STATS_STAGES == 2
        assert (ovs.Q1_STAGES, ovs.DO1_STAGES, ovs.DS_STAGES, ovs.XCHG_STAGES) == (ref.Q1_STAGES, ref.DO1_STAGES, ref.DS_STAGES, ref.XCHG_STAGES)
        assert (ovs.Q1, ovs.Q2, ovs.DO1, ovs.DO2, ovs.K1, ovs.K2, ovs.V, ovs.DS, ovs.XCHG, ovs.DQ, ovs.STATS, ovs.STATS_RING) == (
            ref.Q1,
            ref.Q2,
            ref.DO1,
            ref.DO2,
            ref.K1,
            ref.K2,
            ref.V,
            ref.DS,
            ref.XCHG,
            ref.DQ,
            ref.STATS,
            ref.STATS_RING,
        )
        assert bar_layout_for_roles("16w", "tma_ring", ref.TMEM_MAP, "sm100_oversized") is bar_layout_for_roles("16w", "tma_ring", ref.TMEM_MAP, "sm100")
        assert bar_kind_counts_for_roles("16w", "tma_ring", ref.TMEM_MAP, "sm100_oversized") == bar_kind_counts_for_roles("16w", "tma_ring", ref.TMEM_MAP)
        assert (
            dbg_clk_for_stats_form("tma_ring", None, ref.TMEM_MAP, "sm100_oversized") is rec
            and cfg_for_geometry("tma_ring", ref.TMEM_MAP, "sm100_oversized") is ovs
        )
    assert set(SMEM_MAP_DIAGNOSTIC) == {"sm100_oversized"} and not any(m in SMEM_MAP_DIAGNOSTIC for m in SMEM_MAP_DEFAULT_BY_PART.values())
    assert resolve_smem_map("sm100_oversized", (10, 7)) == "sm100_oversized" and smem_map_code("sm100_oversized", (10, 7), "16w", "tma_ring", "s_tail") == 2
    for cc in ((10, 0), (10, 3)):
        try:
            resolve_smem_map("sm100_oversized", cc)
        except SmemMapNotServed:
            pass
        else:
            raise AssertionError("sm100_oversized must decline by name off cc 10.7")
    for bad in (dict(roles="16w_fa4", stats_form="ldg"), dict(roles="16w", stats_form="ldg")):
        try:
            validate_fields_for_roles(bad["roles"], None, stats_form=bad["stats_form"], tmem_map="s_tail", smem_map="sm100_oversized")
        except ValueError as error:
            assert "smem_map 'sm100_oversized' is not built under" in str(error), str(error)
        else:
            raise AssertionError("sm100_oversized x 16w_fa4 / x ldg must decline by name")
    assert SMEM_MAP_DEFAULT_BY_PART == {(10, 0): "sm100", (10, 3): "sm100", (10, 7): "sm100"} and set(SMEM_MAP_PARTS) == set(c.SMEM_MAPS)
    assert all(m == c.SMEM_MAP for m in SMEM_MAP_DEFAULT_BY_PART.values())  # rubin_b2 is selectable by name only (its A/B/A row flips the part default)
    assert all(cc in SMEM_MAP_PARTS[m] for cc, m in SMEM_MAP_DEFAULT_BY_PART.items())
    assert resolve_smem_map("rubin_b2", (10, 7)) == "rubin_b2" == resolve_smem_map("rubin_b2", [10, 7], "16w", "tma_ring", "dq_576")  # NAMED on cc 10.7
    assert (
        resolve_smem_map(None, (10, 0))
        == "sm100"
        == resolve_smem_map(None, (10, 3))
        == resolve_smem_map(None, (10, 7))
        == resolve_smem_map("sm100", (10, 7))
        == resolve_smem_map(None, (10, 7), tmem_map="dq_576")
        == resolve_smem_map(None, (10, 7), "16w_fa4")
        == resolve_smem_map(None, (10, 7), stats_form="ldg")
    )
    assert (smem_map_code(None, (10, 0)), smem_map_code(None, (10, 7)), smem_map_code("sm100", (10, 7)), smem_map_code("rubin_b2", (10, 7))) == (0, 0, 0, 1)
    for bad_name in ("bogus", "", 1, "RUBIN_B2", "b2"):
        try:
            resolve_smem_map(bad_name, (10, 7))
        except SmemMapNotServed:
            raise AssertionError((bad_name, "a name outside the domain is a plain ValueError, never a decline"))
        except ValueError:
            pass
        else:
            raise AssertionError(bad_name)
    for bad_cc in (None, (9, 0), (12, 0), (10, 9)):
        try:
            resolve_smem_map(None, bad_cc)
        except SmemMapNotServed:
            raise AssertionError((bad_cc, "an unadmitted part has no default: a plain ValueError"))
        except ValueError:
            pass
        else:
            raise AssertionError(bad_cc)
    for bad_cc in ((10, 0), (10, 3)):
        try:
            resolve_smem_map("rubin_b2", bad_cc)
        except SmemMapNotServed as error:
            assert "smem_map 'rubin_b2' is not served on cc" in str(error) and "oversized" in str(error), str(error)
        else:
            raise AssertionError(bad_cc)
    assert cfg_for_part((10, 7), tmem_map="dq_576", smem_map="rubin_b2") is b2 is cfg_for_part((10, 7), "tma_ring", "16w", "dq_576", "rubin_b2")  # BOTH named
    assert cfg_for_part((10, 7), tmem_map="dq_576", smem_map="sm100") is CFG_RING_DQ576 is cfg_for_part((10, 7), tmem_map="dq_576")
    assert cfg_for_part((10, 7)) is CFG_RING is cfg_for_part((10, 7), tmem_map="s_tail") is cfg_for_part((10, 0)) is cfg_for_part((10, 3))  # the defaults
    assert cfg_for_part((10, 7), "ldg", tmem_map="dq_576") is CFG_DQ576 and cfg_for_part((10, 7), roles="16w_fa4") is CFG
    assert cfg_for_geometry("tma_ring", "dq_576") is CFG_RING_DQ576 and cfg_for_geometry("tma_ring", "dq_576", "rubin_b2") is b2
    for bad_part in (dict(smem_map="rubin_b2"), dict(tmem_map="s_tail", smem_map="rubin_b2"), dict(stats_form="ldg", tmem_map="dq_576", smem_map="rubin_b2")):
        try:
            cfg_for_part((10, 7), **bad_part)  # no geometry record exists off the B2 seat (validate_fields_for_roles names the rule first in compile_host)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_part)
    assert set(GEOMETRIES) == {(f, m, "sm100") for f in c.STATS_FORMS for m in c.TMEM_MAPS} | {("tma_ring", "dq_576", "rubin_b2")} | {
        ("tma_ring", m, "sm100_oversized") for m in c.TMEM_MAPS
    }
    assert all(cfg_for_geometry(*k).SMEM_MAP == k[2] and cfg_for_geometry(*k).TMEM_MAP == k[1] for k in GEOMETRIES)
    for bad_geo in (("ldg", "dq_576", "rubin_b2"), ("tma_ring", "s_tail", "rubin_b2"), ("tma_ring", "dq_576", "bogus")):
        try:
            cfg_for_geometry(*bad_geo)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_geo)
    assert validate_fields_for_roles("16w", None, tmem_map="dq_576", smem_map="rubin_b2") == "16w" == validate_fields_for_roles("16w", None, smem_map="sm100")
    assert (
        validate_fields_for_roles("16w", "exchange_off", debug_clk=True, dkv_accum="fp32_reduce", stats_form="tma_ring", tmem_map="dq_576", smem_map="rubin_b2")
        == "16w"
    )
    for bad_kw, text in (
        (dict(tmem_map="s_tail", smem_map="rubin_b2"), "smem_map 'rubin_b2' is not built under tmem_map 's_tail'"),
        (dict(smem_map="rubin_b2"), "smem_map 'rubin_b2' is not built under tmem_map 's_tail'"),
        (dict(stats_form="ldg", tmem_map="dq_576", smem_map="rubin_b2"), "smem_map 'rubin_b2' is not built under stats_form 'ldg'"),
        (dict(smem_map="bogus"), "smem_map must be one of"),
    ):
        try:
            validate_fields_for_roles("16w", None, **bad_kw)
        except ValueError as error:
            assert text in str(error), (bad_kw, str(error))
        else:
            raise AssertionError(bad_kw)
    try:
        validate_fields_for_roles("16w_fa4", None, tmem_map="dq_576", smem_map="rubin_b2")
    except ValueError as error:
        assert "is not built under warp_roles '16w_fa4'" in str(error), str(error)
    else:
        raise AssertionError("rubin_b2 must decline by name under 16w_fa4")
    # the STAMPS record under the map: the same word map; the leader's dq_read_done pair live; the spare re-derived
    db2 = DBG_CLK_RING_DQ576_B2
    assert (db2.SMEM, db2.WORDS, db2.BYTES) == (313936, 88, 352) and db2.SMEM + db2.BYTES == b2.TMEM_HANDLE and db2.SMEM % 16 == 0
    assert db2.WORD == DBG_CLK.WORD and len(db2.WORD) == 86 <= db2.WORDS and db2.MAGIC == DBG_CLK.MAGIC
    assert dbg_clk_for_stats_form("tma_ring", None, "dq_576", "rubin_b2") is db2 is dbg_clk_for_stats_form(None, None, "dq_576", "rubin_b2")
    assert dbg_clk_for_stats_form(None, None, "dq_576") is drd is dbg_clk_for_stats_form(None, None, "dq_576", "sm100")
    assert set(DBG_CLK_BY_GEOMETRY) == set(GEOMETRIES) and all(
        DBG_CLK_BY_GEOMETRY[k].SMEM + DBG_CLK_BY_GEOMETRY[k].BYTES == GEOMETRIES[k].TMEM_HANDLE and DBG_CLK_BY_GEOMETRY[k].WORDS >= len(DBG_CLK.WORD)
        for k in GEOMETRIES
    )
    assert dbg_clk_bytes(1, 1, 256, "tma_ring", "dq_576", "rubin_b2") == 2 * 352 and dbg_clk_bytes(1, 1, 256, None, "dq_576") == 2 * 384
    for bad_map in ("bogus", "", 1):  # None = the sm100 map (legal)
        try:
            dbg_clk_for_stats_form(None, None, "dq_576", bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_map)


validate_config()
