# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped THD d128 accumulation geometry on the cta_group::2 datapath, ARM (ii): the TMEM re-plan with FAv4's rotated MMA order.

Layout-only record for ``kernels/sm100/bprop_d128_accum_grouped_2cta_stail_f16.py`` -- the SECOND pair body, kept beside the
first (``bprop_d128_accum_grouped_2cta_f16.py`` on ``config_sm100_d128_accum_2cta.CFG``, UNTOUCHED) for the one-process A/B.  The
two bodies are selected by the typed adapter field ``tmem_map``: ``"dp_tail"`` (the default = the first body) | ``"s_tail"``
(this one).  Every byte offset, TMEM column and ring depth derives from the geometry below and is re-derived by
:func:`validate_config` at import; the barrier table (``BAR_LAYOUT``) is the barrier-inspector's 34-row SKELETON for the arm
(resources/arm2-barrier-table.md, 2026-10-27 01:05 UTC), pasted, never re-derived.

What moves against the first body (resource-manager 2026-10-27 00:53 UTC tables 1-6 + the inspector's deviations D1-D5):

* TMEM: the dQ half-row image (64 fp32 columns, the M = 128 pair MMA's D image) moves from dP^T's dead half [192, 256) to
  S^T's dead tail [64, 128) (the 64 columns the packed P^T does not reuse): ``DQ_COL = S_COL + PACK_COLS``; 192..256 is DEAD.
  Two alias edges on the S tail: the WRITE edge dQ(t-1) <- the softmax's read of S^T(t) (NEW barrier ``s_read``) and the REUSE
  edge S^T(t+1) <- the epilogue's readout of dQ(t-1) (NEW barrier ``s_tail_empty``); ``dp_empty`` keeps its count and waiter, its
  arrive moves to the dS^T readout.
* ORDER: the leader issues FAv4's rotated stream per q tile: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); the tail issues dQ(hi-1).
  TWO orders on this map (``PAIR_MMA_ORDERS == ("fa4", "fa4b")``, default ``fa4`` until measured): ``fa4b`` issues dQ(t-1) AFTER dV(t)
  (code-writer step 6, 2026-10-27: +512 clk of cover for the dS exchange at zero SMEM, the same barrier table and sites; the kernel's
  ``dq_after_dv`` Constexpr via :func:`dq_after_dv_for_map`); s_first / dk_first are declined BY NAME here (their s_read / s_tail_empty
  site sets differ, inspector section 11).
* SMEM: Q1 2 -> 1 stage (FAv4's ``Q_stage = 1`` under 2-CTA; -16 KiB) funds a DEDICATED dQ fp32 staging ring ``sDQ`` of
  ``DQ_SLOTS`` = 2 x [64 q x 32 d] SW128 slices (today's reduce-add box), two rounds per tile; the dQ staging alias on
  ds_own + XCHG is DELETED (under the rotated order dq_done(t) lands ~150 clk before the dS(t+1) STS, so the alias would gate
  the dS exchange by the reduce-add drain).  Total unchanged: 231424 B (vs-smem predict PASS, banks PASS: resources/*-arm2.*).
* dQ ROUTE (code-writer step 8, 2026-10-27; optimizer 05:15 rank 1 "#3-dry"): the typed field ``dq_route_pair`` = ``"tma"`` (THE default: the
  sDQ ring STS + the store warp's four ``tma_reduce_add`` per tile) | ``"red_global"`` (the fp32 dQ partial leaves the CTA as
  ``red.global.add.v4.f32`` straight from the epilogue warps' registers right after the dQ tcgen05.ld; the sDQ ring PROTOCOL -- dq_full /
  dq_empty arrives and waits, the store warp's per-round loop -- stays, so NO barrier row moves; the STS and the UTMAREDG issue are skipped) |
  ``"red_global_pair"`` (the same reds with lane pairs ``2k / 2k+1`` exchanging one v4 per sector through ``shfl.sync.bfly`` so every
  warp-level red carries FULL 32-B L2 sectors, RULE-GAPS 396).  The address of a lane's words derives from the SAME carriers as the
  TMA route's per-sequence descriptor: the meta prefix ``cu_q[b]`` (``_item_bounds``' ``q_start``, the descriptor's GLOBAL_ADDRESS origin
  in ``emit_seq_descs``) and the ``dq_accum`` tensor's strides (``cast_dq_thd``'s spelling); rows >= ``seq_q`` are skipped by a branch
  (the descriptor's GLOBAL_DIM clip).  The first body (dp_tail) has no red route: every other value declines BY NAME there.
* DAY-3 PROBES + STAMPS (code-writer 2026-10-27 lane A day 3, optimizer 07:45 rank 0, barrier-inspector 08:01): three more ``diag_2cta``
  values defined on THIS map only -- ``"exchange_off"`` (D8s, the inspector's "D8x bytes faked": the XCHG ship skipped and ds_recv_full
  armed with 0 tx bytes, EVERY wait kept = the SAFE exchange-family ceiling; ``"ds_gates_off"`` (D8) is RETIRED here: it removed the arm's
  only back-edge and FAULTED 719), ``"stats_one_chunk"`` (D10: one LDS.128 stats / delta chunk per quarter reused for all eight) and
  ``"sdq_sts_off"`` (D9: the sDQ slice STS + its proxy fence skipped) -- and the typed bool field ``debug_clk`` (the STAMPS lever, the d512
  bodies' ``debug_clk`` pattern): every mbarrier wait of ONE representative warp per role is bracketed by %clock64 and accumulated per
  (role, barrier) into the 100 Int32 words of the bar block's spare (``DBG_CLK``, below), with the epilogue's per-tile hand-off segments,
  dumped per CTA at exit into a workspace region the adapter carves only under the lever; output CORRECT, timing diagnostic; default
  False = no traced code (the default rendering stays byte-identical).
* ROLE SHAPE (code-writer B, 2026-10-27, day 3 rank 1 "X-FA4" = FAv4's role shape on this body; barrier-inspector SKELETON
  resources/x-fa4-barrier-table.md 08:45 UTC, resource-manager resources/xfa4-budgets.md 08:15 UTC): the typed field ``warp_roles`` =
  ``"16w"`` (THE default: today's rendering, byte-identical) | ``"16w_fa4"`` (the SOFTMAX writes sDS own half + the XCHG half straight from
  its 32 packed dS^T words -- today's LDTM image word for word, the same SW128 ``(row % 8) ^ vec`` STS form -- fences ONCE, arrives ``ds_smem``
  (256 = SOFTMAX_LANES), passes the 256-lane XCHG named barrier and its lead warp's elected lane arms ``ds_recv_full`` + ships; a NEW softmax
  ``dq_done`` WAR wait (``dq_war`` st(1), +1 per wait) precedes the first dS STS of every tile; the EPILOGUE warps 8-11 become the REDUCE group
  (dq_done -> x64 dQ ld -> s_tail_empty -> rounds A, B; the item-end dV / dK staging kept; no dS work); ``ds_tmem`` AND ``dp_empty`` are
  DELETED (``BAR_LAYOUT_XFA4``: 32 names / 37 slots, END 230912, re-packed); the dQ ring keeps form A (the store warp issues); s_read kept;
  TMEM / SMEM / the 144-144-80 split UNCHANGED).  Resolved PER MAP (``dp_tail`` serves ``16w`` only); the kernel's ``warp_roles_code``
  Constexpr[int] (``WARP_ROLES_CODES``); the compile key carries the NAME.  Under ``16w_fa4`` the register-direct dQ routes are NOT BUILT and
  ``ds_gates_off`` (D8) is RETIRED (the ds_recv_full arm's back-edge runs through the leader's gate waits): both decline BY NAME.
"""

from dataclasses import dataclass

from .config_sm100_d128_accum_2cta import CFG as CFG_DP_TAIL
from .config_sm100_d128_accum_2cta import Config, diag_2cta_code, reg_split_counts, resolve_diag_2cta  # noqa: F401  (re-exported for the body)

# The typed vocabulary of the pair bodies' TMEM maps (adapter field ``SdpaBwdDslSm100D128.tmem_map``; ``None`` = the default).
TMEM_MAPS = ("dp_tail", "s_tail")
TMEM_MAP_DEFAULT = "dp_tail"  # the first body, byte-identical rendering (md5 c28e6013 @ 4/32/4, 28b0eba0 @ 2/8/2) when the field is unset


@dataclass(frozen=True)
class ConfigSTail(Config):
    """The first body's geometry with the arm-(ii) deltas; every inherited property re-derives from the fields below."""

    # RESOURCE-NOTE: Q1 single stage (FAv4 2-CTA form) funds sDQ 16 KiB; Q1(t+1) window ~2000 clk
    Q1_STAGES: int = 1
    # --- the dedicated dQ staging ring: DQ_SLOTS slices of DQ_SLICE_BYTES, DQ_ROUNDS rounds per tile (DQ_SLOTS * DQ_ROUNDS == DQ_SLICES) ---
    # RESOURCE-NOTE: sDQ 16 KiB dedicated ring, 2 x [64 q x 32 d] fp32 SW128; dS STS waits no dq_empty
    DQ_SLOTS: int = 2
    DQ_ROUNDS: int = 2
    # --- the leader's MMA issue order on this map: fa4 (default until MEASURED) | fa4b (kernel header section 5) ---
    PAIR_MMA_ORDER_DEFAULT: str = "fa4"
    # --- the dQ partial's route out of the CTA on this map: tma (default until MEASURED) | red_global | red_global_pair (header section 11) ---
    DQ_ROUTE_PAIR_DEFAULT: str = "tma"
    # --- the softmax's MASK FORM on this map (lever L5, code-writer D 2026-10-27): fsel (default until MEASURED) | predicated | diag_mask_off ---
    MASK_FORM_DEFAULT: str = "fsel"
    MASK_WORD_COLS: int = 32  # columns per keep-word = tile_dsl.mask.MASK_WORD_COLS (the body asserts the two agree at import)
    TMEM_MAP: str = "s_tail"
    # --- the STAMPS lever (typed adapter field ``debug_clk``; header section 11): default OFF = the shipped rendering, byte-identical ---
    DEBUG_CLK_DEFAULT: bool = False
    DBG_CLK_WORDS: int = 96  # Int32 words per CTA record = the WHOLE bar-block spare [BAR_LAYOUT_END, TMEM_HANDLE) = 384 B since the C2b row
    # dvk_stage_empty took 16 B of it (was 100 words / 400 B at the 34-row layout); validate_config pins the exact fit, the layout uses 82 words
    DBG_CLK_BLOCKED_THRESH: int = 128  # a wait above this many clk really waited (a first-test pass costs ~30-60 clk of issue): the d512 bodies' value
    DBG_CLK_HIST_EDGES: tuple = (100, 400, 1600)  # clk bins of the per-tile "ship issue - ds_tmem pass" segment; bin 0 = the optimizer's < 100 clk kill line
    DBG_CLK_MAGIC: int = 0x5354414D  # "STAM": the record's first word (a decoder refuses a record without it)
    DBG_CLK_RECORD_ALIGN: int = 256  # the stamps region starts at the first 256-B boundary past the chain's workspace regions
    # --- the DIAGNOSTIC probes' constants (header section 11; OUTPUT WRONG BY DESIGN under a probe) ---
    DIAG_EXCHANGE_OFF_TX: int = 0  # D8s: ds_recv_full armed with ZERO tx bytes -> every phase completes at its own arm (no copy is ever in flight)
    DIAG_STATS_CHUNK: int = 0  # D10: the ONE 4-wide stats / delta chunk of each quarter that is loaded and reused for all GROUP_QUARTER // 4 chunks
    # --- the ROLE SHAPE on this map (day 3 rank 1 "X-FA4"): 16w (default = today's rendering) | 16w_fa4 (FAv4's role shape; module docstring) ---
    WARP_ROLES_DEFAULT: str = "16w"
    # RESOURCE-NOTE: XCHG_BARRIER lanes: SOFTMAX_LANES 256 under 16w_fa4, EPILOGUE_LANES 128 under 16w
    SOFTMAX_LEAD_WARP: int = 0  # 16w_fa4: the softmax warp whose elected lane arms ds_recv_full and ships XCHG (FAv4's tidx 0; inspector X4)

    @property
    def PAIR_MMA_ORDERS(self):
        # fa4: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); tail dQ(hi-1) (FAv4's loop rotated onto the tile body; s_read / s_tail_empty
        # sites of resources/arm2-barrier-table.md D1 / D2).  fa4b: dP^T(t), dV(t), dQ(t-1), S^T(t+1), dK(t) -- the in-tile dQ slot one GEMM
        # later (the same sites, guards and counts; `dq_after_dv`).  s_first / dk_first need a DIFFERENT site set on this map: declined by name.
        return ("fa4", "fa4b")

    @property
    def DIAG_2CTA_NAMES_ON_MAP(self):
        # The probes THIS body has code for (the shared code space is config_sm100_d128_accum_2cta.CFG.DIAG_2CTA_NAMES).
        # D7 (dp_empty_wait_off) is MEANINGLESS on this map: dp_empty is arrived at the dS^T readout, four GEMMs before dP^T(t+1) needs
        # it, so skipping its wait measures nothing -> declined by name.  D8 (ds_gates_off) is RETIRED on this map (barrier-inspector
        # 2026-10-27 08:01): skipping the three dS gates removed the ONLY back-edge of the ds_recv_full expect_tx arm (arm(t+1) landed on a
        # tx-pending phase = P3 over-arrival -> CUDA 719 on every launch) -> declined by name; its safe replacement is exchange_off (D8s).
        #   dq_reduce_add_off (D2-pair): the store warp's four reduce-add ISSUES skipped (ring protocol kept; dQ zero)
        #   exchange_off      (D8s):     the epilogue's XCHG ship skipped + ds_recv_full armed with DIAG_EXCHANGE_OFF_TX = 0 bytes; the three dS
        #                                gates, the relay, the XCHG STS / proxy fence / named barrier and every other wait KEPT (the recv half
        #                                is never written: dQ wrong) = the exchange family's ceiling {lever 1, lever 2's exchange part, #4}
        #   stats_one_chunk   (D10):     _p_values / _ds_values load ONE stats / delta chunk per quarter (DIAG_STATS_CHUNK) and reuse it for all
        #                                eight: -DIAG_STATS_LDS_REMOVED_PER_LANE of DIAG_STATS_LDS_PER_LANE LDS.128 per lane per tile, registers /
        #                                compute / waits unchanged (the byte-removal-only residual test of the SMEM-pipe reading; I1's ceiling)
        #   sdq_sts_off       (D9):      _dq_round skips the sDQ slice STS + its proxy fence (the UTMAREDG reads stale staging; -256 wavefronts;
        #                                the dq_empty waits and the dq_full arrive stay) = the second point of the wavefront -> time slope
        return ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off")

    # --- derived: the D10 probe's counts (the softmax's stats / delta reads of header section 4: 4-wide chunks per q quarter, two quarters per group) ---
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
    def DBG_CLK_RECORD_BYTES(self):
        return self.DBG_CLK_WORDS * 4  # Int32 words

    @property
    def DQ_ROUTES_PAIR(self):
        # tma: sDQ STS -> dq_full -> the store warp's tma_reduce_add (today's path).  red_global: red.global.add.v4.f32 from the dQ readout
        # registers, the ring protocol kept (dq_full / dq_empty flips identical, no STS, no UTMAREDG).  red_global_pair: the same reds, lane
        # pairs exchange one v4 per 32-B sector (shfl.sync.bfly 1) so each warp-level red carries full sectors.  Perf-only on dK / dV (bitwise:
        # they never touch the route); dQ differs by the fp32 add ORDER only (non-deterministic under every route, the fp64 gate decides).
        return ("tma", "red_global", "red_global_pair")

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
        return "diag_mask_off"  # the one OUTPUT-WRONG-BY-DESIGN value of MASK_FORMS (the deletion probe of lever L5)

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
        # lever-2 shape (rank 2) is NOT built here (its split 144/64/64/64 at entry 96: resource-manager tables 5-6).
        return ("16w", "16w_fa4")

    # --- derived: the 16w_fa4 role names and lane counts (the SAME warps 8-11 and lanes; renamed: inspector section 1 "RENAMED" rows) ---
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

    # --- derived: the softmax's dS STS image under 16w_fa4 (resource-manager table 1: group g writes vec 4g .. 4g+3 of BOTH 64-q halves) ---
    @property
    def DS_STS_VECS_PER_QUARTER(self):
        return self.PACK_QUARTER_COLS // 4  # 4: a 32-q quarter = 16 packed Int32 words = four 16-B vectors of the lane's 128-B SW128 row

    @property
    def DS_STS_PER_LANE_PER_TILE(self):
        # RESOURCE-NOTE: dS STS from the 32 packed words = today's LDTM image; 4 own + 4 XCHG STS.128/lane
        return self.QUARTERS_PER_GROUP * self.DS_STS_VECS_PER_QUARTER  # 8 per softmax lane per tile = 64 warp-STS per CTA per tile (today's 4 x 16)

    def diag_2cta_names_for_roles(self, roles):
        """The probes a role shape SERVES: ``ds_gates_off`` (D8) is RETIRED under ``16w_fa4`` -- the ds_recv_full arm's back-edge is the softmax's
        dq_done WAR wait, which runs through the leader's gate waits D8 would skip (barrier-inspector section 5.3 / the 08:01 TX-PENDING reading)."""
        if roles == self.WARP_ROLES_DEFAULT:
            return self.DIAG_2CTA_NAMES_ON_MAP
        return tuple(name for name in self.DIAG_2CTA_NAMES_ON_MAP if name != "ds_gates_off")

    def dq_routes_for_roles(self, roles):
        """The dQ routes a role shape SERVES: the register-direct reds are NOT BUILT under ``16w_fa4`` (the reduce group has no red path; PLAN § 11.4
        drops the route) -- ``tma`` only there."""
        return self.DQ_ROUTES_PAIR if roles == self.WARP_ROLES_DEFAULT else (self.DQ_ROUTE_PAIR_DEFAULT,)

    # --- derived: the register-direct dQ route's counts (the 64-column half-row image: one lane = one d half of one q row) ---
    @property
    def DQ_RED_ROW_WORDS(self):
        return self.DQ_COLS  # fp32 words per lane per tile = the lane's 64 contiguous d of its row (== D_HALF, asserted)

    @property
    def DQ_RED_V4_PER_LANE(self):
        # RESOURCE-NOTE: red route: 16 x red.global.add.v4.f32 per lane per tile (128 lanes: 2048 per CTA); 0 sDQ STS, 0 UTMAREDG
        return self.DQ_RED_ROW_WORDS // 4  # 16 x 16-B groups per lane per tile

    @property
    def DQ_RED_SECTORS_PER_LANE(self):
        return self.DQ_RED_ROW_WORDS // 8  # 8 x 32-B sectors per lane per tile (the paired form's unit)

    @property
    def DQ_RED_V4_PER_TILE_PER_CTA(self):
        return self.EPILOGUE_LANES * self.DQ_RED_V4_PER_LANE  # 2048 (the sass-checker's RED count per CTA per tile; 64 warp-level instructions)

    # --- derived: the slab chain gains the sDQ ring between XCHG and STATS (Q -16384, +DQ 16384: SLABS_END unchanged) ---
    @property
    def DQ(self):
        return self.XCHG + self.XCHG_BYTES  # 212992

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
        raise ValueError("the s_tail map has no dQ staging alias on ds_own + XCHG (the dedicated sDQ ring: dq_slot(slot)); dq_slice is the dp_tail map's")

    # --- dkv_accum = fp32_reduce (C2b; resource-manager 08:25 table 9 form (i)): fp32 [M x DQ_BOX] staging slots over the dV / dK bf16 staging
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

    # --- derived: TMEM column map -- the dQ image in S^T's dead tail ---
    @property
    def DQ_COL(self):
        # RESOURCE-NOTE: dQ @64..128 = S^T dead tail: s_read gates dQ(t-1), s_tail_empty gates S^T(t+1)
        return self.S_COL + self.PACK_COLS  # 64

    @property
    def TMEM_DEAD_COL(self):
        return self.DS_COL + self.PACK_COLS  # 192: [TMEM_DEAD_COL, DV_COL) is DEAD (was the first body's dQ image)


CFG = ConfigSTail()


def resolve_tmem_map(name=None):
    """The pair body selector, typed: ``None`` -> ``TMEM_MAP_DEFAULT`` (the first body); any value outside ``TMEM_MAPS`` is a ``ValueError``
    BY NAME (a selector outside its domain declines, never a silent default)."""
    tmem_map = TMEM_MAP_DEFAULT if name is None else name
    if type(tmem_map) is not str or tmem_map not in TMEM_MAPS:
        raise ValueError(f"tmem_map must be one of {TMEM_MAPS}; got {tmem_map!r}")
    return tmem_map


def tmem_map_s_tail(name=None):
    """The chain's ``Constexpr[bool]`` for a resolved map name: the prepared host launches THIS body under ``s_tail``."""
    return resolve_tmem_map(name) == "s_tail"


def config_for_map(name=None):
    """The ``Config`` instance a resolved map name renders with (the first body's CFG or this module's)."""
    return {"dp_tail": CFG_DP_TAIL, "s_tail": CFG}[resolve_tmem_map(name)]


def resolve_pair_mma_order_for_map(name=None, tmem_map=None):
    """The leader MMA issue order ON A MAP, typed: ``None`` -> the map's ``PAIR_MMA_ORDER_DEFAULT``; a name outside the map's
    ``PAIR_MMA_ORDERS`` is a ``ValueError`` BY NAME (on ``s_tail`` only ``fa4`` exists; on ``dp_tail`` ``fa4`` is unreachable: its dQ image
    sits in dP^T's dead half and dp_empty is arrived after the dQ readout, kernel header section 5 of the first body)."""
    resolved = resolve_tmem_map(tmem_map)
    cfg = config_for_map(resolved)
    order = cfg.PAIR_MMA_ORDER_DEFAULT if name is None else name
    if type(order) is not str or order not in cfg.PAIR_MMA_ORDERS:
        raise ValueError(f"pair_mma_order must be one of {cfg.PAIR_MMA_ORDERS} on tmem_map {resolved!r}; got {order!r}")
    return order


def dq_after_dv_for_map(name=None, tmem_map=None):
    """The s_tail body's order ``Constexpr[bool]`` for a (resolved) order name ON A MAP: True iff the order is ``fa4b`` (dQ(t-1) issued after
    dV(t)); False for ``fa4`` and for every first-body order (the first body never reads it).  Declines exactly as
    :func:`resolve_pair_mma_order_for_map` (a name outside the map's domain is a ``ValueError`` BY NAME)."""
    return resolve_pair_mma_order_for_map(name, tmem_map) == "fa4b"


# The dQ route's kernel code space (``Constexpr[int]`` ``dq_route_code``; the body names DQ_ROUTE_TMA / _RED / _RED_PAIR from these).
DQ_ROUTE_PAIR_CODES = {"tma": 0, "red_global": 1, "red_global_pair": 2}


def resolve_dq_route_pair_for_map(name=None, tmem_map=None):
    """The dQ partial's route out of the CTA ON A MAP, typed: ``None`` -> the map's default (``"tma"`` on both maps); a name outside the
    map's domain is a ``ValueError`` BY NAME (on ``s_tail`` the domain is ``CFG.DQ_ROUTES_PAIR``; the first body (``dp_tail``) has NO
    register-direct route: only ``"tma"`` resolves there)."""
    resolved = resolve_tmem_map(tmem_map)
    allowed = CFG.DQ_ROUTES_PAIR if resolved == "s_tail" else (CFG.DQ_ROUTE_PAIR_DEFAULT,)
    route = CFG.DQ_ROUTE_PAIR_DEFAULT if name is None else name
    if type(route) is not str or route not in allowed:
        raise ValueError(f"dq_route_pair must be one of {allowed} on tmem_map {resolved!r}; got {route!r}")
    return route


def dq_route_pair_code(name=None, tmem_map=None):
    """The s_tail body's ``dq_route_code`` Constexpr[int] for a (resolved) route name ON A MAP (``DQ_ROUTE_PAIR_CODES``; 0 = tma on every map);
    declines exactly as :func:`resolve_dq_route_pair_for_map`."""
    return DQ_ROUTE_PAIR_CODES[resolve_dq_route_pair_for_map(name, tmem_map)]


# Why a probe of the shared code space is NOT defined on a map (the typed decline's text; every row reachable: validate_config trips each).
DIAG_2CTA_DECLINE_WHY = {
    ("dp_empty_wait_off", "s_tail"): "dp_empty is arrived at the dS^T readout on the s_tail map, so a skipped dp_empty wait probes nothing",
    ("ds_gates_off", "s_tail"): (
        "RETIRED on the s_tail map: skipping the three dS gates removes the only back-edge of the ds_recv_full expect_tx arm, so the arm of "
        "tile t+1 lands on a tx-pending phase and FAULTS (CUDA 719; barrier-inspector 2026-10-27 08:01) -- the safe exchange ceiling is 'exchange_off'"
    ),
    ("exchange_off", "dp_tail"): "the first body (dp_tail) carries no exchange_off probe: s_tail only",
    ("stats_one_chunk", "dp_tail"): "the first body (dp_tail) carries no stats_one_chunk probe: s_tail only",
    ("sdq_sts_off", "dp_tail"): "the first body (dp_tail) has no dedicated sDQ ring to probe: s_tail only",
}


# The mask form's kernel code space (``Constexpr[int]`` ``mask_form_code``; the body names MASK_FORM_FSEL / _PREDICATED / _DIAG_OFF from these).
MASK_FORM_CODES = {"fsel": 0, "predicated": 1, "diag_mask_off": 2}


def resolve_mask_form_for_map(name=None, tmem_map=None):
    """The softmax's MASK FORM on a map, typed: ``None`` -> the map's default (``"fsel"`` on both maps = today's rendering); a name outside
    the map's domain is a ``ValueError`` BY NAME (on ``s_tail`` the domain is ``CFG.MASK_FORMS``; the first body (``dp_tail``) has ONE form:
    only ``"fsel"`` resolves there).  ``CFG.MASK_FORM_DIAGNOSTIC`` (``"diag_mask_off"``) is the deletion probe: output WRONG BY DESIGN, resolved
    here like the ``diag_2cta`` probes (set on the adapter INSTANCE by the probe driver, keyed apart, refused by the driver's --validate)."""
    resolved = resolve_tmem_map(tmem_map)
    allowed = CFG.MASK_FORMS if resolved == "s_tail" else (CFG.MASK_FORM_DEFAULT,)
    form = CFG.MASK_FORM_DEFAULT if name is None else name
    if type(form) is not str or form not in allowed:
        raise ValueError(f"mask_form must be one of {allowed} on tmem_map {resolved!r}; got {form!r}")
    return form


def mask_form_code(name=None, tmem_map=None):
    """The s_tail body's ``mask_form_code`` Constexpr[int] for a (resolved) form name ON A MAP (``MASK_FORM_CODES``; 0 = fsel on every map);
    declines exactly as :func:`resolve_mask_form_for_map`."""
    return MASK_FORM_CODES[resolve_mask_form_for_map(name, tmem_map)]


def resolve_diag_2cta_for_map(name=None, tmem_map=None):
    """The DIAGNOSTIC probe ON A MAP, typed: the one code space of ``config_sm100_d128_accum_2cta.diag_2cta_code`` (so the kernel constants
    DIAG_* agree across the bodies), with the probes a map does not define declined BY NAME (each map's ``DIAG_2CTA_NAMES_ON_MAP``; the reason
    per (probe, map) in ``DIAG_2CTA_DECLINE_WHY``: a probe never renders as a silent no-op on a body without code for it)."""
    resolved = resolve_tmem_map(tmem_map)
    diag = resolve_diag_2cta(name)
    allowed = CFG_DP_TAIL.DIAG_2CTA_NAMES_ON_MAP if resolved == "dp_tail" else CFG.DIAG_2CTA_NAMES_ON_MAP
    if diag is not None and diag not in allowed:
        raise ValueError(f"diag_2cta {diag!r} is not defined on tmem_map {resolved!r} ({DIAG_2CTA_DECLINE_WHY[(diag, resolved)]}); choose from {allowed}")
    return diag


def resolve_debug_clk_for_map(value=None, tmem_map=None):
    """The STAMPS lever ON A MAP, typed: ``None`` -> ``CFG.DEBUG_CLK_DEFAULT`` (False: the shipped rendering); a bool; anything else (an int, a
    string) is a ``ValueError`` BY NAME; ``True`` on ``dp_tail`` declines BY NAME (the first body carries no stamps lever)."""
    resolved = resolve_tmem_map(tmem_map)
    flag = CFG.DEBUG_CLK_DEFAULT if value is None else value
    if type(flag) is not bool:
        raise ValueError(f"debug_clk must be a bool (None = {CFG.DEBUG_CLK_DEFAULT}); got {flag!r} ({type(flag).__name__})")
    if flag and resolved != "s_tail":
        raise ValueError(f"debug_clk is defined on tmem_map 's_tail' only (the first body carries no stamps lever); got True on tmem_map {resolved!r}")
    return flag


# The role shape's kernel code space (``Constexpr[int]`` ``warp_roles_code``; the body names WARP_ROLES_16W / WARP_ROLES_16W_FA4 from these).
WARP_ROLES_CODES = {"16w": 0, "16w_fa4": 1}
# The barrier rows that have neither producer nor consumer under 16w_fa4 (barrier-inspector X5 / section 6: ds_tmem -- no dS^T readout; dp_empty --
# its ONLY arrive site was that readout; dP^T(t+1)'s overwrite of the packed dS^T(t) is ordered by the in-order tensor pipe after dK(t), of dP^T(t)'s
# fp32 by ds_ready(t) -> dK(t)).  Deleted from BAR_LAYOUT_XFA4 with their states, waits and drains.
DELETED_BARRIERS_XFA4 = ("ds_tmem", "dp_empty")
# The barrier row ADDED under 16w_fa4 (barrier-inspector section 15, 2026-10-27 10:55 UTC, after the a90d6521 hang): `dq_war_done` = the softmax's sDS
# WAR signal, an MMA_COMMIT committed INSIDE `_dq_issue` at the two dQ issue sites only (BEFORE dq_done; never at the zero-trip empty-commit site),
# waited ONLY by the softmax (st(1) pre-armed, one wait per live tile, no zero-trip wait or advance, one unconditional post-loop drain): every flip
# depends on the softmax's own preceding STS, so the pre-armed waiter can never find a flip landed early (F in {w-1, w}) -- the shared `dq_done`
# waiter set of the first build raced the leader's zero-trip EMPTY commit (a zero-trip item as a cluster's first item: a HANG; consecutive zero-trips:
# a lap).  `dq_done` reverts to the reduce group alone (the 04:17 AUDIT row).
ADDED_BARRIERS_XFA4 = ("dq_war_done",)
# The barrier row that has NO USER under 16w_fa4 although it is in the 16w table (the stack fold 2026-10-27): writer C's C2b row `dvk_stage_empty`
# serves `dkv_accum = "fp32_reduce"`, whose fp32 dK / dV staging lives in the 16w EPILOGUE loop; under 16w_fa4 the field DECLINES BY NAME
# (`validate_fields_for_roles`), so the row is absent from BAR_LAYOUT_XFA4 / BarsXFA4 (no init, no arrive, no wait) and writer B's 33-row table,
# its END and its PTX record stand unchanged.
UNSERVED_BARRIERS_XFA4 = ("dvk_stage_empty",)


def resolve_warp_roles_for_map(name=None, tmem_map=None):
    """The ROLE SHAPE on a map, typed: ``None`` -> ``CFG.WARP_ROLES_DEFAULT`` ("16w", today's rendering); a name outside the map's domain is a
    ``ValueError`` BY NAME (``16w_fa4`` exists on ``s_tail`` only: the first body (``dp_tail``) keeps its one role shape)."""
    resolved = resolve_tmem_map(tmem_map)
    allowed = CFG.WARP_ROLES if resolved == "s_tail" else (CFG.WARP_ROLES_DEFAULT,)
    roles = CFG.WARP_ROLES_DEFAULT if name is None else name
    if type(roles) is not str or roles not in allowed:
        raise ValueError(f"warp_roles must be one of {allowed} on tmem_map {resolved!r}; got {roles!r}")
    return roles


def warp_roles_code(name=None, tmem_map=None):
    """The s_tail body's ``warp_roles_code`` Constexpr[int] for a (resolved) role-shape name ON A MAP (``WARP_ROLES_CODES``; 0 = 16w on every map);
    declines exactly as :func:`resolve_warp_roles_for_map`."""
    return WARP_ROLES_CODES[resolve_warp_roles_for_map(name, tmem_map)]


def validate_fields_for_roles(roles, dq_route_pair=None, diag_2cta=None, debug_clk=False, dkv_accum=None):
    """The field bundle a (resolved) role shape SERVES, typed BY NAME: under ``16w_fa4`` a register-direct dQ route is NOT BUILT and ``ds_gates_off``
    (D8) is RETIRED (:meth:`ConfigSTail.dq_routes_for_roles` / :meth:`ConfigSTail.diag_2cta_names_for_roles`).  Called by ``compile_host`` before
    any trace and asserted here; returns ``roles``."""
    if roles not in CFG.WARP_ROLES:
        raise ValueError(f"warp_roles must be one of {CFG.WARP_ROLES}; got {roles!r}")
    # the stack fold: under the DEFAULT shape the MAP resolvers own every domain (dp_tail included: the first body's probes resolve there);
    # the role filter below removes names under a NON-default shape only
    if roles == CFG.WARP_ROLES_DEFAULT:
        return roles
    routes = CFG.dq_routes_for_roles(roles)
    if dq_route_pair is not None and dq_route_pair not in routes:
        raise ValueError(f"dq_route_pair {dq_route_pair!r} is not built under warp_roles {roles!r} (the reduce group has no red path); choose from {routes}")
    diags = CFG.diag_2cta_names_for_roles(roles)
    if diag_2cta is not None and diag_2cta not in diags:
        raise ValueError(
            f"diag_2cta {diag_2cta!r} is not defined under warp_roles {roles!r} (ds_gates_off is RETIRED there: the ds_recv_full arm's back-edge "
            f"runs through the leader's gate waits it would skip); choose from {diags}"
        )
    if debug_clk and roles != CFG.WARP_ROLES_DEFAULT:  # the stack fold: the STAMPS record layout (DBG_CLK.ROLE_WAITS / WORD) is the 16w roles'
        raise ValueError(
            f"debug_clk is not defined under warp_roles {roles!r} (the STAMPS record layout names the 16w roles' waits: no words for the softmax's "
            f"dq_war_done wait or the reduce group; a 16w_fa4 layout is a design step, not a merge); choose warp_roles {CFG.WARP_ROLES_DEFAULT!r}"
        )
    from .d128_grouped_contract import DKV_ACCUM_DEFAULT, resolve_dkv_accum  # local: the contract module is the field's truth, the config stays light

    if resolve_dkv_accum(dkv_accum) != DKV_ACCUM_DEFAULT and roles != CFG.WARP_ROLES_DEFAULT:
        raise ValueError(
            f"dkv_accum {dkv_accum!r} is not built under warp_roles {roles!r} (the fp32 dK / dV staging rounds live in the 16w epilogue loop; the "
            f"reduce group of 16w_fa4 stages bf16 halves only -- the seam is the barrier-inspector's before a re-site); choose warp_roles "
            f"{CFG.WARP_ROLES_DEFAULT!r} or dkv_accum {DKV_ACCUM_DEFAULT!r}"
        )
    return roles


# (name, byte offset, active slots, init count, producer kind, scope) -- barrier-inspector 2026-10-26 (SKELETON) + 2026-10-27 (ARM (ii) rows).
# Pasted from resources/arm2-barrier-table.md section 10 (34 rows / 39 slots, re-packed in declaration order, 16 B per row; END 230944 = 544 B
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
    # dq_empty: store warp after each slot's reduce-add source read per round (D4 closed release); epilogue pre-armed, NO bootstrap (D3)
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
    # C2b (dkv_accum = fp32_reduce; code-writer C 2026-10-27, resource-manager 08:25 table 9 form (i)): the store warp after round 1's fp32 reduce-add
    # source reads drained -> the epilogue's round-2 staging may overwrite the five slots; 1 per item incl. zero-trip; INIT ONLY under dkv_fp32 (no
    # user under the default rendering, which stays byte-identical); AUDIT owed with the X-FA4 AUDIT
    ("dvk_stage_empty", 230944, 1, CFG.STORE_LANES, "thread", "LOCAL"),
)
BAR_LAYOUT_END = 230960  # next free byte of the bar block (384 B of the 944 spare); 35 names / 40 slots, 560 B
BAR_LAYOUT_STATUS = (
    "FILLED by barrier-inspector 2026-10-27 01:05 UTC (SKELETON, ARM (ii), resources/arm2-barrier-table.md sections 1-10): 34 names / 39 slots; "
    "NEW s_read (512) + s_tail_empty (256), dp_empty arrive MOVED to the dS^T readout, xchg_state split (P2), q1 ring 1 stage, dq rings 2 slots "
    "x 2 rounds (D4 per-round release), dq_empty pre-armed without bootstrap (D3), dv_full 160 (fix A) carried; AUDIT from the committed source owed; "
    "+ dvk_stage_empty (C2b, code-writer C 2026-10-27: THREAD 32 store -> epilogue, fp32 rounds only, init under dkv_fp32; AUDIT owed with X-FA4)"
)
BAR_KIND_COUNTS = {"tma_leader": 7, "mma": 9, "thread": 9, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 1}

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
        "p_ready",
        "do2_full",
        "q2_full",
        "ds_ready",
        "sched_full",
    ),
    "softmax": ("s_done", "dp_done", "sched_full"),  # warp 0 of the 8
    "epilogue": ("ds_tmem", "dq_empty", "dq_done", "sched_full"),  # the lead warp (dq_empty = both slots of a round, one word)
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


def _dbg_clk_layout():
    names = list(DBG_CLK_HEADER)
    names += [f"wait:leader:{b}" for b in DBG_CLK_ROLE_WAITS["leader"]]
    names += [f"blocked:leader:{b}" for b in DBG_CLK_ROLE_WAITS["leader"]]
    for role in ("softmax", "epilogue", "follower", "loader", "store"):
        names += [f"wait:{role}:{b}" for b in DBG_CLK_ROLE_WAITS[role]]
    names += list(DBG_CLK_SEGMENTS)
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


DBG_CLK = DbgClk(
    SMEM=BAR_LAYOUT_END,
    WORDS=CFG.DBG_CLK_WORDS,
    BYTES=CFG.DBG_CLK_RECORD_BYTES,
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


def dbg_clk_records(n_batch, nheads, max_skv):
    """CTA records per launch = the s_tail MAIN kernel's DECLARED grid (header section 14: ceil(max_skv / THD_UNIT_ROWS) pairs, at least one, x
    CGA_M CTAs x nheads x n_batch).  Every CTA that RUNS writes its record at exit (a zero-trip CTA writes zeros + the magic); a cluster the CLC
    scheduler cancelled never launches and leaves its two records unwritten (the region is not zeroed: the execute path allocates / converts
    nothing), so a reader keeps the records whose first word is DBG_CLK.MAGIC (the decoder reports the never-launched count)."""
    if any(type(n) is not int or n <= 0 for n in (n_batch, nheads, max_skv)):
        raise ValueError("dbg_clk_records wants positive ints (n_batch, nheads, max_skv)")
    return max((max_skv + CFG.THD_UNIT_ROWS - 1) // CFG.THD_UNIT_ROWS, 1) * CFG.CGA_M * nheads * n_batch


def dbg_clk_bytes(n_batch, nheads, max_skv):
    return dbg_clk_records(n_batch, nheads, max_skv) * DBG_CLK.BYTES


def dbg_clk_region_offset(regions_end):
    """Byte offset of the stamps region inside the chain's workspace payload: the first RECORD_ALIGN boundary at or past the end of the chain's own
    regions (``thd_workspace_offsets``' total).  Pure arithmetic: the prepared host calls it on its runtime Int64, the adapter on Python ints."""
    return (regions_end + (DBG_CLK.RECORD_ALIGN - 1)) & -DBG_CLK.RECORD_ALIGN


# Alias tenants of the time-shared bytes: (tenant, bytes it occupies, the seam the inspector owns).  The first body's dq_staging row is DELETED:
# RESOURCE-NOTE: dq_staging alias on ds_own+XCHG dropped: dq_done(t) lands ~150 clk before dS(t+1)
ALIASES = (
    ("dk_store", "ds_own(rank) + XCHG", "dq_done(t_last) (the dQ MMA read sDS; the ship read XCHG) + dvk_empty before the next item's first dS write"),
    ("dv_store", "K2", "last tile's dq_done -> k2_empty before the next block's K2 load"),
    ("ds_recv(rank)", "DS, the peer-rank half", "written ONLY by the peer's bulk copy; its next write waits the peer's dq_done(t)"),
)

# (name, byte offset, active slots, init count, producer kind, scope) -- barrier-inspector 2026-10-27 08:45 UTC (X-FA4 16w, resources/x-fa4-barrier-table.md
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
    ),  # TWO waiter sets: reduce `dq_state` st(0) (readout) + softmax `dq_war` st(1) (the sDS WAR before every dS STS)
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
    "FILLED by barrier-inspector 2026-10-27 08:45 UTC (SKELETON, X-FA4 16w, resources/x-fa4-barrier-table.md sections 0-14) + the section-15 ZERO-TRIP "
    "CORRECTION 10:55 UTC: 33 names / 38 slots; ds_tmem + dp_empty DELETED (X5 / section 6), ds_smem 128 -> 256 from the SOFTMAX (X2), ds_recv_full armed "
    "by softmax warp 0 after the 256-lane XCHG bar.sync (X3 / X4), dq_war_done NEW = the softmax's WAR signal (section 15.3; dq_done back to the reduce "
    "group alone), s_read kept (X1), dQ ring form A (X6); AUDIT from the committed source owed (section 12 + 15.5 punch lists)"
)


def bar_layout_for_roles(roles):
    """The barrier table a (resolved) role shape renders with: the audited 34-row ``BAR_LAYOUT`` (``16w``) or the inspector's 32-row ``BAR_LAYOUT_XFA4``."""
    return {"16w": BAR_LAYOUT, "16w_fa4": BAR_LAYOUT_XFA4}[roles]


def bar_layout_end_for_roles(roles):
    return {"16w": BAR_LAYOUT_END, "16w_fa4": BAR_LAYOUT_XFA4_END}[roles]


def bar_kind_counts_for_roles(roles):
    return {"16w": BAR_KIND_COUNTS, "16w_fa4": BAR_KIND_COUNTS_XFA4}[roles]


def validate_config():
    """The arm-(ii) deltas on top of the first body's geometry; CPU only (raises at import).  Every number the kernel header quotes is here."""
    c = CFG
    base = CFG_DP_TAIL
    assert c.TMEM_MAP == "s_tail" and TMEM_MAPS == ("dp_tail", "s_tail") and TMEM_MAP_DEFAULT == "dp_tail"
    # unchanged geometry (inherited): MMA shapes, pair extents, boxes, warps, registers, the per-stage slab bytes, the lane constants
    assert (c.M, c.N, c.D, c.K_HW, c.BPE, c.BPE_ACC, c.CTA_MMA, c.CGA_M, c.CGA_N) == (base.M, base.N, base.D, base.K_HW, base.BPE, base.BPE_ACC, 2, 2, 1)
    assert (c.PAIR_KV_ROWS, c.Q_HALF, c.D_HALF, c.COLLECTIVE_M_KV, c.COLLECTIVE_M_Q) == (256, 64, 64, 256, 128)
    assert (c.Q1_BYTES, c.Q2_BYTES, c.DO1_BYTES, c.DO2_BYTES) == (16384,) * 4 and (c.K1_BYTES, c.K2_BYTES, c.V_BYTES, c.DS_BYTES) == (32768,) * 4
    assert (c.DS_HALF_BYTES, c.XCHG_BYTES, c.DQ_SLICES, c.DQ_SLICE_BYTES) == (16384, 16384, 4, 8192)
    assert (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS, c.ENTRY_REGS, c.REG_SPLITS) == (144, 144, 80, 128, base.REG_SPLITS)
    assert (c.MAPS_PER_SEQUENCE, c.MAP_BOX_ROWS, c.THD_UNIT_ROWS, c.Q_TILES_PER_PAIR) == (9, base.MAP_BOX_ROWS, 256, 2)
    assert (c.PACK_QUARTER_COLS, c.PACK_GROUP_COLS, c.K_STEPS_PER_BOX, c.K2_HALF_BYTES, c.DS_A_LBO) == (16, 32, 4, 16384, 8192)
    assert [c.ts_k_perm(k) for k in range(c.N // c.K_HW)] == [base.ts_k_perm(k) for k in range(base.N // base.K_HW)]
    assert (c.TMA_LOAD_FORMS, c.TMA_LOAD_FORM_DEFAULT) == (base.TMA_LOAD_FORMS, "multicast")
    assert (c.STORE_WAIT_FORMS, c.STORE_WAIT_FORM_DEFAULT) == (base.STORE_WAIT_FORMS, "retry")
    assert (c.REG_SPLIT_NAMES, c.REG_SPLIT_DEFAULT, c.DIAG_2CTA_NAMES) == (base.REG_SPLIT_NAMES, "144/144/80", base.DIAG_2CTA_NAMES)
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
    # dkv_accum = fp32_reduce (C2b, form (i)): 5 + 3 fp32 16 KiB slots over [K2, DQ) = the dV / dK bf16 staging bytes + the dS slab; 1024-B aligned,
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
    assert c.TMEM_USED <= c.TMEM_COLS and c.DQ_LANE_SPLIT == 64 and c.TMEM_COLS == base.TMEM_COLS
    # the order and the probes ON THIS MAP (typed; every reader declines by name)
    assert c.PAIR_MMA_ORDERS == ("fa4", "fa4b") and c.PAIR_MMA_ORDER_DEFAULT == "fa4"  # fa4b = dQ(t-1) after dV(t); the default stays fa4 until MEASURED
    assert resolve_pair_mma_order_for_map(None, "s_tail") == "fa4" and resolve_pair_mma_order_for_map("fa4", "s_tail") == "fa4"
    assert resolve_pair_mma_order_for_map("fa4b", "s_tail") == "fa4b"
    assert dq_after_dv_for_map(None, "s_tail") is False and dq_after_dv_for_map("fa4", "s_tail") is False and dq_after_dv_for_map("fa4b", "s_tail") is True
    assert dq_after_dv_for_map(None, None) is False and dq_after_dv_for_map("s_first", "dp_tail") is False and dq_after_dv_for_map("dk_first", None) is False
    assert resolve_pair_mma_order_for_map(None, None) == base.PAIR_MMA_ORDER_DEFAULT and resolve_pair_mma_order_for_map(None, "dp_tail") == "s_first"
    for bad_order, bad_map in (("s_first", "s_tail"), ("dk_first", "s_tail"), ("fa4", "dp_tail"), ("fa4", None), ("fa4b", "dp_tail"), ("fa4b", None)):
        try:
            resolve_pair_mma_order_for_map(bad_order, bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_order, bad_map))
    # the dQ route ON THIS MAP (typed; the default is the TMA path until MEASURED; the first body resolves "tma" only)
    assert c.DQ_ROUTES_PAIR == ("tma", "red_global", "red_global_pair") and c.DQ_ROUTE_PAIR_DEFAULT == "tma" == c.DQ_ROUTES_PAIR[0]
    assert DQ_ROUTE_PAIR_CODES == {"tma": 0, "red_global": 1, "red_global_pair": 2} and tuple(DQ_ROUTE_PAIR_CODES) == c.DQ_ROUTES_PAIR
    assert resolve_dq_route_pair_for_map(None, "s_tail") == "tma" and resolve_dq_route_pair_for_map("tma", "s_tail") == "tma"
    assert (
        resolve_dq_route_pair_for_map("red_global", "s_tail") == "red_global"
        and resolve_dq_route_pair_for_map("red_global_pair", "s_tail") == "red_global_pair"
    )
    assert resolve_dq_route_pair_for_map(None, None) == "tma" and resolve_dq_route_pair_for_map("tma", "dp_tail") == "tma"
    assert (dq_route_pair_code(None, None), dq_route_pair_code(None, "s_tail"), dq_route_pair_code("red_global", "s_tail")) == (0, 0, 1)
    assert dq_route_pair_code("red_global_pair", "s_tail") == 2
    for bad_route, bad_map in (("red_global", "dp_tail"), ("red_global", None), ("red_global_pair", None), ("bogus", "s_tail"), ("", "s_tail"), (1, "s_tail")):
        try:
            resolve_dq_route_pair_for_map(bad_route, bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_route, bad_map))
    # the red route's counts: one lane holds the 64 contiguous d of ONE q row (the M 128 half-row image: lane m % 64 + 64 * (d // 64))
    assert c.DQ_RED_ROW_WORDS == c.DQ_COLS == c.D_HALF == 64 and c.DQ_RED_V4_PER_LANE == 16 and c.DQ_RED_SECTORS_PER_LANE == 8
    assert c.DQ_RED_V4_PER_LANE * 4 == c.DQ_RED_ROW_WORDS == c.DQ_ROUNDS * c.DQ_BOX and c.DQ_RED_SECTORS_PER_LANE * 2 == c.DQ_RED_V4_PER_LANE
    assert c.DQ_RED_V4_PER_TILE_PER_CTA == c.EPILOGUE_LANES * c.DQ_RED_V4_PER_LANE == 2048 and c.EPILOGUE_LANES == 2 * c.Q_HALF
    # the probes ON THIS MAP (D7 and D8 decline BY NAME here; the three day-3 probes decline by name on dp_tail; every decline row reachable)
    assert c.DIAG_2CTA_NAMES_ON_MAP == ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off") and set(c.DIAG_2CTA_NAMES_ON_MAP) < set(
        c.DIAG_2CTA_NAMES
    )
    assert base.DIAG_2CTA_NAMES_ON_MAP == ("dq_reduce_add_off", "dp_empty_wait_off", "ds_gates_off") and "ds_gates_off" not in c.DIAG_2CTA_NAMES_ON_MAP
    assert resolve_diag_2cta_for_map(None, "s_tail") is None and resolve_diag_2cta_for_map("dq_reduce_add_off", "s_tail") == "dq_reduce_add_off"
    assert [resolve_diag_2cta_for_map(n, "s_tail") for n in c.DIAG_2CTA_NAMES_ON_MAP] == list(c.DIAG_2CTA_NAMES_ON_MAP)
    assert (
        resolve_diag_2cta_for_map("dp_empty_wait_off", "dp_tail") == "dp_empty_wait_off" and resolve_diag_2cta_for_map("ds_gates_off", None) == "ds_gates_off"
    )
    for bad_diag, bad_map in DIAG_2CTA_DECLINE_WHY:
        try:
            resolve_diag_2cta_for_map(bad_diag, bad_map)
        except ValueError as error:
            assert DIAG_2CTA_DECLINE_WHY[(bad_diag, bad_map)] in str(error) and f"is not defined on tmem_map {bad_map!r}" in str(error), str(error)
        else:
            raise AssertionError((bad_diag, bad_map, "must decline by name"))
    assert set(DIAG_2CTA_DECLINE_WHY) == {(n, "s_tail") for n in base.DIAG_2CTA_NAMES if n not in c.DIAG_2CTA_NAMES_ON_MAP} | {
        (n, "dp_tail") for n in c.DIAG_2CTA_NAMES_ON_MAP if n not in base.DIAG_2CTA_NAMES_ON_MAP
    }
    assert (c.DIAG_EXCHANGE_OFF_TX, c.DIAG_STATS_CHUNK) == (0, 0) and 0 <= c.DIAG_STATS_CHUNK < c.DIAG_STATS_CHUNKS_PER_QUARTER == 8
    assert (c.DIAG_STATS_LDS_PER_LANE, c.DIAG_STATS_LDS_REMOVED_PER_LANE) == (32, 28) and c.DIAG_STATS_LDS_REMOVED_PER_LANE * c.SOFTMAX_WARPS == 224
    # the STAMPS lever: typed bool, s_tail only; the record fits the bar block's spare; the layout is unique, contiguous, and names real barriers
    assert c.DEBUG_CLK_DEFAULT is False and resolve_debug_clk_for_map(None, "s_tail") is False and resolve_debug_clk_for_map(True, "s_tail") is True
    assert resolve_debug_clk_for_map(False, None) is False and resolve_debug_clk_for_map(None, None) is False
    for bad_flag, bad_map in ((True, "dp_tail"), (True, None), (1, "s_tail"), ("True", "s_tail"), (0, None), ("", "s_tail")):
        try:
            resolve_debug_clk_for_map(bad_flag, bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_flag, bad_map, "debug_clk must decline by name"))
    d = DBG_CLK
    assert d.SMEM == BAR_LAYOUT_END == 230960 and d.WORDS == 96 and d.BYTES == 384 and d.SMEM % 16 == 0
    assert d.SMEM + d.BYTES == c.TMEM_HANDLE and d.BYTES == c.TMEM_HANDLE - BAR_LAYOUT_END  # the record IS the spare: no byte past it, none left
    assert len(d.WORD) <= d.WORDS and sorted(d.WORD.values()) == list(range(len(d.WORD))) and d.MAGIC == c.DBG_CLK_MAGIC != 0
    assert d.BLOCKED_THRESH == 128 and d.HIST_EDGES == (100, 400, 1600) == tuple(sorted(d.HIST_EDGES)) and d.RECORD_ALIGN == 256
    bar_names = {row[0] for row in BAR_LAYOUT}
    for role, waits in d.ROLE_WAITS.items():
        assert waits and len(set(waits)) == len(waits) and set(waits) <= bar_names, role
        assert all(f"wait:{role}:{b}" in d.WORD for b in waits), role
    for b in d.ROLE_WAITS["leader"]:
        assert d.WORD[f"blocked:leader:{b}"] - d.WORD[f"wait:leader:{b}"] == d.LEADER_BLOCKED_STRIDE == 17, b
    assert [d.WORD[n] for n in d.HEADER] == list(range(len(d.HEADER))) and d.HEADER[0] == "magic" and len(d.SEGMENTS) == 18
    assert set(d.ROLE_WAITS) == {"leader", "softmax", "epilogue", "follower", "loader", "store"} and len(d.WORD) == 82
    assert (d.TAG_BACK_IN_TILE, d.TAG_BACK_TAIL, d.TAG_BACK_RELAY) == (2, 1, 1)  # the handshake's tag arithmetic (DbgClk's comment)
    assert (
        dbg_clk_records(1, 1, 1) == c.CGA_M == 2
        and dbg_clk_records(4, 32, 16384) == 64 * 2 * 32 * 4
        and dbg_clk_bytes(1, 1, 256) == 2 * c.DBG_CLK_RECORD_BYTES == 2 * 384
    )
    assert dbg_clk_region_offset(0) == 0 and dbg_clk_region_offset(1) == 256 and dbg_clk_region_offset(256) == 256 and dbg_clk_region_offset(257) == 512
    for bad in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (1.0, 1, 1)):
        try:
            dbg_clk_records(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    # the softmax's MASK FORM on this map (typed; the default is today's FSEL rendering until MEASURED; the first body resolves "fsel" only)
    assert c.MASK_FORMS == ("fsel", "predicated", "diag_mask_off") and c.MASK_FORM_DEFAULT == "fsel" == c.MASK_FORMS[0]
    assert c.MASK_FORM_DIAGNOSTIC == "diag_mask_off" and c.MASK_FORM_DIAGNOSTIC in c.MASK_FORMS and c.MASK_FORM_DIAGNOSTIC != c.MASK_FORM_DEFAULT
    assert MASK_FORM_CODES == {"fsel": 0, "predicated": 1, "diag_mask_off": 2} and tuple(MASK_FORM_CODES) == c.MASK_FORMS
    assert resolve_mask_form_for_map(None, "s_tail") == "fsel" and resolve_mask_form_for_map("fsel", "s_tail") == "fsel"
    assert resolve_mask_form_for_map("predicated", "s_tail") == "predicated" and resolve_mask_form_for_map("diag_mask_off", "s_tail") == "diag_mask_off"
    assert resolve_mask_form_for_map(None, None) == "fsel" and resolve_mask_form_for_map("fsel", "dp_tail") == "fsel"
    assert (mask_form_code(None, None), mask_form_code(None, "s_tail"), mask_form_code("predicated", "s_tail"), mask_form_code("diag_mask_off", "s_tail")) == (
        0,
        0,
        1,
        2,
    )
    for bad_form, bad_map in (
        ("predicated", "dp_tail"),
        ("predicated", None),
        ("diag_mask_off", None),
        ("bogus", "s_tail"),
        ("", "s_tail"),
        (1, "s_tail"),
        ("FSEL", "s_tail"),
    ):
        try:
            resolve_mask_form_for_map(bad_form, bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_form, bad_map))
    # the softmax lane's mask cost per tile: one select per cell of its two 32-column q quarters over one keep-word per quarter; the two groups'
    # quarters tile the 128 q columns, so the CTA's 256 softmax lanes select 128 x 128 cells per tile (the sass census' 64 FSEL + 8 R2P per lane)
    assert c.MASK_WORD_COLS == 32 == c.GROUP_QUARTER and c.QUARTERS_PER_GROUP == 2 and c.SOFTMAX_GROUPS == 2
    assert c.MASK_SELECTS_PER_LANE_PER_TILE == 64 == c.QUARTERS_PER_GROUP * c.GROUP_QUARTER and c.MASK_WORDS_PER_LANE_PER_TILE == 2
    assert c.MASK_SELECTS_PER_LANE_PER_TILE * c.SOFTMAX_GROUPS == c.N and c.MASK_SELECTS_PER_LANE_PER_TILE * c.SOFTMAX_LANES == c.M * c.N
    assert tmem_map_s_tail(None) is False and tmem_map_s_tail("dp_tail") is False and tmem_map_s_tail("s_tail") is True
    assert config_for_map(None) is base and config_for_map("s_tail") is c
    for bad in ("bogus", True, 1, "", "stail", "S_TAIL"):
        try:
            resolve_tmem_map(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
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
    assert [n for n in names if n not in ("s_read", "s_tail_empty", "dvk_stage_empty")] == first_body  # every other row keeps its name and relative order
    assert names[-1] == "dvk_stage_empty"  # the C2b row is appended after the first body's last row (tmem_dealloc)
    assert len(ALIASES) == 3 and [a[0] for a in ALIASES] == ["dk_store", "dv_store", "ds_recv(rank)"]  # the dq_staging alias is deleted
    assert (c.PAIR_EPILOGUE_LANES, c.PAIR_SOFTMAX_LANES, c.DQ_SLICE_LANES, c.STORE_LANES) == (256, 512, 64, 32)
    # --- X-FA4 (warp_roles 16w_fa4; barrier-inspector resources/x-fa4-barrier-table.md section 8): the role shape is typed, resolved PER MAP, and its
    # 32-row table equals the audited 34-row table minus the two deleted rows with ds_smem counted from the softmax, re-packed (config = truth) ---
    assert c.WARP_ROLES == ("16w", "16w_fa4") and c.WARP_ROLES_DEFAULT == "16w" == c.WARP_ROLES[0] and tuple(WARP_ROLES_CODES) == c.WARP_ROLES
    assert WARP_ROLES_CODES == {"16w": 0, "16w_fa4": 1} and DELETED_BARRIERS_XFA4 == ("ds_tmem", "dp_empty") and ADDED_BARRIERS_XFA4 == ("dq_war_done",)
    assert resolve_warp_roles_for_map(None, "s_tail") == "16w" == resolve_warp_roles_for_map("16w", "s_tail") == resolve_warp_roles_for_map(None, None)
    assert resolve_warp_roles_for_map("16w_fa4", "s_tail") == "16w_fa4" and resolve_warp_roles_for_map("16w", "dp_tail") == "16w"
    assert (warp_roles_code(None, None), warp_roles_code(None, "s_tail"), warp_roles_code("16w_fa4", "s_tail"), warp_roles_code("16w", "dp_tail")) == (
        0,
        0,
        1,
        0,
    )
    for bad_roles, bad_map in (
        ("16w_fa4", "dp_tail"),
        ("16w_fa4", None),
        ("bogus", "s_tail"),
        ("", "s_tail"),
        (1, "s_tail"),
        ("16W", "s_tail"),
        ("20w", "s_tail"),
    ):
        try:
            resolve_warp_roles_for_map(bad_roles, bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_roles, bad_map))
    assert (
        validate_fields_for_roles("16w", "red_global", "dq_reduce_add_off") == "16w"  # ds_gates_off is RETIRED on the MAP (writer A): no role serves it
        and validate_fields_for_roles("16w_fa4", "tma", "dq_reduce_add_off") == "16w_fa4"
    )
    assert validate_fields_for_roles("16w_fa4", None, None) == "16w_fa4"
    # the stack fold: debug_clk and dkv_accum = fp32_reduce decline BY NAME under 16w_fa4 and pass under 16w; the unserved row is named
    assert validate_fields_for_roles("16w_fa4", None, None, debug_clk=False, dkv_accum="bf16_partials") == "16w_fa4"
    assert validate_fields_for_roles("16w", None, None, debug_clk=True, dkv_accum="fp32_reduce") == "16w"
    for bad_kw in ({"debug_clk": True}, {"dkv_accum": "fp32_reduce"}):
        try:
            validate_fields_for_roles("16w_fa4", None, None, **bad_kw)
        except ValueError as error:
            assert "is not defined under warp_roles '16w_fa4'" in str(error) or "is not built under warp_roles '16w_fa4'" in str(error), str(error)
        else:
            raise AssertionError(bad_kw)
    assert UNSERVED_BARRIERS_XFA4 == ("dvk_stage_empty",) and all(n in {row[0] for row in BAR_LAYOUT} for n in UNSERVED_BARRIERS_XFA4)
    assert not any(row[0] in UNSERVED_BARRIERS_XFA4 for row in BAR_LAYOUT_XFA4)
    for bad_roles, bad_route, bad_diag in (
        ("16w_fa4", "red_global", None),
        ("16w_fa4", "red_global_pair", None),
        ("16w_fa4", None, "ds_gates_off"),
        ("20w", None, None),
    ):
        try:
            validate_fields_for_roles(bad_roles, bad_route, bad_diag)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_roles, bad_route, bad_diag))
    # the stack: ds_gates_off left the MAP domain (writer A's retirement), so the role filter is the identity on both shapes; D8s / D10 / D9 serve both
    assert (
        c.diag_2cta_names_for_roles("16w") == c.DIAG_2CTA_NAMES_ON_MAP == c.diag_2cta_names_for_roles("16w_fa4")
        and "ds_gates_off" not in c.DIAG_2CTA_NAMES_ON_MAP
    )
    assert c.dq_routes_for_roles("16w") == c.DQ_ROUTES_PAIR and c.dq_routes_for_roles("16w_fa4") == ("tma",)
    # the role-derived lane counts (never a literal in the body): the XCHG named barrier and ds_smem follow the warps that write XCHG / sDS own
    assert c.xchg_barrier_lanes("16w") == c.ds_smem_lanes("16w") == c.EPILOGUE_LANES == 128
    assert c.xchg_barrier_lanes("16w_fa4") == c.ds_smem_lanes("16w_fa4") == c.SOFTMAX_LANES == 256
    assert (c.SOFTMAX_LEAD_WARP, c.REDUCE_LEAD_WARP, c.REDUCE_WARPS, c.REDUCE_LANES, c.PAIR_REDUCE_LANES) == (0, 8, 4, 128, 256)
    assert 0 <= c.SOFTMAX_LEAD_WARP < c.SOFTMAX_WARPS and c.REDUCE_LEAD_WARP == c.SOFTMAX_WARPS and c.PAIR_REDUCE_LANES == c.PAIR_EPILOGUE_LANES
    # the softmax's dS STS image (resource-manager table 1 / 2): 4 vecs per 32-q quarter, 2 quarters per group = 8 STS.128 per lane per tile; the two
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
    assert bar_layout_for_roles("16w") is BAR_LAYOUT and bar_layout_for_roles("16w_fa4") is BAR_LAYOUT_XFA4
    assert (bar_layout_end_for_roles("16w"), bar_layout_end_for_roles("16w_fa4")) == (BAR_LAYOUT_END, BAR_LAYOUT_XFA4_END)
    assert (bar_kind_counts_for_roles("16w"), bar_kind_counts_for_roles("16w_fa4")) == (BAR_KIND_COUNTS, BAR_KIND_COUNTS_XFA4)


validate_config()
