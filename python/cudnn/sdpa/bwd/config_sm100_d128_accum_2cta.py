# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped THD d128 accumulation geometry on the cta_group::2 datapath (one cga2 PAIR per 256 kv rows).

Layout-only record for ``kernels/sm100/bprop_d128_accum_grouped_2cta_f16.py``: every byte offset, TMEM column and
register count derives from the geometry fields below and is re-derived by :func:`validate_config` at import.  The
barrier table (``BAR_LAYOUT``) is FILLED BY the barrier audit; this module only reserves its bytes.

Pair form (per CTA, KV-stationary: CTA ``rank`` owns kv rows ``[128 * rank, +128)`` of the pair's 256-row block and
both CTAs walk the identical q range of ONE q head):

* ``S^T = K Q^T`` and ``dP^T = V dO^T``: collective M = 256, A = own K / V (full d), B N-split = Q / dO rows
  ``[64 * rank, +64)`` x 128 d (layout 1: today's two SW128 boxes [64 x 64]).
* ``dV += P^T dO`` and ``dK += dS^T Q``: collective M = 256, TS A operand from the CTA's OWN TMEM image, B N-split
  along d = dO / Q [128 q x 64 d] (layout 2: one SW128 box, MN-major -- the second Q / dO load).
* ``dQ = dS K``: collective M = 128 (64 q rows per CTA), K-dim = the pair's 256 kv rows; A = dS [64 q x 256 kv] in
  SMEM (own kv half by transposed STS, the peer's half by one DSMEM bulk copy), B N-split along d = K [256 kv x 64 d]
  (the second K layout).  Its D image is the half-row atom: row m -> lane ``m % 64 + 64 * (n // 64)``, 64 columns.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- MMA geometry (per CTA; the collective M is M * CTA_MMA for the kv-stationary GEMMs) ---
    M: int = 128  # kv rows per CTA
    N: int = 128  # q rows per tile
    D: int = 128  # d_qk == d_v
    K_HW: int = 16  # bf16 / fp16 MMA K step on SM10x (one 32-B chunk; 32 is silently wrong)
    BPE: int = 2
    BPE_ACC: int = 4
    CTA_MMA: int = 2  # cta_group::2
    CGA_M: int = 2  # cluster (2, 1, 1)
    CGA_N: int = 1
    HEAD_BOX: int = 64  # bf16 elements per SW128 atom row = TMA box inner extent for Q / K / V / dO
    DQ_BOX: int = 32  # fp32 elements per SW128 atom row = TMA reduce-add box inner extent
    # --- ring depths ---
    Q1_STAGES: int = 2  # S^T lookahead keeps today's two Q stages
    # Q2 single-stage pays K2 (+32 KiB); 2 stages = 14 KiB over the cap. dK(t) frees it early.
    Q2_STAGES: int = 1
    DO_STAGES: int = 1
    KV_STAGES: int = 1
    SCHED_STAGES: int = 2
    # --- warps / registers (16 warps; the follower's MMA warp is quiet) ---
    THREADS: int = 512
    SOFTMAX_WARPS: int = 8
    EPILOGUE_WARPS: int = 4
    SERVICE_WARPS: int = 4  # MMA 12, TMA-load 13, TMA-store 14, scheduler 15
    SOFTMAX_REGS: int = 144
    EPILOGUE_REGS: int = 144
    SERVICE_REGS: int = 80
    REGFILE: int = 65536
    # --- TMEM ---
    TMEM_COLS: int = 512  # one tcgen05.alloc.cta_group::2 of TMEM_COLS on BOTH CTAs; the same constant deallocates
    # --- scaffold (declared budget: mbarriers + TMEM handle + CLC payload; the audit's rows must fit it) ---
    SCAFFOLD_BYTES: int = 1024
    TMEM_HANDLE_BYTES: int = 16
    SCHED_PAYLOAD_BYTES: int = 64
    # --- descriptor / swizzle facts ---
    # layout implies DESC_VERSION 0 (roots < 256 KiB); 1 inherited from the baseline, unmeasured.
    DESC_VERSION: int = 1
    SWIZZLE_BYTES: int = 128
    SWIZZLE_LAYOUT: int = 2
    SOFTMAX_BARRIER: int = 1  # named barrier id; barrier 0 is the CTA sync

    # --- derived: pair / split extents ---
    @property
    def PAIR_KV_ROWS(self):
        return self.M * self.CTA_MMA  # 256: the scheduler's and THD's unit

    @property
    def Q_HALF(self):
        return self.N // self.CTA_MMA  # 64 q rows per CTA (B N-split of S^T / dP^T; A M-split of dQ)

    @property
    def D_HALF(self):
        return self.D // self.CTA_MMA  # 64 d per CTA (B N-split of dV / dK / dQ)

    @property
    def COLLECTIVE_M_KV(self):
        return self.M * self.CTA_MMA  # 256 for S^T, dP^T, dV, dK

    @property
    def COLLECTIVE_M_Q(self):
        return self.N  # 128 for dQ (64 rows per CTA, the half-row D image)

    # --- derived: slab bytes (per stage) ---
    @property
    def Q1_BYTES(self):
        return self.Q_HALF * self.D * self.BPE  # 16 KiB: [64 q x 128 d], two SW128 boxes [64 x HEAD_BOX]

    @property
    def Q2_BYTES(self):
        return self.N * self.D_HALF * self.BPE  # 16 KiB: [128 q x 64 d], one SW128 box, MN-major B

    @property
    def DO1_BYTES(self):
        return self.Q1_BYTES

    @property
    def DO2_BYTES(self):
        return self.Q2_BYTES

    @property
    def K1_BYTES(self):
        return self.M * self.D * self.BPE  # 32 KiB: own [128 kv x 128 d], A of S^T

    @property
    def K2_BYTES(self):
        return self.PAIR_KV_ROWS * self.D_HALF * self.BPE  # 32 KiB: [256 kv x 64 d], B of dQ, one box of 256 rows

    @property
    def V_BYTES(self):
        return self.K1_BYTES

    @property
    def DS_BYTES(self):
        return self.Q_HALF * self.PAIR_KV_ROWS * self.BPE  # 32 KiB: [64 q x 256 kv] K-major, 4 K-chunks of 8 KiB

    @property
    def DS_HALF_BYTES(self):
        return self.DS_BYTES // self.CTA_MMA  # 16 KiB: one CTA's 128 kv columns of the dQ A operand

    @property
    def XCHG_BYTES(self):
        return self.DS_HALF_BYTES  # 16 KiB staging of the half the PEER needs: dS[peer's 64 q x own 128 kv]

    @property
    def DQ_SLICES(self):
        return self.D // self.DQ_BOX  # 4 reduce-add slices per tile

    @property
    def DQ_SLICE_BYTES(self):
        return self.Q_HALF * self.DQ_BOX * self.BPE_ACC  # 8 KiB: fp32 [64 q x 32 d]

    @property
    def DQ_STAGING_BYTES(self):
        return self.DQ_SLICES * self.DQ_SLICE_BYTES  # 32 KiB: half of today's

    @property
    def DVK_STORE_BYTES(self):
        return self.M * self.D * self.BPE  # 32 KiB each: native bf16 dV / dK partial of the CTA's 128 kv rows

    @property
    def DVK_STORE_HALF_BYTES(self):
        return self.DVK_STORE_BYTES // 2  # one [128 kv x HEAD_BOX] store box

    @property
    def STATS_BYTES(self):
        return self.N * self.BPE_ACC  # 512 B: LSE for all 128 q of the tile (the D image spans N = 128 on both CTAs)

    @property
    def DELTA_BYTES(self):
        return self.STATS_BYTES

    # --- derived: byte offsets in DECLARATION order (every slab 1024-B aligned; STATS + DELTA share one 1 KiB) ---
    @property
    def Q1(self):
        return 0

    @property
    def Q2(self):
        return self.Q1 + self.Q1_STAGES * self.Q1_BYTES

    @property
    def DO1(self):
        return self.Q2 + self.Q2_STAGES * self.Q2_BYTES

    @property
    def DO2(self):
        return self.DO1 + self.DO_STAGES * self.DO1_BYTES

    @property
    def K1(self):
        return self.DO2 + self.DO_STAGES * self.DO2_BYTES

    @property
    def K2(self):
        return self.K1 + self.KV_STAGES * self.K1_BYTES

    @property
    def V(self):
        return self.K2 + self.KV_STAGES * self.K2_BYTES

    @property
    def DS(self):
        return self.V + self.KV_STAGES * self.V_BYTES

    @property
    def XCHG(self):
        return self.DS + self.DS_BYTES

    @property
    def ALIAS(self):
        return self.DS  # the time-shared region [DS, STATS): dS A operand + exchange staging | dQ staging | dK store

    @property
    def ALIAS_BYTES(self):
        return self.DS_BYTES + self.XCHG_BYTES  # 48 KiB

    @property
    def STATS(self):
        return self.XCHG + self.XCHG_BYTES

    @property
    def DELTA(self):
        return self.STATS + self.STATS_BYTES

    @property
    def SLABS_END(self):
        return self.DELTA + self.DELTA_BYTES

    @property
    def BARS(self):
        return self.SLABS_END  # mbarrier block [BARS, TMEM_HANDLE); BAR_LAYOUT FILLED BY the barrier audit

    @property
    def SMEM_BYTES(self):
        return self.SLABS_END + self.SCAFFOLD_BYTES

    @property
    def SCHED_PAYLOAD(self):
        return self.SMEM_BYTES - self.SCHED_PAYLOAD_BYTES

    @property
    def TMEM_HANDLE(self):
        return self.SCHED_PAYLOAD - self.TMEM_HANDLE_BYTES

    @property
    def BAR_BLOCK_BYTES(self):
        # scaffold 1024 B = bars [230400, 231344) + TMEM handle + CLC payload; ~42 slots fit (118 max).
        return self.TMEM_HANDLE - self.BARS

    # --- derived: rank-dependent alias geometry (CTA rank = position in the pair) ---
    def ds_own(self, rank):
        """Own kv half of the dQ A operand: K-chunks [2 * rank, 2 * rank + 1] (kv-global K order on both CTAs)."""
        return self.DS + rank * self.DS_HALF_BYTES

    def ds_recv(self, rank):
        """The peer's kv half, landed by the peer's bulk copy; on the PEER's timeline, never reused locally."""
        return self.DS + (1 - rank) * self.DS_HALF_BYTES

    def dq_slice(self, rank, index):
        # dQ fp32 staging aliases DS_OWN + XCHG only; the RECV half is on the peer's timeline.
        pieces = (self.ds_own(rank), self.XCHG)
        per_piece = self.DS_HALF_BYTES // self.DQ_SLICE_BYTES
        return pieces[index // per_piece] + (index % per_piece) * self.DQ_SLICE_BYTES

    def dk_store(self, rank, half):
        # dK store staging aliases DS_OWN+XCHG after the last dq_empty; dvk_empty gates the next dS.
        return (self.ds_own(rank), self.XCHG)[half]

    def dv_store(self, half):
        # dV store staging aliases K2 (dead after the block's last dQ MMA); k2_empty waits the drain.
        return self.K2 + half * self.DVK_STORE_HALF_BYTES

    # --- derived: TMEM column map (fp32 columns; packed bf16 = 2 per column) ---
    @property
    def PACK_COLS(self):
        return self.N * self.BPE // self.BPE_ACC  # 64 columns of packed P^T / dS^T (128 q)

    @property
    def S_COL(self):
        return 0

    @property
    def S_COLS(self):
        return self.N  # S^T [128 kv lanes x 128 q]

    @property
    def P_COL(self):
        return self.S_COL  # packed in place, PACK_COLS wide

    @property
    def DP_COL(self):
        return self.S_COL + self.S_COLS

    @property
    def DP_COLS(self):
        return self.N

    @property
    def DS_COL(self):
        return self.DP_COL  # packed in place, PACK_COLS wide

    @property
    def DQ_COLS(self):
        return self.Q_HALF * self.D * self.BPE_ACC // (self.M * self.BPE_ACC)  # 64: the half-row image

    @property
    def DQ_COL(self):
        # dQ (M=128 image, 64 cols) lands at 192..256 = dP's dead fp32 half; dp_empty covers its readout.
        return self.DP_COL + self.PACK_COLS

    @property
    def DV_COL(self):
        return self.DP_COL + self.DP_COLS

    @property
    def DV_COLS(self):
        return self.D

    @property
    def DK_COL(self):
        return self.DV_COL + self.DV_COLS

    @property
    def DK_COLS(self):
        return self.D

    @property
    def TMEM_USED(self):
        return self.DK_COL + self.DK_COLS

    @property
    def DQ_LANE_SPLIT(self):
        return self.Q_HALF  # row m of dQ -> lane m % 64 + 64 * (d // D_HALF)

    # --- derived: softmax column ownership (two 128-lane groups, interleaved quarters) ---
    @property
    def SOFTMAX_GROUPS(self):
        return self.SOFTMAX_WARPS * 32 // self.M  # 2 groups of 128 lanes (one kv row per lane)

    @property
    def GROUP_QUARTER(self):
        # softmax group g owns q quarters {g, g+2}: packed P/dS are contiguous, no cross-group barrier.
        return self.N // (2 * self.SOFTMAX_GROUPS)  # 32 q columns per quarter

    def group_quarters(self, group):
        """fp32 q-column starts owned by softmax group ``group``; each packs into ``PACK_BASE + GROUP_QUARTER * group``."""
        return (self.GROUP_QUARTER * group, self.N // 2 + self.GROUP_QUARTER * group)

    def ts_k_perm(self, k):
        """K-step ``k`` of the TS A operand (16 q per step) in packed order -> the q row block the B operand walks."""
        steps_per_quarter = self.GROUP_QUARTER // self.K_HW  # 2
        group, rest = divmod(k, 2 * steps_per_quarter)
        half, step = divmod(rest, steps_per_quarter)
        return (self.group_quarters(group)[half] // self.K_HW) + step

    # --- derived: lane constants ---
    @property
    def SOFTMAX_LANES(self):
        return self.SOFTMAX_WARPS * 32

    @property
    def EPILOGUE_LANES(self):
        return self.EPILOGUE_WARPS * 32

    @property
    def STORE_LANES(self):
        return 32

    @property
    def ONE_LANE(self):
        return 1

    @property
    def PAIR_SOFTMAX_LANES(self):
        return self.SOFTMAX_LANES * self.CTA_MMA  # pair-wide fan-ins on the leader (p_ready / ds_ready / dp_empty forms)

    @property
    def SCHED_CONSUMER_WARPS(self):
        return self.SOFTMAX_WARPS + self.EPILOGUE_WARPS + self.SERVICE_WARPS - 1  # every warp but the scheduler, per CTA

    @property
    def SCHED_CREDIT_WARPS(self):
        return self.SCHED_CONSUMER_WARPS  # 15 per CTA: every warp but the scheduler warp credits the CLC slot it decoded

    @property
    def PAIR_SCHED_CREDIT_WARPS(self):
        return self.SCHED_CREDIT_WARPS * self.CTA_MMA  # 30: sched_empty lives on the LEADER (the sole CLC publisher), credits from both CTAs

    @property
    def PAIR_EPILOGUE_LANES(self):
        return self.EPILOGUE_LANES * self.CTA_MMA  # 256: dp_empty on the leader (both CTAs' epilogues read dQ out of dP's plane)

    @property
    def PAIR_STORE_LANES(self):
        return self.STORE_LANES * self.CTA_MMA  # 64: dvk_empty on the leader (both store warps drained their dV / dK stores)

    @property
    def DQ_SLICE_LANES(self):
        # 64: a dQ fp32 slice [Q_HALF q x DQ_BOX d] is written by the TWO epilogue warps holding ONE d half of the M=128 D image
        # (TMEM lane = q % 64 + 64 * (d // 64); warp w reads lanes [32 (w % 4), +32)) -- never the warpgroup's 128.
        return self.EPILOGUE_LANES // 2

    @property
    def TMEM_DEALLOC_ARRIVERS(self):
        return self.CTA_MMA  # 2: the local MMA warp's elected lane + the peer's (tcgen05.dealloc.cta_group::2 frees the PAIR's columns)

    @property
    def MCAST_PAIR_MASK(self):
        return (1 << self.CTA_MMA) - 1  # 0b11: tcgen05.commit .multicast::cluster to both CTAs of the pair (cluster (2,1,1): leader rank 0)

    @property
    def XCHG_BARRIER(self):
        return 2  # named barrier id (EPILOGUE_LANES threads): XCHG stores + fence_proxy -> bar -> ONE elected remote arm + bulk copy (P11 KEEP)

    # --- derived: expect_tx bytes.  cga2 TENSOR loads complete on the PAIR LEADER's mbar whoever issued them (P9): the leader
    # arms BOTH CTAs' boxes (x CTA_MMA); a plain cp.async.bulk DSMEM ship completes on the DESTINATION CTA's own mbar (x 1). ---
    @property
    def Q1_TX(self):
        return self.Q1_BYTES * self.CTA_MMA  # 32768 per stage: 2 boxes [64 x 64] per CTA, both CTAs

    @property
    def Q2_TX(self):
        return self.Q2_BYTES * self.CTA_MMA  # 32768

    @property
    def DO1_HALF_TX(self):
        return self.DO1_BYTES // 2 * self.CTA_MMA  # 16384 per half slot: one [64 x 64] box per CTA, both CTAs (dP^T issued in 2 k-halves)

    @property
    def DO2_TX(self):
        return self.DO2_BYTES * self.CTA_MMA  # 32768

    @property
    def K1_TX(self):
        return self.K1_BYTES * self.CTA_MMA  # 65536

    @property
    def K2_TX(self):
        return self.K2_BYTES * self.CTA_MMA  # 65536: one [256 kv x 64 d] box per CTA, both CTAs

    @property
    def V_TX(self):
        return self.V_BYTES * self.CTA_MMA  # 65536

    @property
    def XCHG_TX(self):
        return self.XCHG_BYTES  # 16384: the peer's cp.async.bulk of ITS XCHG into OUR recv half completes on OUR ds_recv_full (local)

    @property
    def CLC_TX(self):
        return 16  # the clusterlaunchcontrol.try_cancel response, multicast: 16 B on EACH CTA's sched_full copy

    @property
    def ENTRY_REGS(self):
        return self.REGFILE // 32 // (self.SOFTMAX_WARPS + self.EPILOGUE_WARPS + self.SERVICE_WARPS)

    # --- derived: THD / envelope ---
    @property
    def THD_UNIT_ROWS(self):
        return self.PAIR_KV_ROWS  # one CLC work item = one pair = 256 kv rows of one (sequence, kv head, q head)

    @property
    def INT32_ROUNDING_BOUND(self):
        return 2**31 - self.THD_UNIT_ROWS  # max(sq, skv) must stay below it (the grouped contract's 2**31 - 128 today)

    @property
    def Q_TILES_PER_PAIR(self):
        return self.PAIR_KV_ROWS // self.N  # 2: under causal the pair's first q tile is pair * Q_TILES_PER_PAIR (kv row 256 p <-> q row 256 p)

    # --- derived: the 's body constants (2026-10-26); every one is asserted below ---
    @property
    def QUARTERS_PER_GROUP(self):
        return self.N // (self.SOFTMAX_GROUPS * self.GROUP_QUARTER)  # 2: softmax group g owns quarters g and g + SOFTMAX_GROUPS

    @property
    def PACK_QUARTER_COLS(self):
        return self.GROUP_QUARTER * self.BPE // self.BPE_ACC  # 16: packed bf16 columns of one 32-q quarter

    @property
    def PACK_GROUP_COLS(self):
        return 2 * self.PACK_QUARTER_COLS  # 32: softmax group g packs its two quarters into [PACK_GROUP_COLS * g, +32)

    @property
    def K_STEPS_PER_BOX(self):
        return self.HEAD_BOX // self.K_HW  # 4 MMA k-steps per 64-wide d box (dP^T is issued per dO1 half box)

    @property
    def DIAG_2CTA_NAMES(self):
        # The code space of the pair body's DIAGNOSTIC probes (``diag_2cta_code`` = 1 + index; the body names DIAG_* from it).  Each probe
        # removes ONE issue, wait or transfer of the DEFAULT body and keeps every arrive and wait site it does not name, so the barrier table
        # is identical; the OUTPUT IS WRONG BY DESIGN (a timing ceiling, never a result; `--validate` refuses the arm by name; never shipped
        # armed; the default rendering is byte-identical, pinned by PTX md5).
        #   dq_reduce_add_off: the store warp skips the four fp32 dQ tma_reduce_add issues per tile (dQ stays zero)
        #   exchange_off:      the exchange-family ceiling: the XCHG ship is skipped and ds_recv_full is armed with 0 tx bytes, so every phase
        #                      completes at its own arm; EVERY wait kept (the recv half is never written: dQ wrong)
        #   stats_one_chunk:   each stats / delta quarter loads ONE LDS.128 chunk and reuses it for all eight (-448 of the 512 stats
        #                      wavefronts per tile per CTA; registers / compute / waits unchanged)
        #   sdq_sts_off:       the sDQ slice STS + its proxy fence skipped per round; the reduce-add reads stale staging (-256 wavefronts);
        #                      the ring protocol kept
        return ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off")

    @property
    def DV_FULL_LANES(self):
        # 160 = the epilogue's 128 (dV staged over K2) + the loader's 32 (its k2_empty observation of this item): the barrier audit
        # AUDIT 2026-10-26 fix A -- the store's k2_empty arrive follows its dv_full wait, so k2_empty's single waiter is never lapped
        # by a zero-trip item (P14-B back-edge), with no new row.
        return self.EPILOGUE_LANES + self.STORE_LANES

    @property
    def K2_HALF_BYTES(self):
        return self.M * self.D_HALF * self.BPE  # 16 KiB: one 128-row half of the [256 kv x 64 d] K2 slab (two 128-row TMA boxes)

    @property
    def SLICES_PER_D_HALF(self):
        return self.D_HALF // self.DQ_BOX  # 2 dQ slices per d half: the two epilogue warps holding that half publish them

    @property
    def D_HALVES(self):
        return self.D // self.D_HALF  # 2: the 64-wide d halves (dO1 half boxes, native dV / dK store boxes)

    @property
    def KV_HALVES(self):
        return self.PAIR_KV_ROWS // self.M  # 2: the 128-row halves of the pair's K2 slab (two TMA boxes)

    @property
    def DS_A_LBO(self):
        return self.Q_HALF * self.SWIZZLE_BYTES  # MN-major dQ A operand: one 64-q atom per CTA, the leading offset is never stepped

    @property
    def DQ_SLICE_ORDER(self):
        # the store warp's issue order: the XCHG-aliased slices first (the longer exchange path drains first), then the own-half slices
        per_piece = self.DS_HALF_BYTES // self.DQ_SLICE_BYTES
        return tuple(range(per_piece, self.DQ_SLICES)) + tuple(range(0, per_piece))

    # --- derived: the 2-CTA per-sequence tensor-map set (box rows per slot; the d box is HEAD_BOX, the dQ box DQ_BOX fp32) ---
    @property
    def MAP_BOX_ROWS(self):
        # Q1 (own 64 q rows), K, V, dO1 (own 64 q rows), dQ (own 64 q rows, fp32), dK, dV, Q2 (128 q), dO2 (128 q)
        return (self.Q_HALF, self.M, self.M, self.Q_HALF, self.Q_HALF, self.M, self.M, self.N, self.N)

    @property
    def MAPS_PER_SEQUENCE(self):
        return len(self.MAP_BOX_ROWS)  # 9 per-sequence tensor maps (the 1-CTA path has 7)


CFG = Config()


def resolve_diag_2cta(name=None):
    """The pair body's DIAGNOSTIC probe, typed: ``None`` -> no probe (THE default: the shipped body, PTX-identical); a name in
    ``CFG.DIAG_2CTA_NAMES`` -> that probe (output WRONG BY DESIGN); anything else is a ``ValueError`` BY NAME."""
    if name is None:
        return None
    if type(name) is not str or name not in CFG.DIAG_2CTA_NAMES:
        raise ValueError(f"diag_2cta must be None or one of {CFG.DIAG_2CTA_NAMES}; got {name!r}")
    return name


def diag_2cta_code(name=None):
    """The kernel's ``Constexpr[int]`` for a resolved probe name: 0 = no probe, else 1 + the name's index in ``CFG.DIAG_2CTA_NAMES``
    (the kernel compares the code against the config-derived module constants, never a literal)."""
    name = resolve_diag_2cta(name)
    return 0 if name is None else 1 + CFG.DIAG_2CTA_NAMES.index(name)


# (name, byte offset, active slots, init count, producer kind, scope) -- the barrier audit 2026-10-26 (SKELETON mode).
# The lane ledger (SUM(issuing lanes) == init per row, both CTAs) and the issue-order punch list are the kernel header's
# section 5 / 6; this tuple is the single source the kernel's ``Bars`` is built from (one MBarrier per row, same init on BOTH
# CTAs: a leader-waited barrier's follower copy is initialised with the same count and never armed, never waited).
# kinds: tma_leader = TMA_LOAD armed (pred=is_leader & elect) and waited by the pair LEADER only (P9: cga2 tensor loads
#        complete on the leader's mbar; the follower issues its half naming the leader's copy); tma = TMA_LOAD-kind
#        expect_tx(CLC_TX) armed by the LEADER's scheduler on BOTH copies (mapa, CTA scope) before the multicast try_cancel;
#        tma_dsmem = TMA_LOAD-kind armed LOCALLY by OUR epilogue lead warp (expect_tx XCHG_TX, pred=is_lead_warp & elect) and
#        completed by the PEER's cp.async.bulk complete_tx (the d512 role-split's _xfer_full form);
#        mma = tcgen05.commit .multicast::cluster (MCAST_PAIR_MASK) from
#        the leader MMA warp, pred=elect, every CTA waits its own copy; thread = local THREAD arrives (bare, every lane counts);
#        leader = Producer.LEADER relaxed cluster arrive on the LEADER's copy from BOTH CTAs (TMEM data ordered by tcgen05_wait,
#        or a count-only credit; P8); peer = ONE .release.cta remote arrive per warp on the LEADER's copy
#        (arrive_on_peer(release_cta=True) for lane-written SMEM the leader's MMA reads; read_tile_id_arrive_on_leader for the
#        scheduler credit).  Offsets: 16-B aligned arrays in [BARS, TMEM_HANDLE); 4-slot rings take 32 B.
BAR_LAYOUT = (
    # per q tile: operand rings (loader -> leader MMA; leader MMA commit -> both loaders)
    ("q1_full", 230400, 2, CFG.ONE_LANE, "tma_leader", "LEADER"),  # Q1_TX per stage
    ("q1_empty", 230416, 2, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after S^T(t) (same site as s_done)
    ("q2_full", 230432, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # Q2_TX
    ("q2_empty", 230448, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dK(t)
    ("do1_full", 230464, 2, CFG.ONE_LANE, "tma_leader", "LEADER"),  # 2 half slots x DO1_HALF_TX (dP^T in 2 k-halves), ONE do_state
    ("do1_empty", 230480, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dP^T(t) (same site as dp_done)
    ("do2_full", 230496, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # DO2_TX
    ("do2_empty", 230512, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit after dV(t)
    # per kv block
    ("k1_full", 230528, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # K1_TX
    ("k1_empty", 230544, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit at block end (after the loop)
    ("v_full", 230560, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # V_TX
    ("v_empty", 230576, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # commit at block end
    ("k2_full", 230592, 1, CFG.ONE_LANE, "tma_leader", "LEADER"),  # K2_TX
    ("k2_empty", 230608, 1, CFG.STORE_LANES, "thread", "LOCAL"),  # store warp after the dV store drain (dV staging aliases K2)
    # MMA -> compute hand-offs (leader commit, multicast; each CTA's consumers wait their own copy)
    ("s_done", 230624, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # softmax (both CTAs)
    ("dp_done", 230640, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # softmax (both CTAs)
    ("dq_done", 230656, 1, CFG.ONE_LANE, "mma", "LOCAL"),  # epilogue (both CTAs); + one empty commit per zero-trip item
    # compute -> MMA hand-offs (pair-wide fan-ins on the LEADER: the leader's MMA reads both CTAs' TMEM / SMEM)
    ("p_ready", 230672, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # 256 softmax lanes x 2 CTAs; waiter MMA(L) before dV
    ("ds_ready", 230688, 1, CFG.PAIR_SOFTMAX_LANES, "leader", "LEADER"),  # same lanes; waiter MMA(L) before dK
    ("ds_tmem", 230704, 1, CFG.SOFTMAX_LANES, "thread", "LOCAL"),  # softmax -> local epilogue: dS^T packed in OUR TMEM
    ("dp_empty", 230720, 1, CFG.PAIR_EPILOGUE_LANES, "leader", "LEADER"),  # 128 epilogue lanes x 2 after the dQ readout; MMA(L) pre-armed
    ("ds_smem", 230736, 1, CFG.EPILOGUE_LANES, "thread", "LOCAL"),  # own dS half in sDS: waiter = local MMA warp (L: dQ issue, F: relay)
    ("ds_recv_full", 230752, 1, CFG.ONE_LANE, "tma_dsmem", "LOCAL"),  # we arm XCHG_TX (lead warp elect); the PEER's bulk copy lands our recv half
    ("ds_pair_ready", 230768, 1, CFG.ONE_LANE, "peer", "LEADER"),  # follower MMA warp (ds_smem + ds_recv_full observed) -> leader
    # dQ staging ring (epilogue -> store -> epilogue), 4 slices
    ("dq_empty", 230784, 4, CFG.STORE_LANES, "thread", "LOCAL"),  # store warp: bootstrap x 4 per item + after each slice's read
    ("dq_full", 230816, 4, CFG.DQ_SLICE_LANES, "thread", "LOCAL"),  # 64 = the two warps of ONE d half per slice
    # native dV / dK stores
    ("dv_full", 230848, 1, CFG.DV_FULL_LANES, "thread", "LOCAL"),  # epilogue 128 + loader 32 after its k2_empty wait (AUDIT fix A)
    ("dk_full", 230864, 1, CFG.EPILOGUE_LANES, "thread", "LOCAL"),
    ("dvk_empty", 230880, 1, CFG.PAIR_STORE_LANES, "leader", "LEADER"),  # both store warps -> MMA(L): dV / dK TMEM reusable; pre-armed
    # scheduler ring (CLC try_cancel multicast from the LEADER's scheduler; credits land on the leader only)
    ("sched_full", 230896, 2, CFG.ONE_LANE, "tma", "LOCAL"),  # CLC_TX on each CTA's copy; every consumer warp waits its own
    ("sched_empty", 230912, 2, CFG.PAIR_SCHED_CREDIT_WARPS, "peer", "LEADER"),  # 15 warps x 2 CTAs, one release arrive per warp
    # teardown
    ("tmem_dealloc", 230928, 1, CFG.TMEM_DEALLOC_ARRIVERS, "thread", "LOCAL"),  # local elect arrive + the peer's arrive_on_peer
)
BAR_LAYOUT_END = 230944  # next free byte of the bar block (400 B of the 944 spare)
BAR_LAYOUT_STATUS = (
    "FILLED by the barrier audit 2026-10-26: 32 names / 43 slots, ledger + punch list in the kernel header sections 5-6; "
    "AUDIT 2026-10-26 23:10 fix A folded by the : dv_full init EPILOGUE_LANES -> DV_FULL_LANES (160), no new row"
)
BAR_KIND_COUNTS = {"tma_leader": 7, "mma": 9, "thread": 8, "leader": 4, "peer": 2, "tma_dsmem": 1, "tma": 1}

# Alias tenants of the time-shared bytes: (tenant, bytes it occupies, dead-when / the seam the inspector owns).
ALIASES = (
    ("dq_staging", "ds_own(rank) + XCHG", "dq_done (pair commit: the dQ MMA read sDS, the bulk copy read XCHG) -> dq_empty"),
    ("dk_store", "ds_own(rank) + XCHG", "last tile's dq_empty -> dvk_empty before the next block's first dS write"),
    ("dv_store", "K2", "last tile's dq_done -> k2_empty before the next block's K2 load"),
    ("ds_recv(rank)", "DS, the peer-rank half", "written ONLY by the peer's bulk copy; its next write waits the peer's dq_done(t)"),
)


def validate_config():
    """Fixed source geometry, derived byte chain, TMEM map, register pool; CPU only (raises at import)."""
    c = CFG
    assert (c.M, c.N, c.D, c.K_HW, c.BPE, c.BPE_ACC) == (128, 128, 128, 16, 2, 4)
    assert (c.CTA_MMA, c.CGA_M, c.CGA_N) == (2, 2, 1)
    assert (c.PAIR_KV_ROWS, c.Q_HALF, c.D_HALF, c.COLLECTIVE_M_KV, c.COLLECTIVE_M_Q) == (256, 64, 64, 256, 128)
    assert (c.HEAD_BOX * c.BPE, c.DQ_BOX * c.BPE_ACC) == (c.SWIZZLE_BYTES, c.SWIZZLE_BYTES)  # every box row is one SW128 atom
    assert c.D_HALF == c.HEAD_BOX  # a d-split half is exactly one atom wide: layout-2 slabs are single boxes
    assert (c.Q1_STAGES, c.Q2_STAGES, c.DO_STAGES, c.KV_STAGES, c.SCHED_STAGES) == (2, 1, 1, 1, 2)
    # bytes
    assert (c.Q1_BYTES, c.Q2_BYTES, c.DO1_BYTES, c.DO2_BYTES) == (16384,) * 4
    assert (c.K1_BYTES, c.K2_BYTES, c.V_BYTES, c.DS_BYTES) == (32768,) * 4
    assert (c.DS_HALF_BYTES, c.XCHG_BYTES, c.DQ_SLICES, c.DQ_SLICE_BYTES, c.DQ_STAGING_BYTES) == (16384, 16384, 4, 8192, 32768)
    assert (c.DVK_STORE_BYTES, c.DVK_STORE_HALF_BYTES, c.STATS_BYTES, c.DELTA_BYTES) == (32768, 16384, 512, 512)
    # The user's claim ("Q + dQ staging at today's bytes pays K2") does NOT close as stated: today's dQ staging is a
    # tenant of a 64 KiB ALIAS region whose size is max(dS 32, dQ 64, dV/dK 64).  What closes: Q 48 (Q2 one stage, -16),
    # K +32, ALIAS 48 (dQ staging 32 + dS exchange, dV/dK stores moved to K2 / the dS bytes, -16) = today's 225 KiB.
    assert c.Q1_STAGES * c.Q1_BYTES + c.Q2_STAGES * c.Q2_BYTES == 49152
    assert c.DO_STAGES * (c.DO1_BYTES + c.DO2_BYTES) == 32768  # dO bytes == today's dO
    today = {"Q": 65536, "DO": 32768, "K": 32768, "V": 32768, "ALIAS": 65536, "STATS": 1024}
    new = {
        "Q": c.Q1_STAGES * c.Q1_BYTES + c.Q2_STAGES * c.Q2_BYTES,
        "DO": c.DO_STAGES * (c.DO1_BYTES + c.DO2_BYTES),
        "K": c.KV_STAGES * (c.K1_BYTES + c.K2_BYTES),
        "V": c.KV_STAGES * c.V_BYTES,
        "ALIAS": c.ALIAS_BYTES,
        "STATS": c.STATS_BYTES + c.DELTA_BYTES,
    }
    assert sum(new.values()) == sum(today.values()) == c.SLABS_END == 230400
    # the declaration-order chain, every slab 1024-B aligned (STATS + DELTA share one 1 KiB line)
    order = (c.Q1, c.Q2, c.DO1, c.DO2, c.K1, c.K2, c.V, c.DS, c.XCHG, c.STATS, c.DELTA, c.SLABS_END)
    assert order == (0, 32768, 49152, 65536, 81920, 114688, 147456, 180224, 212992, 229376, 229888, 230400)
    assert all(off % 1024 == 0 for off in order[:-2]) and c.DELTA % 512 == 0
    assert (c.ALIAS, c.ALIAS_BYTES) == (c.DS, 49152)
    assert c.DQ_STAGING_BYTES <= c.ALIAS_BYTES - c.DS_HALF_BYTES  # the staging never touches the recv half
    assert c.DVK_STORE_BYTES <= c.K2_BYTES and c.DVK_STORE_BYTES <= c.ALIAS_BYTES - c.DS_HALF_BYTES
    # scaffold
    assert c.SCAFFOLD_BYTES == 1024 and c.SMEM_BYTES == 231424 and c.SMEM_BYTES % 1024 == 0
    assert (c.BARS, c.TMEM_HANDLE, c.SCHED_PAYLOAD) == (230400, 231344, 231360)
    assert c.TMEM_HANDLE % 16 == 0 and c.SCHED_PAYLOAD % 16 == 0 and c.BAR_BLOCK_BYTES == 944
    assert c.SLABS_END + c.SCAFFOLD_BYTES == c.SMEM_BYTES  # usable = cap - scaffold; the launcher queries the cap
    # rank-dependent aliases: own / recv halves are disjoint, dQ slices and the dK halves avoid the recv half
    for rank in (0, 1):
        own, recv = c.ds_own(rank), c.ds_recv(rank)
        assert {own, recv} == {c.DS, c.DS + c.DS_HALF_BYTES} and own != recv
        slices = [c.dq_slice(rank, i) for i in range(c.DQ_SLICES)]
        assert len(set(slices)) == c.DQ_SLICES and all(s % 1024 == 0 for s in slices)
        for s in slices:
            assert not (recv <= s < recv + c.DS_HALF_BYTES)
            assert c.DS <= s and s + c.DQ_SLICE_BYTES <= c.STATS
        assert {c.dk_store(rank, 0), c.dk_store(rank, 1)} == {own, c.XCHG}
    assert (c.dv_store(0), c.dv_store(1)) == (c.K2, c.K2 + 16384)
    # TMEM
    assert (c.S_COL, c.S_COLS, c.P_COL, c.PACK_COLS) == (0, 128, 0, 64)
    assert (c.DP_COL, c.DP_COLS, c.DS_COL, c.DQ_COL, c.DQ_COLS) == (128, 128, 128, 192, 64)
    assert (c.DV_COL, c.DV_COLS, c.DK_COL, c.DK_COLS, c.TMEM_USED, c.TMEM_COLS) == (256, 128, 384, 128, 512, 512)
    assert c.DS_COL + c.PACK_COLS == c.DQ_COL and c.DQ_COL + c.DQ_COLS == c.DV_COL  # dQ fills exactly dP's dead half
    assert c.TMEM_USED <= c.TMEM_COLS and c.DQ_LANE_SPLIT == 64
    # softmax ownership: the two groups' quarters partition the 128 q columns; the packed order is a fixed permutation
    assert c.SOFTMAX_GROUPS == 2 and c.GROUP_QUARTER == 32
    owned = sorted(q for g in range(c.SOFTMAX_GROUPS) for start in c.group_quarters(g) for q in range(start, start + c.GROUP_QUARTER))
    assert owned == list(range(c.N))
    perm = [c.ts_k_perm(k) for k in range(c.N // c.K_HW)]
    assert sorted(perm) == list(range(c.N // c.K_HW)) and perm == [0, 1, 4, 5, 2, 3, 6, 7]
    # registers: SUM(role regs x warps) == entry x warps, every count % 8 == 0, 24 <= regs <= 256
    assert (c.THREADS, c.SOFTMAX_WARPS + c.EPILOGUE_WARPS + c.SERVICE_WARPS) == (512, 16)
    assert (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS, c.ENTRY_REGS) == (144, 144, 80, 128)
    assert all(r % 8 == 0 and 24 <= r <= 256 for r in (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS))
    assert (c.SOFTMAX_WARPS * c.SOFTMAX_REGS + c.EPILOGUE_WARPS * c.EPILOGUE_REGS + c.SERVICE_WARPS * c.SERVICE_REGS) * 32 == c.REGFILE
    assert (c.SOFTMAX_LANES, c.EPILOGUE_LANES, c.STORE_LANES, c.ONE_LANE, c.PAIR_SOFTMAX_LANES, c.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 512, 15)
    # descriptors / THD
    assert (c.DESC_VERSION, c.SWIZZLE_BYTES, c.SWIZZLE_LAYOUT, c.SOFTMAX_BARRIER) == (1, 128, 2, 1)
    assert (c.THD_UNIT_ROWS, c.INT32_ROUNDING_BOUND, c.Q_TILES_PER_PAIR) == (256, 2**31 - 256, 2)
    # the 's body constants (2026-10-26): packed columns, k-steps per box, the K2 half, the dQ A operand's LBO, the slice order
    assert (c.PACK_QUARTER_COLS, c.PACK_GROUP_COLS, c.K_STEPS_PER_BOX, c.K2_HALF_BYTES, c.DS_A_LBO) == (16, 32, 4, 16384, 8192)
    assert c.PACK_GROUP_COLS * c.SOFTMAX_GROUPS == c.PACK_COLS and c.K_STEPS_PER_BOX * (c.D // c.HEAD_BOX) == c.D // c.K_HW
    assert (c.SLICES_PER_D_HALF, c.D_HALVES, c.KV_HALVES) == (2, 2, 2) and c.SLICES_PER_D_HALF * c.D_HALVES == c.DQ_SLICES
    assert c.QUARTERS_PER_GROUP == 2 and c.QUARTERS_PER_GROUP * c.SOFTMAX_GROUPS * c.GROUP_QUARTER == c.N
    assert c.K2_HALF_BYTES * c.CTA_MMA == c.K2_BYTES and c.DQ_SLICE_ORDER == (2, 3, 0, 1) and sorted(c.DQ_SLICE_ORDER) == list(range(c.DQ_SLICES))
    assert c.dq_slice(0, c.DQ_SLICE_ORDER[0]) == c.XCHG and c.dq_slice(1, c.DQ_SLICE_ORDER[2]) == c.ds_own(1)  # XCHG slices first, own-half slices last
    assert c.MAPS_PER_SEQUENCE == 9 and c.MAP_BOX_ROWS == (64, 128, 128, 64, 64, 128, 128, 128, 128)
    # lane constants (P3: every init count is a NAMED constant derived here; the derivation is the comment at the property)
    assert (c.SCHED_CREDIT_WARPS, c.PAIR_SCHED_CREDIT_WARPS) == (15, 30)
    assert (c.PAIR_EPILOGUE_LANES, c.PAIR_STORE_LANES, c.DQ_SLICE_LANES, c.TMEM_DEALLOC_ARRIVERS) == (256, 64, 64, 2)
    assert c.DV_FULL_LANES == c.EPILOGUE_LANES + c.STORE_LANES == 160  # AUDIT fix A: the loader's k2_empty observation is in dv_full's sum
    # the register split (the entry formula and the setmaxnreg directions): 144 / 144 / 80 on 16 warps
    s_regs, e_regs, v_regs = c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS
    assert (c.SOFTMAX_WARPS * s_regs + c.EPILOGUE_WARPS * e_regs + c.SERVICE_WARPS * v_regs) * 32 == c.REGFILE
    assert all(r % 8 == 0 and 24 <= r <= 256 for r in (s_regs, e_regs, v_regs))
    assert s_regs >= c.ENTRY_REGS and e_regs >= c.ENTRY_REGS and v_regs <= c.ENTRY_REGS  # INCREASE / INCREASE / DECREASE
    assert c.DIAG_2CTA_NAMES == ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off") and len(set(c.DIAG_2CTA_NAMES)) == 4
    assert diag_2cta_code(None) == 0 and [diag_2cta_code(n) for n in c.DIAG_2CTA_NAMES] == [1, 2, 3, 4]  # distinct, non-zero codes
    assert c.DQ_SLICE_LANES == c.Q_HALF  # one lane per q row of ONE d half: the M=128 D image's lane split
    assert (c.MCAST_PAIR_MASK, c.XCHG_BARRIER) == (0b11, 2) and c.XCHG_BARRIER not in (0, c.SOFTMAX_BARRIER)
    assert (c.Q1_TX, c.Q2_TX, c.DO1_HALF_TX, c.DO2_TX) == (32768, 32768, 16384, 32768)
    assert (c.K1_TX, c.K2_TX, c.V_TX, c.XCHG_TX, c.CLC_TX) == (65536, 65536, 65536, 16384, 16)
    assert 2 * c.DO1_HALF_TX == c.DO1_BYTES * c.CTA_MMA  # the two half slots together are the whole dO1 stage of both CTAs
    # BAR_LAYOUT: 32 names / 43 slots, 16-B aligned, disjoint, inside [BARS, TMEM_HANDLE); counts by kind; scope by kind
    assert BAR_LAYOUT_STATUS.startswith("FILLED") and len(BAR_LAYOUT) == 32 and sum(row[2] for row in BAR_LAYOUT) == 43
    expected_counts = dict(
        k2_empty=c.STORE_LANES,
        p_ready=c.PAIR_SOFTMAX_LANES,
        ds_ready=c.PAIR_SOFTMAX_LANES,
        ds_tmem=c.SOFTMAX_LANES,
        dp_empty=c.PAIR_EPILOGUE_LANES,
        ds_smem=c.EPILOGUE_LANES,
        ds_pair_ready=c.ONE_LANE,
        dq_empty=c.STORE_LANES,
        dq_full=c.DQ_SLICE_LANES,
        dv_full=c.DV_FULL_LANES,
        dk_full=c.EPILOGUE_LANES,
        dvk_empty=c.PAIR_STORE_LANES,
        sched_empty=c.PAIR_SCHED_CREDIT_WARPS,
        tmem_dealloc=c.TMEM_DEALLOC_ARRIVERS,
    )
    kind_scope = {"tma_leader": "LEADER", "tma": "LOCAL", "tma_dsmem": "LOCAL", "mma": "LOCAL", "thread": "LOCAL", "leader": "LEADER"}
    kind_scope["peer"] = "LEADER"
    addresses, names, kinds_seen = set(), set(), {}
    prev_end = c.BARS
    for name, offset, slots, count, kind, scope in BAR_LAYOUT:
        assert name not in names and slots in (1, 2, 4) and offset % 16 == 0 and offset == prev_end
        names.add(name)
        prev_end = offset + (32 if slots == 4 else 16)
        assert kind in kind_scope and scope == kind_scope[kind]
        kinds_seen[kind] = kinds_seen.get(kind, 0) + 1
        if kind in ("tma_leader", "tma", "tma_dsmem", "mma"):
            assert count == c.ONE_LANE, name  # one elected arm / one commit per phase (a commit lands ONE arrive per target CTA)
        else:
            assert count == expected_counts[name], name
        for i in range(slots):
            address = offset + i * 8
            assert c.BARS <= address and address + 8 <= c.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert prev_end == BAR_LAYOUT_END == 230944 <= c.TMEM_HANDLE and kinds_seen == BAR_KIND_COUNTS
    assert {n for n, *_ in BAR_LAYOUT if n.endswith("_full") and n not in ("sched_full", "ds_recv_full", "dq_full", "dv_full", "dk_full")} == {
        "q1_full",
        "q2_full",
        "do1_full",
        "do2_full",
        "k1_full",
        "v_full",
        "k2_full",
    }  # the seven cga2 tensor-load barriers: LEADER-armed, LEADER-waited (P9)
    assert len(ALIASES) == 4


validate_config()
