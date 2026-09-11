# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST SM100 D128 BF16 NVFP4 QAT dV/dS kernel.

Two CTAs collaborate on a 256-KV by 128-Q tile. QKV fake quantization
precedes this kernel; two GEMMs consume BF16 dS to produce dQ/dK.
Q/K/V intermediates use BSHD, caller dO/dV use native contiguous BHSD,
and dS uses [B,H_chunk,KV,Q]. No layout-copy adapters are needed.

P is quantized in groups of 16 KV lanes for dV only. dS retains FP32 P.
The local-scale floor is 2^-9 and division is round-to-nearest, matching
the public Triton QAT reference. dS folds the attention scale before its
BF16 store, so dQ/dK are numerically close, not bitwise identical.

Derived from NVIDIA VibeTile's two-CTA backward schedule.
This private module has one immutable configuration: BF16, noncausal QAT.
Only its plan-time shape/head-chunk descriptors vary.
"""

from typing import Callable, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass.experimental.primitives import vote_sync, VoteSync  # noqa: F401

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from cutlass._mlir.dialects import arith  # noqa: F401
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from dataclasses import dataclass
from typing import NamedTuple


def _fake_quant_p_scalar(p):
    """Positive P, group=16 KV lanes, PR778 NVFP4 fake-quant contract.

    Native RNE E4M3/E2M1 conversions, exact div.rn at both divisions.
    This is NOT the original FastVideo zero-scale/reciprocal contract.
    All 32 warp lanes must participate. No data-dependent divergent branch.
    """
    return inline_ptx(
        """{
        .reg .f32 m, t, s, x, y;
        .reg .b16 sf8, lo, hi;
        .reg .b8 q4;
        .reg .b32 pair;
        .reg .pred small, all_small;
        // A warp-uniform shortcut proves BOTH 16-lane groups use the floor.
        // Skip all shuffles and scale conversions in the common small-P case.
        setp.le.f32 small, $1, 0f3c400000;
        vote.sync.all.pred all_small, small, -1;
        @all_small bra QAT_MIN_SCALE;
        mov.f32 m, $1;
        shfl.sync.bfly.b32 t, m, 8, 31, -1;
        max.f32 m, m, t;
        shfl.sync.bfly.b32 t, m, 4, 31, -1;
        max.f32 m, m, t;
        shfl.sync.bfly.b32 t, m, 2, 31, -1;
        max.f32 m, m, t;
        shfl.sync.bfly.b32 t, m, 1, 31, -1;
        max.f32 m, m, t;
        div.rn.f32 s, m, 0f40c00000;
        max.f32 s, s, 0f3b000000;
        cvt.rn.satfinite.e4m3x2.f32 sf8, s, s;
        cvt.rn.f16x2.e4m3x2 pair, sf8;
        mov.b32 {lo, hi}, pair;
        cvt.f32.f16 s, lo;
        div.rn.f32 x, $1, s;
        bra QAT_CONVERT;
        QAT_MIN_SCALE:
        mov.f32 s, 0f3b000000;
        mul.f32 x, $1, 0f44000000;
        QAT_CONVERT:
        cvt.rn.satfinite.e2m1x2.f32 q4, x, x;
        cvt.rn.f16x2.e2m1x2 pair, q4;
        mov.b32 {lo, hi}, pair;
        cvt.f32.f16 y, lo;
        mul.f32 $0, y, s;
        }""",
        write_only_types=[cutlass.Float32],
        read_only_args=[p],
    )


def _fake_quant_p_pair(p0, p1):
    """Two query registers; each keeps its own independent 16-KV-lane scale.

    Packs native conversions and amortizes the warp-uniform small-P proof.
    It never shares scales across the two query positions.
    """
    reductions = "\n".join(f"shfl.sync.bfly.b32 t, m{j}, {offset}, 31, -1; max.f32 m{j}, m{j}, t;" for offset in (8, 4, 2, 1) for j in (0, 1))
    return inline_ptx(
        """{
        .reg .f32 m0,m1,t,s0,s1,x0,x1,y0,y1;
        .reg .b16 sf8,lo,hi;
        .reg .b8 q4;
        .reg .b32 pair;
        .reg .pred small,all_small;
        max.f32 t,$2,$3;
        setp.le.f32 small,t,0f3c400000;
        vote.sync.all.pred all_small,small,-1;
        @all_small bra QAT_PAIR_MIN;
        mov.f32 m0,$2;
        mov.f32 m1,$3;
        """
        + reductions
        + """
        div.rn.f32 s0,m0,0f40c00000;
        div.rn.f32 s1,m1,0f40c00000;
        max.f32 s0,s0,0f3b000000;
        max.f32 s1,s1,0f3b000000;
        cvt.rn.satfinite.e4m3x2.f32 sf8,s1,s0;
        cvt.rn.f16x2.e4m3x2 pair,sf8;
        mov.b32 {lo,hi},pair;
        cvt.f32.f16 s0,lo;
        cvt.f32.f16 s1,hi;
        div.rn.f32 x0,$2,s0;
        div.rn.f32 x1,$3,s1;
        bra QAT_PAIR_CONVERT;
        QAT_PAIR_MIN:
        mov.f32 s0,0f3b000000;
        mov.f32 s1,0f3b000000;
        mul.f32 x0,$2,0f44000000;
        mul.f32 x1,$3,0f44000000;
        QAT_PAIR_CONVERT:
        cvt.rn.satfinite.e2m1x2.f32 q4,x1,x0;
        cvt.rn.f16x2.e2m1x2 pair,q4;
        mov.b32 {lo,hi},pair;
        cvt.f32.f16 y0,lo;
        cvt.f32.f16 y1,hi;
        mul.f32 $0,y0,s0;
        mul.f32 $1,y1,s1;
        }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[p0, p1],
    )


def _fake_quant_p_many(*values):
    """Independent query scales, one warp-uniform minimum-scale proof."""
    n = len(values)
    assert n % 2 == 0
    lines = [
        "{",
        ".reg .f32 " + ",".join(f"{a}{j}" for a in ("m", "s", "x", "y") for j in range(n)) + ";",
        ".reg .f32 t;",
        ".reg .b16 sf8,lo,hi;",
        ".reg .b8 q4;",
        ".reg .b32 pair;",
        ".reg .pred small,all_small;",
        f"max.f32 t,${n},${n+1};",
    ]
    lines += [f"max.f32 t,t,${n+j};" for j in range(2, n)]
    lines += ["setp.le.f32 small,t,0f3c400000;", "vote.sync.all.pred all_small,small,-1;", "@all_small bra QAT_QUAD_MIN;"]
    lines += [f"mov.f32 m{j},${j+n};" for j in range(n)]
    lines += [f"shfl.sync.bfly.b32 t,m{j},{off},31,-1; max.f32 m{j},m{j},t;" for off in (8, 4, 2, 1) for j in range(n)]
    lines += [f"div.rn.f32 s{j},m{j},0f40c00000; max.f32 s{j},s{j},0f3b000000;" for j in range(n)]
    for j in range(0, n, 2):
        lines += [
            f"cvt.rn.satfinite.e4m3x2.f32 sf8,s{j+1},s{j};",
            "cvt.rn.f16x2.e4m3x2 pair,sf8;",
            "mov.b32 {lo,hi},pair;",
            f"cvt.f32.f16 s{j},lo;",
            f"cvt.f32.f16 s{j+1},hi;",
        ]
    lines += [f"div.rn.f32 x{j},${j+n},s{j};" for j in range(n)]
    lines += ["bra QAT_QUAD_CONVERT;", "QAT_QUAD_MIN:"]
    lines += [f"mov.f32 s{j},0f3b000000; mul.f32 x{j},${j+n},0f44000000;" for j in range(n)]
    lines += ["QAT_QUAD_CONVERT:"]
    for j in range(0, n, 2):
        lines += [
            f"cvt.rn.satfinite.e2m1x2.f32 q4,x{j+1},x{j};",
            "cvt.rn.f16x2.e2m1x2 pair,q4;",
            "mov.b32 {lo,hi},pair;",
            f"cvt.f32.f16 y{j},lo;",
            f"cvt.f32.f16 y{j+1},hi;",
        ]
    lines += [f"mul.f32 ${j},y{j},s{j};" for j in range(n)] + ["}"]
    return inline_ptx("\n".join(lines), write_only_types=[cutlass.Float32] * n, read_only_args=list(values))


# ============================================================================
# Config — Qwen BPROP V2.  Hand-written (no env overrides) — every axis locked
# Future BF16/FP16 input and MXFP8 input variants will fork.
# ============================================================================


@dataclass(frozen=True)
class BpropQwenCfg:
    QAT_P: bool = True  # PR778 local-scale floor and single-rounded division
    # --- Tile shape ----------------------------------------------------------
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128  # SM100 D128 bring-up; QAT insertion follows baseline validation
    TILE_O: int = 128

    # --- Dtype ---------------------------------------------------------------
    # f16 fork: 0 = FP16, 1 = BF16 inputs (vs the FP8 kernel's 0 = FP8 E4M3).
    DTYPE_QKV: int = 1  # 0 = FP16, 1 = BF16
    DTYPE_O: int = 2  # 2 = BF16 dV/dK output (sg-1 epilogue)
    BPE: int = 2
    BPE_O: int = 2

    # --- Cluster ------------------------------------------------------------
    # dV-ONLY: a SINGLE cga2 sub-group (one pair, CTAs 0,1) does EVERYTHING —
    # the 3 collective matmuls (Q·K, dO·V, dV=P·dO), softmax/dSoftmax, the dS
    # GMEM-workspace store, and the dV BF16 epilogue.  No role split, no sg-1,
    # no cross-sg DSMEM scatter.  The 2 CTAs collaborate on every collective
    # MMA (M = TILE_M·CTA_MMA = 256 kv across the pair).
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    SPLIT_PIPELINE: int = 1
    V2_PIPELINE: int = 1

    # --- Per-tensor swizzle (all 128 B at d=256 FP8) -------------------------
    Q_SWZ_BYTES: int = 128
    dO_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    # sg-1 P / dS SMEM rings (FP8, filled per q-iter by sg-0's DSMEM bulk_copy).
    P_SWZ_BYTES: int = 128
    dS_SWZ_BYTES: int = 128
    # dV / dK output staging swizzle (BF16 epilogue → TMA STG).
    dV_SWZ_BYTES: int = 128
    dK_SWZ_BYTES: int = 128

    # --- BMM K-axis chunking (Rubin F16/BF16 path: TILE_K_HW = 16) -----------
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    # --- Pipeline depths -----------------------------------------------------
    # f16 (BPE=2) doubles every operand buffer.  To fit the 327 KiB oversized
    # cap (~320 KiB): Q + dO (BMM1 dP) rings drop 3->2, dS drops to a single XFER
    # stage.  sdO_dv (BMM2 dV B ring) is DOUBLE-buffered (its 2nd stage aliases
    # the K back-half SMEM freed by the BMM1 K-split — see the module docstring),
    # so the TMA-LDG can prefetch dO_dv[i+1] at ZERO net SMEM.
    STAGES_Q: int = 2
    STAGES_dO: int = 2  # sdO  — BMM1 dP B-operand ring
    STAGES_dO_DV: int = 2  # sdO_dv — BMM2 dV B-operand ring (stage-1 = K alias)
    STAGES_KV: int = 1  # K, V loaded ONCE per K-block

    # TMEM ring depth: S-acc and dP-acc are SINGLE-buffer.  Cross-q-iter
    # overlap comes from the Q·K[i+1] lookahead in the MMA stream, not a
    # parity ring.  The compute-warp `tmem_load(S)->arrive(s_acc_empty)`
    # handshake frees [0..127] before Q·K[i+1] writes it; the [128..255]
    # dP/fp8_P slot runs a token ring (mb_dp_full -> mb_p_ready).
    STAGES_TMEM_S: int = 1
    # f16 P: stored back INTO the S_acc region [P_OFF=32 .. 95] (NOT a separate
    # ring — an f16 P stage is 64 cols, so a 2-stage ring wouldn't fit and the
    # [512..575] cols are reused for the K-half-in-TMEM instead).  Single buffer:
    # the natural matmul order (Q·K -> dO·V -> P·dO) + in-order MMA pipeline
    # makes P reuse safe with NO p_empty (Q·K[i+1] is issued after P·dO[i] reads
    # P[i], so it can't clobber P early).  group-0 writes P cols [32..63],
    # group-1 writes [64..95] — each inside its OWN S-read q-half, so no cross-wg
    # sync between the S tmem_load and the P tmem_store.
    STAGES_TMEM_P: int = 1

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2

    # --- Warp specialization -------------------------------------------------
    # 8 COMPUTE warps = 2 softmax warpgroups of 4.  lane=kv (1 lane/kv-row
    # within a wg, 128 kv-rows/wg).  Each wg owns a q-HALF: wg0 → q[0:64],
    # wg1 → q[64:128] → 64 q-elems/lane (spill cap, per the design).
    SOFTMAX_WARPGROUPS: int = 2
    SOFTMAX_WG_WARPS: int = 4
    CORRECTION_WARPS: int = 0

    # --- Register budget -----------------------------------------------------
    # 8 softmax @ 232 + 4 service @ 40 = 2016 <= 2048 (Rubin 65536/32 per-warp
    # budget across the 12 launched warps).  Each softmax warp only holds a
    # 64-wide q-slice (vs 128 in the role-split kernel) so 232 has ample slack.
    # HW equality constraint: MMA == TMALDG == TMASTG == SCHEDULER.
    SOFTMAX_REGS: int = 232
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    # --- Mask / sink (compile-time; mirror the forward prefill knobs) --------
    # MASK_FLAGS bitmask = MASK_PADDED|MASK_CAUSAL|MASK_SWA (tile_dsl.mask).
    # In BPROP the lane axis is kv and the q-axis is the inner loop, so the
    # mask is the TRANSPOSE of the forward: a fixed kv-block bounds WHICH
    # q-tiles attend (causal: q >= kv; SWA: kv <= q <= kv+W) and the per-cell
    # mask zeroes P on masked (kv,q) cells.  P=0 ⇒ dV/dK/dS/dQ inherit the
    # mask for free.  HAS_SINK is a NO-OP in this kernel (dQ/dK/dV are already
    # sink-correct from the sink-aware LSE the forward wrote); it only gates
    # the standalone dSink reduction the driver runs.
    MASK_FLAGS: int = 0
    SWA_WINDOW: int = 0
    CAUSAL_BOTTOM_RIGHT: int = 0  # align causal diagonal to bottom-right (k <= q + SKV-SQ)
    HAS_SINK: int = 0  # informational only (no main-kernel effect)

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = 0  # 0/1/2 = natural-3D / lpt / lpt_l2 (flat 1-D grid)

    # --- Warp layout (per CTA — single sub-group) ----------------------------
    #   0..3   softmax wg0 — softmax+dSoftmax for q[0:64]
    #   4..7   softmax wg1 — softmax+dSoftmax for q[64:128]
    #   8      MMA leader  — Q·K (S), V·dO (dP), P·dO (dV) in the lookahead order
    #   9      TMALDG      — Q, dO (per q_iter) + K, V (one-shot per K-block)
    #   10     TMASTG      — dS SMEM -> GMEM workspace store + lse/do_dot prefetch
    #   11     scheduler   — try_cancel.multicast::cluster::all
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_WG_LANES: int = 128  # 4 warps * 32 lanes (one wg)
    SOFTMAX_LANES: int = 256  # 8 warps * 32 lanes (both wgs)

    N_BMM2_CHUNKS: int = 2  # TILE_N / 64 — matches FP8 TILE_K_HW=64
    BMM2_CHUNK_SIZE: int = 64


CFG = BpropQwenCfg()

# --- Invariants -------------------------------------------------------------
assert CFG.DTYPE_QKV in (0, 1), "bprop_qwen f16: FP16 (0) / BF16 (1) inputs only"
assert CFG.DTYPE_O in (1, 2), "bprop_qwen: dV/dK grad output is BF16 (2) or FP16 (1)"
assert CFG.TILE_K == 128 and CFG.TILE_O == 128, "SM100 seed: d_qk = d_v = 128"
assert CFG.TILE_M == 128 and CFG.TILE_N == 128
assert CFG.CGA_M == 2 and CFG.CGA_N == 1 and CFG.CTA_MMA == 2, "bprop_qwen dV-only: single cga2 sub-group (CGA_M=2, CTA_MMA=2)"
assert CFG.SPLIT_PIPELINE == 1
assert CFG.V2_PIPELINE == 1
assert CFG.STAGES_Q == 2 and CFG.STAGES_dO == 2, "f16: Q / dO 2-deep ring (BPE=2 SMEM budget)"
assert CFG.STAGES_dO_DV == 2, "f16: sdO_dv 2-deep (stage-1 aliases the BMM1 K-split freed K back-half)"
assert CFG.STAGES_TMEM_P == 1, "f16: P single-buffer in the S_acc region (no separate P ring)"
assert CFG.STAGES_KV == 1, "K, V loaded once per K-block"
assert CFG.STAGES_TMEM_S == 1, "dV-only: S / dP single-buffer (lookahead, not parity)"
assert CFG.SOFTMAX_WARPGROUPS == 2 and CFG.CORRECTION_WARPS == 0, "dV-only: 8 compute warps (2 wg x 4), no correction wg"

# HW equality: MMA == TMALDG == TMASTG == SCHEDULER register counts.
assert CFG.MMA_REGS == CFG.TMALDG_REGS == CFG.TMASTG_REGS == CFG.SCHEDULER_REGS, "MMA / TMALDG / TMASTG / SCHEDULER regs must match (HW equality constraint)"

# Register budget (Rubin): sum of (per-warp-regs) across the LAUNCHED warps
# must be <= 65536/32 = 2048.  dV-only launches 12 warps: 8 softmax + 4 service.
_REG_TOTAL = (
    CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS * CFG.SOFTMAX_REGS
    + CFG.CORRECTION_WARPS * CFG.CORRECTION_REGS
    + CFG.MMA_REGS
    + CFG.TMALDG_REGS
    + CFG.TMASTG_REGS
    + CFG.SCHEDULER_REGS
)
assert _REG_TOTAL <= 2048, f"bprop_qwen dV-only: register budget {_REG_TOTAL} > 2048 (65536/32)"
for _r in (CFG.MMA_REGS, CFG.SOFTMAX_REGS, CFG.OTHER_REGS):
    assert _r % 8 == 0, f"reg counts must be %8: {_r}"
    assert 24 <= _r <= 256, f"reg count out of range [24..256]: {_r}"


# ============================================================================
# Imports from tile_dsl — same surface as the prefill kernels.
# ============================================================================

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    MBarrier,
    Producer,
    Scope,
    # `wait` (free fn) — still used for sched.mb_* (Sched not in Bars).
    wait,
    arrive_expect_tx,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    read_tile_id_arrive,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.mask import (
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat, vec_slice  # noqa: F401
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts, mma_ts_step  # noqa: F401
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma  # noqa: F401
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc  # noqa: F401
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile  # noqa: F401

# === Dtype dispatch ========================================================

if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
else:
    raise NotImplementedError(f"bprop_sdpa_d256_f16: DTYPE_QKV={CFG.DTYPE_QKV} not supported (FP16=0 / BF16=1)")


# dV / dK grad output dtype (BPE_O = 2 for both → no SMEM/alias resize on flip).
def _out_storage_dtype(dtype_o: int):
    if dtype_o == 2:
        return cutlass.BFloat16
    if dtype_o == 1:
        return cutlass.Float16
    raise NotImplementedError(f"bprop_sdpa_d256_fp8: DTYPE_O={dtype_o} not supported (BF16=2 / FP16=1)")


OUT_STORAGE_DTYPE = _out_storage_dtype(CFG.DTYPE_O)


# === Per-CTA SMEM element counts ===========================================
#
# sg-0-only geometry:
#   Q  — A-operand of BMM1 S  = K · Q^T  (N-split on q-axis, _M_PER_CTA q-rows).
#   dO — B-operand of BMM1 dP = V · dO^T (same N-split as Q).
#   K  — A-operand of BMM1 S  (M-split, full TILE_M kv-rows per CTA).
#   V  — A-operand of BMM1 dP (M-split, full TILE_M kv-rows per CTA).
#
# All operands FP8.  Per-CTA bytes:
#   Q  ring (3): 3 × 64 q × 256 d_qk × 1B = 48 KiB
#   dO ring (3): 3 × 64 q × 256 d_v  × 1B = 48 KiB
#   K  (1)     : 128 kv × 256 d_qk × 1B   = 32 KiB
#   V  (1)     : 128 kv × 256 d_v  × 1B   = 32 KiB
#   Total                                 = 160 KiB + mbars/scratch.

_M_PER_CTA = CFG.TILE_N // CFG.CTA_MMA  # 64 — per-CTA q-rows (Q + dO N-split)

qBufferElems = _M_PER_CTA * CFG.TILE_K  # 64 q-rows * 256 d_qk
dOBufferElems = _M_PER_CTA * CFG.TILE_O  # 64 q-rows * 256 d_v
kBufferElems = CFG.TILE_M * CFG.TILE_K  # 128 kv-rows * 256 d_qk
vBufferElems = CFG.TILE_M * CFG.TILE_O  # 128 kv-rows * 256 d_v

# sg-1 per-CTA SMEM buffer element counts.
#   sP, sdS — FP8 BMM2 A-operand (TILE_M kv × TILE_N q-cols, both CTAs in
#             the cga2 pair hold the FULL kv tile, NOT M-split).
#   sdV, sdK — BF16 epilogue staging (TILE_M kv × TILE_O / TILE_K d-cols).
pBufferElems = CFG.TILE_M * CFG.TILE_N  # 128 kv * 128 q-cols
dSBufferElems = CFG.TILE_M * CFG.TILE_N  # 128 kv * 128 q-cols
dVBufferElems = CFG.TILE_M * CFG.TILE_O  # 128 kv * 256 d_v
dKBufferElems = CFG.TILE_M * CFG.TILE_K  # 128 kv * 256 d_qk

# TMA byte transactions (cga2 expect_tx routes both peers' bytes to leader's mbar).
qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
dOTmaTransactionBytes = dOBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA
# sg-1 dV / dK TMA store byte transactions (per-CTA, NOT cga2 multicast —
# each sg-1 CTA stores its own half of the kv tile).
dVTmaTransactionBytes = dVBufferElems * CFG.BPE_O
dKTmaTransactionBytes = dKBufferElems * CFG.BPE_O

# === BMM1 K-split derived geometry ==========================================
# The subtile-major K / Q SMEM is 4 swizzle subtiles of [outer × TMA_QK_GRANU
# (=64 d_qk)].  d_qk[0:128] = subtiles 0,1 (front, contiguous); d_qk[128:256] =
# subtiles 2,3 (back, contiguous).  Offsets are in ALLOC-dtype elements (BPE=2
# for both f16/bf16).  K back-half is UTCCP'd into TMEM[RSVD]; BMM1 S splits
# into mma_ss(front, SMEM-A) + mma_ts(back, TMEM-A).
K_SPLIT_TILE_K = CFG.TILE_K // 2  # 128 — half the d_qk contraction per MMA
K_BACK_OFF_ELEMS = kBufferElems // 2  # 16384 — sK subtiles 2,3 (d_qk[128:256])
Q_BACK_OFF_ELEMS = qBufferElems // 2  # 8192  — sQ subtiles 2,3 (B-operand back)


# === TMEM layout (sg-0 + sg-1) — sg-0: 512 logical / 576 HW;
#                                 sg-1: 512 logical / 576 HW (separate alloc).
# Each cga2 pair has its OWN tcgen05.alloc.cta_group::2 allocation,
# independent across sg-0 and sg-1 (different SMs).
# ============================================================================


@dataclass(frozen=True)
class KernelTmemLayout:
    """dV-only single-subgroup TMEM column map (512 allocated, 416 used).

      cols [  0..127]  S_acc      (FP32, Q·K^T result; SINGLE buffer).  After the
                                    softmax reads S it writes f16 P back INTO
                                    this region at [P_OFF .. P_OFF+P_COLS):
                                      wg0 -> [32..63], wg1 -> [64..95]
                                    Each wg's P-write stays inside its OWN S-read
                                    q-half ([0..63]/[64..127] cols), so no cross-wg
                                    sync between the S tmem_load and the P tmem_st.
      cols [128..255]  dP         (FP32 dP=dO·V^T; read by dSoftmax; SINGLE buf)
      cols [256..383]  dV_acc     (FP32, P·dO result; persistent over q_iters)
      cols [384..415]  K_back     (UTCCP'd K d_qk[64:128] back-half — BMM1 S
                                    splits mma_ss(K_front SMEM)+mma_ts(K_back
                                    TMEM); frees the 16 KiB K back-half SMEM to
                                    alias the sdO_dv 2nd stage).

    S / dP single-buffer (STAGES_TMEM_S=1); P single-buffer in S_acc
    (STAGES_TMEM_P=1) — natural matmul order Q·K -> dO·V -> P·dO + in-order MMA
    make P reuse safe with NO p_empty AND NO s_acc_empty (P·dO[i] reads P before
    Q·K[i+1] overwrites S_acc).  P A-operand width = TILE_N(q)·BPE/4 = 64 cols.
    BMM1 contracts the full K with one SMEM half and one TMEM half.
    """

    TOTAL_COLS: int = 512

    S_OFF: int = 0
    S_COLS: int = 128  # q-cols (TILE_N)

    dP_OFF: int = 128
    dP_COLS: int = 128  # q-cols (TILE_N)

    dV_OFF: int = 256
    dV_COLS: int = 128  # d_v (TILE_O)

    # f16 P lives inside S_acc: base col 32, total 64 cols ([32..95]); each wg
    # owns P_COLS//SOFTMAX_WARPGROUPS = 32 cols (wg0 [32..63], wg1 [64..95]).
    P_OFF: int = 32  # col offset within the S_acc [0..127] region
    P_COLS: int = 64  # full P width = TILE_N*BPE/4 = 128*2/4

    # [384..415] holds the UTCCP'd K d_qk[64:128] back-half (BMM1 K-split).
    RSVD_OFF: int = 384
    RSVD_COLS: int = 32


LAYOUT = KernelTmemLayout()

# Audit: S(128)+dP(128)+dV(128)+RSVD(32) = 416 logical, 512 allocated.
# P reuses the S_acc region [32..95] (not a separate column block).
assert LAYOUT.S_OFF == 0 and LAYOUT.dP_OFF == LAYOUT.S_OFF + LAYOUT.S_COLS
assert LAYOUT.dV_OFF == LAYOUT.dP_OFF + LAYOUT.dP_COLS
assert LAYOUT.RSVD_OFF == LAYOUT.dV_OFF + LAYOUT.dV_COLS
assert LAYOUT.RSVD_OFF + LAYOUT.RSVD_COLS <= LAYOUT.TOTAL_COLS
# P (single buffer) lives inside S_acc: [P_OFF .. P_OFF+P_COLS) must fit [0..128).
assert LAYOUT.P_COLS == (CFG.TILE_N * CFG.BPE) // 4, "P A-operand width = TILE_N*BPE/4 cols"
assert LAYOUT.P_OFF + LAYOUT.P_COLS <= LAYOUT.S_COLS, "f16 P must fit inside the S_acc [0..S_COLS) region"
# wg1's P-half must begin exactly at the S-read q-half boundary (S col = q-idx,
# boundary at TILE_N/WGS = 64), so wg0 P ⊂ S-read[0..64) and wg1 P ⊂ [64..128).
assert (
    LAYOUT.P_OFF + LAYOUT.P_COLS // CFG.SOFTMAX_WARPGROUPS == CFG.TILE_N // CFG.SOFTMAX_WARPGROUPS
), "wg1 P-half must align to the softmax q-half boundary (col 64)"
assert LAYOUT.TOTAL_COLS == 512


# === SMEM swizzle enum mapping =============================================

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_dO = _SWZ_ENUM[CFG.dO_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
SMEM_LAYOUT_dS = _SWZ_ENUM[CFG.dS_SWZ_BYTES]
# dV epilogue stores BF16 via store_swizzled into a 128B-swizzled sdV (mirrors
# the forward d256 O epilogue): TILE_O BF16 = 4 swizzle sub-tiles of
# DV_D_BLOCK = dV_SWZ_BYTES/BPE_O = 64 d_v cols each.  The SmemTile + host TMA
# STG descriptor must use the SAME 128B swizzle (else TMA reads shuffled cells).
SMEM_LAYOUT_dV = _SWZ_ENUM[CFG.dV_SWZ_BYTES]  # 128B swizzle (matches store_swizzled)
SMEM_LAYOUT_dK = 0  # dK not computed in the dV-only kernel (unused)

# TMA iters per inner row (= inner_bytes / swz_bytes).
TMA_QK_ITERS = (CFG.TILE_K * CFG.BPE) // CFG.Q_SWZ_BYTES  # 2
TMA_VO_ITERS = (CFG.TILE_O * CFG.BPE) // CFG.dO_SWZ_BYTES  # 2
TMA_QK_GRANU_ELEMS = CFG.TILE_K // TMA_QK_ITERS  # 128
TMA_VO_GRANU_ELEMS = CFG.TILE_O // TMA_VO_ITERS  # 128

# dV output (BF16) TMA STG: TILE_O × BPE_O = 512 B/row, 128B-swizzled → 4 sub-tiles.
# Mirrors the forward d256 O epilogue: each swizzle sub-tile is DV_D_BLOCK =
# dV_SWZ_BYTES/BPE_O = 64 d_v cols (128 B), so TILE_O=256 → 4 sub-tiles of
# [TILE_M kv × 64 d_v].  Sub-tile slab stride = TILE_M*DV_D_BLOCK (whole slab,
# swizzled layout) — the epilogue store_swizzled writes block b at b*slab + kv*64.
DV_D_BLOCK = CFG.dV_SWZ_BYTES // CFG.BPE_O  # 64
TMA_DV_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.dV_SWZ_BYTES  # 4
TMA_DV_GRANU_ELEMS = DV_D_BLOCK  # 64
DV_BLOCK_SLAB = CFG.TILE_M * DV_D_BLOCK  # 8192 (TILE_M*64)
TMA_DK_ITERS = CFG.TILE_K // 128  # (dK unused)
TMA_DK_GRANU_ELEMS = 128

# sg-1 Q / dO BMM2 operand layout — different from sg-0 (see cga2-mma.md
# § "Collective tile and split axes").
#
#   sg-0 BMM1 dP = V · dO^T  → MMA "A · B^T" with A=V, B=dO.  B SMEM layout
#                              is (N=q-rows, K=d_v); N-split across cga2 →
#                              per-CTA (N_per_cta=TILE_N//CTA_MMA=64,
#                              K=TILE_O=256) = 64 q × 256 d_v.
#   sg-1 BMM2 dV = P  · dO   → MMA "A · B^T" with A=P, B=dO  AND BT=true.
#                              B SMEM physical layout is (K=q-rows, N=d_v)
#                              (matches GMEM layout); N-split across cga2
#                              → per-CTA (K=TILE_N=128, N_per_cta=TILE_O/2
#                              =128) = 128 q × 128 d_v.
#
# Same 16 KiB per stage on disk, completely different per-CTA SHAPE.
# Each subgroup needs its own TMA descriptor (different box_dims) — sg-0
# loads (64, 256), sg-1 loads (128, 128) into the same SMEM offset.
#
# sg-1 TMA params: each inner row = TILE_K//CTA_MMA bytes (128 B for FP8
# d_qk/2), exactly fills 1 swizzle subtile.
TMA_QK_SG1_ITERS = ((CFG.TILE_K // CFG.CTA_MMA) * CFG.BPE) // CFG.Q_SWZ_BYTES  # 1
TMA_VO_SG1_ITERS = ((CFG.TILE_O // CFG.CTA_MMA) * CFG.BPE) // CFG.dO_SWZ_BYTES  # 1
TMA_QK_SG1_GRANU_ELEMS = (CFG.TILE_K // CFG.CTA_MMA) // TMA_QK_SG1_ITERS  # 128
TMA_VO_SG1_GRANU_ELEMS = (CFG.TILE_O // CFG.CTA_MMA) // TMA_VO_SG1_ITERS  # 128

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# sg-0 dO mirrors Q layout (_M_PER_CTA q-rows × TILE_O d_v-cols per CTA).
LEADING_BYTE_OFFSET_dO = 0
STRIDE_BYTE_OFFSET_dO = 8 * CFG.dO_SWZ_BYTES

# sg-1 P / dS / dV / dK SMEM layout offsets.
LEADING_BYTE_OFFSET_P = 0
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES
LEADING_BYTE_OFFSET_dS = 0
STRIDE_BYTE_OFFSET_dS = 8 * CFG.dS_SWZ_BYTES
LEADING_BYTE_OFFSET_dV = 0
STRIDE_BYTE_OFFSET_dV = 8 * CFG.dV_SWZ_BYTES
LEADING_BYTE_OFFSET_dK = 0
STRIDE_BYTE_OFFSET_dK = 8 * CFG.dK_SWZ_BYTES

# sg-1 B-operand SmemDesc leading/stride.  Mirrors matmul.py:328-330
# (BT=true heuristic): when B_PC_COLS // CORE_MATRIX_ROWS (8) > 8 the
# leading byte offset is K × swz_bytes, otherwise 0.  Our sg-1 BMM2:
#   B = dO/Q with cga2 N-split on d-axis → B_PC_COLS = TILE_O/CTA_MMA
#   = 128 (or TILE_K/CTA_MMA = 128 for dK BMM).  128 // 8 = 16 > 8 →
#   leading = K * swz = TILE_N * 128 = 16384.
#   stride = 8 * swz_bytes = 1024.
LEADING_BYTE_OFFSET_Q_SG1 = CFG.TILE_N * CFG.Q_SWZ_BYTES
STRIDE_BYTE_OFFSET_Q_SG1 = 8 * CFG.Q_SWZ_BYTES
LEADING_BYTE_OFFSET_dO_SG1 = CFG.TILE_N * CFG.dO_SWZ_BYTES
STRIDE_BYTE_OFFSET_dO_SG1 = 8 * CFG.dO_SWZ_BYTES

CGA_SIZE = CFG.CGA_M * CFG.CGA_N  # 4

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2  # CTA_MMA == 2 hard-wired

# === dS SMEM ring (LOCAL — softmax store_swizzled -> TMASTG TMA-store) ======
# f16: single XFER stage to fit the 327 KiB cap.  The dSoftmax writes f16 dS
# [kv,q] into the ring slot; the TMASTG warp TMA-stores it to the GMEM dS
# workspace.  1 stage serializes dSoftmax[i+1] behind the TMASTG drain of [i].
# (The BMM1 K-split's freed 32 KiB goes to the sdO_dv 2nd stage, not dS — a dS
# 2nd stage would need a different buffer.)  No DSMEM scatter (single-sub-group).
XFER_STAGES = 1
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)  # 128 B swizzle
P_TMA_ITERS = (CFG.TILE_N * CFG.BPE) // CFG.P_SWZ_BYTES  # 1 (FP8, 128 B swz)
P_D_BLOCK = CFG.TILE_N // P_TMA_ITERS  # 128 cells / chunk
P_BLOCK_BYTES = CFG.TILE_M * P_D_BLOCK  # 16384 elems / chunk
# Total P-xfer payload bytes per slot = TILE_M kv × TILE_N q × BPE = 16 KiB
pXferBytes = CFG.TILE_M * CFG.TILE_N * CFG.BPE  # 16384

# === LSE / do_dot SMEM prefetch ring (sg-0 only) ===========================
# The sg-0 softmax/dSoftmax bodies read lse[q] (for P) and scaled_do_dot[q]
# (for dS) — both indexed by the TILE_N q-cols, IDENTICAL across all 128
# softmax lanes (each lane = a kv-row).  Loaded straight from GMEM that was
# 128x redundant and the dominant long-scoreboard stall (NCU: 85%).  Instead
# the otherwise-idle sg-0 TMASTG warp prefetches the TILE_N lse + TILE_N
# do_dot values into a STATS_STAGES-deep SMEM ring (LDG -> STS -> arrive); the
# softmax wg waits the ring and reads from SMEM (short-scoreboard, on-chip).
# 2 * TILE_N FP32 per slot; the ring aliases sg-0's FREE half of sExcl_raw
# (the sg-1-only sdK region) — zero net SMEM (see SharedStorage).
# Softmax q-half width: each of the 2 wgs owns TILE_N/SOFTMAX_WARPGROUPS = 64
# q-cols (lane=kv, 64 q-elems/lane → spill cap).
_SMX_CHUNK = CFG.TILE_N // CFG.SOFTMAX_WARPGROUPS  # 64

# LDTM (tcgen05.ld.32x32b.xN) per-instruction granularity for the compute-warp
# S/dP TMEM reads.  64 = one .x64 per 64-col q-half; 32/16 split into 2/4 finer
# loads.  Finer loads help only when the load latency is on the critical path;
# with the Q·K[i+1] lookahead in the MMA stream the latency is already hidden,
# so x16/x32 measured within-noise of x64 here — default 64.  (A/B: on the
# natural-order variant w/o lookahead, x16 was ~1.3% faster; see DV_ONLY_PLAN.md.)
_LDTM_NUM = 64
assert _SMX_CHUNK % _LDTM_NUM == 0, "_SMX_CHUNK must be a multiple of _LDTM_NUM"

STATS_STAGES = 2
STATS_LSE_OFF = 0  # FP32 elems within a slot: lse[0..TILE_N)
STATS_DOT_OFF = CFG.TILE_N  #                          do_dot[0..TILE_N)
STATS_SLOT_ELEMS = 2 * CFG.TILE_N  # FP32 elems per ring slot
# dSoftmax register-pressure chunk (EXPERIMENT, mirrors the V2-QK fix): the
# un-chunked dP→dS held chunk_P_0[128] + reg_dP[128] + dot_vec[128] + dS[128]
# live → spill.  Chunking reg_dP/dot_vec/dS into _DSOFT_CHUNK cols (keep
# chunk_P_0[128] full) caps it.  Must be a multiple of 64 and divide TILE_N.
_DSOFT_CHUNK = 64
# log2(e): the softmax body uses exp2, so P = exp(attn_scale·S − lse_nat)
# = exp2(attn_scale·log2e·S − lse_nat·log2e).  attn_scale·log2e rides in the
# `attn_scale_log2e` scalar; the lse·log2e factor is applied IN-KERNEL by the
# TMASTG prefetch warp (scales lse before STS), so the host passes natural-log lse.
_LOG2E = 1.4426950408889634

SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.SOFTMAX_WARPGROUPS


# === Mask plumbing (BPROP = the TRANSPOSE of the forward) ===================
# lane = kv-row, inner loop = q-tile.  A fixed kv-block [kvb, kvb+KVB) bounds
# WHICH q-tiles attend (`_q_loop_bounds` skips the rest); `_mask_p_chunk`
# zeroes P on masked (kv,q) cells of the boundary tiles.  P=0 ⇒ dV/dK/dS/dQ
# inherit the mask for free.  Every MASK_* arm is a Python const_expr → the
# dense (MASK_NONE) path emits NO IR (byte-identical SASS to pre-mask).
_KV_BLOCK_ROWS = CFG.TILE_M * CFG.CTA_MMA  # 256 kv-rows per cga2 pair


def _div_up(a, b: int):
    return (a + cutlass.Int32(b - 1)) // cutlass.Int32(b)


def _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv):
    """[q_lo_tile, q_hi_tile) — the q-tile range that attends this kv-block.

    Transpose of the forward `compute_kv_loop_bounds`:
      causal (k <= q + diag): a query attends a key iff q >= k - diag, so the
        block's earliest key kvb sets q_lo = (kvb - diag)//TILE_N; q_hi = n_q.
      SWA (q-W <= k <= q): q in [kvb, kvb+KVB-1+W] ⇒ q_lo = kvb//TILE_N and
        q_hi = ceil((kvb+KVB+W)/TILE_N).
      padded (k >= seq_kv_len): masks kv ROWS (per-lane) — q-range unchanged.
    `diag = SKV-SQ` only when CAUSAL_BOTTOM_RIGHT.  MASK_NONE → [0, n_q_tiles).
    Uniform across the cga2 pair (kv_block_base is the cluster's kv base)."""
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    q_lo = cutlass.Int32(0)
    q_hi = n_q_tiles
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
        diag = (seqlen_kv - seqlen_q) if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT) else cutlass.Int32(0)
        q_lo = cute.math.max(q_lo, (kv_block_base - diag) // cutlass.Int32(CFG.TILE_N))
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
        q_lo = cute.math.max(q_lo, kv_block_base // cutlass.Int32(CFG.TILE_N))
        q_hi = cute.math.min(q_hi, _div_up(kv_block_base + cutlass.Int32(_KV_BLOCK_ROWS + CFG.SWA_WINDOW), CFG.TILE_N))
    q_lo = cute.math.max(cutlass.Int32(0), q_lo)
    # Empty-kv-block guard (e.g. top-left causal with SKV>SQ: kv-blocks past the
    # last query attend NO q → q_hi<=q_lo → N=0).  An N=0 tile would hang the
    # UNCONDITIONAL MMA prologue on mb_q_full (the TMA q-loop made 0 loads), and
    # a DSL runtime `if` can't skip it (loop-carried PipelineState reassigns
    # don't propagate out of a runtime branch — the DSL-gotchas rule control-flow rule).
    # Instead clamp q_lo in-range and FORCE N>=1: the single forced q-tile is
    # fully masked by _mask_p_chunk → P=0 → dV/dS=0 (correct zeros, dV_acc
    # overwritten via accumulate=(q_iter>q_lo)=False), and every per-iter ring
    # stays balanced.  No-op for non-empty tiles (q_lo<n_q_tiles, q_hi>q_lo).
    q_lo = cute.math.min(q_lo, n_q_tiles - cutlass.Int32(1))
    q_hi = cute.math.max(q_hi, q_lo + cutlass.Int32(1))
    return q_lo, q_hi


def _mask_p_chunk(reg_P, kv_abs, q_col_base, seqlen_kv, causal_diag, N: int):
    """Zero P on masked (kv=lane, q=col) cells.

    kv_abs = per-lane absolute kv-row; q_col_base = absolute q of col 0; N cols.
    Mirrors `tile_dsl.mask.apply_mask_chunk` TRANSPOSED (row=kv, col=q):
      causal : kv_abs > q_abs (+diag for bottom-right)   (key past query)
      SWA    : kv_abs < q_abs - W                         (key left of window)
      padded : kv_abs >= seq_kv_len                       (per-lane pad row)
    MASK_NONE → returns reg_P unchanged (no IR)."""
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return reg_P
    zero = cutlass.Float32(0.0)
    elems = []
    for i in range(N):
        q_abs = q_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
            t = kv_abs >= seqlen_kv
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
            lim = (q_abs + causal_diag) if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT) else q_abs
            t = kv_abs > lim
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
            t = kv_abs < (q_abs - cutlass.Int32(CFG.SWA_WINDOW))
            masked = t if masked is None else (masked | t)
        val = cutlass.Float32(arith.select(masked.ir_value(), zero.ir_value(), reg_P[i].ir_value()))
        elems.append(val)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


# === Named arrival-count constants (P3) =====================================
ONE_LANE = CFG.ONE_LANE  # 1
ONE_WARP = CFG.ONE_WARP  # 32
SOFTMAX_LANES = CFG.SOFTMAX_LANES  # 128 = SOFTMAX_WG_WARPS * 32
MMA_COMMIT_ARRIVES = 1

# READ_TILE_ARRIVERS_TOT — per-CTA arrival count on `mb_read_tile_id`.
# Every persistent-loop warp on every CTA fires `read_tile_id_arrive(...,
# CGA_SIZE=4)` once per iter so the scheduler ring stays in lockstep across
# the cluster.  Each call distributes ONE arrive to each of the 4 peers'
# local mb_read_tile_id slots, so per-CTA arrivals == count of warps
# calling cluster-wide.
#
# dV-only single cga2 sub-group (CGA_SIZE=2).  Per-iter, warps that loop +
# fire read_tile_id_arrive, summed cluster-wide (each call multicasts 1 to
# every peer's local mbar):
#   leader CTA  : 8 softmax + MMA(leader) + TMALDG + TMASTG = 11
#   follower CTA: 8 softmax + TMALDG + TMASTG               = 10  (MMA-quiet no loop)
#   Cluster total                                           = 21
# (AUDIT) re-derive with barrier-inspector once the warp bodies pin the exact
# per-CTA call sites (does the follower TMASTG loop? does either fire from
# scheduler?).  Wrong count → scheduler-ring drift → hang at high cluster counts (P3).
READ_TILE_ARRIVERS_TOT = 21


# === Bars — barrier inventory ===============================================
#
class Bars(NamedTuple):
    """mbarrier inventory — dV-only single cga2 sub-group.

    NOTE: init counts marked (AUDIT) are provisional — re-derive with the
    barrier-inspector once the warp bodies pin the exact arrive call sites
    (P3).  No cross-sg scatter rings (single sub-group).

      mb_q_full[3]/mb_q_empty[3]       TMA Q     -> MMA(Q·K)     (q-split B)
      mb_do_full[3]/mb_do_empty[3]     TMA dO#1  -> MMA(dO·V)    (dP B, BT=f)
      mb_dodv_full[3]/mb_dodv_empty[3] TMA dO#2  -> MMA(S·dO)    (dV B, BT=t)
      mb_k_full[1]/mb_k_empty[1]       TMA K     -> MMA          (one-shot)
      mb_v_full[1]/mb_v_empty[1]       TMA V     -> MMA          (one-shot)
      mb_s_acc_full[1]                 MMA(Q·K)  -> softmax      ("S ready")
      mb_dp_full[1]                    MMA(dO·V) -> softmax      ("dP ready")
      mb_p_ready[1]                    softmax   -> MMA(P·dO)    ("P in S_acc[P_OFF]")
      (NO mb_s_acc_empty — natural order + in-order MMA cover the S_acc WAR)
      mb_stats_full[2]/mb_stats_empty[2]  TMASTG  -> softmax     (lse/do_dot ring)
      mb_ds_smem_full[X]/mb_ds_smem_empty[X] softmax -> TMASTG   (dS SMEM ->
                                                                  GMEM workspace)
      mb_dv_ready[1]/mb_dv_acc_empty[1]   MMA(dV) <-> epilogue   (dV TMEM, post-loop)
      mb_dv_stg_full[1]/mb_dv_stg_empty[1] epilogue <-> TMASTG   (dV BF16 STG)
      mb_tmem_dealloc[1]               softmax/epi -> MMA quiet  (P11)
    """

    mb_q_full: object
    mb_q_empty: object
    mb_do_full: object
    mb_do_empty: object
    mb_dodv_full: object
    mb_dodv_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object
    # BMM1 K-split alias seam: leader MMA commit_mma after UTCCP(K d_qk[128:256]
    # → TMEM) → TMA-LDG (K back-half SMEM dead → safe to clobber with the sdO_dv
    # stage-1 dO load).
    mb_k_utccp_done: object
    mb_s_acc_full: object
    mb_dp_full: object
    mb_dp_empty: object
    mb_p_ready: object
    mb_stats_full: object
    mb_stats_empty: object
    mb_ds_smem_full: object
    mb_ds_smem_empty: object
    mb_dv_ready: object
    mb_dv_acc_empty: object
    mb_dv_stg_full: object
    mb_dv_stg_empty: object
    mb_tmem_dealloc: object


def _make_bprop_bars(CFG):
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    # Every softmax lane (SOFTMAX_LANES = 256, both wgs) on EACH CTA arrives on
    # the leader's mbar → 256 * CTA_MMA = 512 DSMEM arrives on leader.
    SOFT_X_CTA_MMA = SOFTMAX_LANES * CFG.CTA_MMA  # 512

    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_do_full=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_empty=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dodv_full=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_dodv_empty=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # BMM1 K-split alias seam (leader commit_mma multicast → 1 arrive/CTA).
        mb_k_utccp_done=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.MMA_COMMIT),
        # S single-buffer (STAGES_TMEM_S=1).  NO mb_s_acc_empty in the f16 kernel
        # — P lives in S_acc and in-order MMA covers the WAR (see _mma_warp).
        mb_s_acc_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # dP single-buffer.
        mb_dp_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # dP slot freed by dSoftmax (after its tmem_load) -> gates dO·V[i+1]'s
        # reuse.  Pre-armed on the MMA side.
        mb_dp_empty=MBarrier(
            _alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER
        ),  # (AUDIT)
        # f16 P single buffer (in S_acc[P_OFF]): softmax -> MMA "P ready".
        mb_p_ready=MBarrier(
            _alloc(CFG.STAGES_TMEM_P), stages=CFG.STAGES_TMEM_P, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER
        ),  # (AUDIT)
        # lse/do_dot SMEM prefetch ring (TMASTG elect-1 STS -> softmax read).
        mb_stats_full=MBarrier(_alloc(STATS_STAGES), stages=STATS_STAGES, init_count=ONE_WARP, producer=Producer.THREAD),
        mb_stats_empty=MBarrier(_alloc(STATS_STAGES), stages=STATS_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),  # (AUDIT)
        # dS SMEM ring: softmax store_swizzled (all lanes) -> TMASTG TMA-store.
        mb_ds_smem_full=MBarrier(_alloc(XFER_STAGES), stages=XFER_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),  # (AUDIT)
        mb_ds_smem_empty=MBarrier(_alloc(XFER_STAGES), stages=XFER_STAGES, init_count=ONE_LANE, producer=Producer.THREAD),
        # dV epilogue (post-loop, one-shot): MMA(dV) -> epilogue (compute warps).
        mb_dv_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dv_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),  # (AUDIT)
        mb_dv_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),  # (AUDIT)
        mb_dv_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.THREAD),  # (AUDIT)
    )


# === Kernel entry ===========================================================


@cute.kernel
def _kernel(
    # GMEM TMA descriptors — loads
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q  (BMM1 S B, q-split box)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO (BMM1 dP B, dP-box q-split)
    tma_do_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dO (BMM2 dV B, dV-box d_v-split BT)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    # GMEM TMA descriptors — stores
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dV  -> out [B,H,S_kv,d_v] BF16
    tma_ds_desc: cutlass.GridConstant[tmap.TensorMap],  # dS  -> workspace [B,H,S_kv,S_q] FP8
    # GMEM scalar / vector inputs
    lse_tensor: cute.Tensor,  # [B, H_q, S_q]   FP32
    scaled_do_dot_tensor: cute.Tensor,  # [B, H_q, S_q]   FP32 = do_dot * attn_scale
    # Scalars
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_kv_blocks: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    dscale_dO: cutlass.Float32,
    dscale_V: cutlass.Float32,
    dscale_Q: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,  # = attn_scale_in * dscale_V * dscale_dO (C++ fold)
    head_base: cutlass.Int32,  # workspace-chunking: full-tensor head = grid head + head_base
    n_qh_grid: cutlass.Int32,  # grid head extent (== QH_CHUNK); LPT flat-grid decode
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- SharedStorage allocation (single sub-group, dV-only) --------------
    # K + V (one-shot, 64 KiB) + Q ring + 2 dO rings + dS ring + stats.  The
    # dV BF16 epilogue staging ALIASES the K+V backing (both dead post q-loop).

    # Q ring (FP8, 3-stage) — B operand of BMM1 S (q-split, 64 q × 256 d_qk/CTA).
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #1 (FP8, 3-stage) — B operand of BMM1 dP (BT=false, q-split,
    # 64 q × 256 d_v/CTA; leading=0, 2 swizzle subtiles).
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #2 (FP8, 3-stage) — B operand of BMM2 dV (BT=true, d_v-split,
    # 128 q × 128 d_v/CTA; leading=TILE_N*swz, 1 subtile).  SAME dO GMEM data,
    # DIFFERENT per-CTA box + swizzle layout, so it CANNOT share sdO_raw (the
    # Swz128B XOR maps cells to different bytes for the 256B-row vs 128B-row
    # interpretations).  Same 16 KiB/stage element count as sdO_raw.
    # sdO_dv backing + BMM1 K-split alias.  stage-0 of sdO_dv + K + V share ONE
    # backing laid out [sdOdv_s0 | K | V], so sdO_dv stage-1 ALIASES the
    # (post-UTCCP dead) K back-half (subtiles 2,3) at a COMPILE-TIME element
    # offset — no runtime pointer math.  Ring stride (_DODV_STAGE_ELEMS) =
    # dOBufferElems + K_BACK_OFF_ELEMS, so sdO_dv[1] == sK_back.  Net SMEM is
    # unchanged (the 2nd stage costs zero — it reuses the K back-half).  The dV
    # BF16 epilogue aliases the K+V region (sExcl) post-loop.
    _DODV_STAGE_ELEMS = dOBufferElems + K_BACK_OFF_ELEMS
    _sCombined_raw = cutlass.Array(STORAGE_DTYPE, dOBufferElems + kBufferElems + vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdOdv_raw = cutlass.Array(_sCombined_raw.data_ptr(), dOBufferElems, dtype=STORAGE_DTYPE)
    sExcl_raw = cutlass.Array((_sCombined_raw.subview(cutlass.Int32(dOBufferElems))).data_ptr(), kBufferElems + vBufferElems, dtype=STORAGE_DTYPE)
    sK_raw = cutlass.Array(sExcl_raw.data_ptr(), kBufferElems, dtype=STORAGE_DTYPE)
    # NB: ``cutlass.Array.subview(N)`` advances N ELEMENTS (matches SmemTile's
    # element-stride base arithmetic), so the K→V offset is kBufferElems ELEMENTS
    # — no ``* BPE``.  (The fp8 fork's ``kBufferElems * CFG.BPE`` was a no-op only
    # because BPE==1 there; at f16/bf16 BPE==2 it doubled the offset, placing
    # sV's tail past the 327 KiB cap → TMALDG OOB on the 3rd/4th V sub-tile.)
    sV_raw = cutlass.Array((sExcl_raw.subview(cutlass.Int32(kBufferElems))).data_ptr(), vBufferElems, dtype=STORAGE_DTYPE)
    # dV BF16 epilogue staging (aliases sExcl — K+V dead at epilogue).
    sdV_raw = cutlass.Array(sExcl_raw.data_ptr(), dVBufferElems, dtype=OUT_STORAGE_DTYPE)
    # lse/do_dot prefetch ring (FP32) — own small backing (K+V is fully live
    # through the q-loop, so no free corner to alias here).  ~2 KiB.
    sStats_raw = cutlass.Array(cutlass.Float32, STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # dS SMEM ring (FP8, XFER_STAGES-deep) — softmax writes fp8 dS[kv,q] here
    # (store_swizzled), TMASTG warp TMA-stores each slot to the GMEM workspace.
    # LOCAL (no DSMEM scatter).  Per CTA: XFER_STAGES × TILE_M × TILE_N FP8.
    sdS_raw = cutlass.Array(STORAGE_DTYPE, XFER_STAGES * dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    # --- SmemTile wrappers -------------------------------------------------
    sQ = SmemTile(
        desc_version=0,
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.STAGES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_Q,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_QK_GRANU_ELEMS,
    )
    # dO #1 — BMM1 dP B operand (BT=false, 64 q × 256 d_v, leading=0, 2 subtiles).
    sdO = SmemTile(
        desc_version=0,
        base=sdO_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_VO_GRANU_ELEMS,
    )
    # dO #2 — BMM2 dV B operand (BT=true, 128 q × 128 d_v, leading=TILE_N*swz,
    # 1 subtile).  Mirrors the fork-base sg-1 dO view, filled by a second TMA
    # load of the same dO GMEM.  2-stage ring: elems_per_stage is the alias
    # stride (dOBufferElems + K_BACK_OFF_ELEMS) so sdO_dv[1] lands on the
    # K-split-freed K back-half; the tile's actual geometry (leading/stride/TMA
    # params) stays dOBufferElems-shaped.
    sdO_dv = SmemTile(
        desc_version=0,
        base=sdOdv_raw,
        elems_per_stage=_DODV_STAGE_ELEMS,
        stages=CFG.STAGES_dO_DV,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO_SG1,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO_SG1,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_SG1_ITERS,
        tma_granu_elems=TMA_VO_SG1_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_SG1_GRANU_ELEMS,
    )
    sK = SmemTile(
        desc_version=0,
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_K,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
    )
    sV = SmemTile(
        desc_version=0,
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_VO_GRANU_ELEMS,
    )
    # dS — softmax store_swizzled target + TMASTG store source ([kv,q] FP8,
    # 128 B/row → 1 swizzle subtile).
    # f16 dS: TILE_N=128 q × BPE = 256 B/row, 128B-swizzled → P_TMA_ITERS=2
    # sub-tiles of P_D_BLOCK=64 q-cols each (slab stride TILE_M*P_D_BLOCK).
    sdS = SmemTile(
        desc_version=0,
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_dS,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dS,
        layout=SMEM_LAYOUT_dS,
        tma_loads_per_tile=P_TMA_ITERS,
        tma_granu_elems=P_D_BLOCK,
        tma_subtile_stride_elems=CFG.TILE_M * P_D_BLOCK,
    )
    # dV BF16 epilogue staging — 4 swizzle sub-tiles of (TILE_M kv × DV_D_BLOCK
    # d_v), 128B-swizzled (mirrors forward sO); aliases sExcl (K+V dead).  Full
    # dVBufferElems = 64 KiB BF16.  leading/stride = 0 (TMA-STG staging buffer,
    # not an MMA operand — swizzle + tma params fully describe it; matches sO).
    sdV = SmemTile(
        desc_version=0,
        base=sdV_raw,
        elems_per_stage=dVBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_dV,
        tma_loads_per_tile=TMA_DV_ITERS,
        tma_granu_elems=TMA_DV_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_DV_GRANU_ELEMS,
    )

    # --- mbarrier storage (barrier-inspector fills .init() sites) -----------
    # The Bars NamedTuple is allocated below — every field is a flat Int64
    # array sized to the per-mbar-array depth (STAGES_*).  TOTAL = ~40 mbars
    # at 8 B each = ~320 B of mbar storage.
    bars = _make_bprop_bars(CFG)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            # NATURAL: bidy/bidz = blockIdx.{y,z} (head, batch).  LPT/LPT_L2 flat
            # 1-D grid: blockIdx.{y,z}=0 (dead) → REPURPOSE these slots to carry
            # (n_qh_grid, n_batch) so _boot_tile / _decode_tile_payload can decode
            # the linear cluster id with NO extra Sched fields (see those helpers).
            "bidy_init": (n_qh_grid if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidy),
            "bidz_init": (n_batch if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidz),
        }
    )

    # --- cluster role identity (cga2; sg-0 only) ----------------------------
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    partner_cta_id = cta_id_x ^ cutlass.Int32(1)
    sg_id = cta_id_x // cutlass.Int32(CFG.CTA_MMA)  # 0 (always at CGA_M=2)
    is_sg0 = sg_id == cutlass.Int32(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # --- mbar init + cluster fence (P3 + P4) --------------------------------
    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(CFG.STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_dO):
                bars.mb_do_full[s].init()
                bars.mb_do_empty[s].init()
            # sdO_dv ring is single-buffer (STAGES_dO_DV) — separate init loop
            # so we don't index the 1-slot arrays out to STAGES_dO.
            for s in cutlass.range_constexpr(CFG.STAGES_dO_DV):
                bars.mb_dodv_full[s].init()
                bars.mb_dodv_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[s].init()
                bars.mb_v_full[s].init()
                bars.mb_k_empty[s].init()
                bars.mb_v_empty[s].init()
            # K-split alias seam (init = ONE_LANE: leader commit_mma multicast
            # fires 1 arrive per targeted CTA, regardless of issuer lane count).
            bars.mb_k_utccp_done.init()
            # P8: leader-waited empties (dp_empty / p_ready / dv_acc_empty);
            # init differs per-CTA (leader=SOFTMAX_LANES*CTA_MMA, follower=ONE).
            # NO s_acc_empty in the f16 kernel (in-order MMA covers the WAR).
            LEADER_INIT = cutlass.Int32(
                arith.select(
                    is_leader.ir_value(),
                    cutlass.Int32(SOFTMAX_LANES * CFG.CTA_MMA).ir_value(),
                    cutlass.Int32(ONE_LANE).ir_value(),
                )
            )
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_S):
                bars.mb_s_acc_full[p].init()
                bars.mb_dp_full[p].init()
                bars.mb_dp_empty[p].init(override_count=LEADER_INIT)
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_P):
                bars.mb_p_ready[p].init(override_count=LEADER_INIT)
            # dV epilogue handshakes.
            bars.mb_dv_ready.init()
            bars.mb_dv_acc_empty.init(override_count=LEADER_INIT)
            bars.mb_dv_stg_full.init()
            bars.mb_dv_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            # lse/do_dot prefetch ring (mb_stats_empty pre-armed via PipelineState).
            for p in cutlass.range_constexpr(STATS_STAGES):
                bars.mb_stats_full[p].init()
                bars.mb_stats_empty[p].init()
            # dS SMEM ring (mb_ds_smem_empty pre-armed via PipelineState).
            for p in cutlass.range_constexpr(XFER_STAGES):
                bars.mb_ds_smem_full[p].init()
                bars.mb_ds_smem_empty[p].init()

            # Scheduler ring: init on EVERY CTA (try_cancel multicast targets all).
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    # P4 init order: init -> fence -> within-CTA __syncthreads ->
    # cga_arrive/cga_wait.  Pre-armed PipelineState phases handle iter-0
    # waits (P5b); no explicit bootstrap arrives needed.
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    cga_arrive()
    cga_wait()

    # Pair-scoped MMA multicast mask (within the cga2 pair).
    mcast_mask = cutlass.Int32(3) << leader_cta_id
    # TMA multicast mask — cluster-scope (bit i = cluster CTA i).
    tma_mcast_mask = cutlass.Int16(1) << cta_id_x
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # --- warp dispatch (single cga2 sub-group) ------------------------------
    #   0..7   softmax (2 wg) ; 8 MMA ; 9 TMALDG ; 10 TMASTG ; 11 scheduler
    if warp_idx < cutlass.Int32(CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            warp_idx=warp_idx,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            sdS_raw=sdS_raw,
            sStats_raw=sStats_raw,
            sdV_raw=sdV_raw,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_kv_blocks=n_kv_blocks,
            n_qh=n_qh,
            n_batch=n_batch,
            attn_scale=attn_scale,
            attn_scale_log2e=attn_scale_log2e,
            dscale_dO=dscale_dO,
            dscale_V=dscale_V,
            dscale_Q=dscale_Q,
            attn_scale_for_dS=attn_scale_for_dS,
            cta_in_pair=cta_in_pair,
            leader_cta_id=leader_cta_id,
            partner_cta_id=partner_cta_id,
            cta_id_x=cta_id_x,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp(
                sQ=sQ,
                sdO=sdO,
                sdO_dv=sdO_dv,
                sK=sK,
                sV=sV,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                n_kv_blocks=n_kv_blocks,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
            )
        else:
            _mma_warp_quiet(tmem_ptr_i32, bars)

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp(
            tma_q_desc=tma_q_desc,
            tma_do_desc=tma_do_desc,
            tma_do_dv_desc=tma_do_dv_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sdO=sdO,
            sdO_dv=sdO_dv,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_kv_blocks=n_kv_blocks,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            head_base=head_base,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_ds_desc.get_ptr())
        # dS SMEM->workspace store + dV SMEM->GMEM store (stats prefetch moved
        # to the scheduler warp so it runs concurrently with these stores).
        _tmastg_warp(
            tma_dv_desc=tma_dv_desc,
            tma_ds_desc=tma_ds_desc,
            sdV=sdV,
            sdS=sdS,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_qh=n_qh,
            cta_in_pair=cta_in_pair,
            head_base=head_base,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # Scheduler FUSED with lse/do_dot stats prefetch (was on TMASTG).
        _scheduler_stats_warp(
            sched,
            is_cga_first_cta,
            bars=bars,
            sStats_raw=sStats_raw,
            lse_tensor=lse_tensor,
            do_dot_tensor=scaled_do_dot_tensor,
            attn_scale_in=attn_scale_for_dS / (dscale_V * dscale_dO),
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_base=head_base,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === Warp functions ========================================================


@cute.jit
def _decode_linear_bprop(linear, n_qh_grid, n_batch):
    """Flat 1-D (LPT/LPT_L2) tile decode: linear cluster id → (kv_super, head,
    batch).  kv_super is the OUTER axis so the heaviest causal kv-blocks (small
    kv_super, attended by the most queries) land in the first SM wave.
    `n_qh_grid` is the GRID head extent (== QH_CHUNK under head-chunking)."""
    hb = n_qh_grid * n_batch
    kv_super = linear // hb
    within = linear % hb
    head = within % n_qh_grid
    batch = within // n_qh_grid
    return kv_super, head, batch


@cute.jit
def _boot_tile(sched):
    """Decode the FIRST tile (kv_super, head, batch) from the launch blockIdx.
    natural 3-D grid : (bidx//CGA_M, bidy, bidz).
    LPT/LPT_L2 flat   : bidx is the linear cluster base; bidy_init/bidz_init
                        are REPURPOSED to carry (n_qh_grid, n_batch) — dead
                        under a (N,1,1) grid (see _kernel)."""
    linear = sched.bidx_init // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        return (cute.arch.make_warp_uniform(linear), cute.arch.make_warp_uniform(sched.bidy_init), cute.arch.make_warp_uniform(sched.bidz_init))
    kv, h, b = _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)
    return (cute.arch.make_warp_uniform(kv), cute.arch.make_warp_uniform(h), cute.arch.make_warp_uniform(b))


@cute.jit
def _decode_tile_payload(sched, sched_idx):
    """Decode (kv_super_idx, head_idx, batch_idx) from tile_id_smem[idx].

    try_cancel payload = [blockIdx.x, y, z, valid].  Cluster base bidx is t0.
      SCHED_NATURAL (3-D grid): t0 = kv_super*CGA_M, t1 = packed(head=low16,
        batch=high16).
      LPT/LPT_L2 (flat 1-D grid): t0 = linear*CGA_M (y=z=0); decode linearly
        using (n_qh_grid, n_batch) stashed in bidy_init/bidz_init.
    """
    t0 = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
    linear = t0 // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        t1 = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        return linear, t1 & cutlass.Int32(0xFFFF), (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
    return _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)


@cute.jit
def _softmax_warp_group(
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sdS_raw,
    sStats_raw,
    sdV_raw,
    seqlen_q,
    seqlen_kv,
    n_kv_blocks,
    n_qh,
    n_batch,
    attn_scale,
    attn_scale_log2e,
    dscale_dO,
    dscale_V,
    dscale_Q,
    attn_scale_for_dS,
    cta_in_pair,
    leader_cta_id,
    partner_cta_id,
    cta_id_x,
) -> None:
    """8 compute warps (2 wg x 4), single cga2 sub-group, lane=kv.

    wg0 -> q[0:64], wg1 -> q[64:128] (q_half_off).  Per kv-tile, per q-iter:
      softmax : wait mb_s_acc_full; tmem_load S[wg q-half] (cols [0:64]/[64:128]);
                P=exp2(scale*S-lse[q]); f16 P -> tcgen05_st S_acc[P_OFF+p_col_off]
                (wg0 [32:63], wg1 [64:95]); arrive mb_p_ready.  NO mb_s_acc_empty.
      dSoftmax: wait mb_dp_full; tmem_load dP[wg q-half]; arrive mb_dp_empty;
                dS=(attn_scale_for_dS*dP - do_dot[q])*P; f16 dS -> sdS SMEM ring;
                arrive mb_ds_smem_full.
    Post q-loop (per kv-tile): dV epilogue — wait mb_dv_ready; tmem_load dV
      [wg d_v-half]; BF16 -> sdV SMEM; arrive mb_dv_stg_full; arrive
      mb_dv_acc_empty (frees dV TMEM for next tile).

    f16 P is written INTO the S_acc region (single buffer, no separate ring).
    Each wg's P-write stays inside its OWN S-read q-half, so the S tmem_load ->
    P tmem_st needs NO cross-wg bar.sync.  P A-operand is read by the BMM2 dV
    mma_ts (S_acc[P_OFF]).
    attn_scale_for_dS = attn_scale_in * dscale_V * dscale_dO (host pre-fold).
    """
    # ---- Prelude (pair with MMA warp's barrier_arrive at barrier_id=1) ----
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)

    # tid_in_wg: 0..127 = this lane's kv-row within its wg.
    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw & cutlass.Int32(127)
    # wg_id (0/1) -> q-half; CHUNK = 64 q-cols per wg.
    wg_id = (warp_idx - cutlass.Int32(CFG.SOFTMAX_WG0_BASE)) // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    q_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)  # 0 or 64 (q-col offset)
    # f16 P TMEM col offset within S_acc[P_OFF..]: wg0 -> 0 (cols [32..63]),
    # wg1 -> 32 (cols [64..95]).  = _SMX_CHUNK*BPE/4 = 64*2/4 = 32.
    p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()  # consume mb_s_acc_full
    dp_full_state = PipelineState.start()  # consume mb_dp_full
    ds_empty_state = PipelineState.start(phase=1)  # dS SMEM ring slot free
    p_ready_state = PipelineState.start()  # produce mb_p_ready (fp8_P 2-stage ring)
    dv_ready_state = PipelineState.start()  # consume mb_dv_ready
    stats_full_state = PipelineState.start()

    # causal-bottom-right diagonal (k <= q + (SKV-SQ)); 0 for top-left/dense.
    causal_diag = (seqlen_kv - seqlen_q) if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT) else cutlass.Int32(0)
    # per-lane absolute kv-row base (this CTA's M-slice of the pair's kv-block).
    kv_lane_base0 = cta_in_pair * cutlass.Int32(CFG.TILE_M) + tid_in_wg

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        tmem_base = tmem_ptr_i32.load()  # TMEM col base (published by MMA alloc)

        # Mask: q-tile range that attends this kv-block + per-lane abs kv row.
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)
        kv_abs = kv_block_base + kv_lane_base0

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ---- stats: wait ring, read this wg's q-half lse / do_dot ----
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS)
            # ---- 1) softmax: S[wg q-half] -> P (regs) ----
            bars.mb_s_acc_full[s_full_state.idx].wait(s_full_state.phase)
            s_full_state = advance(s_full_state, CFG.STAGES_TMEM_S)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # NO mb_s_acc_empty: P lives in S_acc and the in-order MMA pipeline
            # gates the next Q·K (S_acc writer) behind P·dO[i]'s read of P (which
            # waits mb_p_ready, arrived AFTER this S read completed).
            lse_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw[stats_base + cutlass.Int32(STATS_LSE_OFF) + q_half_off + cutlass.Int32(i)] for i in range(_SMX_CHUNK)), cutlass.Float32
            )
            chunk_P = cute.math.exp2(reg_S.vec * attn_scale_log2e - lse_vec, fastmath=True)
            # Mask (BPROP transpose): zero P on masked (kv=lane, q=col) cells →
            # fp8_P (dV) AND chunk_dS (dS/dK/dQ) both inherit it.  No-op at NONE.
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                chunk_P = _mask_p_chunk(chunk_P, kv_abs, q_iter * cutlass.Int32(CFG.TILE_N) + q_half_off, seqlen_kv, causal_diag, _SMX_CHUNK)

            # ---- 2) f16 P -> S_acc[P_OFF + p_col_off] -> notify MMA (EARLY,
            #         before dSoftmax) so the dV BMM2 overlaps the whole dSoftmax
            #         below.  Single buffer in S_acc; each wg writes inside its OWN
            #         S-read q-half (wg0 [32..63] ⊂ [0..63], wg1 [64..95] ⊂
            #         [64..127]) so the S tmem_load->P tmem_st has no cross-wg race
            #         and needs NO bar.sync; all-lane mb_p_ready gates the MMA. ----
            if cutlass.const_expr(CFG.QAT_P):
                # Lanes are KV rows, registers are Q columns: quantization
                # groups are 16 adjacent lanes, NOT 16 adjacent registers.
                # Keep original FP32 chunk_P below for dS/dQ/dK.
                chunk_P_dv = cutlass.Vector.from_elements(
                    tuple(value for i in range(0, _SMX_CHUNK, 32) for value in _fake_quant_p_many(*(chunk_P[i + j] for j in range(32)))), cutlass.Float32
                )
                chunk_P_f16 = chunk_P_dv.to(STORAGE_DTYPE)
            else:
                chunk_P_f16 = chunk_P.to(STORAGE_DTYPE)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_col_off, cutlass.Float32), chunk_P_f16)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bars.mb_p_ready[p_ready_state.idx].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)

            # ---- 3) dSoftmax: load dP -> FREE the dP slot (mb_dp_empty, symmetric
            #         to mb_s_acc_empty) -> dS = (scale·dP - do_dot)·P -> sdS SMEM
            #         ring -> TMASTG.  Runs CONCURRENTLY with the MMA's dV above. ----
            bars.mb_dp_full[dp_full_state.idx].wait(dp_full_state.phase)
            dp_full_state = advance(dp_full_state, CFG.STAGES_TMEM_S)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dP read into regs -> slot free for dO·V[i+1] (every lane DSMEM-arrives leader).
            bars.mb_dp_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            dot_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw[stats_base + cutlass.Int32(STATS_DOT_OFF) + q_half_off + cutlass.Int32(i)] for i in range(_SMX_CHUNK)), cutlass.Float32
            )
            chunk_dS = (reg_dP.vec * attn_scale_for_dS - dot_vec) * chunk_P
            chunk_dS_f16 = chunk_dS.to(STORAGE_DTYPE)
            ds_slot = ds_empty_state.idx
            bars.mb_ds_smem_empty[ds_slot].wait(ds_empty_state.phase)
            ds_empty_state = advance(ds_empty_state, XFER_STAGES)
            # dS SMEM [kv, q], 128B-swizzled: f16 TILE_N=128 q = 256 B/row =
            # P_TMA_ITERS swizzle sub-tiles of P_D_BLOCK=64 q-cols each.  Each wg
            # owns ONE sub-tile (wg0 -> q[0:64] sub-tile 0, wg1 -> q[64:128] sub-
            # tile 1); store_swizzled block `gblk` at slab `gblk*P_BLOCK_BYTES` +
            # row `tid*P_D_BLOCK` (mirrors the dV epilogue / forward sO).
            gblk = wg_id
            (
                sdS_raw.subview(ds_slot * cutlass.Int32(dSBufferElems) + gblk * cutlass.Int32(P_BLOCK_BYTES) + tid_in_wg * cutlass.Int32(P_D_BLOCK))
            ).data_ptr().store_swizzled(chunk_dS_f16, alignment=128, swizzle=P_SMEM_SWIZZLE)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ds_smem_full[ds_slot].arrive()
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, STATS_STAGES)

        # ---- dV epilogue (per kv-tile): dV TMEM -> BF16 -> sdV SMEM ----
        # sdV is 128B-swizzled: TILE_O d_v = TMA_DV_ITERS swizzle sub-tiles of
        # DV_D_BLOCK cols each (mirrors the forward d256 O epilogue).  Each wg
        # owns TILE_O/SOFTMAX_WARPGROUPS d_v cols == (TMA_DV_ITERS/WGS) blocks;
        # store_swizzled block b at sub-tile slab `gblk*DV_BLOCK_SLAB` + row
        # `tid*DV_D_BLOCK`, with the 128B swizzle XOR (descriptor matches).
        bars.mb_dv_ready.wait(dv_ready_state.phase)
        dv_ready_state = advance(dv_ready_state, 1)
        _DV_BLOCKS_PER_WG = TMA_DV_ITERS // CFG.SOFTMAX_WARPGROUPS  # 4//2 = 2
        for _b in cutlass.range_constexpr(_DV_BLOCKS_PER_WG):
            gblk = wg_id * cutlass.Int32(_DV_BLOCKS_PER_WG) + cutlass.Int32(_b)
            reg_dV = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dV_OFF) + gblk * cutlass.Int32(DV_D_BLOCK), num_elems=DV_D_BLOCK)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dV_acc = Σ P_fp8·dO_fp8; true dV = dscale_dO · dV_acc (P∈[0,1] needs
            # no scale; dO carries dscale_dO).  BF16 out.
            dV_bf16 = (reg_dV.vec * dscale_dO).to(OUT_STORAGE_DTYPE)
            (sdV_raw.subview(gblk * cutlass.Int32(DV_BLOCK_SLAB) + tid_in_wg * cutlass.Int32(DV_D_BLOCK))).data_ptr().store_swizzled(
                dV_bf16, alignment=128, swizzle=P_SMEM_SWIZZLE
            )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dv_stg_full.arrive()
        # Free dV TMEM for the next kv-tile's S·dO[0] (accumulate=False) overwrite.
        bars.mb_dv_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        # ---- Scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # ---- P11: release the MMA's TMEM alloc ----
    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


@cute.jit
def _mma_warp(
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    tmem_ptr_i32,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    n_kv_blocks,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
) -> None:
    """MMA leader (single cga2 sub-group) — NATURAL-order 3-matmul stream.

    Per kv-tile (K, V one-shot), the q-iter MMA ISSUE order is the natural
    Q·K -> dO·V -> P·dO (NO Q·K[i+1] lookahead):

        for i in q_lo..q_hi-1:  Q·K[i] ; dO·V[i] ; P·dO[i]

      BMM1 S  = Q·K^T  -> S_acc  TMEM [S_OFF]   (A=K full SMEM, B=Q[i]) mma_ss
      BMM1 dP = dO·V^T -> dP     TMEM [dP_OFF]  (A=V, B=dO_dP[i])       mma_ss
      BMM2 dV = P·dO   -> dV_acc TMEM [dV_OFF]  (A=P TMEM S_acc[P_OFF], B=dO_dv[i] BT)
                          mma_ts ; accumulate=(i>q_lo)

    Handshakes (S / dP / P all single-buffer):
      mb_s_acc_full  Q·K  -> softmax ("S[i] ready")
      mb_dp_full     dO·V -> softmax ("dP[i] ready")
      mb_dp_empty    gates dO·V[i+1] (dSoftmax loaded dP[i])      [pre-armed]
      mb_p_ready     softmax -> P·dO[i] ("P[i] in S_acc[P_OFF]")
    NO mb_s_acc_empty and NO mb_p_empty: P lives inside S_acc, and the in-order
    MMA pipeline serializes Q·K[i+1] (S_acc writer) AFTER P·dO[i] (P reader) and
    transitively after softmax[i]'s S read (p_ready[i] gates P·dO[i]).
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=False)
    # Pair with the softmax wg's barrier_cta_sync(1) at top-of-body.
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()  # dO_dP ring (BMM1 dP)
    dodv_full_state = PipelineState.start()  # dO_dv ring (BMM2 dV)
    # Consumer handshakes: pre-arm dp_empty so the first dO·V[0] passes (no
    # dSoftmax load yet).  p_ready is a real consumer wait.  NO s_acc_empty —
    # in-order MMA covers the S_acc WAR (see docstring).
    dp_empty_state = PipelineState.start(phase=1)
    p_ready_state = PipelineState.start(phase=0)
    dv_empty_state = PipelineState.start(phase=0)  # epilogue drained dV_acc

    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    # kv_super tracked for the mask q-loop bounds (the MMA only needs the
    # q-COUNT + the accumulate predicate; absolute q lands in softmax/tmaldg).
    kv_super_idx, _, _ = _boot_tile(sched)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))  # dO·V output (FP32 dP)
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_OFF))
    tmem_P = tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))  # f16 P inside S_acc[P_OFF] (dV mma_ts A)

    # ---- Descriptors ------------------------------------------------------
    # BMM1 S = K·Q^T (A=K M=kv, B=Q N=q, K=d_qk).  idesc shape (n/m/k_dim) is
    # K-independent; the K-split half desc below reuses it (only K differs).
    idesc_bmm1_s = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=1
    )
    # BMM1 dP = V·dO^T (A=V M=kv, B=dO N=q, K=d_v).
    idesc_bmm1_dp = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=1
    )
    bmm1_dp_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_O,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_dp,
        kind=MMA_KIND,
    )
    # BMM2 dV = P·dO (A=fp8_P TMEM M=kv K=q, B=dO_dv N=d_v BT, K=q).
    idesc_bmm2_dv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_O, m_dim=CFG.TILE_M * CFG.CTA_MMA, a_major=0, b_major=1, k_dim=1
    )
    bmm2_dv_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm2_dv,
        kind=MMA_KIND,
    )

    # BMM1 S K-split: the d_qk=256 contraction runs as two K=128 halves.  ONE
    # MmaDesc (K=128, shape == idesc_bmm1_s) drives BOTH the front mma_ss (A=K
    # front SMEM) and the back mma_ts (A=K back-half in TMEM); the operand source
    # is decided by the mma_ss/mma_ts call, not the desc.  tmem_advance_A =
    # TKH*BPE/4 = 8 cols/step × 8 steps = 64 cols = RSVD_COLS (UTCCP'd K back-half).
    bmm1_s_half_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=K_SPLIT_TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_s,
        kind=MMA_KIND,
    )
    tmem_K_back = tmem_raw.subview(cutlass.Int32(LAYOUT.RSVD_OFF))
    # Subtile-major UTCCP walk (tcgen05.cp.128x128b; cf. SM100 d512_f16).
    # 16 calls × 4 cols = 64 cols = K back-half (d_qk[128:256], subtiles 2,3).
    _UTCCP_BYTES_PER_CALL = 16
    _UTCCP_TMEM_COLS_PER_CALL = _UTCCP_BYTES_PER_CALL // 4  # 4
    _UTCCP_PER_SUBTILE = CFG.K_SWZ_BYTES // _UTCCP_BYTES_PER_CALL  # 8
    _UTCCP_SUBTILE_DESC_STRIDE = (CFG.TILE_M * CFG.K_SWZ_BYTES) // _UTCCP_BYTES_PER_CALL  # 1024
    _UTCCP_N_CALLS = LAYOUT.RSVD_COLS // _UTCCP_TMEM_COLS_PER_CALL  # 16

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Mask: q-tile range that attends this kv-block (uniform across the pair).
        # The prologue handles q_lo; the loop runs [q_lo, q_hi); dV accumulate
        # restarts at q_lo (NOT 0) — rule the cuda-kernels rules §2.
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        # K + V — one-shot per kv-tile.
        bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase)
        bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase)
        desc_K = sK[k_full_state.idx].desc()
        desc_V = sV[v_full_state.idx].desc()

        # K-split: UTCCP K's d_qk[128:256] back-half (subtiles 2,3) SMEM → TMEM
        # [RSVD] so BMM1 S can read it as a TMEM-A mma_ts operand, then commit so
        # the TMA-LDG learns the K back-half SMEM is dead (sdO_dv stage-1 alias
        # seam).  Leader-only UTCCP self-fills BOTH peers' TMEM (rule §17); the
        # subsequent mma_ts is in the same MMA pipeline → no leader-side wait.
        desc_K_back = sK[k_full_state.idx].shifted(K_BACK_OFF_ELEMS).desc()
        if nvvm.elect_sync():
            for _qk in cutlass.range_constexpr(_UTCCP_N_CALLS):
                _s = _qk // _UTCCP_PER_SUBTILE
                _kk = _qk % _UTCCP_PER_SUBTILE
                _desc_off = _s * _UTCCP_SUBTILE_DESC_STRIDE + _kk
                nvvm.tcgen05_cp(
                    nvvm.Tcgen05CpShape.SHAPE_128X128B,
                    tmem_K_back.subview(_qk * _UTCCP_TMEM_COLS_PER_CALL),
                    desc_K_back + _desc_off,
                    group=CTA_GROUP_KIND,
                )
        # commit_mma multicast → 1 arrive per CTA's mb_k_utccp_done; fires
        # AFTER the UTCCP stream drains (TMA-LDG alias seam, P13).  Predicated
        # (P16) to match the kernel's other commit_mma arrives — single-warp
        # so perf-neutral; the tcgen05_cp loop stays inside the elect branch.
        bars.mb_k_utccp_done.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())

        # ---- NATURAL order: Q·K[i] ; dO·V[i] ; P·dO[i], per q_iter ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ----- BMM1 S = Q·K[i] -> S_acc (K-split: mma_ss + mma_ts) -----
            # No s_acc_empty gate: in-order MMA serializes this write AFTER the
            # previous iter's P·dO read of S_acc[P_OFF] (and transitively after
            # softmax[i-1]'s S read via mb_p_ready).
            # S = K_front·Q_front^T (mma_ss, overwrite) + K_back·Q_back^T
            # (mma_ts, A=K back-half in TMEM, B=Q back subtiles 2,3 in SMEM,
            # accumulate onto the front half) = exact K·Qᵀ.  Both consume
            # sQ[stage] SMEM, so mb_q_empty fires only after the back mma_ts.
            bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase)
            mma_ss(bmm1_s_half_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S, accumulate=False)
            mma_ts(bmm1_s_half_desc, tmem_K_back, sQ[q_full_state.idx].shifted(Q_BACK_OFF_ELEMS).desc(), tmem_S, accumulate=True)
            elect_p = nvvm.elect_sync()
            bars.mb_s_acc_full[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ----- BMM1 dP = dO·V[i] -> dP (mma_ss) -----
            # dp_empty gates reuse of the single dP slot (dSoftmax[i-1] read it);
            # pre-armed so dO·V[0] passes.
            bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase)
            dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)
            bars.mb_do_full[do_full_state.idx].wait(do_full_state.phase)
            mma_ss(bmm1_dp_desc, desc_V, sdO[do_full_state.idx].desc(), tmem_dP)
            elect_p = nvvm.elect_sync()
            bars.mb_dp_full[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_do_empty[do_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            do_full_state = advance(do_full_state, CFG.STAGES_dO)

            # ----- BMM2 dV += P·dO[i] -> dV_acc (mma_ts, A=P in S_acc[P_OFF]) -----
            # P[i] was stored by softmax into S_acc[P_OFF..P_OFF+P_COLS) (single
            # buffer).  No p_empty: the next iter's Q·K (S_acc writer) is in-order
            # AFTER this read, so it can't clobber P early.
            bars.mb_p_ready[p_ready_state.idx].wait(p_ready_state.phase)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)
            bars.mb_dodv_full[dodv_full_state.idx].wait(dodv_full_state.phase)
            mma_ts(bmm2_dv_desc, tmem_P, sdO_dv[dodv_full_state.idx].desc(), tmem_dV, accumulate=(q_iter > q_lo))
            elect_p = nvvm.elect_sync()
            bars.mb_dodv_empty[dodv_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dodv_full_state = advance(dodv_full_state, CFG.STAGES_dO_DV)

        # dV accumulation complete for this kv-tile -> epilogue (compute warps).
        bars.mb_dv_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        # dV TMEM read drained by epilogue before the next kv-tile's S·dO[0]
        # (accumulate=False) overwrites dV_acc — gate the next tile here.
        bars.mb_dv_acc_empty[dv_empty_state.idx].wait(dv_empty_state.phase)
        dv_empty_state = advance(dv_empty_state, 1)

        # End-of-tile: release K + V.
        if nvvm.elect_sync():
            bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
            bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
        k_full_state = advance(k_full_state, CFG.STAGES_KV)
        v_full_state = advance(v_full_state, CFG.STAGES_KV)

        # ---- Scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_super_idx, _, _ = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # ---- TMEM dealloc ----
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    """Quiet MMA warp on the non-leader CTA of the cga2 pair: collective
    tmem_alloc (the leader's UTCCP-free collective MMAs write this CTA's TMEM
    half), fire the named barriers the softmax wg waits on, then wait
    mb_tmem_dealloc + release TMEM."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=False)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _tmastg_warp(
    tma_dv_desc,
    tma_ds_desc,
    sdV,
    sdS,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    n_qh,
    cta_in_pair,
    head_base,
) -> None:
    """TMA-store service warp (single sub-group):
      per q-iter store the dS slot -> GMEM workspace [B,H,S_kv,S_q]
        (mb_ds_smem_full wait -> TMA store -> mb_ds_smem_empty);
      per kv-tile store dV -> GMEM out [B,H,S_kv,d_v] (mb_dv_stg_full wait ->
        TMA store -> mb_dv_stg_empty).
    The lse/do_dot stats prefetch now lives on the SCHEDULER warp
    (_scheduler_stats_warp) so it runs concurrently with these TMA stores
    instead of serialized behind them on one warp (latency-hiding for the
    compute warp's LDS.128 LSE/DOT reads).
    """
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_ds = GmemTileTma(tma_ds_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    ds_full_state = PipelineState.start()  # consume mb_ds_smem_full
    dv_full_state = PipelineState.start()  # consume mb_dv_stg_full

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        kv_block_base = kv_super_idx * cutlass.Int32(CFG.TILE_M * CFG.CTA_MMA)
        KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)
        # Mask: only the in-range q-tiles produce dS.  The skipped (out-of-band)
        # q-tiles' dS-workspace regions stay ZERO (driver zero-inits the
        # workspace per head-chunk under masks) so dK/dQ = dS·Q / dSᵀ·K are correct.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)

            # dS store for THIS q-iter (after softmax produced it).
            ds_slot = ds_full_state.idx
            bars.mb_ds_smem_full[ds_slot].wait(ds_full_state.phase)
            ds_full_state = advance(ds_full_state, XFER_STAGES)
            tma_store_tile(
                sdS[ds_slot],
                # workspace [B,H,S_kv,S_q] → coords innermost-first
                # (S_q, S_kv, H, B); cf. ds_box (1,1,TILE_M,TILE_N).
                tma_ds(q_col_base, kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_idx),
            )
            tma_store_commit()
            tma_store_wait()
            if nvvm.elect_sync():
                bars.mb_ds_smem_empty[ds_slot].arrive()

        # (c) dV store for THIS kv-tile (after the epilogue).  dV → FULL output
        # [B,H_q,S_kv,d_v] so use the full-tensor head (head_idx + head_base);
        # dS above stays CHUNK-local (workspace is [B,H_chunk,S_kv,S_q]).
        bars.mb_dv_stg_full.wait(dv_full_state.phase)
        dv_full_state = advance(dv_full_state, 1)
        tma_store_tile(
            sdV[0],
            tma_dv(cutlass.Int32(0), head_idx + head_base, kv_block_base + KV_ROW_OFFSET_PEER, batch_idx),
        )
        tma_store_commit()
        tma_store_wait()
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()

        # Scheduler tail.
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _tmaldg_warp(
    tma_q_desc,
    tma_do_desc,
    tma_do_dv_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    n_kv_blocks,
    n_qh,
    n_batch,
    qh_per_kh,
    head_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
) -> None:
    """TMA-LDG warp (single cga2 sub-group).

    Per kv-tile: K, V one-shot (M-split kv, full d).  Per q_iter:
      Q       — BMM1 S B (N-split on q: TILE_N/CTA_MMA q-rows × full d_qk).
      dO      — BMM1 dP B (N-split on q: TILE_N/CTA_MMA q-rows × full d_v).
      dO_dv   — BMM2 dV B (N-split on d_v: full TILE_N q-rows × TILE_O/CTA_MMA
                d_v cols, BT=true).  SECOND load of the SAME dO GMEM into a
                separate SMEM ring with the dV-box descriptor.
    """
    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_do_dv = GmemTileTma(tma_do_dv_desc)
    tma_v = GmemTileTma(tma_v_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    # Workspace-chunking: Q/K/V/dO read the FULL [B,H_q,...] tensors, so index
    # by the full-tensor head (grid-local head + head_base).
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)

    # dO_dv per-CTA inner (d_v) offset; q-axis is FULL (no per-CTA q offset).
    DO_DV_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)
    dOdvTmaBytes = (CFG.TILE_N * (CFG.TILE_O // CFG.CTA_MMA)) * CFG.BPE * CFG.CTA_MMA

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    dodv_empty_state = PipelineState.start(phase=1)
    # K-split alias seam: consumer of the leader MMA's per-tile mb_k_utccp_done
    # commit (real producer, NOT pre-armed — fires on tile 0's UTCCP).
    utccp_done_state = PipelineState.start(phase=0)
    # sdV aliases K storage. K's MMA consumer completion alone does not
    # release that storage: the previous tile's dV TMA store must finish too.
    # Phase 1 is immediately free before the first tile (initial phase 0).
    dv_storage_empty_state = PipelineState.start(phase=1)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(CFG.TILE_M * CFG.CTA_MMA)
        K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)
        Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_M_PER_CTA)
        # Mask: only load Q/dO for the q-tiles that attend this kv-block.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        # ---- K + V — one-shot per kv-tile (M-split kv, full d) ----
        bars.mb_dv_stg_empty.wait(dv_storage_empty_state.phase)
        dv_storage_empty_state = advance(dv_storage_empty_state, 1)
        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
        if is_leader:
            if nvvm.elect_sync():
                bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kTmaTransactionBytes)
        tma_load_tile(
            sK[k_empty_state.idx],
            tma_k(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_k_full[k_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
        if is_leader:
            if nvvm.elect_sync():
                bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=vTmaTransactionBytes)
        tma_load_tile(
            sV[v_empty_state.idx],
            tma_v(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_v_full[v_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

        # K-split alias seam: sdO_dv stage-1 aliases the K back-half SMEM, so
        # the dO_dv loads (and the whole q-loop) must wait until the leader MMA
        # has UTCCP'd the K back-half into TMEM (mb_k_utccp_done).  Once per
        # tile — within-tile dO_dv prefetch still pipelines.  (MMA does UTCCP →
        # commit BEFORE it needs Q, so no deadlock against mb_q_full below.)
        bars.mb_k_utccp_done.wait(utccp_done_state.phase)
        utccp_done_state = advance(utccp_done_state, 1)

        # ---- Q + dO + dO_dv — per q_iter (mask-bounded range) ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_N)

            # Q (N-split on q).
            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_q_full[q_empty_state.idx].arrive(n_bytes=qTmaTransactionBytes)
            tma_load_tile(
                sQ[q_empty_state.idx],
                tma_q(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_q_full[q_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)

            # dO (dP-view, N-split on q).
            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_do_full[do_empty_state.idx].arrive(n_bytes=dOTmaTransactionBytes)
            tma_load_tile(
                sdO[do_empty_state.idx],
                tma_do(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)

            # dO_dv (dV-view, N-split on d_v; full q rows).
            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_dodv_full[dodv_empty_state.idx].arrive(n_bytes=dOdvTmaBytes)
            tma_load_tile(
                sdO_dv[dodv_empty_state.idx],
                tma_do_dv(DO_DV_OFFSET_PEER, full_head, q_row_base, batch_idx),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)

        # ---- Scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: cga2 _empty ring drain — OUTSIDE the persistent loop.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.STAGES_Q):
            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase)
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)
        for _qs in cutlass.range_constexpr(CFG.STAGES_dO):
            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase)
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)
        # sdO_dv ring is single-buffer — drain it once (NOT STAGES_dO times, or
        # the 2nd wait hangs on a 1-stage ring).  P15.
        for _qs in cutlass.range_constexpr(CFG.STAGES_dO_DV):
            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase)
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)


@cute.jit
def _scheduler_stats_warp(
    sched,
    is_cga_first_cta,
    bars,
    sStats_raw,
    lse_tensor,
    do_dot_tensor,
    attn_scale_in,
    seqlen_q,
    seqlen_kv,
    head_base,
) -> None:
    """Persistent tile scheduler (try_cancel protocol) FUSED with the
    lse/do_dot stats prefetch.

    Moving the stats prefetch off the TMASTG warp (where it was serialized
    behind the dS/dV TMA stores) onto this otherwise-idle scheduler warp lets
    the stats ring fill run concurrently with the GMEM stores — the compute
    warps' LDS.128 LSE/DOT reads (the dominant L1TEX latency stall) get their
    ring filled promptly.

    Per loop iter:
      (A) prefetch the CURRENTLY-processed tile's lse/do_dot -> sStats ring
          (one slot per q-iter).  Tile context tracked one-behind the
          scheduling: bootstrap = blockIdx, then the decoded payload.
      (B) the standard try_cancel scheduler protocol for the NEXT tile
          (mirrors scheduler_warp_loop).
    `lse·log2(e)` / `do_dot·attn_scale_in` folded here (host passes RAW); only
    head/batch context is needed (stats are [batch, head, q]).  This warp does
    NOT fire read_tile_id (it is the scheduler) → READ_TILE_ARRIVERS unchanged.
    """
    lse_arr = cutlass.make_array_view(lse_tensor)
    dot_arr = cutlass.make_array_view(do_dot_tensor)
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    _PER_LANE = CFG.TILE_N // 32  # constexpr 4

    # Context of the tile the consumers are CURRENTLY processing (one behind
    # the scheduling).  Bootstrap = this CTA's blockIdx (== consumers' init).
    cur_kv_super, cur_head, cur_batch = _boot_tile(sched)

    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        # lse / do_dot are FULL [B,H_q,S_q] tensors → index by full-tensor head.
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        # Mask: prefetch stats ONLY for the q-tiles the consumers process (so
        # the stats ring count matches the softmax consumer's [q_lo, q_hi)).
        kv_block_base = cur_kv_super * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)
        # ---- (A) stats prefetch for the CURRENT tile (in-range q-iters) ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase)
            stats_empty_state = advance(stats_empty_state, STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                sStats_raw[slot_base + cutlass.Int32(STATS_LSE_OFF) + col] = lse_arr[cur_batch, cur_full_head, q_col_base + col] * _LOG2E
                sStats_raw[slot_base + cutlass.Int32(STATS_DOT_OFF) + col] = dot_arr[cur_batch, cur_full_head, q_col_base + col] * attn_scale_in
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        # ---- (B) try_cancel scheduler protocol for the NEXT tile ----
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync() and is_cga_first_cta:
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        validity = (sched.tile_id_smem.subview(state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid = validity & cutlass.Int32(1)
        # Decode the NEXT tile's context -> becomes "current" for the next loop.
        cur_kv_super, cur_head, cur_batch = _decode_tile_payload(sched, state.idx)
        state = advance(state, CFG.SCHEDULER_STAGES)


# === Host launcher =========================================================


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    do_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    dv_tensor: cute.Tensor,  # out  [B, S_kv, H, d_v]  BF16
    ds_tensor: cute.Tensor,  # out  [B, H, S_kv, S_q]  FP8 workspace (dK/dQ matmul A)
    lse_tensor: cute.Tensor,
    scaled_do_dot_tensor: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    dscale_dO: cutlass.Float32,
    dscale_V: cutlass.Float32,
    dscale_Q: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,
    head_base: cutlass.Int32,
    seqlen_kv_real: cutlass.Int32,  # PADDED real KV length (mask kv>=this); = SKV when dense
    stream: _cuda_driver.CUstream,
) -> None:
    # problem_size[5] = QH_CHUNK: grid head-count + dS-workspace head extent
    # (≤ QH for the workspace-chunked path; == QH for the single-shot path).
    B, QH, KH, SQ, SKV, QH_CHUNK = problem_size

    # TMA box specs ([B, S, H, D] layout; box = (1 batch, S-rows, 1 head, D-cols)).
    qk_box_q = (1, _M_PER_CTA, 1, TMA_QK_GRANU_ELEMS)  # Q  (BMM1 S B, q-split)
    qk_box_k = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)  # K  (M-split kv)
    do_box = (1, _M_PER_CTA, 1, TMA_VO_GRANU_ELEMS)  # dO (BMM1 dP B, q-split)
    v_box = (1, CFG.TILE_M, 1, TMA_VO_GRANU_ELEMS)  # V  (M-split kv)
    # dO dV-view (BMM2 dV B, BT=true): TILE_N q-rows × per-sub-tile d_v granu.
    # f16: TILE_O/CTA_MMA = 128 d_v × 2 B = 256 B/row > the 128 B swizzle atom,
    # so the box inner dim is the GRANU (TMA_VO_SG1_GRANU_ELEMS = 64), walked by
    # TMA_VO_SG1_ITERS sub-tiles (matches the sdO_dv SmemTile).
    do_dv_box = (1, CFG.TILE_N, 1, TMA_VO_SG1_GRANU_ELEMS)
    # dV STG: 128B-swizzled box per sub-tile (TILE_M kv × DV_D_BLOCK BF16 d-cols);
    # TMA_DV_ITERS=4 sub-tiles.  Matches the store_swizzled sdV layout.
    dv_box = (1, CFG.TILE_M, 1, TMA_DV_GRANU_ELEMS)
    # dS workspace STG: TILE_M kv × per-sub-tile q granu (f16: P_D_BLOCK=64,
    # P_TMA_ITERS=2 sub-tiles — TILE_N q × 2 B = 256 B/row > the 128 B swizzle
    # atom).  Workspace is [B, H, S_kv, S_q] (box dims 2=S_kv, 3=S_q) so the
    # dK/dQ matmul A operand is a free [B*H, S_kv, S_q] reshape (no permute).
    ds_box = (1, 1, CFG.TILE_M, P_D_BLOCK)
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor, box_dims=qk_box_q, stride_order=stride_order, swizzle=_tma_swz(CFG.Q_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=do_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_dv_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=do_dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor, box_dims=qk_box_k, stride_order=stride_order, swizzle=_tma_swz(CFG.K_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor, box_dims=v_box, stride_order=stride_order, swizzle=_tma_swz(CFG.V_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # dV STG: BF16 cells written via store_swizzled (128B) into sdV -> descriptor
    # swizzle must match (s128b), like the dS / forward-O store paths.
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(
        dv_tensor, box_dims=dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dV_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # dS STG: softmax store_swizzled's fp8 dS into sdS with P_SMEM_SWIZZLE
    # (128 B) -> descriptor swizzle must match (s128b).
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_tensor, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )

    # Grid: one cga2 cluster per (kv_block, head, batch) tile.  Head axis spans
    # QH_CHUNK heads per launch (== QH single-shot); the driver loops chunks and
    # bumps head_base so the union covers all QH heads.
    # SCHEDULER_POLICY: 0=natural 3-D grid (kv_block, head, batch); 1/2=LPT/
    # LPT_L2 flat 1-D grid (kv_super OUTER → heaviest causal kv-blocks first).
    # Both ride the SAME persistent try_cancel scheduler; the policy only sets
    # the launch shape + the per-tile decode (_boot_tile / _decode_tile_payload).
    cluster_kv_block = CFG.TILE_M * CFG.CTA_MMA
    kv_blocks = (SKV + cluster_kv_block - 1) // cluster_kv_block
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        grid_shape = (kv_blocks * CFG.CGA_M, QH_CHUNK, B)
    else:
        grid_shape = (kv_blocks * QH_CHUNK * B * CFG.CGA_M, 1, 1)

    _kernel(
        tma_q_desc,
        tma_do_desc,
        tma_do_dv_desc,
        tma_k_desc,
        tma_v_desc,
        tma_dv_desc,
        tma_ds_desc,
        lse_tensor,
        scaled_do_dot_tensor,
        # kernel `seqlen_kv` = REAL length (drives the PADDED mask); the grid +
        # kv_blocks + TMA descriptors stay on the allocated compile-time SKV.
        cutlass.Int32(SQ),
        seqlen_kv_real,
        cutlass.Int32(kv_blocks),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        dscale_dO,
        dscale_V,
        dscale_Q,
        attn_scale_for_dS,
        head_base,
        cutlass.Int32(QH_CHUNK),
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


def compile(b: int, qh: int, kh: int, sq: int, skv: int, qh_chunk: int = 0) -> Callable:
    """Compile a fixed QAT configuration from plan-time descriptors only."""
    if b != 1 or qh != kh or sq != skv or sq <= 0 or sq % 256:
        raise ValueError("SM100 QAT requires B=1, MHA, and equal positive 256-aligned lengths")
    if qh_chunk == 0:
        qh_chunk = qh
    if qh_chunk <= 0 or qh <= 0 or qh % qh_chunk:
        raise ValueError("head chunk must be a positive divisor of H")
    fake_q = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (b, sq, qh, CFG.TILE_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_do = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (b, sq, qh, CFG.TILE_O),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (b, skv, kh, CFG.TILE_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (b, skv, kh, CFG.TILE_O),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_dv = cute.runtime.make_fake_compact_tensor(
        OUT_STORAGE_DTYPE,
        (b, skv, qh, CFG.TILE_O),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    # Logical [B,S,H,D] views over the caller's contiguous BHSD dO/dV.
    # TMA consumes the declared strides directly: no adapter-side copies.
    fake_do = cute.runtime.make_fake_tensor(
        STORAGE_DTYPE,
        (b, sq, qh, CFG.TILE_O),
        (sq * qh * CFG.TILE_O, CFG.TILE_O, sq * CFG.TILE_O, 1),
        assumed_align=16,
    )
    fake_dv = cute.runtime.make_fake_tensor(
        OUT_STORAGE_DTYPE,
        (b, skv, qh, CFG.TILE_O),
        (skv * qh * CFG.TILE_O, CFG.TILE_O, skv * CFG.TILE_O, 1),
        assumed_align=16,
    )
    # dS workspace [B, H_chunk, S_kv, S_q] BF16 (consumed by dK = dS·Q / dQ = dSᵀ·K
    # GEMMs).  Head dim is the CHUNK count so the workspace stays ≤ the cap.
    fake_ds = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (b, qh_chunk, skv, sq),  # [B, H_chunk, S_kv, S_q] (matmul A reshape)
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_lse = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (b, qh, sq),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_scaled_do_dot = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (b, qh, sq),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dv,
        fake_ds,
        fake_lse,
        fake_scaled_do_dot,
        (b, qh, kh, sq, skv, qh_chunk),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),  # dscale_dO
        cutlass.Float32(0.0),  # dscale_V
        cutlass.Float32(0.0),  # dscale_Q
        cutlass.Float32(0.0),  # attn_scale_for_dS
        cutlass.Int32(0),  # head_base
        cutlass.Int32(skv),  # seqlen_kv_real (default = allocated SKV; driver overrides for --padded)
        _cuda_driver.CUstream(0),
        options="--enable-tvm-ffi",
    )
