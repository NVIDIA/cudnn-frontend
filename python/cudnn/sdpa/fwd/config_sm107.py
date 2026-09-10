# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the Frost SM107 (Rubin) DSL SDPA-forward flavors.

**Why this is a separate module from** :mod:`cudnn.sdpa.fwd.config_sm100`, and
not a widening of it: the Rubin kernels are a different kernel *lineage*, not
the Blackwell kernels recompiled. They implement different pipelines, so they
need different warp populations, stage depths and MMA k-steps. The two module's
flavors overlap in NAME (``d128`` / ``d256`` / ``d512``) and in almost nothing
else, and the overlap is what makes sharing dangerous rather than economical.

That is not a theoretical concern — it cost a debugging session. The SM107
``d256`` per-tensor FP8 kernel was routed through ``make_cfg_d256`` (Blackwell),
which derives its whole warp layout from a *split-P* switch that only the
Blackwell d256 kernel implements::

    SOFTMAX_WARPGROUPS   rubin-body=1     blackwell-cfg=2
    TOTAL_WARPS          rubin-body=12    blackwell-cfg=16
    THREADS_PER_CTA      rubin-body=384   blackwell-cfg=512
    MMA_WARP_ID          rubin-body=8     blackwell-cfg=12
    READ_TILE_ARRIVERS   rubin-body=11    blackwell-cfg=15

The last line is the fatal one: ``READ_TILE_ARRIVERS`` is the init count of the
scheduler's tile-handoff mbarrier, so a count of 15 against 11 actual arrivers
can never be reached and the *first* tile handoff never completes. The kernel
hung at every shape, down to ``B=1 H=1 S=128`` — 100 % GPU, no fault, no wrong
number, nothing for a sanitizer to find. Keeping Rubin's geometry in its own
module is what makes that class of mismatch impossible rather than merely
unlikely.

Provenance: every value here is the one the Rubin kernel BODIES were written and
validated against (the pre-upstream ``sdpa_config_{llama,dsv3,qwen,
dsv4}`` flavors, keyed here by op geometry per the engine contract, not by model
name). ``test_sdpa_fwd_config_sm107.py`` pins the ones a kernel body would
deadlock or corrupt on if they drifted.

Shared with SM100 by design: :class:`TemplateParams` (the engine-contract record
the loader injects) and the mask / dtype / scheduler vocabulary. Those are the
interface; everything below is the Rubin geometry behind it.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier, by ``graph_analyzer.probe`` /
``api_dsl.SdpaFwdDslSm100.check_support`` — a ``ValueError`` from here means
those support checks have a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from cudnn.frost.tile_dsl.constants import (
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# The engine-contract record is deliberately SHARED: one lowering builds it, and
# an arch-specific copy would let the two drift in ways the loader cannot see.
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

__all__ = [
    "TemplateParams",
    "resolve_dtype_o",
    "TmaIters",
    "CfgD128",
    "CfgD192",
    "CfgD256",
    "CfgD512",
    "make_cfg_d128",
    "make_cfg_d128_mxfp8",
    "make_cfg_d192",
    "make_cfg_d192_mxfp8",
    "make_cfg_d256",
    "make_cfg_d256_mxfp8",
    "make_cfg_d512",
    "make_cfg_d512_mxfp8",
    "SMEM_CAP_BYTES",
    "SM107_FP8_THD_SHAPES",
    "SM107_F16_THD_SHAPES",
]


# ---------------------------------------------------------------------------
# Rubin architecture constants
# ---------------------------------------------------------------------------

# Rubin SM10.7 oversized SMEM cap.  Reaching past the standard 227 KiB requires
# the launcher to set ALLOW_OVERSIZED_SHARED_MEMORY; the d128 FP8 9-stage ring
# and the d256/d512 flavors all depend on it.
SMEM_CAP_BYTES = 327 * 1024

# ...but the CAPACITY is not the budget. The shipped sm107/prefill_d128_fp8.py
# sizes its own guard against 320 KiB ("327 KiB capacity minus reserves"), and
# the kernels additionally spend ~2 KiB on barriers, the scheduler ring and the
# TMEM pointer. Validating Q/K/V/O against the raw 327 KiB waves through an
# allocation that silently overflows: d192 f16 at STAGES_KV=4 sums to exactly
# 320 KiB of Q/K/V/O, passed a 327 KiB check, and produced an output that was
# 50 % ZEROS with a numerically EXACT LSE (BMM1 and the softmax are fine; only
# the clobbered O half is wrong). Check against this, not the capacity.
SMEM_USABLE_BYTES = 320 * 1024
_SMEM_FIXED_OVERHEAD = 2 * 1024  # barriers + scheduler + tmem-ptr slack

# TMEM columns.  Rubin raises the Blackwell 512 to 576, which needs
# `tmem_alloc(..., is_exclusive=True)` (see tile_dsl/tmem.py).
TMEM_TOTAL_COLS = 576

_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16 = 0, 1, 2, 3

# Head-dim shapes whose Rubin PER-TENSOR FP8 kernel carries the THD/varlen leg.
#
# ONE definition, consumed by BOTH the engine row (`engines._sm100_fp8_spec`'s
# ``thd_d_shapes`` on the Rubin arm) and the standalone adapter's THD gate
# (``api_dsl.SdpaFwdDslSm100.check_support``).  They are two enforcement points
# for one fact, and contract rule 8b' exists because keeping two copies in step
# by hand does not work: widen only the row and a graph enters the ranked list
# then dies with a bare NotImplementedError in check_support; widen only the
# wrapper and the row still declines.  Sharing the constant makes disagreement
# unrepresentable rather than merely tested.
#
# Membership rule: the shape's body must carry the FROST THD contract -- the
# 14-arg build_thd_meta_o_descs_kernel, 4B+4 metadata, (b+3) O-descriptor slots,
# the persistent claim-counter scheduler, the dead-unit O-store guard and the
# packed-total-clamped runtime K/V descriptors.
#
# All four qualify as of 2026-09-09.  d192xd128 came free -- it IS the shipped
# d128 body, only the config factory differs -- and d256 / d512 were moved onto
# the contract (frost_dev/port_thd_contract.py); they previously called the
# setup kernel with the pre-upstream 7-arg signature and allocated 3B+2 where
# the shared decode reads 4B+4.  Confirmed on w2u1g-lc-0030: 43 passed / 0
# failed across the per-tensor FP8 THD suite.
SM107_FP8_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

# f16/bf16 flavor names whose kernel body HAS been ported to the FROST
# setup-kernel contract (the 14-arg build_thd_meta_o_descs_kernel + the 4B+4
# metadata the shared decode reads).  Keyed by the `flavor` string
# `_validate_params` already receives, so adding a ported flavor is one entry.
_F16_THD_FLAVORS = frozenset({"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512"})

# Head-dim shapes whose Rubin f16/bf16 kernel carries the THD/varlen leg -- the
# same one-definition-two-consumers arrangement as SM107_FP8_THD_SHAPES above
# (engine row + standalone adapter gate; contract rule 8b').  Must stay in step
# with _F16_THD_FLAVORS, which is the same fact keyed by config-flavor name.
SM107_F16_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})


# ---------------------------------------------------------------------------
# Derivation helpers (Rubin values — several differ from the SM100 twins)
# ---------------------------------------------------------------------------


def bpe(dtype: int) -> int:
    """Bytes per element. FP8/MXFP8 = 1, BF16/FP16 = 2."""
    return 1 if dtype <= _DTYPE_E5M2 else 2


def resolve_dtype_o(params: TemplateParams) -> int:
    """``TemplateParams.dtype_o`` uses -1 as "inherit dtype_qkv"; every consumer
    must resolve it before use. Passing the raw -1 into ``bpe()`` would silently
    yield BPE_O=1 (an FP8-sized O buffer) for a half-precision output."""
    return params.dtype_qkv if params.dtype_o < 0 else params.dtype_o


def tile_k_hw(dtype_qkv: int) -> int:
    """HW k-step in elements.

    **This is the single most arch-sensitive value in the file.** Rubin runs the
    dense 2xFP8 MMA at K=64 per instruction and pairs it with idesc ``k_dim=1``;
    Blackwell has no K=64 FP8 path and uses K=32 with ``k_dim=0``. The pairing is
    arch-OPPOSITE, and a mismatch does not fail — it silently scrambles rows of
    the accumulator (see rules/mma-tma-matrix.md S1). The kernel bodies here
    hardcode ``k_dim=1``, so FP8/MXFP8 must be 64.

    BF16/FP16 use 16 on both arches: the 2-chunk (K=32) f16 path is silently
    wrong on SM10x.
    """
    return 64 if dtype_qkv <= _DTYPE_E5M2 else 16


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    """V is split along d_v under cga2, so its swizzle follows the PER-CTA inner
    stride; Q/K/O use the full-tile inner extent."""
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} is not a multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int) -> int:
    return 128 if (tile_o * bpe_o) % 128 == 0 else 64


def rescale_threshold(dtype_qkv: int) -> float:
    """4.0 for FP8/MXFP8 (low precision, rescale aggressively); 8.0 for
    BF16/FP16 (defer rescale to amortize the alpha == 1 fast path)."""
    return 4.0 if dtype_qkv <= _DTYPE_E5M2 else 8.0


def sdpa_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    tile_o: int,
    tiles_q: int,
    stages_kv: int,
    cta_mma: int,
    bpe_qkv: int,
    bpe_o: int,
    qo_alias: bool = False,
) -> int:
    """Q/K/V/O SMEM footprint for the classic prefill pipeline.

    K (seq rows) and V (d_v cols) shrink by CTA_MMA; Q/O are per-CTA full.
    ``qo_alias`` = O reuses Q's slab in the epilogue (max instead of sum).
    """
    s_q = tiles_q * (tile_m * tile_k) * bpe_qkv
    s_o = tiles_q * (tile_m * tile_o) * bpe_o
    s_k = stages_kv * (tile_n * tile_k // cta_mma) * bpe_qkv
    s_v = stages_kv * (tile_o * tile_n // cta_mma) * bpe_qkv
    return (max(s_q, s_o) if qo_alias else s_q + s_o) + s_k + s_v


@dataclass(frozen=True)
class TmaIters:
    """TMA inner-loop iteration counts and per-iteration element granularity."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    """TMA inner-loop iters when d*BPE exceeds the swizzle atom: each TMA-tile
    load expands into inner_bytes/swz_b sub-loads stepped by swz_b/BPE elems."""
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"TMA inner bytes {inner_bytes} is not a multiple of swizzle {swz_b}")
    return inner_bytes // swz_b


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, cfg.BPE, cfg.V_SWZ_BYTES)
    return TmaIters(QK_ITERS=qk, VO_ITERS=vo, QK_GRANU_ELEMS=cfg.TILE_K // qk, VO_GRANU_ELEMS=cfg.TILE_O // vo)


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding. The kernels fold
    trace-time branches on these bits; the band VALUES ride WINDOW_LEFT /
    WINDOW_RIGHT (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present:
        flags |= MASK_PADDED
    return flags


def _validate_params(flavor: str, k: TemplateParams) -> None:
    """Guard the TemplateParams a Rubin flavor can express. Every rejection here
    must also be a Capabilities decline — reaching this is an engine-row bug."""
    if k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_qkv must be 0=E4M3/1=E5M2/2=BF16/3=FP16 (got {k.dtype_qkv}); Rubin has no TF32 prefill kernel")
    dtype_o = resolve_dtype_o(k)
    if dtype_o not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_o must be 0..3 (got {k.dtype_o})")
    if k.dtype_qkv > _DTYPE_E5M2 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires dtype_o == dtype_qkv; got dtype_o={dtype_o}")
    if k.sched_policy not in (None, SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: sched_policy must be NATURAL/LPT/LPT_L2 or None (got {k.sched_policy})")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.split_kv and k.split_kv > 1:
        raise ValueError(f"{flavor}: split_kv > 1 is not wired in the SM107 kernels (no SplitHelpers)")
    # THD/varlen is per-FLAVOR on the Rubin line, not per-dtype.  Every
    # QUANTIZED flavor carries it; on the f16/bf16 side only the flavors whose
    # BODY has been ported to the FROST setup-kernel contract do -- the rest
    # still call it with the pre-upstream 7-arg signature against a 14-arg
    # helper, and allocate the 3B+2 metadata buffer where the SHARED decode
    # (_common_blackwell._thd_decode) reads 4B+4 with a batch_remap.  That
    # mismatch is a HANG or a wrong batch, not an arity error, so it is declined
    # here rather than left to fail deep in a trace.
    if k.thd_varlen and k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2) and flavor not in _F16_THD_FLAVORS:
        raise ValueError(
            f"{flavor}: THD/varlen on the SM107 f16/bf16 line is served by {sorted(_F16_THD_FLAVORS)} only "
            f"(got dtype_qkv={k.dtype_qkv}); the other flavors' setup-kernel call sites are not ported"
        )


def _band_fields(params: TemplateParams) -> Tuple[int, int, int, int, int]:
    """(MASK_FLAGS, WINDOW_LEFT, WINDOW_RIGHT, BOTTOM_RIGHT, HAS_SINK).

    WINDOW_* are 0 when their mask bit is unset. Passing window_left without
    window_right is silently wrong rather than partial — the KV-tile trim already
    uses WINDOW_RIGHT, so the element-level mask must receive it too."""
    return (
        _mask_flags_from(params),
        params.window_left or 0,
        params.window_right or 0,
        int(params.bottom_right),
        int(params.has_sink),
    )


# ---------------------------------------------------------------------------
# Shared field surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CfgSm107:
    """Fields every Rubin prefill flavor carries.

    Per-flavor factories set every field explicitly; the defaults exist so the
    dataclass is constructible in a doc/test context, not as a fallback.
    """

    # --- tile geometry
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128  # d_qk
    TILE_O: int = 128  # d_v

    # --- dtypes
    DTYPE_QKV: int = _DTYPE_FP16  # 0=E4M3 1=E5M2 2=BF16 3=FP16
    DTYPE_O: int = _DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    # --- cluster
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    SPLIT_PIPELINE: int = 0

    # --- swizzles
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    # --- MMA k-step (arch-sensitive; see tile_k_hw)
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    # --- pipeline depth
    TILES_Q: int = 2
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 4

    # --- warp roles
    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    # --- graph-derived features
    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64

    # --- warp population (READ_TILE_ARRIVERS must equal the number of warps that
    # actually call read_tile_id_arrive; a mismatch is an unreachable mbarrier
    # count, i.e. a hang at every shape)
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15

    # --- mbarrier lane constants
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 15


@dataclass(frozen=True)
class CfgD128(_CfgSm107):
    """d_qk = d_v = 128 (the classic 2-warpgroup pipeline)."""

    QO_ALIAS: int = 0


@dataclass(frozen=True)
class CfgD192(CfgD128):
    """d_qk = 192, d_v = 128. Same pipeline as d128; Q/K swizzle drops to 64 B at
    FP8 because TILE_K * 1 = 192 is not a multiple of 128."""


@dataclass(frozen=True)
class CfgD256(_CfgSm107):
    """d_qk = d_v = 256. Q-union-O SMEM alias, single softmax warpgroup, a
    Q.K(i+1) -> S.V(i) lookahead MMA stream with two parity S_acc TMEM slots."""


@dataclass(frozen=True)
class CfgD512(_CfgSm107):
    """d_qk = d_v = 512, cga4x1 role-split: the cluster's 4 CTAs form two 2-CTA
    sub-groups (sg0 = BMM1 + softmax + ship, sg1 = recv + correction + BMM2)."""

    XFER_STAGES: int = 2
    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 2


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_regs(cfg, flavor: str) -> None:
    """The four hardware register constraints. Equality and alignment are HW
    requirements, not style: violating them gives wrong results or a crash, with
    no diagnostic."""
    _check(
        [
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS,
                f"{flavor}: MMA/TMALDG/TMASTG/SCHEDULER regs must be equal",
            ),
            (
                cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512,
                f"{flavor}: register budget over 512",
            ),
            (
                all(r % 8 == 0 for r in (cfg.MMA_REGS, cfg.CORRECTION_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8",
            ),
            (
                all(24 <= r <= 256 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS) if r),
                f"{flavor}: per-warp register counts must be within 24..256",
            ),
        ]
    )


def _validate_warp_layout(cfg, flavor: str, *, expect_total: int, expect_arrivers: int) -> None:
    """Pin the warp population against the count the kernel BODY implements.

    This is the check whose absence hung the d256 FP8 port: the body's role
    dispatch and its ``read_tile_id_arrive`` call sites are fixed, so a config
    that disagrees produces an mbarrier init count nothing can reach."""
    _check(
        [
            (cfg.TOTAL_WARPS == expect_total, f"{flavor}: kernel body has {expect_total} warps, config says {cfg.TOTAL_WARPS}"),
            (cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32, f"{flavor}: THREADS_PER_CTA must be TOTAL_WARPS*32"),
            (
                cfg.READ_TILE_ARRIVERS == expect_arrivers,
                f"{flavor}: READ_TILE_ARRIVERS must be {expect_arrivers} for this body (got {cfg.READ_TILE_ARRIVERS}); "
                f"a wrong count is an unreachable scheduler mbarrier, i.e. a hang at EVERY shape",
            ),
        ]
    )


def _validate_dtype_k_step(cfg, flavor: str) -> None:
    """FP8/MXFP8 must ride Rubin's K=64 dense path (the bodies hardcode idesc
    k_dim=1); f16/bf16 must stay 1-chunk at 16."""
    want = tile_k_hw(cfg.DTYPE_QKV)
    _check(
        [
            (
                cfg.TILE_K_HW_BMM1 == want and cfg.TILE_K_HW_BMM2 == want,
                f"{flavor}: TILE_K_HW must be {want} at DTYPE_QKV={cfg.DTYPE_QKV} on Rubin "
                f"(got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}); a mismatch against the body's "
                f"idesc k_dim silently scrambles accumulator rows",
            ),
        ]
    )


def _validate_smem(cfg, flavor: str, *, qo_alias: bool) -> None:
    used = (
        sdpa_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.TILES_Q, cfg.STAGES_KV, cfg.CTA_MMA, cfg.BPE, cfg.BPE_O, qo_alias=qo_alias)
        + _SMEM_FIXED_OVERHEAD
    )
    if used > SMEM_USABLE_BYTES:
        raise ValueError(
            f"{flavor}: SMEM {used // 1024} KiB (Q/K/V/O + {_SMEM_FIXED_OVERHEAD // 1024} KiB fixed) exceeds the "
            f"{SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout at STAGES_KV={cfg.STAGES_KV}, CTA_MMA={cfg.CTA_MMA}. "
            f"Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated (measured: O came back "
            f"50 % zeros with an exact LSE), so this must raise here"
        )


def _validate_swizzles(cfg, flavor: str) -> None:
    _check(
        [
            (cfg.Q_SWZ_BYTES in (64, 128), f"{flavor}: Q swizzle must be 64/128 B"),
            (cfg.K_SWZ_BYTES in (64, 128), f"{flavor}: K swizzle must be 64/128 B"),
            (cfg.V_SWZ_BYTES in (32, 64, 128), f"{flavor}: V swizzle must be 32/64/128 B"),
            (cfg.O_SWZ_BYTES in (64, 128), f"{flavor}: O swizzle must be 64/128 B"),
        ]
    )


def _validate_masks(cfg, flavor: str) -> None:
    _check(
        [
            (
                not (cfg.BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)),
                f"{flavor}: bottom-right alignment requires a causal band",
            ),
            (
                not (cfg.THD_VARLEN and not (cfg.MASK_FLAGS & MASK_PADDED)),
                f"{flavor}: THD/varlen implies per-sequence padded masking",
            ),
            (
                not (cfg.THD_VARLEN and not cfg.SEQ_KV_LENS_PRESENT),
                f"{flavor}: THD/varlen must force SEQ_KV_LENS_PRESENT=1, or every sequence attends the whole pack",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# d128 / d192 -- the classic 2-warpgroup pipeline (16 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 8 softmax (2 warpgroups) + 4 correction + MMA + TMALDG + TMASTG +
# scheduler.  15 of the 16 warps arrive on the scheduler ring (all but the
# scheduler warp itself).


_D128_TOTAL_WARPS = 16
_D128_READ_TILE_ARRIVERS = 15


def _stages_kv_d128(dtype_qkv: int, cta_mma: int, *, mxfp8: bool, tile_k: int) -> int:
    """Rubin d128-family ring depth.

    Per-tensor FP8 at d128/cga2 runs a **9-stage** ring: a Rubin-specific tuning
    carried by the shipped ``sm107/prefill_d128_fp8.py`` (which previously spelled
    it as a post-hoc ``dataclasses.replace``), and the reason that kernel needs
    the oversized-SMEM launch mode.

    It is deliberately narrow. At cga1 the per-CTA K/V slabs double and 9 stages
    need 352 KiB, past the 327 KiB cap; at d192 (``tile_k=192``) the Q/K slabs
    grow and the tuning was never measured. Both fall back to 4, which is what
    the kernel bodies were validated with.
    """
    if dtype_qkv <= _DTYPE_E5M2:
        if not mxfp8 and tile_k == 128 and cta_mma == 2:
            return 9
        return 4
    # d192's wider Q/K slabs do not leave room for a 4-deep ring: Q/K/V/O alone
    # sum to exactly 320 KiB, which overflows the usable carveout once barriers
    # are counted. Measured on Rubin at n_kv=2: depth 4 -> 50 % of O zeroed,
    # cos = 0.0015; depth 2 -> cos = 1.000000. d128 at depth 4 sums to 256 KiB
    # and passes every n_kv, so this is a d192 constraint, not a family one.
    if tile_k == 192:
        return 2
    return 2 * cta_mma


def _make_cfg_d128_family(params: TemplateParams, *, flavor: str, tile_k: int, tile_o: int, mxfp8: bool):
    _validate_params(flavor, params)
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_n = 128
    stages_kv = _stages_kv_d128(params.dtype_qkv, cta_mma, mxfp8=mxfp8, tile_k=tile_k)
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    cls = CfgD192 if tile_k == 192 else CfgD128

    cfg = cls(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=2,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=2,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=192,
        CORRECTION_REGS=88,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D128_TOTAL_WARPS,
        THREADS_PER_CTA=_D128_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=4,
        CORR_WARP_BASE=8,
        MMA_WARP_ID=12,
        TMALDG_WARP_ID=13,
        TMASTG_WARP_ID=14,
        SCHED_WARP_ID=15,
        READ_TILE_ARRIVERS=_D128_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D128_TOTAL_WARPS, expect_arrivers=_D128_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _validate_smem(cfg, flavor, qo_alias=bool(cfg.QO_ALIAS))
    return cfg, _tma_iters(cfg)


def make_cfg_d128(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128", tile_k=128, tile_o=128, mxfp8=False)


def make_cfg_d128_mxfp8(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128 mxfp8", tile_k=128, tile_o=128, mxfp8=True)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, f16 and per-tensor FP8.

    Same family as ``make_cfg_d128`` with a wider K: the pre-upstream base kernel
    was flavor-generic (one body served d128 and d192xd128 by swapping the config),
    and ``CfgD192`` only widens ``TILE_K``.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128", tile_k=192, tile_o=128, mxfp8=False)


def make_cfg_d192_mxfp8(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, block-scale MXFP8.

    CGA2 ONLY, and that is a DESCRIPTOR constraint rather than a tuning choice.
    At ``cta_mma=1`` the K/V rings are not halved, so the four scale-factor tiles
    -- allocated last -- start at 256/258/276/278 KiB, i.e. past the **256 KiB
    version-0 tcgen05 descriptor window**.  A version-0 SF descriptor there wraps
    to offset 0 and the UTCCP copies Q DATA bytes into the SF TMEM columns:
    ``LSE = +inf`` and ``O = NaN`` on 100 % of cells, at every shape (the exact
    d512 MXFP8 failure in rules/mma-tma-matrix.md S6).  At ``cta_mma=2`` the
    highest slab sits at 200 KiB and version 0 is provably safe.

    So the Rubin MXFP8 engine row deliberately declares NO ``cgas_by_d_shape``
    entry for (192, 128), leaving it on the row default ``cgas={2}``.  Lifting
    that needs ``DESC_VERSION`` derived from the layout AND the version-1 SF path
    validated on Rubin -- which is NOT a free widening: setting
    ``desc_version=1`` on the d128/d256 MXFP8 tiles turned 21 green tests red
    (2026-09-08), so the bit is not a transparent superset.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128 mxfp8", tile_k=192, tile_o=128, mxfp8=True)


# ---------------------------------------------------------------------------
# d256 -- Q-union-O alias, ONE softmax warpgroup (12 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 softmax + 4 correction + MMA + TMALDG + TMASTG + scheduler.
# The arriver count is per-CTA-group: ((SOFTMAX_WG*4) + CORR + 2) * CGA_SIZE
# + (CGA_M / CTA_MMA).

_D256_TOTAL_WARPS = 12


def _d256_read_tile_arrivers(cta_mma: int) -> int:
    return ((1 * 4) + 4 + 2) * (cta_mma * 1) + (cta_mma // cta_mma)


def _make_cfg_d256_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    _validate_params(flavor, params)
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 256
    tile_n = 128
    # STAGES_KV is a TUNING KNOB again (2..4), and the note that said otherwise
    # was WRONG about why.
    #
    # It used to be pinned to 2, blaming a body that "conflates the KV ring
    # index with the 2-slot S_acc parity". That diagnosis does not survive
    # reading the body: every parity site derives parity from the ABSOLUTE
    # `kv_loop & 1`, and K/V ride separate PipelineStates advanced in lockstep
    # with the TMA warp, so the body is depth-agnostic.
    #
    # The real cause was the >256 KiB tcgen05 descriptor wrap. Buffers are laid
    # out sQO | sK[stages] | sV[stages]; at depth 4 the last two V stages start
    # at 256 KiB and 288 KiB, and a VERSION-0 descriptor truncates
    # `start_address` to 14 bits, so those stages alias the bottom of SMEM. The
    # answer goes wrong from the KV iteration that first touches a wrapped
    # stage -- which is why it looked like a ring bug and why it only shows up
    # at S_kv > 256.
    #
    # Measured on Rubin, d256 f16, cos vs an fp32 reference, n_kv = 2/3/4/8:
    #   depth 4, desc v0:  1.0000 / 0.6806 / 0.4980 / 0.4640   <- the old data
    #   depth 4, desc v1:  1.0000 / 1.0000 / 1.0000 / 1.0000   dense AND causal
    #   depth 3, desc v0:  1.0000 / 1.0000 / 1.0000 / 1.0000   (stays under)
    # `prefill_d256_f16.DESC_VERSION` is now DERIVED from the layout, so any
    # depth that fits SMEM is correct by construction. Do not re-literal it.
    stages_kv = params.stages_kv if getattr(params, "stages_kv", None) else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    arrivers = _d256_read_tile_arrivers(cta_mma)

    cfg = CfgD256(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        # The body is a SINGLE-warpgroup pipeline. Two warpgroups is the
        # Blackwell d256 split-P design and does not exist here.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=96,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D256_TOTAL_WARPS,
        THREADS_PER_CTA=_D256_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused at SOFTMAX_WARPGROUPS=1
        CORR_WARP_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        READ_TILE_ARRIVERS=arrivers,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D256_TOTAL_WARPS, expect_arrivers=arrivers)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _check(
        [
            (cfg.TILES_Q == 1, f"{flavor}: the d256 pipeline mandates TILES_Q == 1"),
            (cfg.SOFTMAX_WARPGROUPS == 1, f"{flavor}: the d256 pipeline mandates SOFTMAX_WARPGROUPS == 1"),
            (
                2 <= cfg.STAGES_KV <= 4,
                f"{flavor}: STAGES_KV must be in 2..4 (got {cfg.STAGES_KV}); the upper bound is the "
                f"{SMEM_USABLE_BYTES // 1024} KiB Rubin carveout, and the kernel derives its tcgen05 "
                f"descriptor version from the resulting layout so every depth in range is correct",
            ),
        ]
    )
    # d256 always Q-union-O aliases.
    _validate_smem(cfg, flavor, qo_alias=True)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256", mxfp8=False)


def make_cfg_d256_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256 mxfp8", mxfp8=True)


# ---------------------------------------------------------------------------
# d512 -- cga4x1 role split (8 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 compute (softmax on sg0 / correction on sg1) + MMA + TMALDG +
# TMASTG + scheduler.  READ_TILE_ARRIVERS counts across the whole cluster:
#   compute warps (4/CTA, both sgs)          = 4 * CGA_M * CGA_N          = 16
#   TMALDG        (1/CTA, both sgs)          = 1 * CGA_M * CGA_N          =  4
#   sg0 MMA leader (lead CTA only)           = (CGA_M*CGA_N)/(2*CTA_MMA)  =  1
#   sg1 MMA all   (P12 forwarder + leader)   = (CGA_M*CGA_N)/2            =  2
#   sg1 TMASTG    (sg1 CTAs only)            = CTA_MMA * CGA_N            =  2
#                                                                   total = 25

_D512_TOTAL_WARPS = 8
_D512_CGA_M, _D512_CGA_N, _D512_CTA_MMA = 4, 1, 2
_D512_READ_TILE_ARRIVERS = (
    ((1 * 4) + 1) * (_D512_CGA_M * _D512_CGA_N)
    + (_D512_CGA_M * _D512_CGA_N) // (2 * _D512_CTA_MMA)
    + (_D512_CGA_M * _D512_CGA_N) // 2
    + _D512_CTA_MMA * _D512_CGA_N
)


def _make_cfg_d512_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    _validate_params(flavor, params)
    if params.cta_mma != _D512_CTA_MMA:
        raise ValueError(f"{flavor}: the role-split pipeline is cga4x1 / CTA_MMA=2 only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 512
    tile_n = 128
    is_fp8 = params.dtype_qkv <= _DTYPE_E5M2
    # FP8 (BPE=1) fits a deeper K/V ring and a 4-deep cross-sg P ring; f16/bf16
    # (BPE=2) are SMEM-cap driven down to 2/2.
    stages_kv = 3 if is_fp8 else 2
    xfer_stages = 4 if is_fp8 else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)

    cfg = CfgD512(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=_D512_CGA_M,
        CGA_N=_D512_CGA_N,
        CTA_MMA=_D512_CTA_MMA,
        SPLIT_PIPELINE=1,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, _D512_CTA_MMA, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        XFER_STAGES=xfer_stages,
        # One 4-warp compute group; correction is fused into it, sub-group
        # conditional, so there are no separate correction warps or registers.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=0,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        BMM2_N_PER_CALL=256,
        BMM2_LOOP_N_BLOCKS=tile_o // 256,
        N_BMM2_CHUNKS=2,
        BMM2_CHUNK_SIZE=256,
        TOTAL_WARPS=_D512_TOTAL_WARPS,
        THREADS_PER_CTA=_D512_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused under the role split
        CORR_WARP_BASE=0,  # correction fused with the compute group
        MMA_WARP_ID=4,
        TMALDG_WARP_ID=5,
        TMASTG_WARP_ID=6,
        SCHED_WARP_ID=7,
        READ_TILE_ARRIVERS=_D512_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D512_TOTAL_WARPS, expect_arrivers=_D512_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _check([(cfg.SPLIT_PIPELINE == 1, f"{flavor}: the d512 flavor is role-split (SPLIT_PIPELINE=1)")])
    return cfg, _tma_iters(cfg)


def make_cfg_d512(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    return _make_cfg_d512_family(params, flavor="sm107 d512", mxfp8=False)


def make_cfg_d512_mxfp8(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    return _make_cfg_d512_family(params, flavor="sm107 d512 mxfp8", mxfp8=True)
