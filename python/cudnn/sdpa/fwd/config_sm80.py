# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time flavor configs for the SM80 SDPA forward kernels.

One frozen dataclass per kernel flavor (the model families the head-dim
envelopes are named after), all served by the same kernel skeleton
(``kernels/sm80/prefill_f16.py``; qwen d=256 by ``sm80/prefill_d256_f16.py``).
Knob provenance per flavor:

- **gptoss** (d_qk = d_v = 64): picked from the A100-PCIE-40GB perf sweep. At
  d=64 the SV mma has 8 n_frags instead of 16, so M_BLOCKS=2 per warp fits the
  register budget that spills at d=128. Best point ``(tile_m=128, tile_n=64,
  num_warps=4)`` matches FA to within ~5-7 % on H=32 H_kv=2 GQA shapes at
  SQ ∈ {2K, 4K, 8K}. For causal the sweep additionally wants ``--sched lpt_l2
  --sched-l2-mib 16`` on top — driver-side override, not pinned here.
- **llama** (d_qk = d_v = 128): swept on A100-PCIE-40GB; ``(tile_m=128,
  tile_n=64, num_warps=8)`` matches Phase-3 baseline of ~97 % FA-2.8.3 at
  SQ=8K — keep unless a perf-node sweep finds a better point.
- **dsv3** (d_qk = 192, d_v = 128): first flavor with d_qk != d_v; the shared
  skeleton's K vs V row-stride plumbing handles the asymmetry without a fork.
  Knobs are an initial guess — not swept yet; the perf-node sweep will pin it.
- **qwen** (d_qk = d_v = 256): SMEM at the pinned point: sQ_buf (64 KiB,
  aliased with the sV ring after the prologue Q→reg) + sK_buf (64 KiB,
  2-stage) = 128 KiB, under A100's 164 KiB opt-in. ~240-250 regs/thread —
  under SM80's 256/thread cap, but only at m_blocks = 1: TILE_M must equal
  num_warps*16 at d=256 (any larger m_blocks pushes O_acc + Q_frag over).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int
    D_V: int
    TILE_M: int
    TILE_N: int
    NUM_WARPS: int


GPTOSS_CFG = Cfg(D_QK=64, D_V=64, TILE_M=128, TILE_N=64, NUM_WARPS=4)
LLAMA_CFG = Cfg(D_QK=128, D_V=128, TILE_M=128, TILE_N=64, NUM_WARPS=8)
DSV3_CFG = Cfg(D_QK=192, D_V=128, TILE_M=128, TILE_N=64, NUM_WARPS=8)
QWEN_CFG = Cfg(D_QK=256, D_V=256, TILE_M=128, TILE_N=64, NUM_WARPS=8)


# ---------------------------------------------------------------------------
# TemplateParams — the compile-time identity of one SM80 kernel specialization.
#
# Mirrors config_sm100/config_sm120: a frozen, hashable record injected into
# the kernel template as the ``FROST_TEMPLATE_PARAMS`` module global by
# ``frost.template_loader.load_template``, so ``cutlass.const_expr`` folding
# specializes the traced code per parameter set. Everything here is PLAN-TIME
# data (graph declaration + capability row + knobs) — never a runtime tensor
# value (AGENTS.md Hard Rule 4). Shape axes (b/h/sq/skv, the actual head dim
# under the envelope, the SWA width) stay arguments of the template module's
# ``compile()`` and its per-shape lru cache; THD packed token totals compile
# DYNAMIC (``cute.sym_int``) there and are never part of any key.
# ---------------------------------------------------------------------------
from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

_SM80_SEQ_TILES_N = (64, 128)


@dataclass(frozen=True)
class TemplateParams:
    """One SM80 prefill kernel specialization (the module-identity axes)."""

    # dtype: fp16 or bf16 I/O (one mma pipeline serves both).
    io_bf16: bool = False
    # Flavor envelope tile dims (the compile-time D box; the actual head dim
    # is a compile() argument and may be smaller — loads zero-fill past it).
    d_qk: int = 128
    d_v: int = 128
    # Tile geometry (swept-and-frozen per flavor; see the Cfg table above).
    tile_m: int = 128
    num_warps: int = 8
    tile_n: int = 64
    # Mask family. right_bound stays a RUNTIME argument (it widens the causal
    # band without changing the traced structure).
    is_causal: bool = False
    has_swa: bool = False
    causal_bottom_right: bool = False
    # Optional operands (compile-time ABI presence; a missing-but-required or
    # provided-but-uncompiled operand raises at execute — Hard Rule 1).
    has_seq_kv_lens: bool = False
    has_seq_q_lens: bool = False
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)).
    stats_log2: bool = False
    has_bias: bool = False
    bias_is_fp32: bool = False
    has_rope: bool = False
    # Packed varlen (wrapper-only today; the engine row declares thd=False).
    thd_varlen: bool = False
    # Tile-scheduler policy, in the SHARED frost vocabulary
    # (tile_dsl.constants.SCHED_*; the kernel's grid mapping interprets
    # NATURAL as its plain 3-D grid). The L2 budget for SCHED_LPT_L2 is
    # flavor-tuned plan-time data, so it rides here too.
    sched_policy: int = SCHED_NATURAL
    sched_l2_mib: int = 32
    # False compiles the LSE store out entirely (the template None-specializes
    # the LSE argument) — a stats-less graph binds no LSE buffer at any level.
    has_lse: bool = True


def validate_params(p: TemplateParams) -> None:
    """Raising validator — a failure here means the capability row or the
    adapter lied about what this template can serve (README Rule 2)."""
    if p.tile_n not in _SM80_SEQ_TILES_N:
        raise ValueError(f"sm80: tile_n must be one of {_SM80_SEQ_TILES_N}; got {p.tile_n}")
    if p.num_warps not in (4, 8):
        raise ValueError(f"sm80: num_warps must be 4 or 8; got {p.num_warps}")
    if p.tile_m % (p.num_warps * 16) != 0:
        raise ValueError(f"sm80: tile_m ({p.tile_m}) must be a multiple of num_warps*16 ({p.num_warps * 16})")
    if p.d_qk % 16 != 0 or p.d_qk <= 0:
        raise ValueError(f"sm80: template d_qk must be a positive multiple of 16 (m16n8k16 K); got {p.d_qk}")
    if p.d_v % 16 != 0 or p.d_v <= 0:
        raise ValueError(f"sm80: template d_v must be a positive multiple of 16 (cp.async + STG.128 epilogue); got {p.d_v}")
    if (p.d_qk, p.d_v) not in {(cfg.D_QK, cfg.D_V) for cfg in (GPTOSS_CFG, LLAMA_CFG, DSV3_CFG, QWEN_CFG)}:
        raise ValueError(f"sm80: (d_qk, d_v) = ({p.d_qk}, {p.d_v}) is not a swept flavor envelope")
    if p.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"sm80: sched_policy must be a tile_dsl SCHED_* value; got {p.sched_policy}")
    if p.sched_l2_mib <= 0:
        raise ValueError(f"sm80: sched_l2_mib must be > 0; got {p.sched_l2_mib}")
    if p.causal_bottom_right and not (p.is_causal or p.has_swa):
        raise ValueError("sm80: causal_bottom_right requires is_causal and/or has_swa (nothing to align otherwise)")
    if p.thd_varlen and (p.has_rope or p.has_seq_kv_lens or p.has_seq_q_lens or p.has_bias):
        raise ValueError("sm80: THD carries lengths via cu_seqlens; rope / bias / dense seq-lens are dense-only")


def params_for_flavor(flavor: str, **overrides) -> TemplateParams:
    """A TemplateParams seeded from one swept flavor's Cfg row."""
    cfg = {"gptoss": GPTOSS_CFG, "llama": LLAMA_CFG, "dsv3": DSV3_CFG, "qwen": QWEN_CFG}[flavor]
    base = dict(d_qk=cfg.D_QK, d_v=cfg.D_V, tile_m=cfg.TILE_M, num_warps=cfg.NUM_WARPS, tile_n=cfg.TILE_N)
    base.update(overrides)
    p = TemplateParams(**base)
    validate_params(p)
    return p


# ---------------------------------------------------------------------------
# SM89 (Ada / L20) — the same kernel skeleton on a 99 KiB-SMEM part.
#
# The SM80 row is pinned to cc 8.0 exactly because the *shared* skeleton has to
# cover the d=256 flavor, whose pinned point needs sQ_buf (64 KiB) + sK_buf
# (64 KiB) = 128 KiB, inside A100s 164 KiB opt-in SMEM limit. An Ada part
# ---------------------------------------------------------------------------
# SM89 (Ada / L20) — the same kernel skeleton on a 99 KiB-SMEM part.
#
# The SM80 row is pinned to cc 8.0 exactly because the shared skeleton has to
# cover the d=256 flavor, whose pinned point needs sQ_buf (64 KiB) + sK_buf
# (64 KiB) = 128 KiB, inside A100's 164 KiB opt-in SMEM limit.  An Ada part
# exposes 99 KiB per block (101376 B on the L20, queried), so only flavors whose
# allocation fits can be served there.  d=64 routes to ``prefill_f16`` (the
# shared skeleton, NOT ``prefill_d256_f16``); its pinned gptoss point
# (tile_m=128, tile_n=64, num_warps=4) allocates
#
#     sQ_buf = max(tile_m*d_qk, 2*tile_n*d_v, tile_m*d_v) = 8192 elems = 16 KiB
#     sK_buf = 2*tile_n*d_qk                                = 8192 elems = 16 KiB
#                                                        total = 32 KiB
#
# which is not the binding constraint on Ada at all.  The flavor geometry is
# therefore identical to the SM80 gptoss point; only the device gate and the
# advertised capability box differ.  Nothing here widens the SM80 gate: the
# SM80 row stays cc 8.0 exactly and keeps serving A100.
# ---------------------------------------------------------------------------

SM89_SUPPORTED_FLAVORS = ("gptoss",)

# The template box the SM89 row lowers onto: d64 gptoss, same as A100's pinned
# point.  Kept as a name here so a later Ada sweep can move it in one place.
SM89_FLAVOR = "gptoss"


def smem_bytes(d_qk: int, d_v: int, tile_m: int, tile_n: int, elem_bytes: int = 2) -> int:
    """Static SMEM a specialization of ``prefill_f16`` allocates.

    Mirrors the allocation site verbatim (``SQ_BUF_ELEMS`` / ``SK_BUF_ELEMS``)
    so the plan-time resource gate and the kernel cannot drift:
    ``(max(tile_m*d_qk, 2*tile_n*d_v, tile_m*d_v) + 2*tile_n*d_qk) * elem_bytes``.
    A lower bound on the real footprint (alignment padding excluded).
    """
    elems_q = tile_m * d_qk
    elems_kv_v = tile_n * d_v
    elems_kv_k = tile_n * d_qk
    return (max(elems_q, 2 * elems_kv_v, tile_m * d_v) + 2 * elems_kv_k) * elem_bytes


def sm89_smem_bytes(params: "TemplateParams") -> int:
    """SMEM for one SM89 TemplateParams.  Reads the FLAVOR head dims (the compile
    box), not the graph's actual head dim: the allocation is a compile-time
    constant and does not shrink for a head dim below the box."""
    return smem_bytes(params.d_qk, params.d_v, params.tile_m, params.tile_n)
