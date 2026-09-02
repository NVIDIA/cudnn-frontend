# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time flavor configs for the SM80 SDPA backward kernel.

Single source of truth for the shared 2-sub-group backward pipeline
(``kernels/bprop_f16_sm80.py``); all shape-dependent sizes are derived in the
kernel from these fields — do NOT inline ``128`` / ``64`` literals there.
The pipeline shape the knobs describe:

  * 1 CTA owns one **KV-tile** (TILE_KV rows of K/V) for a (batch, head);
    the CTA loops over Q-tiles (TILE_Q rows of Q/dO/O).
  * Two sub-groups of ``WARPS_PER_SG`` warps each (256 threads at the default
    4+4 split): sg0 runs ``S = K·Qᵀ`` + softmax + ``dV = P·dO``; sg1 runs
    ``dP = V·dOᵀ`` + dSoftmax + ``dK = dS·Q``. Both groups split ``dQ`` by
    d-col — the swizzle is handled d-agnostically via
    ``load_b_smem_x4(col_base=...)``.

Flavors (d is read from the tensor shapes; the kernel runs both unchanged):

- **llama** (d_qk = d_v = 128): TILE_Q=64.
- **gptoss** (d_qk = d_v = 64): at d=64 the per-thread register / SMEM
  footprint is roughly half of llama's, which buys a **128-row Q-tile**:
  TILE_Q=128 halves the number of Q-iters, so the fixed per-iter overhead
  (5 CTA barriers + Q/dO reload + do_dot/LSE reads) is amortized over 2× the
  MMA work — the dominant cost at d=64 where each tile's BMM flops halve.
  The dQ MMA then runs DQ_M_BLOCKS = TILE_Q // (WARPS_PER_SG*16) = 2 m-blocks
  per warp (the same M_BLOCKS trick the forward gptoss kernel uses at d=64).
  SMEM at TILE_Q=128 is ~144 KiB < the A100 163 KiB dynamic-SMEM cap. SQ must
  be a multiple of TILE_Q=128; SKV a multiple of TILE_KV=64.
  NOTE: the shipped gptoss-sm80 (d=64) BPROP kernel is
  ``kernels/bprop_d64_f16_sm80.py``, which uses a FIXED tile shape
  (m_block=64, n_block=128) and does not read this config. GPTOSS_CFG
  configures the shared path when run at d=64.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int  # head dim for Q/K (= matmul K-dim of BMM1)
    D_V: int  # head dim for V/dO/O (= N of dV, dP K-dim of BMM2)
    TILE_KV: int  # KV rows owned per CTA (M of dV/dK; K-reduce of dQ)
    TILE_Q: int  # Q rows per inner iter (N of S/dP; M of dQ)
    WARPS_PER_SG: int  # warps per sub-group → 2 sub-groups = 8 warps = 256 thr


LLAMA_CFG = Cfg(D_QK=128, D_V=128, TILE_KV=64, TILE_Q=64, WARPS_PER_SG=4)
GPTOSS_CFG = Cfg(D_QK=64, D_V=64, TILE_KV=64, TILE_Q=128, WARPS_PER_SG=4)


# ---------------------------------------------------------------------------
# TemplateParams — the compile-time identity of one SM80 backward kernel
# specialization. Mirrors fwd/config_sm80.py (#689) and config_sm120: a
# frozen, hashable record injected into the kernel template as the
# ``FROST_TEMPLATE_PARAMS`` module global by ``frost.template_loader``.
# Everything here is PLAN-TIME data — never a runtime tensor value (Hard
# Rule 4). Shape axes (b/h/hk/sq/skv, the actual head dims under the flavor
# envelope, the SWA width) stay arguments of the template module's
# ``compile()`` and its per-shape lru cache; THD packed token totals compile
# DYNAMIC (``cute.sym_int``) there and are never part of any key (issue #604).
# ---------------------------------------------------------------------------
from cudnn.frost.tile_dsl.constants import SCHED_LPT as _SCHED_LPT  # noqa: E402
from cudnn.frost.tile_dsl.constants import SCHED_NATURAL as _SCHED_NATURAL  # noqa: E402


@dataclass(frozen=True)
class TemplateParams:
    """One SM80 backward kernel specialization (the module-identity axes)."""

    # dtype: fp16 or bf16 I/O (one mma pipeline serves both).
    io_bf16: bool = False
    # Flavor envelope head dims (the compile-time D box; the actual dims are
    # compile() arguments and may be smaller — the host pads to the box).
    d_qk: int = 128
    d_v: int = 128
    # Backward pipeline geometry (see the Cfg table above; tile_kv must equal
    # warps_per_sg*16 and tile_q a multiple of it).
    tile_kv: int = 64
    tile_q: int = 64
    warps_per_sg: int = 4
    # Mask family. right_bound stays a RUNTIME argument (it widens the causal
    # band without changing the traced structure).
    is_causal: bool = False
    has_swa: bool = False
    causal_bottom_right: bool = False
    # Optional operands / outputs (compile-time ABI presence; a
    # missing-but-required or provided-but-uncompiled operand raises at
    # execute — Hard Rule 1).
    has_seq_kv_lens: bool = False
    has_seq_q_lens: bool = False
    has_bias: bool = False  # bias input => dBias accumulator output
    bias_is_fp32: bool = True
    bias_broadcast: bool = True  # bias batch dim 1 (broadcast) vs B
    has_sink: bool = False  # sinks input => dSink output (standalone reduction)
    has_rope: bool = False
    # Deterministic dQ: the kv-ordered gmem-semaphore relay (forces
    # SCHED_DEFAULT; the semaphore is carved caller scratch).
    deterministic: bool = False
    # Packed varlen (wrapper-only today; the engine row declares thd=False).
    thd_varlen: bool = False
    # Tile-scheduler policy in the SHARED frost vocabulary
    # (tile_dsl.constants.SCHED_*): the bwd grid interprets NATURAL as its
    # plain kv-major grid and LPT as the kv-major LPT remap (LPT_L2 is a
    # forward-only policy today).
    sched_policy: int = _SCHED_NATURAL


def validate_bwd_params(p: TemplateParams) -> None:
    """Raising validator — a failure here means the capability row or the
    adapter lied about what this template can serve."""
    if p.tile_kv != p.warps_per_sg * 16:
        raise ValueError(f"sm80 bwd: tile_kv ({p.tile_kv}) must equal warps_per_sg*16 ({p.warps_per_sg * 16})")
    if p.tile_q % (p.warps_per_sg * 16) != 0:
        raise ValueError(f"sm80 bwd: tile_q ({p.tile_q}) must be a multiple of warps_per_sg*16 ({p.warps_per_sg * 16})")
    if p.d_qk % 16 != 0 or p.d_v % 16 != 0 or p.d_qk <= 0 or p.d_v <= 0:
        raise ValueError(f"sm80 bwd: template head dims must be positive multiples of 16; got ({p.d_qk}, {p.d_v})")
    if p.d_qk < p.d_v:
        raise ValueError(f"sm80 bwd: d_qk ({p.d_qk}) must be >= d_v ({p.d_v}) (the per-sub-group split)")
    if (p.d_qk // 2) % 16 != 0:
        raise ValueError(f"sm80 bwd: d_qk//2 ({p.d_qk // 2}) must be a multiple of 16 (ldmatrix.x4 N//8 even)")
    if p.d_v % 32 != 0:
        raise ValueError(f"sm80 bwd: d_v ({p.d_v}) must be a multiple of 32 (do_dot warp reduce)")
    if p.has_rope and p.d_qk > 128:
        raise ValueError("sm80 bwd: RoPE requires d_qk <= 128 (the sDQ SMEM staging exceeds the A100 budget beyond that)")
    if p.sched_policy not in (_SCHED_NATURAL, _SCHED_LPT):
        raise ValueError(f"sm80 bwd: sched_policy must be SCHED_NATURAL or SCHED_LPT; got {p.sched_policy}")
    if p.deterministic and p.sched_policy != _SCHED_NATURAL:
        raise ValueError("sm80 bwd: deterministic dQ requires SCHED_NATURAL (the kv-ordered semaphore relay)")
    if p.causal_bottom_right and not (p.is_causal or p.has_swa):
        raise ValueError("sm80 bwd: causal_bottom_right requires is_causal and/or has_swa (nothing to align otherwise)")
    if p.thd_varlen and (p.has_bias or p.has_rope or p.has_sink or p.has_seq_kv_lens or p.has_seq_q_lens):
        raise ValueError("sm80 bwd: THD carries lengths via cu_seqlens; bias / rope / sink / dense seq-lens are dense-only")
    if p.thd_varlen and p.deterministic:
        # The dQ-relay semaphore is sized at compile time from the dense sq;
        # THD compiles sq as a dynamic sym_int so there is nothing to size it
        # from (the FE support surface already rejects deterministic + ragged).
        raise ValueError("sm80 bwd: deterministic dQ is dense-only (THD compiles sq dynamic; the relay semaphore has no plan-time size)")


def bwd_params_for_flavor(flavor: str, **overrides) -> TemplateParams:
    """A backward TemplateParams seeded from one flavor's bprop Cfg row.

    Only llama (the d<=128 shared-pipeline point) and gptoss (the d=64
    wide-Q-tile point) have swept bprop rows; dsv3/qwen ride the llama
    pipeline shape with their own envelope dims (the kernel derives
    qo_stages / drop-sDQ from d_qk).
    """
    cfg = {"llama": LLAMA_CFG, "gptoss": GPTOSS_CFG}.get(flavor, LLAMA_CFG)
    # Seed the envelope dims too, so a flavor name alone cannot yield tiles
    # from one flavor and head dims from another (callers with a different
    # envelope — dsv3/qwen on the llama pipeline — override d_qk/d_v).
    base = dict(d_qk=cfg.D_QK, d_v=cfg.D_V, tile_kv=cfg.TILE_KV, tile_q=cfg.TILE_Q, warps_per_sg=cfg.WARPS_PER_SG)
    base.update(overrides)
    p = TemplateParams(**base)
    validate_bwd_params(p)
    return p
