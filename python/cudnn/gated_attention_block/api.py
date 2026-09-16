# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, forward — SKELETON (no kernel is wired yet).

FROST's first MODEL-LEVEL block: a SET of FROST kernels behind ONE API, one
workspace, one call. Every stage is a FROST kernel; there are no cuBLAS or
cuDNN-graph call-outs for the GEMMs (an explicit scoping decision).

Geometry is Qwen3.5's gated attention (provenance in comments only — the shipped
name is by op geometry, per the FROST engine contract § 8).

.. note::

   **MAINTENANCE.** This docstring is the block's design record in the code, and
   it makes claims that stop being true the moment a stage lands or two stages
   merge. Update it in the SAME commit as: any stage becoming real, any fusion
   that deletes a stage or a workspace region, any dtype the block starts to
   serve, and any change to the ``qkvg_offsets`` / :class:`SavedForBackward`
   contracts. A stale design record is worse than none, because it gets trusted.
   The per-stage **Fusion status** paragraphs below are the parts that go stale
   first.

The op graph this file implements, in pipeline order::

    h [B, S, d_model]                                    (post input_layernorm)
     |
     |  (1) QKV+GATE projection   h @ W_qkvg^T
     +----------------------------------------------------------------+
     |  Q [B,S,H_q,D] | GATE [B,S,H_q,D] | K [B,S,H_kv,D] | V [B,S,H_kv,D]
     |
     |  (2)+(3) ONE kernel: QK-RMSNorm per head over D, then partial mRoPE on
     |          the first ROPE_DIM. Q and K only -- V is NOT normed. Dims
     |          [ROPE_DIM, D) pass through. fp32 throughout, ONE rounding.
     |
     |  (4) SDPA        O = softmax(QK^T * scale + mask) V        GQA H_q/H_kv
     |
     |  (5) gate        O_gated = O * sigmoid(GATE)     elementwise, BEFORE (6)
     |
     |  (6) out projection   O_gated @ W_o^T
     v
    out [B, S, d_model]

**FUSION KNOB (2026-09-11): ``fuse_gate=True`` folds stage (5) into stage (4)'s
epilogue.**  The sigmoid gate is a PRODUCTION feature of the Rubin d256 SDPA
kernels (``sdpa/fwd/kernels/sm107/prefill_d256_{f16,fp8}.py`` behind
``TemplateParams.epilogue_gate``; the engine rows claim it through
``epilogue_gate_d_shapes``): a TMA-staged GATE read + ``O *= sigmoid(GATE)``
after the dead-row select.  ``_Sdpa`` reaches it through the shipped adapter
(``SdpaFwdDslSm100(sample_gate=...)`` / ``execute(gate=...)``) -- the block
owns no SDPA kernel in ANY configuration.  Inference only (no pre-gate ``O``
for ``dG``); composes with ``fuse_norm_rope`` into the 3-launch block
proj(+norm+rope) -> sdpa(+gate) -> out_proj.

**FUSION KNOB (2026-09-11): ``fuse_norm_rope=True`` folds stages (2)+(3) into
stage (1)'s epilogue** (``_FusedQkvProjection`` over the fork
``kernels/proj_gemm_norm_rope.py``): Q/K tiles are normed + rotated on the fp32
accumulator and the slab is written once. Inference / in-place only (no
``q_pre``); OFF by default until the perf node ranks it.

**STATUS (2026-09-11): every stage is REAL.** The two projections drive the
shipped FROST GEMM, stages (2)+(3) are one FROST kernel of this block's own,
stage (4) is the shipped FROST SDPA, stage (5) is this block's elementwise
kernel. Two OPTIONAL fusions, both off by default until the perf node ranks
them: ``fuse_norm_rope`` (stages (2)+(3) into (1)'s epilogue) and ``fuse_gate``
(stage (5) into (4)'s epilogue); with both on the block is THREE launches,
``proj(+norm+rope) -> sdpa(+gate) -> out_proj``. The block targets **Rubin
(SM107) only** for now — every other arch is declined explicitly rather than
served slowly.

**The default pipeline is deliberately UNFUSED**: five stages, five launches
(four under ``inplace_qkv``), every intermediate materialised in the caller's
workspace. That is the honest baseline; it is also the point. Per the design
record, the API boundary is the expensive thing to change later and the
partitioning behind it is cheap, so the stage cuts below move behind knobs
(see "Fusion status" per stage) while this file's signature does not.

Two structural facts that will not move, and that shape everything here:

* **(6) can never share a kernel with (4).** ``o_proj`` contracts over
  ``H_q * D`` — ALL heads — while an SDPA CTA owns one head. Fusing it needs a
  cross-CTA split-K reduction. This is why the block is necessarily a *set* of
  kernels and why the API, not any one kernel, is the deliverable.
* **(5) must run AFTER the SDPA epilogue's dead-row substitution.** A row with
  no unmasked column gets ``O := 0`` by SELECT (``sdpa-invariants.md`` § 2);
  multiplying accumulator residue by ``sigmoid(GATE)`` before that substitution
  propagates NaN. Whenever (5) folds into (4)'s epilogue, it goes after the
  select, never before.

Precision roadmap
-----------------

**bf16 throughout by default** (fp32 accumulate in every GEMM and in the
softmax).  **FP8 (2026-09-11): an E4M3 pipeline with static per-tensor scales
is wired in TWO configurations** -- pass an e4m3 ``h``/weights and a
:class:`QuantSpec`:

* **UNFUSED** (both fusion knobs off, 7 launches): FP8 projections with the
  descale folded into a scalar-multiply epilogue (bf16 out), bf16 norm+RoPE in
  place, two quantize passes (Q/K/V, then gated O) feeding the Rubin per-tensor
  FP8 SDPA and the FP8 out projection.  The quantize passes are the visible
  price of "unfused".
* **FULLY FUSED** (``fuse_norm_rope=True, fuse_gate=True``, 3 launches): the
  projection fork ``kernels/proj_gemm_norm_rope_fp8.py`` descales, norms,
  rotates AND quantizes in its epilogue -- e4m3 Q / K / V into COMPACT
  per-tensor ``q8`` / ``k8`` / ``v8`` (each == compact BSHD ``[B, S, H, d]``,
  exactly what the unfused FP8 SDPA reads), the GATE as bf16 into ``gate16``
  -- and the production FP8 d256 SDPA (``prefill_d256_fp8.py`` with
  ``epilogue_gate`` on and ``has_amax_o=False``: the block runs static scales
  and reads no ``Amax_O``, so the atomic is compiled out) reads those compact
  buffers, multiplies by ``sigmoid(gate16)`` after the dead-row select and
  writes e4m3 ``o8`` for the out projection.  No bf16 slab, no quantize
  passes, no bf16 O: 33792 B/token of workspace against 68608 unfused
  (-51 %).  (Round 1 wrote ONE token-major ``qkv8 [T, n_qkv]`` slab and read
  it STRIDED: measured +7.2 % on the SDPA at 32K -- K/V lines refetched at a
  9216 B stride once the gate stream evicts them -- where compact reads are
  -3.5 %; STATUS.md "SDPA decomposition".  Hence compact per tensor.)  The two
  are NOT bit-identical (the unfused path rounds to bf16 twice); both are
  scored against the fake-quant fp32 oracle.
* Anything in between (one knob) is a typed decline naming both knobs; FP8 is
  inference-only, and FP8 + ``seq_lens_present`` is declined while the FP8
  d256 SDPA kernel hangs on an empty KV entry (see ``__init__``).

MXFP8 follows the same route -- the quantization rides in the PROJECTION
EPILOGUE rather than as separate passes, because stage (1) already owns the only
full pass over Q/GATE/K/V (emitting them pre-quantized costs one epilogue and
saves a 34 GiB round trip at 1M tokens).  Three things that decision reaches
into, listed so the current code does not foreclose them:

* stage (1)'s epilogue must be able to emit **two layouts of scale factors** for
  the operands that are consumed transposed. V is quantized COLUMNWISE for the
  BMM2, and its MXFP8 scale factors are D-PLANE-MAJOR in GMEM while Q/K's are
  per-tile contiguous — reusing one SF descriptor builder for all three is a
  silent wrong answer that only shows up at ``d > 128`` and ``S > TILE_N``
  (``mma-tma-matrix.md`` § 7).
* ``TILE_K_HW`` / idesc ``k_dim`` are **arch-opposite** for FP8: Blackwell wants
  ``k_dim=0`` at ``TILE_K_HW=32``, Rubin ``k_dim=1`` at 64. Read
  ``mma-tma-matrix.md`` § 1 before picking either.
* the RMSNorm in stage (2) and the sigmoid in stage (5) stay fp32-compute
  regardless of the storage dtype.

Public signatures grow APPEND-ONLY (a ``QuantSpec`` argument lands at the end of
the argument list with a ``None`` default), so nothing below needs a placeholder
field today.

Design record (fusion analysis, backward graph, multi-GPU/CP, measurement
protocol): ``frost_dev/plans/qwen_gated_attention_block_plan.md``.
Backward: :mod:`cudnn.gated_attention_block.api_bwd` — read this file first, the
backward is defined against the :class:`SavedForBackward` contract below.
PyTorch oracle and perf baseline:
``test/python/fe_api/gated_attention_block/reference.py``.

Not in scope for v1, in the order they are likely to land: THD / varlen packing;
MXFP8 (above; per-tensor FP8 is wired); the graph-API engine row. This is a frontend-only OSS API
first; a manifest family + ``Capabilities`` comes when the stages exist to be
honest about.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.workspace import WorkspaceLayout

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Stage (1) layout — exactly what the QKV+GATE matmul computes and where it lands
# ---------------------------------------------------------------------------
#
# ONE GEMM, four logical outputs. This section is the contract every other stage
# is written against, so it is spelled out in full rather than left to the
# kernel.
#
# THE GEMM
# --------
#   A  = h        [M, K]   M = B*S, K = d_model      row-major, bf16
#   B  = W_qkvg   [N, K]   N = (2*H_q + 2*H_kv) * D  row-major, bf16   (read transposed)
#   C  =          [M, N]   fp32 accumulate, bf16 epilogue store
#
# i.e. the ordinary ``nn.Linear`` TN form, ``out = h @ W_qkvg.T``. At the 397B
# full-attention layer: M = B*S, N = 17408, K = 4096.
#
# THE N AXIS
# ----------
# N is four contiguous blocks, each head-major with D innermost:
#
#     col   0                  H_q*D            2*H_q*D    +H_kv*D   +H_kv*D
#           |------- Q -------|------ GATE -----|--- K ---|--- V ---|
#   397B:   0                 8192              16384     16896     17408
#           |<--- 32 heads -->|<--- 32 heads --->|<- 2 ->|<- 2 ->|
#
# Within a block, head ``j`` owns columns ``[j*D, (j+1)*D)``. Q and GATE are
# adjacent because that is how the model produces them (one double-width
# ``q_proj``), not for any kernel reason.
#
# THE ALIGNMENT INVARIANT, AND WHY IT IS LOAD-BEARING
# --------------------------------------------------
# Every block boundary is a multiple of ``QKVG_TILE_ALIGN`` (64), so for any
# supported ``TILE_N`` **no GEMM output tile ever straddles two blocks**
# (``qkvg_tile_plan`` proves it, ``validate`` enforces it). Consequences:
#
#   * the epilogue can specialize PER BLOCK on a per-tile constant rather than
#     predicating column-by-column -- which is what makes the future
#     quantization epilogue cheap (Q/K rowwise, V columnwise, GATE passthrough,
#     each with its own scale-factor layout);
#   * each block's compact row stride is 16-byte aligned at bf16, so a TMA
#     descriptor over it is legal.
#
# At 397B the boundaries are 8192 / 16384 / 16896, all multiples of 256, so any
# TILE_N in {64, 128, 256} works. The invariant is checked, not assumed: a
# geometry with, say, H_kv*D = 96 raises at build time instead of producing a
# straddling tile nobody notices.
#
# WHERE THE TILES ARE STORED: FOUR COMPACT BUFFERS, NOT ONE FUSED ONE
# -------------------------------------------------------------------
# The epilogue selects one of FOUR output descriptors from the N-tile index and
# writes a BSHD-COMPACT buffer per block:
#
#     Q     [B, S, H_q,  D]   strides (S*H_q*D,  H_q*D,  D, 1)
#     GATE  [B, S, H_q,  D]   strides (S*H_q*D,  H_q*D,  D, 1)
#     K     [B, S, H_kv, D]   strides (S*H_kv*D, H_kv*D, D, 1)
#     V     [B, S, H_kv, D]   strides (S*H_kv*D, H_kv*D, D, 1)
#
# The tempting alternative -- one fused ``[B, S, N]`` buffer, with Q/K/V as
# strided views -- is REJECTED, and the reason is specific rather than
# aesthetic. Such a view has a padded token stride (N, not H*D). The SDPA's
# dense path accepts that layout (``sdpa/graph_analyzer.dense_layout_ok``
# explicitly allows padded strides) and then NORMALIZES it: ``_to_bshd`` is
# ``transpose(1, 2)`` followed by ``.contiguous()`` if the view is neither already
# contiguous nor at the strides the artifact was compiled for
# (``SdpaFwdDsl._to_bshd`` in ``sdpa/fwd/api_dsl.py``). That is a **gather copy of Q**, which
# at 1M tokens is 16 GiB moved twice, silently, inside a block whose whole
# premise is that it does not do that (AGENTS.md Rule 2).
#
# The SDPA's THD arm does address such strides natively -- its own docstring
# names this exact case, "a K/V view of a kv-interleaved [T, 2, H, D] buffer
# ... the layout torch.nn.attention.varlen users produce by slicing a fused KV
# projection" (``api_dsl.py`` ``_thd_check_strides_native``). So the fused-buffer
# layout is reachable, just not on the dense path today. Writing four compact
# buffers gets zero-copy on BOTH arms with no SDPA change, and it is also
# strictly less memory: 17 GiB of Q+GATE+K+V rather than a 34 GiB fused slab
# plus per-tensor copies.
#
# LAYOUT VARIANTS CONSIDERED AND NOT TAKEN (revisit with a measurement, not an
# argument):
#   * **Q/GATE interleaved per head** (``[q_0 g_0 q_1 g_1 ...]``). Still a legal
#     rank-4 view (head stride 2*D), and it is what a fused epilogue would want
#     if one CTA produced a head's Q and its gate together. Buys nothing while
#     the two land in separate buffers anyway, and costs a checkpoint permute.
#   * **K/V interleaved** (``[T, 2, H_kv, D]``). Better KV-tile locality for the
#     SDPA, and a shape the THD path already handles. Worth measuring once the
#     SDPA is the bottleneck; K and V are 1/16 of Q here, so it is a small
#     lever.
#
# THE CALLER'S SIDE
# -----------------
# ``build_fused_qkvg_weight`` assembles ``W_qkvg`` from the three checkpoint
# matrices ONCE, at load time. It takes ``q_gate_layout`` because the model's
# own convention for splitting the double-width ``q_proj`` is not something to
# guess: "flat" chunks ``[..., 2*H_q*D]`` into all-Q then all-GATE, "per_head"
# views it as ``[..., H_q, 2*D]`` first and chunks the head dim. The two differ
# by a row permutation of ``q_proj.weight``, and picking wrong is a silent wrong
# answer -- so it is a parameter, checked by a round-trip test, and never a
# hot-path conversion either way.


QKVG_TILE_ALIGN = 64
_QK_NORM_ROPE_THREADS = 128
_ELEMENTWISE_THREADS = 128
_ELEMENTWISE_ROWS_PER_GROUP = 2
# H is fixed per compiled artifact either way (it is in the compile-cache key),
# so baking it into the address math costs no generality -- it only turns a
# software integer divide per row into a shift. Runtime arm kept for A/B.
_ELEMENTWISE_CONST_HEAD_COUNT = True
_QK_NORM_ROPE_ROWS_PER_GROUP = 2
_QK_NORM_ROPE_DEFER_SECONDARY_LOADS = True  # keep in step with kernels/qk_norm_rope.py DEFAULT_DEFER_SECONDARY_LOADS
_QK_NORM_ROPE_TILE_ROWS = 16  # keep in step with kernels/qk_norm_rope_tma.py DEFAULT_TILE_ROWS
_QK_NORM_ROPE_STAGES = 2  # keep in step with kernels/qk_norm_rope_tma.py DEFAULT_STAGES
"""Smallest GEMM ``TILE_N`` the block intends to support.

Every ``qkvg`` block boundary must be a multiple of this so no output tile
straddles two blocks -- see the alignment invariant above.
"""


class ProjBlock(IntEnum):
    """Which of stage (1)'s four outputs an N column belongs to.

    The values are the block ORDER along N and are part of the layout contract:
    ``qkvg_offsets`` returns them in this order and ``dQKVG`` in the backward
    reuses it, so one wgrad GEMM produces ``dW_qkvg`` in the layout the forward
    consumes.
    """

    Q = 0
    GATE = 1
    K = 2
    V = 3


# ---------------------------------------------------------------------------
# 2. Geometry — the block's compile-time contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedAttentionBlockGeometry:
    """Everything the stages specialize on. Plan-time only, never runtime data.

    Compile keys are plan-time-only (``python/cudnn/AGENTS.md`` Rule 4): every
    field here is derivable from the declaration — shapes, dtypes, flags — so
    one compiled artifact re-binds any batch/sequence.

    Qwen3.5-397B (TP=1) fills this as ``d_model=4096, h_q=32, h_kv=2,
    d_head=256, rope_dim=64`` — GQA 16x, ``scale = 256 ** -0.5 = 1/16``.
    """

    d_model: int
    h_q: int
    h_kv: int
    d_head: int  # d_qk == d_v; the SDPA flavor is picked from this
    rope_dim: int  # leading dims of d_head that rotate; [rope_dim, d_head) pass through

    qk_norm_eps: float = 1e-6
    attn_scale: Optional[float] = None  # None -> d_head ** -0.5

    # Mask. NOT hardcoded causal: ColQwen3.5 is bidirectional, and the mask is a
    # SCHEDULING-REGIME selector here, not just a correctness knob -- under a
    # window the SDPA collapses to a constant ~9 KV tiles while the gate GEMM is
    # fixed, which can invert the partitioning the causal arm wants.
    is_causal: bool = True
    causal_bottom_right: bool = False
    window_left: int = -1
    window_right: int = -1

    # -- derived scalars ----------------------------------------------------

    @property
    def scale(self) -> float:
        """Softmax scale actually used."""
        return self.attn_scale if self.attn_scale is not None else float(self.d_head) ** -0.5

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head."""
        return self.h_q // self.h_kv

    @property
    def n_qkvg(self) -> int:
        """Output width of the fused stage-(1) projection."""
        return sum(self.qkvg_block_widths)

    # -- the N-axis map (see "Stage (1) layout" above) ----------------------

    @property
    def qkvg_block_widths(self) -> tuple[int, int, int, int]:
        """Column count of each block, in ``ProjBlock`` order."""
        return (
            self.h_q * self.d_head,
            self.h_q * self.d_head,
            self.h_kv * self.d_head,
            self.h_kv * self.d_head,
        )

    @property
    def qkvg_offsets(self) -> tuple[int, int, int, int]:
        """Starting column of each block, in ``ProjBlock`` order.

        **This ordering is API, append-only forever** — it is what a caller
        concatenates its checkpoint weights into, ONCE at load time (never per
        execute: Rule 1 bans conversions on the hot path).
        """
        offs, acc = [], 0
        for w in self.qkvg_block_widths:
            offs.append(acc)
            acc += w
        return tuple(offs)

    # -- the fully fused FP8 pipeline: Q + K + V width (no GATE band) -----------

    @property
    def n_qkv(self) -> int:
        """``n_qkvg`` minus the GATE band: the e4m3 bytes per token the fused
        FP8 projection writes across its three COMPACT outputs ``q8`` / ``k8``
        / ``v8`` (``h_q*d + 2*h_kv*d``).  A width, not a slab: since round 2
        nothing in the block addresses a ``[T, n_qkv]`` buffer."""
        return self.n_qkvg - self.h_q * self.d_head

    @property
    def qkvg_heads(self) -> tuple[int, int, int, int]:
        """Head count of each block, in ``ProjBlock`` order."""
        return (self.h_q, self.h_q, self.h_kv, self.h_kv)

    def block_for_column(self, col: int) -> tuple[ProjBlock, int, int]:
        """``col`` in ``[0, N)`` -> ``(block, head, column within head)``."""
        if not 0 <= col < self.n_qkvg:
            raise ValueError(f"column {col} out of range [0, {self.n_qkvg})")
        for block, (off, width) in enumerate(zip(self.qkvg_offsets, self.qkvg_block_widths)):
            if col < off + width:
                local = col - off
                return ProjBlock(block), local // self.d_head, local % self.d_head
        raise AssertionError("unreachable: widths sum to n_qkvg")

    def qkvg_tile_plan(self, tile_n: int) -> tuple[ProjBlock, ...]:
        """Destination block of every stage-(1) output tile, at this ``TILE_N``.

        This is what the epilogue's descriptor select decodes at run time, and
        it exists as plain Python so the alignment invariant is checkable
        without a GPU: the plan is well-defined exactly when no tile straddles a
        block boundary, which :meth:`validate` requires.
        """
        if tile_n <= 0 or self.n_qkvg % tile_n != 0:
            raise ValueError(f"TILE_N={tile_n} must be positive and divide N={self.n_qkvg}")
        plan = []
        for t in range(self.n_qkvg // tile_n):
            first = self.block_for_column(t * tile_n)[0]
            last = self.block_for_column((t + 1) * tile_n - 1)[0]
            if first != last:
                raise ValueError(
                    f"TILE_N={tile_n} tile {t} straddles {first.name} and {last.name}; " f"block widths {self.qkvg_block_widths} are not {tile_n}-aligned"
                )
            plan.append(first)
        return tuple(plan)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on any geometry the stages cannot express.

        Never a module-level ``assert`` (contract § 7): anything derived from
        user input raises, so it survives ``python -O`` and names itself. Each
        message says what the failure would have LOOKED like, because a config
        error that reaches a kernel does not announce itself.
        """
        for label, value in (("d_model", self.d_model), ("h_q", self.h_q), ("h_kv", self.h_kv), ("d_head", self.d_head)):
            if value <= 0:
                raise ValueError(f"{label} must be > 0, got {value}")

        if self.h_q % self.h_kv != 0:
            raise ValueError(f"h_q ({self.h_q}) must be divisible by h_kv ({self.h_kv}) for GQA/MQA broadcast")

        if not 0 <= self.rope_dim <= self.d_head:
            raise ValueError(f"rope_dim must be in [0, d_head={self.d_head}], got {self.rope_dim}")
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even (rotate_half pairs i with i + rope_dim//2), got {self.rope_dim}")

        if not self.qk_norm_eps > 0.0:
            raise ValueError(f"qk_norm_eps must be > 0, got {self.qk_norm_eps}")
        if self.attn_scale is not None and not self.attn_scale > 0.0:
            raise ValueError(f"attn_scale must be > 0 when given, got {self.attn_scale}")

        # The stage-(1) alignment invariant. Without it an output tile can
        # straddle two blocks, and the epilogue would have to predicate per
        # column instead of specializing per tile -- which is not a correctness
        # bug today but forecloses the quantization epilogue entirely.
        for block, width in zip(ProjBlock, self.qkvg_block_widths):
            if width % QKVG_TILE_ALIGN != 0:
                raise ValueError(
                    f"{block.name} block width {width} (= heads * d_head) must be a multiple of QKVG_TILE_ALIGN={QKVG_TILE_ALIGN}, "
                    f"else a GEMM output tile straddles two of Q/GATE/K/V"
                )

        # Mask knobs. -1 means "unbounded on that side"; anything else must be a
        # real, non-negative distance.
        for label, value in (("window_left", self.window_left), ("window_right", self.window_right)):
            if value < -1:
                raise ValueError(f"{label} must be >= 0, or -1 for unbounded; got {value}")
        if self.causal_bottom_right and not self.is_causal:
            raise ValueError("causal_bottom_right=True requires is_causal=True")
        # The SDPA lowers ONE band: window_right widens the causal diagonal, so
        # it is meaningless without a diagonal to widen. Caught here for a
        # message that names the block's own field rather than the adapter's.
        if self.window_right >= 0 and not self.is_causal:
            raise ValueError("window_right requires is_causal=True (it widens the causal diagonal, it does not create one)")


def build_fused_qkvg_weight(
    w_q_gate: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    q_gate_layout: str = "flat",
) -> torch.Tensor:
    """Assemble ``W_qkvg [N, d_model]`` from the checkpoint's three matrices.

    **Load time only.** The result is what stage (1) reads; nothing here ever
    runs on the execute path.

    Parameters
    ----------
    w_q_gate
        ``q_proj.weight``, ``[2*H_q*D, d_model]`` — the model's double-width
        query projection, holding both Q and the output gate.
    w_k, w_v
        ``k_proj.weight`` / ``v_proj.weight``, ``[H_kv*D, d_model]``.
    q_gate_layout
        How ``w_q_gate``'s rows split into Q and GATE, which is a property of
        the checkpoint and is NOT inferable from the tensor:

        ``"flat"``
            ``[all Q heads | all GATE heads]`` — the split a
            ``chunk(2, dim=-1)`` on the flat ``[..., 2*H_q*D]`` output performs.
        ``"per_head"``
            ``[Q_0 | GATE_0 | Q_1 | GATE_1 | ...]`` — the split a
            ``view(..., H_q, 2*D)`` followed by ``chunk(2, dim=-1)`` performs.

        Get this wrong and every gate is applied to the wrong head: the output
        is finite, plausible, and wrong, with no error anywhere. Verify it
        against the model you are loading rather than trusting a default.
    """
    geometry.validate()
    d = geometry.d_head
    hq, hkv = geometry.h_q, geometry.h_kv
    if w_q_gate.shape[0] != 2 * hq * d:
        raise ValueError(f"w_q_gate must have {2 * hq * d} rows (2*h_q*d_head), got {w_q_gate.shape[0]}")
    for name, w in (("w_k", w_k), ("w_v", w_v)):
        if w.shape[0] != hkv * d:
            raise ValueError(f"{name} must have {hkv * d} rows (h_kv*d_head), got {w.shape[0]}")
    for name, w in (("w_q_gate", w_q_gate), ("w_k", w_k), ("w_v", w_v)):
        if w.shape[1] != geometry.d_model:
            raise ValueError(f"{name} must have {geometry.d_model} columns (d_model), got {w.shape[1]}")

    if q_gate_layout == "flat":
        w_q, w_gate = w_q_gate[: hq * d], w_q_gate[hq * d :]
    elif q_gate_layout == "per_head":
        per_head = w_q_gate.view(hq, 2 * d, geometry.d_model)
        w_q = per_head[:, :d].reshape(hq * d, geometry.d_model)
        w_gate = per_head[:, d:].reshape(hq * d, geometry.d_model)
    else:
        raise ValueError(f"q_gate_layout must be 'flat' or 'per_head', got {q_gate_layout!r}")

    return torch.cat([w_q, w_gate, w_k, w_v], dim=0).contiguous()


# ---------------------------------------------------------------------------
# RoPE table contract
# ---------------------------------------------------------------------------
#
# ``cos`` / ``sin`` are PRECOMPUTED per-token tables of shape
# ``[B, S, ROPE_DIM]`` (B may be 1 and broadcast), in the block's storage dtype
# or fp32. Element ``i`` of the rotated slice pairs with element
# ``i + ROPE_DIM // 2`` -- the NeoX / ``rotate_half`` convention that HF
# ``apply_rotary_pos_emb`` and vLLM's default ``RotaryEmbedding`` use, with the
# halves duplicated so ``cos[..., i] == cos[..., i + ROPE_DIM // 2]``:
#
#     x_rot = x[..., :ROPE_DIM]
#     out   = x_rot * cos + rotate_half(x_rot) * sin
#     where rotate_half(v) = cat(-v[..., H:], v[..., :H]), H = ROPE_DIM // 2
#
# The duplication is redundant (2x a 128 KiB table at S=1K) and kept anyway
# because it makes the tables byte-identical to what a HF or vLLM caller already
# holds -- so no conversion happens on the hot path (Rule 1).
#
# **Taking a TABLE rather than position ids is the load-bearing choice.** mRoPE
# (per-section positions), plain RoPE, YaRN and any NTK scaling all differ only
# in how the table is built; the caller builds it (``cudnn.yarn`` already ships
# the YaRN half) and the kernel never learns which. This is also what makes
# ColQwen3.5 and the vision tower reachable without touching stage (3).
#
# NOT served, and declined explicitly rather than silently mis-rotated:
# the GPT-J / INTERLEAVED pairing (``i`` with ``i+1``). DeepSeek's
# interleaved-in / halves-out variant lives in
# ``gemm/cutedsl/dense/proj_rope_mxfp8`` if it is ever needed here.


# ---------------------------------------------------------------------------
# 3. Workspace — every intermediate the unfused chain materialises
# ---------------------------------------------------------------------------


_WS_ALIGN = 256


def _align_up(n: int, a: int = _WS_ALIGN) -> int:
    return (n + a - 1) // a * a


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _view(ws: torch.Tensor, offset: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """A typed, shaped VIEW of the caller's uint8 workspace at ``offset``.

    Views only — never a copy, never an allocation (Rule 1). Offsets are
    ``_WS_ALIGN``-aligned so the dtype reinterpretation is always legal.
    """
    n = 1
    for x in shape:
        n *= int(x)
    nbytes = n * _itemsize(dtype)
    return ws[offset : offset + nbytes].view(dtype).view(*shape)


def _cols(proj: torch.Tensor, col_offset: int, h: int, d: int) -> torch.Tensor:
    """A ``[T, h, d]`` view of ``h*d`` COLUMNS of the fused ``[T, N]`` projection.

    Heads are contiguous within a token (stride ``d``, elem 1) but the token
    stride is ``N``, not ``h*d`` — a padded stride. Every consumer in this block
    addresses that natively; the one that would not is the SDPA, which is why V
    gets its own compaction stage.
    """
    t, n = int(proj.shape[0]), int(proj.shape[1])
    return torch.as_strided(proj, (t, h, d), (n, d, 1), storage_offset=proj.storage_offset() + col_offset)


@dataclass(frozen=True)
class _Intermediates:
    """Byte offsets into the caller's workspace, one per materialised tensor.

    ``get_workspace_size()`` must be honest and ``execute()`` must not allocate
    (contract § 10 / Rule 1), so every intermediate is reserved at build time and
    carved as a view at execute time.

    At Qwen3.5-397B, B=1 S=1Mi bf16 these are NOT small: PROJ alone is 34 GiB and
    Q/O are 16 GiB each, while K and V are 1 GiB (GQA is 16x and ``h_kv`` is 2).
    That asymmetry decides both the fusion order here and the context-parallel
    strategy in the design record § 9.2.

    ``-1`` means the tensor never lands in HBM: ``gate`` lives in ``proj``'s
    columns and ``o_gated`` aliases ``o`` (stage (5) gates in place).  Under the
    FULLY FUSED FP8 pipeline ``proj`` and ``o`` are ``-1`` too: the fused
    projection writes the COMPACT e4m3 ``q8`` / ``k8`` / ``v8`` (the same three
    slots the unfused FP8 quantize passes fill) + ``gate16`` (bf16 GATE), and
    the gated FP8 SDPA writes ``o8`` directly.
    """

    proj: int  # [T, N]            stage (1) output; holds Q | GATE | K | V  (-1 when FP8 fully fused)
    q: int  # [T, H_q,  D]     compact, post-norm, post-RoPE
    gate: int  # -1: a column slice of proj
    k: int  # [T, H_kv, D]     compact
    v: int  # [T, H_kv, D]     compact (stage 3b)
    o: int  # [T, H_q,  D]     SDPA output, gated in place by (5)  (-1 when FP8 fully fused)
    o_gated: int  # -1: aliases o
    engine_scratch: int  # the sub-engines' own workspace (GEMM / SDPA)

    total_bytes: int
    base_align: int
    # FP8 pipelines only (-1 otherwise): compact e4m3 Q/K/V the FP8 SDPA reads
    # -- written by the quantize stages (unfused) or by the projection fork's
    # epilogue (fully fused) -- and the e4m3 O the FP8 out_proj reads.
    q8: int = -1  # [T, H_q,  D]
    k8: int = -1  # [T, H_kv, D]
    v8: int = -1  # [T, H_kv, D]
    o8: int = -1  # [T, H_q * D]
    # FULLY FUSED FP8 only (-1 otherwise): the fused projection's bf16 GATE.
    gate16: int = -1  # [T, H_q, D]  bf16, compact GATE


def _plan_workspace(
    geom: GatedAttentionBlockGeometry,
    b: int,
    s: int,
    dtype: torch.dtype,
    want_lse: bool,
    want_rstd: bool,
    inplace_qkv: bool = False,
    fp8: bool = False,
    fp8_fused: bool = False,
) -> _Intermediates:
    """Reserve every intermediate, in stage order, and report the total.

    Reserved in the order the stages write them, so a future fusion that deletes
    one leaves a contiguous prefix rather than a hole — forking stage (1) to
    write four compact buffers deletes ``proj`` and stage (3b) together.

    Scratch only. Anything the BACKWARD needs leaves through
    :class:`SavedForBackward`, into storage the caller owns, because a workspace
    is dead the moment ``execute()`` returns. ``lse`` is likewise a caller
    tensor: it is an OUTPUT, not an intermediate.
    """
    del want_lse, want_rstd
    e = _itemsize(dtype)
    t = b * s
    off = 0
    offsets = {}
    if fp8_fused:
        # FULLY FUSED FP8 (fuse_norm_rope + fuse_gate under a QuantSpec): the
        # projection fork writes COMPACT e4m3 Q / K / V into `q8` / `k8` / `v8`
        # (the unfused pipeline's own slots -- the gated FP8 SDPA reads exactly
        # what the ungated one reads, compact BSHD at token_stride 0) and the
        # bf16 GATE into `gate16`; the gated FP8 SDPA writes e4m3 `o8`.  No bf16 slab,
        # no bf16 O, no quantize passes: 33792 B/token at the 397B geometry
        # against 68608 unfused (-51 %).  Same byte total as round 1's
        # [T, n_qkv] slab (q8 + k8 + v8 == t * n_qkv), laid out per tensor.
        slots = [
            ("q8", t * geom.h_q * geom.d_head),
            ("k8", t * geom.h_kv * geom.d_head),
            ("v8", t * geom.h_kv * geom.d_head),
            ("gate16", t * geom.h_q * geom.d_head * e),
            ("o8", t * geom.h_q * geom.d_head),
        ]
        for name, nbytes in slots:
            offsets[name] = off
            off += _align_up(nbytes)
        return _Intermediates(
            proj=-1,
            q=-1,
            gate=-1,
            k=-1,
            v=-1,
            o=-1,
            o_gated=-1,
            engine_scratch=off,
            total_bytes=off,
            base_align=_WS_ALIGN,
            q8=offsets["q8"],
            k8=offsets["k8"],
            v8=offsets["v8"],
            o8=offsets["o8"],
            gate16=offsets["gate16"],
        )
    # IN-PLACE: Q and K are normed back over their own columns of `proj`, and V
    # is never moved, so the SDPA reads all three straight out of the slab at
    # its PADDED token stride. The three compact buffers stop existing -- 26% of
    # the block's workspace at every shape (18 GiB at 1Mi tokens). What makes it
    # legal is that the forward's own kernels are row-local (each lane holds its
    # whole [D] row in registers before storing) and the SDPA now compiles its
    # TMA descriptors AT the declared strides rather than repacking.
    # It is NOT legal under save_for_backward: the slab IS q_pre/k_pre, which
    # the RMSNorm backward needs and cannot safely reconstruct -- see
    # SavedForBackward. The caller-facing guard is in GatedAttentionBlock.
    slots = [("proj", t * geom.n_qkvg * e)]
    if not inplace_qkv and not fp8:
        slots += [
            ("q", t * geom.h_q * geom.d_head * e),
            ("k", t * geom.h_kv * geom.d_head * e),
            ("v", t * geom.h_kv * geom.d_head * e),
        ]
    if fp8:
        # FP8: norm+RoPE stays in place on the bf16 slab; the quantize stages
        # then write COMPACT e4m3 Q/K/V (1 B/elem -- half the size of the bf16
        # compact buffers they replace, and they double as V's compaction), the
        # SDPA writes bf16 O, the gate is in place, and O is quantized into o8
        # for the FP8 out_proj.
        slots += [
            ("q8", t * geom.h_q * geom.d_head),
            ("k8", t * geom.h_kv * geom.d_head),
            ("v8", t * geom.h_kv * geom.d_head),
        ]
    slots += [("o", t * geom.h_q * geom.d_head * e)]
    if fp8:
        slots += [("o8", t * geom.h_q * geom.d_head)]
    for name, nbytes in slots:
        offsets[name] = off
        off += _align_up(nbytes)
    engine = off
    return _Intermediates(
        proj=offsets["proj"],
        q=offsets.get("q", -1),
        gate=-1,
        k=offsets.get("k", -1),
        v=offsets.get("v", -1),
        o=offsets["o"],
        o_gated=-1,
        engine_scratch=engine,
        total_bytes=engine,
        base_align=_WS_ALIGN,
        q8=offsets.get("q8", -1),
        k8=offsets.get("k8", -1),
        v8=offsets.get("v8", -1),
        o8=offsets.get("o8", -1),
    )


# ---------------------------------------------------------------------------
# 4. Saved-tensor contract — the forward/backward boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedForBackward:
    """What the forward hands the backward. **Append-only forever**, like a
    manifest slot: a field's meaning cannot change once a checkpoint or an
    autograd graph has been built against it.

    Sizes are per token; the 1M-token column is one full-attention layer of the
    397B at B=1 bf16, which is the regime that makes the trade real.

    ============ ================= ============ ===================================
    field         shape             at S=1Mi     why
    ============ ================= ============ ===================================
    ``h``         [B,S,d_model]     8 GiB        stage (1) wgrad AND recompute source
    ``gate``      [B,S,H_q,D]       16 GiB       stage (5) backward; NOT recomputable
                                                 without half the stage-(1) GEMM
    ``o``         [B,S,H_q,D]       16 GiB       ``dG`` needs pre-gate O; SDPA bwd
                                                 needs O too
    ``lse``       [B,H_q,S] fp32    134 MiB      SDPA bwd cannot run without it
    ``rstd_q``    [B,S,H_q] fp32    134 MiB      RMSNorm bwd; tiny, always save
    ``rstd_k``    [B,S,H_kv] fp32   8 MiB        idem
    ``q_pre``     [B,S,H_q,D]       16 GiB       pre-norm Q -- SAVE or RECOMPUTE
    ``k_pre``     [B,S,H_kv,D]      1 GiB        idem
    ============ ================= ============ ===================================

    **This dataclass is the block's one genuinely novel value proposition, and
    the reason to build it even if the first A/B is flat.** ``gate`` is the same
    size as ``o``; saving it roughly doubles the attention activation footprint,
    and recomputing it re-runs half the stage-(1) GEMM. Likewise ``q_pre`` /
    ``k_pre`` are either saved (17 GiB) or recomputed from ``h`` by re-running
    the Q and K slices of the projection. **Both trades are invisible at the op
    level and a decomposed graph cannot make either one** — a block can, and can
    expose them as a knob.

    Do NOT try to reconstruct ``q_pre`` from the normed Q by dividing out
    ``w_q_norm``: it is undefined wherever a norm weight is zero, and it is
    numerically hostile wherever one is small.

    ``o_gated`` is deliberately absent: it is ``o * sigmoid(gate)``, so the
    backward's ``dW_o`` stage recomputes it elementwise from two tensors it
    already holds rather than storing a third 16 GiB copy.

    A field left ``None`` means "recompute me", and the backward decides how
    from :class:`~cudnn.gated_attention_block.api_bwd.RecomputePolicy`.
    """

    h: torch.Tensor
    gate: torch.Tensor
    o: torch.Tensor
    lse: torch.Tensor
    rstd_q: torch.Tensor
    rstd_k: torch.Tensor
    q_pre: Optional[torch.Tensor] = None
    k_pre: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# 4b. FP8 (E4M3, per-tensor STATIC scales) — the QuantSpec contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSpec:
    """Static per-tensor FP8 scales for the block (inference-style quantization).

    Passing a ``QuantSpec`` (and FP8 ``h`` / weights) selects the FP8 pipeline;
    ``None`` is the bf16/f16 block.  All scales are Python floats fixed at plan
    time -- calibrated offline, like the weights' own scales -- so the execute
    path does NO amax pass and NO host readback; the block materialises them as
    1-element fp32 device tensors ONCE in ``compile()`` (the SDPA adapter does
    the same for its identity descales).

    Conventions (``x_real = x_fp8 * descale``; ``x_fp8 = sat_e4m3(x_real * scale)``):

    ============== ============================================================
    ``descale_h``      dequant multiplier of the FP8 input ``h``
    ``descale_w_qkvg`` dequant multiplier of the FP8 ``W_qkvg``
    ``descale_w_o``    dequant multiplier of the FP8 ``W_o``
    ``scale_q/k/v``    quant scales applied to the bf16 post-norm Q, post-norm K
                       and V before the SDPA; the SDPA receives ``1/scale`` as
                       its descales
    ``scale_o``        quant scale applied to the bf16 gated O before ``out_proj``
    ============== ============================================================

    UNFUSED: the two projections fold ``descale_a * descale_w`` into ONE
    scalar-multiply epilogue of the FROST GEMM (fp32 accumulate, bf16 out), so
    the slab and O stay bf16 and the norm+RoPE and gate kernels run unchanged.
    FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``): the same scalars ride
    INSIDE the two kernels -- ``alpha_qkvg`` and ``scale_q/k/v`` as one fp32 ``[4]``
    device vector read by the projection fork's epilogue (which writes e4m3
    Q/K/V and a bf16 GATE), ``1/scale_q/k/v`` and ``scale_o`` as the production
    FP8 SDPA's ``descale_q/k/v`` / ``scale_o`` execute tensors (it folds
    ``descale_v * scale_o`` into ``inv_sum`` in-kernel and writes e4m3 O).  Only
    E4M3 is served today;
    E5M2 is a knob away once a use case asks for it.
    """

    descale_h: float
    descale_w_qkvg: float
    descale_w_o: float
    scale_q: float
    scale_k: float
    scale_v: float
    scale_o: float
    dtype: torch.dtype = torch.float8_e4m3fn

    def validate(self) -> None:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"QuantSpec: only torch.float8_e4m3fn is served, got {self.dtype}")
        for name in ("descale_h", "descale_w_qkvg", "descale_w_o", "scale_q", "scale_k", "scale_v", "scale_o"):
            v = float(getattr(self, name))
            if not (v > 0.0) or v != v or v in (float("inf"),):
                raise ValueError(f"QuantSpec.{name} must be a finite positive float, got {v}")

    @property
    def alpha_qkvg(self) -> float:
        return self.descale_h * self.descale_w_qkvg

    @property
    def alpha_o(self) -> float:
        return (1.0 / self.scale_o) * self.descale_w_o


# ---------------------------------------------------------------------------
# 5. Stages — one FROST kernel each, in pipeline order
# ---------------------------------------------------------------------------


class _Stage(ABC):
    """One kernel of the block.

    Deliberately the same three-phase shape as ``APIBase`` (support / compile /
    execute) so a stage can be promoted to a standalone public API, or absorbed
    into its neighbour, without the block's own API moving.
    """

    name: str

    @abstractmethod
    def check_support(self) -> None:
        """Raise ``NotImplementedError`` / ``ValueError`` if this stage cannot
        serve the declaration. Never degrade silently, never adapt (Rule 2)."""

    @abstractmethod
    def compile(self) -> None:
        """Build the artifact. Plan-time keys only (Rule 4)."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Launch. No allocation, no D2H read, no implicit conversion
        (Rules 1 / 3), and everything ordered on the LAUNCH stream (Rule 5)."""


class _Projection(_Stage):
    """(1) and (6) — the block's two dense projections, ONE implementation.

    They are the same op at different shapes, so there is one stage class and
    two factories rather than two kernels::

        (1) qkv_gate_proj   h       [M, d_model]  @ W_qkvg^T  ->  [M, N_qkvg]
        (6) out_proj        O_gated [M, H_q*D]    @ W_o^T     ->  [M, d_model]

    **Neither writes a kernel.** Both drive the shipped FROST GEMM
    (``gemm/frost/kernel_templates/sm100_matmul.py`` — persistent, double-TMEM,
    CLC-scheduled, tuned tile catalog), pinned by name in the graph's ranked
    plan list. Details and the layout convention: ``kernels/proj_gemm.py``.

    **Fusion status — both are FORK candidates, and the template is built to be
    forked** (the SDPA backward already forked it for a 2-D batch, with a
    bidirectional "apply fixes both ways" note). Stage (1) will need a fork for
    the four-output-buffer epilogue of § 1 and, later, the FP8/MXFP8
    quantization epilogue; stage (1)'s GATE columns are also the block's one
    piece of independent MMA work, dependency-legal to run during the SDPA's
    softmax phase below ~12K tokens. **Measure against the unforked engine
    first** — it is the baseline any fork has to beat, and this stage is how you
    get that number.

    Stage (6) can never fuse into the SDPA: it contracts over all heads.
    """

    def __init__(self, *, m: int, k: int, n: int, dtype: torch.dtype, name: str, out_dtype: Optional[torch.dtype] = None, alpha: bool = False) -> None:
        self.name = name
        self.m = int(m)
        self.k = int(k)
        self.n = int(n)
        self.dtype = dtype
        # FP8: inputs e4m3, fp32 accumulate, `out_dtype` (bf16) out, and `alpha`
        # = descale_a * descale_w folded into the GEMM's scalar-multiply epilogue.
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.alpha = bool(alpha)
        self._plan = None

    def check_support(self) -> None:
        if self.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            raise NotImplementedError(f"{self.name}: bf16/f16/fp8-e4m3 only, got {self.dtype}")
        if self.dtype == torch.float8_e4m3fn and self.k % 16:
            raise NotImplementedError(f"{self.name}: FP8 needs K % 16 == 0 (TMA 16-byte rule at 1 B/elem), got K={self.k}")
        if self.out_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: output dtype must be bf16/f16, got {self.out_dtype}")

    def compile(self) -> None:
        from .kernels.proj_gemm import build_proj_gemm

        self._plan = build_proj_gemm(m=self.m, k=self.k, n=self.n, dtype=self.dtype, label=self.name, out_dtype=self.out_dtype, alpha=self.alpha)

    def workspace_bytes(self) -> int:
        """Bytes the FROST GEMM needs. Its own, separate from the block's
        intermediates — the block reserves a region for it (contract § 10)."""
        if self._plan is None:
            raise RuntimeError("call compile() before workspace_bytes()")
        return self._plan.workspace_bytes

    def flops(self) -> int:
        """``2*M*N*K`` — the denominator for an MMA SOL number."""
        return 2 * self.m * self.n * self.k

    def execute(
        self, a: torch.Tensor, w: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor, handle=None, alpha: Optional[torch.Tensor] = None, *, stream=None
    ) -> None:
        """``stream`` is the block's launch stream (a raw ``CUstream`` int); the
        runner carries it onto both GEMM routes -- see ``run_proj_gemm`` (Rule 5)."""
        from .kernels.proj_gemm import run_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        run_proj_gemm(self._plan, a, w, out, workspace, handle, alpha=alpha, stream=stream)


def _qkv_gate_projection(
    geom: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, out_dtype: Optional[torch.dtype] = None, alpha: bool = False
) -> _Projection:
    """Stage (1). At the 397B full-attention layer: ``M x 17408 x 4096``."""
    return _Projection(m=batch * seq_len, k=geom.d_model, n=geom.n_qkvg, dtype=dtype, name="qkv_gate_proj", out_dtype=out_dtype, alpha=alpha)


def _out_projection(
    geom: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, out_dtype: Optional[torch.dtype] = None, alpha: bool = False
) -> _Projection:
    """Stage (6). At the 397B full-attention layer: ``M x 4096 x 8192``."""
    return _Projection(m=batch * seq_len, k=geom.h_q * geom.d_head, n=geom.d_model, dtype=dtype, name="out_proj", out_dtype=out_dtype, alpha=alpha)


class _Quantize(_Stage):
    """(3q) / (5q) per-tensor FP8 cast: ``dst = sat_e4m3(src * scale)`` over ``[T, H, D]``.

    Kernel: ``kernels/quantize.py`` (a streaming pass in the mould of
    ``elementwise.py``; the source may be a strided slab column slice, the
    destination is compact, so for Q/K/V this stage IS the compaction).  The
    scale is a 1-element fp32 device tensor read in-kernel -- no host readback.
    It exists because the UNFUSED FP8 pipeline needs FP8 operands for the SDPA
    and the out projection while the norm+RoPE and gate kernels stay bf16.  The
    fully fused FP8 pipeline (``fuse_norm_rope`` + ``fuse_gate``) folds both
    casts into the producing epilogues and does not build this stage at all.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.quantize import validate_shape

        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        from .kernels.quantize import compile_quantize

        self._recipe = compile_quantize(dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, threads_per_cta=_ELEMENTWISE_THREADS)

    def moved_bytes(self) -> int:
        from .kernels.quantize import moved_bytes

        return moved_bytes(self.batch * self.seq_len, self.heads, self.geom.d_head, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(self, src: torch.Tensor, dst: torch.Tensor, scale: torch.Tensor, current_stream=None) -> None:
        from .kernels.quantize import run_quantize

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_quantize(self._recipe, src, dst, scale, stream=stream)


class _FusedQkvProjection(_Stage):
    """(1)+(2)+(3) in ONE kernel: the QKV+GATE projection with per-head RMSNorm
    and partial RoPE applied to the Q and K tiles INSIDE the GEMM epilogue.

    **This one IS a fork** -- ``kernels/proj_gemm_norm_rope.py``, the rendered
    shipped GEMM at the block's tile config plus an epilogue arm.  What makes the
    fusion tile-local, and therefore cheap, is the S1 layout contract: at
    ``d_head == 256`` one CTA output tile is exactly one head, so the RMSNorm
    reduction over D is thread-local (one epilogue thread owns one row of the
    tile) and the rotate_half partner sits in the same thread.  No new barrier,
    no new SMEM, no cross-CTA exchange; the kernel docstring has the design.

    It writes the SAME ``[T, N]`` slab the unfused chain norms in place, so the
    SDPA reads it exactly as under ``inplace_qkv`` -- which is why the fused
    path REQUIRES in-place: there is no pre-norm Q/K anywhere.  Training
    (``save_for_backward``) therefore keeps the unfused chain until the fork
    also writes ``q_pre``/``k_pre`` (the same trade every overwriting fusion in
    this block makes; see :class:`SavedForBackward`).  ``rstd`` IS emitted.

    Numerics: the norm runs on the fp32 ACCUMULATOR with one bf16 rounding,
    where the unfused chain norms the bf16-rounded projection -- the fused
    result is the more accurate one, so the oracle is the fp32 reference, never
    bit-identity with the pair.

    Rubin (SM107) only: the fork is a rendering for the sm_107a ``128x256``
    2-CTA config and its dtype constants are baked in -- ONE rendering per
    dtype family.  A different ``d_head`` or ``rope_dim`` the epilogue cannot
    tile is declined here, with the tile constraint in the message, before the
    template is ever loaded.

    **FP8 arm (``quant`` given, e4m3 ``h``/``W_qkvg``):** the second rendering,
    ``kernels/proj_gemm_norm_rope_fp8.py`` (selected by
    ``NormRopeFusionParams.quant_fp8``), whose epilogue also QUANTIZES: Q/K tiles
    are descaled (``alpha_qkvg``), normed + rotated on fp32 and written as e4m3
    (``* scale_q`` / ``* scale_k``) into COMPACT per-tensor ``q8`` / ``k8``
    buffers, V tiles as ``e4m3(alpha * acc * scale_v)`` into ``v8``, GATE tiles
    as bf16 into ``gate16`` -- four TMA-store descriptors, tile class -> buffer
    + column remap (round 2; round 1's single ``qkv8`` slab made the SDPA read
    strided and cost +7.2 % at 32K).  The four scalars ride as ONE fp32 ``[4]``
    device vector ``[alpha_qkvg, scale_q, scale_k, scale_v]`` (``qscal``)
    materialised in :meth:`compile`, read in-kernel -- never module-param floats.
    Inference only (``want_rstd`` must be False).  Launched through
    ``run_fused_proj_gemm_fp8`` (the bf16 runner is not overloaded).
    """

    name = "qkv_gate_proj_norm_rope"
    _TILE_N = 256  # the rendering's per-CTA output width; one head per tile needs d_head == this
    _SUBTILE_N = 32  # epilogue subtile; rope_dim must be a whole number of subtile PAIRS
    _FORK_PATH_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "proj_gemm_norm_rope_fp8.py")

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_rstd: bool,
        norm_source: str = "ldg_early",
        quant: Optional[QuantSpec] = None,
        device=None,
    ) -> None:
        self.geom = geometry
        self.m = int(batch * seq_len)
        self.k = int(geometry.d_model)
        self.n = int(geometry.n_qkvg)
        self.dtype = dtype
        self.want_rstd = bool(want_rstd)
        self.norm_source = str(norm_source)
        # FP8: the QuantSpec whose alpha_qkvg / scale_q / scale_k / scale_v the
        # fork's epilogue applies; `device` is where `qscal` is materialised.
        self.quant = quant
        self.device = device
        self._plan = None
        self._qscal = None

    @property
    def fp8(self) -> bool:
        return self.quant is not None

    def params(self):
        from .kernels.proj_gemm import NormRopeFusionParams

        g = self.geom
        kw = {}
        if self.fp8:
            # Only the FP8 arm names the field, so the bf16 compile key (and the
            # bf16 rendering's cache) is byte-identical to before the FP8 fork.
            kw["quant_fp8"] = True
        return NormRopeFusionParams(
            d_head=g.d_head, rope_dim=g.rope_dim, h_q=g.h_q, h_kv=g.h_kv, eps=g.qk_norm_eps, want_rstd=self.want_rstd, norm_source=self.norm_source, **kw
        )

    def _fp8_fork_available(self) -> Optional[str]:
        """None when the FP8 fork + its runner ABI are present; else the reason they are not."""
        import dataclasses

        from .kernels import proj_gemm

        if "quant_fp8" not in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}:
            return "NormRopeFusionParams has no `quant_fp8` field"
        if not hasattr(proj_gemm, "run_fused_proj_gemm_fp8"):
            return "kernels/proj_gemm.py has no `run_fused_proj_gemm_fp8`"
        if not self._runner_writes_compact_qkv():
            return "`run_fused_proj_gemm_fp8` still has the round-1 slab ABI (no `out_k8`): the compact q8/k8/v8 GEMM fork has not landed"
        if not os.path.exists(self._FORK_PATH_FP8):
            return f"{os.path.basename(self._FORK_PATH_FP8)} is not in kernels/"
        return None

    @staticmethod
    def _runner_writes_compact_qkv() -> bool:
        """True when the FP8 runner carries the round-2 ABI
        ``(plan, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, qscal, *, stream)``.
        Round 1 took one ``out_qkv8`` slab; handing it the compact buffers
        positionally would bind ``out_k8`` as the gate, so the block declines
        (check_support) and refuses (execute_fp8) rather than mis-binding."""
        import inspect

        from .kernels import proj_gemm

        fn = getattr(proj_gemm, "run_fused_proj_gemm_fp8", None)
        return fn is not None and "out_k8" in inspect.signature(fn).parameters

    def check_support(self) -> None:
        from .kernels.proj_gemm import validate_norm_rope_params

        g = self.geom
        # Geometry first, so the declines below read the same on every device.
        if g.d_head != self._TILE_N:
            raise NotImplementedError(
                f"{self.name}: the fused epilogue needs one GEMM output tile == one head, i.e. d_head == {self._TILE_N} "
                f"(the rendering's per-CTA N tile); got d_head={g.d_head}. Use the unfused chain (fuse_norm_rope=False)."
            )
        if g.rope_dim % (2 * self._SUBTILE_N) or not 0 < g.rope_dim < g.d_head:
            raise NotImplementedError(
                f"{self.name}: rope_dim must be a positive multiple of {2 * self._SUBTILE_N} and < d_head so each rotate_half pair "
                f"spans whole epilogue subtiles; got rope_dim={g.rope_dim}"
            )
        if self.fp8:
            if self.dtype != torch.float8_e4m3fn:
                raise NotImplementedError(f"{self.name}: the FP8 fork is rendered for e4m3 h / W_qkvg, got {self.dtype}")
            if self.k % 16:
                # TMA 16-byte rule at 1 B/elem (mirrors _Projection's FP8 gate).
                raise NotImplementedError(f"{self.name}: FP8 needs K % 16 == 0 (TMA 16-byte rule at 1 B/elem), got K={self.k}")
            if self.want_rstd:
                raise NotImplementedError(f"{self.name}: the FP8 fork is inference-only (no rstd output)")
            missing = self._fp8_fork_available()
            if missing is not None:
                raise NotImplementedError(f"{self.name}: the FP8 fused projection fork has not landed in this checkout ({missing})")
        elif self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the fork is rendered for bf16 only (e4m3 needs a QuantSpec), got {self.dtype}")
        validate_norm_rope_params(self.params())
        if torch.cuda.is_available():
            cc = tuple(torch.cuda.get_device_capability())
            if cc != _SM107_CC:
                raise NotImplementedError(f"{self.name}: rendered for sm_107a (Rubin); this device is SM{cc[0]}{cc[1]}")

    def compile(self) -> None:
        from .kernels.proj_gemm import build_fused_proj_gemm

        self._plan = build_fused_proj_gemm(self.params())
        if self.fp8:
            # Plan-time constant (contract § 10): the fork reads these four once
            # per epilogue warp; the execute path never allocates or converts.
            q = self.quant
            dev = self.device if self.device is not None else torch.device("cuda")
            self._qscal = torch.tensor([q.alpha_qkvg, q.scale_q, q.scale_k, q.scale_v], dtype=torch.float32, device=dev)

    def workspace_bytes(self) -> int:
        """None: no split-K, no scratch -- the kernel writes the slab directly."""
        return 0

    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def execute(self, a, w, out, w_q_norm, w_k_norm, cos, sin, rstd_q=None, rstd_k=None, *, stream) -> None:
        """bf16 arm: ONE ``[M, N]`` slab out (Q/K columns normed + rotated)."""
        from .kernels.proj_gemm import run_fused_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        if self.fp8:
            raise ValueError(f"{self.name}: this stage was declared FP8; use execute_fp8(...)")
        run_fused_proj_gemm(self._plan, a, w, out, w_q_norm, w_k_norm, cos, sin, rstd_q, rstd_k, stream=stream)

    def execute_fp8(self, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, *, stream) -> None:
        """FP8 arm: COMPACT e4m3 ``q8 [M, h_q*d]`` / ``k8 [M, h_kv*d]`` / ``v8 [M, h_kv*d]``
        + bf16 ``gate16 [M, h_q*d]`` out (all 2-D, contiguous; the SDPA views the
        same bytes ``[B, S, H, d]``).

        The positional order is the runner's leading order on purpose --
        ``frost_dev/probe_fp8_fused_gemm_ab.py`` captures these arguments and
        replays them through ``run_fused_proj_gemm_fp8(plan, *args, qscal, **kw)``.
        ``a`` is the e4m3 ``[M, K]`` view of ``h`` (the runner binds it rank-3
        ``[1, M, K]`` exactly as the bf16 runner does), ``w`` the e4m3
        ``[N_qkvg, K]`` checkpoint-layout weight; ``qscal`` is the fp32 ``[4]``
        ``[alpha_qkvg, scale_q, scale_k, scale_v]`` materialised in :meth:`compile`.
        """
        from .kernels.proj_gemm import run_fused_proj_gemm_fp8

        if self._plan is None or self._qscal is None:
            raise RuntimeError("call compile() before execute_fp8()")
        if not self.fp8:
            raise ValueError(f"{self.name}: this stage was declared bf16; use execute(...)")
        if not self._runner_writes_compact_qkv():
            # Never hand the round-1 slab runner three compact buffers positionally.
            raise NotImplementedError(f"{self.name}: {self._fp8_fork_available()}")
        run_fused_proj_gemm_fp8(self._plan, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, self._qscal, stream=stream)


class _QkNormRope(_Stage):
    """(2)+(3) per-head RMSNorm over D then partial RoPE, on Q and K, ONE kernel.

    **V is not normed and is not touched here.** Q and K ride one launch over a
    flat row space, so K's 1/16-of-Q traffic costs no second launch and the
    cos/sin tables stay hot across a token's heads.

    Emits ``rstd_q`` / ``rstd_k`` (fp32, one per (token, head)) when the block
    saves for backward — 134 MiB + 8 MiB at 1M tokens, cheap enough that
    recomputing them would be the odd choice.

    Norm and rotation both run in fp32 with a SINGLE rounding at the end. An
    unfused torch chain rounds twice, so this is a slightly different (and more
    accurate) function; the oracle is written to match
    (``reference.qk_norm_rope_reference``).

    **Fusion status: this stage IS the fusion.** Stages (2)+(3) were the
    largest single stage at every sequence length in the torch baseline, which
    is why they were built before the projections. Measured on Rubin at the 397B
    geometry: **18.9x the torch chain, taking the stage from 36-53% of the block
    to 1.5-2.6%** — and **36.7% of a cold 10777 GB/s copy ceiling** at the
    shipped knobs, so there is real headroom left. Being pure bandwidth, the
    harness reports achieved GB/s and fraction-of-ceiling, never TFLOP/s.

    **Quote no fraction for this stage that was not L2-FLUSHED.** The earlier
    "58%" was hot-cache and against another node's ceiling; re-measured cold on
    2026-09-10 it is 36.7%. The kernel docstring carries the full table and the
    register-pressure diagnosis; ``frost_dev/probe_norm_rope_bw.py`` reproduces
    it with hot and cold columns side by side.

    Q/K may be updated IN PLACE (``q_out is q``): every lane reads its whole row
    before any lane stores, and the RoPE partner shuffle stays inside the row's
    own lane group.

    Kernel: ``kernels/qk_norm_rope.py`` — at ``kernels/`` level, not under an
    arch package, because it is plain vectorized LDG/STG with no tcgen05 and no
    arch-specific path (engine contract § 8).
    """

    name = "qk_norm_rope"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_rstd: bool,
        threads_per_cta: int = _QK_NORM_ROPE_THREADS,
        rows_per_group: int = _QK_NORM_ROPE_ROWS_PER_GROUP,
        defer_secondary_loads: bool = _QK_NORM_ROPE_DEFER_SECONDARY_LOADS,
        const_head_counts: bool = True,
        impl: str = "auto",
        tile_rows: Optional[int] = None,  # None = fit it to the geometry
        stages: int = _QK_NORM_ROPE_STAGES,
        use_pdl: bool = False,
        dynamic_token_stride: bool = True,
    ) -> None:
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.want_rstd = bool(want_rstd)
        self.threads_per_cta = int(threads_per_cta)
        self.rows_per_group = int(rows_per_group)
        self.defer_secondary_loads = bool(defer_secondary_loads)
        self.const_head_counts = bool(const_head_counts)
        self.impl = str(impl)
        self.tile_rows = None if tile_rows is None else int(tile_rows)
        self.stages = int(stages)
        self.use_pdl = bool(use_pdl)
        self.dynamic_token_stride = bool(dynamic_token_stride)
        self._recipe = None
        self._impl = None

    def resolve_tile_rows(self) -> int:
        """Rows per TMA tile: the caller's value, or the best one for this shape.

        A TMA tile is ``tile_rows`` rows of the ``[T, H, D]`` operand, so the
        tiling is only expressible when ``tile_rows`` divides ``h_q``, is a
        multiple of ``h_kv``, and spreads evenly over the CTA's warps. The
        measured optimum on Rubin is 16, but ``h_q`` is 8 on plenty of models and
        16 simply does not tile there.

        So ``tile_rows=None`` (the default) FITS it: the largest legal value not
        exceeding the measured preference. That is a selection, like
        ``impl="auto"`` -- an EXPLICIT ``tile_rows`` that does not fit raises
        instead of being quietly replaced.
        """
        g = self.geom
        warps = max(1, self.threads_per_cta // 32)
        legal = lambda r: r > 0 and g.h_q % r == 0 and r % g.h_kv == 0 and r % warps == 0
        if self.tile_rows is not None:
            if not legal(self.tile_rows):
                raise NotImplementedError(
                    f"tile_rows={self.tile_rows} cannot tile this geometry: it must divide h_q={g.h_q}, "
                    f"be a multiple of h_kv={g.h_kv}, and spread over {warps} warps"
                )
            return self.tile_rows
        fits = [r for r in range(_QK_NORM_ROPE_TILE_ROWS, 0, -1) if legal(r)]
        return fits[0] if fits else 0

    def resolve_impl(self) -> str:
        """Which of the two kernels this stage will run: ``"ldg"`` or ``"tma"``.

        Both are kept on purpose. They compute the SAME function bit-for-bit
        (asserted by ``test_qk_norm_rope.py``), and each owns an arch:

        * **tma** stages every tile through SMEM with a TMA ring. It needs
          ``cp.async.bulk.tensor``, i.e. **SM90 or newer**, and it is the faster
          kernel wherever it runs -- +45.7% on the Rubin dev node, +55.7% on the
          perf node, both cold and against the LDG kernel that already carries
          the deferred-load fix.
        * **ldg** is plain vectorized global load/store with no SMEM staging. It
          is the ONLY option on SM80, which has no TMA at all, and it is the
          fallback for any geometry the TMA tiling cannot express.

        ``impl="auto"`` picks tma when the arch AND the geometry allow it. That
        is a SELECTION, not a silent capability fallback: asking for ``"tma"``
        explicitly on a part or a shape that cannot serve it RAISES rather than
        quietly running the other kernel (Rule 1).
        """
        if self.impl not in ("auto", "ldg", "tma"):
            raise ValueError(f'impl must be one of "auto", "ldg", "tma"; got {self.impl!r}')
        if self.impl == "ldg":
            return "ldg"
        from cudnn.frost.device import ambient_device, compute_capability

        from .kernels.qk_norm_rope_tma import validate_shape as _tma_validate

        g = self.geom
        major, minor = compute_capability(ambient_device())
        arch_ok = major >= 9  # cp.async.bulk.tensor exists from Hopper on
        # OUTSIDE the try: an EXPLICIT tile_rows that does not fit is a knob the
        # caller asked for and we cannot honor, so it must propagate. Only the
        # fitted value is allowed to decide "tma is not expressible here".
        tile_rows = self.resolve_tile_rows()
        try:
            _tma_validate(g.d_head, g.rope_dim, g.h_q, g.h_kv, tile_rows, self.threads_per_cta)
            shape_ok, why = True, ""
        except (ValueError, NotImplementedError) as exc:
            shape_ok, why = False, str(exc)
        if self.impl == "tma":
            if not arch_ok:
                raise NotImplementedError(f"impl='tma' needs sm_90 or newer for cp.async.bulk.tensor; this device is sm_{major}{minor}")
            if not shape_ok:
                raise NotImplementedError(f"impl='tma' cannot tile this geometry: {why}")
            return "tma"
        return "tma" if (arch_ok and shape_ok) else "ldg"

    def check_support(self) -> None:
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"qk_norm_rope serves bf16/f16 only, got {self.dtype}")
        if self.resolve_impl() == "tma":
            from .kernels.qk_norm_rope_tma import validate_shape as _tma_validate

            g = self.geom
            _tma_validate(g.d_head, g.rope_dim, g.h_q, g.h_kv, self.resolve_tile_rows(), self.threads_per_cta)
        else:
            from .kernels.qk_norm_rope import validate_shape

            validate_shape(self.geom.d_head, self.geom.rope_dim, self.threads_per_cta)

    def compile(self) -> None:
        g = self.geom
        self._impl = self.resolve_impl()
        if self._impl == "tma":
            from .kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma

            self._recipe = compile_qk_norm_rope_tma(
                dtype=self.dtype,
                h_q=g.h_q,
                h_kv=g.h_kv,
                d=g.d_head,
                rope_dim=g.rope_dim,
                eps=g.qk_norm_eps,
                want_rstd=self.want_rstd,
                tile_rows=self.resolve_tile_rows(),
                stages=self.stages,
                threads_per_cta=self.threads_per_cta,
            )
            return
        from .kernels.qk_norm_rope import compile_qk_norm_rope

        self._recipe = compile_qk_norm_rope(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            want_rstd=self.want_rstd,
            threads_per_cta=self.threads_per_cta,
            rows_per_group=self.rows_per_group,
            defer_secondary_loads=self.defer_secondary_loads,
            const_head_counts=self.const_head_counts,
            use_pdl=self.use_pdl,
            dynamic_token_stride=self.dynamic_token_stride,
        )

    def moved_bytes(self) -> int:
        """HBM traffic of one launch — the denominator for the SOL number."""
        from .kernels.qk_norm_rope import moved_bytes

        g = self.geom
        return moved_bytes(self.batch * self.seq_len, g.h_q, g.h_kv, g.d_head, elem_bytes=2, want_rstd=self.want_rstd)

    def execute(
        self,
        q: torch.Tensor,  # [B, S, H_q,  D]
        k: torch.Tensor,  # [B, S, H_kv, D]
        w_q_norm: torch.Tensor,  # [D]
        w_k_norm: torch.Tensor,  # [D]
        cos: torch.Tensor,  # [B, S, ROPE_DIM]
        sin: torch.Tensor,  # [B, S, ROPE_DIM]
        q_out: Optional[torch.Tensor] = None,  # defaults to in place
        k_out: Optional[torch.Tensor] = None,
        rstd_q: Optional[torch.Tensor] = None,  # [B, S, H_q]  fp32
        rstd_k: Optional[torch.Tensor] = None,  # [B, S, H_kv] fp32
        current_stream=None,
        flat: bool = False,  # accepted and ignored: rank decides
    ) -> None:
        """Flatten ``[B, S, ...]`` to ``[T, ...]`` and launch.

        ``.view()`` only — never ``reshape``: these buffers are compact by
        construction (§ 1), so a view is exact and a copy would be a silent
        extra kernel (Rule 1).
        """
        from .kernels.qk_norm_rope import run_qk_norm_rope
        from .kernels.qk_norm_rope_tma import run_qk_norm_rope_tma

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        g = self.geom
        t = self.batch * self.seq_len
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(q.device).cuda_stream

        def _flat(x, h):
            """Accept ``[B, S, H, D]`` or an already-flat ``[T, H, D]``.

            The block hands STRIDED ``[T, H, D]`` views of the fused projection,
            which cannot be ``.view()``-ed; a standalone caller hands compact
            ``[B, S, H, D]``. Never ``reshape``: it may copy (Rule 1).
            """
            if x is None:
                return None
            return x if x.ndim == 3 else x.view(t, h, g.d_head)

        runner = run_qk_norm_rope_tma if self._impl == "tma" else run_qk_norm_rope
        runner(
            self._recipe,
            _flat(q, g.h_q),
            _flat(k, g.h_kv),
            _flat(q if q_out is None else q_out, g.h_q),
            _flat(k if k_out is None else k_out, g.h_kv),
            w_q_norm,
            w_k_norm,
            cos.view(t, g.rope_dim),
            sin.view(t, g.rope_dim),
            None if rstd_q is None else rstd_q.view(t, g.h_q),
            None if rstd_k is None else rstd_k.view(t, g.h_kv),
            stream=stream,
        )


# Rubin. The block is Rubin-only for now: the SDPA flavor it needs
# ((256, 256) f16) exists on SM100 too, but nothing here has been validated
# there, and serving an unvalidated arch silently is worse than declining it.
_SM107_CC = (10, 7)


def _bhsd_desc(b: int, h: int, s: int, d: int, dtype: torch.dtype, device, name: str, token_stride: int = 0) -> TensorDesc:
    """Descriptor for a BSHD-COMPACT buffer presented as logical BHSD.

    The SDPA's operand contract is rank-4 ``(B, H, S, D)``; the block's buffers
    are ``[B, S, H, D]`` compact, so what it hands over is ``.transpose(1, 2)``
    — strides ``(S*H*D, D, H*D, 1)``. That is exactly the layout the SDPA's own
    normalization recognises as already-canonical: ``_to_bshd`` transposes back
    and finds it contiguous, so it returns the view unchanged and **no copy
    happens** (``SdpaFwdDsl._to_bshd`` in ``sdpa/fwd/api_dsl.py``). This is the whole reason stage (1)
    writes four compact buffers instead of one fused slab — see § 1.

    Built as a descriptor rather than from a sample tensor so declaring the
    block costs no device allocation.
    """
    # token_stride = 0 means COMPACT (h*d). A larger value declares a column
    # slice of a wider row -- Q/K/V inside the fused projection, token stride
    # n_qkvg. The SDPA compiles its descriptors at whatever this declares and
    # binds the view directly (api_dsl._bshd_zero_copy_stride), so a padded
    # stride costs no copy; what it must not do is OVERLAP, hence the >= check.
    ts = int(token_stride) if token_stride else h * d
    if ts < h * d:
        raise ValueError(f"{name}: token_stride {ts} is smaller than h*d={h*d}; that would alias distinct rows")
    shape = (b, h, s, d)
    stride = (s * ts, d, ts, 1)
    stride_order = tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))
    return TensorDesc(dtype=dtype, shape=shape, stride=stride, stride_order=stride_order, device=device, name=name)


def _lse_desc(b: int, h: int, s: int, device, name: str = "lse") -> TensorDesc:
    """Descriptor for the ``[B, H_q, S]`` fp32 log-sum-exp, head-major compact."""
    shape = (b, h, s)
    stride = (h * s, s, 1)
    stride_order = tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))
    return TensorDesc(dtype=torch.float32, shape=shape, stride=stride, stride_order=stride_order, device=device, name=name)


class _Sdpa(_Stage):
    """(4) ``O = softmax(Q K^T * scale + mask) V``, GQA-broadcast H_q/H_kv.

    **This stage writes no kernel, in ANY configuration.** It drives the shipped
    FROST forward (``cudnn.sdpa.fwd.api_dsl.SdpaFwdDslSm100``), which routes to
    the SM107 sibling kernels automatically from the live device —
    ``rubin=(self._device_cc == (10, 7))`` in ``SdpaFwdDslSm100.compile`` — so the
    ``(256, 256)`` flavor Qwen3.5 needs lands on
    ``sdpa/fwd/kernels/sm107/prefill_d256_f16.py`` (bf16 / f16) or, under
    ``pertensor_fp8=True``, ``prefill_d256_fp8.py`` (e4m3).  There is no flag to
    pass; there IS an arch to check, which :meth:`check_support` does.

    **``fuse_gate=True`` is the SAME adapter with a gate descriptor.**  The
    sigmoid-gate epilogue -- a TMA-staged read of GATE and
    ``O := O * sigmoid(GATE)`` AFTER the dead-row select, per element -- is a
    production feature of both d256 kernels behind
    ``TemplateParams.epilogue_gate`` (``SdpaFwdDslSm100.template_params()``
    sets it from ``sample_gate``; the engine rows claim it through
    ``Capabilities.epilogue_gate_d_shapes``).  Declaring the stage with
    ``sample_gate=<GATE descriptor>`` selects that specialization and
    ``execute(gate=...)`` binds the tensor.  GATE may be a column slice of the
    ``[T, N]`` slab (``gate_token_stride``): the adapter compiles the gate's
    TMA descriptor at the declared stride exactly as it does Q/K/V, so no copy
    happens anywhere -- and stage (5) disappears.

    **FP8** (``dtype == e4m3``): ``pertensor_fp8=True``, ``dtype_o`` (bf16 on the
    unfused pipeline, e4m3 on the fully fused one) and ``has_amax_o=False`` --
    the block runs static per-tensor scales and never reads ``Amax_O``, so the
    kernel's atomicMax is compiled out rather than written into a slot nobody
    reads.  ``descale_q/k/v`` and (e4m3 O only) ``scale_o`` ride ``execute``;
    the kernel folds ``descale_v * scale_o`` into ``inv_sum`` in-kernel.  The
    block feeds it COMPACT ``q8`` / ``k8`` / ``v8`` (``token_stride == 0``):
    round 1's strided slab reads cost +7.2 % on the SDPA at 32K -- K/V lines
    refetched at a 9216 B stride once the gate stream evicts them -- where
    compact reads are -3.5 % (STATUS.md "SDPA decomposition").

    Three things settled here rather than inherited:

    * **``sched_policy`` is chosen EXPLICITLY, and it is LPT at d=256 -- bf16
      AND FP8.** `sdpa_fwd_prefill_sm107` (PR #1001) and, since 2026-09-11,
      `sdpa_fwd_prefill_sm107_fp8` advertise ``{NATURAL, LPT}`` for the
      (256, 256) flavor this block uses. ``_sched_policy`` reads the live
      device's row and picks LPT exactly where that row claims it, NATURAL for
      anything else -- gated on the d-shape the row actually claims rather than
      on "is Rubin". Stated rather than left to the standalone wrapper's ``None``
      derivation, which still excludes Rubin wholesale and would quietly give
      the win back.
    * **LSE is optional here and mandatory for training.** Prefill inference
      never reads it, so a stats-less block pays nothing; ``save_for_backward``
      implies it, because the SDPA backward cannot run without it.
    * **Mask arms are ``const_expr``-folded**, so a dense PASS proves nothing
      about the causal/window path. Validate at least one config per arm.

    ``seq_lens`` is the per-batch valid KV length (dense padding mask). Q-side
    trimming (``seq_q_lens``) and THD/varlen are NOT wired — both are declines,
    not silent no-ops.
    """

    name = "sdpa"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        want_lse: bool,
        seq_lens_present: bool = False,
        token_stride: int = 0,
        fuse_gate: bool = False,
        gate_token_stride: int = 0,
        o_dtype: Optional[torch.dtype] = None,
        gate_dtype: Optional[torch.dtype] = None,
    ) -> None:
        # token_stride != 0 => Q/K/V are column slices of the fused projection
        # and are read in place at that stride (the adapter compiles its TMA
        # descriptors at the declared strides).  0 => the compact buffers.
        self.token_stride = int(token_stride)
        # FP8 (dtype == e4m3): per-tensor descales (and scale_o) ride execute();
        # O comes out in `o_dtype` -- bf16 on the unfused pipeline, e4m3 on the
        # fully fused one.
        self.o_dtype = o_dtype if o_dtype is not None else dtype
        self.fp8 = dtype == torch.float8_e4m3fn
        # fuse_gate => the kernel's epilogue_gate specialization reads GATE (a
        # slab column slice at gate_token_stride; 0 = compact like O) in
        # `gate_dtype` -- the block's activation dtype; None = Q's dtype -- and
        # writes O gated in place of O.
        self.fuse_gate = bool(fuse_gate)
        self.gate_token_stride = int(gate_token_stride)
        self.gate_dtype = gate_dtype
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        self.want_lse = bool(want_lse)
        self.seq_lens_present = bool(seq_lens_present)
        self._impl = None

    def _build_impl(self):

        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        g = self.geom
        b, s, d = self.batch, self.seq_len, g.d_head
        ts = self.token_stride
        # has_amax_o=False: static scales, no Amax_O consumer -> the FP8 kernel
        # compiles the atomicMax out (and execute() refuses an amax_o tensor).
        kw = dict(pertensor_fp8=True, dtype_o=self.o_dtype, has_amax_o=False) if self.fp8 else {}
        if self.fuse_gate:
            # The gate descriptor selects the epilogue_gate specialization
            # (template_params().epilogue_gate); its stride is compiled in like
            # Q/K/V's, so a slab column slice binds with no copy.
            kw["sample_gate"] = _bhsd_desc(b, g.h_q, s, d, self.gate_dtype or self.dtype, self.device, "gate", token_stride=self.gate_token_stride)
        return SdpaFwdDslSm100(
            _bhsd_desc(b, g.h_q, s, d, self.dtype, self.device, "q", token_stride=ts),
            _bhsd_desc(b, g.h_kv, s, d, self.dtype, self.device, "k", token_stride=ts),
            _bhsd_desc(b, g.h_kv, s, d, self.dtype, self.device, "v", token_stride=ts),
            _bhsd_desc(b, g.h_q, s, d, self.o_dtype, self.device, "o"),
            _lse_desc(b, g.h_q, s, self.device) if self.want_lse else None,
            is_causal=g.is_causal,
            causal_bottom_right=g.causal_bottom_right,
            window_size_left=None if g.window_left < 0 else g.window_left,
            window_size_right=None if g.window_right < 0 else g.window_right,
            scale_softmax=g.scale,
            seq_kv_lens_present=self.seq_lens_present,
            sched_policy=self._sched_policy(),
            **kw,
        )

    @classmethod
    def _row_capabilities(cls, arch: str, fp8: bool):
        """The ``Capabilities`` of the SDPA-forward engine row for ``(arch, dtype family)``.

        A missing / renamed row is a typed decline, not a bare ``StopIteration``
        escaping ``check_support`` (engine-contract § 2): this sits on the decline
        path of every ``fuse_gate=True`` block, on every device.
        """
        from cudnn.sdpa.fwd import engines

        row = engines.engine_name(arch=arch, fp8=fp8)
        caps = next((spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == row), None)
        if caps is None:
            raise NotImplementedError(f"{cls.name}: no SDPA forward engine row {row!r} in cudnn.sdpa.fwd.engines.ENGINE_SPECS")
        return caps

    def _sched_policy(self) -> int:
        """LPT where the engine row says it is validated, NATURAL everywhere else.

        The causal load is triangular, so natural row-major order leaves the last
        CTAs of a wave with almost nothing to do. LPT rebalances it: measured
        standalone on the bf16 d256 kernel at +18.2 / +11.6 / +1.1 % of causal
        SOL at S = 4096 / 8192 / 32768, recovering 40 / 51 / 29 % of the
        causal-vs-dense gap, and dense-neutral; on the per-tensor FP8 d256 kernel
        +5.2 / +5.9 / +5.6 / +2.0 / +2.3 % at S = 2K..32K (perf node,
        launch-interleaved, O bit-identical to NATURAL).

        READ OFF THE ROW, never transcribed: the SM107 f16 and per-tensor FP8
        rows claim LPT per flavor through `sched_policies_by_d_shape` ((256, 256)
        on both, (192, 128) on FP8), and this picks LPT exactly when the row the
        live device would use has it in the effective domain for (d, d). Asking
        for LPT on another flavor would be asking for something no row claims
        (contract § 4: honoured or ineligible, never substituted). Two copies of
        one fact drift (engine-contract § 8b'), so there is no shape literal
        here. The policy is compile-time; the arm not chosen is never traced.
        """
        from cudnn.frost.device import ambient_device, compute_capability
        from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL

        d = self.geom.d_head
        arch = "sm107" if compute_capability(ambient_device()) == (10, 7) else "sm100"
        caps = self._row_capabilities(arch, self.fp8)
        domain = dict(caps.sched_policies_by_d_shape).get((d, d), caps.sched_policies)
        return SCHED_LPT if SCHED_LPT in domain else SCHED_NATURAL

    def _check_gate_geometry(self) -> None:
        """Decline ``fuse_gate`` on a head dim -- or a gate dtype -- the Rubin row does not claim.

        The adapter picks its kernel flavor from d_head (d128 / d192x128 / d256
        / d512) and the gate epilogue is wired on ONE of them.  READ OFF THE ROW
        (``Capabilities.epilogue_gate`` / ``epilogue_gate_d_shapes`` of the SM107
        engine the block runs on), the same way ``_sched_policy`` reads LPT: two
        copies of one fact drift (engine-contract § 8b'), so there is no shape
        literal here.  The adapter's standalone twin re-checks the same claim
        after the cc gate; THIS pin runs first so a d64 block declines with the
        head dim in the message on EVERY device (a d128 block would otherwise
        pass eligibility here and, on Rubin, be declined only by the adapter --
        or, before the twin existed, run a gate nobody wired at 2-4x the cost).
        Mirrors ``engines.mismatch()``: a row without the claim declines every
        head dim; a row claiming it with no shape set claims every flavor; a row
        with no ``epilogue_gate_dtypes`` reads GATE in Q's dtype.
        """
        g = self.geom
        caps = self._row_capabilities("sm107", self.fp8)
        if not caps.epilogue_gate:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True is not served by the {'FP8' if self.fp8 else 'f16/bf16'} SM107 SDPA engine row. "
                "Use fuse_gate=False (stage (5) runs as its own launch)"
            )
        shapes = caps.epilogue_gate_d_shapes
        if shapes is not None and (g.d_head, g.d_head) not in shapes:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True is served at (d_head, d_head) in {sorted(shapes)}; got {(g.d_head, g.d_head)}. "
                "Use fuse_gate=False for other head dims"
            )
        # The gate's dtype is a row claim too (``epilogue_gate_dtypes``; None =
        # "G in Q's dtype", exactly ``engines.mismatch()``'s reading -- the f16
        # row; the FP8 row claims bf16).  Pinned HERE, before the cc gate, so a
        # standalone fp8 ``_Sdpa(fuse_gate=True)`` that forgot ``gate_dtype=``
        # (the block always passes its activation dtype) names the knob instead
        # of reading as the cc decline off Rubin or as the adapter's dtype
        # ValueError on it.  No dtype literal: the domain is the row's.
        gate_dtype = self.gate_dtype or self.dtype
        if caps.epilogue_gate_dtypes is None:
            served = (self.dtype,)
        else:
            from cudnn.sdpa.graph_analyzer import to_torch_dtype

            served = tuple(to_torch_dtype(dt) for dt in sorted(caps.epilogue_gate_dtypes, key=str))
        if gate_dtype not in served:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True reads GATE in {' / '.join(str(t) for t in served)} on the "
                f"{'FP8' if self.fp8 else 'f16/bf16'} SM107 SDPA engine row; got gate_dtype={gate_dtype}. "
                "Pass gate_dtype= (the block's activation dtype) or use fuse_gate=False"
            )

    def check_support(self) -> None:
        # Geometry first, so the decline reads the same on every device (the
        # block's own stages ahead of this one are cc-independent too).
        if self.fuse_gate:
            self._check_gate_geometry()
        cc = torch.cuda.get_device_capability(self.device)
        if tuple(cc) != _SM107_CC:
            raise NotImplementedError(f"gated_attention_block targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found SM{cc[0]}{cc[1]}")
        self._impl = self._build_impl()
        # The adapter's own contract check: the dense S % 128 decline and the
        # FP8 envelope, the gate descriptor's shape / dtype / TMA-expressible
        # stride, and the standalone twins of the rows' gate claims (arch,
        # head dims, MXFP8, THD, paged, split, PackGQA -- engine-contract § 8b).
        self._impl.check_support()

    def compile(self) -> None:
        if self._impl is None:
            raise RuntimeError("call check_support() before compile()")
        self._impl.compile()

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch the SDPA carves from the block's workspace.

        0 on every block configuration (dense, unsplit: ``api_dsl.py``
        ``SdpaFwdDslSm100.scratch_workspace_bytes``); the block reserves
        ``max(..., 1)`` so the carve stays well-formed either way.
        """
        if self._impl is None:
            raise RuntimeError("call check_support() before scratch_workspace_bytes()")
        return int(self._impl.scratch_workspace_bytes())

    def execute(
        self,
        q: torch.Tensor,  # [B, S, H_q,  D]  compact or a slab column slice (token_stride)
        k: torch.Tensor,  # [B, S, H_kv, D]
        v: torch.Tensor,  # [B, S, H_kv, D]
        o: torch.Tensor,  # [B, S, H_q,  D]  compact, written
        lse: Optional[torch.Tensor] = None,  # [B, H_q, S] fp32
        seq_lens: Optional[torch.Tensor] = None,  # [B] int32, per-batch valid KV length
        workspace: Optional[torch.Tensor] = None,
        current_stream=None,
        gate: Optional[torch.Tensor] = None,  # [B, S, H_q, D] slab slice or compact gate16; fuse_gate only
        descale_q: Optional[torch.Tensor] = None,  # FP8 only: 1-element fp32 device tensors
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,  # FP8 with e4m3 O only: 1-element fp32 device tensor
    ) -> None:
        """Hand the BSHD buffers over as BHSD views. No copy — see :func:`_bhsd_desc`.

        ONE call into the adapter for every configuration; the FP8 scales and
        the gate ride as keyword arguments the adapter validates against the
        specialization it compiled (``gate`` <-> ``sample_gate``, ``amax_o``
        refused under ``has_amax_o=False``).
        """
        if self._impl is None:
            raise RuntimeError("call compile() before execute()")
        if self.fp8:
            if descale_q is None or descale_k is None or descale_v is None:
                raise ValueError(f"{self.name}: the FP8 SDPA needs descale_q/k/v (1-element fp32 device tensors)")
            # The kernel folds scale_o into inv_sum REGARDLESS of O's dtype, so
            # on a bf16 O it would scale the output the block's own quantize
            # pass then scales again.  None binds the adapter's cached 1.0.
            if scale_o is not None and self.o_dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"{self.name}: scale_o is consumed only when O is e4m3 (this stage writes {self.o_dtype}); "
                    "the unfused FP8 pipeline applies scale_o in its quantize pass"
                )
            if scale_o is None and self.o_dtype == torch.float8_e4m3fn:
                # A caller bug, not a case to paper over with a per-execute fill (Rule 1).
                raise ValueError(f"{self.name}: an e4m3 O needs scale_o (1-element fp32 device tensor)")
        elif descale_q is not None or descale_k is not None or descale_v is not None or scale_o is not None:
            raise ValueError(f"{self.name}: descales / scale_o are only consumed by the FP8 pipeline")
        if self.fuse_gate and gate is None:
            raise ValueError(f"{self.name}: fuse_gate=True requires the GATE tensor at execute")
        if not self.fuse_gate and gate is not None:
            raise ValueError(f"{self.name}: gate is only consumed under fuse_gate=True (the SDPA's epilogue gate); this stage was declared without it")
        kw = {}
        if self.fp8:
            kw.update(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o)
        if self.fuse_gate:
            kw["gate"] = gate.transpose(1, 2)
        self._impl.execute(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            o.transpose(1, 2),
            lse_tensor=lse,
            seq_kv_lens=seq_lens,
            workspace=workspace,
            current_stream=current_stream,
            **kw,
        )


class _ElementwiseStage(_Stage):
    """Shared body of stages (5) and (3b): one pass over ``[T, H, D]``.

    Both are the same kernel with a ``const_expr`` on whether a gate operand
    exists — see ``kernels/elementwise.py``. Being pure streaming with no
    reduction and no shuffle, this is the closest thing in the block to a copy
    and should sit closest to the HBM ceiling.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, heads: int, has_gate: bool, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.heads = int(heads)
        self.has_gate = bool(has_gate)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.elementwise import validate_shape

        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: bf16/f16 only, got {self.dtype}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        from .kernels.elementwise import compile_elementwise_gate

        self._recipe = compile_elementwise_gate(
            dtype=self.dtype,
            h=self.heads,
            d=self.geom.d_head,
            has_gate=self.has_gate,
            threads_per_cta=_ELEMENTWISE_THREADS,
            rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
            const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
        )

    def moved_bytes(self) -> int:
        """HBM traffic of one launch — the denominator for the SOL number."""
        from .kernels.elementwise import moved_bytes

        return moved_bytes(self.batch * self.seq_len, self.heads, self.geom.d_head, elem_bytes=_itemsize(self.dtype), has_gate=self.has_gate)

    def _run(self, src, gate, dst, current_stream):
        from .kernels.elementwise import run_elementwise_gate

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_elementwise_gate(self._recipe, src, gate, dst, stream=stream)


class _SigmoidGate(_ElementwiseStage):
    """(5) ``O_gated = O * sigmoid(GATE)``, elementwise over ``[T, H_q, D]``.

    **Hazard, and the reason ordering matters:** it must consume the SDPA's
    *substituted* O. For a fully-masked row the epilogue selects ``O := 0``;
    gating accumulator residue instead of that zero propagates NaN, and residue
    really can be a NaN bit pattern (``* 0`` is not a fix -- the substitution is
    a SELECT). Under context parallelism a rank can legitimately hold a chunk
    whose KV range is empty, so this is reachable, not theoretical.

    GATE is read as a strided COLUMN SLICE of the fused projection — no repack.
    O is gated IN PLACE, so ``O_gated`` costs no buffer.

    **Fusion status: folding this into the SDPA epilogue is the obvious next
    increment** — a pure epilogue change, one extra input in O's exact layout,
    removing a full HBM round trip of O. It pulls against the backward, which
    wants pre-gate ``O`` for ``dG``: a fused epilogue would have to write both,
    or training keeps this stage split. Decide that deliberately.
    """

    def __init__(self, geometry, *, batch, seq_len, dtype):
        super().__init__(geometry, batch=batch, seq_len=seq_len, dtype=dtype, heads=geometry.h_q, has_gate=True, name="sigmoid_gate")

    def execute(self, o: torch.Tensor, gate: torch.Tensor, out: torch.Tensor, current_stream=None) -> None:
        self._run(o, gate, out, current_stream)


class _VCompaction(_ElementwiseStage):
    """(3b) copy V out of the fused projection into a compact buffer.

    **This stage exists only because stage (1) is the UNFORKED FROST GEMM.** Q
    and K are de-interleaved for free by (2)+(3), which already read and write
    them; V is untouched between the projection and the SDPA, so without this it
    would reach the SDPA as a padded-stride view that ``_to_bshd`` silently
    ``.contiguous()``-copies — a hidden kernel inside the adapter (Rule 2).
    Doing it here makes the cost NAMED and MEASURED instead.

    Forking stage (1) to write four compact buffers (§ 1) deletes this outright.
    Until then it is 1/16 of Q's traffic, which is why it is an acceptable v1.
    """

    def __init__(self, geometry, *, batch, seq_len, dtype):
        super().__init__(geometry, batch=batch, seq_len=seq_len, dtype=dtype, heads=geometry.h_kv, has_gate=False, name="compact_v")

    def execute(self, v_src: torch.Tensor, v_dst: torch.Tensor, current_stream=None) -> None:
        self._run(v_src, None, v_dst, current_stream)


# ---------------------------------------------------------------------------
# 6. The public API
# ---------------------------------------------------------------------------


class GatedAttentionBlockFwd(APIBase):
    """Gated attention block, forward. One call, one workspace, three to five launches.

    No ``Sm1xx`` suffix on the class: arch is a directory axis under ``kernels/``
    and a per-stage dispatch, not part of the user-visible name. A shipped name
    cannot be renamed, so the arch stays out of it.

    **The default pipeline (``inplace_qkv``, no fusion knobs) is FOUR launches;
    the two fusion knobs take it to three**::

        (1) proj         h            -> PROJ [T, N]            FROST GEMM        | fuse_norm_rope: (1)+(2)+(3) in ONE
        (2+3) norm+rope  PROJ[Q],[K]  -> in place  (+rstd)      this block's kernel| launch (the GEMM fork's epilogue)
        (3b) compact     PROJ[V]      -> V_c                    only when inplace_qkv=False
        (4) sdpa         PROJ[Q,K,V]  -> O  (+LSE)              FROST SDPA        | fuse_gate: (4)+(5) in ONE launch
        (5) gate         O, PROJ[G]   -> O in place             this block's kernel| (the SDPA's epilogue_gate)
        (6) out_proj     O            -> out                    FROST GEMM

    Both fusions are inference-only: each overwrites a tensor the backward needs
    (``q_pre``/``k_pre``, pre-gate ``O``) and is declined with ``save_for_backward``.

    Stage (3b) exists only because stage (1) is the UNFORKED FROST GEMM, which
    writes one fused ``[T, N]``. Q and K are de-interleaved for FREE by (2+3),
    which already reads and writes them; V is untouched between the projection
    and the SDPA, so without (3b) it would reach the SDPA as a padded-stride view
    that ``_to_bshd`` silently ``.contiguous()``-copies — a hidden kernel inside
    the adapter, which Rule 2 bans. **Forking stage (1) to write four compact
    buffers (§ 1) deletes (3b) outright**; until then it is the measurable price
    of not having forked, and V is 1/16 of Q so the price is small.

    ``O_gated`` needs no buffer: stage (5) gates O in place.
    """

    def __init__(
        self,
        sample_h: torch.Tensor,  # [B, S, d_model]
        sample_w_qkvg: torch.Tensor,  # [N, d_model], N = (2*H_q + 2*H_kv) * D
        sample_w_q_norm: torch.Tensor,  # [D]
        sample_w_k_norm: torch.Tensor,  # [D]
        sample_cos: torch.Tensor,  # [B, S, ROPE_DIM] -- see "RoPE table contract"
        sample_sin: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_w_o: torch.Tensor,  # [d_model, H_q * D]
        sample_out: torch.Tensor,  # [B, S, d_model]
        geometry: GatedAttentionBlockGeometry,
        *,
        return_lse: bool = False,
        save_for_backward: bool = False,  # implies return_lse; see SavedForBackward
        seq_lens_present: bool = False,
        inplace_qkv: Optional[bool] = None,  # None -> not save_for_backward
        fuse_norm_rope: bool = False,  # stages (2)+(3) inside stage (1)'s epilogue; needs inplace_qkv
        fuse_gate: bool = False,  # stage (5) inside stage (4)'s epilogue; inference only (no pre-gate O)
        quant: Optional[QuantSpec] = None,  # FP8 (E4M3) pipeline with static per-tensor scales; None = bf16/f16
    ):
        super().__init__()
        self._warn_experimental_api()
        self.geom = geometry
        self.dtype = sample_h.dtype
        self.device = sample_h.device
        # FP8: `h` and both weights arrive as e4m3 with a QuantSpec; every
        # activation the block's own kernels touch (slab, O, out, cos/sin, norm
        # weights) is bf16 -- the "activation dtype".  bf16/f16: the two coincide.
        self.quant = quant
        if (self.dtype == torch.float8_e4m3fn) != (quant is not None):
            raise ValueError(
                "FP8 needs both halves: an e4m3 `h` AND a QuantSpec (static scales). "
                f"Got h.dtype={self.dtype}, quant={'set' if quant is not None else 'None'}."
            )
        if quant is not None:
            quant.validate()
        self.act_dtype = torch.bfloat16 if quant is not None else self.dtype
        if sample_h.ndim != 3:
            raise ValueError(f"sample_h must be [B, S, d_model], got {tuple(sample_h.shape)}")
        self.batch, self.seq_len, d_model = (int(x) for x in sample_h.shape)
        if d_model != geometry.d_model:
            raise ValueError(f"sample_h last dim {d_model} != geometry.d_model {geometry.d_model}")
        self.save_for_backward = bool(save_for_backward)
        self.return_lse = bool(return_lse) or self.save_for_backward
        self.seq_lens_present = bool(seq_lens_present)
        # IN-PLACE Q/K/V. Norm+RoPE writes back over its own columns of the
        # fused projection and V is never moved, so stage (3b) disappears and
        # the three compact buffers with it -- 26% of the workspace.
        # Default: ON for inference, OFF the moment the backward needs q_pre /
        # k_pre. Explicit True with save_for_backward RAISES rather than
        # silently costing the caller a tensor the backward cannot rebuild.
        self.inplace_qkv = (not self.save_for_backward) if inplace_qkv is None else bool(inplace_qkv)
        if self.inplace_qkv and self.save_for_backward:
            raise ValueError(
                "inplace_qkv=True is incompatible with save_for_backward=True: norming in place destroys q_pre/k_pre, "
                "which the RMSNorm backward needs and cannot reconstruct (dividing out the norm weight is undefined at "
                "a zero weight and hostile at a small one -- see SavedForBackward). Pass inplace_qkv=False, or recompute "
                "q_pre/k_pre from h by re-running the Q and K slices of the projection."
            )

        # FUSED norm+RoPE: stage (1)'s fork norms the Q/K tiles in its epilogue
        # and writes the slab the in-place path already reads.  OFF by default
        # until the perf node has ranked it against the unfused chain; it is
        # never a silent choice.  It has no pre-norm Q/K to hand a backward, so
        # it rides the same guard as inplace_qkv.
        self.fuse_norm_rope = bool(fuse_norm_rope)
        if self.fuse_norm_rope and not self.inplace_qkv:
            raise ValueError(
                "fuse_norm_rope=True writes normed Q/K straight into the projection slab (the in-place layout) and never "
                "materialises q_pre/k_pre, so it requires inplace_qkv=True and is incompatible with save_for_backward=True. "
                "Pass fuse_norm_rope=False for training."
            )

        # FUSED GATE: the production d256 SDPA's epilogue_gate specialization
        # reads GATE in its epilogue and writes O_gated, so stage (5) is not
        # built.  It overwrites the one tensor the backward's dG needs (pre-gate
        # O -- SavedForBackward), so training declines it here, the same way
        # inplace_qkv / fuse_norm_rope do.
        self.fuse_gate = bool(fuse_gate)
        # FP8 serves exactly TWO configurations: UNFUSED (7 launches) and FULLY
        # FUSED (3 launches: proj(+norm+rope+quant) -> sdpa(+gate, e4m3 O) ->
        # out_proj).  Each half-fused combination would be its own
        # specialization (bf16 O + a quantize pass, or a gate read from the
        # bf16 slab) that nobody has validated -- declined, typed, naming both knobs.
        if quant is not None and self.fuse_gate != self.fuse_norm_rope:
            raise NotImplementedError(
                "the FP8 pipeline is either fully fused or unfused: fuse_norm_rope and fuse_gate must BOTH be True "
                f"(3 launches) or BOTH be False (7 launches); got fuse_norm_rope={self.fuse_norm_rope}, fuse_gate={self.fuse_gate}"
            )
        if quant is not None and self.save_for_backward:
            raise NotImplementedError("the FP8 pipeline is inference-only for now (no q_pre/k_pre/pre-gate O contract under quantization)")
        if quant is not None and self.seq_lens_present:
            # KERNEL BUG, not a design choice: the Rubin per-tensor FP8 d256 SDPA
            # (`sdpa/fwd/kernels/sm107/prefill_d256_fp8.py`) HANGS (exit 124) on a
            # batch entry with seq_kv_lens == 0 -- first launch at S=1000, second
            # launch at S=512; non-empty padding is fine; the bf16 sibling handles
            # the identical case.  Repro: `frost_dev/probe_sdpa_fp8_d256.py
            # --lsepad-dead`.  Declined here until the kernel's empty-mainloop path
            # is fixed, because a hang is worse than a decline.
            raise NotImplementedError(
                "FP8 + seq_lens_present is declined: the Rubin FP8 d256 SDPA kernel hangs on an empty (seq_kv_lens == 0) batch entry "
                "(frost_dev/probe_sdpa_fp8_d256.py --lsepad-dead). Use bf16 for padded batches until that kernel is fixed."
            )
        if self.fuse_gate and self.save_for_backward:
            raise ValueError(
                "fuse_gate=True writes O_gated in place of O and never materialises the pre-gate O, which the backward's "
                "dG needs (see SavedForBackward: `o` is saved because dG needs pre-gate O; o_gated is recomputable, o is not). "
                "Pass fuse_gate=False for training."
            )

        self._descs = {
            "w_qkvg": self._make_tensor_desc(sample_w_qkvg, name="w_qkvg"),
            "w_q_norm": self._make_tensor_desc(sample_w_q_norm, name="w_q_norm"),
            "w_k_norm": self._make_tensor_desc(sample_w_k_norm, name="w_k_norm"),
            "cos": self._make_tensor_desc(sample_cos, name="cos"),
            "sin": self._make_tensor_desc(sample_sin, name="sin"),
            "w_o": self._make_tensor_desc(sample_w_o, name="w_o"),
            "out": self._make_tensor_desc(sample_out, name="out"),
        }

        fp8 = quant is not None
        # FULLY FUSED FP8: both knobs under a QuantSpec (the both-or-neither
        # decline above makes `fp8 and fuse_gate` == `fp8 and fuse_norm_rope`).
        self.fp8_fused = fp8 and self.fuse_gate and self.fuse_norm_rope
        act = self.act_dtype
        self._quant_dev = None  # the QuantSpec as device scalars, materialised in compile()
        if self.fuse_norm_rope:
            # bf16: writes the [T, N] slab.  FP8: the second rendering writes the
            # compact e4m3 q8/k8/v8 + the bf16 gate16 buffer, quantizing in its epilogue.
            self._proj = _FusedQkvProjection(
                geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, want_rstd=self.save_for_backward, quant=quant, device=self.device
            )
            self._norm_rope = None  # lives in the fused epilogue
        else:
            # FP8: e4m3 x e4m3 -> fp32 -> * (descale_h * descale_w_qkvg) -> bf16 slab.
            self._proj = _qkv_gate_projection(geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, out_dtype=act, alpha=fp8)
            self._norm_rope = _QkNormRope(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, want_rstd=self.save_for_backward)
        # Stage (3b) exists ONLY to give the SDPA a compact V. In-place needs no
        # such thing, so the stage is not built at all rather than built and
        # skipped -- a stage that is never run should not be in `_stages`,
        # where it would still be compiled and still report support.  Under
        # FP8 the quantize stages compact V (and Q, K) on the way to e4m3.
        self._compact_v = None if (self.inplace_qkv or fp8) else _VCompaction(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act)
        # (3q) UNFUSED FP8 only: bf16 slab slices -> compact e4m3 Q/K/V.  Two
        # recipes (h_q and h_kv); K and V share the h_kv one.  Fully fused FP8
        # quantizes in the projection fork's epilogue and builds none of them.
        quantize = fp8 and not self.fp8_fused
        self._quant_q = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, name="quantize_q") if quantize else None
        self._quant_kv = (
            _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_kv, name="quantize_kv") if quantize else None
        )
        if self.fp8_fused:
            # Q/K/V are the COMPACT e4m3 q8/k8/v8 the projection fork writes
            # (token_stride 0 -- what the unfused FP8 SDPA reads too), the GATE
            # is the compact bf16 gate16, O comes out e4m3 (o8).  The gated FP8
            # SDPA specialization (epilogue_gate) is compiled at compact strides.
            sdpa_token_stride, sdpa_gate_token_stride, sdpa_o_dtype = 0, geometry.h_q * geometry.d_head, torch.float8_e4m3fn
        else:
            # bf16: in-place reads the slab at its padded stride; FP8 unfused
            # reads the compact e4m3 buffers.  GATE is always a slab column slice.
            sdpa_token_stride, sdpa_gate_token_stride, sdpa_o_dtype = (geometry.n_qkvg if self.inplace_qkv else 0) if not fp8 else 0, geometry.n_qkvg, act
        self._sdpa = _Sdpa(
            geometry,
            batch=self.batch,
            seq_len=self.seq_len,
            dtype=self.dtype,
            device=self.device,
            want_lse=self.return_lse,
            seq_lens_present=self.seq_lens_present,
            token_stride=sdpa_token_stride,
            fuse_gate=self.fuse_gate,
            gate_token_stride=sdpa_gate_token_stride,
            o_dtype=sdpa_o_dtype,
            gate_dtype=act,  # gate16 / the slab's GATE columns are the activation dtype (bf16 under FP8)
        )
        # Stage (5) lives in the SDPA kernel's gate epilogue under fuse_gate --
        # not built rather than built and skipped (it would still compile).
        self._gate = None if self.fuse_gate else _SigmoidGate(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act)
        # (5q) UNFUSED FP8 only: bf16 gated O -> compact e4m3 for the out
        # projection (same [T, H_q, D] shape as Q, so the Q recipe serves it).
        self._quant_o = self._quant_q
        self._out_proj = _out_projection(geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, out_dtype=act, alpha=fp8)
        self._stages = tuple(
            st for st in (self._proj, self._norm_rope, self._compact_v, self._quant_q, self._quant_kv, self._sdpa, self._gate, self._out_proj) if st is not None
        )
        self._ws = None

    # -- support ------------------------------------------------------------

    def check_support(self) -> bool:
        """Validate the declaration, then ask every stage in turn.

        Whatever this ACCEPTS, the stages must address natively — acceptance is a
        promise about the execute path, not about what the adapter can patch up
        (Rule 2). A ``ValueError`` escaping a stage's own config builder after
        this returned True is a bug HERE, not user error.
        """
        self.geom.validate()
        if self.save_for_backward and not self.return_lse:
            raise ValueError("save_for_backward requires the LSE: the SDPA backward cannot run without it")
        g = self.geom
        self._check_tensor_shape(self._descs["w_qkvg"], (g.n_qkvg, g.d_model), "w_qkvg")
        self._check_tensor_shape(self._descs["w_o"], (g.d_model, g.h_q * g.d_head), "w_o")
        for nm in ("w_q_norm", "w_k_norm"):
            self._check_tensor_shape(self._descs[nm], (g.d_head,), nm)
        for nm in ("cos", "sin"):
            self._check_tensor_shape(self._descs[nm], (self.batch, self.seq_len, g.rope_dim), nm)
        self._check_tensor_shape(self._descs["out"], (self.batch, self.seq_len, g.d_model), "out")
        # dtype contract: bf16/f16 everywhere, or (FP8) e4m3 h + weights with
        # every activation-side tensor in the bf16 activation dtype.
        for nm in ("w_qkvg", "w_o"):
            if self._descs[nm].dtype != self.dtype:
                raise ValueError(f"{nm} must have h's dtype {self.dtype}, got {self._descs[nm].dtype}")
        for nm in ("w_q_norm", "w_k_norm", "cos", "sin", "out"):
            if self._descs[nm].dtype != self.act_dtype:
                raise ValueError(f"{nm} must be the activation dtype {self.act_dtype}, got {self._descs[nm].dtype}")
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    # -- workspace ----------------------------------------------------------

    def _layout(self) -> "_Intermediates":
        return _plan_workspace(
            self.geom,
            self.batch,
            self.seq_len,
            self.act_dtype,
            self.return_lse,
            self.save_for_backward,
            self.inplace_qkv,
            fp8=self.quant is not None,
            fp8_fused=self.fp8_fused,
        )

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide: every intermediate, plus each
        sub-engine's own scratch at a reserved offset.

        Honest and never exceeded (contract § 10) — that is what keeps the block
        CUDA-graph capturable with stable pointers.
        """
        self._ensure_support_checked()
        lay = self._layout()
        engine = max(self._proj.workspace_bytes(), self._out_proj.workspace_bytes(), self._sdpa.scratch_workspace_bytes(), 1)
        return lay.total_bytes + _align_up(engine)

    # -- compile ------------------------------------------------------------

    def compile(self) -> None:
        """Build all five artifacts. Plan-time keys only, so every execute-path
        dispatch is a guaranteed cache hit (Rule 4)."""
        self._ensure_support_checked()
        for st in self._stages:
            st.compile()
        self._ws = self._layout()
        if self.quant is not None:
            # Plan-time constants (contract § 10 allows compile-time buffers; the
            # execute path never allocates): one fp32 device scalar per scale.
            q = self.quant

            def _dev(v: float) -> torch.Tensor:
                return torch.full((1,), float(v), dtype=torch.float32, device=self.device)

            self._quant_dev = dict(
                alpha_qkvg=_dev(q.alpha_qkvg),
                alpha_o=_dev(q.alpha_o),
                scale_q=_dev(q.scale_q),
                scale_k=_dev(q.scale_k),
                scale_v=_dev(q.scale_v),
                scale_o=_dev(q.scale_o),
                descale_q=_dev(1.0 / q.scale_q),
                descale_k=_dev(1.0 / q.scale_k),
                descale_v=_dev(1.0 / q.scale_v),
            )

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        h: torch.Tensor,
        w_qkvg: torch.Tensor,
        w_q_norm: torch.Tensor,
        w_k_norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        saved: Optional[SavedForBackward] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the five stages in pipeline order.

        No allocation, no D2H read, no implicit conversion: every intermediate is
        a strided VIEW of the caller's workspace.
        """
        if self._ws is None:
            raise RuntimeError("call compile() before execute()")
        g = self.geom
        t = self.batch * self.seq_len
        # THE launch stream (Rule 5): the caller's ``current_stream``, else torch's
        # current stream on h's device -- resolved once, here, and handed to EVERY
        # stage.  The CuTe-DSL kernels take the raw CUstream int directly; the two
        # FROST GEMMs take it through ``run_proj_gemm(stream=)`` (the JIT plan's
        # own ``stream=``, or a cached per-(device, stream) cuDNN handle bound to
        # it on the graph route); the SDPA adapter takes it as ``current_stream``.
        # No stage derives its own stream from torch's current one, so a block
        # run under ``with torch.cuda.stream(s):`` -- or with an explicit stream --
        # cannot split across two streams (the GEMMs used to launch on the
        # default stream regardless, and the SDPA read an unwritten slab).
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(h.device).cuda_stream
        ws = self._ws
        req = self.get_workspace_size()
        if workspace.numel() < req:
            raise ValueError(f"workspace is {workspace.numel()} bytes, need {req}")

        fp8 = self.quant is not None
        act = self.act_dtype
        if self.fp8_fused:
            self._execute_fp8_fused(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream)
            return
        proj = _view(workspace, ws.proj, (t, g.n_qkvg), act)
        # In-place: there ARE no compact Q/K/V buffers -- the SDPA reads the
        # slab columns directly, so these are the same views stage (2)+(3)
        # normed over. Bound below, after `q_src`/`k_src`/`v_src` exist.
        q_c = k_c = v_c = None
        if not self.inplace_qkv and not fp8:
            q_c = _view(workspace, ws.q, (t, g.h_q, g.d_head), act)
            k_c = _view(workspace, ws.k, (t, g.h_kv, g.d_head), act)
            v_c = _view(workspace, ws.v, (t, g.h_kv, g.d_head), act)
        o = _view(workspace, ws.o, (t, g.h_q, g.d_head), act)
        engine_ws = workspace[ws.engine_scratch :]
        if fp8:
            e4 = torch.float8_e4m3fn
            q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
            k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
            v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
            o8 = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4)
            qd = self._quant_dev

        o_q, o_g, o_k, o_v = g.qkvg_offsets
        # Column slices of the fused projection, as strided views. Every consumer
        # addresses these strides natively -- no repack anywhere (Rule 2).
        q_src = _cols(proj, o_q, g.h_q, g.d_head)
        gate_src = _cols(proj, o_g, g.h_q, g.d_head)
        k_src = _cols(proj, o_k, g.h_kv, g.d_head)
        v_src = _cols(proj, o_v, g.h_kv, g.d_head)

        rstd_q = rstd_k = None
        if self.save_for_backward:
            if saved is None:
                raise ValueError("save_for_backward=True requires a SavedForBackward to write through")
            rstd_q, rstd_k = saved.rstd_q, saved.rstd_k

        if self.fuse_norm_rope:
            # (1)+(2)+(3): the fork norms + rotates the Q/K tiles in its
            # epilogue and writes the slab; rstd (if wanted) comes out of the
            # same launch.
            self._proj.execute(
                h.view(t, g.d_model),
                w_qkvg,
                proj,
                w_q_norm,
                w_k_norm,
                cos,
                sin,
                rstd_q if rstd_q is None else rstd_q.view(t, g.h_q),
                rstd_k if rstd_k is None else rstd_k.view(t, g.h_kv),
                stream=stream,
            )
        else:
            # (1) [FP8: e4m3 x e4m3, epilogue * alpha_qkvg, bf16 slab]
            self._proj.execute(h.view(t, g.d_model), w_qkvg, proj, engine_ws, alpha=qd["alpha_qkvg"] if fp8 else None, stream=stream)
            # (2)+(3). q_out/k_out=None means IN PLACE, which is the stage's own
            # default and is safe by construction: every lane holds its whole [D]
            # row in registers before it stores, and no lane touches another's.
            self._norm_rope.execute(
                q_src, k_src, w_q_norm, w_k_norm, cos, sin, q_out=q_c, k_out=k_c, rstd_q=rstd_q, rstd_k=rstd_k, current_stream=stream, flat=True
            )
        if fp8:
            # (3q) bf16 slab slices -> compact e4m3.  This IS the compaction the
            # FP8 SDPA needs (its adapter path takes no declared slab strides).
            self._quant_q.execute(q_src, q8, qd["scale_q"], current_stream=stream)
            self._quant_kv.execute(k_src, k8, qd["scale_k"], current_stream=stream)
            self._quant_kv.execute(v_src, v8, qd["scale_v"], current_stream=stream)
            q_c, k_c, v_c = q8, k8, v8
        elif self.inplace_qkv:
            # The normed Q/K are the slab columns; V was never moved. All three
            # go to the SDPA at the slab's token stride -- no stage (3b), no
            # compact buffers, no copy anywhere.
            q_c, k_c, v_c = q_src, k_src, v_src
        else:
            # (3b) -- exists only to hand the SDPA a compact V.
            self._compact_v.execute(v_src, v_c, current_stream=stream)
        # (4) [+ (5) under fuse_gate: the SDPA gates the SUBSTITUTED O inside
        # its epilogue, after the dead-row select, and writes O_gated.]
        self._sdpa.execute(
            q_c.view(self.batch, self.seq_len, g.h_q, g.d_head),
            k_c.view(self.batch, self.seq_len, g.h_kv, g.d_head),
            v_c.view(self.batch, self.seq_len, g.h_kv, g.d_head),
            o.view(self.batch, self.seq_len, g.h_q, g.d_head),
            lse=lse,
            seq_lens=seq_lens,
            workspace=engine_ws,
            current_stream=cuda.CUstream(stream),
            gate=gate_src.view(self.batch, self.seq_len, g.h_q, g.d_head) if self.fuse_gate else None,
            descale_q=qd["descale_q"] if fp8 else None,
            descale_k=qd["descale_k"] if fp8 else None,
            descale_v=qd["descale_v"] if fp8 else None,
        )
        if not self.fuse_gate:
            # (5) -- gates the SUBSTITUTED O: the SDPA epilogue already selected
            # O := 0 on dead rows, so no residue reaches the sigmoid.
            self._gate.execute(o, gate_src, o, current_stream=stream)
        if fp8:
            # (5q) bf16 gated O -> e4m3 for the FP8 out projection; (6) folds
            # descale_o * descale_w_o into its epilogue.
            self._quant_o.execute(o, o8, qd["scale_o"], current_stream=stream)
            self._out_proj.execute(o8.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)
        else:
            # (6)
            self._out_proj.execute(o.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, stream=stream)

    def _execute_fp8_fused(self, h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream) -> None:
        """The FULLY FUSED FP8 pipeline: three launches, three workspace buffers.

        ::

            (1') proj+norm+rope+quant  h8, W8              -> q8 [T, H_q, D] / k8, v8 [T, H_kv, D] e4m3 (COMPACT), gate16 [T, H_q, D] bf16
            (4') sdpa+gate             q8, k8, v8, gate16  -> o8 [T, H_q, D] e4m3   (descale_v*scale_o folded, gate after the select)
            (6)  out_proj              o8                  -> out                    (alpha_o epilogue)

        No bf16 slab, no bf16 O, no quantize pass: every intermediate is a view
        of the five slots ``_plan_workspace`` reserved.  Q/K/V are compact BSHD
        (``token_stride == 0``), the SAME layout the unfused FP8 SDPA reads --
        the 2-D ``[T, h*d]`` views go to the GEMM runner, the ``[B, S, H, d]``
        views of the same bytes to the gated FP8 SDPA; nothing is strided,
        nothing is repacked (Rule 2).
        """
        g = self.geom
        b, s = self.batch, self.seq_len
        t = b * s
        ws = self._ws
        qd = self._quant_dev
        e4 = torch.float8_e4m3fn
        q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
        k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
        v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
        gate16 = _view(workspace, ws.gate16, (t, g.h_q, g.d_head), self.act_dtype)
        o8 = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4)
        engine_ws = workspace[ws.engine_scratch :]
        # (1'): alpha_qkvg / scale_q / scale_k / scale_v ride the stage's qscal vector.
        # The runner takes the four outputs 2-D ([T, h*d]); the [T, H, D] / [B, S, H, D] views are for the SDPA.
        self._proj.execute_fp8(
            h.view(t, g.d_model),
            w_qkvg,
            q8.view(t, g.h_q * g.d_head),
            k8.view(t, g.h_kv * g.d_head),
            v8.view(t, g.h_kv * g.d_head),
            gate16.view(t, g.h_q * g.d_head),
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            stream=stream,
        )
        # (4'): e4m3 in, e4m3 out; the SDPA gates the SUBSTITUTED O (after the
        # dead-row select, per element) and casts once, with saturation.
        self._sdpa.execute(
            q8.view(b, s, g.h_q, g.d_head),
            k8.view(b, s, g.h_kv, g.d_head),
            v8.view(b, s, g.h_kv, g.d_head),
            o8.view(b, s, g.h_q, g.d_head),
            lse=lse,
            seq_lens=seq_lens,
            workspace=engine_ws,
            current_stream=cuda.CUstream(stream),
            gate=gate16.view(b, s, g.h_q, g.d_head),
            descale_q=qd["descale_q"],
            descale_k=qd["descale_k"],
            descale_v=qd["descale_v"],
            scale_o=qd["scale_o"],
        )
        # (6): e4m3 O_gated @ W_o^T with (1/scale_o) * descale_w_o in the epilogue.
        self._out_proj.execute(o8.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)


# ---------------------------------------------------------------------------
# 7. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------


def gated_attention_block_forward(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: torch.Tensor,
    w_k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    save_for_backward: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate outputs + workspace, cache the compiled block, and run it.

    Returns ``{"out": ..., "lse": ..., "saved": ...}``; the optional entries are
    ``None`` unless requested. Allocation happens HERE and only here — the class
    API above never allocates.
    """
    raise NotImplementedError("gated_attention_block_forward")
