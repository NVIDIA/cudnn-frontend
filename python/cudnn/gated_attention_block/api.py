# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, forward -- projection, QK-norm/RoPE, SDPA, gate, out
projection as one FROST block (bf16 / FP8 / MXFP8, optional MXFP4 weights and fp4 O).

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
     |          ``geometry.qk_norm=False`` drops the RMSNorm: RoPE-only Q/K, no
     |          norm weights (pass None), no rstd; [ROPE_DIM, D) copied bit-exactly.
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

* **UNFUSED** (both fusion knobs off; 7 stages = 9 kernel launches, the Q/K/V
  quantize stage being three launches): FP8 projections with the descale
  folded into a scalar-multiply epilogue (bf16 out), bf16 norm+RoPE in place,
  two quantize passes (Q/K/V, then gated O) feeding the Rubin per-tensor FP8
  SDPA and the FP8 out projection.  The quantize passes are the visible price
  of "unfused".
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
  inference-only.  FP8 + ``seq_lens_present`` (a dense padding mask, incl. an
  EMPTY entry) is SERVED since 2026-09-15: the Rubin FP8 d256 SDPA's
  empty-KV-entry hang is gone (8/8 fresh processes at S=1000 / 512), and the
  block's dead-entry oracle test pins ``out[dead] == 0`` exactly.

**MXFP8 (2026-09-15, PR-B): an E4M3 + per-32-block E8M0 pipeline is wired in
the same TWO configurations** -- pass e4m3 ``h`` / ``W_qkvg`` codes, their
F8_128x4 scale-factor blobs (``sample_h_sf`` / ``sample_w_qkvg_sf`` at
declaration, ``h_sf`` / ``w_qkvg_sf`` at execute) and an :class:`MxQuantSpec`:

* **UNFUSED** (9 stages = 9 kernel launches -- the same launch count as
  unfused FP8, whose 7 stages also issue 9): the block-scale FROST GEMM (``block_scale=True``,
  the E8M0 dequant is exact and happens IN the MMA, so there is no ``alpha``)
  writes the bf16 slab; norm+RoPE in place; THREE ``quantize_mxfp8`` launches
  (Q / K ROWWISE along D, V COLUMNWISE along S -- ``kernels/quantize_mxfp8.py``)
  write compact e4m3 ``q8`` / ``k8`` / ``v8`` + the SDPA's own F8_128x4 SF
  blobs (``sf_q`` / ``sf_k`` / ``sf_v``, one 1 KiB tile per ``(b, h, 128-row
  tile)``; V's is D-PLANE-MAJOR, ``mma-tma-matrix.md`` § 7); the production
  ``sm107/prefill_d256_mxfp8.py`` (``SdpaFwdDslSm100(pertensor_fp8=False)``,
  NATURAL at (256, 256), cga1) writes bf16 O; the bf16 gate; a PER-TENSOR
  ``quantize_o`` (``MxQuantSpec.scale_o``; D1 -- never the rowwise recipe) and
  the per-tensor FP8 ``out_proj`` with ``alpha_o = descale_w_o / scale_o``.
* **FULLY FUSED** (``fuse_norm_rope=True, fuse_gate=True``, 3 launches): the
  MXFP8 GEMM fork twin (``kernels/proj_gemm_norm_rope_mxfp8.py`` via
  ``run_fused_proj_gemm_mxfp8``) norms / rotates / BLOCK-quantizes in its
  epilogue and writes ``q8`` / ``k8`` / ``v8`` + ``sf_q`` / ``sf_k`` / ``sf_v``
  + bf16 ``gate16``; the gated production MXFP8 SDPA writes e4m3 O UNSCALED
  (the kernel has no per-tensor ``scale_o`` -- D8: ``MxQuantSpec.scale_o``
  must be 1.0 on this path, typed decline otherwise); FP8 ``out_proj``.  The
  fork twin is feature-detected on the runner name (``_fork_supports_field``),
  so the typed ``NotImplementedError`` returns only on a checkout without the
  twin -- never a silently un-quantized path.
* Same envelope as FP8: inference-only, both knobs or neither, e4m3 only
  (E5M2 is a typed decline), ``d_model % 128 == 0`` (whole SF atoms along K),
  padding (``seq_lens_present``) served with the same dead-entry contract.

**fp4 (2026-09-17): two modes RIDE the MXFP8 pipeline as appended
:class:`MxQuantSpec` fields** (unrepresentable on :class:`QuantSpec` / bf16
rather than declined); every existing caller and every existing workspace
offset is byte-identical (pinned by a frozen layout snapshot):

* **MXFP4 weights** (``w_qkvg_dtype=torch.float4_e2m1fn_x2``): ``W_qkvg``
  arrives as packed e2m1 codes ``[n_qkvg, d_model // 2]`` (two per byte, LOW
  nibble = even k; the STORAGE shape is what ``check_support`` checks) with the
  UNCHANGED E8M0 / 32 ``w_qkvg_sf``; stage (1) runs the FROST catalog's MIXED
  block-scale row (``fp8_e4m3 x fp4_e2m1``, E8M0 per 32 on both sides) --
  the same 9 launches, no new stage, no new slot.  UNFUSED only: the fused
  MXFP8 projection fork is rendered for an e4m3 B, so ``fuse_norm_rope`` with
  an e2m1 ``W_qkvg`` is a feature-detected typed ``NotImplementedError``
  (``NormRopeFusionParams.weight_fp4``), inverting the day the arm lands.
* **fp4 O** (``o_fp4=Fp4Format.NVFP4 | Fp4Format.MXFP4``; ONE enum member =
  e2m1 codes x scale dtype x block, e4m3 / 16 or E8M0 / 32, so an illegal
  pairing cannot be spelled): the per-tensor tail (``quantize_o`` +
  ``alpha_o`` FP8 ``out_proj``) is replaced by ``kernels/quantize_fp4.py``
  (bf16 gated O -> e2m1 codes ``o4`` + the out-projection GEMM's PADDED
  F8_128x4 blob ``sf_o``, sized by ``proj_gemm.sf_blob_bytes`` -- the GEMM
  contract, never the SDPA's ``_sf_slot_bytes``) and the fp4 x fp4
  block-scale ``out_proj`` against an e2m1 ``W_o`` ``[d_model, H_q*D // 2]``
  of the SAME format with its blob ``sample_w_o_sf`` / ``w_o_sf`` (appended
  arguments, required iff ``o_fp4``, refused otherwise).  No per-tensor scale
  on either side: ``scale_o`` and ``descale_w_o`` MUST be 1.0 (typed
  ``ValueError`` -- the block-scale GEMM has no alpha to carry them).
  UNFUSED: still 9 launches (``quantize_fp4_o`` in ``quantize_o``'s place);
  FULLY FUSED: **4 launches** -- the gated MXFP8 SDPA writes **bf16** O
  (``sdpa_o_dtype = act``; no SDPA ``Capabilities`` change, so the
  support-matrix tracker is untouched), then the fp4 quantize, then the fp4
  ``out_proj``.  Workspace: ``o8`` is not reserved (-1); ``o4`` / ``sf_o`` are
  appended at the END of the arm.  ``d_head % (4 * block) == 0`` (whole
  4-block scale words per head; d=256 passes both formats), else a typed
  ``NotImplementedError``.  A dead ragged entry quantizes to codes 0 exactly
  (NVFP4 scale = the ``2^-9`` e4m3 floor, MXFP4 scale byte ``0x00``).
* Both compose (row 9: the mixed GEMM at (1), the fp4 tail at (5q')/(6')); both
  are inference-only like every quantized pipeline (``save_for_backward`` is
  the same typed decline).  NOT served: an fp4 ``h``, an fp4 ``W_o`` against
  an e4m3 O, e4m3 scales at block 32 / E8M0 at block 16, a global (per-tensor)
  scale on either fp4 side.

Three facts the MXFP8 design rests on, kept here because a port will drop them:

* V is quantized COLUMNWISE for the BMM2, and its MXFP8 scale factors are
  D-PLANE-MAJOR in GMEM while Q/K's are per-tile contiguous -- reusing one SF
  layout for all three is a silent wrong answer that only shows up at
  ``d > 128`` and ``S > TILE_N`` (``mma-tma-matrix.md`` § 7).  The SF-order
  S-sweep in ``test_block_mxfp8.py`` is the detector.
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
an MXFP8 (e4m3 block-scaled) O / ``out_proj`` (D1 keeps the e4m3 O per-tensor; the
block-scaled out projection exists only in the fp4 formats above); the graph-API
engine row. This is a frontend-only OSS API first; a manifest family +
``Capabilities`` comes when the stages exist to be honest about.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Tuple, Union

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
_QUANTIZE_MXFP8_THREADS = 256  # keep in step with kernels/quantize_mxfp8.py DEFAULT_THREADS_PER_CTA (>= the 64 burst lanes of a 1 KiB SF tile)
_QUANTIZE_FP4_THREADS = 256  # keep in step with kernels/quantize_fp4.py DEFAULT_THREADS_PER_CTA (>= the 128 burst lanes of a 2 KiB nvfp4 SF tile at d=256)
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

    # QK-RMSNorm on/off. False: RoPE-only Q/K -- no per-head RMSNorm, so no
    # norm weights (the block takes None in their slots, both directions
    # checked) and no rstd (SavedForBackward.rstd_q/rstd_k are None). The
    # kernels fold the norm out at trace time (presence of the weight tensors,
    # like want_rstd); the dims [rope_dim, d_head) are then a bit-exact copy.
    # ``qk_norm_eps`` stays validated > 0 regardless (D10): the fused fork's
    # own validator re-checks it after eligibility, and relaxing it here would
    # leak a post-eligibility ValueError.
    qk_norm: bool = True

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
        if not self.qk_norm and self.rope_dim == 0:
            raise ValueError("qk_norm=False with rope_dim=0 leaves stage (2)+(3) an identity copy; drop the stage instead")
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
    # MXFP8 pipelines only (-1 otherwise): the SDPA's F8_128x4 E8M0 scale-factor
    # blobs for Q / K / V -- ``b * h * ceil(s/128) * (128 * d/32)`` bytes each
    # (the adapter's ``_reshape_sf`` count; 1 KiB per (b, h, 128-row tile) at
    # d=256).  Written by the quantize_mxfp8 stages (unfused) or by the MXFP8
    # projection fork's epilogue (fully fused).
    sf_q: int = -1
    sf_k: int = -1
    sf_v: int = -1
    # fp4 O (``MxQuantSpec.o_fp4``) only (-1 otherwise): the packed e2m1 gated O the fp4 out_proj reads
    # -- ``[T, H_q*D/2]`` bytes, viewed ``float4_e2m1fn_x2`` -- and its PADDED F8_128x4 scale blob over
    # ``(rows=T, K=H_q*D)``, ``proj_gemm.sf_blob_bytes(T, H_q*D, block)`` bytes: the GEMM's contract, NOT the
    # SDPA's ``_sf_slot_bytes`` (a different byte ORDER and count).  ``o8`` is -1 under o_fp4 (never written,
    # so never reserved); on the fused arm ``o`` (bf16) takes its place, since the gated SDPA then writes bf16 O.
    o4: int = -1
    sf_o: int = -1


_SF_TILE_ROWS = 128  # rows of one F8_128x4 scale-factor atom == the SDPA's Q / KV tile height (keep in step with kernels/quantize_mxfp8.py SF_TILE_ROWS)
_SF_BLOCK = 32  # elements per E8M0 scale (MXFP8 block size)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _sf_slot_bytes(b: int, h: int, s: int, d: int) -> int:
    """Bytes of ONE of the SDPA's F8_128x4 scale-factor blobs (Q, K or V) over ``[B, H, S, D]``.

    ``B * H * ceil(S/128) * (128 * D/32)`` -- exactly the adapter's ``_reshape_sf``
    count (``b*h*n_tiles*SF_SMEM_SIZE``, 1 KiB per tile at d=256) and the
    ``kernels/quantize_mxfp8.py`` ``sf_bytes`` contract the quantize stages
    write; Q/K (rowwise) and V (columnwise, D-plane-major) have the SAME byte
    count, only the byte ORDER differs.  Derived, never a literal.
    """
    return b * h * _ceil_div(s, _SF_TILE_ROWS) * (_SF_TILE_ROWS * d // _SF_BLOCK)


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
    mxfp8: bool = False,
    o_fp4: Optional[Fp4Format] = None,
) -> _Intermediates:
    """Reserve every intermediate, in stage order, and report the total.

    ``mxfp8`` (appended) adds the three SDPA scale-factor blobs ``sf_q`` /
    ``sf_k`` / ``sf_v`` (:func:`_sf_slot_bytes`) at the END of either layout, so
    every FP8 offset is byte-identical to before MXFP8 existed.

    ``o_fp4`` (appended; an :class:`Fp4Format`) appends ``o4`` (packed e2m1 gated O,
    ``t*h_q*d_head // 2`` bytes) and ``sf_o`` (its padded F8_128x4 blob,
    ``proj_gemm.sf_blob_bytes(t, h_q*d_head, block)`` bytes -- the GEMM contract)
    at the END of either layout and does NOT reserve ``o8`` (a slot never written
    is not reserved); on the fused arm the bf16 ``o`` the gated SDPA then writes
    takes ``o8``'s place.  Every layout with ``o_fp4=None`` is byte-identical to
    before (pinned by ``test_workspace_layout_is_byte_identical_without_fp4``).

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
            # fp4 O (row 10): the gated SDPA writes bf16 `o` here instead of e4m3 `o8`; quantize_fp4 reads it.
            ("o", t * geom.h_q * geom.d_head * e) if o_fp4 is not None else ("o8", t * geom.h_q * geom.d_head),
        ]
        if mxfp8:
            slots += _sf_slots(geom, b, s)
        if o_fp4 is not None:
            slots += _o_fp4_slots(geom, t, o_fp4)
        for name, nbytes in slots:
            offsets[name] = off
            off += _align_up(nbytes)
        return _Intermediates(
            proj=-1,
            q=-1,
            gate=-1,
            k=-1,
            v=-1,
            o=offsets.get("o", -1),
            o_gated=-1,
            engine_scratch=off,
            total_bytes=off,
            base_align=_WS_ALIGN,
            q8=offsets["q8"],
            k8=offsets["k8"],
            v8=offsets["v8"],
            o8=offsets.get("o8", -1),
            gate16=offsets["gate16"],
            sf_q=offsets.get("sf_q", -1),
            sf_k=offsets.get("sf_k", -1),
            sf_v=offsets.get("sf_v", -1),
            o4=offsets.get("o4", -1),
            sf_o=offsets.get("sf_o", -1),
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
    if fp8 and o_fp4 is None:
        slots += [("o8", t * geom.h_q * geom.d_head)]
    if mxfp8:
        # MXFP8: the SDPA's F8_128x4 SF blobs, written by the three quantize_mxfp8 stages.
        slots += _sf_slots(geom, b, s)
    if o_fp4 is not None:
        # fp4 O (rows 8 / 9): the quantize_fp4 stage's packed codes + the out_proj GEMM's scale blob, at the END.
        slots += _o_fp4_slots(geom, t, o_fp4)
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
        sf_q=offsets.get("sf_q", -1),
        sf_k=offsets.get("sf_k", -1),
        sf_v=offsets.get("sf_v", -1),
        o4=offsets.get("o4", -1),
        sf_o=offsets.get("sf_o", -1),
    )


def _sf_slots(geom: GatedAttentionBlockGeometry, b: int, s: int) -> list:
    """The three MXFP8 SDPA scale-factor slots, in (Q, K, V) order."""
    return [
        ("sf_q", _sf_slot_bytes(b, geom.h_q, s, geom.d_head)),
        ("sf_k", _sf_slot_bytes(b, geom.h_kv, s, geom.d_head)),
        ("sf_v", _sf_slot_bytes(b, geom.h_kv, s, geom.d_head)),
    ]


def _o_fp4_code_bytes(geom: GatedAttentionBlockGeometry, t: int) -> int:
    """Bytes of the packed e2m1 gated O: ``T * H_q * D / 2`` (two codes per byte along K = H_q*D)."""
    return t * geom.h_q * geom.d_head // 2


def _o_fp4_slots(geom: GatedAttentionBlockGeometry, t: int, o_fp4: Fp4Format) -> list:
    """The two fp4-O slots, in the order quantize_fp4 writes them: ``o4`` (codes), ``sf_o`` (the out_proj GEMM's
    PADDED F8_128x4 blob over ``(rows=T, K=H_q*D)`` at the format's block -- ``proj_gemm.sf_blob_bytes``)."""
    from .kernels.proj_gemm import sf_blob_bytes

    return [
        ("o4", _o_fp4_code_bytes(geom, t)),
        ("sf_o", sf_blob_bytes(t, geom.h_q * geom.d_head, o_fp4.block_size)),
    ]


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
                                                 (None iff geometry.qk_norm is False)
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
    from :class:`~cudnn.gated_attention_block.api_bwd.RecomputePolicy` --
    EXCEPT ``rstd_q`` / ``rstd_k``, which are ``None`` iff
    ``geometry.qk_norm`` is False: there is no RMSNorm, stage (B6) does not
    exist, and nothing could recompute them. The forward REQUIRES them to be
    ``None`` in that case (and tensors otherwise), both directions typed.
    """

    h: torch.Tensor
    gate: torch.Tensor
    o: torch.Tensor
    lse: torch.Tensor
    rstd_q: Optional[torch.Tensor]  # None iff geometry.qk_norm is False (positional slot kept: append-only)
    rstd_k: Optional[torch.Tensor]  # idem
    q_pre: Optional[torch.Tensor] = None
    k_pre: Optional[torch.Tensor] = None


def _check_norm_weights_agree(qk_norm: bool, w_q_norm, w_k_norm, *, prefix: str = "") -> None:
    """``geometry.qk_norm`` and the two norm-weight slots must agree, BOTH ways.

    Typed ``ValueError`` naming the knob. Load-bearing rather than cosmetic:
    ``APIBase._make_tensor_desc(None)`` and ``_check_tensor_shape(None)`` both
    return ``None`` SILENTLY, so a norm-on block declared with ``None`` weights
    would sail through declaration and die in ``check_support``'s dtype loop with
    an untyped ``AttributeError`` -- or, worse, hand the kernel a null weight
    pointer. Called at declaration (``sample_*``) and again at every ``execute``.
    """
    have_q, have_k = w_q_norm is not None, w_k_norm is not None
    q_nm, k_nm = f"{prefix}w_q_norm", f"{prefix}w_k_norm"
    if have_q != have_k:
        raise ValueError(
            f"{q_nm} and {k_nm} must be given together or both be None (geometry.qk_norm={qk_norm}); "
            f"got {q_nm}={'tensor' if have_q else 'None'}, {k_nm}={'tensor' if have_k else 'None'}"
        )
    if qk_norm and not have_q:
        raise ValueError(
            f"geometry.qk_norm=True (QK-RMSNorm on) requires both [D] norm weights, but {q_nm} / {k_nm} are None. "
            "Pass them, or declare GatedAttentionBlockGeometry(qk_norm=False) for RoPE-only Q/K."
        )
    if not qk_norm and have_q:
        raise ValueError(
            f"geometry.qk_norm=False (RoPE-only Q/K, no RMSNorm) takes no norm weights, but {q_nm} / {k_nm} were given. "
            "Pass None for both, or declare GatedAttentionBlockGeometry(qk_norm=True)."
        )


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
# 4c. MXFP8 (E4M3 codes + per-32-block E8M0 scales, F8_128x4) — the MxQuantSpec contract
# ---------------------------------------------------------------------------


MXFP8_BLOCK_SIZE = 32
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_SF_DTYPES = tuple(t for t in (torch.uint8, _E8M0) if t is not None)
_FP4_X2 = getattr(torch, "float4_e2m1fn_x2", None)  # packed e2m1: ONE byte = two codes along the contiguous axis
# The W_qkvg code dtypes an MxQuantSpec may name: e4m3 (MXFP8 x MXFP8) or e2m1 (the catalog's MIXED
# block-scale row, MXFP8 h x MXFP4 W -- `kernel_registry._BLOCK_SCALE_CASES` fp8_e4m3 x fp4_e2m1, E8M0 / 32).
_W_QKVG_DTYPES = tuple(t for t in (torch.float8_e4m3fn, _FP4_X2) if t is not None)


class Fp4Format(Enum):
    """The two fp4 OUTPUT formats the block quantizes its gated O to (``MxQuantSpec.o_fp4``) -- and, by
    construction, the format of the fp4 ``W_o`` the block-scale out projection multiplies it with.

    ONE member = (e2m1 codes, scale dtype, scale block): an illegal pairing cannot be spelled.  Exactly the
    two rows the FROST block-scale catalog serves for two e2m1 sides (``kernel_registry._BLOCK_SCALE_CASES``):

    ========= ============================ ======= ======================================================
    member    scale dtype                  block   catalog row
    ========= ============================ ======= ======================================================
    ``NVFP4`` ``torch.float8_e4m3fn``      16      ``fp4_e2m1 x fp4_e2m1`` with ``fp8_e4m3`` scales per 16
    ``MXFP4`` ``torch.float8_e8m0fnu``     32      ``fp4_e2m1 x fp4_e2m1`` with ``fp8_e8m0`` scales per 32
    ========= ============================ ======= ======================================================

    Members are NAMED after ``kernels/quantize_fp4.py``'s format keys (``"nvfp4"`` / ``"mxfp4"``): the
    quantize stage takes the member itself and ``fp4_format(member)`` cross-checks ``block_size`` against
    the kernel's table, so the enum and the kernel cannot drift apart silently.  No global (per-tensor)
    scale in either format: ``MxQuantSpec.scale_o`` / ``descale_w_o`` are pinned to 1.0 under ``o_fp4``.
    """

    NVFP4 = ("nvfp4", 16)  # e2m1 x e4m3 scales per 16 along K
    MXFP4 = ("mxfp4", 32)  # e2m1 x E8M0 scales per 32 along K

    @property
    def fmt_name(self) -> str:
        """The ``kernels/quantize_fp4.py`` format key (``"nvfp4"`` / ``"mxfp4"``)."""
        return self.value[0]

    @property
    def block_size(self) -> int:
        """Elements per scale along K (16 for NVFP4, 32 for MXFP4)."""
        return self.value[1]

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        """The scale blob's torch storage dtype: ``float8_e4m3fn`` (NVFP4) / ``float8_e8m0fnu`` (MXFP4); a blob may
        also arrive as plain ``uint8`` bytes."""
        return torch.float8_e4m3fn if self is Fp4Format.NVFP4 else _E8M0

    @property
    def sf_cudnn_dtype(self):
        """The ``cudnn.data_type`` the out-projection GEMM DECLARES its scale tensors in -- which selects the MMA's
        scale format (``FP8_E4M3`` for NVFP4, ``FP8_E8M0`` for MXFP4)."""
        import cudnn

        return cudnn.data_type.FP8_E4M3 if self is Fp4Format.NVFP4 else cudnn.data_type.FP8_E8M0


@dataclass(frozen=True)
class MxQuantSpec:
    """MXFP8 pipeline: ``h`` / ``W_qkvg`` / Q / K / V carry per-32-block E8M0 scales
    (cuDNN's F8_128x4 order), so only ``out_proj``'s per-tensor pair survives.

    A SIBLING of :class:`QuantSpec`, not a flag on it (PR-B D3): five of
    ``QuantSpec``'s seven floats are meaningless under block scales, and a
    separate dataclass makes the illegal combinations unrepresentable while
    leaving every FP8 caller byte-identical.  Passing an ``MxQuantSpec`` (with
    e4m3 ``h`` / weights AND the two scale-factor blobs ``sample_h_sf`` /
    ``sample_w_qkvg_sf``) selects the MXFP8 pipeline.

    ============== ============================================================
    ``descale_w_o``  dequant multiplier of the per-tensor FP8 ``W_o`` (D1: the
                     out projection stays per-tensor FP8)
    ``scale_o``      quant scale applied to the bf16 gated O before ``out_proj``
                     (UNFUSED path).  On the FULLY FUSED path the production
                     MXFP8 SDPA writes e4m3 O UNSCALED (its ABI has no
                     ``scale_o``), so it MUST be 1.0 there -- D8, typed decline.
    ``dtype``        the code dtype; only E4M3 is served (E5M2 declines typed)
    ``block_size``   32 -- the MX block; anything else is a ``ValueError``
    ``w_qkvg_dtype`` the ``W_qkvg`` code dtype (appended, default e4m3 = today's
                     MXFP8 x MXFP8 GEMM).  ``torch.float4_e2m1fn_x2`` selects an
                     MXFP4 weight: e2m1 codes stored ``[n_qkvg, d_model // 2]``
                     (two per byte, LOW nibble = even k) against the e4m3 ``h``
                     -- the FROST catalog's MIXED block-scale row (``fp8_e4m3 x
                     fp4_e2m1``, E8M0 scales per 32); ``w_qkvg_sf`` is UNCHANGED
                     (the same E8M0 / 32 F8_128x4 blob over ``n_qkvg x d_model``).
                     Unfused pipeline only: the fused MXFP8 projection fork is
                     rendered for an e4m3 B (typed decline at ``check_support``).
    ``o_fp4``        (appended, default ``None`` = today's per-tensor e4m3 O).  An
                     :class:`Fp4Format` member selects the fp4 OUTPUT mode: the gated
                     O is block-quantized to e2m1 codes + that format's scale blob by
                     a ``quantize_fp4`` launch, and the out projection becomes the
                     fp4 x fp4 block-scale GEMM against an e2m1 ``W_o`` of the SAME
                     format -- stored ``[d_model, h_q*d_head // 2]`` as
                     ``torch.float4_e2m1fn_x2`` with its F8_128x4 blob
                     (``sample_w_o_sf`` / ``w_o_sf``,
                     ``proj_gemm.sf_blob_bytes(d_model, h_q*d_head, block)`` bytes).
                     Neither side carries a per-tensor scale: ``scale_o`` and
                     ``descale_w_o`` MUST be 1.0 (typed ``ValueError``).  Served on
                     the unfused pipeline (config row 8 / 9: ``quantize_o`` becomes
                     ``quantize_fp4_o``, still 9 launches) and on the fully fused one
                     (row 10: the gated MXFP8 SDPA writes bf16 O, then the fp4
                     quantize, then the fp4 out projection -- 4 launches).
    ============== ============================================================

    ``h`` / ``W_qkvg`` arrive PRE-quantized by the caller (D2: static, offline,
    like the weights' own scales -- the block runs no amax pass): e4m3 codes plus
    ONE F8_128x4 E8M0 blob each, PADDED to whole 128-row x 4-block atoms --
    ``kernels.proj_gemm.sf_blob_bytes(rows, d_model)`` bytes over ``rows = B*S``
    (``h``) / ``n_qkvg`` (``W_qkvg``), pad rows / blocks ``0x00``.
    ``descale_w_o * (1/scale_o)`` is ``alpha_o``, the out projection's epilogue.
    """

    descale_w_o: float
    scale_o: float = 1.0
    dtype: torch.dtype = torch.float8_e4m3fn
    block_size: int = MXFP8_BLOCK_SIZE
    w_qkvg_dtype: torch.dtype = torch.float8_e4m3fn  # appended: torch.float4_e2m1fn_x2 -> MXFP4 W_qkvg (the mixed row)
    o_fp4: Optional[Fp4Format] = None  # appended: Fp4Format.NVFP4 | MXFP4 -> fp4 gated O + fp4 W_o of the same format, block-scale out_proj

    @property
    def w_qkvg_fp4(self) -> bool:
        """``W_qkvg`` is packed e2m1 (``torch.float4_e2m1fn_x2``): the mixed MXFP8 x MXFP4 stage (1)."""
        return _FP4_X2 is not None and self.w_qkvg_dtype == _FP4_X2

    def validate(self, *, fused: bool = False) -> None:
        """Typed declines; ``fused=True`` adds the D8 unit-``scale_o`` rule of the fully fused path."""
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"MxQuantSpec: only torch.float8_e4m3fn codes are served (E5M2 is a knob away), got {self.dtype}")
        if int(self.block_size) != MXFP8_BLOCK_SIZE:
            raise ValueError(f"MxQuantSpec.block_size must be {MXFP8_BLOCK_SIZE} (one E8M0 scale per 32-element MX block), got {self.block_size}")
        if self.w_qkvg_dtype not in _W_QKVG_DTYPES:
            raise NotImplementedError(
                f"MxQuantSpec.w_qkvg_dtype: W_qkvg codes are torch.float8_e4m3fn (MXFP8 x MXFP8) or torch.float4_e2m1fn_x2 (the FROST "
                f"block-scale catalog's mixed row, kernel_registry._BLOCK_SCALE_CASES fp8_e4m3 x fp4_e2m1 with E8M0 scales per 32); got "
                f"{self.w_qkvg_dtype}. A uint8 blob of packed e2m1 codes is spelled .view(torch.float4_e2m1fn_x2), never uint8."
            )
        if self.o_fp4 is not None and not isinstance(self.o_fp4, Fp4Format):
            raise TypeError(f"MxQuantSpec.o_fp4 must be an Fp4Format member (Fp4Format.NVFP4 | Fp4Format.MXFP4) or None, got {self.o_fp4!r}")
        if self.o_fp4 is not None and _FP4_X2 is None:
            # Mirrors quantize_fp4.check_torch_fp4_dtypes: the o4 buffer and W_o are VIEWED as the packed e2m1 dtype, so a
            # torch without it cannot serve rows 8-10 -- declined here, before _expected_weight_dtypes reports a None dtype.
            raise NotImplementedError(f"MxQuantSpec.o_fp4 needs torch.float4_e2m1fn_x2 for the packed e2m1 O and W_o (torch {torch.__version__} has none)")
        for name in ("descale_w_o", "scale_o"):
            v = float(getattr(self, name))
            if not (v > 0.0) or v != v or v in (float("inf"),):
                raise ValueError(f"MxQuantSpec.{name} must be a finite positive float, got {v}")
        if self.o_fp4 is not None:
            # No global scale on either fp4 side (plan section 1): the O quantizer writes local block scales
            # only, and W_o dequantizes through its own blob in the MMA.  A non-unit value here would be
            # silently dropped by the block-scale GEMM (no alpha epilogue) -- refused instead.
            if float(self.scale_o) != 1.0:
                raise ValueError(f"MxQuantSpec.scale_o: a block-scaled O has no per-tensor scale; pass scale_o=1.0 under o_fp4 (got {self.scale_o})")
            if float(self.descale_w_o) != 1.0:
                raise ValueError(
                    f"MxQuantSpec.descale_w_o: under o_fp4 W_o dequantizes through w_o_sf (its F8_128x4 scale blob); pass descale_w_o=1.0 "
                    f"(got {self.descale_w_o})"
                )
        elif fused and float(self.scale_o) != 1.0:
            # D8 (per-tensor e4m3 O on the fully fused path).  Skipped under o_fp4: scale_o == 1.0 is already pinned above.
            raise NotImplementedError(
                f"MxQuantSpec.scale_o must be 1.0 on the FULLY FUSED MXFP8 path (got {self.scale_o}): the production MXFP8 SDPA writes "
                "e4m3 O UNSCALED (its ABI has no per-tensor scale_o -- PR-B D8). Use scale_o=1.0, or the unfused pipeline."
            )

    @property
    def alpha_o(self) -> float:
        return (1.0 / self.scale_o) * self.descale_w_o


def _check_sf_blob(sf: torch.Tensor, name: str, rows: int, k: int, *, block: int = MXFP8_BLOCK_SIZE, sf_dtypes: Tuple[torch.dtype, ...] = _SF_DTYPES) -> None:
    """A caller-supplied F8_128x4 scale-factor blob: dtype, PADDED byte count
    (``proj_gemm.sf_blob_bytes(rows, k, block)``), contiguity, 16-B alignment -- typed, before any kernel.

    ``block`` / ``sf_dtypes`` (appended, defaults = the MXFP8 E8M0 / 32 contract) generalise it to the fp4
    blobs: one scale byte per ``block`` elements along K, stored as any of ``sf_dtypes`` (``uint8`` plus
    the format's own fp8 storage dtype -- ``float8_e8m0fnu`` for MX scales, ``float8_e4m3fn`` for NVFP4)."""
    from .kernels.proj_gemm import sf_blob_bytes, sf_padded_dims

    if sf.dtype not in sf_dtypes:
        want = " or ".join(str(t) for t in sf_dtypes)
        raise ValueError(f"{name} must be {want} (scale bytes in F8_128x4 order), got {sf.dtype}")
    need = sf_blob_bytes(rows, k, block)
    if sf.numel() != need:
        rows_pad, k4 = sf_padded_dims(rows, k, block)
        raise ValueError(
            f"{name} has {sf.numel()} bytes; the F8_128x4 blob over {rows} rows x K={k} at block {block} is {need} = {rows_pad} (rows padded to 128) "
            f"x {k4} (K/{block} blocks padded to 4) -- whole 512-B atoms, pad rows / blocks 0x00 (kernels.proj_gemm.sf_blob_bytes)"
        )
    if not sf.is_contiguous():
        raise ValueError(f"{name} must be contiguous (an opaque F8_128x4 byte blob bound by storage order)")
    if sf.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned (TMA-fed)")


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

    def __init__(
        self,
        *,
        m: int,
        k: int,
        n: int,
        dtype: torch.dtype,
        name: str,
        out_dtype: Optional[torch.dtype] = None,
        alpha: bool = False,
        block_scale: bool = False,
        w_dtype: Optional[torch.dtype] = None,
        block_size: int = MXFP8_BLOCK_SIZE,
        sf_dtype=None,
    ) -> None:
        self.name = name
        self.m = int(m)
        self.k = int(k)
        self.n = int(n)
        self.dtype = dtype
        # FP8: inputs e4m3, fp32 accumulate, `out_dtype` (bf16) out, and `alpha`
        # = descale_a * descale_w folded into the GEMM's scalar-multiply epilogue.
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.alpha = bool(alpha)
        # MXFP8: e4m3 codes + per-32-block E8M0 scale factors for A (over M) and W
        # (over N), dequantized IN the MMA (`block_scale_dequantize`) -- no alpha.
        # The caller hands the two F8_128x4 blobs to execute(sf_a=, sf_w=).
        self.block_scale = bool(block_scale)
        # fp4 (append-only, default = today's behaviour): `w_dtype` names W's dtype apart from
        # A's (an e2m1 weight against an e4m3 activation is the catalog's MIXED block-scale
        # row; two e2m1 sides are NVFP4 / MXFP4), `block_size` the scale block along K (32,
        # or 16 for NVFP4) and `sf_dtype` the scale dtype (a `cudnn.data_type`; None = the
        # pair's own default).  The served pairs are `kernels.proj_gemm.block_scale_pairing`.
        self.w_dtype = w_dtype if w_dtype is not None else dtype
        self.block_size = int(block_size)
        self.sf_dtype = sf_dtype
        self._plan = None

    def check_support(self) -> None:
        from .kernels.proj_gemm import block_scale_pairing

        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        a_dtypes = (torch.bfloat16, torch.float16, torch.float8_e4m3fn) + ((fp4,) if (self.block_scale and fp4 is not None) else ())
        if self.dtype not in a_dtypes:
            raise NotImplementedError(f"{self.name}: bf16/f16/fp8-e4m3 (and fp4-e2m1 under block_scale) only, got {self.dtype}")
        if self.dtype == torch.float8_e4m3fn and self.k % 16:
            raise NotImplementedError(f"{self.name}: FP8 needs K % 16 == 0 (TMA 16-byte rule at 1 B/elem), got K={self.k}")
        if self.out_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: output dtype must be bf16/f16, got {self.out_dtype}")
        if self.block_scale:
            # The pairing table is the GEMM driver's (one source); its ValueError is the block's typed decline.
            try:
                block_scale_pairing(dtype=self.dtype, w_dtype=self.w_dtype, sf_dtype=self.sf_dtype, block_size=self.block_size, label=self.name)
            except ValueError as exc:
                raise NotImplementedError(str(exc)) from None
            if self.alpha:
                raise ValueError(f"{self.name}: block_scale=True carries its descale in the per-block scale factors; alpha must be False")
            if self.k % MXFP8_BLOCK_SIZE:
                # 32 also for NVFP4 (two 16-blocks): it is the fp4 TMA rule (16 bytes = 32 codes) as much as the scale block.
                raise NotImplementedError(
                    f"{self.name}: block_scale=True needs K % {MXFP8_BLOCK_SIZE} == 0 (whole scale blocks; the fp4 TMA rule), got K={self.k}"
                )
        elif self.w_dtype != self.dtype:
            raise NotImplementedError(
                f"{self.name}: a per-weight dtype (A {self.dtype}, W {self.w_dtype}) exists only as a block-scale row; the dense GEMM takes one dtype"
            )

    def compile(self) -> None:
        from .kernels.proj_gemm import build_proj_gemm

        self._plan = build_proj_gemm(
            m=self.m,
            k=self.k,
            n=self.n,
            dtype=self.dtype,
            label=self.name,
            out_dtype=self.out_dtype,
            alpha=self.alpha,
            block_scale=self.block_scale,
            sf_dtype=self.sf_dtype,
            w_dtype=self.w_dtype,
            block_size=self.block_size,
        )

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
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        handle=None,
        alpha: Optional[torch.Tensor] = None,
        sf_a: Optional[torch.Tensor] = None,
        sf_w: Optional[torch.Tensor] = None,
        *,
        stream=None,
    ) -> None:
        """``sf_a`` / ``sf_w`` (block-scale plans only, both required): the PADDED
        F8_128x4 E8M0 blobs of ``a`` (over its M rows) and ``w`` (over its N rows).
        ``stream`` is the block's launch stream (a raw ``CUstream`` int); the
        runner carries it onto both GEMM routes -- see ``run_proj_gemm`` (Rule 5)."""
        from .kernels.proj_gemm import run_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        if self.block_scale and (sf_a is None or sf_w is None):
            raise ValueError(f"{self.name}: block_scale=True needs both scale-factor blobs (sf_a=, sf_w=); no silent unit scale (Rule 1)")
        if not self.block_scale and (sf_a is not None or sf_w is not None):
            raise ValueError(f"{self.name}: this projection has no block-scale dequant; refusing to drop sf_a / sf_w silently")
        run_proj_gemm(self._plan, a, w, out, workspace, handle, alpha=alpha, sf_a=sf_a, sf_w=sf_w, stream=stream)


def _qkv_gate_projection(
    geom: GatedAttentionBlockGeometry,
    *,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
    block_scale: bool = False,
    w_dtype: Optional[torch.dtype] = None,
) -> _Projection:
    """Stage (1). At the 397B full-attention layer: ``M x 17408 x 4096``.  ``w_dtype`` (default:
    ``dtype``) is the weight's dtype -- ``torch.float4_e2m1fn_x2`` for an MXFP4 ``W_qkvg`` against
    e4m3 ``h`` (the block-scale MIXED row; the E8M0 / 32 scale blob is unchanged)."""
    return _Projection(
        m=batch * seq_len,
        k=geom.d_model,
        n=geom.n_qkvg,
        dtype=dtype,
        name="qkv_gate_proj",
        out_dtype=out_dtype,
        alpha=alpha,
        block_scale=block_scale,
        w_dtype=w_dtype,
    )


def _out_projection(
    geom: GatedAttentionBlockGeometry,
    *,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
    block_scale: bool = False,
    w_dtype: Optional[torch.dtype] = None,
    block_size: int = MXFP8_BLOCK_SIZE,
    sf_dtype=None,
) -> _Projection:
    """Stage (6). At the 397B full-attention layer: ``M x 4096 x 8192``.  ``block_scale`` + the fp4
    trio (``dtype=w_dtype=torch.float4_e2m1fn_x2``, ``block_size`` 16 | 32, ``sf_dtype`` E4M3 | E8M0)
    is the fp4 gated-O x fp4 ``W_o`` projection; the defaults are the per-tensor / bf16 path as before."""
    return _Projection(
        m=batch * seq_len,
        k=geom.h_q * geom.d_head,
        n=geom.d_model,
        dtype=dtype,
        name="out_proj",
        out_dtype=out_dtype,
        alpha=alpha,
        block_scale=block_scale,
        w_dtype=w_dtype,
        block_size=block_size,
        sf_dtype=sf_dtype,
    )


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


class _QuantizeMxfp8(_Stage):
    """(3q, MXFP8) block quantize: bf16 ``[T, H, D]`` -> compact e4m3 ``[T, H, D]`` + an F8_128x4 E8M0 SF blob.

    Kernel: ``kernels/quantize_mxfp8.py``.  ONE stage class, TWO arms selected by
    ``axis`` -- ``"row"`` (Q / K: 32-element blocks along D, the BMM1 contraction;
    SF tile ``(b, h, s_tile)`` = 1024 contiguous bytes) and ``"col"`` (V: blocks
    along S, the BMM2 contraction; SF D-PLANE-MAJOR, plane stride
    ``B*H*n_tiles*512``) -- the exact byte layouts the production MXFP8 SDPA's
    ``_build_sf_desc`` reads (``mma-tma-matrix.md`` § 7).  The kernel writes EVERY
    SF byte of every 128-row tile (pad rows -> ``0x00``), so a KV tail is served
    exactly like the torch oracle pads it (D6).  The source may be a strided slab
    column slice; the destination is compact, so for Q/K/V this stage IS the
    compaction.  The fully fused MXFP8 pipeline folds all three into the
    projection fork's epilogue and builds none of them.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, axis: str, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self.axis = str(axis)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.quantize_mxfp8 import AXES, validate_shape

        if self.axis not in AXES:
            raise ValueError(f"{self.name}: axis must be one of {AXES} ('row' for Q/K, 'col' for V), got {self.axis!r}")
        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        validate_shape(self.geom.d_head, _QUANTIZE_MXFP8_THREADS, self.axis)

    def compile(self) -> None:
        from .kernels.quantize_mxfp8 import compile_quantize_mxfp8

        self._recipe = compile_quantize_mxfp8(dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, axis=self.axis, threads_per_cta=_QUANTIZE_MXFP8_THREADS)

    def sf_bytes(self) -> int:
        """Bytes of the SF blob this stage writes (== the SDPA adapter's ``_reshape_sf`` count)."""
        return _sf_slot_bytes(self.batch, self.heads, self.seq_len, self.geom.d_head)

    def moved_bytes(self) -> int:
        """HBM traffic of one launch: 2 B in, 1 B code + 1/32 B SF out per element."""
        from .kernels.quantize_mxfp8 import moved_bytes

        return moved_bytes(self.batch * self.seq_len, self.heads, self.geom.d_head, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(
        self, src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, *, batch: Optional[int] = None, seq_len: Optional[int] = None, current_stream=None
    ) -> None:
        """``src`` ``[T, H, D]`` (strided ok), ``dst`` compact e4m3 ``[T, H, D]``, ``sf`` uint8 flat (``sf_bytes()`` bytes)."""
        from .kernels.quantize_mxfp8 import run_quantize_mxfp8

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        b = self.batch if batch is None else int(batch)
        s = self.seq_len if seq_len is None else int(seq_len)
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_quantize_mxfp8(self._recipe, src, dst, sf, batch=b, seq_len=s, stream=stream)


class _QuantizeFp4(_Stage):
    """(5q') fp4 block quantize of the gated O: compact bf16/f16 ``[T, H_q, D]`` -> e2m1 codes ``[T, H_q*D/2]``
    (two per byte, bound as ``float4_e2m1fn_x2``) + the out-projection GEMM's PADDED F8_128x4 scale blob
    over ``(rows=T, K=H_q*D)``.

    Kernel: ``kernels/quantize_fp4.py``.  ONE stage class, TWO formats selected by ``fmt`` (the
    ``Fp4Format`` member, or its name): ``NVFP4`` -- e4m3 scale per 16 (``e4m3_rn(max(amax/6, 2^-9))``,
    codes by ``div.rn.f32``) -- and ``MXFP4`` -- E8M0 scale per 32 (``cvt.rp.ue8m0(amax/6)``, codes by the
    exact power-of-two reciprocal).  The blob is sized by ``proj_gemm.sf_blob_bytes`` (the GEMM contract,
    NOT the SDPA's ``_sf_slot_bytes``) and the kernel writes EVERY byte of it (pad rows ``0x00``), so the
    block-scale out projection reads it with no re-layout.  The source is the compact gated ``o``; the
    stage is what turns it into the fp4 A operand of ``o4 @ W_o^T``.  Built under ``MxQuantSpec.o_fp4`` (config
    rows 8-10) in ``quantize_o``'s place, on the unfused and the fully fused MXFP8 pipeline.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, fmt, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self.fmt = fmt
        self._recipe = None

    def _format(self) -> tuple:
        """``(name, block, sf_e4m3)`` of ``fmt`` -- ``ValueError`` for anything but the two served formats."""
        from .kernels.quantize_fp4 import fp4_format

        return fp4_format(self.fmt)

    def check_support(self) -> None:
        from .kernels.quantize_fp4 import validate_shape

        _, block, _ = self._format()
        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        if self.geom.d_head % (4 * block):
            raise NotImplementedError(
                f"{self.name}: d_head={self.geom.d_head} is not a multiple of 4*block = {4 * block}, so one head's scales are not whole F8_128x4 atoms"
            )
        validate_shape(self.geom.d_head, _QUANTIZE_FP4_THREADS, block)

    def compile(self) -> None:
        from .kernels.quantize_fp4 import compile_quantize_fp4

        self._recipe = compile_quantize_fp4(dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, fmt=self.fmt, threads_per_cta=_QUANTIZE_FP4_THREADS)

    def rows(self) -> int:
        return self.batch * self.seq_len

    def code_bytes(self) -> int:
        """Bytes of the packed e2m1 codes: ``T * H_q * D / 2``."""
        return self.rows() * self.heads * self.geom.d_head // 2

    def sf_bytes(self) -> int:
        """Bytes of the padded F8_128x4 blob this stage writes == what the block-scale out projection binds."""
        from .kernels.proj_gemm import sf_blob_bytes

        _, block, _ = self._format()
        return sf_blob_bytes(self.rows(), self.heads * self.geom.d_head, block)

    def moved_bytes(self) -> int:
        """HBM traffic of one launch: 2 B in, 1/2 B code + 1/block B SF out per element."""
        from .kernels.quantize_fp4 import moved_bytes

        _, block, _ = self._format()
        return moved_bytes(self.rows(), self.heads, self.geom.d_head, block, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(self, src: torch.Tensor, dst4: torch.Tensor, sf: torch.Tensor, *, current_stream=None) -> None:
        """``src`` compact ``[T, H_q, D]``; ``dst4`` uint8 / ``float4_e2m1fn_x2`` of ``code_bytes()``; ``sf`` uint8 of ``sf_bytes()``."""
        from .kernels.quantize_fp4 import run_quantize_fp4

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_quantize_fp4(self._recipe, src, dst4, sf, stream=stream)


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

    **MXFP8 arm (``quant`` is an :class:`MxQuantSpec`, e4m3 codes + F8_128x4 SF
    blobs for ``h`` / ``W_qkvg``):** the THIRD rendering, the block-scale twin
    ``kernels/proj_gemm_norm_rope_mxfp8.py`` (``NormRopeFusionParams.quant_mxfp8``;
    PR-B section 3.1), whose epilogue norms / rotates the DEQUANTIZED fp32
    accumulator (the E8M0 dequant happens in the MMA -- no alpha, no qscal) and
    BLOCK-quantizes: Q / K rowwise along D, V columnwise along S, e4m3 codes into
    compact ``q8`` / ``k8`` / ``v8`` and E8M0 scale factors into the SDPA's own
    F8_128x4 blobs ``sf_q`` / ``sf_k`` / ``sf_v`` (Q/K per-tile contiguous, V
    D-plane-major); GATE as bf16 ``gate16``.  Launched through
    ``run_fused_proj_gemm_mxfp8`` (frozen ABI, plan 3.1).  Feature-detected on
    the runner name + the fork file: until slice S6 lands the twin this arm is a
    typed ``NotImplementedError`` at ``check_support``, never a silently
    un-quantized path.  Declines ``S % 128 != 0 and B > 1`` (GEMM M-tiles
    straddle sequences; the unfused path serves it).
    """

    name = "qkv_gate_proj_norm_rope"
    _TILE_N = 256  # the rendering's per-CTA output width; one head per tile needs d_head == this
    _SUBTILE_N = 32  # epilogue subtile; rope_dim must be a whole number of subtile PAIRS
    _FORK_PATH_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "proj_gemm_norm_rope_fp8.py")
    _FORK_PATH_MXFP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "proj_gemm_norm_rope_mxfp8.py")
    _MXFP8_RUNNER = "run_fused_proj_gemm_mxfp8"
    _WEIGHT_FP4_FIELD = "weight_fp4"  # the NormRopeFusionParams field the fork's e2m1-B arm will append (feature-detected)

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_rstd: bool,
        norm_source: str = "ldg_early",
        quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,
        device=None,
    ) -> None:
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.m = int(batch * seq_len)
        self.k = int(geometry.d_model)
        self.n = int(geometry.n_qkvg)
        self.dtype = dtype
        self.want_rstd = bool(want_rstd)
        self.norm_source = str(norm_source)
        # FP8: the QuantSpec whose alpha_qkvg / scale_q / scale_k / scale_v the
        # fork's epilogue applies; `device` is where `qscal` is materialised.
        # MXFP8: an MxQuantSpec selects the block-scale twin (no scalars ride in).
        if quant is not None and not isinstance(quant, (QuantSpec, MxQuantSpec)):
            raise TypeError(f"{self.name}: quant must be a QuantSpec or an MxQuantSpec, got {type(quant).__name__}")
        self.quant = quant
        self.device = device
        self._plan = None
        self._qscal = None

    @property
    def fp8(self) -> bool:
        """The per-tensor FP8 fork (a ``QuantSpec``)."""
        return isinstance(self.quant, QuantSpec)

    @property
    def mxfp8(self) -> bool:
        """The block-scale MXFP8 fork twin (an ``MxQuantSpec``)."""
        return isinstance(self.quant, MxQuantSpec)

    def params(self):
        from .kernels.proj_gemm import NormRopeFusionParams

        g = self.geom
        kw = {}
        if self.fp8:
            # Only the FP8 arm names the field, so the bf16 compile key (and the
            # bf16 rendering's cache) is byte-identical to before the FP8 fork.
            kw["quant_fp8"] = True
        if self.mxfp8:
            # Same only-when-set idiom for the block-scale twin; a fork without
            # the field is a typed decline, never a norm-only bf16 artifact.
            if not self._fork_supports_field("quant_mxfp8"):
                raise NotImplementedError(f"{self.name}: {self._mxfp8_fork_available()}")
            kw["quant_mxfp8"] = True
            if self.quant.w_qkvg_fp4:
                # An e2m1 B is its own rendering (Uint8 B SMEM, the B4X16_P64 form, half the
                # expect-tx bytes); a fork without the field must never render an e4m3-B artifact
                # for an fp4 weight, so the same only-when-set idiom guards the key too.
                if not self._fork_supports_field(self._WEIGHT_FP4_FIELD):
                    raise NotImplementedError(self._weight_fp4_unsupported_msg())
                kw[self._WEIGHT_FP4_FIELD] = True
        if not g.qk_norm:
            # Same only-when-set idiom: norm-on keys are spelled identically to
            # today.  The field lands with the fork edits (PR-B slice S2); until
            # then a RoPE-only fused projection is a typed decline, never a
            # silently norm-ON artifact.
            if not self._fork_supports_qk_norm():
                raise NotImplementedError(self._qk_norm_off_unsupported_msg())
            kw["qk_norm"] = False
        return NormRopeFusionParams(
            d_head=g.d_head, rope_dim=g.rope_dim, h_q=g.h_q, h_kv=g.h_kv, eps=g.qk_norm_eps, want_rstd=self.want_rstd, norm_source=self.norm_source, **kw
        )

    @staticmethod
    def _fork_supports_qk_norm() -> bool:
        """True once ``NormRopeFusionParams`` carries ``qk_norm`` (the GEMM forks'
        RoPE-only epilogue, PR-B slice S2). Feature-detected so either landing
        order works: this slice can ship before or after the fork edits."""
        import dataclasses

        from .kernels import proj_gemm

        return "qk_norm" in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}

    def _qk_norm_off_unsupported_msg(self) -> str:
        return (
            f"{self.name}: geometry.qk_norm=False (RoPE-only Q/K) with fuse_norm_rope=True needs the GEMM forks' RoPE-only "
            "epilogue (NormRopeFusionParams.qk_norm), which has not landed in this checkout. Use the unfused chain "
            "(fuse_norm_rope=False) for qk_norm=False."
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

    def _weight_fp4_unsupported_msg(self) -> str:
        return (
            f"{self.name}: the fused MXFP8 projection fork is rendered for an e4m3 B; W_qkvg is torch.float4_e2m1fn_x2 "
            f"(MxQuantSpec.w_qkvg_dtype) and NormRopeFusionParams has no `{self._WEIGHT_FP4_FIELD}` arm in this checkout. "
            "Use the unfused pipeline (fuse_norm_rope=False, fuse_gate=False): its block-scale GEMM serves the mixed MXFP8 x MXFP4 row."
        )

    @staticmethod
    def _fork_supports_field(field: str) -> bool:
        import dataclasses

        from .kernels import proj_gemm

        return field in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}

    def _mxfp8_fork_available(self) -> Optional[str]:
        """None when the MXFP8 fork twin + its runner ABI (plan 3.1) are present; else the reason they are not.

        Feature-detected (the ``_fp8_fork_available`` idiom) so the block ships
        before the twin: ``NormRopeFusionParams.quant_mxfp8``, the runner
        ``run_fused_proj_gemm_mxfp8`` carrying the frozen SF-output ABI
        (``out_sf_q`` / ``out_sf_k`` / ``out_sf_v``), and the fork file."""
        import inspect

        from .kernels import proj_gemm

        if not self._fork_supports_field("quant_mxfp8"):
            return "NormRopeFusionParams has no `quant_mxfp8` field"
        fn = getattr(proj_gemm, self._MXFP8_RUNNER, None)
        if fn is None:
            return f"kernels/proj_gemm.py has no `{self._MXFP8_RUNNER}` (the MXFP8 GEMM fork twin, PR-B slice S6, has not landed)"
        params = inspect.signature(fn).parameters
        if not {"out_sf_q", "out_sf_k", "out_sf_v", "sf_a", "sf_w"} <= set(params):
            return f"`{self._MXFP8_RUNNER}` does not carry the frozen PR-B 3.1 ABI (sf_a, sf_w, out_sf_q/k/v)"
        if not os.path.exists(self._FORK_PATH_MXFP8):
            return f"{os.path.basename(self._FORK_PATH_MXFP8)} is not in kernels/"
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
        if not g.qk_norm:
            # RoPE-only epilogue: no RMSNorm, so no rstd can be wanted, and the
            # fork must know the knob (typed decline until slice S2 lands it).
            if self.want_rstd:
                raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and emits no rstd; want_rstd must be False")
            if not self._fork_supports_qk_norm():
                raise NotImplementedError(self._qk_norm_off_unsupported_msg())
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
        elif self.mxfp8:
            if self.dtype != torch.float8_e4m3fn:
                raise NotImplementedError(f"{self.name}: the MXFP8 fork twin is rendered for e4m3 codes, got {self.dtype}")
            if self.quant.w_qkvg_fp4 and not self._fork_supports_field(self._WEIGHT_FP4_FIELD):
                # Config row 11: MXFP4 W_qkvg + fuse_norm_rope.  Feature-detected like
                # `quant_mxfp8`, so this decline INVERTS the day the fork's e2m1-B arm lands.
                raise NotImplementedError(self._weight_fp4_unsupported_msg())
            if self.k % MXFP8_BLOCK_SIZE:
                raise NotImplementedError(f"{self.name}: MXFP8 needs K % {MXFP8_BLOCK_SIZE} == 0 (one E8M0 scale per block), got K={self.k}")
            if self.want_rstd:
                raise NotImplementedError(f"{self.name}: the MXFP8 fork twin is inference-only (no rstd output)")
            if self.seq_len % _SF_TILE_ROWS and self.batch > 1:
                # The fork's SF stores are decoded from the flat GEMM row; a 128-row
                # M-tile straddling two sequences would have to scatter its SF
                # atom across two (b, s_tile) units.  v1 declines (plan 3.1 / Q9).
                raise NotImplementedError(
                    f"{self.name}: the fully fused MXFP8 pipeline needs S % {_SF_TILE_ROWS} == 0 when B > 1 (a GEMM M-tile must not straddle two "
                    f"sequences' scale-factor tiles); got B={self.batch}, S={self.seq_len}. Use the unfused MXFP8 pipeline."
                )
            missing = self._mxfp8_fork_available()
            if missing is not None:
                raise NotImplementedError(f"{self.name}: the MXFP8 fused projection fork twin has not landed in this checkout ({missing})")
        elif self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the fork is rendered for bf16 only (e4m3 needs a QuantSpec / MxQuantSpec), got {self.dtype}")
        validate_norm_rope_params(self.params())
        if torch.cuda.is_available():
            cc = tuple(torch.cuda.get_device_capability())
            if cc != _SM107_CC:
                raise NotImplementedError(f"{self.name}: rendered for sm_107a (Rubin); this device is SM{cc[0]}{cc[1]}")

    def compile(self) -> None:
        from .kernels.proj_gemm import build_fused_proj_gemm

        # params() itself declines qk_norm=False on a fork without the knob, so
        # a compile() reached without check_support() cannot build a norm-ON
        # artifact for a norm-OFF geometry.
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
        """bf16 arm: ONE ``[M, N]`` slab out (Q/K columns normed + rotated).

        ``w_q_norm`` / ``w_k_norm`` are ``None`` (both) under ``geometry.qk_norm=False``
        and tensors otherwise -- checked here, both directions, before the runner."""
        from .kernels.proj_gemm import run_fused_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        if self.fp8:
            raise ValueError(f"{self.name}: this stage was declared FP8; use execute_fp8(...)")
        if self.mxfp8:
            raise ValueError(f"{self.name}: this stage was declared MXFP8; use execute_mxfp8(...)")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
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
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        if not self._runner_writes_compact_qkv():
            # Never hand the round-1 slab runner three compact buffers positionally.
            raise NotImplementedError(f"{self.name}: {self._fp8_fork_available()}")
        run_fused_proj_gemm_fp8(self._plan, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, self._qscal, stream=stream)

    def execute_mxfp8(
        self, a, sf_a, w, sf_w, out_q8, out_k8, out_v8, out_gate16, out_sf_q, out_sf_k, out_sf_v, w_q_norm, w_k_norm, cos, sin, *, stream
    ) -> None:
        """MXFP8 arm (frozen runner ABI, PR-B plan 3.1): e4m3 codes ``a [M, K]`` + its
        F8_128x4 blob ``sf_a``, ``w [N_qkvg, K]`` + ``sf_w``; COMPACT e4m3 ``q8`` / ``k8`` /
        ``v8`` + their SDPA F8_128x4 blobs ``sf_q`` / ``sf_k`` / ``sf_v`` (Q/K rowwise
        per-(b, h, s_tile) 1024-B tiles; V columnwise D-plane-major) + bf16 ``gate16`` out.
        ``batch`` / ``seq_len`` ride as keywords: the SF tiles are decoded per sequence."""
        from .kernels import proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute_mxfp8()")
        if not self.mxfp8:
            raise ValueError(f"{self.name}: this stage was not declared MXFP8; use execute(...) / execute_fp8(...)")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        missing = self._mxfp8_fork_available()
        if missing is not None:
            raise NotImplementedError(f"{self.name}: {missing}")
        getattr(proj_gemm, self._MXFP8_RUNNER)(
            self._plan,
            a,
            sf_a,
            w,
            sf_w,
            out_q8,
            out_k8,
            out_v8,
            out_gate16,
            out_sf_q,
            out_sf_k,
            out_sf_v,
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            batch=self.batch,
            seq_len=self.seq_len,
            stream=stream,
        )


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

    **``geometry.qk_norm=False`` -- RoPE only.** Both kernels fold the RMSNorm
    out at trace time (``compile_qk_norm_rope[_tma](apply_norm=False)``: the
    weight slots are traced as ``None``, exactly the presence switch ``want_rstd``
    uses), so there is no sum-of-squares pass, no rsqrt, no weight load and no
    rstd; the dims ``[rope_dim, d_head)`` come out BIT-EXACT. The stage name and
    position do not change (``"qk_norm_rope"`` stays in ``_stages``); ``execute``
    takes ``None`` for both weights and refuses tensors, both directions typed.

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
        g = self.geom
        if not g.qk_norm:
            # RoPE-only: nothing to norm, so nothing to emit an rstd from, and
            # with no RoPE either the stage would be an identity copy.
            if self.want_rstd:
                raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and emits no rstd; want_rstd must be False")
            if g.rope_dim == 0:
                raise ValueError(f"{self.name}: geometry.qk_norm=False with rope_dim=0 is an identity copy of Q/K; drop the stage instead")
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
                apply_norm=g.qk_norm,
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
            apply_norm=g.qk_norm,
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
        w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],  # [D]
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

        The norm weights must agree with ``geometry.qk_norm`` in both directions
        (typed ``ValueError`` naming the knob) -- checked here, before the recipe.
        """
        from .kernels.qk_norm_rope import run_qk_norm_rope
        from .kernels.qk_norm_rope_tma import run_qk_norm_rope_tma

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        g = self.geom
        _check_norm_weights_agree(g.qk_norm, w_q_norm, w_k_norm)
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

    **MXFP8** (``mxfp8=True``, e4m3 codes + F8_128x4 E8M0 scale factors): the
    SAME adapter with ``pertensor_fp8=False`` -> ``sm107/prefill_d256_mxfp8.py``
    (the production block-scale kernel; the per-tensor fork machinery is gone,
    so nothing can route block-scaled data into the per-tensor kernel).  The
    three SF blobs ride ``execute(sf_q=, sf_k=, sf_v=)`` and are REQUIRED;
    ``descale_q/k/v`` / ``scale_o`` are REFUSED (the adapter would silently
    ignore them -- the E8M0 dequant is in-MMA and a gated e4m3 O is UNSCALED,
    D8).  ``sched_policy`` is read off the MXFP8 row
    (``engine_name(arch, mxfp8=True)``: NATURAL at (256, 256) -- the per-tensor
    row's LPT claim does NOT transfer, D5) and ``cta_mma`` is left to the
    adapter (1 for the quantized d256 flavor).  ``has_amax_o=False`` as under FP8.

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
        mxfp8: bool = False,
    ) -> None:
        # token_stride != 0 => Q/K/V are column slices of the fused projection
        # and are read in place at that stride (the adapter compiles its TMA
        # descriptors at the declared strides).  0 => the compact buffers.
        self.token_stride = int(token_stride)
        # FP8 (dtype == e4m3): per-tensor descales (and scale_o) ride execute();
        # O comes out in `o_dtype` -- bf16 on the unfused pipeline, e4m3 on the
        # fully fused one.  `fp8` = the fp8-CLASS dtype; `mxfp8` selects the
        # block-scale kernel (pertensor_fp8=False), `pertensor` the scalar one.
        self.o_dtype = o_dtype if o_dtype is not None else dtype
        self.fp8 = dtype == torch.float8_e4m3fn
        self.mxfp8 = bool(mxfp8)
        if self.mxfp8 and not self.fp8:
            raise ValueError(f"{self.name}: mxfp8=True needs e4m3 Q/K/V codes, got dtype={dtype}")
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
        # pertensor_fp8 picks the kernel FAMILY: True -> prefill_d256_fp8.py
        # (scalar descales), False -> prefill_d256_mxfp8.py (block scales).
        kw = dict(pertensor_fp8=self.pertensor, dtype_o=self.o_dtype, has_amax_o=False) if self.fp8 else {}
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

    @property
    def pertensor(self) -> bool:
        """The per-tensor FP8 kernel family (scalar descales); False for bf16 and for MXFP8."""
        return self.fp8 and not self.mxfp8

    @property
    def _family(self) -> str:
        return "MXFP8" if self.mxfp8 else "FP8" if self.fp8 else "f16/bf16"

    @classmethod
    def _row_capabilities(cls, arch: str, fp8: bool, mxfp8: bool = False):
        """The ``Capabilities`` of the SDPA-forward engine row for ``(arch, dtype family)``.

        ``fp8`` names the per-tensor FP8 row, ``mxfp8`` the block-scale one
        (``engine_name(arch, fp8=..., mxfp8=...)``); a caller passing the fp8-class
        flag with ``mxfp8=True`` gets the MXFP8 row -- the two rows make DIFFERENT
        claims (LPT, gate dtypes), so the family must never be conflated.

        A missing / renamed row is a typed decline, not a bare ``StopIteration``
        escaping ``check_support`` (engine-contract § 2): this sits on the decline
        path of every ``fuse_gate=True`` block, on every device.
        """
        from cudnn.sdpa.fwd import engines

        row = engines.engine_name(arch=arch, fp8=bool(fp8) and not mxfp8, mxfp8=bool(mxfp8))
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
        # MXFP8 reads the MXFP8 row (D5): at (256, 256) it claims NATURAL only --
        # the per-tensor row's LPT does not transfer, and `sched_policy=None`
        # would let the adapter's auto knobs pick LPT for a masked fp8-class d256.
        caps = self._row_capabilities(arch, self.fp8, self.mxfp8)
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
        caps = self._row_capabilities("sm107", self.fp8, self.mxfp8)
        if not caps.epilogue_gate:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True is not served by the {self._family} SM107 SDPA engine row. "
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
                f"{self._family} SM107 SDPA engine row; got gate_dtype={gate_dtype}. "
                "Pass gate_dtype= (the block's activation dtype) or use fuse_gate=False"
            )

    def check_support(self) -> None:
        # Geometry first, so the decline reads the same on every device (the
        # block's own stages ahead of this one are cc-independent too).
        if self.fuse_gate:
            self._check_gate_geometry()
        if self.mxfp8 and self.fuse_gate:
            # The block-scale kernel is reached ONLY through the production
            # adapter's `sample_gate` (PR-A / S7).  An adapter without it has no
            # gated MXFP8 path at all -- decline rather than reach for any
            # per-tensor fork, which cannot take block scales.
            import inspect

            from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

            if "sample_gate" not in inspect.signature(SdpaFwdDslSm100.__init__).parameters:
                raise NotImplementedError(
                    f"{self.name}: MXFP8 + fuse_gate needs the production adapter's `sample_gate` (the shared epilogue_gate hook); "
                    "this checkout's SdpaFwdDslSm100 has none. Use fuse_gate=False."
                )
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
        sf_q: Optional[torch.Tensor] = None,  # MXFP8 only (all three REQUIRED): F8_128x4 E8M0 blobs, uint8, the adapter's _reshape_sf byte count
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
    ) -> None:
        """Hand the BSHD buffers over as BHSD views. No copy — see :func:`_bhsd_desc`.

        ONE call into the adapter for every configuration; the FP8 scales and
        the gate ride as keyword arguments the adapter validates against the
        specialization it compiled (``gate`` <-> ``sample_gate``, ``amax_o``
        refused under ``has_amax_o=False``).

        MXFP8 (``mxfp8=True``): ``sf_q`` / ``sf_k`` / ``sf_v`` are REQUIRED and
        ``descale_q/k/v`` / ``scale_o`` are REFUSED -- the adapter's MXFP8 path
        accepts and silently ignores the scalars (``_execute_mxfp8`` never reads
        them), and a silently-ignored scale is a wrong answer nobody reports.
        A non-MXFP8 stage refuses the SF blobs for the mirror-image reason.
        """
        if self._impl is None:
            raise RuntimeError("call compile() before execute()")
        if self.mxfp8:
            if sf_q is None or sf_k is None or sf_v is None:
                raise ValueError(f"{self.name}: the MXFP8 SDPA needs sf_q/sf_k/sf_v (F8_128x4 E8M0 scale-factor blobs, uint8)")
            if descale_q is not None or descale_k is not None or descale_v is not None or scale_o is not None:
                raise ValueError(
                    f"{self.name}: descale_q/k/v and scale_o are per-tensor FP8 scalars; the MXFP8 kernel dequantizes with its block "
                    "scale factors in the MMA and writes an e4m3 O UNSCALED (the adapter would silently ignore them -- refused instead)"
                )
        elif sf_q is not None or sf_k is not None or sf_v is not None:
            raise ValueError(f"{self.name}: sf_q/sf_k/sf_v are the MXFP8 scale-factor blobs; this stage was declared {self._family}")
        if self.pertensor:
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
        elif not self.mxfp8 and (descale_q is not None or descale_k is not None or descale_v is not None or scale_o is not None):
            raise ValueError(f"{self.name}: descales / scale_o are only consumed by the FP8 pipeline")
        if self.fuse_gate and gate is None:
            raise ValueError(f"{self.name}: fuse_gate=True requires the GATE tensor at execute")
        if not self.fuse_gate and gate is not None:
            raise ValueError(f"{self.name}: gate is only consumed under fuse_gate=True (the SDPA's epilogue gate); this stage was declared without it")
        kw = {}
        if self.pertensor:
            kw.update(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o)
        if self.mxfp8:
            kw.update(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v)
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
                         geometry.qk_norm=False: RoPE only -- same stage, same launch, no norm weights (None), no rstd
        (3b) compact     PROJ[V]      -> V_c                    only when inplace_qkv=False
        (4) sdpa         PROJ[Q,K,V]  -> O  (+LSE)              FROST SDPA        | fuse_gate: (4)+(5) in ONE launch
        (5) gate         O, PROJ[G]   -> O in place             this block's kernel| (the SDPA's epilogue_gate)
        (6) out_proj     O            -> out                    FROST GEMM

    Both fusions are inference-only: each overwrites a tensor the backward needs
    (``q_pre``/``k_pre``, pre-gate ``O``) and is declined with ``save_for_backward``.

    **MXFP8** (an :class:`MxQuantSpec` + e4m3 codes + F8_128x4 SF blobs for ``h`` /
    ``W_qkvg``), UNFUSED (9 stages = 9 kernel launches)::

        (1)  proj          h8+sf_h, W8+sf_w -> PROJ [T, N] bf16   block-scale FROST GEMM (E8M0 dequant in-MMA, no alpha)
        (2+3) norm+rope    in place                                this block's kernel
        (3q) quantize x3   PROJ[Q]/[K] rowwise, PROJ[V] columnwise -> q8/k8/v8 + sf_q/sf_k/sf_v   kernels/quantize_mxfp8.py
        (4)  sdpa          q8,k8,v8 + SF -> O bf16                 production prefill_d256_mxfp8.py (NATURAL, cga1)
        (5)  gate          O, PROJ[G] -> O in place                this block's kernel
        (5q) quantize_o    O -> o8 (PER-TENSOR scale_o, D1)        kernels/quantize.py
        (6)  out_proj      o8 -> out (alpha_o = descale_w_o / scale_o)   per-tensor FP8 FROST GEMM

    and FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``, 3 launches):
    ``proj(+norm+rope+block-quant -> q8/k8/v8 + sf_q/k/v + gate16) -> sdpa(+gate, e4m3 O UNSCALED;
    scale_o must be 1.0, D8) -> out_proj`` (the ``run_fused_proj_gemm_mxfp8`` fork twin is
    feature-detected; a checkout without it gets a typed decline).  Padding (``seq_lens_present``) is served under
    FP8 and MXFP8 like bf16: a dead entry (``seq_lens[b] == 0``) yields ``out[b] == 0`` exactly.

    **MXFP4 W_qkvg** (``MxQuantSpec.w_qkvg_dtype = torch.float4_e2m1fn_x2``: e2m1 codes ``[N, d_model // 2]``,
    two per byte, ``w_qkvg_sf`` unchanged): the SAME nine unfused stages with stage (1) on the mixed
    MXFP8 x MXFP4 block-scale row::

        (1)  proj          h8+sf_h, W4+sf_w -> PROJ [T, N] bf16   block-scale FROST GEMM, mixed row (fp8_e4m3 x fp4_e2m1, E8M0/32)

    UNFUSED only -- ``fuse_norm_rope`` with an e2m1 ``W_qkvg`` is a typed decline (the fork twin is
    rendered for an e4m3 B; feature-detected on ``NormRopeFusionParams.weight_fp4``).

    **fp4 O** (``MxQuantSpec.o_fp4 = Fp4Format.NVFP4 | MXFP4`` + an e2m1 ``W_o`` ``[d_model, H_q*D // 2]``
    with its F8_128x4 blob ``sample_w_o_sf`` / ``w_o_sf``): the per-tensor tail of BOTH MXFP8
    configurations is replaced -- UNFUSED (still 9 launches)::

        (5q') quantize_fp4_o  O bf16 -> o4 [T, H_q*D/2] e2m1 + sf_o (the GEMM's padded blob)   kernels/quantize_fp4.py
        (6')  out_proj        o4, W_o4 -> out                        fp4 x fp4 block-scale FROST GEMM (scales dequant in-MMA, no alpha)

    and FULLY FUSED (4 launches: the gated MXFP8 SDPA writes **bf16** O, then (5q') and (6')).
    ``scale_o`` / ``descale_w_o`` are pinned to 1.0 (no global scale in either format); ``o8`` is not
    reserved, ``o4`` / ``sf_o`` are appended at the end of the workspace arm.  Composes with the MXFP4
    ``W_qkvg`` above on the unfused pipeline (the mixed GEMM at (1), the fp4 tail at (5q') / (6')).

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
        sample_w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False -- same positions
        sample_w_k_norm: Optional[torch.Tensor],  # [D]
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
        quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,  # FP8 (E4M3) per-tensor static scales, or MXFP8 (MxQuantSpec); None = bf16/f16
        sample_h_sf: Optional[torch.Tensor] = None,  # MXFP8 only: F8_128x4 E8M0 blob of h, proj_gemm.sf_blob_bytes(B*S, d_model) bytes (uint8 / e8m0)
        sample_w_qkvg_sf: Optional[torch.Tensor] = None,  # MXFP8 only: F8_128x4 E8M0 blob of W_qkvg, sf_blob_bytes(n_qkvg, d_model) bytes
        sample_w_o_sf: Optional[torch.Tensor] = None,  # fp4 O only (MxQuantSpec.o_fp4): the e2m1 W_o's F8_128x4 blob, sf_blob_bytes(d_model, h_q*d_head, block)
    ):
        super().__init__()
        self._warn_experimental_api()
        self.geom = geometry
        self.dtype = sample_h.dtype
        self.device = sample_h.device
        # FP8: `h` and both weights arrive as e4m3 with a QuantSpec; every
        # activation the block's own kernels touch (slab, O, out, cos/sin, norm
        # weights) is bf16 -- the "activation dtype".  bf16/f16: the two coincide.
        # MXFP8: an MxQuantSpec instead, plus the two scale-factor blobs.
        if quant is not None and not isinstance(quant, (QuantSpec, MxQuantSpec)):
            raise TypeError(f"quant must be a QuantSpec (per-tensor FP8) or an MxQuantSpec (MXFP8), got {type(quant).__name__}")
        self.quant = quant
        self.mxfp8 = isinstance(quant, MxQuantSpec)
        if (self.dtype == torch.float8_e4m3fn) != (quant is not None):
            raise ValueError(
                "FP8 needs both halves: an e4m3 `h` AND a QuantSpec / MxQuantSpec (static scales). "
                f"Got h.dtype={self.dtype}, quant={'set' if quant is not None else 'None'}."
            )
        have_sf = sample_h_sf is not None or sample_w_qkvg_sf is not None
        if self.mxfp8 != have_sf or (self.mxfp8 and (sample_h_sf is None or sample_w_qkvg_sf is None)):
            raise ValueError(
                "MXFP8 needs all three halves: e4m3 `h` / `W_qkvg` codes, an MxQuantSpec AND both F8_128x4 E8M0 scale-factor blobs "
                f"(sample_h_sf, sample_w_qkvg_sf). Got quant={type(quant).__name__ if quant is not None else 'None'}, "
                f"sample_h_sf={'tensor' if sample_h_sf is not None else 'None'}, sample_w_qkvg_sf={'tensor' if sample_w_qkvg_sf is not None else 'None'}."
            )
        if self.mxfp8:
            # D8 rides the declaration: a fully fused MXFP8 block writes e4m3 O
            # UNSCALED, so scale_o != 1.0 is refused HERE (typed), not silently dropped.
            quant.validate(fused=bool(fuse_gate) and bool(fuse_norm_rope))
        elif quant is not None:
            quant.validate()
        # fp4 O (MxQuantSpec.o_fp4): the e2m1 W_o needs its scale blob, and a blob needs the mode -- both
        # halves or neither, at declaration (the field lives on MxQuantSpec only, so a QuantSpec / bf16
        # block can name the blob but never the mode: refused here, typed).
        self.o_fp4: Optional[Fp4Format] = quant.o_fp4 if self.mxfp8 else None
        if (self.o_fp4 is not None) != (sample_w_o_sf is not None):
            raise ValueError(
                "fp4 O needs both halves: MxQuantSpec.o_fp4 (Fp4Format.NVFP4 | MXFP4) AND sample_w_o_sf (the F8_128x4 scale blob of the e2m1 W_o, "
                f"proj_gemm.sf_blob_bytes(d_model, h_q*d_head, block) bytes). Got o_fp4={self.o_fp4}, "
                f"sample_w_o_sf={'tensor' if sample_w_o_sf is not None else 'None'}" + ("" if self.mxfp8 else " (only an MxQuantSpec carries o_fp4)") + "."
            )
        # geometry.qk_norm vs the two weight slots, BOTH directions, at
        # declaration (again at every execute).  A None sample would otherwise
        # be swallowed by _make_tensor_desc and surface as an untyped
        # AttributeError in check_support's dtype loop.
        _check_norm_weights_agree(geometry.qk_norm, sample_w_q_norm, sample_w_k_norm, prefix="sample_")
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
        # The two training guards are KEPT under qk_norm=False (D11): the
        # SavedForBackward contract still writes q_pre/k_pre out of place and
        # no block-level save_for_backward=True execute has validated a
        # relaxation (PR-B plan § 6 Q10).  Only the REASON differs, so the
        # message must not claim an RMSNorm backward that does not exist.
        _why_pre = (
            "norming in place destroys q_pre/k_pre, which the RMSNorm backward needs and cannot reconstruct (dividing out "
            "the norm weight is undefined at a zero weight and hostile at a small one -- see SavedForBackward)"
            if geometry.qk_norm
            else "rotating in place overwrites q_pre/k_pre, which the SavedForBackward contract still hands the backward "
            "out of place under qk_norm=False (RoPE-only; relaxing this needs a block-level training validation first)"
        )
        if self.inplace_qkv and self.save_for_backward:
            raise ValueError(
                f"inplace_qkv=True is incompatible with save_for_backward=True: {_why_pre}. Pass inplace_qkv=False, or recompute "
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
                f"fuse_norm_rope=True writes {'normed' if geometry.qk_norm else 'rotated'} Q/K straight into the projection slab "
                "(the in-place layout) and never materialises q_pre/k_pre, so it requires inplace_qkv=True and is incompatible "
                "with save_for_backward=True. Pass fuse_norm_rope=False for training."
            )

        # FUSED GATE: the production d256 SDPA's epilogue_gate specialization
        # reads GATE in its epilogue and writes O_gated, so stage (5) is not
        # built.  It overwrites the one tensor the backward's dG needs (pre-gate
        # O -- SavedForBackward), so training declines it here, the same way
        # inplace_qkv / fuse_norm_rope do.
        self.fuse_gate = bool(fuse_gate)
        # FP8 / MXFP8 serve exactly TWO configurations: UNFUSED (7 FP8 stages or
        # 9 MXFP8 stages -- 9 kernel launches either way, the FP8 Q/K/V quantize
        # stage being three launches) and FULLY FUSED (3 launches:
        # proj(+norm+rope+quant) -> sdpa(+gate, e4m3 O) -> out_proj).  Each
        # half-fused combination would be its own specialization (bf16 O + a
        # quantize pass, or a gate read from the bf16 slab) that nobody has
        # validated -- declined, typed, naming both knobs.
        _family = "MXFP8" if self.mxfp8 else "FP8"
        if quant is not None and self.fuse_gate != self.fuse_norm_rope:
            raise NotImplementedError(
                f"the {_family} pipeline is either fully fused or unfused: fuse_norm_rope and fuse_gate must BOTH be True "
                f"(3 launches) or BOTH be False ({'9' if self.mxfp8 else '7'} stages, 9 kernel launches); "
                f"got fuse_norm_rope={self.fuse_norm_rope}, fuse_gate={self.fuse_gate}"
            )
        if quant is not None and self.save_for_backward:
            raise NotImplementedError(f"the {_family} pipeline is inference-only for now (no q_pre/k_pre/pre-gate O contract under quantization)")
        # seq_lens_present (a dense padding mask, incl. an EMPTY entry) is SERVED
        # under FP8 and MXFP8 since 2026-09-15.  The decline that used to sit here
        # ("the Rubin FP8 d256 SDPA hangs on seq_kv_lens == 0") is retired: the
        # rebased kernels pass 8/8 fresh processes at S=1000 and S=512 with a dead
        # entry, and the MXFP8 d256 leading-zero-length-KV L0 test passes on Rubin.
        # The block's own dead-entry oracle tests (test_block_fp8.py /
        # test_block_mxfp8.py: seq_lens=[s, 0] -> out[1] == 0 EXACTLY) pin it.
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
            # MXFP8 only (None otherwise): the caller's F8_128x4 blobs, validated in check_support.
            "h_sf": self._make_tensor_desc(sample_h_sf, name="h_sf"),
            "w_qkvg_sf": self._make_tensor_desc(sample_w_qkvg_sf, name="w_qkvg_sf"),
            # fp4 O only (None otherwise): the e2m1 W_o's F8_128x4 blob, validated in check_support.
            "w_o_sf": self._make_tensor_desc(sample_w_o_sf, name="w_o_sf"),
        }

        fp8 = quant is not None  # fp8-CLASS pipeline (per-tensor FP8 or MXFP8): e4m3 h / weights, bf16 activations
        mxfp8 = self.mxfp8
        # FULLY FUSED: both knobs under a quant spec (the both-or-neither decline
        # above makes `fp8 and fuse_gate` == `fp8 and fuse_norm_rope`).
        # `fp8_fused` names the per-tensor pipeline (test-visible, unchanged);
        # `mxfp8_fused` the block-scale one; `quant_fused` either.
        self.fp8_fused = fp8 and not mxfp8 and self.fuse_gate and self.fuse_norm_rope
        self.mxfp8_fused = mxfp8 and self.fuse_gate and self.fuse_norm_rope
        self.quant_fused = self.fp8_fused or self.mxfp8_fused
        act = self.act_dtype
        self._quant_dev = None  # the QuantSpec / MxQuantSpec as device scalars, materialised in compile()
        # rstd exists only where a norm exists: under qk_norm=False a training
        # block saves lse / q_pre / k_pre but no rstd (SavedForBackward.rstd_*
        # are None -- required, both directions, at execute).
        want_rstd = self.save_for_backward and geometry.qk_norm
        if self.fuse_norm_rope:
            # bf16: writes the [T, N] slab.  FP8: the second rendering writes the
            # compact e4m3 q8/k8/v8 + the bf16 gate16 buffer, quantizing in its
            # epilogue.  MXFP8: the block-scale twin (typed decline until S6 lands).
            self._proj = _FusedQkvProjection(
                geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, want_rstd=want_rstd, quant=quant, device=self.device
            )
            self._norm_rope = None  # lives in the fused epilogue
        else:
            # FP8: e4m3 x e4m3 -> fp32 -> * (descale_h * descale_w_qkvg) -> bf16 slab.
            # MXFP8: e4m3 x e4m3 with the E8M0 block scales dequantized IN the MMA -> bf16 slab (no alpha).
            # MXFP8 with an e2m1 W_qkvg (MxQuantSpec.w_qkvg_dtype): the same GEMM on the catalog's
            # MIXED row (e4m3 h x e2m1 W, E8M0 / 32) -- same stage list, same workspace (config row 7).
            self._proj = _qkv_gate_projection(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                out_dtype=act,
                alpha=fp8 and not mxfp8,
                block_scale=mxfp8,
                w_dtype=quant.w_qkvg_dtype if mxfp8 else None,
            )
            self._norm_rope = _QkNormRope(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, want_rstd=want_rstd)
        # Stage (3b) exists ONLY to give the SDPA a compact V. In-place needs no
        # such thing, so the stage is not built at all rather than built and
        # skipped -- a stage that is never run should not be in `_stages`,
        # where it would still be compiled and still report support.  Under
        # FP8 / MXFP8 the quantize stages compact V (and Q, K) on the way to e4m3.
        self._compact_v = None if (self.inplace_qkv or fp8) else _VCompaction(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act)
        # (3q) UNFUSED FP8 only: bf16 slab slices -> compact e4m3 Q/K/V.  Two
        # recipes (h_q and h_kv); K and V share the h_kv one.  Fully fused FP8
        # quantizes in the projection fork's epilogue and builds none of them.
        # (3q) UNFUSED MXFP8: THREE block-quantize stages -- Q and K ROWWISE
        # (two recipes, h_q / h_kv), V COLUMNWISE (its own recipe: a different
        # kernel arm AND a different SF byte order) -- each writing compact e4m3
        # + the SDPA's F8_128x4 SF blob.
        quantize = fp8 and not self.quant_fused
        self._quant_q = self._quant_kv = self._quant_k = self._quant_v = None
        if quantize and not mxfp8:
            self._quant_q = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, name="quantize_q")
            self._quant_kv = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_kv, name="quantize_kv")
        elif quantize:
            _mxq = lambda heads, axis, name: _QuantizeMxfp8(
                geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=heads, axis=axis, name=name
            )  # noqa: E731
            self._quant_q = _mxq(geometry.h_q, "row", "quantize_mxfp8_q")
            self._quant_k = _mxq(geometry.h_kv, "row", "quantize_mxfp8_k")
            self._quant_v = _mxq(geometry.h_kv, "col", "quantize_mxfp8_v")
        if self.quant_fused:
            # Q/K/V are the COMPACT e4m3 q8/k8/v8 the projection fork writes
            # (token_stride 0 -- what the unfused quantized SDPA reads too), the
            # GATE is the compact bf16 gate16, O comes out e4m3 (o8).  The gated
            # FP8 / MXFP8 SDPA specialization (epilogue_gate) is compiled at
            # compact strides.
            # fp4 O (row 10): the gated MXFP8 SDPA writes bf16 O (the adapter serves a bf16 O for fp8-class input --
            # no SDPA row changes) and the quantize_fp4 stage turns it into the fp4 out_proj's A operand.
            sdpa_token_stride, sdpa_gate_token_stride = 0, geometry.h_q * geometry.d_head
            sdpa_o_dtype = torch.float8_e4m3fn if self.o_fp4 is None else act
        else:
            # bf16: in-place reads the slab at its padded stride; FP8 / MXFP8
            # unfused read the compact e4m3 buffers.  GATE is always a slab column slice.
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
            gate_dtype=act,  # gate16 / the slab's GATE columns are the activation dtype (bf16 under FP8 / MXFP8)
            mxfp8=mxfp8,  # pertensor_fp8=False -> the production block-scale kernel; NATURAL read off the MXFP8 row (D5)
        )
        # Stage (5) lives in the SDPA kernel's gate epilogue under fuse_gate --
        # not built rather than built and skipped (it would still compile).
        self._gate = None if self.fuse_gate else _SigmoidGate(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act)
        # (5q) UNFUSED FP8 only: bf16 gated O -> compact e4m3 for the out
        # projection (same [T, H_q, D] shape as Q, so the Q recipe serves it).
        # UNFUSED MXFP8: an EXPLICIT per-tensor recipe (D1) -- never the rowwise
        # MX recipe, whose SF blob nothing downstream would read.
        # (5q') fp4 O (MXFP8, unfused OR fused): ONE block-quantize launch (bf16 gated O -> e2m1 codes + the
        # out_proj GEMM's padded F8_128x4 blob) replaces the per-tensor quantize_o; the out projection becomes
        # the fp4 x fp4 block-scale GEMM (no alpha: both sides dequantize through their blobs in the MMA).
        quant_o_distinct = mxfp8 and (quantize or self.o_fp4 is not None)
        if self.o_fp4 is not None:
            self._quant_o = _QuantizeFp4(
                geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, fmt=self.o_fp4, name="quantize_fp4_o"
            )
            self._out_proj = _out_projection(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=_FP4_X2,
                out_dtype=act,
                alpha=False,
                block_scale=True,
                w_dtype=_FP4_X2,
                block_size=self.o_fp4.block_size,
                sf_dtype=self.o_fp4.sf_cudnn_dtype,
            )
        else:
            if quantize and mxfp8:
                self._quant_o = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, name="quantize_o")
            else:
                self._quant_o = self._quant_q
            self._out_proj = _out_projection(geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, out_dtype=act, alpha=fp8)
        # Stage order == pipeline order.  `_quant_o` is listed only where it is a
        # DISTINCT stage (MXFP8: quantize_o, or quantize_fp4_o under o_fp4); under
        # FP8 it aliases `_quant_q`, listed once.
        self._stages = tuple(
            st
            for st in (
                self._proj,
                self._norm_rope,
                self._compact_v,
                self._quant_q,
                self._quant_kv,
                self._quant_k,
                self._quant_v,
                self._sdpa,
                self._gate,
                self._quant_o if quant_o_distinct else None,
                self._out_proj,
            )
            if st is not None
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
        self._check_declaration()
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    def _expected_weight_dtypes(self) -> dict:
        """``{"w_qkvg": dtype, "w_o": dtype}`` -- what each weight descriptor must carry.  ``h``'s dtype
        (bf16 / f16, or e4m3 under a QuantSpec / MxQuantSpec) unless the spec names a packed e2m1 weight:
        ``MxQuantSpec.w_qkvg_dtype`` for ``w_qkvg`` (the mixed block-scale row).  ONE place for the
        per-weight contract, so a further fp4 weight is one more entry here, not a second loop."""
        w_qkvg = self.quant.w_qkvg_dtype if self.mxfp8 else self.dtype
        w_o = _FP4_X2 if self.o_fp4 is not None else self.dtype
        return {"w_qkvg": w_qkvg, "w_o": w_o}

    @staticmethod
    def _fp4_storage_shape(desc: TensorDesc) -> Tuple[int, ...]:
        """The STORAGE shape of a ``float4_e2m1fn_x2`` descriptor.  ``_make_tensor_desc`` reports the LOGICAL
        shape of a packed-fp4 tensor (the innermost, stride-1 extent DOUBLED -- pinned by
        ``test_make_tensor_desc_reports_the_logical_k_of_an_fp4x2_tensor``); this halves that one axis back,
        so the caller-facing check speaks the shape the caller allocated (``[N, K // 2]``)."""
        inner = desc.stride_order[0]  # ascending stride: [0] is the stride-1 axis the descriptor doubled
        return tuple(int(n) // 2 if i == inner else int(n) for i, n in enumerate(desc.shape))

    def _check_weight(self, nm: str, rows: int, k: int, expect: torch.dtype) -> None:
        """One weight, dtype THEN shape, typed.  A packed e2m1 weight is checked against its STORAGE shape
        ``(rows, k // 2)``; every other dtype against ``(rows, k)`` exactly as before."""
        d = self._descs[nm]
        fp4_expected = _FP4_X2 is not None and expect == _FP4_X2
        if not fp4_expected:
            self._check_tensor_shape(d, (rows, k), nm)
            if d.dtype == _FP4_X2:
                need = (
                    "declare it with MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)"
                    if nm == "w_qkvg"
                    else "an e2m1 w_o is the block's fp4 O output mode: declare it with MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4) plus sample_w_o_sf"
                )
                raise ValueError(
                    f"{nm} is torch.float4_e2m1fn_x2 but this block expects {expect}: a packed e2m1 {nm} rides the block-scale "
                    f"MXFP8 pipeline only -- {need}; a QuantSpec / bf16 block has no GEMM row for it"
                )
            if d.dtype != expect:
                raise ValueError(f"{nm} must have h's dtype {expect}, got {d.dtype}")
            return
        if d.dtype != _FP4_X2:
            hint = (
                " -- torch 2.13 can VIEW but not cast to fp4: hand over the packed codes as storage.view(torch.float4_e2m1fn_x2), not uint8"
                if d.dtype == torch.uint8
                else ""
            )
            field = "MxQuantSpec.w_qkvg_dtype" if nm == "w_qkvg" else "MxQuantSpec.o_fp4"
            raise ValueError(f"{nm} must be torch.float4_e2m1fn_x2 ({field} names an e2m1 {nm}), got {d.dtype}{hint}")
        storage = self._fp4_storage_shape(d)
        if storage != (rows, k // 2):
            raise ValueError(
                f"{nm} is fp4 storage {storage} (logical {tuple(int(x) for x in d.shape)}); a packed e2m1 {nm} is [{rows}, {k} // 2 = {k // 2}] "
                f"-- two codes per byte along K, LOW nibble = even k.  A LOGICAL [{rows}, {k}] fp4 tensor holds twice the data the GEMM declares."
            )

    def _check_declaration(self) -> None:
        """Everything ``check_support`` verifies BEFORE asking the stages (geometry, every descriptor's shape
        and dtype, the MXFP8 blob contract) -- device-agnostic, so a test can pin the caller contract on
        any GPU while the stages' own ``check_support`` still gates the arch."""
        self.geom.validate()
        if self.save_for_backward and not self.return_lse:
            raise ValueError("save_for_backward requires the LSE: the SDPA backward cannot run without it")
        g = self.geom
        want = self._expected_weight_dtypes()
        self._check_weight("w_qkvg", g.n_qkvg, g.d_model, want["w_qkvg"])
        self._check_weight("w_o", g.d_model, g.h_q * g.d_head, want["w_o"])
        # Under qk_norm=False the two norm-weight descriptors are None (the
        # constructor checked both directions), so they are skipped here: the
        # shape check would pass silently and the dtype loop would raise an
        # untyped AttributeError.
        norm_names = ("w_q_norm", "w_k_norm") if g.qk_norm else ()
        for nm in norm_names:
            self._check_tensor_shape(self._descs[nm], (g.d_head,), nm)
        for nm in ("cos", "sin"):
            self._check_tensor_shape(self._descs[nm], (self.batch, self.seq_len, g.rope_dim), nm)
        self._check_tensor_shape(self._descs["out"], (self.batch, self.seq_len, g.d_model), "out")
        # dtype contract: bf16/f16 everywhere, or (FP8 / MXFP8) e4m3 h + weights (an e2m1 W_qkvg
        # under MxQuantSpec.w_qkvg_dtype -- checked per weight above) with every activation-side
        # tensor in the bf16 activation dtype.
        for nm in norm_names + ("cos", "sin", "out"):
            if self._descs[nm].dtype != self.act_dtype:
                raise ValueError(f"{nm} must be the activation dtype {self.act_dtype}, got {self._descs[nm].dtype}")
        if self.mxfp8:
            self._check_mxfp8_declaration()

    def _check_mxfp8_declaration(self) -> None:
        """The MXFP8 caller contract, typed, before any stage: whole SF atoms along
        K (``d_model % 128``), and the two F8_128x4 blobs' dtype / PADDED byte count
        / contiguity (``proj_gemm.sf_blob_bytes``).  16-B alignment needs a data
        pointer, so it is checked on the real tensors at ``execute``."""
        from .kernels.proj_gemm import sf_blob_bytes

        g = self.geom
        if g.d_model % _SF_TILE_ROWS:
            # v1 contract: K = d_model in whole 128-element F8_128x4 atoms (4 blocks of
            # 32), so `h_sf` / `w_qkvg_sf` carry no partially-used 4-block words and
            # the quantizer / GEMM / oracle agree on every byte.  Rows (T, n_qkvg) ARE
            # padded (sf_padded_dims); only the K axis is pinned.
            raise NotImplementedError(
                f"MXFP8 needs geometry.d_model % {_SF_TILE_ROWS} == 0 (whole F8_128x4 scale-factor atoms along the contraction: 4 blocks of "
                f"{MXFP8_BLOCK_SIZE}); got d_model={g.d_model}. Use the per-tensor FP8 or bf16 pipeline for this width."
            )
        t = self.batch * self.seq_len
        # (name, rows, K, block, accepted storage dtypes) of every caller blob: the two MXFP8 ones (E8M0 / 32),
        # plus the e2m1 W_o's under o_fp4 (the format's own scale dtype and block; K = the out_proj's H_q*D).
        blobs = [("h_sf", t, g.d_model, MXFP8_BLOCK_SIZE, _SF_DTYPES), ("w_qkvg_sf", g.n_qkvg, g.d_model, MXFP8_BLOCK_SIZE, _SF_DTYPES)]
        if self.o_fp4 is not None:
            block = self.o_fp4.block_size
            if g.d_head % (4 * block):
                # One head's scales must be whole 4-block F8_128x4 words, so the per-head quantize CTA owns whole
                # atoms of the out_proj blob and no two heads share a byte (d=256 passes both formats).
                raise NotImplementedError(
                    f"fp4 O ({self.o_fp4.name}) needs geometry.d_head % {4 * block} == 0 (whole 4-block scale words per head at block {block}); "
                    f"got d_head={g.d_head}. Use the per-tensor e4m3 O (o_fp4=None) for this head dim."
                )
            blobs.append(("w_o_sf", g.d_model, g.h_q * g.d_head, block, (torch.uint8, self.o_fp4.sf_torch_dtype)))
        for nm, rows, k, block, dtypes in blobs:
            d = self._descs[nm]
            if d.dtype not in dtypes:
                want = " or ".join(str(x) for x in dtypes)
                raise ValueError(f"sample_{nm} must be {want} (scale bytes in F8_128x4 order), got {d.dtype}")
            numel = 1
            for x in d.shape:
                numel *= int(x)
            need = sf_blob_bytes(rows, k, block)
            if numel != need:
                raise ValueError(
                    f"sample_{nm} has {numel} bytes; the PADDED F8_128x4 blob over {rows} rows x K={k} at block {block} is {need} "
                    f"(ceil(rows/128)*128 x ceil(K/{block}/4)*4 -- kernels.proj_gemm.sf_blob_bytes; pad rows / blocks 0x00)"
                )
            # Contiguity from the descriptor strides (an opaque blob bound by storage order).
            expect, ok = 1, True
            for size, stride in zip(reversed(d.shape), reversed(d.stride)):
                if int(size) != 1 and int(stride) != expect:
                    ok = False
                expect *= int(size)
            if not ok:
                raise ValueError(f"sample_{nm} must be contiguous, got shape {tuple(d.shape)} strides {tuple(d.stride)}")

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
            fp8_fused=self.quant_fused,
            mxfp8=self.mxfp8,
            o_fp4=self.o_fp4,
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
        self._quant_dev = self._make_quant_dev()

    def _make_quant_dev(self) -> Optional[dict]:
        """The quant spec's scalars as 1-element fp32 device tensors (plan-time constants, contract § 10;
        the execute path never allocates).  ``None`` for bf16; ``{}`` under ``o_fp4`` -- neither ``alpha_o``
        nor ``scale_o`` exists there (both pinned 1.0: the fp4 out_proj has no alpha epilogue and the fp4
        quantizer takes no scale)."""
        if self.o_fp4 is not None:
            return {}
        if self.mxfp8:
            # MXFP8: only the out projection's per-tensor pair survives (D1) --
            # `alpha_o` for the GEMM epilogue, `scale_o` for the unfused quantize_o.
            q = self.quant
            return dict(
                alpha_o=torch.full((1,), float(q.alpha_o), dtype=torch.float32, device=self.device),
                scale_o=torch.full((1,), float(q.scale_o), dtype=torch.float32, device=self.device),
            )
        if self.quant is not None:
            # Plan-time constants (contract § 10 allows compile-time buffers; the
            # execute path never allocates): one fp32 device scalar per scale.
            q = self.quant

            def _dev(v: float) -> torch.Tensor:
                return torch.full((1,), float(v), dtype=torch.float32, device=self.device)

            return dict(
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
        return None

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        h: torch.Tensor,
        w_qkvg: torch.Tensor,
        w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        saved: Optional[SavedForBackward] = None,
        current_stream: Optional[cuda.CUstream] = None,
        h_sf: Optional[torch.Tensor] = None,  # MXFP8 only (both REQUIRED): the F8_128x4 E8M0 blobs of h and W_qkvg (sample_* byte counts)
        w_qkvg_sf: Optional[torch.Tensor] = None,
        w_o_sf: Optional[torch.Tensor] = None,  # fp4 O only (REQUIRED there): the F8_128x4 blob of the e2m1 W_o (sample_w_o_sf's byte count)
    ) -> None:
        """Launch the five stages in pipeline order.

        No allocation, no D2H read, no implicit conversion: every intermediate is
        a strided VIEW of the caller's workspace.

        ``w_q_norm`` / ``w_k_norm`` must agree with ``geometry.qk_norm`` in both
        directions (typed ``ValueError``), and so must ``saved.rstd_q`` /
        ``saved.rstd_k`` under ``save_for_backward`` (tensors iff qk_norm).

        ``h_sf`` / ``w_qkvg_sf`` (appended): REQUIRED under MXFP8 (both, checked
        for dtype / byte count / contiguity / 16-B alignment, typed), REFUSED
        otherwise.  ``w_o_sf`` (appended): REQUIRED under ``MxQuantSpec.o_fp4``
        (same checks at the format's block and scale dtype), REFUSED otherwise.
        """
        if self._ws is None:
            raise RuntimeError("call compile() before execute()")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        if self.mxfp8:
            if h_sf is None or w_qkvg_sf is None:
                raise ValueError("MXFP8 execute needs both scale-factor blobs: h_sf (over B*S rows) and w_qkvg_sf (over n_qkvg rows); no silent unit scale")
            _check_sf_blob(h_sf, "h_sf", self.batch * self.seq_len, self.geom.d_model)
            _check_sf_blob(w_qkvg_sf, "w_qkvg_sf", self.geom.n_qkvg, self.geom.d_model)
            for nm, sf in (("h_sf", h_sf), ("w_qkvg_sf", w_qkvg_sf)):
                if sf.device != h.device:
                    raise ValueError(f"{nm} must live on h's device {h.device}, got {sf.device}")
        elif h_sf is not None or w_qkvg_sf is not None:
            raise ValueError("h_sf / w_qkvg_sf are the MXFP8 scale-factor blobs; this block was declared without an MxQuantSpec")
        if self.o_fp4 is not None:
            if w_o_sf is None:
                raise ValueError(
                    f"fp4 O ({self.o_fp4.name}) execute needs w_o_sf (the F8_128x4 scale blob of the e2m1 W_o, over d_model rows x K=h_q*d_head); "
                    "no silent unit scale"
                )
            _check_sf_blob(
                w_o_sf,
                "w_o_sf",
                self.geom.d_model,
                self.geom.h_q * self.geom.d_head,
                block=self.o_fp4.block_size,
                sf_dtypes=(torch.uint8, self.o_fp4.sf_torch_dtype),
            )
            if w_o_sf.device != h.device:
                raise ValueError(f"w_o_sf must live on h's device {h.device}, got {w_o_sf.device}")
        elif w_o_sf is not None:
            raise ValueError("w_o_sf is the e2m1 W_o's scale blob of the fp4 O mode; this block was declared without MxQuantSpec.o_fp4")
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
        mxfp8 = self.mxfp8
        act = self.act_dtype
        if self.fp8_fused:
            self._execute_fp8_fused(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream)
            return
        if self.mxfp8_fused:
            self._execute_mxfp8_fused(h, h_sf, w_qkvg, w_qkvg_sf, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream, w_o_sf=w_o_sf)
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
        sfq = sfk = sfv = None
        if fp8:
            e4 = torch.float8_e4m3fn
            q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
            k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
            v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
            o8 = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4) if ws.o8 >= 0 else None  # not reserved under o_fp4
            qd = self._quant_dev
        o4 = sfo = None
        if self.o_fp4 is not None:
            o4, sfo = self._fp4_o_views(workspace, ws, t)
        if mxfp8:
            # The SDPA's own F8_128x4 SF blobs (flat uint8), written by the three quantize stages.
            sfq = _view(workspace, ws.sf_q, (_sf_slot_bytes(self.batch, g.h_q, self.seq_len, g.d_head),), torch.uint8)
            sfk = _view(workspace, ws.sf_k, (_sf_slot_bytes(self.batch, g.h_kv, self.seq_len, g.d_head),), torch.uint8)
            sfv = _view(workspace, ws.sf_v, (_sf_slot_bytes(self.batch, g.h_kv, self.seq_len, g.d_head),), torch.uint8)

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
            if g.qk_norm:
                if saved.rstd_q is None or saved.rstd_k is None:
                    raise ValueError("save_for_backward=True with geometry.qk_norm=True needs SavedForBackward.rstd_q and rstd_k tensors to write through")
            elif saved.rstd_q is not None or saved.rstd_k is not None:
                raise ValueError(
                    "geometry.qk_norm=False (RoPE-only) computes no RMSNorm and writes no rstd: SavedForBackward.rstd_q and rstd_k must be None "
                    "(stage B6 does not exist)"
                )
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
            if mxfp8:
                # (1) [MXFP8: e4m3 codes x e4m3 codes with the E8M0 block scales
                # dequantized IN the MMA (block_scale_dequantize) -> bf16 slab; no alpha]
                self._proj.execute(h.view(t, g.d_model), w_qkvg, proj, engine_ws, sf_a=h_sf, sf_w=w_qkvg_sf, stream=stream)
            else:
                # (1) [FP8: e4m3 x e4m3, epilogue * alpha_qkvg, bf16 slab]
                self._proj.execute(h.view(t, g.d_model), w_qkvg, proj, engine_ws, alpha=qd["alpha_qkvg"] if fp8 else None, stream=stream)
            # (2)+(3) -- on EVERY unfused pipeline (bf16 / FP8 / MXFP8; the S5 bring-up
            # bisect caught an MXFP8 arm that skipped this call: GEMM exact, SDPA
            # exact, end-to-end cos 0.73 -- the normed-Q/K comparison is the tell).
            # q_out/k_out=None means IN PLACE, which is the stage's own
            # default and is safe by construction: every lane holds its whole [D]
            # row in registers before it stores, and no lane touches another's.
            self._norm_rope.execute(
                q_src, k_src, w_q_norm, w_k_norm, cos, sin, q_out=q_c, k_out=k_c, rstd_q=rstd_q, rstd_k=rstd_k, current_stream=stream, flat=True
            )
        if mxfp8:
            # (3q) bf16 slab slices -> compact e4m3 + the SDPA's F8_128x4 SF blobs:
            # Q / K ROWWISE (blocks along D), V COLUMNWISE (blocks along S,
            # D-plane-major SF).  This IS the compaction the MXFP8 SDPA needs.
            self._quant_q.execute(q_src, q8, sfq, batch=self.batch, seq_len=self.seq_len, current_stream=stream)
            self._quant_k.execute(k_src, k8, sfk, batch=self.batch, seq_len=self.seq_len, current_stream=stream)
            self._quant_v.execute(v_src, v8, sfv, batch=self.batch, seq_len=self.seq_len, current_stream=stream)
            q_c, k_c, v_c = q8, k8, v8
        elif fp8:
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
            # per-tensor FP8: scalar descales; MXFP8: the three SF blobs (and NO scalars -- _Sdpa refuses them)
            descale_q=qd["descale_q"] if (fp8 and not mxfp8) else None,
            descale_k=qd["descale_k"] if (fp8 and not mxfp8) else None,
            descale_v=qd["descale_v"] if (fp8 and not mxfp8) else None,
            sf_q=sfq,
            sf_k=sfk,
            sf_v=sfv,
        )
        if not self.fuse_gate:
            # (5) -- gates the SUBSTITUTED O: the SDPA epilogue already selected
            # O := 0 on dead rows, so no residue reaches the sigmoid.
            self._gate.execute(o, gate_src, o, current_stream=stream)
        if self.o_fp4 is not None:
            # (5q') bf16 gated O -> e2m1 codes + the out_proj GEMM's F8_128x4 blob (block scales, no per-tensor
            # scale); (6') fp4 x fp4 block-scale out projection, both blobs dequantized IN the MMA (no alpha).
            self._quant_o.execute(o, o4, sfo, current_stream=stream)
            self._out_proj.execute(o4, w_o, out.view(t, g.d_model), engine_ws, sf_a=sfo, sf_w=w_o_sf, stream=stream)
        elif fp8:
            # (5q) bf16 gated O -> e4m3 for the FP8 out projection (PER-TENSOR
            # scale_o under both families, D1); (6) folds (1/scale_o) *
            # descale_w_o into its epilogue.
            self._quant_o.execute(o, o8, qd["scale_o"], current_stream=stream)
            self._out_proj.execute(o8.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)
        else:
            # (6)
            self._out_proj.execute(o.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, stream=stream)

    def _fp4_o_views(self, workspace: torch.Tensor, ws: "_Intermediates", t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two fp4-O workspace views: ``o4`` -- the packed e2m1 gated O as ``float4_e2m1fn_x2 [T, H_q*D/2]``
        (a re-view of the uint8 bytes, no copy; what ``run_proj_gemm`` binds as the fp4 A operand) -- and ``sfo``, the
        flat uint8 blob of ``proj_gemm.sf_blob_bytes(T, H_q*D, block)`` bytes the quantize stage writes and the
        block-scale GEMM reads."""
        from .kernels.proj_gemm import sf_blob_bytes

        g = self.geom
        o4 = _view(workspace, ws.o4, (t, g.h_q * g.d_head // 2), torch.uint8).view(_FP4_X2)
        sfo = _view(workspace, ws.sf_o, (sf_blob_bytes(t, g.h_q * g.d_head, self.o_fp4.block_size),), torch.uint8)
        return o4, sfo

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

        ``w_q_norm`` / ``w_k_norm`` are ``None`` (both) under ``geometry.qk_norm=False``
        -- already checked by ``execute``; the fork's runner receives them as-is.
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

    def _execute_mxfp8_fused(self, h, h_sf, w_qkvg, w_qkvg_sf, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream, w_o_sf=None) -> None:
        """The FULLY FUSED MXFP8 pipeline: three launches, eight workspace slots.

        ::

            (1'') proj+norm+rope+block-quant  h8+sf_h, W8+sf_w      -> q8 / k8 / v8 e4m3 (COMPACT) + sf_q / sf_k / sf_v (F8_128x4) + gate16 bf16
            (4'') sdpa+gate                   q8, k8, v8, SF, gate16 -> o8 e4m3 UNSCALED (block scales dequant in-MMA; gate after the select)
            (6)   out_proj                    o8                    -> out   (alpha_o = descale_w_o / 1.0)

        Mirrors :meth:`_execute_fp8_fused` with the three SF slots added: the
        fork's runner (``run_fused_proj_gemm_mxfp8``, frozen ABI) writes the
        SDPA's own SF layouts (Q/K per-(b, h, s_tile) 1024-B tiles, V
        D-plane-major), so the gated production MXFP8 SDPA reads exactly what
        the unfused pipeline's quantize stages would have written.  No scalar
        rides into the SDPA (``_Sdpa`` refuses ``descale_*`` / ``scale_o`` under
        MXFP8); ``MxQuantSpec.scale_o == 1.0`` was pinned at declaration (D8).

        ``w_o_sf`` (appended) -- fp4 O (config row 10, FOUR launches): the gated SDPA writes **bf16** ``o``
        (``o8`` is not reserved), ``quantize_fp4_o`` turns it into ``o4`` + ``sf_o``, and the out projection
        is the fp4 x fp4 block-scale GEMM over ``sf_o`` / ``w_o_sf`` (no alpha).
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
        # e4m3 O for the per-tensor out_proj, or (o_fp4) bf16 O for the quantize_fp4 stage -- one slot exists, never both.
        o_sdpa = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4) if self.o_fp4 is None else _view(workspace, ws.o, (t, g.h_q, g.d_head), self.act_dtype)
        sfq = _view(workspace, ws.sf_q, (_sf_slot_bytes(b, g.h_q, s, g.d_head),), torch.uint8)
        sfk = _view(workspace, ws.sf_k, (_sf_slot_bytes(b, g.h_kv, s, g.d_head),), torch.uint8)
        sfv = _view(workspace, ws.sf_v, (_sf_slot_bytes(b, g.h_kv, s, g.d_head),), torch.uint8)
        engine_ws = workspace[ws.engine_scratch :]
        # (1''): the runner takes the four data outputs 2-D ([T, h*d]) + the three SF blobs flat.
        self._proj.execute_mxfp8(
            h.view(t, g.d_model),
            h_sf,
            w_qkvg,
            w_qkvg_sf,
            q8.view(t, g.h_q * g.d_head),
            k8.view(t, g.h_kv * g.d_head),
            v8.view(t, g.h_kv * g.d_head),
            gate16.view(t, g.h_q * g.d_head),
            sfq,
            sfk,
            sfv,
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            stream=stream,
        )
        # (4''): e4m3 in (+ block scales), e4m3 out UNSCALED (or bf16 out under o_fp4); the SDPA gates the
        # SUBSTITUTED O (after the dead-row select, per element) and casts once.
        self._sdpa.execute(
            q8.view(b, s, g.h_q, g.d_head),
            k8.view(b, s, g.h_kv, g.d_head),
            v8.view(b, s, g.h_kv, g.d_head),
            o_sdpa.view(b, s, g.h_q, g.d_head),
            lse=lse,
            seq_lens=seq_lens,
            workspace=engine_ws,
            current_stream=cuda.CUstream(stream),
            gate=gate16.view(b, s, g.h_q, g.d_head),
            sf_q=sfq,
            sf_k=sfk,
            sf_v=sfv,
        )
        if self.o_fp4 is not None:
            # (5q') bf16 O_gated -> e2m1 codes + the out_proj blob; (6') fp4 x fp4 block-scale out projection (no alpha).
            o4, sfo = self._fp4_o_views(workspace, ws, t)
            self._quant_o.execute(o_sdpa, o4, sfo, current_stream=stream)
            self._out_proj.execute(o4, w_o, out.view(t, g.d_model), engine_ws, sf_a=sfo, sf_w=w_o_sf, stream=stream)
        else:
            # (6): e4m3 O_gated @ W_o^T with descale_w_o (scale_o == 1.0) in the epilogue.
            self._out_proj.execute(o_sdpa.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)


# ---------------------------------------------------------------------------
# 7. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------


def gated_attention_block_forward(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
    w_k_norm: Optional[torch.Tensor],
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
