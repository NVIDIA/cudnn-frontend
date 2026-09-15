# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, backward — SKELETON (no kernel is wired yet).

Read :mod:`cudnn.gated_attention_block.api` first. This module is defined
against that file's :class:`~cudnn.gated_attention_block.api.SavedForBackward`
contract and its ``qkvg_offsets`` ordering; neither is restated here.

.. note::

   **MAINTENANCE.** Same rule as the forward: this docstring and the per-stage
   **Fusion status** paragraphs are the design record in the code. Update them
   in the same commit as any stage becoming real, any fusion, any dtype.

The op graph, in pipeline order::

    dY [B, S, d_model]
     |
     +-- (B1) out_proj wgrad    dW_o = O_gated^T @ dY          K = B*S
     |        INDEPENDENT of everything below -- see "the filler" note
     |
     +-- (B2) out_proj dgrad    dO_gated = dY @ W_o
              |
              |  (B3) gate backward, elementwise, s = sigmoid(GATE):
              |         dO = dO_gated * s
              |         dG = dO_gated * O * s * (1 - s)
              |
              |  (B4) SDPA backward  (dO, Q, K, V, O, LSE) -> dQ, dK, dV
              |
              |  (B5) RoPE^T   inverse rotation on the first ROPE_DIM of dQ, dK
              |  (B6) RMSNorm backward, per head over D
              |         -> dQ_pre, dK_pre, dW_q_norm, dW_k_norm
              |
              |  concat(dQ_pre | dG | dK_pre | dV) = dQKVG   (qkvg_offsets order)
              |
     +-- (B7) qkv+gate wgrad    dW_qkvg = h^T @ dQKVG         K = B*S
     +-- (B8) qkv+gate dgrad    dh      = dQKVG @ W_qkvg
              v
             dh [B, S, d_model]

**Six GEMMs and two attention kernels under one API.** That is roughly three
times the forward, and it is the reason the forward's saved-tensor contract was
pinned before any of it was written.

Where the fusion opportunity lives — it MOVES to the other end
--------------------------------------------------------------

The forward's opportunity was the gate GEMM hiding in the softmax shadow. The
backward's is different, and one of the two obvious candidates is a trap:

* **``dW_o`` (B1) is unambiguous independent filler.** It needs only
  ``O_gated`` (recomputable elementwise from saved ``o`` and ``gate``) and
  ``dY`` (available at entry). It depends on no part of the SDPA backward and
  can run concurrently with all of it. Large, too: ``M=8192, N=4096, K=B*S``.
  This is the backward's analogue of the forward's gate GEMM.
* **The ``o_proj`` DGRAD (B2) is per-head independent** — it contracts over
  ``d_model``, not over heads — so structurally it is the OPPOSITE of the
  forward's stage-(6) constraint and it IS legal to fuse as an SDPA-backward
  prologue. **Do not cash that in.** ``dO`` is consumed by both backward stages
  with DIFFERENT tilings (dK/dV loops q-tiles for a fixed kv-tile; dQ loops
  kv-tiles for a fixed q-tile), so it gets materialized once regardless and the
  prologue fusion saves nothing on the dK/dV side. Legal is not worth it. Filed
  as available to a later partitioning.
* **(B3) folds into the SDPA-backward prologue cheaply** — ``dO`` is per-element
  on a tensor the backward already reads. Same character as the forward's
  stage-(5) increment, and it should land the same way.
* **The two wgrads (B1, B7) contract over TOKENS** (``K = B*S``), not over a
  feature dim. Different tiling from everything else in the block; the classic
  split-K case. They reuse nothing from the dgrad path.

Determinism
-----------

Any split-K reduction in B1 / B7, and any cross-rank merge, reduces in a FIXED
order — never arrival order. Run-to-run bit differences in a gradient are a
support burden out of all proportion to the perf they buy. This is a contract
decision, not an implementation detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.frost.workspace import WorkspaceLayout

from .api import GatedAttentionBlockGeometry, SavedForBackward, _Stage

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Recompute policy — the knob a decomposed graph cannot offer
# ---------------------------------------------------------------------------


class RecomputePolicy(Enum):
    """How much of the forward the backward re-runs instead of reading.

    The forward's ``SavedForBackward`` says which tensors EXIST; this says what
    to do about the ones that do not. Both halves of the trade are real at
    scale, which is the whole argument for owning them at the block level.

    ``SAVE_ALL``
        Nothing recomputed. ``q_pre`` / ``k_pre`` came out of the forward
        (+17 GiB at 1M tokens). Fastest backward, largest footprint.

    ``RECOMPUTE_QK_PRE``
        Re-run the Q and K column slices of the forward's stage-(1) GEMM from
        the saved ``h`` to rebuild the pre-norm operands. Costs a partial GEMM;
        saves the 17 GiB. **Expected default.**

    ``RECOMPUTE_GATE``
        Additionally drop ``gate`` from the save set and re-run its slice too
        (another 16 GiB saved). Only sensible together with the above, since it
        is the same GEMM — at which point it is a full stage-(1) recompute.

    Whatever is recomputed must be recomputed BIT-IDENTICALLY to the forward, or
    gradients acquire a noise floor that looks like a kernel bug. Same tile
    config, same accumulation order, same epilogue rounding.
    """

    SAVE_ALL = 0
    RECOMPUTE_QK_PRE = 1
    RECOMPUTE_GATE = 2


# ---------------------------------------------------------------------------
# 2. Workspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BwdIntermediates:
    """Byte offsets into the caller's workspace for the backward's scratch.

    ``dqkvg`` is one buffer in ``qkvg_offsets`` order so B7/B8 each see a single
    contiguous operand — the whole reason the forward's concat ordering is
    contract rather than convenience.

    ``-1`` means the producing stage was fused away.
    """

    do_gated: int  # [B, S, H_q,  D]
    do: int  # [B, S, H_q,  D]   gated dO, the SDPA-bwd input
    dqkvg: int  # [B, S, N]         holds dQ | dG | dK | dV, then dQ_pre | dG | dK_pre | dV
    o_gated: int  # [B, S, H_q,  D]   recomputed for B1, or -1 if B1 reads it fused
    sdpa_bwd_ws: int  # whatever the reused SDPA backward asks for (it has its own carver)
    recompute: int  # staging for RecomputePolicy re-runs, or -1

    total_bytes: int
    base_align: int


def _plan_bwd_workspace(geom: GatedAttentionBlockGeometry, b: int, s: int, dtype: torch.dtype, policy: RecomputePolicy) -> _BwdIntermediates:
    """Reserve every backward intermediate and report the total.

    Note ``dqkvg`` is written in TWO passes over the same memory: B4 lands
    dQ/dK/dV in it, then B5/B6 rewrite the Q and K slices in place into
    dQ_pre/dK_pre while dG (from B3) and dV pass through untouched. Sizing it
    once as ``[B, S, N]`` is what makes that in-place walk legal.
    """
    layout = WorkspaceLayout()
    del geom, b, s, dtype, policy, layout
    raise NotImplementedError("_plan_bwd_workspace")


# ---------------------------------------------------------------------------
# 3. Stages — one FROST kernel each, in pipeline order
# ---------------------------------------------------------------------------


class _OutProjWgrad(_Stage):
    """(B1) ``dW_o = O_gated^T @ dY``. Contracts over TOKENS: ``K = B*S``.

    ``O_gated`` is not saved — recompute it elementwise as
    ``o * sigmoid(gate)`` from tensors the backward already holds, either in a
    prologue or fused into this GEMM's operand load.

    **Fusion status: this is the backward's filler.** It depends on nothing
    below it, so it is the work to overlap the SDPA backward with. Getting that
    overlap is a scheduling question (stream order, or a persistent kernel that
    owns both), not a kernel-fusion one.

    Split-K over tokens, reduced in fixed order (see module docstring).

    Kernel: ``kernels/<arch>/out_proj_wgrad_*.py``.
    """

    name = "out_proj_wgrad"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _OutProjDgrad(_Stage):
    """(B2) ``dO_gated = dY @ W_o``, contracting over ``d_model``.

    **Fusion status: per-head independent, therefore legal as an SDPA-backward
    prologue — and deliberately NOT taken.** See the module docstring: ``dO`` is
    read with two different tilings, so it is materialized either way.

    Kernel: ``kernels/<arch>/out_proj_dgrad_*.py``.
    """

    name = "out_proj_dgrad"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _SigmoidGateBwd(_Stage):
    """(B3) gate backward, elementwise over ``[B, S, H_q, D]``::

        s  = sigmoid(GATE)
        dO = dO_gated * s
        dG = dO_gated * O * s * (1 - s)

    Two outputs, one pass, three input reads. Compute ``s`` in fp32 regardless
    of storage dtype; ``s * (1 - s)`` in bf16 loses bits exactly where the gate
    saturates and the gradient matters least — but it is free to do right.

    ``dG`` is written straight into the ``dqkvg`` scratch at the GATE offset and
    never touched again: B5 and B6 walk only the Q and K slices.

    **Hazard, mirroring the forward's stage (5):** ``O`` here is the SDPA's
    *substituted* output. For a dead row it is an exact zero, which correctly
    makes ``dG`` zero; if a future fused epilogue ever hands this stage
    pre-substitution residue instead, ``dG`` becomes NaN for rows whose forward
    output was fine.

    **Fusion status:** folds into the SDPA-backward prologue cheaply — it is
    per-element on ``dO``, which that kernel already reads.

    Kernel: ``kernels/<arch>/sigmoid_gate_bwd_*.py``.
    """

    name = "sigmoid_gate_bwd"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _SdpaBwd(_Stage):
    """(B4) ``(dO, Q, K, V, O, LSE) -> dQ, dK, dV``, GQA-reduced over H_q/H_kv.

    **Writes no new kernel**: drives the shipped FROST backward under
    ``cudnn/sdpa/bwd/``, which is already a multi-kernel staged pipeline behind
    one API with a caller-provided workspace. A block backward extends that
    pattern; it does not invent it.

    Consumes ``Q`` and ``K`` **post-norm, post-RoPE** — the same tensors the
    forward's SDPA saw, not the pre-norm ones. Under
    ``RecomputePolicy.SAVE_ALL`` they are separate saved tensors from
    ``q_pre`` / ``k_pre``; under a recompute policy they are rebuilt by re-running
    stages (1)-(3), which is why the recompute must be bit-identical.

    ``LSE`` is mandatory here. That is the whole reason
    ``save_for_backward`` implies ``return_lse`` in the forward.

    Kernel: reused, ``sdpa/bwd/kernels/``.
    """

    name = "sdpa_bwd"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _PartialRopeBwd(_Stage):
    """(B5) inverse rotation on the first ``ROPE_DIM`` dims of dQ and dK.

    RoPE is orthogonal, so the backward is the transpose — the same table with
    the sine negated::

        dx_rot = dy_rot * cos - rotate_half(dy_rot) * sin

    Dims ``[ROPE_DIM, D)`` pass through, in place, exactly as in the forward.
    Same ``cos`` / ``sin`` tables the forward was given; the caller re-supplies
    them rather than the block saving a copy (they are ~128 KiB and rebuilding
    them is a caller concern anyway).

    **Fusion status:** merges with B6 for the same reason (2) and (3) merge in
    the forward — one pass over dQ, one over dK.

    Kernel: ``kernels/<arch>/qk_norm_rope_bwd_*.py`` (shared with B6).
    """

    name = "rope_bwd"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _QkRmsNormBwd(_Stage):
    """(B6) per-head RMSNorm backward over D, on dQ and dK. **Not on dV.**

    Two kinds of output, and they want different reductions::

        dx = (rstd / D) * (D * dy * w - x_hat * sum_j(dy_j * w_j * x_hat_j))
        dW = sum over ALL tokens and ALL heads of (dy * x_hat)

    ``dx`` is a per-row reduction over D — trivially parallel, one CTA per
    (token, head). ``dW_q_norm`` / ``dW_k_norm`` are ``[D]`` accumulated over
    ``B*S*H`` rows, i.e. a global reduction: a deterministic tree or a fixed-order
    split-K, never atomics (module docstring, "Determinism").

    Needs ``x`` (pre-norm Q/K) and ``rstd``. ``rstd`` is always saved — it is
    134 MiB. ``x`` is the :class:`RecomputePolicy` decision. Do **not** try to
    recover ``x`` by dividing the normed value by ``w``: undefined at a zero
    norm weight, hostile at a small one.

    Writes dQ_pre / dK_pre in place over the Q and K slices of ``dqkvg``.

    Kernel: ``kernels/<arch>/qk_norm_rope_bwd_*.py`` (shared with B5).
    """

    name = "qk_norm_bwd"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _QkvGateWgrad(_Stage):
    """(B7) ``dW_qkvg = h^T @ dQKVG``. Contracts over TOKENS: ``K = B*S``.

    Emits the weight gradient in the exact ``qkvg_offsets`` layout the forward
    consumes, so a caller never re-slices. At 397B: ``17408 x 4096``.

    Split-K over tokens, fixed-order reduction. Same tiling family as B1 and
    nothing else in the block.

    Kernel: ``kernels/<arch>/qkv_gate_wgrad_*.py``.
    """

    name = "qkv_gate_wgrad"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


class _QkvGateDgrad(_Stage):
    """(B8) ``dh = dQKVG @ W_qkvg``, contracting over N.

    The block's output gradient. Nothing downstream of it here.

    Kernel: ``kernels/<arch>/qkv_gate_dgrad_*.py``.
    """

    name = "qkv_gate_dgrad"

    def check_support(self) -> None:
        raise NotImplementedError(self.name)

    def compile(self) -> None:
        raise NotImplementedError(self.name)

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError(self.name)


# ---------------------------------------------------------------------------
# 4. The public API
# ---------------------------------------------------------------------------


class GatedAttentionBlockBwd(APIBase):
    """Gated attention block, backward. One call, one workspace, eight stages.

    Built against a forward that ran with ``save_for_backward=True``. It does
    not re-derive the geometry: pass the same
    :class:`~cudnn.gated_attention_block.api.GatedAttentionBlockGeometry`
    instance, and the constructor checks the saved tensors against it rather
    than trusting either alone.
    """

    def __init__(
        self,
        sample_dy: torch.Tensor,  # [B, S, d_model]
        sample_saved: SavedForBackward,
        sample_w_qkvg: torch.Tensor,  # [N, d_model]
        sample_w_q_norm: torch.Tensor,  # [D]
        sample_w_k_norm: torch.Tensor,  # [D]
        sample_cos: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_sin: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_w_o: torch.Tensor,  # [d_model, H_q * D]
        geometry: GatedAttentionBlockGeometry,
        *,
        recompute: RecomputePolicy = RecomputePolicy.RECOMPUTE_QK_PRE,
        # Which input gradients the caller actually wants. A partial request
        # skips whole GEMMs (dh alone needs neither wgrad), so this is a real
        # scheduling input, not a convenience -- and it must be fixed at build
        # time, because it changes which artifacts exist.
        need_dh: bool = True,
        need_dw_qkvg: bool = True,
        need_dw_o: bool = True,
        need_dw_norms: bool = True,
    ):
        super().__init__()
        self._warn_experimental_api()
        raise NotImplementedError("GatedAttentionBlockBwd.__init__")

    # -- support ------------------------------------------------------------

    def check_support(self) -> bool:
        """Validate the saved set against the geometry and the policy, then ask
        every enabled stage.

        Declines that belong here: a ``SavedForBackward`` missing a tensor the
        chosen :class:`RecomputePolicy` does not rebuild; ``lse`` absent (the
        forward ran inference-only); a ``need_*`` combination that leaves no
        work; a dtype outside the precision roadmap.
        """
        raise NotImplementedError("GatedAttentionBlockBwd.check_support")

    # -- workspace ----------------------------------------------------------

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide, INCLUDING whatever the reused SDPA
        backward reports — it carves from the same buffer, at an offset this
        layout reserves for it."""
        raise NotImplementedError("GatedAttentionBlockBwd.get_workspace_size")

    # -- compile ------------------------------------------------------------

    def compile(self) -> None:
        """Build the artifacts for the enabled stages only."""
        self._ensure_support_checked()
        raise NotImplementedError("GatedAttentionBlockBwd.compile")

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        dy: torch.Tensor,
        saved: SavedForBackward,
        w_qkvg: torch.Tensor,
        w_q_norm: torch.Tensor,
        w_k_norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        dh: Optional[torch.Tensor] = None,
        dw_qkvg: Optional[torch.Tensor] = None,
        dw_o: Optional[torch.Tensor] = None,
        dw_q_norm: Optional[torch.Tensor] = None,
        dw_k_norm: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the enabled stages onto one stream.

        Order, with B1 deliberately first so it is in flight across everything
        that follows (module docstring, "the filler")::

            (B1) out_proj_wgrad    saved.o, saved.gate, dy   -> dw_o
            (B2) out_proj_dgrad    dy, w_o                   -> ws.do_gated
            (B3) sigmoid_gate_bwd  ws.do_gated, saved.o,
                                   saved.gate                -> ws.do, ws.dqkvg[GATE]
            (B4) sdpa_bwd          ws.do, Q, K, V,
                                   saved.o, saved.lse        -> ws.dqkvg[Q,K,V]
            (B5) rope_bwd          ws.dqkvg[Q,K]             -> in place
            (B6) qk_norm_bwd       ws.dqkvg[Q,K], rstd,
                                   q_pre, k_pre              -> in place, + dw_*_norm
            (B7) qkv_gate_wgrad    saved.h, ws.dqkvg         -> dw_qkvg
            (B8) qkv_gate_dgrad    ws.dqkvg, w_qkvg          -> dh

        A ``need_*`` that was False at build time means the corresponding output
        argument must be ``None`` here — a provided-but-uncompiled tensor raises
        rather than being silently ignored (Rule 1, both directions).

        No allocation, no D2H read, no implicit conversion.
        """
        raise NotImplementedError("GatedAttentionBlockBwd.execute")


# ---------------------------------------------------------------------------
# 5. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------


def gated_attention_block_backward(
    dy: torch.Tensor,
    saved: SavedForBackward,
    w_qkvg: torch.Tensor,
    w_q_norm: torch.Tensor,
    w_k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    recompute: RecomputePolicy = RecomputePolicy.RECOMPUTE_QK_PRE,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate gradients + workspace, cache the compiled block, and run it.

    Returns ``{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}``. Which
    entries are non-``None`` follows ``requires_grad`` on the corresponding
    forward inputs, snapshotted here — the same policy a torch autograd
    ``Function`` would apply, and the natural place to wire one later.
    """
    raise NotImplementedError("gated_attention_block_backward")
