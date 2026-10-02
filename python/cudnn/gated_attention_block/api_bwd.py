# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, backward -- bf16 / fp16 over a proj_slab record: the unfused assembly plus its first fused step.

Read :mod:`cudnn.gated_attention_block.api` first. This module is defined
against that file's :class:`~cudnn.gated_attention_block.api.SavedForBackward`
contract and its ``qkvg_offsets`` ordering; neither is restated here.

.. note::

   **MAINTENANCE.** Same rule as the forward: this docstring, the launch and
   workspace tables below and the per-stage **Fusion status** paragraphs are
   the as-built record of this backward. Update them in the same commit as any stage
   changing, any fusion, any dtype; the measured numbers behind the tolerances and
   the launch counts live in ``test_block_backward.py``'s module docstring.

The op graph, in LAUNCH order -- one stream, nothing overlaps anything::

    dY [B, S, d_model]
     |
     +-- (B2) out_proj dgrad     dO_gated = dY @ W_o                          -> ws.do_gated [T, H_q, D]
     |   (B3) gate backward, elementwise, s = sigmoid(GATE):
     |          dO      = dO_gated * s                                        -> IN PLACE over ws.do_gated
     |          dG      = dO_gated * O * s * (1 - s)                          -> ws.dqkvg [GATE band]
     |          O_gated = O * s                        (need_dw_o only)       -> ws.o_gated
     +-- (B1) out_proj wgrad     dW_o = dY^T @ O_gated             K = B*S    (AFTER B3: it takes O_gated from B3's third output)
     |
     |   recompute  Q, K = norm + RoPE(q_pre, k_pre)  -- the forward's stage (2)+(3) kernel over the proj_slab bands
     |              V    = compact copy of the V band                         -> ws.recompute / recompute_k / recompute_v
     |   (B4) SDPA backward   (dO, Q, K, V, O, LSE) -> dQ, dK, dV  COMPACT    -> ws.dq / ws.dk / ws.dv
     |   (B5) RoPE^T + (B6) RMSNorm backward -- ONE kernel over dQ and dK (+ the bit-exact dV copy):
     |          ws.dqkvg [Q band] = dQ_pre, [K band] = dK_pre, [V band] = dV; fp32 dW_norm partials per CTA
     |        + the fixed-order reduce of the partials                        -> dW_q_norm, dW_k_norm   (qk_norm only)
     |
     +-- (B7) qkv+gate wgrad     dW_qkvg = dQKVG^T @ h              K = B*S
     +-- (B8) qkv+gate dgrad     dh      = dQKVG @ W_qkvg
              v
             dh [B, S, d_model]

Orientation follows the forward's ``nn.Linear`` TN form (``api.py``):
``QKVG = h @ W_qkvg^T`` with ``W_qkvg [N, d_model]`` and
``Y = O_gated @ W_o^T`` with ``W_o [d_model, H_q*D]``. For ``Y = X @ W^T`` the
gradients are ``dW = dY^T @ X`` and ``dX = dY @ W`` -- so both wgrads above
land directly in their weight's own ``[out, in]`` layout, and both dgrads
read the weight un-transposed (``kernels/proj_gemm.py::run_wgrad_gemm`` /
``run_dgrad_gemm``: M-major A / N-major B plans, zero-copy views, the declared
strides asserted before every bind).

**Six GEMMs and two attention kernels under one API.** That is roughly three
times the forward, and it is the reason the forward's saved-tensor contract was
pinned before any of it was written.

Two corrections to the first skeleton, both load-bearing:

* **B4 writes COMPACT dQ / dK / dV** (workspace slots), never the ``dqkvg``
  bands: the Rubin backward adapter (``sdpa/bwd/api_dsl_sm107.py``) requires
  every one of its eight io operands to be BSHD-physical -- stride exactly
  ``(S*H*D, D, H*D, 1)`` -- and its main kernel is compiled against compact
  fakes. The elementwise backward kernel (B5+B6) then writes the ``dqkvg``
  bands, reading the compact slots; the ``[T, N]`` slab B7 / B8 consume is
  still written exactly once per column.
* **B1 runs after B3**, taking ``O_gated`` from the gate-backward kernel's
  optional third output instead of a separate ``o * sigmoid(gate)`` launch (two
  reads and one launch saved). It is still the filler: it depends on nothing
  the SDPA backward produces.

**Fusion knob: ``fuse_gate_bwd`` (default False).**  The SDPA backward's first
launch is ``delta = rowsum(dO * O)`` over the very ``dO`` B3 just wrote and the
``O`` it just read; with the knob the gate-backward kernel emits ``delta`` as a
fourth output (``kernels/sigmoid_gate_bwd.py``: the bf16-ROUNDED ``dO`` it
stores, summed in the chain's own ``dot_do_o`` order, the pad rows zeroed) into a
block-owned fp32 ``[B, H_q, S_pad]`` region, and the adapter is built with
``external_delta=True`` and handed that tensor -- one launch and one read each of
``O`` and ``dO`` fewer, the adapter's own ``delta`` region gone from its scratch.
Bitwise the unfused block (same fp32 operations in the same order; pinned by
``test_fused_gate_bwd_is_bitwise_the_unfused_block``), so it is a performance
knob in the Rule-9 sense: the same function under either value.

Launch table (one stream; ``g = h_q / h_kv``, ``c`` = the adapter's head
chunks, ``q`` = the adapter's dQ GEMM launches per chunk -- 1 under its
single-launch dQ rendering (``MatmulTemplateParams.b_head_group = g``, the
shipped default), ``g`` under the per-group-member twin; all ``need_*`` True)::

    #   stage                           launches
    1   B2  run_dgrad_gemm              1
    2   B3  sigmoid_gate_bwd            1            (+ delta = rowsum(dO * O) as a 4th output under fuse_gate_bwd)
    3   B1  run_wgrad_gemm              1            (need_dw_o)
    4   Q/K recompute (_QkNormRope)     1
    5   V compaction (_VCompaction)     1
    6   B4  SdpaBwdDslSm107.execute     3 + c*(2+q)  fill_i32 + dot_do_o + c x [main + dK GEMM + q x dQ GEMM] + dkv_reduce (g > 1)
                                          + 3 at S % 128 != 0 (q / dO / lse pads), + 2 at S % 256 != 0 (k / v pads),
                                          + 2 fold copy-outs when GQA and kv-padded (+1 when MHA and kv-padded);
                                          MHA (g = 1): no dkv_reduce -> 2 + 3c;
                                          fuse_gate_bwd: no dot_do_o (external_delta) -> 2 + c*(2+q)
    7   B5+B6 qk_norm_rope_bwd          1
    8   dW_norm reduce                  1            (need_dw_norms)
    9   B7  run_wgrad_gemm              1            (need_dw_qkvg)
    10  B8  run_dgrad_gemm              1            (need_dh)
                                       ---
                                        12 + c*(2+q)  -- 15 at the test geometry (h_q=8, h_kv=2, q=1; 22 at S=1000), 15 / 18 at 397B (c=1 / 2)
                                        11 + c*(2+q)  under fuse_gate_bwd -- 14 / 21 at the test geometry, 14 / 17 at 397B

The count is CHECKED by CUPTI in the tests (``test_launch_count_is_honest``:
15 / 22 at the test geometry, 14 / 21 with ``fuse_gate_bwd``, MEASURED on Rubin
cc 10.7), never quoted from this formula: the padded / zero-fill / MHA arms
change it, and ``q`` is read off the adapter's dQ record
(``prepared_host._dq_launches``), never assumed.

Workspace table (``WorkspaceLayout(align=256)``, ``e`` = activation bytes,
``T = B*S``; every region is a slot of the CALLER's one uint8 buffer)::

    region            shape             dtype   writer                      reader
    do_gated (= do)   [T, H_q, D]       act     B2; then B3 in place         B3; B4 (as dO)
    dqkvg             [T, N]            act     B3 (GATE), B5+B6 (Q, K, V)   B7, B8
    o_gated           [T, H_q, D]       act     B3 (og)          need_dw_o  B1
    recompute (q)     [T, H_q, D]       act     _QkNormRope                  B4
    recompute_k       [T, H_kv, D]      act     _QkNormRope                  B4
    recompute_v       [T, H_kv, D]      act     _VCompaction                 B4
    dq / dk / dv      compact           act     B4                           B5+B6
    dw_partials_q/k   [n_ctas_x, D]     fp32    B5+B6         need_dw_norms  the reduce   (EXACTLY n_ctas_for(recipe, T) rows)
    sdpa_bwd_ws       opaque            uint8   the adapter's own carver (delta -- unless fuse_gate_bwd --, ONE dS chunk, pads, GQA partials)
    gemm_scratch      opaque            uint8   the FROST GEMM (max(plan.workspace_bytes) over B1 / B2 / B7 / B8, never 0)
    delta             [B, H_q, S_pad]   fp32    B3 (4th output)  fuse_gate_bwd  B4 (external_delta; S_pad = the adapter's external_delta_shape)

The adapter's dS chunk dominates at scale: ``qh_chunk x S_q_pad x S_kv_pad x 2 B``
with ``qh_chunk`` a multiple of the GQA group -- 4.25 / 8.50 / 33.0 GiB at
S = 8K / 16K / 32K for the 397B geometry at B=1 (MEASURED by constructing the
adapter). ``gemm_scratch`` is the max with the backend's OWN heuristic plan's
number, which at the test geometry auto-splits 3-way (12 MiB) although the
forced JIT that runs needs none; 0 at 397B.

One engine, no fallback (AGENTS.md Rule 9, stated as a design decision):
the block binds ``SdpaBwdDslSm107`` directly, exactly as the forward binds its
FROST SDPA class, so it declines every non-Rubin device with a typed message
and never falls back to the cuDNN backend's d=256 backward engine. The graph
route (``manifest.py`` selection) is a later option if a second arch needs it.

P0 limits (all typed, at declaration -- ``check_support``): bf16 / fp16;
Rubin (SM107); ``d_head = 256``; ``seq_len >= 2`` (S_q = 1 is decode, out of
the ``sdpa_bwd_sm107`` prefill bodies' scope); ``d_model % 256 == 0`` (the
forced 256-wide GEMM tile behind the determinism contract below; ``h_q * D``
satisfies it through ``d_head = 256``); ``saved.proj_slab`` present (the
gate-copy record -- V is a slab band with no field of its own -- is a follow-up PR);
no ``seq_lens`` (the ``sdpa_bwd_sm107`` row declines padding; a follow-up PR
flips it) -- declined from ``seq_lens_present``, from the sample record's
``seq_lens`` at declaration AND from every record handed to ``execute``
(the tensor's presence is the fact, never its values); ``window_left > 0``
only, ``window_right`` unbounded (or 0) only; ``dw_norm_dtype = torch.float32``
only.

Determinism
-----------

No atomics on the whole chain, so two executes are bitwise equal (pinned on
Rubin by ``test_two_runs_are_bitwise`` and ``test_a_caller_stream_orders_every_stage``,
workspace poisoned in between, on Rubin cc 10.7): the four GEMMs run the FORCED
``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma`` tile at one split-K
slice (no reducer). That premise is ENFORCED, not assumed: ``check_support``
declines a geometry the forced tile cannot take (``d_model % 256 != 0``;
``kernels/proj_gemm.py::_forced_tile_config``), and ``compile()`` refuses,
typed, a plan that fell back to the heuristic (a different tile, route and
possibly a split-K reducer) -- so a served block never runs an unpinned GEMM.
The bf16 / fp16 Rubin backward chain has no atomic and its GQA fold is
a fixed-order reduction; the elementwise kernels are per element; ``dW_norm``
is per-CTA partials plus a fixed-order reduce launch. Bitwise PER KNOB SET and
per device (the SM-fill CTA count enters the partial order) -- never claimed
across devices. This is a contract decision, not an implementation detail.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import stream_context
from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.workspace import WorkspaceLayout

from .api import (
    _SM107_CC,
    _WS_ALIGN,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    SavedForBackward,
    _bhsd_desc,
    _check_norm_weights_agree,
    _cols,
    _itemsize,
    _QkNormRope,
    _Stage,
    _VCompaction,
    _view,
    saved_slab_views,
)

_ACT_DTYPES = (torch.bfloat16, torch.float16)


# ---------------------------------------------------------------------------
# 1. Recompute policy — the knob a decomposed graph cannot offer
# ---------------------------------------------------------------------------


class RecomputePolicy(Enum):
    """How much of the forward the backward re-runs instead of reading.

    The forward's ``SavedForBackward`` says which tensors EXIST; this says what
    to do about the ones that do not. Both halves of the trade are real at
    scale, which is the whole argument for owning them at the block level.

    Which ``SavedForBackward`` fields each policy READS, and what it REBUILDS
    (the schema is ``api.py``'s; it is append-only and is not widened here):

    ``SAVE_ALL``
        Reads the pre-norm Q / K from the save set. With a ``proj_slab`` record
        (the forward's default save mode, and the ONLY record served today)
        they are its column bands -- as are the GATE and V -- so nothing is
        recomputed at all: the fastest backward, the largest footprint
        (34 KiB/token at the 397B geometry).

    ``RECOMPUTE_QK_PRE``
        "Rebuild what is ``None``". With a ``proj_slab`` record
        nothing is ``None`` and the policy reads the bands exactly like
        ``SAVE_ALL``. With a gate-copy record (``proj_slab=None``, a compact
        ``gate``) it re-runs the Q, K and V column slices of the forward's
        stage-(1) GEMM from ``saved.h`` -- a partial GEMM in exchange for the
        16 KiB/token the gate-copy save keeps. **Expected default**; the
        gate-copy half is a follow-up PR and :meth:`GatedAttentionBlockBwd.check_support`
        declines such a record typed until it lands.

    ``RECOMPUTE_GATE``
        RESERVED -- not servable against the current schema. It would
        additionally drop the GATE from the save set and re-run its slice too
        (another 16 GiB saved; the same GEMM, so at that point it is a full
        stage-(1) recompute). But the training forward ALWAYS saves the GATE
        -- as the compact ``gate`` (gate-copy save mode) or as a band of
        ``proj_slab`` (proj_slab save mode, where ``gate`` may be ``None``
        because ``saved_slab_views(proj_slab, ...)[1]`` derives it) -- and the
        stage-(1) GEMM always produces the GATE columns, so today there is
        nothing to drop. It becomes real when the forward learns to skip the
        GATE columns; until then :meth:`GatedAttentionBlockBwd.check_support`
        declines it rather than recomputing a tensor that is present.

    Two operands the schema does NOT hold, under every policy:

    * **Post-norm, post-RoPE Q and K are never saved.** The SDPA backward (B4)
      consumes them, and they are rebuilt from ``q_pre`` / ``k_pre`` by
      re-running the forward's stages (2)+(3) kernel (``api._QkNormRope``) --
      one bandwidth pass, no GEMM; the rstd it recomputes over the same inputs
      IS the forward's, and the SAVED ``rstd_q`` / ``rstd_k`` feed B6.
    * **V is a column slice of stage (1) and has no field of its own.** With a
      ``proj_slab`` it is the V band, compacted by one elementwise copy for the
      adapter (whose operands must be BSHD-physical). An appended ``Optional``
      ``v`` slot on ``SavedForBackward`` is a schema change and is not made here.

    Whatever is recomputed must be recomputed BIT-IDENTICALLY to the forward, or
    gradients acquire a noise floor that looks like a kernel bug. Same tile
    config, same accumulation order, same epilogue rounding.

    Under ``geometry.qk_norm=False`` the policy still decides ``q_pre`` /
    ``k_pre`` (the RoPE-only backward reads them through the same slots), but
    there is no ``rstd`` to save or rebuild -- stage (B6) does not exist.
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
    contiguous operand -- the whole reason the forward's concat ordering is
    contract rather than convenience. It is written exactly once per column:
    the GATE band by B3, the Q / K / V bands by B5+B6 (from the COMPACT B4 slots
    ``dq`` / ``dk`` / ``dv`` -- the adapter's BSHD-physical contract).

    ``-1`` means the region does not exist for this declaration (a ``need_*``
    False, a knob folded out) or aliases another one (``do``).

    Append-only: the first eight fields keep their positions; the block backward appended
    the rest (the gate-copy follow-up appends its recompute slabs after them).
    """

    do_gated: int  # [T, H_q, D]   B2's output
    do: int  # -1: ALIASES do_gated (B3 writes dO in place -- `do` MAY alias `dog`, kernels/sigmoid_gate_bwd.py)
    dqkvg: int  # [T, N]        dQ_pre | dG | dK_pre | dV, written ONLY by the elementwise kernels
    o_gated: int  # [T, H_q, D]   B3's third output (`og`) for B1; -1 when need_dw_o is False (has_og=False recipe)
    sdpa_bwd_ws: int  # the adapter's scratch_workspace_bytes(), 256-B aligned offset (it carves 128-B chunks inside)
    recompute: int  # [T, H_q, D]   rebuilt post-norm / post-RoPE Q (the adapter's input; never -1)

    total_bytes: int
    base_align: int
    # APPENDED with the block backward
    recompute_k: int = -1  # [T, H_kv, D]  rebuilt post-norm / post-RoPE K
    recompute_v: int = -1  # [T, H_kv, D]  compact V (copied from the proj_slab band)
    dq: int = -1  # [T, H_q, D]   B4 output (compact)
    dk: int = -1  # [T, H_kv, D]
    dv: int = -1  # [T, H_kv, D]
    dw_partials_q: int = -1  # [n_ctas_q, D] fp32 -- TWO planes, not one: the kernel takes two contiguous tensors
    dw_partials_k: int = -1  # [n_ctas_k, D] fp32    (exactly n_ctas_for(recipe, T) rows each); -1 without need_dw_norms
    gemm_scratch: int = -1  # max(plan.workspace_bytes for B1, B2, B7, B8), never 0 (max(.., 1))
    n_ctas_q: int = 0  # the plane row counts the carve was sized for (the reduce is launched with t=T and re-derives them)
    n_ctas_k: int = 0
    sdpa_bwd_bytes: int = 0  # the adapter's slice length (the region is padded to the carve alignment)
    gemm_scratch_bytes: int = 0
    # APPENDED with fuse_gate_bwd: B3's delta = rowsum(dO * O) for the adapter's external_delta; -1 / () when the knob is off
    delta: int = -1  # [B, H_q, S_pad] fp32 (the adapter's external_delta_shape)
    delta_shape: tuple = ()


def _plan_bwd_workspace(
    geom: GatedAttentionBlockGeometry,
    b: int,
    s: int,
    dtype: torch.dtype,
    policy: RecomputePolicy,
    *,
    need: Optional[dict] = None,
    sdpa_bwd_bytes: int = 0,
    gemm_scratch_bytes: int = 0,
    n_ctas_q: int = 0,
    n_ctas_k: int = 0,
    delta_shape: Optional[tuple] = None,
) -> _BwdIntermediates:
    """Reserve every backward intermediate and report the total.

    The positional head is the declaration; the keyword facts come from
    ``compile()`` -- the adapter's scratch, the four plans' ``workspace_bytes``
    and the dW-partial plane rows ``n_ctas_for(recipe, T)`` exist only once the
    artifacts do, which is why :meth:`GatedAttentionBlockBwd.get_workspace_size`
    requires ``compile()`` first. ``need`` = ``{"dw_o", "dw_norms"}`` flags
    (missing = True); ``policy`` is kept for the gate-copy follow-up's recompute slabs (a
    ``proj_slab`` record carves none).  ``delta_shape`` (``fuse_gate_bwd``: the adapter's
    ``external_delta_shape``, ``(B, H_q, S_pad)``) carves the fp32 ``delta`` region B3 writes
    and B4 reads; None carves none (the adapter keeps its own).

    Regions are ``_WS_ALIGN`` (256 B) aligned so every typed ``_view`` and the
    adapter's own 128-B carve are legal; ``gemm_scratch`` is ``max(.., 1)`` so
    the slice handed to the GEMM driver is never empty.
    """
    need = dict(need or {})
    want_og = bool(need.get("dw_o", True))
    want_dw = bool(need.get("dw_norms", geom.qk_norm))
    t, e, d = int(b) * int(s), _itemsize(dtype), geom.d_head
    layout = WorkspaceLayout(align=_WS_ALIGN)
    do_gated = layout.add(t * geom.h_q * d * e)
    dqkvg = layout.add(t * geom.n_qkvg * e)
    o_gated = layout.add(t * geom.h_q * d * e) if want_og else -1
    recompute = layout.add(t * geom.h_q * d * e)
    recompute_k = layout.add(t * geom.h_kv * d * e)
    recompute_v = layout.add(t * geom.h_kv * d * e)
    dq = layout.add(t * geom.h_q * d * e)
    dk = layout.add(t * geom.h_kv * d * e)
    dv = layout.add(t * geom.h_kv * d * e)
    if want_dw:
        if n_ctas_q < 1 or n_ctas_k < 1:
            raise ValueError(f"need_dw_norms needs the dW partial plane rows from the compiled recipe (n_ctas_q={n_ctas_q}, n_ctas_k={n_ctas_k})")
        dw_partials_q = layout.add(int(n_ctas_q) * d * 4)
        dw_partials_k = layout.add(int(n_ctas_k) * d * 4)
    else:
        dw_partials_q = dw_partials_k = -1
        n_ctas_q = n_ctas_k = 0
    if sdpa_bwd_bytes < 1:
        raise ValueError(f"the SDPA backward adapter's scratch must be known (got {sdpa_bwd_bytes} bytes); it is a pure function of the geometry")
    sdpa_bwd_ws = layout.add(int(sdpa_bwd_bytes))
    gemm_scratch_bytes = max(int(gemm_scratch_bytes), 1)
    gemm_scratch = layout.add(gemm_scratch_bytes)
    delta = -1
    if delta_shape is not None:
        delta_shape = tuple(int(x) for x in delta_shape)
        if len(delta_shape) != 3 or delta_shape[0] != b or delta_shape[1] != geom.h_q or delta_shape[2] < s:
            raise ValueError(f"delta_shape must be the adapter's (B={b}, H_q={geom.h_q}, S_pad >= {s}), got {delta_shape}")
        delta = layout.add(int(math.prod(delta_shape)) * 4)
    return _BwdIntermediates(
        do_gated=do_gated,
        do=-1,
        dqkvg=dqkvg,
        o_gated=o_gated,
        sdpa_bwd_ws=sdpa_bwd_ws,
        recompute=recompute,
        total_bytes=layout.size,
        base_align=layout.base_align,
        recompute_k=recompute_k,
        recompute_v=recompute_v,
        dq=dq,
        dk=dk,
        dv=dv,
        dw_partials_q=dw_partials_q,
        dw_partials_k=dw_partials_k,
        gemm_scratch=gemm_scratch,
        n_ctas_q=int(n_ctas_q),
        n_ctas_k=int(n_ctas_k),
        sdpa_bwd_bytes=int(sdpa_bwd_bytes),
        gemm_scratch_bytes=gemm_scratch_bytes,
        delta=delta,
        delta_shape=tuple(delta_shape) if delta_shape is not None else (),
    )


# ---------------------------------------------------------------------------
# 3. Stages — one FROST kernel each, in pipeline order
# ---------------------------------------------------------------------------


class _GemmStage(_Stage):
    """Shared body of the four projection GEMMs (B1, B2, B7, B8) on the shipped
    FROST GEMM through ``kernels/proj_gemm.py``'s backward drivers.

    ``kind`` picks the driver and the plan's majors: ``"wgrad"`` is
    ``dW[rows, cols] = dy_like[T, rows]^T @ x[T, cols]`` (A M-major, B N-major,
    ``m=rows, k=T, n=cols``); ``"dgrad"`` is ``dX[T, N] = dy_like[T, K] @ w[K, N]``
    with ``w`` the UN-transposed row-major weight (A K-major, B N-major,
    ``m=T, k=K, n=N``). Both bind zero-copy views of contiguous ``[rows, cols]``
    storage and refuse anything else, typed, before any launch -- never hand
    them a column band of the slab.

    The tile is the block's FORCED ``..._cluster2x1_2ctamma`` config whenever
    ``n % 256 == 0`` (every backward GEMM of the block), at one split-K slice on
    the JIT route; ``plan.tile_config_name`` / ``plan.route`` / ``plan.jit``
    record what runs and the tests pin them (a fallback to the heuristic is a
    FAILURE: different config, route and possibly a split-K reducer).
    """

    kind: str = ""

    def __init__(self, *, m: int, k: int, n: int, dtype: torch.dtype, label: str) -> None:
        self.m, self.k, self.n = int(m), int(k), int(n)
        self.dtype = dtype
        self.label = label
        self.plan = None

    @property
    def majors(self) -> tuple:
        return ("m", "n") if self.kind == "wgrad" else ("k", "n")

    def check_support(self) -> None:
        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: the backward GEMM drivers serve bf16 / fp16 only, got {self.dtype}")
        # The TMA 16-byte contiguous-extent rule falls on the MN-major operands (build_proj_gemm's
        # _check_mn_major_tma_rule would say the same at compile(); the block says it at declaration).
        elems16 = 16 // _itemsize(self.dtype)
        a_major, _b_major = self.majors
        if a_major == "m" and self.m % elems16:
            raise ValueError(
                f"{self.name}: A is M-major (a transposed operand), so M={self.m} must be a multiple of {elems16} ({self.dtype}: the TMA 16-byte rule)"
            )
        if self.n % elems16:
            raise ValueError(f"{self.name}: B is N-major, so N={self.n} must be a multiple of {elems16} ({self.dtype}: the TMA 16-byte rule)")

    def compile(self) -> None:
        from .kernels.proj_gemm import build_proj_gemm

        a_major, b_major = self.majors
        self.plan = build_proj_gemm(m=self.m, k=self.k, n=self.n, dtype=self.dtype, label=self.label, a_major=a_major, b_major=b_major)

    def workspace_bytes(self) -> int:
        if self.plan is None:
            raise RuntimeError(f"{self.name}: call compile() before workspace_bytes()")
        return int(self.plan.workspace_bytes)

    def execute(self, dy_like: torch.Tensor, other: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor, *, stream) -> None:
        from .kernels.proj_gemm import run_dgrad_gemm, run_wgrad_gemm

        if self.plan is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        runner = run_wgrad_gemm if self.kind == "wgrad" else run_dgrad_gemm
        runner(self.plan, dy_like, other, out, workspace, stream=stream)


class _OutProjWgrad(_GemmStage):
    """(B1) ``dW_o = dY^T @ O_gated``. Contracts over TOKENS: ``K = B*S``.

    ``W_o`` is ``[d_model, H_q*D]`` and the forward computes ``O_gated @ W_o^T``,
    so the weight gradient is ``dY^T @ O_gated`` and lands in ``W_o``'s own
    layout -- no transpose on the way out. ``O_gated`` is not saved: B3 writes
    it as its optional third output (``has_og`` = ``need_dw_o``), so this stage
    runs AFTER B3 and reads the workspace ``o_gated`` slot.

    **Fusion status: this is the backward's filler.** It depends on nothing
    the SDPA backward produces, so it is the work to overlap it with. Getting
    that overlap is a scheduling question (a second stream and events, or PDL
    -- a later PR), not a kernel-fusion one.

    Forced tile at one split-K slice; a pinned split (``split_k >= 2``) would
    reduce in fixed order (module docstring, "Determinism").
    """

    name = "out_proj_wgrad"
    kind = "wgrad"


class _OutProjDgrad(_GemmStage):
    """(B2) ``dO_gated = dY @ W_o``, contracting over ``d_model``.

    **Fusion status: per-head independent, therefore legal as an SDPA-backward
    prologue -- and deliberately NOT taken.** ``dO`` is read with two different
    tilings by the SDPA backward (dK/dV loop q-tiles for a fixed kv-tile; dQ
    loops kv-tiles for a fixed q-tile), so it is materialized either way and
    the prologue fusion saves nothing on the dK/dV side.
    """

    name = "out_proj_dgrad"
    kind = "dgrad"


class _QkvGateWgrad(_GemmStage):
    """(B7) ``dW_qkvg = dQKVG^T @ h``. Contracts over TOKENS: ``K = B*S``.

    ``W_qkvg`` is ``[N, d_model]`` and the forward computes ``h @ W_qkvg^T``, so
    the weight gradient is ``dQKVG^T @ h``: ``[N, d_model]``, the exact
    ``qkvg_offsets`` row layout the forward consumes, so a caller never
    re-slices. At 397B: ``17408 x 4096``. Same tiling family as B1 and nothing
    else in the block; reads ``saved.h`` as ``[T, d_model]``.
    """

    name = "qkv_gate_wgrad"
    kind = "wgrad"


class _QkvGateDgrad(_GemmStage):
    """(B8) ``dh = dQKVG @ W_qkvg``, contracting over N.

    The block's output gradient. Nothing downstream of it here.
    """

    name = "qkv_gate_dgrad"
    kind = "dgrad"


class _SigmoidGateBwd(_Stage):
    """(B3) gate backward, elementwise over ``[T, H_q, D]``::

        s       = sigmoid(GATE)
        dO      = dO_gated * s                      (IN PLACE over dO_gated)
        dG      = dO_gated * O * s * (1 - s)        (-> the GATE band of dqkvg)
        O_gated = O * s                             (optional third output, for B1 -- has_og = need_dw_o)

    Three input reads, two (or three) writes, one pass. ``s`` in fp32 regardless
    of storage dtype; ``s * (1 - s)`` in bf16 loses bits exactly where the gate
    saturates and the gradient matters least -- but it is free to do right.

    ``dG`` is written straight into the ``dqkvg`` slab at the GATE offset and
    never touched again: B5+B6 write only the Q, K and V bands. GATE is read as
    a strided column band of ``saved.proj_slab`` -- every operand carries its own
    token stride, so no repack anywhere.

    **Hazard, mirroring the forward's stage (5):** ``O`` here is the SDPA's
    *substituted* output (``saved.o``). For a dead row it is an exact zero,
    which correctly makes ``dG`` zero; if a future fused epilogue ever hands
    this stage pre-substitution residue instead, ``dG`` becomes NaN for rows
    whose forward output was fine. P0 declines padding, so the only dead rows
    would be mask-made ones, and with ``S_q == S_kv`` a causal / windowed row
    always keeps its diagonal (``window_left == 0`` is declined first).

    **Fusion status: the SDPA backward's ``dot_do_o`` pre-pass is folded in HERE
    under ``fuse_gate_bwd``** (``want_delta``): a fourth output ``delta[b, h, q] =
    rowsum(dO * O)`` in the chain's own order over the bf16-rounded ``dO`` this
    kernel stores, so the adapter (``external_delta=True``) skips its first launch
    and its second read of ``O`` and ``dO``; bitwise the unfused chain.  The other
    direction -- folding this kernel into the SDPA backward's prologue -- stays
    untaken: ``dO`` is read under two tilings there (``_OutProjDgrad``).

    Kernel: ``kernels/sigmoid_gate_bwd.py`` (plain LDG/STG, any CuTe-DSL device).
    """

    name = "sigmoid_gate_bwd"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, want_og: bool, want_delta: bool = False) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_og = bool(want_og)
        self.want_delta = bool(want_delta)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.sigmoid_gate_bwd import DEFAULT_THREADS_PER_CTA, validate_shape

        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: bf16 / fp16 only, got {self.dtype}")
        validate_shape(self.geom.d_head, DEFAULT_THREADS_PER_CTA)

    def compile(self) -> None:
        from .kernels.sigmoid_gate_bwd import compile_sigmoid_gate_bwd

        self._recipe = compile_sigmoid_gate_bwd(
            dtype=self.dtype, h=self.geom.h_q, d=self.geom.d_head, has_og=self.want_og, has_seq_lens=False, has_delta=self.want_delta
        )

    def execute(self, dog, o, gate, do, dg, og, *, stream, delta=None) -> None:
        """``delta`` (``want_delta`` only): the fp32 ``[B, H_q, S_pad]`` region the adapter reads as its external delta."""
        from .kernels.sigmoid_gate_bwd import run_sigmoid_gate_bwd

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        run_sigmoid_gate_bwd(
            self._recipe, dog, o, gate, do, dg, og=og, seq_lens=None, s=self.seq_len if delta is not None else None, stream=stream, delta=delta
        )


class _SdpaBwd(_Stage):
    """(B4) ``(dO, Q, K, V, O, LSE) -> dQ, dK, dV``, GQA-reduced over H_q/H_kv.

    **Writes no new kernel**: drives the shipped Rubin d=256 backward adapter
    ``cudnn.sdpa.bwd.api_dsl_sm107.SdpaBwdDslSm107`` -- a graph-free building
    block (construct over nine ``TensorDesc``s, ``check_support()``,
    ``compile()``, ``execute(..., workspace=<uint8 slice>, current_stream=)``)
    that carves its own scratch (delta, ONE dS chunk, padded staging, the GQA
    partials) out of the slice the block reserves for it and allocates nothing.
    ONE engine class, no backend fallback (module docstring, Rule 9).

    Consumes ``Q`` and ``K`` **post-norm, post-RoPE** -- the same tensors the
    forward's SDPA saw -- and ``V``, all COMPACT ``[B, S, H, D]`` (the adapter
    requires BSHD-physical operands: the block rebuilds Q / K through the
    forward's ``_QkNormRope`` from the ``proj_slab`` bands and compacts V with
    ``_VCompaction``, then hands the ``.transpose(1, 2)`` views the binder
    demands -- exactly what the forward's ``_Sdpa`` stage does). Writes COMPACT
    ``dQ`` / ``dK`` / ``dV`` into the workspace slots B5+B6 read.

    ``LSE`` is mandatory here and binds as the forward wrote it: ``[B, H_q, S]``
    fp32 NATURAL log -- the backward applies ``log2e`` itself at its stats
    prefetch (the block's forward never sets
    ``STATS_LOG2``). That is the whole reason ``save_for_backward`` implies
    ``return_lse`` in the forward.

    Declared with ``deterministic=False`` (the row declines ``True``); the
    chain has no atomics and the block's two-run bitwise test is the pin.
    Masks: the block's ``is_causal`` / ``causal_bottom_right`` / ``window_left``
    (> 0) / ``window_right`` (0) map one-to-one onto the adapter's; what the row
    cannot serve is declined by the block first, naming its own field.

    **Fusion status:** under ``fuse_gate_bwd`` (``external_delta``) the adapter is
    built with ``external_delta=True`` and :meth:`execute` hands it B3's ``delta``
    region, so the chain's ``dot_do_o`` launch does not exist and its scratch has
    no ``delta`` region (``delta_shape`` is the adapter's contract for the carve).
    """

    name = "sdpa_bwd"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, device, external_delta: bool = False) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.device = device
        self.external_delta = bool(external_delta)
        self._impl = None

    def _build_impl(self):
        from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

        g, b, s, d, act, dev = self.geom, self.batch, self.seq_len, self.geom.d_head, self.dtype, self.device
        # The row REQUIRES rank-4 (B, H_q, S_q, 1) stats with exactly this stride; saved.lse [B, H_q, S] binds to it as is
        # (the binder checks contiguity + element count only for Stats).
        stats = TensorDesc(dtype=torch.float32, shape=(b, g.h_q, s, 1), stride=(g.h_q * s, s, 1, 1), stride_order=(3, 2, 1, 0), device=dev, name="stats")
        return SdpaBwdDslSm107(
            sample_q=_bhsd_desc(b, g.h_q, s, d, act, dev, "q"),
            sample_k=_bhsd_desc(b, g.h_kv, s, d, act, dev, "k"),
            sample_v=_bhsd_desc(b, g.h_kv, s, d, act, dev, "v"),
            sample_o=_bhsd_desc(b, g.h_q, s, d, act, dev, "o"),
            sample_do=_bhsd_desc(b, g.h_q, s, d, act, dev, "dO"),
            sample_stats=stats,
            sample_dq=_bhsd_desc(b, g.h_q, s, d, act, dev, "dQ"),
            sample_dk=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dK"),
            sample_dv=_bhsd_desc(b, g.h_kv, s, d, act, dev, "dV"),
            is_causal=bool(g.is_causal),
            causal_bottom_right=bool(g.causal_bottom_right),
            window_size_left=None if g.window_left < 0 else int(g.window_left),
            window_size_right=None if g.window_right < 0 else int(g.window_right),
            deterministic=False,
            scale_softmax=float(g.scale),
            seq_kv_lens_present=False,
            external_delta=self.external_delta,
        )

    @property
    def delta_shape(self) -> tuple:
        """The adapter's ``external_delta_shape`` -- ``(B, H_q, S_pad)`` fp32, the region B3 fills under ``fuse_gate_bwd``."""
        if self._impl is None:
            self._impl = self._build_impl()
        return tuple(int(x) for x in self._impl.external_delta_shape)

    def check_support(self) -> None:
        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: the sdpa_bwd_sm107 row serves bf16 / fp16 only, got {self.dtype}")
        if self.geom.d_head != 256:
            raise NotImplementedError(f"{self.name}: the Rubin d256 backward serves d_head = 256 exactly, got {self.geom.d_head}")
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        self._impl = self._build_impl()
        self._impl.check_support()

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the geometry: callable right after construction (no compile)."""
        if self._impl is None:
            self._impl = self._build_impl()
        return int(self._impl.scratch_workspace_bytes())

    def compile(self) -> None:
        if self._impl is None:
            self._impl = self._build_impl()
        self._impl.compile()

    def execute(self, q, k, v, o, do, lse, dq, dk, dv, *, workspace: torch.Tensor, stream, delta=None) -> None:
        """Every io tensor COMPACT ``[B, S, H, D]``; transposed here into the ``(B, H, S, D)`` view the binder demands.
        ``delta`` (``external_delta`` only): B3's fp32 ``[B, H_q, S_pad]`` region, the adapter's ``delta_tensor``."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        self._impl.execute(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            o.transpose(1, 2),
            do.transpose(1, 2),
            lse,
            dq.transpose(1, 2),
            dk.transpose(1, 2),
            dv.transpose(1, 2),
            workspace=workspace,
            current_stream=cuda.CUstream(int(stream)),
            delta_tensor=delta,
        )


class _QkNormRopeBwd(_Stage):
    """(B5)+(B6) ONE kernel: inverse RoPE then RMSNorm backward on dQ and dK, the
    V band copied bit-exactly, fp32 ``dW_norm`` partials -- plus the fixed-order
    reduce launch. Merged for the same reason (2) and (3) merge in the forward:
    one pass over dQ, one over dK.

    (B5) RoPE is orthogonal, so the backward is the transpose. The shipped form
    is the EXACT adjoint for ANY table, ``dx = dy*cos - rotate_half(dy*sin)``
    (``y = C x + S R x`` with ``R^T = -R`` gives ``dx = C dy - R (S dy)``; the
    naive ``dy*cos - rotate_half(dy)*sin`` is exact only for duplicated-half
    tables, which ``build_rope_tables`` happens to guarantee -- pinned by the
    random-table test). Dims ``[ROPE_DIM, D)`` pass through. Same ``cos`` /
    ``sin`` tables the forward was given; the caller re-supplies them rather
    than the block saving a copy.

    (B6) per-head RMSNorm backward over D, on dQ and dK -- **not on dV**::

        dx = rstd * (g*w - x_hat * mean(g*w*x_hat))     with g = RoPE^T(dy), x_hat = x * rstd
        dW = sum over ALL tokens and ALL heads of (g * x_hat)

    ``dx`` is a per-row reduction over D (one lane group per row); ``dW_q_norm``
    / ``dW_k_norm`` are ``[D]`` accumulated over ``B*S*H`` rows -- per-CTA fp32
    partials (``[n_ctas_q, D]`` / ``[n_ctas_k, D]``, EXACTLY ``n_ctas_for(recipe,
    T)`` rows: the reduce sums every row it is handed) plus a second launch that
    reduces them in a FIXED order, never atomics (module docstring,
    "Determinism"). Needs ``x`` (pre-norm Q / K: the ``proj_slab`` bands) and the
    SAVED ``rstd``. Do **not** try to recover ``x`` by dividing the normed value
    by ``w``: undefined at a zero norm weight, hostile at a small one.

    Writes dQ_pre / dK_pre / dV into the Q / K / V bands of ``dqkvg``, reading
    the COMPACT B4 slots.

    **Under ``geometry.qk_norm=False``** the RoPE-only adjoint is traced
    (``apply_norm=False``): no norm weights, no rstd, no partials, no reduce --
    both directions typed by the kernel's own presence checks.

    Kernel: ``kernels/qk_norm_rope_bwd.py`` (LDG/STG + one butterfly shuffle +
    a 4 KiB SMEM combine; any CuTe-DSL device).
    """

    name = "qk_norm_rope_bwd"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, want_dw: bool) -> None:
        self.geom = geometry
        self.batch, self.seq_len = int(batch), int(seq_len)
        self.dtype = dtype
        self.want_dw = bool(want_dw)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.qk_norm_rope_bwd import DEFAULT_THREADS_PER_CTA, validate_shape

        if self.dtype not in _ACT_DTYPES:
            raise NotImplementedError(f"{self.name}: bf16 / fp16 only, got {self.dtype}")
        if self.want_dw and not self.geom.qk_norm:
            raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and therefore no dW_norm; need_dw_norms must be False")
        validate_shape(self.geom.d_head, self.geom.rope_dim, DEFAULT_THREADS_PER_CTA)

    def compile(self) -> None:
        from .kernels.qk_norm_rope_bwd import compile_qk_norm_rope_bwd

        g = self.geom
        # rows_per_group=None -> the MEASURED per-arm optimum (1 with the norm, 2 RoPE-only); never pass 1 for the RoPE-only arm.
        self._recipe = compile_qk_norm_rope_bwd(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            apply_norm=bool(g.qk_norm),
            want_dw=self.want_dw,
            has_seq_lens=False,
        )

    def n_ctas(self) -> tuple:
        """``(n_ctas_q, n_ctas_k, n_ctas_v)`` for this declaration's ``T`` -- the dW partial plane rows."""
        from .kernels.qk_norm_rope_bwd import n_ctas_for

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before n_ctas()")
        return n_ctas_for(self._recipe, self.batch * self.seq_len)

    def execute(self, dq, dk, dv, xq, xk, rstd_q, rstd_k, w_q, w_k, cos, sin, out_q, out_k, out_v, plane_q, plane_k, *, stream) -> None:
        from .kernels.qk_norm_rope_bwd import run_qk_norm_rope_bwd

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        run_qk_norm_rope_bwd(
            self._recipe,
            dq,
            dk,
            dv,
            xq,
            xk,
            rstd_q,
            rstd_k,
            w_q,
            w_k,
            cos,
            sin,
            out_q,
            out_k,
            out_v,
            dw_partials_q=plane_q,
            dw_partials_k=plane_k,
            seq_lens=None,
            stream=stream,
        )

    def reduce(self, plane_q, plane_k, dw_q_norm, dw_k_norm, *, stream) -> None:
        """The second launch: the partial planes -> ``dW_q_norm`` / ``dW_k_norm`` (fp32 ``[D]``), fixed order; ``t=T`` arms the plane cross-check."""
        from .kernels.qk_norm_rope_bwd import run_dw_norm_reduce

        if self._recipe is None:
            raise RuntimeError(f"{self.name}: call compile() before reduce()")
        run_dw_norm_reduce(self._recipe, plane_q, plane_k, dw_q_norm, dw_k_norm, stream=stream, t=self.batch * self.seq_len)


# ---------------------------------------------------------------------------
# 4. Host-only checks shared by check_support and execute
# ---------------------------------------------------------------------------


def _byte_range(t: torch.Tensor):
    """``[lo, hi)`` of the bytes a (possibly strided) tensor can touch; ``None`` for an empty one."""
    if t.numel() == 0:
        return None
    span = 1 + sum((int(sz) - 1) * int(st) for sz, st in zip(t.shape, t.stride()) if int(sz) > 1)
    lo = int(t.data_ptr())
    return lo, lo + span * t.element_size()


def _check_no_overlap(written, read) -> None:
    """No buffer the backward WRITES overlaps any other buffer it binds (a caller aliasing ``saved.o`` onto the
    workspace, or ``dh`` onto ``dy``, would silently corrupt an operand another stage still reads). Read-only
    inputs may alias each other freely (shared weights, ``saved.gate`` inside ``saved.proj_slab``)."""
    for i, (wn, wt) in enumerate(written):
        wr = _byte_range(wt)
        if wr is None:
            continue
        for name, ten in written[i + 1 :] + read:
            if ten is None:
                continue
            r = _byte_range(ten)
            if r is None:
                continue
            if wr[0] < r[1] and r[0] < wr[1]:
                raise ValueError(
                    f"{wn} overlaps {name} in memory ([{wr[0]:#x}, {wr[1]:#x}) vs [{r[0]:#x}, {r[1]:#x})): the backward writes {wn} while "
                    f"other stages still read {name}; give every output and the workspace its own storage"
                )


def _check_saved_record(saved: SavedForBackward, geom: GatedAttentionBlockGeometry, b: int, s: int, act: torch.dtype, dev, *, at: str) -> tuple:
    """Validate a :class:`SavedForBackward` record against the declaration (host-only, typed, names the field) and
    return ``(proj [T, N] view of proj_slab, o [T, H_q, D] view of saved.o)``.  ``at`` = ``"declaration"`` (the sample
    record at ``check_support``: a gate-copy record is a typed ``NotImplementedError`` naming the follow-up) or ``"execute"``
    (a record disagreeing with the declaration is a ``ValueError``)."""
    if not isinstance(saved, SavedForBackward):
        raise ValueError(f"saved must be a SavedForBackward record, got {type(saved).__name__}")
    # The record's two PRESENCE facts come first, before any buffer is validated, so a gate-copy or a padded record
    # gets the answer that matters (the typed decline) and not a complaint about one of its buffers.
    if saved.proj_slab is None:
        msg = (
            "SavedForBackward.proj_slab is None (the gate-copy save mode): P0 of the block backward serves proj_slab records only -- "
            "V is a band of the slab with no field of its own, so a gate-copy record needs the stage-(1) column-slice recompute "
            "(RecomputePolicy.RECOMPUTE_QK_PRE over saved.h), which a follow-up PR lands. Run the training forward with saved_gate_copy=False"
        )
        raise NotImplementedError(msg) if at == "declaration" else ValueError(msg + " (this block was declared over a proj_slab record)")
    if saved.seq_lens is not None:
        # A PADDED forward's record carries its seq_lens tensor; its PRESENCE is the fact (never its values -- no device
        # read). Declined at declaration (P0), and refused at execute for a record that contradicts the dense declaration:
        # a padded record run through the dense chain reads LSE = -inf / O = 0 on its dead rows and every gradient is NaN.
        msg = (
            "saved.seq_lens is a tensor (a PADDED forward's record): padding is not served by the block backward yet -- the "
            "sdpa_bwd_sm107 row declines seq_kv_lens_present; a follow-up PR flips it with the row's `padded` capability"
        )
        raise (
            NotImplementedError(msg)
            if at == "declaration"
            else ValueError(msg + " -- and this block was declared WITHOUT padding (seq_lens_present=False), so the record contradicts the declaration")
        )
    check = GatedAttentionBlockFwd._check_saved_tensor
    t, d = b * s, geom.d_head
    if geom.qk_norm:
        if saved.rstd_q is None or saved.rstd_k is None:
            raise ValueError("geometry.qk_norm=True: SavedForBackward.rstd_q and rstd_k are required (the forward wrote them; stage B6 consumes them)")
        check("saved.rstd_q", saved.rstd_q, (b, s, geom.h_q), torch.float32, dev)
        check("saved.rstd_k", saved.rstd_k, (b, s, geom.h_kv), torch.float32, dev)
    elif saved.rstd_q is not None or saved.rstd_k is not None:
        raise ValueError("geometry.qk_norm=False: SavedForBackward.rstd_q / rstd_k must be None (the forward wrote none; stage B6 does not exist)")
    check("saved.h", saved.h, (b, s, geom.d_model), act, dev)
    check("saved.lse", saved.lse, (b, geom.h_q, s), torch.float32, dev)
    check("saved.o", saved.o, (b, s, geom.h_q, d), act, dev)
    ps = saved.proj_slab
    if not isinstance(ps, torch.Tensor):
        raise ValueError(f"saved.proj_slab must be a tensor, got {type(ps).__name__}")
    if ps.dtype != act:
        raise ValueError(f"saved.proj_slab must be the activation dtype {act} (the projection GEMM's output dtype), got {ps.dtype}")
    if ps.device != dev:
        raise ValueError(f"saved.proj_slab must live on dy's device {dev}, got {ps.device}")
    want_q, want_gate, want_k, _want_v = saved_slab_views(ps, geom, b, s)  # numel / contiguity / 16-B alignment, typed
    for nm, given, want in (("gate", saved.gate, want_gate), ("q_pre", saved.q_pre, want_q), ("k_pre", saved.k_pre, want_k)):
        if given is None:
            continue
        ok = (
            isinstance(given, torch.Tensor)
            and given.data_ptr() == want.data_ptr()
            and given.dtype == want.dtype
            and tuple(given.shape) == tuple(want.shape)
            and tuple(given.stride()) == tuple(want.stride())
        )
        if not ok:
            raise ValueError(
                f"saved.{nm} must alias saved.proj_slab's band exactly as saved_slab_views(proj_slab, geometry, batch, seq_len) spells it "
                f"([B, S, heads, D], token stride n_qkvg, storage offset at qkvg_offsets), or be None (the backward derives it from the slab)"
            )
    return ps.view(t, geom.n_qkvg), saved.o.view(t, geom.h_q, d)


# ---------------------------------------------------------------------------
# 5. The public API
# ---------------------------------------------------------------------------


class GatedAttentionBlockBwd(APIBase):
    """Gated attention block, backward. One call, one workspace, eight stages.

    Built against a forward that ran with ``save_for_backward=True`` in the
    proj_slab save mode. It does not re-derive the geometry: pass the same
    :class:`~cudnn.gated_attention_block.api.GatedAttentionBlockGeometry`
    instance, and ``check_support`` checks the saved tensors against it rather
    than trusting either alone.

    Lifecycle, exactly the forward's: ``check_support()``; ``compile()``;
    ``get_workspace_size()`` (needs the compiled artifacts -- see the method);
    ``execute(...)`` any number of times, allocation-free.
    """

    def __init__(
        self,
        sample_dy: torch.Tensor,  # [B, S, d_model]
        sample_saved: SavedForBackward,
        sample_w_qkvg: torch.Tensor,  # [N, d_model]
        sample_w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False -- same positions
        sample_w_k_norm: Optional[torch.Tensor],  # [D]
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
        # None -> geometry.qk_norm: the norm-weight gradients exist exactly when
        # the norm does. An explicit True under qk_norm=False is a typed
        # decline (there is no dW_*_norm to compute), never a silent False.
        need_dw_norms: Optional[bool] = None,
        # APPENDED with the block backward, keyword-only, defaulted, LAST -- append-only forever.
        # The padding the forward ran with is a DECLARATION fact; P0 declines it typed (the
        # sdpa_bwd_sm107 row serves no seq_lens), from this flag OR from sample_saved.seq_lens.
        seq_lens_present: bool = False,
        # The dtype of dW_q_norm / dW_k_norm.  P0 serves fp32 ONLY (the kernel writes fp32
        # partials and the reduce fp32 [D] outputs; a cast would be an extra launch nobody measured).
        dw_norm_dtype: torch.dtype = torch.float32,
        # Fusion knob (module docstring, "Fusion knob"): the gate-backward kernel also emits the SDPA backward's
        # delta = rowsum(dO * O), and the adapter is built with external_delta=True -- one launch and one read each of
        # O and dO fewer, bitwise the unfused block.  Performance-only: the same function under either value.
        fuse_gate_bwd: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        # The four declaration-time contracts (typed, in this order, BEFORE anything else -- pinned by
        # test_block_end_to_end.py::test_bwd_declaration_contracts_fire_first):
        _check_norm_weights_agree(geometry.qk_norm, sample_w_q_norm, sample_w_k_norm, prefix="sample_")
        if need_dw_norms is None:
            need_dw_norms = bool(geometry.qk_norm)
        elif need_dw_norms and not geometry.qk_norm:
            raise ValueError(
                "need_dw_norms=True with geometry.qk_norm=False: a RoPE-only block has no norm weights, so there is no dW_q_norm / dW_k_norm to compute"
            )
        if not geometry.qk_norm and (sample_saved.rstd_q is not None or sample_saved.rstd_k is not None):
            raise ValueError("geometry.qk_norm=False: SavedForBackward.rstd_q / rstd_k must be None (the forward wrote none; stage B6 does not exist)")
        self.need_dw_norms = bool(need_dw_norms)
        if not isinstance(recompute, RecomputePolicy):
            raise TypeError(f"recompute must be a RecomputePolicy, got {type(recompute).__name__}")
        if sample_dy.ndim != 3:
            raise ValueError(f"sample_dy must be [B, S, d_model], got {tuple(sample_dy.shape)}")
        self.batch, self.seq_len, d_model = (int(x) for x in sample_dy.shape)
        if d_model != geometry.d_model:
            raise ValueError(f"sample_dy last dim {d_model} != geometry.d_model {geometry.d_model}")
        self.geom = geometry
        self.act_dtype = sample_dy.dtype
        self.device = sample_dy.device
        self.recompute = recompute
        self.need_dh, self.need_dw_qkvg, self.need_dw_o = bool(need_dh), bool(need_dw_qkvg), bool(need_dw_o)
        self.seq_lens_present = bool(seq_lens_present)
        self.dw_norm_dtype = dw_norm_dtype
        self.fuse_gate_bwd = bool(fuse_gate_bwd)
        # The declaration's samples, re-read by check_support (shapes / dtypes / the record's presence facts; no device read).
        self._samples = dict(
            dy=sample_dy,
            saved=sample_saved,
            w_qkvg=sample_w_qkvg,
            w_q_norm=sample_w_q_norm,
            w_k_norm=sample_w_k_norm,
            cos=sample_cos,
            sin=sample_sin,
            w_o=sample_w_o,
        )
        g, act, b, s = geometry, self.act_dtype, self.batch, self.seq_len
        t, dm, hd, n = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg
        # Stages, in launch order.  Building them costs no device work; the GEMM stages get a plan at compile().
        self._out_proj_dgrad = _OutProjDgrad(m=t, k=dm, n=hd, dtype=act, label="out_proj_dgrad")
        self._gate_bwd = _SigmoidGateBwd(g, batch=b, seq_len=s, dtype=act, want_og=self.need_dw_o, want_delta=self.fuse_gate_bwd)
        self._out_proj_wgrad = _OutProjWgrad(m=dm, k=t, n=hd, dtype=act, label="out_proj_wgrad") if self.need_dw_o else None
        self._recompute_qk = _QkNormRope(g, batch=b, seq_len=s, dtype=act, want_rstd=False)
        self._compact_v = _VCompaction(g, batch=b, seq_len=s, dtype=act)
        self._sdpa = _SdpaBwd(g, batch=b, seq_len=s, dtype=act, device=self.device, external_delta=self.fuse_gate_bwd)
        self._norm_bwd = _QkNormRopeBwd(g, batch=b, seq_len=s, dtype=act, want_dw=self.need_dw_norms)
        self._qkv_gate_wgrad = _QkvGateWgrad(m=n, k=t, n=dm, dtype=act, label="qkv_gate_wgrad") if self.need_dw_qkvg else None
        self._qkv_gate_dgrad = _QkvGateDgrad(m=t, k=n, n=dm, dtype=act, label="qkv_gate_dgrad") if self.need_dh else None
        self._stages = [
            st
            for st in (
                self._out_proj_dgrad,
                self._gate_bwd,
                self._out_proj_wgrad,
                self._recompute_qk,
                self._compact_v,
                self._sdpa,
                self._norm_bwd,
                self._qkv_gate_wgrad,
                self._qkv_gate_dgrad,
            )
            if st is not None
        ]
        self._ws: Optional[_BwdIntermediates] = None

    # -- facts ------------------------------------------------------------------

    @property
    def gemm_plans(self) -> dict:
        """``{label: ProjGemmPlan}`` of the compiled GEMM stages (the tests pin tile / route / majors)."""
        return {st.label: st.plan for st in self._stages if isinstance(st, _GemmStage)}

    def _check_inputs(self, dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o) -> None:
        """The non-record operands: shape / dtype / device / contiguity / 16-B alignment, typed, naming the operand.
        Every one is a TMA-loaded GEMM operand or a 16-B vector-load operand of an elementwise kernel."""
        g, b, s, act, dev = self.geom, self.batch, self.seq_len, self.act_dtype, self.device
        check = GatedAttentionBlockFwd._check_saved_tensor
        check("dy", dy, (b, s, g.d_model), act, dev)
        check("w_qkvg", w_qkvg, (g.n_qkvg, g.d_model), act, dev)
        check("w_o", w_o, (g.d_model, g.h_q * g.d_head), act, dev)
        _check_norm_weights_agree(g.qk_norm, w_q_norm, w_k_norm)
        if g.qk_norm:
            check("w_q_norm", w_q_norm, (g.d_head,), act, dev)
            check("w_k_norm", w_k_norm, (g.d_head,), act, dev)
        check("cos", cos, (b, s, g.rope_dim), act, dev)
        check("sin", sin, (b, s, g.rope_dim), act, dev)

    # -- support ------------------------------------------------------------

    def check_support(self) -> bool:
        """Validate the declaration, the saved set against the geometry and the
        policy, then ask every enabled stage.  Every decline is typed and, where
        the row the block binds is the reason, names it -- in this order:

        ``geometry.validate()``; the activation dtype (bf16 / fp16);
        ``RecomputePolicy.RECOMPUTE_GATE`` (reserved -- the GATE is always
        saved); a gate-copy record (``saved.proj_slab`` None: a follow-up PR); a
        ``need_*`` combination that leaves no work; ``dw_norm_dtype`` other than
        fp32; padding (``seq_lens_present`` or ``sample_saved.seq_lens`` -- the
        ``sdpa_bwd_sm107`` row declines it, a follow-up PR flips it; no device read); the record
        buffers (shape / dtype / contiguity / 16-B alignment, the slab's bands
        aliasing); the operands; ``seq_len >= 2`` (S_q = 1 is decode, which the
        adapter would refuse with an untyped ``ValueError``); ``rope_dim > 0``;
        the TMA 16-byte rule on the MN-major GEMM operands; the forced GEMM
        tile's precondition (``d_model % 256 == 0`` -- the determinism
        contract's premise, never a silent heuristic fallback); the mask knobs
        the row cannot serve (``window_left == 0``, ``window_right > 0``); Rubin
        only; then each stage's own ``check_support`` (the adapter's contract
        last).

        After ``compile()`` the declaration's sample tensors are released (the
        convenience wrapper caches the block for the process lifetime; it must
        hold artifacts, not the first call's buffers), so a repeated call is a
        no-op returning the already-established ``True``.
        """
        if self._samples is None:
            return True
        g, act = self.geom, self.act_dtype
        g.validate()
        if act not in _ACT_DTYPES:
            raise NotImplementedError(f"gated_attention_block backward serves bf16 / fp16 only (the elementwise kernels and the sdpa_bwd_sm107 row), got {act}")
        if self.recompute is RecomputePolicy.RECOMPUTE_GATE:
            raise NotImplementedError(
                "RecomputePolicy.RECOMPUTE_GATE is reserved: the training forward always saves the GATE (as saved.gate or as a proj_slab band), "
                "so there is nothing to drop; use SAVE_ALL or RECOMPUTE_QK_PRE"
            )
        sv = self._samples["saved"]
        if not isinstance(sv, SavedForBackward):
            raise ValueError(f"sample_saved must be a SavedForBackward record, got {type(sv).__name__}")
        if sv.proj_slab is None:
            _check_saved_record(sv, g, self.batch, self.seq_len, act, self.device, at="declaration")  # raises the typed gate-copy decline
        if not (self.need_dh or self.need_dw_qkvg or self.need_dw_o or self.need_dw_norms):
            raise ValueError("no work: need_dh, need_dw_qkvg, need_dw_o and need_dw_norms are all False -- nothing to compute")
        if self.dw_norm_dtype != torch.float32:
            raise NotImplementedError(
                f"dw_norm_dtype={self.dw_norm_dtype}: P0 writes dW_q_norm / dW_k_norm in fp32 only (the kernel's partials and its reduce are fp32); a cast "
                "would be an extra launch nobody measured -- pass torch.float32"
            )
        if self.seq_lens_present or sv.seq_lens is not None:
            raise NotImplementedError(
                "padding (seq_lens) is not served by the block backward yet: the sdpa_bwd_sm107 row declines seq_kv_lens_present, so a padded "
                "forward's save set (saved.seq_lens is a tensor, or seq_lens_present=True) is declined here at declaration -- a follow-up PR flips it "
                "with the row's `padded` capability"
            )
        _check_saved_record(sv, g, self.batch, self.seq_len, act, self.device, at="declaration")
        self._check_inputs(*(self._samples[k] for k in ("dy", "w_qkvg", "w_q_norm", "w_k_norm", "cos", "sin", "w_o")))
        if self.seq_len < 2:
            raise NotImplementedError(
                f"seq_len={self.seq_len}: the sdpa_bwd_sm107 row's prefill bodies serve S_q >= 2 (S_q = 1 is decode, out of the block backward's scope)"
            )
        if g.rope_dim == 0:
            raise NotImplementedError(
                "rope_dim=0 (no RoPE) is not exercised by the block backward yet; the norm+RoPE backward kernel is traced with rope_dim > 0"
            )
        elems16 = 16 // _itemsize(act)
        for label, extent in (("d_model", g.d_model), ("h_q * d_head", g.h_q * g.d_head), ("n_qkvg", g.n_qkvg)):
            if extent % elems16:
                raise ValueError(
                    f"{label}={extent} must be a multiple of {elems16} ({act}): it is the contiguous extent of an MN-major GEMM operand (the TMA 16-byte rule)"
                )
        from .kernels.proj_gemm import _forced_tile_config

        for label, extent in (("d_model", g.d_model), ("h_q * d_head", g.h_q * g.d_head)):
            if _forced_tile_config(extent) is None:
                raise NotImplementedError(
                    f"{label}={extent} is not a multiple of 256: the four backward GEMMs run the block's FORCED 256-wide N tile "
                    "(kernels/proj_gemm.py::_forced_tile_config) at one split-K slice -- the premise of the bitwise-determinism contract -- "
                    "and the block serves no heuristic fallback (a different tile, route and possibly a split-K reducer)"
                )
        if g.window_left == 0:
            raise NotImplementedError(
                "geometry.window_left=0 is not served: the sdpa_bwd_sm107 row needs window_left > 0 (a zero-width window keeps only the diagonal); use -1 for unbounded"
            )
        if g.window_right > 0:
            raise NotImplementedError(
                f"geometry.window_right={g.window_right} is not served: the sdpa_bwd_sm107 row accepts an unbounded (or 0) right band only"
            )
        dev = torch.device(self.device)
        cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
        if cc != _SM107_CC:
            raise NotImplementedError(
                f"gated_attention_block backward targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found "
                + (f"SM{cc[0]}{cc[1]}" if cc is not None else str(dev))
            )
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    # -- workspace ----------------------------------------------------------

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide, INCLUDING the reused SDPA backward's
        scratch (it carves from the same buffer, at an offset this layout
        reserves for it) and the GEMMs' scratch.

        **Requires ``compile()`` first** (``RuntimeError`` otherwise): the dW
        partial plane rows come from the compiled recipe's SM-fill cap, and the
        four ``plan.workspace_bytes`` exist only after the plans do -- the same
        order the forward's own ``get_workspace_size`` imposes
        (``check_support(); compile(); get_workspace_size()``).

        Closed form (bf16, ``e = 2``, per token): ``(6*H_q + 6*H_kv)*D*e`` block
        regions -- ``dO``, the ``dqkvg`` slab ``(2*H_q + 2*H_kv)*D``, ``o_gated``, the
        recomputed Q / K / V, compact ``dQ`` / ``dK`` / ``dV`` (``(5*H_q + 6*H_kv)*D*e``
        without ``o_gated``, i.e. ``need_dw_o=False``) -- + the adapter's
        ``delta + dv_part + dk_part`` (32 KiB/token at 397B) + ONE dS chunk
        ``qh_chunk x S_q_pad x S_kv_pad x e`` (4.25 / 8.50 / 33.0 GiB at S = 8K /
        16K / 32K, 397B, B=1) + ``(n_ctas_q + n_ctas_k) x D x 4`` dW partials +
        ``max(plan.workspace_bytes)`` (12 MiB at the test geometry, 0 at 397B).
        Under ``fuse_gate_bwd`` the adapter's ``delta`` moves out of its scratch
        into the block's own ``delta`` region of the same size (``B x H_q x S_pad x 4``).
        Honest and never exceeded.
        """
        if self._ws is None:
            raise RuntimeError(
                "GatedAttentionBlockBwd.get_workspace_size() needs compile() first: the dW partial plane rows and the four GEMM plans' "
                "workspace_bytes exist only once the artifacts do (check_support(); compile(); get_workspace_size())"
            )
        return int(self._ws.total_bytes)

    def _layout(self) -> _BwdIntermediates:
        if self._ws is None:
            raise RuntimeError("call compile() before _layout()")
        return self._ws

    # -- compile ------------------------------------------------------------

    def compile(self) -> None:
        """Build the artifacts for the enabled stages only, then the workspace carve."""
        self._ensure_support_checked()
        for st in self._stages:
            st.compile()
        # The determinism contract's premise, verified on the plans that will run: the forced tile compiled (plan.jit) under
        # its own name. The driver logs a WARNING and takes the graph heuristic when the forced config is refused for a
        # shape -- a different tile, route and possibly a split-K reducer -- which the block does not serve.
        from .kernels.proj_gemm import _forced_tile_config

        for st in self._stages:
            if isinstance(st, _GemmStage):
                want, plan = _forced_tile_config(st.n), st.plan
                if plan.jit is None or plan.tile_config_name != want:
                    raise NotImplementedError(
                        f"{st.label} (m={st.m}, k={st.k}, n={st.n}, {st.dtype}): the forced tile {want} did not compile for this shape and the "
                        f"plan fell back to the heuristic (tile={plan.tile_config_name}, route={plan.route}); the block backward serves the "
                        "forced tile only -- the premise of its bitwise-determinism contract (module docstring, Determinism)"
                    )
        n_ctas_q, n_ctas_k = (0, 0)
        if self.need_dw_norms:
            n_ctas_q, n_ctas_k, _n_v = self._norm_bwd.n_ctas()
        gemm_scratch = max([st.workspace_bytes() for st in self._stages if isinstance(st, _GemmStage)] + [1])
        self._ws = _plan_bwd_workspace(
            self.geom,
            self.batch,
            self.seq_len,
            self.act_dtype,
            self.recompute,
            need=dict(dw_o=self.need_dw_o, dw_norms=self.need_dw_norms),
            sdpa_bwd_bytes=self._sdpa.scratch_workspace_bytes(),
            gemm_scratch_bytes=gemm_scratch,
            n_ctas_q=n_ctas_q,
            n_ctas_k=n_ctas_k,
            delta_shape=self._sdpa.delta_shape if self.fuse_gate_bwd else None,
        )
        self._compiled_kernel = self._ws  # APIBase's "compiled" marker
        # The declaration's tensors are not needed past here: hold artifacts and facts, never the sample buffers (the
        # convenience wrapper caches this object for the process lifetime -- at 397B, S=32K that would pin ~1.6 GiB of
        # the first call's proj_slab / dy / h / o).
        self._samples = None

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        dy: torch.Tensor,
        saved: SavedForBackward,
        w_qkvg: torch.Tensor,
        w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        dh: Optional[torch.Tensor] = None,
        dw_qkvg: Optional[torch.Tensor] = None,
        dw_o: Optional[torch.Tensor] = None,
        dw_q_norm: Optional[torch.Tensor] = None,  # must be None under qk_norm=False (need_dw_norms resolves False)
        dw_k_norm: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the enabled stages, in this order, onto the ONE launch stream
        (module docstring: the launch table).  Everything is stream-ordered, so
        nothing overlaps anything else; B1 is issued right after B3 because
        that is when its operand exists, and it depends on nothing below it --
        that independence is what makes it the candidate filler::

            (B2) out_proj_dgrad     dy, w_o                     -> ws.do_gated
            (B3) sigmoid_gate_bwd   ws.do_gated, saved.o,
                                    proj_slab[GATE]             -> ws.do_gated (in place), ws.dqkvg[GATE], ws.o_gated
                                                                   (+ ws.delta = rowsum(dO * O) under fuse_gate_bwd)
            (B1) out_proj_wgrad     dy, ws.o_gated              -> dw_o
                 qk_norm_rope       proj_slab[Q], [K]           -> ws.recompute, ws.recompute_k   (post-norm / post-RoPE)
                 compact_v          proj_slab[V]                -> ws.recompute_v
            (B4) sdpa_bwd           ws.do_gated, ws.recompute*,
                                    saved.o, saved.lse
                                    (+ ws.delta: no dot_do_o)   -> ws.dq, ws.dk, ws.dv   (COMPACT)
            (B5+B6) qk_norm_rope_bwd ws.dq/dk/dv, proj_slab[Q], [K],
                                    saved.rstd_*, w_*_norm      -> ws.dqkvg[Q], [K], [V]; ws.dw_partials_*
                 dw_norm reduce     ws.dw_partials_*            -> dw_q_norm, dw_k_norm       (qk_norm only)
            (B7) qkv_gate_wgrad     ws.dqkvg, saved.h           -> dw_qkvg
            (B8) qkv_gate_dgrad     ws.dqkvg, w_qkvg            -> dh

        A ``need_*`` that was False at build time means the corresponding output
        argument must be ``None`` here -- a provided-but-uncompiled tensor raises
        rather than being silently ignored (Rule 1, both directions). The norm
        weights follow ``geometry.qk_norm`` the same way (both ``None`` iff
        False). ``seq_lens`` is refused (the block was declared without padding).

        No allocation, no D2H read, no implicit conversion. The record and every
        operand are re-validated (host-only) on each call, and no written buffer
        may overlap any other buffer the backward binds.
        """
        if self._ws is None:
            raise RuntimeError("call compile() before execute()")
        g, b, s, act, dev = self.geom, self.batch, self.seq_len, self.act_dtype, self.device
        t, dm, hd, n, d = b * s, g.d_model, g.h_q * g.d_head, g.n_qkvg, g.d_head
        if seq_lens is not None:
            raise ValueError(
                "seq_lens was given but this block was declared without padding (seq_lens_present=False; P0 declines it anyway -- the "
                "sdpa_bwd_sm107 row serves no seq_lens): pass None"
            )
        # Outputs, both directions (Rule 1).
        check = GatedAttentionBlockFwd._check_saved_tensor
        for name, need, ten, shape, dtype, knob in (
            ("dh", self.need_dh, dh, (b, s, dm), act, "need_dh"),
            ("dw_qkvg", self.need_dw_qkvg, dw_qkvg, (n, dm), act, "need_dw_qkvg"),
            ("dw_o", self.need_dw_o, dw_o, (dm, hd), act, "need_dw_o"),
            ("dw_q_norm", self.need_dw_norms, dw_q_norm, (d,), self.dw_norm_dtype, "need_dw_norms"),
            ("dw_k_norm", self.need_dw_norms, dw_k_norm, (d,), self.dw_norm_dtype, "need_dw_norms"),
        ):
            if need and ten is None:
                raise ValueError(f"{name} is required: this block was declared with {knob}=True")
            if not need and ten is not None:
                raise ValueError(
                    f"{name} was given but this block was declared with {knob}=False; a provided-but-uncompiled output is refused rather than silently ignored"
                )
            if need:
                if name.startswith("dw_") and name.endswith("_norm") and ten.dtype != dtype:
                    raise ValueError(f"{name} must be {dtype} (dw_norm_dtype={self.dw_norm_dtype}; P0 serves fp32 only), got {ten.dtype}")
                check(name, ten, shape, dtype, dev)
        if workspace is None:
            raise ValueError("workspace is required: get_workspace_size() bytes of uint8 on dy's device")
        req = self.get_workspace_size()
        if workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.device != dev:
            raise ValueError(
                f"workspace must be a contiguous uint8 tensor on {dev}, got {workspace.dtype} strides {tuple(workspace.stride())} on {workspace.device}"
            )
        if workspace.numel() < req:
            raise ValueError(f"workspace is {workspace.numel()} bytes, need {req}")
        if workspace.data_ptr() % self._ws.base_align:
            raise ValueError(f"workspace base must be {self._ws.base_align}-byte aligned (the carve assumes it), got data_ptr={workspace.data_ptr():#x}")
        proj, o_flat = _check_saved_record(saved, g, b, s, act, dev, at="execute")
        self._check_inputs(dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o)
        _check_no_overlap(
            [
                (nm, ten)
                for nm, ten in (("dh", dh), ("dw_qkvg", dw_qkvg), ("dw_o", dw_o), ("dw_q_norm", dw_q_norm), ("dw_k_norm", dw_k_norm), ("workspace", workspace))
                if ten is not None
            ],
            [
                ("dy", dy),
                ("saved.h", saved.h),
                ("saved.o", saved.o),
                ("saved.lse", saved.lse),
                ("saved.rstd_q", saved.rstd_q),
                ("saved.rstd_k", saved.rstd_k),
                ("saved.proj_slab", saved.proj_slab),
                ("w_qkvg", w_qkvg),
                ("w_o", w_o),
                ("w_q_norm", w_q_norm),
                ("w_k_norm", w_k_norm),
                ("cos", cos),
                ("sin", sin),
            ],
        )
        # THE launch stream (Rule 5): the caller's, else torch's current stream on dy's device -- resolved once and handed
        # to EVERY stage (the CuTe-DSL kernels and the GEMM drivers take the raw int; the adapter a CUstream of it).
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(dev).cuda_stream
        ws = self._ws
        do_gated = _view(workspace, ws.do_gated, (t, g.h_q, d), act)
        dqkvg = _view(workspace, ws.dqkvg, (t, n), act)
        o_gated = _view(workspace, ws.o_gated, (t, g.h_q, d), act) if self.need_dw_o else None
        rq = _view(workspace, ws.recompute, (t, g.h_q, d), act)
        rk = _view(workspace, ws.recompute_k, (t, g.h_kv, d), act)
        rv = _view(workspace, ws.recompute_v, (t, g.h_kv, d), act)
        dq = _view(workspace, ws.dq, (t, g.h_q, d), act)
        dk = _view(workspace, ws.dk, (t, g.h_kv, d), act)
        dv = _view(workspace, ws.dv, (t, g.h_kv, d), act)
        plane_q = _view(workspace, ws.dw_partials_q, (ws.n_ctas_q, d), torch.float32) if self.need_dw_norms else None
        plane_k = _view(workspace, ws.dw_partials_k, (ws.n_ctas_k, d), torch.float32) if self.need_dw_norms else None
        sdpa_ws = workspace[ws.sdpa_bwd_ws : ws.sdpa_bwd_ws + ws.sdpa_bwd_bytes]
        gemm_ws = workspace[ws.gemm_scratch : ws.gemm_scratch + ws.gemm_scratch_bytes]
        delta = _view(workspace, ws.delta, ws.delta_shape, torch.float32) if self.fuse_gate_bwd else None
        o_q, o_g, o_k, o_v = g.qkvg_offsets
        q_pre_b = _cols(proj, o_q, g.h_q, d)
        gate_b = _cols(proj, o_g, g.h_q, d)
        k_pre_b = _cols(proj, o_k, g.h_kv, d)
        v_b = _cols(proj, o_v, g.h_kv, d)
        dy2 = dy.view(t, dm)

        # (B2) dO_gated = dY @ W_o
        self._out_proj_dgrad.execute(dy2, w_o, do_gated.view(t, hd), gemm_ws, stream=stream)
        # (B3) dO (in place), dG -> the GATE band, O_gated (need_dw_o), delta = rowsum(dO * O) (fuse_gate_bwd)
        self._gate_bwd.execute(do_gated, o_flat, gate_b, do_gated, _cols(dqkvg, o_g, g.h_q, d), o_gated, stream=stream, delta=delta)
        # (B1) dW_o = dY^T @ O_gated -- the filler, issued as soon as its operand exists
        if self.need_dw_o:
            self._out_proj_wgrad.execute(dy2, o_gated.view(t, hd), dw_o, gemm_ws, stream=stream)
        # Q / K rebuilt post-norm / post-RoPE from the slab bands (the forward's stage (2)+(3) kernel; rstd recomputed by
        # the SAME kernel over the SAME inputs = the forward's), V compacted -- the adapter's operands must be BSHD-physical.
        self._recompute_qk.execute(q_pre_b, k_pre_b, w_q_norm, w_k_norm, cos, sin, q_out=rq, k_out=rk, current_stream=stream)
        self._compact_v.execute(v_b, rv, current_stream=stream)
        # (B4) the Rubin d256 backward chain -> COMPACT dQ / dK / dV
        self._sdpa.execute(
            rq.view(b, s, g.h_q, d),
            rk.view(b, s, g.h_kv, d),
            rv.view(b, s, g.h_kv, d),
            saved.o,
            do_gated.view(b, s, g.h_q, d),
            saved.lse,
            dq.view(b, s, g.h_q, d),
            dk.view(b, s, g.h_kv, d),
            dv.view(b, s, g.h_kv, d),
            workspace=sdpa_ws,
            stream=stream,
            delta=delta,
        )
        # (B5+B6) RoPE^T + RMSNorm backward into the Q / K bands, dV into the V band, fp32 dW partials
        norm = g.qk_norm
        self._norm_bwd.execute(
            dq,
            dk,
            dv,
            q_pre_b if norm else None,
            k_pre_b if norm else None,
            saved.rstd_q.view(t, g.h_q) if norm else None,
            saved.rstd_k.view(t, g.h_kv) if norm else None,
            w_q_norm,
            w_k_norm,
            cos.view(t, g.rope_dim),
            sin.view(t, g.rope_dim),
            _cols(dqkvg, o_q, g.h_q, d),
            _cols(dqkvg, o_k, g.h_kv, d),
            _cols(dqkvg, o_v, g.h_kv, d),
            plane_q,
            plane_k,
            stream=stream,
        )
        if self.need_dw_norms:
            self._norm_bwd.reduce(plane_q, plane_k, dw_q_norm, dw_k_norm, stream=stream)
        # (B7) dW_qkvg = dQKVG^T @ h
        if self.need_dw_qkvg:
            self._qkv_gate_wgrad.execute(dqkvg, saved.h.view(t, dm), dw_qkvg, gemm_ws, stream=stream)
        # (B8) dh = dQKVG @ W_qkvg
        if self.need_dh:
            self._qkv_gate_dgrad.execute(dqkvg, w_qkvg, dh.view(t, dm), gemm_ws, stream=stream)


# ---------------------------------------------------------------------------
# 6. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------

_BWD_CACHE: dict = {}


def _detach(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """A view sharing storage that the kernels can take through DLPack (a tensor with ``requires_grad`` cannot be exported)."""
    return None if x is None else x.detach()


def gated_attention_block_backward(
    dy: torch.Tensor,
    saved: SavedForBackward,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    recompute: RecomputePolicy = RecomputePolicy.RECOMPUTE_QK_PRE,
    current_stream: Optional[cuda.CUstream] = None,
    fuse_gate_bwd: bool = False,
) -> TupleDict:
    """Allocate gradients + workspace, cache the compiled block, and run it.

    Returns ``{"dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"}``. Which
    entries are non-``None`` follows ``requires_grad`` on the corresponding
    forward inputs (``saved.h``, ``w_qkvg``, ``w_o``, ``w_q_norm`` / ``w_k_norm``),
    snapshotted here -- the same policy a torch autograd ``Function`` would
    apply, and the natural place to wire one later; nothing requiring a
    gradient is a ``ValueError``. ``dw_q_norm`` / ``dw_k_norm`` are ``None``
    under ``geometry.qk_norm=False``. The tensors are handed to the block
    DETACHED (views, no copy). The compiled block is cached per declaration
    (shapes, dtypes, device, geometry, needs, policy); the workspace and the
    gradients are allocated per call -- this is the convenience path, the class
    is the allocation-free one. Both are allocated, and the block is run, on the
    LAUNCH stream (``current_stream``, else torch's current stream): the caching
    allocator orders a buffer's reuse only against the stream it was allocated
    on, so a workspace allocated on the ambient stream for a side-stream launch
    would be freed into the ambient pool at return and handed to the caller's
    next allocation while the backward is still writing it.  ``fuse_gate_bwd``
    (appended, default off) is the block's fusion knob, part of the cache key.
    """
    need_dh = bool(saved.h.requires_grad)
    need_dw_qkvg = bool(w_qkvg.requires_grad)
    need_dw_o = bool(w_o.requires_grad)
    need_dw_norms = bool(geometry.qk_norm and ((w_q_norm is not None and w_q_norm.requires_grad) or (w_k_norm is not None and w_k_norm.requires_grad)))
    if not (need_dh or need_dw_qkvg or need_dw_o or need_dw_norms):
        raise ValueError("gated_attention_block_backward: nothing requires a gradient (saved.h, w_qkvg, w_o, w_q_norm / w_k_norm all have requires_grad=False)")
    saved_d = dataclasses.replace(
        saved,
        h=_detach(saved.h),
        gate=_detach(saved.gate),
        o=_detach(saved.o),
        lse=_detach(saved.lse),
        rstd_q=_detach(saved.rstd_q),
        rstd_k=_detach(saved.rstd_k),
        q_pre=_detach(saved.q_pre),
        k_pre=_detach(saved.k_pre),
        proj_slab=_detach(saved.proj_slab),
        seq_lens=saved.seq_lens,
    )
    dy_d, w_qkvg_d, w_q_d, w_k_d, cos_d, sin_d, w_o_d = (_detach(x) for x in (dy, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o))
    key = (
        tuple(dy.shape),
        dy.dtype,
        str(dy.device),
        tuple(sorted(dataclasses.asdict(geometry).items())),
        need_dh,
        need_dw_qkvg,
        need_dw_o,
        need_dw_norms,
        recompute,
        seq_lens is not None or saved.seq_lens is not None,
        bool(fuse_gate_bwd),
    )
    blk = _BWD_CACHE.get(key)
    if blk is None:
        blk = GatedAttentionBlockBwd(
            dy_d,
            saved_d,
            w_qkvg_d,
            w_q_d,
            w_k_d,
            cos_d,
            sin_d,
            w_o_d,
            geometry,
            recompute=recompute,
            need_dh=need_dh,
            need_dw_qkvg=need_dw_qkvg,
            need_dw_o=need_dw_o,
            need_dw_norms=need_dw_norms,
            seq_lens_present=(seq_lens is not None) or (saved.seq_lens is not None),
            fuse_gate_bwd=fuse_gate_bwd,
        )
        blk.check_support()
        blk.compile()
        _BWD_CACHE[key] = blk
    dev = dy.device
    # Rule 5 / recipe R2: the per-call scratch and the gradients are allocated, and the block is launched, on the
    # LAUNCH stream.  The workspace reference dies at return; allocated on torch's ambient stream while
    # ``current_stream`` names a side stream it would be freed into the ambient pool and could back the caller's next
    # allocation while the backward is still writing it (silent gradient corruption under load).  ``stream_context``
    # is a no-op for ``None`` and for a handle equal to torch's current stream.
    with stream_context(current_stream, dev):
        workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=dev)
        dh = torch.empty_like(saved_d.h) if need_dh else None
        dw_qkvg = torch.empty_like(w_qkvg_d) if need_dw_qkvg else None
        dw_o = torch.empty_like(w_o_d) if need_dw_o else None
        dw_q_norm = torch.empty(geometry.d_head, dtype=torch.float32, device=dev) if need_dw_norms else None
        dw_k_norm = torch.empty(geometry.d_head, dtype=torch.float32, device=dev) if need_dw_norms else None
        blk.execute(
            dy_d,
            saved_d,
            w_qkvg_d,
            w_q_d,
            w_k_d,
            cos_d,
            sin_d,
            w_o_d,
            dh=dh,
            dw_qkvg=dw_qkvg,
            dw_o=dw_o,
            dw_q_norm=dw_q_norm,
            dw_k_norm=dw_k_norm,
            workspace=workspace,
            seq_lens=seq_lens,
            current_stream=current_stream,
        )
    return TupleDict(dh=dh, dw_qkvg=dw_qkvg, dw_o=dw_o, dw_q_norm=dw_q_norm, dw_k_norm=dw_k_norm)
