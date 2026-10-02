# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapters over the FROST SM107 (Rubin) d=256 SDPA backward chains.

Three rows share this module -- ``sdpa_bwd_sm107`` (bf16 / fp16, :class:`SdpaBwdDslSm107`),
``sdpa_bwd_sm107_fp8`` (per-tensor FP8 E4M3, :class:`SdpaBwdDslSm107Fp8`) and
``sdpa_bwd_sm107_mxfp8`` (block-scale MXFP8, :class:`SdpaBwdDslSm107Mxfp8`; its own
section at the end of this doc) -- because they share one chain shape.  Each is a
TWO-kernel backward around a dS workspace (bf16 / fp16 on the half row; e4m3 -- or bf16,
the A/B twin -- on the fp8 row; bf16 on the MXFP8 row's P-c chain), followed by the two
gradient GEMMs and a fold:

    stage 1  delta = rowsum(dO * O)                      bprop_chain_common.dot_do_o{,_scaled}_host
    stage 2  dV (in TMEM, stored per Q head) + dS -> a   sm107/bprop_d256_{f16,fp8}.py
             [B, H_chunk, S_kv, S_q] GMEM workspace
    stage 3  dK = dS . Q,  dQ = dS^T . K                 bprop_matmul_blackwell.py (two renderings at the
                                                          d = 256 cluster tile: 2x1, 256 x 256, no N padding;
                                                          the fp8 row renders its K64 fp8 arm with a descale /
                                                          quantize epilogue)
    stage 4  GQA fold of the per-Q-head dK / dV partials  dkv_reduce_host (half) /
             (+ descale, amax, scale, cast on the fp8 row)  fold_quant_host (fp8: dV always, dK under GQA)

The workspace is KV-MAJOR (``[.., S_kv, S_q]``, q contiguous) -- the layout the stage-2
kernel writes without a transpose -- so the stage-3 operand majors are the OPPOSITE of
the SM100 chain's ``[S_q, S_kv]`` workspace: dK reads dS as ``A[kv, q]`` K-major
(``a_is_m_major=False``) and dQ reads dS^T as ``A[q, kv]`` M-major (``a_is_m_major=True``).
The causal K-trim MODES do not flip with the layout (they follow which axis is the
output row): dK trims the low q tiles (``CAUSAL_K_LO``), dQ the high kv blocks
(``CAUSAL_K_HI``), rounded to the kernel's 256-row kv block; a sliding window adds the
band's second edge to both (``MatmulTemplateParams.causal_window``: dK ends after the
window, dQ starts at it), so each GEMM visits only the band's k tiles.  The trim is a
CORRECTNESS-neutral subset of what stage 2 wrote: the bodies round every kv block's q
range outward to the GEMMs' 256-row pair (``config_sm107.q_write_tiles``) and every
GEMM bound rounds outward inside it (``bprop_matmul_blackwell._causal_k_range``), so no
mask needs the workspace zero-filled -- ``_stage3_needs_zero_fill`` keeps the per-execute
fill only for the untrimmed twin and for the one geometry a q pair is written by no kv
block (a top-left window with ``S_q > roundup(S_kv + W, 256)``); the poisoned-workspace
tests are the proof (``test_masked_stage3_reads_only_what_stage2_wrote``).

Both renderings take the d = 256 cluster tile (``MatmulTemplateParams.cgrp_tile_mn =
(256, 256)``, ``_stage3_cgrp_tile_mn``): cluster 2x1, one 256-row x 256-col tile per
2-CTA pair, six 32 KiB operand stages, the 256-column accumulator double-buffered in
TMEM.  The SM100 chain's (512, 512) tile would spend the N-rank pair of every cluster
on columns 256..511 that a d = 256 problem does not have -- half of every cluster's
MMA work as TMA-OOB zeros and clipped stores (measured 23 % of the bf16 peak in
useful work).  The rule is this adapter's only: the SM100 adapter never sets the
field, so its d512 renderings are unchanged (``STAGE3_D256_TILE`` is the bitwise
pin's twin).

Shapes: the kernels compile with EVERY extent concrete and TMA-strided, and require
``S_q % 128 == 0`` and ``S_kv % 256 == 0``.  A graph that is not a multiple is served by
padding: Q / dO (and the LSE, with ``+inf`` so ``P = 0``) are staged into zero-filled
padded copies carved from the workspace; K / V likewise, and a padded S_kv selects the
kernels' padded-mask specialization with the uniform real length, so every padded kv
row's dS / dV is exactly zero.  The GEMMs then read real-extent slices and write the
caller's tensors directly; only a padded-kv GQA graph folds through a padded staging
copy.  Everything the chain needs is carved from the caller's workspace in one fixed
order (:meth:`_scratch_plan`), so ``scratch_workspace_bytes()`` is a build-time function.

Per-batch kv lengths (``seq_kv_lens_present=True`` at construction, then
``execute(seq_kv_lens=<[B] int32>)``): the HALF row binds the caller's lengths in place of
the uniform fill and the same padded-mask arm reads ``seq_kv_lens[b]`` -- a kv row at or
past its batch's length is select-dead (P = 0 -> dS = dV = 0 exactly; a zero length is a
dead batch whose dQ / dK / dV come back as exact zeros, whatever its LSE holds).
``seq_kv_lens[b]`` must satisfy ``0 <= len <= S_kv``: device data the host cannot validate
without a synchronization, so an out-of-range value is the caller's contract violation (as
on the forward).  Under bottom-right causal the diagonal becomes per batch
(``seq_kv_lens[b] - S_q``, the forward's convention) while the stage-3 K-trim is computed
from the uniform ``S_kv - S_q``, and three rules keep the GEMMs reading only zeros or what
the kernel wrote: :func:`_stage3_needs_zero_fill` keeps the dS zero-fill for exactly that
arm (the trim can reach tiles a shorter batch's band did not write);
:func:`_stage3_trim_window` drops a sliding window from the trim (a window edge anchored on
the uniform diagonal would SKIP live tiles of a shorter batch -- dQ / dK missing, finite, no
crash); and the fill runs ahead of EVERY batch / head chunk rather than once per execute
(``prepared_host.host_f16``: the skipped set is per batch, so a chunk's workspace slot may
hold the previous batch's dS in tiles the next batch's narrower band does not write).  A
top-left band does not move with the length and needs none of the three.  The fp8 / MXFP8
bodies take ONE uniform ``seqlen_kv_real``, so their adapters
decline per-batch lengths; no body threads per-batch Q lengths (``seq_q_lens``), and a
GRAPH padding mask always carries ``seq_len_q`` as well (the frontend requires both), which
is why every row keeps ``Capabilities.padded = False`` and the graph form stays declined at
eligibility rather than served while ignoring the q lengths.

One exception, declined rather than served wrong: **bottom-right causal on the fp8 row
needs ``S_q % 128 == 0``.**  The bottom-right diagonal is ``S_kv - S_q`` in REAL rows.
The f16 body takes the real lengths (``sq_real`` / ``skv_real`` on its ``compile()``,
``SQ_REAL`` in its problem_size); the fp8 body's ABI has no ``seqlen_q_real`` -- it
derives the diagonal and the q-tile trim from the PADDED compile extent ``SQ`` (its kv
term IS the real length, the runtime ``seqlen_kv_real``), so a ragged S_q would shift
both by ``pad - S_q`` rows: finite, wrong dQ / dK / dV near the diagonal, no crash.  The
row declines it at eligibility (``Capabilities.bottom_right_s_q_multiple``) and
:meth:`SdpaBwdDslSm107Fp8._check_support_family` backstops.  A ragged S_kv is fine on
both rows.  Follow-up: thread ``seqlen_q_real`` through the fp8 body like the f16 one
and drop the decline.

The dS workspace is the dominant allocation (``B * H * S_kv * S_q * bpe_ds`` bytes):
heads (and, on the half row, batches -- the fp8 body has no ``batch_base``) are chunked
to fit ``_SM107_WS_BUDGET_BYTES`` and the chain loops over chunks with ``head_base`` /
``batch_base``.

Both rows are PREPARED launches (``bwd/prepared.py``; ``bwd/prepared_sm107.py`` +
``kernels/sm107/prepared_host.py``): ``compile()`` builds ONE pointer-host artifact that
runs the whole chain -- the padding copies, the ``seq_kv`` fill and the dS zero-fill,
``dot``, every chunk's main kernel + stage-3 GEMMs, the fold (and the fp8 upcast / fold +
quantize passes) -- from device pointers and the caller's workspace, and records it as
``self._prepared`` (a ``BwdLaunchSpec``).  The graph plan binds the normalized variant pack
straight into it (``engines.lower_dsl_bwd*`` -> ``PreparedBwdLaunch``); :meth:`execute` is
the standalone twin over torch tensors.  No torch op runs on the execute path.

**An externally computed delta (half row only).**  ``SdpaBwdDslSm107(external_delta=True)``
declares that the caller hands stage 1's result to ``execute(..., delta_tensor=)``: a
contiguous fp32 ``[B, H_q, S_q_pad]`` tensor (``external_delta_shape``; ``S_q_pad`` =
``S_q`` rounded up to the 128-row q tile, zeros past ``S_q``) on the plan's device, 16-B
aligned, holding the RAW ``rowsum(dO * O)`` -- the gated attention block's sigmoid-gate
backward produces it while it already reads O and dO, in ``dot_do_o``'s own reduction order
(``gated_attention_block/kernels/sigmoid_gate_bwd.py``), so the fused and the unfused block are
bitwise equal.  Under the flag the chain launches no ``dot``, reads O once less, and the
workspace carve has no ``delta`` region (``scratch_workspace_bytes()`` shrinks by it); the
operand is validated like Stats before any bind (dtype, shape, strides, device, alignment --
each a typed ``ValueError``), and a plan built without the flag refuses a delta (Rule 1,
both directions).  A plan fact, not an eligibility fact: ``Capabilities`` and the graph path
are untouched (no graph declares a delta; the prepared launch frames the slot as absent).
The fp8 and MXFP8 rows decline the flag -- their delta is the dot of their own payloads
(the DEscaled fp8 dot of the scaled pre-pass; the ``o_f16`` / ``dO_f16`` ports' dot), computed
by their own pre-pass.  The flag composes with the per-batch kv lengths above: two
independent plan facts, each deciding its own appended operand slot (the lengths, then the
delta), and a plan built with both takes both at ``execute``.

FP8 (cuDNN ``sdpa_fp8_backward``): the twelve scalar descales / scales are 1-element
fp32 DEVICE tensors, read by the kernels -- never folded on the host.  Stage 2 consumes
descale_q/k/v/dO/s, scale_s and scale_dP, publishes ``dS_q = e4m3(dS * scale_dP)`` (dS
in TRUE units, attn_scale folded) to the e4m3 workspace and its per-Q-head dV in bf16
(``dtype_o = BF16``: the pre-quantization value), and folds ``amax_dP`` in-kernel over the
fp32 dS BEFORE the scale and the cast.  The GEMMs render the template's fp8 K64 arm over
the e4m3 dS and the e4m3 Q / K payloads (no upcast copies) and undo both scalings in their
epilogue: dQ = ``QUANT`` straight into the caller's dQ (``acc * descale_dP * descale_k``
-> ``amax_dQ`` -> ``* scale_dQ`` -> the gradient dtype); dK likewise into the caller's dK
at MHA, or ``DESCALE`` to bf16 per-Q-head TRUE-unit partials under GQA, which stage 4 sums
in fixed order BEFORE it folds ``amax_dK``, applies ``scale_dK`` and casts.  dV always
goes through stage 4 (fold + ``amax_dV`` + ``scale_dV`` + cast).  ``FP8_DS_DTYPE = BF16``
restores the pre-quantized chain end to end (bf16 dS, bf16 GEMMs over EXACT e4m3 -> bf16
upcasts of Q / K, three fold + quantize passes; ``descale_dP`` / ``scale_dP`` unused) --
the A/B and oracle twin.

MXFP8 (cuDNN ``sdpa_mxfp8_backward``):
ONE kernel plus scale-factor plumbing (``sm107/bprop_d256_mxfp8.py``, the fp8 body's
pipeline with the F8_128x4 E8M0 scale factors dequantizing INSIDE every block-scale MMA).
The kernel consumes the ROWWISE payloads q / k / v / dO with their SF and the COLUMNWISE
dO_T with its D-plane-major SF; q_T / k_T (and their SF) are stage-3 operands only.  P is
quantized with the fixed 2^8 scale (byte 119; ``p_scale_log2`` pinned to 8), dV leaves the
kernel as bf16 TRUE-unit per-Q-head partials, dS = attn_scale * P (dP - delta) from the
fp32 P.  **dS policy P-b** (``MXFP8_DS_SF_POLICY`` = ``config_sm107.DS_SF_POLICY_DEFAULT`` =
``DS_SF_P_B``, a module constant read when the adapter is built, never a knob -- what ships):
the kernel quantizes the fp32 dS to e4m3 per 32-element block along BOTH axes
and writes two payloads (``ds_dk`` per 32-q block of a kv row, ``ds_dq`` per 32-kv block of a q
column) plus their F8_128x4 E8M0 atoms (``sf_ds_dk`` / ``sf_ds_dq``); the stage-3 GEMMs render
the block-scale arm (``MatmulTemplateParams.block_scale``: dK = ds_dk . q_T with the ``sf_ds_dk``
atoms as SFA and the columnwise ``sf_q_T`` as SFB, dQ = ds_dq^T . k_T with ``sf_ds_dq`` / ``sf_k_T``),
dequantizing in the MMA -- no dequant pass, EPI_NONE, bf16 true-unit gradients; a ragged S_q /
S_kv re-stages the columnwise q_T / k_T scale factors with their pad groups zeroed (the MMA
reads whole atoms).  Rubin-line only (the arm's 576-column exclusive TMEM).  Its stage-2
workspace chunks against the same ``_SM107_WS_BUDGET_BYTES`` as every other sm107 row (8 GiB: at
2 + 2/32 bytes per dS element the 8K H=128 head chunk is 32 heads / 4 launches).  **dS policy
P-c** (``MXFP8_DS_SF_POLICY = DS_SF_P_C``): a bf16 dS
workspace and the bf16 stage-3 renderings over the EXACTLY dequantized bf16 q_T / k_T
(``prepared_host.dequant_mxfp8_to_bf16_host``) -- the oracle twin (``quantize_ds=False``), kept
selectable.  No amax outputs (a graph that requests them is declined, typed), no per-tensor scalars.  Padding adds one obligation the other rows
do not have: the producer's SF tensors cover ``ceil128(S)`` rows / groups with UNDEFINED pad
bytes, and the kernel reads them (a 0xFF is an E8M0 NaN -> NaN dV on every kv row), so
``host_mxfp8`` re-stages ``sf_q / sf_dO / sf_dO_T`` (S_q % 128 != 0) and ``sf_k / sf_v``
(S_kv % 256 != 0, grown to the kernel's 256-row pad) with the pad bytes zeroed.
Gradients are bf16 only for now (the bf16 GEMM stores its io dtype; an fp16 arm is a
follow-up).  Bottom-right causal keeps the fp8 row's ``S_q % 128 == 0`` rule (the body
derives the diagonal from its padded S_q).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import TensorDesc
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
from cudnn.sdpa.bwd import config_sm107 as _cfg
from cudnn.sdpa.bwd import prepared_sm107 as _prepared
from cudnn.frost.tile_dsl.thd import THD_BWD_MAPS_META_WORDS
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl, _SM100_MATMUL_FILE, _sm100_device_clusters, _sm100_kernel_path
from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, EPI_DESCALE, EPI_NONE, EPI_QUANT, MatmulTemplateParams, vec_bytes_epi_for
from cudnn.sdpa.fwd.api_dsl import ws_align

_SM107_D = 256
# The bodies' compile geometry: the q loop walks 128-row q tiles, a cga2 pair owns a 256-row kv block
# (``config_sm107.q_pad_rows`` / ``kv_pad_rows``).  The adapter pads S_q / S_kv up to these and stages
# the padded operands (module doc).  ``_SM107_Q_PAD`` is also the fp8 row's bottom-right alignment
# claim (``engines.Capabilities.bottom_right_s_q_multiple``; pinned equal by the fp8 suite).
_SM107_Q_PAD = 128
_SM107_KV_PAD = 256  # also the stage-3 K-trim granularity (`causal_gran`) and the kernels' q write pair (`config_sm107.q_write_tiles`)
_SM107_KERNEL_FILES = {
    _cfg.FAMILY_F16: "sm107/bprop_d256_f16.py",
    _cfg.FAMILY_FP8: "sm107/bprop_d256_fp8.py",
    _cfg.FAMILY_MXFP8: "sm107/bprop_d256_mxfp8.py",
}
_SM107_TEMPLATE_TAGS = {
    _cfg.FAMILY_F16: "sdpa_bwd_sm107_main_f16",
    _cfg.FAMILY_FP8: "sdpa_bwd_sm107_main_fp8",
    _cfg.FAMILY_MXFP8: "sdpa_bwd_sm107_main_mxfp8",
}
_SM107_MM_TAGS = {"dk": "sdpa_bwd_sm107_mm_dk", "dq": "sdpa_bwd_sm107_mm_dq"}
# The stage-2 dS workspace budget EVERY sm107 row chunks against (``_sm107_chunks``): ONE constant for the half row, the
# per-tensor fp8 row and both of the MXFP8 row's dS policies.  Above it the (batch, head) chunk shrinks and the chain runs
# more launches over the same total work; every extra launch costs a pipeline fill / drain and, under a causal mask, the
# LPT tail of a smaller head set.  Chunk arithmetic at B=1 H=128 (divisor chunks, the largest that fits):
#   S=8K : bf16 / fp16 dS (2 B per element, 128 MiB per head)                -> 64 heads / 2 launches  (4 GiB: 32 heads / 4)
#          per-tensor fp8 e4m3 dS (1 B, 64 MiB per head)                       -> 128 heads / 1 launch  (4 GiB: 64 heads / 2)
#          MXFP8 block-scaled dS (2 + 2/32 B: two payloads + atoms, 132 MiB)   -> 32 heads / 4 launches (4 GiB: 16 heads / 8)
#   S=16K: 512 / 256 / 528 MiB per head -> 16 / 32 / 8 heads: every row still chunks.
# MEASURED on Rubin (cc 10.7, 212 SMs, SM clock 2376 MHz), B=1 H=128/128 S=8192, whole row, 3 rounds, control twins
# within 0.9 %: the bf16 row at 8 GiB over 4 GiB (32-head / 4 launches -> 64-head / 2) +3.7 % causal, +0.8 % dense (within
# the control spread); the MXFP8 block-scaled chain at 8 GiB over 4 GiB (16-head -> 32-head chunks) +1.4 % dense / +9.3 %
# causal (the control twin's value; the arm read +11.6 % against a slot whose control pair spread 2 %); the bf16-dS chain
# forced from 32-head to 16-head chunks -0.9 % dense / -6.6 % causal.  The gain is the launch count: a causal chunk ends in
# an LPT tail the next chunk cannot fill.  A chunking constant only: the plan still reports the whole carve through
# ``scratch_workspace_bytes`` and the caller allocates it (a caller that cannot hold it bounds the plan with the graph's
# ``deselect_workspace_greater_than``, a typed decline before any launch); when nothing fits, ``_sm107_chunks`` returns the
# smallest legal chunk and the reported size stays honest.  The SM100 chain keeps its own ``_SM100_WS_BUDGET_BYTES``.
_SM107_WS_BUDGET_BYTES = 8 << 30
# Stage-3 causal K-trim.  True = trimmed renderings (what ships).  False = both GEMMs
# rendered ``CAUSAL_K_NONE`` -- the correctness pin's twin (bitwise-equal gradients,
# since the zero-filled workspace makes the trim an optimization) and the A/B for its
# perf value on Rubin.  A module constant, not a knob: it must never differ per plan.
STAGE3_CAUSAL_TRIM: bool = True
# Stage-3 cluster tile.  True = the d = 256 rendering (``MatmulTemplateParams.cgrp_tile_mn = (256, 256)``: cluster 2x1,
# 256 x 256 per pair, no N padding at d = 256, the accumulator double-buffered) -- what ships on the Rubin line.  False =
# the SM100 chain's (512, 512) rendering, which at d = 256 computes 256 columns of padding per cluster tile: the bitwise
# pin's twin (same k-tile walk, same 256x256x16 instruction, so identical bits) and the A/B base.  A module constant,
# not a knob: it must never differ per plan.
STAGE3_D256_TILE: bool = True
_STAGE3_TILE_D256 = (256, 256)
_STAGE3_TILE_PADDED = (512, 512)
# Stage-3 dQ launch shape under GQA.  True = ONE dQ launch per (batch, head) chunk: the dQ rendering takes
# ``MatmulTemplateParams.b_head_group = group`` (its B = K is indexed by ``h // group``, the K head the group's Q heads share)
# over the whole dS chunk and the whole dQ chunk -- what ships.  False = one launch per group MEMBER over every ``group``-th
# Q head (``b_head_group = 1``: B batched per head, so each launch's A / dQ heads line up with its K heads): the bitwise
# pin's twin (the same k-tile walk per output tile, so identical bits) and the A/B base.  At H_q / H_kv = 16 the member
# launches were sixteen under-one-wave launches (2 KV heads x 32 M tiles = 64 clusters on 212 SMs, thinned further by the
# causal trim): 0.58 ms against 0.31 ms for the dK GEMM of the same FLOPs (Rubin, B=1 H_q=32 H_kv=2 S=8K causal bf16).
# MHA (group 1) renders and launches identically either way.  A module constant read at CALL time (``_stage3_params``),
# not a knob: it must never differ per plan.
DQ_SINGLE_LAUNCH: bool = True
_DTYPE_CODE = {torch.bfloat16: DTYPE_BF16, torch.float16: DTYPE_FP16, torch.float8_e4m3fn: DTYPE_E4M3}
_TORCH_DTYPE = {code: dt for dt, code in _DTYPE_CODE.items()}
# The fp8 row's dS workspace dtype (plan Q2(a)).  DTYPE_E4M3 = what ships: dS_q = e4m3(dS * scale_dP), the stage-3 GEMMs
# render the fp8 K64 arm with the descale_dP / descale_{q|k} epilogue (dQ and MHA dK quantized in the GEMM, no Q / K
# upcast copies, half the workspace bytes).  DTYPE_BF16 = the pre-quantized chain (bf16 dS, bf16 GEMMs over exact e4m3
# -> bf16 upcasts, three fold + quantize passes): the A/B base and the twin the fp8 suite runs every accept case on.
# A module constant read when the adapter is built, not a knob: it must never differ per plan.
FP8_DS_DTYPE: int = DTYPE_E4M3
# The MXFP8 row's dS scale-factor policy (``config_sm107.DS_SF_P_B`` / ``DS_SF_P_C``), read when the adapter is BUILT -- the
# ``FP8_DS_DTYPE`` pattern: a module constant, never a knob (numerics-changing: a flip re-runs the accept matrix and re-writes the
# support-matrix cell).  ``DS_SF_P_B`` (= ``config_sm107.DS_SF_POLICY_DEFAULT``, what ships): the block-scaled chain -- the kernel
# writes two 1x32-scaled e4m3 dS payloads (ds_dk per 32-q block, ds_dq per 32-kv block) plus their F8_128x4 E8M0 atoms, and the
# stage-3 GEMMs render the block-scale arm (``MatmulTemplateParams.block_scale``) over them and the columnwise q_T / k_T with their
# own scale factors, dequantizing IN the MMA -- no dequant pass, the two payloads' bytes equal the bf16 chain's plus 1/16 for the
# atoms; Rubin-line only (the arm's 576-column exclusive TMEM allocation).  ``DS_SF_P_C``: the bf16-dS oracle twin -- a bf16 dS
# workspace into the bf16 stage-3 renderings over the EXACTLY dequantized bf16 q_T / k_T -- kept selectable.  The flip to P-b
# followed the Rubin A/B (cc 10.7, 212 SMs, SM clock 2376 MHz; B=1 H=128/128 S=8192): whole row +22.0 % dense / +17.1 % causal
# over P-c, both at a 4 GiB stage-2 budget (the main kernel 1.27x / 1.54x the bf16-dS body, both stage-3 GEMMs 1.8x faster and the two
# dequant passes gone), every accept cell within the fp8 recipe, the dS payloads bit-exact against the oracle's 1x32 quantization.
MXFP8_DS_SF_POLICY: int = _cfg.DS_SF_POLICY_DEFAULT


def _sm107_chunks(b: int, h_q: int, group: int, s_q_pad: int, s_kv_pad: int, bpe_ds: int, budget: int = _SM107_WS_BUDGET_BYTES, batch_chunking: bool = True):
    """``(b_chunk, qh_chunk)`` for one stage-2 launch: the LARGEST (batch, head) chunk whose dS
    workspace ``b_chunk * qh_chunk * S_kv * S_q * bpe`` fits ``budget``.

    Divisors, not floors, so no launch has a ragged tail and one artifact serves every
    chunk; the head chunk is a multiple of the GQA group so a chunk's Q heads map onto
    whole KV heads (``config_sm107.validate_head_chunk``).  Heads shrink first (a
    smaller head chunk keeps the batch loop out of the way); the batch is chunked only
    when even ``group`` heads at the full batch do not fit, and only on the half row
    (``batch_chunking``): the fp8 body walks the whole batch in-grid.  When nothing
    fits the smallest legal chunk is returned and the size is still honest.
    """
    per = s_q_pad * s_kv_pad * bpe_ds
    heads = [c for c in range(h_q, 0, -1) if h_q % c == 0 and c % group == 0]
    batches = [c for c in range(b, 0, -1) if b % c == 0] if batch_chunking else [b]
    for bc in batches:
        for hc in heads:
            if bc * hc * per <= budget:
                return bc, hc
    return batches[-1], group


def _stage3_cgrp_tile_mn(sm: int, d: int, d256_tile: Optional[bool] = None) -> tuple:
    """The stage-3 cluster tile for this arch and head dim: ``(256, 256)`` on the Rubin line at d = 256 (the tile whose
    N equals the head dim, so no cluster computes padding), the SM100 chain's ``(512, 512)`` anywhere else.  The rule
    lives HERE, in the sm107 adapter, so the SM100 adapter's records keep their default and render the constants they
    always did.  ``d256_tile`` defaults to ``STAGE3_D256_TILE``, read at CALL time so the bitwise pin can flip it."""
    if d256_tile is None:
        d256_tile = STAGE3_D256_TILE
    return _STAGE3_TILE_D256 if (d256_tile and 107 <= sm <= 119 and d == _SM107_D) else _STAGE3_TILE_PADDED


def _stage3_params(
    dtype_code: int,
    causal: bool,
    shift: int,
    gran: int,
    trim: Optional[bool] = None,
    *,
    cgrp_tile_mn: tuple,
    epi_modes: tuple = (EPI_NONE, EPI_NONE),
    dtype_out: int = -1,
    window: Optional[int] = None,
    gqa_group: int = 1,
    dq_single_launch: Optional[bool] = None,
    block_scale: bool = False,
):
    """The two stage-3 renderings ``(dK, dQ)`` for the KV-MAJOR ``[S_kv, S_q]`` workspace.

    dK = dS . Q  : A = dS[kv, q]   -- M = kv, K = q, q contiguous -> K-major; K starts at kv's block (LO), ends after the window
    dQ = dS^T . K: A = dS^T[q, kv] -- M = q,  K = kv, q contiguous -> M-major; K starts at the window, ends after q's block (HI)

    ``shift`` is how far the written band extends past the plain ``kv <= q`` diagonal
    (bottom-right: ``S_kv - S_q``); ``gran`` the kernel's kv write block (256) = the q pair
    it rounds its q range to (``config_sm107.q_write_tiles``).  ``window`` is the graph's
    ``window_left`` (the kernel keeps ``kv >= q + shift - W``), None = no window; a window
    alone (no causal) renders the band with the diagonal dropped (``causal_diag=False``).
    ``trim`` defaults to the module constant, read at CALL time so the bitwise pin can flip
    it; off, both records are ``CAUSAL_K_NONE`` and read every tile.  ``cgrp_tile_mn`` is the
    cluster tile ``_stage3_cgrp_tile_mn`` picked -- required, so a caller cannot fall into the
    padded rendering by omission.  ``dtype_code`` is the dS WORKSPACE dtype (the GEMMs' A
    operand): E4M3 selects the template's fp8 K64 arm, whose epilogue each rendering names in
    ``epi_modes = (dK, dQ)`` (``EPI_DESCALE`` -> bf16 true-unit partials, ``EPI_QUANT`` -> the
    quantized gradient in ``dtype_out`` + amax); the half row's bf16 / fp16 records keep
    ``EPI_NONE`` and the inherited output dtype.  ``gqa_group`` (``H_q / H_kv``) with
    ``dq_single_launch`` (None = the module constant ``DQ_SINGLE_LAUNCH``, read at CALL time so
    the bitwise pin can flip it) sets the dQ record's ``b_head_group``: the group when one launch
    covers a whole head chunk (its B = K is indexed by ``h // group``), 1 for the per-member loop
    and always at MHA -- the dK record's B = Q is per Q head and keeps 1.  ``block_scale`` (appended, default False) renders the MXFP8 block-scale arm
    over an E4M3 dS workspace whose 32-element K blocks carry E8M0 scale atoms (``MatmulTemplateParams.block_scale``): both
    renderings stay ``EPI_NONE`` (the MMA dequantizes; the accumulator is the true-unit gradient) with the inherited bf16 output.
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
    if dq_single_launch is None:
        dq_single_launch = DQ_SINGLE_LAUNCH
    if int(gqa_group) < 1:
        raise ValueError(f"sm107 stage 3: gqa_group must be >= 1 (H_q / H_kv); got {gqa_group}")
    dq_b_head_group = int(gqa_group) if (dq_single_launch and int(gqa_group) > 1) else 1
    if window is not None and int(window) <= 0:
        # The template spells "no window" as causal_window == 0, while the kernels' SWA arm at W = 0 keeps exactly one key per
        # row -- the two would disagree on what was written.  Unreachable through the rows (check_support / config_sm107
        # decline window_left <= 0); refused here so it stays that way.
        raise ValueError(f"sm107 stage 3: a sliding window needs window_left > 0 (the rows decline window_left <= 0); got {window}")
    band = trim and (causal or window is not None)
    lo = CAUSAL_K_LO if band else CAUSAL_K_NONE
    hi = CAUSAL_K_HI if band else CAUSAL_K_NONE
    common = dict(
        b_is_n_major=True,
        causal_gran=gran,
        causal_shift=shift if band else 0,
        vec_bytes_epi=vec_bytes_epi_for(_SM107_D, 2),
        dtype_qkv=dtype_code,
        cgrp_tile_mn=tuple(cgrp_tile_mn),
        causal_window=int(window) if (band and window is not None) else 0,
        causal_diag=bool(causal) if band else True,
        block_scale=bool(block_scale),
    )
    dk_mode, dq_mode = epi_modes
    return (
        MatmulTemplateParams(a_is_m_major=False, causal_mode=lo, epi_mode=dk_mode, dtype_out=dtype_out if dk_mode == EPI_QUANT else -1, **common),
        MatmulTemplateParams(
            a_is_m_major=True, causal_mode=hi, epi_mode=dq_mode, dtype_out=dtype_out if dq_mode == EPI_QUANT else -1, b_head_group=dq_b_head_group, **common
        ),
    )


def _stage3_needs_zero_fill(
    causal: bool,
    window: Optional[int],
    bottom_right: bool,
    s_q_pad: int,
    s_kv_pad: int,
    gran: int,
    trim: Optional[bool] = None,
    cgrp_tile_m: int = 256,
    per_batch_kv: bool = False,
) -> bool:
    """Whether the dS workspace must be zero-filled before the main kernel writes it (once per execute, the WHOLE chunk).

    The stage-3 GEMMs read only tiles the main kernel wrote: the kernel rounds every kv block's q range OUTWARD to the
    GEMMs' ``gran``-row pair (``config_sm107.q_write_tiles``) and every K-trim bound rounds outward inside that band
    (``bprop_matmul_blackwell._causal_k_range``), so a masked cell the GEMM reads is a zero the kernel STORED, not one the
    fill left.  Proven per mask by the poisoned-workspace tests (``test_masked_stage3_reads_only_what_stage2_wrote``,
    both suites: top-left / bottom-right causal, aligned and ragged, a window at 640 and at a non-multiple of the tile,
    window + bottom-right, window without causal).  Three cases still need the fill:

    * the untrimmed twin (``trim=False``): both GEMMs render ``CAUSAL_K_NONE`` and read every tile, skipped ones included
      (the bitwise pin's base);
    * a cluster M tile WIDER than the kernel's write block (``cgrp_tile_m > gran``: the (512, 512) / (512, 256) rows the
      ``STAGE3_D256_TILE = False`` twin renders on this line, the SM100 chain's tile) -- a 512-row M tile straddles two
      256-row kv blocks whose bands differ, so no per-tile K range stays inside what both wrote (the template's
      tight-trim invariant, ``_causal_k_range``);
    * a TOP-LEFT window with ``S_q > roundup(S_kv + W, gran)``: the last kv block (base ``S_kv_pad - gran``) writes q up
      to ``roundup(S_kv_pad + W, gran)`` and the q pairs past it are written by NO block, while the dQ GEMM's never-empty
      clamp still reads one k tile of them.  Bottom-right anchors the window on the diagonal (``S_kv - S_q``), so its
      last block reaches the last q row and the case cannot arise there.

    A fourth case comes with the caller's PER-BATCH kv lengths (``per_batch_kv``, appended, default False -- the half row's
    standalone ``seq_kv_lens``): under BOTTOM-RIGHT causal the kernel's diagonal is per batch (``seq_kv_lens[b] - S_q``) while
    the GEMMs' trim is computed from the uniform ``S_kv - S_q``, so a batch whose length is short of S_kv has a band the
    trim's K range can reach below (dK) or past (dQ) -- tiles the kernel never wrote, every cell of them masked.  The fill
    makes those reads exact zeros; a top-left band (causal or window) does not move with the length, so it needs nothing.
    Two companions of this case live elsewhere: the trim drops a sliding window for it (``_stage3_trim_window``), and the
    prepared host runs the fill ahead of every chunk, not once per execute (``prepared_host.host_f16``) -- the skipped set is
    per BATCH, so a later chunk's batch would otherwise read the previous batch's dS out of its workspace slot.

    Dense (no mask) never needs it: every tile is written.  The fill is a whole-chunk ``cudaMemset``-class kernel outside
    the harness's kernel-time filter -- 0.44 ms per 8K backward when it ran under every mask (MASK_FLOPS.md).
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
    if not (causal or window is not None):
        return False
    if per_batch_kv and bottom_right:
        return True
    if not trim or cgrp_tile_m > gran:
        return True
    if window is not None and not bottom_right:
        last_written_q = -(-(s_kv_pad + int(window)) // gran) * gran
        return s_q_pad > last_written_q
    return False


def _stage3_thd_needs_zero_fill(causal: bool, window: Optional[int], gran: int, trim: Optional[bool] = None, cgrp_tile_m: int = 256) -> bool:
    """The THD twin of :func:`_stage3_needs_zero_fill`: whether the kv-blocked dS workspace must be zero-filled per execute.

    Under THD the stage-3 GEMMs render the per-sequence K-trim (``MatmulTemplateParams.thd_varlen`` with a trimmed
    ``causal_mode``; ``bprop_matmul_blackwell._thd_causal_k_range``): every bound is sequence-local, derived from the
    sequence's REAL lengths and its own diagonal (``thd_causal_bottom_right``), rounded outward exactly like the dense trim
    (the causal edge to the kernel's 256-row q pair, the window edge to the k tile), so every tile a GEMM reads was written
    by the sequence's own kv blocks -- and a tile whose band is EMPTY (a kv block no query attends, a q pair with no key in
    its band, an empty reduction side) gets an empty K range and a SELECT-zero store instead of the dense arm's never-empty
    clamp.  That last point is why the dense rule's two per-geometry exceptions do not exist here: the per-batch
    bottom-right case (the THD shift is per sequence by construction) and the top-left window with
    ``S_q > roundup(S_kv + W, gran)`` (its unwritten q pairs are exactly the empty-band tiles).  Proven per mask by the
    poisoned-workspace THD tests (every ``_run_graph`` case runs over a 0xFF workspace) and the host tile walk
    ``test_stage3_thd_band_arithmetic``.  Two cases keep the fill: the untrimmed twin (``trim=False``: ``CAUSAL_K_NONE`` on
    both GEMMs, every tile read) and a cluster M tile wider than the kernel's write block (``cgrp_tile_m > gran``, the
    (512, 512) twin's tile).  Dense THD never needs it.
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
    if not (causal or window is not None):
        return False
    return (not trim) or cgrp_tile_m > gran


def _stage3_trim_window(window: Optional[int], causal: bool, bottom_right: bool, per_batch_kv: bool) -> Optional[int]:
    """The sliding window the stage-3 K-trim may use: the graph's ``window_left``, or None -- the window dropped from the
    trim -- under PER-BATCH kv lengths with bottom-right causal.

    The kernel anchors BOTH band edges on the per-batch diagonal (``seq_kv_lens[b] - S_q``: ``bprop_d256_f16._mask_p_chunk``,
    ``_q_loop_bounds``), the GEMMs on the uniform one (``S_kv - S_q``: ``bprop_matmul_blackwell._causal_k_range``).  The CAUSAL
    edge survives the mismatch: it rounds outward from the uniform diagonal, which sits at or past every per-batch one
    (``len_b <= S_kv``), so the plain bottom-right range is a superset of every batch's band and whatever it reaches beyond
    the kernel's band is the zero-fill's (``_stage3_needs_zero_fill``).  The WINDOW edge does not: for a batch shorter than
    S_kv the live dS tiles sit past the uniform window's ``k_hi`` on dK and below its ``k_lo`` on dQ, and the GEMMs skip
    them -- that batch's dQ / dK come back missing most of their mass, finite, no crash (measured on Rubin at lengths
    [1024, 700, 300, 0], S_q 512, S_kv 1024, window 200: dQ cosine 0.65 against the oracle, the two shorter batches'
    max|diff| equal to max|ref|; the same shape without the window, and with a top-left window, exact).  So the trim reads
    the plain bottom-right band for that arm -- the kernel still writes only its per-batch window band, the fill makes the
    rest exact zeros -- at the cost of the window's trim on stage 3 for that arm alone.  A top-left band does not move with
    the length and keeps its window; so does every uniform-length graph.
    """
    if per_batch_kv and causal and bottom_right and window is not None:
        return None
    return window


def _bshd_physical_ok(desc: TensorDesc) -> bool:
    """True when a logical-BHSD desc sits on compact BSHD storage (the rows' layout claim)."""
    b, h, s, d = (int(x) for x in desc.shape)
    return tuple(int(x) for x in desc.stride) == (s * h * d, d, h * d, 1)


def _thd_packed_ok(desc: TensorDesc) -> bool:
    """True when a ragged port's envelope desc describes PACKED BSHD rows the THD chain addresses with its own strides -- what
    ``engines.mismatch()`` admits for a row without ``thd_head_stride``: element stride 1, head stride D (wildcarded at one head),
    token stride >= H * D and a multiple of 8 elements (every head base 16-byte aligned).  The batch stride is not consulted (a
    ragged port's sequences start at its ragged offsets).  Every descriptor and view the chain builds carries these strides, so a
    padded token stride costs nothing to serve; admitting exactly what ``mismatch()`` admits keeps a bare ValueError out of the
    lowering."""
    b, h, s, d = (int(x) for x in desc.shape)
    _, hs, ts, es = (int(x) for x in desc.stride)
    head_ok = h == 1 or hs == d
    return (d == 1 or es == 1) and head_ok and ts >= h * (hs if h > 1 else d) and ts % 8 == 0


class SdpaBwdDslSm107(SdpaBwdDsl):
    """``sdpa_bwd_sm107``: d_qk = d_v = 256, bf16 / fp16, on the Rubin line (cc 10.7-11.9)."""

    _FAMILY = _cfg.FAMILY_F16
    _NAME = "sdpa_bwd_sm107"
    _BATCH_CHUNKING = True
    _IO_DTYPES = (torch.bfloat16, torch.float16)
    # The f16 body reads ``seq_kv_lens[batch]`` under its padded-mask arm (``_resolve_seqlen_kv``), so the caller's per-batch
    # kv lengths are served (module doc); the fp8 / MXFP8 bodies take one uniform ``seqlen_kv_real`` and their rows say False.
    _PER_BATCH_KV_LENS = True
    # The f16 body serves THD / varlen (packed [1, T, H, D] operands through packed-total-clamped runtime descriptors, a kv-blocked
    # dS workspace, per-sequence lengths and the device claim counter from the metadata buffer -- ``sm107/bprop_d256_f16.py``
    # "THD / varlen"); the fp8 / MXFP8 bodies take one uniform ``seqlen_kv_real`` and their rows say False.
    _THD_SUPPORTED = True

    def __init__(
        self,
        *args,
        external_delta: bool = False,
        thd: bool = False,
        max_total_seq_len_q: Optional[int] = None,
        max_total_seq_len_kv: Optional[int] = None,
        thd_stats_token_major: bool = False,
        thd_stats_head_stride: Optional[int] = None,
        **kwargs,
    ) -> None:
        """``external_delta`` (appended, default off): the caller computes stage 1's ``delta = rowsum(dO * O)`` and hands it to
        :meth:`execute` as ``delta_tensor`` (module docstring, "An externally computed delta"); the chain then launches no
        ``dot`` and carves no ``delta`` region.  A plan fact -- it decides the artifact and the workspace -- so it is fixed at
        construction, like ``amax_requested`` on the fp8 row.

        The THD plan-time facts (``thd``, the declared packed totals, the packed Stats packing) are the base constructor's, re-declared
        HERE because the engines' lowering forwards a fact only when it appears in the adapter's OWN signature (``lower_dsl_bwd``
        filters its extra constructor arguments by ``inspect.signature(adapter_cls.__init__)``, and a bare ``**kwargs`` hides them --
        the same rule the SM100 adapter's constructor spells out): without them a ragged graph would reach ``check_support`` as a
        DENSE plan and fail its Stats-layout check instead of being served."""
        self.external_delta = bool(external_delta)
        super().__init__(
            *args,
            thd=thd,
            max_total_seq_len_q=max_total_seq_len_q,
            max_total_seq_len_kv=max_total_seq_len_kv,
            thd_stats_token_major=thd_stats_token_major,
            thd_stats_head_stride=thd_stats_head_stride,
            **kwargs,
        )

    # --- geometry ------------------------------------------------------------------
    def _initialize_implementation(self) -> None:
        q_shape = tuple(int(x) for x in self.q_desc.shape)  # logical BHSD
        k_shape = tuple(int(x) for x in self.k_desc.shape)
        self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk = q_shape
        self.h_kv, self.s_k_max = int(k_shape[1]), int(k_shape[2])
        self.head_dim_v = int(tuple(self.v_desc.shape)[3])
        self.dtype = self.q_desc.dtype
        self.grad_dtype = self.dq_desc.dtype
        self._bpe = self.dtype.itemsize
        # dS workspace dtype: the io dtype on the half chain; e4m3 (shipped) or bf16 (the twin) on the fp8 chain (`_ds_torch_dtype`).
        self._ds_dtype = self._ds_torch_dtype()
        self._bpe_ds = self._ds_dtype.itemsize
        # attn_scale is OPTIONAL on the graph: None = absent -> 1/sqrt(d) (the SM100 adapter's story).  An EXPLICIT 0.0 is a
        # valid declared scale -- uniform P, dQ = dK = 0 exactly, dV = sum(dO) / S_kv -- that the analyzer preserves
        # (`scale = float(attn_scale)`), so it is preserved here too; `or == 0.0` ran such a graph at 1/sqrt(d) and returned
        # nonzero dQ / dK (Codex review on #1212).  Pinned by test_explicit_zero_attn_scale_* (both rows, host + Rubin).
        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim_qk)
        # Tile-rounded COMPILE shape: the q loop walks 128-row q tiles, a cga2 pair owns a
        # 256-row kv block.  The padded operands are staged (zero-filled), see the module doc.
        self._sq_pad = -(-self.s_q_max // _SM107_Q_PAD) * _SM107_Q_PAD
        self._skv_pad = -(-self.s_k_max // _SM107_KV_PAD) * _SM107_KV_PAD
        self._q_padded = self._sq_pad != self.s_q_max
        self._kv_padded = self._skv_pad != self.s_k_max
        self._gqa_group = self.h_q // max(self.h_kv, 1)
        # THD / varlen (the half row): the declared shapes carry the ENVELOPE (B, H, S_max, D); the token capacity is the packed
        # buffers' own extent tightened by the declared totals (``_thd_total``: a MIN -- a declaration only shrinks what the
        # buffers hold; the kernels clamp their descriptors to the live ``cu_*[B]`` on device).  The dS workspace is BLOCKED over
        # packed kv tokens at the kernel's 256-row kv block: every sequence's block is padded up to it, so B blocks cost at most
        # 255 rows each (``_ws_rows_cap``); the q columns are uniform at the padded q envelope.  No staging under THD (the packed
        # path addresses the caller's buffers directly), no batch chunking (one packed batch); the head chunk divides a per-head
        # slab of ``R_kv_cap x S_q_pad`` against the shared budget.
        self._thd_lse_token_major = bool(getattr(self, "thd_stats_token_major", False)) and self.thd
        self._thd_lse_head_stride = int(getattr(self, "thd_stats_head_stride", 0) or 0) if (self.thd and not self._thd_lse_token_major) else 0
        self._thd_units = 0  # the persistent grid's cluster count: a DEVICE fact, decided in compile()
        if self.thd:
            self._t_q_cap = _thd_total(self.s_q_max * self.batch_size, self.max_total_seq_len_q)
            self._t_kv_cap = _thd_total(self.s_k_max * self.batch_size, self.max_total_seq_len_kv)
            self._ws_rows_cap = -(-(self._t_kv_cap + self.batch_size * _SM107_KV_PAD) // _SM107_KV_PAD) * _SM107_KV_PAD
            self._q_padded = self._kv_padded = False
            self._b_chunk, self._qh_chunk = _sm107_chunks(
                1,
                self.h_q,
                self._gqa_group,
                self._sq_pad,
                self._ws_rows_cap,
                self._ds_chunk_bytes_per_elem(),
                budget=self._ws_budget_bytes(),
                batch_chunking=False,
            )
        else:
            self._b_chunk, self._qh_chunk = _sm107_chunks(
                self.batch_size,
                self.h_q,
                self._gqa_group,
                self._sq_pad,
                self._skv_pad,
                self._ds_chunk_bytes_per_elem(),
                budget=self._ws_budget_bytes(),
                batch_chunking=self._BATCH_CHUNKING,
            )
        # Whether the dS workspace is zero-filled per execute: decided in `compile()`, where the stage-3 cluster tile is
        # known (`_stage3_needs_zero_fill` / `_stage3_thd_needs_zero_fill`: the two-sided K-trim -- per sequence under THD --
        # reads only what the kernel wrote, so only the untrimmed twin, the wide-tile twin and, on the dense path, one
        # top-left-window geometry need it; the poisoned-workspace tests pin the rest).
        self._zero_ws = None
        # The dQ rendering's B head group (`MatmulTemplateParams.b_head_group`), copied off the record `compile()` builds so the
        # prepared host launches exactly what was rendered (`prepared_sm107._config` -> `prepared_host._stage3`): 1 = one dQ
        # launch per GQA group member (and MHA), the group = one launch per chunk (`DQ_SINGLE_LAUNCH`).
        self._dq_b_head_group = 1
        self._compiled = None
        self._prepared = None

    # --- capability backstop ---------------------------------------------------------
    def check_support(self) -> bool:
        """Re-check what the Capabilities row promised (backstops, never the gate)."""
        n = self._NAME
        self._value_error_if(
            self.head_dim_qk != _SM107_D or self.head_dim_v != _SM107_D, f"{n}: d_qk = d_v = {_SM107_D} exactly; got {self.head_dim_qk} / {self.head_dim_v}"
        )
        self._value_error_if(self.h_kv < 1 or self.h_q % self.h_kv != 0, f"{n}: h_q ({self.h_q}) must be a multiple of h_kv ({self.h_kv})")
        self._value_error_if(self.dtype not in self._IO_DTYPES, f"{n}: Q/K/V/O/dO dtype {self.dtype} not served")
        self._value_error_if(self.s_q_max < 1 or self.s_k_max < 1, f"{n}: empty sequences are not served (S_q={self.s_q_max}, S_kv={self.s_k_max})")
        self._value_error_if(self.s_q_max == 1, f"{n}: s_q == 1 (decode) is out of scope for the prefill bodies")
        for name, desc in (
            ("q", self.q_desc),
            ("k", self.k_desc),
            ("v", self.v_desc),
            ("o", self.o_desc),
            ("dO", self.do_desc),
            ("dQ", self.dq_desc),
            ("dK", self.dk_desc),
            ("dV", self.dv_desc),
        ):
            if self.thd:
                self._value_error_if(
                    not _thd_packed_ok(desc),
                    f"{n}: {name} must be packed BSHD rows under THD (element stride 1, head stride D, token stride >= H * D and a multiple of 8 "
                    f"elements); got stride {tuple(desc.stride)}",
                )
            else:
                self._value_error_if(not _bshd_physical_ok(desc), f"{n}: {name} must be BSHD-physical (stride order 3,1,2,0); got stride {tuple(desc.stride)}")
        st_shape, st_stride = tuple(int(x) for x in self.stats_desc.shape), tuple(int(x) for x in self.stats_desc.stride)
        if self.thd:
            # Packed Stats: the ctor's packing flags (token-major (T, H) / head-major (1, QH, head_stride)) decide how the kernel
            # reads it; the declared dims are the envelope's.  A head stride SHORTER than the packed total puts the later heads'
            # rows past the buffer (the kernel reads [0, h, row] at that stride for every row < T_q).
            self._value_error_if(
                st_shape != (self.batch_size, self.h_q, self.s_q_max, 1), f"{n} THD: stats must be declared (B, H_q, S_q, 1); got dim {st_shape}"
            )
            self._value_error_if(
                not self._THD_SUPPORTED,
                f"{n}: THD / ragged is not implemented on this row -- the body takes ONE uniform real kv length (seqlen_kv_real); the bf16 / fp16 "
                f"row (sdpa_bwd_sm107) serves THD",
            )
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                f"{n} THD: THD carries its per-sequence lengths in the metadata buffer, not as seq_*_lens_present tensors (mutually exclusive)",
            )
            # The externally computed delta is a DENSE contract ([B, H_q, S_q_pad] fp32, one row per envelope position): the THD
            # chain's delta is PACKED head-major at ceil128(T_q) and computed by its own dot_do_o over the packed O / dO, and no
            # producer emits the packed layout.  Declined typed here, before any plan is built; a delta_tensor at execute is then
            # refused by the plan-fact check (external_delta=False) -- never silently ignored.
            self._value_error_if(
                self.external_delta,
                f"{n} THD: external_delta is not served on the packed chain -- its delta is dot_do_o over the packed O / dO in the head-major "
                f"[1, H_q, ceil128(T_q)] layout the THD main kernel reads, and a caller's dense [B, H_q, S_q_pad] delta has no packed twin; "
                f"build the THD plan without external_delta (the chain launches its own pre-pass)",
            )
            # The declared totals are REQUIRED, and the reason is the workspace: scratch_workspace_bytes() is a BUILD-time function,
            # and the blocked dS row count, delta's row stride and the GQA partials are all fixed from the packed token capacity
            # before any buffer exists.  Undeclared, that capacity falls back to B * S_max -- more tokens than a packed buffer holds.
            self._value_error_if(
                self.max_total_seq_len_q is None or self.max_total_seq_len_kv is None,
                f"{n} THD: max_total_seq_len_q and max_total_seq_len_kv must be declared (the kv-blocked dS workspace, delta and the GQA partials "
                f"are sized from the packed token totals at build time)",
            )
            self._value_error_if(
                self._t_q_cap < 1 or self._t_kv_cap < 1,
                f"{n} THD: the packed token capacities must be positive; got T_q={self._t_q_cap}, T_kv={self._t_kv_cap}",
            )
            self._value_error_if(
                self._thd_lse_token_major and bool(self.thd_stats_head_stride),
                f"{n} THD: thd_stats_head_stride is head-major-only (token-major (T, H) Stats is compact)",
            )
            self._value_error_if(
                bool(self._thd_lse_head_stride) and self._thd_lse_head_stride < self._t_q_cap,
                f"{n} THD: Stats head stride {self._thd_lse_head_stride} must cover the packed token total {self._t_q_cap}",
            )
        else:
            self._value_error_if(
                st_shape != (self.batch_size, self.h_q, self.s_q_max, 1) or st_stride != (self.h_q * self.s_q_max, self.s_q_max, 1, 1),
                f"{n}: stats must be contiguous (B, H_q, S_q, 1); got dim {st_shape} stride {st_stride}",
            )
        # Per-batch lengths: the half body reads seq_kv_lens[b] under its padded-mask arm (the standalone surface,
        # `seq_kv_lens_present=True` + `execute(seq_kv_lens=)`); no body threads a per-batch Q length, and the fp8 / MXFP8
        # bodies take ONE uniform real kv length, so those stay declined.  The graph rows keep `Capabilities.padded = False`:
        # a graph padding mask carries seq_len_q AND seq_len_kv by construction (the frontend requires both), and serving it
        # while ignoring the q lengths would be silently wrong on any q length < S_q.
        self._value_error_if(
            self.seq_q_lens_present, f"{n}: per-batch Q lengths (seq_q_lens) are not implemented -- the body threads only the per-batch kv length (seq_kv_lens)"
        )
        self._value_error_if(
            self.seq_kv_lens_present and not self._PER_BATCH_KV_LENS,
            f"{n}: per-batch kv lengths (seq_kv_lens) are not implemented -- this body takes ONE uniform real kv length (seqlen_kv_real)",
        )
        self._value_error_if(
            self.deterministic, f"{n}: use_deterministic_algorithm is not claimed yet (the chain has no atomics; the two-run bitwise test decides)"
        )
        self._value_error_if(self.sink_desc is not None or self.dsink_desc is not None, f"{n}: sink / dSink are not implemented")
        self._value_error_if(self.bias_desc is not None or self.dbias_desc is not None, f"{n}: bias / dBias are not implemented")
        self._value_error_if(
            self.window_size_right not in (None, 0), f"{n}: causal right-band widening (window_right={self.window_size_right}) is not implemented"
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left <= 0, f"{n}: sliding window needs window_left > 0; got {self.window_size_left}"
        )
        self._value_error_if(self.causal_bottom_right and not self.is_causal, f"{n}: bottom-right alignment requires a causal mask")
        self._check_support_family()
        self._is_supported = True
        return True

    def _check_support_family(self) -> None:
        self._value_error_if(self.grad_dtype != self.dtype, f"{self._NAME}: dQ/dK/dV dtype {self.grad_dtype} must match the io dtype {self.dtype}")

    def _ds_torch_dtype(self):
        """The dS workspace dtype: the io dtype (the bf16 / fp16 GEMM renderings read it as the io dtype)."""
        return self.dtype

    def _ds_chunk_bytes_per_elem(self):
        """Workspace bytes per dS element the (batch, head) chunking budgets (``_sm107_chunks``): the dS dtype's size -- ONE
        payload -- on every chain but the MXFP8 row's block-scaled one, which carries two payloads and two scale-factor tensors."""
        return self._bpe_ds

    def _ws_budget_bytes(self) -> int:
        """The stage-2 dS workspace budget the (batch, head) chunking honours (``_sm107_chunks``): ``_SM107_WS_BUDGET_BYTES`` on
        every row and every dS policy -- one constant, one place (the MXFP8 row used to carry its own)."""
        return _SM107_WS_BUDGET_BYTES

    # --- workspace: ONE ordered plan, carved identically by the prepared host ----------
    def _scratch_shapes(self):
        """``[(name, shape, dtype)]`` in carve order -- every buffer the chain touches that is
        not a caller tensor, as the compact region the pointer host views it as
        (``prepared_sm107._regions``), so :meth:`scratch_workspace_bytes` and the artifact
        cannot disagree."""
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp = self._sq_pad, self._skv_pad
        kv_rows = skvp if self._kv_padded else skv
        gqa = self._gqa_group > 1
        if self.thd:
            # THD (the half row): delta packed head-major at ceil128(T_q) -- always the chain's own (the THD row declines
            # ``external_delta``: a caller's dense [B, H_q, S_q_pad] delta has no packed twin, ``check_support``); ONE head chunk of
            # the kv-BLOCKED dS workspace; the metadata buffer (5B+5 words) followed by the main kernel's (5 + B) tensor maps on the
            # next 128-B boundary (``tile_dsl.thd.THD_BWD_MAPS_META_WORDS``); stage 3's (B + 1) patched descriptors; the per-Q-head
            # partials over the PACKED kv axis under GQA.  No staging (the packed path reads the caller's buffers), no folds.
            tq, tkv = self._t_q_cap, self._t_kv_cap
            plan = [
                ("delta", (1, h, -(-tq // 128) * 128), torch.float32),
                ("ds_ws", (1, self._qh_chunk, self._ws_rows_cap, sqp), self._ds_dtype),
                ("seq_kv", (THD_BWD_MAPS_META_WORDS(b, 5 + b),), torch.int32),
                ("desc_words", ((b + 1) * 16,), torch.int64),
            ]
            if gqa:
                plan += [("dv_part", (1, tkv, h, d), self.dtype), ("dk_part", (1, tkv, h, d), self.dtype)]
            return plan
        plan = []
        if not self.external_delta:
            # stage 1's delta: [B, H_q, ceil128(S_q)] fp32 -- dot_do_o writes the rounded extent, zeros past S_q.  Under
            # external_delta the caller's tensor of the same shape (`external_delta_shape`) replaces the region.
            plan.append(("delta", (b, h, sqp), torch.float32))
        plan += [
            # the dS workspace of one launch: [b_chunk, qh_chunk, S_kv_pad, S_q_pad], kv-major
            ("ds_ws", (self._b_chunk, self._qh_chunk, skvp, sqp), self._ds_dtype),
            # stage 2's per-batch kv lengths (read only under the padded arm) + stage 3's dead THD ABI slot
            ("seq_kv", (b,), torch.int32),
            ("desc_words", (1,), torch.int64),
        ]
        if self._q_padded:
            plan += [("q_pad", (b, sqp, h, d), self.dtype), ("do_pad", (b, sqp, h, d), self.dtype), ("lse_pad", (b, h, sqp), torch.float32)]
        if self._kv_padded:
            plan += [("k_pad", (b, skvp, hkv, d), self.dtype), ("v_pad", (b, skvp, hkv, d), self.dtype)]
        plan += self._family_scratch_shapes(kv_rows, gqa)
        return plan

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        b, h, hkv, d = self.batch_size, self.h_q, self.h_kv, _SM107_D
        plan = []
        # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding
        if gqa or self._kv_padded:
            plan.append(("dv_part", (b, kv_rows, h, d), self.dtype))
        # stage 3's dK per Q head: the caller's dK when MHA (real rows written)
        if gqa:
            plan.append(("dk_part", (b, kv_rows, h, d), self.dtype))
            if self._kv_padded:
                # dkv_reduce folds at the workspace's row extent; the real rows are copied out
                plan += [("dk_fold", (b, kv_rows, hkv, d), self.dtype), ("dv_fold", (b, kv_rows, hkv, d), self.dtype)]
        return plan

    def _scratch_plan(self):
        """``[(name, numel, dtype)]`` -- :meth:`_scratch_shapes` flattened (the byte sum's and the tests' form)."""
        return [(name, math.prod(shape), dtype) for name, shape, dtype in self._scratch_shapes()]

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the compile geometry (delta -- unless ``external_delta`` -- + one dS chunk +
        padded staging + GQA partials): the artifact carves all of it from the caller's buffer."""
        return sum(ws_align(numel * dtype.itemsize) for _name, numel, dtype in self._scratch_plan())

    @property
    def external_delta_shape(self) -> tuple:
        """The delta's contract, ``(B, H_q, S_q_pad)``: what stage 1 writes into the workspace region, and what
        ``execute(delta_tensor=)`` must carry under ``external_delta`` -- fp32, contiguous, zeros on the pad rows
        ``[S_q, S_q_pad)``, 16-B aligned, on the plan's device."""
        return (self.batch_size, self.h_q, self._sq_pad)

    def _check_external_delta(self, delta_tensor) -> None:
        """The ``delta_tensor`` contract at execute, host-only and before any bind: both directions of the plan fact, then
        the exact ``dot_do_o`` layout (a strided or padded-differently delta would be read as garbage by the main kernel's
        stats prefetch, never a fault)."""
        n = self._NAME
        if not self.external_delta:
            self._value_error_if(
                delta_tensor is not None,
                f"{n}: delta_tensor was given but this plan was built with external_delta=False (its own dot_do_o pre-pass computes delta); pass None",
            )
            return
        self._value_error_if(
            delta_tensor is None,
            f"{n}: delta_tensor is required by this plan (external_delta=True): fp32 contiguous {self.external_delta_shape} on the plan's device",
        )
        shape = tuple(int(x) for x in delta_tensor.shape)
        self._value_error_if(
            delta_tensor.dtype != torch.float32,
            f"{n}: delta_tensor must be fp32 (the chain's dot_do_o layout), got {delta_tensor.dtype}",
        )
        self._value_error_if(
            shape != tuple(self.external_delta_shape) or not delta_tensor.is_contiguous(),
            f"{n}: delta_tensor must be a CONTIGUOUS [B, H_q, S_q_pad] = {self.external_delta_shape} tensor (S_q_pad = S_q rounded up to {_SM107_Q_PAD}, zeros past "
            f"S_q), got shape {shape} with strides {tuple(delta_tensor.stride())}",
        )
        self._value_error_if(
            not delta_tensor.is_cuda or delta_tensor.device != self.q_desc.device,
            f"{n}: delta_tensor must be on the plan's device {self.q_desc.device}, got {delta_tensor.device}",
        )
        self._value_error_if(delta_tensor.data_ptr() % 16 != 0, f"{n}: delta_tensor base must be 16-byte aligned, got {delta_tensor.data_ptr():#x}")

    # --- compilation -------------------------------------------------------------------
    def _template_params(self):
        return _cfg.TemplateParams(
            dtype_qkv=_DTYPE_CODE[self.dtype],
            window_right=0 if self.is_causal else None,  # set => causal; right-band widening is declined
            window_left=self.window_size_left,
            bottom_right=self.causal_bottom_right,
            # A padded S_kv selects the padded-mask arm with the UNIFORM real length, so the
            # zero-filled pad rows produce P = 0 -> dS = dV = 0 there (and, on the fp8 row,
            # stay out of the amax folds); the caller's per-batch kv lengths (half row) select
            # the same arm reading seq_kv_lens[b].  Dense otherwise: the arm folds out.
            seq_kv_lens_present=(self._kv_padded or self.seq_kv_lens_present) and not self.thd,
            # THD: the per-sequence lengths come from the metadata buffer (mutually exclusive with the dense length flags).
            thd_varlen=self.thd,
            dtype_o=self._dtype_o_code(),
            **self._template_params_family(),
        )

    def _template_params_family(self) -> dict:
        return {}  # the half row's record inherits its dS dtype (the io dtype)

    def _dtype_o_code(self) -> int:
        return -1  # inherit the io dtype

    def _stage3_records(self, mod, tile_mn):
        """The (dK, dQ) stage-3 renderings: bf16 / fp16 over the io-dtype workspace, no epilogue."""
        if self.thd:
            # THD renders the same two-sided K-trim as the dense path, PER SEQUENCE (``bprop_matmul_blackwell._thd_causal_k_range``:
            # every bound sequence-local, the sequence's real lengths from the metadata; bottom-right spelled as
            # ``thd_causal_bottom_right`` -- the diagonal offset is ``s_kv[b] - s_q[b]`` per sequence, so the record's constant
            # ``causal_shift`` stays 0), with the THD arm on and the rows named KV-major (``thd_rows_kv``: the token side of each
            # GEMM's reduction flips against the SM100 chain's q-major workspace).  The window edge keeps the graph's window: the
            # THD trim anchors it on the per-sequence diagonal exactly as the kernel does.  dQ once per head chunk under GQA
            # (``b_head_group = group`` through ``DQ_SINGLE_LAUNCH``, read at call time like the dense records).
            p_dk, p_dq = _stage3_params(
                _DTYPE_CODE[self._ds_dtype],
                bool(self.is_causal),
                0,
                _cfg.kv_pad_rows(mod.CFG),
                cgrp_tile_mn=tile_mn,
                window=self.window_size_left,
                gqa_group=self._gqa_group,
            )
            thd_br = bool(self.is_causal and self.causal_bottom_right) and p_dk.causal_mode != CAUSAL_K_NONE
            return (
                replace(p_dk, thd_varlen=True, thd_rows_kv=True, thd_causal_bottom_right=thd_br),
                replace(p_dq, thd_varlen=True, thd_rows_kv=True, thd_causal_bottom_right=thd_br),
            )
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        # Per-batch kv lengths under bottom-right read the plain bottom-right band: the window edge is the kernel's alone.
        window = _stage3_trim_window(self.window_size_left, bool(self.is_causal), bool(self.causal_bottom_right), bool(self.seq_kv_lens_present))
        return _stage3_params(
            _DTYPE_CODE[self._ds_dtype],
            bool(self.is_causal),
            shift,
            _cfg.kv_pad_rows(mod.CFG),
            cgrp_tile_mn=tile_mn,
            window=window,
            gqa_group=self._gqa_group,
        )

    def _compile_plan(self, mod, mm_dk, mm_dq):
        if self.thd:
            return _prepared.compile_plan_thd(self, mod, mm_dk, mm_dq)
        return _prepared.compile_plan(self, mod, mm_dk, mm_dq)

    def compile(self) -> None:
        """Plan-time JIT: the stage-2 template specialized on this graph's masks / dtype, the
        two stage-3 GEMM renderings, and the ONE pointer-host artifact that launches the
        whole chain over them (``prepared_sm107.compile_plan``), recorded as ``_prepared``."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        mod = load_template(_sm100_kernel_path(_SM107_KERNEL_FILES[self._FAMILY]), self._template_params(), tag=_SM107_TEMPLATE_TAGS[self._FAMILY])
        # The bodies' own geometry guards (tile multiples, chunk divisors) -- cheap, and the plan is wrong if they fire.
        _cfg.validate_head_chunk(self.h_q, self.h_kv, self._qh_chunk)
        # Stage 3 reads the dS workspace in ITS dtype (`_stage3_records`: the io dtype on the half chain; on the fp8 chain the
        # e4m3 workspace through the fp8 K64 arm + its epilogue, or the bf16 twin through the bf16 renderings).
        # The d = 256 cluster tile (no N padding) on the Rubin line; `prepared_sm107._sm` resolves the same device the
        # prepared artifact is compiled for.
        tile_mn = _stage3_cgrp_tile_mn(_prepared._sm(self), _SM107_D)
        if self.thd:
            # The per-sequence K-trim reads only tiles the kernel wrote (`_stage3_thd_needs_zero_fill`): no fill under any mask
            # the row serves; the untrimmed twin and the wide-tile twin keep it (`prepared_host.host_f16_thd`).  The persistent
            # grid: min(the unit upper bound, the device's cluster count) -- occupancy-sized, a cluster whose first unit is past
            # the device live total runs one forced fully-masked tile.
            self._zero_ws = _stage3_thd_needs_zero_fill(bool(self.is_causal), self.window_size_left, _SM107_KV_PAD, cgrp_tile_m=tile_mn[0])
            self._thd_units = max(
                1,
                min(
                    _cfg.thd_units_upper_bound(mod.CFG, self._t_kv_cap, self.batch_size, self._qh_chunk),
                    _sm100_device_clusters(self.q_desc.device, mod.CFG.CGA_M),
                ),
            )
        else:
            self._zero_ws = _stage3_needs_zero_fill(
                bool(self.is_causal),
                self.window_size_left,
                bool(self.causal_bottom_right),
                self._sq_pad,
                self._skv_pad,
                _SM107_KV_PAD,
                cgrp_tile_m=tile_mn[0],
                per_batch_kv=bool(self.seq_kv_lens_present),
            )
        p_dk, p_dq = self._stage3_records(mod, tile_mn)
        # The host launches dQ the way its rendering indexes B: ONE source of truth, the record (`prepared_host._dq_launches`
        # refuses a value that is neither 1 nor the group).
        self._dq_b_head_group = int(p_dq.b_head_group)
        mm_dk = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dk, tag=_SM107_MM_TAGS["dk"])
        mm_dq = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dq, tag=_SM107_MM_TAGS["dq"])
        self._prepared = self._compile_plan(mod, mm_dk, mm_dq)
        self._compiled = self._prepared.artifact
        return self._compiled

    # --- execution ---------------------------------------------------------------------
    def _refuse_unclaimed(self, seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor) -> None:
        for name, t in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(t is not None, f"{self._NAME}: {name} is not implemented")
        if self.thd:
            # Both per-sequence length tensors are REQUIRED under THD ((B,) lengths or (B+1,) prefixes, int32, contiguous; bind()
            # derives the form from numel): the setup launch builds the metadata from them.
            self._value_error_if(
                seq_q_lens is None or seq_kv_lens is None,
                f"{self._NAME} THD: execute needs seq_q_lens AND seq_kv_lens ((B,) lengths or (B+1,) prefix sums, int32, contiguous); "
                f"given q: {seq_q_lens is not None}, kv: {seq_kv_lens is not None}",
            )
            return
        self._value_error_if(seq_q_lens is not None, f"{self._NAME}: per-batch Q lengths (seq_q_lens) are not implemented")
        # The lengths operand is a plan fact (prepared_sm107.compile_plan binds it exactly when seq_kv_lens_present); bind()
        # would refuse the mismatch too, this names it.
        self._value_error_if(
            (seq_kv_lens is not None) != bool(self.seq_kv_lens_present),
            f"{self._NAME}: seq_kv_lens must be given exactly when the plan was built with seq_kv_lens_present=True "
            f"(built with {bool(self.seq_kv_lens_present)}, given: {seq_kv_lens is not None})",
        )

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        delta_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """The standalone twin of the graph plan: the same prepared artifact, bound from torch
        tensors (``prepared_sm107.execute_standalone``).  Every operand must carry the plan's
        geometry; the workspace is the caller's (``scratch_workspace_bytes()`` bytes).
        ``seq_kv_lens`` ([B] int32, contiguous, on the plan's device) is required exactly when
        the plan was built with ``seq_kv_lens_present=True`` (module doc); every entry must
        satisfy ``0 <= len <= S_kv`` -- device data, not validated here.  ``delta_tensor``
        (appended) is required exactly when the plan was built with ``external_delta=True``:
        fp32 contiguous ``external_delta_shape`` on the plan's device (module docstring, "An
        externally computed delta").  The two are independent plan facts: a plan built with
        both takes both."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        self._check_external_delta(delta_tensor)
        self.compile()
        if self.thd:
            # The THD plan's two appended slots are the two length tensors; a delta never reaches here (`_check_external_delta`
            # refused it above: the THD row is built with external_delta=False, `check_support`).
            tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor, seq_q_lens, seq_kv_lens)
        else:
            tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor, seq_kv_lens, delta_tensor)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


def _thd_total(capacity: int, declared: Optional[int]) -> int:
    """Token capacity, tightened by the caller's declared packed total -- always a MIN (the SM100 adapter's rule): a
    declaration can only shrink what the buffers can hold, so a stale or oversized one cannot push an access outside the
    caller's allocation.  It sizes the workspace; it does NOT make the extents exact (it is a maximum while the row that must
    read as zero is the current ``cu_*[B]``), hence the kernels' device-side descriptor clamps."""
    return capacity if declared is None else min(capacity, max(int(declared), 0))


_FP8_GRAD_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float16)


class SdpaBwdDslSm107Fp8(SdpaBwdDslSm107):
    """``sdpa_bwd_sm107_fp8``: cuDNN ``sdpa_fp8_backward`` at d = 256 on the Rubin line.

    FP8 E4M3 Q / K / V / O / dO with scalar descales in, fp32 Stats, gradients in the
    graph's dtype (E4M3 with ``scale_dQ / dK / dV``, or bf16 / fp16) plus the requested
    ``amax_dQ / dK / dV / dP`` out.  See the module doc for where each scalar is applied.
    """

    _FAMILY = _cfg.FAMILY_FP8
    _NAME = "sdpa_bwd_sm107_fp8"
    _BATCH_CHUNKING = False  # the fp8 body has no batch_base: the whole batch is in-grid
    _IO_DTYPES = (torch.float8_e4m3fn,)
    _PER_BATCH_KV_LENS = False  # the fp8 body takes ONE uniform `seqlen_kv_real` (its padded arm + amax row gate), no per-batch read
    _THD_SUPPORTED = False  # no THD arm in the fp8 body (uniform real lengths, no packed descriptors); the f16 row serves THD

    def __init__(self, *args, amax_requested=(), **kwargs) -> None:
        """``amax_requested``: the subset of ``("amax_dQ", "amax_dK", "amax_dV", "amax_dP")`` the
        graph declared as real outputs.  A plan fact: it decides which amax pointers the
        artifact binds (an unrequested one is None-specialized and its fold atomics fold
        out), so the lowering passes it at construction."""
        names = tuple(amax_requested)
        unknown = [n for n in names if n not in _prepared.FP8_AMAX]
        if unknown:
            raise ValueError(f"{self._NAME}: unknown amax outputs {unknown}; expected a subset of {_prepared.FP8_AMAX}")
        self.amax_requested = frozenset(names)
        super().__init__(*args, **kwargs)

    def _check_support_family(self) -> None:
        n = self._NAME
        self._value_error_if(
            self.external_delta,
            f"{n}: external_delta is not served on the fp8 row: its delta is the DEscaled dot of the fp8 O / dO payloads (descale_o * descale_dO), "
            "computed by its own scaled pre-pass; only the bf16 / fp16 row takes a caller's delta",
        )
        self._value_error_if(self.grad_dtype not in _FP8_GRAD_DTYPES, f"{n}: dQ/dK/dV dtype {self.grad_dtype} not in {_FP8_GRAD_DTYPES}")
        self._value_error_if(
            self.dk_desc.dtype != self.grad_dtype or self.dv_desc.dtype != self.grad_dtype,
            f"{n}: dQ/dK/dV must share one dtype; got {self.dq_desc.dtype}/{self.dk_desc.dtype}/{self.dv_desc.dtype}",
        )
        for name, desc in (("k", self.k_desc), ("v", self.v_desc), ("o", self.o_desc), ("dO", self.do_desc)):
            self._value_error_if(desc.dtype != self.dtype, f"{n}: {name} is an FP8 payload and must share Q's dtype {self.dtype}; got {desc.dtype}")
        # The fp8 body derives the bottom-right diagonal (S_kv - S_q) and the q-tile trim from the PADDED
        # S_q (no seqlen_q_real in its ABI; the f16 body threads it), so a ragged S_q would be served
        # silently wrong by (pad - S_q) rows.  The row already declines this at eligibility
        # (Capabilities.bottom_right_s_q_multiple); this is the backstop.  A ragged S_kv is fine: the
        # kernel's kv term is the runtime real length.
        self._value_error_if(
            self.causal_bottom_right and self._q_padded,
            f"{n}: bottom-right causal needs S_q % {_SM107_Q_PAD} == 0 (the fp8 body derives the diagonal from the padded S_q; "
            f"its ABI has no seqlen_q_real); got S_q={self.s_q_max}; follow-up: thread seqlen_q_real like the f16 body",
        )

    def _dtype_o_code(self) -> int:
        # Stage 2 publishes dV in bf16 (the pre-quantization value) for the fold + quantize pass.
        return DTYPE_BF16

    def _ds_torch_dtype(self):
        """e4m3 (``FP8_DS_DTYPE = DTYPE_E4M3``, shipped: dS_q = e4m3(dS * scale_dP) for the fp8 GEMM arm) or bf16 (the twin)."""
        if FP8_DS_DTYPE not in (DTYPE_E4M3, DTYPE_BF16):
            raise ValueError(f"{self._NAME}: FP8_DS_DTYPE must be DTYPE_E4M3 or DTYPE_BF16; got {FP8_DS_DTYPE}")
        return _TORCH_DTYPE[FP8_DS_DTYPE]

    @property
    def _ds_fp8(self) -> bool:
        return self._ds_dtype == torch.float8_e4m3fn

    def _template_params_family(self) -> dict:
        return dict(dtype_ds=_DTYPE_CODE[self._ds_dtype])

    def _stage3_records(self, mod, tile_mn):
        """e4m3 dS: the fp8 K64 arm -- dQ ``EPI_QUANT`` into the caller's dQ (amax_dQ in the epilogue); dK ``EPI_QUANT``
        into the caller's dK at MHA, ``EPI_DESCALE`` (bf16 true-unit per-Q-head partials, quantized AFTER the GQA fold)
        otherwise.  bf16 dS: the bf16 renderings, no epilogue (stage 4 folds + quantizes all three)."""
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        gran = _cfg.kv_pad_rows(mod.CFG)
        window = self.window_size_left
        if not self._ds_fp8:
            return _stage3_params(
                _DTYPE_CODE[self._ds_dtype], bool(self.is_causal), shift, gran, cgrp_tile_mn=tile_mn, window=window, gqa_group=self._gqa_group
            )
        dk_mode = EPI_QUANT if self._gqa_group == 1 else EPI_DESCALE
        return _stage3_params(
            DTYPE_E4M3,
            bool(self.is_causal),
            shift,
            gran,
            cgrp_tile_mn=tile_mn,
            epi_modes=(dk_mode, EPI_QUANT),
            dtype_out=_DTYPE_CODE[self.grad_dtype],
            window=window,
            gqa_group=self._gqa_group,
        )

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        plan = [("dv_part", (b, kv_rows, h, d), torch.bfloat16)]  # stage 2's per-Q-head dV_true, bf16
        if self._ds_fp8:
            # e4m3 dS: the GEMMs read the e4m3 payloads directly and quantize dQ (and MHA dK) in their epilogue; only the
            # GQA dK partials (bf16, TRUE units: EPI_DESCALE) go through stage 4.
            if gqa:
                plan.append(("dk_part", (b, kv_rows, h, d), torch.bfloat16))
        else:
            plan += [
                ("dk_part", (b, kv_rows, h, d), torch.bfloat16),  # stage 3's per-Q-head dS . Q8 (descale_q pending)
                ("dq_ws", (b, sq, h, d), torch.bfloat16),  # stage 3's dS^T . K8 (descale_k pending)
                ("q_bf16", (b, sq, h, d), torch.bfloat16),  # Q8 upcast EXACTLY (the bf16 GEMM's B operand)
                ("k_bf16", (b, skv, hkv, d), torch.bfloat16),  # K8 upcast EXACTLY
            ]
        plan.append(("amax_scratch", (8,), torch.float32))  # stage 2's dV amax (recomputed by stage 4) + any amax the graph left virtual
        return plan

    def _compile_plan(self, mod, mm_dk, mm_dq):
        return _prepared.compile_plan_fp8(self, mod, mm_dk, mm_dq)

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        # sdpa_fp8_backward operands (append-only extension of the shared signature)
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_o: Optional[torch.Tensor] = None,
        descale_dO: Optional[torch.Tensor] = None,
        descale_s: Optional[torch.Tensor] = None,
        descale_dP: Optional[torch.Tensor] = None,
        scale_s: Optional[torch.Tensor] = None,
        scale_dQ: Optional[torch.Tensor] = None,
        scale_dK: Optional[torch.Tensor] = None,
        scale_dV: Optional[torch.Tensor] = None,
        scale_dP: Optional[torch.Tensor] = None,
        amax_dQ: Optional[torch.Tensor] = None,
        amax_dK: Optional[torch.Tensor] = None,
        amax_dV: Optional[torch.Tensor] = None,
        amax_dP: Optional[torch.Tensor] = None,
    ) -> None:
        """The standalone twin of the graph plan over the ``sdpa_fp8_backward`` operand set: the
        twelve scalars are 1-element fp32 DEVICE tensors (read in-kernel), an amax tensor is
        given exactly when the plan was built with it requested (``amax_requested``)."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        given = dict(
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            descale_o=descale_o,
            descale_dO=descale_dO,
            descale_dP=descale_dP,
            scale_dQ=scale_dQ,
            scale_dK=scale_dK,
            scale_dV=scale_dV,
            scale_dP=scale_dP,
        )
        for name in _prepared.FP8_SCALARS:
            self._value_error_if(given[name] is None, f"{self._NAME}: {name} is required by sdpa_fp8_backward")
        amaxes = dict(amax_dQ=amax_dQ, amax_dK=amax_dK, amax_dV=amax_dV, amax_dP=amax_dP)
        for name in _prepared.FP8_AMAX:
            self._value_error_if(
                (amaxes[name] is not None) != (name in self.amax_requested),
                f"{self._NAME}: {name} must be given exactly when the plan requested it (requested: {sorted(self.amax_requested)})",
            )
        self.compile()
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        tensors += tuple(given[name] for name in _prepared.FP8_SCALARS) + tuple(amaxes[name] for name in _prepared.FP8_AMAX)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


# The MXFP8 row's gradient dtypes: bf16 only for now.  The P-c chain's stage 3 is the bf16 renderings, which store their io dtype
# (config_sm100 refuses another output dtype under EPI_NONE); dV leaves the kernel in the same dtype.  fp16 = a final cast pass per
# gradient (a follow-up with its own accept case), never a widening of this tuple without it.
_MXFP8_GRAD_DTYPES = (torch.bfloat16,)
_MXFP8_SF_ROLES = ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v", "sf_do", "sf_do_T")
_MXFP8_BLOCK = _cfg.MX_BLOCK  # 32 elements per E8M0 scale
_MXFP8_SF_ATOM_ROWS = _cfg.SF_ATOM_ROWS  # 128: the F8_128x4 row pad


class SdpaBwdDslSm107Mxfp8(SdpaBwdDslSm107):
    """``sdpa_bwd_sm107_mxfp8``: cuDNN ``sdpa_mxfp8_backward`` at d = 256 on the Rubin line (module doc, MXFP8 section).

    The constructor takes the SM100 MXFP8 adapter's keyword surface (``engines.lower_dsl_bwd_mxfp8`` is reused unchanged):
    ``sample_q_T / k_T / do_T / do_f16`` and the seven ``sample_sf_*``; ``sample_o`` carries the ``o_f16`` port, ``sample_do``
    the ROWWISE e4m3 dO.  ``p_scale_log2`` is pinned to 8: the kernel folds the byte 119 at trace time (the ONLY constant form
    of the fused scaled cvt that assembles -- the ``cvt.u8.u32`` immediate of the 16-pack), so any other value is a typed decline, not a parameter.
    """

    _FAMILY = _cfg.FAMILY_MXFP8
    _NAME = "sdpa_bwd_sm107_mxfp8"
    _BATCH_CHUNKING = False  # the MXFP8 body (the fp8 body's pipeline) has no batch_base: the whole batch is in-grid
    _IO_DTYPES = (torch.float8_e4m3fn,)
    _PER_BATCH_KV_LENS = False  # the MXFP8 body takes ONE uniform `seqlen_kv_real` like the fp8 body, no per-batch read
    _THD_SUPPORTED = False  # no THD arm in the MXFP8 body (uniform real lengths, per-sequence scale-factor pads unsolved); the f16 row serves THD

    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_o,
        sample_do,
        sample_stats,
        sample_dq,
        sample_dk,
        sample_dv,
        *,
        sample_q_T,
        sample_k_T,
        sample_do_T,
        sample_do_f16,
        sample_sf_q,
        sample_sf_q_T,
        sample_sf_k,
        sample_sf_k_T,
        sample_sf_v,
        sample_sf_do,
        sample_sf_do_T,
        p_scale_log2: int = _cfg.MXFP8_P_SCALE_LOG2,
        **kwargs,
    ) -> None:
        # Stashed raw; the descs are built in _initialize_implementation, which the base __init__ calls once APIBase's state exists.
        self._mxfp8_samples = dict(
            q_T=sample_q_T,
            k_T=sample_k_T,
            dO_T=sample_do_T,
            dO_f16=sample_do_f16,
            sf_q=sample_sf_q,
            sf_q_T=sample_sf_q_T,
            sf_k=sample_sf_k,
            sf_k_T=sample_sf_k_T,
            sf_v=sample_sf_v,
            sf_do=sample_sf_do,
            sf_do_T=sample_sf_do_T,
        )
        self.p_scale_log2 = int(p_scale_log2)
        # The dS policy is read ONCE, here (``MXFP8_DS_SF_POLICY``, a module constant like ``FP8_DS_DTYPE``), before the base
        # geometry asks for the dS dtype; its derived dS facts (payloads, atoms, bytes) come from the config's own record of the
        # policy, never a literal.
        self._ds_policy = int(MXFP8_DS_SF_POLICY)
        if self._ds_policy not in (_cfg.DS_SF_P_C, _cfg.DS_SF_P_B):
            raise NotImplementedError(
                f"{self._NAME}: MXFP8_DS_SF_POLICY must be DS_SF_P_C ({_cfg.DS_SF_P_C}, bf16 dS) or DS_SF_P_B ({_cfg.DS_SF_P_B}, block-scaled e4m3 dS both "
                f"ways); the P-a tile-scale policy ({_cfg.DS_SF_P_A}) is not built; got {self._ds_policy}"
            )
        self._ds_cfg = _cfg.make_cfg_d256_bwd(_cfg.TemplateParams(dtype_qkv=DTYPE_E4M3, ds_sf_policy=self._ds_policy), _cfg.FAMILY_MXFP8)
        super().__init__(sample_q, sample_k, sample_v, sample_o, sample_do, sample_stats, sample_dq, sample_dk, sample_dv, **kwargs)

    @property
    def _ds_block_scaled(self) -> bool:
        """True under the P-b chain: two 1x32-scaled e4m3 dS payloads + E8M0 atoms into the block-scale stage-3 GEMM arm."""
        return self._ds_policy == _cfg.DS_SF_P_B

    # --- geometry ------------------------------------------------------------------
    def _initialize_implementation(self) -> None:
        s = self._mxfp8_samples
        self.q_T_desc = self._make_tensor_desc(s["q_T"], name="q_T")
        self.k_T_desc = self._make_tensor_desc(s["k_T"], name="k_T")
        self.do_T_desc = self._make_tensor_desc(s["dO_T"], name="dO_T")
        self.do_f16_desc = self._make_tensor_desc(s["dO_f16"], name="dO_f16")
        self.sf_descs = {n: self._make_tensor_desc(s[n], name=n) for n in _MXFP8_SF_ROLES}
        super()._initialize_implementation()
        self.out_dtype = self.o_desc.dtype  # the half-precision side: o_f16 / dO_f16 / dQ / dK / dV

    @staticmethod
    def _ceil128(n: int) -> int:
        return -(-n // _MXFP8_SF_ATOM_ROWS) * _MXFP8_SF_ATOM_ROWS

    def _sf_expected_bytes(self, graph_sf: str) -> int:
        """Byte count of a graph SF tensor under cuDNN's F8_128x4 padding rules (rows to 128, block columns to 4) -- the
        SM100 adapter's function; the ONLY property of an SF tensor the row trusts (its declared dims are the producer's,
        two of its strides are rewritten by the C++ node before lowering)."""
        b, hq, hk, sq, sk, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, self.head_dim_qk

        def _blocks4(n: int) -> int:  # 32-element blocks along an axis, padded to a multiple of 4 (one F8_128x4 atom column group)
            return -(-(-(-n // _MXFP8_BLOCK)) // 4) * 4

        if graph_sf in ("sf_q", "sf_do"):
            return b * hq * self._ceil128(sq) * _blocks4(d)
        if graph_sf in ("sf_k", "sf_v"):
            return b * hk * self._ceil128(sk) * _blocks4(d)
        if graph_sf in ("sf_q_T", "sf_do_T"):
            return b * hq * _blocks4(sq) * self._ceil128(d)
        if graph_sf == "sf_k_T":
            return b * hk * _blocks4(sk) * self._ceil128(d)
        raise ValueError(graph_sf)

    # --- capability backstop ---------------------------------------------------------
    def _check_support_family(self) -> None:
        n = self._NAME
        self._value_error_if(
            self.external_delta,
            f"{n}: external_delta is not served on the mxfp8 row: its delta is the dot of its own f16 ports (o_f16 / dO_f16), computed by its own "
            "pre-pass; only the bf16 / fp16 row takes a caller's delta",
        )
        self._value_error_if(
            self.p_scale_log2 != _cfg.MXFP8_P_SCALE_LOG2,
            f"{n}: p_scale_log2 is pinned to {_cfg.MXFP8_P_SCALE_LOG2} (the kernel folds the P scale byte {127 - _cfg.MXFP8_P_SCALE_LOG2} at trace "
            f"time -- the only constant form of the fused scaled cvt that assembles); got {self.p_scale_log2}",
        )
        self._value_error_if(
            self.out_dtype not in _MXFP8_GRAD_DTYPES,
            f"{n}: o_f16 / dO_f16 / dQ / dK / dV dtype {self.out_dtype} not in {_MXFP8_GRAD_DTYPES} (the P-c chain's bf16 stage-3 renderings "
            f"store bf16; the fp16 arm is a follow-up)",
        )
        for name, desc in (("dO_f16", self.do_f16_desc), ("dQ", self.dq_desc), ("dK", self.dk_desc), ("dV", self.dv_desc)):
            self._value_error_if(desc.dtype != self.out_dtype, f"{n}: {name} dtype {desc.dtype} != o_f16 {self.out_dtype}")
        for name, desc in (
            ("k", self.k_desc),
            ("v", self.v_desc),
            ("dO", self.do_desc),
            ("q_T", self.q_T_desc),
            ("k_T", self.k_T_desc),
            ("dO_T", self.do_T_desc),
        ):
            self._value_error_if(desc.dtype != self.dtype, f"{n}: {name} is an FP8 payload and must share Q's dtype {self.dtype}; got {desc.dtype}")
        for name, desc in (("q_T", self.q_T_desc), ("k_T", self.k_T_desc), ("dO_T", self.do_T_desc), ("dO_f16", self.do_f16_desc)):
            self._value_error_if(not _bshd_physical_ok(desc), f"{n}: {name} must be BSHD-physical (stride order 3,1,2,0); got stride {tuple(desc.stride)}")
        for name, desc in self.sf_descs.items():
            have = int(math.prod(int(x) for x in desc.shape))
            want = self._sf_expected_bytes(name)
            self._value_error_if(have != want, f"{n}: {name} has {have} bytes; the F8_128x4 layout for this shape needs {want}")
        # V's scale factor is ROWWISE in the backward (the C++ node's reference math dequantizes V like K); a
        # columnwise-shaped binding has the same byte count and would be a wrong dV, so the SHAPE is asserted for sf_v alone.
        sf_v_shape = tuple(int(x) for x in self.sf_descs["sf_v"].shape)
        self._value_error_if(
            len(sf_v_shape) != 4 or sf_v_shape[2] != self._ceil128(self.s_k_max) or sf_v_shape[3] != self.head_dim_qk // _MXFP8_BLOCK,
            f"{n}: sf_v must be the ROWWISE F8_128x4 tensor (B, H_kv, ceil128(S_kv), {self.head_dim_qk // _MXFP8_BLOCK}); got {sf_v_shape}",
        )
        # The MXFP8 body derives the bottom-right diagonal from its PADDED S_q (its seqlen_q_real feeds the q-pad band only):
        # the fp8 row's rule, at eligibility (Capabilities.bottom_right_s_q_multiple) and here as the backstop.
        self._value_error_if(
            self.causal_bottom_right and self._q_padded,
            f"{n}: bottom-right causal needs S_q % {_SM107_Q_PAD} == 0 (the body derives the diagonal from the padded S_q); got S_q={self.s_q_max}",
        )
        # The block-scaled dS chain (P-b) renders the stage-3 block-scale GEMM arm: a 576-column EXCLUSIVE TMEM allocation and the
        # K64 tcgen05.mma.block_scale -- the Rubin line only (the row's own range; ``prepared_host._check_target`` backstops at
        # compile, this declines typed BEFORE a plan is built on any other part).
        if self._ds_block_scaled:
            sm = _prepared._sm(self)
            self._value_error_if(
                not 107 <= sm <= 119,
                f"{n}: the block-scaled dS policy (P-b: two 1x32-scaled e4m3 dS payloads into the block-scale stage-3 GEMM arm, a 576-column exclusive "
                f"TMEM allocation) is served on the Rubin line (SM107-SM119) only; this device is SM{sm}",
            )

    def _dtype_o_code(self) -> int:
        return _DTYPE_CODE[self.out_dtype]  # the kernel's dV partials in the gradient dtype (bf16), TRUE units

    def _ds_torch_dtype(self):
        """The dS workspace dtype follows the policy: P-c a bf16 workspace (the bf16 renderings read it as bf16); P-b the e4m3
        payload (``ds_ws`` is ``ds_dk``, scaled per 32-q block -- the dK GEMM's A; ``ds_dq`` is carved next to it)."""
        return torch.float8_e4m3fn if self._ds_block_scaled else torch.bfloat16

    def _ds_chunk_bytes_per_elem(self):
        """P-b: ``DS_PAYLOADS`` e4m3 payloads + ``DS_SF_ATOMS`` scale-factor tensors of one byte per ``SF_BLOCK`` elements (2 + 2/32 per
        dS element: the bf16 chain's 2 B plus 1/16); P-c: the bf16 payload (the config's ``DS_PAYLOADS`` is 1, ``DS_SF_ATOMS`` 0)."""
        c = self._ds_cfg
        return c.DS_PAYLOADS * c.BPE_DS + c.DS_SF_ATOMS / c.SF_BLOCK

    def _template_params_family(self) -> dict:
        from cudnn.frost.device import compute_capability, resolve_device

        cc = compute_capability(resolve_device(self.q_desc.device))
        return dict(
            # The fused scale-and-pack cvt assembles for sm_107a only (the helper's trace-time backstop refuses it elsewhere).
            scaled_fp8_pack=(tuple(cc) == (10, 7)),
            # A ragged S_q takes the q < seqlen_q_real band of the transposed mask (a SEPARATE const_expr arm, not a MASK_FLAGS bit).
            mask_q_pad=self._q_padded,
            ds_sf_policy=self._ds_policy,
        )

    def _stage3_records(self, mod, tile_mn):
        """P-c: the base class's bf16 renderings over the bf16 dS.  P-b: the block-scale arm (``MatmulTemplateParams.block_scale``)
        over the e4m3 payloads -- dK reads ``ds_dk`` [kv, q] K-major with the ``sf_ds_dk`` atoms, dQ reads ``ds_dq`` as [q, kv]
        M-major with ``sf_ds_dq``; B is the columnwise q_T / k_T payload with its D-plane-major SF.  EPI_NONE on both (the MMA
        dequantizes; the fp32 accumulator is the true-unit gradient, stored bf16)."""
        if not self._ds_block_scaled:
            return super()._stage3_records(mod, tile_mn)
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        # The block-scale arm launches dQ once per GQA group member (its SFB descriptor is indexed per A / C head), so its dQ record
        # keeps b_head_group == 1 whatever DQ_SINGLE_LAUNCH says; the single launch is the plain renderings' form (a follow-up here).
        return _stage3_params(
            DTYPE_E4M3,
            bool(self.is_causal),
            shift,
            _cfg.kv_pad_rows(mod.CFG),
            cgrp_tile_mn=tile_mn,
            window=self.window_size_left,
            block_scale=True,
            gqa_group=self._gqa_group,
            dq_single_launch=False,
        )

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp, groups = self._sq_pad, self._skv_pad, d // _MXFP8_BLOCK
        plan = []
        if self._q_padded:
            # dO_T pads like dO; the Q-side SF slabs are re-staged with the rows / groups past S_q zeroed (prepared_host._pad_sf_atoms).
            plan += [
                ("do_T_pad", (b, sqp, h, d), self.dtype),
                ("sf_q_pad", (b, h, sqp, groups), torch.uint8),
                ("sf_do_pad", (b, h, sqp, groups), torch.uint8),
                ("sf_doT_pad", (b, h, groups, sqp), torch.uint8),
            ]
        if self._kv_padded:
            # The kernel's K / V SF descriptors span the PADDED kv tiles (S_kv_pad / 128 per head): the slabs grow to that extent
            # with the rows past S_kv zeroed (both pad classes: S_kv % 256 in (0, 128] appends a whole tile, (128, 256) pads inside one).
            plan += [("sf_k_pad", (b, hkv, skvp, groups), torch.uint8), ("sf_v_pad", (b, hkv, skvp, groups), torch.uint8)]
        if self._ds_block_scaled:
            # P-b.  The kernel's second dS payload (ds_dq: scaled per 32-kv block; the first, ds_dk, is the shared ``ds_ws`` region)
            # and the two E8M0 atom tensors, one F8_128x4 atom per (kv tile, q tile) -- ``config_sm107.sf_workspace_bytes`` each --
            # at the launch's chunk geometry (the workspace contract: sf_ds_dk [B, H_chunk, S_kv/128, S_q/128, 512], sf_ds_dq the
            # transpose).  Then the GEMMs' SFB operands: the columnwise q_T / k_T scale factors are read as WHOLE atoms by the
            # block-scale MMA (the dequant pass that read real extents only is gone), so a ragged S_q / S_kv re-stages them with
            # the pad groups zeroed, at the SF tensor's own ceil128 tile count (the GEMM's K tiles) -- 0 x 2^-127 of a zero
            # payload is exactly 0, while a producer's 0xFF pad byte would be 0 x NaN.
            tq, tkv = sqp // _MXFP8_SF_ATOM_ROWS, skvp // _MXFP8_SF_ATOM_ROWS
            atom = _cfg.SF_ATOM_BYTES
            plan += [
                ("ds_dq", (self._b_chunk, self._qh_chunk, skvp, sqp), self._ds_dtype),
                ("sf_ds_dk", (self._b_chunk, self._qh_chunk, tkv, tq, atom), torch.uint8),
                ("sf_ds_dq", (self._b_chunk, self._qh_chunk, tq, tkv, atom), torch.uint8),
            ]
            if self._q_padded:
                plan.append(("sf_qT_pad", (b, h, groups, sqp), torch.uint8))
            if self._kv_padded:
                plan.append(("sf_kT_pad", (b, hkv, groups, self._ceil128(skv)), torch.uint8))
        else:
            # Stage 3's B operands: the columnwise q_T / k_T dequantized EXACTLY to bf16 (real extents).
            plan += [("q_T_bf16", (b, sq, h, d), torch.bfloat16), ("k_T_bf16", (b, skv, hkv, d), torch.bfloat16)]
        # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding; stage 3's dK per Q head: the caller's dK when MHA.
        if gqa or self._kv_padded:
            plan.append(("dv_part", (b, kv_rows, h, d), self.out_dtype))
        if gqa:
            plan.append(("dk_part", (b, kv_rows, h, d), self.out_dtype))
            if self._kv_padded:
                plan += [("dk_fold", (b, kv_rows, hkv, d), self.out_dtype), ("dv_fold", (b, kv_rows, hkv, d), self.out_dtype)]
        return plan

    def _compile_plan(self, mod, mm_dk, mm_dq):
        return _prepared.compile_plan_mxfp8(self, mod, mm_dk, mm_dq)

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        *,
        # sdpa_mxfp8_backward operands (the SM100 MXFP8 adapter's keyword surface; append-only)
        q_T_tensor: Optional[torch.Tensor] = None,
        k_T_tensor: Optional[torch.Tensor] = None,
        do_T_tensor: Optional[torch.Tensor] = None,
        do_f16_tensor: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_q_T: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_k_T: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        sf_do: Optional[torch.Tensor] = None,
        sf_do_T: Optional[torch.Tensor] = None,
    ) -> None:
        """The standalone twin of the graph plan over the ``sdpa_mxfp8_backward`` operand set: ``o_tensor`` is ``o_f16``,
        ``do_tensor`` the rowwise e4m3 dO; the SF tensors are the F8_128x4 uint8 blobs (byte count checked, dims free)."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        extras = dict(
            q_T=q_T_tensor,
            k_T=k_T_tensor,
            do_T=do_T_tensor,
            do_f16=do_f16_tensor,
            sf_q=sf_q,
            sf_q_T=sf_q_T,
            sf_k=sf_k,
            sf_k_T=sf_k_T,
            sf_v=sf_v,
            sf_do=sf_do,
            sf_do_T=sf_do_T,
        )
        missing = [name for name, t in extras.items() if t is None]
        self._value_error_if(bool(missing), f"{self._NAME}: execute needs {missing}")
        self.compile()
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        tensors += tuple(extras[name] for name in _prepared.MXFP8_PAYLOADS + _prepared.MXFP8_SF)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


__all__ = [
    "SdpaBwdDslSm107",
    "SdpaBwdDslSm107Fp8",
    "SdpaBwdDslSm107Mxfp8",
    "FP8_DS_DTYPE",
    "MXFP8_DS_SF_POLICY",
    "DQ_SINGLE_LAUNCH",
    "STAGE3_CAUSAL_TRIM",
    "STAGE3_D256_TILE",
    "_stage3_needs_zero_fill",
    "_stage3_thd_needs_zero_fill",
    "_stage3_trim_window",
]
