# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapters over the FROST SM107 (Rubin) d=256 SDPA backward chains.

Two rows share this module -- ``sdpa_bwd_sm107`` (bf16 / fp16, :class:`SdpaBwdDslSm107`)
and ``sdpa_bwd_sm107_fp8`` (per-tensor FP8 E4M3, :class:`SdpaBwdDslSm107Fp8`) -- because
they share one chain shape.  Each is a TWO-kernel backward around a dS workspace
(bf16 / fp16 on the half row; e4m3 -- or bf16, the A/B twin -- on the fp8 row),
followed by the two gradient GEMMs and a fold:

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
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import TensorDesc
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
from cudnn.sdpa.bwd import config_sm107 as _cfg
from cudnn.sdpa.bwd import prepared_sm107 as _prepared
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl, _SM100_MATMUL_FILE, _SM100_WS_BUDGET_BYTES, _sm100_kernel_path
from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, EPI_DESCALE, EPI_NONE, EPI_QUANT, MatmulTemplateParams, vec_bytes_epi_for
from cudnn.sdpa.fwd.api_dsl import ws_align

_SM107_D = 256
# The bodies' compile geometry: the q loop walks 128-row q tiles, a cga2 pair owns a 256-row kv block
# (``config_sm107.q_pad_rows`` / ``kv_pad_rows``).  The adapter pads S_q / S_kv up to these and stages
# the padded operands (module doc).  ``_SM107_Q_PAD`` is also the fp8 row's bottom-right alignment
# claim (``engines.Capabilities.bottom_right_s_q_multiple``; pinned equal by the fp8 suite).
_SM107_Q_PAD = 128
_SM107_KV_PAD = 256  # also the stage-3 K-trim granularity (`causal_gran`) and the kernels' q write pair (`config_sm107.q_write_tiles`)
_SM107_KERNEL_FILES = {_cfg.FAMILY_F16: "sm107/bprop_d256_f16.py", _cfg.FAMILY_FP8: "sm107/bprop_d256_fp8.py"}
_SM107_TEMPLATE_TAGS = {_cfg.FAMILY_F16: "sdpa_bwd_sm107_main_f16", _cfg.FAMILY_FP8: "sdpa_bwd_sm107_main_fp8"}
_SM107_MM_TAGS = {"dk": "sdpa_bwd_sm107_mm_dk", "dq": "sdpa_bwd_sm107_mm_dq"}
# Same budget as the SM100 chain: above it the chunk shrinks and the chain runs more
# launches over the same total work.
_SM107_WS_BUDGET_BYTES = _SM100_WS_BUDGET_BYTES
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
_DTYPE_CODE = {torch.bfloat16: DTYPE_BF16, torch.float16: DTYPE_FP16, torch.float8_e4m3fn: DTYPE_E4M3}
_TORCH_DTYPE = {code: dt for dt, code in _DTYPE_CODE.items()}
# The fp8 row's dS workspace dtype (plan Q2(a)).  DTYPE_E4M3 = what ships: dS_q = e4m3(dS * scale_dP), the stage-3 GEMMs
# render the fp8 K64 arm with the descale_dP / descale_{q|k} epilogue (dQ and MHA dK quantized in the GEMM, no Q / K
# upcast copies, half the workspace bytes).  DTYPE_BF16 = the pre-quantized chain (bf16 dS, bf16 GEMMs over exact e4m3
# -> bf16 upcasts, three fold + quantize passes): the A/B base and the twin the fp8 suite runs every accept case on.
# A module constant read when the adapter is built, not a knob: it must never differ per plan.
FP8_DS_DTYPE: int = DTYPE_E4M3


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
    ``EPI_NONE`` and the inherited output dtype.
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
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
    )
    dk_mode, dq_mode = epi_modes
    return (
        MatmulTemplateParams(a_is_m_major=False, causal_mode=lo, epi_mode=dk_mode, dtype_out=dtype_out if dk_mode == EPI_QUANT else -1, **common),
        MatmulTemplateParams(a_is_m_major=True, causal_mode=hi, epi_mode=dq_mode, dtype_out=dtype_out if dq_mode == EPI_QUANT else -1, **common),
    )


def _stage3_needs_zero_fill(
    causal: bool, window: Optional[int], bottom_right: bool, s_q_pad: int, s_kv_pad: int, gran: int, trim: Optional[bool] = None, cgrp_tile_m: int = 256
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

    Dense (no mask) never needs it: every tile is written.  The fill is a whole-chunk ``cudaMemset``-class kernel outside
    the harness's kernel-time filter -- 0.44 ms per 8K backward when it ran under every mask (MASK_FLOPS.md).
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
    if not (causal or window is not None):
        return False
    if not trim or cgrp_tile_m > gran:
        return True
    if window is not None and not bottom_right:
        last_written_q = -(-(s_kv_pad + int(window)) // gran) * gran
        return s_q_pad > last_written_q
    return False


def _bshd_physical_ok(desc: TensorDesc) -> bool:
    """True when a logical-BHSD desc sits on compact BSHD storage (the rows' layout claim)."""
    b, h, s, d = (int(x) for x in desc.shape)
    return tuple(int(x) for x in desc.stride) == (s * h * d, d, h * d, 1)


class SdpaBwdDslSm107(SdpaBwdDsl):
    """``sdpa_bwd_sm107``: d_qk = d_v = 256, bf16 / fp16, on the Rubin line (cc 10.7-11.9)."""

    _FAMILY = _cfg.FAMILY_F16
    _NAME = "sdpa_bwd_sm107"
    _BATCH_CHUNKING = True
    _IO_DTYPES = (torch.bfloat16, torch.float16)

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
        self._b_chunk, self._qh_chunk = _sm107_chunks(
            self.batch_size, self.h_q, self._gqa_group, self._sq_pad, self._skv_pad, self._bpe_ds, batch_chunking=self._BATCH_CHUNKING
        )
        # Whether the dS workspace is zero-filled per execute: decided in `compile()`, where the stage-3 cluster tile is
        # known (`_stage3_needs_zero_fill`: the two-sided K-trim reads only what the kernel wrote, so only the untrimmed
        # twin, the wide-tile twin and one top-left-window geometry need it; the poisoned-workspace tests pin the rest).
        self._zero_ws = None
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
            self._value_error_if(not _bshd_physical_ok(desc), f"{n}: {name} must be BSHD-physical (stride order 3,1,2,0); got stride {tuple(desc.stride)}")
        st_shape, st_stride = tuple(int(x) for x in self.stats_desc.shape), tuple(int(x) for x in self.stats_desc.stride)
        self._value_error_if(
            st_shape != (self.batch_size, self.h_q, self.s_q_max, 1) or st_stride != (self.h_q * self.s_q_max, self.s_q_max, 1, 1),
            f"{n}: stats must be contiguous (B, H_q, S_q, 1); got dim {st_shape} stride {st_stride}",
        )
        # v1 declines (plan Q4): each flips together with its accept test and tracker line.
        self._value_error_if(self.thd, f"{n}: THD / ragged is not implemented")
        self._value_error_if(self.seq_kv_lens_present or self.seq_q_lens_present, f"{n}: padding masks (seq lens) are not implemented")
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
        plan = [
            # stage 1's delta: [B, H_q, ceil128(S_q)] fp32 -- dot_do_o writes the rounded extent, zeros past S_q
            ("delta", (b, h, sqp), torch.float32),
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
        """A pure function of the compile geometry (delta + one dS chunk + padded staging +
        GQA partials): the artifact carves all of it from the caller's buffer."""
        return sum(ws_align(numel * dtype.itemsize) for _name, numel, dtype in self._scratch_plan())

    # --- compilation -------------------------------------------------------------------
    def _template_params(self):
        return _cfg.TemplateParams(
            dtype_qkv=_DTYPE_CODE[self.dtype],
            window_right=0 if self.is_causal else None,  # set => causal; right-band widening is declined
            window_left=self.window_size_left,
            bottom_right=self.causal_bottom_right,
            # A padded S_kv selects the padded-mask arm with the UNIFORM real length, so the
            # zero-filled pad rows produce P = 0 -> dS = dV = 0 there (and, on the fp8 row,
            # stay out of the amax folds).  Dense otherwise: the arm folds out.
            seq_kv_lens_present=self._kv_padded,
            dtype_o=self._dtype_o_code(),
            **self._template_params_family(),
        )

    def _template_params_family(self) -> dict:
        return {}  # the half row's record inherits its dS dtype (the io dtype)

    def _dtype_o_code(self) -> int:
        return -1  # inherit the io dtype

    def _stage3_records(self, mod, tile_mn):
        """The (dK, dQ) stage-3 renderings: bf16 / fp16 over the io-dtype workspace, no epilogue."""
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        return _stage3_params(
            _DTYPE_CODE[self._ds_dtype], bool(self.is_causal), shift, _cfg.kv_pad_rows(mod.CFG), cgrp_tile_mn=tile_mn, window=self.window_size_left
        )

    def _compile_plan(self, mod, mm_dk, mm_dq):
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
        self._zero_ws = _stage3_needs_zero_fill(
            bool(self.is_causal), self.window_size_left, bool(self.causal_bottom_right), self._sq_pad, self._skv_pad, _SM107_KV_PAD, cgrp_tile_m=tile_mn[0]
        )
        p_dk, p_dq = self._stage3_records(mod, tile_mn)
        mm_dk = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dk, tag=_SM107_MM_TAGS["dk"])
        mm_dq = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dq, tag=_SM107_MM_TAGS["dq"])
        self._prepared = self._compile_plan(mod, mm_dk, mm_dq)
        self._compiled = self._prepared.artifact
        return self._compiled

    # --- execution ---------------------------------------------------------------------
    def _refuse_unclaimed(self, seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor) -> None:
        for name, t in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(t is not None, f"{self._NAME}: {name} is not implemented")
        self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, f"{self._NAME}: padding masks (seq lens) are not implemented")

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
    ) -> None:
        """The standalone twin of the graph plan: the same prepared artifact, bound from torch
        tensors (``prepared_sm107.execute_standalone``).  Every operand must carry the plan's
        geometry; the workspace is the caller's (``scratch_workspace_bytes()`` bytes)."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        self.compile()
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


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
            return _stage3_params(_DTYPE_CODE[self._ds_dtype], bool(self.is_causal), shift, gran, cgrp_tile_mn=tile_mn, window=window)
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


__all__ = ["SdpaBwdDslSm107", "SdpaBwdDslSm107Fp8", "FP8_DS_DTYPE", "STAGE3_CAUSAL_TRIM", "STAGE3_D256_TILE", "_stage3_needs_zero_fill"]
