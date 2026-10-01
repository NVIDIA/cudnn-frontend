# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the FROST SM100 SDPA-backward stage-2 flavor (d512).

Stage 2 of the three-stage backward: it consumes Q / K / V / dO / LSE / do_dot
and produces the ``S`` and ``dS`` workspaces that stage 3's three GEMMs reduce
into dV / dK / dQ.

Geometry is fixed here; the per-graph compile-time parameters (dtype, mask,
padding, scheduler policy) arrive as a :class:`TemplateParams` built by the
adapter in :mod:`cudnn.sdpa.bwd.api_dsl`. Nothing in this module reads an
environment variable.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier by the engine's ``Capabilities`` —
a ``ValueError`` from here means that row has a gap.

Layout rationale: the
Rubin SM107 backward's 320 KiB single-CTA layout does not fit SM100's 227 KiB.
This flavor instead forks the *forward* d512 SM100 kernel's cga4x1 role split —
sub-group 0 runs BMM1 (Q.K^T -> S_acc) and the softmax, sub-group 1 runs BMM2
(dO.V^T -> dS_acc) and the dS epilogue — which gives each sub-group its own
227 KiB of SMEM and its own 512 TMEM columns. That is what makes the operand
resident in TMEM on BOTH sides (Q on sg0, dO on sg1) and restores the
accumulator double-buffer the Rubin kernel had to drop.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stage-3 causal K-trim modes (see MatmulTemplateParams.causal_mode).
CAUSAL_K_NONE = 0
CAUSAL_K_LO = 1
CAUSAL_K_HI = 2
# Stage-3 epilogue modes (see MatmulTemplateParams.epi_mode).
EPI_NONE = 0
EPI_DESCALE = 1
EPI_QUANT = 2
from typing import Optional, Tuple

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# SM100 usable dynamic SMEM per CTA (228 KiB physical, 227 KiB usable).  Mirrors
# ``sdpa.fwd.config_sm100._SM100_MAX_DYN_SMEM``; duplicated rather than imported
# so the backward pass does not depend on the forward's config module.
_SM100_MAX_DYN_SMEM = 227 * 1024

# TMEM columns on Blackwell.  576 is Rubin, and needs ``is_exclusive=True``,
# which is fenced out of the public cutlass-dsl wheel — do not raise this.
_SM100_TMEM_COLS = 512


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph compile-time parameters threaded into the stage-2 template.

    Shapes are deliberately absent: they ride the template's per-shape
    ``compile()`` cache. Batch and head are absent for the same reason *and*
    because they are host-loop coordinates — the kernel takes ``head_base`` /
    ``batch_base`` as runtime Int32 arguments.
    """

    dtype_qkv: int = DTYPE_BF16
    # Mask band. ``window_right`` set => causal; ``window_left`` set => SWA.
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    bottom_right: bool = False
    seq_kv_lens_present: bool = False
    seq_q_lens_present: bool = False
    # THD / varlen: Q/K/V/O/dO and the gradients are PACKED [1, T, H, D] and the
    # per-sequence lengths come from a device metadata buffer, not from the
    # shapes.  Mutually exclusive with the dense per-batch length tensors above
    # -- both describe "ragged", and threading two sources of truth for the same
    # fact is how they drift apart.
    thd_varlen: bool = False
    sched_policy: int = SCHED_NATURAL
    # Tuning knob: halves of the fp32 S tile shipped sg0 -> sg1 per kv step.
    # 2 keeps the cross-sub-group ring 2-deep at an identical byte footprint;
    # 1 ships the whole tile and hard-stalls sg0 on sg1.
    xfer_halves: int = 2


@dataclass(frozen=True)
class MatmulTemplateParams:
    """Per-GEMM compile-time parameters for the stage-3 gradient GEMMs.

    Lives HERE rather than in the kernel file because a template loaded through
    ``frost.template_loader`` is executed before it is registered in
    ``sys.modules``, and a module-scope ``@dataclass`` needs its own module to be
    importable while the decorator runs. Same reason ``TemplateParams`` above is
    defined here and not in the stage-2 template.

    Only the operand-major pair varies: the three stage-3 GEMMs share one tile
    config, and the rendered bodies for the different majors are textually
    identical apart from ten constants (see the kernel's ``_MAJOR_CONSTS``).

        dV = S^T.dO, dK = dS^T.Q -> a_is_m_major=True   (kv is contiguous and is M)
        dQ = dS.K                -> a_is_m_major=False  (kv is contiguous and is K)

    ``b_is_n_major`` is True for all three: D is contiguous in BSHD and D is N.
    """

    a_is_m_major: bool = False
    b_is_n_major: bool = True
    # THD / varlen: the S/dS A operand is the BLOCKED workspace, B and the
    # output are PACKED, and the per-group offsets come from the metadata the
    # setup launch published.  Which axis is ragged follows from
    # ``a_is_m_major``, so it needs no parameter of its own.
    thd_varlen: bool = False
    # Causal K-trim.  Stage 2 under a causal mask leaves the tiles above the
    # diagonal UNWRITTEN, so this is a correctness requirement before it is an
    # optimization: a stage-3 GEMM must not read them.
    #
    #   CAUSAL_K_NONE  dense -- full K range.
    #   CAUSAL_K_LO    dV = S^T.dO and dK = dS^T.Q.  Output row is kv; S[q,kv]
    #                  is written only for q >= the stage-2 block containing kv,
    #                  so K (= q) STARTS there.
    #   CAUSAL_K_HI    dQ = dS.K.  Output row is q; dS[q,kv] is written only for
    #                  kv < the end of q's stage-2 block, so K (= kv) ENDS there.
    #
    # ``causal_gran`` is stage 2's write granularity in elements -- its
    # ``TILE_M * CTA_MMA`` (256), because its kv bound is taken over the whole
    # cluster q span.  Rounding to it is what makes "never read an unwritten
    # tile" exact rather than approximate.
    causal_mode: int = CAUSAL_K_NONE
    causal_gran: int = 0
    # How far the non-zero band extends PAST the plain kv <= q diagonal, in
    # elements.  Two things push it out and they add:
    #   * `diagonal_band_right_bound` (band widening) -- kv <= q + W;
    #   * bottom-right alignment      -- kv <= q + (S_kv - S_q).
    # Stage 2 writes those columns, so a trim that ignores them cuts away real
    # data. Negative is legal (bottom-right with S_kv < S_q pulls the band IN).
    causal_shift: int = 0
    # Epilogue store vector, in BYTES.  The output N is the head dim, and the
    # epilogue stores N in `vec_bytes_epi / sizeof(out)` element chunks -- so 32
    # requires d % 16 == 0 and 16 only d % 8 == 0.  Rendering the upstream
    # template at an N % 8 shape changes THIS CONSTANT AND NOTHING ELSE (the
    # bodies are textually identical), which is why it is a plain knob here.
    # Use `vec_bytes_epi_for(d, bpe)` rather than setting it by hand.
    vec_bytes_epi: int = 32
    # IO dtype of A/B (and, at the default `dtype_out`, D), as a DTYPE_* code.
    # Must match the stage-2 template's dS workspace dtype: stage 3 reads the
    # S/dS workspace stage 2 wrote.  BF16 / FP16 are both 2 B/element and both
    # take the Tcgen05MMAKind.F16 path, so between them this changes only the
    # dtype TOKENS in the rendered body -- every byte-size constant (swizzle,
    # box dims, SMEM staging) is width-driven and unaffected.  E4M3 selects the
    # fp8 ARM (the sm107 d256 fp8 chain: an e4m3 dS workspace scaled by scale_dP
    # against the e4m3 Q / K payloads): Rubin's dense-FP8 K64 MMA form
    # (256x256x64, idesc k_dim=1, Tcgen05MMAKind.F8F6F4), 128 e4m3 per K stage
    # in the same 128-B swizzle row, the (256, 256) row's tile, and one of the
    # epilogues below -- NEVER EPI_NONE (an undescaled fp8 accumulator has no
    # consumer).  The K64 form is silently WRONG on Blackwell (rules/
    # mma-tma-matrix.md S1); only the sm107 adapter renders it and
    # `kernels/sm107/prepared_host._check_target` is the runtime backstop.
    dtype_qkv: int = DTYPE_BF16
    # The CLUSTER's output tile ``(M, N)``, selecting one row of the template's
    # tile-constants table (``bprop_matmul_blackwell._TILE_ROWS``).  The N tile
    # is what matters: the grid covers ``ceil(n / N)`` cluster tiles along the
    # head dim, so a rendering whose N exceeds the head dim computes PADDING --
    # at d = 256 the default (512, 512) row (cluster 2x2, 512 x 512) spends the
    # two N-rank CTAs of every cluster on columns 256..511 that do not exist
    # (TMA-OOB zero loads, clipped stores): half of every cluster's MMA work.
    #   (512, 512)  cluster (2,2,1), CTA tile 256 x 128, 4 stages, one 512-col
    #               accumulator -- the SM100 d512 chain's rendering (N = 512
    #               fills it; measured faster than 2x1 there, see the template).
    #   (256, 256)  cluster (2,1,1), CTA tile 128 x 128, 6 stages, TWO 256-col
    #               accumulator stages (the epilogue overlaps the next tile's
    #               mainloop) -- no padding at d = 256; the sm107 d256 chain.
    #   (512, 256)  cluster (2,1,1), CTA tile 256 x 128, 4 stages, one 512-col
    #               accumulator -- an A/B alternate, selected by no adapter.
    # Append-only, defaulted: the SM100 adapter never sets it, so its records
    # render exactly the constants they always did.
    cgrp_tile_mn: tuple = (512, 512)
    # The fp8 arm's EPILOGUE (dtype_qkv == DTYPE_E4M3 only; the bf16 / fp16 rows
    # round the fp32 accumulator once into the io dtype and nothing else):
    #   EPI_NONE     acc.to(out)                          bf16 / fp16 rows only
    #   EPI_DESCALE  (acc * d0 * d1).to(out)              the TRUE-unit gradient in
    #                bf16: d0 * d1 = descale_dP * descale_{q|k} (the e4m3 dS
    #                workspace carries scale_dP, the e4m3 Q / K payload its own
    #                descale) -- the per-Q-head dK partial the GQA fold sums
    #                BEFORE quantizing
    #   EPI_QUANT    t = acc * d0 * d1; amax = max |t| (one int32-bit-pattern
    #                atomicMax per epilogue warp onto a caller-ZEROED fp32 [1]);
    #                (t * s_out).to(out) -- the quantized gradient straight into
    #                the caller's dQ / dK (e4m3 with scale_dQ / dK, or bf16 /
    #                fp16 with scale 1.0) plus its amax_dQ / dK
    # The four scalars ride as TRAILING `Optional[cute.Tensor] = None` arguments
    # of the template's `_host` (descale_0, descale_1, scale_out, amax); at
    # EPI_NONE every use folds out, so the SM100 chain's positional 7-argument
    # call renders exactly what it always did.
    epi_mode: int = EPI_NONE
    # Output (D) dtype code.  -1 = inherit: dtype_qkv on the bf16 / fp16 rows,
    # BF16 on the fp8 arm (the DESCALE true-unit value).  E4M3 needs EPI_QUANT.
    dtype_out: int = -1
    # The band's SECOND edge (append-only, defaulted: every rendering that existed
    # before these two fields -- the SM100 d512 chain's ten, the sm107 dense and
    # plain-causal ones -- renders exactly what it did; the proof is a PTX md5 per
    # rendering, frost_dev/results/bwd_d256_sm107/parity/renderings/RENDERINGS.md).
    #   ``causal_window``  the sliding window W in elements, as the stage-2 kernel
    #                      applies it (it keeps kv >= q + shift - W, i.e. W + 1
    #                      keys per row -- ``compute_q_loop_bounds`` / the bodies'
    #                      ``_mask_p_chunk``); 0 = no window.  With a window each
    #                      mode gains its second bound: CAUSAL_K_LO (M = kv, K = q)
    #                      an UPPER q bound (q <= kv - shift + W), CAUSAL_K_HI
    #                      (M = q, K = kv) a LOWER kv bound (kv >= q + shift - W),
    #                      both rounded OUTWARD to the k tile.  Without it the
    #                      window's zero tiles -- (S - W) / S of every K range -- are
    #                      read and multiplied (measured 80 % of the sm107 d256
    #                      SWA-640 backward, fractal 2026-09-24, PERF.md).
    #   ``causal_diag``    whether the causal edge kv <= q + shift is part of the
    #                      band.  False = a window WITHOUT a causal diagonal (SWA
    #                      only): ``causal_mode`` then just names the output axis,
    #                      the diagonal's bound is dropped and ``causal_shift`` is 0.
    # A window needs a mode (LO / HI names the axis) and a band needs an edge
    # (``causal_diag`` or a window); ``validate_matmul_params`` refuses the rest.
    # See ``bprop_matmul_blackwell._causal_k_range`` for the tile arithmetic and
    # the invariant the stage-2 kernel owes it.
    causal_window: int = 0
    causal_diag: bool = True
    # The B operand's head GROUP (append-only, defaulted: every rendering that existed before this field renders exactly
    # what it did -- the proof is a PTX md5 per shipped stage-3 record, rendered with and without the field).  The (b, h)
    # batch decodes ONE head ``h`` from the flat CLC batch index; A and C take it as is, B takes ``h // b_head_group`` and
    # its descriptor's head extent is ``n_head // b_head_group``.  1 = B batched per A/C head (every rendering before this
    # field).  Under GQA the dQ GEMM's B is K, shared by ``group`` consecutive Q heads: ``b_head_group = group`` lets ONE
    # launch cover every Q head of a chunk (A = the whole dS chunk, C = the whole dQ chunk, B = the chunk's KV heads),
    # where ``b_head_group = 1`` needed one launch per group MEMBER over every ``group``-th head -- sixteen under-one-wave
    # launches at H_q / H_kv = 16 (a Rubin d=256 backward at B=1 H_q=32 H_kv=2 S=8K causal spent 0.58 ms in them against
    # 0.31 ms for the dK GEMM of the same FLOPs).  The runtime ``n_head`` must be a multiple of it (the sm107 adapter's head
    # chunk is a multiple of the GQA group, ``config_sm107.validate_head_chunk``).  Not offered on the THD leg (the packed B
    # descriptor's head extent was not validated there; ``validate_matmul_params`` refuses it).
    b_head_group: int = 1
    # The BLOCK-SCALE (MXFP8) arm -- append-only, defaulted: every record built before this field existed renders exactly
    # what it did (a PTX md5 per shipped rendering pins it).  True selects the F8_128x4 block-scaled K64 MMA on the fp8
    # (256, 256) row: A and B are e4m3 payloads whose 32-element K blocks each carry an E8M0 scale byte
    # (``STAGE3_MX_BLOCK``); the scales ride their own SMEM rings declared AHEAD of the operand rings, are UTCCP'd into
    # 4 (SFA) + 8 (SFB) TMEM columns past the two accumulator stages (a 576-column EXCLUSIVE allocation,
    # ``STAGE3_BLOCK_SCALE_TMEM_COLS`` -- the Rubin line only) and dequantize IN the MMA (``tcgen05.mma.block_scale`` kind
    # MXF8F6F4, BLOCK32, idesc ``k_dim=1``, scale ids 0 / 2 for the two k-blocks of a 128-B K stage), so the fp32 accumulator
    # is the TRUE-unit gradient and the epilogue is EPI_NONE (one bf16 / fp16 rounding).  Operands, per the MXFP8 d256
    # backward's workspace contract: A = the block-scaled dS workspace with its SF atoms ``[B, H, M/128, K/128, 512]``
    # (dK: M = kv, K = q, K-major; dQ: M = q, K = kv, M-major); B = the COLUMNWISE-quantized Q / K payload with its
    # D-plane-major columnwise SF (``sdpa.kernels._mxfp8_sf.build_columnwise_sf_desc``'s layout, both D planes per stage).
    # The two SF tensors ride as TRAILING ``Optional[cute.Tensor]`` arguments of the template's ``_host`` (``sfa_0``,
    # ``sfb_0``); the kernel's two SF tensor-map parameters are None-specialized away when this is False.
    block_scale: bool = False
    # THD: which TOKEN axis the blocked S/dS workspace's ROWS are (appended; requires ``thd_varlen``).  False = the SM100
    # d512 chain's Q-major workspace (rows = packed q tokens, so the k-major dQ GEMM's A offset lands on M and its B is K at
    # ``cu_k``; the m-major dV / dK GEMMs' A offset lands on K and their B is dO / Q at ``cu_q``).  True = a KV-major
    # workspace blocked over packed kv tokens (the sm107 d256 chain): the ROW-OFFSET placement is unchanged (the blocked row
    # axis is K for an m-major A and M for a k-major A in both layouts) but the token side FLIPS -- the k-major dK GEMM
    # reduces over q tokens (B = Q at ``cu_q``, ``k_len = s_q``, C = dK rows at ``cu_k``) and the m-major dQ GEMM over kv
    # tokens (B = K at ``cu_k``, ``k_len = s_kv``, C = dQ rows at ``cu_q``).  Keyed on the operand major alone (the
    # pre-field spelling) the dK GEMM would pair its Q operand with ``cu_k`` and reduce over ``s_kv``: finite, plausible,
    # wrong for every sequence but the first.  At the default every keyed expression equals the pre-field one, so the
    # SM100 THD renderings stay PTX-identical.
    thd_rows_kv: bool = False


# The cluster tiles the stage-3 template renders (see ``MatmulTemplateParams.cgrp_tile_mn``).
STAGE3_CGRP_TILES = ((512, 512), (256, 256), (512, 256))
# The fp8 arm is rendered at this row only (see ``MatmulTemplateParams.dtype_qkv``).
STAGE3_FP8_CGRP_TILE = (256, 256)
STAGE3_EPI_MODES = (EPI_NONE, EPI_DESCALE, EPI_QUANT)
# The block-scale arm (``MatmulTemplateParams.block_scale``): the E8M0 block along K, and the TMEM the arm allocates -- the
# 512 accumulator columns of the (256, 256) row plus 4 SFA + 8 SFB scale columns, in the Rubin line's 576-column EXCLUSIVE
# allocation (a 512-column part cannot serve it; ``kernels/sm107/prepared_host._check_target`` is the runtime backstop).
STAGE3_MX_BLOCK = 32
STAGE3_BLOCK_SCALE_TMEM_COLS = 576


def matmul_out_dtype(params: MatmulTemplateParams) -> int:
    """The stage-3 output (D) dtype code ``dtype_out`` resolves to: itself when set, else the io dtype on the bf16 / fp16
    rows and BF16 on the fp8 arm (the DESCALE epilogue's true-unit value)."""
    out = int(getattr(params, "dtype_out", -1))
    if out >= 0:
        return out
    return DTYPE_BF16 if int(params.dtype_qkv) == DTYPE_E4M3 else int(params.dtype_qkv)


def validate_matmul_params(params: MatmulTemplateParams) -> None:
    """Backstop on the stage-3 GEMM params, mirroring ``make_cfg_d512``'s role
    for stage 2. Public because the template calls it; reaching a raise here
    means the adapter built a record the Capabilities row should not have
    admitted."""
    if tuple(params.cgrp_tile_mn) not in STAGE3_CGRP_TILES:
        raise ValueError(
            f"SDPA bwd stage 3: cgrp_tile_mn must be one of {STAGE3_CGRP_TILES} (the template's tile-constants rows; the N tile "
            f"must not exceed the head dim or the cluster computes padding); got {params.cgrp_tile_mn!r}."
        )
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16, DTYPE_E4M3):
        raise ValueError(
            f"SDPA bwd stage 3: dtype_qkv must be DTYPE_BF16 ({DTYPE_BF16}), DTYPE_FP16 ({DTYPE_FP16}) or DTYPE_E4M3 ({DTYPE_E4M3}, the fp8 arm); "
            f"got {params.dtype_qkv}."
        )
    fp8 = params.dtype_qkv == DTYPE_E4M3
    epi_mode = int(getattr(params, "epi_mode", EPI_NONE))
    if epi_mode not in STAGE3_EPI_MODES:
        raise ValueError(f"SDPA bwd stage 3: epi_mode must be one of {STAGE3_EPI_MODES} (EPI_NONE / EPI_DESCALE / EPI_QUANT); got {epi_mode}.")
    if fp8 and tuple(params.cgrp_tile_mn) != STAGE3_FP8_CGRP_TILE:
        raise ValueError(
            f"SDPA bwd stage 3: the fp8 arm (dtype_qkv=DTYPE_E4M3) is rendered at the {STAGE3_FP8_CGRP_TILE} row only -- its constants are lifted "
            f"from the upstream Rubin rendering of that config (cluster 2x1, 128 x 128 x 128 e4m3 per CTA, K64 MMA) and no other row was validated; "
            f"got cgrp_tile_mn={params.cgrp_tile_mn!r}."
        )
    if fp8 and params.thd_varlen:
        raise ValueError("SDPA bwd stage 3: the fp8 arm has no THD / varlen leg (the sm107 d256 chain is dense BSHD only).")
    block_scale = bool(getattr(params, "block_scale", False))
    if block_scale and not fp8:
        raise ValueError(
            f"SDPA bwd stage 3: block_scale is the MXFP8 arm of the fp8 rendering (dtype_qkv=DTYPE_E4M3: e4m3 payloads with an E8M0 scale per "
            f"{STAGE3_MX_BLOCK} K elements); got dtype_qkv={params.dtype_qkv}."
        )
    if block_scale and epi_mode != EPI_NONE:
        raise ValueError(
            f"SDPA bwd stage 3: block_scale dequantizes IN the MMA (the fp32 accumulator is the true-unit gradient), so its epilogue is EPI_NONE -- "
            f"a descale / quantize epilogue (epi_mode={epi_mode}) belongs to the per-tensor fp8 arm."
        )
    if block_scale and int(getattr(params, "b_head_group", 1)) != 1:
        raise ValueError(
            f"SDPA bwd stage 3: block_scale indexes its B scale-factor descriptor per A / C head, so a block-scale record keeps b_head_group == 1 "
            f"(the dQ GEMM runs once per GQA group member); got b_head_group={params.b_head_group}.  The single-launch dQ (b_head_group == the "
            f"group) is the plain renderings' form; the block-scale arm takes it in a follow-up."
        )
    if (epi_mode != EPI_NONE) != (fp8 and not block_scale):
        raise ValueError(
            f"SDPA bwd stage 3: the descale / quantize epilogue (epi_mode={epi_mode}) belongs to the fp8 arm and the fp8 arm requires one: an e4m3 dS "
            f"workspace carries scale_dP and the e4m3 Q / K payloads their descale, so the accumulator must be descaled (EPI_DESCALE) or descaled + "
            f"quantized (EPI_QUANT) before it is stored; the bf16 / fp16 rows store the accumulator as is (EPI_NONE); got dtype_qkv={params.dtype_qkv}."
        )
    out = matmul_out_dtype(params)
    if out not in (DTYPE_BF16, DTYPE_FP16, DTYPE_E4M3):
        raise ValueError(f"SDPA bwd stage 3: dtype_out must be -1 (inherit), DTYPE_BF16, DTYPE_FP16 or DTYPE_E4M3; got {params.dtype_out}.")
    if out == DTYPE_E4M3 and epi_mode != EPI_QUANT:
        raise ValueError("SDPA bwd stage 3: an E4M3 output needs EPI_QUANT (an unscaled fp8 store of the accumulator has no consumer).")
    if not fp8 and out != params.dtype_qkv:
        raise ValueError(f"SDPA bwd stage 3: the bf16 / fp16 rows store the io dtype (dtype_out must be -1 or dtype_qkv={params.dtype_qkv}); got {out}.")
    if params.vec_bytes_epi not in (16, 32):
        raise ValueError(f"SM100 SDPA bwd d512 stage 3: vec_bytes_epi must be 16 or 32; got {params.vec_bytes_epi}.")
    window = int(getattr(params, "causal_window", 0))
    diag = bool(getattr(params, "causal_diag", True))
    if window < 0:
        raise ValueError(f"SDPA bwd stage 3: causal_window must be >= 0 (0 = no window; W = window_left as the stage-2 kernel applies it); got {window}.")
    if window > 0 and params.causal_mode == CAUSAL_K_NONE:
        raise ValueError(
            f"SDPA bwd stage 3: a window (causal_window={window}) needs causal_mode CAUSAL_K_LO or CAUSAL_K_HI to name the output axis -- "
            f"CAUSAL_K_NONE renders the full K range and would silently read every window-masked zero tile."
        )
    if not diag and params.causal_mode == CAUSAL_K_NONE:
        raise ValueError(
            "SDPA bwd stage 3: causal_diag=False only means something on a trimmed rendering (causal_mode LO / HI); render CAUSAL_K_NONE for dense."
        )
    if not diag and window == 0:
        raise ValueError(
            "SDPA bwd stage 3: a band with neither edge (causal_diag=False, causal_window=0) is dense -- render CAUSAL_K_NONE; a trimmed mode with no bound "
            "would still take the never-empty clamp path."
        )
    if not diag and params.causal_shift != 0:
        raise ValueError(
            f"SDPA bwd stage 3: causal_shift ({params.causal_shift}) is the causal diagonal's offset; without the diagonal (causal_diag=False) it must be 0 "
            f"(bottom-right alignment requires a causal band on every row)."
        )
    bhg = getattr(params, "b_head_group", 1)
    if isinstance(bhg, bool) or not isinstance(bhg, int) or bhg < 1:
        raise ValueError(
            f"SDPA bwd stage 3: b_head_group must be a positive int (1 = B batched per A/C head; the GQA group for a dQ GEMM whose B is the shared "
            f"K head); got {bhg!r}."
        )
    if bhg > 1 and params.thd_varlen:
        raise ValueError(
            f"SDPA bwd stage 3: b_head_group > 1 ({bhg}) has no THD / varlen leg (the packed B descriptor's head extent was not validated there; "
            f"the sm107 d256 chain that uses it is dense BSHD only)."
        )
    if bool(getattr(params, "thd_rows_kv", False)) and not params.thd_varlen:
        raise ValueError(
            "SDPA bwd stage 3: thd_rows_kv names the token axis of the THD blocked workspace's rows and means nothing on a dense rendering "
            "-- it requires thd_varlen=True."
        )
    if params.thd_varlen and params.causal_mode != CAUSAL_K_NONE:
        # The causal K-trim assumes the workspace is one dense rectangle per
        # (batch, head), which is exactly what THD's blocked layout is not: the
        # trim's `causal_gran` / `causal_shift` arithmetic is in ABSOLUTE
        # workspace rows. Rewriting it per group is a follow-up, not a silent
        # approximation -- so a CAUSAL packed graph is served by rendering stage
        # 3 UNTRIMMED (`SdpaBwdDslSm100.compile` forces CAUSAL_K_NONE and the
        # adapter zero-fills the workspace instead). This raise is what keeps
        # that the only spelling: it is not a decline of causal under THD.
        raise ValueError("SM100 SDPA bwd d512 stage 3: THD with a causal K-trim is not implemented (render it untrimmed; the caller zero-fills)")


def vec_bytes_epi_for(d: int, bpe: int = 2) -> int:
    """Widest legal stage-3 epilogue store vector for a head dim of ``d``.

    The epilogue writes the output's N (= the head dim) in
    ``vec_bytes_epi / bpe`` element chunks, and the compiled artifact declares
    that as a symbolic divisibility -- so a mismatched d fails at CALL time with
    *"expected to be divisible by 16"*, not at build time.  32 B is the fast
    path and needs ``d % 16 == 0``; 16 B relaxes that to ``d % 8 == 0``.
    """
    per32 = 32 // bpe
    return 32 if d % per32 == 0 else 16


def _validate_params(params: TemplateParams) -> None:
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(
            f"SM100 SDPA bwd d512: dtype_qkv must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); "
            f"got {params.dtype_qkv}. FP8/MXFP8 backward is not implemented on this arch."
        )
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"SM100 SDPA bwd d512: window_left must be non-negative; got {params.window_left}")
    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"SM100 SDPA bwd d512: window_right must be non-negative; got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("SM100 SDPA bwd d512: bottom_right alignment requires a causal upper bound (window_right)")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("SM100 SDPA bwd d512: seq_q_lens_present requires seq_kv_lens_present (padding mask)")
    if params.thd_varlen and (params.seq_kv_lens_present or params.seq_q_lens_present):
        raise ValueError(
            "SM100 SDPA bwd d512: thd_varlen is mutually exclusive with seq_kv_lens_present / "
            "seq_q_lens_present -- THD carries its per-sequence lengths in the metadata buffer"
        )
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"SM100 SDPA bwd d512: sched_policy must be one of NATURAL/LPT/LPT_L2; got {params.sched_policy}")
    if params.xfer_halves not in (1, 2):
        raise ValueError(f"SM100 SDPA bwd d512: xfer_halves must be 1 or 2; got {params.xfer_halves}")


def _mask_flags_from(params: TemplateParams) -> int:
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.seq_kv_lens_present or params.thd_varlen:
        # THD is padded BY CONSTRUCTION: every sequence has its own S_q / S_kv,
        # so the tile tails need the per-cell mask exactly as a padding mask
        # does.  The lengths just come from the metadata buffer instead.
        flags |= MASK_PADDED
    return flags


def bpe(dtype: int) -> int:
    return 1 if dtype <= DTYPE_E5M2 else 2


@dataclass(frozen=True)
class CfgBwdD512:
    """Stage-2 geometry: d_qk = d_v = 512, SM100, cga4x1 role split.

    Identifiers are keyed by op geometry, not by the model this was tuned for
    (frost-engine-contract.md §8).
    """

    TILE_M: int = 128  # q rows per CTA
    TILE_N: int = 128  # kv cols per kernel tile (collective MMA N)
    TILE_K: int = 512  # d_qk
    TILE_O: int = 512  # d_v

    DTYPE_QKV: int = DTYPE_BF16
    BPE: int = 2

    # cga4x1 = two sub-groups of two CTAs.  sg0: BMM1 + softmax; sg1: BMM2 + dS.
    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    DO_SWZ_BYTES: int = 128
    S_SWZ_BYTES: int = 128

    # f16/bf16 is 1-chunk only on SM10x; 2-chunk (32) is silently wrong.
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    STAGES_KV: int = 2  # K ring on sg0 / V ring on sg1 (SMEM-cap driven)
    STAGES_ACC: int = 2  # S_acc / dS_acc parity slots (TMEM-cap driven)
    XFER_HALVES: int = 2  # fp32 S halves shipped sg0 -> sg1
    SCHEDULER_STAGES: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    SOFTMAX_WG_WARPS: int = 4
    CORRECTION_WARPS: int = 0

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    TOTAL_WARPS: int = 8
    THREADS_PER_CTA: int = 8 * 32
    SOFTMAX_WG0_BASE: int = 0
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TMASTG_WARP_ID: int = 6
    SCHED_WARP_ID: int = 7

    # --- mbarrier arrival counts ---------------------------------------
    # P3: an init count must equal the EXACT sum of producer arrivals per
    # phase, and the constant belongs next to its derivation.  Note
    # ``.arrive()`` fires on every LANE of the calling warp, not once per warp.
    ONE_LANE: int = 1  # a single elect_sync'd lane
    ONE_WARP: int = 32  # an un-elected THREAD arrive from one warp
    COMPUTE_LANES: int = 4 * 32  # SOFTMAX_WG_WARPS * 32, the compute warp group
    # *_acc_empty: every lane of the compute WG on BOTH CTAs of the pair
    # arrive_on_peer's the pair leader.
    ACC_EMPTY_ARRIVERS: int = 4 * 32 * 2  # COMPUTE_LANES * CTA_MMA

    # Scheduler ring.  Each calling warp delivers exactly ONE arrive to each
    # CTA of the cluster, so this is the count of (warp, CTA) instances that
    # call read_tile_id_arrive.  EVERY warp role except the scheduler warp
    # itself calls it, on every CTA of the cluster:
    #     (SOFTMAX_WG_WARPS + TMA-LDG + TMA-STG + MMA) * CGA_SIZE
    #   = (4 + 1 + 1 + 1) * 4 = 28
    # The MMA term counts BOTH arms: the non-leader is a quiet warp but it
    # still runs the persistent loop and must stay in step, so it arrives too.
    #
    # The forward d512's 25 is NOT comparable and must not be copied: its
    # TMA-STG runs on sg1 only, and only its sg1 non-leader MMA warp arrives.
    #
    # Getting this too LOW is the dangerous direction: the barrier completes
    # before every role has read the payload, so the ring advances a tile early
    # every iteration and the kernel wedges intermittently -- it does not fail
    # cleanly (P3).  An init of 26 against 28 arrivers cost a debugging session.
    READ_TILE_ARRIVERS: int = 28

    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    # Row granularity of the BLOCKED S/dS workspace under THD: sequence b owns
    # ceil(S_q[b] / this) * this rows at row_off[b].  It is TILE_M, not the
    # cluster span -- see tile_dsl/thd.write_thd_row_offsets for why it is
    # bracketed below by stage 3's k-tile and above by nothing but waste.
    WS_BLOCK_ROWS: int = 128

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL


# ---------------------------------------------------------------------------
# Resource derivations — the single source of truth for both the validator
# below and the kernel's allocation block.  Never inline these numbers.
# ---------------------------------------------------------------------------


def operand_bytes(cfg: CfgBwdD512) -> Tuple[int, int, int, int]:
    """(Q, dO, K-per-stage, V-per-stage) SMEM bytes for ONE CTA.

    K and V are split along the collective MMA-N across the CTA pair, so each
    CTA holds ``TILE_N / CTA_MMA`` rows of them; Q and dO are whole-block.
    """
    q = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    do = cfg.TILE_M * cfg.TILE_O * cfg.BPE
    k = (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE
    v = (cfg.TILE_N * cfg.TILE_O // cfg.CTA_MMA) * cfg.BPE
    return q, do, k, v


def xfer_bytes(cfg: CfgBwdD512) -> int:
    """fp32 S staged for the cross-sub-group ship, summed over the ring.

    fp32 and not the io dtype on purpose: the Rubin reference never transfers S
    at all (it multiplies the fp32 register copy straight into dS), so shipping
    a rounded S across the role split would be an accuracy regression the
    reference does not have.  Halving the tile keeps the ring 2-deep
    at an identical footprint.
    """
    return cfg.XFER_HALVES * cfg.TILE_M * (cfg.TILE_N // cfg.XFER_HALVES) * 4


def s_tma_iters(cfg: CfgBwdD512) -> int:
    """Workspace TMA subtiles per stored tile.

    One row of a subtile must be exactly one swizzle atom, so this is
    ``(TILE_N * BPE) // S_SWZ_BYTES`` -- 2 at bf16/TILE_N=128/128 B.  The
    validator pins ``XFER_HALVES`` to it: one shipped fp32 half is exactly one
    stored subtile is exactly one softmax chunk.
    """
    return (cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES


def cast_bytes(cfg: CfgBwdD512) -> int:
    """io-dtype staging for one workspace tile (S on sg0, dS on sg1).

    Separate from the fp32 xfer buffer so the fp32 ring slot is released on
    cast completion instead of on ``tma_store_wait``.
    """
    return cfg.TILE_M * cfg.TILE_N * cfg.BPE


def smem_bytes(cfg: CfgBwdD512) -> Tuple[int, int]:
    """(sg0, sg1) SMEM tensor bytes per CTA.

    Each sub-group's operand slab is a max-union alias, not a sum: the Q (resp.
    dO) staging buffer is dead once the UTCCP has moved it to TMEM, so the K
    (resp. V) ring reuses those exact bytes behind the alias-seam barrier.
    """
    q, do, k, v = operand_bytes(cfg)
    sg0_alias = max(q, cfg.STAGES_KV * k)
    sg1_alias = max(do, cfg.STAGES_KV * v)
    common = xfer_bytes(cfg) + cast_bytes(cfg)
    return sg0_alias + common, sg1_alias + common


def tmem_cols(cfg: CfgBwdD512) -> Tuple[int, int]:
    """(sg0, sg1) TMEM columns.

    An accumulator is fp32 and TILE_N wide => TILE_N columns per parity slot.
    A 16-bit operand packs 2 elements per 32-bit column, so [TILE_M x d] costs
    ``d * BPE / 4`` columns -- 256 at d=512, which is exactly what leaves room
    for the two accumulator slots under the 512-column cap.
    """
    acc = cfg.STAGES_ACC * cfg.TILE_N
    return acc + (cfg.TILE_K * cfg.BPE) // 4, acc + (cfg.TILE_O * cfg.BPE) // 4


def _validate_cfg_d512(cfg: CfgBwdD512) -> None:
    """Consistency checks on the (mostly hardcoded) stage-2 d512 geometry."""
    sg0_smem, sg1_smem = smem_bytes(cfg)
    sg0_tmem, sg1_tmem = tmem_cols(cfg)
    checks = (
        # --- register split: all four are hardware constraints -------------
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "bwd d512: MMA/TMALDG/TMASTG/Scheduler regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "bwd d512: register budget over 512"),
        (
            cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0,
            "bwd d512: per-role regs must be multiples of 8",
        ),
        (24 <= cfg.SOFTMAX_REGS <= 256 and 24 <= cfg.MMA_REGS <= 256, "bwd d512: per-warp regs must be within [24, 256]"),
        # --- geometry ------------------------------------------------------
        (cfg.TILE_M == 128 and cfg.TILE_N == 128, "bwd d512: TILE_M = TILE_N = 128"),
        (cfg.TILE_K == 512 and cfg.TILE_O == 512, "bwd d512: d_qk = d_v = 512"),
        (cfg.CGA_M == 4 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, "bwd d512: cga4x1 / CTA_MMA=2 only"),
        (cfg.CGA_M // cfg.CTA_MMA == 2, "bwd d512 (role split): exactly two sub-groups (CGA_M / CTA_MMA == 2)"),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 0, "bwd d512 (role split): one softmax warpgroup, no correction warp"),
        (cfg.TILE_N % cfg.XFER_HALVES == 0, "bwd d512: XFER_HALVES must divide TILE_N"),
        # A four-way alignment that the store layout, the ship ring and the
        # softmax register budget all depend on: one fp32 S half == one softmax
        # chunk == one workspace TMA subtile == one 128 B swizzle atom per row.
        # (TILE_N * BPE) // S_SWZ_BYTES is the subtile count; at TILE_N=128,
        # BPE=2, 128 B swizzle that is 2, and XFER_HALVES=2 matches it.  The
        # forward asserts the same coincidence as P_TMA_ITERS == P_XFER_HALVES.
        (
            cfg.XFER_HALVES == (cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES,
            f"bwd d512: XFER_HALVES ({cfg.XFER_HALVES}) must equal the workspace TMA subtile count "
            f"({(cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES}) -- one ship half must be exactly one stored subtile",
        ),
        # --- MMA: bf16/fp16 is 1-chunk only on SM10x -----------------------
        (
            cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16,
            "bwd d512 f16/bf16: TILE_K_HW must be 16 (1-chunk only on SM10x -- 2-chunk silently wrong)",
        ),
        (cfg.BPE == bpe(cfg.DTYPE_QKV), "bwd d512: BPE must match DTYPE_QKV"),
        # The blocked workspace's granularity is bracketed on BOTH sides: at
        # least stage 3's k-tile, so a dV/dK k-loop never reads past a block
        # into the next sequence's live rows; and at least stage 2's per-CTA
        # store box, so an out-of-range box is a boolean skip rather than a
        # clipped per-sequence descriptor.  TILE_M satisfies both.
        (cfg.WS_BLOCK_ROWS == cfg.TILE_M, "bwd d512: WS_BLOCK_ROWS must be TILE_M (stage 2's per-CTA store box)"),
        (cfg.WS_BLOCK_ROWS % 64 == 0, "bwd d512: WS_BLOCK_ROWS must be a multiple of stage 3's 64-row k-tile"),
        (cfg.THD_VARLEN == 0 or (cfg.MASK_FLAGS & MASK_PADDED) != 0, "bwd d512: THD implies the padded per-cell mask"),
        # The workspace dtype IS the io dtype.  An earlier Rubin revision tied
        # it to an independent output dtype and produced a silent mismatch
        # against the stage-3 GEMMs, which read the workspace as the io dtype.
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "bwd d512: workspace/io dtype must be BF16 or FP16"),
        # --- swizzle -------------------------------------------------------
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.DO_SWZ_BYTES == 128,
            "bwd d512: Q/K/V/dO swizzle must all be 128B",
        ),
        # Not only a swizzle width: S_SWZ_BYTES alone sets the fp32 ship's
        # 256 B per-lane row, which S_XFER_SWIZZLE's sshift is derived from.
        # A moved S mismatches that swizzle -- bank-conflicted, not wrong.
        (cfg.S_SWZ_BYTES == 128, f"bwd d512: S swizzle must be 128B (sets the fp32 ship's 256 B row); got {cfg.S_SWZ_BYTES}"),
        # --- caps ----------------------------------------------------------
        (
            sg0_smem <= _SM100_MAX_DYN_SMEM,
            f"bwd d512: sg0 SMEM {sg0_smem / 1024:.1f} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (
            sg1_smem <= _SM100_MAX_DYN_SMEM,
            f"bwd d512: sg1 SMEM {sg1_smem / 1024:.1f} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (
            sg0_tmem <= _SM100_TMEM_COLS,
            f"bwd d512: sg0 TMEM carve {sg0_tmem} over the {_SM100_TMEM_COLS}-column Blackwell cap",
        ),
        (
            sg1_tmem <= _SM100_TMEM_COLS,
            f"bwd d512: sg1 TMEM carve {sg1_tmem} over the {_SM100_TMEM_COLS}-column Blackwell cap",
        ),
        # --- masks ---------------------------------------------------------
        (not (cfg.BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)), "bwd d512: bottom_right requires a causal band"),
        (
            cfg.READ_TILE_ARRIVERS == (cfg.SOFTMAX_WG_WARPS + 3) * cfg.CGA_M * cfg.CGA_N,
            f"bwd d512: READ_TILE_ARRIVERS ({cfg.READ_TILE_ARRIVERS}) must equal "
            f"{(cfg.SOFTMAX_WG_WARPS + 3) * cfg.CGA_M * cfg.CGA_N} "
            "= (softmax warps + TMA-LDG + TMA-STG + MMA) * CGA_SIZE -- every role but the scheduler, on every CTA",
        ),
        (cfg.COMPUTE_LANES == cfg.SOFTMAX_WG_WARPS * 32, "bwd d512: COMPUTE_LANES must be SOFTMAX_WG_WARPS * 32"),
        # v1 scope, deliberately narrow so the engine row cannot over-promise:
        # dense only, natural (grid-mapped) schedule.  Masks and persistence are
        # Phase 4 / Phase 5; each needs its own capabilities row + reject test.
        # Causal is implemented (tile-level triangular skip + a post-exp2
        # per-cell mask on the diagonal tiles).  SWA and padding are NOT, and
        # neither is bottom-right alignment: under bottom-right with
        # S_kv < S_q the leading q tiles get an EMPTY kv range, and this
        # kernel has no empty-tile path -- every cross-CTA ring would have to
        # fire a bookkeeping arrive so the four CTAs stay in step.  Top-left
        # causal always keeps kv_right >= 1, which is why it needs none.
        # Causal (top-left and bottom-right) and SWA are implemented: the tile
        # bounds come from the shared `compute_kv_loop_bounds` and the per-cell
        # mask from `apply_mask_chunk`, both driven by MASK_FLAGS.
        #
        # An EMPTY kv range (which bottom-right and SWA both admit) needs no
        # special path here: every ring except the per-tile operand one is
        # per-kv-iteration, so a zero-trip loop fires nothing, and
        # `_residual_depth(0, stages)` is 0. The three conditions that make that
        # safe all hold -- all four CTAs derive IDENTICAL bounds (they are keyed
        # on the cluster's q span), `read_tile_id_arrive` sits ABOVE the kv loop,
        # and the cast/xfer buffers are not aliased into the operand slab.
        #
        # Padding (`seq_kv_lens`) is still out: it needs the PER-BATCH kv length
        # at the bounds and mask sites, and this kernel threads only the scalar.
        # Padding IS implemented, for a UNIFORM length: the engine rounds the
        # compile shape up to the tile and passes the real S_q / S_kv, and the
        # kernel masks the tail (kv side through apply_mask_chunk, q side by
        # zeroing the row). That is what makes a sequence length which is not a
        # multiple of the tile work at all. A PER-BATCH seq_len still needs the
        # per-batch value threaded to the bounds and mask sites.
        # LPT / LPT_L2 need lpt_tile_coords and its L2-residency model, which
        # bwd/kernels/sm100/_common.py deliberately does not copy.
        (cfg.SCHEDULER_POLICY == SCHED_NATURAL, "bwd d512 v1: only SCHED_NATURAL is implemented (LPT/LPT_L2 need the L2 tile-coord model)"),
        (cfg.ACC_EMPTY_ARRIVERS == cfg.COMPUTE_LANES * cfg.CTA_MMA, "bwd d512: ACC_EMPTY_ARRIVERS must be COMPUTE_LANES * CTA_MMA"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> CfgBwdD512:
    _validate_params(params)
    b = bpe(params.dtype_qkv)
    cfg = CfgBwdD512(
        DTYPE_QKV=params.dtype_qkv,
        BPE=b,
        XFER_HALVES=params.xfer_halves,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        BOTTOM_RIGHT=int(params.bottom_right),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy,
    )
    _validate_cfg_d512(cfg)
    return cfg
