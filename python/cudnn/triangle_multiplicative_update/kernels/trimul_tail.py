# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 -- LayerNorm of the contraction output ``y`` (over the channel), LayerNorm of the pair representation ``z`` (recomputed, the K1 operand),
the dual GEMM (output projection of ``u = LN(y)`` ‖ output gate of ``z_in = LN(z)``), sigmoid * mul -> the pair-major update.

For every pair row ``(b, i, j)`` (the contraction wrote ``y`` CHANNEL-major, ``y[c][b * N + i][j]``, with the ``j`` axis padded like K1's planes)::

    u      = LayerNorm_c(y[:, b N + i, j]; norm_out_weight, norm_out_bias, eps)      fp32 statistics, bf16 operand   (MN-major rows)
    z_in   = LayerNorm_{c_z}(z[b, i, j, :]; norm_in_weight, norm_in_bias, eps)        the same arithmetic as K1       (K-major rows)
    out[b, i, j, :] = bf16( sigmoid(z_in W_g^T + b_g) * (u W_o^T + b_o) )             pair-major [B, N, N, c_z]; the caller adds the residual

(AlphaFold 3 Algorithm 12 / 13 steps 5-6; cuEquivariance's ``p_out_weight`` [c_z, c], ``g_out_weight`` [c_z, c_z], optional biases.)  The
output is not masked -- the pair mask acted on ``a`` and ``b`` inside K1 and is already inside ``y``.  The per-row ``(mean, rstd)`` of ``u``
(LayerNorm 2) are written once for the backward (K1 wrote LayerNorm 1's).

This module writes no kernel: it is the triangle multiplicative update's SECOND instance of the LayerNorm-prologue dual GEMM substrate
``cudnn.gemm.frost.ln_prologue`` -- the ``pairing = "accumulators"`` arm (two GEMMs of N = c_z, the value from the MN-major ``y`` operand,
the gate from the K-major ``z`` operand, the pairing across the two accumulators, a pair-major store); K1 is the ``"columns"`` arm of the same
body.  Two NUMERICS MODES -- op attributes, never performance knobs (they change the function, so no heuristic or autotuner sets them; both OFF):
``affine_fold`` (gamma_out / beta_out folded into ``W_o`` / ``b_o`` on the host by :func:`cudnn.gemm.frost.ln_prologue.fold_output_affine`; the
kernel's LayerNorm 2 is the pure ``(y - mean) * rstd``; a second bf16 rounding of the weights and the bias, measured ~2.3x the unfolded max-abs
error) and ``fuse_residual`` (``out = bf16(z + update)`` in one rounding -- the fused block form; the cuEquivariance-signature API rounds twice
like the reference).  Performance knobs (the same function, bitwise, under every value): ``a0_prologue`` (``"mn_major"`` normalises ``y`` in
its MN-major SMEM layout; ``"transpose"`` writes the normalised row K-major -- the fallback form), ``clc_stages``, ``prefetch_a_l2`` (a measured
dead end, kept as the control of that experiment).

Traffic model (:func:`tail_bytes`): read ``y`` (2 B per pair-channel) and ``z`` (2 B), write the update (2 B), plus the statistics (8 B per
pair); FLOPs (:func:`tail_flop`): ``2 * B N^2 * (c * c_z + c_z * c_z)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from cudnn.gemm.frost.ln_prologue import (
    K_PAD_MULTIPLE,
    MIN_N_RES_MN_MAJOR,
    LnDualGemmParams,
    LnDualGemmPlan,
    RowBlockGeometry,
    build_ln_dual_gemm,
    fold_output_affine,
    interleave_value_gate_bias,
    row_block_geometry,
    run_ln_dual_gemm_tail,
)

_BF16_BYTES = 2


def tail_params(
    c_z: int,
    c: int,
    *,
    has_bias: bool,
    want_stats: bool = True,
    a0_prologue: str = "mn_major",
    affine_fold: bool = False,
    fuse_residual: bool = False,
    prefetch_a_l2: bool = False,
    eps: float = 1e-5,
    clc_stages: Optional[int] = None,
    diag: str = "",
    ln_pass_form: Optional[str] = None,
    epi_math: Optional[str] = None,
    min_blocks_per_mp: Optional[int] = None,
    ln_once: Optional[str] = None,
) -> LnDualGemmParams:
    """The substrate's compile-time parameters of K3 for the ``(c_z, c)`` geometry.  Both GEMMs share one K (the ring stage is one k-tile of
    every operand), so ``c == c_z`` for now -- the Protenix-v2 ``c_z = 256, c = 128`` geometry is a typed decline until the per-GEMM k-tile
    count lands.  ``ln_pass_form`` / ``epi_math`` / ``min_blocks_per_mp`` (performance only, bitwise the same output; the last one is the
    launch's ``.minnctapersm`` hint under which ptxas honours the body's register split) default to the substrate's."""
    if int(c) != int(c_z):
        raise ValueError(f"K3 on the substrate needs c == c_z (one K for the value GEMM over y and the gate GEMM over z), got c={c}, c_z={c_z}")
    kw = {} if clc_stages is None else {"clc_stages": int(clc_stages)}
    if diag:
        kw["diag"] = str(diag)
    if ln_pass_form is not None:
        kw["ln_pass_form"] = str(ln_pass_form)
    if epi_math is not None:
        kw["epi_math"] = str(epi_math)
    if min_blocks_per_mp is not None:
        kw["min_blocks_per_mp"] = (
            min_blocks_per_mp if isinstance(min_blocks_per_mp, bool) else int(min_blocks_per_mp)
        )  # a bool reaches the typed decline as itself
    if ln_once is not None:
        kw["ln_once"] = str(ln_once)  # the LayerNorm-once-per-row-block lever (performance only, bitwise; the substrate's typed vocabulary)
    return LnDualGemmParams(
        k=int(c),
        c_out=int(c_z),
        passes=1,
        has_bias=bool(has_bias),
        want_stats=bool(want_stats),
        has_mask=False,
        skip_masked_tiles=False,
        eps=float(eps),
        pairing="accumulators",
        a0_major="mn",
        a0_prologue=str(a0_prologue),
        affine_fold=bool(affine_fold),
        fuse_residual=bool(fuse_residual),
        prefetch_a_l2=bool(prefetch_a_l2),
        **kw,
    )


def n_pad(n_res: int) -> int:
    """The padded inner extent of the channel-major ``y`` (the same rule as K1's planes: a multiple of 8)."""
    return -(-int(n_res) // K_PAD_MULTIPLE) * K_PAD_MULTIPLE


def y_shape(c: int, batch: int, n_res: int) -> Tuple[int, int, int]:
    """``[c, B * N, n_pad]`` -- the channel-major contraction output K3 reads (K2's output has the planes' row geometry)."""
    return (int(c), int(batch) * int(n_res), n_pad(n_res))


def stack_tail_biases(p_out_bias, g_out_bias):
    """cuEquivariance's ``p_out_bias``, ``g_out_bias`` (``[c_z]``) -> the substrate's interleaved ``[2 c_z]`` (value block, gate block per 128)."""
    return interleave_value_gate_bias(p_out_bias, g_out_bias).contiguous()


def fold_tail_affine(p_out_weight, p_out_bias, norm_out_weight, norm_out_bias):
    """``affine_fold``: the folded ``(W_o', b_o')`` (bf16) for ``tail_params(..., affine_fold=True)``; ``p_out_bias`` may be None
    (the fold then yields the bias ``W_o beta_out`` alone, which the artifact must take with ``has_bias=True``)."""
    return fold_output_affine(p_out_weight, p_out_bias, norm_out_weight, norm_out_bias)


def build_trimul_tail(params: LnDualGemmParams) -> LnDualGemmPlan:
    """Compile (cached per params) -- shapes are runtime."""
    if not params.dual:
        raise ValueError("the triangle multiplicative update's K3 is the accumulators arm of the substrate; build it from tail_params(...)")
    return build_ln_dual_gemm(params)


def run_trimul_tail(
    plan: LnDualGemmPlan,
    y,  # [c, B * N, n_pad] bf16 contiguous (channel-major; y_shape)
    z,  # [B, N, N, c_z] bf16 contiguous
    p_out_weight,  # [c_z, c] bf16 (cuEquivariance's convention; the FOLDED W_o' under affine_fold)
    g_out_weight,  # [c_z, c_z] bf16
    out,  # [B, N, N, c_z] bf16 contiguous -- the update (the caller adds the residual unless fuse_residual)
    norm_out_weight,  # [c] bf16 (LayerNorm 2; bound but ignored by the kernel under affine_fold)
    norm_out_bias,  # [c] bf16
    norm_in_weight,  # [c_z] bf16 (LayerNorm 1, as K1)
    norm_in_bias,  # [c_z] bf16
    *,
    biases=None,  # [2 c_z] bf16 from stack_tail_biases, iff params.has_bias
    stats=None,  # [B * N * N, 2] fp32 (mean, rstd) of LayerNorm 2, iff params.want_stats
    residual=None,  # [B, N, N, c_z] bf16, iff params.fuse_residual (normally z itself)
    stream,
    k_block=None,  # k's per 256-row pair tile (the substrate's rule when None; >= 64 here -- the MN-major box; bitwise the same output)
) -> RowBlockGeometry:
    """Launch K3.  No allocation, no conversion."""
    if z.ndim != 4 or z.shape[1] != z.shape[2]:
        raise ValueError(f"z must be [B, N, N, c_z], got {tuple(z.shape)}")
    batch, n_res, _, c_z = (int(s) for s in z.shape)
    if c_z != plan.params.c_out or plan.params.k != c_z:
        raise ValueError(f"z has c_z={c_z} but the plan was compiled for c_out={plan.params.c_out}, K={plan.params.k}")
    if n_res < MIN_N_RES_MN_MAJOR:
        raise ValueError(f"K3 needs N >= {MIN_N_RES_MN_MAJOR} (the MN-major 64-row TMA box of one pair row block), got N={n_res}")
    if tuple(y.shape) != y_shape(plan.params.k, batch, n_res):
        raise ValueError(f"y must be {y_shape(plan.params.k, batch, n_res)} (channel-major, inner index padded), got {tuple(y.shape)}")
    if tuple(out.shape) != (batch, n_res, n_res, c_z):
        raise ValueError(f"out must be [B, N, N, c_z] = {(batch, n_res, n_res, c_z)}, got {tuple(out.shape)}")
    if residual is not None and tuple(residual.shape) != (batch, n_res, n_res, c_z):
        raise ValueError(f"residual must be [B, N, N, c_z] = {(batch, n_res, n_res, c_z)}, got {tuple(residual.shape)}")
    rows = batch * n_res * n_res
    return run_ln_dual_gemm_tail(
        plan,
        y,
        z.view(rows, c_z),
        p_out_weight,
        g_out_weight,
        out.view(rows, c_z),
        norm_out_weight,
        norm_out_bias,
        norm_in_weight,
        norm_in_bias,
        n_res=n_res,
        bn=batch * n_res,
        bias=biases,
        stats=stats,
        residual=residual.view(rows, c_z) if residual is not None else None,
        stream=stream,
        k_block=k_block,
    )


def tail_bytes(c_z: int, c: int, batch: int, n_res: int, *, with_stats: bool = True) -> int:
    """Compulsory HBM traffic of K3: read ``y`` and ``z`` once, write the update once (bf16), plus the statistics write."""
    pairs = int(batch) * int(n_res) ** 2
    total = pairs * (int(c) * _BF16_BYTES + 2 * int(c_z) * _BF16_BYTES)
    if with_stats:
        total += pairs * 8
    return total


def tail_flop(c_z: int, c: int, batch: int, n_res: int) -> int:
    """``2 * (B N^2) * (c c_z + c_z c_z)`` -- the value GEMM over ``y`` and the gate GEMM over ``z``."""
    return 2 * int(batch) * int(n_res) ** 2 * (int(c) * int(c_z) + int(c_z) * int(c_z))


__all__ = [
    "LnDualGemmParams",
    "LnDualGemmPlan",
    "MIN_N_RES_MN_MAJOR",
    "RowBlockGeometry",
    "build_trimul_tail",
    "fold_tail_affine",
    "n_pad",
    "row_block_geometry",
    "run_trimul_tail",
    "stack_tail_biases",
    "tail_bytes",
    "tail_flop",
    "tail_params",
    "y_shape",
]
