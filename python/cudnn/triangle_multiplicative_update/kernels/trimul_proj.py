# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K1 -- LayerNorm of the pair representation + the gated dual projections + the pair mask -> the channel-major planes ``a``, ``b``.

For every pair row ``(b, i, k)`` of ``z`` (``[B, N, N, c_z]`` bf16) and each pass ``p`` in (``a``, ``b``)::

    z_in      = LayerNorm_{c_z}(z[b, i, k, :]; norm_in_weight, norm_in_bias, eps)             fp32 statistics, bf16 operand
    pass a    a[b, i, k, :] = sigmoid(z_in W_ag^T + b_ag) * (z_in W_ap^T + b_ap) * mask[b, i, k]
    pass b    b[b, i, k, :] = sigmoid(z_in W_bg^T + b_bg) * (z_in W_bp^T + b_bp) * mask[b, i, k]

stored CHANNEL-major as ``planes[p][c][b * N + i][k]`` with the ``k`` axis padded to a multiple of 8 (``n_pad``) -- the contract the
contraction (K2, ``kernels.contraction``) consumes: per channel ``c`` and batch ``b`` the ``[N, N]`` matrix at rows ``b * N ..``
with row stride ``n_pad``.  The per-row ``(mean, rstd)`` of the LayerNorm are written once for the backward.

This module writes no kernel: it is the triangle multiplicative update's instance of the LayerNorm-prologue gated dual GEMM
substrate ``cudnn.gemm.frost.ln_prologue`` (one persistent launch whose tile list is (pass, row block); the mask is
applied in fp32 before the one bf16 rounding; a masked row is exactly ``+0.0``; a 256-row tile whose rows are all masked skips its
MMA by a warp vote).  The weight convention is cuEquivariance's: ``p_in_weight, g_in_weight`` are ``[2 c, c_z]`` with rows
``[a; b]`` -- :func:`stack_projection_weights` turns them into the substrate's ``[2, 2 c, c_z]`` interleaved stack ONCE (plan time).

Traffic model (:func:`proj_bytes`): read ``z`` (2 B per pair-channel), write ``a`` and ``b`` (2 + 2 B per pair-channel), plus the
mask (2 B per pair) and the statistics (8 B per pair); FLOPs (:func:`proj_flop`): ``2 * B N^2 * c_z * 4 c``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from cudnn.gemm.frost.ln_prologue import (
    K_PAD_MULTIPLE,
    LnDualGemmParams,
    LnDualGemmPlan,
    RowBlockGeometry,
    build_ln_dual_gemm,
    interleave_value_gate,
    interleave_value_gate_bias,
    row_block_geometry,
    run_ln_dual_gemm,
)

#: The two projections (``a``, ``b``) are the substrate's two passes over one LayerNormed activation.
PASSES = 2
_BF16_BYTES = 2


def proj_params(
    c_z: int,
    c: int,
    *,
    has_bias: bool,
    has_mask: bool,
    want_stats: bool = True,
    skip_masked_tiles: bool = True,
    eps: float = 1e-5,
    clc_stages: Optional[int] = None,
    diag: str = "",
    pass_order: Optional[str] = None,
    ln_pass_form: Optional[str] = None,
    epi_math: Optional[str] = None,
    min_blocks_per_mp: Optional[int] = None,
    ln_once: Optional[str] = None,
    ln_once_b_stages: Optional[int] = None,
) -> LnDualGemmParams:
    """The substrate's compile-time parameters for the ``(c_z, c)`` geometry (K = ``c_z``, ``c`` output channels per pass);
    ``clc_stages``, ``pass_order``, ``ln_pass_form``, ``epi_math`` and ``min_blocks_per_mp`` (performance only, bitwise the same planes; the
    last one is the launch's ``.minnctapersm`` hint under which ptxas honours the body's register split) default to the substrate's."""
    kw = {} if clc_stages is None else {"clc_stages": int(clc_stages)}
    if diag:
        kw["diag"] = str(diag)
    if pass_order is not None:
        kw["pass_order"] = str(pass_order)
    if ln_pass_form is not None:
        kw["ln_pass_form"] = str(ln_pass_form)
    if epi_math is not None:
        kw["epi_math"] = str(epi_math)
    if min_blocks_per_mp is not None:
        kw["min_blocks_per_mp"] = (
            min_blocks_per_mp if isinstance(min_blocks_per_mp, bool) else int(min_blocks_per_mp)
        )  # a bool reaches the typed decline as itself
    if ln_once is not None:
        kw["ln_once"] = str(ln_once)
    if ln_once_b_stages is not None:
        kw["ln_once_b_stages"] = (
            ln_once_b_stages if isinstance(ln_once_b_stages, bool) else int(ln_once_b_stages)
        )  # the LN-once item's B-ring depth (appended; a bool reaches the typed decline as itself)  # the LayerNorm-once-per-row-block lever (performance only, bitwise; the substrate's typed vocabulary)
    return LnDualGemmParams(
        k=int(c_z),
        c_out=int(c),
        passes=PASSES,
        has_bias=bool(has_bias),
        want_stats=bool(want_stats),
        has_mask=bool(has_mask),
        skip_masked_tiles=bool(skip_masked_tiles),
        eps=float(eps),
        **kw,
    )


def n_pad(n_res: int) -> int:
    """The padded ``k`` extent of the channel-major planes (a multiple of 8: a 16-byte-aligned TMA row stride)."""
    return -(-int(n_res) // K_PAD_MULTIPLE) * K_PAD_MULTIPLE


def planes_shape(c: int, batch: int, n_res: int) -> Tuple[int, int, int, int]:
    """``[2, c, B * N, n_pad]`` -- the ``a`` / ``b`` workspace planes K1 writes and K2 reads."""
    return (PASSES, int(c), int(batch) * int(n_res), n_pad(n_res))


def plane_strides(batch: int, n_res: int) -> Tuple[int, int, int]:
    """Per channel and batch, the ``[N, N]`` matrix ``plane[c][b * N + i][k]``: ``(channel, row i, col k)`` element strides
    ``(B * N * n_pad, n_pad, 1)`` -- what the contraction graph declares over these planes (the batch ``b`` is a row offset)."""
    npd = n_pad(n_res)
    return (int(batch) * int(n_res) * npd, npd, 1)


def stack_projection_weights(p_in_weight, g_in_weight):
    """cuEquivariance's ``p_in_weight, g_in_weight`` (``[2 c, c_z]``, rows ``[a; b]``) -> the substrate's ``[2, 2 c, c_z]`` stack:
    pass 0 = (W_ap, W_ag), pass 1 = (W_bp, W_bg), each interleaved per 128-channel block (``interleave_value_gate``).  A copy; plan time."""
    import torch

    if p_in_weight.shape != g_in_weight.shape or p_in_weight.ndim != 2 or p_in_weight.shape[0] % 2:
        raise ValueError(f"p_in_weight / g_in_weight must both be [2 c, c_z], got {tuple(p_in_weight.shape)} and {tuple(g_in_weight.shape)}")
    c = p_in_weight.shape[0] // 2
    w_a = interleave_value_gate(p_in_weight[:c], g_in_weight[:c])
    w_b = interleave_value_gate(p_in_weight[c:], g_in_weight[c:])
    return torch.stack([w_a, w_b], dim=0).contiguous()


def stack_projection_biases(p_in_bias, g_in_bias):
    """The ``[2 c]`` biases (rows ``[a; b]``) in the ``[2, 2 c]`` interleaved order of :func:`stack_projection_weights`."""
    import torch

    if p_in_bias.shape != g_in_bias.shape or p_in_bias.ndim != 1 or p_in_bias.shape[0] % 2:
        raise ValueError(f"p_in_bias / g_in_bias must both be [2 c], got {tuple(p_in_bias.shape)} and {tuple(g_in_bias.shape)}")
    c = p_in_bias.shape[0] // 2
    return torch.stack([interleave_value_gate_bias(p_in_bias[:c], g_in_bias[:c]), interleave_value_gate_bias(p_in_bias[c:], g_in_bias[c:])], dim=0).contiguous()


def build_trimul_proj(params: LnDualGemmParams, arch: str = "sm107") -> LnDualGemmPlan:
    """Compile (cached per (params, arch)) -- shapes are runtime.  ``arch`` names the substrate's rendered body (``"sm107"`` Rubin, ``"sm100"``
    Blackwell; ``cudnn.gemm.frost.ln_prologue.body_arch_for_cc`` picks it from the device's compute capability)."""
    if params.passes != PASSES:
        raise ValueError(f"the triangle multiplicative update's K1 has exactly {PASSES} passes (a, b), got params.passes={params.passes}")
    return build_ln_dual_gemm(params, arch)


def run_trimul_proj(
    plan: LnDualGemmPlan,
    z,  # [B, N, N, c_z] bf16 contiguous
    w_stacked,  # [2, 2 c, c_z] bf16 from stack_projection_weights
    planes,  # [2, c, B * N, n_pad] bf16 contiguous (planes_shape)
    norm_in_weight,  # [c_z] bf16
    norm_in_bias,  # [c_z] bf16
    *,
    bias_stacked=None,  # [2, 2 c] bf16 from stack_projection_biases, iff params.has_bias
    mask=None,  # [B, N, N] bf16 pair mask, iff params.has_mask
    stats=None,  # [B * N * N, 2] fp32 (mean, rstd), iff params.want_stats
    stream,
    k_block=None,  # k's per 256-row pair tile (the substrate's rule when None; a performance knob, bitwise the same planes)
) -> RowBlockGeometry:
    """Launch K1.  No allocation, no conversion (the mask must already be bf16 ``[B, N, N]``; the planes are the caller's workspace)."""
    if z.ndim != 4 or z.shape[1] != z.shape[2]:
        raise ValueError(f"z must be [B, N, N, c_z], got {tuple(z.shape)}")
    batch, n_res, _, c_z = (int(s) for s in z.shape)
    if c_z != plan.params.k:
        raise ValueError(f"z has c_z={c_z} but the plan was compiled for K={plan.params.k}")
    if mask is not None and tuple(mask.shape) != (batch, n_res, n_res):
        raise ValueError(f"mask must be [B, N, N] = {(batch, n_res, n_res)}, got {tuple(mask.shape)}")
    return run_ln_dual_gemm(
        plan,
        z.view(batch * n_res * n_res, c_z),
        w_stacked,
        planes,
        norm_in_weight,
        norm_in_bias,
        n_res=n_res,
        bn=batch * n_res,
        bias=bias_stacked,
        mask=mask.reshape(-1) if mask is not None else None,
        stats=stats,
        stream=stream,
        k_block=k_block,
    )


def proj_bytes(c_z: int, c: int, batch: int, n_res: int, *, with_mask: bool = True, with_stats: bool = True) -> int:
    """Compulsory HBM traffic of K1: read ``z`` once, write ``a`` and ``b`` once (bf16), plus the mask read and the statistics write."""
    pairs = int(batch) * int(n_res) ** 2
    total = pairs * (int(c_z) * _BF16_BYTES + 2 * int(c) * _BF16_BYTES)
    if with_mask:
        total += pairs * _BF16_BYTES
    if with_stats:
        total += pairs * 8
    return total


def proj_flop(c_z: int, c: int, batch: int, n_res: int) -> int:
    """``2 * (B N^2) * c_z * (4 c)`` -- two passes of a ``[rows, c_z] x [c_z, 2 c]`` GEMM."""
    return 2 * int(batch) * int(n_res) ** 2 * int(c_z) * 4 * int(c)


__all__ = [
    "PASSES",
    "LnDualGemmParams",
    "LnDualGemmPlan",
    "RowBlockGeometry",
    "build_trimul_proj",
    "n_pad",
    "plane_strides",
    "planes_shape",
    "proj_bytes",
    "proj_flop",
    "proj_params",
    "row_block_geometry",
    "run_trimul_proj",
    "stack_projection_biases",
    "stack_projection_weights",
]
