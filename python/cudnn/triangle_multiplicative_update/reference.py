# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp64 reference of the triangle multiplicative update (AlphaFold 3 Algorithms 12 / 13), cuEquivariance's parameter convention.

For a pair representation ``x`` of shape ``[B, N, N, c_z]`` (``[N, N, c_z]`` is read as ``B = 1``), ``direction`` in
``("outgoing", "incoming")`` and an optional pair mask ``m`` of shape ``[B, N, N]`` (``None`` = all valid)::

    z_in       = LayerNorm_{c_z}(x; norm_in_weight, norm_in_bias, eps)                       over the channel axis
    a          = sigmoid(z_in . g_in_weight[:c]^T + g_in_bias[:c]) * (z_in . p_in_weight[:c]^T + p_in_bias[:c]) * m
    b          = sigmoid(z_in . g_in_weight[c:]^T + g_in_bias[c:]) * (z_in . p_in_weight[c:]^T + p_in_bias[c:]) * m
    y[.., i, j, :] = sum_k a[.., i, k, :] * b[.., j, k, :]        (outgoing)      /      sum_k a[.., k, i, :] * b[.., k, j, :]   (incoming)
    u          = LayerNorm_c(y; norm_out_weight, norm_out_bias, eps)
    out        = sigmoid(z_in . g_out_weight^T + g_out_bias) * (u . p_out_weight^T + p_out_bias)                  [B, N, N, c_z]

``p_in_weight`` / ``g_in_weight`` are ``[2 c, c_z]`` with rows ``[a; b]`` (cuEquivariance's packing), ``p_out_weight`` is ``[c_z, c]``,
``g_out_weight`` ``[c_z, c_z]``; every bias is optional (``None`` = zero).  The caller adds the residual; the output is not masked (the
mask acted on ``a`` and ``b``).

Two rounding models, selected by ``rounding``:

* ``"none"`` -- the pure fp64 formula above (the mathematical definition; cuEquivariance's reference differs from it only by its own
  bf16 roundings).
* ``"kernel"`` (default) -- the fp64 formula with the FROST kernels' bf16 rounding points inserted, so an oracle comparison measures the
  kernels' arithmetic and nothing else: ``z_in`` rounded once to bf16 (the MMA operand of K1 and of K3's gate GEMM), ``a`` and ``b``
  rounded once to bf16 AFTER the mask multiplies in full precision (K1's planes; a zero mask row is exactly ``+0.0``), ``y`` rounded once
  to bf16 (K2's output), ``u`` rounded once to bf16 (K3's value operand).  The final ``out`` is left in fp64 (the kernel rounds it once
  to bf16 -- the oracle's tolerance accounts for that half-ulp).  Mask semantics: the kernel multiplies the mask in fp32 BEFORE the bf16
  rounding of ``a`` / ``b``; cuEquivariance multiplies a bf16-rounded product -- identical for ``{0, 1}`` and power-of-two masks, one bf16
  ulp apart on fractional weights.

Everything runs in fp64 on the tensors' device; the inputs may be any real dtype (a bool mask is cast).  Memory: the fp64 ``a``, ``b`` and
``y`` are ``B N^2 c`` doubles each (604 MB at ``N = 768, c = 128``).
"""

from __future__ import annotations

from typing import Optional

ROUNDINGS = ("kernel", "none")


def _ensure_dims(t, n: int, name: str):
    if t.ndim == n - 1:
        return t.unsqueeze(0)
    if t.ndim != n:
        raise ValueError(f"{name} must have {n} (or {n - 1}, batchless) dimensions, got {tuple(t.shape)}")
    return t


def _bf16_round(t, enabled: bool):
    import torch

    return t.to(torch.bfloat16).to(torch.float64) if enabled else t


def _layer_norm_fp64(rows, weight, bias, eps):
    """Two-pass fp64 LayerNorm over the last axis with the biased variance (the kernels' statistics)."""
    mean = rows.mean(dim=-1, keepdim=True)
    var = ((rows - mean) ** 2).mean(dim=-1, keepdim=True)
    out = (rows - mean) / (var + eps).sqrt()
    if weight is not None:
        out = out * weight.to(rows.dtype)
    if bias is not None:
        out = out + bias.to(rows.dtype)
    return out


def triangle_multiplicative_update_reference(
    x,
    direction: str = "outgoing",
    mask=None,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    p_in_bias=None,
    g_in_weight=None,
    g_in_bias=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    p_out_bias=None,
    g_out_weight=None,
    g_out_bias=None,
    eps: float = 1e-5,
    *,
    rounding: str = "kernel",
):
    """The fp64 update ``[B, N, N, c_z]`` (see the module docstring; ``rounding`` = ``"kernel"`` or ``"none"``)."""
    import torch

    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}, got {rounding!r}")
    for name, w in (("norm_in_weight", norm_in_weight), ("norm_in_bias", norm_in_bias), ("p_in_weight", p_in_weight), ("g_in_weight", g_in_weight)):
        if w is None:
            raise ValueError(f"{name} is required by the reference (no implicit initialisation)")
    for name, w in (("norm_out_weight", norm_out_weight), ("norm_out_bias", norm_out_bias), ("p_out_weight", p_out_weight), ("g_out_weight", g_out_weight)):
        if w is None:
            raise ValueError(f"{name} is required by the reference (no implicit initialisation)")
    kr = rounding == "kernel"
    x4 = _ensure_dims(x, 4, "x").to(torch.float64)
    batch, n_res, n2, c_z = x4.shape
    if n_res != n2:
        raise ValueError(f"x must be [B, N, N, c_z] (a square pair representation), got {tuple(x4.shape)}")
    if p_in_weight.ndim != 2 or p_in_weight.shape[0] % 2 or p_in_weight.shape[1] != c_z or p_in_weight.shape != g_in_weight.shape:
        raise ValueError(f"p_in_weight / g_in_weight must both be [2 c, c_z = {c_z}], got {tuple(p_in_weight.shape)} and {tuple(g_in_weight.shape)}")
    c = p_in_weight.shape[0] // 2
    if tuple(p_out_weight.shape) != (c_z, c) or tuple(g_out_weight.shape) != (c_z, c_z):
        raise ValueError(
            f"p_out_weight must be [c_z, c] = {(c_z, c)} and g_out_weight [c_z, c_z] = {(c_z, c_z)}, got {tuple(p_out_weight.shape)} / {tuple(g_out_weight.shape)}"
        )
    f64 = lambda t: None if t is None else t.to(torch.float64)  # noqa: E731
    rows = x4.reshape(-1, c_z)
    z_in = _bf16_round(_layer_norm_fp64(rows, f64(norm_in_weight), f64(norm_in_bias), eps), kr)
    if mask is not None:
        m = _ensure_dims(mask, 3, "mask")
        if tuple(m.shape) != (batch, n_res, n_res):
            raise ValueError(f"mask must be [B, N, N] = {(batch, n_res, n_res)}, got {tuple(m.shape)}")
        m = m.to(torch.float64).reshape(-1, 1)
    else:
        m = None
    wp, wg = f64(p_in_weight), f64(g_in_weight)
    bp, bg = f64(p_in_bias), f64(g_in_bias)
    planes = []
    for half in (slice(0, c), slice(c, 2 * c)):
        val = z_in @ wp[half].T
        gate = z_in @ wg[half].T
        if bp is not None:
            val = val + bp[half]
        if bg is not None:
            gate = gate + bg[half]
        o = torch.sigmoid(gate) * val
        if m is not None:
            o = o * m
            o = torch.where(m == 0, torch.zeros_like(o), o)  # a masked row is EXACTLY zero (a select, never x * 0)
        planes.append(_bf16_round(o, kr).reshape(batch, n_res, n_res, c))
    a, b = planes
    if direction == "outgoing":
        y = torch.einsum("bikc,bjkc->bijc", a, b)
    else:
        y = torch.einsum("bkic,bkjc->bijc", a, b)
    y = _bf16_round(y, kr).reshape(-1, c)
    u = _bf16_round(_layer_norm_fp64(y, f64(norm_out_weight), f64(norm_out_bias), eps), kr)
    val = u @ f64(p_out_weight).T
    gate = z_in @ f64(g_out_weight).T
    if p_out_bias is not None:
        val = val + f64(p_out_bias)
    if g_out_bias is not None:
        gate = gate + f64(g_out_bias)
    out = torch.sigmoid(gate) * val
    return out.reshape(batch, n_res, n_res, c_z)


__all__ = ["ROUNDINGS", "triangle_multiplicative_update_reference"]
