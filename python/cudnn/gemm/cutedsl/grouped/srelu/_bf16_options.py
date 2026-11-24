# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side activation options and the CuTe DSL gate of the BF16 grouped srelu/dsrelu kernels.

THIS MODULE IMPORTS NOTHING FROM ``cutlass`` (python/cudnn/AGENTS.md Rule 7). The BF16 kernel modules
(``moe_grouped_gemm_srelu.py``, ``dsrelu/moe_grouped_gemm_dsrelu.py``) import ``cudnn.frost.tile_dsl``, and
through it ``cutlass.experimental``, at module scope. The facades and wrappers validate the activation
options of both backends and route BF16 calls with the functions below, and import a BF16 kernel module
only behind :func:`bf16_kernel_requirement_error`, so the block-scaled and JAX paths never import it.
"""

from __future__ import annotations

import math
import struct
from typing import Optional, Tuple

__all__ = [
    "ACT_INPUTS",
    "DSL_FLOOR",
    "TANH_MODES",
    "bf16_kernel_requirement_error",
    "check_tanh_mode",
    "resolve_bf16_srelu_options",
]

# Where the activation input comes from: the FP32 accumulator ("acc") or bf16_rn of it ("saved").
ACT_INPUTS = ("acc", "saved")
# How tanh is evaluated when a clamp scale is set (see moe_grouped_gemm_srelu.py's module docstring).
TANH_MODES = ("accurate", "approx")
# The CuTe DSL the BF16 kernels are qualified on. It is above CUTEDSL_MIN_VERSION, the floor of the
# cudnn.frost.tile_dsl helpers they use, so the gate passes it as an explicit floor (python/cudnn/AGENTS.md Rule 7).
DSL_FLOOR = (4, 8, 0)


def _check_clamp_scale(clamp_scale: float) -> None:
    """Raise unless ``s`` is positive with ``fp32(s)`` and ``fp32(1/s)`` finite and nonzero.

    The epilogue bakes both as FP32 immediates (same condition as the dSReLU kernel's check).
    """
    s_value = float(clamp_scale)
    try:
        s_ok = math.isfinite(s_value) and s_value > 0.0
        s_ok = s_ok and struct.unpack("<f", struct.pack("<f", s_value))[0] > 0.0
        s_ok = s_ok and struct.unpack("<f", struct.pack("<f", 1.0 / s_value))[0] > 0.0
    except OverflowError:
        s_ok = False
    if not s_ok:
        raise ValueError(f"clamp_scale must be positive with s and 1/s finite and nonzero in FP32, " f"got {clamp_scale}")


def check_tanh_mode(tanh_mode) -> str:
    """Return ``tanh_mode`` when it is ``"approx"`` (the public default) or ``"accurate"``.

    Anything else, ``None`` included, raises ``ValueError``, on both backends and whether or not
    ``tanh_clamp_scale`` is set.
    """
    if not isinstance(tanh_mode, str) or tanh_mode not in TANH_MODES:
        raise ValueError(f"tanh_mode must be 'approx' or 'accurate', got {tanh_mode!r}")
    return tanh_mode


def resolve_bf16_srelu_options(tanh_clamp_scale: Optional[float], tanh_mode: str) -> Tuple[Optional[float], str]:
    """Validate the activation options of the BF16 kernels and return ``(clamp_scale, tanh_mode)`` as
    the kernels take them.

    With a clamp scale, ``"approx"`` selects MUFU ``tanh.approx.f32`` and ``"accurate"`` the Cephes
    FP32 tanh. The plain squared-ReLU (``tanh_clamp_scale=None``) evaluates no tanh: its mode is
    still validated, then normalized to ``"approx"`` so that both values build the same kernel.
    """
    if tanh_clamp_scale is not None:
        if isinstance(tanh_clamp_scale, bool):
            raise ValueError(f"tanh_clamp_scale must be None or a positive float, got {tanh_clamp_scale!r}")
        _check_clamp_scale(tanh_clamp_scale)
        tanh_clamp_scale = float(tanh_clamp_scale)
    tanh_mode = check_tanh_mode(tanh_mode)
    if tanh_clamp_scale is None:
        tanh_mode = "approx"
    return tanh_clamp_scale, tanh_mode


def bf16_kernel_requirement_error(what: str) -> Optional[str]:
    """Rule 7's gate of the BF16 kernels, run before their modules are imported: a message naming the
    installed CuTe DSL when it is below ``DSL_FLOOR`` or exports no ``cutlass.experimental.primitives``;
    ``None`` when the kernels can be imported."""
    from cudnn.frost.buffers import cutedsl_primitives_error, cutedsl_requirement_error

    return cutedsl_requirement_error(what, floor=DSL_FLOOR) or cutedsl_primitives_error(what)
