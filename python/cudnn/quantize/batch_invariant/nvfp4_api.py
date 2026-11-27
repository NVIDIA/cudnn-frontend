# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/batch_invariant/quantization/nvfp4_api.py (26075a1331ff).

"""Torch-facing entry point of the batch-invariant NVFP4 quantizer.

:func:`nvfp4_quantize_forward` runs the per-tensor class ``nvfp4.quant`` (the global scale is an input) or, with
``per_token=True``, the per-token class ``nvfp4.quant.pertoken`` (the global scale is derived per row and returned). The
output contract is FlashInfer's: ``fp4`` a ``uint8`` tensor of shape ``(*input.shape[:-1], K / 2)`` (two e2m1 codes per
byte, element ``2j`` in the low nibble), ``sf`` the ``uint8`` e4m3 block scales of the chosen layout as a
``(rows, padded columns)`` view (rows padded to 128 / 8 for the swizzled layouts, columns to 4; no padding for the linear
layout) and, per token, the fp32 per-row scale. Every unsupported input raises; nothing falls back.

:func:`nvfp4_quantize_rows` is the generic entry the tests drive (``config=`` to force a member of the enumerated space,
``sm_count=`` to ask the dispatcher for another part's pick, ``out=`` for preallocated outputs).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .api import MAX_ROW_BYTES, _check_device, _device_support_pdl, _require_cutedsl, _rows, _unsupported, device_sm_count, layout_of
from .layout import padded_sf_cols, padded_sf_rows, sf_numel
from .nvfp4_schedule import BLOCK, QuantConfig, default_config, num_blocks
from .nvfp4_spec import PER_TENSOR, PER_TOKEN, Variant

CLASS_NVFP4 = "nvfp4.quant"
#: The compute capabilities the class is stated and tested for: SM100, SM103 and SM107.
NVFP4_QUANT_ARCHS = ((10, 0), (10, 3), (10, 7))
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


def _scale_tensor(scale, ref: torch.Tensor) -> torch.Tensor:
    """The global scale as a 1-element fp32 device tensor: a host float becomes a device fill (``torch.full``,
    CUDA-graph capturable); a tensor on another device is copied to the input's device, as the operators' own paths
    do (a copy is not capturable)."""
    if not isinstance(scale, torch.Tensor):
        return torch.full((1,), float(scale), dtype=torch.float32, device=ref.device)
    return scale.to(device=ref.device, dtype=torch.float32).reshape(-1)[:1].contiguous()


def nvfp4_quantize_rows(
    a: torch.Tensor,
    a_global_sf: Union[float, torch.Tensor],
    layout="128x4",
    *,
    per_token: bool = False,
    out: Optional[Tuple[torch.Tensor, ...]] = None,
    config: Optional[QuantConfig] = None,
    enable_pdl: bool = False,
    sm_count: Optional[int] = None,
):
    """Quantize ``a`` (bf16 / f16 / fp32, last dim ``K`` a multiple of 16) to NVFP4 (the fp32 input is the same class:
    the leaf is the fp32 value of the element after the exact widening, so the bf16 / f16 bits are unchanged).

    Returns ``(fp4, sf)``: ``fp4`` uint8 of shape ``(..., K / 2)`` and ``sf`` the flat uint8 e4m3 block scales in the
    layout ``layout`` names (``"128x4"``, ``"8x4"`` or ``"linear"``; padding bytes zero). With ``per_token=True`` a third
    tensor, the per-row fp32 scale, is returned and ``a_global_sf`` is the inverse base scale (TE: ``1 / (448 * 6)``);
    otherwise ``a_global_sf`` is the per-tensor global scale, an input (a host float or a 1-element fp32 device tensor).
    Bitwise identical for every ``config`` in :func:`nvfp4_schedule.all_valid_configs`, for every batch composition,
    every SM count and the three layouts."""
    from .nvfp4_kernel import get_compiled_kernel

    if a.dtype not in _DTYPE_STR:
        raise ValueError(f"input dtype {a.dtype} not supported (bf16 / f16 / fp32)")
    layout = layout_of(layout)
    K = a.shape[-1]
    num_blocks(K)
    x2 = _rows(a, K, "input")
    M = x2.shape[0]
    variant: Variant = PER_TOKEN if per_token else PER_TENSOR
    gs = _scale_tensor(a_global_sf, a)
    padded_M = padded_sf_rows(M, layout)
    nsf = sf_numel(M, K, BLOCK, layout)
    if out is None:
        fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=a.device)
        sf = torch.empty(nsf, dtype=torch.uint8, device=a.device)
    else:
        fp4, sf = out[0], out[1]
        fp4 = _rows(fp4, K // 2, "fp4 out")
        assert fp4.shape == (M, K // 2) and fp4.dtype == torch.uint8
        sf = sf.view(-1)
        assert sf.dtype == torch.uint8 and sf.is_contiguous() and sf.numel() >= nsf
        sf = sf[:nsf]
    pts = torch.empty(M, dtype=torch.float32, device=a.device) if per_token else gs
    elem_bytes = a.element_size()
    if config is None:
        config = default_config(K, M, elem_bytes, per_token, sm_count or device_sm_count(a.device), layout)
    else:
        config.validate(K, elem_bytes)
    if x2.data_ptr() % 16:
        # an under-aligned storage offset: the 16-byte vector loads need the base aligned (a copy is a placement)
        x2 = x2.clone(memory_format=torch.contiguous_format)
    assert x2.stride(0) % (16 // elem_bytes) == 0 or M == 1, "input row stride must be a multiple of 16 bytes"
    if fp4.data_ptr() % 16 or sf.data_ptr() % 16:
        raise ValueError("the fp4 and sf output buffers must be 16-byte aligned")
    assert fp4.stride(0) % 8 == 0 or M == 1, "fp4 row stride must be a multiple of 8 bytes"
    kernel = get_compiled_kernel(_DTYPE_STR[a.dtype], K, config, variant, bool(enable_pdl), layout, a.device.index)
    kernel(x2, fp4, sf, pts, gs, M, K, padded_M)
    fp4 = fp4.view(*a.shape[:-1], K // 2)
    if per_token:
        return fp4, sf, pts
    return fp4, sf


def nvfp4_quantize_forward(
    input: torch.Tensor,
    global_scale: Union[float, torch.Tensor],
    sf_layout="128x4",
    enable_pdl: Optional[bool] = None,
    per_token: bool = False,
    *,
    config: Optional[QuantConfig] = None,
    sm_count: Optional[int] = None,
):
    """Quantize ``input`` (bf16, f16 or fp32, shape ``(..., K)``, ``K`` a multiple of 16) to NVFP4.

    ``per_token=False`` is the class ``nvfp4.quant``: ``global_scale`` is the per-tensor global scale, a host float or a
    one-element fp32 tensor. It returns ``(fp4, sf)``. ``per_token=True`` is the class ``nvfp4.quant.pertoken``:
    ``global_scale`` is the inverse base scale (Transformer Engine: ``1 / (448 * 6)``), each row derives its own global
    scale from its amax, and the call returns ``(fp4, sf, per_token_scale)`` with ``per_token_scale`` an fp32 ``(M,)`` vector.

    ``fp4`` is a ``uint8`` tensor ``(*input.shape[:-1], K / 2)``; ``sf`` holds the ``uint8`` e4m3 block scales, one per 16
    elements, of ``sf_layout`` as a ``(rows, columns)`` view: ``"128x4"`` (rows padded to 128 and columns to 4), ``"8x4"``
    (rows to 8, columns to 4) or ``"linear"`` (``M x K / 16``). FlashInfer's ``SfLayout`` members are accepted as well.
    ``enable_pdl=None`` follows the device.

    A row's codes, scale bytes and per-token scale depend on the row alone: they are the same for any batch size, row
    position, tile configuration and SM count. ``config`` / ``sm_count`` force a configuration / another part's dispatcher
    pick (members of the enumerated space: the bits never depend on them). Runs on SM100, SM103 and SM107 on the current
    stream of ``input.device``. Raises for any input outside the class; it never falls back."""
    _require_cutedsl("NVFP4 quantization")
    cls = CLASS_NVFP4 + (".pertoken" if per_token else "")
    _check_device(input.device, cls, NVFP4_QUANT_ARCHS)
    if input.dtype not in _DTYPE_STR:
        raise _unsupported(cls, f"input dtype {input.dtype}", "bf16, f16 or fp32 inputs")
    if input.dim() < 1:
        raise _unsupported(cls, f"a {input.dim()}-D input", "inputs of shape (..., K)")
    K = int(input.shape[-1])
    if K <= 0 or K % BLOCK:
        raise _unsupported(cls, f"K = {K}", f"K a positive multiple of {BLOCK}")
    if K * input.element_size() > MAX_ROW_BYTES:
        raise _unsupported(
            cls,
            f"K = {K} for {input.dtype}",
            f"rows of at most {MAX_ROW_BYTES} bytes (K <= {MAX_ROW_BYTES // input.element_size()})",
        )
    if isinstance(global_scale, torch.Tensor):
        if global_scale.numel() != 1:
            raise _unsupported(cls, f"a global scale of {global_scale.numel()} elements", "one scalar global scale")
    else:
        try:
            float(global_scale)
        except (TypeError, ValueError):
            raise _unsupported(cls, f"a global scale of type {type(global_scale).__name__}", "a float or a 1-element tensor") from None
    try:
        layout = layout_of(sf_layout)
    except ValueError:
        raise _unsupported(cls, f"the scale-factor layout {sf_layout!r}", "'128x4', '8x4' or 'linear'") from None
    try:
        x2 = _rows(input, K, "input")
    except ValueError:
        x2 = input.reshape(-1, K)  # a copy: the input could not be viewed as rows (a placement, not numerics)
    if x2.data_ptr() % 16 or (x2.shape[0] > 1 and (x2.stride(0) * x2.element_size()) % 16):
        # An under-aligned storage offset or a row stride that is not a multiple of 16 bytes: the 16-byte vector loads
        # need both. The copy is a placement, not numerics.
        x2 = x2.clone(memory_format=torch.contiguous_format)
    if enable_pdl is None:
        enable_pdl = _device_support_pdl(input.device)
    lead = input.shape[:-1]
    pcols = padded_sf_cols(K, BLOCK, layout)
    if x2.shape[0] == 0:  # no rows: nothing to launch; the outputs keep their documented shapes
        fp4 = torch.empty((*lead, K // 2), dtype=torch.uint8, device=input.device)
        sf = torch.empty(sf_numel(0, K, BLOCK, layout), dtype=torch.uint8, device=input.device).view(-1, pcols)
        if per_token:
            return fp4, sf, torch.empty(0, dtype=torch.float32, device=input.device)
        return fp4, sf
    res = nvfp4_quantize_rows(x2, global_scale, layout, per_token=per_token, config=config, enable_pdl=bool(enable_pdl), sm_count=sm_count)
    fp4, sf = res[0].view(*lead, K // 2), res[1].view(-1, pcols)
    if per_token:
        return fp4, sf, res[2]
    return fp4, sf
