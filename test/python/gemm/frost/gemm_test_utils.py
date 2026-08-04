# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared gates and helpers for the FROST GEMM test suite."""

from __future__ import annotations

import re

import cudnn
import pytest
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name

# --- GPU / arch gate -------------------------------------------------------


def _active_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()

# Every e2e test in this suite JITs sm100-family templates, valid only on
# 100 <= SM < 120 (see kernel_registry.PIPELINE_ARCH_RANGES) — gate on arch, not just
# GPU presence, so wrong-arch machines skip instead of failing in the JIT.
requires_sm100 = pytest.mark.skipif(
    _SM is None or not (100 <= _SM < 120),
    reason="needs a Blackwell-family GPU (100 <= SM < 120), have " + ("none" if _SM is None else f"sm_{_SM}"),
)


# --- plan / config resolution ----------------------------------------------


class Plan:
    """JIT-compiles a recorded graph with a forced tile config (bypassing the
    FROST engine's auto-select). Exposes chain / binding / block_scale /
    aux_names; callable with a variant pack."""

    def __init__(self, graph, config=None, cta_group=2, scheduler="clc", force_stg_epi=False):
        self.g = graph
        kw = dict(cta_group=cta_group, scheduler=scheduler, force_stg_epi=force_stg_epi)
        if config is not None:
            kw["config"] = config
        self._compiled = jit_from_cudnn_graph(graph, **kw)
        self.chain = self._compiled.chain
        self.binding = self._compiled.binding
        self.block_scale = self.chain.has_block_scale
        self.aux_names = [t.name for t in self.chain.aux_tensors]

    def __call__(self, variant_pack):
        return self._compiled(variant_pack)


LEGACY_RE = re.compile(r"^(CONFIG_sm\d+_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def resolve(legacy_name):
    """Legacy config-name (with _Nctamma/_static, kept as readable test IDs) ->
    (pure-geometry config, cta_group, scheduler)."""
    m = LEGACY_RE.match(legacy_name)
    assert m, legacy_name
    return by_name(m.group(1)), int(m.group(2)), "static" if m.group(3) else "clc"


def kw(legacy_name):
    """resolve() packaged as jit/Plan kwargs."""
    config, cta_group, scheduler = resolve(legacy_name)
    return dict(config=config, cta_group=cta_group, scheduler=scheduler)


# --- variant packs ----------------------------------------------------------


def vp(compiled, a, b, outs, *aux):
    """Variant-pack dict {cuDNN tensor: buffer}: A/B operands, outputs, then aux."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    d = {bd.a_operands[0]: a, bd.b_operands[0]: b}
    d.update({o: buf for o, buf in zip(bd.outputs, outs)})
    d.update({x: buf for x, buf in zip(bd.aux, aux)})
    return d


def vp_bs(compiled, a, b, outs, sfa, sfb, *aux, fto=None):
    """Block-scale variant-pack (A/B + SFA/SFB + outputs + aux); pass ``fto``
    for the MoE grouped variant's first_token_offset."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    d = {
        bd.a_operands[0]: a,
        bd.b_operands[0]: b,
        bd.sfa_operands[0]: sfa,
        bd.sfb_operands[0]: sfb,
    }
    if fto is not None:
        d[bd.first_token_offset] = fto
    d.update({o: buf for o, buf in zip(bd.outputs, outs)})
    d.update({x: buf for x, buf in zip(bd.aux, aux)})
    return d


def vp_mg(compiled, gemm_pairs, outs, *aux, fto=None):
    """Multi-GEMM variant-pack: dedup per-GEMM (a, b) pairs by identity into
    the binding's distinct A/B slots (first-appearance order); + outputs + aux.
    Pass ``fto`` for the MoE grouped variant's first_token_offset."""
    bd = compiled.binding
    a_seen, b_seen = [], []
    for ag, bg in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    d = {} if fto is None else {bd.first_token_offset: fto}
    d.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    d.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    d.update({o: buf for o, buf in zip(bd.outputs, outs)})
    d.update({x: buf for x, buf in zip(bd.aux, aux)})
    return d


# --- block-scale / quantization reference helpers ---------------------------

# FP4 E2M1 value table, indexed by the 4-bit code.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    """Reorder a (rows, cols) SF tensor into the F8_128x4 blocked layout."""
    rows, cols = x.shape
    nrb, ncb = ceil_div(rows, 128), ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def unpack_fp4(u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Expand byte-packed FP4 pairs to values via a 16-entry LUT."""
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def rand_e8m0(shape, dev):
    # E8M0 holds a power-of-2; small exponents around bias 127 keep the FP32 ref in range.
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def block_quant_ref(x, block_size, out_dtype, scale_dtype):
    """Torch reference for the block-quant epilogue: per-block amax scale
    (E8M0 scales round toward +inf) + quantized output."""
    blocks = x.view(1, x.shape[0], x.shape[1] // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(scale_f > 0, torch.pow(2.0, torch.ceil(torch.log2(safe))), 0.0)
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    q = q.to(out_dtype).view(1, x.shape[0], x.shape[1])
    return q, scale


def reduction_ref(x: torch.Tensor, mode, dims: tuple[int, ...]) -> torch.Tensor:
    if mode == cudnn.reduction_mode.AMAX:
        return x.abs().amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MAX:
        return x.amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MIN:
        return x.amin(dim=dims, keepdim=True)
    return x.sum(dim=dims, keepdim=True)


def reduction_dims(out_dims: tuple[int, int, int], full: tuple[int, int, int]):
    return tuple(i for i, (out_extent, full_extent) in enumerate(zip(out_dims, full)) if out_extent == 1 and full_extent != 1)


def assert_block_scale_reduction_close(actual, expected, mode):
    if mode == cudnn.reduction_mode.ADD:
        torch.testing.assert_close(actual, expected, atol=2.0, rtol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


# --- shared MoE fixtures -----------------------------------------------------

# 36 expert offsets (BxE > E stress pattern) shared by the MoE grouped tests.
FULL_EXPERT_REDUCE_OFFSETS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    127,
    255,
    383,
    483,
    515,
    643,
    718,
    924,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
]
