# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped d128 workspace: FP32 dQ/Delta and native per-Q-head dK/dV partials."""

from .d128_contract import WORKSPACE_ALIGNMENT, WORKSPACE_SLACK, workspace_layout as dense_mha_workspace_layout
from .d128_thd_contract import workspace_layout as thd_mha_workspace_layout, workspace_offsets as thd_mha_workspace_offsets

ABI_VERSION = 1
DENSE = "accumulation_grouped_dense"
THD = "accumulation_grouped_thd"
PARTIAL_ITEMSIZE = 2  # BF16 for BF16 IO, FP16 for FP16 IO; final group accumulation is FP32.
ACCUM_ITEMSIZE = 4  # the fp32 dK / dV accumulation workspace under dkv_accum = "fp32_reduce" (one buffer per KV head)
# The typed op-level field ``SdpaBwdDslSm100D128.dkv_accum`` (C2b, 2026-10-27): how the GQA group's dK / dV are accumulated on the THD chain.
# A ROUNDING CHANGE between the two forms, hence an op ATTRIBUTE the caller sets (never a knob): "bf16_partials" = one bf16 partial per Q
# head + the fp32 ``dkv_reduce`` fold launch (a FIXED summation order; the shipping chain, byte-identical when the field is unset);
# "fp32_reduce" = every Q-head owner reduce-adds its fp32 dK / dV tile (``cp.reduce.async.bulk.tensor`` add.f32) into ONE per-KV-head fp32
# workspace (hardware order, non-deterministic: `use_deterministic_algorithm` is declined on this route anyway), the fold launch gone, ONE bf16
# rounding at the output in ``cast_dq_thd``.  Served by the 1-CTA grouped THD body and the cta_group::2 pair body; declined BY NAME on the
# dense grouped chain and on the MHA routes.
DKV_ACCUM_FORMS = ("bf16_partials", "fp32_reduce")
DKV_ACCUM_DEFAULT = DKV_ACCUM_FORMS[0]


def resolve_dkv_accum(value=None):
    """``None`` -> ``DKV_ACCUM_DEFAULT``; a value outside ``DKV_ACCUM_FORMS`` is a ``ValueError`` BY NAME (a selector outside its domain declines,
    never a silent default)."""
    form = DKV_ACCUM_DEFAULT if value is None else value
    if type(form) is not str or form not in DKV_ACCUM_FORMS:
        raise ValueError(f"dkv_accum must be one of {DKV_ACCUM_FORMS}; got {form!r}")
    return form


def dkv_accum_fp32(value=None):
    """The chain's ``Constexpr[bool]`` for a (resolved) form: True under ``fp32_reduce``; declines exactly as :func:`resolve_dkv_accum`."""
    return resolve_dkv_accum(value) == "fp32_reduce"


def thd_fp32_accum_bytes(rounded_k, kv_heads):
    """Bytes of ONE fp32 dK / dV accumulation buffer ``[1, rounded_k, kv_heads, 128]`` (shared host / trace arithmetic; accepts Int64)."""
    return rounded_k * kv_heads * 128 * ACCUM_ITEMSIZE


def validate_fp32_accum_fits(heads, kv_heads):
    """The fp32 per-KV-head buffer is carved INSIDE the bf16 per-Q-head partial region of the same tensor, so the advertised workspace is
    unchanged: ``rounded_k x kv_heads x 128 x 4 <= rounded_k x heads x 128 x 2`` holds on every grouped ratio (``heads >= 2 x kv_heads``)."""
    if kv_heads * ACCUM_ITEMSIZE > heads * PARTIAL_ITEMSIZE:
        raise ValueError(f"dkv_accum 'fp32_reduce' needs Hq >= 2 x Hkv for the fp32 accumulator to fit the partial region; got Hq {heads}, Hkv {kv_heads}")


def validate_dimensions(batch, heads, kv_heads, sq, skv):
    """Shared declaration-only launch checks for the adapter and row matcher."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, kv_heads, sq, skv)):
        raise ValueError("grouped dimensions must be positive integers")
    if heads <= kv_heads or heads % kv_heads:
        raise ValueError("requires Hq > Hkv and Hq divisible by Hkv")
    if max(sq, skv) > 2**31 - 128:
        raise ValueError("sequence envelope exceeds safe Int32 tile rounding")
    if batch > 65535 or heads > 65535 or batch * heads * ((skv + 127) // 128) > 2**31 - 1:
        raise ValueError("launch envelope exceeds the grid or Int32 work-id range")


def validate_dense_bounds(batch, heads, kv_heads, sq, skv):
    validate_dimensions(batch, heads, kv_heads, sq, skv)
    if sq % 128 or skv % 128:
        raise ValueError("dense sequence lengths must be multiples of 128")
    # The main kernel indexes dense Stats and Delta with Int32 row offsets.
    if batch * heads * sq > 2**31 - 1:
        raise ValueError("dense Stats/Delta span exceeds the Int32 index range")
    # The shared fold computes bidx*256+tidx in Int32 before its widened
    # element-address math. Aligned Skv makes this vector count block-aligned.
    if batch * skv * kv_heads * 16 > 2**31 - 1:
        raise ValueError("dense KV reduction exceeds the Int32 vector index range")
    if max(sq, skv) * heads * 128 * 2 >= 2**40:
        raise ValueError("native BSHD byte stride exceeds the tensor-map stride range")
    if sq * heads * 128 * 4 >= 2**40:
        raise ValueError("FP32 dQ byte stride exceeds the tensor-map stride range")


def validate_thd_bounds(batch, heads, kv_heads, sq, skv, tq, tkv, tq_phys, tkv_phys):
    validate_dimensions(batch, heads, kv_heads, sq, skv)
    if any(type(n) is not int or n <= 0 for n in (tq, tkv, tq_phys, tkv_phys)):
        raise ValueError("THD declared packed capacities must be positive integers")
    if max(tq, tkv) > 2**31 - 1:
        raise ValueError("compact packed capacity exceeds the Int32 prefix ABI")
    if max(tq_phys, tkv_phys) > 2**32 - 1:
        raise ValueError("physical token capacity exceeds the tensor-map dimension range")
    if max(tq_phys * heads, tkv_phys * kv_heads) * 128 * 2 >= 2**40:
        raise ValueError("physical BSHD byte stride exceeds the tensor-map stride range")
    if ((tq + 127) // 128 * 128) * heads * 128 * 4 >= 2**40:
        raise ValueError("FP32 dQ byte stride exceeds the tensor-map stride range")
    if ((tkv + 127) // 128 * 128) * heads * 128 * 2 >= 2**40:
        raise ValueError("native KV partial byte stride exceeds the tensor-map stride range")


def _align(value):
    return (value + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT


def dense_workspace_layout(batch, heads, sq, skv):
    """Preserve the dense MHA prefix; append two native [B,Skv,Hq,128] regions."""
    if type(skv) is not int or skv <= 0:
        raise ValueError("grouped d128 KV workspace extent must be a positive integer")
    regions, required = dense_mha_workspace_layout(batch, heads, sq)
    dk = _align(required - WORKSPACE_SLACK)
    size = batch * skv * heads * 128 * PARTIAL_ITEMSIZE
    dv = dk + _align(size)
    total = dv + size + WORKSPACE_SLACK
    if total > 2**63 - 1:
        raise ValueError("grouped d128 workspace exceeds the signed pointer range")
    shape, strides = (batch, skv, heads, 128), (skv * heads * 128, heads * 128, 128, 1)
    return regions + ((dk, shape, strides), (dv, shape, strides)), total


def thd_workspace_offsets(batch, heads, tq, tkv):
    """Shared host/trace arithmetic; no value reads from device metadata."""
    rounded_q, delta, meta, origins, maps, prefix_end = thd_mha_workspace_offsets(batch, heads, tq)
    rounded_k = (tkv + 127) // 128 * 128
    dk = _align(prefix_end)
    size = rounded_k * heads * 128 * PARTIAL_ITEMSIZE
    dv = dk + _align(size)
    return rounded_q, rounded_k, delta, meta, origins, maps, dk, dv, dv + size


def thd_workspace_layout(batch, heads, tq, tkv):
    """Append native compact-KV partials; explicit output origins use no scratch."""
    if type(tkv) is not int or tkv <= 0:
        raise ValueError("grouped d128 packed KV capacity must be a positive integer")
    regions, _ = thd_mha_workspace_layout(batch, heads, tq)
    _, rounded_k, _, _, _, _, dk, dv, total = thd_workspace_offsets(batch, heads, tq, tkv)
    if total + WORKSPACE_SLACK > 2**63 - 1:
        raise ValueError("grouped d128 packed workspace exceeds the signed pointer range")
    shape, strides = (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1)
    return regions + ((dk, shape, strides), (dv, shape, strides)), total + WORKSPACE_SLACK
