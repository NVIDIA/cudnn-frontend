# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pointer entry for dense d128 per-tensor FP8 delta, accumulation and conversion."""

from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_fp8 import host as accumulation_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_fp8_helpers import cast_dq_fp8_host, delta_zero_fp8_host

HEAD_DIM = globals().get("FROST_TEMPLATE_PARAMS", 128)

FP8_SCALARS = (
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_s",
    "scale_s",
    "descale_o",
    "descale_dO",
    "descale_dP",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
)
FP8_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")
_SCALAR_GEOMETRY = ((1,), (1,))


@cute.jit
def _view(ptr: cute.Pointer, geometry: cutlass.Constexpr):
    return cute.make_tensor(ptr, cute.make_layout(geometry[0], stride=geometry[1]))


@cute.jit
def _optional_view(ptr: Optional[cute.Pointer]):
    if cutlass.const_expr(ptr is None):
        return None
    return _view(ptr, _SCALAR_GEOMETRY)


@cute.jit
def _scratch(payload: cutlass.Int64, region: cutlass.Constexpr):
    offset, shape, strides = region
    pointer = cute.make_ptr(cutlass.Float32, payload + cutlass.Int64(offset), cute.AddressSpace.gmem, assumed_align=256)
    return _view(pointer, (shape, strides))


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    descale_q_ptr: cute.Pointer,
    descale_k_ptr: cute.Pointer,
    descale_v_ptr: cute.Pointer,
    descale_s_ptr: cute.Pointer,
    scale_s_ptr: cute.Pointer,
    descale_o_ptr: cute.Pointer,
    descale_do_ptr: cute.Pointer,
    descale_dp_ptr: cute.Pointer,
    scale_dq_ptr: cute.Pointer,
    scale_dk_ptr: cute.Pointer,
    scale_dv_ptr: cute.Pointer,
    scale_dp_ptr: cute.Pointer,
    amax_dq_ptr: Optional[cute.Pointer],
    amax_dk_ptr: Optional[cute.Pointer],
    amax_dv_ptr: Optional[cute.Pointer],
    amax_dp_ptr: Optional[cute.Pointer],
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
):
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    lse = _view(lse_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    descale_q, descale_k, descale_v = _view(descale_q_ptr, _SCALAR_GEOMETRY), _view(descale_k_ptr, _SCALAR_GEOMETRY), _view(descale_v_ptr, _SCALAR_GEOMETRY)
    descale_s, scale_s = _view(descale_s_ptr, _SCALAR_GEOMETRY), _view(scale_s_ptr, _SCALAR_GEOMETRY)
    descale_o, descale_do, descale_dp = _view(descale_o_ptr, _SCALAR_GEOMETRY), _view(descale_do_ptr, _SCALAR_GEOMETRY), _view(descale_dp_ptr, _SCALAR_GEOMETRY)
    scale_dq, scale_dk, scale_dv, scale_dp = (
        _view(scale_dq_ptr, _SCALAR_GEOMETRY),
        _view(scale_dk_ptr, _SCALAR_GEOMETRY),
        _view(scale_dv_ptr, _SCALAR_GEOMETRY),
        _view(scale_dp_ptr, _SCALAR_GEOMETRY),
    )
    amax_dq, amax_dk, amax_dv, amax_dp = _optional_view(amax_dq_ptr), _optional_view(amax_dk_ptr), _optional_view(amax_dv_ptr), _optional_view(amax_dp_ptr)
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    dq_accum = _scratch(payload, regions[0])
    delta = _scratch(payload, regions[1])
    delta_zero_fp8_host(o, do, delta, dq_accum, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp, stream)
    accumulation_host(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_accum,
        dk,
        dv,
        descale_q,
        descale_k,
        descale_v,
        descale_s,
        scale_s,
        descale_do,
        descale_dp,
        scale_dk,
        scale_dv,
        scale_dp,
        amax_dk,
        amax_dv,
        amax_dp,
        scale,
        causal,
        stream,
    )
    cast_dq_fp8_host(dq_accum, dq, descale_dp, descale_k, scale_dq, amax_dq, dq.element_type, stream)


def compile_host(geometry, regions, grad_dtype, causal, sm, amax_requested, window=0):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    global accumulation_host, cast_dq_fp8_host, delta_zero_fp8_host
    head_dim = geometry[0][0][3]
    if head_dim == 64 and HEAD_DIM != 64:
        module = load_template(__file__, 64, tag="sdpa_bwd_d64_fp8_host")
        return module.compile_host(geometry, regions, grad_dtype, causal, sm, amax_requested, window)
    if head_dim != HEAD_DIM:
        raise ValueError("prepared fp8 accumulation head dimension does not match the kernel specialization")
    if window and HEAD_DIM != 64:
        raise ValueError("the fp8 accumulation kernel serves a sliding window at d64 only")
    if HEAD_DIM == 64:
        here = Path(__file__)
        accumulation_host = load_template(str(here.with_name("bprop_d128_accum_fp8.py")), (64, int(window)), tag=f"sdpa_bwd_d64_fp8_accum_w{int(window)}").host
        helpers = load_template(str(here.with_name("bprop_d128_fp8_helpers.py")), 64, tag="sdpa_bwd_d64_fp8_helpers")
        cast_dq_fp8_host, delta_zero_fp8_host = helpers.cast_dq_fp8_host, helpers.delta_zero_fp8_host
    types = [cutlass.Float8E4M3FN] * 5 + [cutlass.Float32] + [grad_dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    args += [cute.runtime.make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4) for _ in FP8_SCALARS]
    args += [cute.runtime.make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4) if requested else None for requested in amax_requested]
    return compile_cached(
        host,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        regions,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(
            (
                "sdpa_bwd_sm100_d128_fp8_accumulation",
                ABI_VERSION,
                geometry,
                regions,
                str(grad_dtype),
                bool(causal),
                sm,
                tuple(amax_requested),
                HEAD_DIM,
                int(window),
            )
        ),
        symbol="frost_sdpa_bwd_sm100_d128_fp8_accumulation_prepared",
    )
