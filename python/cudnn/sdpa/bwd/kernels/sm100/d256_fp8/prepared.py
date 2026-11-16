# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared D256 FP8 delta, dK/dV and dQ chain; no execution allocations."""

from pathlib import Path
import threading
from typing import Optional
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION
from cudnn.sdpa.bwd.kernels.sm100.fp8_pair.helpers import delta_d192_fp8_host as delta_host
from cudnn.sdpa.bwd.kernels.sm100.prepared_d192_fp8_host import FP8_SCALARS

SPEC = globals().get("FROST_TEMPLATE_PARAMS")
_SCALAR_GEOMETRY = ((1,), (1,))
dkdv_host = dq_host = None
_BIND_LOCK = threading.Lock()


def _bind():
    # load_template holds its own import lock: bind only after this module has
    # finished loading. The kernel pair is immutable for this specialization.
    global dkdv_host, dq_host
    with _BIND_LOCK:
        if dkdv_host is None:
            body = load_template(str(Path(__file__).with_name("kernel.py")), SPEC, tag="sdpa_bwd_sm100_d256_fp8")
            dkdv_host, dq_host = body.dkdv_host, body.dq_host


def _specialized(spec):
    return load_template(__file__, spec, tag="sdpa_bwd_sm100_d256_fp8_prepared")


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
    ratio: cutlass.Constexpr[int],
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
    delta = _scratch(payload, regions[0])
    delta_host(o, do, delta, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp, stream)
    dkdv_host(
        q,
        k,
        v,
        do,
        lse,
        delta,
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
        ratio,
    )
    dq_host(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        descale_q,
        descale_k,
        descale_v,
        descale_s,
        scale_s,
        descale_do,
        descale_dp,
        scale_dq,
        scale_dp,
        amax_dq,
        None,
        scale,
        causal,
        stream,
        ratio,
    )


def compile_host(geometry, regions, grad_dtype, causal, sm, amax_requested, dqk=256, dv=256, ratio=1, window=0, *, input_dtype=cutlass.Float8E4M3FN):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    spec = (int(window), "e5m2" if input_dtype == cutlass.Float8E5M2 else "e4m3")
    if spec != SPEC:
        return _specialized(spec).compile_host(geometry, regions, grad_dtype, causal, sm, amax_requested, dqk, dv, ratio, window, input_dtype=input_dtype)
    _bind()
    head_dims = (dqk, dv)
    types = [input_dtype] * 5 + [cutlass.Float32] + [grad_dtype] * 3
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
        int(ratio),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(
            (
                "sdpa_bwd_sm100_d256_fp8",
                ABI_VERSION,
                geometry,
                regions,
                str(grad_dtype),
                str(input_dtype),
                bool(causal),
                sm,
                tuple(amax_requested),
                head_dims,
                int(ratio),
                int(window),
            )
        ),
        symbol="frost_sdpa_bwd_sm100_d256_fp8_prepared",
    )
