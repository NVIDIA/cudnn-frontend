# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pointer entry for dense per-tensor FP8 delta, dK / dV and dQ launches at D_QK in {128, 192}, D_V = 128."""

import pathlib

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template

from cudnn.sdpa.bwd.d128_contract import ABI_VERSION, WORKSPACE_SLACK
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import host as dkdv_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dq_fp8 import host as dq_host
from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_fp8_helpers import delta_d192_fp8_host as delta_host

HEAD_DIMS = globals().get("FROST_TEMPLATE_PARAMS", (192, 128))
EXP2_FMA_SPLIT_SMS = frozenset((100,))

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


def workspace_layout(batch, heads, sq):
    """Return the aligned-payload-relative delta region and the advertised allocation bytes."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq)):
        raise ValueError("dense d192 fp8 workspace dimensions must be positive integers")
    rounded = (sq + 127) // 128 * 128
    regions = ((0, (batch, heads, rounded), (heads * rounded, rounded, 1)),)
    total = 4 * batch * heads * rounded + WORKSPACE_SLACK
    if total > 2**63 - 1:
        raise ValueError("dense d192 fp8 workspace exceeds the signed pointer range")
    return regions, total


def sf_geometry(batch, heads, rows, d=128):
    """(shape, strides) of a rowwise block-scale tensor viewed as (batch, head, 128-row tile, 4 per 128-wide group, 128) bytes."""
    tiles, groups = (int(rows) + 127) // 128, (int(d) + 127) // 128
    return ((int(batch), int(heads), tiles, 4 * groups, 128), (int(heads) * tiles * 512 * groups, tiles * 512 * groups, 512 * groups, 128, 1))


def sf_geometry_t(batch, heads, rows, d):
    """(shape, strides) of a columnwise block-scale tensor viewed as (128-row d tile, head plane, 128-row tile, 4, 128) bytes."""
    tiles, dtiles, planes = (int(rows) + 127) // 128, (int(d) + 127) // 128, int(batch) * int(heads)
    return ((dtiles, planes, tiles, 4, 128), (planes * tiles * 512, tiles * 512, 512, 128, 1))


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


def compile_host(geometry, regions, grad_dtype, causal, sm, amax_requested, dqk=192, dv=128, ratio=1):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    global dkdv_host, dq_host
    head_dims = (int(dqk), int(dv))
    if head_dims != tuple(HEAD_DIMS):
        return load_template(__file__, head_dims, tag=f"sdpa_bwd_d{dqk}x{dv}_fp8_host").compile_host(
            geometry, regions, grad_dtype, causal, sm, amax_requested, dqk, dv, ratio
        )
    exp2_fma_split = int(sm) in EXP2_FMA_SPLIT_SMS
    if head_dims != (192, 128) or exp2_fma_split:
        here = pathlib.Path(__file__).resolve()
        params = (*head_dims, exp2_fma_split)
        tag = f"sdpa_bwd_d{dqk}x{dv}_fp8{'_fma' if exp2_fma_split else ''}"
        dkdv_host = load_template(str(here.with_name("bprop_d192_dkdv_fp8.py")), params, tag=f"{tag}_dkdv").host
        dq_host = load_template(str(here.with_name("bprop_d192_dq_fp8.py")), params, tag=f"{tag}_dq").host
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
        int(ratio),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(
            ("sdpa_bwd_sm100_d192_fp8", ABI_VERSION, geometry, regions, str(grad_dtype), bool(causal), sm, tuple(amax_requested), head_dims, int(ratio))
        ),
        symbol="frost_sdpa_bwd_sm100_d192_fp8_prepared",
    )


@cute.jit
def host_mx(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    qt_ptr: cute.Pointer,
    kt_ptr: cute.Pointer,
    dot_ptr: cute.Pointer,
    do16_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    sfqt_ptr: cute.Pointer,
    sfk_ptr: cute.Pointer,
    sfkt_ptr: cute.Pointer,
    sfv_ptr: cute.Pointer,
    sfdo_ptr: cute.Pointer,
    sfdot_ptr: cute.Pointer,
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    sf_geoms: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    ratio: cutlass.Constexpr[int],
    stream: driver.CUstream,
):
    """The block-scale (MXFP8) chain: delta from the half O / dO, then the dK/dV and dQ kernels over the fp8 payloads, their transposed copies and the seven scale tensors."""
    q, k, v = _view(q_ptr, geometry[0]), _view(k_ptr, geometry[1]), _view(v_ptr, geometry[2])
    o16, do = _view(o_ptr, geometry[3]), _view(do_ptr, geometry[4])
    lse = _view(lse_ptr, geometry[5])
    dq, dk, dv = _view(dq_ptr, geometry[6]), _view(dk_ptr, geometry[7]), _view(dv_ptr, geometry[8])
    qt, kt, dot = _view(qt_ptr, geometry[0]), _view(kt_ptr, geometry[1]), _view(dot_ptr, geometry[4])
    do16 = _view(do16_ptr, geometry[3])
    sf_q, sf_qt = _view(sfq_ptr, sf_geoms[0]), _view(sfqt_ptr, sf_geoms[1])
    sf_k, sf_kt = _view(sfk_ptr, sf_geoms[2]), _view(sfkt_ptr, sf_geoms[3])
    sf_v, sf_do, sf_dot = _view(sfv_ptr, sf_geoms[4]), _view(sfdo_ptr, sf_geoms[5]), _view(sfdot_ptr, sf_geoms[6])
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    delta = _scratch(payload, regions[0])
    dot_do_o_host(o16, do16, delta, None, None, 128, HEAD_DIMS[0], HEAD_DIMS[1], 64, False, False, stream)
    dkdv_host(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        scale,
        causal,
        stream,
        ratio,
        qt=qt,
        dot=dot,
        sf_q=sf_q,
        sf_qt=sf_qt,
        sf_k=sf_k,
        sf_v=sf_v,
        sf_do=sf_do,
        sf_dot=sf_dot,
    )
    dq_host(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        scale,
        causal,
        stream,
        ratio,
        kt=kt,
        sf_q=sf_q,
        sf_do=sf_do,
        sf_k=sf_k,
        sf_kt=sf_kt,
        sf_v=sf_v,
    )


def compile_host_mx(geometry, sf_geoms, regions, out_dtype, causal, sm, ratio=1, dqk=128, dv=128, window=0):
    """Compile the block-scale chain from declarations only; the kernel templates run in block-scale mode with a left sliding window of ``window`` keys (0 = none)."""
    global dkdv_host, dq_host
    head_dims = (int(dqk), int(dv))
    if head_dims != tuple(HEAD_DIMS):
        return load_template(__file__, head_dims, tag=f"sdpa_bwd_d{dqk}x{dv}_mxfp8_host").compile_host_mx(
            geometry, sf_geoms, regions, out_dtype, causal, sm, ratio, dqk, dv, window
        )
    here = pathlib.Path(__file__).resolve()
    params = (*head_dims, int(sm) in EXP2_FMA_SPLIT_SMS, True, int(window))
    tag = f"sdpa_bwd_d{dqk}x{dv}_mxfp8_w{int(window)}"
    dkdv_host = load_template(str(here.with_name("bprop_d192_dkdv_fp8.py")), params, tag=f"{tag}_dkdv").host
    dq_host = load_template(str(here.with_name("bprop_d192_dq_fp8.py")), params, tag=f"{tag}_dq").host
    fp8 = cutlass.Float8E4M3FN
    types = (fp8, fp8, fp8, out_dtype, fp8, cutlass.Float32, out_dtype, out_dtype, out_dtype, fp8, fp8, fp8, out_dtype) + (cutlass.Int8,) * 7
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host_mx,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        sf_geoms,
        regions,
        bool(causal),
        int(ratio),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(
            ("sdpa_bwd_sm100_d192_mxfp8", ABI_VERSION, geometry, sf_geoms, regions, str(out_dtype), bool(causal), sm, head_dims, int(ratio), int(window))
        ),
        symbol="frost_sdpa_bwd_sm100_d192_mxfp8_prepared",
    )
