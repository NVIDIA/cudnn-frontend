# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer entry for dense grouped d128 with native dK/dV partials."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from ...d128_grouped_contract import ABI_VERSION
from .bprop_d128_accum_grouped_helpers import host_dense


@cute.jit
def _view(ptr: cute.Pointer, geometry: cutlass.Constexpr):
    return cute.make_tensor(ptr, cute.make_layout(geometry[0], stride=geometry[1]))


@cute.jit
def _scratch(payload: cutlass.Int64, region: cutlass.Constexpr, dtype: cutlass.Constexpr):
    offset, shape, strides = region
    pointer = cute.make_ptr(dtype, payload + cutlass.Int64(offset), cute.AddressSpace.gmem, assumed_align=256)
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
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    dq_accum = _scratch(payload, regions[0], cutlass.Float32)
    delta = _scratch(payload, regions[1], cutlass.Float32)
    dk_part = _scratch(payload, regions[2], q.element_type)
    dv_part = _scratch(payload, regions[3], q.element_type)
    host_dense(q, k, v, o, do, lse, dq, dk, dv, dq_accum, delta, dk_part, dv_part, scale, causal, stream)


def compile_host(geometry, regions, dtype, causal, sm):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
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
        cache_key=repr(("sdpa_bwd_sm100_d128_grouped_dense", ABI_VERSION, geometry, regions, str(dtype), bool(causal), sm)),
        symbol="frost_sdpa_bwd_sm100_d128_grouped_dense_prepared",
    )
