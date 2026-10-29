# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pointer entry for dense d128 delta, accumulation and conversion."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from pathlib import Path

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_f16 import host as accumulation_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_helpers import cast_dq_host, delta_zero_host

HEAD_DIM = globals().get("FROST_TEMPLATE_PARAMS", 128)


@cute.jit
def _view(ptr: cute.Pointer, geometry: cutlass.Constexpr):
    return cute.make_tensor(ptr, cute.make_layout(geometry[0], stride=geometry[1]))


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
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
    window: cutlass.Constexpr[int] = 0,
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
    dq_accum = _scratch(payload, regions[0])
    delta = _scratch(payload, regions[1])
    delta_zero_host(o, do, delta, dq_accum, stream)
    accumulation_host(q, k, v, do, lse, delta, dq_accum, dk, dv, scale, causal, stream, window)
    cast_dq_host(dq_accum, dq, q.element_type, stream)


def compile_host(geometry, regions, dtype, causal, sm, window=0):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    global accumulation_host
    head_dim = geometry[0][0][3]
    if head_dim == 64 and HEAD_DIM != 64:
        module = load_template(__file__, 64, tag="sdpa_bwd_d64_accum_host")
        return module.compile_host(geometry, regions, dtype, causal, sm, window)
    if head_dim != HEAD_DIM:
        raise ValueError("prepared accumulation head dimension does not match the kernel specialization")
    if HEAD_DIM == 64:
        accumulation_host = load_template(str(Path(__file__).with_name("bprop_d128_accum_f16.py")), 64, tag="sdpa_bwd_d64_accum").host
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
        int(window),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(("sdpa_bwd_sm100_d128_accumulation", ABI_VERSION, geometry, regions, str(dtype), bool(causal), sm, int(window), HEAD_DIM)),
        symbol="frost_sdpa_bwd_sm100_d128_accumulation_prepared",
    )
