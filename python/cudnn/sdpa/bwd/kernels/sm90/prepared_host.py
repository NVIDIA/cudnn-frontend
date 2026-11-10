# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""One pointer entry for the Hopper dense d64 / d128 backward pair: delta, the dK / dV kernel, the dQ kernel."""

import threading
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import delta_only_host

_TEMPLATE = globals().get("FROST_TEMPLATE_PARAMS", (128, 1))
HEAD_DIM, RATIO = int(_TEMPLATE[0]), int(_TEMPLATE[1])
_BIND_LOCK = threading.Lock()
dkdv_host = dq_host = None


def _bind():
    """Bind the kernel pair of this head dimension and head ratio on the first compile; it never changes afterwards."""
    global dkdv_host, dq_host
    with _BIND_LOCK:
        if dkdv_host is None:
            module = load_template(str(Path(__file__).with_name("bprop_f16.py")), (HEAD_DIM, RATIO), tag=f"sdpa_bwd_sm90_d{HEAD_DIM}_g{RATIO}")
            dkdv_host, dq_host = module.dkdv_host, module.dq_host


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
    delta = _scratch(payload, regions[0])
    delta_only_host(o, do, delta, stream)
    dkdv_host(q, k, v, do, lse, delta, dk, dv, scale, causal, stream)
    dq_host(q, k, v, do, lse, delta, dq, scale, causal, stream)


def compile_host(geometry, regions, dtype, causal, ratio=1):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    head_dim = geometry[0][0][3]
    if (head_dim, int(ratio)) != (HEAD_DIM, RATIO):
        module = load_template(__file__, (head_dim, int(ratio)), tag=f"sdpa_bwd_sm90_d{head_dim}_g{int(ratio)}_host")
        return module.compile_host(geometry, regions, dtype, causal, ratio)
    _bind()
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
        options="--enable-tvm-ffi --gpu-arch sm_90a",
        cache_key=repr(("sdpa_bwd_sm90", ABI_VERSION, geometry, regions, str(dtype), bool(causal), HEAD_DIM, RATIO)),
        symbol="frost_sdpa_bwd_sm90_prepared",
    )
