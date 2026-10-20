# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer entry for the single-launch dense d128 backward kernel."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from .bprop_d128_f16 import host as tensor_host


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
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
):
    # The shared backward binder carries a workspace address even for a plan
    # requiring zero bytes. This unused scalar accepts a null address.
    q = cute.make_tensor(q_ptr, cute.make_layout(geometry[0][0], stride=geometry[0][1]))
    k = cute.make_tensor(k_ptr, cute.make_layout(geometry[1][0], stride=geometry[1][1]))
    v = cute.make_tensor(v_ptr, cute.make_layout(geometry[2][0], stride=geometry[2][1]))
    o = cute.make_tensor(o_ptr, cute.make_layout(geometry[3][0], stride=geometry[3][1]))
    do = cute.make_tensor(do_ptr, cute.make_layout(geometry[4][0], stride=geometry[4][1]))
    lse = cute.make_tensor(lse_ptr, cute.make_layout(geometry[5][0], stride=geometry[5][1]))
    dq = cute.make_tensor(dq_ptr, cute.make_layout(geometry[6][0], stride=geometry[6][1]))
    dk = cute.make_tensor(dk_ptr, cute.make_layout(geometry[7][0], stride=geometry[7][1]))
    dv = cute.make_tensor(dv_ptr, cute.make_layout(geometry[8][0], stride=geometry[8][1]))
    tensor_host(q, k, v, o, do, lse, dq, dk, dv, scale, causal, stream)


def compile_host(geometry, dtype, causal, sm):
    """Compile only plan-time geometry; bind storage and scale at execution."""
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(("sdpa_bwd_sm100_d128", geometry, str(dtype), bool(causal), sm)),
        symbol="frost_sdpa_bwd_sm100_d128_prepared",
    )
