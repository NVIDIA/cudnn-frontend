# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed delta -> dK/dV -> optional ordered folds -> dQ launch chain."""

import threading
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION as DENSE_ABI_VERSION
from cudnn.sdpa.bwd.kernels.sm100._d128_chain import delta_only_host, fold_dkv_host
from cudnn.sdpa.bwd.kernels.sm100._prepared_d128_common import _view, _scratch_fp32_region

_TEMPLATE = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_TEMPLATE) if isinstance(_TEMPLATE, int) else int(_TEMPLATE[0])

PAIR_RATIO = 1 if isinstance(_TEMPLATE, int) or len(_TEMPLATE) < 2 else int(_TEMPLATE[1])
PAIR_SPLIT = 1 if isinstance(_TEMPLATE, int) or len(_TEMPLATE) < 3 else int(_TEMPLATE[2])
_PAIR_BIND_LOCK = threading.Lock()
pair_dkdv_host = pair_dq_host = None


def _bind_pair():
    """Bind the kernel pair of this head dimension, group ratio and split on the first compile; it never changes afterwards."""
    global pair_dkdv_host, pair_dq_host
    with _PAIR_BIND_LOCK:
        if pair_dkdv_host is None:
            kernels = Path(__file__).with_name("kernel_dkv.py")
            pair_dkdv_host = load_template(
                str(kernels), (HEAD_DIM, PAIR_RATIO, PAIR_SPLIT), tag=f"sdpa_bwd_d{HEAD_DIM}_g{PAIR_RATIO}_s{PAIR_SPLIT}_dkdv"
            ).host_accum
            pair_dq_host = load_template(str(kernels.with_name("kernel_dq.py")), HEAD_DIM, tag=f"sdpa_bwd_d{HEAD_DIM}_dq").host


@cute.jit
def host_pair(
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
    delta = _scratch_fp32_region(payload, regions[0])
    delta_only_host(o, do, delta, stream)
    if cutlass.const_expr(PAIR_SPLIT > 1):
        dv_part = _scratch_fp32_region(payload, regions[1])
        dk_part = _scratch_fp32_region(payload, regions[2])
        pair_dkdv_host(q, k, v, do, lse, delta, None, dk_part, dv_part, scale, causal, stream, window)
        fold_dkv_host(dv_part, dv, PAIR_SPLIT, stream)
        fold_dkv_host(dk_part, dk, PAIR_SPLIT, stream)
    else:
        pair_dkdv_host(q, k, v, do, lse, delta, None, dk, dv, scale, causal, stream, window)
    pair_dq_host(q, k, v, do, lse, delta, dq, scale, causal, stream, window, PAIR_RATIO)


def compile_host_pair(geometry, regions, dtype, causal, sm, window=0, ratio=1, split=1):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    head_dim = geometry[0][0][3]
    if (head_dim, int(ratio), int(split)) != (HEAD_DIM, PAIR_RATIO, PAIR_SPLIT):
        module = load_template(__file__, (head_dim, int(ratio), int(split)), tag=f"sdpa_bwd_d{head_dim}_g{int(ratio)}_s{int(split)}_pair_host")
        return module.compile_host_pair(geometry, regions, dtype, causal, sm, window, ratio, split)
    _bind_pair()
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host_pair,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        regions,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        int(window),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(
            ("sdpa_bwd_sm100_d128_pair", DENSE_ABI_VERSION, geometry, regions, str(dtype), bool(causal), sm, int(window), HEAD_DIM, PAIR_RATIO, PAIR_SPLIT)
        ),
        symbol="frost_sdpa_bwd_sm100_d128_pair_prepared",
    )
