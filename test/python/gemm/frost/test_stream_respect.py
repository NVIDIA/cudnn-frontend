# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream-respect regression for FROST GEMM engines.

A FROST engine must run on the stream carried by the cuDNN handle passed to
``graph.execute()`` — and therefore be CUDA-graph-capturable. Both are asserted
through the NATIVE public graph API only (``graph.execute(variant_pack, workspace,
handle)`` + ``cudnn.set_stream(handle, s)``), never the internal compiled executor:
it was the direct-executor test path (default stream only) that let the original
stream-0 hardcode slip through unnoticed.

The capture check is the cheap, deterministic, nsys-free detector. If the kernel
launches on a hardcoded default stream instead of the handle's (capture) stream,
nothing lands in the captured graph, so replay is a no-op and the output keeps its
zeroed sentinel -> the assertion fires.
"""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import requires_sm100, to_blocked

pytestmark = [pytest.mark.L0, requires_sm100]

import cudnn

_BF, _H, _F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, cudnn.data_type.FLOAT
_FP8, _E2M1 = cudnn.data_type.FP8_E4M3, cudnn.data_type.FP4_E2M1
_REORD = cudnn.tensor_reordering.F8_128x4


def _build_dense_bf16(M, N, K):
    torch.manual_seed(0)
    a = torch.randint(-3, 4, (1, M, K), device="cuda").to(torch.bfloat16)
    b = torch.randint(-3, 4, (1, N, K), device="cuda").to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g = cudnn.pygraph(io_data_type=_BF, intermediate_data_type=_F32, compute_data_type=_F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(_BF)
    return g, {A: a, B: b, C: c}, c


def _build_block_scale_fp4(M, N, K):
    torch.manual_seed(0)
    bs = 16
    sf_k = K // bs
    a = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sfa = torch.randint(1, 4, (M, sf_k), device="cuda").to(torch.float8_e4m3fn)
    sfb = torch.randint(1, 4, (N, sf_k), device="cuda").to(torch.float8_e4m3fn)
    c = torch.zeros(1, M, N, dtype=torch.bfloat16, device="cuda")
    g = cudnn.pygraph(io_data_type=_H, intermediate_data_type=_F32, compute_data_type=_F32)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=_E2M1)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=_E2M1)
    SFA = g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=_FP8, reordering_type=_REORD)
    SFB = g.tensor(name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=_FP8, reordering_type=_REORD)
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[bs, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(_BF)
    vp = {A: a, B: b, SFA: to_blocked(sfa).view(1, M, sf_k), SFB: to_blocked(sfb).view(1, N, sf_k), C: c}
    return g, vp, c


def _build_frost_plan(g):
    """Pin the FROST entry of the ranked plan list and build it."""
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    if "frost_gemm" not in names:
        pytest.skip(f"no FROST engine claimed this graph (plans={names})")
    g.select_plan(names.index("frost_gemm"))
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == "frost_gemm"
    return "frost_gemm"


_BUILDERS = {
    "dense_bf16": lambda: _build_dense_bf16(256, 256, 256),
    "block_scale_fp4": lambda: _build_block_scale_fp4(256, 256, 512),
}


@pytest.mark.parametrize("pattern", list(_BUILDERS))
def test_frost_gemm_respects_handle_stream_and_is_capturable(pattern):
    g, vp, c = _BUILDERS[pattern]()
    _build_frost_plan(g)
    ws = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda") if g.get_workspace_size() else None

    h = cudnn.create_handle()

    # Reference: execute on the default stream through the public API.
    c.zero_()
    g.execute(vp, ws, handle=h)
    torch.cuda.synchronize()
    ref = c.clone()
    assert ref.abs().sum().item() > 0, "reference output is all-zero; the capture check cannot distinguish an empty replay"

    # (1) Bind a NON-default stream to the handle: the result must match the
    # default-stream run (a wrong stream would corrupt ordering / results).
    s = torch.cuda.Stream()
    cudnn.set_stream(h, s.cuda_stream)
    c.zero_()
    with torch.cuda.stream(s):
        g.execute(vp, ws, handle=h)
    s.synchronize()
    torch.testing.assert_close(c.float(), ref.float(), rtol=0, atol=0)

    # (2) CUDA-graph capture on that stream, then replay. A stream-0 hardcode
    # makes the capture EMPTY, so replay leaves c at its zeroed sentinel.
    with torch.cuda.stream(s):
        for _ in range(3):
            g.execute(vp, ws, handle=h)  # warm up / JIT-compile before capture
    s.synchronize()
    cg = torch.cuda.CUDAGraph()
    c.zero_()
    with torch.cuda.graph(cg, stream=s):
        g.execute(vp, ws, handle=h)
    c.zero_()
    cg.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(c.float(), ref.float(), rtol=0, atol=0)
