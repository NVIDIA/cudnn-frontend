# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream-respect regression for the FROST SDPA fwd engines.

A FROST SDPA engine must run on the stream carried by the cuDNN handle passed to
``graph.execute()`` — and therefore be CUDA-graph-capturable. Asserted through the
NATIVE public graph API only (``graph.execute(vp, ws, handle)`` +
``cudnn.set_stream(handle, s)``), with CUDA-graph capture as the cheap,
deterministic, nsys-free detector (a stream-0 hardcode makes the capture empty ->
zeroed replay -> failure). Mirrors test/python/gemm/frost/test_stream_respect.py.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl, _dsl_installed

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

_B, _H, _S = 2, 8, 256
_HALF, _F32 = cudnn.data_type.HALF, cudnn.data_type.FLOAT


def _build_causal_sdpa(d):
    dims = (_B, _H, _S, d)
    strides = (_S * _H * d, d, _H * d, 1)
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    q = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(strides)
    return g, q, k, v, o


@pytest.mark.parametrize("d", [256, 512])
def test_frost_sdpa_respects_handle_stream_and_is_capturable(d):
    torch.manual_seed(0)
    q_gpu = torch.randn(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa(d)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    python = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not python:
        pytest.skip(f"no FROST SDPA engine claimed the graph for d={d}")
    g.select_plan(python[0])  # the stream contract under test is the engine's
    g.check_support()
    g.build_plans()
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8) if g.get_workspace_size() else None

    h = cudnn.create_handle()
    vp = {q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}

    def run():
        g.execute(vp, ws, handle=h)

    # Reference on the default stream.
    o_gpu.zero_()
    run()
    torch.cuda.synchronize()
    ref = o_gpu.clone()
    assert ref.abs().sum().item() > 0, "reference output is all-zero; the capture check cannot distinguish an empty replay"

    # (1) Bind a NON-default stream to the handle: result must match the default run.
    s = torch.cuda.Stream()
    cudnn.set_stream(handle=h, stream=s.cuda_stream)
    o_gpu.zero_()
    with torch.cuda.stream(s):
        run()
    s.synchronize()
    torch.testing.assert_close(o_gpu.float(), ref.float(), rtol=0, atol=0)

    # (2) CUDA-graph capture on that stream, then replay.
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    s.synchronize()
    cg = torch.cuda.CUDAGraph()
    o_gpu.zero_()
    with torch.cuda.graph(cg, stream=s):
        run()
    o_gpu.zero_()
    cg.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(o_gpu.float(), ref.float(), rtol=0, atol=0)
