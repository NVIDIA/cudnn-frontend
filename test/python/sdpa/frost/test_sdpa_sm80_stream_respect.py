# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream-respect regression for the SM80 FROST SDPA engines.

Mirrors test_sdpa_stream_respect.py for the SM80 rows: the engines must run
on the stream carried by the cuDNN handle passed to ``graph.execute()`` —
and therefore be CUDA-graph-capturable.  Asserted through the native public
graph API only (``graph.execute(vp, ws, handle)`` + ``cudnn.set_stream``),
with CUDA-graph capture as the cheap, deterministic detector.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
import cudnn.sdpa  # noqa: F401 — the SM80 capability tables live here
from frost_test_utils import select_engine


def _is_sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(torch.cuda.current_device()) == (8, 0)


def _dsl_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not (_is_sm80() and _dsl_available()), reason="needs an SM80 (A100) GPU with cutlass"),
]

_B, _H, _S, _D = 2, 8, 256, 128
_HALF, _F32 = cudnn.data_type.HALF, cudnn.data_type.FLOAT
_STRIDES = (_S * _H * _D, _D, _H * _D, 1)


def _mk_buf():
    return torch.randn(_B, _S, _H, _D, device="cuda", dtype=torch.float16).transpose(1, 2)


def _build_fwd():
    dims = (_B, _H, _S, _D)
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    q = g.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="q")
    k = g.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="k")
    v = g.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(_D), is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(_STRIDES)
    return g, {"q": q, "k": k, "v": v, "o": o}


def _build_and_pin(g, engine):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine)
    g.check_support()
    g.build_plans()


def _assert_stream_respect(run, out_bufs, handle):
    """Default-stream reference, non-default-stream agreement, then CUDA-graph
    capture + replay agreement (an ignored handle stream makes the capture
    empty -> zeroed replay -> mismatch)."""
    for b in out_bufs:
        b.zero_()
    run()
    torch.cuda.synchronize()
    refs = [b.clone() for b in out_bufs]
    assert all(r.abs().sum().item() > 0 for r in refs), "all-zero reference; capture check cannot distinguish an empty replay"

    s = torch.cuda.Stream()
    cudnn.set_stream(handle=handle, stream=s.cuda_stream)
    for b in out_bufs:
        b.zero_()
    with torch.cuda.stream(s):
        run()
    s.synchronize()
    for b, r in zip(out_bufs, refs):
        torch.testing.assert_close(b.float(), r.float(), rtol=0, atol=0)

    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    s.synchronize()
    cg = torch.cuda.CUDAGraph()
    for b in out_bufs:
        b.zero_()
    with torch.cuda.graph(cg, stream=s):
        run()
    for b in out_bufs:
        b.zero_()
    cg.replay()
    torch.cuda.synchronize()
    for b, r in zip(out_bufs, refs):
        torch.testing.assert_close(b.float(), r.float(), rtol=0, atol=0)


def test_sm80_fwd_respects_handle_stream_and_is_capturable():
    torch.manual_seed(0)
    g, t = _build_fwd()
    _build_and_pin(g, "sdpa_fwd_prefill_sm80")
    q_gpu, k_gpu, v_gpu = _mk_buf(), _mk_buf(), _mk_buf()
    o_gpu = torch.empty_like(q_gpu)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8) if g.get_workspace_size() else None
    h = cudnn.create_handle()
    vp = {t["q"]: q_gpu, t["k"]: k_gpu, t["v"]: v_gpu, t["o"]: o_gpu}

    _assert_stream_respect(lambda: g.execute(vp, ws, handle=h), [o_gpu], h)


def test_sm80_bwd_respects_handle_stream_and_is_capturable():
    torch.manual_seed(0)
    dims = (_B, _H, _S, _D)
    # Forward (engine) to produce O / stats.
    gf = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    q = gf.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="q")
    k = gf.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="k")
    v = gf.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="v")
    o, stats = gf.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(_D), generate_stats=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(_STRIDES)
    stats.set_output(True).set_data_type(_F32)
    _build_and_pin(gf, "sdpa_fwd_prefill_sm80")
    q_gpu, k_gpu, v_gpu = _mk_buf(), _mk_buf(), _mk_buf()
    o_gpu = torch.empty_like(q_gpu)
    stats_gpu = torch.empty(_B, _H, _S, 1, device="cuda", dtype=torch.float32)
    gf.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu, stats: stats_gpu}, None)
    torch.cuda.synchronize()

    gb = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    qb = gb.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="q")
    kb = gb.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="k")
    vb = gb.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="v")
    ob = gb.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="o")
    dob = gb.tensor(dim=dims, stride=_STRIDES, data_type=_HALF, name="dO")
    statsb = gb.tensor(dim=(_B, _H, _S, 1), stride=(_H * _S, _S, 1, 1), data_type=_F32, name="stats")
    dq, dk, dv = gb.sdpa_backward(q=qb, k=kb, v=vb, o=ob, dO=dob, stats=statsb, attn_scale=1.0 / math.sqrt(_D), use_causal_mask=True)
    for x in (dq, dk, dv):
        x.set_output(True).set_data_type(_HALF)
    _build_and_pin(gb, "sdpa_bwd_sm80")

    do_gpu = _mk_buf()
    dq_gpu, dk_gpu, dv_gpu = torch.empty_like(q_gpu), torch.empty_like(k_gpu), torch.empty_like(v_gpu)
    h = cudnn.create_handle()
    vp = {qb: q_gpu, kb: k_gpu, vb: v_gpu, ob: o_gpu, dob: do_gpu, statsb: stats_gpu, dq: dq_gpu, dk: dk_gpu, dv: dv_gpu}

    _assert_stream_respect(lambda: gb.execute(vp, None, handle=h), [dq_gpu, dk_gpu, dv_gpu], h)
