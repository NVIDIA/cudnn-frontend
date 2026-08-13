# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 3 regression: a FROST SDPA ``execute()`` reads no device memory to the host.

A D2H read makes execute synchronous. The user-visible consequences are that the
path cannot be CUDA-graph captured (covered by test_sdpa_stream_respect.py) and
that its cost becomes the depth of the queue rather than its own work. This test
catches the cause instead of the symptom, so it needs no capture support and
names the offending accessor: every torch D2H accessor is made to raise for the
duration of one execute.

Scoped to the dense f16 rows, which are clean. The THD and per-tensor FP8 rows
still read device memory back; each needs a kernel-side change and is listed
under Rule 3 in python/cudnn/AGENTS.md. Widen the parametrization as they land.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine
from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0]

_B, _H, _S = 2, 8, 256
_HALF, _F32 = cudnn.data_type.HALF, cudnn.data_type.FLOAT

# Every torch accessor that copies device memory back to the host.
_D2H_ACCESSORS = ("item", "tolist", "cpu", "numpy", "__float__", "__int__", "__bool__")


def _build(d):
    dims = (_B, _H, _S, d)
    strides = (_S * _H * d, d, _H * d, 1)
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    q = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(strides)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    python = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not python:
        pytest.skip(f"no FROST SDPA engine claimed the graph for d={d}")
    g.select_plan(python[0])
    g.check_support()
    g.build_plans()
    mk = lambda: torch.randn(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)  # noqa: E731
    vp = {q: mk(), k: mk(), v: mk(), o: torch.empty(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)}
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8) if g.get_workspace_size() else None
    return g, vp, ws


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("d", [256, 512])
def test_execute_reads_no_device_memory_to_the_host(monkeypatch, d):
    g, vp, ws = _build(d)
    g.execute(vp, ws)  # compile/warm outside the ban: JIT may legitimately sync
    torch.cuda.synchronize()

    caught = []

    def _forbid(name):
        def guard(self, *a, **kw):
            if self.is_cuda:
                caught.append(name)
                raise AssertionError(f"execute() called Tensor.{name}() on a CUDA tensor — see Rule 3 in python/cudnn/AGENTS.md")
            return _originals[name](self, *a, **kw)

        return guard

    _originals = {n: getattr(torch.Tensor, n) for n in _D2H_ACCESSORS}
    for n in _D2H_ACCESSORS:
        monkeypatch.setattr(torch.Tensor, n, _forbid(n))

    g.execute(vp, ws)
    torch.cuda.synchronize()
    assert not caught
