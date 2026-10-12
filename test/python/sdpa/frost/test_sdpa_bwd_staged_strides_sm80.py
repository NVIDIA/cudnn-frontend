# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical wide addresses through the retained packed staging path."""

import pytest
import torch

from frost_test_utils import requires_dsl
from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _wide_buffer
from sdpa.frost.test_sdpa_bwd_thd_sm80 import _build_thd_bwd_graph, _check, _plan_graph, _thd_case

pytestmark = [
    requires_dsl,
    pytest.mark.L1,
    pytest.mark.gpu_exclusive,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
@pytest.mark.parametrize("product", [False, True])
def test_staged_packed_physical_stride(role, product, monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    lengths = (5,) if product else (2, 2)
    case = _thd_case(lengths, lengths, 2, 96, torch.bfloat16, hkv=1)
    graph, pack, outputs = _build_thd_bwd_graph(case)
    grads = [torch.full_like(x, float("nan")) for x in (case.q, case.k, case.v)]
    pack.update(zip(outputs, grads))
    ports = dict(graph._thd_test_ports, **dict(zip(("dq", "dk", "dv"), outputs)))
    ref = ports[role]
    wide, backing = _wide_buffer(pack[ref], product=product, axis=1)
    pack[ref] = wide
    ts, dim = wide.stride(1), wide.shape[-1]
    ref.set_stride([max(lengths) * ts, dim, ts, 1])
    ro_name = role + "_ro" if role not in ("dq", "dk", "dv") else ref.get_name() + "_ro"
    for ro in list(pack):
        if ro.get_name() == ro_name:
            cu = case.cu_k if role in ("k", "v", "dk", "dv") else case.cu_q
            pack[ro] = (torch.tensor(cu, dtype=torch.int64, device="cuda") * ts).reshape(len(lengths) + 1, 1, 1, 1)
    if role in ("dq", "dk", "dv"):
        grads[("dq", "dk", "dv").index(role)] = wide
    _plan_graph(graph)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xBD)
    graph.execute(pack, workspace)
    _check(case, *grads)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(pack, workspace)
        case.do.mul_(1.25)
        pack[ports["do"]].copy_(case.do)
        for grad in grads:
            grad.fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        _check(case, *grads)
    finally:
        capture.reset()
    assert backing.numel() >= wide.numel()
