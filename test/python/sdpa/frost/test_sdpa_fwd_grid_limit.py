# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST SDPA forward past 65535 heads or batch (#1573).

The NATURAL scheduler launches heads and batch on gridDim.y/z, which CUDA caps
at 65535, so a pinned NATURAL plan is declined when it is built and the
recommended plans ride the flat LPT grid.
"""

import math

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine

pytestmark = pytest.mark.L1


def _frost_shape():
    """(S, D) a FROST forward row serves on this device; the SM90 row serves d512 only."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (9, 0):
        return 64, 512
    if (major, minor) == (8, 0) or (major == 10 and minor <= 6):
        return 128, 64
    return None


if _frost_shape() is None:
    pytest.skip("needs an A100, H100 or SM100 device for the FROST forward rows", allow_module_level=True)
S, D = _frost_shape()
SCALE = 1.0 / math.sqrt(D)


def _graph(b, h):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    stride = [S * h * D, D, h * D, 1]
    q, k, v = (g.tensor(name=n, dim=[b, h, S, D], stride=stride, data_type=cudnn.data_type.HALF) for n in "qkv")
    o, _ = g.sdpa(q, k, v, is_inference=True, attn_scale=SCALE)
    o.set_output(True).set_data_type(cudnn.data_type.HALF).set_dim([b, h, S, D]).set_stride(stride)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    return g, (q, k, v, o)


@pytest.mark.parametrize("b,h", [(65536, 1), (1, 65536)], ids=["batch", "heads"])
def test_frost_forward_past_65535_runs_on_the_flat_grid(monkeypatch, b, h):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    g, ports = _graph(b, h)
    frost = [i for i, plan in enumerate(g.plans) if is_python_engine(plan.engine_id)]
    if not frost:
        pytest.skip("no FROST forward row claims this graph here")
    engine_id, knobs = g.get_engine_and_knobs_at_index(frost[0])
    assert knobs[cudnn.knob_type.SCHED_POLICY] != 0

    natural = {**knobs, cudnn.knob_type.SCHED_POLICY: 0}
    g.create_execution_plan(engine_id, natural)
    g.select_plan(g.get_execution_plan_count() - 1)
    with pytest.raises(Exception, match="65535"):
        g.check_support()
        g.build_plans()

    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, S, h, D, dtype=torch.float16, device="cuda").transpose(1, 2) for _ in range(3))
    o = torch.full_like(q, float("nan"))
    workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    g.execute(dict(zip(ports, (q, k, v, o))), workspace)
    torch.cuda.synchronize()
    for bi, hi in ((0, 0), (b - 1, h - 1), (b // 2, h // 2)):
        p = torch.softmax((q[bi, hi].double() @ k[bi, hi].double().T) * SCALE, dim=-1)
        torch.testing.assert_close(o[bi, hi].float(), (p @ v[bi, hi].double()).float(), atol=2e-3, rtol=1e-2)
