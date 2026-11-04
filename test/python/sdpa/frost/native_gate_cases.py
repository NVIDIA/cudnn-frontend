# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared fused-gate graph fixture, independent of architecture test collectors."""

import cudnn
import pytest
import torch


def _case(kind, gate, *, b=2):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("existing fused SDPA output gate requires SM107")
    if kind in ("fp8", "mx"):
        from test_sdpa_prepared_quantized_gate import _helper

        return _helper(kind == "mx")._case(d=256, arch="sm107", gate=gate, b=b)
    from frost_test_utils import select_engine

    torch.manual_seed(623)
    dtype = getattr(torch, kind)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if kind == "float16" else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    bufs, ts, vp = {}, {}, {}
    for name, heads in (("q", 4), ("k", 2), ("v", 2)):
        bufs[name] = (torch.randn(b, 128, heads, 256, device="cuda") * 0.4).to(dtype).transpose(1, 2)
        ts[name] = g.tensor_like(bufs[name])
        vp[ts[name]] = bufs[name]
    bufs["gate"], ts["gate"] = gate, g.tensor_like(gate)
    vp[ts["gate"]] = gate
    o, lse = g.sdpa(q=ts["q"], k=ts["k"], v=ts["v"], attn_scale=256**-0.5, generate_stats=True)
    shape, strides = [b, 4, 128, 256], [128 * 4 * 256, 256, 4 * 256, 1]
    o.set_dim(shape).set_stride(strides)
    o = g.mul(a=o, b=g.sigmoid(input=ts["gate"]))
    o.set_output(True).set_dim(shape).set_stride(strides)
    lse.set_output(True).set_dim([b, 4, 128, 1]).set_stride([4 * 128, 128, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    bufs["o"] = torch.empty((b, 128, 4, 256), device="cuda", dtype=dtype).transpose(1, 2)
    bufs["lse"] = torch.empty((b, 4, 128), device="cuda")
    ts.update(o=o, lse=lse)
    vp.update({o: bufs["o"], lse: bufs["lse"]})
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, "sdpa_fwd_prefill_sm107", pack_gqa=False)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, ws, bufs, ts
