# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 f16/bf16 forward: ``stats_use_log2`` Stats are the base-2 LSE on every template and through the split combine."""

import math

import pytest
import torch
import cudnn

from frost_test_utils import requires_blackwell_geforce, requires_dsl
import test_sdpa_prepared_fp8 as fp8

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


# (64, 64) / (128, 128): the general template; (256, 256): the D256 template; (320, 320) / (512, 512): the D512
# template (envelope-served and native).
@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (256, 256), (320, 320), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,skv", [(1, 512), (77, 384)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split"])
def test_sm120_stats_use_log2(d, dv, dtype, sq, skv, causal, split, cudnn_handle):
    torch.manual_seed(120 + d + sq)
    b, h, hk = 2, 4, 2
    q = torch.randn(b, sq, h, d, dtype=dtype, device="cuda").transpose(1, 2)
    k = torch.randn(b, skv, hk, d, dtype=dtype, device="cuda").transpose(1, 2)
    v = torch.randn(b, skv, hk, dv, dtype=dtype, device="cuda").transpose(1, 2)
    o = torch.full((b, sq, h, dv), 77, dtype=dtype, device="cuda").transpose(1, 2)
    stats = torch.full((b, h, sq, 1), 77.0, device="cuda")
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nq, nk, nv = (graph.tensor_like(t) for t in (q, k, v))
    out, st = graph.sdpa(q=nq, k=nk, v=nv, attn_scale=1 / math.sqrt(d), generate_stats=True, stats_use_log2=True, use_causal_mask=causal)
    out.set_output(True).set_dim(o.shape).set_stride(o.stride()).set_data_type(io)
    st.set_output(True).set_dim(stats.shape).set_stride(stats.stride()).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    # The first ranked SM120 plan of the requested kind: a split plan's base-2 LSE comes out of the split combine.
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    index = next(
        (i for i, n in enumerate(names) if n.startswith("sdpa_fwd_prefill_sm120[") and (getattr(graph.plans[i].knobs, "split_kv", 1) > 1) == split),
        None,
    )
    if index is None and not split:
        pytest.skip(f"the SM120 engine ranks only split plans for this shape: {names}")
    assert index is not None, f"no split SM120 plan in {names}"
    graph.select_plan(index)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    graph.execute({nq: q, nk: k, nv: v, out: o, st: stats}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    kk = k.double().repeat_interleave(h // hk, 1)
    vv = v.double().repeat_interleave(h // hk, 1)
    scores = q.double() @ kk.transpose(-1, -2) / math.sqrt(d)
    if causal:
        scores.masked_fill_(torch.arange(skv, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None], -float("inf"))
    lse = scores.logsumexp(-1)
    expected_o = (scores - lse[..., None]).exp() @ vv
    torch.testing.assert_close(stats[..., 0].double(), lse * math.log2(math.e), atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(o.double(), expected_o, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("sq,skv", [(1, 512), (77, 384)])
@pytest.mark.parametrize("split_kv", [1, 4])
def test_sm120_fp8_stats_use_log2(d, dv, sq, skv, split_kv):
    """Per-tensor FP8 on the SM120 FP8 templates (d <= 256 and the d512 flavor): base-2 Stats, unsplit and split."""
    g, vp, ws, bufs, _ = fp8._case(arch="sm120", d=d, dv=dv, sq=sq, skv=skv, split_kv=split_kv, sdpa_kwargs=dict(stats_use_log2=True))
    g.execute(vp, ws)
    torch.cuda.synchronize()
    fp8._check({k: v for k, v in bufs.items() if k != "lse"}, thd=False, sq=sq, skv=skv)
    _, stats, _ = fp8._reference(bufs, thd=False, sq=sq, skv=skv)
    torch.testing.assert_close(bufs["lse"], stats * math.log2(math.e), atol=2e-3, rtol=2e-4)
