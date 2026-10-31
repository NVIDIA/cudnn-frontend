# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged D128 retains the architecture-specific route and row-max instruction."""

import inspect
import math

import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
def test_paged_rowmax_architecture_replay(dtype, packed):
    import cudnn
    from cudnn.engines.manifest import family_for
    from cudnn.frost.buffers import cutedsl_arch_requirement_error

    if not torch.cuda.is_available():
        pytest.skip("requires SM100, SM103 or SM107")
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100, SM103 or SM107")
    if cc == (10, 7) and (reason := cutedsl_arch_requirement_error(cc)) is not None:
        pytest.skip(reason)
    engine = "sdpa_fwd_prefill_sm107" if cc == (10, 7) else "sdpa_fwd_prefill_sm100"
    io_type = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    nq, nk, h, hk, d, page = 137, 1155, 8, 2, 128, 16
    npages = (nk + page - 1) // page
    torch.manual_seed(261026 + nq)
    q = torch.randn(nq, h, d, device="cuda", dtype=dtype)
    k = torch.randn(npages, page, hk, d, device="cuda", dtype=q.dtype).transpose(1, 2)
    v = torch.randn_like(k)
    o = torch.empty_like(q)
    lse = torch.empty(nq, h, device="cuda", dtype=torch.float32)
    table = torch.randperm(npages, device="cuda", dtype=torch.int32).view(1, 1, npages, 1)
    qlens = torch.tensor([nq], device="cuda", dtype=torch.int32).view(1, 1, 1, 1)
    klens = torch.tensor([nk], device="cuda", dtype=torch.int32).view(1, 1, 1, 1)
    qo = torch.tensor([0, nq * h * d], device="cuda", dtype=torch.int32).view(2, 1, 1, 1)
    so = torch.tensor([0, nq * h], device="cuda", dtype=torch.int32).view(2, 1, 1, 1)
    graph = cudnn.pygraph(io_data_type=io_type, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qn = graph.tensor(dim=(1, h, nq, d), stride=(nq * h * d, d, h * d, 1), data_type=io_type)
    kn, vn, tn, qln, kln, qon, son = [graph.tensor_like(t) for t in (k, v, table, qlens, klens, qo, so)]
    qn.set_ragged_offset(qon)
    on, sn = graph.sdpa(
        q=qn,
        k=kn,
        v=vn,
        generate_stats=True,
        attn_scale=1 / math.sqrt(d),
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        seq_len_q=qln,
        seq_len_kv=kln,
        paged_attention_k_table=tn,
        paged_attention_v_table=tn,
        paged_attention_max_seq_len_kv=nk,
        max_total_seq_len_q=nq,
    )
    on.set_output(True).set_dim((1, h, nq, d)).set_stride((nq * h * d, d, h * d, 1)).set_ragged_offset(qon)
    sn.set_output(True).set_dim((1, h, nq, 1)).set_stride((nq * h, 1, h, 1)).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(son)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(
        family_for(graph).offered_ids()[engine],
        {
            cudnn.knob_type.TILE_CGA_M: 2,
            cudnn.knob_type.SPLIT_KV: 1,
            cudnn.knob_type.PACK_GQA: int(packed),
            cudnn.knob_type.SCHED_POLICY: 1,
            cudnn.knob_type.TILE_M: 128,
            cudnn.knob_type.TILE_N: 128,
        },
    )
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    api = inspect.getclosurevars(graph._compiled_plans[graph._plan_index]._compiled.default_stream).nonlocals["api"]
    route = api._k_mod.__file__
    assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.THD_VARLEN
    if cc == (10, 7):
        assert route.endswith("sm107/prefill_d128_f16.py"), route
        native_source = inspect.getsource(api._k_mod)
        assert "tmem_load_max_reduction_tile(" in native_source
        assert "_tmem_load_max_reduction(" in native_source
    else:
        assert route.endswith("sm100/prefill_d128_f16.py"), route
        assert bool(api._k_mod.FUSED_LDTM_STAT) == (cc == (10, 3))
        assert api._k_mod._D128_PAGED_JOINT_ROW_MAX == (cc == (10, 0) and dtype == torch.bfloat16 and packed)
    pack = {qn: q, kn: k, vn: v, on: o, sn: lse, tn: table, qln: qlens, kln: klens, qon: qo, son: so}
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

    def run():
        graph.execute(pack, workspace)

    def check(nk_live):
        torch.cuda.synchronize()
        kk = k[table.flatten().long()].transpose(0, 1).reshape(hk, npages * page, d)[:, :nk_live].double().repeat_interleave(h // hk, 0)
        vv = v[table.flatten().long()].transpose(0, 1).reshape(hk, npages * page, d)[:, :nk_live].double().repeat_interleave(h // hk, 0)
        for lo in range(0, nq, 64):
            hi = min(nq, lo + 64)
            score = q[lo:hi].double().transpose(0, 1) @ kk.transpose(1, 2) / math.sqrt(d)
            score.masked_fill_(
                torch.arange(nk_live, device="cuda")[None, :] > torch.arange(nk_live - nq + lo, nk_live - nq + hi, device="cuda")[:, None], -torch.inf
            )
            ref = (score.softmax(-1) @ vv).transpose(0, 1)
            stats = score.logsumexp(-1).T
            torch.testing.assert_close(o[lo:hi].double(), ref, atol=0.004, rtol=0.004)
            torch.testing.assert_close(lse[lo:hi].double(), stats, atol=0.0006, rtol=0)

    run()
    check(nk)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            run()
        o.fill_(19)
        lse.fill_(19)
        workspace.fill_(255)
        capture.replay()
        check(nk)
        table.copy_(torch.roll(table, 1, 2))
        q.mul_(0.5)
        k.mul_(0.75)
        klens.fill_(nk - 7)
        o.fill_(19)
        lse.fill_(19)
        workspace.fill_(255)
        capture.replay()
        check(nk - 7)
    finally:
        capture.reset()
