# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 paged loads preserve independent pools/tables under rebinding and replay."""

import math

import pytest
import torch
import cudnn

from cudnn.api_base import TensorDesc
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80, _sm80_pick_flavor
from frost_test_utils import _SM, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]

_POINTS = [
    (64, 64, 16, "HND", "NHD", "none"),
    (128, 128, 64, "NHD", "HND", "causal"),
    (192, 128, 128, "HND", "HND", "br_swa"),
    (256, 256, 16, "NHD", "NHD", "br_swa"),
    (48, 32, 64, "NHD", "HND", "none"),
    (96, 80, 16, "HND", "NHD", "br"),
    (136, 112, 64, "HND", "NHD", "swa"),
    (248, 200, 128, "NHD", "HND", "causal"),
]


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_native_paged_adapter_rebind_replay(point, dtype, thd, graph_mode, cudnn_handle, monkeypatch, single_q=False):
    import cutlass.cute as cute

    index = _POINTS.index(point) * 4 + (2 if dtype == torch.float16 else 0) + int(thd)
    observations = []
    d, dv, page, klayout, vlayout, mask = point
    torch.manual_seed(721 + index)
    b, h, hk, max_sq, max_skv, tq = 3, 8, 2, 65, 129, 89
    if single_q:
        max_sq = 1
    cap = (tq + 64 if graph_mode else tq) if thd else max_sq
    qbatch = 1 if thd else b
    npages = (max_skv + page - 1) // page
    causal, br, swa = mask in ("causal", "br", "br_swa"), mask in ("br", "br_swa"), mask in ("swa", "br_swa")
    flavor = _sm80_pick_flavor(d, dv)
    fd, fv = {"gptoss": (64, 64), "llama": (128, 128), "dsv3": (192, 128), "qwen": (256, 256)}[flavor]
    m, n, w = (128, 64, 4) if fd == 64 else (64, 32 if fd == 256 else 64, 4)

    def allocate():
        q = torch.randn(qbatch, cap, h, d + 8, dtype=dtype, device="cuda")[..., :d]
        backing_o = torch.full((qbatch, cap, h, dv + 8), 77.0, dtype=dtype, device="cuda")
        o = backing_o[..., :dv]
        backing_l = torch.full((qbatch, h, cap * 2), 77.0, device="cuda")
        stats = backing_l[..., :cap] if thd else backing_l[..., ::2]
        pools = []
        for width, layout, extra in ((d, klayout, 3), (dv, vlayout, 5)):
            count = b * npages + extra
            raw = torch.empty((count, hk, page, width + 8) if layout == "HND" else (count, page, hk, width + 8), dtype=dtype, device="cuda")
            pools.append((raw.permute(0, 2, 1, 3) if layout == "HND" else raw)[..., :width])
        kt = torch.empty(npages * 2, b + 1, device="cuda", dtype=torch.int32).t()[:b, ::2]
        vt = torch.empty(b, npages * 2 + 1, device="cuda", dtype=torch.int32)[:, :npages]
        qlens = torch.empty(b + 1 if thd else b, device="cuda", dtype=torch.int32)
        klens = torch.empty_like(qlens)
        return [q, *pools, o, stats, klens, qlens, kt, vt], backing_o, backing_l

    buffers, ob, lb = allocate()
    q, k, v, o, stats, klens, qlens, kt, vt = buffers
    descs = []
    for name, tensor in zip(("q", "k", "v", "o"), (q, k, v, o)):
        view = tensor.transpose(1, 2)
        shape = (b, h, max_sq, tensor.shape[-1]) if thd and name in ("q", "o") else tuple(view.shape)
        descs.append(
            TensorDesc(
                dtype=dtype,
                shape=shape,
                stride=tuple(view.stride()),
                stride_order=tuple(sorted(range(4), key=lambda i: view.stride(i))),
                device=torch.device("cuda:0"),
                name=name,
            )
        )
    lse_desc = TensorDesc(
        dtype=torch.float32,
        shape=(b, h, max_sq),
        stride=tuple(stats.stride()),
        stride_order=tuple(sorted(range(3), key=lambda i: stats.stride(i))),
        device=torch.device("cuda:0"),
        name="lse",
    )
    api = SdpaFwdDslSm80(
        *descs,
        sample_lse=lse_desc,
        is_causal=causal,
        causal_bottom_right=br,
        window_size_left=31 if swa else None,
        stats_log2=br,
        tile_m=m,
        tile_n=n,
        sched_policy=0 if thd else (2 if causal else 0),
        thd=thd,
        seq_kv_lens_present=not thd,
        seq_q_lens_present=not thd,
        cu_seq_q_lens=thd,
        cu_seq_kv_lens=thd,
        max_total_seq_len_q=cap if thd else None,
        paged_page_size=page,
        paged_max_seq_len_kv=max_skv,
        paged_table_stride=tuple(kt.stride()),
        paged_table_v_stride=tuple(vt.stride()),
    )
    assert api.check_support()
    if graph_mode:
        io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
        fe = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        inputs = [fe.tensor(name=desc.name, dim=desc.shape, stride=desc.stride, data_type=io) for desc in descs[:3]]
        lengths = [fe.tensor(name=name, dim=(b + int(thd), 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32) for name in ("qlen", "klen")]
        tables = [
            fe.tensor(name=name, dim=(b, 1, npages, 1), stride=(table.stride(0), 0, table.stride(1), 1), data_type=cudnn.data_type.INT32)
            for name, table in (("kt", kt), ("vt", vt))
        ]
        opts = {("cu_seq_len_q" if thd else "seq_len_q"): lengths[0], ("cu_seq_len_kv" if thd else "seq_len_kv"): lengths[1]}
        if thd:
            opts["max_total_seq_len_q"] = cap
        out, st = fe.sdpa(
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            name="paged",
            attn_scale=1 / math.sqrt(d),
            generate_stats=True,
            use_padding_mask=True,
            use_causal_mask=causal and not br,
            use_causal_mask_bottom_right=causal and br,
            sliding_window_length=32 if swa else None,
            stats_use_log2=br,
            paged_attention_k_table=tables[0],
            paged_attention_v_table=tables[1],
            paged_attention_max_seq_len_kv=max_skv,
            **opts,
        )
        out.set_output(True).set_data_type(io).set_dim(descs[3].shape).set_stride(descs[3].stride)
        st.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((*lse_desc.shape, 1)).set_stride((*lse_desc.stride, 1))
        origins = []
        if thd:
            for name, node in (("q", inputs[0]), ("o", out), ("stats", st)):
                origin = fe.tensor(name=name + "_origin", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
                node.set_ragged_offset(origin)
                origins.append(origin)
        fe.validate()
        fe.build_operation_graph()
        fe.create_execution_plans([cudnn.heur_mode.A])
        select_engine(fe, "sdpa_fwd_prefill_sm80")
        fe.check_support()
        fe.build_plans()
        artifact = fe
        workspace = torch.empty(max(1, fe.get_workspace_size()), dtype=torch.uint8, device="cuda")
    else:
        api.compile()
        artifact = api._compiled_kernel
        workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda") if thd else None
    for binding in range(2):
        if binding:
            buffers, ob, lb = allocate()
        q, k, v, o, stats, klens, qlens, kt, vt = buffers
        graph = torch.cuda.CUDAGraph()
        origin_buffers = [torch.empty(b + 1, dtype=torch.int64, device="cuda") for _ in range(3)] if graph_mode and thd else []
        try:
            phases = (((1, 0, 1), (129, 33, 0)), ((0, 1, 1), (33, 129, 0))) if single_q else (((65, 8, 16), (129, 33, 0)), ((8, 33, 0), (33, 129, 0)))
            for phase, (qs, ks) in enumerate(phases):
                cq, ck = [0], [0]
                for x, y in zip(qs, ks):
                    cq.append(cq[-1] + x)
                    ck.append(ck[-1] + y)
                qlens.copy_(torch.tensor(cq if thd else qs, device="cuda", dtype=torch.int32))
                klens.copy_(torch.tensor(ck if thd else ks, device="cuda", dtype=torch.int32))
                qstarts = ([5, 75, 105] if phase == 0 else [3, 49, 113]) if graph_mode and thd else cq[:-1]
                ostarts = [x + 7 for x in qstarts] if graph_mode and thd else qstarts
                lstarts = [x + 13 for x in qstarts] if graph_mode and thd else qstarts
                if origin_buffers:
                    for buf, starts, stride in zip(origin_buffers, (qstarts, ostarts, lstarts), (q.stride(1), o.stride(1), stats.stride(2))):
                        buf.copy_(torch.tensor([x * stride for x in starts] + [cap * stride], dtype=torch.int64, device="cuda"))
                q.mul_(-0.7)
                k.fill_(float("nan"))
                v.fill_(float("nan"))
                kt.fill_(-171)
                vt.fill_(-173)
                refs = []
                for batch, (sq, sk) in enumerate(zip(qs, ks)):
                    kk = torch.randn(sk, hk, d, dtype=dtype, device="cuda")
                    vv = torch.randn(sk, hk, dv, dtype=dtype, device="cuda")
                    for pg in range((sk + page - 1) // page):
                        ki = (batch * npages + pg + phase + binding) % k.shape[0]
                        vi = (v.shape[0] - 1 - batch * npages - pg - phase - binding) % v.shape[0]
                        kt[batch, pg] = ki
                        vt[batch, pg] = vi
                        count = min(page, sk - pg * page)
                        k[ki, :count].copy_(kk[pg * page : pg * page + count])
                        v[vi, :count].copy_(vv[pg * page : pg * page + count])
                    qq = q[0, qstarts[batch] : qstarts[batch] + sq] if thd else q[batch, :sq]
                    scores = qq.transpose(0, 1).double() @ kk.transpose(0, 1).double().repeat_interleave(h // hk, 0).transpose(-1, -2) / math.sqrt(d)
                    qp = torch.arange(sq, device="cuda")[:, None] + (sk - sq if br else 0)
                    kp = torch.arange(sk, device="cuda")[None, :]
                    if causal:
                        scores.masked_fill_(kp > qp, -float("inf"))
                    if swa:
                        scores.masked_fill_(kp < qp - 31, -float("inf"))
                    ls = scores.logsumexp(-1)
                    oo = torch.nan_to_num(scores.softmax(-1), nan=0) @ vv.transpose(0, 1).double().repeat_interleave(h // hk, 0)
                    refs.append((oo.transpose(0, 1), ls * (math.log2(math.e) if br else 1)))
                ob.fill_(77)
                lb.fill_(77)

                def run():
                    before = torch.cuda.get_sync_debug_mode()
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        with monkeypatch.context() as guard:

                            def forbidden(*args, **kwargs):
                                raise AssertionError("paged execute must not compile, allocate, or copy data")

                            guard.setattr(cute, "compile", forbidden)
                            guard.setattr(torch, "empty", forbidden)
                            guard.setattr(torch, "zeros", forbidden)
                            guard.setattr(torch.Tensor, "contiguous", forbidden)
                            guard.setattr(torch.Tensor, "clone", forbidden)
                            if graph_mode:
                                vp = dict(
                                    zip(
                                        (*inputs, out, st, *lengths, *tables, *origins),
                                        (
                                            q.transpose(1, 2),
                                            k.transpose(1, 2),
                                            v.transpose(1, 2),
                                            o.transpose(1, 2),
                                            stats,
                                            qlens,
                                            klens,
                                            kt,
                                            vt,
                                            *origin_buffers,
                                        ),
                                    )
                                )
                                previous = cudnn.get_stream(cudnn_handle)
                                try:
                                    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
                                    fe.execute(vp, workspace, handle=cudnn_handle)
                                finally:
                                    cudnn.set_stream(cudnn_handle, previous)
                            else:
                                api.execute(
                                    q.transpose(1, 2),
                                    k.transpose(1, 2),
                                    v.transpose(1, 2),
                                    o.transpose(1, 2),
                                    stats,
                                    seq_q_lens=qlens,
                                    seq_kv_lens=klens,
                                    workspace=workspace,
                                    block_table=kt,
                                    block_table_v=vt,
                                )
                    finally:
                        torch.cuda.set_sync_debug_mode(before)
                    assert (fe if graph_mode else api._compiled_kernel) is artifact

                if not graph_mode and phase == 0 and binding == 0 and point == _POINTS[0] and dtype == torch.bfloat16:
                    for bad_table in (None, kt.to(torch.int64), kt[: b - 1], kt.contiguous()):
                        with pytest.raises(ValueError):
                            api.execute(
                                q.transpose(1, 2),
                                k.transpose(1, 2),
                                v.transpose(1, 2),
                                o.transpose(1, 2),
                                stats,
                                seq_q_lens=qlens,
                                seq_kv_lens=klens,
                                workspace=workspace,
                                block_table=bad_table,
                                block_table_v=vt,
                            )
                        assert torch.all(ob == 77) and torch.all(lb == 77)
                if phase == 0:
                    run()
                else:
                    graph.replay()
                for batch, (sq, sk) in enumerate(zip(qs, ks)):
                    actual_o = o[0, ostarts[batch] : ostarts[batch] + sq] if thd else o[batch, :sq]
                    actual_l = stats[0, :, lstarts[batch] : lstarts[batch] + sq] if thd else stats[batch, :, :sq]
                    torch.testing.assert_close(actual_o.double(), refs[batch][0], atol=4e-3, rtol=1e-2)
                    torch.testing.assert_close(actual_l.double(), refs[batch][1], atol=2e-3, rtol=1e-3)
                    if not thd:
                        assert torch.all(o[batch, sq:] == 0)
                        assert torch.all(stats[batch, :, sq:] == -float("inf"))
                assert torch.all(ob[..., dv:] == 77)
                assert torch.all(lb[..., cap:] == 77) if thd else torch.all(lb[..., 1::2] == 77)
                if thd:
                    om = torch.ones(cap, dtype=torch.bool, device="cuda")
                    lm = torch.ones_like(om)
                    for so, sl, count in zip(ostarts, lstarts, qs):
                        om[so : so + count] = False
                        lm[sl : sl + count] = False
                    assert torch.all(o[0, om] == 77) and torch.all(stats[0, :, lm] == 77)
                rec = dict(index=index, point=point, dtype=str(dtype), thd=thd, binding=binding, phase=phase, pass_full_reference=True)
                observations.append(rec)
                if phase == 0:
                    with torch.cuda.graph(graph):
                        run()
        finally:
            graph.reset()
    assert len(observations) == 4
