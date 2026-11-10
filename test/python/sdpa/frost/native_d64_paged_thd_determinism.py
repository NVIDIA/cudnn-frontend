# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-payload compact THD batch checks for native SM107 D64 page loads."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from sm107_half_test_utils import _rubin_only


@pytest.mark.parametrize(
    "dtype_name,cga,packed,tile_n",
    [
        ("float16", 1, False, 128),
        ("bfloat16", 2, False, 64),
        ("float16", 2, True, 128),
        ("bfloat16", 1, True, 64),
    ],
)
def test_thd_batch_and_golden(dtype_name, cga, packed, tile_n, kv_name=None):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _rubin_only()
    dtype = getattr(torch, dtype_name)
    kv_dtype = getattr(torch, kv_name or dtype_name)
    h, kh, sq, d, page, blocks = 8, 2, 17, 64, 16, 17
    rng = torch.Generator().manual_seed(93612)
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    k = (torch.randn(89, kh, page, d, generator=rng) * 0.3).to(kv_dtype).cuda()
    v = (torch.randn(89, page, kh, d, generator=rng) * 0.3).to(kv_dtype).cuda().transpose(1, 2)
    kt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    qlens, klens = [17, 9, 0, 1, 16], [257, 1, 0, 33, 129]
    sink = torch.linspace(-0.5, 0.5, h, device="cuda")
    inputs = dict(q=q, k=k, v=v, kt=kt, vt=vt, sink=sink)
    before = {n: t.clone() for n, t in inputs.items()}
    plans = {}

    def run(order):
        b = len(order)
        lengths = [qlens[i] for i in order]
        kv_lengths = [klens[i] for i in order]
        total = sum(lengths)
        qb = torch.full((b, sq, h, d), torch.nan, device="cuda", dtype=dtype)
        ob = torch.full_like(qb, -123.0)
        lb = torch.full((b, sq, h), -123.0, device="cuda")
        qflat, oflat, lflat = qb.view(-1, h, d), ob.view(-1, h, d), lb.view(-1, h)
        offset = 0
        for sequence, nq in zip(order, lengths):
            qflat[offset : offset + nq].copy_(q[sequence, :, :nq].transpose(0, 1))
            offset += nq
        qbind, obind, lbind = qb.transpose(1, 2), ob.transpose(1, 2), lb.transpose(1, 2)
        index = torch.tensor(order, device="cuda")
        tables = kt.index_select(0, index), vt.index_select(0, index)
        ql = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        kl = torch.tensor(kv_lengths, dtype=torch.int32, device="cuda")
        if b not in plans:
            api = SdpaFwdDslSm100(
                qbind,
                k,
                v,
                obind,
                lbind,
                thd=True,
                is_causal=True,
                causal_bottom_right=True,
                window_size_left=73,
                has_sink=True,
                seq_kv_lens_present=True,
                paged_page_size=page,
                paged_max_seq_len_kv=page * blocks,
                cga=cga,
                split_kv=1,
                pack_gqa=packed,
                tile_n=tile_n,
            )
            api.check_support()
            api.compile()
            assert api.flavor == (64, 64)
            assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
            assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.THD_VARLEN
            plans[b] = api
        api = plans[b]
        ws = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")
        qb_before = qb.clone()
        api.execute(qbind, k, v, obind, lflat, sinks=sink, seq_q_lens=ql, seq_kv_lens=kl, block_table=tables[0], block_table_v=tables[1], workspace=ws)
        assert torch.all(oflat[total:] == -123), "packed output capacity tail was written"
        assert torch.all(lflat[total:] == -123), "packed Stats capacity tail was written"
        assert_bitwise({"q": qb}, {"q": qb_before}, "packed Q bytes")
        assert_bitwise(inputs, before, "source bytes")
        offset = 0
        for sequence, nq, nk in zip(order, lengths, kv_lengths):
            if not nq:
                continue
            qq = q[sequence, :, :nq].cpu().double()
            kk = k[kt[sequence].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
            vv = v[vt[sequence].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
            scores = qq @ kk.transpose(-1, -2) * d**-0.5
            diagonal = torch.arange(nq)[:, None] + nk - nq
            keys = torch.arange(nk)[None, :]
            scores.masked_fill_((keys > diagonal) | (keys < diagonal - 73), -torch.inf)
            all_scores = torch.cat([scores, sink.cpu().double().view(h, 1, 1).expand(h, nq, 1)], -1)
            ref_o = (all_scores.softmax(-1)[..., :nk] @ vv).transpose(0, 1)
            ref_lse = all_scores.logsumexp(-1).t()
            torch.testing.assert_close(oflat[offset : offset + nq].cpu().double(), ref_o, atol=0.003, rtol=0.02)
            torch.testing.assert_close(lflat[offset : offset + nq].cpu().double(), ref_lse, atol=0.0006, rtol=0)
            offset += nq
        return {"o": oflat[:total], "lse": lflat[:total]}

    target = assert_replays(lambda: run([0]), repeats=4)
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order))
        offset = 0
        for sequence in order:
            nq = qlens[sequence]
            if sequence == 0:
                assert_bitwise({n: actual[n][offset : offset + nq] for n in target}, target, f"packed origin {offset}")
            offset += nq
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    storage = f"-{kv_name}" if kv_name is not None else ""
    assert_golden(f"sdpa-half-d64-thd-paged-{dtype_name}{storage}-cga{cga}-pack{int(packed)}-n{tile_n}", inputs, actual)
