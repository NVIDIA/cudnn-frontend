# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Public D256 paged packed-split execution with fixed sequence payloads."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from sdpa.frost.test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as _check_origins


@pytest.mark.parametrize(
    "dtype,layout,stats,splits,packed,page,heads",
    [
        ("float16", "hnd", "token", 2, False, 8, 16),
        ("bfloat16", "hnd", "head", 9, True, 16, 16),
        ("float16", "nhd", "token", 3, True, 128, 16),
        ("bfloat16", "mixed", "head", 5, True, 256, 24),
    ],
    ids=["fp16-unpacked-page8", "bf16-packed-page16", "fp16-packed-page128", "bf16-partial-packed-page256"],
)
def test_origins_and_replay(dtype, layout, stats, splits, packed, page, heads):
    _check_origins(256, dtype, layout, stats, 2, splits, packed, page_size=page, kv_bound=1024, cpu_reference=True, heads_override=heads)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_batch_and_golden(dtype_name):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    dtype = getattr(torch, dtype_name)
    h, kh, sq, d, page, blocks, splits = 8, 2, 129, 256, 16, 17, 5
    rng = torch.Generator().manual_seed(9361256)
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    k = (torch.randn(89, kh, page, d, generator=rng) * 0.3).to(dtype).cuda()
    v = (torch.randn(89, kh, page, d, generator=rng) * 0.3).to(dtype).cuda()
    kt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    qlens, klens = [129, 65, 0, 1, 33], [257, 1, 0, 33, 129]
    inputs = dict(q=q, k=k, v=v, kt=kt, vt=vt)
    before = snapshot(inputs)
    plans = {}

    def run(order):
        batch = len(order)
        lengths, kv_lengths = [qlens[i] for i in order], [klens[i] for i in order]
        total = sum(lengths)
        qb = torch.full((batch, sq, h, d), torch.nan, device="cuda", dtype=dtype)
        ob = torch.full_like(qb, -123.0)
        lb = torch.full((batch, sq, h), -123.0, device="cuda")
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
        if batch not in plans:
            api = SdpaFwdDslSm100(
                qbind,
                k,
                v,
                obind,
                lbind,
                thd=True,
                is_causal=True,
                causal_bottom_right=True,
                seq_kv_lens_present=True,
                paged_page_size=page,
                paged_max_seq_len_kv=page * blocks,
                cga=2,
                split_kv=splits,
                pack_gqa=True,
            )
            api.check_support()
            api.compile()
            assert api.flavor == (256, 256) and api.packed_thd_split
            assert api._k_mod.__file__.endswith("sm100/d256_f16_lookahead/kernel.py")
            assert api._k_mod.CFG.PACK_GQA and api._k_mod.CFG.SPLIT_KV == splits
            plans[batch] = api
        api = plans[batch]
        workspace = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")
        q_before = qb.clone()
        api.execute(qbind, k, v, obind, lflat, seq_q_lens=ql, seq_kv_lens=kl, block_table=tables[0], block_table_v=tables[1], workspace=workspace)
        assert torch.all(oflat[total:] == -123), "packed output capacity tail was written"
        assert torch.all(lflat[total:] == -123), "packed Stats capacity tail was written"
        assert_bitwise({"q": qb}, {"q": q_before}, "packed Q bytes")
        assert_bitwise(inputs, before, "source bytes")
        refs = {}
        offset = 0
        for sequence, nq, nk in zip(order, lengths, kv_lengths):
            if not nq:
                continue
            if sequence not in refs:
                qq = q[sequence, :, :nq].cpu().double()
                kk = k[kt[sequence].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
                vv = v[vt[sequence].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
                scores = qq @ kk.transpose(-1, -2) * d**-0.5
                scores.masked_fill_(torch.arange(nk)[None, :] > torch.arange(nq)[:, None] + nk - nq, -torch.inf)
                refs[sequence] = (scores.softmax(-1).nan_to_num() @ vv).transpose(0, 1), scores.logsumexp(-1).t()
            ref_o, ref_lse = refs[sequence]
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
                assert_bitwise({name: actual[name][offset : offset + nq] for name in target}, target, f"packed origin {offset}")
            offset += nq
    neighbor_q = q[1].clone()
    q[1].neg_()
    before = snapshot(inputs)
    changed = snapshot(run([1, 0, 2]))
    start = qlens[1]
    assert_bitwise({name: changed[name][start : start + qlens[0]] for name in target}, target, "changed neighbor")
    q[1].copy_(neighbor_q)
    before = snapshot(inputs)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-half-d256-thd-paged-packed-split5-{dtype_name}", inputs, actual)
