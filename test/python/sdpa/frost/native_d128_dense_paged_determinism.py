# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense-Q native D128 paging: physical layouts, replay and batch invariance."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from sm107_half_test_utils import _rubin_only

_CASES = [
    ("bfloat16", "bfloat16", 1, 1, 64),
    ("float16", "float16", 2, 4, 128),
    ("bfloat16", "float8_e4m3fn", 2, 1, 64),
    ("float16", "float8_e5m2", 1, 4, 128),
    ("bfloat16", "bfloat16", 1, 4, 64),
    ("bfloat16", "float8_e4m3fn", 2, 4, 64),
]


def _runner(dtype_name, kv_name, cga, split, tile_n, head_dim=128):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _rubin_only()
    dtype, kv_dtype = getattr(torch, dtype_name), getattr(torch, kv_name)
    h, kh, sq, d, page, blocks = 8, 2, 17, head_dim, 16, 17
    rng = torch.Generator().manual_seed(73461)
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    # Independent pool layouts and aligned interior head padding.
    k = (torch.randn(89, kh, page, d + 16, generator=rng) * 0.3).to(kv_dtype).cuda()[..., :d]
    v = (torch.randn(89, page, kh, d + 16, generator=rng) * 0.3).to(kv_dtype).cuda()[..., :d].transpose(1, 2)
    kt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(89, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    ql = torch.tensor([17, 1, 0, 9, 16], device="cuda", dtype=torch.int32)
    kl = torch.tensor([257, 33, 0, 129, 1], device="cuda", dtype=torch.int32)
    sink = torch.linspace(-0.5, 0.5, h, device="cuda")
    source = dict(q=q, k=k, v=v, kt=kt, vt=vt, ql=ql, kl=kl, sink=sink)
    before = {n: t.clone() for n, t in source.items()}
    plans = {}

    def buffers(order):
        index = torch.tensor(order, device="cuda")
        b = len(order)
        # Native descriptors must preserve head padding and the declared BSHD order.
        qb = torch.empty(b, sq, h, d + 8, device="cuda", dtype=dtype)[..., :d].transpose(1, 2)
        qb.copy_(q.index_select(0, index))
        ob_storage = torch.full((b, sq, h, d + 16), -123.0, device="cuda", dtype=dtype)
        ob = ob_storage[..., :d].transpose(1, 2)
        return dict(
            q=qb,
            k=k,
            v=v,
            o=ob,
            o_storage=ob_storage,
            lse=torch.empty(b, h, sq, device="cuda"),
            kt=kt.index_select(0, index).t().contiguous().t(),
            vt=vt.index_select(0, index).t().contiguous().t(),
            ql=ql.index_select(0, index),
            kl=kl.index_select(0, index),
            sink=sink,
        )

    def reference(t):
        assert torch.all(t["o_storage"][..., d:] == -123), "output padding overwritten"
        expected_o = torch.zeros_like(t["o"], device="cpu", dtype=torch.float64)
        expected_lse = torch.full(t["lse"].shape, -torch.inf, dtype=torch.float64)
        for i, (nq, nk) in enumerate(zip(t["ql"].tolist(), t["kl"].tolist())):
            if not nq:
                continue
            qq = t["q"][i, :, :nq].cpu().double()
            kk = t["k"][t["kt"][i].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
            vv = t["v"][t["vt"][i].long()].cpu().transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].double().repeat_interleave(h // kh, 0)
            scores = qq @ kk.transpose(-1, -2) * d**-0.5
            diagonal = torch.arange(nq)[:, None] + nk - nq
            keys = torch.arange(nk)[None, :]
            scores.masked_fill_((keys > diagonal) | (keys < diagonal - 73), -torch.inf)
            all_scores = torch.cat([scores, sink.cpu().double().view(h, 1, 1).expand(h, nq, 1)], -1)
            expected_o[i, :, :nq] = all_scores.softmax(-1)[..., :nk] @ vv
            expected_lse[i, :, :nq] = all_scores.logsumexp(-1)
        torch.testing.assert_close(t["o"].cpu().double(), expected_o, atol=0.003, rtol=0.02)
        torch.testing.assert_close(t["lse"].cpu().double(), expected_lse, atol=0.0006, rtol=0)
        return {n: t[n] for n in ("o", "lse")}

    def prepare(t):
        b = len(t["q"])
        if b not in plans:
            api = SdpaFwdDslSm100(
                t["q"],
                k,
                v,
                t["o"],
                t["lse"],
                is_causal=True,
                causal_bottom_right=True,
                window_size_left=73,
                has_sink=True,
                seq_q_lens_present=True,
                seq_kv_lens_present=True,
                paged_page_size=page,
                paged_max_seq_len_kv=page * blocks,
                paged_table_stride=t["kt"].stride(),
                paged_table_v_stride=t["vt"].stride(),
                cga=cga,
                split_kv=split,
                pack_gqa=True,
                tile_n=tile_n,
            )
            api.check_support()
            api.compile()
            assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
            assert api.flavor == (head_dim, head_dim)
            assert api._k_mod.CFG.PAGED_KV and not api._k_mod.CFG.THD_VARLEN
            plans[b] = api
        return plans[b]

    def execute(api, t, ws):
        api.execute(
            t["q"],
            t["k"],
            t["v"],
            t["o"],
            t["lse"],
            sinks=t["sink"],
            seq_q_lens=t["ql"],
            seq_kv_lens=t["kl"],
            block_table=t["kt"],
            block_table_v=t["vt"],
            workspace=ws,
        )

    def run(order, capture=False):
        t = buffers(order)
        api = prepare(t)
        ws = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")
        t["o"].fill_(torch.nan)
        t["lse"].fill_(torch.nan)
        execute(api, t, ws)
        result = reference(t)
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    execute(api, t, ws)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            try:
                t["q"].mul_(0.5)
                t["ql"].copy_(t["ql"].flip(0))
                t["kl"].copy_(t["kl"].flip(0))
                t["kt"].copy_(t["kt"].flip(1))
                t["vt"].copy_(t["vt"].flip(1))
                t["o"].fill_(torch.nan)
                t["lse"].fill_(torch.nan)
                ws.fill_(255)
                graph.replay()
                result = reference(t)
            finally:
                torch.cuda.synchronize()
                graph.reset()
        assert_bitwise(source, before, "input bytes")
        return result

    return run, before


@pytest.mark.parametrize("dtype_name,kv_name,cga,split,tile_n", _CASES)
def test_replay_batch_and_golden(dtype_name, kv_name, cga, split, tile_n, head_dim=128):
    run, inputs = _runner(dtype_name, kv_name, cga, split, tile_n, head_dim=head_dim)
    target = assert_replays(lambda: run([0]), repeats=4)
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({n: actual[n][row : row + 1] for n in target}, target, f"batch row {row}")
    run([0, 1, 2, 3, 4], capture=True)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-half-d{head_dim}-dense-paged-{dtype_name}-{kv_name}-cga{cga}-split{split}-n{tile_n}", inputs, actual)
