# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay, batch invariance and refill stress for half-width D128 decode.

This is forward-only coverage of the standalone API. Graph entry points are
covered by test_sdpa_fwd_decode_live_columns_sm100; backward uses other kernels.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]
CASES = [(torch.float16, 1, 8, 128, 128, 1), (torch.bfloat16, 1, 8, 128, 128, 4), (torch.float16, 2, 4, 80, 120, 1), (torch.bfloat16, 2, 4, 128, 128, 4)]
IDS = ["fp16-q1-g8-split1", "bf16-q1-g8-split4", "fp16-q2-g4-d80-v120", "bf16-q2-g4-split4"]


def _runner(dtype, sq, h, d, dv, split):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    rng = torch.Generator().manual_seed(73199)
    page, blocks, pool = 16, 33, 47
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    # Separate K/V page maps, layouts, and head padding are runtime strides.
    k = (torch.randn(pool, 1, page, d + 16, generator=rng) * 0.3).to(dtype).cuda()[..., :d]
    v = (torch.randn(pool, page, 1, dv + 8, generator=rng) * 0.3).to(dtype).cuda()[..., :dv].transpose(1, 2)
    kt = torch.stack([torch.randperm(pool, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(pool, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    ql = torch.tensor([sq, max(sq - 1, 0), 0, sq, 1], device="cuda", dtype=torch.int32)
    kl = torch.tensor([513, 129, 0, 1, 257], device="cuda", dtype=torch.int32)
    sink = torch.linspace(-0.5, 0.5, h, device="cuda")
    inputs = dict(q=q, k=k, v=v, kt=kt, vt=vt, ql=ql, kl=kl, sink=sink)
    before = {name: value.clone() for name, value in inputs.items()}
    plans = {}

    def buffers(order):
        index = torch.tensor(order, device="cuda")
        b = len(order)
        qb = torch.empty(b, sq, h, d + 8, device="cuda", dtype=dtype)[..., :d].transpose(1, 2)
        qb.copy_(q.index_select(0, index))
        return dict(
            q=qb,
            o=torch.empty(b, h, sq, dv, device="cuda", dtype=dtype),
            lse=torch.empty(b, h, sq, device="cuda"),
            kt=kt.index_select(0, index).t().contiguous().t(),
            vt=vt.index_select(0, index).t().contiguous().t(),
            ql=ql.index_select(0, index),
            kl=kl.index_select(0, index),
        )

    def reference(t):
        expected = torch.zeros(t["o"].shape, dtype=torch.float64)
        lse = torch.full(t["lse"].shape, -torch.inf, dtype=torch.float64)
        for b, (nq, nk) in enumerate(zip(t["ql"].tolist(), t["kl"].tolist())):
            if nq == 0:
                continue
            qq = t["q"][b, :, :nq].cpu().double()
            kk = k[t["kt"][b].long()].cpu().reshape(blocks * page, d)[:nk].double()
            vv = v[t["vt"][b].long()].cpu().reshape(blocks * page, dv)[:nk].double()
            scores = qq @ kk.t() * d**-0.5
            diagonal = torch.arange(nq)[:, None] + nk - nq
            keys = torch.arange(nk)[None, :]
            scores.masked_fill_((keys > diagonal) | (keys < diagonal - 73), -torch.inf)
            all_scores = torch.cat([scores, sink.cpu().double().view(h, 1, 1).expand(h, nq, 1)], -1)
            expected[b, :, :nq] = all_scores.softmax(-1)[..., :nk] @ vv
            lse[b, :, :nq] = all_scores.logsumexp(-1)
        torch.testing.assert_close(t["o"].cpu().double(), expected, atol=0.003, rtol=0.02)
        torch.testing.assert_close(t["lse"].cpu().double(), lse, atol=0.0006, rtol=0)
        return {name: t[name] for name in ("o", "lse")}

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
                cga=1,
                split_kv=split,
                pack_gqa=True,
            )
            api.check_support()
            api.compile()
            assert api._k_mod.__file__.endswith("sm100/d256_f16_swap_ab/kernel.py")
            assert api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 128
            assert api._k_mod.N_Q == 16 and api._k_mod.COLS == 8
            plans[b] = api
        api = plans[b]
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")

        def launch():
            api.execute(
                t["q"], k, v, t["o"], t["lse"], sinks=sink, seq_q_lens=t["ql"], seq_kv_lens=t["kl"], block_table=t["kt"], block_table_v=t["vt"], workspace=ws
            )

        return launch, ws

    def run(order, capture=False):
        t = buffers(order)
        launch, ws = prepare(t)
        t["o"].fill_(torch.nan)
        t["lse"].fill_(torch.nan)
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            launch()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        result = reference(t)
        if capture:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            baseline = snapshot(result)
            try:
                v.mul_(0.5)
                t["o"].fill_(torch.nan)
                t["lse"].fill_(torch.nan)
                ws.fill_(255)
                graph.replay()
                reference(t)
                v.copy_(before["v"])
                graph.replay()
                assert_bitwise(reference(t), baseline, "restored retained replay")
                t["ql"].copy_(t["ql"].flip(0))
                t["kl"].copy_(t["kl"].flip(0))
                t["kt"].copy_(t["kt"].flip(1))
                t["vt"].copy_(t["vt"].flip(1))
                graph.replay()
                result = reference(t)
            finally:
                torch.cuda.synchronize()
                graph.reset()
                v.copy_(before["v"])
        assert_bitwise(inputs, before, "input mutation")
        return result

    return run, before, buffers, prepare


@pytest.mark.parametrize("dtype,sq,h,d,dv,split", CASES, ids=IDS)
def test_live_columns_replay_batch(dtype, sq, h, d, dv, split):
    run, _, _, _ = _runner(dtype, sq, h, d, dv, split)
    target = assert_replays(lambda: run([0]), repeats=4)
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 256 + [0] + [2] * 256):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({name: actual[name][row : row + 1] for name in target}, target, f"batch row {row}")
    run([0, 1, 2, 3, 4], capture=True)


@pytest.mark.parametrize("dtype,sq,h,d,dv,split", CASES, ids=IDS)
def test_live_columns_golden(dtype, sq, h, d, dv, split):
    run, inputs, _, _ = _runner(dtype, sq, h, d, dv, split)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-live8-{dtype}-q{sq}-g{h}-d{d}-v{dv}-split{split}", inputs, actual)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_live_columns_warm_refill(dtype):
    from warm_stress import StressCase, assert_warm_stress

    run, _, buffers, prepare = _runner(dtype, 1, 8, 128, 128, 1)
    reference = snapshot(run([0]))
    b = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    t = buffers([0] * b)
    launch, _ = prepare(t)
    launch()
    expected = {name: value.expand((b,) + value.shape[1:]).contiguous() for name, value in reference.items()}
    assert_bitwise({name: t[name] for name in expected}, expected, "stress batch invariance")
    # The 73-wide sliding window straddles two 128-key tiles at length 513;
    # K/V alternate through the three-slot ring, so the first stage is refilled.
    assert_warm_stress(
        StressCase("D128 live-column decode", launch, {name: t[name] for name in expected}, thread_blocks=b, refills_per_launch=1, reference=expected),
        warm=256,
        poisoned=256,
    )
