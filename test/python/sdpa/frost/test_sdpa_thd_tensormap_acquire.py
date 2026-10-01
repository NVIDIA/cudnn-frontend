# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable K/V maps must be acquired again after every THD setup launch."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(not 100 <= _SM <= 106, reason="SM100 half templates")]


@pytest.mark.parametrize("d,dv,cga", [(256, 256, 2), (192, 128, 1), (192, 128, 2)], ids=["d256_cga2", "d192_d128_cga1", "d192_d128_cga2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=1300)
def test_thd_tensormaps_rebind_and_replay(d, dv, cga, dtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    q_lengths = [193, 97, 33]
    b, h, hk, sq, sk = 3, 8, 2, 193, 257
    tq = sum(q_lengths)

    def tensor_view(raw, heads, length, width):
        return raw.as_strided((b, heads, length, width), (length * heads * width, width, heads * width, 1))

    def inputs(k_lengths):
        q = torch.full((b * sq, h, d), float("nan"), device="cuda", dtype=dtype)
        k = torch.full((b * sk, hk, d), float("nan"), device="cuda", dtype=dtype)
        v = torch.full((b * sk, hk, dv), float("nan"), device="cuda", dtype=dtype)
        q[:tq].normal_()
        k[: sum(k_lengths)].normal_()
        v[: sum(k_lengths)].normal_()
        out = torch.empty((b * sq, h, dv), device="cuda", dtype=dtype)
        stats = torch.empty((b * sq, h), device="cuda", dtype=torch.float32)
        return dict(
            q=q,
            k=k,
            v=v,
            out=out,
            stats=stats,
            q_lens=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"),
            k_lens=torch.tensor(k_lengths, dtype=torch.int32, device="cuda"),
        )

    def views(x):
        return (
            tensor_view(x["q"], h, sq, d),
            tensor_view(x["k"], hk, sk, d),
            tensor_view(x["v"], hk, sk, dv),
            tensor_view(x["out"], h, sq, dv),
            x["stats"].as_strided((b, h, sq), (sq * h, 1, h)),
        )

    first_lengths = [257, 161, 65]
    first = inputs(first_lengths)
    q, k, v, out, stats = views(first)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=stats,
        thd=True,
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens_present=True,
        cga=cga,
        sched_policy=1,
        split_kv=1,
        pack_gqa=False,
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.CFG.THD_VARLEN and not api._k_mod.PAGED_KV
    assert api._k_mod.CFG.CTA_MMA == cga
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def run(x):
        api.execute(*views(x)[:4], x["stats"], seq_q_lens=x["q_lens"], seq_kv_lens=x["k_lens"], workspace=workspace)

    def poison(x):
        x["out"].fill_(float("nan"))
        x["stats"].fill_(float("nan"))

    def check(x, k_lengths):
        qb = kb = 0
        for nq, nk in zip(q_lengths, k_lengths):
            q = x["q"][qb : qb + nq].double().transpose(0, 1)
            k = x["k"][kb : kb + nk].double().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            v = x["v"][kb : kb + nk].double().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            scores = q @ k.transpose(-1, -2) * d**-0.5
            mask = torch.arange(nk, device="cuda")[None, :] > torch.arange(nq, device="cuda")[:, None] + nk - nq
            scores.masked_fill_(mask, -float("inf"))
            prob = scores.softmax(-1)
            expected = prob @ v
            bound = torch.finfo(dtype).eps / 2 * (prob @ v.abs() + expected.abs()) + 2e-5
            error = (x["out"][qb : qb + nq].transpose(0, 1).double() - expected).abs()
            assert torch.all(error <= bound), float((error / bound).max())
            torch.testing.assert_close(x["stats"][qb : qb + nq].T.double(), scores.logsumexp(-1), atol=2e-4, rtol=0)
            qb += nq
            kb += nk

    poison(first)
    run(first)
    check(first, first_lengths)

    # Reuse the compiled plan with new pointers and a different packed total.
    second_lengths = [241, 145, 47]
    second = inputs(second_lengths)
    poison(second)
    run(second)
    check(second, second_lengths)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(second)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(second)
        poison(second)
        graph.replay()
        check(second, second_lengths)

        # Setup must republish the shorter extent on replay. The retired K/V
        # rows become NaN, so a stale descriptor corrupts the final partial tile.
        third_lengths = [225, 129, 35]
        second["k_lens"].copy_(torch.tensor(third_lengths, dtype=torch.int32))
        second["q"][:tq].normal_()
        for name in ("k", "v"):
            second[name][: sum(third_lengths)].normal_()
            second[name][sum(third_lengths) :].fill_(float("nan"))
        poison(second)
        graph.replay()
        check(second, third_lengths)
    finally:
        graph.reset()
