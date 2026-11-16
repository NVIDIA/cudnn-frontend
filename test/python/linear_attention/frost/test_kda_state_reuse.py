# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact summary/state aliasing preserves the FP32 piece-chain recurrence."""

import pytest
import torch

from determinism_utils import assert_bitwise, snapshot
from linear_attention.test_la import frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def test_forward_state_arena_policy():
    from cudnn.linear_attention.frost.kda_engine import _forward_head_divisor

    gib = 1024**3
    # Preserve the previous wave count below the live-arena threshold.
    assert _forward_head_divisor(96, 40 * gib, 2048, 148, 30 * gib) == 2
    assert _forward_head_divisor(96, 64 * gib, 4096, 148, 48 * gib) == 4
    assert _forward_head_divisor(96, 128 * gib, 8192, 148, 96 * gib) == 8
    # Exact equality stays below the threshold; do not split away parallelism.
    assert _forward_head_divisor(96, 64 * gib, 4096, 148, 32 * gib) == 2
    assert _forward_head_divisor(96, 128 * gib, 4, 148, 96 * gib) == 2
    assert _forward_head_divisor(3, 128 * gib, 8192, 148, 96 * gib) == 1


@pytest.mark.parametrize("key_dim,value_dim", [(64, 64), (128, 128), (64, 128), (128, 64)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pieces", [1, 3])
@pytest.mark.parametrize("has_seed", [False, True], ids=["zero-seed", "seeded"])
def test_state_chain_summary_alias_capture(key_dim, value_dim, reverse, pieces, has_seed):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, run_state_chain

    batch, heads = 3, 4
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            generator = torch.Generator(device="cuda").manual_seed(20261115)
            h = torch.randn(batch * pieces, heads, value_dim, key_dim, device="cuda", generator=generator) * 0.01
            m = torch.randn(batch * pieces, heads, key_dim, key_dim, device="cuda", generator=generator) * 0.001
            m += torch.eye(key_dim, device="cuda") * 0.9
            seed = torch.randn(batch, heads, value_dim, key_dim, device="cuda", generator=generator) * 0.01
            seed_arg = seed if has_seed else None
            storage = torch.full((h.numel() + 64,), 17.0, device="cuda")
            alias = storage[: h.numel()].view_as(h)
            separate = torch.empty_like(h)
            tails = [torch.empty_like(seed) for _ in range(2)]
            products = [torch.empty(batch, heads, key_dim, key_dim, device="cuda") for _ in range(2)]
            compiled = build_state_chain(
                heads_out=heads,
                dim_v=value_dim,
                dim_k=key_dim,
                pieces=pieces,
                reverse=reverse,
                transpose=reverse,
                has_seed=has_seed,
                has_tail=True,
                emit_summary=True,
                device=torch.cuda.current_device(),
                opt_level=3,
            )

            def run_inplace():
                alias.copy_(h)
                run_state_chain(compiled, batch, alias, m, alias, seed_arg, tails[1], products[1], stream.cuda_stream)

            run_inplace()
            with torch.cuda.graph(graph, stream=stream):
                run_inplace()
            for _ in range(3):
                h.mul_(-0.8)
                seed.add_(0.001)
                run_state_chain(compiled, batch, h, m, separate, seed_arg, tails[0], products[0], stream.cuda_stream)
                expected = snapshot(dict(state=separate, tail=tails[0], product=products[0]))
                alias.fill_(float("nan"))
                tails[1].fill_(float("nan"))
                products[1].fill_(float("nan"))
                graph.replay()
                actual = snapshot(dict(state=alias, tail=tails[1], product=products[1]))
                assert_bitwise(actual, expected, "in-place FP32 state chain")
                assert all(torch.isfinite(value).all() for value in actual.values())
                assert (storage[h.numel() :] == 17).all()
    finally:
        stream.synchronize()
        graph.reset()
        torch.cuda.current_stream().wait_stream(stream)
