# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated Hopper D128 device lengths, including changes after graph capture."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_d128_sm90 import _inputs, _plan
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires Hopper")]


def _reference(q, k, v, q_lengths, kv_lengths, mask, scale=128**-0.5):
    """FP32 attention on each live request, with declared padding written explicitly."""
    out = torch.zeros_like(q)
    stats = torch.full(q.shape[:3], -float("inf"), dtype=torch.float32, device=q.device)
    for batch, (qlen, klen) in enumerate(zip(q_lengths, kv_lengths)):
        if not qlen or not klen:
            continue
        keys = k[batch, :, :klen].float().repeat_interleave(q.shape[1] // k.shape[1], dim=0)
        values = v[batch, :, :klen].float().repeat_interleave(q.shape[1] // v.shape[1], dim=0)
        scores = (q[batch, :, :qlen].float() @ keys.transpose(-1, -2)) * scale
        if mask != "none":
            rows = torch.arange(qlen, device=q.device).view(-1, 1)
            cols = torch.arange(klen, device=q.device).view(1, -1)
            scores.masked_fill_(cols > rows + (klen - qlen if mask == "br" else 0), -float("inf"))
        stats[batch, :, :qlen] = torch.logsumexp(scores, dim=-1)
        out[batch, :, :qlen] = (torch.softmax(scores, dim=-1).nan_to_num() @ values).to(out.dtype)
    return out, stats


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["none", "tl", "br"])
@pytest.mark.parametrize("lengths", ["kv", "both"])
@torch_fork_set_rng(seed=1729)
def test_d128_runtime_lengths_replay(dtype, mask, lengths):
    # A whole declared Q tile must still mask a partial *runtime* Q tile.
    sq, skv = 256, 385
    q, k, v = _inputs(4, 4, 2, sq, skv, dtype)
    q_lens = torch.full((4,), sq, dtype=torch.int32, device="cuda") if lengths in ("q", "both") else None
    kv_lens = torch.full((4,), skv, dtype=torch.int32, device="cuda") if lengths in ("kv", "both") else None
    api, o, stats = _plan(
        q,
        k,
        v,
        stats=True,
        is_causal=mask != "none",
        causal_bottom_right=mask == "br",
        seq_q_lens_present=q_lens is not None,
        seq_kv_lens_present=kv_lens is not None,
    )
    assert not api.params.persistent and not api.params.causal_walk and not api.params.causal_claim
    assert api.scratch_workspace_bytes() == 0

    def execute():
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=stats, seq_q_lens=q_lens, seq_kv_lens=kv_lens)

    previous_debug = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        execute()
    finally:
        torch.cuda.set_sync_debug_mode(previous_debug)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        execute()
    try:
        for q_values, kv_values in (
            ([sq] * 4, [skv] * 4),
            ([0, 65, 127, sq], [skv, 0, 31, skv - 1]),
            ([sq, 129, 0, 1], [0, 1, skv, 129]),
        ):
            if q_lens is not None:
                q_lens.copy_(torch.tensor(q_values, dtype=torch.int32, device="cuda"))
            else:
                q_values = [sq] * 4
            if kv_lens is not None:
                kv_lens.copy_(torch.tensor(kv_values, dtype=torch.int32, device="cuda"))
            else:
                kv_values = [skv] * 4
            v.add_(0.125)
            o.fill_(float("nan"))
            stats.fill_(float("nan"))
            captured.replay()
            torch.cuda.current_stream().synchronize()
            expected_o, expected_stats = _reference(q, k, v, q_values, kv_values, mask)
            torch.testing.assert_close(o, expected_o, atol=5e-2, rtol=3e-2)
            torch.testing.assert_close(stats, expected_stats, atol=2e-2, rtol=2e-2)
    finally:
        captured.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [128**-0.5, 0.0, -0.125])
@torch_fork_set_rng(seed=1731)
def test_d128_lengths_without_stats_on_partial_q_tail(dtype, scale):
    # Declared Q > KV is legal with lengths. Bottom-right causal has fully
    # masked leading rows, a whole empty Q tile, and a partial declared tail.
    sq, skv = 257, 129
    q, k, v = _inputs(4, 4, 2, sq, skv, dtype)
    q_values, kv_values = [sq, 129, 0, 65], [skv, 0, skv, 31]
    q_lens = torch.tensor(q_values, dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor(kv_values, dtype=torch.int32, device="cuda")
    api, o, stats = _plan(
        q,
        k,
        v,
        is_causal=True,
        causal_bottom_right=True,
        scale_softmax=scale,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
    )
    assert stats is None and api.scratch_workspace_bytes() == 0
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, seq_q_lens=q_lens, seq_kv_lens=kv_lens)
    expected, _ = _reference(q, k, v, q_values, kv_values, "br", scale)
    torch.testing.assert_close(o, expected, atol=5e-2, rtol=3e-2)
