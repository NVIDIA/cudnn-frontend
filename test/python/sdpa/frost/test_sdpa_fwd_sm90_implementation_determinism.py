# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed SM90 D64 forward records; no backward engine belongs to this migration."""

import pytest
import torch
from determinism_utils import assert_bitwise, assert_replays, snapshot, assert_golden
from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_sm90_implementation_contract import _graph

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native Hopper")]


def _inputs(dtype, b=3, h=4, sq=65, skv=257):
    torch.manual_seed(2751)
    q = torch.randn(b, sq, h, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k, v = (torch.randn(b, skv, 1, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
    return q, k, v


def _run(q, k, v, *, causal=False, scheduler=0):
    g, vp, ws, outputs, _ = _graph(q, k, v, causal=causal, scheduler=scheduler)

    def execute():
        g.execute(vp, ws)
        return outputs

    return execute


def _reference(q, k, v, outputs, causal):
    scores = (q.double() @ k.double().repeat_interleave(q.shape[1], dim=1).transpose(-1, -2)) * 0.125
    if causal:
        rows = torch.arange(q.shape[2], device="cuda")[:, None]
        keys = torch.arange(k.shape[2], device="cuda")[None, :]
        scores.masked_fill_(keys > rows, -float("inf"))
    ref = scores.softmax(-1) @ v.double().repeat_interleave(q.shape[1], dim=1)
    torch.testing.assert_close(outputs["o"], ref.to(q.dtype), atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(outputs["lse"], scores.logsumexp(-1).float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
@pytest.mark.parametrize("scheduler", [0, 1, 2], ids=["natural", "lpt", "lpt_l2"])
def test_fixed_d64_replay_and_batch_invariance(dtype, causal, scheduler):
    q, k, v = _inputs(dtype)
    run = _run(q, k, v, causal=causal, scheduler=scheduler)
    expected = assert_replays(run, repeats=4)
    _reference(q, k, v, expected, causal)
    for indexes in ([1], [2, 1, 0], [1, 0, 1, 2]):
        ids = torch.tensor(indexes, device="cuda")
        qt, kt, vt = (t.index_select(0, ids) for t in (q, k, v))
        actual = snapshot(_run(qt, kt, vt, causal=causal, scheduler=scheduler)())
        want = {name: t.index_select(0, ids) for name, t in expected.items()}
        assert_bitwise(actual, want, f"batch {indexes}")
    q2, k2, v2 = (t.clone() for t in (q, k, v))
    q2[0].neg_()
    k2[0].mul_(2)
    v2[0].zero_()
    actual = snapshot(_run(q2, k2, v2, causal=causal, scheduler=scheduler)())
    assert_bitwise({n: t[1:] for n, t in actual.items()}, {n: t[1:] for n, t in expected.items()}, "changed neighbor")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_fixed_d64_warm_stage_refills(dtype):
    from warm_stress import StressCase, assert_warm_stress

    q, k, v = _inputs(dtype, b=8, h=16, sq=129, skv=257)
    run = _run(q, k, v)
    outputs = run()
    _reference(q, k, v, outputs, False)
    # O is BSHD physical: compare the actual storage view, never a contiguous copy.
    physical = {"o": outputs["o"].transpose(1, 2), "lse": outputs["lse"]}
    assert all(t.is_contiguous() for t in physical.values())
    assert_warm_stress(
        StressCase(name=f"sm90-explicit-d64-{dtype}", launch=run, outputs=physical, thread_blocks=8 * 16 * 3, refills_per_launch=4), warm=3000, poisoned=3000
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_fixed_d64_runtime_golden(dtype):
    q, k, v = _inputs(dtype, b=1, h=4, sq=65, skv=129)
    outputs = assert_replays(_run(q, k, v, causal=True), repeats=3)
    _reference(q, k, v, outputs, True)
    assert_golden(f"sdpa-sm90-explicit-d64-causal-{dtype}", {"q": q, "k": k, "v": v}, outputs)
