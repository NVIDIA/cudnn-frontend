# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``sdpa_bwd_sm90`` deterministic mode: raw-byte replay and batch invariance.

``use_deterministic_algorithm=True`` orders the dQ accumulation across KV tiles
with the SM80 relay; dK / dV are written once per CTA and the GQA partials are
reduced in a fixed order. The target sequence spans several KV tiles (a tail
included), so the ordered dQ path is exercised. Each case keeps the target's
inputs and upstream gradient fixed and compares dQ / dK / dV bytes across
fixed-input replays under concurrent work, B1 / B3 / B129 batches with
duplicated and reordered copies, and a CUDA Graph replay after a neighbor
changes; an independent FP64 reference checks the values once.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays
from frost_test_utils import requires_bwd_engine_row, requires_dsl, select_engine
from test_sdpa_bwd_dsl_sm90 import ENGINE, _bhsd, _build, _ref

pytestmark = [pytest.mark.L0, requires_dsl, requires_bwd_engine_row(ENGINE)]

_H, _HK, _S = 4, 2, 200
_ROLES = ("q", "k", "v", "o", "do", "stats")


@pytest.fixture(autouse=True)
def _enable_frost(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _sequence(d, dtype, seed):
    """One batch element (B = 1): inputs, upstream gradient, and the forward's O / Stats."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        q, k, v, do = _bhsd(1, _H, _S, d, dtype), _bhsd(1, _HK, _S, d, dtype), _bhsd(1, _HK, _S, d, dtype), _bhsd(1, _H, _S, d, dtype)
    mask = dict(causal=True, bottom_right=False, window_left=None, window_right=None)
    o, stats, dq, dk, dv, _ = _ref(q, k, v, do, scale=d**-0.5, padding=None, sink=None, **mask)
    return dict(q=q, k=k, v=v, o=o, do=do, stats=stats), (dq, dk, dv)


def _batch(sequences, d, dtype):
    """Concatenate sequences along B in compact BSHD storage (Stats contiguous)."""

    def cat(role):
        if role == "stats":
            return torch.cat([s[role] for s in sequences], 0).contiguous()
        storage = torch.cat([s[role].transpose(1, 2) for s in sequences], 0).contiguous()
        return storage.transpose(1, 2)

    tensors = {r: cat(r) for r in _ROLES}
    b = len(sequences)
    grads = [_bhsd(b, n, _S, d, dtype, empty=True) for n in (_H, _HK, _HK)]
    return tensors, grads


def _graph(tensors, grads, d):
    g, pack, _ = _build(
        tuple(tensors[r] for r in _ROLES) + tuple(grads),
        scale=d**-0.5,
        causal=True,
        bottom_right=False,
        window_left=None,
        window_right=None,
        deterministic=True,
        lens=None,
        sink=None,
    )
    select_engine(g, ENGINE)
    g.check_support()
    g.build_plans()
    assert g.selected_engine is not None and g.selected_engine.name == ENGINE
    workspace = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    return g, pack, workspace


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128])
def test_sdpa_bwd_sm90_deterministic_replay_and_batch(d, dtype):
    target, refs = _sequence(d, dtype, seed=149)
    neighbor, _ = _sequence(d, dtype, seed=151)
    changed_neighbor, _ = _sequence(d, dtype, seed=153)
    expected = None
    for order in ([0], [0, 1, 0], [1, 0, 1], [0] * 129):
        tensors, grads = _batch([target if i == 0 else neighbor for i in order], d, dtype)
        g, pack, workspace = _graph(tensors, grads, d)

        def outputs(index):
            return {name: t[index] for name, t in zip(("dq", "dk", "dv"), grads)}

        def run():
            for t in grads:
                t.fill_(float("nan"))
            g.execute(pack, workspace)
            return outputs(order.index(0))

        got = assert_replays(run, repeats=3)
        if expected is None:
            expected = got
            tol = {"atol": 2e-2 if dtype == torch.float16 else 5e-2, "rtol": 5e-2}
            for name, ref in zip(("dq", "dk", "dv"), refs):
                torch.testing.assert_close(got[name].float(), ref[0].float(), **tol)
        for i, source in enumerate(order):
            if source == 0:
                assert_bitwise(outputs(i), expected, f"batch size {len(order)}, sequence {i}")
        if order == [0, 1, 0]:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                g.execute(pack, workspace)
            changed, _ = _batch([target, changed_neighbor, target], d, dtype)
            for r in _ROLES:
                tensors[r].copy_(changed[r])
            for t in grads:
                t.fill_(float("nan"))
            capture.replay()
            torch.cuda.synchronize()
            assert_bitwise(outputs(0), expected, "changed neighbor replay")
            assert_bitwise(outputs(2), expected, "duplicate after changed neighbor replay")
            capture.reset()
    assert_golden(f"sdpa-sm90-bwd-d{d}-gqa-causal-det-{dtype}", {r: target[r] for r in _ROLES}, expected)
