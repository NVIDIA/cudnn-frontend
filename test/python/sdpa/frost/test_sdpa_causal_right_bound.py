# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conflicting band-mask spellings: a causal flag with an explicit right bound,
both causal flags, or a sliding window with an explicit left bound.

The Python binding rejects these combinations for every SDPA entry point except
sdpa_backward, which applies them in order so the explicit left / right bound
wins. The FROST analyzer must agree with both, instead of silently picking one.
"""

from __future__ import annotations

import math

import cudnn
import pytest
import torch

from cudnn.sdpa import graph_analyzer as ga
from frost_test_utils import requires_dsl, select_engine

B, H, S, D = 2, 4, 256, 64
DTYPE = cudnn.data_type.HALF

_SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an SM80 (A100) GPU",
)


def _strides(h, s, d):
    return (s * h * d, d, h * d, 1)


def _graph():
    return cudnn.pygraph(io_data_type=DTYPE, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)


def _fwd_graph(d=D, **mask):
    g = _graph()
    dims, strides = (B, H, S, d), _strides(H, S, d)
    q, k, v = (g.tensor(dim=dims, stride=strides, data_type=DTYPE, name=n) for n in ("q", "k", "v"))
    o, stats = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1 / math.sqrt(d), generate_stats=True, **mask)
    o.set_output(True).set_dim(dims).set_stride(strides).set_data_type(DTYPE)
    stats.set_output(True).set_dim((B, H, S, 1)).set_stride((H * S, S, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    return g, (q, k, v, o, stats)


def _bwd_graph(d=D, **mask):
    g = _graph()
    dims, strides = (B, H, S, d), _strides(H, S, d)
    q, k, v, o, do = (g.tensor(dim=dims, stride=strides, data_type=DTYPE, name=n) for n in ("q", "k", "v", "o", "dO"))
    stats = g.tensor(dim=(B, H, S, 1), stride=(H * S, S, 1, 1), data_type=cudnn.data_type.FLOAT, name="stats")
    dq, dk, dv = g.sdpa_backward(name="sb", q=q, k=k, v=v, o=o, dO=do, stats=stats, attn_scale=1 / math.sqrt(d), **mask)
    for t in (dq, dk, dv):
        t.set_output(True).set_dim(dims).set_stride(strides).set_data_type(DTYPE)
    return g, (q, k, v, o, do, stats, dq, dk, dv)


def _fp8_fwd_graph(**mask):
    g = _graph()
    dims, strides = (B, H, S, 128), _strides(H, S, 128)
    q, k, v = (g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name=n) for n in ("q", "k", "v"))
    scales = {
        name: g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name=name)
        for name in ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
    }
    o, _, _, _ = g.sdpa_fp8(q=q, k=k, v=v, attn_scale=0.1, generate_stats=False, **scales, **mask)
    o.set_output(True).set_dim(dims).set_stride(strides).set_data_type(DTYPE)
    return g, None


# ---------------------------------------------------------------------------
# Analyzer: GPU-free
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask, message",
    [
        (dict(use_causal_mask=True, diagonal_band_right_bound=64), "use_causal_mask and diagonal_band_right_bound cannot be set at the same time"),
        (dict(use_causal_mask=True, diagonal_band_right_bound=0), "use_causal_mask and diagonal_band_right_bound cannot be set at the same time"),
        (
            dict(use_causal_mask_bottom_right=True, diagonal_band_right_bound=64),
            "use_causal_mask_bottom_right and diagonal_band_right_bound cannot be set at the same time",
        ),
        (dict(use_causal_mask=True, use_causal_mask_bottom_right=True), "use_causal_mask and use_causal_mask_bottom_right cannot both be true"),
        (dict(sliding_window_length=32, diagonal_band_left_bound=16), "sliding window and left_bound cannot be set at the same time"),
    ],
    ids=["causal_r64", "causal_r0", "causal_br_r64", "both_causal_flags", "window_and_left"],
)
def test_forward_rejects_what_the_binding_rejects(mask, message):
    g, _ = _fwd_graph(**mask)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is not None, "a combination the binding rejects must not reach a FROST engine"
    assert message in facts.invalid


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask, message",
    [
        (dict(use_causal_mask=True, right_bound=64), "use_causal_mask and diagonal_band_right_bound cannot be set at the same time"),
        (dict(sliding_window=32, left_bound=16), "sliding window and left_bound cannot be set at the same time"),
    ],
    ids=["causal_r64", "window_and_left"],
)
def test_fp8_forward_rejects_its_own_spellings(mask, message):
    g, _ = _fp8_fwd_graph(**mask)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is not None, "a combination the binding rejects must not reach a FROST engine"
    assert message in facts.invalid


@pytest.mark.L0
def test_backward_explicit_left_bound_wins_over_the_window():
    g, _ = _bwd_graph(sliding_window_length=32, diagonal_band_left_bound=16)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, facts.invalid if facts else None
    assert facts.window_left == 15  # diagonal_band_left_bound=16, applied after the window as in the binding


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask, right_bound, bottom_right",
    [
        (dict(use_causal_mask=True, diagonal_band_right_bound=64), 64, False),
        (dict(use_causal_mask_bottom_right=True, diagonal_band_right_bound=64), 64, True),
        (dict(use_causal_mask=True, diagonal_band_right_bound=0), 0, False),
    ],
    ids=["causal_r64", "causal_br_r64", "causal_r0"],
)
def test_backward_explicit_right_bound_wins_as_in_the_binding(mask, right_bound, bottom_right):
    g, _ = _bwd_graph(**mask)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, facts.invalid if facts else None
    assert facts.right_bound == right_bound
    assert facts.right_band_widening == (right_bound > 0)
    assert facts.causal == (right_bound == 0)
    assert facts.bottom_right == bottom_right


@pytest.mark.L0
@pytest.mark.parametrize("build", [_fwd_graph, _bwd_graph], ids=["fwd", "bwd"])
@pytest.mark.parametrize(
    "mask, right_bound, window_left",
    [
        (dict(use_causal_mask=True), 0, None),
        (dict(diagonal_band_right_bound=64), 64, None),
        (dict(), None, None),
        (dict(use_causal_mask=True, sliding_window_length=32), 0, 31),
        (dict(use_causal_mask=True, diagonal_band_left_bound=16), 0, 15),
    ],
    ids=["causal", "right_only", "unmasked", "causal_window", "causal_left"],
)
def test_single_spellings_are_unchanged(build, mask, right_bound, window_left):
    g, _ = build(**mask)
    facts = ga.analyze(g)
    assert facts is not None and facts.invalid is None, facts.invalid if facts else None
    assert facts.right_bound == right_bound
    assert facts.window_left == window_left


# ---------------------------------------------------------------------------
# SM80 FROST rows on an A100
# ---------------------------------------------------------------------------


def _reference(q, k, v, right_bound, *, need_grad=False):
    """FP64 attention over (B, H, S, D) views with a top-left band ``j <= i + right_bound``."""
    q64, k64, v64 = (t.double().requires_grad_(need_grad) for t in (q, k, v))
    scores = q64 @ k64.transpose(-1, -2) / math.sqrt(q.shape[-1])
    i = torch.arange(S, device=q.device).view(S, 1)
    j = torch.arange(S, device=q.device).view(1, S)
    if right_bound is not None:
        scores = scores.masked_fill(j > i + right_bound, float("-inf"))
    lse = scores.logsumexp(-1)
    out = (scores - lse[..., None]).exp() @ v64
    return out, lse, (q64, k64, v64)


def _bshd(d, seed):
    torch.manual_seed(seed)
    return torch.randn(B, S, H, d, device="cuda", dtype=torch.float16).transpose(1, 2)


def _run(graph, engine, pack):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, engine)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute(pack, workspace)
    torch.cuda.synchronize()


@pytest.mark.L0
@_SM80
@requires_dsl
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("right_bound", [16, 64])
def test_sm80_forward_serves_right_band_widening(d, right_bound):
    q, k, v = (_bshd(d, seed) for seed in (1, 2, 3))
    o = torch.empty_like(q)
    stats = torch.empty(B, H, S, 1, device="cuda", dtype=torch.float32)
    g, (qt, kt, vt, ot, st) = _fwd_graph(d=d, diagonal_band_right_bound=right_bound)
    _run(g, "sdpa_fwd_prefill_sm80", {qt: q, kt: k, vt: v, ot: o, st: stats})
    ref_o, ref_lse, _ = _reference(q, k, v, right_bound)
    torch.testing.assert_close(o.double(), ref_o, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(stats.squeeze(-1).double(), ref_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@_SM80
@requires_dsl
@pytest.mark.parametrize("d", [64, 128])
def test_sm80_backward_causal_flag_with_right_bound_uses_the_bound(d):
    """sdpa_backward(use_causal_mask=True, diagonal_band_right_bound=R) is the R band,
    as the binding builds it; a FROST row that forced R to 0 would return causal gradients."""
    right_bound = 64
    q, k, v, do = (_bshd(d, seed) for seed in (4, 5, 6, 7))
    ref_o, ref_lse, (q64, k64, v64) = _reference(q, k, v, right_bound, need_grad=True)
    ref_o.backward(do.double())
    o = torch.empty_like(q).copy_(ref_o.detach())  # keeps the BSHD strides the graph declares
    stats = ref_lse.detach().float().unsqueeze(-1).contiguous()
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    g, (qt, kt, vt, ot, dot, st, dqt, dkt, dvt) = _bwd_graph(d=d, use_causal_mask=True, diagonal_band_right_bound=right_bound)
    _run(g, "sdpa_bwd_sm80", {qt: q, kt: k, vt: v, ot: o, dot: do, st: stats, dqt: dq, dkt: dk, dvt: dv})
    for got, ref in ((dq, q64.grad), (dk, k64.grad), (dv, v64.grad)):
        torch.testing.assert_close(got.double(), ref, atol=2e-2, rtol=2e-2)
