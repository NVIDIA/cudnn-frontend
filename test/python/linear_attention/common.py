# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the GDN (Gated DeltaNet) test suite.

Availability guards, run dispatcher, and comparison tolerances shared by the
fprop and bprop test files.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()

try:
    import cuda.tile  # noqa: F401 — the engines' own availability gate

    from cudnn.linear_attention.ops import gated_delta_net, kimi_delta_attention

    _HAS_CUTILE = True
except ImportError:
    _HAS_CUTILE = False

GDN_MARKS = [
    pytest.mark.L0,
    pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA"),
    pytest.mark.skipif(not _HAS_CUTILE, reason="needs the cuda.tile runtime"),
]

KDA_MARKS = GDN_MARKS  # same gate: both ops are lazy exports of the same package

FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}

# (H, HV) pairs: H = q/k heads, HV = v/g/beta heads (GVA when HV > H).
HEAD_CONFIGS = [(1, 1), (3, 3), (1, 2), (2, 4), (16, 32), (16, 64)]
HEAD_CONFIGS_SMALL = [(1, 1), (2, 4)]


def assert_engine_declines(graph, engine_name: str) -> None:
    """``engine_name`` must not serve ``graph``.

    Asserted against the ranked plan list rather than through a failing
    ``build()``: a decline only advances the walk, so a sibling engine
    (cuTile vs FROST) may still serve the graph — and does wherever both are
    installed, which is why the build-fails form passes on a box missing one
    of them and fails in CI."""
    try:
        graph.create_execution_plans()
    except Exception:
        return  # nothing claimed it at all, which is a stronger decline
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert not any(n.startswith(engine_name) for n in names), f"{engine_name} claimed a graph it must decline; plans={names}"


def run_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Run GDN through the public custom op (the op is THD-only: dense
    ``[B, T, ...]`` inputs are flattened with ``cu_seqlens = [0, T, 2T, ...]``;
    packed ``[1, total, ...]`` inputs are squeezed to the op's rank-3 THD
    layout). Returns ``(o, final_state)`` with ``final_state`` normalized to
    ``None`` when not requested."""
    g = g.float()  # the op requires kernel-native fp32 gates; callers convert
    beta = beta.float()
    if cu_seqlens is None:
        B, T = q.shape[0], q.shape[1]
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * T
        o, fs = gated_delta_net(
            q.reshape(B * T, *q.shape[2:]),
            k.reshape(B * T, *k.shape[2:]),
            v.reshape(B * T, *v.shape[2:]),
            g.reshape(B * T, *g.shape[2:]),
            beta.reshape(B * T, *beta.shape[2:]),
            cu,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        o = o.reshape(B, T, *o.shape[1:])
    else:
        o, fs = gated_delta_net(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            g.squeeze(0),
            beta.squeeze(0),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        o = o.unsqueeze(0)
    if fs is None or fs.numel() == 0:
        fs = None
    return o, fs


def run_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Run KDA through the public custom op (the op is THD-only: dense
    ``[B, T, ...]`` inputs are flattened with ``cu_seqlens = [0, T, 2T, ...]``).
    ``g`` is the per-key-channel log decay ([..., HV, K]); ``beta`` is scalar
    ([..., HV]). Returns ``(o, final_state)`` with ``final_state`` normalized
    to ``None`` when not requested."""
    g = g.float()  # the op requires kernel-native fp32 gates; callers convert
    beta = beta.float()
    if cu_seqlens is None:
        B, T = q.shape[0], q.shape[1]
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * T
        o, fs = kimi_delta_attention(
            q.reshape(B * T, *q.shape[2:]),
            k.reshape(B * T, *k.shape[2:]),
            v.reshape(B * T, *v.shape[2:]),
            g.reshape(B * T, *g.shape[2:]),
            beta.reshape(B * T, *beta.shape[2:]),
            cu,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        o = o.reshape(B, T, *o.shape[1:])
    else:
        o, fs = kimi_delta_attention(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            g.squeeze(0),
            beta.squeeze(0),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
        o = o.unsqueeze(0)
    if fs is None or fs.numel() == 0:
        fs = None
    return o, fs


DETERMINISM_REPEATS = int(os.environ.get("DETERMINISM_REPEATS", "8"))


def bitwise_bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def assert_bitwise_runs(launch, repeats=DETERMINISM_REPEATS, label=""):
    """``launch()`` returns a tuple of freshly-written output tensors.  Launch
    ``repeats`` times back to back (single sync at the end) and require every
    run to match run 0 bit for bit (barrier/fence races are timing-dependent;
    any mismatching bit is a failure, there is no tolerance)."""
    runs = [launch() for _ in range(repeats)]
    torch.cuda.synchronize()
    for out in runs[0]:
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite output in run 0"
    for r, outs in enumerate(runs[1:], start=1):
        for i, (a, b) in enumerate(zip(runs[0], outs)):
            assert torch.equal(bitwise_bits(a), bitwise_bits(b)), f"{label}: output {i} differs between run 0 and run {r}"


def assert_concurrent_stream_runs(launch_a, launch_b, s1, s2, repeats=DETERMINISM_REPEATS):
    """Two concurrent kernel instances on separate streams must not perturb
    each other: every repeat must match its own single-stream baseline."""
    with torch.cuda.stream(s1):
        base_a = launch_a()
    torch.cuda.synchronize()
    with torch.cuda.stream(s2):
        base_b = launch_b()
    torch.cuda.synchronize()
    for r in range(repeats):
        with torch.cuda.stream(s1):
            out_a = launch_a()
        with torch.cuda.stream(s2):
            out_b = launch_b()
        torch.cuda.synchronize()
        for label, base, outs in (("A", base_a, out_a), ("B", base_b, out_b)):
            for i, (x, y) in enumerate(zip(base, outs)):
                assert torch.equal(bitwise_bits(x), bitwise_bits(y)), f"stream {label} output {i} differs on concurrent run {r}"
