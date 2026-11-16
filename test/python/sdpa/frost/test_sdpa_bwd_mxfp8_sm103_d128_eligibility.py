# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm103_d128_mxfp8`` without an SM103: the row's graph declaration bounds and its launch plan.

The device is faked (compute capability, SM count, DSL version), so these run on any CUDA machine: the row's admission
is its ``mismatch()`` on the analyzed graph, and the launch plan is host arithmetic over cluster capacities.
"""

from __future__ import annotations

import math

import pytest

import cudnn
import test_sdpa_bwd_mxfp8_sm103_d128 as row

pytestmark = pytest.mark.L0

# B300: 74 two-CTA and 33 four-CTA clusters of a one-CTA-per-SM kernel (cudnn.frost.occupancy on cc 10.3)
_B300_CLUSTERS = {2: 74, 4: 33}


@pytest.fixture(autouse=True)
def _sm103(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 3))
    monkeypatch.setattr(ga, "_device_sm_count", lambda: 148)
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")))
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)


def _decline_reason(b=1, hq=8, hkv=2, s=1024, scale=1 / math.sqrt(128), **kw):
    """Why the row declines the graph, or None if it would serve it (its own ``mismatch()``: the plan list also holds
    backend plans)."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    spec = next(sp for sp in ENGINE_SPECS if sp.name == row._ENGINE)
    try:
        g, _t, _outs = row._build_graph(b, hq, hkv, s, scale=scale, **kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts, None)


# --------------------------------------------------------------------------- #
# admission                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(hq=8, hkv=8),
        dict(hq=8, hkv=1),
        dict(s=100),
        dict(s=1000),
        dict(b=8, hq=4, hkv=2, s=512),
        dict(use_causal_mask=True),
        dict(s=1000, use_causal_mask=True),
        dict(use_causal_mask_bottom_right=True),
        dict(s=1000, use_causal_mask_bottom_right=True),
        dict(declare_amax=True),
        dict(declare_amax=("dQ",)),
        dict(use_deterministic_algorithm=True),
        dict(scale=2.0**12),
    ],
    ids=[
        "gqa",
        "mha",
        "mqa",
        "s100",
        "s1000",
        "b8",
        "causal",
        "causal-tail",
        "causal-bottom-right",
        "causal-bottom-right-tail",
        "amax",
        "amax-dq",
        "deterministic",
        "scale-max",
    ],
)
def test_admits(kw):
    assert _decline_reason(**kw) is None


@pytest.mark.parametrize(
    "kw",
    [
        dict(stride_fn=row._bshd_stride),
        dict(sq=512),
        dict(sq=512, use_causal_mask_bottom_right=True),
        dict(use_causal_mask=True, left_bound=64),
        dict(right_bound=16),
        dict(scale=2.0**12 * 2),
        dict(scale=0.0),
        dict(scale=-0.125),
        dict(scale=float("inf")),
        dict(b=8, hq=128, hkv=8, s=2**21),  # B * H_q * S = 2^31 rows: past the kernels' int32 row index
    ],
    ids=["bshd", "sq-ne-skv", "bottom-right-sq-ne-skv", "window", "band", "scale-too-big", "scale-zero", "scale-negative", "scale-inf", "int32-rows"],
)
def test_declines(kw):
    assert _decline_reason(**kw) is not None


def test_declines_strided_stats():
    """The kernels read Stats as a contiguous (B, H_q, S_q, 1) fp32 array, and the row does not claim strided_stats,
    so the shared gate declines any other Stats layout (here head-minor, the BSHD order) instead of reading it wrong.
    The seven scale tensors need no gate here: the frontend itself rejects an MXFP8 backward graph whose descale
    tensors are not F8_128x4 (include/cudnn_frontend/node/sdpa_fp8_bwd.h)."""
    reason = _decline_reason(stats_stride=[8 * 1024, 1, 8, 1])
    assert reason is not None and "stats must be contiguous" in reason, reason


@pytest.mark.parametrize("cc", [(10, 0), (12, 0), (10, 7)])
def test_declines_other_devices(monkeypatch, cc):
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    assert _decline_reason() is not None


@pytest.mark.parametrize("version", ["4.7.0", "4.8.0"])
def test_declines_below_the_dsl_floor(monkeypatch, version):
    """Rule 7: the row's own floor (nvidia-cutlass-dsl 4.8.1, above the library's) is checked at eligibility, not at build."""
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    reason = _decline_reason()
    assert reason is not None and "4.8.1" in reason


# --------------------------------------------------------------------------- #
# launch plan (host arithmetic, no device)                                    #
# --------------------------------------------------------------------------- #


def _plan(s, b, hq, hkv, causal=False, batch_invariant=True, **kw):
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_sm103 as G

    g = G.geometry(s, b, hq, hkv, 1 / math.sqrt(128), False, causal)
    return G.plan(g, _B300_CLUSTERS.__getitem__, batch_invariant=batch_invariant, **kw)


@pytest.mark.parametrize(
    "shape,batch_invariant,expected",
    [
        # (S, B, Hq, Hkv, causal) -> (walk split, cluster, dK/dV grid, dQ clusters), with B300's cluster capacities
        ((1024, 1, 88, 8, False), True, (2, 4, 128, 74)),
        ((1024, 8, 88, 8, False), True, (2, 4, 128, 74)),
        ((1024, 8, 88, 8, False), False, (1, 4, 128, 74)),
        ((16384, 8, 88, 8, False), True, (1, 4, 132, 74)),
        ((2048, 1, 88, 1, False), True, (2, 4, 32, 74)),
        ((4096, 1, 8, 1, False), True, (2, 4, 64, 74)),
        ((1000, 1, 8, 2, True), True, (2, 4, 32, 32)),
        ((384, 1, 4, 4, True), True, (1, 4, 16, 8)),
        ((100, 1, 8, 2, False), True, (2, 4, 8, 8)),
        ((512, 8, 4, 2, False), True, (2, 4, 128, 64)),
    ],
)
def test_launch_plan(shape, batch_invariant, expected):
    p = _plan(*shape, batch_invariant=batch_invariant)
    assert (p.split, p.kv_cluster, p.kv_grid, p.dq_clusters) == expected
    assert p.kv_grid % p.kv_cluster == 0 and p.kv_grid <= p.kv_cap


def test_deterministic_plan_is_per_batch_element():
    """The walk split is the plan's only rounding-relevant choice: under use_deterministic_algorithm it is decided from
    one batch element, so every B gets the B = 1 decision; otherwise a large batch keeps the walk unsplit."""
    for b in (1, 2, 8, 64):
        assert _plan(1024, b, 4, 2, batch_invariant=True).split == _plan(1024, 1, 4, 2).split == 2
    assert _plan(1024, 8, 4, 2, batch_invariant=False).split == 1


def test_forced_split_must_be_built():
    from cudnn.sdpa.bwd.kernels.sm100 import _geometry_sm103 as G

    g = G.geometry(1152, 1, 8, 8, 1 / math.sqrt(128), False, True)  # causal MHA, 9 query blocks (odd): no walk split
    assert not g.has_split
    with pytest.raises(ValueError):
        G.plan(g, _B300_CLUSTERS.__getitem__, split=2)
