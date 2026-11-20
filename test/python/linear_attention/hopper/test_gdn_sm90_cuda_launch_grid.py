# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gdn_hopper_cuda`` admission at the launch-grid bound, in both directions.

The backward scan launches one grid row per (sequence, head), the backward's prep and
gradient kernels one per head, and the unsplit forward one per sequence. So
``check_support`` declines a graph whose sequences x heads exceeds 65535, forward and
backward alike, and admits it at 65535. Host only: the SM and the NVRTC probe are
stubbed, and the check stops at the first test after the grid bound.
"""

from types import SimpleNamespace

import pytest

from cudnn.linear_attention.hopper import gdn_cuda_engine

pytestmark = pytest.mark.L0


class _PastTheGridCheck(Exception):
    """Raised by the first check after the grid bound: the graph got past it."""


@pytest.fixture
def engine(monkeypatch):
    from cudnn.frost import buffers
    from cudnn.linear_attention.cake import compiler

    try:
        from cudnn.linear_attention.hopper import cuda_bwd_host, cuda_host  # noqa: F401 -- check_support imports them
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the sm90 host modules do not import here: {exc}")

    def past(*args, **kwargs):
        raise _PastTheGridCheck

    monkeypatch.setattr(buffers, "current_sm", lambda: gdn_cuda_engine.HOPPER_SM)
    monkeypatch.setattr(compiler, "cuda_include_dirs", lambda: [])
    monkeypatch.setattr(compiler, "nvrtc_version", lambda: (13, 4))
    monkeypatch.setattr(gdn_cuda_engine, "declared_layout_reason", past)
    return gdn_cuda_engine.GdnHopperCudaEngine()


def _graph(n_seq, heads, backward):
    """A one-node GDN graph whose facts pass every check before the grid bound."""
    facts = SimpleNamespace(
        op="GDN",
        invalid=None,
        is_bwd=backward,
        thd_layout=True,
        d_qk=gdn_cuda_engine.HEAD_DIM,
        d_v=gdn_cuda_engine.HEAD_DIM,
        h_q=heads,
        h_k=heads,
        h_v=heads,
        gate_channels=0,
        gates_at_ho=True,
        total_t=n_seq,
        n_seq=n_seq,
    )
    return SimpleNamespace(_facts_for=lambda analyzer: facts, nodes=(object(),))


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize(
    "n_seq, heads, admitted",
    [
        pytest.param(65535, 1, True, id="sequences-65535"),
        pytest.param(65536, 1, False, id="sequences-65536"),
        pytest.param(1, 65535, True, id="heads-65535"),
        pytest.param(1, 65536, False, id="heads-65536"),
        pytest.param(255, 257, True, id="pairs-65535"),
        pytest.param(256, 256, False, id="pairs-65536"),
    ],
)
def test_launch_grid_bound(engine, n_seq, heads, admitted, backward):
    graph = _graph(n_seq, heads, backward)
    if admitted:
        with pytest.raises(_PastTheGridCheck):
            engine.check_support(graph)
    else:
        with pytest.raises(NotImplementedError, match="exceeds a launch grid"):
            engine.check_support(graph)
