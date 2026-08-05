# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline (no-GPU) checks for the analytic tile/cluster/scheduler selection.

There is ONE selection path for every graph type, so these cover the support constraints
that used to sit around the old N-bucket rule rather than a single branch of it.
Selection is pure geometry, so every invariant is checkable on CPU:
  * whatever it returns must be a real CATALOG entry;
  * it must never emit a geometry the kernel templates cannot run
    (2-CTA needs an even cluster M; block-scale needs 128-multiples);
  * it must never ask for the static scheduler when the caller has not confirmed a
    static template exists — mainloop-fusion and MoE graphs have no static variant;
  * multi-GEMM must stay inside the shared N budget, on cta_group=1 / "clc";
  * N-major B must get a per-CTA N extent that is a whole number of swizzle groups;
  * ``K`` is optional, and omitting it must still resolve to a runnable config.
"""

from __future__ import annotations

import pytest

from cudnn.gemm.frost.tile_config import by_name, select_config

MS = (1, 4, 16, 32, 64, 96, 128, 129, 256, 512, 1024, 4096)
NS = (32, 64, 128, 256, 512, 1024, 4096, 8192, 10240)
KS = (256, 1024, 2048, 4096, 8192)


@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("supports_static", [False, True])
def test_analytic_selection_is_always_runnable(block_scale, supports_static):
    for M in MS:
        for N in NS:
            for K in KS:
                cfg, cta_group, sched = select_config(M, N, 1, K=K, block_scale=block_scale, supports_static=supports_static)
                assert by_name(cfg.name) is cfg, f"{cfg.name} not in CATALOG"
                assert sched in ("clc", "static")
                if cta_group == 2:
                    assert cfg.cgrp_size_m % 2 == 0, f"2-CTA needs even cluster M: {cfg.name}"
                if block_scale:
                    assert cfg.cta_tile_m % 128 == 0 and cfg.cta_tile_n % 128 == 0
                    # block-scaled graphs stay on clc: static and clc are within noise of
                    # each other there and clc measured slightly better on both arches
                    assert sched == "clc"
                if not supports_static:
                    assert sched == "clc", "static requested where no static template exists"


def test_static_only_above_one_m_tile():
    """The static scheduler is selected for plain matmul above one M-tile, clc below."""
    for N in (4096, 8192):
        _, _, small = select_config(64, N, 1, K=4096, supports_static=True)
        _, _, large = select_config(1024, N, 1, K=4096, supports_static=True)
        assert small == "clc"
        assert large == "static"


def test_omitting_k_is_accepted():
    """K is optional for callers that do not have it to hand: the small-K bias goes
    neutral and everything else still resolves to a real CATALOG entry."""
    for M in MS:
        for N in NS:
            for num_gemms in (1, 2, 4):
                cfg, cta_group, sched = select_config(M, N, num_gemms)
                assert by_name(cfg.name) is cfg
                assert sched == "clc"
                if cta_group == 2:
                    assert cfg.cgrp_size_m % 2 == 0


def test_multi_gemm_budget_and_constraints():
    """One selection path, but the multi-GEMM support constraints still hold: the N
    tile is capped by the shared 256-wide budget, and only the 1ctamma CLC template
    implements multi-GEMM."""
    for ng in (2, 4, 8):
        cap = max(32, min(256, 256 // ng))
        for M in (64, 512, 4096):
            cfg, cta_group, sched = select_config(M, 8192, ng, K=4096, supports_static=True)
            assert cfg.cta_tile_n <= cap
            assert cta_group == 1, "multi-GEMM is 1ctamma-only"
            assert sched == "clc", "multi-GEMM is CLC-only"


def test_n_major_b_lifts_the_n_tile():
    """N-major B is TMA-loaded a swizzle group at a time, so the per-CTA N extent must
    be a whole number of groups -- under the analytic tile choice too."""
    for eb in (1, 2):
        group_elems = 128 // eb
        for M in (64, 512):
            cfg, cta_group, _ = select_config(M, 8192, 1, K=4096, b_n_major=True, b_elem_bytes=eb)
            assert cfg.cta_tile_n % (group_elems * cta_group) == 0


def test_moe_and_mainloop_never_get_static():
    """MoE and mainloop-fusion graphs have no static template. The caller signals that
    with supports_static=False; selection must honour it."""
    for M in (64, 512, 4096):
        for N in (1024, 8192):
            _, _, sched = select_config(M, N, 1, K=4096, supports_static=False)
            assert sched == "clc"
