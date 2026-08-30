# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline (no-GPU) checks for the analytic tile/cluster selection.

There is ONE selection path for every graph type, so these cover the support constraints
that used to sit around the old N-bucket rule rather than a single branch of it.
Selection is pure geometry, so every invariant is checkable on CPU:
  * whatever it returns must be a real CATALOG entry;
  * it must never emit a geometry the kernel templates cannot run
    (2-CTA needs an even cluster M; block-scale needs 128-multiples);
  * multi-GEMM must stay inside the shared N budget, on cta_group=1;
  * N-major B must get a per-CTA N extent that is a whole number of swizzle groups;
  * ``K`` is optional, and omitting it must still resolve to a runnable config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from cudnn.gemm.frost.tile_config import by_name, select_config

pytestmark = pytest.mark.L0

MS = (1, 4, 16, 32, 64, 96, 128, 129, 256, 512, 1024, 4096)
NS = (32, 64, 128, 256, 512, 1024, 4096, 8192, 10240)
KS = (256, 1024, 2048, 4096, 8192)


@pytest.mark.parametrize("block_scale", [False, True])
def test_analytic_selection_is_always_runnable(block_scale):
    for M in MS:
        for N in NS:
            for K in KS:
                cfg = select_config(M, N, 1, K=K, block_scale=block_scale)
                assert by_name(cfg.name) is cfg, f"{cfg.name} not in CATALOG"
                if cfg.cta_group == 2:
                    assert cfg.cga_size_m % 2 == 0, f"2-CTA needs even cluster M: {cfg.name}"
                if block_scale:
                    assert cfg.cta_tile_m % 128 == 0 and cfg.cta_tile_n % 128 == 0


def test_omitting_k_is_accepted():
    """K is optional for callers that do not have it to hand: the small-K bias goes
    neutral and everything else still resolves to a real CATALOG entry."""
    for M in MS:
        for N in NS:
            for num_gemms in (1, 2, 4):
                cfg = select_config(M, N, num_gemms)
                assert by_name(cfg.name) is cfg
                if cfg.cta_group == 2:
                    assert cfg.cga_size_m % 2 == 0


def test_multi_gemm_budget_and_constraints():
    """One selection path, but the multi-GEMM support constraints still hold: the N
    tile is capped by the shared 256-wide budget, and only the 1ctamma template
    implements multi-GEMM."""
    for ng in (2, 4, 8):
        cap = max(32, min(256, 256 // ng))
        for M in (64, 512, 4096):
            cfg = select_config(M, 8192, ng, K=4096)
            assert cfg.cta_tile_n <= cap
            assert cfg.cta_group == 1, "multi-GEMM is 1ctamma-only"


@pytest.mark.parametrize("M", [384, 768, 1408])
def test_terminal_quant_strategy_forces_one_cta_and_rescores_cluster(M):
    """Ultra-like terminal-quant up projections use 1-CTA; the matching down
    geometry keeps the default 2-CTA policy.  The cluster remains a scorer result."""
    from cudnn.gemm.frost import tile_config as tc

    sm_count = 148
    up = select_config(
        M,
        5120,
        1,
        K=2048,
        block_scale=True,
        sm_count=sm_count,
        force_cta_group=1,
    )
    down = select_config(M, 2048, 1, K=5120, block_scale=True, sm_count=sm_count)
    assert up.cta_group == 1
    assert down.cta_group == 2

    pool = [cluster for cluster in tc._CLUSTERS_1D + tc._CLUSTERS_2D if not tc._hang_prone(up.cta_tile_m, up.cta_tile_n, *cluster)]
    expected = max(
        pool,
        key=lambda cluster: tc._cluster_score(
            M,
            5120,
            up.cta_tile_m,
            up.cta_tile_n,
            1,
            *cluster,
            sm_count,
        ),
    )
    assert (up.cga_size_m, up.cga_size_n) == expected


def test_force_cta_group_rejects_invalid_or_unsupported_values():
    for value in (True, 0, 3):
        with pytest.raises(ValueError, match="force_cta_group"):
            select_config(384, 5120, 1, force_cta_group=value)
    with pytest.raises(NotImplementedError, match="multi-GEMM is 1ctamma-only"):
        select_config(384, 5120, 2, force_cta_group=2)


@pytest.mark.parametrize("M", [384, 768, 1408])
def test_planner_applies_one_cta_to_dense_and_grouped_terminal_quant_chain(monkeypatch, M):
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)

    def chain(*, N, K, quants, moe_groups=None):
        return SimpleNamespace(
            matmul=SimpleNamespace(M=M * (moe_groups or 1), N=N, K=K, b_major="k", b_dtype="fp4_e2m1"),
            moe=None if moe_groups is None else SimpleNamespace(num_groups=moe_groups),
            num_gemms=1,
            has_block_scale=True,
            has_moe=moe_groups is not None,
            quants=quants,
        )

    up = compiler.plan_config(chain(N=5120, K=2048, quants=[object()]))
    down = compiler.plan_config(chain(N=2048, K=5120, quants=[]))
    moe_quant = compiler.plan_config(chain(N=5120, K=2048, quants=[object()], moe_groups=64))
    assert up.cta_group == 1
    assert down.cta_group == 2
    assert moe_quant.cta_group == 1


def test_n_major_b_lifts_the_n_tile():
    """N-major B is TMA-loaded a swizzle group at a time, so the per-CTA N extent must
    be a whole number of groups -- under the analytic tile choice too."""
    for eb in (1, 2):
        group_elems = 128 // eb
        for M in (64, 512):
            cfg = select_config(M, 8192, 1, K=4096, b_n_major=True, b_elem_bytes=eb)
            assert cfg.cta_tile_n % (group_elems * cfg.cta_group) == 0


def test_a_new_pipeline_must_register_its_hardware_facts():
    """A family that registers a config class but forgets a per-pipeline table
    must raise, not inherit another family's value: the tables are hardware
    facts, and a wrong MMA-inst K renders a descriptor that is silently wrong."""
    import dataclasses

    from cudnn.gemm.frost import tile_config as tc

    @dataclasses.dataclass(frozen=True)
    class ConfigSmFake(tc.TileConfig):
        pass

    tc._CONFIG_CLASS_BY_PIPELINE["sm_fake"] = ConfigSmFake
    try:
        with pytest.raises(NotImplementedError, match="MMA-inst K width not known for pipeline"):
            tc.as_pipeline(tc.DEFAULT_CONFIG, "sm_fake")
    finally:
        del tc._CONFIG_CLASS_BY_PIPELINE["sm_fake"]
