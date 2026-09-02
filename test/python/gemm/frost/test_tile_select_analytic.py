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

import pytest

from cudnn.gemm.frost.tile_config import by_name, select_config

MS = (1, 4, 16, 32, 64, 96, 128, 129, 256, 512, 1024, 4096)
NS = (32, 64, 128, 256, 512, 1024, 4096, 8192, 10240)
KS = (256, 1024, 2048, 4096, 8192)


@pytest.mark.parametrize("block_scale", [False, True])
def test_analytic_selection_is_always_runnable(block_scale):
    for M in MS:
        for N in NS:
            for K in KS:
                cfg, cta_group = select_config(M, N, 1, K=K, block_scale=block_scale)
                assert by_name(cfg.name) is cfg, f"{cfg.name} not in CATALOG"
                if cta_group == 2:
                    assert cfg.cgrp_size_m % 2 == 0, f"2-CTA needs even cluster M: {cfg.name}"
                if block_scale:
                    assert cfg.cta_tile_m % 128 == 0 and cfg.cta_tile_n % 128 == 0


def test_omitting_k_is_accepted():
    """K is optional for callers that do not have it to hand: the small-K bias goes
    neutral and everything else still resolves to a real CATALOG entry."""
    for M in MS:
        for N in NS:
            for num_gemms in (1, 2, 4):
                cfg, cta_group = select_config(M, N, num_gemms)
                assert by_name(cfg.name) is cfg
                if cta_group == 2:
                    assert cfg.cgrp_size_m % 2 == 0


def test_multi_gemm_budget_and_constraints():
    """One selection path, but the multi-GEMM support constraints still hold: the N
    tile is capped by the shared 256-wide budget, and only the 1ctamma template
    implements multi-GEMM."""
    for ng in (2, 4, 8):
        cap = max(32, min(256, 256 // ng))
        for M in (64, 512, 4096):
            cfg, cta_group = select_config(M, 8192, ng, K=4096)
            assert cfg.cta_tile_n <= cap
            assert cta_group == 1, "multi-GEMM is 1ctamma-only"


def test_n_major_b_lifts_the_n_tile():
    """N-major B is TMA-loaded a swizzle group at a time, so the per-CTA N extent must
    be a whole number of groups -- under the analytic tile choice too."""
    for eb in (1, 2):
        group_elems = 128 // eb
        for M in (64, 512):
            cfg, cta_group = select_config(M, 8192, 1, K=4096, b_n_major=True, b_elem_bytes=eb)
            assert cfg.cta_tile_n % (group_elems * cta_group) == 0


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
