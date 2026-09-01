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


@pytest.mark.L0
@pytest.mark.parametrize("M", [384, 768, 1408])
def test_terminal_quant_strategy_can_select_one_cta(M):
    """The strategy override selects a runnable 1-CTA config without pinning a
    private tile or cluster identity."""
    config = select_config(M, 5120, 1, K=2048, block_scale=True, sm_count=148, force_cta_group=1)
    assert config.cta_group == 1
    assert by_name(config.name) is config


@pytest.mark.L0
def test_force_cta_group_rejects_invalid_or_unsupported_values():
    """Invalid overrides and a 2-CTA multi-GEMM request fail explicitly."""
    for value in (True, 0, 3):
        with pytest.raises(ValueError, match="force_cta_group"):
            select_config(384, 5120, 1, force_cta_group=value)
    with pytest.raises(NotImplementedError, match="multi-GEMM is 1ctamma-only"):
        select_config(384, 5120, 2, force_cta_group=2)


def _block_scaled_chain(*, M, N=5120, K=2048, batch=1, quantized=True, moe_groups=None):
    """Build the semantic IR consumed by the planner, without GPU storage."""
    from cudnn.gemm.frost.fusion_ir import (
        BlockQuantizeSpec,
        BlockScaleSpec,
        FusionChain,
        MatmulSpec,
        MoeSpec,
        OutputSpec,
        gemm_source,
    )

    if moe_groups is not None and batch != 1:
        raise ValueError("MoE analyzer IR always has batch=1")
    matmul = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=batch,
        a_batch=1,
        b_batch=batch,
        a_dtype="fp4_e2m1",
        b_dtype="fp4_e2m1",
    )
    block_scale = BlockScaleSpec(
        a_dtype="fp4_e2m1",
        b_dtype="fp4_e2m1",
        block_size_a=(1, 16),
        block_size_b=(16, 1),
        sf_dtype_a="fp8_e4m3",
        sf_dtype_b="fp8_e4m3",
        sfa_reorder="F8_128x4",
        sfb_reorder="F8_128x4",
        dequant_compute_a="fp32",
        dequant_compute_b="fp32",
        dequant_out_a="fp32",
        dequant_out_b="fp32",
    )
    source = gemm_source(0)
    quants = [BlockQuantizeSpec(source_ref=source, block_size=16, scale_dtype="fp8_e4m3")] if quantized else []
    output = OutputSpec(source_ref=source, dtype="fp4_e2m1", quant_idx=0) if quantized else OutputSpec(source_ref=source, dtype="bf16")
    moe = None if moe_groups is None else MoeSpec(num_experts=moe_groups, num_groups=moe_groups)
    return FusionChain(matmul=matmul, block_scale=block_scale, moe=moe, quants=quants, output_specs=[output])


@pytest.mark.L0
@pytest.mark.parametrize("M,expected_cta_group", [(384, 1), (1408, 1), (1409, 2), (4096, 2)])
def test_planner_bounds_terminal_quant_one_cta_to_measured_total_rows(monkeypatch, M, expected_cta_group):
    """The terminal-quant override stops at the measured total-row boundary."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    config = compiler.plan_config(_block_scaled_chain(M=M))
    assert config.cta_group == expected_cta_group


@pytest.mark.L0
def test_planner_uses_total_m_not_unobservable_moe_row_distribution(monkeypatch):
    """A low average cannot extend the override past total M; naturally M-starved
    grouped problems may still select 1-CTA through the ordinary heuristic."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    assert compiler.plan_config(_block_scaled_chain(M=2816, moe_groups=2)).cta_group == 2
    assert compiler.plan_config(_block_scaled_chain(M=2816, moe_groups=23)).cta_group == 1


@pytest.mark.L0
def test_planner_counts_every_dense_batch_row(monkeypatch):
    """A small per-batch M does not bypass the total declared-row boundary."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    assert compiler.plan_config(_block_scaled_chain(M=384, batch=3)).cta_group == 1
    assert compiler.plan_config(_block_scaled_chain(M=384, batch=4)).cta_group == 2


@pytest.mark.L0
def test_non_quantizing_graph_keeps_default_cta_policy(monkeypatch):
    """A block-scaled graph without a materialized quantizer is unaffected."""
    from cudnn.gemm.frost import compiler, kernel_registry

    monkeypatch.setattr(kernel_registry, "preferred_strategy", lambda _chain, config: config)
    config = compiler.plan_config(_block_scaled_chain(M=384, N=2048, K=5120, quantized=False))
    assert config.cta_group == 2


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
