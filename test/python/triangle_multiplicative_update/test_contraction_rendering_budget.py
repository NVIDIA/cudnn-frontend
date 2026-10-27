# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K2 contraction's per-arch pipeline budget, GPU-free.

The dense GEMM template derives its A/B ring depth from the part's per-CTA SMEM budget and its accumulator layout from the part's TMEM
column count -- neither is stored in a tile config (``TileConfig``'s docstring), so ONE set of K2 tile configs renders a different pipeline
on each arch while the K loop (128-byte K tiles of 64 bf16, the per-tile accumulate order) stays the same code.  The two parts the
contraction has been rendered and measured on: a 227 KiB / 512-column part (compute capability 10.0: 6 stages, a non-exclusive
512-column allocation) and a 327 KiB / 576-column part (10.7: 9 stages, an exclusive 576-column allocation), the 256-column accumulator
region double-buffered on both.  This pins the derivation those constants come from -- the SMEM arithmetic once the template's fixed
reserve and the TMA-store staging buffer are off the top, and the TMEM table -- so a change to any of them surfaces here rather than as a
pipeline that no longer matches the body it was measured with."""

import pytest

import cudnn
import cudnn.gemm.frost  # noqa: F401  -- installs the pygraph recorder hook the analyzer reads
from cudnn.gemm.frost import tile_config
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import TEMPLATES
from cudnn.gemm.frost.sm100 import compiler as sm100_compiler
from cudnn.triangle_multiplicative_update.kernels import contraction as k2

pytestmark = [pytest.mark.L0]

K2_TILE_CONFIGS = (
    k2.DESIGN_TILE_CONFIG,
    k2.MULTICAST_A2_TILE_CONFIG,
    k2.MULTICAST_A4_TILE_CONFIG,
    k2.SMALL_N_TILE_CONFIG,
    k2.SMALL_N_MULTICAST_B2_TILE_CONFIG,
)
DENSE_TEMPLATE = next(t for t in TEMPLATES if t.file == "sm100_matmul.py")

# (arch, the per-CTA SMEM budget the driver reports for the part, A/B stages, TMEM columns, exclusive allocation).  227 KiB is the
# Blackwell opt-in limit and 327 KiB Rubin's oversized carveout; 512 / 576 are the two parts' TMEM column counts.
PARTS = [
    pytest.param(100, 227 * 1024, 6, 512, False, id="cc10.0-227KiB-512cols"),
    pytest.param(107, 327 * 1024, 9, 576, True, id="cc10.7-327KiB-576cols"),
]


@pytest.mark.parametrize("arch,smem_budget,stages,tmem_cols,exclusive", PARTS)
@pytest.mark.parametrize("config_name", K2_TILE_CONFIGS)
def test_k2_pipeline_depth_and_tmem_follow_the_part(monkeypatch, arch, smem_budget, stages, tmem_cols, exclusive, config_name):
    """One direction: the operand majors do not enter the ring-depth / TMEM arithmetic (``test_contraction_graph.py`` pins the majors)."""
    monkeypatch.setattr(tile_config, "_sm_smem_budget_bytes", lambda device=None: smem_budget)
    cfg = tile_config.by_name(config_name)
    chain = analyze(k2.contraction_graph(cudnn, 128, 512, "outgoing"))
    # Every K2 config stages A 16 KiB + B 16 KiB per K tile: 128 A rows + 128 B rows (B's N is halved per CTA under the pair MMA) x 128 B.
    per_stage = (cfg.cta_tile_m + cfg.cta_smem_tile_n) * cfg.cta_tile_k_bytes
    assert per_stage == 32 * 1024
    # The TMA-store staging buffer of the bf16 output: 2 slots x 32 rows x 256 B + 16, the `SMEM-D 16400B` both renderings read back.
    smem_d = sm100_compiler._smem_d_bytes(cfg, chain)
    assert smem_d == 2 * 32 * 256 + 16
    reserve = DENSE_TEMPLATE.smem_fixed_reserve
    assert cfg.max_ab_stages(smem_fixed_reserve=reserve, extra_smem_bytes=smem_d) == (smem_budget - reserve - smem_d) // per_stage == stages
    # TMEM: the part's column count decides the allocation mode and the accumulator depth; 256 columns per stage fit twice on both parts.
    total = sm100_compiler._tmem_cols_for_arch(arch)
    assert total == tmem_cols
    assert (total > sm100_compiler._MAX_NON_EXCLUSIVE_TMEM_COLS) is exclusive
    acc_cols = chain.num_gemms * cfg.mma_size_m * sm100_compiler._epi_tile_cols(cfg)
    assert acc_cols <= 256 and 2 * acc_cols <= total


def test_the_k2_budget_is_derived_not_stored(monkeypatch):
    """The pipeline depth is NOT a tile-config field: the same name renders a different depth under a different budget."""
    cfg = tile_config.by_name(k2.DESIGN_TILE_CONFIG)
    assert not hasattr(cfg, "ab_stages")
    depths = {}
    for budget in (227 * 1024, 327 * 1024):
        monkeypatch.setattr(tile_config, "_sm_smem_budget_bytes", lambda device=None, b=budget: b)
        depths[budget] = cfg.max_ab_stages(smem_fixed_reserve=DENSE_TEMPLATE.smem_fixed_reserve, extra_smem_bytes=0)
    assert depths == {227 * 1024: 7, 327 * 1024: 10}  # the pre-staging maxima both renderings' first `ab_stages` assignment reads back
