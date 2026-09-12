# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``frost_gemm`` plans carry their tile config as public knobs.

The family heuristics list the automatic strategy WITH its TileConfig spelled
in the shared ``cudnn.knob_type`` vocabulary, so ``(engine_id, knobs)`` pins
the exact kernel and replays on a fresh graph; a record that names no canonical
config is declined, never snapped. The frost slot is opt-in (conftest)."""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import requires_sm100

import cudnn
from cudnn.engines import is_backend_engine, is_python_engine
from cudnn.gemm.frost.knobs import GemmKnobs
from cudnn.gemm.frost.tile_config import CATALOG, DEFAULT_CONFIG, by_name

pytestmark = pytest.mark.L0

M, N, K = 256, 512, 256


# ---------------------------------------------------------------------------
# Pure: knobs <-> TileConfig <-> public dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [DEFAULT_CONFIG] + list(CATALOG[::97]), ids=lambda c: c.name)
def test_knobs_name_every_catalog_config_exactly(cfg):
    knobs = GemmKnobs.from_config(cfg)
    assert knobs.config_name == cfg.name
    assert knobs.to_config() == cfg
    public = knobs.to_public()
    assert all(isinstance(k, cudnn.knob_type) and isinstance(v, int) for k, v in public.items())
    assert GemmKnobs.from_public(public) == knobs
    assert GemmKnobs.from_public({int(k): v for k, v in public.items()}) == knobs  # a JSON cache's int keys


def test_knobs_speak_the_shared_vocabulary():
    kt = cudnn.knob_type
    knobs = GemmKnobs.from_config(by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"))
    public = knobs.to_public()
    # backend types where the meaning matches ...
    assert public[kt.TILE_M] == 128 and public[kt.TILE_N] == 256 and public[kt.TILEK] == 128
    assert public[kt.TILE_CGA_M] == 2 and public[kt.TILE_CGA_N] == 1
    assert public[kt.SPLIT_K_SLC] == 1 and public[kt.SWAP_AB] == 0
    for knob in (kt.TILE_M, kt.TILE_N, kt.TILEK, kt.TILE_CGA_M, kt.TILE_CGA_N, kt.SPLIT_K_SLC, kt.SWAP_AB):
        assert not cudnn.is_frontend_knob_type(knob)
    # ... frontend-only ones for the axes the backend has no word for
    assert public[kt.PIPELINE_ARCH] == 100 and public[kt.CTA_GROUP] == 2
    assert public[kt.MMA_TILE_M] == 128 and public[kt.MMA_TILE_N] == 256 and public[kt.MMA_TILE_K] == 32
    for knob in (kt.PIPELINE_ARCH, kt.CTA_GROUP, kt.MMA_TILE_M, kt.MMA_TILE_N, kt.MMA_TILE_K):
        assert cudnn.is_frontend_knob_type(knob)
    # a CTA-scoped family spells no CTA_GROUP; the warp grid appears only on warp-scoped families
    assert kt.WARPS_M not in public and kt.WARPS_N not in public


def test_split_k_and_swap_ab_round_trip():
    from dataclasses import replace

    cfg = replace(DEFAULT_CONFIG, split_k_slices=4, swap_ab=True)
    knobs = GemmKnobs.from_config(cfg)
    assert knobs.split_k_slices == 4 and knobs.swap_ab is True
    assert knobs.to_config() == cfg
    public = knobs.to_public()
    assert public[cudnn.knob_type.SPLIT_K_SLC] == 4 and public[cudnn.knob_type.SWAP_AB] == 1
    assert GemmKnobs.from_public(public).to_config() == cfg


def test_from_public_rejects_foreign_or_incomplete_records():
    kt = cudnn.knob_type
    good = GemmKnobs.from_config(DEFAULT_CONFIG).to_public()
    with pytest.raises(ValueError, match="not a tile axis"):
        GemmKnobs.from_public({**good, kt.SPLIT_KV: 2})  # an SDPA axis
    incomplete = dict(good)
    del incomplete[kt.MMA_TILE_K]
    with pytest.raises(ValueError, match="missing required axes: MMA_TILE_K"):
        GemmKnobs.from_public(incomplete)
    with pytest.raises(ValueError, match="must be an int"):
        GemmKnobs.from_public({**good, kt.TILE_N: "256"})
    with pytest.raises(ValueError, match="SWAP_AB must be 0 or 1"):
        GemmKnobs.from_public({**good, kt.SWAP_AB: 2})
    # a well-formed record that names no config the family constructors admit
    # (N=100 is not a tcgen05 N tile; the geometry check refuses it)
    with pytest.raises((KeyError, NotImplementedError)):
        GemmKnobs.from_public({**good, kt.TILE_N: 100}).to_config()


# ---------------------------------------------------------------------------
# Through the graph API on SM100
# ---------------------------------------------------------------------------


def _build_matmul_bias_relu():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.BFLOAT16)
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="bs")
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate()
    g.build_operation_graph()
    return g, (A, B, bias, Y)


def _operands():
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    bias_t = torch.randn(1, 1, N, dtype=torch.bfloat16).cuda()
    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.float(), b.float()) + bias_t.float()).to(torch.bfloat16)
    return a, b, bias_t, ref


def _frost_indices(g):
    return [i for i in range(g.get_execution_plan_count()) if is_python_engine(g.get_engine_and_knobs_at_index(i)[0])]


@requires_sm100
def test_frost_gemm_plan_carries_its_tile_config_as_knobs():
    from cudnn.gemm.frost.compiler import plan_config
    from cudnn.gemm.frost.heuristics import analyze_facts

    kt = cudnn.knob_type
    g, _ = _build_matmul_bias_relu()
    g.create_execution_plans([cudnn.heur_mode.A])
    frost = _frost_indices(g)
    assert len(frost) == 1, "one candidate today: the automatic strategy, spelled out"
    assert any(is_backend_engine(g.get_engine_and_knobs_at_index(i)[0]) for i in range(g.get_execution_plan_count()))

    engine_id, knobs = g.get_engine_and_knobs_at_index(frost[0])
    assert isinstance(knobs, dict) and knobs, "a frost_gemm plan is no longer knob-less"
    for required in (kt.PIPELINE_ARCH, kt.TILE_M, kt.TILE_N, kt.TILEK, kt.MMA_TILE_M, kt.MMA_TILE_N, kt.MMA_TILE_K, kt.TILE_CGA_M, kt.TILE_CGA_N):
        assert required in knobs, (required, knobs)
    assert knobs[kt.PIPELINE_ARCH] in (100, 103)
    name = g.get_plan_name_at_index(frost[0])
    assert name.startswith("frost_gemm[") and "TILE_M=" in name and "PIPELINE_ARCH=" in name, name

    # The listed knobs ARE the config the engine builds: same name as the automatic pick.
    facts = analyze_facts(g)
    assert facts is not None
    auto = plan_config(facts.chain, dynamic_shapes=facts.dynamic_shapes)
    assert GemmKnobs.from_public(knobs).config_name == auto.name


@requires_sm100
def test_frost_gemm_knobs_replay_and_run():
    g, _ = _build_matmul_bias_relu()
    g.create_execution_plans([cudnn.heur_mode.A])
    frost = _frost_indices(g)
    engine_id, knobs = g.get_engine_and_knobs_at_index(frost[0])
    name = g.get_plan_name_at_index(frost[0])

    # Replay on a fresh graph: the record round-trips, builds, runs, and matches.
    g2, (A, B, bias, Y) = _build_matmul_bias_relu()
    g2.create_execution_plan(engine_id, knobs)
    last = g2.get_execution_plan_count() - 1
    assert g2.get_engine_and_knobs_at_index(last) == (engine_id, knobs)
    assert g2.get_plan_name_at_index(last) == name
    g2.check_support()
    g2.build_plans()
    a, b, bias_t, ref = _operands()
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(max(g2.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g2.execute_plan_at_index({A: a, B: b, bias: bias_t, Y: y}, ws, index=last)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@requires_sm100
def test_frost_gemm_declines_a_record_naming_no_canonical_config():
    kt = cudnn.knob_type
    g, _ = _build_matmul_bias_relu()
    g.create_execution_plans([cudnn.heur_mode.A])
    engine_id, knobs = g.get_engine_and_knobs_at_index(_frost_indices(g)[0])
    g2, _ = _build_matmul_bias_relu()
    bad = {**knobs, kt.TILE_N: 100}  # tcgen05 N tiles are multiples of 8; nothing to snap to
    g2.create_execution_plan(engine_id, bad)
    g2.select_plan(g2.get_execution_plan_count() - 1)
    with pytest.raises((NotImplementedError, cudnn.cudnnGraphNotSupportedError, RuntimeError), match="frost_gemm"):
        g2.check_support()
        g2.build_plans()


def test_analyzer_declines_a_graph_without_a_gemm():
    from cudnn.gemm.frost.heuristics import analyze_facts

    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="x", dim=[1, 8, 8], stride=[64, 8, 1], data_type=cudnn.data_type.BFLOAT16)
    y = g.relu(input=x, name="r")
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    assert analyze_facts(g) is None
