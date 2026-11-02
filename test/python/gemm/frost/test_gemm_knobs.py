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

from gemm_test_utils import requires_sm100, requires_sm120, skip_unless_pipeline_active

import cudnn
from cudnn.engines import is_backend_engine, is_python_engine
from cudnn.gemm.frost.knobs import GemmKnobs
from cudnn.gemm.frost.tile_config import CATALOG, DEFAULT_CONFIG, by_name

pytestmark = pytest.mark.L0

M, N, K = 256, 512, 256


# ---------------------------------------------------------------------------
# Pure: knobs <-> TileConfig <-> public dict
# ---------------------------------------------------------------------------


def test_knobs_name_the_whole_catalog():
    """Every catalog entry round-trips (one test, not ~16k parametrized ids)."""
    for cfg in (DEFAULT_CONFIG, *CATALOG):
        knobs = GemmKnobs.from_config(cfg)
        assert knobs.config_name == cfg.name, cfg.name
        assert knobs.to_config() == cfg, cfg.name
        assert GemmKnobs.from_public(knobs.to_public()) == knobs, cfg.name


@pytest.mark.parametrize("cfg", [DEFAULT_CONFIG, *CATALOG[::97]], ids=lambda c: c.name)
def test_knobs_name_a_catalog_config_exactly(cfg):
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


def test_cluster_k_reduce_config_and_knobs():
    """Round-trip the explicit reduction mode and preserve legacy workspace records."""
    from dataclasses import replace

    workspace = replace(DEFAULT_CONFIG, split_k_slices=4)
    cluster = replace(workspace, cluster_k_reduce=True)
    assert workspace.cluster_shape == (2, 1, 1) and workspace.workspace_split_k
    assert cluster.cluster_shape == (2, 1, 4) and not cluster.workspace_split_k
    assert cluster.cga_tile_mnk(2)[2] == 4 * cluster.cta_tile_k(2)
    assert workspace.name != cluster.name
    assert by_name(cluster.name) == cluster
    knobs = GemmKnobs.from_config(cluster)
    public = knobs.to_public()
    assert public[cudnn.knob_type.CLUSTER_K_REDUCE] == 1
    assert GemmKnobs.from_public(public).to_config() == cluster
    legacy = GemmKnobs.from_config(workspace).to_public()
    cluster_reduce = legacy.pop(cudnn.knob_type.CLUSTER_K_REDUCE)
    assert cluster_reduce == 0
    assert GemmKnobs.from_public(legacy).to_config() == workspace
    with pytest.raises(ValueError, match="CLUSTER_K_REDUCE must be 0 or 1"):
        GemmKnobs.from_public({**public, cudnn.knob_type.CLUSTER_K_REDUCE: 2})


def test_cluster_k_reduce_config_rejects_invalid_combinations():
    """Reject non-boolean mode values and impossible full-cluster geometries."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="cluster_k_reduce must be bool"):
        replace(DEFAULT_CONFIG, cluster_k_reduce=1)
    with pytest.raises(NotImplementedError, match="at least two"):
        replace(DEFAULT_CONFIG, cluster_k_reduce=True)
    with pytest.raises(NotImplementedError, match="CGA limit"):
        replace(DEFAULT_CONFIG, split_k_slices=16, cluster_k_reduce=True)


@requires_sm100
def test_cluster_k_reduce_declines_sm120_before_compiling(monkeypatch):
    """Reject an unsupported cluster mode before entering the SM120 compiler."""
    from dataclasses import replace
    from cudnn.gemm.frost.sm120 import compiler
    from cudnn.gemm.frost.tile_config import as_pipeline

    graph, _ = _build_matmul_bias_relu()
    cfg = replace(as_pipeline(DEFAULT_CONFIG, "sm120"), split_k_slices=4, cluster_k_reduce=True)

    def unexpected_compile(*args, **kwargs):
        """Fail if an unsupported config or runtime extent triggers compilation."""
        pytest.fail("an unsupported cluster reduction must be declined before compilation")

    monkeypatch.setattr(compiler, "_import_kernel", unexpected_compile)
    with pytest.raises(NotImplementedError, match="cluster_k_reduce"):
        compiler.jit_from_cudnn_graph(graph, cfg)


@requires_sm100
@pytest.mark.parametrize("slices", [1, 4])
def test_sm120_workspace_split_k_executes_on_sm100_family(slices):
    """Keep the SM120 workspace path executable on an admitted SM100-family device."""
    from dataclasses import replace
    from cudnn.gemm.frost.sm120 import compiler
    from cudnn.gemm.frost.tile_config import as_pipeline
    from cudnn.frost.workspace import Workspace

    graph, (A, B, bias, Y) = _build_matmul_bias_relu()
    cfg = replace(as_pipeline(DEFAULT_CONFIG, "sm120"), split_k_slices=slices)
    plan = compiler.jit_from_cudnn_graph(graph, cfg)
    assert not plan.config.cluster_k_reduce
    assert plan.workspace_bytes == (0 if slices == 1 else slices * M * N * 4)
    scratch = torch.empty(max(plan.workspace_bytes, 1), dtype=torch.uint8, device="cuda")
    workspace = Workspace(scratch, plan.workspace_bytes, "sm120 workspace split-K") if plan.workspace_bytes else None
    a, b, bias_t, reference = _operands()
    out = torch.full_like(reference, float("nan"))
    plan({A: a, B: b, bias: bias_t, Y: out}, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


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
    with pytest.raises(ValueError, match="SPLIT_K_SLC must be >= 1"):
        GemmKnobs.from_public({**good, kt.SPLIT_K_SLC: 0})  # would otherwise spell (and replay as) one slice
    # a well-formed record that names no config the family constructors admit
    # (N=100 is not a tcgen05 N tile; the geometry check refuses it)
    with pytest.raises((KeyError, NotImplementedError)):
        GemmKnobs.from_public({**good, kt.TILE_N: 100}).to_config()


def test_replayed_split_k_record_declines_under_dynamic_shapes():
    """The automatic pick never adds split-K to a dynamic-shape graph; a
    replayed record does not get to bypass that rule. The knobs branch of
    plan_config never reads the chain, so none is needed here."""
    from dataclasses import replace

    from cudnn.gemm.frost.compiler import plan_config

    knobs = GemmKnobs.from_config(replace(DEFAULT_CONFIG, split_k_slices=2))
    assert plan_config(None, dynamic_shapes=False, knobs=knobs).split_k_slices == 2
    with pytest.raises(NotImplementedError, match="dynamic shapes"):
        plan_config(None, dynamic_shapes=True, knobs=knobs)


@requires_sm100
def test_calibrated_heuristic_lists_eight_unique_configs(monkeypatch):
    """Propose distinct replayable plans while retaining workspace reduction by default."""
    from cudnn.gemm.frost import compiler, planning, tile_config
    from cudnn.gemm.frost.fusion_ir import FusionChain, MatmulSpec, OutputSpec
    from cudnn.gemm.frost.heuristics import ENGINE, GemmFacts, recommend

    skip_unless_pipeline_active(DEFAULT_CONFIG)
    device = planning.DeviceProperties(100, "NVIDIA B200", 148, 132644864)
    monkeypatch.setattr(planning, "current_device_properties", lambda: device)
    monkeypatch.setattr(tile_config, "_sm_count", lambda: 148)
    monkeypatch.setattr(compiler, "_sm_count", lambda: 148)
    monkeypatch.setattr(compiler, "_current_arch", lambda: 100)
    chain = FusionChain(matmul=MatmulSpec(M=256, N=256, K=16384), output_specs=[OutputSpec(source_ref=-1, dtype="bf16")])
    facts = GemmFacts(chain, dynamic_shapes=False)
    plans = recommend("A", facts, {ENGINE: 42})
    assert len(plans) == 8
    assert all(not p.knobs.cluster_k_reduce for p in plans)
    assert len({p.knobs for p in plans}) == 8
    assert all(p.engine_id == 42 for p in plans)
    assert plans[0].knobs.to_config() == compiler.plan_config(chain)
    for plan in plans:
        config = plan.knobs.to_config()
        compiler.probe_chain(chain, config)
        assert GemmKnobs.from_public(plan.knobs.to_public()).to_config() == config
    assert recommend("FALLBACK", facts, {ENGINE: 42}) == plans


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
    assert len(frost) == 1, "uncalibrated fused graphs retain their single recommendation"
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
@pytest.mark.parametrize(
    "a_dtype,b_dtype,output_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "bf16", "fp16"),
        ("fp16", "fp16", "bf16"),
        ("fp8_e4m3", "fp8_e4m3", "bf16"),
        ("fp8_e5m2", "fp8_e5m2", "fp16"),
        ("fp8_e4m3", "fp8_e5m2", "fp16"),
        ("fp8_e5m2", "fp8_e4m3", "bf16"),
    ],
)
@pytest.mark.parametrize("profile_device", ["b200", "rubin"])
def test_eight_calibrated_plans_deduplicate_across_modes_and_replay(monkeypatch, profile_device, a_dtype, b_dtype, output_dtype):
    """Replay calibrated proposals on the active tcgen05 GPU for numerics."""
    from cudnn.gemm.frost import compiler, planning, tile_config

    skip_unless_pipeline_active(DEFAULT_CONFIG)
    if profile_device == "b200":
        monkeypatch.setattr(planning, "current_device_properties", lambda: planning.DeviceProperties(100, "NVIDIA B200", 148, 132644864))
        monkeypatch.setattr(tile_config, "_sm_count", lambda: 148)
        monkeypatch.setattr(compiler, "_sm_count", lambda: 148)
    elif torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin profile requires an SM107 GPU")
    batch, m, n, k = (3, 192, 160, 4160) if a_dtype != b_dtype else (1, 128, 128, 1024)
    dtypes = {
        "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16),
        "fp16": (cudnn.data_type.HALF, torch.float16),
        "fp8_e4m3": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn),
        "fp8_e5m2": (cudnn.data_type.FP8_E5M2, torch.float8_e5m2),
    }
    a_cudnn, a_torch = dtypes[a_dtype]
    b_cudnn, b_torch = dtypes[b_dtype]
    output_cudnn, output_torch = dtypes[output_dtype]

    def graph():
        g = cudnn.pygraph(io_data_type=output_cudnn, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        a = g.tensor(name="A", dim=[batch, m, k], stride=[m * k, k, 1], data_type=a_cudnn)
        b = g.tensor(name="B", dim=[batch, k, n], stride=[k * n, 1, k], data_type=b_cudnn)
        y = g.matmul(A=a, B=b, name="mm")
        y.set_output(True).set_data_type(output_cudnn)
        g.validate()
        g.build_operation_graph()
        return g, a, b, y

    original, _, _, _ = graph()
    original.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    records = [original.get_engine_and_knobs_at_index(i) for i in _frost_indices(original)]
    assert len(records) == 8
    configs = [GemmKnobs.from_public(knobs).to_config() for _, knobs in records]
    assert len(set(configs)) == 8
    if a_dtype in ("bf16", "fp16"):
        assert {cfg.swap_ab for cfg in configs} == {False, True}
        assert any(cfg.split_k_slices == 1 for cfg in configs)
        assert any(cfg.split_k_slices & (cfg.split_k_slices - 1) for cfg in configs)

    torch.manual_seed(918)
    a = torch.randint(-2, 3, (batch, m, k), device="cuda").to(a_torch)
    b = torch.randint(-2, 3, (batch, n, k), device="cuda").to(b_torch)
    reference = (a.float() @ b.float().transpose(-1, -2)).to(output_torch)
    for engine_id, knobs in records:
        replay, A, B, Y = graph()
        replay.create_execution_plan(engine_id, knobs)
        index = replay.get_execution_plan_count() - 1
        replay.select_plan(index)
        replay.check_support()
        replay.build_plans()
        assert replay.get_engine_and_knobs_at_index(index) == (engine_id, knobs)
        y = torch.full((batch, m, n), float("nan"), dtype=output_torch, device="cuda")
        workspace = torch.empty(max(1, replay.get_workspace_size()), dtype=torch.uint8, device="cuda")
        replay.execute_plan_at_index({A: a, B: b, Y: y}, workspace, index=index)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, reference, rtol=0, atol=0)


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
@pytest.mark.parametrize("cluster_reduce", [False, True])
def test_cluster_splitk_public_replay_workspace(cluster_reduce):
    """Honor the public reduction-mode knob and its reported workspace requirement."""
    from dataclasses import replace

    g, _ = _build_matmul_bias_relu()
    g.create_execution_plans([cudnn.heur_mode.A])
    engine_id, _ = g.get_engine_and_knobs_at_index(_frost_indices(g)[0])
    geometry = "128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg = replace(by_name("CONFIG_sm100_" + geometry), split_k_slices=4, cluster_k_reduce=cluster_reduce)
    knobs = GemmKnobs.from_config(cfg).to_public()
    replay, (A, B, bias, Y) = _build_matmul_bias_relu()
    replay.create_execution_plan(engine_id, knobs)
    index = replay.get_execution_plan_count() - 1
    replay.select_plan(index)
    replay.check_support()
    replay.build_plans()
    assert replay.get_engine_and_knobs_at_index(index) == (engine_id, knobs)
    expected_workspace = 0 if cluster_reduce else 4 * M * N * 4
    assert replay.get_workspace_size() == expected_workspace
    workspace = torch.empty(max(expected_workspace, 1), dtype=torch.uint8, device="cuda")
    a, b, bias_t, _ = _operands()
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):
        reference = torch.relu(torch.einsum("bmk,bnk->bmn", a.float(), b.float()) + bias_t.float()).to(torch.bfloat16)
        y.fill_(float("nan"))
        torch.cuda.set_sync_debug_mode("error")
        try:
            replay.execute({A: a, B: b, bias: bias_t, Y: y}, workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        torch.testing.assert_close(y, reference, rtol=0, atol=0)
        a.mul_(2)


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


@requires_sm120
def test_frost_gemm_declines_a_split_k_record_for_a_tile_invariant_fp32_sum(monkeypatch):
    """SPLIT_K_SLC is a performance knob, so a record may not change what the plan computes. The split-K
    reducer adds an fp32 sum over all of M and N with one same-address atomic per output vector, which
    drifts (a constant-input AVG at 3000x3000x1024 came back 2.8% low at split 2): the engine never lists
    such a split, and a replayed record that adds one declines before any kernel is rendered."""
    from dataclasses import replace

    from cudnn.gemm.frost import compiler

    # Past the automatic pick's 2048-byte K floor (the module's K=256 is not), so only the drift rule keeps the
    # listed record unsplit: without it, this graph lists SPLIT_K_SLC=2 on a 188-SM RTX PRO 6000.
    k = 4096

    def graph():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[1, M, k], stride=[M * k, k, 1], data_type=cudnn.data_type.BFLOAT16)
        B = g.tensor(name="B", dim=[1, k, N], stride=[k * N, 1, k], data_type=cudnn.data_type.BFLOAT16)
        Y = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        R = g.reduction(input=Y, mode=cudnn.reduction_mode.AVG, name="red")
        R.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_output(True).set_data_type(cudnn.data_type.FLOAT)
        g.validate()
        g.build_operation_graph()
        return g

    listed = graph()
    listed.create_execution_plans([cudnn.heur_mode.A])
    engine_id, knobs = listed.get_engine_and_knobs_at_index(_frost_indices(listed)[0])
    assert knobs[cudnn.knob_type.SPLIT_K_SLC] == 1
    record = GemmKnobs.from_config(replace(GemmKnobs.from_public(knobs).to_config(), split_k_slices=2)).to_public()

    def no_render(src):
        pytest.fail("a declined record must not render a kernel")

    monkeypatch.setattr(compiler, "_import_kernel", no_render)
    replay = graph()
    replay.create_execution_plan(engine_id, record)
    replay.select_plan(replay.get_execution_plan_count() - 1)
    replay.check_support()
    with pytest.raises(NotImplementedError, match="frost_gemm: .*fp32 avg reduction over all of M and N"):
        replay.build_plans()


@pytest.mark.parametrize("swapped", (False, True), ids=("plain", "swap_ab"))
def test_sm120_split_k_gate_declines_tile_invariant_fp32_sums(swapped):
    """Host-only (no kernel, no GPU work): the SM120 split-K gate that explicit configs, replayed records and the
    automatic pick share declines an fp32 sum over all of M and N, per batch or over the batch too, also under
    swap-AB (``swap_ab`` swaps a reduction's M and N extents). Modes exact under atomics (AMAX, an int32 ADD)
    and per-row / per-column sums keep the split."""
    from dataclasses import replace

    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.sm120 import compiler

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), split_k_slices=2, swap_ab=swapped)

    def reason(mode, dim, batch=1, dtype=cudnn.data_type.FLOAT):
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
        B = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
        Y = g.relu(input=g.matmul(A=A, B=B, name="mm"), name="r")
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        R = g.reduction(input=Y, mode=getattr(cudnn.reduction_mode, mode), compute_data_type=dtype, name="red")
        R.set_dim(dim).set_stride([dim[1] * dim[2], dim[2], 1]).set_output(True).set_data_type(dtype)
        chain = analyze(g)
        # probe_chain and jit_from_cudnn_graph gate the swapped view of a swap-AB config
        return compiler._splitk_reject_reason(swap_ab(chain) if swapped else chain, cfg)

    for mode in ("ADD", "AVG", "NORM1", "NORM2"):
        for batch, dim in ((1, [1, 1, 1]), (2, [2, 1, 1]), (2, [1, 1, 1])):
            assert f"fp32 {mode.lower()} reduction over all of M and N" in (reason(mode, dim, batch) or ""), (mode, dim)
    assert reason("AMAX", [1, 1, 1]) is None
    assert reason("ADD", [1, 1, 1], dtype=cudnn.data_type.INT32) is None
    for mode in ("ADD", "AVG"):
        assert reason(mode, [1, M, 1]) is None  # per row
        assert reason(mode, [1, 1, N]) is None  # per column


def test_analyzer_declines_a_graph_without_a_gemm():
    from cudnn.gemm.frost.heuristics import analyze_facts

    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="x", dim=[1, 8, 8], stride=[64, 8, 1], data_type=cudnn.data_type.BFLOAT16)
    y = g.relu(input=x, name="r")
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    assert analyze_facts(g) is None
