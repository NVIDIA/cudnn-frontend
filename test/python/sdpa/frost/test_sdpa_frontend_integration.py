# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend integration for the FROST DSL SDPA engines: they join the ONE ranked
plan list the graph API builds at create_execution_plans() — discovered from
``cudnn/engines/manifest.py``, no registration call and no environment variable —
and any entry of that list is selectable with select_plan()."""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import MANIFEST, is_backend_engine, is_python_engine

from cudnn.sdpa.fwd.engines import engine_name

_FROST = engine_name(512)  # matches the D=512 graphs below
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(torch.cuda.current_device()) == (10, 0)


def _dsl_deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


_SM100 = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100 (Blackwell) GPU")
_SM100_DSL = pytest.mark.skipif(
    not (_is_sm100() and _dsl_deps_available()),
    reason="needs an SM100 (Blackwell) GPU with cutlass installed",
)

# The default pytest.ini addopts is `-m L0`; mark the whole module so it runs.
pytestmark = pytest.mark.L0

B, H, S, D = 2, 8, 256, 512


def _build_causal_sdpa(dtype=cudnn.data_type.HALF, d=D, **sdpa_kwargs):
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=dims, stride=strides, data_type=dtype, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=dtype, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=dtype, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=True, **sdpa_kwargs)
    o.set_output(True).set_dim(dims).set_stride(strides)
    return g, q, k, v, o


def _plan(g):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return g


def _plan_names(g):
    return [g.get_plan_name_at_index(i) for i in range(len(g.plans))]


def _index_of(g, name):
    names = _plan_names(g)
    assert name in names, f"no plan named {name!r} in {names}"
    return names.index(name)


def _pin(g, name):
    """Pin the ranked entry named ``name``. Today's placeholder frontend
    heuristics rank the backend's plans first, so a test that means a specific
    engine says so by index (graph.plans / select_plan)."""
    g.select_plan(_index_of(g, name))
    return g


def test_engine_family_is_in_the_manifest():
    """The SDPA fwd family ships in the library's static table — discovery is
    the library's job, not a registration call the user has to make."""
    (row,) = [r for r in MANIFEST if r.factory == "FrostSdpaFwdEngines"]
    assert is_python_engine(row.engine_id)
    assert row.id_end > row.engine_id + 1  # a family: a whole id block, one id per cell


@_GPU
def test_ineligible_graph_lists_no_dsl_engine():
    """ALiBi (a feature no FROST SDPA engine serves) validates on any GPU, so no
    SM100 needed to check the engine declines it — and a declined graph has no
    python entry to select at all. (Small head dims are no longer a rejection:
    the head-dim ENVELOPE serves any d <= flavor via TMA zero-padding.)"""
    g, q, k, v, o = _build_causal_sdpa(use_alibi_mask=True)
    _plan(g)
    assert _FROST not in _plan_names(g)
    assert not any(is_python_engine(p.engine_id) for p in g.plans)


@_SM100
def test_eligible_graph_lists_matching_dsl_engine():
    """ONE list: the engine that claims the graph is an entry of the same list
    the backend's plans are in, and get_execution_plan_count() counts both."""
    g, q, k, v, o = _build_causal_sdpa()
    _plan(g)
    names = _plan_names(g)
    assert _FROST in names
    assert g.get_execution_plan_count() == len(names) == len(g.plans)
    assert is_python_engine(g.plans[names.index(_FROST)].engine_id)


@_SM100
def test_both_frost_and_backend_entries_are_selectable():
    """Either side of the ranked list can be pinned. Deliberately says nothing
    about which one ranks first — that is the placeholder in
    engines/heuristics.py, and a cost model will change it."""
    g, q, k, v, o = _build_causal_sdpa()
    _plan(g)
    _pin(g, _FROST)
    assert g.selected_engine.name == _FROST
    backend = next(i for i, p in enumerate(g.plans) if is_backend_engine(p.engine_id))
    g.select_plan(backend)
    assert g.selected_engine is None  # None == the backend path


@_SM100_DSL
def test_select_dsl_engine_runs_and_matches_torch():
    q_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa()
    _plan(g)
    _pin(g, _FROST)
    g.check_support()
    g.build_plans()
    # Honest workspace: this graph is inference (no Stats output), so the
    # engine carves a dummy LSE (B*H*S fp32) from the caller's buffer.
    ws_size = g.get_workspace_size()
    assert ws_size >= B * H * S * 4

    ws = torch.empty(ws_size, device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_gpu,
        k_gpu,
        v_gpu,
        is_causal=True,
        scale=1.0 / (D**0.5),
    )
    torch.testing.assert_close(o_gpu, ref, atol=5e-2, rtol=3e-2)


@_SM100_DSL
def test_default_plan_runs_and_matches_torch():
    """No select_plan at all: the default entry of the ranked list builds and
    executes (the backend today), so the unified list never breaks the classic
    path."""
    q_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa()
    _plan(g)
    g.check_support()
    g.build_plans()

    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_gpu,
        k_gpu,
        v_gpu,
        is_causal=True,
        scale=1.0 / (D**0.5),
    )
    torch.testing.assert_close(o_gpu, ref, atol=5e-2, rtol=3e-2)


@_SM100_DSL
def test_envelope_lists_every_covering_flavor_smallest_first():
    """Head-dim ENVELOPE: a (64,64) graph is eligible for every f16 flavor, and
    the family proposes them smallest-covering-first, so the tightest tile is
    the first python entry of the ranked list; the padded run matches torch."""
    d = 64
    q_gpu = torch.randn(B, S, H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, S, H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, S, H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, S, H, d, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa(d=d)
    _plan(g)
    names = _plan_names(g)
    for flavor in (128, 256, 512):
        assert engine_name(flavor) in names  # every covering flavor
    python = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    assert names[python[0]] == engine_name(128)  # tightest flavor first
    g.select_plan(python[0])
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == engine_name(128)

    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_gpu,
        k_gpu,
        v_gpu,
        is_causal=True,
        scale=1.0 / (d**0.5),
    )
    torch.testing.assert_close(o_gpu, ref, atol=5e-2, rtol=3e-2)


@pytest.mark.skip(
    reason="graph.set_engine_knobs() was removed with the monkey-patch dispatch layer and has no replacement in this MR. "
    "Knobs now ride on the plan (engines.base.PlanConfig.knobs) — an engine proposes one entry per knob set and the "
    "caller picks with select_plan(); re-enable once the SDPA fwd family proposes its SdpaFwdKnobs domain as plans."
)
@_SM100
def test_knob_request_gates_selection():
    """An unsupported knob request must not silently degrade: the plan carrying
    it should simply not be in the list (or decline when pinned)."""


@_SM100
def test_no_magic_import_required():
    """Discovery is the library's job: a fresh process that only imports cudnn
    (never cudnn.sdpa) still gets the FROST engine as an entry of graph.plans
    and can pin it — the manifest imports the engine module lazily, and there is
    no environment variable to set."""
    import subprocess
    import sys

    code = (
        "import torch, cudnn\n"
        "b,h,s,d = 1,2,256,512\n"
        "q_gpu = torch.randn(b,s,h,d, device='cuda', dtype=torch.float16).transpose(1,2)\n"
        "k_gpu, v_gpu, o_gpu = q_gpu.clone(), q_gpu.clone(), torch.empty_like(q_gpu)\n"
        "g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF,\n"
        "                  intermediate_data_type=cudnn.data_type.FLOAT,\n"
        "                  compute_data_type=cudnn.data_type.FLOAT)\n"
        "q = g.tensor_like(q_gpu); k = g.tensor_like(k_gpu); v = g.tensor_like(v_gpu)\n"
        "o, _ = g.sdpa(name='s', q=q, k=k, v=v, attn_scale=0.08, generate_stats=False, use_causal_mask=True)\n"
        "o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())\n"
        "g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])\n"
        "names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]\n"
        "assert 'sdpa_fwd_prefill_sm100_d512' in names, names\n"
        "g.select_plan(names.index('sdpa_fwd_prefill_sm100_d512'))\n"
        "g.check_support()\n"
        "print('ELIGIBLE-WITHOUT-IMPORT')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert "ELIGIBLE-WITHOUT-IMPORT" in out.stdout, out.stderr[-2000:]


@_SM100_DSL
def test_plan_name_contract():
    """ONE list: get_execution_plan_count() / get_plan_name_at_index() describe
    graph.plans, python entries and backend entries alike. deselect_engines bars
    a name from the build walk WITHOUT renumbering the list — an index a caller
    already holds never moves."""
    g, *_ = _build_causal_sdpa()
    _plan(g)
    count = g.get_execution_plan_count()
    names = _plan_names(g)
    assert count == len(names) == len(g.plans)
    frost = names.index(_FROST)
    assert is_python_engine(g.plans[frost].engine_id)
    assert all(is_backend_engine(p.engine_id) or is_python_engine(p.engine_id) for p in g.plans)
    g.check_support()
    g.build_plans()

    # FROST deselected: same list, same names, same indices — but the barred
    # entry is skipped and a backend plan serves the graph.
    g2, *_ = _build_causal_sdpa()
    _plan(g2)
    g2.deselect_engines([_FROST])
    assert _plan_names(g2) == names
    g2.check_support()
    g2.build_plans()
    assert g2.selected_engine is None


@_SM100_DSL
def test_workspace_carve_no_per_execute_allocs_and_guards():
    """The dummy-LSE (inference) path carves scratch from the caller's
    workspace: steady-state executes make ZERO torch CUDA allocations, an
    undersized buffer raises instead of corrupting, and get_workspace_size
    reports the real requirement (not 0)."""
    q_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa()
    _plan(g)
    _pin(g, _FROST)
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == _FROST

    ws_size = g.get_workspace_size()
    assert ws_size >= B * H * S * 4  # dummy LSE: b*h_q*s_q fp32

    # Undersized / absent workspace: loud failure, no silent allocation.
    with pytest.raises(ValueError, match="workspace"):
        g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, torch.empty(1, device="cuda", dtype=torch.uint8))
    with pytest.raises((ValueError, TypeError), match="workspace"):
        g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, None)

    ws = torch.empty(ws_size, device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)  # warm-up: one-time dummy caches fill
    torch.cuda.synchronize()

    stats_key = "allocation.all.allocated"
    before = torch.cuda.memory_stats().get(stats_key, 0)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats().get(stats_key, 0)
    assert after == before, f"LSE path made {after - before} per-execute CUDA allocation(s); scratch must be carved from the workspace"

    ref = torch.nn.functional.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, is_causal=True, scale=1.0 / (D**0.5))
    torch.testing.assert_close(o_gpu, ref, atol=5e-2, rtol=3e-2)
