# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend integration: the GEMM engine ``frost_gemm`` joins the ONE ranked
plan list the graph API builds at create_execution_plans() — discovered from
``cudnn/engines/manifest.py``, no registration call and no environment variable.
Exercises its presence in the list, select_plan/deselect_engines, the build walk
falling through a declining plan, ineligible graphs, and the wrapper.Graph path.
"""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import requires_sm100

import cudnn
from cudnn.engines import MANIFEST, OUT_OF_TREE_ID_BASE, PlanConfig, Router, is_backend_engine, is_python_engine

pytestmark = pytest.mark.L0

_GPU = requires_sm100

M, N, K = 256, 256, 128
_FROST = "frost_gemm"


def _build_matmul_bias_relu():
    """A recorded bf16 matmul + per-col bias + relu graph → (g, A, B, bias, Y)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.BFLOAT16)
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="bs")
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, A, B, bias, Y


def _operands():
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    bias_t = torch.randn(1, 1, N, dtype=torch.bfloat16).cuda()
    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.float(), b.float()) + bias_t.float()).to(torch.bfloat16)
    return a, b, bias_t, ref


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


def _pin_frost(g):
    """Pin the FROST entry of the ranked list. The placeholder frontend
    heuristics rank the backend's plans first, so a test that means FROST says
    so by index (graph.plans / select_plan) instead of relying on position."""
    g.select_plan(_index_of(g, _FROST))
    return g


def _first_backend_index(g):
    return next(i for i, p in enumerate(g.plans) if is_backend_engine(p.engine_id))


def test_engine_is_in_the_manifest():
    """The engine ships in the library's static table — discovery is the
    library's job, not a registration call the user has to make."""
    (row,) = [r for r in MANIFEST if r.name == _FROST]
    assert is_python_engine(row.engine_id)
    assert row.module == "cudnn.gemm.frost.engine" and row.factory == "FrostGemmEngines"
    assert list(row.slots) == [_FROST], "the manifest assigns this engine its id"


@_GPU
def test_select_frost_engine_runs_frost():
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _plan(g)
    _pin_frost(g)
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == _FROST
    assert g.get_workspace_size() == 0  # honest 0: this executor carves no per-execute scratch
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


_OVERRIDE_SHAPES = [(256, 256, 128), (1024, 512, 512), (200, 768, 256), (333, 512, 264)]


def _build_matmul_uids(m, n, k):
    """A plain bf16 matmul with fixed uids (needed for override_uids)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", uid=1, dim=[1, m, k], stride=[m * k, k, 1], data_type=cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", uid=2, dim=[1, k, n], stride=[k * n, 1, k], data_type=cudnn.data_type.BFLOAT16)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_uid(3)
    return g, A, B, C


def _mm_operands(m, n, k):
    a = torch.empty(1, m, k, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    b = torch.empty(1, n, k, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    return a, b, ref


@_GPU
def test_override_shape_frost():
    """A FROST plan built for one problem size runs OTHER sizes through the native
    override-shape API — ``get_workspace_size_plan_at_index`` /
    ``execute_plan_at_index`` with ``override_uids/shapes/strides`` — and through
    plain ``execute`` with new-shape buffers. The compiled kernel is shape-agnostic
    (M/N/K symbolic), so no rebuild; results are bit-exact (small-int inputs)."""
    h = cudnn.create_handle()
    m0, n0, k0 = 256, 256, 128
    g, _A, _B, _C = _build_matmul_uids(m0, n0, k0)
    _plan(g)
    frost = _index_of(g, _FROST)
    g.select_plan(frost)
    g.check_support()
    g.build_plans()  # one JIT compile, at the anchor shape

    for m, n, k in _OVERRIDE_SHAPES:
        a, b, ref = _mm_operands(m, n, k)
        ou = [1, 2, 3]
        osh = [[1, m, k], [1, k, n], [1, m, n]]
        ost = [[m * k, k, 1], [k * n, 1, k], [m * n, n, 1]]

        # (a) native override-shape API: workspace query + indexed execute.
        wsz = g.get_workspace_size_plan_at_index(frost, h, ou, osh, ost)
        assert wsz == 0  # FROST owns its workspace at any shape
        ws = torch.empty(max(wsz, 1), device="cuda", dtype=torch.uint8)
        c = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
        g.execute_plan_at_index(
            {1: a, 2: b, 3: c},
            ws,
            frost,
            handle=h,
            override_uids=ou,
            override_shapes=osh,
            override_strides=ost,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(c, ref, atol=0, rtol=0)

        # (b) plain execute() with override args (FROST reads the new shape from
        # the buffers; override_* accepted for API parity).
        c2 = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
        g.execute(
            {1: a, 2: b, 3: c2},
            ws,
            handle=h,
            override_uids=ou,
            override_shapes=osh,
            override_strides=ost,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(c2, ref, atol=0, rtol=0)


@_GPU
def test_misaligned_buffer_rejected():
    """A 16-byte-misaligned operand (a byte-offset view) is rejected with a clean
    ValueError rather than faulting deep in TMA descriptor creation — the runtime
    honors the compile-time ``assumed_align=16`` contract. Especially relevant to
    override-shape, where callers may pass sliced / offset views."""
    h = cudnn.create_handle()
    m, n, k = 256, 256, 128
    g, _A, _B, _C = _build_matmul_uids(m, n, k)
    _plan(g)
    _pin_frost(g)
    g.check_support()
    g.build_plans()

    a, b, ref = _mm_operands(m, n, k)
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)

    # aligned baseline works
    c = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
    g.execute({1: a, 2: b, 3: c}, ws, handle=h)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref, atol=0, rtol=0)

    # 16-misaligned A (base pointer offset 8 bytes) → clean ValueError, no CUDA fault
    nbytes = 1 * m * k * 2
    storage = torch.empty(nbytes + 16, dtype=torch.uint8, device="cuda")
    a_mis = storage[8 : 8 + nbytes].view(torch.bfloat16).view(1, m, k)
    a_mis.copy_(a)
    assert a_mis.data_ptr() % 16 != 0
    c2 = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="alignment .* < required"):
        g.execute({1: a_mis, 2: b, 3: c2}, ws, handle=h)


@_GPU
@pytest.mark.parametrize("route", ["default", "frost", "deselected"])
def test_ranked_list_routes_and_all_routes_agree(route):
    """The same graph runs on whichever entry of the ranked list is selected:
    the default, the FROST entry pinned by select_plan, or the backend after
    deselect_engines bars FROST. All three produce the right numbers.

    The default route asserts nothing about WHICH engine served it — that is
    the placeholder in engines/heuristics.py, and a cost model will change it."""
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _plan(g)
    if route == "frost":
        _pin_frost(g)
    elif route == "deselected":
        g.deselect_engines([_FROST])
    g.check_support()
    g.build_plans()
    if route == "frost":
        assert g.selected_engine.name == _FROST
        # Honest workspace: 0 because this executor carves no per-execute
        # scratch from the caller (its TMA-descriptor buffer is a one-time,
        # plan-owned allocation) — not a blanket FROST 0.
        assert g.get_workspace_size() == 0
    elif route == "deselected":
        assert g.selected_engine is None  # FROST barred -> the backend served it
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_build_walk_falls_through_a_declining_plan(caplog):
    """A plan that declines at build time is logged and the walk moves to the
    next entry — here a python engine ranked ahead of the backend, so the graph
    still builds and executes natively with no exception reaching the user
    (a select_plan pin is strict instead; see build_plans)."""
    import logging

    from cudnn.engines import BaseEngine

    class Boom(BaseEngine):
        name = "frost_fake_always_fails"
        engine_id = OUT_OF_TREE_ID_BASE + 1

        def build_plan(self, graph, plan, ctx=None):
            raise NotImplementedError("frost build boom")

        def execute(self, graph, tensor_data, ctx=None):
            raise AssertionError("should never run")

    boom = Boom()

    class BoomFirst(Router):
        def plan(self, graph, engines):
            return [PlanConfig(boom.engine_id)] + graph.backend_plan_entries()

    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    g.set_router(BoomFirst()).register_backend(boom)
    _plan(g)
    assert g.get_plan_name_at_index(0) == "frost_fake_always_fails"
    g.check_support()
    with caplog.at_level(logging.INFO, logger="cudnn.pygraph"):
        g.build_plans()  # entry 0 declines -> the walk lands on the backend
    assert any("declined at build time" in r.getMessage() for r in caplog.records)
    assert g.selected_engine is None and is_backend_engine(g.plans[g._plan_index].engine_id)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_frost_is_one_entry_of_the_ranked_list():
    """ONE list: get_execution_plan_count() counts the backend's plans AND the
    python ones, and FROST appears exactly once."""
    g, _A, _B, _bias, _Y = _build_matmul_bias_relu()
    _plan(g)
    names = _plan_names(g)
    assert names.count(_FROST) == 1
    assert g.get_execution_plan_count() == len(names)
    assert sum(1 for p in g.plans if is_python_engine(p.engine_id)) == 1


class _LoweredSpy:
    """Records deselect_engines calls reaching the lowered C++ graph, forwarding
    everything else untouched."""

    def __init__(self, real):
        self._real = real
        self.deselected = []

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name != "deselect_engines":
            return attr

        def _record(names, *args, **kwargs):
            self.deselected.append(list(names))
            return attr(names, *args, **kwargs)

        return _record


def _spy_on_lowered(g):
    spy = _LoweredSpy(g.__dict__["_lowered_graph"])
    g.__dict__["_lowered_graph"] = spy
    return spy


def _native_engine_token(g):
    """The cuDNN engine name (e.g. "eng0") of the leading NATIVE plan."""
    return g.get_plan_name_at_index(_first_backend_index(g)).split("_")[0]


@_GPU
def test_deselect_native_engine_reaches_cudnn():
    """deselect_engines is the classic API and stays a passthrough to the
    lowered C++ graph — pygraph bars the name across the whole ranked list AND
    forwards it, so a backend engine name still reaches cuDNN itself."""
    g, *_ = _build_matmul_bias_relu()
    _plan(g)
    native = _native_engine_token(g)
    spy = _spy_on_lowered(g)
    assert g.deselect_engines([native]) is g  # fluent, like the other setters
    assert spy.deselected == [[native]]


@_GPU
def test_deselect_mixed_frost_and_native():
    """A mixed list bars the python engine locally AND forwards to cuDNN; the
    barred FROST entry is skipped by the build walk."""
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _plan(g)
    native = _native_engine_token(g)
    spy = _spy_on_lowered(g)
    g.deselect_engines([_FROST, native])
    assert _FROST in g._barred_names
    assert spy.deselected == [[_FROST, native]]
    g.build_plans()
    assert g.selected_engine is None  # FROST barred -> a backend plan serves it
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


def _build_moe(S=512, N=256, K=256, E=4):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="fto", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g, tok, w, fto, out, (S, N, K, E)


@_GPU
def test_moe_reports_and_uses_caller_workspace():
    """The MoE plan needs a per-CTA TMA-descriptor scratch. get_workspace_size()
    must report it (not 0) and execute() must carve from the CALLER's buffer, so
    the pointer is caller-owned and stable — the workspace contract in
    cudnn/gemm/frost/engine.py."""
    g, tok, w, fto, out, (S, N, K, E) = _build_moe()
    _plan(g)
    _pin_frost(g)
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == _FROST
    wsz = g.get_workspace_size()
    assert wsz > 0  # honest: this plan really does own per-CTA scratch

    t = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    wt = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    off = torch.tensor([0, S // 4, S // 2, 3 * S // 4], dtype=torch.int32, device="cuda")
    o = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(wsz, dtype=torch.uint8, device="cuda")
    g.execute({tok: t, w: wt, fto: off, out: o}, ws)
    torch.cuda.synchronize()

    bounds = off.tolist() + [S]
    ref = torch.empty_like(o)
    for gi in range(E):
        lo, hi = bounds[gi], bounds[gi + 1]
        ref[0, lo:hi] = (t[0, lo:hi].float() @ wt[gi].float().T).to(torch.bfloat16)
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-1, rtol=2e-2)

    with pytest.raises(ValueError, match="needs a .*-byte workspace"):
        g.execute({tok: t, w: wt, fto: off, out: o}, torch.empty(16, dtype=torch.uint8, device="cuda"))


@_GPU
def test_build_convenience_then_select():
    """build() runs the whole sequence on the default entry; select_plan then
    re-pins FROST and build_plans compiles it."""
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    g.build([cudnn.heur_mode.A])
    _pin_frost(g)
    g.build_plans()
    assert g.selected_engine.name == _FROST
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


def test_ineligible_graph_not_listed():
    """fp32 matmul (no fp32 MMA path) is declined → no python entry at all."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, 64, 64],
        stride=[64 * 64, 64, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    B = g.tensor(
        name="B",
        dim=[1, 64, 64],
        stride=[64 * 64, 1, 64],
        data_type=cudnn.data_type.FLOAT,
    )
    C = g.matmul(A=A, B=B, name="mm", compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _plan(g)
    assert not any(is_python_engine(p.engine_id) for p in g.plans)


@_GPU
def test_wrapper_graph_path():
    """wrapper.Graph(heuristics=[A]) plans and executes the fused graph."""
    from cudnn.wrapper import Graph

    a, b, bias_t, ref = _operands()
    with Graph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        heuristics=[cudnn.heur_mode.A],
        handle="auto",
    ) as g:
        A = g.tensor(
            name="A",
            dim=[1, M, K],
            stride=[M * K, K, 1],
            data_type=cudnn.data_type.BFLOAT16,
        )
        B = g.tensor(
            name="B",
            dim=[1, K, N],
            stride=[K * N, 1, K],
            data_type=cudnn.data_type.BFLOAT16,
        )
        bias = g.tensor(
            name="bias",
            dim=[1, 1, N],
            stride=[N, N, 1],
            data_type=cudnn.data_type.BFLOAT16,
        )
        C = g.matmul(A=A, B=B, name="mm")
        Cb = g.bias(input=C, bias=bias, name="bs")
        Y = g.relu(input=Cb, name="r")
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g({A: a, B: b, bias: bias_t, Y: y})
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)
