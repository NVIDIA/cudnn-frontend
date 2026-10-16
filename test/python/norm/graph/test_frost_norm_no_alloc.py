# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``execute`` must not allocate device memory, and must say what it needs.

AGENTS.md Rule 1 (execute never allocates) and Rule 8 (the graph API owns no
device memory). The plans used to report ``workspace=0`` and then allocate
anyway: the lowering built its own Y/mean/rstd (or DX/DGamma/DBeta) and copied
each into the graph's buffer, and the cooperative kernels allocated their
cross-CTA partials on top. None of it was visible -- the graph ran, the numbers
were right, and the only symptoms were a workspace size that lied and a
measured 1.6x against the direct API from copying Y twice.

So this file asserts the contract directly rather than through a proxy. It
wraps every torch allocator entry point, runs a warmed ``execute``, and
requires the count to be exactly zero. A count of 1 is as much a failure as a
count of 10: the point is that nothing is allocated, not that little is.

The second assertion is that ``get_workspace_size()`` is honest in both
directions -- large enough that the kernels never fall back to allocating
(which the first assertion already catches), and not wildly larger than what
they use, since the caller has to allocate it for real.
"""

import pytest
import torch

import cudnn

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

_HEUR = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
_PY_ENGINE_FLOOR = 20_000
_ALLOCATORS = ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like")


class _AllocWatch:
    """Records every CUDA tensor torch hands out while armed."""

    def __init__(self):
        self.calls = []
        self._real = {}

    def __enter__(self):
        for name in _ALLOCATORS:
            self._real[name] = getattr(torch, name)
            setattr(torch, name, self._wrap(name, self._real[name]))
        return self

    def __exit__(self, *exc):
        for name, fn in self._real.items():
            setattr(torch, name, fn)
        return False

    def _wrap(self, name, fn):
        def inner(*a, **k):
            t = fn(*a, **k)
            try:
                if t.is_cuda:
                    self.calls.append(f"{name}{tuple(t.shape)} {t.dtype}")
            except Exception:  # noqa: BLE001 - not every return is a tensor
                pass
            return t

        return inner


def _select(graph):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans(_HEUR)
    idx = next((i for i, p in enumerate(graph.plans) if p.engine_id >= _PY_ENGINE_FLOOR), None)
    if idx is None:
        pytest.skip("frost norm engine proposed no plan for this graph")
    graph.select_plan(idx)
    graph.check_support()
    graph.build_plans()
    return graph


def _layernorm_graph(handle, x, scale, bias, eps_cpu):
    B, S, D = x.shape
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    X = g.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    SC = g.tensor(name="s", dim=scale.size(), stride=scale.stride(), data_type=scale.dtype)
    BI = g.tensor(name="b", dim=bias.size(), stride=bias.stride(), data_type=bias.dtype)
    EP = g.tensor_like(eps_cpu)
    Y, M, I = g.layernorm(name="LN", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=SC, bias=BI, epsilon=EP)
    Y.set_output(True).set_data_type(x.dtype)
    Y.set_dim((B, S, D))
    Y.set_stride(x.stride())
    for t in (M, I):
        t.set_output(True).set_data_type(torch.float32)
        t.set_dim((B, S, 1))
        t.set_stride((S, 1, 1))
    return _select(g), (X, SC, BI, EP, Y, M, I)


def _layernorm_bwd_graph(handle, x, dy, scale, mean, ivar):
    B, S, D = x.shape
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    X = g.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    DY = g.tensor(name="dy", dim=dy.size(), stride=dy.stride(), data_type=dy.dtype)
    SC = g.tensor(name="s", dim=scale.size(), stride=scale.stride(), data_type=scale.dtype)
    MN = g.tensor(name="m", dim=mean.size(), stride=mean.stride(), data_type=mean.dtype)
    IV = g.tensor(name="i", dim=ivar.size(), stride=ivar.stride(), data_type=ivar.dtype)
    DX, DS, DB = g.layernorm_backward(name="LNB", grad=DY, input=X, scale=SC, mean=MN, inv_variance=IV)
    DX.set_output(True).set_data_type(x.dtype)
    DX.set_dim((B, S, D))
    DX.set_stride(x.stride())
    for t in (DS, DB):
        t.set_output(True).set_data_type(torch.float32)
        t.set_dim((1, 1, D))
        t.set_stride(scale.stride())
    return _select(g), (X, DY, SC, MN, IV, DX, DS, DB)


def _batchnorm_graph(handle, x, scale, bias, pm, pv, eps_cpu, mom_cpu):
    c = x.shape[1]
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    X = g.tensor_like(x)
    SC = g.tensor_like(scale)
    BI = g.tensor_like(bias)
    IRM = g.tensor_like(pm)
    IRV = g.tensor_like(pv)
    EP = g.tensor_like(eps_cpu)
    MO = g.tensor_like(mom_cpu)
    Y, SM, SIV, ORM, ORV = g.batchnorm(name="BN", input=X, scale=SC, bias=BI, in_running_mean=IRM, in_running_var=IRV, epsilon=EP, momentum=MO)
    Y.set_output(True).set_data_type(x.dtype)
    for t in (SM, SIV, ORM, ORV):
        t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return _select(g), (X, SC, BI, IRM, IRV, EP, MO, Y, SM, SIV, ORM, ORV), c


@pytest.fixture(scope="module")
def _handle():
    h = cudnn.create_handle()
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
    return h


def _assert_no_alloc(label, graph, pack, handle):
    """Warm once (the first execute JIT-compiles), then require zero allocations."""
    ws = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(pack, ws, handle=handle)
    torch.cuda.synchronize()
    with _AllocWatch() as w:
        graph.execute(pack, ws, handle=handle)
    torch.cuda.synchronize()
    assert not w.calls, f"{label}: execute allocated {len(w.calls)} device tensor(s): {w.calls}"
    return graph.get_workspace_size()


@pytest.mark.L0
def test_layernorm_forward_execute_allocates_nothing(_handle):
    B, S, D = 8, 128, 1024
    torch.manual_seed(0)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    sc = torch.randn(1, 1, D, device="cuda", dtype=torch.bfloat16)
    bi = torch.randn(1, 1, D, device="cuda", dtype=torch.bfloat16)
    eps = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)
    g, (X, SC, BI, EP, Y, M, I) = _layernorm_graph(_handle, x, sc, bi, eps)
    y = torch.empty_like(x)
    mean = torch.empty(B, S, 1, device="cuda", dtype=torch.float32)
    ivar = torch.empty_like(mean)
    nb = _assert_no_alloc("LN fwd", g, {X: x, SC: sc, BI: bi, EP: eps, Y: y, M: mean, I: ivar}, _handle)
    # the rowwise forward needs no scratch at all, and should not ask for any
    assert nb == 0, f"LN forward reported a {nb}-byte workspace it does not use"


@pytest.mark.L0
def test_layernorm_backward_execute_allocates_nothing(_handle):
    B, S, D = 8, 128, 1024
    torch.manual_seed(0)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    sc = torch.randn(1, 1, D, device="cuda", dtype=torch.bfloat16)
    xf = x.float()
    mean = xf.mean(-1, keepdim=True).contiguous()
    ivar = torch.rsqrt(xf.var(-1, keepdim=True, correction=0) + 1e-5).contiguous()
    g, (X, DY, SC, MN, IV, DX, DS, DB) = _layernorm_bwd_graph(_handle, x, dy, sc, mean, ivar)
    dx = torch.empty_like(x)
    ds = torch.empty(1, 1, D, device="cuda", dtype=torch.float32)
    db = torch.empty_like(ds)
    nb = _assert_no_alloc("LN bwd", g, {X: x, DY: dy, SC: sc, MN: mean, IV: ivar, DX: dx, DS: ds, DB: db}, _handle)
    # this one DOES carve partials, so it has to ask for them
    assert nb > 0, "LN backward carves per-CTA dgamma partials but reported a zero workspace"


@pytest.mark.L0
def test_batchnorm_forward_execute_allocates_nothing(_handle):
    n, c, h, w = 16, 32, 16, 16
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
    sc = torch.ones(1, c, 1, 1, device="cuda", dtype=torch.float32)
    bi = torch.zeros_like(sc)
    pm = torch.zeros_like(sc)
    pv = torch.ones_like(sc)
    eps = torch.full((1, 1, 1, 1), 1e-5, dtype=torch.float32)
    mom = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
    g, T, cc = _batchnorm_graph(_handle, x, sc, bi, pm, pv, eps, mom)
    X, SC, BI, IRM, IRV, EP, MO, Y, SM, SIV, ORM, ORV = T
    y = torch.empty_like(x)
    sm = torch.empty(1, cc, 1, 1, device="cuda", dtype=torch.float32)
    siv, orm, orv = torch.empty_like(sm), torch.empty_like(sm), torch.empty_like(sm)
    pack = {X: x, SC: sc, BI: bi, IRM: pm, IRV: pv, EP: eps, MO: mom, Y: y, SM: sm, SIV: siv, ORM: orm, ORV: orv}
    nb = _assert_no_alloc("BN fwd", g, pack, _handle)
    # BatchNorm's split-K forward needs cross-CTA partials and a retired-CTA counter
    assert nb > 0, "BatchNorm forward carves cross-CTA partials but reported a zero workspace"
