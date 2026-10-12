# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared packed SM80 backward: current lengths, pointers and address widths."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from sdpa.frost.test_sdpa_bwd_thd_sm80 import _build_thd_bwd_graph, _gap_columns, _gapped, _plan_graph, _ref_bwd, _run_graph, _thd_case
from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _wide_buffer

pytestmark = [requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires SM80")]


def _prepared(graph):
    plan = graph._compiled_plans[graph._plan_index]
    assert plan._prepared is not None
    return plan._prepared.spec


def _check(case, *grads):
    """Check magnitude as well as direction against per-sequence FP64 gradients."""
    group = case.h // case.hkv
    for i, (nq, nk) in enumerate(zip(case.lens_q, case.lens_kv)):
        qs = slice(case.cu_q[i], case.cu_q[i] + nq)
        ks = slice(case.cu_k[i], case.cu_k[i] + nk)
        got = [x[0, sl].transpose(0, 1).double() for x, sl in zip(grads, (qs, ks, ks))]
        if nq == 0 or nk == 0:
            for value in got:
                torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
            continue
        q, k, v, do = [x[0, sl].transpose(0, 1) for x, sl in zip((case.q, case.k, case.v, case.do), (qs, ks, ks, qs))]
        expected = list(_ref_bwd(q, k.repeat_interleave(group, 0), v.repeat_interleave(group, 0), do, case.scale, causal=case.causal))
        for j in (1, 2):
            expected[j] = expected[j].reshape(case.hkv, group, *expected[j].shape[1:]).sum(1)
        for value, want in zip(got, expected):
            # GQA dV sums query-head groups and can exceed unit magnitude.
            # Scale the absolute allowance with the reference, retaining the
            # independent relative-norm guard against a magnitude regression.
            atol = 0.003 * max(1.0, float(want.abs().max()))
            torch.testing.assert_close(value, want, atol=atol, rtol=0.03)
            assert torch.linalg.vector_norm(value - want) <= 0.03 * torch.linalg.vector_norm(want) + 1e-7


def _pack(case, stats_layout, deterministic):
    gaps = {role: 8 for role in ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
    graph, pack, outputs = _build_thd_bwd_graph(case, stats_layout=stats_layout, gaps=gaps, use_deterministic_algorithm=deterministic)
    grads = tuple(_gapped(torch.full_like(x, float("nan")), 8) for x in (case.q, case.k, case.v))
    pack.update(zip(outputs, grads))
    return graph, pack, grads


def _check_tail(case, grads):
    _check(case, *grads)
    for grad, live in zip(grads, (case.t_q, case.t_kv, case.t_kv)):
        assert torch.isfinite(grad[0, :live]).all()
        assert torch.isnan(grad[0, live:]).all()
        assert torch.isnan(_gap_columns(grad)).all()


@pytest.mark.L0
@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
def test_changed_lengths_rebind_and_replay(d, dv, stats_layout):
    cases = [
        _thd_case(ql, kl, 4, d, torch.bfloat16, hkv=2, d_v=dv, cap_q=80, cap_kv=96, poison=True, seed=seed)
        for ql, kl, seed in [((33, 17, 0), (25, 41, 0), 7), ((11, 31, 0), (0, 19, 31), 8), ((7, 9, 21), (15, 0, 21), 9)]
    ]
    graph, pack, grads = _pack(cases[0], stats_layout, True)
    _plan_graph(graph)
    _prepared(graph)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xBD)
    graph.execute(pack, workspace)
    _check_tail(cases[0], grads)
    _, second, second_grads = _pack(cases[1], stats_layout, True)
    second_names = {ref.get_name(): buf for ref, buf in second.items()}
    bindings = {ref: second_names[ref.get_name()] for ref in pack}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.execute(bindings, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    _check_tail(cases[1], second_grads)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(bindings, workspace)
        _, third, _ = _pack(cases[2], stats_layout, True)
        third_names = {ref.get_name(): buf for ref, buf in third.items()}
        for ref, buf in bindings.items():
            buf.copy_(third_names[ref.get_name()])
        workspace.fill_(0xBD)
        capture.replay()
        _check_tail(cases[2], second_grads)
    finally:
        capture.reset()


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
@pytest.mark.parametrize("hkv", [1, 2])
def test_no_tensor_plumbing(stats_layout, hkv, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import api_dsl

    case, graph, pack, workspace, grads = _run_graph((33, 17), (25, 41), h=2, hkv=hkv, stats_layout=stats_layout)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared THD backward reconstructed tensors, allocated or compiled")

    with monkeypatch.context() as patch:
        for name in ("view", "reshape", "as_strided", "transpose", "permute", "contiguous", "copy_", "zero_"):
            patch.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(api_dsl, "_fd_tvm", forbidden, raising=False)
        patch.setattr(cute, "compile", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            graph.execute(pack, workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _prepared(graph)
    _check(case, *grads)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("d", [128, 256])
def test_physical_packed_token_stride(role, product, d):
    lengths = (5,) if product else (2, 2)
    case = _thd_case(lengths, lengths, 2, d, torch.bfloat16, hkv=1)
    graph, pack, outputs = _build_thd_bwd_graph(case)
    grads = [torch.full_like(x, float("nan")) for x in (case.q, case.k, case.v)]
    pack.update(zip(outputs, grads))
    ports = dict(graph._thd_test_ports, **dict(zip(("dq", "dk", "dv"), outputs)))
    ref = ports[role]
    wide, backing = _wide_buffer(pack[ref], product=product, axis=1)
    pack[ref] = wide
    ts, dim = wide.stride(1), wide.shape[-1]
    ref.set_stride([max(lengths) * ts, dim, ts, 1])
    # Ragged offsets remain consistent with the port declaration, even though
    # this engine derives contiguous sequence prefixes from lengths on device.
    ro_name = role + "_ro" if role not in ("dq", "dk", "dv") else ref.get_name() + "_ro"
    for ro in list(pack):
        if ro.get_name() == ro_name:
            cu = case.cu_k if role in ("k", "v", "dk", "dv") else case.cu_q
            pack[ro] = (torch.tensor(cu, dtype=torch.int64, device="cuda") * ts).reshape(len(lengths) + 1, 1, 1, 1)
    if role in ("dq", "dk", "dv"):
        grads[("dq", "dk", "dv").index(role)] = wide
    _plan_graph(graph)
    _prepared(graph)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xBD)
    graph.execute(pack, workspace)
    _check(case, *grads)
    assert backing.numel() >= wide.numel()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("d", [128, 256])
def test_physical_packed_stats_head_stride(product, d):
    heads = 5 if product else 2
    case = _thd_case((2, 2), (2, 2), heads, d, torch.bfloat16, hkv=1)
    graph, pack, outputs = _build_thd_bwd_graph(case)
    grads = tuple(torch.full_like(x, float("nan")) for x in (case.q, case.k, case.v))
    pack.update(zip(outputs, grads))
    stats = next(ref for ref in pack if ref.get_name() == "stats")
    compact = pack[stats].reshape(1, heads, -1)
    # One extra guard head provides the full declared final-head extent without
    # initializing the enormous unused gaps. Narrowed products stay in storage.
    padded = torch.cat((compact, torch.zeros_like(compact[:, :1])), dim=1)
    wide, backing = _wide_buffer(padded, product=product, axis=1)
    stride = wide.stride(1)
    stats.set_stride([heads * stride, stride, 1, 1])
    pack[stats] = backing.as_strided((heads * stride,), (1,), wide.storage_offset())
    _plan_graph(graph)
    _prepared(graph)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xBD)
    graph.execute(pack, workspace)
    _check(case, *grads)
