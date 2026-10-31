# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed wrappers bind logical widths and caller strides without layout copies."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_bwd_staged_sm80 import _assert_kernel_count

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires SM80")]


def _pitched(tensor, index):
    _, tokens, heads, width = tensor.shape
    head_stride = width + 8 * (index + 1)
    token_stride = heads * head_stride + 16
    owner = torch.full((max(tokens, 1) * token_stride + 16,), float("nan"), dtype=tensor.dtype, device=tensor.device)
    view = owner.as_strided(tensor.shape, (0, token_stride, head_stride, 1), 8)
    view.copy_(tensor)
    return view


def _forbid_copy(monkeypatch):
    import builtins

    original = builtins.__import__

    def checked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cudnn.sdpa.packed_copy_sm80" or (name == "cudnn.sdpa" and "packed_copy_sm80" in (fromlist or ())):
            pytest.fail("native packed wrapper reached a pad/trim/compact data copy")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", checked)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv", [(48, 32), (64, 96), (96, 80), (192, 112), (224, 208)])
def test_forward_native_widths_strides_and_replay(d, dv, dtype, monkeypatch):
    from cudnn.sdpa.fwd import prepared_sm80_thd
    from test_sdpa_sm80_thd_forward_prepared import _check, _inputs, _prefix, _run

    lengths_q, lengths_k = (33, 0, 17), (41, 5, 23)
    tensors = tuple(_pitched(t, index) for index, t in enumerate(_inputs(d, dv, dtype, capq=64, capkv=80)))
    cq, ck = _prefix(lengths_q), _prefix(lengths_k)
    expected_ptrs = tuple(t.data_ptr() for t in tensors)
    original = prepared_sm80_thd.execute
    seen = []

    def execute(launch, buffers, *args):
        pointers = tuple(t.data_ptr() for t in buffers[:3])
        assert pointers == expected_ptrs
        seen.append(pointers)
        return original(launch, buffers, *args)

    monkeypatch.setattr(prepared_sm80_thd, "execute", execute)
    _forbid_copy(monkeypatch)
    kwargs = dict(causal=True, bottom=True, window=32, maxq=40)
    first = _run(tensors, cq, ck, **kwargs)
    _check(tensors, first, lengths_q, lengths_k, causal=True, bottom=True, window=32)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            output = _run(tensors, cq, ck, **kwargs)
        _assert_kernel_count(graph, 2)  # output initialization and attention
        for t in tensors:
            t.mul_(0.7)
        lengths_q, lengths_k = (17, 9, 23), (23, 7, 31)
        cq.copy_(_prefix(lengths_q))
        ck.copy_(_prefix(lengths_k))
        output["o_tensor"].fill_(float("nan"))
        graph.replay()
        _check(tensors, output, lengths_q, lengths_k, causal=True, bottom=True, window=32)
        assert len(seen) == 2
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv", [(64, 96), (16, 256)])
@pytest.mark.parametrize("packed", [False, True])
def test_backward_graph_independent_head_dimensions(d, dv, dtype, packed, monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    _forbid_copy(monkeypatch)
    if packed:
        from test_sdpa_bwd_thd_sm80 import _run_graph, _check

        case, graph, pack, workspace, outputs = _run_graph(
            (33, 0, 17), (41, 5, 23), h=4, hkv=2, d=d, d_v=dv, dtype=dtype, poison=True, pad_cap=16, use_causal_mask=True
        )

        def check():
            _check(case, *outputs)

    else:
        from test_sdpa_bwd_prepared_sm80 import _case, _check

        case = _case(d, dv, dtype=dtype, hk=2, layout="gapped", sq=33, skv=41)
        graph, pack, workspace = case.graph, case.pack, case.workspace
        outputs = tuple(case.bufs[n] for n in ("dq", "dk", "dv"))

        def check():
            _check(case)

    check()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(pack, workspace)
        for output in outputs:
            output.fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        check()
    finally:
        capture.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv", [(48, 32), (64, 96), (96, 80), (192, 112), (224, 208)])
def test_backward_native_widths_strides_and_replay(d, dv, dtype, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from test_sdpa_bwd_thd_sm80 import _thd_case
    from test_sdpa_bwd_prepared_thd_sm80 import _check
    from test_sdpa_sm80_thd_wrapper_prepared import _prefix, _wrapper

    def make(lq, lk, seed):
        case = _thd_case(lq, lk, 4, d, dtype, hkv=2, d_v=dv, cap_q=80, cap_kv=96, poison=True, causal=True, seed=seed)
        for index, role in enumerate(("q", "k", "v", "o", "do")):
            setattr(case, role, _pitched(getattr(case, role), index))
        h, tq = case.lse.shape[1:]
        hs = tq + 13
        owner = torch.full(((h - 1) * hs + tq,), float("nan"), device="cuda")
        case.lse = owner.as_strided((1, h, tq), (0, hs, 1)).copy_(case.lse)
        return case

    case = make((33, 0, 17), (41, 5, 23), 79)
    cq, ck = _prefix(case.cu_q), _prefix(case.cu_k)
    expected_ptrs = tuple(getattr(case, role).data_ptr() for role in ("q", "k", "v", "o", "do"))
    original = SdpaBwdDslSm80.execute
    seen = []

    def execute(api, *args, **kwargs):
        assert tuple(t.data_ptr() for t in args[:5]) == expected_ptrs
        seen.append(api)
        return original(api, *args, **kwargs)

    monkeypatch.setattr(SdpaBwdDslSm80, "execute", execute)
    _forbid_copy(monkeypatch)
    kwargs = dict(deterministic=True, max_s_q=40, max_s_kv=48)
    first = _wrapper(case, cq, ck, **kwargs)
    _check(case, *(first[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            output = _wrapper(case, cq, ck, **kwargs)
        # Combined output/accumulator init, packed metadata, delta, main,
        # dQ cast and two GQA reductions are the native algorithm chain.
        _assert_kernel_count(graph, 7)
        changed = make((17, 9, 23), (23, 7, 31), 97)
        for role in ("q", "k", "v", "o", "do", "lse"):
            getattr(case, role).copy_(getattr(changed, role))
        cq.copy_(_prefix(changed.cu_q))
        ck.copy_(_prefix(changed.cu_k))
        for value in output.values():
            value.fill_(float("nan"))
        graph.replay()
        _check(changed, *(output[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
        assert len(seen) == 2
        for key, live in (("dq_tensor", changed.t_q), ("dk_tensor", changed.t_kv), ("dv_tensor", changed.t_kv)):
            assert torch.count_nonzero(output[key][:, live:]) == 0
    finally:
        graph.reset()
