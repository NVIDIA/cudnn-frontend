# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native large-head backward layouts with no IO staging."""

from types import SimpleNamespace
from itertools import permutations
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import cudnn

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_bwd_dsl_sm100 import _reference, _causal_keep, _check_prepared, _io_dtype, _plan_index

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _case(*, dtype=torch.bfloat16, d=512, hkv=2, causal=True, roles=("q", "do", "dq", "dv"), b=2, h=4, orders=None, singleton_stride=None):
    from cudnn.sdpa.bwd import api_dsl

    torch.manual_seed(5142)
    sq, skv = 129, 143
    tensors, owners = {}, {}
    for name, heads, seq in (("q", h, sq), ("k", hkv, skv), ("v", hkv, skv), ("do", h, sq), ("o", h, sq), ("dq", h, sq), ("dk", hkv, skv), ("dv", hkv, skv)):
        width = d + 24 if name in roles else d
        shape = (b, heads, seq, width)
        order = (orders or {}).get(name, (0, 2, 1)) + (3,)
        owner = torch.full(tuple(shape[i] for i in order), 17, dtype=dtype, device="cuda")
        tensor = owner[..., 8 : d + 8] if name in roles else owner
        tensor = tensor.permute(*(order.index(i) for i in range(4)))
        if singleton_stride is not None:
            strides = tuple(singleton_stride if n == 1 and i < 3 else st for i, (n, st) in enumerate(zip(tensor.shape, tensor.stride())))
            tensor = tensor.as_strided(tensor.shape, strides)
        if name in ("q", "k", "v", "do"):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
        tensors[name], owners[name] = tensor, owner
    keep = _causal_keep(sq, skv) if causal else None
    o, stats, _, dq, dk, dv = _reference(*(tensors[n] for n in ("q", "k", "v", "do")), keep, h // hkv)
    tensors["o"].copy_(o)
    tensors["stats"] = stats.unsqueeze(-1).contiguous()
    graph = cudnn.pygraph(io_data_type=_io_dtype(dtype), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    refs = {
        n: graph.tensor(
            name=n, dim=list(tensors[n].shape), stride=list(tensors[n].stride()), data_type=cudnn.data_type.FLOAT if n == "stats" else _io_dtype(dtype)
        )
        for n in ("q", "k", "v", "o", "do", "stats")
    }
    outputs = graph.sdpa_backward(
        q=refs["q"], k=refs["k"], v=refs["v"], o=refs["o"], dO=refs["do"], stats=refs["stats"], attn_scale=d**-0.5, use_causal_mask=causal
    )
    for name, out in zip(("dq", "dk", "dv"), outputs):
        out.set_output(True).set_data_type(_io_dtype(dtype)).set_stride(list(tensors[name].stride()))
        refs[name] = out
    with patch.object(api_dsl, "_sm100_head_chunk", side_effect=lambda *a, group=1, **kw: group):
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        index = _plan_index(graph)
        assert index is not None
        graph.select_plan(index)
        graph.check_support()
        graph.build_plans()
    size = graph.get_workspace_size()
    ws_owner = torch.full((size + 513,), 0xA7, dtype=torch.uint8, device="cuda")
    workspace = ws_owner[256 : 256 + size + 1]
    pack = {refs[n]: value for n, value in tensors.items()}
    return SimpleNamespace(
        graph=graph,
        refs=refs,
        tensors=tensors,
        owners=owners,
        roles=roles,
        pack=pack,
        workspace=workspace,
        workspace_owner=ws_owner,
        keep=keep,
        group=h // hkv,
        expected=(dq, dk, dv),
    )


def _check(case):
    _check_prepared(case)
    for name in case.roles:
        owner = case.owners[name]
        assert torch.all(owner[..., :8] == 17)
        assert torch.all(owner[..., -16:] == 17)
    assert torch.all(case.workspace_owner[:256] == 0xA7)
    assert torch.all(case.workspace_owner[-256:] == 0xA7)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("d,hkv,causal", [(264, 4, False), (384, 2, True), (512, 1, True)])
@pytest.mark.parametrize("roles", [("q", "do", "dq", "dv"), ("k", "v", "o", "dk")])
def test_native_rebind_stream_capture(dtype, d, hkv, causal, roles, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import prepared

    monkeypatch.setattr(prepared, "_bind_python", lambda *a, **k: pytest.fail("half native backward entered the Python binder"))

    def forbidden(*args, **kwargs):
        raise AssertionError("native backward used the legacy tensor compiler or DLPack")

    with monkeypatch.context() as guards:
        guards.setattr(cute.runtime, "make_fake_compact_tensor", forbidden)
        guards.setattr(cute.runtime, "from_dlpack", forbidden)
        case = _case(dtype=dtype, d=d, hkv=hkv, causal=causal, roles=roles)
        guards.setattr(cute, "compile", forbidden)
        case.graph.execute(case.pack, case.workspace)
        _check(case)
        # Rebind the same graph to new allocations without changing its layout.
        for name, owner in tuple(case.owners.items()):
            old = case.tensors[name]
            owner = owner.clone()
            case.owners[name] = owner
            case.tensors[name] = owner.as_strided(old.shape, old.stride(), old.storage_offset())
        case.tensors["stats"] = case.tensors["stats"].clone()
        case.workspace_owner = case.workspace_owner.clone()
        size = case.graph.get_workspace_size()
        case.workspace = case.workspace_owner[256 : 256 + size + 1]
        case.pack = {case.refs[n]: value for n, value in case.tensors.items()}
        case.graph.execute(case.pack, case.workspace)
        _check(case)
        handle = cudnn.create_handle()
        launch, other = torch.cuda.Stream(), torch.cuda.Stream()
        cudnn.set_stream(handle, launch.cuda_stream)
        capture = torch.cuda.CUDAGraph()
        try:
            launch.wait_stream(torch.cuda.current_stream())
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture, stream=launch):
                with torch.cuda.stream(other):
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        case.graph.execute(case.pack, case.workspace, handle=handle)
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
            case.tensors["q"].mul_(0.75)
            case.tensors["do"].mul_(1.25)
            o, stats, _, *grads = _reference(*(case.tensors[n] for n in ("q", "k", "v", "do")), case.keep, case.group)
            case.tensors["o"].copy_(o)
            case.tensors["stats"].copy_(stats.unsqueeze(-1))
            case.expected = grads
            for name in ("dq", "dk", "dv"):
                case.tensors[name].fill_(float("nan"))
            case.workspace.fill_(0xBD)
            capture.replay()
            _check(case)
        finally:
            capture.reset()
            cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_native_standalone_and_runtime_layout(dtype, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl, prepared, prepared_sm100

    monkeypatch.setattr(prepared, "_bind_python", lambda *a, **k: pytest.fail("half standalone backward entered the Python binder"))

    case = _case(dtype=dtype)
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=512**-0.5)
    api.compile()
    assert api._prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    args = {name + "_tensor": value for name, value in case.tensors.items()}
    api.execute(**args, workspace=workspace)
    _check_prepared(case)
    wrong = dict(args, q_tensor=case.tensors["q"].contiguous())
    with monkeypatch.context() as guards:
        api._prepared = replace(api._prepared, fn=lambda *a: pytest.fail("invalid metadata reached the kernel"), native_binding=True)
        with pytest.raises(ValueError, match="geometry|strides"):
            api.execute(**wrong, workspace=workspace)
    launches = []
    monkeypatch.setattr(prepared_sm100, "execute_standalone", lambda *a: launches.append(a))
    with pytest.raises(ValueError, match="padding masks"):
        api.execute(**args, workspace=workspace, seq_q_lens=torch.ones(2, dtype=torch.int32, device="cuda"))
    assert not launches


@pytest.mark.parametrize("unaligned", [False, True])
def test_sm100_direct_adapter_declines_old_dsl(unaligned, monkeypatch):
    from cudnn.frost import buffers
    from cudnn.sdpa.bwd import api_dsl

    q = torch.empty(2, 4, 128, 515 if unaligned else 512, device="cuda", dtype=torch.bfloat16)[..., :512]
    stats = torch.empty(2, 4, 128, device="cuda", dtype=torch.float32)
    api = api_dsl.SdpaBwdDslSm100(q, q, q, q, q, stats, q, q, q)
    monkeypatch.setattr(buffers, "cutedsl_requirement_error", lambda name: f"{name} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    monkeypatch.setattr(api_dsl, "load_template", lambda *a, **k: pytest.fail("old DSL reached a kernel template"))
    with pytest.raises(NotImplementedError, match=r"SdpaBwdDslSm100.*4\.7\.0.*4\.6\.2"):
        api.compile()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("explicit_stream", (False, True))
@pytest.mark.parametrize("adapter", ("sm100", "sm107"))
def test_direct_standalone_with_another_device_current(dtype, explicit_stream, adapter, monkeypatch):
    """Both tensor adapters must select the plan device before entering its host."""
    from cuda.bindings import driver
    from cudnn.sdpa.bwd import api_dsl, prepared, prepared_sm100, prepared_sm107

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    target = torch.cuda.current_device()
    other = next(i for i in range(torch.cuda.device_count()) if i != target)
    case = _case(dtype=dtype, roles=())
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=512**-0.5)
    api.compile()
    assert api._prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=target)
    args = {name + "_tensor": value for name, value in case.tensors.items()}
    # The SM107 adapter consumes the same prepared spec contract. Exercise its
    # device/stream boundary against a real SM100 host, independently of ISA.
    if adapter == "sm107":
        monkeypatch.setattr(prepared_sm100, "execute_standalone", prepared_sm107.execute_standalone)
    monkeypatch.setattr(prepared, "_bind_python", lambda *a, **k: pytest.fail("half backward entered the Python binder"))
    real_execute = prepared.execute
    expected_stream = torch.cuda.default_stream(target).cuda_stream

    def checked_execute(*a, **kw):
        assert torch.cuda.current_device() == target, "direct backward launch must run in the operand context"
        assert a[3] == expected_stream, "implicit stream must belong to the operand device"
        return real_execute(*a, **kw)

    monkeypatch.setattr(prepared, "execute", checked_execute)

    def run():
        with torch.cuda.device(other):
            api.execute(**args, workspace=workspace, current_stream=driver.CUstream(expected_stream) if explicit_stream else None)
            assert torch.cuda.current_device() == other

    # Include the default-stream sentinel: a foreign current device makes its
    # raw handle select the wrong context even though tensor facts are valid.
    with torch.cuda.stream(torch.cuda.default_stream(target)):
        run()
    torch.cuda.synchronize(target)
    _check_prepared(case)
    stream = torch.cuda.Stream(device=target)
    stream.wait_stream(torch.cuda.current_stream(target))
    expected_stream = stream.cuda_stream
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream(target).wait_stream(stream)
    _check_prepared(case)
    captured = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream(target))
    with torch.cuda.graph(captured, stream=stream):
        run()
    torch.cuda.current_stream(target).wait_stream(stream)
    case.tensors["q"].mul_(0.75)
    case.tensors["do"].mul_(1.25)
    o, stats, _, *grads = _reference(*(case.tensors[n] for n in ("q", "k", "v", "do")), case.keep, case.group)
    case.tensors["o"].copy_(o)
    case.tensors["stats"].copy_(stats.unsqueeze(-1))
    case.expected = grads
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    captured.replay()
    _check_prepared(case)
    captured.reset()
    args["k_tensor"] = case.tensors["k"].float()
    with torch.cuda.stream(stream), torch.cuda.device(other):
        with pytest.raises(ValueError, match="k must"):
            run()
        assert torch.cuda.current_device() == other


def _kernel_count(case):
    from cuda.bindings import runtime as cudart

    capture = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        graph = cudart.cudaGraph_t(capture.raw_cuda_graph())
        error, _, count = cudart.cudaGraphGetNodes(graph, 0)
        assert error == cudart.cudaError_t.cudaSuccess
        error, nodes, _ = cudart.cudaGraphGetNodes(graph, count)
        assert error == cudart.cudaError_t.cudaSuccess
        kinds = [cudart.cudaGraphNodeGetType(node) for node in nodes]
        assert all(error == cudart.cudaError_t.cudaSuccess for error, _ in kinds)
        capture.replay()
        _check(case)
        return sum(kind == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel for _, kind in kinds)
    finally:
        capture.reset()


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_independent_orders_bind_original_pointers(order, dtype, monkeypatch):
    from cudnn.sdpa.bwd import prepared_sm100

    roles = ("q", "k", "v", "o", "do", "dq", "dk", "dv")
    all_orders = list(permutations(range(3)))
    orders = {role: all_orders[(all_orders.index(order) + i) % 6] for i, role in enumerate(roles)}
    control = _case(d=264, dtype=dtype, roles=())
    expected_kernels = _kernel_count(control)
    frames = []
    original = prepared_sm100.compile_plan

    def compile_plan(*args):
        spec = original(*args)

        def launch(*frame):
            frames.append(frame)
            return spec.fn(*frame)

        return replace(spec, fn=launch, native_binding=True)

    monkeypatch.setattr(prepared_sm100, "compile_plan", compile_plan)
    case = _case(d=264, dtype=dtype, roles=roles, orders=orders)
    assert case.graph.get_workspace_size() == control.graph.get_workspace_size()
    assert _kernel_count(case) == expected_kernels
    assert frames
    assert tuple(frames[-1][:9]) == tuple(case.tensors[n].data_ptr() for n in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"))


@pytest.mark.parametrize("b,h,hkv", [(1, 4, 2), (2, 1, 1), (1, 1, 1)])
@pytest.mark.parametrize("unused", [0, 3, 2**33 + 1])
def test_singleton_pitches_execute_and_replay(b, h, hkv, unused):
    case = _case(d=264, b=b, h=h, hkv=hkv, singleton_stride=unused)
    _kernel_count(case)
