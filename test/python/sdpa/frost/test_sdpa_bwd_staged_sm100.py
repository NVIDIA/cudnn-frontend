# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The retained dense conversion layouts share the prepared large-head chain."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import cudnn

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_bwd_dsl_sm100 import _reference, _causal_keep, _check_prepared, _io_dtype, _plan_index

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _case(*, dtype=torch.bfloat16, d=512, hkv=2, causal=True, roles=("q", "do", "dq", "dv")):
    from cudnn.sdpa.bwd import api_dsl

    torch.manual_seed(5142)
    b, h, sq, skv = 2, 4, 129, 143
    tensors, owners = {}, {}
    for name, heads, seq in (("q", h, sq), ("k", hkv, skv), ("v", hkv, skv), ("do", h, sq), ("o", h, sq), ("dq", h, sq), ("dk", hkv, skv), ("dv", hkv, skv)):
        width = d + 3 if name in roles else d
        owner = torch.full((b, seq, heads, width), 17, dtype=dtype, device="cuda")
        tensor = owner[..., 1 : d + 1] if name in roles else owner
        tensor = tensor.permute(0, 2, 1, 3)
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
        assert torch.all(owner[..., :1] == 17)
        assert torch.all(owner[..., -2:] == 17)
    assert torch.all(case.workspace_owner[:256] == 0xA7)
    assert torch.all(case.workspace_owner[-256:] == 0xA7)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("d,hkv,causal", [(264, 4, False), (384, 2, True), (512, 1, True)])
@pytest.mark.parametrize("roles", [("q", "do", "dq", "dv"), ("k", "v", "o", "dk")])
def test_staged_rebind_stream_capture(dtype, d, hkv, causal, roles, monkeypatch):
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("staged backward used the legacy tensor compiler or DLPack")

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
def test_staged_standalone_and_runtime_layout(dtype, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl, prepared_sm100

    case = _case(dtype=dtype)
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=512**-0.5)
    api.compile()
    assert api._prepared is None and api._staged_prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    args = {name + "_tensor": value for name, value in case.tensors.items()}
    api.execute(**args, workspace=workspace)
    _check_prepared(case)
    wrong = dict(args, q_tensor=case.tensors["q"].contiguous())
    with monkeypatch.context() as guards:
        guards.setattr(prepared_sm100, "_copy", lambda *a, **k: pytest.fail("invalid metadata reached a prepared staging write"))
        with pytest.raises(ValueError, match="declared shape and strides"):
            api.execute(**wrong, workspace=workspace)
    launches = []
    monkeypatch.setattr(prepared_sm100, "execute_staged", lambda *a: launches.append(a))
    with pytest.raises(ValueError, match="padding masks"):
        api.execute(**args, workspace=workspace, seq_q_lens=torch.ones(2, dtype=torch.int32, device="cuda"))
    assert not launches


@pytest.mark.parametrize("staged", [False, True])
def test_sm100_direct_adapter_declines_old_dsl(staged, monkeypatch):
    from cudnn.frost import buffers
    from cudnn.sdpa.bwd import api_dsl

    q = torch.empty(2, 4, 128, 515 if staged else 512, device="cuda", dtype=torch.bfloat16)[..., :512]
    stats = torch.empty(2, 4, 128, device="cuda", dtype=torch.float32)
    api = api_dsl.SdpaBwdDslSm100(q, q, q, q, q, stats, q, q, q)
    monkeypatch.setattr(buffers, "cutedsl_requirement_error", lambda name: f"{name} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    monkeypatch.setattr(api_dsl, "load_template", lambda *a, **k: pytest.fail("old DSL reached a kernel template"))
    with pytest.raises(NotImplementedError, match=r"SdpaBwdDslSm100.*4\.7\.0.*4\.6\.2"):
        api.compile()
