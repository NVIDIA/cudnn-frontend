# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 D128 MHA packed backward: pinned native routing and CPU FP64 oracle."""

import pytest
import torch

import cudnn
from d128_thd_test_utils import (
    GRADS,
    ROLES,
    adapter_arguments,
    check_case,
    copy_case_storage,
    make_adapter,
    make_case,
    make_graph,
    make_workspace,
    poison,
    variant_pack,
)
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM not in (120, 121), reason="requires SM120 or SM121")]


def _graph_case(case):
    graph, refs = make_graph(case, engine_name="sdpa_bwd_sm120", engine_id=20600)
    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    assert spec.name == "sdpa_bwd_sm80"  # The shared native ABI retains its output-alias guards.
    assert len(spec.roles) == 24 and spec.native is not None
    return graph, refs, variant_pack(case, refs), make_workspace(graph.get_workspace_size())


def _adapter_arguments(case):
    # The shared mma.sync standalone adapter consumes logical BHSD views;
    # graph bindings remain raw caller storage under the graph declaration.
    args = adapter_arguments(case)
    for role in ROLES:
        if role != "stats":
            args[role + "_tensor"] = case.tensors[role].transpose(1, 2)
    return args


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_layout", ["TH1", "HN"])
def test_frost_thd_standalone(dtype, stats_layout):
    from cudnn.sdpa.bwd.api_dsl_sm120_thd import SdpaBwdDslSm120Thd

    case = make_case((65, 0, 3), (17, 129, 0), dtype=dtype, stats_layout=stats_layout)
    api = make_adapter(case, api_type=SdpaBwdDslSm120Thd)
    assert api._params.qo_stages == 1 and api._params.dq_smem_coalesce is False
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    api.execute(**_adapter_arguments(case), workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize(
    "lens_q,lens_kv,dtype,causal",
    [
        ((1, 63, 64, 65, 127, 128, 129), (129, 128, 127, 65, 64, 63, 1), torch.bfloat16, False),
        ((129, 0, 65, 1), (3, 257, 0, 129), torch.float16, True),
        ((0, 0), (1, 129), torch.float16, False),
        ((1, 129), (0, 0), torch.bfloat16, True),
        ((0, 0), (0, 0), torch.float16, False),
    ],
)
@pytest.mark.parametrize("stats_layout", ["TH1", "HN"])
def test_frost_thd_independent_origins(lens_q, lens_kv, dtype, causal, stats_layout):
    envelope = (128, 128) if not any(lens_q) and not any(lens_kv) else None
    case = make_case(lens_q, lens_kv, dtype=dtype, causal=causal, stats_layout=stats_layout, envelope=envelope)
    assert len(set(case.origins.values())) == len(ROLES)
    graph, _, pack, workspace = _graph_case(case)
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)


@pytest.mark.parametrize("stats_layout,dtype", [("TH1", torch.bfloat16), ("HN", torch.float16)])
def test_frost_thd_rebind_and_retained_capture(stats_layout, dtype):
    options = dict(dtype=dtype, stats_layout=stats_layout, envelope=(257, 257), capacities=(1024, 1024))
    old = make_case((129, 0, 65), (3, 257, 0), **options)
    fresh = make_case((1, 127, 0), (129, 0, 65), seed=151, layout_shift=7, **options)
    changed = make_case((0, 193, 1), (17, 0, 129), seed=153, layout_shift=13, **options)
    graph, refs, pack, workspace = _graph_case(old)
    fresh_workspace = make_workspace(graph.get_workspace_size())
    stream, ambient = torch.cuda.Stream(), torch.cuda.Stream()
    handle, capture = cudnn.create_handle(), torch.cuda.CUDAGraph()
    cudnn.set_stream(handle, stream.cuda_stream)
    try:
        poison(old, workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ambient):
            graph.execute(pack, workspace.tensor, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        check_case(old, workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(ambient):
                graph.execute(pack, workspace.tensor, handle=handle)
        poison(fresh, fresh_workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ambient):
            graph.execute(variant_pack(fresh, refs), fresh_workspace.tensor, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        check_case(fresh, fresh_workspace)
        snapshots = {r: fresh.tensors[r].clone() for r in GRADS}
        copy_case_storage(old, changed)
        for _ in range(3):
            poison(old, workspace)
            capture.replay()
            check_case(old, workspace)
        for role in GRADS:
            torch.testing.assert_close(fresh.tensors[role], snapshots[role], rtol=0, atol=0)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


def test_frost_thd_hot_path_has_five_native_launches(monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import prepared

    case = make_case((129, 0, 65), (3, 257, 0), stats_layout="HN")
    graph, _, pack, workspace = _graph_case(case)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]

    def forbidden(*args, **kwargs):
        raise AssertionError("packed execute allocated, read device data, rebuilt a tensor or compiled")

    sync = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            case.lengths["seq_q"][0].item()
        with monkeypatch.context() as patch:
            patch.setattr(cute, "compile", forbidden)
            patch.setattr(prepared, "facts_of_roles", forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full", "tensor"):
                patch.setattr(torch, name, forbidden)
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "cpu", "item", "tolist"):
                patch.setattr(torch.Tensor, name, forbidden)
            graph.execute(pack, workspace.tensor)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode(sync)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.execute(pack, workspace.tensor)
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(names) == 5, names  # accumulator clear, metadata, delta, five-GEMM main, dQ cast
    assert all("cudnn" in n for n in names), names
    assert not any("dtoh" in n.lower() or "device to host" in n.lower() for n in names), names
    check_case(case, workspace)


@pytest.mark.parametrize("prefix_q,prefix_kv", [(True, False), (False, True)])
@pytest.mark.parametrize("offset_dtype,units", [(torch.int32, "elements"), (torch.int64, "tokens")])
def test_frost_thd_standalone_prefixes_stay_independent(prefix_q, prefix_kv, offset_dtype, units):
    from cudnn.sdpa.bwd.api_dsl_sm120_thd import SdpaBwdDslSm120Thd

    case = make_case((65, 0, 3), (17, 129, 0), stats_layout="HN", offset_dtype=offset_dtype, units=units)
    api = make_adapter(case, api_type=SdpaBwdDslSm120Thd)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = _adapter_arguments(case)
    for role, lengths, prefix in (("seq_q_lens", case.lens_q, prefix_q), ("seq_kv_lens", case.lens_kv, prefix_kv)):
        if prefix:
            values = [37]
            for n in lengths:
                values.append(values[-1] + n)
            args[role] = torch.tensor(values, dtype=torch.int32, device="cuda")
    poison(case, workspace)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("invalid", ["missing_length", "short_length", "stats_dtype", "short_origin", "short_dv", "gradient_alias", "workspace"])
def test_frost_thd_current_bindings_validate_before_writes(invalid):
    case = make_case((65, 0, 3), (17, 129, 0), stats_layout="HN")
    graph, refs, pack, workspace = _graph_case(case)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)
    poison(case, workspace)
    saved_workspace = workspace.raw.clone()
    saved_outputs = {r: case.raw[r].clone() for r in GRADS}
    pack = dict(pack)
    ws = workspace.tensor
    if invalid == "missing_length":
        del pack[refs["seq_q"]]
    elif invalid == "short_length":
        pack[refs["seq_q"]] = case.lengths["seq_q"][:-1]
    elif invalid == "stats_dtype":
        pack[refs["stats"]] = case.tensors["stats"].half()
    elif invalid == "short_origin":
        pack[refs["q_ro"]] = case.offsets["q"][:-1]
    elif invalid == "short_dv":
        pack[refs["dv"]] = torch.empty(16, dtype=case.dtype, device="cuda")
    elif invalid == "gradient_alias":
        pack[refs["dv"]] = case.tensors["dk"]
    else:
        ws = None
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        graph.execute(pack, ws)
    torch.testing.assert_close(workspace.raw, saved_workspace, rtol=0, atol=0)
    for role in GRADS:
        torch.testing.assert_close(case.raw[role], saved_outputs[role], rtol=0, atol=0, equal_nan=True)
    poison(case, workspace)
    graph.execute(variant_pack(case, refs), workspace.tensor)
    check_case(case, workspace)


def test_frost_thd_changed_prefix_replay():
    from cudnn.sdpa.bwd.api_dsl_sm120_thd import SdpaBwdDslSm120Thd

    options = dict(dtype=torch.float16, stats_layout="HN", envelope=(129, 129), capacities=(512, 512))
    case = make_case((65, 0, 3), (17, 129, 0), **options)
    changed = make_case((0, 97, 1), (17, 0, 129), seed=157, layout_shift=5, **options)
    api = make_adapter(case, api_type=SdpaBwdDslSm120Thd)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = _adapter_arguments(case)

    def prefix(lengths, base):
        values = [base]
        for n in lengths:
            values.append(values[-1] + n)
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    args["seq_q_lens"], args["seq_kv_lens"] = prefix(case.lens_q, 37), prefix(case.lens_kv, 51)
    capture, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    try:
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        check_case(case, workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            api.execute(**args, workspace=workspace.tensor, current_stream=stream.cuda_stream)
        copy_case_storage(case, changed)
        args["seq_q_lens"].copy_(prefix(changed.lens_q, 73))
        args["seq_kv_lens"].copy_(prefix(changed.lens_kv, 101))
        for _ in range(3):
            poison(case, workspace)
            capture.replay()
            check_case(case, workspace)
    finally:
        capture.reset()


@pytest.mark.parametrize("invalid", ["length_dtype", "scale_zero", "scale_negative", "scale_nan"])
def test_frost_thd_standalone_rejects_current_metadata(invalid):
    from cudnn.sdpa.bwd.api_dsl_sm120_thd import SdpaBwdDslSm120Thd

    case = make_case((65, 0, 3), (17, 129, 0))
    api = make_adapter(case, api_type=SdpaBwdDslSm120Thd)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = _adapter_arguments(case)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)
    poison(case, workspace)
    saved = workspace.raw.clone()
    if invalid == "length_dtype":
        args["seq_q_lens"] = case.lengths["seq_q"].long()
    else:
        args["scale_softmax"] = {"scale_zero": 0.0, "scale_negative": -0.1, "scale_nan": float("nan")}[invalid]
    with pytest.raises(ValueError, match="int32|positive"):
        api.execute(**args, workspace=workspace.tensor)
    torch.testing.assert_close(workspace.raw, saved, rtol=0, atol=0)
