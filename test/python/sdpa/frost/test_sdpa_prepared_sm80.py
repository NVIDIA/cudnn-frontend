# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM80 layouts, current bindings and the prepared execution contract."""

from dataclasses import replace
from types import SimpleNamespace

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, select_engine

pytestmark = [requires_dsl, pytest.mark.skipif(torch.cuda.get_device_capability() != (8, 0), reason="requires SM80")]


def _case(
    dq=128,
    dv=128,
    *,
    dtype=torch.bfloat16,
    layout="bshd",
    stats=True,
    features=False,
    causal=False,
    scheduler=0,
    wide=None,
    product=False,
    wide_axis="batch",
    heads=(4, 2),
    scale=None,
):
    scale = dq**-0.5 if scale is None else scale
    torch.manual_seed(3217)
    b, h, hk, sq, skv = (5 if product else 2), *heads, 33, 65
    if wide_axis == "row":
        b, sq, skv = 1, (3 if product else 2), (3 if product else 2)
    elif wide_axis == "tile":
        b, sq, skv = 1, 2, 65
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs, storage, refs = {}, {}, {}
    for name, heads, seq, d in (("q", h, sq, dq), ("k", hk, skv, dq), ("v", hk, skv, dv), ("o", h, sq, dv)):
        if layout == "bhsd":
            raw = torch.randn(b, heads, seq, d, dtype=dtype, device="cuda") * 0.3
            view = raw
        else:
            raw = torch.randn(b, seq, heads, d + (8 if layout == "gapped" else 0), dtype=dtype, device="cuda") * 0.3
            view = raw[..., :d].transpose(1, 2)
        if name == "o":
            raw.fill_(123)
        storage[name], bufs[name] = raw, view
    if stats:
        storage["stats"] = torch.full((b, h, sq * 2), float("nan"), device="cuda")
        bufs["stats"] = storage["stats"][..., ::2].unsqueeze(-1)
    if features:
        bufs["seq_q"] = torch.tensor([sq - 3, 0], dtype=torch.int32, device="cuda")
        bufs["seq_kv"] = torch.tensor([skv - 7, 0], dtype=torch.int32, device="cuda")
        bufs["sink"] = torch.randn(h, device="cuda")
        bufs["bias"] = torch.randn(1, h, sq, skv, device="cuda") * 0.1
    if wide is not None:
        import ctypes

        src = bufs[wide]
        axis = 0 if wide_axis == "batch" else 2
        stride = list(src.stride())
        stride[axis] += 2**25 if wide_axis == "tile" else (2**30 if product else 2**32)
        stride = tuple(stride)
        elements = 1 + sum((n - 1) * st for n, st in zip(src.shape, stride))
        origin = 2**31 if product else 0
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if (elements + origin) * src.element_size() + 512 * 2**20 > free:
            pytest.skip("physical Int64 probe needs room for one guarded wide buffer")
        try:
            backing = torch.empty(elements + origin, device="cuda", dtype=src.dtype)
        except torch.OutOfMemoryError:
            pytest.skip("another worker consumed the guarded wide-buffer allocation capacity")
        if product:
            shape = list(src.shape)
            shape[axis] = 1
            for index in range(src.shape[axis]):
                decoy = origin + ctypes.c_int32(index * stride[axis]).value
                backing.as_strided(shape, src.stride(), decoy).fill_(float("nan"))
        else:
            backing.as_strided(src.shape, src.stride()).fill_(float("nan"))
        bufs[wide] = backing.as_strided(src.shape, stride, origin).copy_(src)
        storage[wide] = backing
    for name, buf in bufs.items():
        if name not in ("o", "stats"):
            if name in ("seq_q", "seq_kv"):
                refs[name] = graph.tensor(name=name, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
            elif name == "sink":
                refs[name] = graph.tensor(name=name, dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
            else:
                refs[name] = graph.tensor_like(buf, name=name)
    optional = dict(use_padding_mask=True, seq_len_q=refs["seq_q"], seq_len_kv=refs["seq_kv"], sink_token=refs["sink"], bias=refs["bias"]) if features else {}
    o, lse = graph.sdpa(q=refs["q"], k=refs["k"], v=refs["v"], attn_scale=scale, generate_stats=stats, use_causal_mask=causal, **optional)
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride()).set_data_type(io)
    refs["o"] = o
    if stats:
        lse.set_output(True).set_dim(bufs["stats"].shape).set_stride(bufs["stats"].stride()).set_data_type(cudnn.data_type.FLOAT)
        refs["stats"] = lse
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    selected = select_engine(graph, "sdpa_fwd_prefill_sm80")
    if selected.knobs.sched_policy != scheduler:
        graph.create_execution_plan(selected.engine_id, replace(selected.knobs, sched_policy=scheduler))
        graph.select_plan(len(graph.plans) - 1)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    pack = {refs[name]: buf for name, buf in bufs.items()}
    return SimpleNamespace(graph=graph, bufs=bufs, storage=storage, refs=refs, pack=pack, workspace=workspace, features=features, causal=causal, scale=scale)


def _check(case):
    q, k, v = (case.bufs[name].double() for name in ("q", "k", "v"))
    group = q.shape[1] // k.shape[1]  # Q heads per KV head; the kernel reads K/V head h // group
    scores = q @ k.repeat_interleave(group, 1).transpose(-1, -2) * case.scale
    sq, skv = q.shape[2], k.shape[2]
    if case.features:
        scores += case.bufs["bias"].double()
        scores.masked_fill_(torch.arange(skv, device="cuda")[None, None, None, :] >= case.bufs["seq_kv"][:, None, None, None], float("-inf"))
    if case.causal:
        scores.masked_fill_(torch.arange(skv, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None], float("-inf"))
    all_scores = torch.cat((scores, case.bufs["sink"].double()[None, :, None, None].expand(q.shape[0], -1, sq, 1)), -1) if case.features else scores
    stats = all_scores.logsumexp(-1)
    probs = torch.nan_to_num((scores - stats[..., None]).exp(), nan=0)
    out = probs @ v.repeat_interleave(group, 1)
    if case.features:
        padded = torch.arange(sq, device="cuda")[None, :] >= case.bufs["seq_q"][:, None]
        out.masked_fill_(padded[:, None, :, None], 0)
        stats.masked_fill_(padded[:, None, :], float("-inf"))
    torch.testing.assert_close(case.bufs["o"].double(), out, atol=2e-3, rtol=2e-2)
    if "stats" in case.bufs:
        torch.testing.assert_close(case.bufs["stats"].squeeze(-1).double(), stats, atol=8e-4, rtol=8e-4)


@pytest.mark.L0
@pytest.mark.parametrize("dq,dv", [(64, 64), (96, 128), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["bshd", "bhsd", "gapped"])
def test_sm80_prepared_native_layouts(dq, dv, dtype, layout):
    case = _case(dq, dv, dtype=dtype, layout=layout)
    assert case.graph._compiled_plans[case.graph._plan_index]._prepared is not None
    assert case.graph.get_workspace_size() == 0
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "stats"):
                # clone preserves compact/permuted layouts; a gapped declaration
                # needs its original strides and a new allocation explicitly.
                old = case.bufs[name]
                new = torch.empty_strided(old.shape, old.stride(), device="cuda", dtype=old.dtype).copy_(old)
                if name in ("q", "k", "v"):
                    new.mul_(0.7 + iteration * 0.2)
                else:
                    new.fill_(float("nan"))
                case.bufs[name] = new
                case.pack[case.refs[name]] = new
        case.graph.execute(case.pack, case.workspace)
        _check(case)
        if layout == "gapped":
            assert torch.all(case.storage["o"][..., dv:] == 123)


@pytest.mark.L0
@pytest.mark.parametrize("dq,dv", [(64, 64), (128, 128), (192, 128), (256, 256)], ids=["d64", "d128", "d192_d128", "d256"])
@pytest.mark.parametrize("heads", [(8, 4), (8, 2), (8, 1)], ids=["g2", "g4", "mqa"])
def test_sm80_prepared_gqa_ratios(dq, dv, heads):
    """Native dense GQA/MQA where the group size H_q // H_kv differs from H_kv.

    Every other case here runs H_q=4, H_kv=2, where the two counts are equal, so
    a kernel that uses one for the other (dividing the Q head by H_kv, or sizing
    the LPT_L2 KV groups and row ranks with the wrong count) still reads the
    right K/V head. 8:4 and 8:2 tell the counts apart; MQA folds every Q head
    onto one KV head. Pinned to LPT_L2, the SM80 heuristic's causal choice and
    the only decode that reads H_kv, on all four flavors, so the dedicated d256
    kernel is covered as well as the generic one."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    case = _case(dq, dv, causal=True, scheduler=SCHED_LPT_L2, heads=heads)
    assert case.graph._compiled_plans[case.graph._plan_index]._prepared is not None
    case.graph.execute(case.pack, case.workspace)
    _check(case)


@pytest.mark.L0
@pytest.mark.parametrize("scheduler", [0, 1, 2])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_sm80_prepared_optional_operands_capture(scheduler, stats, causal):
    case = _case(features=True, scheduler=scheduler, stats=stats, causal=causal, layout="bhsd")
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            case.graph.execute(case.pack, case.workspace, handle)
        torch.cuda.current_stream().wait_stream(stream)
        for factor in (0.6, 1.4):
            case.bufs["sink"].mul_(factor)
            case.bufs["v"].mul_(factor)
            case.bufs["o"].fill_(float("nan"))
            capture.replay()
            _check(case)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.L0
@pytest.mark.parametrize("standalone", [False, True])
def test_sm80_prepared_execute_has_no_tensor_plumbing(standalone, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    case = _case(layout="bhsd")
    if standalone:
        api = SdpaFwdDslSm80(*(case.bufs[name] for name in ("q", "k", "v", "o", "stats")))
        api.check_support()
        api.compile()
        run = lambda: api.execute(*(case.bufs[name] for name in ("q", "k", "v", "o", "stats")))
    else:
        run = lambda: case.graph.execute(case.pack, case.workspace)
    run()
    torch.cuda.synchronize()
    with monkeypatch.context() as guard:

        def forbidden(*a, **k):
            pytest.fail("prepared execute must not construct tensors, allocate, compile, or copy")

        for name in ("view", "reshape", "transpose", "permute", "contiguous", "as_strided", "copy_", "zero_"):
            guard.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "empty_strided", "zeros", "ones"):
            guard.setattr(torch, name, forbidden)
        guard.setattr(cute, "compile", forbidden)
        guard.setattr(cute.runtime, "from_dlpack", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            run()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _check(case)


@pytest.mark.L0
@pytest.mark.parametrize("role", ["q", "stats", "bias"])
@pytest.mark.parametrize("ordered", [False, True])
def test_sm80_prepared_raw_storage_and_override_rejection(role, ordered):
    case = _case(features=True, layout="gapped")
    tensor = case.bufs[role]
    span = 1 + sum((n - 1) * st for n, st in zip(tensor.shape, tensor.stride()))
    case.pack[case.refs[role]] = tensor.as_strided((span,), (1,))
    uids = [t.get_uid() for t in case.pack]
    values = list(case.pack.values())

    def run(**kwargs):
        if ordered:
            case.graph.execute(values, case.workspace, tensor_uids=uids, **kwargs)
        else:
            case.graph.execute(case.pack, case.workspace, **kwargs)

    run()
    _check(case)
    plan = case.graph._compiled_plans[case.graph._plan_index]
    calls = []
    original = plan._prepared.spec
    plan._prepared.spec = replace(original, fn=lambda *args: calls.append(args))
    try:
        shape, strides = list(case.refs[role].get_dim()), list(case.refs[role].get_stride())
        strides[-2] += 8
        with pytest.raises((ValueError, RuntimeError), match="geometry|stride|override|storage"):
            run(override_uids=[case.refs[role].get_uid()], override_shapes=[shape], override_strides=[strides])
        assert not calls
    finally:
        plan._prepared.spec = original


@pytest.mark.L0
@pytest.mark.parametrize("bad", ["dtype", "layout", "alignment", "short", "missing_stats", "unexpected_sink", "cpu"])
def test_sm80_prepared_standalone_invalid_binding(bad):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    case = _case(layout="bhsd")
    values = [case.bufs[name] for name in ("q", "k", "v", "o", "stats")]
    api = SdpaFwdDslSm80(*values)
    api.check_support()
    api.compile()
    calls = []
    api._sm80_spec = replace(api._sm80_spec, fn=lambda *args: calls.append(args))
    extra = {}
    if bad == "dtype":
        values[0] = values[0].float()
    elif bad == "layout":
        values[0] = values[0].transpose(1, 2).contiguous().transpose(1, 2)
    elif bad == "alignment":
        values[0] = torch.empty(values[0].numel() + 1, dtype=values[0].dtype, device="cuda")[1:].view(values[0].shape)
    elif bad == "short":
        values[0] = values[0][:1]
    elif bad == "missing_stats":
        values[4] = None
    elif bad == "unexpected_sink":
        extra["sinks"] = torch.zeros(4, device="cuda")
    else:
        values[0] = values[0].cpu()
    with pytest.raises(ValueError):
        api.execute(*values, **extra)
    assert not calls


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "stats"])
@pytest.mark.parametrize("product", [False, True], ids=["wide_stride", "wide_product"])
@pytest.mark.parametrize("axis", ["batch", "row"])
@pytest.mark.parametrize("d", [64, 256])
def test_sm80_prepared_physical_int64(role, product, axis, d):
    case = _case(dq=d, dv=d, wide=role, product=product, wide_axis=axis)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        case.bufs["o"].fill_(float("nan"))
        case.bufs["stats"].fill_(float("nan"))
        capture.replay()
        _check(case)
    finally:
        capture.reset()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("d", [64, 256])
def test_sm80_prepared_physical_tile_product(role, d):
    case = _case(dq=d, dv=d, wide=role, product=True, wide_axis="tile")
    case.graph.execute(case.pack, case.workspace)
    _check(case)


@pytest.mark.L0
def test_sm80_prepared_standalone_flat_operands():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    case = _case(features=True)
    case.bufs["stats"] = case.bufs["stats"].contiguous()
    api = SdpaFwdDslSm80(
        *(case.bufs[name] for name in ("q", "k", "v", "o", "stats")),
        has_sink=True,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        bias_present=True,
        bias_fp32=True,
    )
    api.check_support()
    api.compile()
    # The first bias plane is the existing broadcast contract; a second plane
    # and the alternative flat Stats/sink views must not change that operation.
    bias = torch.cat((case.bufs["bias"], torch.full_like(case.bufs["bias"], 1000)), 0)
    api.execute(
        *(case.bufs[name] for name in ("q", "k", "v", "o")),
        case.bufs["stats"].view(-1),
        sinks=case.bufs["sink"].view(2, 2),
        seq_q_lens=case.bufs["seq_q"],
        seq_kv_lens=case.bufs["seq_kv"],
        bias_tensor=bias,
    )
    _check(case)


@pytest.mark.L0
@pytest.mark.parametrize("dq,dv", [(64, 64), (128, 128), (256, 256)])
@pytest.mark.parametrize("scale", [0.7, 1.0, 4.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("features", [False, True])
def test_sm80_prepared_scale_above_ln2(dq, dv, scale, causal, features):
    # scale * log2(e) > 1 overflows a fully masked tile row's -FLT_MAX fill to -inf.
    case = _case(dq, dv, causal=causal, features=features, scale=scale)
    case.graph.execute(case.pack, case.workspace)
    torch.cuda.synchronize()
    _check(case)
