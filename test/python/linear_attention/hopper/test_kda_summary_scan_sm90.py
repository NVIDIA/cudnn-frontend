# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-only KDA reverse summaries reuse the ordinary SM90 backward scan."""

from contextlib import nullcontext

import pytest
import torch

import cudnn
import test_kda_summary_sm90 as fp32
from cudnn.linear_attention import kimi_delta_attention_summary, kimi_delta_attention_summary_bwd
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

ENGINE = "kda_summary_hopper_scan"
pytestmark = fp32.pytestmark


@pytest.fixture(autouse=True)
def enable_scan_engine(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def run(case):
    return kimi_delta_attention_summary_bwd(
        case["q"],
        case["k"],
        case["g"],
        case["beta"],
        case["dO"],
        case["cu"],
        batch_invariant=True,
        plan_name=ENGINE,
    )


def expected(case):
    return fp32.reference(case, backward=True, seeded=False)[0]


def ordinary_backward_state(case):
    """Run the complete ordinary backward; its reverse scan is the byte oracle."""
    from cudnn.linear_attention.hopper import cuda_bwd_host

    total, heads, dim = case["q"].shape
    seqs = case["cu"].numel() - 1
    state = torch.zeros(seqs, heads, dim, dim, dtype=torch.float32, device="cuda")
    d_initial_state = torch.empty_like(state)
    dq, dk, dv = (torch.empty_like(case["q"]) for _ in range(3))
    dg = torch.empty_like(case["g"])
    dbeta = torch.empty_like(case["beta"])
    workspace = torch.empty(cuda_bwd_host.workspace_bytes(total, heads, seqs), dtype=torch.uint8, device="cuda")
    cuda_bwd_host.launch(
        device=torch.cuda.current_device(),
        stream=torch.cuda.current_stream().cuda_stream,
        workspace=workspace.data_ptr(),
        dO=case["dO"].data_ptr(),
        q=case["q"].data_ptr(),
        k=case["k"].data_ptr(),
        v=case["v"].data_ptr(),
        g=case["g"].data_ptr(),
        beta=case["beta"].data_ptr(),
        cu_seqlens=case["cu"].data_ptr(),
        initial_state=state.data_ptr(),
        d_final_state=state.data_ptr(),
        dq=dq.data_ptr(),
        dk=dk.data_ptr(),
        dv=dv.data_ptr(),
        dg=dg.data_ptr(),
        dbeta=dbeta.data_ptr(),
        d_initial_state=d_initial_state.data_ptr(),
        total_tokens=total,
        n_heads=heads,
        n_seqs=seqs,
        scale=dim**-0.5,
    )
    return d_initial_state


def build_graph(case):
    from cudnn.linear_attention.ops.common import select_plan
    from cudnn.linear_attention.ops.kda import build_kda_summary_bwd_graph

    total, heads, dim = case["q"].shape
    seqs = case["cu"].numel() - 1
    graph, tensors = build_kda_summary_bwd_graph(
        total,
        seqs,
        heads,
        heads,
        heads,
        heads,
        dim,
        dim,
        cudnn.data_type.BFLOAT16,
        cudnn.data_type.FLOAT,
        cudnn.data_type.FLOAT,
        None,
        cudnn.data_type.INT64 if case["cu"].dtype == torch.int64 else cudnn.data_type.INT32,
        dim**-0.5,
        False,
        True,
        False,
        False,
        False,
        None,
        "log",
        output_transition=False,
    )
    select_plan(graph, ENGINE)
    graph.build_plans()
    output = torch.empty(seqs, heads, dim, dim, dtype=torch.float32, device="cuda")
    bindings = {
        tensors["q"]: case["q"],
        tensors["k"]: case["k"],
        tensors["g"]: case["g"],
        tensors["beta"]: case["beta"],
        tensors["dO"]: case["dO"],
        tensors["cu"]: case["cu"],
        tensors["d_initial_state"]: output,
    }
    return graph, tensors, bindings, output


@pytest.mark.parametrize("gate", [-0.001, -5.0], ids=["weak-decay", "strong-decay"])
def test_scan_matches_fp64(gate):
    case = fp32.make_case((1, 15, 16, 17, 0, 33, 65), heads=2, gate=gate)
    actual = run(case)
    reference = expected(case)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert_bitwise({"state": actual}, {"state": ordinary_backward_state(case)}, "ordinary backward scan")
    torch.testing.assert_close(actual.double(), reference, atol=1e-3, rtol=1e-2)


def test_scan_determinism_and_batch_invariance():
    target = fp32.make_case((33,), heads=2)
    before = {name: value.clone() for name, value in target.items()}
    replay = assert_replays(lambda: {"state": run({name: value.clone() for name, value in target.items()})}, repeats=4)
    assert_bitwise(replay, {"state": ordinary_backward_state(target)}, "ordinary backward scan")
    torch.testing.assert_close(replay["state"].double(), expected(target), atol=1e-3, rtol=1e-2)

    cases = [target, fp32.make_case((17,), seed=121), fp32.make_case((0,), seed=717)]
    for order in ((0, 1, 2, 0), (2, 0, 1, 0), tuple([1] * 20 + [0, 2, 0])):
        packed = {key: torch.cat([cases[i][key] for i in order]) for key in ("q", "k", "v", "g", "beta", "dO", "state")}
        lengths = [cases[i]["k"].shape[0] for i in order]
        packed["cu"] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        got = snapshot({"state": run(packed)})
        for index, source in enumerate(order):
            if source == 0:
                assert_bitwise({"state": got["state"][index : index + 1]}, replay, "batch packing")

    assert_bitwise(target, before, "input mutation")
    inputs = {name: target[name] for name in ("q", "k", "g", "beta", "dO", "cu")}
    assert_golden("kda_summary_scan_sm90_bwd_bf16", inputs, replay)


def test_scan_workspace_capture_and_changed_inputs(cudnn_handle):
    case = fp32.make_case((17, 0, 65), heads=2)
    graph, tensors, bindings, output = build_graph(case)
    workspace_bytes = graph.get_workspace_size()
    from cudnn.linear_attention.hopper import cuda_bwd_host

    total, heads, _ = case["q"].shape
    seqs = case["cu"].numel() - 1
    zero_bytes = seqs * heads * 128 * 128 * 4
    assert workspace_bytes == cuda_bwd_host.reverse_workspace_bytes(total, heads, seqs) + zero_bytes
    assert cuda_bwd_host.reverse_workspace_bytes(total, heads, seqs) < cuda_bwd_host.workspace_bytes(total, heads, seqs) // 2
    assert workspace_bytes > output.numel() * output.element_size()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    capture = torch.cuda.CUDAGraph()
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        workspace.fill_(0xFF)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.execute(bindings, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(output.double(), expected(case), atol=1e-3, rtol=1e-2)
        eager = output.clone()

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            graph.execute(bindings, workspace, handle=cudnn_handle)

        for bounds, q_scale, do_scale in (([0, 19, 19, 82], -1.0, 0.75), ([0, 33, 82, 82], 0.5, -1.25)):
            case["cu"].copy_(torch.tensor(bounds, dtype=torch.int32, device="cuda"))
            case["q"].mul_(q_scale)
            case["dO"].mul_(do_scale)
            case["g"].mul_(1.5)
            output.fill_(float("nan"))
            workspace.fill_(0xFF)
            stream.wait_stream(torch.cuda.current_stream())
            capture.replay()
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(output.double(), expected(case), atol=1e-3, rtol=1e-2)

        rebound = {tensor: value.clone() for tensor, value in bindings.items()}
        fresh_workspace = torch.full_like(workspace, 0xFF)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.execute(rebound, fresh_workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        assert_bitwise({"state": rebound[tensors["d_initial_state"]]}, {"state": output}, "fresh addresses")

        # The original warm result proves that changed-input replay did not
        # accidentally preserve the capture-time output.
        assert not torch.equal(eager.view(torch.uint8), output.view(torch.uint8))
    finally:
        capture.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)


def test_short_workspace_rejects_before_output_write(cudnn_handle):
    case = fp32.make_case((65,), heads=1)
    graph, _, bindings, output = build_graph(case)
    required = graph.get_workspace_size()
    backing = torch.empty(required, dtype=torch.uint8, device="cuda")
    output.fill_(123)
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        with pytest.raises(ValueError, match="workspace|capacity|small"):
            graph.execute(bindings, backing[:-1], handle=cudnn_handle)
        assert (output == 123).all()
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize(
    "option, reason",
    [
        ("batch", "batch_invariant"),
        ("transition", "G-only"),
        ("seed", "unseeded"),
        ("cu64", "int32 cu_seqlens"),
        ("strided", "packed"),
    ],
)
def test_scan_contract_declines(option, reason):
    from cudnn.linear_attention.hopper.summary_engine import KdaSummaryHopperScanEngine
    from cudnn.linear_attention.ops.kda import build_kda_summary_bwd_graph

    case = fp32.make_case((17,), heads=1)
    total, heads, dim = case["q"].shape
    graph, tensors = build_kda_summary_bwd_graph(
        total,
        1,
        heads,
        heads,
        heads,
        heads,
        dim,
        dim,
        cudnn.data_type.BFLOAT16,
        cudnn.data_type.FLOAT,
        cudnn.data_type.FLOAT,
        cudnn.data_type.FLOAT if option == "seed" else None,
        cudnn.data_type.INT64 if option == "cu64" else cudnn.data_type.INT32,
        dim**-0.5,
        False,
        option != "batch",
        False,
        False,
        False,
        None,
        "log",
        output_transition=option == "transition",
    )
    if option == "strided":
        tensors["q"].set_stride([heads * dim * 2, dim, 1])
    with pytest.raises(NotImplementedError, match=reason):
        KdaSummaryHopperScanEngine().check_support(graph)


def test_scan_plan_is_reverse_only():
    case = fp32.make_case((17,), heads=1)
    with pytest.raises((cudnn.cudnnGraphNotSupportedError, NotImplementedError), match=ENGINE):
        kimi_delta_attention_summary(
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            case["cu"],
            batch_invariant=True,
            plan_name=ENGINE,
        )


@pytest.mark.parametrize(
    "tokens,heads,seqs,expected_scan,expected_grid",
    [
        (64, 8, 1, "k_scan_t<16, 128, true, true, true>", (8, 8, 1)),
        (65, 3, 1, "k_scan_t<32, 256, true, false, true>", (4, 3, 1)),
        (65, 3, 2, "k_scan_t<32, 256, false, false, true>", (4, 6, 1)),
    ],
)
def test_host_launch_omits_forward_scan_and_token_gradients(monkeypatch, tokens, heads, seqs, expected_scan, expected_grid):
    from cudnn.linear_attention.hopper import cuda_bwd_host

    launches = []

    class Library:
        @staticmethod
        def function(name):
            return name

    def launch(function, *args, **kwargs):
        launches.append((function, kwargs.get("grid"), kwargs.get("what")))

    monkeypatch.setattr(cuda_bwd_host, "library", lambda device: Library())
    monkeypatch.setattr(cuda_bwd_host, "device_context", lambda device: nullcontext())
    monkeypatch.setattr(cuda_bwd_host.compiler, "launch", launch)
    cuda_bwd_host.launch(
        device=0,
        stream=0,
        workspace=0x100000,
        dO=1,
        q=2,
        k=3,
        v=4,
        g=5,
        beta=6,
        cu_seqlens=7,
        initial_state=0,
        d_final_state=8,
        dq=0,
        dk=0,
        dv=0,
        dg=0,
        dbeta=0,
        d_initial_state=9,
        total_tokens=tokens,
        n_heads=heads,
        n_seqs=seqs,
        scale=128**-0.5,
        gradient_outputs=False,
    )
    names = [entry[0] for entry in launches]
    expected_prep = f"k_prep<{'true' if seqs == 1 and tokens % 64 == 0 else 'false'}, true>"
    assert expected_prep in names
    assert expected_scan in names
    assert next(entry[1] for entry in launches if entry[0] == expected_scan) == expected_grid
    assert not any(name.startswith("k_bwd") for name in names)
    assert not any(name.startswith("k_scan_t") and name != expected_scan for name in names)
