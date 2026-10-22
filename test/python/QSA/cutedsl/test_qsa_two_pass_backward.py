# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GQA index-list sparse (QSA) backward on SM100 / SM103, the internal two-pass backend."""

from unittest.mock import patch

import pytest
import torch
from cuda.bindings import driver as cuda

from cudnn.QSA.two_pass.api import _QsaTwoPassBackward, _backward
from cudnn.api_base import TensorDesc
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

from QSA.cutedsl.two_pass_reference import assert_grads_close, make_inputs, reference_backward

_INPUT_NAMES = ("q", "k", "v", "out", "dout", "lse", "topk_idxs")


@pytest.fixture(autouse=True)
def supported_device(request):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("QSA backward requires SM100 or SM103")
    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("QSA backward requires CuTe DSL >= 4.7.0")
    if getattr(request.node, "callspec", None) is not None and request.node.callspec.params.get("h_kv") == 1:
        from cudnn.QSA.two_pass.api import _dsl_cuda_version

        if _dsl_cuda_version()[0] < 13:
            pytest.skip("QSA 12Q/1KV requires the CUDA 13 build of CuTe DSL")


def _metadata(tensor):
    stride = tensor.stride()
    order = tuple(sorted(range(tensor.ndim), key=lambda i: (stride[i], tensor.shape[i])))
    return TensorDesc(tensor.dtype, tuple(tensor.shape), stride, order, tensor.device)


def _check_against_reference(inputs, grads):
    expected = reference_backward(*(inputs[name] for name in _INPUT_NAMES))
    for name, actual, ref in zip(("dq", "dk", "dv"), grads, expected):
        assert_grads_close(name, actual, ref)


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [2, 1])  # the model's 24 / 2 heads, and the 12 / 1 of one of two tensor-parallel ranks
@pytest.mark.parametrize(
    "seq_len, grad_scale",
    [
        (2, 1.0),  # tail tokens only (a single token has dQ = dK = 0, which a relative bound cannot score)
        (5, 1.0),  # one complete block and a tail
        (2051, 1.0),  # the longest sequence whose lists are all whole causal prefixes, so the bitmap is filled without the bitmap kernel
        (2052, 1.0),  # the shortest sequence on which the last query drops a block and the bitmap kernel runs
        (4099, 1.0),  # every query in the dense kernel, sparse rows above 2050 included
        (16384, 1.0),  # every query in the dense kernel with the larger dense unit
        (22527, 1.0),  # the longest sequence served by the dense kernel alone
        (22528, 1.0),  # the shortest sequence with a sparse remainder: a single query range on the key-major kernel
        (22529, 1.0),  # one full query range and a second range holding a single query, which with one KV head is an unpaired item
        (24000, 1.0),  # dense, query-major and key-major kernels in one KV pass and two query ranges
        (24000, 1e-7),  # training-scale gradients, which rule out reduced-range accumulators
        (32769, 1.0),  # the long-sequence crossover, two KV passes and a partial last block
    ],
)
def test_qsa_backward_matches_reference(seq_len, grad_scale, h_kv):
    inputs = make_inputs(seq_len, seed=seq_len, grad_scale=grad_scale, h_kv=h_kv)
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [2, 1])
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        [0, 3, 2054, 24583],  # a 3-token sequence, a whole-prefix one starting at 3 and one past the crossover starting at 2054 (both off the 4-token grid)
        [0, 4100, 28105, 28107],  # dense, sparse with a 3525-query remainder, then 2 tokens starting at 28105
        [0, 500, 500, 3000],  # an empty sequence between two dense ones
        [0, 1000, 3051, 5102],  # every sequence within 2051 tokens, so the bitmap is filled without the bitmap kernel
        [0, 24000],  # one sequence through the packed path
        [0, 3, 24580, 49157],  # multiple windows and two sparse sequences with unaligned origins
        [0, 32769],  # partial final record window
    ],
)
def test_qsa_backward_packed_sequences(cu_seqlens, h_kv):
    inputs = make_inputs(cu_seqlens[-1], seed=cu_seqlens[-1], h_kv=h_kv, cu_seqlens=cu_seqlens)
    max_seq_len = max(e - s for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]))
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES), cu_seqlens=inputs["cu_seqlens"], max_seq_len=max_seq_len)
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
def test_qsa_backward_accepts_the_padded_index_width(h_kv):
    # 2052 columns: the fourth tail slot is never read, only the row stride changes
    inputs = make_inputs(24000, seed=11, h_kv=h_kv, width=2052)
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
@pytest.mark.parametrize("grad_scale", [1.0, 1e-7])
@pytest.mark.parametrize("seq_len, cu_seqlens", [(4099, None), (32769, None), (32769, [0, 3, 32769])])
def test_fp32_output_and_head_major_lse(seq_len, cu_seqlens, h_kv, grad_scale):
    inputs = make_inputs(
        seq_len,
        seed=seq_len,
        h_kv=h_kv,
        cu_seqlens=cu_seqlens,
        width=2052,
        grad_scale=grad_scale,
        saved_output_dtype=torch.float32,
    )
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES), cu_seqlens=inputs.get("cu_seqlens"), max_seq_len=seq_len)
    _check_against_reference(inputs, tuple(grads))


@pytest.mark.L0
def test_native_head_major_lse_needs_only_scalar_alignment():
    inputs = make_inputs(17, seed=42, h_kv=2, saved_output_dtype=torch.float32)
    storage = torch.empty(inputs["lse"].numel() + 1, dtype=torch.float32, device="cuda")
    lse = storage[1:].view_as(inputs["lse"])
    lse.copy_(inputs["lse"])
    inputs["lse"] = lse
    assert lse.data_ptr() % 16 == 4
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, tuple(grads))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
@pytest.mark.parametrize("num_seqs", [0, 3])
def test_two_pass_records_workspace_is_bounded(h_kv, num_seqs):
    from math import prod
    from cudnn.QSA.two_pass import qsa_bwd_sm100 as kernel

    for length in (32769, 131072, 262144):
        plan = kernel.LaunchPlan(length, 148, h_kv, num_seqs, length)
        records = next(prod(shape) * dtype.itemsize for name, shape, dtype in plan.chunks if name == "pair_records")
        assert records < 1024**3
        if length == 131072:
            assert plan.workspace_bytes() < (3 if h_kv == 1 else 5) * 1024**3


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
@pytest.mark.parametrize("saved_output_dtype", [torch.bfloat16, torch.float32])
def test_packed_window_graph_rebuilds_changed_sequence_metadata(h_kv, saved_output_dtype):
    # The same plan crosses windows for one input, has only dense queries for
    # the next, then crosses sparse windows and sequence origins on the third.
    bounds = ([0, 3, 32769], [0, 16384, 32769], [0, 2, 32769])
    samples = [make_inputs(32769, seed=101 + i, h_kv=h_kv, cu_seqlens=cu, saved_output_dtype=saved_output_dtype) for i, cu in enumerate(bounds)]
    static = {name: value.clone() for name, value in samples[0].items()}
    args = [static[name] for name in _INPUT_NAMES]
    plan = _QsaTwoPassBackward(*args, sample_cu_seqlens=static["cu_seqlens"], max_seq_len=32769)
    plan.compile()
    outputs = [torch.empty_like(static[name]) for name in ("q", "k", "v")]
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(*args, *outputs, workspace=workspace, cu_seqlens=static["cu_seqlens"])
    for sample in samples:
        for name in static:
            static[name].copy_(sample[name])
        workspace.fill_(0xFF)
        for output in outputs:
            output.fill_(float("nan"))
        graph.replay()
        _check_against_reference(sample, outputs)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("h_kv", [2, 1])
@pytest.mark.parametrize("seq_len", [65536, 262144])
def test_qsa_backward_long_sequences(seq_len, h_kv):
    inputs = make_inputs(seq_len, seed=seq_len, h_kv=h_kv)
    grads = _backward(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L0
def test_wrapper_on_explicit_stream_with_preallocated_outputs(monkeypatch):
    from cudnn.QSA.two_pass import qsa_bwd_sm100 as kernel

    streams = []
    launch = kernel.launch

    def record(fn):
        def call(*fn_args):
            # Every compiled launcher takes its stream as the last argument.
            streams.append(int(fn_args[-1]))
            return fn(*fn_args)

        return call

    monkeypatch.setattr(kernel, "launch", lambda plan, fns, *rest: launch(plan, {name: record(fn) for name, fn in fns.items()}, *rest))
    inputs = make_inputs(28000, seed=3)
    args = [inputs[name] for name in _INPUT_NAMES]
    dq, dk, dv = torch.empty_like(inputs["q"]), torch.empty_like(inputs["k"]), torch.empty_like(inputs["v"])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    grads = _backward(*args, dq=dq, dk=dk, dv=dv, stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.current_stream().wait_stream(side)
    assert streams and all(stream == side.cuda_stream for stream in streams)  # one launch per kernel of the dense + sparse sequence
    assert grads["dq"] is dq and grads["dk"] is dk and grads["dv"] is dv
    _check_against_reference(inputs, (dq, dk, dv))


@pytest.mark.L0
@pytest.mark.parametrize("seq_len", [4099, 28000])  # the dense-only and the dense + sparse launch sequences
@pytest.mark.parametrize("saved_output_dtype", [torch.bfloat16, torch.float32])
def test_execute_allocates_nothing_and_never_synchronizes(seq_len, saved_output_dtype, monkeypatch, compile_allocates_nothing):
    from cudnn.QSA.two_pass import qsa_bwd_sm100 as kernel

    # An empty artifact cache makes this a cold compile.
    monkeypatch.setattr(kernel, "_COMPILED", {})
    inputs = make_inputs(seq_len, seed=5, saved_output_dtype=saved_output_dtype)
    args = [inputs[name] for name in _INPUT_NAMES]
    plan = _QsaTwoPassBackward(*args)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    outputs = [torch.empty_like(inputs[name]) for name in ("q", "k", "v")]
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    plan.execute(*args, *outputs, workspace=workspace)
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            plan.execute(*args, *outputs, workspace=workspace)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    _check_against_reference(inputs, outputs)


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [2, 1])
@pytest.mark.parametrize("seq_len", [4099, 28000])  # the dense-only and the dense + sparse launch sequences
def test_first_execute_under_cuda_graph_capture_replays_new_inputs(seq_len, h_kv, monkeypatch):
    from cudnn.QSA.two_pass import qsa_bwd_sm100 as kernel

    # Fresh artifacts make the capture hold their first launch.
    monkeypatch.setattr(kernel, "_COMPILED", {})
    first, second = make_inputs(seq_len, seed=21, h_kv=h_kv), make_inputs(seq_len, seed=22, h_kv=h_kv)
    static = {name: first[name].clone() for name in _INPUT_NAMES}
    args = [static[name] for name in _INPUT_NAMES]
    plan = _QsaTwoPassBackward(*args)
    plan.compile()
    grads = [torch.empty_like(first[name]) for name in ("q", "k", "v")]
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(*args, *grads, workspace=workspace)
    for inputs in (first, second):
        for name in _INPUT_NAMES:
            static[name].copy_(inputs[name])
        graph.replay()
        _check_against_reference(inputs, grads)


@pytest.mark.L0
def test_execute_requires_workspace_and_compiled_contract():

    inputs = make_inputs(64, seed=7)
    args = [inputs[name] for name in _INPUT_NAMES]
    plan = _QsaTwoPassBackward(*args)
    plan.compile()
    outputs = [torch.empty_like(inputs[name]) for name in ("q", "k", "v")]
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(*args, *outputs)
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="dq must match"):
        plan.execute(*args, outputs[0][:32], *outputs[1:], workspace=workspace)


def _sample_descs(seq_len=64, h_kv=2):
    q = torch.empty((seq_len, 12 * h_kv, 256), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((seq_len, h_kv, 256), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((seq_len, 12 * h_kv), dtype=torch.float32, device="cuda")
    idx = torch.empty((seq_len, 2051), dtype=torch.int32, device="cuda")
    return {name: _metadata(t) for name, t in zip(_INPUT_NAMES, (q, kv, kv, q, q, lse, idx))}


def _with(descs, name, **changes):
    d = descs[name]
    fields = {"dtype": d.dtype, "shape": d.shape, "stride": d.stride, "stride_order": d.stride_order, "device": d.device} | changes
    return {**descs, name: TensorDesc(fields["dtype"], fields["shape"], fields["stride"], fields["stride_order"], fields["device"])}


@pytest.mark.L0
@pytest.mark.parametrize(
    "name, changes, error, match",
    [
        ("q", {"dtype": torch.float16}, ValueError, "q"),
        ("k", {"shape": (64, 4, 256), "stride": (1024, 256, 1)}, NotImplementedError, "KV heads"),
        ("k", {"shape": (64, 1, 256), "stride": (256, 256, 1)}, ValueError, "q must have shape"),  # 24 query heads over one KV head
        ("q", {"shape": (64, 24, 128), "stride": (3072, 128, 1)}, ValueError, "q must have shape"),
        ("topk_idxs", {"shape": (64, 2048), "stride": (2048, 1)}, ValueError, "topk_idxs must have shape"),
        ("lse", {"dtype": torch.bfloat16}, ValueError, "lse"),
        ("dout", {"stride": (256, 64 * 256, 1), "stride_order": (2, 0, 1)}, NotImplementedError, "contiguous dout"),
    ],
)
def test_check_support_declines(name, changes, error, match):

    descs = _with(_sample_descs(), name, **changes)
    with pytest.raises(error, match=match):
        _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
def test_check_support_accepts_both_head_counts(h_kv):
    descs = _sample_descs(h_kv=h_kv)
    assert _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
def test_one_kv_head_declines_the_cuda12_compiler(monkeypatch):
    from cudnn.QSA.two_pass import api

    monkeypatch.setattr(api, "_dsl_cuda_version", lambda: (12, 9))
    descs = _sample_descs(seq_len=24000, h_kv=1)
    with pytest.raises(NotImplementedError, match="CUDA 13 build"):
        _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()
    descs = _sample_descs(h_kv=1)  # dense-only geometry does not compile the affected key kernel
    assert _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()
    descs = _sample_descs(h_kv=2)
    assert _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
def test_check_support_declines_malformed_sequence_bounds():
    descs = _sample_descs()
    args = [descs[n] for n in _INPUT_NAMES]
    cu = _metadata(torch.tensor([0, 30, 64], dtype=torch.int32, device="cuda"))
    assert _QsaTwoPassBackward(*args, sample_cu_seqlens=cu, max_seq_len=34).check_support()
    with pytest.raises(ValueError, match="cu_seqlens"):
        _QsaTwoPassBackward(*args, sample_cu_seqlens=_metadata(torch.tensor([0, 30, 64], dtype=torch.int64, device="cuda"))).check_support()
    with pytest.raises(ValueError, match="cu_seqlens"):
        _QsaTwoPassBackward(*args, sample_cu_seqlens=_metadata(torch.tensor([64], dtype=torch.int32, device="cuda"))).check_support()
    with pytest.raises(ValueError, match="max_seq_len"):
        _QsaTwoPassBackward(*args, sample_cu_seqlens=cu, max_seq_len=65).check_support()


@pytest.mark.L0
def test_execute_requires_the_sampled_sequence_bounds():
    inputs = make_inputs(64, seed=9, cu_seqlens=[0, 30, 64])
    args = [inputs[name] for name in _INPUT_NAMES]
    plan = _QsaTwoPassBackward(*args, sample_cu_seqlens=inputs["cu_seqlens"])
    plan.compile()
    outputs = [torch.empty_like(inputs[name]) for name in ("q", "k", "v")]
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="cu_seqlens"):
        plan.execute(*args, *outputs, workspace=workspace)
    plan.execute(*args, *outputs, workspace=workspace, cu_seqlens=inputs["cu_seqlens"])
    _check_against_reference(inputs, outputs)


@pytest.mark.L0
def test_check_support_declines_other_softmax_scales():

    descs = _sample_descs()
    assert _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES), softmax_scale=1 / 16).check_support()
    with pytest.raises(NotImplementedError, match="softmax_scale"):
        _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES), softmax_scale=0.1).check_support()


@pytest.mark.L0
def test_check_support_bounds_the_sequence_length():

    seq_len = (2**31 - 1) // (24 * 256) + 1
    descs = {}
    for name, (shape, dtype) in {
        "q": ((seq_len, 24, 256), torch.bfloat16),
        "k": ((seq_len, 2, 256), torch.bfloat16),
        "v": ((seq_len, 2, 256), torch.bfloat16),
        "out": ((seq_len, 24, 256), torch.bfloat16),
        "dout": ((seq_len, 24, 256), torch.bfloat16),
        "lse": ((seq_len, 24), torch.float32),
        "topk_idxs": ((seq_len, 2051), torch.int32),
    }.items():
        stride = tuple(int(torch.tensor(shape[i + 1 :]).prod()) for i in range(len(shape)))
        descs[name] = TensorDesc(dtype, shape, stride, tuple(reversed(range(len(shape)))), torch.device("cuda", torch.cuda.current_device()))
    with pytest.raises(ValueError, match="sequence length"):
        _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
def test_old_dsl_and_other_architecture_rejected():

    descs = _sample_descs()
    old_dsl = (True, ("nvidia-cutlass-dsl", "4.6.2"))
    with patch("cudnn.frost.buffers.cutedsl_state", return_value=old_dsl):
        with pytest.raises(NotImplementedError, match="4.7"):
            _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        with pytest.raises(NotImplementedError, match="SM100"):
            _QsaTwoPassBackward(*(descs[n] for n in _INPUT_NAMES)).check_support()
