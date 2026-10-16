# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the index-list sparse (QSA) backward on SM100, ``cudnn.sdpa.bwd.SparseGqaBwdDslSm100``."""

from unittest.mock import patch

import pytest
import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import TensorDesc
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

from sparse_gqa_bwd_reference import assert_grads_close, make_inputs, reference_backward

_INPUT_NAMES = ("q", "k", "v", "out", "dout", "lse", "topk_idxs")


@pytest.fixture(autouse=True)
def supported_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("QSA backward requires SM100")
    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("QSA backward requires CuTe DSL >= 4.7.0")


def _metadata(tensor):
    stride = tensor.stride()
    order = tuple(sorted(range(tensor.ndim), key=lambda i: (stride[i], tensor.shape[i])))
    return TensorDesc(tensor.dtype, tuple(tensor.shape), stride, order, tensor.device)


def _check_against_reference(inputs, grads):
    expected = reference_backward(*(inputs[name] for name in _INPUT_NAMES))
    for name, actual, ref in zip(("dq", "dk", "dv"), grads, expected):
        assert_grads_close(name, actual, ref)


@pytest.mark.L0
@pytest.mark.parametrize(
    "seq_len, grad_scale",
    [
        (2, 1.0),  # tail tokens only (a single token has dQ = dK = 0, which a relative bound cannot score)
        (5, 1.0),  # one complete block and a tail
        (4099, 1.0),  # every query in the dense kernel, sparse rows above 2050 included
        (16384, 1.0),  # every query in the dense kernel with the larger dense unit
        (24577, 1.0),  # a single query on the query-major kernel
        (28000, 1.0),  # dense and query-major kernels, two KV passes
        (28000, 1e-7),  # training-scale gradients, which rule out reduced-range accumulators
        (32769, 1.0),  # three KV passes and a partial last block
    ],
)
def test_qsa_backward_matches_reference(seq_len, grad_scale):
    from cudnn.sdpa import bwd

    inputs = make_inputs(seq_len, seed=seq_len, grad_scale=grad_scale)
    grads = bwd.sparse_gqa_bwd_wrapper_sm100(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("seq_len", [65536, 262144])
def test_qsa_backward_long_sequences(seq_len):
    from cudnn.sdpa import bwd

    inputs = make_inputs(seq_len, seed=seq_len)
    grads = bwd.sparse_gqa_bwd_wrapper_sm100(*(inputs[name] for name in _INPUT_NAMES))
    _check_against_reference(inputs, (grads["dq"], grads["dk"], grads["dv"]))


@pytest.mark.L0
def test_wrapper_on_explicit_stream_with_preallocated_outputs(monkeypatch):
    from cudnn.sdpa import bwd
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_sparse_d256_f16 as kernel

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
    grads = bwd.sparse_gqa_bwd_wrapper_sm100(*args, dq=dq, dk=dk, dv=dv, stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.current_stream().wait_stream(side)
    assert streams == [side.cuda_stream] * 6
    assert grads["dq"] is dq and grads["dk"] is dk and grads["dv"] is dv
    _check_against_reference(inputs, (dq, dk, dv))


@pytest.mark.L0
@pytest.mark.parametrize("seq_len", [4099, 28000])  # the dense-only and the dense + sparse launch sequences
def test_execute_allocates_nothing_and_never_synchronizes(seq_len, monkeypatch, compile_allocates_nothing):
    from cudnn.sdpa import bwd
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_sparse_d256_f16 as kernel

    # An empty artifact cache makes this a cold compile.
    monkeypatch.setattr(kernel, "_COMPILED", {})
    inputs = make_inputs(seq_len, seed=5)
    args = [inputs[name] for name in _INPUT_NAMES]
    plan = bwd.SparseGqaBwdDslSm100(*args)
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
@pytest.mark.parametrize("seq_len", [4099, 28000])  # the dense-only and the dense + sparse launch sequences
def test_first_execute_under_cuda_graph_capture_replays_new_inputs(seq_len, monkeypatch):
    from cudnn.sdpa import bwd
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_sparse_d256_f16 as kernel

    # Fresh artifacts make the capture hold their first launch.
    monkeypatch.setattr(kernel, "_COMPILED", {})
    first, second = make_inputs(seq_len, seed=21), make_inputs(seq_len, seed=22)
    static = {name: first[name].clone() for name in _INPUT_NAMES}
    args = [static[name] for name in _INPUT_NAMES]
    plan = bwd.SparseGqaBwdDslSm100(*args)
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
    from cudnn.sdpa import bwd

    inputs = make_inputs(64, seed=7)
    args = [inputs[name] for name in _INPUT_NAMES]
    plan = bwd.SparseGqaBwdDslSm100(*args)
    plan.compile()
    outputs = [torch.empty_like(inputs[name]) for name in ("q", "k", "v")]
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(*args, *outputs)
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="dq must match"):
        plan.execute(*args, outputs[0][:32], *outputs[1:], workspace=workspace)


def _sample_descs(seq_len=64):
    q = torch.empty((seq_len, 24, 256), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((seq_len, 2, 256), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((seq_len, 24), dtype=torch.float32, device="cuda")
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
        ("k", {"shape": (64, 4, 256), "stride": (1024, 256, 1)}, ValueError, "k must have shape"),
        ("q", {"shape": (64, 24, 128), "stride": (3072, 128, 1)}, ValueError, "q must have shape"),
        ("topk_idxs", {"shape": (64, 2048), "stride": (2048, 1)}, ValueError, "topk_idxs must have shape"),
        ("lse", {"dtype": torch.bfloat16}, ValueError, "lse"),
        ("dout", {"stride": (256, 64 * 256, 1), "stride_order": (2, 0, 1)}, NotImplementedError, "contiguous dout"),
    ],
)
def test_check_support_declines(name, changes, error, match):
    from cudnn.sdpa import bwd

    descs = _with(_sample_descs(), name, **changes)
    with pytest.raises(error, match=match):
        bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
def test_check_support_declines_other_softmax_scales():
    from cudnn.sdpa import bwd

    descs = _sample_descs()
    assert bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES), softmax_scale=1 / 16).check_support()
    with pytest.raises(NotImplementedError, match="softmax_scale"):
        bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES), softmax_scale=0.1).check_support()


@pytest.mark.L0
def test_check_support_bounds_the_sequence_length():
    from cudnn.sdpa import bwd

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
        bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES)).check_support()


@pytest.mark.L0
def test_old_dsl_and_other_architecture_rejected():
    from cudnn.sdpa import bwd

    descs = _sample_descs()
    old_dsl = (True, ("nvidia-cutlass-dsl", "4.6.2"))
    with patch("cudnn.frost.buffers.cutedsl_state", return_value=old_dsl):
        with pytest.raises(NotImplementedError, match="4.7"):
            bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES)).check_support()
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        with pytest.raises(NotImplementedError, match="SM100"):
            bwd.SparseGqaBwdDslSm100(*(descs[n] for n in _INPUT_NAMES)).check_support()
