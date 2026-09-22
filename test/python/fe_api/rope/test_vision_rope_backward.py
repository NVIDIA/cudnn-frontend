# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerics, runtime layout contracts, and replay for the DSv4.1 vision VJP."""

from unittest.mock import patch

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = [pytest.mark.L0]


@pytest.fixture(autouse=True)
def supported_device():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("vision RoPE backward requires SM100")
    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("vision RoPE backward requires CuTe DSL >= 4.7.0")


def operands(n, seed=19):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, dtype):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)

    q = rand((16, n, 64), torch.bfloat16).transpose(0, 1)
    k = rand((16, 64, n), torch.bfloat16).permute(2, 0, 1)
    v = rand((16, n, 64), torch.bfloat16).transpose(0, 1)
    angles = rand((n, 1, 32), torch.float32)
    return q, k, v, angles.cos(), angles.sin()


def reference(args):
    q, k, v, cosine, sine = args

    def rotate(grad):
        first, second = grad.float().chunk(2, dim=-1)
        return torch.cat((first * cosine + second * sine, second * cosine - first * sine), dim=-1).to(torch.bfloat16)

    return torch.cat([t.reshape(q.shape[0], 1024) for t in (rotate(q), rotate(k), v)], dim=-1)


def metadata(tensor):
    stride = tensor.stride()
    order = tuple(sorted(range(tensor.ndim), key=lambda i: (stride[i], tensor.shape[i])))
    return TensorDesc(tensor.dtype, tuple(tensor.shape), stride, order, tensor.device)


def plan_for(args, *, descriptors=False):
    plan = cudnn.VisionRoPEBackward(*(metadata(t) for t in args) if descriptors else args)
    assert plan.check_support()
    assert plan.scratch_workspace_bytes() == 0
    plan.compile()
    return plan


@pytest.mark.parametrize("tokens", [1, 31, 32, 33, 1521, 1610, 4070, 5476, 8418, 8649])
def test_numerics_and_fresh_buffers(tokens):
    args = operands(tokens)
    plan = plan_for(args, descriptors=True)
    out = torch.empty((tokens, 3072), device="cuda", dtype=torch.bfloat16)
    for seed in (11, 29):
        args = operands(tokens, seed)
        snapshots = [t.clone(memory_format=torch.preserve_format) for t in args]
        expected = reference(args)
        out.fill_(float("nan"))
        assert plan.execute(*args, out) is out
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        assert torch.equal(out.view(tokens, 3, 16, 64)[:, 2].view(torch.int16), args[2].view(torch.int16))
        for actual, saved in zip(args, snapshots):
            dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
            assert torch.equal(actual.view(dtype), saved.view(dtype))


def test_dynamic_tokens_do_not_compile_on_execute():
    args = operands(31)
    plan = plan_for(args)
    from cudnn.rope import api

    with patch.object(api, "_compile", side_effect=AssertionError("execute compiled")):
        for n in (33, 1610, 1521):
            args = operands(n)
            out = torch.empty((n, 3072), device="cuda", dtype=torch.bfloat16)
            plan.execute(*args, out)
            torch.testing.assert_close(out, reference(args), atol=0, rtol=0)


def test_explicit_stream_graph_and_no_sync():
    args = operands(1610)
    plan = plan_for(args)
    out = torch.empty((1610, 3072), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    import cuda.bindings.driver as cuda

    handle = cuda.CUstream(stream.cuda_stream)
    with torch.cuda.stream(stream):
        plan.execute(*args, out, current_stream=handle)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        plan.execute(*args, out, current_stream=handle)
    for seed in (31, 59):
        fresh = operands(1610, seed)
        expected = reference(fresh)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for actual, value in zip(args, fresh):
                actual.copy_(value)
            out.fill_(float("nan"))
        # Launch explicitly on a stream different from torch's current one.
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan.execute(*args, out, current_stream=handle)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("index", range(5))
def test_bad_runtime_dtype_rejected(index):
    args = list(operands(33))
    plan = plan_for(args)
    args[index] = args[index].float() if index < 3 else args[index].bfloat16()
    out = torch.empty((33, 3072), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        plan.execute(*args, out)


@pytest.mark.parametrize("index", range(3))
def test_bad_stride_rejected_at_plan_and_execute(index):
    args = list(operands(33))
    plan = plan_for(args)
    args[index] = args[index].contiguous()
    with pytest.raises(ValueError, match="strides"):
        cudnn.VisionRoPEBackward(*args).check_support()
    out = torch.empty((33, 3072), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="strides"):
        plan.execute(*args, out)


@pytest.mark.parametrize("index", range(5))
def test_output_overlap_rejected(index):
    args = list(operands(33))
    plan = plan_for(args)
    # Include interior aliases, and FP32 tables viewed over BF16 output storage.
    storage = torch.empty(33 * 3072 + 64, device="cuda", dtype=torch.bfloat16)
    out = storage[: 33 * 3072].view(33, 3072)
    original = args[index]
    alias = storage[8:].view(torch.float32) if original.dtype == torch.float32 else storage[8:]
    args[index] = alias.as_strided(original.shape, original.stride())
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(*args, out)


def test_output_layout_alignment_and_uncompiled_rejected():
    args = operands(33)
    out = torch.empty((33, 3072), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="compile"):
        cudnn.VisionRoPEBackward(*args).execute(*args, out)
    plan = plan_for(args)
    bad_stride = torch.empty((33, 3073), device="cuda", dtype=torch.bfloat16)[:, :3072]
    with pytest.raises(ValueError, match="grad_qkv"):
        plan.execute(*args, bad_stride)
    misaligned = torch.empty(33 * 3072 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(33, 3072)
    with pytest.raises(ValueError, match="aligned"):
        plan.execute(*args, misaligned)


def test_wrapper_and_backend():
    args = operands(1521)
    out = cudnn.vision_rope_backward_wrapper(*args, backend="frost")["grad_qkv"]
    torch.testing.assert_close(out, reference(args), atol=0, rtol=0)
    with pytest.raises(ValueError, match="backend"):
        cudnn.VisionRoPEBackward(*args, backend="unknown").check_support()


def test_wrong_device_and_grid_bound():
    args = operands(33)
    descs = [metadata(t) for t in args]
    first = descs[0]
    descs[0] = TensorDesc(first.dtype, first.shape, first.stride, first.stride_order, torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA device"):
        cudnn.VisionRoPEBackward(*descs).check_support()
    for n in (0, 32 * (2**31 - 1) + 1):
        bad = TensorDesc(first.dtype, (n, 16, 64), (64, n * 64, 1), first.stride_order, first.device)
        with pytest.raises(ValueError, match="CUDA x grid"):
            cudnn.VisionRoPEBackward(bad, *map(metadata, args[1:])).check_support()


def test_old_dsl_and_other_architecture_rejected():
    from cudnn.rope import api

    args = operands(33)
    old_dsl = (True, ("nvidia-cutlass-dsl", "4.6.2"))
    with patch.object(api, "cutedsl_state", return_value=old_dsl), patch("cudnn.frost.buffers.cutedsl_state", return_value=old_dsl):
        with pytest.raises(RuntimeError, match="4.7"):
            cudnn.VisionRoPEBackward(*args).check_support()
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        with pytest.raises(NotImplementedError, match="SM100"):
            cudnn.VisionRoPEBackward(*args).check_support()
