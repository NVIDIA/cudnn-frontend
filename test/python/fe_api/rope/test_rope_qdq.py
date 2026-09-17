# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact quantization, prepared ABI, and model-shaped RoPE QDQ coverage."""

import math

import pytest
import torch

import cudnn


@pytest.fixture(autouse=True)
def supported_device():
    pytest.importorskip("triton", minversion="3.7.0")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("FROST RoPE QDQ requires SM100")


def reference(x, cache, positions, quantization):
    """Source FMA -> BF16 -> group32 UE8M0 QDQ, including signed zeros."""
    logical = x if quantization == "fp4" else x.view(-1, 1, 512)
    cosine, sine = cache[positions.long()].split(32, dim=-1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    a, b = logical[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
    negative_bs = ((b * sine).view(torch.int32) ^ -2147483648).view(torch.float32)
    tail = torch.stack((torch.addcmul(negative_bs, a, cosine), torch.addcmul(a * sine, b, cosine)), dim=-1).flatten(-2)
    rotated = torch.cat((logical[..., :-64].float(), tail.to(torch.bfloat16).float()), dim=-1)
    groups = rotated.reshape(*logical.shape[:-1], logical.shape[-1] // 32, 32)
    limit, floor = (6.0, 6 * 2.0**-126) if quantization == "fp4" else (448.0, 1e-4)
    bits = (groups.abs().amax(-1).clamp_min(floor) * (1.0 / limit)).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    normalized = (groups / scale[..., None]).clamp(-limit, limit)
    if quantization == "fp4":
        magnitude = normalized.abs()
        q = torch.zeros_like(normalized)
        for threshold, value, even_upper in (
            (0.25, 0.5, False),
            (0.75, 1.0, True),
            (1.25, 1.5, False),
            (1.75, 2.0, True),
            (2.5, 3.0, False),
            (3.5, 4.0, True),
            (5.0, 6.0, False),
        ):
            q = torch.where(magnitude >= threshold if even_upper else magnitude > threshold, value, q)
        q = torch.copysign(q, normalized)
    else:
        q = normalized.to(torch.float8_e4m3fn).float()
    return (q * scale[..., None]).to(torch.bfloat16).reshape(x.shape)


def inputs(shape, dtype=torch.int32, offsets=(0, 0, 0), quantization="fp4", generation=0):
    torch.manual_seed(907 + generation)
    n = shape[0] if quantization == "fp4" else math.prod(shape[:-1])
    parent = torch.full((math.prod(shape) + offsets[0] + 8,), -37, dtype=torch.bfloat16, device="cuda")
    x = parent[offsets[0] : offsets[0] + math.prod(shape)].view(shape)
    x.copy_(torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (1 + generation))
    p = max(n + 3, 17)
    cache_parent = torch.full((p * 64 + offsets[1] + 8,), -37, device="cuda", dtype=torch.float32)
    cache = cache_parent[offsets[1] : offsets[1] + p * 64].view(p, 64)
    angles = torch.arange(p, device="cuda")[:, None].float() * torch.linspace(0.0003, 0.13, 32, device="cuda")[None, :]
    cache.copy_(torch.cat((angles.cos(), angles.sin()), dim=-1))
    positions_parent = torch.full((n + offsets[2] + 8,), -37, device="cuda", dtype=dtype)
    positions = positions_parent[offsets[2] : offsets[2] + n]
    positions.copy_((torch.arange(n, device="cuda") * (generation + 1) + 1) % p)
    return x, cache, positions, parent, cache_parent, positions_parent


def assert_bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("quantization,heads", [("fp4", 1), ("fp4", 4), ("fp4", 32), ("fp8", 1)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_prepared_dynamic_length_alignment_and_guards(quantization, heads, dtype, monkeypatch):
    import triton
    from cudnn.rope import api

    dim = 128 if quantization == "fp4" else 512
    initial = inputs((1, heads, dim), dtype, quantization=quantization)
    op = cudnn.RopeQDQInplace(*initial[:3], quantization=quantization)
    assert op.check_support()
    op.compile()
    assert op.scratch_workspace_bytes() == 0
    cached = op._compiled_kernel
    cache_info = api._compile.cache_info()

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("execute attempted compilation")

    monkeypatch.setattr(triton, "compile", unexpected_compile)
    for generation, (n, offsets) in enumerate(((1, (0, 0, 0)), (5, (1, 1, 1)), (17, (2, 1, 1)), (64, (8, 0, 0)))):
        x, cache, positions, parent, cp, pp = inputs((n, heads, dim), dtype, offsets, quantization, generation)
        expected = reference(x, cache, positions, quantization)
        expected_parent, cache_before, pos_before = parent.clone(), cp.clone(), pp.clone()
        expected_parent[offsets[0] : offsets[0] + x.numel()].copy_(expected.flatten())
        result = op.execute(x, cache, positions)
        assert result["out"] is x
        assert_bits(parent, expected_parent)
        torch.testing.assert_close(cp, cache_before, rtol=0, atol=0)
        torch.testing.assert_close(pp, pos_before, rtol=0, atol=0)
    assert op._compiled_kernel is cached
    assert api._compile.cache_info() == cache_info


@pytest.mark.L0
@pytest.mark.parametrize("quantization", ["fp4", "fp8"])
def test_exact_midpoints_and_signed_zero(quantization):
    x, cache, positions, *_ = inputs((5, 1, 128 if quantization == "fp4" else 512), quantization=quantization)
    pattern = torch.tensor([0.0, -0.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.0, -5.0, 6.0, -6.0], device="cuda")
    x.flatten().copy_(pattern.repeat((x.numel() + pattern.numel() - 1) // pattern.numel())[: x.numel()])
    cache[:, :32].fill_(1)
    cache[:, 32:].zero_()
    expected = reference(x, cache, positions, quantization)
    result = cudnn.rope_qdq_inplace(x, cache, positions, quantization=quantization)
    assert result["out"] is x
    assert_bits(x, expected)


@pytest.mark.L0
@pytest.mark.parametrize("quantization", ["fp4", "fp8"])
def test_graph_changed_inputs_no_tensor_work_on_execute(quantization):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorWork(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"unexpected tensor operation on execute: {func}")

    args = inputs((17, 1, 128 if quantization == "fp4" else 512), offsets=(1, 1, 1), quantization=quantization)
    x, cache, positions = args[:3]
    op = cudnn.RopeQDQInplace(x, cache, positions, quantization=quantization)
    op.compile()
    launch = torch.cuda.Stream()
    launch.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=launch):
        with NoTensorWork():
            op.execute(x, cache, positions, current_stream=launch.cuda_stream)
    last = None
    for generation in (1, 2):
        new = inputs(tuple(x.shape), offsets=(1, 1, 1), quantization=quantization, generation=generation)
        expected = reference(*new[:3], quantization)
        launch.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(launch):
            x.copy_(new[0])
            cache.copy_(new[1])
            positions.copy_(new[2])
            graph.replay()
        torch.cuda.current_stream().wait_stream(launch)
        assert_bits(x, expected)
        if last is not None:
            assert not torch.equal(x.view(torch.int16), last)
        last = x.view(torch.int16).clone()
    # The detector must reject both a sync and an otherwise hidden copy.
    torch.cuda.set_sync_debug_mode("error")
    try:
        op.execute(x, cache, positions, current_stream=launch.cuda_stream)
        with pytest.raises(RuntimeError, match="synchroniz"):
            x[0, 0, 0].item()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(AssertionError, match="unexpected tensor operation"):
        with NoTensorWork():
            x.copy_(x)
    launch.synchronize()


@pytest.mark.L0
def test_declaration_runtime_mismatch_and_alias_rejection():
    x, cache, positions, *_ = inputs((5, 4, 128))
    with pytest.raises(ValueError, match="backend"):
        cudnn.RopeQDQInplace(x, cache, positions, quantization="fp4", backend="unknown").check_support()
    with pytest.raises(NotImplementedError, match="strides"):
        cudnn.RopeQDQInplace(x.transpose(0, 1), cache, positions, quantization="fp4").check_support()
    op = cudnn.RopeQDQInplace(x, cache, positions, quantization="fp4")
    with pytest.raises(RuntimeError, match="compile"):
        op.execute(x, cache, positions)
    op.compile()
    with pytest.raises(ValueError, match="positions"):
        op.execute(x, cache, positions.to(torch.int64))
    with pytest.raises(ValueError, match="planned H"):
        op.execute(x[:, :1].contiguous(), cache, positions)
    alias_cache = x.view(torch.float32).view(-1, 64)
    with pytest.raises(ValueError, match="overlap"):
        op.execute(x, alias_cache, positions)
    with pytest.raises(ValueError, match="autograd"):
        op.execute(x.detach().requires_grad_(), cache, positions)


@pytest.mark.L1
@pytest.mark.parametrize("n", [4096, 16384])
@pytest.mark.parametrize("heads,ratio", [(32, 1), (4, 1), (1, 1), (1, 2)])
def test_model_shaped_fp4(n, heads, ratio):
    x, cache, positions, *_ = inputs((n // ratio, heads, 128))
    positions.copy_((torch.arange(x.shape[0], device="cuda") * ratio) % cache.shape[0])
    expected = reference(x, cache, positions, "fp4")
    cudnn.rope_qdq_inplace(x, cache, positions, quantization="fp4")
    assert_bits(x, expected)


@pytest.mark.L1
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("sequence", [1, 5, 4096, 16384])
def test_model_shaped_fp8(batch, sequence):
    x, cache, positions, *_ = inputs((batch, sequence, 512), quantization="fp8")
    expected = reference(x, cache, positions, "fp8")
    cudnn.rope_qdq_inplace(x, cache, positions, quantization="fp8")
    assert_bits(x, expected)
