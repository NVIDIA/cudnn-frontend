# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared tail RoPE arithmetic, dynamic ABI, and asynchronous execution contracts."""

import pytest
import torch

import cudnn


@pytest.fixture(autouse=True)
def supported_device():
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("Frost tail RoPE requires CuTe DSL >=4.7.0")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Frost tail RoPE requires SM100")


def inputs(shape, generation=0):
    torch.manual_seed(648 + generation)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    angles = torch.arange(shape[0], dtype=torch.float32, device="cuda")[:, None] * torch.linspace(0.001, 0.3, 32, device="cuda")[None, :]
    return x, angles.cos(), angles.sin(), torch.full_like(x, float("nan"))


def reference(x, cosine, sine):
    """Explicit FP32 FMA sequence with an exact sign flip before real FMA."""
    tail = x.view(x.shape[0], -1, x.shape[-1])[..., -64:].float()
    a, b = tail[..., 0::2], tail[..., 1::2]
    c, s = cosine[:, None], sine[:, None]
    negative_bs = ((b * s).view(torch.int32) ^ -2147483648).view(torch.float32)
    real, imag = torch.addcmul(negative_bs, a, c), torch.addcmul(a * s, b, c)
    out = x.clone().view(x.shape[0], -1, x.shape[-1])
    out[..., -64::2] = real.to(torch.bfloat16)
    out[..., -63::2] = imag.to(torch.bfloat16)
    return out.view(x.shape)


def assert_bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.L0
def test_fma_order_boundary_signed_zero_and_subnormals():
    x, cosine, sine, out = inputs((4, 128))
    pattern = torch.tensor([0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**-126, -(2.0**-126), 1.0, -1.0], dtype=torch.bfloat16, device="cuda")
    x.copy_(pattern.repeat(x.numel() // pattern.numel()).view(x.shape))
    # This BF16 rounding boundary distinguishes fma(b,c,round(a*s)) from
    # fma(a,s,round(b*c)); the constants are an authored regression fixture.
    x[0, -64] = -0.88671875
    x[0, -63] = 0.28125
    cosine[0, 0] = -0.9531887769699097
    sine[0, 0] = -0.3023758828639984
    expected = reference(x, cosine, sine)
    assert int(expected[0, -63].view(torch.int16)) == 14368
    a, b = x[0, -64:].float()[0:2]
    wrong = torch.addcmul(b * cosine[0, 0], a, sine[0, 0]).to(torch.bfloat16)
    assert int(wrong.view(torch.int16)) == 14367
    plan = cudnn.TailRoPEForward(x, cosine, sine, out)
    plan.compile()
    plan.execute(x, cosine, sine, out)
    assert_bits(out, expected)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 128), marks=pytest.mark.L0),
        pytest.param((4, 32, 128), marks=pytest.mark.L0),
        pytest.param((17, 3, 128), marks=pytest.mark.L0),
        pytest.param((4096, 64, 512), marks=pytest.mark.L1),
        pytest.param((16384, 64, 512), marks=pytest.mark.L1),
        pytest.param((4096, 1, 512), marks=pytest.mark.L0),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_numerical_source_fma_and_readonly(shape, inverse):
    x, cosine, sine, out = inputs(shape)
    if inverse:
        sine.neg_()
    expected = reference(x, cosine, sine)
    originals = tuple(t.clone() for t in (x, cosine, sine))
    plan = cudnn.TailRoPEForward(x, cosine, sine, out)
    assert plan.check_support() and plan.scratch_workspace_bytes() == 0
    plan.compile()
    assert plan.execute(x, cosine, sine, out)["out"] is out
    assert_bits(out, expected)
    for actual, saved in zip((x, cosine, sine), originals):
        torch.testing.assert_close(
            actual.view(torch.int16 if actual.dtype == torch.bfloat16 else torch.int32),
            saved.view(torch.int16 if saved.dtype == torch.bfloat16 else torch.int32),
            rtol=0,
            atol=0,
        )


@pytest.mark.L0
def test_dynamic_length_no_execute_compile_or_allocations(monkeypatch):
    import cutlass.cute as cute
    from cudnn.rope import tail

    initial = inputs((1, 32, 128))
    plan = cudnn.TailRoPEForward(*initial)
    plan.compile()
    plan.execute(*initial)
    torch.cuda.synchronize()
    compiled, cache = plan._compiled_kernel, tail._compile.cache_info()

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated, synchronized, or compiled")

    for generation, n in enumerate((0, 5, 257, 4096)):
        values = inputs((n, 32, 128), generation)
        expected = reference(*values[:3]) if n else values[0].clone()
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                guard.setattr(torch, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan.execute(*values)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        assert_bits(values[-1], expected)
    assert plan._compiled_kernel is compiled and tail._compile.cache_info() == cache


@pytest.mark.L0
def test_graph_replay_changes_inputs_and_poison():
    values = inputs((257, 32, 128))
    plan = cudnn.TailRoPEForward(*values)
    plan.compile()
    plan.execute(*values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(*values)
    previous = None
    for generation in (1, 2):
        changed = inputs(values[0].shape, generation)
        for target, source in zip(values[:3], changed[:3]):
            target.copy_(source)
        values[-1].fill_(float("nan"))
        expected = reference(*values[:3])
        graph.replay()
        assert_bits(values[-1], expected)
        if previous is not None:
            assert not torch.equal(previous, values[-1])
        previous = values[-1].clone()


@pytest.mark.L0
def test_explicit_nondefault_stream_and_wrapper():
    x, cosine, sine, out = inputs((257, 32, 128))
    plan = cudnn.TailRoPEForward(x, cosine, sine, out)
    plan.compile()
    plan.execute(x, cosine, sine, out)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        x.fill_(7)
        cosine.fill_(1)
        sine.zero_()
    plan.execute(x, cosine, sine, out, current_stream=stream.cuda_stream)
    stream.synchronize()
    assert_bits(out, torch.full_like(out, 7))
    result = cudnn.tail_rope(x, cosine, sine, stream=stream.cuda_stream)
    stream.synchronize()
    assert list(result.keys()) == ["out"] and result["out"].data_ptr() != x.data_ptr()
    assert_bits(result["out"], out)


@pytest.mark.L0
def test_metadata_device_binding_is_stable(monkeypatch):
    from cudnn.api_base import TensorDesc
    from cudnn.rope import tail

    tensors = inputs((1, 128))
    descs = [TensorDesc(t.dtype, tuple(t.shape), tuple(t.stride()), tuple(reversed(range(t.ndim))), torch.device("cuda")) for t in tensors]
    actual_device = torch.cuda.current_device()
    plan = cudnn.TailRoPEForward(*descs)
    assert all(d.device.index is None for d in descs)
    assert all(d.device.index == actual_device for d in plan._descs.values())
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_device", lambda: actual_device + 1)
        patch.setattr(torch.cuda, "get_device_capability", lambda device: ((10, 0) if device.index == actual_device else (9, 0)))
        patch.setattr(tail, "_compile", lambda h, d, device, capability: calls.append((h, d, device, capability)) or object())
        plan.compile()
    assert calls == [(1, 128, actual_device, (10, 0))]


@pytest.mark.L0
@pytest.mark.parametrize("field", range(4))
def test_runtime_alignment_and_layout_rejection(field):
    values = list(inputs((4, 32, 128)))
    plan = cudnn.TailRoPEForward(*values)
    plan.compile()
    value = values[field]
    parent = torch.empty(value.numel() + 1, dtype=value.dtype, device=value.device)
    bad = parent[1:].view(value.shape)
    values[field] = bad
    with pytest.raises(ValueError, match="aligned"):
        plan.execute(*values)
    values[field] = torch.empty((*value.shape, 2), dtype=value.dtype, device=value.device)[..., 0]
    with pytest.raises(ValueError, match="contiguous"):
        plan.execute(*values)
    with pytest.raises(NotImplementedError, match="contiguous"):
        cudnn.TailRoPEForward(*values).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("field", [0, 1, 2])
def test_output_overlap_rejected(field):
    values = list(inputs((4, 128)))
    if field:
        owner = torch.empty(512, dtype=torch.float32, device="cuda")
        values[field] = owner[:128].view(4, 32)
        values[-1] = owner.view(torch.bfloat16)[:512].view(4, 128)
    else:
        values[-1] = values[0]
    plan = cudnn.TailRoPEForward(*values)
    plan.compile()
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(*values)


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["backend", "dtype", "shape", "rank", "grad", "device"])
def test_unsupported_declarations(kind):
    values = list(inputs((4, 128)))
    kwargs = {}
    if kind == "backend":
        kwargs["backend"] = "unknown"
    elif kind == "dtype":
        values[0] = values[0].float()
    elif kind == "shape":
        values[2] = values[2][:3]
    elif kind == "rank":
        values[0] = values[0].flatten()
    elif kind == "grad":
        values[0].requires_grad_()
    else:
        values = [value.cpu() for value in values]
    with pytest.raises((ValueError, NotImplementedError)):
        cudnn.TailRoPEForward(*values, **kwargs).check_support()


@pytest.mark.L0
def test_old_dsl_declines_before_kernel_load(monkeypatch):
    from cudnn.frost import buffers

    plan = cudnn.TailRoPEForward(*inputs((1, 128)))
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(RuntimeError, match="4.7.0"):
        plan.check_support()


@pytest.mark.L1
def test_signed_i64_extent_beyond_two_billion():
    # Exercise the first rows past 2**31 elements, not only the boundary itself.
    free, _ = torch.cuda.mem_get_info()
    if free < 12 * 2**30:
        pytest.skip("Wide-index stress test requires 12 GiB free")
    shape = (65537, 64, 512)
    x = torch.full(shape, 1, dtype=torch.bfloat16, device="cuda")
    out = torch.full_like(x, float("nan"))
    cosine = torch.ones(shape[0], 32, dtype=torch.float32, device="cuda")
    sine = torch.zeros_like(cosine)
    plan = cudnn.TailRoPEForward(x, cosine, sine, out)
    plan.compile()
    plan.execute(x, cosine, sine, out)
    assert torch.equal(out.view(torch.int16), x.view(torch.int16))
