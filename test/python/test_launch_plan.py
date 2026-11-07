# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch plans (cudnn.launch_plan): a planned function's repeated calls launch its kernels from C++."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")


_compiled = {}


def scale_add():
    """y += alpha * x over n elements, compiled once with tvm-ffi."""
    if "kernel" in _compiled:
        return _compiled["kernel"]
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def kernel(gx: cute.Tensor, gy: cute.Tensor, alpha: cutlass.Float32, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gy[i] = gy[i] + alpha * gx[i]

    @cute.jit
    def host(mx: cute.Tensor, my: cute.Tensor, alpha: cutlass.Float32, n: cutlass.Int32, stream: cuda.CUstream):
        kernel(mx, my, alpha, n).launch(grid=((n + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)

    x = torch.zeros(256, device="cuda")
    _compiled["kernel"] = cute.compile(
        host,
        from_dlpack(x, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(x, assumed_align=16).mark_layout_dynamic(),
        cutlass.Float32(1.0),
        cutlass.Int32(256),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options="--enable-tvm-ffi",
    )
    return _compiled["kernel"]


def make_api(calls, extra=None, env=()):
    """An API whose GPU work is two launches, a scratch buffer it fills and drops, and a zeroed output."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    @planned(env=env)
    def api(x, *, alpha):
        calls.append(alpha)
        kernel = scale_add()
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        scratch = torch.empty_like(x).fill_(1.0)
        out = torch.zeros_like(x)
        launch(kernel, x, scratch, alpha, x.numel(), stream)
        launch(kernel, scratch, out, 1.0, x.numel(), stream)
        if extra is not None:
            extra(out)
        return out, x

    return api


def test_plan_runs_the_launches():
    calls = []
    api = make_api(calls)
    for i in range(4):
        x = torch.randn(1000, device="cuda")
        out, same = api(x, alpha=0.5)
        torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
        assert same is x
    assert len(calls) == 2  # the first call and the recorded one; the plan runs the rest
    (plan,) = api.plans.values()
    assert type(plan).__name__ == "_Plan"


def test_other_torch_work_is_not_planned():
    calls = []
    api = make_api(calls, extra=lambda out: out.add_(0.0))
    for _ in range(3):
        x = torch.randn(1000, device="cuda")
        torch.testing.assert_close(api(x, alpha=2.0)[0], 1 + 2.0 * x, rtol=0, atol=0)
    assert len(calls) == 3
    assert list(api.plans.values()) == [2]


def test_unwatched_raw_launch_is_caught_when_watched():
    """A kwargs wrapper over a raw tvm_ffi.Function (as the compiled cache reloads kernels) called directly declines."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    from cudnn.launch_plan import watch

    raw = make_kwargs_wrapper(scale_add().__tvm_ffi_object__(), arg_names=["x", "y", "alpha", "n", "stream"])
    watch(raw)
    calls = []
    api = make_api(calls, extra=lambda out: raw(out, out, 0.0, out.numel(), torch.cuda.current_stream().cuda_stream))
    for _ in range(3):
        x = torch.randn(1000, device="cuda")
        torch.testing.assert_close(api(x, alpha=1.0)[0], 1 + x, rtol=0, atol=0)
    assert list(api.plans.values()) == [2]


def test_misaligned_call_runs_the_function():
    calls = []
    api = make_api(calls)
    for _ in range(2):
        api(torch.randn(1000, device="cuda"), alpha=1.0)
    x = torch.randn(1001, device="cuda")[1:]  # 4-byte aligned; the recorded call's x was 16-byte aligned
    with pytest.raises(ValueError, match="Misaligned"):  # the function ran: its kernel takes 16-byte aligned data only
        api(x, alpha=1.0)
    assert len(calls) == 3


def test_plans_off(monkeypatch):
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "ENABLED", False)

    def api(x):
        return x

    assert launch_plan.planned(api) is api


def test_overlapping_arguments_run_the_function():
    """A call whose argument tensors overlap differently from the recorded call's runs the function."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    calls = []

    @planned
    def accumulate(x, y):
        calls.append(1)
        launch(scale_add(), x, y, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return y

    for _ in range(3):
        x, y = torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")
        expected = x + y
        torch.testing.assert_close(accumulate(x, y), expected, rtol=0, atol=0)
    assert len(calls) == 2
    shared = torch.randn(2048, device="cuda")
    expected = shared[:1024] * 2
    accumulate(shared[:1024], shared[:1024])  # same signature, but x and y alias
    assert len(calls) == 3
    torch.testing.assert_close(shared[:1024], expected, rtol=0, atol=0)


def test_environment_joins_the_key(monkeypatch):
    calls = []
    api = make_api(calls, env=("CUDNN_TEST_LAUNCH_PLAN_KNOB",))
    x = torch.randn(1000, device="cuda")
    for _ in range(3):
        api(x, alpha=1.0)
    assert len(calls) == 2
    monkeypatch.setenv("CUDNN_TEST_LAUNCH_PLAN_KNOB", "1")
    api(x, alpha=1.0)
    assert len(calls) == 3


def test_plans_are_bounded(monkeypatch):
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "MAX_PLANS", 2)
    api = make_api([])
    for n in (256, 512, 768, 1024):
        for _ in range(3):
            api(torch.randn(n, device="cuda"), alpha=1.0)
    assert len(api.plans) == 2


def test_integer_stream_argument():
    """A named stream parameter carrying a plain handle is bound per call, fills included, on any stream."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    calls = []

    @planned(stream="current_stream")
    def api(x, *, current_stream):
        calls.append(1)
        with torch.cuda.stream(torch.cuda.ExternalStream(current_stream)):
            out = torch.full_like(x, -1.0)
            launch(scale_add(), x, out, 1.0, x.numel(), cuda.CUstream(current_stream))
        return out

    side = torch.cuda.Stream()
    for handle in [torch.cuda.current_stream().cuda_stream] * 3 + [side.cuda_stream] * 4:
        x = torch.randn(1 << 20, device="cuda")
        side.wait_stream(torch.cuda.current_stream())
        out = api(x, current_stream=handle)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x - 1, rtol=0, atol=0)
    assert len(calls) == 4  # two calls on each relation of the stream to torch's current one, then plans
    assert all(type(p).__name__ == "_Plan" for p in api.plans.values())


def test_overlapping_arguments_at_another_offset_run_the_function():
    """A plan recorded with two arguments as one tensor does not serve them overlapping at another offset."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    calls = []

    @planned
    def accumulate(x, y):
        calls.append(1)
        launch(scale_add(), x, y, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return y

    for _ in range(3):
        same = torch.randn(1024, device="cuda")
        accumulate(same, same)
    assert len(calls) == 2
    shared = torch.randn(2048, device="cuda")
    accumulate(shared[:1024], shared[16:1040])  # same signature, overlapping, but 64 bytes apart
    assert len(calls) == 3
