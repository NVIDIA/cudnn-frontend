# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch plans (cudnn.launch_plan): a planned function's repeated calls launch its kernels from C++."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "ENABLED", True)  # these tests count calls, whatever mode the run asks for
    monkeypatch.setattr(launch_plan, "VERIFY", False)
    monkeypatch.setattr(launch_plan, "HOT_CALLS", 2)  # record a signature's second call


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
    assert list(api.plans.values()) == [-1]


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
    assert list(api.plans.values()) == [-1]


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
    assert len(calls) == 3  # the first call, then one recorded call for each relation of the stream to torch's current one
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


def test_a_signature_is_recorded_once_hot(monkeypatch):
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "HOT_CALLS", 5)
    calls = []
    api = make_api(calls)
    x = torch.randn(1000, device="cuda")
    for i in range(8):
        torch.testing.assert_close(api(x, alpha=0.5)[0], 1 + 0.5 * x, rtol=0, atol=0)
        assert len(calls) == min(i + 1, 5)  # the function runs until its fifth call, which is recorded


def declined(body, calls=3):
    """Whether a planned function with ``body`` (x, out, stream) -> None after its launch gets no plan."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    @planned
    def api(x):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, 1.0, x.numel(), stream)
        body(x, out, stream)
        return out

    for _ in range(calls):
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(api(x), x, rtol=0, atol=0)
    return list(api.plans.values()) == [-1]


def test_a_plain_call_is_planned():
    assert not declined(lambda x, out, stream: None)


def test_reading_a_tensor_is_not_planned():
    assert declined(lambda x, out, stream: x[0].item())
    assert declined(lambda x, out, stream: x[:2].tolist())
    assert declined(lambda x, out, stream: x.cpu())


def test_a_direct_kernel_call_is_not_planned():
    assert declined(lambda x, out, stream: scale_add()(out, out, 0.0, out.numel(), stream))


def test_a_tensor_the_api_owns_is_not_planned():
    """A kernel argument the call was neither passed nor allocated has no place in a later call."""
    from cudnn.launch_plan import launch

    owned = torch.zeros(1024, device="cuda")
    assert declined(lambda x, out, stream: launch(scale_add(), owned, out, 1.0, out.numel(), stream))


def test_a_tensor_the_call_keeps_is_not_planned():
    kept = []
    assert declined(lambda x, out, stream: kept.append(torch.empty_like(x)))


def test_a_host_tensor_argument_is_not_planned():
    """A host tensor's values can decide the launches without the recording seeing it."""
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    @planned
    def api(x, scale):
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, float(scale[0]), x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    x = torch.ones(1024, device="cuda")
    for value in (1.0, 1.0, 1.0, 5.0):
        torch.testing.assert_close(api(x, torch.tensor([value])), value * x, rtol=0, atol=0)
    assert list(api.plans.values()) == [-1]


def test_equal_values_of_other_types_are_other_signatures():
    from cudnn.launch_plan import _key

    x = torch.ones(8, device="cuda")
    keys = {_key([x, value, flag], (), None) for value in (1, 1.0, True) for flag in (0, False)}
    assert len(keys) == 6


def test_verify_mode_checks_every_call(monkeypatch):
    """In verify mode a call is recorded and replayed on copies of its arguments, and has to come out the same."""
    import itertools

    from cuda.bindings import driver as cuda

    from cudnn import launch_plan
    from cudnn.launch_plan import launch, planned

    monkeypatch.setattr(launch_plan, "VERIFY", True)
    monkeypatch.setattr(launch_plan, "VERIFIED", launch_plan.VERIFIED.copy())
    counter = itertools.count(1)

    def body(x, y, alpha):
        scratch = torch.empty_like(x).fill_(1.0)
        out = torch.empty_like(x)  # the second launch reads it: a plan has to fill it as the call left it
        out.zero_()
        launch(scale_add(), x, scratch, alpha, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        launch(scale_add(), scratch, out, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        launch(scale_add(), out, y, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out, torch.empty(0, device="cuda")

    pure = planned(lambda x, y: body(x, y, 0.5))
    impure = planned(lambda x, y: body(x, y, float(next(counter))))  # what it launches does not follow from its arguments
    x, y = torch.randn(1000, device="cuda"), torch.zeros(1000, device="cuda")
    matched = launch_plan.VERIFIED["matched"]
    for i in range(2):
        out, _ = pure(x, y)
        torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
        torch.testing.assert_close(y, (i + 1) * out, rtol=0, atol=0)  # the argument it updates is updated once per call
    assert launch_plan.VERIFIED["matched"] == matched + 2
    with pytest.raises(AssertionError, match="does not reproduce the call"):
        impure(x, y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # a capture cannot be checked: the call just runs
        out, _ = pure(x, y)
    graph.replay()
    torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
    assert launch_plan.VERIFIED["matched"] == matched + 2


def test_a_call_keys_the_same_however_its_arguments_are_passed():
    from cuda.bindings import driver as cuda

    from cudnn.launch_plan import launch, planned

    calls = []

    @planned
    def api(x, alpha=0.5, *, n=None):
        calls.append(alpha)
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, alpha, n or x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    x = torch.randn(1000, device="cuda")
    for out in (api(x), api(x, 0.5), api(x, alpha=0.5), api(x=x, n=None), api(alpha=0.5, x=x)):
        torch.testing.assert_close(out, 0.5 * x, rtol=0, atol=0)
    assert len(calls) == 2 and len(api.plans) == 1
    for bad in (lambda: api(x, 0.5, alpha=0.5), lambda: api(x, 0.5, None), lambda: api(x, beta=1.0), lambda: api()):
        with pytest.raises(TypeError):  # a planned signature does not answer for a call the function rejects
            bad()


def test_a_shape_that_does_not_repeat_is_only_counted(monkeypatch):
    """Under changing shapes a call costs a count by its first argument's shape: no key, and nothing kept per signature."""
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "HOT_CALLS", 16)
    monkeypatch.setattr(launch_plan, "_key", lambda *a: pytest.fail("a shape seen fewer than 8 times was keyed"))
    calls = []
    api = make_api(calls)
    for n in range(256, 256 + 3000):
        api(torch.zeros(n, device="cuda"), alpha=1.0)
    assert len(calls) == 3000 and not api.plans
