# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The C++ launcher (cudnn.cpp_launcher): a function's repeated calls launch its kernels from C++."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    from cudnn import cpp_launcher

    monkeypatch.setattr(cpp_launcher, "ENABLED", True)  # these tests count calls, whatever mode the run asks for
    monkeypatch.setattr(cpp_launcher, "VERIFY", False)
    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", 2)  # record a signature's second call


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

    from cudnn.cpp_launcher import launch, cpp_launched

    @cpp_launched(env=env)
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


def test_recording_runs_the_launches():
    calls = []
    api = make_api(calls)
    for i in range(4):
        x = torch.randn(1000, device="cuda")
        out, same = api(x, alpha=0.5)
        torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
        assert same is x
    assert len(calls) == 2  # the first call and the recorded one; the recording runs the rest
    (recording,) = api.recordings.values()
    assert type(recording).__name__ == "_Recording"


def test_a_kernel_with_many_arguments_is_recorded():
    """A recording carries however many arguments a kernel takes; a fused FROST launch takes more than a hundred."""
    import tvm_ffi

    from cudnn.cpp_launcher import launch, cpp_launched

    seen = []
    sink = tvm_ffi.convert(lambda *values: seen.append(values))

    @cpp_launched
    def api(x):
        launch(sink, *range(300))
        return x

    x = torch.zeros(8, device="cuda")
    for _ in range(4):
        assert api(x) is x
    assert seen == [tuple(range(300))] * 4
    (recording,) = api.recordings.values()
    assert type(recording).__name__ == "_Recording"


def test_float_arguments_are_replayed_bit_for_bit():
    """A kernel's float arguments reach a replay with their bits, the infinities and NaN that JSON cannot carry included."""
    import math
    import struct

    import tvm_ffi

    from cudnn.cpp_launcher import launch, cpp_launched

    values = (math.inf, -math.inf, math.nan, -0.0, 0.1, 5e-324)
    bits = [struct.pack("d", v) for v in values]
    seen = []
    sink = tvm_ffi.convert(lambda *got: seen.append([struct.pack("d", v) for v in got]))

    @cpp_launched
    def api(x):
        launch(sink, *values)
        return x

    x = torch.zeros(8, device="cuda")
    for _ in range(4):
        assert api(x) is x
    assert seen == [bits] * 4
    (recording,) = api.recordings.values()
    assert type(recording).__name__ == "_Recording"


class OddFloat(float):
    """A float whose truth value and repr say nothing of its value."""

    def __bool__(self):
        raise ValueError("no truth value")

    def __repr__(self):
        return "odd"


@pytest.mark.parametrize("number", ["float", "numpy.float32", "numpy.float64", "OddFloat"])
def test_a_negative_zero_argument_is_not_the_recorded_zero(number):
    """0.0 and -0.0 compare equal and are different kernel arguments: each has its own recording, whatever the float's type."""
    import struct

    import numpy
    import tvm_ffi

    from cudnn.cpp_launcher import launch, cpp_launched

    number = {"float": float, "numpy.float32": numpy.float32, "numpy.float64": numpy.float64, "OddFloat": OddFloat}[number]
    seen = []
    sink = tvm_ffi.convert(lambda got: seen.append(struct.pack("d", got)))

    @cpp_launched
    def api(x, scale):
        launch(sink, float(scale))
        return x

    x = torch.zeros(8, device="cuda")
    scales = [number(v) for v in (0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 1.0, 1.0, 2.0, 2.0)]
    for scale in scales:
        assert api(x, scale) is x
    assert seen == [struct.pack("d", scale) for scale in scales]
    assert [type(r).__name__ for r in api.recordings.values()] == ["_Recording"] * 4


def test_a_number_that_gives_no_double_is_keyed_by_its_value():
    import struct

    import numpy
    import tvm_ffi

    from cudnn.cpp_launcher import launch, cpp_launched

    class NoDouble(numpy.float32):
        def __float__(self):
            raise TypeError("no double")

    seen = []
    sink = tvm_ffi.convert(lambda got: seen.append(struct.pack("d", got)))

    @cpp_launched
    def api(x, scale):
        launch(sink, scale.item())
        return x

    x = torch.zeros(8, device="cuda")
    for value in (0.5, 0.5, 0.5, 0.25, 0.25):
        assert api(x, NoDouble(value)) is x
    assert seen == [struct.pack("d", value) for value in (0.5, 0.5, 0.5, 0.25, 0.25)]
    assert [type(r).__name__ for r in api.recordings.values()] == ["_Recording"] * 2


def test_a_recorded_function_called_inside_a_recording_runs_its_python():
    """A function with a recording of its own, called while another call is recorded, runs its Python: a replay's launches
    are made in C++, where the outer recording would not see them."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import cpp_launched, launch

    @cpp_launched
    def inner(x):
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, 2.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    @cpp_launched
    def outer(x):
        mid = inner(x)
        out = torch.zeros_like(x)
        launch(scale_add(), mid, out, 3.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    for _ in range(4):
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(inner(x), 2 * x, rtol=0, atol=0)
    assert [type(r).__name__ for r in inner.recordings.values()] == ["_Recording"]
    for _ in range(5):
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(outer(x), 6 * x, rtol=0, atol=0)
    assert [type(r).__name__ for r in outer.recordings.values()] == ["_Recording"]


def test_other_torch_work_is_not_recorded():
    calls = []
    api = make_api(calls, extra=lambda out: out.add_(0.0))
    for _ in range(3):
        x = torch.randn(1000, device="cuda")
        torch.testing.assert_close(api(x, alpha=2.0)[0], 1 + 2.0 * x, rtol=0, atol=0)
    assert len(calls) == 3
    assert list(api.recordings.values()) == [-1]


def test_unwatched_raw_launch_is_caught_when_watched():
    """A kwargs wrapper over a raw tvm_ffi.Function (as the compiled cache reloads kernels) called directly declines."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    from cudnn.cpp_launcher import watch

    raw = make_kwargs_wrapper(scale_add().__tvm_ffi_object__(), arg_names=["x", "y", "alpha", "n", "stream"])
    watch(raw)
    calls = []
    api = make_api(calls, extra=lambda out: raw(out, out, 0.0, out.numel(), torch.cuda.current_stream().cuda_stream))
    for _ in range(3):
        x = torch.randn(1000, device="cuda")
        torch.testing.assert_close(api(x, alpha=1.0)[0], 1 + x, rtol=0, atol=0)
    assert list(api.recordings.values()) == [-1]


def test_misaligned_call_runs_the_function():
    calls = []
    api = make_api(calls)
    for _ in range(2):
        api(torch.randn(1000, device="cuda"), alpha=1.0)
    x = torch.randn(1001, device="cuda")[1:]  # 4-byte aligned; the recorded call's x was 16-byte aligned
    with pytest.raises(ValueError, match="Misaligned"):  # the function ran: its kernel takes 16-byte aligned data only
        api(x, alpha=1.0)
    assert len(calls) == 3


def test_launcher_off(monkeypatch):
    from cudnn import cpp_launcher

    monkeypatch.setattr(cpp_launcher, "ENABLED", False)

    def api(x):
        return x

    assert cpp_launcher.cpp_launched(api) is api


def test_overlapping_arguments_run_the_function():
    """A call whose argument tensors overlap differently from the recorded call's runs the function."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    calls = []

    @cpp_launched
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
    api = make_api(calls, env=("CUDNN_TEST_CPP_LAUNCHER_KNOB",))
    x = torch.randn(1000, device="cuda")
    for _ in range(3):
        api(x, alpha=1.0)
    assert len(calls) == 2
    monkeypatch.setenv("CUDNN_TEST_CPP_LAUNCHER_KNOB", "1")
    api(x, alpha=1.0)
    assert len(calls) == 3


def test_recordings_are_bounded(monkeypatch):
    from cudnn import cpp_launcher

    monkeypatch.setattr(cpp_launcher, "MAX_RECORDINGS", 2)
    api = make_api([])
    for n in (256, 512, 768, 1024):
        for _ in range(3):
            api(torch.randn(n, device="cuda"), alpha=1.0)
    assert len(api.recordings) == 2


def test_integer_stream_argument():
    """A named stream parameter carrying a plain handle is bound per call, fills included, on any stream."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    calls = []

    @cpp_launched(stream="current_stream")
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
    assert all(type(p).__name__ == "_Recording" for p in api.recordings.values())


def test_overlapping_arguments_at_another_offset_run_the_function():
    """A recording recorded with two arguments as one tensor does not serve them overlapping at another offset."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    calls = []

    @cpp_launched
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
    from cudnn import cpp_launcher

    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", 5)
    calls = []
    api = make_api(calls)
    x = torch.randn(1000, device="cuda")
    for i in range(8):
        torch.testing.assert_close(api(x, alpha=0.5)[0], 1 + 0.5 * x, rtol=0, atol=0)
        assert len(calls) == min(i + 1, 5)  # the function runs until its fifth call, which is recorded


def declined(body, calls=3):
    """Whether a ``cpp_launched`` function with ``body`` (x, out, stream) -> None after its launch gets no recording."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    @cpp_launched
    def api(x):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, 1.0, x.numel(), stream)
        body(x, out, stream)
        return out

    for _ in range(calls):
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(api(x), x, rtol=0, atol=0)
    return list(api.recordings.values()) == [-1]


def test_a_plain_call_is_recorded():
    assert not declined(lambda x, out, stream: None)


def test_reading_a_tensor_is_not_recorded():
    assert declined(lambda x, out, stream: x[0].item())
    assert declined(lambda x, out, stream: x[:2].tolist())
    assert declined(lambda x, out, stream: x.cpu())


def test_a_direct_kernel_call_is_not_recorded():
    assert declined(lambda x, out, stream: scale_add()(out, out, 0.0, out.numel(), stream))


def test_a_tensor_the_api_owns_is_not_recorded():
    """A kernel argument the call was neither passed nor allocated has no place in a later call."""
    from cudnn.cpp_launcher import launch

    owned = torch.zeros(1024, device="cuda")
    assert declined(lambda x, out, stream: launch(scale_add(), owned, out, 1.0, out.numel(), stream))


def test_declared_scratch_is_replayed_in_a_buffer_of_its_own():
    """A buffer the API keeps, declared scratch, does not decline the recording; a replay leaves the API's buffer alone."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import cpp_launched, launch, scratch

    owned = torch.zeros(1024, device="cuda")

    @cpp_launched
    def api(x):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        out = torch.zeros_like(x)
        launch(scale_add(), x, scratch(owned).fill_(0.0), 1.0, x.numel(), stream)  # owned = x
        launch(scale_add(), owned, out, 2.0, x.numel(), stream)  # out = 2 x
        return out

    for _ in range(2):
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(api(x), 2 * x, rtol=0, atol=0)
    (recording,) = api.recordings.values()
    assert type(recording).__name__ == "_Recording"
    owned.fill_(7.0)
    x = torch.randn(1024, device="cuda")
    torch.testing.assert_close(api(x), 2 * x, rtol=0, atol=0)
    assert bool((owned == 7.0).all())


def test_a_tupledict_result_is_rebuilt():
    from cuda.bindings import driver as cuda

    from cudnn._tupledict import TupleDict
    from cudnn.cpp_launcher import cpp_launched, launch

    @cpp_launched
    def api(x):
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, 3.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return TupleDict(out=out, same=x, none=None)

    for _ in range(4):
        x = torch.randn(256, device="cuda")
        result = api(x)
        assert type(result) is TupleDict and list(result.keys()) == ["out", "same", "none"]
        out, same, none = result
        torch.testing.assert_close(out, 3 * x, rtol=0, atol=0)
        assert same is x and none is None
    assert [type(r).__name__ for r in api.recordings.values()] == ["_Recording"]


def test_a_tensor_the_call_keeps_is_not_recorded():
    kept = []
    not_recorded = declined(lambda x, out, stream: kept.append(torch.empty_like(x)))
    assert not_recorded


def test_a_host_tensor_argument_is_not_recorded():
    """A host tensor's values can decide the launches without the recording seeing it."""
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    @cpp_launched
    def api(x, scale):
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, float(scale[0]), x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    x = torch.ones(1024, device="cuda")
    for value in (1.0, 1.0, 1.0, 5.0):
        torch.testing.assert_close(api(x, torch.tensor([value])), value * x, rtol=0, atol=0)
    assert list(api.recordings.values()) == [-1]


def test_equal_values_of_other_types_are_other_signatures():
    from cudnn.cpp_launcher import _key

    x = torch.ones(8, device="cuda")
    keys = {_key([x, value, flag], (), None) for value in (1, 1.0, True) for flag in (0, False)}
    assert len(keys) == 6


def test_verify_mode_checks_every_call(monkeypatch):
    """In verify mode a call is recorded and replayed on copies of its arguments, and has to come out the same."""
    import itertools

    from cuda.bindings import driver as cuda

    from cudnn import cpp_launcher
    from cudnn.cpp_launcher import launch, cpp_launched

    monkeypatch.setattr(cpp_launcher, "VERIFY", True)
    monkeypatch.setattr(cpp_launcher, "VERIFIED", cpp_launcher.VERIFIED.copy())
    counter = itertools.count(1)

    def body(x, y, alpha):
        scratch = torch.empty_like(x).fill_(1.0)
        out = torch.empty_like(x)  # the second launch reads it: a recording has to fill it as the call left it
        out.zero_()
        launch(scale_add(), x, scratch, alpha, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        launch(scale_add(), scratch, out, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        launch(scale_add(), out, y, 1.0, x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out, torch.empty(0, device="cuda")

    pure = cpp_launched(lambda x, y: body(x, y, 0.5))
    impure = cpp_launched(lambda x, y: body(x, y, float(next(counter))))  # what it launches does not follow from its arguments
    x, y = torch.randn(1000, device="cuda"), torch.zeros(1000, device="cuda")
    matched = cpp_launcher.VERIFIED["matched"]
    for i in range(2):
        out, _ = pure(x, y)
        torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
        torch.testing.assert_close(y, (i + 1) * out, rtol=0, atol=0)  # the argument it updates is updated once per call
    assert cpp_launcher.VERIFIED["matched"] == matched + 2
    with pytest.raises(AssertionError, match="does not reproduce the call"):
        impure(x, y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # a capture cannot be checked: the call just runs
        out, _ = pure(x, y)
    graph.replay()
    torch.testing.assert_close(out, 1 + 0.5 * x, rtol=0, atol=0)
    assert cpp_launcher.VERIFIED["matched"] == matched + 2


def test_verify_mode_compares_a_float_argument_by_its_bits(monkeypatch):
    """A NaN is not equal to itself, and it is the same argument in the call and in its replay."""
    import math

    import tvm_ffi

    from cudnn import cpp_launcher
    from cudnn.cpp_launcher import launch, cpp_launched

    monkeypatch.setattr(cpp_launcher, "VERIFY", True)
    monkeypatch.setattr(cpp_launcher, "VERIFIED", cpp_launcher.VERIFIED.copy())
    sink = tvm_ffi.convert(lambda got: None)

    @cpp_launched
    def api(x, scale):
        launch(sink, scale)
        return x

    matched = cpp_launcher.VERIFIED["matched"]
    x = torch.zeros(8, device="cuda")
    assert api(x, math.nan) is x
    assert cpp_launcher.VERIFIED["matched"] == matched + 1


def test_a_call_keys_the_same_however_its_arguments_are_passed():
    from cuda.bindings import driver as cuda

    from cudnn.cpp_launcher import launch, cpp_launched

    calls = []

    @cpp_launched
    def api(x, alpha=0.5, *, n=None):
        calls.append(alpha)
        out = torch.zeros_like(x)
        launch(scale_add(), x, out, alpha, n or x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    x = torch.randn(1000, device="cuda")
    for out in (api(x), api(x, 0.5), api(x, alpha=0.5), api(x=x, n=None), api(alpha=0.5, x=x)):
        torch.testing.assert_close(out, 0.5 * x, rtol=0, atol=0)
    assert len(calls) == 2 and len(api.recordings) == 1
    for bad in (lambda: api(x, 0.5, alpha=0.5), lambda: api(x, 0.5, None), lambda: api(x, beta=1.0), lambda: api()):
        with pytest.raises(TypeError):  # a recorded signature does not answer for a call the function rejects
            bad()


def test_a_shape_that_does_not_repeat_is_only_counted(monkeypatch):
    """Under changing shapes a call costs a count by its first argument's shape: no key, and nothing kept per signature."""
    from cudnn import cpp_launcher

    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", 16)
    monkeypatch.setattr(cpp_launcher, "_key", lambda *a: pytest.fail("a shape seen fewer than 8 times was keyed"))
    calls = []
    api = make_api(calls)
    for n in range(256, 256 + 3000):
        api(torch.zeros(n, device="cuda"), alpha=1.0)
    assert len(calls) == 3000 and not api.recordings
