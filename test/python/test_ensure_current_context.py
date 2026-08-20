# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ensure_current_context`` binds the context a python plan's work runs in to
the calling thread. Two halves, and only the first used to hold: a cold thread
must end up bound at all, and a thread bound to ANOTHER GPU's context must be
moved off it -- a context is not interchangeable just because it exists.

The second half is not cosmetic. A real stream carries its context, so a
cross-context launch is rejected outright (``CUDA_ERROR_INVALID_HANDLE``); the
legacy default stream (handle 0) carries none, and under a foreign context it
runs the work on THAT context's GPU, where the pointers are invalid -- an async
fault at some later sync rather than an error at the launch."""

import threading

import pytest

from cudnn._device import _primary_context, device_count, ensure_current_context, is_available

pytestmark = pytest.mark.L0


@pytest.fixture
def drv():
    d = pytest.importorskip("cuda.bindings.driver")
    if not is_available():
        pytest.skip("no CUDA device")
    entry = d.cuCtxGetCurrent()[1]
    yield d
    d.cuCtxSetCurrent(entry)  # these tests move the thread's context on purpose


def _current(drv):
    err, ctx = drv.cuCtxGetCurrent()
    return int(ctx) if int(err) == 0 else 0


def _bind_primary(drv, ordinal):
    """Retain ``ordinal``'s primary context and make it current on this thread."""
    dev = drv.cuDeviceGet(ordinal)[1]
    ctx = drv.cuDevicePrimaryCtxRetain(dev)[1]
    drv.cuCtxSetCurrent(ctx)
    return int(ctx)


def _two_devices():
    if device_count() < 2:
        pytest.skip("needs two GPUs to tell 'a context' from 'the right context'")
    return 0, 1


def _on_a_cold_thread(body):
    """Run ``body`` on a thread that has never bound a context. Returns its dict."""
    seen = {}
    worker = threading.Thread(target=body, args=(seen,))
    worker.start()
    worker.join()
    if "exc" in seen:
        raise seen["exc"]
    return seen


def test_binds_a_cold_thread(drv):
    _bind_primary(drv, 0)  # the process has a context; the worker below does not

    def body(seen):
        try:
            seen["before"] = _current(drv)
            ensure_current_context(0, 0)
            seen["after"] = _current(drv)
        except BaseException as exc:  # noqa: BLE001
            seen["exc"] = exc

    seen = _on_a_cold_thread(body)
    assert seen["before"] == 0, "the worker was already bound, so this no longer covers the cold path"
    assert seen["after"] != 0


@pytest.mark.parametrize("handle", ["null", "legacy", "per_thread"])
def test_replaces_a_context_on_another_device(drv, handle):
    """All three default-stream handles resolve against the CALLING thread's
    current context, so none of them can name the GPU: the device decides, and a
    context on the wrong one must be replaced rather than accepted."""
    a, b = _two_devices()
    ctx_a, ctx_b = _bind_primary(drv, a), _bind_primary(drv, b)
    assert ctx_a != ctx_b
    stream = {"null": 0, "legacy": int(drv.CU_STREAM_LEGACY), "per_thread": int(drv.CU_STREAM_PER_THREAD)}[handle]
    assert int(drv.cuStreamGetCtx(stream)[1]) == ctx_b  # follows the thread, names nothing

    drv.cuCtxSetCurrent(drv.CUcontext(ctx_a))
    ensure_current_context(stream, b)
    assert _current(drv) == ctx_b, "left the thread on another GPU's context"


def test_follows_the_streams_context(drv):
    """A real stream carries the context the work runs in; it wins over whatever
    the thread happens to have current."""
    a, b = _two_devices()
    ctx_a, ctx_b = _bind_primary(drv, a), _bind_primary(drv, b)
    stream = drv.cuStreamCreate(0)[1]  # created under ctx_b, so it belongs to it
    try:
        drv.cuCtxSetCurrent(drv.CUcontext(ctx_a))
        ensure_current_context(int(stream), a)  # device says a, the stream says b
        assert _current(drv) == ctx_b, "the stream's context did not win"
    finally:
        drv.cuCtxSetCurrent(drv.CUcontext(ctx_b))
        drv.cuStreamDestroy(stream)


def test_leaves_an_already_correct_context_alone(drv):
    """The steady state is a no-op: no rebind, no primary-context churn."""
    ctx = _bind_primary(drv, 0)
    ensure_current_context(0, 0)
    assert _current(drv) == ctx
    ensure_current_context(0, 0)
    assert _current(drv) == ctx


def test_an_unnamed_device_does_not_override_a_bound_context(drv):
    """With no device named there is nothing to correct: a bound driver context
    is authoritative (frost.device.ambient_device's first rung), so it must
    survive even when the runtime's thread-local slot names another GPU."""
    a, b = _two_devices()
    _bind_primary(drv, a)
    ctx_b = _bind_primary(drv, b)
    torch = pytest.importorskip("torch")
    torch.cuda.set_device(a)  # runtime slot -> a, while the driver context is b's
    drv.cuCtxSetCurrent(drv.CUcontext(ctx_b))
    ensure_current_context(0, None)
    assert _current(drv) == ctx_b, "overrode a bound context on the runtime's word"


def test_the_primary_context_is_retained_once_per_device(drv):
    """An unbalanced retain per execute would grow the usage count without bound."""
    a, b = _two_devices()
    for _ in range(20):  # alternating default-stream execution over two GPUs
        ensure_current_context(0, a)
        ensure_current_context(0, b)
    for ordinal in (a, b):
        dev = drv.cuDeviceGet(ordinal)[1]
        # cuDevicePrimaryCtxGetState reports active/flags, not the count, so assert
        # the cache instead: one retained handle per ordinal, reused.
        assert _primary_context(ordinal) is _primary_context(ordinal)
        assert int(drv.cuDevicePrimaryCtxGetState(dev)[2]) == 1  # still active, not churned
