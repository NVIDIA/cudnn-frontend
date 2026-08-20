# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ensure_current_context`` binds the context a plan's work runs in to the
calling thread. Two halves, and only the first used to hold: a cold thread must
end up bound at all, and a thread bound to ANOTHER GPU's context must be moved
off it. The second is not cosmetic -- a default stream under a foreign context
runs the work on that context's GPU, where the pointers are invalid."""

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
    """No default-stream handle can name a GPU -- each resolves against the
    calling thread's current context -- so ``device`` decides."""
    a, b = _two_devices()
    ctx_a, ctx_b = _bind_primary(drv, a), _bind_primary(drv, b)
    assert ctx_a != ctx_b
    stream = {"null": 0, "legacy": int(drv.CU_STREAM_LEGACY), "per_thread": int(drv.CU_STREAM_PER_THREAD)}[handle]
    assert int(drv.cuStreamGetCtx(stream)[1]) == ctx_b  # follows the thread, names nothing

    drv.cuCtxSetCurrent(drv.CUcontext(ctx_a))
    ensure_current_context(stream, b)
    assert _current(drv) == ctx_b, "left the thread on another GPU's context"


def test_follows_the_streams_context(drv):
    """A real stream carries its context, and it wins over ``device``."""
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
    """Steady state is a no-op: no rebind, no primary-context churn."""
    ctx = _bind_primary(drv, 0)
    ensure_current_context(0, 0)
    assert _current(drv) == ctx
    ensure_current_context(0, 0)
    assert _current(drv) == ctx


def test_an_unnamed_device_does_not_override_a_bound_context(drv):
    """No device named means nothing to correct: a bound context is
    authoritative (``ambient_device``'s first rung)."""
    a, b = _two_devices()
    _bind_primary(drv, a)
    ctx_b = _bind_primary(drv, b)
    torch = pytest.importorskip("torch")
    torch.cuda.set_device(a)  # runtime slot -> a, while the driver context is b's
    drv.cuCtxSetCurrent(drv.CUcontext(ctx_b))
    ensure_current_context(0, None)
    assert _current(drv) == ctx_b, "overrode a bound context on the runtime's word"


def test_the_primary_context_is_retained_once_per_device(drv):
    """A retain per execute would grow the usage count without bound."""
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


def test_a_backend_graph_runs_on_a_cold_thread(drv):
    """The C++ execute funnel binds one too, for the same reason.

    cuDNN's runtime-compiled engines launch through the driver and read the
    calling thread's stack; the precompiled ones go through the runtime and are
    unaffected, so the fused ``matmul + relu + relu`` is what exercises this."""
    torch = pytest.importorskip("torch")
    import cudnn

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 matmul needs sm80+")
    a = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()

    handle = cudnn.create_handle()
    try:
        g = cudnn.pygraph(handle=handle, io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
        ta, tb = g.tensor_like(a), g.tensor_like(b)
        y = g.relu(input=g.relu(input=g.matmul(A=ta, B=tb)))
        y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        try:
            g.build([cudnn.heur_mode.A])
        except cudnn.cudnnGraphNotSupportedError as exc:
            pytest.skip(f"no engine for the fused graph on this arch/backend: {exc}")
        ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
        pack = {ta: a, tb: b, y: out}
        g.execute(pack, ws, handle=handle)  # warm, on this thread
        torch.cuda.synchronize()

        def body(seen):
            # No torch op before execute -- even out.zero_() binds the context.
            seen["before"] = _current(drv)
            try:
                g.execute(pack, ws, handle=handle)
            except BaseException as exc:  # noqa: BLE001
                seen["exc"] = exc
            seen["after"] = _current(drv)

        seen = _on_a_cold_thread(body)
        assert seen["before"] == 0, "the worker was already bound, so this no longer covers the cold path"
        assert seen["after"] != 0
        torch.cuda.synchronize()
        expected = torch.relu(torch.relu(a.float() @ b.float()))
        assert (out.float() - expected).abs().max().item() < 1.0
    finally:
        cudnn.destroy_handle(handle)
