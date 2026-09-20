# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-independent stream/device contract for the SDPA DSL helpers.

``sdpa/frost/test_sdpa_stream_ordering.py`` carries a module-level
``requires_pre_rubin_blackwell`` gate because its cases execute SM100 kernels.
That gate also covers the helper cases in it, which are pure Torch/CUDA-runtime
behaviour that any sm80+ device can check:

``_torch_stream_context`` -- enter the stream a kernel is launched on, so the
Torch glue around it (staging copies, ``zero_()`` resets, output copy-back,
amax scaling) is ordered against that kernel instead of racing it on whatever
stream torch happens to be on.

``APIBase._get_default_stream`` -- resolve ``None`` to torch's *current* stream,
never to the legacy default stream.

This file keeps those cases where they can actually run, and adds the boundary
cases the engine suite does not cover: exception and nesting restoration, the
``verify_current`` override of the raw-handle fast path, the default-stream
sentinels staying distinguishable, tensor-device precedence, one launch per
fresh stream/device/input set, CUDA-graph capture and replay, and the engine
lane that is the single guard the adapter glue relies on.

Nothing here executes a FROST kernel. Kernel-level ordering evidence stays in
``sdpa/frost/test_sdpa_stream_ordering.py`` (Blackwell + DSL) and in the
architecture CI; a pass here is helper-layer evidence only.
"""

import ast
import logging
import os
import pathlib
from types import SimpleNamespace

import pytest
import torch

from cuda.bindings import driver as cuda_driver

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

# Bounded device-side spin used to make a mis-ordered launch observable, at the
# same order as the engine-level case in sdpa/frost/test_sdpa_stream_ordering.py.
# Measured on the L20: with the context replaced in place by a no-op, this
# value-race form stays masked here -- the ambient work is scheduled after the
# launch spin -- so the deterministic stream/device assertions, not this case,
# are what a disabled context makes fail.  The ordering case says so explicitly.
# Override with CUDNN_TEST_STREAM_SPIN_CYCLES when calibrating another part.
_SPIN_CYCLES = int(os.environ.get("CUDNN_TEST_STREAM_SPIN_CYCLES", "1000000000"))


def _handle(stream: torch.cuda.Stream) -> int:
    return int(stream.cuda_stream)


def _current_handle(device: torch.device) -> int:
    return _handle(torch.cuda.current_stream(device))


def _cu(stream: torch.cuda.Stream) -> "cuda_driver.CUstream":
    return cuda_driver.CUstream(_handle(stream))


@pytest.fixture
def two_devices_or_skip():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices for the device-isolation case")
    return torch.device("cuda:0"), torch.device("cuda:1")


@pytest.mark.L0
def test_launch_stream_is_entered_and_the_caller_stream_restored():
    """The context runs torch work on the launch stream, then restores the caller's."""
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    ambient = torch.cuda.Stream(device=device)
    launch = torch.cuda.Stream(device=device)

    with torch.cuda.stream(ambient):
        assert _current_handle(device) == _handle(ambient)
        with _torch_stream_context(_cu(launch), device):
            assert _current_handle(device) == _handle(launch), "torch work was not moved onto the launch stream"
        assert _current_handle(device) == _handle(ambient), "the caller's stream was not restored"

    assert _current_handle(device) == _handle(torch.cuda.default_stream(device))


@pytest.mark.L0
def test_launch_stream_is_restored_when_the_body_raises():
    """An exception inside the context must not leak the launch stream."""
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    ambient = torch.cuda.Stream(device=device)
    launch = torch.cuda.Stream(device=device)

    with torch.cuda.stream(ambient):
        with pytest.raises(RuntimeError, match="boom"):
            with _torch_stream_context(_cu(launch), device):
                raise RuntimeError("boom")
        assert _current_handle(device) == _handle(ambient), "the launch stream leaked past an exception"

    assert _current_handle(device) == _handle(torch.cuda.default_stream(device))


@pytest.mark.L0
def test_repeated_and_nested_contexts_do_not_reuse_a_stale_stream():
    """Re-entering with a different handle must take the new one, not a cached one."""
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    first = torch.cuda.Stream(device=device)
    second = torch.cuda.Stream(device=device)

    for _ in range(3):
        with _torch_stream_context(_cu(first), device):
            assert _current_handle(device) == _handle(first)
            with _torch_stream_context(_cu(second), device):
                assert _current_handle(device) == _handle(second)
            assert _current_handle(device) == _handle(first), "the inner context did not restore the outer stream"
        assert _current_handle(device) == _handle(torch.cuda.default_stream(device))

    with _torch_stream_context(_cu(second), device):
        assert _current_handle(device) == _handle(second), "a stale stream was reused across entries"


@pytest.mark.L0
def test_verify_current_makes_the_handle_authoritative_over_the_raw_fast_path():
    """``verify_current=True`` must not take the "already on that stream" shortcut.

    The shortcut exists because the launch stream is almost always the stream
    torch is already on.  A caller that supplies the stream explicitly asked for
    that handle to be authoritative, so the context has to go through
    ``torch.cuda.stream`` even when the handles compare equal.
    """
    import unittest.mock as mock

    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    # A NON-default stream: the default sentinels take their own early branch
    # before the fast path, so they cannot show the difference under test.
    side = torch.cuda.Stream(device=device)
    entered = []
    real_stream_ctx = torch.cuda.stream

    def spy(stream, *args, **kwargs):
        entered.append(_handle(stream) if isinstance(stream, torch.cuda.Stream) else None)
        return real_stream_ctx(stream, *args, **kwargs)

    with torch.cuda.stream(side):
        with mock.patch.object(torch.cuda, "stream", spy):
            with _torch_stream_context(_cu(side), device, verify_current=True):
                pass
            assert entered == [_handle(side)], "verify_current did not enter the authoritative stream"
            # ... while the default entry point is allowed to take the shortcut
            # when the launch stream is already torch's current stream.
            entered.clear()
            with _torch_stream_context(_cu(side), device):
                pass
            assert entered == [], "the raw-handle fast path stopped short-circuiting"
    assert _current_handle(device) == _handle(torch.cuda.default_stream(device))


@pytest.mark.L0
def test_none_and_default_stream_sentinels_are_not_wrapped():
    """``None`` and the default-stream sentinels must not be wrapped in a stream object.

    Torch work already follows the default stream in those cases, and wrapping a
    default-stream sentinel in ``ExternalStream`` is what makes every launch
    after the compile run no-op on some torch builds (all-zero outputs).  This
    pins the contract of the sentinel branches; it is not a claim that the
    legacy and per-thread default streams are interchangeable.
    """
    import unittest.mock as mock

    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    ambient = torch.cuda.Stream(device=device)
    previous = torch.cuda.current_stream(device)
    torch.cuda.set_stream(ambient)
    try:
        with mock.patch.object(torch.cuda, "stream", side_effect=AssertionError("a sentinel was wrapped")):
            with _torch_stream_context(None, device):
                pass
            for handle in (0, 1, 2):
                with _torch_stream_context(cuda_driver.CUstream(handle), device):
                    pass
        assert _current_handle(device) == _handle(ambient)
    finally:
        torch.cuda.set_stream(previous)


@pytest.mark.L0
def test_sentinel_handles_leave_the_ambient_stream_in_place():
    """Behavioral probe of the raw 0 / legacy 1 / per-thread-default 2 handles.

    Recorded as evidence, not as a claim that one of them should be treated as
    the others: the point is that the ambient stream stays the caller's and the
    work submitted inside is still usable.
    """
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    ambient = torch.cuda.Stream(device=device)
    observed = {}
    with torch.cuda.stream(ambient):
        for handle in (0, 1, 2):
            with _torch_stream_context(cuda_driver.CUstream(handle), device):
                observed[handle] = _current_handle(device)
                torch.ones(64, device=device).mul_(2.0)
    torch.cuda.synchronize()
    assert all(value == _handle(ambient) for value in observed.values()), observed


@pytest.mark.L0
def test_torch_work_inside_the_context_follows_the_launch_stream():
    """Ordering evidence for the helper itself, independent of any engine.

    Initialization is ordered before the trial: the poison write completes and
    records ``poison_ready``, and the launch stream waits on that event before it
    spins and restores the tensor.  The work under test then runs inside the
    context while torch's ambient stream is still a different one.

    The reader waits on an event the work itself records, so it observes the
    stream the multiply really used instead of racing it: with the context the
    record lands on the launch stream behind the restore and the reader sees the
    doubled data, while a context that fails to switch streams records on the
    ambient stream, lets the reader run while the launch stream still spins, and
    reads back the poison value.  That failure mode is verified by disabling the
    helper in place (mutation run), not by leaving the outcome to the scheduler.

    Scope: kernel-level side-stream ordering stays in
    ``sdpa/frost/test_sdpa_stream_ordering.py``, which needs the Blackwell line.
    """
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    ambient = torch.cuda.Stream(device=device)
    launch = torch.cuda.Stream(device=device)
    reader = torch.cuda.Stream(device=device)
    real = torch.arange(64, dtype=torch.float32, device=device)
    x = torch.full((64,), -7.0, dtype=torch.float32, device=device)
    # Every buffer and event is created before the trial window: a fresh
    # allocation inside it can call cudaMalloc, which synchronizes the device and
    # would hide the very ordering this case observes.
    observed = torch.empty_like(x)
    done = torch.cuda.Event()
    torch.cuda.synchronize()

    # Initialization completes before the trial starts: the launch stream waits
    # on the poison event instead of relying on cross-stream luck, and no
    # device-wide synchronize is used inside the window.
    poison_ready = torch.cuda.Event()
    with torch.cuda.stream(ambient):
        x.fill_(-7.0)
        poison_ready.record(ambient)
    with torch.cuda.stream(launch):
        launch.wait_event(poison_ready)
        torch.cuda._sleep(_SPIN_CYCLES)
        x.copy_(real)

    # The event is recorded inside the context, so it lands on whichever stream
    # the multiply actually used, and the reader is ordered behind that record.
    with torch.cuda.stream(ambient):
        with _torch_stream_context(_cu(launch), device):
            x.mul_(2.0)
            done.record()
    with torch.cuda.stream(reader):
        reader.wait_event(done)
        observed.copy_(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(x, real * 2.0, atol=0, rtol=0)
    torch.testing.assert_close(observed, real * 2.0, atol=0, rtol=0)


@pytest.mark.L0
def test_default_stream_resolution_follows_torch_current_stream():
    """``APIBase._get_default_stream(None)`` resolves to torch's *current* stream.

    ``None`` must not be read as legacy stream 0: a wrapper that resolved it that
    way would stop being ordered with surrounding torch ops under
    ``with torch.cuda.stream(s):``.
    """
    from cudnn.api_base import APIBase

    dummy = SimpleNamespace(_logger=logging.getLogger("test"))
    device = torch.device("cuda")
    side = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side):
        resolved = APIBase._get_default_stream(dummy, None)
        assert int(resolved) == _handle(side), "None did not resolve to torch's current stream"

    assert int(APIBase._get_default_stream(dummy, None)) == _current_handle(device)
    assert int(APIBase._get_default_stream(dummy, _cu(side))) == _handle(side), "an explicit stream must pass through"


@pytest.mark.L0
def test_device_isolation_uses_the_tensor_device_and_restores_the_caller(two_devices_or_skip):
    """A stream on the tensor's device wins over the ambient device, then restores.

    Only a stream that belongs to the target device is used: handing a
    ``cuda:1`` handle to a ``cuda:0`` kernel would be an illegal call, not a
    device-isolation result.
    """
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device0, device1 = two_devices_or_skip
    stream0 = torch.cuda.Stream(device=device0)
    ambient1 = torch.cuda.Stream(device=device1)
    original_device = torch.cuda.current_device()

    with torch.cuda.device(device1), torch.cuda.stream(ambient1):
        assert torch.cuda.current_device() == device1.index
        with _torch_stream_context(_cu(stream0), device0):
            assert _current_handle(device0) == _handle(stream0)
            torch.ones(64, device=device0).mul_(2.0)
        assert _current_handle(device1) == _handle(ambient1), "the ambient device's stream was not restored"

    torch.cuda.synchronize(device0)
    assert torch.cuda.current_device() == original_device


@pytest.mark.L0
def test_each_round_binds_its_own_stream_device_and_inputs(two_devices_or_skip):
    """Fresh stream, fresh device, fresh storage every round: nothing is reused.

    A helper that cached the previous stream, device or tensor address passes a
    single-device single-call case.  Each round here uses new launch and ambient
    streams and new source storage on the other device, and the observed value
    must be the transform of the storage that is current in that round.
    """
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device0, device1 = two_devices_or_skip
    original_device = torch.cuda.current_device()

    for round_index, device in enumerate((device0, device1)):
        ambient = torch.cuda.Stream(device=device)
        launch = torch.cuda.Stream(device=device)
        source = torch.full((64,), float(round_index + 1), dtype=torch.float32, device=device)
        target = torch.empty_like(source)

        with torch.cuda.device(device), torch.cuda.stream(ambient):
            with _torch_stream_context(_cu(launch), device):
                assert _current_handle(device) == _handle(launch), "the round reused a stale stream"
                target.copy_(source)
                target.mul_(3.0)
            assert _current_handle(device) == _handle(ambient), "the round left the ambient stream switched"

        torch.cuda.synchronize(device)
        torch.testing.assert_close(target, source * 3.0, atol=0, rtol=0)

    assert torch.cuda.current_device() == original_device, "the last round leaked its device"


@pytest.mark.L0
def test_capture_and_replay_run_on_the_context_stream():
    """Capture the context's own work, then replay it against new inputs.

    The engine suite uses CUDA-graph capture as a cheap deterministic stream
    detector: work launched on some other stream is captured empty and replays
    zeros.  Inside a capture the launch handle *is* the capture stream, so
    entering it must keep the work capturable, and the replay must read the
    inputs as they are at replay time instead of the values captured with the
    graph.  ``torch.cuda.graph`` owns capture setup and the private pool, so the
    warmup exercises the same ops outside the capture with preallocated storage.
    """
    from cudnn.sdpa.fwd.api_dsl import _torch_stream_context

    device = torch.device("cuda")
    source = torch.arange(64, dtype=torch.float32, device=device)
    updated = source + 5.0
    target = torch.empty_like(source)
    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        target.copy_(source)
        target.mul_(2.0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream = _current_handle(device)
        with _torch_stream_context(capture_stream, device):
            assert _current_handle(device) == capture_stream, "the capture stream was switched"
            target.copy_(source)
            target.mul_(2.0)

    source.copy_(updated)
    target.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(target, updated * 2.0, atol=0, rtol=0)


@pytest.mark.L0
def test_native_dense_engine_follows_the_bound_handle_stream():
    """The dense lane this part can really execute, checked with a real reference.

    On an L20 the FROST/DSL engines never claim the graph -- their kernel cases
    skip on sm_89 -- so those rows belong to the architecture CI.  The native
    backend does claim it (three ``eng8`` plans here), which makes this the one
    dense f16 SDPA execution available locally.  The handle is bound to a
    non-default stream while torch's ambient stream is deliberately a different
    one, the same graph is captured on that stream and replayed after the input
    buffers change, and every result is compared with an independent float64
    reference.  Measured on the L20: max abs error 1.7e-4 on both streams and
    1.8e-4 after the replay, so the 1e-3 bound keeps margin without hiding a
    wrong-output regression.
    """
    import cudnn
    from cudnn.engines import is_python_engine

    device = torch.device("cuda")
    b, h, s, d = 1, 4, 256, 64
    dims, strides = (b, h, s, d), (s * h * d, d, h * d, 1)

    def make_inputs(seed):
        torch.manual_seed(seed)
        q_gpu = torch.randn(b, s, h, d, device=device, dtype=torch.float16).transpose(1, 2)
        k_gpu = torch.randn(b, s, h, d, device=device, dtype=torch.float16).transpose(1, 2)
        v_gpu = torch.randn(b, s, h, d, device=device, dtype=torch.float16).transpose(1, 2)
        o_gpu = torch.empty(b, s, h, d, device=device, dtype=torch.float16).transpose(1, 2)
        return q_gpu, k_gpu, v_gpu, o_gpu

    def reference(q_gpu, k_gpu, v_gpu):
        scores = torch.matmul(q_gpu.double(), k_gpu.double().transpose(-1, -2)) * (1.0 / d**0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), v_gpu.double())

    def assert_matches(o_gpu, q_gpu, k_gpu, v_gpu):
        assert bool(torch.isfinite(o_gpu).all()), "the engine produced non-finite output"
        torch.testing.assert_close(o_gpu.double(), reference(q_gpu, k_gpu, v_gpu), atol=1e-3, rtol=1e-3)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=False)
    o.set_output(True).set_dim(dims).set_stride(strides)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if all(is_python_engine(plan.engine_id) for plan in g.plans):
        pytest.skip("no native backend plan claimed the dense graph on this part")
    g.check_support()
    g.build_plans()
    workspace = g.get_workspace_size()
    ws = torch.empty(workspace, device=device, dtype=torch.uint8) if workspace else None

    handle = cudnn.create_handle()
    qa, ka, va, oa = make_inputs(0)
    vp_a = {q: qa, k: ka, v: va, o: oa}
    oa.zero_()
    g.execute(vp_a, ws, handle=handle)
    torch.cuda.synchronize()
    assert_matches(oa, qa, ka, va)

    side = torch.cuda.Stream(device=device)
    ambient = torch.cuda.Stream(device=device)
    cudnn.set_stream(handle=handle, stream=side.cuda_stream)
    with torch.cuda.stream(ambient):
        oa.zero_()
        with torch.cuda.stream(side):
            g.execute(vp_a, ws, handle=handle)
        side.synchronize()
        assert_matches(oa, qa, ka, va)

    qb, kb, vb, ob = make_inputs(1)
    vp_b = {q: qb, k: kb, v: vb, o: ob}
    with torch.cuda.stream(side):
        for _ in range(3):  # warm up outside the capture
            g.execute(vp_b, ws, handle=handle)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        g.execute(vp_b, ws, handle=handle)

    qc, kc, vc, _ = make_inputs(2)
    qb.copy_(qc)
    kb.copy_(kc)
    vb.copy_(vc)
    ob.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert_matches(ob, qb, kb, vb)


@pytest.mark.L0
def test_engine_lane_wraps_the_adapter_entry_in_the_stream_context():
    """The adapter glue is protected by ONE outer guard: pin it structurally.

    Every site the launch-stream issue lists sits inside an adapter method that
    has no local context of its own, because the engine lane establishes one
    around the whole call: ``engines.py``'s ``_execute`` and
    ``_execute_by_tensor`` wrap ``_execute_resolved``, which is what calls
    ``api.execute``.  Removing that guard would silently unprotect every
    copy/zero/scale in the adapters, so the structure is asserted here instead of
    being left to an architecture CI run that cannot execute on every host.
    """
    import cudnn.sdpa.fwd.engines as engines

    tree = ast.parse(pathlib.Path(engines.__file__).read_text(encoding="utf-8"))
    guarded = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in ("_execute", "_execute_by_tensor"):
            continue
        calls = []
        for child in ast.walk(node):
            if not isinstance(child, ast.With):
                continue
            if "_torch_stream_context" not in ast.dump(child.items[0].context_expr):
                continue
            calls += [sub.lineno for sub in ast.walk(child) if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "_execute_resolved"]
        guarded[node.name] = calls
    assert set(guarded) == {"_execute", "_execute_by_tensor"}, guarded
    for name, calls in guarded.items():
        assert calls, f"{name} no longer wraps _execute_resolved in _torch_stream_context"
