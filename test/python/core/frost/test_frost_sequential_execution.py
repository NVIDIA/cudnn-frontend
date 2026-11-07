# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only composition checks use metadata addresses that are never read.

The separate CUDA case runs non-gather pointwise steps through the real buffer
protocols, workspace transport and changed-input capture replay.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from cudnn.frost import buffers, execution
from cudnn.frost.execution import BufferBinding, ExecutionStep, SequentialPlan
from cudnn.frost.workspace import Workspace

pytestmark = pytest.mark.L0


class MetadataBuffer:
    def __init__(self, pointer, shape=(2, 3), strides=(3, 1), dtype="float32", device=0):
        self.pointer, self.shape, self.strides, self.dtype, self.device = pointer, shape, strides, dtype, device

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return self.strides

    def __dlpack_device__(self):
        return (2, self.device)


@pytest.fixture
def metadata_probe(monkeypatch):
    monkeypatch.setattr(execution, "device_context", lambda device: nullcontext())
    # Intercept metadata observation only: no synthetic CUDA address is exported
    # or dereferenced. Real CUDA coverage below does not use this fixture.
    monkeypatch.setattr(
        buffers, "probe", lambda value: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.__dlpack_device__()[1])
    )


def _workspace(nbytes, pointer=0x100000):
    return Workspace.over(SimpleNamespace(workspace=pointer, workspace_bytes=nbytes, device=0), 0, "host sequence")


def _binding(key, role="input", **kwargs):
    return BufferBinding(key, kwargs.pop("shape", (2, 3)), kwargs.pop("strides", (3, 1)), kwargs.pop("dtype", "float32"), role, **kwargs)


def _sequence(events, *, decline=False):
    bindings = (_binding("x"), _binding("temporary", "temporary", strides=(7, 2)), _binding("y", "output"))

    def first(values, scratch, stream):
        assert tuple(values["temporary"].stride()) == (7, 2)
        assert tuple(values["temporary"].shape) == (2, 3)
        events.append(("prepare first", scratch.data_ptr(), scratch.numel(), stream))
        return lambda: events.append("enqueue first")

    def second(values, scratch, stream):
        events.append(("prepare second", scratch.data_ptr(), scratch.numel(), stream))
        if decline:
            raise ValueError("downstream preparation failed")
        return lambda: events.append("enqueue second")

    return SequentialPlan(
        bindings,
        (
            ExecutionStep("first", ("x",), ("temporary",), first, 128),
            ExecutionStep("second", ("temporary",), ("y",), second, 384),
        ),
        device=0,
    )


def test_frost_sequence_prepares_all_steps_and_reuses_only_local_scratch(metadata_probe):
    events = []
    plan = _sequence(events)
    assert plan.workspace_bytes == 512  # 128-byte tensor region + max(128, 384)
    assert plan.bound == ("x", "y")
    pack = {"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000), "unused": object()}
    plan(pack, workspace=_workspace(512), stream=17)
    assert events[0] == ("prepare first", 0x100080, 128, 17)
    assert events[1] == ("prepare second", 0x100080, 384, 17)
    assert events[2:] == ["enqueue first", "enqueue second"]
    events.clear()
    pack["x"] = MetadataBuffer(0x400000)
    plan(pack, workspace=_workspace(512, 0x500000), stream=19)
    assert events[0][1:] == (0x500080, 128, 19)


def test_frost_sequence_downstream_failure_has_no_enqueue(metadata_probe):
    events = []
    plan = _sequence(events, decline=True)
    with pytest.raises(ValueError, match="downstream"):
        plan({"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000)}, workspace=_workspace(512))
    assert len(events) == 2
    assert all(event[0].startswith("prepare") for event in events)


@pytest.mark.parametrize("bad", ["missing", "dtype", "shape", "stride", "device", "alias", "capacity", "workspace_alias"])
def test_frost_sequence_binding_failures_precede_every_prepare(metadata_probe, bad):
    events = []
    plan = _sequence(events)
    pack = {"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000)}
    workspace = _workspace(512)
    if bad == "missing":
        del pack["x"]
    elif bad in ("dtype", "shape", "stride", "device"):
        attribute, value = {"dtype": ("dtype", "float16"), "shape": ("shape", (3, 2)), "stride": ("strides", (4, 1)), "device": ("device", 1)}[bad]
        setattr(pack["x"], attribute, value)
    elif bad == "alias":
        pack["y"] = pack["x"]
    elif bad == "capacity":
        workspace = _workspace(511)
    else:
        workspace = _workspace(512, 0x200000)
    with pytest.raises(ValueError):
        plan(pack, workspace=workspace)
    assert events == []


@pytest.mark.parametrize("case", ["unknown", "before_producer", "read_only", "duplicate", "missing_producer"])
def test_frost_sequence_checks_declared_dataflow_at_build(case):
    bindings = [_binding("x"), _binding("t", "temporary"), _binding("y", "output")]
    steps = [ExecutionStep("first", ("x",), ("t",), lambda *_: lambda: None), ExecutionStep("second", ("t",), ("y",), lambda *_: lambda: None)]
    if case == "unknown":
        steps[0] = ExecutionStep("first", ("unknown",), ("t",), lambda *_: lambda: None)
    elif case == "before_producer":
        steps.reverse()
    elif case == "read_only":
        steps[0] = ExecutionStep("first", ("x",), ("x", "t"), lambda *_: lambda: None)
    elif case == "duplicate":
        bindings.append(bindings[0])
    else:
        bindings.append(_binding("unwritten", "output"))
    with pytest.raises(ValueError):
        SequentialPlan(bindings, steps, device=0)


def test_frost_sequence_explicit_inout_and_empty_temporaries(metadata_probe):
    events = []
    plan = SequentialPlan(
        (_binding("state", "inout"),), (ExecutionStep("update", ("state",), ("state",), lambda *_: lambda: events.append("update")),), device=0
    )
    plan({"state": MetadataBuffer(0x200000)})
    assert events == ["update"]
    empty = SequentialPlan(
        (
            _binding("x", shape=(0, 3)),
            _binding("t", "temporary", shape=(0, 3)),
            _binding("y", "output", shape=(0, 3)),
        ),
        (
            ExecutionStep("first", ("x",), ("t",), lambda *_: lambda: None),
            ExecutionStep("second", ("t",), ("y",), lambda *_: lambda: None),
        ),
        device=0,
    )
    assert empty.workspace_bytes == 0
    empty({"x": MetadataBuffer(0, shape=(0, 3)), "y": MetadataBuffer(0, shape=(0, 3))})


@pytest.mark.parametrize("kwargs", [dict(strides=(0, 1), role="output"), dict(strides=(-3, 1)), dict(shape=(1 << 62, 2)), dict(strides=(1 << 62, 1))])
def test_frost_sequence_rejects_unsafe_geometry(kwargs):
    with pytest.raises(ValueError):
        _binding("x", **kwargs)


def test_frost_sequence_nongather_cuda_capture_and_no_device_allocation():
    import torch
    from cudnn._torch_stream import stream_context

    if not torch.cuda.is_available():
        pytest.skip("CUDA pointwise composition requires a GPU")
    device = torch.cuda.current_device()
    x = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    y = torch.ones_like(x)
    scale = torch.tensor(3.0, device="cuda")
    backing = torch.empty((2, 8), device="cuda")
    out = backing[:, :6:2]
    stream = torch.cuda.Stream()
    bindings = (
        _binding("x"),
        _binding("y"),
        _binding("scale", shape=(), strides=()),
        _binding("t", "temporary"),
        _binding("out", "output", strides=tuple(out.stride())),
    )

    def prepare_add(values, scratch, stream):
        temporary = torch.utils.dlpack.from_dlpack(values["t"])

        def enqueue():
            with stream_context(stream, device):
                torch.add(values["x"], values["y"], out=temporary)

        return enqueue

    def prepare_mul(values, scratch, stream):
        temporary = torch.utils.dlpack.from_dlpack(values["t"])

        def enqueue():
            with stream_context(stream, device):
                torch.mul(temporary, values["scale"], out=values["out"])

        return enqueue

    plan = SequentialPlan(
        bindings,
        (
            ExecutionStep("add", ("x", "y"), ("t",), prepare_add),
            ExecutionStep("multiply", ("t", "scale"), ("out",), prepare_mul),
        ),
        device=device,
    )
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    pack = {"x": x, "y": y, "scale": scale, "out": out}
    stream.wait_stream(torch.cuda.current_stream())
    plan(pack, workspace=workspace, stream=stream.cuda_stream)
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    debug = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        for _ in range(3):
            plan(pack, workspace=workspace, stream=stream.cuda_stream)
    finally:
        torch.cuda.set_sync_debug_mode(debug)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured, stream=stream):
            plan(pack, workspace=workspace, stream=stream.cuda_stream)
        x.add_(7)
        y.mul_(2)
        out.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            captured.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out, (x + y) * scale, rtol=0, atol=0)
        previous = out.clone()
        with torch.cuda.stream(stream):
            captured.replay()
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(out.contiguous().view(torch.uint8), previous.contiguous().view(torch.uint8))
    finally:
        captured.reset()


def test_frost_sequence_restores_device_scope_after_prepare_failure(metadata_probe, monkeypatch):
    scopes, events = [], []

    @contextmanager
    def scoped_device(device):
        scopes.append(("enter", device))
        try:
            yield
        finally:
            scopes.append(("restore", device))

    monkeypatch.setattr(execution, "device_context", scoped_device)
    plan = _sequence(events, decline=True)
    with pytest.raises(ValueError, match="downstream"):
        plan({"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000)}, workspace=_workspace(512))
    assert scopes == [("enter", 0), ("restore", 0)]
    assert len(events) == 2


def test_frost_sequence_empty_values_have_no_alias_range(metadata_probe):
    plan = SequentialPlan(
        (_binding("x"), _binding("y", "output", shape=(0, 3))),
        (ExecutionStep("empty", ("x",), ("y",), lambda *_: lambda: None, 128),),
        device=0,
    )
    # A zero-element view can point into a nonempty input or workspace without
    # reading/writing their storage.
    for pointer in (0x200004, 0x100004):
        plan({"x": MetadataBuffer(0x200000), "y": MetadataBuffer(pointer, shape=(0, 3))}, workspace=_workspace(128))


def test_frost_sequence_cai_only_device_is_not_assumed_zero(metadata_probe, monkeypatch):
    runtime = pytest.importorskip("cuda.bindings.runtime")
    calls = []
    plan = SequentialPlan(
        (_binding("x"),),
        (ExecutionStep("read", ("x",), (), lambda *_: lambda: calls.append("enqueue")),),
        device=0,
    )
    value = SimpleNamespace(data_ptr=lambda: 0x200000, stride=lambda: (3, 1), shape=(2, 3), dtype="float32")
    monkeypatch.setattr(buffers, "probe", lambda _: (0x200000, (2, 3), (3, 1), "float32", 0))
    monkeypatch.setattr(runtime, "cudaPointerGetAttributes", lambda ptr: (runtime.cudaError_t.cudaSuccess, SimpleNamespace(device=1)))
    with pytest.raises(ValueError, match="compiled device"):
        plan({"x": value})
    assert calls == []
    monkeypatch.setattr(runtime, "cudaPointerGetAttributes", lambda ptr: (runtime.cudaError_t.cudaSuccess, SimpleNamespace(device=0)))
    plan({"x": value})
    assert calls == ["enqueue"]


def test_frost_sequence_preserves_positional_stream_and_workspace(metadata_probe):
    events = []
    plan = _sequence(events)
    plan({"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000)}, 17, _workspace(512))
    assert events[0][-1] == events[1][-1] == 17
    from cudnn.frost.execution import VariantPackPlan

    class LegacyExecutor:
        workspace_bytes = 128

        def __call__(self, pack, workspace=None, stream=None):
            return pack, workspace, stream

    adapted = VariantPackPlan(LegacyExecutor())
    assert adapted.workspace_bytes == 128
    assert adapted("pack", 19, "workspace") == ("pack", "workspace", 19)


@pytest.mark.parametrize("key", ["x", "y"])
def test_frost_sequence_alignment_failure_precedes_all_steps(metadata_probe, key):
    events = []
    plan = SequentialPlan(
        (_binding("x", alignment=16), _binding("y", "output", alignment=4)),
        (ExecutionStep("copy", ("x",), ("y",), lambda *_: lambda: events.append("enqueue")),),
        device=0,
    )
    pack = {"x": MetadataBuffer(0x200000), "y": MetadataBuffer(0x300000)}
    pack[key].pointer += 1
    with pytest.raises(ValueError, match="pointer alignment"):
        plan(pack)
    assert events == []


@pytest.mark.parametrize("alignment", [0, -1, 3, 1 << 63])
def test_frost_sequence_rejects_invalid_alignment(alignment):
    with pytest.raises(ValueError, match="alignment"):
        _binding("x", alignment=alignment)
