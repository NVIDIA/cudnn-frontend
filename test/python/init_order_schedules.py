# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedules and stand-ins for the cases that hold the order of a launch behind the initialization of an entry-owned
allocation (``cudnn.frost.init_order``), shared by the helper's own cases and by the suites of the direct-call entries.

The DEVICE schedules park a stream behind a delay kernel, let an entry allocate on it and launch from ANOTHER stream while
that work is still queued.  What every such case must do, or it shows nothing:

* WARM the entry first with a real launch: the first launch of a newly compiled kernel has been observed to wait until the
  other streams' work is done, which hides the overlap;
* take its side stream from :func:`side_stream`: ``torch.cuda.Stream()`` is one of a small pool of streams handed out in
  turn, so a "new" stream can be one the case already uses.  For the same reason a capture runs on a stream of the case's
  own (``torch.cuda.graph(graph, stream=...)``; the context's default is one stream shared by every capture of the process).
  A stream that is to be destroyed is made with the driver (:func:`recycled_handle`): torch's own are never destroyed;
* give the side stream's allocator pool the blocks the entry will allocate (:func:`pooled`), so no allocation synchronizes
  the schedule away;
* warm torch's own kernels the parked stream will run (:func:`warm_fills`): they are loaded at their first use, and that
  first launch waits like an entry's;
* query the marker :func:`park` returns behind its call: still pending = the call overlapped the parked work and did not
  wait on the host;
* run inside the ``settled_records`` fixture: the cases of every entry share the device's records.

The HOST stand-ins (:func:`stand_ins`, :func:`host_allocations`) replace what the ordering touches -- events, streams, the
driver's capture-mode exchange and capture status, the device context -- and keep ONE log of their calls; :func:`on_device`
gives an entry built for a stand-in device its operands.

Not a test module (no ``test_`` prefix): imported by name from the suites (``test/python`` is on their path)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost import capture, init_order

DELAY = 1_000_000_000  # cycles of the delay kernel: long enough that the case's calls overlap the parked work


def side_stream(dev, *others):
    """A stream of ``dev`` whose RAW HANDLE differs from the current stream's and from that of every stream in ``others``."""
    taken = {torch.cuda.current_stream(dev).cuda_stream, *(s.cuda_stream for s in others)}
    for _ in range(64):
        stream = torch.cuda.Stream(device=dev)
        if stream.cuda_stream not in taken:
            return stream
    raise AssertionError("torch handed out no stream with a handle of its own")


def park(stream, work=None, cycles=DELAY):
    """Park ``stream`` behind a delay kernel, run ``work()`` on it, and return an event recorded behind both: pending until
    the delay has run."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")
    with torch.cuda.stream(stream):
        torch.cuda._sleep(cycles)
        if work is not None:
            work()
        marker = torch.cuda.Event()
        marker.record(stream)
    return marker


def pooled(dev, make_blocks, stream=None):
    """``stream`` (a side stream when none is given) with the blocks ``make_blocks()`` allocates in its allocator pool, so
    an allocation of those sizes on it is a pool hit."""
    stream = side_stream(dev) if stream is None else stream
    with torch.cuda.stream(stream):
        blocks = make_blocks()
        del blocks
    torch.cuda.synchronize()
    return stream


@contextmanager
def recycled_handle(dev):
    """``(first, successor)``: a stream of the case's own, made with the driver so that it can be DESTROYED, and a function
    that destroys it -- the work it has queued still runs -- and returns the stream the driver makes NEXT, which has been
    observed to get the destroyed stream's raw handle (the case skips when it does not: the schedule shows nothing then)."""
    drv = pytest.importorskip("cuda.bindings.driver")
    raw = []

    def make():
        with torch.cuda.device(dev):
            err, handle = drv.cuStreamCreate(drv.CUstream_flags.CU_STREAM_NON_BLOCKING)
        assert int(err) == 0, f"cuStreamCreate failed: {err}"
        raw.append(handle)
        return torch.cuda.ExternalStream(int(handle), device=dev)

    def successor():
        gone = raw.pop()
        handle = int(gone)
        assert int(drv.cuStreamDestroy(gone)[0]) == 0  # returns at once; the stream's queued work is not cancelled
        stream = make()
        if stream.cuda_stream != handle:
            pytest.skip("the driver gave the new stream another raw handle: the schedule shows nothing")
        return stream

    try:
        yield make(), successor
    finally:
        torch.cuda.synchronize(dev)
        for handle in raw:
            drv.cuStreamDestroy(handle)


@contextmanager
def raw_capture(graph, stream):
    """A RAW capture of ``graph`` on ``stream`` around the block (``capture_begin`` does not synchronize the device, which
    ``torch.cuda.graph`` does when it is entered), ENDED also when the block raises: a capture left open makes the context
    unusable for every later case.  The block runs on the caller's current stream.  ``stream`` None: no capture."""
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        graph.capture_begin()
    try:
        yield
    finally:
        with torch.cuda.stream(stream):
            graph.capture_end()


def previous_owner(make, value):
    """A tensor allocated by ``make()``, filled with ``value`` and freed, on the current stream: the address of its block.
    Called on a parked stream right before an entry allocates there, the block goes to the entry while the fill is queued."""
    old = make()
    old.fill_(value)
    address = old.data_ptr()
    del old
    return address


@contextmanager
def deterministic_fill():
    """``torch.use_deterministic_algorithms(True)`` around an ALLOCATING call: torch then fills ``torch.empty`` itself, on
    the allocating stream -- an initialization that does not depend on which block the allocator hands out."""
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def warm_fills(dev, *dtypes):
    """One fill of each dtype under deterministic algorithms, synchronized: the first use of a torch fill kernel has been
    observed to wait until the other streams' work is done, which would hide the overlap of a schedule that queues a fill
    (:func:`deterministic_fill`, :func:`previous_owner`) on a parked stream."""
    with deterministic_fill():
        for dtype in dtypes:
            torch.empty(16, dtype=dtype, device=dev).fill_(1)
    torch.cuda.synchronize(dev)


def _settle(when):
    """Every device idle and its records walked once: none may be left (a record is dropped by the first walk behind its
    event).  The records are emptied in place whatever was found, so a failing case does not reach the next one."""
    for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        pending = init_order.pending_of(torch.device("cuda", index))
        torch.cuda.synchronize(index)
        try:
            if pending:
                init_order.order_behind(pending, None, ())
            assert not pending, f"{len(pending)} record(s) of cuda:{index} left {when} although the device is idle"
        finally:
            pending.clear()


@pytest.fixture
def settled_records():
    """Around an ordering case on a device: it starts and ends with an idle device and no record."""
    _settle("by an earlier case")
    yield
    _settle("by this case")


def stand_ins(monkeypatch):
    """Stand-ins for what the ordering touches, with ONE log of their calls (``calls``), on a registry of their own (the
    process's records are put back when the case ends).  ``current``: what ``torch.cuda.current_stream`` answers;
    ``capturing``: the raw handles that capture; ``Driver.fails``: the capture-mode exchange is refused.  A synchronization
    of a stream or of the device raises."""
    calls = []

    class Event:
        def __init__(self, complete=False, fails=False):
            self.complete, self.fails = complete, fails

        def record(self, stream):
            calls.append(("record", stream.cuda_stream))

        def query(self):
            calls.append(("query", self))
            if self.fails:
                raise RuntimeError("the event's query failed")
            return self.complete

        def synchronize(self):
            calls.append(("synchronize", self))
            self.complete = True

    class Stream:
        def __init__(self, handle):
            self.cuda_stream = handle

        def wait_event(self, event):
            calls.append(("wait_event", self.cuda_stream, event))

        def synchronize(self):
            raise AssertionError("a stream was synchronized")

        wait_stream = synchronize

    class Driver:
        class CUstreamCaptureMode:
            CU_STREAM_CAPTURE_MODE_RELAXED = "relaxed"

        mode, fails = "global", False

        @classmethod
        def cuThreadExchangeStreamCaptureMode(cls, mode):
            if cls.fails:
                return 1, cls.mode
            previous, cls.mode = cls.mode, mode
            calls.append(("capture mode", mode))
            return 0, previous

    state = SimpleNamespace(calls=calls, Event=Event, Stream=Stream, Driver=Driver, current=Stream(100), capturing=set())

    @contextmanager
    def device_context(device):
        calls.append(("enter", device))
        try:
            yield
        finally:
            calls.append(("leave", device))

    def no_synchronization(*args, **kwargs):
        raise AssertionError("the device was synchronized")

    monkeypatch.setattr(init_order, "_RECORDS", {})
    monkeypatch.setattr(init_order, "device_context", device_context)
    monkeypatch.setattr("cudnn._device._driver", lambda: Driver)
    monkeypatch.setattr(capture, "is_capturing", lambda handle: calls.append(("is capturing", handle)) or handle in state.capturing)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.current.cuda_stream in state.capturing)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: state.current)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", no_synchronization)
    return state


def host_allocations(monkeypatch, dev, calls):
    """``torch.empty`` / ``torch.zeros`` for ``dev`` answered with host memory and logged in ``calls`` as ``(name, shape)``:
    an entry built for ``dev`` allocates without a device."""

    def allocation(name, real):
        def allocate(*shape, device=None, **kw):
            if device != dev:
                return real(*shape, device=device, **kw)
            calls.append((name, tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape))
            return real(*shape, **kw)

        return allocate

    monkeypatch.setattr(torch, "empty", allocation("empty", torch.empty))
    monkeypatch.setattr(torch, "zeros", allocation("zeros", torch.zeros))


def on_device(dev, *tensors):
    """``tensors`` (host memory) as tensors that report ``dev`` as their device: the operands of an entry built for ``dev``,
    so a call runs every check it makes in front of its launch without a device."""

    class OnDevice(torch.Tensor):
        device = property(lambda self: dev)

    return tuple(t.as_subclass(OnDevice) for t in tensors)


def spans_of(tensors):
    """The whole byte range of each tensor, written out here: what a record of their allocation holds."""
    return tuple((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors)
