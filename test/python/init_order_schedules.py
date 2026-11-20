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
  A stream that is to be destroyed is made with the driver (:func:`recycled_handle`): torch's own are never destroyed --
  and the schedule that destroys it runs in a process of its own (:func:`own_process`): torch keeps what a stream allocated
  under that stream's raw handle;
* give the side stream's allocator pool the blocks the entry will allocate (:func:`pooled`), so no allocation synchronizes
  the schedule away;
* warm torch's own kernels the parked stream will run (:func:`warm_fills`): they are loaded at their first use, and that
  first launch waits like an entry's;
* query the marker :func:`park` returns behind its call: still pending = the call overlapped the parked work and did not
  wait on the host;
* run inside the ``settled_records`` fixture: the cases of every entry share the device's records (a schedule in a
  process of its own is settled there).

The HOST stand-ins (:func:`stand_ins`, :func:`host_allocations`) replace what the ordering touches -- events, streams, the
driver's capture-mode exchange and capture status, the device context -- and keep ONE log of their calls; :func:`on_device`
gives an entry built for a stand-in device its operands.

Not a test module (no ``test_`` prefix): imported by name from the suites (``test/python`` is on their path)."""

import importlib
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost import capture, init_order

DELAY = 1_000_000_000  # cycles of the delay kernel: long enough that the case's calls overlap the parked work
OUTCOME = "SCHEDULE OUTCOME:"  # the last line a schedule's own process prints, in front of ``passed`` or ``skipped: <reason>``
_OWN_PROCESS = False  # True in the process :func:`own_process` starts for a schedule
REHEARSALS = 4  # pairs :func:`recycled_handle` tries, at most, for one whose next stream overlaps the destroyed stream's work


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


def own_process(schedule, *args, timeout=300):
    """Run ``schedule(*args)`` -- a function at the top of a test module, ``args`` plain values -- in a NEW interpreter and
    bring its outcome back: return when it passed, skip with its reason when it skipped, fail with everything the process
    printed otherwise (an assertion, the deadline, a signal).  For a schedule that DESTROYS a stream torch allocated under
    (:func:`recycled_handle`).  torch keeps the memory a stream allocated under that stream's RAW HANDLE and synchronizes
    the handle when it returns the memory to the driver (``torch.cuda.empty_cache()``, which the suite's memory gate calls
    behind every case): once the last stream of the handle is destroyed that call reads a handle no stream has -- an error
    at the teardown of every later case, or the end of the process.  Keeping the last stream alive covers the schedule
    whose next stream gets the handle; it cannot cover the one whose next stream gets ANOTHER handle: the buffers
    allocated under the destroyed handle are alive then, and nothing in the process can give that handle a stream again.
    So the whole schedule runs where no later case runs: the new process ends with the schedule, without the interpreter's
    shutdown, and the process of the suite never holds memory under that handle.

    The new process has the caller's ``sys.path`` (pytest's changes are not inherited) and must import the caller's
    ``cudnn``; the device's records are settled there behind a schedule that passed.  ``timeout`` (seconds) is the budget
    of the schedule's imports, compiles and launches: the new process dumps its stacks and ends itself shortly before it.
    A trip on a launch is a hang of that launch -- audit its waits; never a larger number."""
    import cudnn

    code = (
        "import faulthandler, json, sys\n"
        f"faulthandler.dump_traceback_later({0.9 * timeout!r}, exit=True)\n"
        "path, package, module, function, args = json.loads(sys.argv[1])\n"
        "sys.path[:] = path\n"
        "import init_order_schedules\n"
        "init_order_schedules._run_schedule(package, module, function, args)\n"
    )
    spec = json.dumps([sys.path, cudnn.__file__, schedule.__module__, schedule.__name__, args])

    def failed(how, out, err):
        out, err = (s.decode("utf-8", errors="replace") if isinstance(s, bytes) else (s or "") for s in (out, err))  # a timeout carries bytes
        report = f"the schedule {schedule.__name__}{args!r} {how} in its own process\n--- its output ---\n{out}\n--- its errors and stacks ---\n{err}"
        pytest.fail(report, pytrace=False)

    try:
        ran = subprocess.run([sys.executable, "-u", "-c", code, spec], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as late:
        failed(f"did not end within {timeout} s", late.stdout, late.stderr)
    last = ran.stdout.rstrip().rsplit("\n", 1)[-1]
    if ran.returncode == 0 and last == f"{OUTCOME} passed":
        return
    if ran.returncode == 0 and last.startswith(f"{OUTCOME} skipped: "):
        pytest.skip(last[len(f"{OUTCOME} skipped: ") :])
    failed(f"ended with exit code {ran.returncode}", ran.stdout, ran.stderr)


def _run_schedule(package, module, function, args):
    """The process :func:`own_process` starts: the schedule, the device's records settled behind it, its outcome as the last
    line of the output, and the end of the process WITHOUT the interpreter's shutdown (nothing of torch runs behind a
    destroyed stream handle).  Exit code 0 for a schedule that passed or skipped, 1 with the traceback for any other end."""
    global _OWN_PROCESS
    code = 1
    try:
        import cudnn

        assert Path(cudnn.__file__).resolve() == Path(package).resolve(), f"the schedule's process imported {cudnn.__file__}, the suite {package}"
        _OWN_PROCESS = True
        getattr(importlib.import_module(module), function)(*args)
        _settle("by the schedule")
        print(f"{OUTCOME} passed")
        code = 0
    except pytest.skip.Exception as skipped:
        print(f"{OUTCOME} skipped: {skipped.msg}")
        code = 0
    except BaseException:  # noqa: BLE001  (reported: the traceback is what the case fails with)
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


@contextmanager
def recycled_handle(dev):
    """``(first, successor)``: a stream of the schedule's own, made with the driver so that it can be DESTROYED, and a
    function that destroys it -- the work it has queued still runs -- and returns the stream the driver makes NEXT.  The
    schedule shows something only when that stream got the destroyed stream's raw handle AND its work overlaps what the
    destroyed stream still has queued.  Both have been observed; so has a FIRST such pair of a process whose next stream
    ran its work behind the destroyed stream's (the driver then orders a launch on it by itself), with the pairs behind it
    overlapping.  So pairs are rehearsed here, with a delay and a fill of torch's own, until one overlaps; ``successor()``
    reads both conditions for the schedule's pair and skips when one does not hold.

    ONLY IN A PROCESS OF ITS OWN (:func:`own_process` says why); anywhere else it is refused before a stream is made.  No
    other stream is destroyed here: the last stream of a handle stays alive for as long as torch can hold memory under the
    handle -- until the process ends."""
    if not _OWN_PROCESS:
        raise RuntimeError("recycled_handle destroys a stream torch allocated under: run the schedule in a process of its own (own_process)")
    drv = pytest.importorskip("cuda.bindings.driver")
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")
    made = []  # (the driver's stream, torch's view of it), the newest last
    probe = torch.empty(16, dtype=torch.uint8, device=dev).fill_(0)  # on the caller's stream: the fill is warm for the pairs

    def make():
        with torch.cuda.device(dev):
            err, handle = drv.cuStreamCreate(drv.CUstream_flags.CU_STREAM_NON_BLOCKING)
        assert int(err) == 0, f"cuStreamCreate failed: {err}"
        made.append((handle, torch.cuda.ExternalStream(int(handle), device=dev)))
        return made[-1][1]

    def replace():
        """The newest stream destroyed and the next one made: ``(stream, overlaps)``.  ``stream`` is None when the driver gave
        it another raw handle; ``overlaps``: a fill on it completed while the destroyed stream's work was still pending."""
        gone, old = made.pop()
        last = torch.cuda.Event()
        last.record(old)  # behind everything the stream has queued
        assert int(drv.cuStreamDestroy(gone)[0]) == 0  # returns at once; the stream's queued work is not cancelled
        stream = make()
        if stream.cuda_stream != old.cuda_stream:
            return None, False
        with torch.cuda.stream(stream):
            probe.fill_(1)
            mine = torch.cuda.Event()
            mine.record(stream)
        while not mine.query() and not last.query():  # ends with the destroyed stream's work at the latest
            pass
        return stream, not last.query()

    def successor():
        stream, overlaps = replace()
        if stream is None:
            pytest.skip("the driver gave the new stream another raw handle: the schedule shows nothing")
        if not overlaps:
            pytest.skip("the driver ran the new stream's work behind the destroyed stream's: the schedule shows nothing")
        return stream

    for _ in range(REHEARSALS):
        with torch.cuda.stream(make()):
            torch.cuda._sleep(DELAY // 8)
        if replace()[1]:
            break
    torch.cuda.synchronize(dev)
    try:
        yield make(), successor
    finally:
        torch.cuda.synchronize(dev)  # the queued work is done; the streams in ``made`` are left to the end of the process


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
