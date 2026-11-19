# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The order of a launch behind the initialization of an entry-owned allocation (``cudnn.frost.init_order``), host-only, on
stand-ins for the events, the streams, the driver and the device context (``init_order_schedules.stand_ins``): what a call
does for one record by its state; which records a launch is ordered behind; what a record of an allocation holds; the
records of a device as ONE object; host threads; the relaxed capture check of ``cudnn.frost.capture``.  The schedules on a
device are in the suites of the entries that use the helper."""

import copy
import pickle
import subprocess
import sys
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

from cudnn.frost import capture, init_order  # noqa: E402
from init_order_schedules import spans_of, stand_ins  # noqa: E402

pytestmark = pytest.mark.L0

DEV = torch.device("cuda", 5)  # a name for the stand-ins: nothing here touches a device
RELAXED, RESTORED = ("capture mode", "relaxed"), ("capture mode", "global")
INIT, OTHER = 0x7F0000001000, 0x7F0000002000  # raw stream handles: the allocating stream's and another's


def _put(pending, event, handle=INIT, spans=((1000, 2000),)):
    serial = next(init_order._SERIAL)
    pending[serial] = (event, handle, tuple(spans))
    return serial


def _walk(s, pending, stream, written):
    """One ``order_behind``: what it did to events and streams, inside ONE relaxed capture check under the device."""
    s.calls.clear()
    init_order.order_behind(pending, stream, written)
    assert s.calls[:2] == [("enter", DEV), RELAXED] and s.calls[-2:] == [RESTORED, ("leave", DEV)] and s.Driver.mode == "global"
    assert s.calls.count(RELAXED) == 1, "one exchange per call"
    return s.calls[2:-2]


def test_one_record_is_decided_by_its_state(monkeypatch):
    """Complete: dropped by ANY call, hit or not.  Pending and not hit: one query, nothing enqueued, kept -- also under a
    capture (no host wait for bytes the launch does not write).  Pending and hit: the launch stream waits for the event and
    the record is kept for the next stream -- on ANY raw handle, the allocating stream's too (a new stream can have been
    given the handle of a destroyed one, and the per-thread handle is another stream in every thread; on the allocating
    stream itself the wait is free).  Hit under a capture: the host waits for this ONE event without querying it, the record
    is dropped and NO stream wait is enqueued (a capturing stream cannot wait for an event recorded outside the capture) --
    on the allocating stream's handle like on any other.  The capture status of the legacy default stream is never read (it
    cannot capture, and the driver answers for it only on a host thread with a current context: a thread whose first CUDA
    call is this launch has none); its launch gets the stream wait like any other.  No branch synchronizes a stream or the
    device (the stand-ins raise).  A failed read restores the capture check; a failed exchange reads and drops nothing, and
    stops only a launch that writes a recorded allocation."""
    s = stand_ins(monkeypatch)
    pending = init_order.pending_of(DEV)
    inside, outside = ((1500, 1600),), ((5000, 5100),)

    def decide(complete, written, handle, captured=False, init=INIT):
        pending.clear()
        s.capturing = {handle} if captured else set()
        event = s.Event(complete)
        serial = _put(pending, event, init)
        got = _walk(s, pending, s.Stream(int(str(handle))), written)  # an equal handle, another int object
        return [c if c[0] == "is capturing" else c[:-1] if c[0] == "wait_event" else c[0] for c in got], serial in pending

    status = lambda handle: ("is capturing", handle)  # noqa: E731
    assert decide(False, outside, OTHER) == (["query"], True)
    assert decide(False, outside, OTHER, captured=True) == (["query"], True), "not hit under a capture: no host wait"
    assert decide(True, outside, OTHER) == (["query"], False), "complete: dropped by a call that does not write it"
    assert decide(True, outside, OTHER, captured=True) == (["query"], False)
    assert decide(False, inside, OTHER) == ([status(OTHER), "query", ("wait_event", OTHER)], True)
    assert decide(False, inside, INIT) == ([status(INIT), "query", ("wait_event", INIT)], True), "the allocating stream's handle: waited for too"
    assert decide(False, inside, 2, init=2) == ([status(2), "query", ("wait_event", 2)], True), "the per-thread stream"
    assert decide(True, inside, OTHER) == ([status(OTHER), "query"], False)
    assert decide(True, inside, INIT) == ([status(INIT), "query"], False), "complete: no wait on any handle"
    assert decide(False, inside, OTHER, captured=True) == ([status(OTHER), "synchronize"], False)
    assert decide(False, inside, INIT, captured=True) == ([status(INIT), "synchronize"], False), "captured on the allocating stream: no stream wait"
    for legacy in (0, 1):
        assert decide(False, inside, legacy) == (["query", ("wait_event", legacy)], True), "the legacy default stream: no status read"

    pending.clear()
    s.calls.clear()
    init_order.order_behind(pending, s.Stream(OTHER), inside)
    assert not s.calls, "without a record nothing is entered, exchanged or read"

    failing = _put(pending, s.Event(False, fails=True))
    with pytest.raises(RuntimeError, match="the event's query failed"):
        init_order.order_behind(pending, s.Stream(OTHER), inside)
    assert s.Driver.mode == "global" and failing in pending, "the capture check is restored behind a failed read"

    pending.clear()
    kept = _put(pending, s.Event(True))
    s.Driver.fails = True
    s.calls.clear()
    with pytest.raises(RuntimeError, match="cuThreadExchangeStreamCaptureMode failed"):
        init_order.order_behind(pending, s.Stream(OTHER), inside)
    assert s.calls == [("enter", DEV), ("leave", DEV)] and kept in pending, "nothing is read or dropped when the check cannot be relaxed"
    s.calls.clear()
    init_order.order_behind(pending, s.Stream(OTHER), outside)  # writes no recorded byte: it proceeds
    assert s.calls == [("enter", DEV), ("leave", DEV)] and kept in pending


def test_which_records_a_launch_is_ordered_behind(monkeypatch):
    """By ADDRESS, half-open: every range the launch writes against every range of a record (not range by range in
    order); ranges that only touch are not hit; an empty written range hits nothing; the ranges may come as a one-shot
    iterable.  Every record that is hit is waited for.  The events of ONE stream complete in the order they were
    recorded: behind a stream's first pending record its later ones are pending without a query (a hit one still gets its
    wait), so a walk queries once per record it drops and once per allocating stream that holds a pending record.  A
    record of another device is neither read nor dropped."""
    s = stand_ins(monkeypatch)
    pending, launch = init_order.pending_of(DEV), s.Stream(OTHER)

    def waited(written, *records):
        """The indices of ``records`` (each: its spans) the launch waits for; every record pending on INIT."""
        pending.clear()
        events = [s.Event(False) for _ in records]
        for event, spans in zip(events, records):
            _put(pending, event, INIT, spans)
        got = _walk(s, pending, launch, written)
        assert len(pending) == len(records) and [c[0] for c in got].count("query") == 1, "one query for the one allocating stream"
        return [events.index(c[2]) for c in got if c[0] == "wait_event"]

    two = ((1000, 2000), (3000, 4000))  # one allocation of two tensors
    assert waited(((1500, 1600), (9000, 9100)), two) == [0]
    assert waited(((9000, 9100), (3999, 4001)), two) == [0], "the second written range over the second tensor"
    assert waited(((3500, 3600), (9000, 9100)), two) == [0], "the FIRST written range over the SECOND tensor"
    assert waited(((9000, 9100), (1999, 2000)), two) == [0], "the second written range over the first tensor"
    assert waited(((900, 1000), (2000, 3000), (4000, 4100)), two) == [], "ranges that only touch"
    assert waited(((1500, 1500), (3500, 3400)), two) == [], "an empty written range hits nothing"
    assert waited(((0, 10**9),), two, ((5000, 6000),), ((7000, 7001),)) == [0, 1, 2], "every record that is hit"
    assert waited(((7000, 7001), (1000, 1001)), ((1000, 2000),), ((7000, 8000),)) == [0, 1]
    assert waited((r for r in ((7000, 7001), (1000, 1001))), ((1000, 2000),), ((7000, 8000),)) == [0, 1], "a one-shot iterable"

    # the order of one stream: complete records in front are dropped one query each, the first pending one ends the queries
    pending.clear()
    done = [s.Event(True), s.Event(True), s.Event(False), s.Event(True), s.Event(False)]
    serials = [_put(pending, e, INIT, ((1000 * i, 1000 * i + 10),)) for i, e in enumerate(done)]
    elsewhere = s.Event(True)
    other_stream = _put(pending, elsewhere, 300, ((90000, 90010),))
    got = _walk(s, pending, launch, ((4000, 4001),))
    assert got == [("query", done[0]), ("query", done[1]), ("query", done[2]), ("is capturing", OTHER), ("wait_event", OTHER, done[4]), ("query", elsewhere)]
    assert sorted(pending) == serials[2:], "a later record of a pending stream is kept unread; another stream's is read"
    assert other_stream not in pending

    # another device's record is another object's: neither read nor dropped
    far = init_order.pending_of(torch.device("cuda", 6))
    theirs = _put(far, s.Event(True))
    _walk(s, pending, launch, ((0, 10**9),))
    assert theirs in far and len(far) == 1


def test_record_holds_the_event_the_allocating_stream_and_every_range(monkeypatch):
    """``record``: under the device, the capture status of ITS current stream, then an event recorded on that stream; the
    record holds the stream's raw handle and the byte range of every tensor (an empty one is skipped).  Nothing is recorded
    for a device without streams, for an allocation without a byte, or inside a capture.  A record keeps a key of its own:
    one made behind a drop does not take the place of a pending one.  Allocations that no launch follows are swept of
    complete records once the device holds more than the mark."""
    s = stand_ins(monkeypatch)
    pending = init_order.pending_of(DEV)
    a, b, empty = torch.empty(3, 5, dtype=torch.int16), torch.zeros(7, dtype=torch.uint8), torch.empty(0, 4)
    s.current = s.Stream(321)
    init_order.record(pending, (t for t in (a, empty, b)))
    assert s.calls == [("enter", DEV), ("record", 321), ("leave", DEV)]
    ((event, handle, spans),) = pending.values()
    assert isinstance(event, s.Event) and handle == 321 and spans == spans_of((a, b)) and spans[0][1] - spans[0][0] == 30
    s.calls.clear()
    init_order.record(pending, (empty,))
    init_order.record(init_order.pending_of(torch.device("cpu")), (a, b))
    assert not s.calls and len(pending) == 1 and not init_order.pending_of(torch.device("cpu"))
    s.capturing = {321}
    init_order.record(pending, (a,))
    assert s.calls == [("enter", DEV), ("leave", DEV)] and len(pending) == 1, "inside a capture nothing is recorded"

    s.capturing = set()
    pending.clear()
    init_order.record(pending, (a,))
    init_order.record(pending, (b,))
    first, second = pending
    pending[first][0].complete = True
    init_order.order_behind(pending, None, ())  # a walk that writes nothing: the complete record is dropped
    init_order.record(pending, (a,))
    assert second in pending and len(pending) == 2 and len({first, *pending}) == 3, "a new record took the place of a pending one"

    s.calls.clear()
    pending.clear()
    monkeypatch.setattr(s.Event, "__init__", lambda self, complete=True, fails=False: self.__dict__.update(complete=complete, fails=fails))
    for _ in range(init_order.SWEEP_AT):
        init_order.record(pending, (a,))
    assert len(pending) == init_order.SWEEP_AT, "below the mark nothing is read"
    assert not [c for c in s.calls if c[0] == "query"]
    init_order.record(pending, (a,))
    assert not pending and pending.sweep_at == init_order.SWEEP_AT, "the complete records are dropped by the allocation that passes the mark"


def test_span_is_the_part_a_launch_touches_or_the_whole_footprint():
    t = torch.empty(6, 8, dtype=torch.int16)
    start = t.data_ptr()
    assert init_order.span(t) == (start, start + 96) and init_order.span(t, 16) == (start, start + 32)
    assert init_order.span(t[:, ::2]) == (start, start + 2 * (5 * 8 + 3 * 2 + 1)), "the bounds of a strided view"
    nothing, one = init_order.span(t[1:1]), init_order.span(torch.empty((), dtype=torch.float32))
    assert nothing[0] == nothing[1] and one[1] - one[0] == 4, "an empty view addresses no byte; a 0-d tensor one element"


class _Holder:
    """What an entry does at construction."""

    def __init__(self):
        self._pending = init_order.pending_of(DEV)


def test_the_records_of_a_device_are_one_object(monkeypatch):
    """Every entry of a device holds THE records of that device: the same object for the same device however often it is
    asked for, another for another device; a copy, a deep copy or a pickle of an object that holds it refers to the same
    object -- with a record pending too (an event is neither copied nor pickled) -- so a copied entry reads the device's
    records (the tensors a deep copy clones are not recorded: that is the copier's).  A CUDA device without an index is
    refused.  The truth test an entry makes in front of a launch is no Python-level call.
    The module needs neither torch nor the CuTe DSL."""
    s = stand_ins(monkeypatch)
    pending = init_order.pending_of(DEV)
    assert init_order.pending_of(torch.device("cuda", 5)) is pending and init_order.pending_of(torch.device("cuda", 6)) is not pending
    assert isinstance(pending, dict) and pending.device == DEV
    with pytest.raises(ValueError, match="per CUDA device index"):
        init_order.pending_of(torch.device("cuda"))

    for filled in (False, True):
        if filled:
            _put(pending, s.Event(False))
        holder = _Holder()
        for twin in (copy.copy(holder), copy.deepcopy(holder), pickle.loads(pickle.dumps(holder)), copy.deepcopy(pending)):
            assert getattr(twin, "_pending", twin) is pending and bool(pending) == filled

    counted = []
    sys.setprofile(lambda frame, event, arg: counted.append(event) if event in ("call", "c_call") else None)
    try:
        if pending:
            pass
    finally:
        sys.setprofile(None)
    assert counted == ["c_call"], "the truth test made a Python-level call"  # the one c_call is sys.setprofile(None) itself

    code = "import sys, cudnn.frost.init_order; print(sorted(m for m in sys.modules if m in ('torch', 'cutlass')))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


def test_host_threads_share_the_records_without_a_lock(monkeypatch):
    """A record that another host thread dropped behind this walk's copy is skipped and the walk goes on; one dropped WHILE
    this call reads its event (a query, or the host wait under a capture) is dropped again without an error; one recorded
    meanwhile is left for the next call.  Then two real threads walk while a third records: no exception, no record left."""
    s = stand_ins(monkeypatch)
    pending, launch = init_order.pending_of(DEV), s.Stream(OTHER)
    everything = ((0, 10**9),)

    class Racing(s.Event):
        """An event whose read lets "another thread" act first."""

        def __init__(self, complete, act):
            super().__init__(complete)
            self.act = act

        def query(self):
            self.act()
            return super().query()

        def synchronize(self):
            self.act()
            super().synchronize()

    gone, last = s.Event(False), s.Event(False)
    first = Racing(False, lambda: pending.pop(victim))
    _put(pending, first, INIT, ((1000, 2000),))
    victim = _put(pending, gone, 300, ((3000, 4000),))
    _put(pending, last, 400, ((5000, 6000),))
    got = _walk(s, pending, launch, everything)
    assert got == [("is capturing", OTHER), ("query", first), ("wait_event", OTHER, first), ("query", last), ("wait_event", OTHER, last)]
    assert victim not in pending and len(pending) == 2, "a record dropped behind the copy: not read, and the walk goes on"

    for captured in (False, True):
        pending.clear()
        s.capturing = {OTHER} if captured else set()
        late = s.Event(False)
        racing = Racing(True, lambda: (pending.pop(mine), _put(pending, late, 500)))
        mine = _put(pending, racing)
        _walk(s, pending, launch, everything)
        assert list(pending.values()) == [(late, 500, ((1000, 2000),))], "dropped twice without an error; the new record waits for the next call"

    s.capturing = set()
    pending.clear()
    monkeypatch.setattr(s.Event, "__init__", lambda self, complete=True, fails=False: self.__dict__.update(complete=complete, fails=fails))
    a = torch.empty(4)
    failures, stop = [], threading.Event()

    def walker():
        try:
            while not stop.is_set() or pending:
                if pending:
                    init_order.order_behind(pending, launch, everything)
        except BaseException as err:  # noqa: BLE001  (reported by the assertion below)
            failures.append(err)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads = [threading.Thread(target=walker, daemon=True) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for _ in range(3000):
            init_order.record(pending, (a,))
    finally:  # also when ``record`` raises: the walkers end with the case, they do not keep the session alive
        stop.set()
        sys.setswitchinterval(previous)
    for t in threads:
        t.join(60)
    assert not failures and not pending and not any(t.is_alive() for t in threads)


def test_relaxed_capture_check_exchanges_this_threads_mode_and_restores_it(monkeypatch):
    s = stand_ins(monkeypatch)
    with capture.relaxed_capture_check() as relaxed:
        assert relaxed is True and s.Driver.mode == "relaxed"
    assert s.calls == [RELAXED, RESTORED] and s.Driver.mode == "global"
    with pytest.raises(KeyError):
        with capture.relaxed_capture_check():
            raise KeyError("the block raised")
    assert s.Driver.mode == "global", "restored when the block raises"
    s.Driver.fails = True
    ran = []
    with pytest.raises(RuntimeError, match="cuThreadExchangeStreamCaptureMode failed: 1"):
        with capture.relaxed_capture_check():
            ran.append("the block ran")
    with capture.relaxed_capture_check(required=False) as relaxed:
        ran.append(relaxed)
    assert ran == [False], "not required: the block is told to read nothing"
    monkeypatch.setattr("cudnn._device._driver", lambda: None)
    with capture.relaxed_capture_check() as relaxed:
        assert relaxed is True, "without a visible CUDA device nothing can be capturing"
