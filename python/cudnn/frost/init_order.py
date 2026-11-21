# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Order a launch behind the initialization of an entry-owned allocation, on whatever stream the launch runs.

A direct-call entry that allocates an output or a scratch buffer for itself does so with torch, on torch's current stream of
its device AT THAT MOMENT: the allocation, a zero-fill, a fill torch itself enqueues for ``torch.empty`` under deterministic
algorithms, and whatever that stream still owes a block the allocator recycles are all STREAM WORK of that stream.  A launch
on another stream is not ordered behind it: the kernel's bytes are written first and the late fill lands on top of them.  The
first launch of a newly compiled kernel has been observed to wait until the other streams' work is done, which hides this; a
warm launch does not wait.

This module is the ONE implementation of the ordering.  An entry makes three calls:

* at construction, ``self._pending = init_order.pending_of(self.device)``: the device's records, kept by reference;
* ``init_order.record(self._pending, tensors)`` right behind an allocation and its fills, BEFORE the entry stores the tensors
  where a launch can find them: an event is recorded on torch's current stream of the device and kept, with the address range
  of every tensor of the allocation, among the device's records;
* in front of EVERY launch, behind the launch stream's resolution::

      stream = torch.cuda.current_stream(self.device)
      if self._pending:  # empty in the steady state: a warm call does nothing here
          init_order.order_behind(self._pending, stream, ranges_the_launch_writes)

  ``ranges_the_launch_writes`` are half-open byte ranges (:func:`span`): every operand the launch WRITES -- an output the
  entry owns, an output the caller passed, a caller-owned operand the kernel updates in place -- over the bytes this launch
  can write of it (the live rows of a contiguous slab; the whole footprint where the written part is device data).  A
  superset is correct; a range that leaves written bytes out is the defect.  An entry that allocates nothing makes the first
  and the third call: its launch can write an allocation ANOTHER entry of the device recorded.

THE RULE (:func:`order_behind`).  The ordering follows the ALLOCATION, never the way a tensor reached the call: every range
the launch writes is compared, by address, with every range of every record of the launch's device -- of this entry or of
any other.  A record that is hit and whose initialization may be pending: an eager launch stream waits for the event
(``cudaStreamWaitEvent``; the host does not wait) and the record stays for the next stream -- whatever its raw handle, the
allocating stream's too: a handle does not name a stream for good, and on the allocating stream itself the event is already
behind the launch in stream order, so that wait is free; a capturing stream -- it cannot wait for an event recorded outside
the capture, so the HOST waits for that ONE event (``cudaEventSynchronize``) before the captured launch and no stream wait
is enqueued.  That is the one host wait of the ordering: it is set-up, bounded by what the allocating stream had queued,
never part of a replay, and a caller avoids it by allocating before the capture and letting that stream's work finish
(``torch.cuda.graph`` synchronizes the device when it is entered; a raw capture does not).  Never a synchronization of a
stream or of the device.  A record whose event is complete is dropped by ANY call that walks the device's records, hit or
not, so every flow reaches "no record" and a warm call then pays one truth test.  Operands a launch only READS are not
ordered here, and neither are the caller's own reads and writes of a returned tensor outside a call.

WHAT A CALL COSTS WHILE A RECORD IS PENDING: every call of every entry of the device walks the device's records -- address
arithmetic per record, one capture-mode exchange, and event queries: one per record it drops and one per allocating stream
that still holds a pending record (the events of one stream complete in the order they were recorded, so behind a stream's
first pending record its later ones are taken as pending without a query).  Allocations that no launch follows are swept
of complete records by a later allocation that finds more records than the mark (:data:`SWEEP_AT`, or twice what the last
sweep left); until then the complete ones stay, one event each.

THE RECORDS are kept per device for the life of the process, ONE object for every entry of that device
(:func:`pending_of`); a copy or a pickle of an entry refers to the same object (the copy's tensors are another matter: a
deep copy made after an allocation clones the buffers with torch, on the copying thread's stream, and nothing records the
clones -- copy an entry before it allocates).  A record holds no tensor: an event, the raw handle of the allocating stream
and integers.  It stays true when its tensors are freed -- until the event completes that stream may still write those
bytes -- so a launch into a block that a later allocation reused is ordered behind it too, and it is gone at the first call
behind its completion.  Host threads: every change is one dictionary operation and a walk reads a copy; no lock.  A record
another host thread dropped is skipped, and a drop never raises.  THE LAUNCH STREAM IS NEVER COMPARED WITH THE ALLOCATING
ONE: a raw handle is all a record can keep of a stream, the driver can give the handle of a destroyed stream to the next
new one while the destroyed stream's queued work still runs, and the per-thread stream handle is another stream in every
host thread.  The handle a record keeps only saves event queries (the paragraph above), where taking two streams for one
leaves a complete record for a later walk and never drops a pending one.

Not recorded: an allocation on a device without streams (a host entry), and one made INSIDE a stream capture -- its fill is
a node of the captured graph, ordered in front of the captured launches by the graph itself, and an event recorded there
cannot be queried.  An allocation on the legacy default stream while a blocking stream captures raises torch's own error
(no event can be recorded there).  No torch import at module level.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Tuple

from cudnn._torch_stream import DEFAULT_STREAM_HANDLES, device_context  # imports no framework at module level

from . import capture

#: an allocation that finds more records than this on its device drops the complete ones (then at twice what was left)
SWEEP_AT = 64
_SERIAL = itertools.count()


class Records(dict):
    """The records of ONE device: ``serial -> (event, raw handle of the allocating stream, the byte ranges of the
    allocation's tensors)`` for every entry-owned allocation whose initialization MAY still be pending.  Empty in the steady
    state, and its truth is all an entry reads then.  The object is the device's for the life of the process: copying or
    pickling it (inside an entry) yields the same object, never a private set of records."""

    __slots__ = ("device", "sweep_at")

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sweep_at = SWEEP_AT

    def __reduce__(self):  # what a copy, a deep copy and a pickle are built from: the device's own object
        return pending_of, (self.device,)


_RECORDS: Dict[tuple, Records] = {}


def pending_of(device) -> Records:
    """The records of ``device`` (a ``torch.device``; a CUDA device with its index): THE object, shared by every entry of
    the device.  An entry keeps it at construction, tests its truth in front of a launch and hands it to :func:`record` and
    :func:`order_behind`.  ``ValueError`` for a CUDA device without an index (it would name another set of records)."""
    if device.type == "cuda" and device.index is None:
        raise ValueError("cudnn.frost.init_order: the records are kept per CUDA device index; resolve the device first")
    key = (device.type, device.index)
    found = _RECORDS.get(key)
    return found if found is not None else _RECORDS.setdefault(key, Records(device))


def span(t, elements=None) -> Tuple[int, int]:
    """``[start, end)`` in bytes of the storage a tensor view addresses: its first ``elements`` elements (the part of a
    contiguous operand a launch touches), else its whole footprint -- the strides included, so the bounds of a strided view."""
    start = t.data_ptr()
    if elements is None:
        elements = 1
        for extent, stride in zip(t.shape, t.stride()):
            if extent == 0:
                return start, start
            elements += (extent - 1) * stride
    return start, start + elements * t.element_size()


def record(pending: Records, tensors: Iterable) -> None:
    """Record the initialization of ONE entry-owned allocation (``tensors``: everything it allocated together) as stream work
    of torch's current stream of the device of ``pending``.  Call it right behind the last allocation / fill and before the
    tensors are stored where a launch can find them (a host thread that finds the tensors then finds the record)."""
    device = pending.device
    if device.type != "cuda":
        return
    import torch

    spans = tuple(s for s in map(span, tensors) if s[0] < s[1])
    if not spans:
        return
    with device_context(device):  # torch's capture status is that of the CURRENT device's current stream
        if torch.cuda.is_current_stream_capturing():  # torch's own query: an allocation needs no driver binding (a launch does)
            return
        stream = torch.cuda.current_stream(device)
        done = torch.cuda.Event()
        done.record(stream)  # behind the allocation, its fills and all this stream still owes a recycled block
    pending[next(_SERIAL)] = (done, stream.cuda_stream, spans)
    if len(pending) > pending.sweep_at:  # allocations no launch followed: drop what is complete
        try:
            order_behind(pending, None, ())
        except ImportError:  # a host without the driver binding: nothing launches there either
            pass
        pending.sweep_at = max(SWEEP_AT, 2 * len(pending))


def order_behind(pending: Records, stream, written: Iterable[Tuple[int, int]]) -> None:
    """Order a launch on ``stream`` (the torch stream of the device of ``pending`` it is about to run on) that writes the byte
    ranges ``written`` behind every record of the device it hits and whose initialization may be pending, and drop the
    records whose event is complete (the module docstring's rule).  Call it only when ``pending`` is not empty.  Every read
    of an event runs with this thread's capture check relaxed (a query while ANY thread captures in the default mode is
    refused and invalidates that capture) and the check is restored also when a read raises.  When it cannot be relaxed
    nothing is read and nothing is dropped: ``RuntimeError`` for a launch that writes a recorded allocation, a plain return
    for one that does not.  ``stream`` is not used by a walk that writes nothing (``written`` empty: a drop-only walk)."""
    walk = pending.copy()  # a copy: other host threads record and drop while this call walks
    if not walk:
        return
    written = tuple(w for w in written if w[0] < w[1])  # once: the ranges are walked per record; an empty one hits nothing
    hits = set()
    for serial, (_, _, spans) in walk.items():
        for b0, b1 in spans:
            for a0, a1 in written:
                if a0 < b1 and b0 < a1:
                    hits.add(serial)
                    break
            else:
                continue
            break
    capturing = None
    held = set()  # the allocating streams found holding a pending record: their later records are pending too
    with device_context(pending.device), capture.relaxed_capture_check(required=bool(hits)) as relaxed:
        if not relaxed:
            return
        try:
            for serial, (done, init_handle, _) in walk.items():
                if serial not in pending:  # another host thread saw it complete and dropped it: nothing to order behind
                    continue
                hit = serial in hits
                if hit:
                    if capturing is None:
                        handle = stream.cuda_stream
                        # the legacy default stream cannot capture, and the driver reads its status only for a host thread
                        # with a current context: a thread whose first CUDA call is this launch has none yet
                        capturing = handle not in DEFAULT_STREAM_HANDLES and capture.is_capturing(handle)
                    if capturing:
                        done.synchronize()  # this ONE event, on the host: a captured launch cannot carry the dependency
                        pending.pop(serial, None)
                        continue
                if init_handle not in held:
                    if done.query():
                        pending.pop(serial, None)  # complete: dropped by any call, hit or not
                        continue
                    held.add(init_handle)
                if hit:  # on the allocating stream's raw handle too: a new stream can have got it, and on the same one it is free
                    stream.wait_event(done)  # the launch stream waits, the host does not; the record stays for the next stream
        finally:
            walk = done = None  # a precaution: a dropped event is released here, while the capture check is still relaxed


__all__ = ["order_behind", "pending_of", "record", "span"]
