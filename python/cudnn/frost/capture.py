# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph capture status of a stream, and the ONE typed refusal every FROST host path spells the same way.

A serving stack captures a block's execute into a CUDA graph with no eager warm-up, on a stream the block never ran on.
Anything a FROST host path creates lazily on its first use -- a cuDNN handle bound to the launch stream, a compiled plan,
a blocking host read of device values -- is then either illegal inside the capture (the capture is invalidated, or the
call fails with a stream-capture error) or, worse, captured with its first-use values and replayed for ever.  The host
paths that have such a lazy step ask here first and refuse TYPED, naming the route that would have invalidated the capture
and the remedy, instead of letting the driver report an "operation not permitted when stream is capturing" from the
middle of a launch sequence.

:func:`capture_state` is the status query (``cuStreamGetCaptureInfo``), legal under every capture mode;
:func:`is_capturing` its boolean; :func:`check_not_capturing` the refusal.  ``stream`` is a raw ``CUstream`` handle
(``torch.cuda.Stream.cuda_stream`` / ``torch.cuda.current_stream().cuda_stream``; ``0`` = the legacy default stream).  The
driver answers for a stream of its own handle from any host thread; for the legacy default stream it needs a context
current on the calling thread and raises ``CUDA_ERROR_INVALID_CONTEXT`` on a thread that has made no CUDA call yet -- that
stream cannot capture, so a caller that may run on such a thread does not ask for it (``cudnn.frost.init_order`` does not).

The callers today: the gated attention block's projection GEMM driver (``gated_attention_block/kernels/proj_gemm.py``:
``compile()`` refused under capture; the GRAPH route's cuDNN handle creation refused under capture) and the grouped GEMM's
debug-only device-value validation (``gemm/cutedsl/grouped/backend_utils.py``), which keeps its own remedy text.

:func:`relaxed_capture_check` is for the opposite need: a host path that must READ a CUDA event while a capture may be open
somewhere in the process.  Its caller today: ``cudnn.frost.init_order`` (the order of a direct-call entry's launches behind the
initialization of the buffers it allocates for itself); ``buffers.DeviceBuffer.__del__`` keeps its own teardown-tolerant form.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

#: What a refused caller is told when it names no remedy of its own.
DEFAULT_REMEDY = (
    "compile at warm-up, before the capture begins, and run one eager execute on the capture stream before capturing "
    "(it creates every lazily created resource); the capture itself may then hold nothing but the launches"
)

# cuStreamGetCaptureInfo refuses the LEGACY default stream while another stream captures in global mode
# (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906); that stream cannot be capturing, so the refusal reads as "not capturing".
# The one blind spot: a lazy resource created on the legacy stream WHILE a global-mode capture is live elsewhere is itself
# an unsafe call (cudaStreamCaptureModeGlobal invalidates that capture), and this query cannot see it -- a caller that runs
# on the legacy stream beside a capturing one is outside what the helper can refuse; the block's launch streams are explicit.
_CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906


def capture_state(stream) -> Tuple[bool, Optional[int]]:
    """``(capturing, capture_id)`` of the raw ``CUstream`` handle ``stream``.

    Every stream joined to one capture reports that capture's id; a stream whose capture has been INVALIDATED is still
    held by it and reads ``(True, None)`` (the driver gives no id for a dead capture).  Without a visible CUDA device
    nothing can be capturing: ``(False, None)``.  A driver error other than the legacy-stream refusal above is raised --
    a failed status query is a bug to see, not a reason to guess.
    """
    from cudnn._device import _driver

    drv = _driver()
    if drv is None:
        return False, None
    out = drv.cuStreamGetCaptureInfo(drv.CUstream(int(stream)))
    err = int(out[0])
    if err != 0:
        if err == _CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            return False, None
        raise RuntimeError(f"cuStreamGetCaptureInfo({int(stream):#x}) failed: {out[0]!r}")
    status = out[1]
    if status == drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        return False, None
    if status == drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        return True, int(out[2])
    return True, None  # CU_STREAM_CAPTURE_STATUS_INVALIDATED


def is_capturing(stream) -> bool:
    """True when ``stream`` (a raw ``CUstream`` handle) is inside a CUDA-graph capture, active or invalidated."""
    return capture_state(stream)[0]


def check_not_capturing(stream, what: str, *, remedy: Optional[str] = None) -> None:
    """Raise a typed ``RuntimeError`` when ``stream`` is capturing; a no-op otherwise.

    ``what`` names the ROUTE that cannot run inside a capture (``"frost gemm graph route (cuDNN handle creation)"``,
    ``"<label>: FROST GEMM compile"``) and ``remedy`` what the caller should do instead -- :data:`DEFAULT_REMEDY` when the
    route has no remedy of its own.  The message carries both, so whoever runs a serving stack reads which step of which
    route broke the capture and how to avoid it, never a bare driver error.
    """
    if not is_capturing(stream):
        return
    raise RuntimeError(f"{what}: refused under CUDA-graph capture (stream {int(stream):#x} is capturing); {remedy or DEFAULT_REMEDY}")


@contextmanager
def relaxed_capture_check(required: bool = True) -> Iterator[bool]:
    """Run the block with THIS thread's stream-capture check relaxed and restore the previous mode behind it, also when the
    block raises.  For a host path that must READ a CUDA event (``cudaEventQuery``, ``cudaEventSynchronize``) while a stream
    capture may be open in the process: in the default (global) capture mode such a call is refused while ANY thread
    captures, and the refusal invalidates that capture -- another thread's too.  Relaxing lifts the refusal of this thread's
    calls and nothing else: the caller reads events recorded OUTSIDE the capture only.

    Yields ``True`` when the block may read (the check is relaxed; also without a visible CUDA device, where nothing can be
    capturing).  When the mode cannot be exchanged: ``RuntimeError`` and the block does not run (``required``), else the
    block runs with ``False`` and must read nothing.
    """
    from cudnn._device import _driver

    drv = _driver()
    if drv is None:
        yield True
        return
    err, previous = drv.cuThreadExchangeStreamCaptureMode(drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED)
    if int(err) != 0:
        if required:
            raise RuntimeError(f"cuThreadExchangeStreamCaptureMode failed: {err}")
        yield False
        return
    try:
        yield True
    finally:
        drv.cuThreadExchangeStreamCaptureMode(previous)


__all__ = ["DEFAULT_REMEDY", "capture_state", "check_not_capturing", "is_capturing", "relaxed_capture_check"]
