# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class ``cudnn.Handle``.

The backend ``cudnnHandle_t`` binds a device and carries the current stream, but
on the FE side the handle used to be a bare int with nowhere to hang per-handle
state — so that state accreted as side tables (the ``_handle_to_stream`` dict)
and per-engine device queries (frost's ``current_device()``). ``Handle`` gives
the handle a home for ``{backend_handle, device, stream}`` while staying a drop-in for the
old int: it converts to the raw ``intptr_t`` via ``__index__`` anywhere a binding
wants it, so code that forwards a handle to ``graph.execute`` / ``set_stream`` /
the C++ layer is unchanged.
"""

from __future__ import annotations

# The device-fact layer lives in cudnn._device (the FE's single owner of a GPU's
# properties); Handle.device is the DeviceInfo for the handle's ordinal.
from ._device import DeviceInfo, device_info


class Handle:
    """A cuDNN handle as a first-class FE object: owns the backend
    ``cudnnHandle_t`` (``backend_handle``), the stream it runs on (``stream`` —
    the source of truth that used to live in the module-global
    ``_handle_to_stream`` dict), and its device (``device``, lazy, cached per
    ordinal). The naming anticipates the front end BEING "cudnn": this object is
    the ``handle``; the wrapped ``cudnnHandle_t`` is the ``backend_handle``.

    The backend handle is handed to C++ EXPLICITLY via ``to_backend_handle()``
    at the handoff sites (all in ``_pygraph`` and this module — none elsewhere),
    so there is no hidden path from a Handle to the backend: a Handle that reaches
    a C++ binding unconverted fails loudly rather than being silently coerced. The dunders are therefore minimal on purpose —
    NO ``__index__``/``__int__`` (no implicit int coercion) — and
    ``__eq__``/``__hash__``/``__bool__`` are left at the object defaults
    (identity equality, identity hash, always-truthy), which is what the
    surrounding code needs: the handle stays a valid dict key, stays truthy in
    ``if handle:`` guards, and does not raise on ``wrapper.py``'s ``== 'auto'``.
    """

    __slots__ = ("backend_handle", "_ordinal", "stream")

    def __init__(self, backend_handle: int, ordinal: int | None = None):
        self.backend_handle = int(backend_handle)
        self._ordinal = ordinal
        self.stream = None  # None = default stream, until set_stream() is called

    @property
    def device(self) -> DeviceInfo:
        ordinal = self._ordinal
        if ordinal is None:
            from .frost.device import current_device

            ordinal = self._ordinal = current_device()
        return device_info(ordinal)

    def __repr__(self) -> str:
        return f"cudnn.Handle(backend_handle=0x{self.backend_handle:x}, cuda:{self._ordinal})"


def to_backend_handle(handle):
    """The backend cudnnHandle_t to hand to the C++ layer: a Handle's
    ``backend_handle``, a foreign raw-int handle unchanged, or None. The ONE
    explicit conversion from the first-class Handle to the int the bindings take."""
    return handle.backend_handle if isinstance(handle, Handle) else handle
