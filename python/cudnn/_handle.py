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

import functools


@functools.lru_cache(maxsize=None)
def _device_info(ordinal: int) -> "DeviceInfo":
    """One DeviceInfo per ordinal (never keyed by Handle: the per-device driver
    caches assume a stable ordinal, and two handles on the same GPU share facts)."""
    return DeviceInfo(ordinal)


class DeviceInfo:
    """Device facts for a CUDA ordinal, read from the frost driver introspector
    (``cudnn.frost.device``, itself lru_cached per ordinal). This is the single
    device-info surface for the FE — every field a caller needs off a handle.

    The introspector is imported lazily so a plain cuDNN-backend user who never
    touches ``handle.device`` does not pull in the frost/cuda-python stack.
    """

    __slots__ = ("ordinal",)

    def __init__(self, ordinal: int):
        self.ordinal = int(ordinal)

    @property
    def compute_capability(self) -> tuple[int, int]:
        from .frost.device import compute_capability

        return compute_capability(self.ordinal)

    @property
    def sm_version(self) -> int:
        """Packed ``major * 10 + minor`` (derived, so it cannot drift from the tuple)."""
        major, minor = self.compute_capability
        return major * 10 + minor

    @property
    def sm_count(self) -> int:
        from .frost.device import multiprocessor_count

        return multiprocessor_count(self.ordinal)

    @property
    def shared_memory_per_block_optin(self) -> int:
        from .frost.device import shared_memory_per_block_optin

        return shared_memory_per_block_optin(self.ordinal)

    @property
    def oversized_shared_memory_per_block(self) -> int:
        from .frost.device import oversized_shared_memory_per_block

        return oversized_shared_memory_per_block(self.ordinal)

    @property
    def l2_cache_bytes(self) -> int:
        from .frost.device import l2_cache_bytes

        return l2_cache_bytes(self.ordinal)

    @property
    def device_name(self) -> str:
        from .frost.device import device_name

        return device_name(self.ordinal)

    def __repr__(self) -> str:
        return f"DeviceInfo(cuda:{self.ordinal})"


class Handle:
    """A cuDNN handle as a first-class FE object: owns the backend
    ``cudnnHandle_t`` (``backend_handle``), the stream it runs on (``stream`` —
    the source of truth that used to live in the module-global
    ``_handle_to_stream`` dict), and its device (``device``, lazy, cached per
    ordinal). The naming anticipates the front end BEING "cudnn": this object is
    the ``handle``; the wrapped ``cudnnHandle_t`` is the ``backend_handle``.

    The backend handle is handed to C++ EXPLICITLY via ``to_backend_handle()`` /
    ``unwrap_handles()`` at the handoff sites (all in ``_pygraph`` and this
    module — none elsewhere), so there is no hidden path from a Handle to the
    backend: a Handle that reaches a C++ binding unconverted fails loudly rather
    than being silently coerced. The dunders are therefore minimal on purpose —
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
        return _device_info(ordinal)

    def __repr__(self) -> str:
        return f"cudnn.Handle(backend_handle=0x{self.backend_handle:x}, cuda:{self._ordinal})"


def to_backend_handle(handle):
    """The backend cudnnHandle_t to hand to the C++ layer: a Handle's
    ``backend_handle``, a foreign raw-int handle unchanged, or None. The ONE
    explicit conversion from the first-class Handle to the int the bindings take."""
    return handle.backend_handle if isinstance(handle, Handle) else handle


def unwrap_handles(args, kwargs):
    """Replace any Handle in a passthrough ``(*args, **kwargs)`` with its
    ``backend_handle``, so an opaque forwarder (get_workspace_size, cuda-graph,
    deserialize) can hand C++ a plain int without knowing the handle's position."""
    return tuple(to_backend_handle(a) for a in args), {k: to_backend_handle(v) for k, v in kwargs.items()}
