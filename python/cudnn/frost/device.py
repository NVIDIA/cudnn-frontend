# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which GPU a FROST plan is built for — shared by every FROST engine.

The device is recorded on the compiled plan and compared against
:func:`current_device` at execute time, so a plan whose baked constants
describe one GPU fails loudly instead of launching on another.

That comparison is about the LAUNCH, not the buffers. cuDNN's own variant pack
carries pointers, uids and a workspace and no device at all, so an operand's
device is not something the front end has an opinion about.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _driver():
    """The cuInit'd driver module, or ``None`` when no CUDA device is visible."""
    import cuda.bindings.driver as drv

    if int(drv.cuInit(0)[0]) != 0:
        return None
    return drv


def _ck(err, *rest):
    if int(err) != 0:
        drv = _driver()
        detail = drv.cuGetErrorString(err)[1].decode() if drv is not None else ""
        raise RuntimeError(f"cudnn.frost: CUDA driver error {err}{f': {detail}' if detail else ''}")
    return rest[0] if len(rest) == 1 else None


def is_available() -> bool:
    """True when at least one CUDA device is visible to this process."""
    drv = _driver()
    return drv is not None and int(_ck(*drv.cuDeviceGetCount())) > 0


def device_count() -> int:
    drv = _driver()
    return 0 if drv is None else int(_ck(*drv.cuDeviceGetCount()))


def _device_handle(device: int):
    """``CUdevice`` for an ordinal. Needs only cuInit — creates no context."""
    drv = _driver()
    if drv is None:
        raise RuntimeError("cudnn.frost: no CUDA device visible")
    count = int(_ck(*drv.cuDeviceGetCount()))
    if not 0 <= device < count:
        raise ValueError(f"cudnn.frost: cuda:{device} does not exist ({count} device(s) visible)")
    return _ck(*drv.cuDeviceGet(device))


def current_device() -> int:
    """CUDA device index a plan built right now would target.

    A bound driver context wins — it is process-wide and authoritative. Before
    anything has allocated there may be none, and ``cudaSetDevice`` (what
    ``torch.cuda.set_device`` drives) has only moved the runtime's thread-local
    slot, so that is the second rung."""
    drv = _driver()
    if drv is None:
        raise RuntimeError("cudnn.frost: no CUDA device visible")
    if int(_ck(*drv.cuCtxGetCurrent())) != 0:
        return int(_ck(*drv.cuCtxGetDevice()))
    import cuda.bindings.runtime as rt

    err, index = rt.cudaGetDevice()
    if int(err) != 0:
        raise RuntimeError(f"cudnn.frost: cudaGetDevice failed: {err}")
    return int(index)


def resolve_device(device=None) -> int:
    """Normalize ``None`` / int / ``"cuda:N"`` / ``torch.device`` to an index.

    ``None`` (and a device without an explicit index) means the current device."""
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    kind = getattr(device, "type", None)
    index = getattr(device, "index", None)
    if kind is None:
        kind, _, tail = str(device).partition(":")
        index = int(tail) if tail else None
    if kind != "cuda":
        raise ValueError(f"cudnn.frost: expected a CUDA device, got {device}")
    return current_device() if index is None else int(index)


@functools.lru_cache(maxsize=None)
def compute_capability(device: int) -> tuple[int, int]:
    drv = _driver()
    handle = _device_handle(device)
    attr = drv.CUdevice_attribute
    major = int(_ck(*drv.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle)))
    minor = int(_ck(*drv.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle)))
    return major, minor


@functools.lru_cache(maxsize=None)
def multiprocessor_count(device: int) -> int:
    drv = _driver()
    handle = _device_handle(device)
    return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, handle)))


@functools.lru_cache(maxsize=None)
def shared_memory_per_block_optin(device: int) -> int:
    drv = _driver()
    handle = _device_handle(device)
    return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, handle)))


# CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK. Named in CUDA 13.4's
# cuda.h; cuda-python's CUdevice_attribute does not carry it yet, so ask by ordinal.
_ATTR_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK = 150


@functools.lru_cache(maxsize=None)
def oversized_shared_memory_per_block(device: int) -> int:
    """Per-CTA SMEM ceiling in the *oversized* carveout (327 KiB vs the 227 KiB
    opt-in limit on SM 10.7), which the part gives by shrinking L1 to 8 kB — free for
    a TMA-fed GEMM. 0 when the driver has no such mode."""
    drv = _driver()
    handle = _device_handle(device)
    err, value = drv.cuDeviceGetAttribute(_ATTR_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK, handle)
    return int(value) if int(err) == 0 else 0


@functools.lru_cache(maxsize=None)
def l2_cache_bytes(device: int) -> int:
    drv = _driver()
    handle = _device_handle(device)
    return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, handle)))


@functools.lru_cache(maxsize=None)
def device_name(device: int) -> str:
    drv = _driver()
    return _ck(*drv.cuDeviceGetName(256, _device_handle(device))).split(b"\x00")[0].decode()


class device_context:
    """Bind ``device``'s primary context for the enclosing block, then restore
    whatever was bound before. The retain/release pair is refcounted, so this
    composes with a process that already owns the same primary context."""

    def __init__(self, device: int):
        self._device = device
        self._drv = None
        self._handle = None
        self._previous = None

    def __enter__(self):
        self._drv = _driver()
        if self._drv is None:
            raise RuntimeError("cudnn.frost: no CUDA device visible")
        self._handle = _device_handle(self._device)
        self._previous = _ck(*self._drv.cuCtxGetCurrent())
        _ck(*self._drv.cuCtxSetCurrent(_ck(*self._drv.cuDevicePrimaryCtxRetain(self._handle))))
        return self

    def __exit__(self, *exc):
        _ck(*self._drv.cuCtxSetCurrent(self._previous))
        _ck(*self._drv.cuDevicePrimaryCtxRelease(self._handle))
        return False
