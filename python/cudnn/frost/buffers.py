# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-buffer introspection and zero-copy views over the DLPack protocol.

The engines exchange buffers with the kernels exclusively through DLPack:
caller buffers pass through untouched, and workspace carves are exposed as
:class:`DeviceView` objects built directly over (pointer, shape, dtype) —
no tensor-library dependency on the execute path.

* :func:`probe` — (ptr, shape, strides, dtype-name, device-id) of any buffer
  exposing ``__cuda_array_interface__`` or ``__dlpack__``.
* :class:`DeviceView` — a DLPack-capable view over a raw device pointer;
  what workspace carving hands to the kernel entries.
* :func:`current_sm` — the active device's SM number via ``cuda.bindings``.
"""

from __future__ import annotations

import ctypes
import struct

from cudnn import _pybind_module

# ---------------------------------------------------------------------------
# DLPack ABI (dlpack.h v0.8 layout; the unversioned "dltensor" capsule)
# ---------------------------------------------------------------------------

_KDL_CUDA = 2


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER_T = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER_T),
]


_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

# name -> (DLPack type code, bits); typestr -> name for the CAI path
# name -> (DLPack type code, bits). Names match how torch spells them, so one
# table serves both a dtype read off a buffer and one named in a graph.
# Sub-byte types are deliberately absent: DTYPE_ITEMSIZE below is bits // 8, so
# fp4 would land on 0 and take every byte/element conversion with it.
DTYPES = {
    "float32": (2, 32),
    "float16": (2, 16),
    "bfloat16": (4, 16),
    "float64": (2, 64),
    "int64": (0, 64),
    "int32": (0, 32),
    "int8": (0, 8),
    "uint8": (1, 8),
    "bool": (6, 8),
    "float8_e4m3fn": (10, 8),
    "float8_e5m2": (12, 8),
    "float8_e8m0fnu": (14, 8),
}
_TYPESTR = {"<f4": "float32", "<f2": "float16", "<f8": "float64", "<i8": "int64", "<i4": "int32", "|i1": "int8", "|u1": "uint8", "|b1": "bool"}
_CODE_BITS = {v: k for k, v in DTYPES.items()}

DTYPE_ITEMSIZE = {name: bits // 8 for name, (code, bits) in DTYPES.items()}


def dtype_name(buf) -> str:
    """Buffer dtype as a bare name ("torch.bfloat16" and "bfloat16" both map
    to "bfloat16")."""
    return str(buf.dtype).split(".")[-1]


def data_ptr(buf) -> int:
    """Device address of a tensor-like (``data_ptr()`` or the CUDA array interface)."""
    fn = getattr(buf, "data_ptr", None)
    if fn is not None:
        return fn()
    return buf.__cuda_array_interface__["data"][0]


class DeviceView:
    """Zero-copy DLPack view over a raw CUDA pointer.

    The view owns no memory — the underlying allocation (workspace or caller
    buffer) must outlive it. Row-major contiguous.

    Not a concept an engine has to learn: it is what a variant-pack slot or a
    workspace region turns into on the way to a kernel, because CuTe needs an
    object exposing ``__dlpack__`` and neither a pointer nor a Tensor record
    is one."""

    def __init__(self, ptr: int, shape, dtype: str, device_id: int):
        self._ptr = int(ptr)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self._device_id = int(device_id)

    def data_ptr(self) -> int:
        return self._ptr

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def reshape(self, *shape):
        """Contiguous reshape (supports one ``-1`` wildcard)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        numel = 1
        for s in self.shape:
            numel *= s
        wild = [i for i, s in enumerate(shape) if s == -1]
        fixed = 1
        for s in shape:
            if s != -1:
                fixed *= int(s)
        if wild:
            if len(wild) > 1 or fixed == 0 or numel % fixed:
                raise ValueError(f"cannot reshape {self.shape} to {shape}")
            shape = tuple(numel // fixed if s == -1 else int(s) for s in shape)
        else:
            if fixed != numel:
                raise ValueError(f"cannot reshape {self.shape} to {shape}")
        return DeviceView(self._ptr, shape, self.dtype, self._device_id)

    view = reshape

    def view_as(self, other):
        return self.reshape(tuple(other.shape))

    def contiguous(self):
        return self  # row-major contiguous by construction

    def element_size(self) -> int:
        return DTYPE_ITEMSIZE[self.dtype]

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        n = DTYPE_ITEMSIZE[self.dtype]
        for s in self.shape:
            n *= s
        return n

    def __dlpack_device__(self):
        return (_KDL_CUDA, self._device_id)

    def __dlpack__(self, *, stream=None, **_kwargs):
        """Delegate to a slot, which owns the struct it hands out.

        A view has nowhere to put a struct that must outlive the capsule: it
        cannot know when the consumer is done with it, and cute's from_dlpack
        keeps the pointer rather than copying the DLTensor. Holding the struct
        on the view and shipping a no-op deleter -- which is what this did --
        is a use-after-free the moment a consumer outlives the view.
        """
        code, bits = DTYPES[self.dtype]
        return _pybind_module.make_operand_buffer(self._ptr, list(self.shape), code, bits, self._device_id).__dlpack__()


class DeviceBuffer(DeviceView):
    """A uint8 device allocation this process owns, for the paths where no
    caller buffer exists. Allocated on the CURRENT context, so the caller owns
    the context this memory belongs to."""

    def __init__(self, nbytes: int, device_id: int):
        from cuda.bindings import driver as _drv

        err, ptr = _drv.cuMemAlloc(int(nbytes))
        if int(err) != 0:
            raise RuntimeError(f"cudnn.frost: cuMemAlloc({int(nbytes)}) failed: {err}")
        super().__init__(int(ptr), (int(nbytes),), "uint8", device_id)

    def __del__(self):
        # At interpreter teardown the context can already be gone, which makes
        # the free fail on memory the driver has reclaimed anyway.
        try:
            from cuda.bindings import driver as _drv

            _drv.cuMemFree(self.data_ptr())
        except Exception:  # noqa: BLE001
            pass


def probe(buf):
    """(ptr, shape, strides_in_elements_or_None, dtype_name, device_id) of a
    device buffer, via ``__cuda_array_interface__`` when available (torch,
    CuPy, numba) else the ``__dlpack__`` protocol.

    Raises for a buffer it cannot read. Callers that would rather have the
    geometry of a buffer whose DTYPE has no name here — fp8, fp4, anything
    sub-byte — want :func:`_dlpack_geometry`, which separates the two
    failures."""
    geometry = _dlpack_geometry(buf)
    if geometry is None:
        raise TypeError(f"buffer of type {type(buf).__name__} exposes neither __cuda_array_interface__ nor __dlpack__")
    if geometry[3] is None:
        raise TypeError(f"unsupported buffer dtype for {type(buf).__name__}")
    return geometry


def _dlpack_geometry(buf):
    """``probe``'s reading, with its two declines made distinguishable.

    Returns None when the buffer exposes neither protocol — nothing but a
    pointer will ever come out of it. Returns the 5-tuple with ``dtype_name``
    set to None when the buffer IS readable but its dtype has no name in
    ``DTYPES``; dim and stride are real in that case and worth keeping.
    """
    try:
        # torch's property RAISES for dtypes CAI can't express (bf16) instead
        # of being absent — treat any failure as "no CAI" and use DLPack
        cai = getattr(buf, "__cuda_array_interface__", None)
    except Exception:  # noqa: BLE001
        cai = None
    if cai is not None:
        dtype = _TYPESTR.get(cai["typestr"])
        # an unmapped typestr (e.g. torch presents bf16 as raw '<V2') falls
        # through to the DLPack path, which carries the true dtype
        if dtype is not None:
            ptr, _ro = cai["data"]
            strides_b = cai.get("strides")
            itemsize = DTYPE_ITEMSIZE[dtype]
            strides = tuple(s // itemsize for s in strides_b) if strides_b else None
            dev = getattr(buf, "device", None)
            device_id = getattr(dev, "index", None) or 0
            return int(ptr), tuple(cai["shape"]), strides, dtype, int(device_id)

    dl = getattr(buf, "__dlpack__", None)
    if dl is None:
        return None
    # stream=-1 is DLPack's "the caller handles synchronisation; do no
    # bookkeeping". This only reads metadata, so it never needed any -- and the
    # default makes torch call record_stream, which is illegal inside a CUDA
    # graph capture.
    try:
        capsule = dl(stream=-1)
    except TypeError:  # a producer whose __dlpack__ predates the stream kwarg
        capsule = dl()
    raw = _PyCapsule_GetPointer(capsule, b"dltensor")
    mt = ctypes.cast(raw, ctypes.POINTER(_DLManagedTensor)).contents
    t = mt.dl_tensor
    shape = tuple(t.shape[i] for i in range(t.ndim))
    strides = tuple(t.strides[i] for i in range(t.ndim)) if t.strides else None
    dtype = _CODE_BITS.get((t.dtype.code, t.dtype.bits)) if t.dtype.lanes == 1 else None
    ptr = (t.data or 0) + t.byte_offset
    device_id = t.device.device_id
    # release: mark the capsule consumed and run its deleter
    _PyCapsule_SetName(capsule, b"used_dltensor")
    if mt.deleter:
        mt.deleter(ctypes.cast(raw, ctypes.POINTER(_DLManagedTensor)))
    return int(ptr), shape, strides, dtype, int(device_id)


def is_contiguous(shape, strides) -> bool:
    if strides is None:
        return True
    expect = 1
    for dim, stride in zip(reversed(shape), reversed(strides)):
        if dim != 1 and stride != expect:
            return False
        expect *= dim
    return True


def memset_zero_async(ptr: int, nbytes: int, stream) -> None:
    """Stream-ordered zero-fill of a device range via ``cuda.bindings``."""
    from cuda.bindings import runtime as _rt

    res = _rt.cudaMemsetAsync(int(ptr), 0, int(nbytes), int(stream) if stream is not None else 0)
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"cudaMemsetAsync failed: {err}")


_WORD_FORMAT = {"fp32": "<f", "int32": "<i"}


def init_word(dtype: str, value) -> int:
    """The 32-bit pattern that writes ``value`` to a buffer of ``dtype``.

    A memset moves bits, not numbers, so the value has to be packed as the dtype
    the kernel will read it back as. int32's reduction identities are the ends
    of its range and are exactly where that bites: -2**31 packed as float is
    0xcf000000 where the kernel wants 0x80000000.
    """
    fmt = _WORD_FORMAT.get(dtype)
    if fmt is None:
        raise NotImplementedError(f"no 32-bit fill pattern for dtype {dtype!r}")
    return int.from_bytes(struct.pack(fmt, value), "little")


def fill_word_async(ptr: int, count: int, word: int, stream) -> None:
    """Stream-ordered fill of ``count`` CONTIGUOUS 32-bit words with ``word``.

    An engine that seeds a caller's buffer owns that operation itself: reaching
    for ``tensor.fill_()`` works only while the buffer happens to be a torch
    tensor, and queues on torch's current stream rather than the one the kernel
    will run on. Every seed a reduction uses is a 32-bit pattern, so the
    driver's D32 memset covers them without a kernel -- see ``init_word`` for
    turning a value into one.

    :func:`fill_word_strided_async` is the same fill for a buffer that is not
    one dense run.
    """
    from cuda.bindings import driver as _drv

    res = _drv.cuMemsetD32Async(int(ptr), int(word), int(count), int(stream) if stream is not None else 0)
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"cuMemsetD32Async failed: {err}")


def _fill_word_2d_async(ptr: int, pitch_words: int, width: int, height: int, word: int, stream) -> None:
    from cuda.bindings import driver as _drv

    res = _drv.cuMemsetD2D32Async(int(ptr), int(pitch_words) * 4, int(word), int(width), int(height), int(stream) if stream is not None else 0)
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"cuMemsetD2D32Async failed: {err}")


def collapse_layout(shape, strides) -> list:
    """``(extent, stride)`` outermost first, with unit axes dropped and adjacent
    axes merged where one exactly fills the other's gap.

    A padded output is usually dense underneath its declared rank -- a rank-3
    ``(1, M, 1)`` tap is one strided run, and a contiguous one is a single dense
    run whatever rank it was declared at. Merging first is what keeps the fill
    below down to one memset in both cases.
    """
    axes = sorted(((int(d), int(s)) for d, s in zip(shape, strides) if int(d) != 1), key=lambda ds: -ds[1])
    out: list = []
    for extent, stride in axes:
        if out and out[-1][1] == extent * stride:
            out[-1] = (out[-1][0] * extent, stride)
        else:
            out.append((extent, stride))
    return out


def strided_fill_plan(shape, strides) -> "list | None":
    """The 2D memsets that cover a strided region exactly once, or None.

    None means the region writes some element twice -- a stride of 0 over a real
    extent, or an outer stride that does not clear the axis below it. That is a
    write race whichever buffer it is, so it is refused rather than issued; the
    caller decides how to say so.

    Returned before anything is written, which is the point: the seed's
    preconditions have to be settled while the caller's buffer is still
    untouched, and a plan is what lets several outputs all be checked before the
    first of them is filled.

    Each entry is ``(offset, pitch, width, height)`` in ELEMENTS. The driver's
    2D memset takes a pitch, so a per-row scalar tap is one entry rather than one
    per row (the reading that made this look expensive: 572 us at one memset per
    row); what remains is one entry per point of whatever axis is left outside
    the 2D region, which for a rank-3 output is the batch and is usually 1.
    """
    if any(int(s) == 0 and int(d) != 1 for d, s in zip(shape, strides)):
        return None
    axes = collapse_layout(shape, strides)
    # Non-overlapping iff each axis clears the whole span of the one below it.
    # `pitch >= width` is this rule at the innermost pair and misses the rest:
    # shape (2, 2) stride (2, 2) has width 1 and passes it, and lands both axes
    # on the same element.
    for (_outer_extent, outer_stride), (inner_extent, inner_stride) in zip(axes, axes[1:]):
        if outer_stride < inner_extent * inner_stride:
            return None
    if not axes:
        return [(0, 1, 1, 1)]
    # The innermost run is the memset's width when it is dense; otherwise every
    # element stands alone and the width is one.
    width, rest = (axes[-1][0], axes[:-1]) if axes[-1][1] == 1 else (1, axes)
    if not rest:
        return [(0, width, width, 1)]
    height, pitch = rest[-1]
    offsets = [0]
    for extent, stride in reversed(rest[:-1]):
        offsets = [base + i * stride for base in offsets for i in range(extent)]
    return [(base, pitch, width, height) for base in offsets]


def apply_fill_plan(ptr: int, plan, word: int, stream) -> None:
    """Issue a plan from :func:`strided_fill_plan`, stream-ordered."""
    for offset, pitch, width, height in plan:
        if height == 1:
            fill_word_async(ptr + offset * 4, width, word, stream)
        else:
            _fill_word_2d_async(ptr + offset * 4, pitch, width, height, word, stream)


def fill_word_strided_async(ptr: int, shape, strides, elem_bytes: int, word: int, stream) -> None:
    """Plan and issue in one call, for a caller with a single region to seed.

    The engine owns seeding a reduction output, and a padded one is the case
    that used to send it back to the caller's ``fill_()`` -- the last place
    anything here wrote through a buffer it does not own, and the reason a
    perfectly legal call had to fall off the fast path. A caller with SEVERAL
    regions wants :func:`strided_fill_plan` for all of them first: this one
    cannot know whether the next region is refusable, so it would leave the
    earlier ones filled.
    """
    if elem_bytes != 4:
        raise NotImplementedError(f"frost: a reduction seed is a 32-bit pattern; this output stores {elem_bytes}-byte elements")
    plan = strided_fill_plan(shape, strides)
    if plan is None:
        raise ValueError(f"frost: a reduction output cannot write an element twice (shape {tuple(shape)} stride {tuple(strides)})")
    apply_fill_plan(ptr, plan, word, stream)


# The CuTe primitives these engines lower through landed in 4.7.0; older DSLs
# fail during codegen with errors that name a missing attribute rather than the
# version, so the check belongs where an engine can still decline.
CUTEDSL_MIN_VERSION = (4, 7, 0)

_DSL_STATE = None


def cutedsl_state():
    """``(installed, version)`` for the CuTe DSL, without importing it.

    Support checks must be able to say "my optional dependency is absent" at
    CHECK time, not discover it when lowering imports the adapter -- a decline
    there is honest but late, and the plan was already in the ranked list.
    ``find_spec`` resolves without executing (7 ms vs ~1 s for the real import)
    and the version comes from package metadata (5 ms).

    ``version`` is ``(distribution, version)`` or None: the public wheel is
    nvidia-cutlass-dsl, internal RCs ship as nvidia-cutlass-dsl-internal, and
    the two number themselves differently. Presence is what gates; the pair only
    refines the message and feeds :func:`cutedsl_too_old`.
    """
    global _DSL_STATE
    if _DSL_STATE is None:
        import importlib.metadata
        import importlib.util

        try:
            installed = importlib.util.find_spec("cutlass") is not None
        except (ImportError, ValueError):
            installed = False
        version = None
        for dist in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-internal"):
            try:
                version = (dist, importlib.metadata.version(dist))
                break
            except importlib.metadata.PackageNotFoundError:
                pass
        _DSL_STATE = (installed, version)
    return _DSL_STATE


def cutedsl_too_old(version):
    """``(distribution, version)`` is the public wheel, below the floor.

    Absent, unparsable, or an internal RC counts as NOT too old: internal builds
    carry their own numbering ("0.3.0+2026...") that the public floor cannot
    judge, and refusing on a string we failed to read would decline on a machine
    that works.
    """
    if not version:
        return False
    dist, ver = version
    if dist != "nvidia-cutlass-dsl":
        return False
    try:
        parts = tuple(int(x) for x in ver.split("+", 1)[0].split(".")[:3])
    except ValueError:
        return False
    return len(parts) == 3 and parts < CUTEDSL_MIN_VERSION


def current_device_id():
    """Active CUDA device id, or None when there is no device.

    Same two-probe shape as current_sm() below: a missing ``cuda-python`` must
    not look like a missing GPU.
    """
    try:
        from cuda.bindings import runtime as _rt

        err, dev = _rt.cudaGetDevice()
        if int(err) == 0:
            return int(dev)
    except Exception:  # noqa: BLE001 — fall through to the driver
        pass
    try:
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        if lib.cuCtxGetDevice(ctypes.byref(dev)) == 0 or lib.cuDeviceGet(ctypes.byref(dev), 0) == 0:
            return int(dev.value)
    except Exception:  # noqa: BLE001
        pass
    return None


def _sm_via_cuda_bindings():
    from cuda.bindings import runtime as _rt

    err, dev = _rt.cudaGetDevice()
    if int(err) != 0:
        return None
    err, major = _rt.cudaDeviceGetAttribute(_rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev)
    if int(err) != 0:
        return None
    err, minor = _rt.cudaDeviceGetAttribute(_rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev)
    if int(err) != 0:
        return None
    return major * 10 + minor


def _sm_via_driver():
    """The same query through libcuda directly — no python package needed."""
    import ctypes

    lib = ctypes.CDLL("libcuda.so.1")
    if lib.cuInit(0) != 0:
        return None
    dev = ctypes.c_int()
    if lib.cuCtxGetDevice(ctypes.byref(dev)) != 0 and lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        return None
    major, minor = ctypes.c_int(), ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, MINOR = 76
    if lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) != 0:
        return None
    if lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) != 0:
        return None
    return major.value * 10 + minor.value


def current_sm():
    """Active device's SM number (major*10 + minor), or None when there is no
    CUDA device.

    Two probes: a missing ``cuda-python`` in the image must not look like a
    missing GPU — that read cost every arch-gated engine its place in the plan
    list once already."""
    for probe in (_sm_via_cuda_bindings, _sm_via_driver):
        try:
            sm = probe()
        except Exception:  # noqa: BLE001 — try the next probe
            continue
        if sm is not None:
            return sm
    return None
