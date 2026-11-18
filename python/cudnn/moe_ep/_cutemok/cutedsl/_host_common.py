# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Host-side helpers shared by the CuTe DSL hosts (no CUTLASS import at module import).

Device, CUDA context and stream lookups; tensor validation and stride normalization for the TMA
descriptors; the workspace-owned peer address tables; the validated-call and launch-plan caches of the
repeated-call paths; and the optional on-disk cache of compiled executors.  The launch-plan cache keys on
the identity of its source tensors through weak references and re-checks their address and layout, so a
freed or changed source is a miss, never a stale view.
"""

from __future__ import annotations

import contextlib
import threading

from .persistent_bf16_fwd_contract import COMM_TILE_SHAPES, DEFAULT_COMM_TILE_SHAPE, CommTileShape

_CONTEXT_KEYS: dict = {}


def current_cuda_context_key() -> int:
    """Return the current CUDA context handle; executors are owned per context.

    The handle is read once per device and reused for that device.  This assumes PyTorch's model
    of one context per device, its primary context: a caller that makes another context current on
    a device it already used is not tracked, and the cached executors then fail with a CUDA error
    rather than run."""
    device_index = current_device_index()
    key = _CONTEXT_KEYS.get(device_index)
    if key is not None:
        return key
    from cuda.bindings import driver as cuda

    result, context = cuda.cuCtxGetCurrent()
    if result != cuda.CUresult.CUDA_SUCCESS or context is None:
        raise RuntimeError(f"cuCtxGetCurrent failed: {result}")
    try:
        key = int(context)
    except TypeError:
        value = getattr(context, "value", None)
        if value is None:
            raise RuntimeError("CUDA context handle is not representable as an integer")
        key = int(value)
    _CONTEXT_KEYS[device_index] = key
    return key


def current_raw_stream(device) -> int:
    """Raw handle of PyTorch's current stream on ``device`` (an index or a ``torch.device``),
    without building a ``torch.cuda.Stream`` object per call."""
    import torch

    if not isinstance(device, int):
        device = device.index if device.index is not None else torch.cuda.current_device()
    return torch._C._cuda_getCurrentRawStream(device)


#: Raw stream handle -> its ``CUstream``: a handle names the same stream for as long as the stream lives,
#: and a recycled handle names its new stream just as well, so one wrapper per value serves every call.
_STREAM_HANDLES: dict = {}


def current_stream_handle(device):
    """``CUstream`` of PyTorch's current stream on ``device`` (see current_raw_stream)."""
    raw = current_raw_stream(device)
    handle = _STREAM_HANDLES.get(raw)
    if handle is None:
        from cuda.bindings import driver as cuda

        handle = _STREAM_HANDLES[raw] = cuda.CUstream(raw)
    return handle


def current_device_index() -> int:
    """``torch.cuda.current_device()`` without its per-call lazy-initialization check, for the launch
    paths, which only run once CUDA tensors exist (CUDA is initialized)."""
    import torch

    return torch._C._cuda_getDevice()


_CAPABILITIES: dict = {}
_PROPERTIES: dict = {}


def _device_index(device) -> int:
    index = device if type(device) is int else device.index
    return current_device_index() if index is None else index


def device_capability(device) -> tuple[int, int]:
    """``torch.cuda.get_device_capability(device)``, read once per device."""
    index = _device_index(device)
    capability = _CAPABILITIES.get(index)
    if capability is None:
        import torch

        capability = _CAPABILITIES[index] = torch.cuda.get_device_capability(index)
    return capability


def device_properties(device):
    """``torch.cuda.get_device_properties(device)``, read once per device."""
    index = _device_index(device)
    properties = _PROPERTIES.get(index)
    if properties is None:
        import torch

        properties = _PROPERTIES[index] = torch.cuda.get_device_properties(index)
    return properties


#: Bytes a TMA descriptor's base address and non-unit strides must be a multiple of.
TMA_ALIGNMENT_BYTES = 16


def validate_tensor(name, tensor, *, shape, dtype, device, sliced: bool = False) -> None:
    """Rejects a tensor the kernels cannot address.

    Args:
        sliced: accept a tensor whose innermost dimension is contiguous but whose outer strides are
            larger than its own extents imply - a slice of a larger allocation.  Each weight gets
            its own TMA descriptor and the kernels never assume two of them are adjacent (the
            routed Gate/Up interleave exists in shared memory only), so the outer strides only have
            to satisfy TMA's alignment.  This admits, for example, the routed Gate and Up as the
            two halves of one combined ``[E, 2I, H]`` parameter without copying them apart.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must be {dtype} {tuple(shape)} on {device}")
    if tensor.data_ptr() % TMA_ALIGNMENT_BYTES:
        raise ValueError(f"{name} data pointer must be {TMA_ALIGNMENT_BYTES}-byte aligned")
    if tensor.is_contiguous():
        return
    if not sliced:
        raise ValueError(f"{name} must be contiguous")
    elements = TMA_ALIGNMENT_BYTES // tensor.element_size()
    strides, extents = tensor.stride(), tuple(shape)
    if strides[-1] != 1:
        raise ValueError(f"{name} must be contiguous in its last dimension; strides are {strides}")
    inner = 1
    for axis in range(len(extents) - 1, -1, -1):
        if strides[axis] < inner:
            raise ValueError(f"{name} overlaps itself: shape {extents}, strides {strides}")
        if axis != len(extents) - 1 and strides[axis] % elements:
            raise ValueError(
                f"{name} stride {strides[axis]} of dimension {axis} must be a multiple of "
                f"{elements} elements ({TMA_ALIGNMENT_BYTES} bytes); strides are {strides}"
            )
        inner = strides[axis] * extents[axis]


def normalize_situ_beta(situ_beta, swiglu_limit) -> tuple[float, float] | None:
    """SiTU-GLU's two betas as the kernel's FP32 values, or None for SwiGLU.  Two positive finite
    numbers, checked after rounding to FP32 (where the kernel sees them); never together with a clamp."""
    import math
    import struct

    if situ_beta is None:
        return None
    error = "situ_beta must contain two positive finite FP32 numbers"
    if not isinstance(situ_beta, tuple) or len(situ_beta) != 2 or any(type(beta) not in (int, float) for beta in situ_beta):
        raise ValueError(error)
    try:
        beta = struct.unpack("ff", struct.pack("ff", *situ_beta))
    except (OverflowError, struct.error) as exc:
        raise ValueError(error) from exc
    if any(not math.isfinite(value) or value <= 0 for value in beta):
        raise ValueError(error)
    if swiglu_limit is not None:
        raise ValueError("SiTU cannot be combined with swiglu_limit")
    return beta


def row_major_view(tensor):
    """Normalize the strides of a validated tensor without copying.

    Singleton and empty dimensions of a *contiguous* tensor may carry arbitrary strides, and the
    CuTe DSL ABI descriptors expect canonical row-major metadata, so those are rewritten.

    A tensor that is not contiguous is returned untouched.  Its strides are the truth: it is a
    slice of a larger allocation -- for example the routed Gate and Up weights as the two halves
    of one combined ``[E, 2I, H]`` parameter -- and rewriting them to the canonical ones would
    describe a different region of memory, with no error, returning wrong data.
    """
    shape = tensor.shape
    if not tensor.is_contiguous():
        return tensor
    if 1 not in shape and 0 not in shape:
        return tensor
    strides, stride = [], 1
    for extent in reversed(tensor.shape):
        strides.append(stride)
        stride *= int(extent)
    return tensor.as_strided(tensor.shape, tuple(reversed(strides)))


def peer_address_table(addresses, device, tables: dict):
    """int64 device tensor of the ``ep_size`` peer base addresses.

    Symmetric buffers are allocated once per workspace, so the table for a given address
    tuple is built once and kept in ``tables``, a dict owned by that workspace
    (``workspace_peer_tables``): it is released with the workspace, and no process-global
    cache holds CUDA tensors.  The kernel indexes it by the runtime peer rank; ``ep_size``
    is the tuple length, never a constant.
    """
    import torch

    addresses = tuple(int(address) for address in addresses)
    if not addresses:
        raise ValueError("peer address table must not be empty")
    key = (str(device), addresses)
    table = tables.get(key)
    if table is None:
        table = torch.tensor(addresses, dtype=torch.int64, device=device)
        tables[key] = table
    return table


def workspace_peer_tables(workspace) -> dict:
    """The workspace-owned cache of peer address tables (created on first use)."""

    tables = getattr(workspace, "_cutedsl_peer_tables", None)  # duck-typed workspaces may lack the field
    if tables is None:
        tables = workspace._cutedsl_peer_tables = {}
    return tables


#: The workspace's symmetric buffers whose peer addresses the megakernels read from a device table.
PEER_TABLE_BUFFERS = ("x_buffer", "combine_buffer", "d_y_buffer", "d_x_routed_buffer", "router_weight_buffer", "d_router_weight_buffer")


def build_workspace_peer_tables(workspace) -> None:
    """Build every peer address table of the workspace (one host-to-device copy each) when it is created, so that
    no forward or backward builds one: inside a CUDA graph capture that copy fails.  The views of the workspace
    (``CuteMoKPlan.workspace_for``) share its table dict."""

    tables = workspace_peer_tables(workspace)
    for name in PEER_TABLE_BUFFERS:
        peer_address_table(getattr(workspace, f"{name}_ptrs"), workspace.device, tables)


def tensor_layout_signature(tensors) -> tuple[object, ...]:
    """Hashable layout of ``tensors`` for the validated-call caches of the small operators:
    type, dtype, device, shape, strides and contiguity (no address)."""

    return tuple((type(t), t.dtype, t.device, tuple(t.shape), t.stride(), t.is_contiguous()) for t in tensors)


def typed_key(*values) -> tuple[object, ...]:
    """Hashable key of scalar arguments and flat lists / tuples of scalars that keeps the type of
    every value and element, so that ``True``, ``1`` and ``1.0``, or a tuple and a list, never share
    an entry validated for one of them.  Built on every call: one C-level map per list, no Python
    call per element.  A nested list is unhashable here and goes to the full validation."""

    return tuple((type(value), tuple(value), tuple(map(type, value))) if isinstance(value, (list, tuple)) else (type(value), value) for value in values)


class ValidatedCalls:
    """Call signature -> executor key, for signatures whose full validation passed.

    A signature holds every layout, scalar and device value the validation reads; the address
    checks (alignment, overlap) are repeated on every call by the caller.  The cache is cleared
    when it reaches ``limit`` entries."""

    def __init__(self, limit: int = 64) -> None:
        self._entries: dict = {}
        self._limit = limit

    def get(self, signature):
        return self._entries.get(signature)

    def put(self, signature, key) -> None:
        if len(self._entries) >= self._limit:
            self._entries.clear()
        self._entries[signature] = key

    def clear(self) -> None:
        self._entries.clear()


class LaunchPlanCache:
    """Launch plans: the checked executor and argument template of one set of source tensors and
    scalars, built once; a call only fills the per-call slots.

    An entry is keyed by the identity of its source tensors plus ``extra`` (a typed_key) and
    remembered with weak references and each source's address, shape, strides and dtype, so a
    freed, re-pointed, resized or re-typed source is a miss (rebuild), never a stale view.  The
    plans themselves must not hold the sources.  ``limit`` entries (one per layer and geometry in
    use) are kept per instance; the plans of freed sources are dropped at the next ``put``."""

    def __init__(self, limit: int = 256) -> None:
        self._entries: dict = {}
        self._limit = limit

    @staticmethod
    def _view_key(views) -> tuple:
        return tuple((view.data_ptr(), view.shape, view.stride(), view.dtype, view.device) for view in views)

    def get(self, sources, extra, views=()):
        """The plan of ``sources`` (by identity, see the class) and ``views`` (by address and layout).

        ``views`` are tensors the launch takes afresh on every call (the plan's template holds none of
        them) and that callers rebuild per call as new objects over the same memory -- for example Gate /
        Up row views narrowed from a combined FC1 parameter on every call -- so a new view object
        of the same address, shape, strides, dtype and device finds the plan its predecessor built."""
        entry = self._entries.get((tuple(map(id, sources)), self._view_key(views), extra))
        if entry is None:
            return None
        refs, layouts, value = entry
        for ref, (pointer, shape, stride, dtype), source in zip(refs, layouts, sources):
            if ref() is not source or source.data_ptr() != pointer or source.shape != shape or source.stride() != stride or source.dtype != dtype:
                return None
        return value

    def put(self, sources, extra, value, views=()) -> None:
        import weakref

        # Drop the plans of freed sources first: their templates may hold a freed workspace's peer tables.
        self._entries = {key: entry for key, entry in self._entries.items() if all(ref() is not None for ref in entry[0])}
        if len(self._entries) >= self._limit:
            self._entries.clear()
        self._entries[(tuple(map(id, sources)), self._view_key(views), extra)] = (
            tuple(weakref.ref(source) for source in sources),
            tuple((source.data_ptr(), source.shape, source.stride(), source.dtype) for source in sources),
            value,
        )

    def clear(self) -> None:
        self._entries.clear()


class SlotTemplate:
    """A launch-argument list split into kept values and per-call slots, derived rather than hand-numbered.

    ``build`` takes one complete argument list (what the full launch path built for a call) and the call's
    *per-call sources*: every tensor, stream or object the call may pass differently next time -- the
    caller's tensors, the workspace buffers, the fresh state.  Each argument that is one of the sources
    (identity), or ``row_major_view`` / ``routed_weight_view``-like view of one (same address, the view's
    layout), becomes a slot refilled from the next call's sources; ``named`` maps argument positions to
    per-call scalars passed by name (``fill(..., **scalars)``).  Every other argument is kept.  ``build``
    refuses (returns None) when a kept tensor lies in the memory of any per-call source: a derivation it
    does not recognize would otherwise keep a stale view, so the caller keeps its full path for that call.
    """

    __slots__ = ("values", "slots", "named")

    def __init__(self, values, slots, named):
        self.values, self.slots, self.named = values, slots, named

    @classmethod
    def build(cls, args, sources, transforms, named=None):
        import torch

        values = list(args)
        slots = []
        source_ids = {id(source): index for index, source in enumerate(sources)}
        ranges = []
        for source in sources:
            if isinstance(source, torch.Tensor) and source.numel():
                storage = source.untyped_storage()
                ranges.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
        named = dict(named or {})
        for position, value in enumerate(values):
            if position in named.values():
                values[position] = None
                continue
            index = source_ids.get(id(value))
            if index is not None and sources[index] is value:
                slots.append((position, index, -1))
                values[position] = None
                continue
            if not isinstance(value, torch.Tensor):
                continue
            match = None
            for index, source in enumerate(sources):
                if not isinstance(source, torch.Tensor) or source.device != value.device or source.data_ptr() != value.data_ptr():
                    continue
                for code, transform in enumerate(transforms):
                    derived = transform(source)
                    if (
                        isinstance(derived, torch.Tensor)
                        and derived.shape == value.shape
                        and derived.stride() == value.stride()
                        and derived.dtype == value.dtype
                    ):
                        match = (position, index, code)
                        break
                if match is not None:
                    break
            if match is not None:
                slots.append(match)
                values[position] = None
                continue
            # A kept tensor: it must not live in any per-call source's memory.
            if value.numel():
                pointer = value.data_ptr()
                if any(low <= pointer < high for low, high in ranges):
                    return None
        return cls(values, tuple(slots), tuple(named.items()))

    def fill(self, sources, transforms, **scalars):
        args = self.values.copy()
        for position, index, code in self.slots:
            source = sources[index]
            args[position] = source if code < 0 else transforms[code](source)
        for name, position in self.named:
            args[position] = scalars[name]
        return args


def tensor_layout(tensor) -> tuple[object, ...]:
    """What a per-call tensor of a launch plan must keep: type, shape, strides, dtype, device."""
    return (type(tensor), tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def layouts_match(tensors, layouts, alignment: int) -> bool:
    """Whether per-call tensors (fresh every call, so not part of a plan key) have the layouts the
    plan was validated with, and ``alignment``-byte aligned addresses."""
    for tensor, (cls, shape, stride, dtype, device) in zip(tensors, layouts):
        if (
            type(tensor) is not cls
            or tensor.shape != shape
            or tensor.stride() != stride
            or tensor.dtype != dtype
            or tensor.device != device
            or tensor.data_ptr() % alignment
        ):
            return False
    return True


def checked_template(full, per_call_slots, fill, message: str) -> tuple[object, ...]:
    """``full`` (a complete launch-argument list) with its per-call slots emptied, after checking
    once that ``fill`` rebuilds exactly those slots; the kept template then holds no per-call or
    caller-owned tensor."""

    full = tuple(full)
    per_call = {slot % len(full) for slot in per_call_slots}
    filled = fill([None] * len(full))
    for slot in set(range(len(full))) - per_call:
        filled[slot] = full[slot]
    if not all(same_launch_arg(a, b) for a, b in zip(filled, full)):
        raise RuntimeError(message)
    return tuple(None if slot in per_call else arg for slot, arg in enumerate(full))


def same_launch_arg(a, b) -> bool:
    """Whether two launch arguments are interchangeable: tensors by address, dtype, shape and
    strides; streams by handle; scalars by type and value.  Used once per new template to check
    that its per-call slot positions match the full argument builder."""
    import torch
    from cuda.bindings import driver as cuda

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.data_ptr() == b.data_ptr()
            and a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape)
            and a.stride() == b.stride()
        )
    if isinstance(a, cuda.CUstream):
        return isinstance(b, cuda.CUstream) and int(a) == int(b)
    return type(a) is type(b) and a == b


#: The width of an H / I tail: half a 256-wide tile.  The megakernels' tail instantiations are generated for
#: exactly this many columns (their modules assert it against their own tile constants).
TAIL_WIDTH = 128


def has_tail(*sizes: int) -> bool:
    """Whether any of these H / I extents ends in a 128-wide tail (size % 256 != 0): the compile-time
    ``has_tail`` flag of the megakernels and of the backward's metadata builder.

    Compile time because the hidden and the intermediate size are model parameters and the flag only
    selects whether the tail code is generated: an aligned model runs loops without predicates or tail
    branches.  It doubles the executors of the kernels that take it -- one for every aligned shape, one
    for every shape with a tail, none per size -- as one more entry of their executor keys, which a model
    keeps for its whole run.  The values of H and I stay runtime arguments: nothing is specialised on a
    size."""
    # The precondition under which the tail instantiations are generated for a static TAIL_WIDTH: every extent
    # is whole TAIL_WIDTH-wide halves of a tile, so a tail is exactly TAIL_WIDTH wide.  The public entries
    # validate their sizes before they get here; this is the kernels' own guard.
    for size in sizes:
        if size % TAIL_WIDTH != 0 or size % (2 * TAIL_WIDTH) not in (0, TAIL_WIDTH):
            raise ValueError(f"an H / I extent must be a multiple of {TAIL_WIDTH}; got {size}")
    return any(size % (2 * TAIL_WIDTH) != 0 for size in sizes)


_WIDE_COMM_TILE_SHAPE = CommTileShape(16, 1024)
assert _WIDE_COMM_TILE_SHAPE in COMM_TILE_SHAPES


def comm_tile_shape(hidden_size: int, requested=None) -> CommTileShape:
    """The communication tile shape of the BF16 megakernels for a routed hidden size: one of the contract's
    COMM_TILE_SHAPES, the compile-time parameter of the communication CTAs of the forward and the backward
    megakernel and an entry of their executor keys.  ``requested`` None (the hosts' own default) takes 32 x 512 for
    every hidden size; ``CuteMoKConfig`` defaults to ``"auto"``.

    ``requested`` (``CuteMoKConfig.comm_tile_shape``, the ``comm_tile_shape`` keyword of ``forward_bf16`` /
    ``backward_bf16``): a CommTileShape or its (rows, columns), returned as the contract's CommTileShape; None
    takes the default.  ``"auto"`` picks by the hidden size alone: 16 x 1024 when it is a multiple of 1024 (no
    partial column block; measured no slower than 32 x 512 at any token count, faster where the communication
    CTAs are the bound), 32 x 512 otherwise.  The hidden size is a model parameter, so this stays one variant per
    model.  This is the only selection point of both hosts; they validate ``hidden_size`` themselves (their
    plans)."""
    if requested is None:
        return DEFAULT_COMM_TILE_SHAPE
    if isinstance(requested, str) and requested == "auto":
        return _WIDE_COMM_TILE_SHAPE if hidden_size % _WIDE_COMM_TILE_SHAPE.columns == 0 else DEFAULT_COMM_TILE_SHAPE
    shape = requested if isinstance(requested, CommTileShape) else (CommTileShape(*requested) if isinstance(requested, tuple) and len(requested) == 2 else None)
    if shape not in COMM_TILE_SHAPES:
        allowed = " or ".join(str((s.rows, s.columns)) for s in COMM_TILE_SHAPES)
        raise ValueError(f'comm_tile_shape must be {allowed} (rows, columns) or "auto"; got {requested!r}')
    return COMM_TILE_SHAPES[COMM_TILE_SHAPES.index(shape)]


__all__ = [
    "TAIL_WIDTH",
    "comm_tile_shape",
    "has_tail",
    "normalize_situ_beta",
    "LaunchPlanCache",
    "PEER_TABLE_BUFFERS",
    "ValidatedCalls",
    "build_workspace_peer_tables",
    "checked_template",
    "compiled_only",
    "current_cuda_context_key",
    "current_raw_stream",
    "current_stream_handle",
    "layouts_match",
    "peer_address_table",
    "row_major_view",
    "same_launch_arg",
    "tensor_layout",
    "tensor_layout_signature",
    "typed_key",
    "workspace_peer_tables",
]


# ---------------------------------------------------------------------------------------------
# Compiled executors kept across processes
# ---------------------------------------------------------------------------------------------
#: The directory compiled executors are kept in between processes; unset, every process compiles.
#: The kernels are otherwise compiled by every rank of every job on its first call, and
#: ``cute.compile`` bypasses CuTe DSL's own file cache (it compiles with ``no_cache``).  A shared
#: directory (one per toolchain, e.g. on the job's filesystem) lets every later job load what the
#: first one compiled.
EXECUTOR_CACHE_DIR_ENV = "MOK_CUTEDSL_CACHE_DIR"
_SOURCE_FINGERPRINT: list[str] = []


def _source_fingerprint() -> str:
    """A hash of every source file of the CuTe DSL package: any change to the kernels -- including
    one that forgets to bump a codegen epoch -- makes every kept executor unreachable."""
    if not _SOURCE_FINGERPRINT:
        import hashlib
        from pathlib import Path

        digest = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.rglob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        _SOURCE_FINGERPRINT.append(digest.hexdigest())
    return _SOURCE_FINGERPRINT[0]


def _toolchain() -> tuple[object, ...]:
    from importlib import metadata

    from ._arch import COMPILE_OPTIONS  # the target and TVM FFI are compile options, not environment variables

    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return (version("nvidia-cutlass-dsl"), version("apache-tvm-ffi"), COMPILE_OPTIONS)


_EXECUTING = threading.local()


@contextlib.contextmanager
def compiled_only():
    """The execute step of a plan (``cudnn.moe_ep._cutemok.functional.forward_execute``): inside, a kernel variant that is not in
    its process cache raises instead of being compiled or loaded.  The plan step compiles every variant its
    configuration launches, so a miss here is a call the plan was not made for -- and under a CUDA graph capture
    a compile would run in the middle of the recording.  Every ``cute.compile`` of the package goes through
    ``persistent_executor``, which is where this is checked."""
    depth = getattr(_EXECUTING, "depth", 0)
    _EXECUTING.depth = depth + 1
    try:
        yield
    finally:
        _EXECUTING.depth = depth


def persistent_executor(disk_key, compile_executor):
    """The executor ``compile_executor()`` returns, kept in ``MOK_CUTEDSL_CACHE_DIR`` across processes.

    ``disk_key`` is the caller's executor-cache key without what is particular to a process (the
    device index and the CUDA context): everything that selects the generated code.  The kept file
    is named by a hash of it, of the package source and of the toolchain, so a kept executor is
    only ever loaded for exactly the code it was compiled from.  A miss compiles, then exports the
    executor (CuTe DSL's ``export_to_c``) to a temporary file renamed into place, so ranks that miss
    together never read a partial file; a kept file that fails to load is compiled over.

    Called on a miss of a host's process cache, so it never compiles or loads during a CUDA graph capture of the
    current stream: that raises.  Warm up (one eager call of each configuration) or make a plan before capturing.
    """
    import hashlib
    import os

    import torch

    if getattr(_EXECUTING, "depth", 0):
        raise RuntimeError(
            f"CuteMoK execute: the kernel variant {disk_key[:1]} was not compiled by the plan of this call; "
            "make a plan for this configuration first (kernels are not compiled in the execute step)"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"CuteMoK: the kernel variant {disk_key[:1]} must be compiled before CUDA graph capture; run the call "
            "once before capturing it (warm-up), or make a plan (kernels are not compiled during a capture)"
        )
    directory = os.environ.get(EXECUTOR_CACHE_DIR_ENV)
    if not directory:
        return DirectExecutor(compile_executor())
    name = "mok_" + hashlib.sha256(repr((_source_fingerprint(), _toolchain(), disk_key)).encode()).hexdigest()[:40]
    path = os.path.join(directory, name + ".o")
    if os.path.exists(path):
        try:
            import cutlass.cute as cute

            return DirectExecutor(getattr(cute.runtime.load_module(path, enable_tvm_ffi=True), name))
        except Exception as error:  # a damaged or foreign file: compile it again
            import warnings

            warnings.warn(f"CuteMoK: ignoring the kept executor {path}: {error}")
    executor = compile_executor()
    try:
        os.makedirs(directory, exist_ok=True)
        # export_to_c picks the format from the extension, so the temporary name keeps ".o".
        temporary = os.path.join(directory, f"{name}.{os.getpid()}.{id(executor)}.tmp.o")
        executor.export_to_c(temporary, name)
        os.replace(temporary, path)
    except Exception as error:  # keeping it is an optimization, never a failure
        import warnings

        warnings.warn(f"CuteMoK: could not keep the executor in {directory}: {error}")
    return DirectExecutor(executor)


def _flattens(argument) -> bool:
    """Whether CuTe DSL's TVM FFI call would flatten ``argument`` (a dataclass instance)."""
    from dataclasses import is_dataclass

    return is_dataclass(argument) and not isinstance(argument, type)


class DirectExecutor:
    """A CuTe DSL TVM FFI executor called without its Python-side argument flattening.

    CuTe DSL's ``TVMFFIJitCompiledFunction.__call__`` passes every argument through
    ``_flatten_dataclass_arg`` (an ``is_dataclass`` check and a generator step each) before the C-level
    ``tvm_ffi.Function.__call__``; with the megakernels' dozens of arguments that is avoidable host time
    on every launch.  The first call checks its arguments: when none is a dataclass -- the launch
    paths pass tensors, scalars and streams -- every later call goes straight to
    ``tvm_ffi.Function.__call__``; otherwise this executor keeps the flattening call.  Other attributes
    are the executor's.
    """

    __slots__ = ("executor", "_call")

    def __init__(self, executor):
        self.executor = executor
        self._call = None

    def __call__(self, *args, **kwargs):
        call = self._call
        if call is None or kwargs:
            if kwargs:
                return self.executor(*args, **kwargs)
            call = self._call = self._select(args)
        return call(*args)

    def _select(self, args):
        import functools

        try:
            import tvm_ffi
        except ImportError:
            return self.executor
        if isinstance(self.executor, tvm_ffi.Function) and not any(map(_flattens, args)):
            return functools.partial(tvm_ffi.Function.__call__, self.executor)
        return self.executor

    def __getattr__(self, name):
        return getattr(self.executor, name)


def process_independent(key) -> tuple[object, ...]:
    """An executor-cache key without its ``("device", ...)`` and ``("context", ...)`` entries:
    the part that selects code, for ``persistent_executor``."""
    return tuple(part for part in key if not (isinstance(part, tuple) and len(part) == 2 and part[0] in ("device", "context")))
