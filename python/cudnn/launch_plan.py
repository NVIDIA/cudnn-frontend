# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch plans: a planned function's repeated calls launch its kernels from C++ and skip its Python.

:func:`planned` wraps an API function whose GPU work is kernel launches made
through :func:`launch`, plus torch allocations and fills. The first call for a
set of arguments (each tensor's dtype, shape, strides and device, and the other
arguments' values) runs as usual. The second runs as usual too and records what
it does: each :func:`launch` (the compiled kernel and its positional arguments),
the tensors it allocates and fills, and what it returns. The recording becomes a
plan when every kernel argument follows from the call: an argument tensor or a
view into one, a tensor the call allocates (one it returns, or scratch it drops),
the stream, or a constant (a value, or a buffer the API keeps across calls).
Every later call allocates the returned tensors and the scratch and hands their
addresses and the stream to ``_LaunchPlan``, which fills and launches in C++ in
the recorded order; the function's Python does not run.

A call the plan could not reproduce is not planned: a torch operation on a CUDA
tensor other than an allocation, a fill or a view; a CuTeDSL kernel launched
other than through :func:`launch`; a kernel argument that is none of the above;
or an allocation on a stream that is neither the current one nor a stream
argument. Kernels launched through a raw ``tvm_ffi.Function`` other than through
:func:`launch` cannot be seen, so a planned function must launch every kernel
through :func:`launch`.

A later call runs the function, not the plan, whenever the function could have
behaved differently: its argument tensors are less aligned than the recorded
call's (up to 256 bytes), or they overlap each other differently (overlapping
tensors must sit at the same byte offset from each other as in the recorded call). The key also
holds the current CUDA device and the environment variables the function names
(``planned(env=...)``). Each planned function keeps its 256 most recently used
signatures (``CUDNN_FRONTEND_LAUNCH_PLANS_MAX``).

``CUDNN_FRONTEND_LAUNCH_PLANS=0`` (read at import) turns plans off.
"""

import atexit
import collections
import functools
import inspect
import json
import logging
import os
import threading
import types
import weakref
from numbers import Integral, Real

ENABLED = os.environ.get("CUDNN_FRONTEND_LAUNCH_PLANS", "1") != "0"
MAX_PLANS = int(os.environ.get("CUDNN_FRONTEND_LAUNCH_PLANS_MAX", "256"))  # signatures kept per planned function

_state = threading.local()  # .capture: the call being recorded on this thread
_capturing = threading.Lock()  # one recording at a time: it patches the DSL's call paths
_UNPLANNED = object()  # the plan does not apply to this call: run the function
_FACTORIES = {
    "empty",
    "empty_strided",
    "empty_like",
    "new_empty",
    "new_empty_strided",
    "zeros",
    "zeros_like",
    "new_zeros",
    "ones",
    "ones_like",
    "new_ones",
    "full",
    "full_like",
    "new_full",
}
_FILL_VALUES = {"zeros": 0, "zeros_like": 0, "new_zeros": 0, "ones": 1, "ones_like": 1, "new_ones": 1}
_ARENA_ALIGN = 512
_ALIGN_LIMIT = 256  # the largest alignment of an argument tensor a later call must keep
_all_plans = []  # every planned function's plans, released at exit before the extension unloads
_stream_types = None
_Recorder = None  # the TorchDispatchMode recordings run under, made on first use
_watched = weakref.WeakSet()  # tvm-ffi kwargs wrappers over raw functions, which the DSL's call paths do not cover


def launch(fn, *args):
    """``fn(*args)``: launch a compiled kernel (a ``tvm_ffi.Function`` or a CuTeDSL tvm-ffi compiled function).

    While a planned call is recorded, the launch is recorded with it."""
    capture = getattr(_state, "capture", None)
    if capture is None:
        return fn(*args)
    return capture.launch(fn, args)


def watch(wrapper):
    """Let recordings see direct calls of ``wrapper``, a tvm-ffi kwargs wrapper over a raw ``tvm_ffi.Function``.

    A raw function is called below every Python call path a recording can patch, so a kernel reloaded
    as one (``cudnn.frost.compiled_cache``) and called other than through :func:`launch` would be missed;
    a watched wrapper's call instead declines the recording."""
    _watched.add(wrapper)
    return wrapper


def planned(fn=None, *, env=(), stream=None):
    """``fn`` with its calls run by a plan once its second call for the same arguments recorded one.

    ``env``: names of environment variables ``fn`` reads per call; their values join the key.
    ``stream``: the name of ``fn``'s parameter that carries the launch stream when it may be a plain integer
    handle (a ``CUstream`` or an object with ``__cuda_stream__`` is recognized in any argument)."""
    if fn is None:
        return lambda f: planned(f, env=env, stream=stream)
    if not ENABLED:
        return fn
    names = list(inspect.signature(fn).parameters)
    stream = (names.index(stream) if stream in names else -1, stream)
    plans = collections.OrderedDict()  # key -> plan, or the number of calls that planned nothing; least recent first
    _all_plans.append(plans)

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            key = _key(args, kwargs, env, stream)
            plan = plans.get(key, 0)
        except TypeError:  # an argument the key cannot hold
            return fn(*args, **kwargs)
        if type(plan) is not int:
            try:
                plans.move_to_end(key)
            except KeyError:  # another thread dropped it
                pass
            result = plan(args, kwargs)
            return fn(*args, **kwargs) if result is _UNPLANNED else result
        if plan != 1 or getattr(_state, "capture", None) is not None:  # first call, gave up, or inside a recording
            if plan == 0:
                plans[key] = 1
                while len(plans) > MAX_PLANS:
                    plans.popitem(last=False)
            return fn(*args, **kwargs)
        result, plans[key] = _record(fn, args, kwargs, stream)
        return result

    run.plans = plans
    return run


def _key(args, kwargs, env, stream):
    import torch

    tensor = torch.Tensor
    device = torch._C._cuda_getDevice()

    def launch_stream(v):
        # whether it is torch's current stream decides which allocations a recording can place on it
        return _Stream, _stream_handle(v) == torch._C._cuda_getCurrentRawStream(device)

    def item(v):
        if isinstance(v, tensor):
            return (v.dtype, v.shape, v.stride(), v.device)
        if _stream_like(v):
            return launch_stream(v)
        if isinstance(v, (list, tuple)):
            if any(isinstance(x, tensor) for x in _flatten(v)):
                raise TypeError("a tensor inside a sequence")
            return tuple(item(x) for x in v)
        hash(v)
        return v

    position, name = stream
    keyed = (
        tuple(launch_stream(v) if i == position and v is not None else item(v) for i, v in enumerate(args)),
        tuple((k, launch_stream(v) if k == name and v is not None else item(v)) for k, v in kwargs.items()),
        device,
    )
    return keyed + tuple(os.environ.get(name) for name in env) if env else keyed


class _Stream:
    """The key of a stream argument: the stream is the call's, not part of the signature."""


def _stream_like(v):
    global _stream_types
    if _stream_types is None:
        from cuda.bindings import driver as cuda

        _stream_types = cuda.CUstream
    return isinstance(v, _stream_types) or hasattr(type(v), "__cuda_stream__")


def _stream_handle(v):
    return v.__cuda_stream__()[1] if hasattr(type(v), "__cuda_stream__") else int(v)


def _arguments(args, kwargs):
    """(accessor, value) for each argument; an accessor is a position or a keyword."""
    return [(i, v) for i, v in enumerate(args)] + list(kwargs.items())


def _get(args, kwargs, accessor):
    return args[accessor] if type(accessor) is int else kwargs[accessor]


def _record(fn, args, kwargs, stream):
    """Run ``fn`` recording it: (its result, the plan or 2 when there is none)."""
    with _capturing:
        try:
            capture = _Capture(args, kwargs, stream)
        except _Decline as e:
            logging.getLogger(__name__).info("launch plan: %s keeps running in Python: %s", getattr(fn, "__qualname__", fn), e)
            return fn(*args, **kwargs), 2
        _state.capture = capture
        try:
            with capture.recorder(), capture.strays():
                result = fn(*args, **kwargs)
        finally:
            _state.capture = None
    try:
        plan = capture.plan(result)
    except _Decline as e:
        logging.getLogger(__name__).info("launch plan: %s keeps running in Python: %s", getattr(fn, "__qualname__", fn), e)
        plan = None
    return result, plan if plan is not None else 2


class _Decline(Exception):
    pass


class _Allocation:
    def __init__(self, tensor, stream, index):
        self.tensor = tensor  # held for the recording, so no other allocation takes its address
        self.storage = tensor.untyped_storage()._cdata
        self.start = tensor.untyped_storage().data_ptr()
        self.bytes = tensor.untyped_storage().nbytes()
        self.layout = (tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.storage_offset())
        self.stream = stream
        self.index = index


class _Capture:
    """One planned call, recorded."""

    def __init__(self, args, kwargs, stream):
        import torch

        from cudnn._torch_stream import _raw_current_stream

        self.torch = torch
        self.raw_stream = _raw_current_stream
        self.inputs = [(a, v) for a, v in _arguments(args, kwargs) if isinstance(v, torch.Tensor) and v.is_cuda]
        if not self.inputs:
            raise _Decline("no CUDA tensor argument")
        self.device = self.inputs[0][1].device
        streams = [(a, v) for a, v in _arguments(args, kwargs) if v is not None and (a in stream or _stream_like(v))]
        self.stream_arg = streams[0][0] if streams else None
        self.ambient = self.current_stream()
        self.stream = _stream_handle(streams[0][1]) if streams else self.ambient
        self.allocations = []
        self.events = []  # ("fill", tensor, width, value, count) / ("launch", function, entry, positional)
        self.declined = None
        self.launching = False

    def current_stream(self):
        handle = self.raw_stream(self.torch, self.device)
        return self.torch.cuda.current_stream(self.device).cuda_stream if handle is None else handle

    def decline(self, reason):
        if self.declined is None:
            self.declined = reason

    def launch(self, fn, args):
        entry, positional = _positional(fn, args)
        if entry is None:
            self.decline(f"{type(fn).__name__} is no tvm-ffi function")
        else:
            self.events.append(("launch", fn, entry, positional))
        self.launching = True
        try:
            return fn(*args)
        finally:
            self.launching = False

    def recorder(self):
        global _Recorder
        if _Recorder is None:  # one class for every recording: torch keeps per-mode-class state
            from torch.utils._python_dispatch import TorchDispatchMode

            class Recorder(TorchDispatchMode):
                @classmethod
                def _should_skip_dynamo(cls):  # torch would wrap __torch_dispatch__ in dynamo, importing it (~1 s)
                    return False

                def __init__(self, capture):
                    super().__init__()
                    self.capture = capture

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    out = func(*args, **(kwargs or {}))
                    self.capture.observe(func, args, kwargs or {}, out)
                    return out

            _Recorder = Recorder
        return _Recorder(self)

    def observe(self, func, args, kwargs, out):
        torch = self.torch
        name = func.overloadpacket.__name__
        tensors = [t for t in _flatten((args, kwargs, out)) if isinstance(t, torch.Tensor)]
        if not any(t.is_cuda for t in tensors):
            return
        if name in _FACTORIES:
            for t in _flatten(out):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    self.allocations.append(_Allocation(t, self.current_stream(), len(self.allocations)))
                    value = _FILL_VALUES.get(name, args[-1] if name.startswith(("full", "new_full")) else None)
                    if value is not None:
                        self.fill(t, value)
        elif name in ("fill_", "zero_") and func.__name__ != "fill_.Tensor":
            self.fill(args[0], 0 if name == "zero_" else args[1])
        else:
            sources = {t.untyped_storage()._cdata for t in _flatten((args, kwargs)) if isinstance(t, torch.Tensor)}
            results = [t for t in _flatten(out) if isinstance(t, torch.Tensor)]
            if func._schema.is_mutable or not results or any(t.untyped_storage()._cdata not in sources for t in results):
                self.decline(f"torch operation {func} on a CUDA tensor")

    def fill(self, t, value):
        torch = self.torch
        if t.numel() == 0:
            return
        if not t.is_contiguous():
            return self.decline("a fill of a non-contiguous tensor")
        size = t.element_size()
        raw = torch.tensor([value], dtype=t.dtype).view(torch.uint8).tolist()
        if len(set(raw)) == 1:
            width, pattern = 1, raw[0]
        elif size in (2, 4):
            width, pattern = size, int.from_bytes(bytes(raw), "little")
        elif size == 8 and raw[:4] == raw[4:]:
            width, pattern = 4, int.from_bytes(bytes(raw[:4]), "little")
        else:
            return self.decline(f"a fill of {t.dtype} with {value}")
        self.events.append(("fill", t, width, pattern, t.numel() * size // width))

    def strays(self):
        """Patch the DSL's call paths so a kernel launched other than through launch() declines the plan."""
        import contextlib

        capture = self

        @contextlib.contextmanager
        def patched():
            classes = []
            try:
                from cutlass.base_dsl.jit_executor import JitCompiledFunction
                from cutlass.cutlass_dsl import tvm_ffi_provider

                classes = [JitCompiledFunction] + [getattr(tvm_ffi_provider, n) for n in dir(tvm_ffi_provider) if "JitCompiledFunction" in n]
            except ImportError:
                pass
            saved = []
            guarded = []
            for wrapper in list(_watched):
                scope = wrapper.__globals__
                guarded.append((scope, scope["__i_target_func"]))
                scope["__i_target_func"] = _Guarded(scope["__i_target_func"], capture)
            for cls in classes:
                if isinstance(cls, type) and "__call__" in cls.__dict__:
                    original = cls.__dict__["__call__"]

                    def call(self, *a, original_call=original, **k):
                        if getattr(_state, "capture", None) is capture and not capture.launching:
                            capture.decline(f"{type(self).__name__} launched other than through launch_plan.launch")
                        return original_call(self, *a, **k)

                    saved.append((cls, original))
                    cls.__call__ = call
            try:
                yield
            finally:
                for cls, original in saved:
                    cls.__call__ = original
                for scope, target in guarded:
                    scope["__i_target_func"] = target

        return patched()

    def plan(self, result):
        """The plan for the recorded call, or None; raises _Decline with the reason."""
        from torch.multiprocessing.reductions import StorageWeakRef

        torch = self.torch
        if self.declined is not None:
            raise _Decline(self.declined)
        if not any(e[0] == "launch" for e in self.events):
            raise _Decline("it launched no kernel")
        inputs = [(a, t, t.data_ptr(), t.data_ptr() + _extent(t)) for a, t in self.inputs]
        regions = [(start, end, ("input", i)) for i, (_, _, start, end) in enumerate(inputs) if end > start]
        regions += [(a.start, a.start + a.bytes, ("allocation", a.index)) for a in self.allocations if a.bytes]

        def place(address, size=1):
            for start, end, where in regions:
                if start <= address and address + size <= end:
                    return where, address - start
            return None, address

        held = {}  # id -> object kept alive as a constant
        steps = [self.step(event, place, held) for event in self.events]
        storages = {a.storage: a for a in self.allocations}
        returned = {}
        for t in _flatten(result):
            if isinstance(t, torch.Tensor) and t.untyped_storage()._cdata in storages:
                returned[t.untyped_storage()._cdata] = storages[t.untyped_storage()._cdata]
        weak = [(a, StorageWeakRef(a.tensor.untyped_storage())) for a in self.allocations]
        for a in self.allocations:
            a.tensor = None
        self.events = None
        kept = {}  # allocation index -> storage the API keeps across calls
        for a, ref in weak:
            if a.storage not in returned and not ref.expired():
                kept[a.index] = torch.UntypedStorage._new_with_weak_ptr(ref.cdata)
        return _Plan(self, inputs, returned, kept, steps, held, result)

    def step(self, event, place, held):
        if event[0] == "fill":
            _, t, width, pattern, count = event
            where, offset = place(t.data_ptr(), t.numel() * t.element_size())
            if where is None:
                held[id(t)] = t
            return ("fill", where, offset, width, pattern, count)
        _, fn, entry, positional = event
        held[id(fn)], held[id(entry)] = fn, entry
        return ("launch", entry, [self.argument(v, place, held) for v in positional])

    def argument(self, v, place, held):
        """How one kernel argument follows from the call (see ``_LaunchPlan``)."""
        import torch
        import tvm_ffi
        from cuda.bindings import driver as cuda

        cls = type(v)
        if v is None:
            return None
        if isinstance(v, bool):
            return {"bool": v}
        if hasattr(cls, "__cuda_stream__") or isinstance(v, cuda.CUstream):
            if _stream_handle(v) != self.stream:
                raise _Decline("a kernel runs on a stream that is not the call's")
            return {"stream": True}
        if isinstance(v, Integral):
            where, offset = place(int(v))
            return {"int": int(v)} if where is None else {"where": where, "offset": offset, "as": "int"}
        if isinstance(v, Real):
            return {"float": float(v)}
        if isinstance(v, tvm_ffi.Tensor) or hasattr(cls, "__tvm_ffi_object__"):
            obj = v if isinstance(v, tvm_ffi.Tensor) else v.__tvm_ffi_object__()
            if not isinstance(obj, tvm_ffi.Tensor):
                raise _Decline(f"a kernel argument {cls.__name__}")
            return self.tensor(obj, place, held, v)
        if isinstance(v, torch.Tensor) or hasattr(cls, "__dlpack_c_exchange_api__"):
            return self.tensor(tvm_ffi.from_dlpack(v), place, held, v)
        if hasattr(cls, "__tvm_ffi_opaque_ptr__"):
            address = int(v.__tvm_ffi_opaque_ptr__())
            where, offset = place(address)
            return {"pointer": address} if where is None else {"where": where, "offset": offset, "as": "pointer"}
        if hasattr(cls, "__tvm_ffi_int__"):
            return {"int": int(v.__tvm_ffi_int__())}
        if hasattr(cls, "__tvm_ffi_float__"):
            return {"float": float(v.__tvm_ffi_float__())}
        raise _Decline(f"a kernel argument {cls.__name__}")

    def tensor(self, t, place, held, original):
        if t.device.dlpack_device_type() != 2:
            raise _Decline("a kernel argument on the host")
        d = t.dtype
        layout = {"as": "tensor", "dim": list(t.shape), "stride": list(t.strides), "dtype": [int(d.type_code), int(d.bits), int(d.lanes)]}
        if 0 in tuple(t.shape) or t.data_ptr() == 0:
            return {"address": 0, **layout}
        size = (1 + sum((n - 1) * s for n, s in zip(t.shape, t.strides))) * ((d.bits * d.lanes + 7) // 8)
        where, offset = place(t.data_ptr(), size)
        if where is None:
            held[id(original)] = original
            return {"address": t.data_ptr(), **layout}
        return {"where": where, "offset": offset, **layout}


class _Guarded:
    """A watched wrapper's target while a recording runs: declines it unless called through launch()."""

    def __init__(self, target, capture):
        self.target, self.capture = target, capture

    def __call__(self, *args):
        if getattr(_state, "capture", None) is self.capture and not self.capture.launching:
            self.capture.decline("a tvm-ffi function launched other than through launch_plan.launch")
        return self.target(*args)


class _Plan:
    """A recorded call's fills and launches, executed by ``_LaunchPlan``."""

    def __init__(self, capture, inputs, returned, kept, steps, held, result):
        import torch

        from cudnn import _compiled_module

        self.torch, self.raw_stream, self.device = torch, capture.raw_stream, capture.device
        self.stream_arg = capture.stream_arg
        self.inputs = tuple(a for a, *_ in inputs)
        align = [min(_ALIGN_LIMIT, start & -start) if start else 1 for _, _, start, _ in inputs]
        extent = [end - start for _, _, start, end in inputs]
        overlaps = [  # (i, j, byte offset of i from j): a later call must place overlapping tensors exactly the same
            [i, j, inputs[i][2] - inputs[j][2]]
            for i in range(len(inputs))
            for j in range(i)
            if extent[i] and extent[j] and inputs[i][2] < inputs[j][3] and inputs[j][2] < inputs[i][3]
        ]
        slots = {("input", i): i for i in range(len(inputs))}
        on_launch_stream = lambda a: a.stream != capture.ambient and a.stream == capture.stream  # noqa: E731
        if any(a.stream not in (capture.ambient, capture.stream) for a in capture.allocations):
            raise _Decline("an allocation on a stream that is neither the current one nor the call's")
        self.outputs = []  # (shape, stride, dtype, on the launch stream)
        for a in returned.values():
            slots[("allocation", a.index)] = len(align)
            align.append(1)
            shape, stride, dtype, offset = a.layout
            if offset:
                raise _Decline("a returned tensor is not at the start of its allocation")
            self.outputs.append((shape, stride, dtype, on_launch_stream(a)))
        self.arenas = []  # (bytes, on the launch stream)
        placed = {}  # allocation index -> (slot, offset) in an arena
        for launch_stream in (False, True):
            size = 0
            for a in capture.allocations:
                if a.index in kept or a.storage in returned or on_launch_stream(a) != launch_stream:
                    continue
                placed[a.index] = (len(align), size)
                size += -(-a.bytes // _ARENA_ALIGN) * _ARENA_ALIGN
            if size:
                self.arenas.append((size, launch_stream))
                align.append(1)
        self.kept = kept

        def locate(where, offset):
            if where is None:
                return {"slot": -1, "offset": offset}
            if where in slots:
                return {"slot": slots[where], "offset": offset}
            if where[1] in kept:
                return {"slot": -1, "offset": kept[where[1]].data_ptr() + offset}
            slot, base = placed[where[1]]
            return {"slot": slot, "offset": base + offset}

        recorded = []
        for step in steps:
            if step[0] == "fill":
                _, where, offset, width, pattern, count = step
                recorded.append({"memset": {**locate(where, offset), "width": width, "value": pattern, "count": count}})
            else:
                _, entry, args = step
                out = []
                for a in args:
                    if isinstance(a, dict) and "where" in a:
                        a = dict(a)
                        a.update(locate(tuple(a.pop("where")), a.pop("offset")))
                    elif isinstance(a, dict) and "address" in a:
                        a = dict(a)
                        a.update({"slot": -1, "offset": a.pop("address")})
                    out.append(a)
                recorded.append({"function": entry.__chandle__(), "args": out})
        self.held = held
        self.op = _compiled_module._LaunchPlan(
            json.dumps({"device": self.device.index or 0, "align": align, "extent": extent, "overlaps": overlaps, "steps": recorded})
        )
        storages = {s: (i, a.layout[2]) for i, (s, a) in enumerate(returned.items())}
        self.result = _template(result, capture.inputs, storages, torch)

    def __call__(self, args, kwargs):
        torch, device = self.torch, self.device
        if self.stream_arg is not None:
            stream = _stream_handle(_get(args, kwargs, self.stream_arg))
        else:
            handle = self.raw_stream(torch, device)
            stream = torch.cuda.current_stream(device).cuda_stream if handle is None else handle
        buffers = self.allocate(stream)
        addresses = [_get(args, kwargs, a).data_ptr() for a in self.inputs] + [b.data_ptr() for b in buffers]
        if not self.op.execute(addresses, stream):
            return _UNPLANNED
        return _build(self.result, args, kwargs, buffers)

    def allocate(self, stream):
        """The returned tensors and the arenas, each on the stream the recorded call allocated it on."""
        torch, device = self.torch, self.device
        side = None
        if any(on for *_, on in self.outputs) or any(on for _, on in self.arenas):
            handle = self.raw_stream(torch, device)
            if handle is not None and handle != stream:
                side = torch.cuda.ExternalStream(stream, device=device)
        buffers = []
        for shape, stride, dtype, on in self.outputs:
            if on and side is not None:
                with torch.cuda.stream(side):
                    buffers.append(torch.empty_strided(shape, stride, dtype=dtype, device=device))
            else:
                buffers.append(torch.empty_strided(shape, stride, dtype=dtype, device=device))
        for size, on in self.arenas:
            if on and side is not None:
                with torch.cuda.stream(side):
                    buffers.append(torch.empty(size, dtype=torch.uint8, device=device))
            else:
                buffers.append(torch.empty(size, dtype=torch.uint8, device=device))
        return buffers


def _positional(fn, args):
    """(the tvm-ffi function, the positional arguments it receives) for ``fn(*args)``, or (None, args)."""
    import tvm_ffi

    execution = getattr(fn, "execution_args", None)
    if execution is not None and execution.has_pointer_address_arg_specs:
        args = execution.convert_python_pointer_args_for_tvm_ffi(args, {})[0]
    wrapper = getattr(fn, "_kwargs_wrapper", None) or (fn if "__i_target_func" in getattr(fn, "__globals__", {}) else None)
    if wrapper is not None:  # a tvm-ffi kwargs wrapper: run its code against a recorder to learn the positional call
        target = wrapper.__globals__["__i_target_func"]
        target = target.target if isinstance(target, _Guarded) else target
        got = []
        scope = dict(wrapper.__globals__, __i_target_func=lambda *a: got.append(a))
        rerun = types.FunctionType(wrapper.__code__, scope, wrapper.__name__, wrapper.__defaults__, wrapper.__closure__)
        rerun.__kwdefaults__ = wrapper.__kwdefaults__
        rerun(*args)
        return (target, got[0]) if isinstance(target, tvm_ffi.Function) else (None, args)
    if isinstance(fn, tvm_ffi.Function):
        return fn, args
    return None, args


def _flatten(value):
    if isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    else:
        yield value


def _template(value, inputs, storages, torch):
    """How to rebuild the recorded call's result from a later call's arguments and buffers."""
    if isinstance(value, torch.Tensor):
        for accessor, t in inputs:
            if t is value:
                return ("input", accessor)
        storage = value.untyped_storage()._cdata
        if storage not in storages:
            raise _Decline("it returns a tensor it did not allocate and was not passed")
        index, dtype = storages[storage]
        if value.dtype != dtype:
            raise _Decline("it returns a tensor viewed as another dtype")
        return ("output", index, tuple(value.shape), value.stride(), value.storage_offset())
    if isinstance(value, (list, tuple)):
        return (type(value), [_template(v, inputs, storages, torch) for v in value])
    if isinstance(value, dict):
        return (type(value), {k: _template(v, inputs, storages, torch) for k, v in value.items()})
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    raise _Decline(f"it returns a {type(value).__name__}")


def _build(template, args, kwargs, buffers):
    kind = template[0]
    if kind == "output":
        _, i, shape, stride, offset = template
        t = buffers[i]
        return t if offset == 0 and t.shape == shape and t.stride() == stride else t.as_strided(shape, stride, offset)
    if kind == "input":
        return _get(args, kwargs, template[1])
    if kind == "value":
        return template[1]
    if issubclass(kind, dict):
        return kind({k: _build(v, args, kwargs, buffers) for k, v in template[1].items()})
    return kind(_build(v, args, kwargs, buffers) for v in template[1])


def _extent(t):
    """Bytes from ``t``'s first element to just past its last."""
    if t.numel() == 0:
        return 0
    return (1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))) * t.element_size()


@atexit.register
def _release():
    for plans in _all_plans:
        plans.clear()
