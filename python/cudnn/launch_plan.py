# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch plans: a planned function's repeated calls launch its kernels from C++ and skip its Python.

:func:`planned` wraps an API function whose GPU work is kernel launches made
through :func:`launch`, plus torch allocations and fills. Calls are keyed by
their arguments (each tensor's dtype, shape, strides and device, and the other
arguments' values and types). A signature runs as usual until it has been called
``HOT_CALLS`` times (the first half of them counted by the first argument's shape
alone, which is all a shape that never repeats costs); that call runs as usual
too and records what it does: each
:func:`launch` (the compiled kernel and its positional arguments), the tensors it
allocates and fills, and what it returns. The recording becomes a plan when
every kernel argument follows from the call: an argument tensor or a view into
one, a tensor the call allocates (one it returns, or scratch it drops), the
stream, or a constant value. Every later call allocates the returned tensors and
the scratch and hands their addresses and the stream to ``_LaunchPlan``, which
fills and launches in C++ in the recorded order; the function's Python does not
run.

A call the plan could not reproduce is not planned: a host tensor argument (its
values could decide the launches); a torch operation on a CUDA tensor other than
an allocation, a fill or a view; a CuTeDSL kernel launched other than through
:func:`launch`; a kernel argument that is none of the above (a tensor or address
the call was neither passed nor allocated, such as a buffer the API keeps); a
tensor the call allocates and keeps; or an allocation on a stream that is neither
the current one nor a stream argument. Kernels launched through a raw
``tvm_ffi.Function`` other than through :func:`launch` cannot be seen, so a
planned function must launch every kernel through :func:`launch`, and whatever
it does must follow from its arguments (or the environment variables it names).

A later call runs the function, not the plan, whenever the function could have
behaved differently: its argument tensors are less aligned than the recorded
call's (up to 256 bytes), or they overlap each other differently (overlapping
tensors must sit at the same byte offset from each other as in the recorded call). The key also
holds the current CUDA device and the environment variables the function names
(``planned(env=...)``). Each planned function keeps its 256 most recently used
signatures (``CUDNN_FRONTEND_LAUNCH_PLANS_MAX``).

``CUDNN_FRONTEND_LAUNCH_PLANS`` (read at import): ``0`` turns plans off;
``verify`` checks every planned call instead of speeding it up (the call runs,
then is recorded and replayed on copies of its arguments, and a plan that does
not reproduce the call's results bit for bit raises).
"""

import atexit
import collections
import contextlib
import functools
import inspect
import json
import logging
import os
import sys
import threading
import types
import weakref
from numbers import Integral, Real
from operator import itemgetter

MODE = os.environ.get("CUDNN_FRONTEND_LAUNCH_PLANS", "1")
ENABLED = MODE != "0"
VERIFY = MODE == "verify"
MAX_PLANS = int(os.environ.get("CUDNN_FRONTEND_LAUNCH_PLANS_MAX", "256"))  # signatures kept per planned function
HOT_CALLS = 16  # the call of a signature that is recorded: a recording costs about what this many planned calls save.
# Its first half is counted by the first argument's shape alone, so shapes that keep changing do not pay for a key.
VERIFIED = collections.Counter()  # verify mode: calls whose plan matched, and why calls were not planned

_state = threading.local()  # .capture: the call being recorded on this thread; .verifying: a call being verified
_capturing = threading.Lock()  # one recording at a time: it patches the DSL's call paths
_UNPLANNED = object()  # the plan does not apply to this call: run the function
_DECLINED = -1  # a signature whose recording gave no plan
_MISSING = object()  # a parameter the call leaves out and that has no default
_VALUES = frozenset({bool, int, float, str, type(None)})  # argument types keyed by value
_TENSORS = set()  # torch.Tensor and torch.nn.Parameter, once torch is loaded
_EMPTY = {"empty", "empty_strided", "empty_like", "new_empty", "new_empty_strided"}
_FILL_VALUES = {"zeros": 0, "zeros_like": 0, "new_zeros": 0, "ones": 1, "ones_like": 1, "new_ones": 1}
_FACTORIES = _EMPTY | set(_FILL_VALUES) | {"full", "full_like", "new_full"}
_ARENA_ALIGN = 512
_ALIGN_LIMIT = 256  # the largest alignment of an argument tensor a later call must keep
_all_plans = []  # every planned function's plans, released at exit before the extension unloads
_torch = None  # torch, imported on first use
_raw_current_stream = None  # cudnn._torch_stream._raw_current_stream
_stream_types = None
_Recorder = None  # the TorchDispatchMode recordings run under, made on first use
_Zeroing = None  # the TorchDispatchMode that zero-fills new tensors (verify mode)
_holder = None  # the tvm-ffi function that hands a plan's kernels to C++; False where plans cannot run
_watched = weakref.WeakSet()  # tvm-ffi kwargs wrappers over raw functions, which the DSL's call paths do not cover
log = logging.getLogger(__name__)


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
    """``fn`` with its calls run by a plan once a signature's ``HOT_CALLS``-th call recorded one.

    ``env``: names of environment variables ``fn`` reads per call; their values join the key.
    ``stream``: the name of ``fn``'s parameter that carries the launch stream when it may be a plain integer
    handle (a ``CUstream`` or an object with ``__cuda_stream__`` is recognized in any argument)."""
    if fn is None:
        return lambda f: planned(f, env=env, stream=stream)
    parameters = list(inspect.signature(fn).parameters.values())
    if not ENABLED or any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
        return fn
    bind = _binder(parameters)
    stream = next((i for i, p in enumerate(parameters) if p.name == stream), None)
    if VERIFY:
        return _verified(fn, bind, env, stream)
    plans = collections.OrderedDict()  # key -> plan, calls seen so far, or _DECLINED; least recent first
    _all_plans.append(plans)
    first = parameters[0].name if parameters else None
    warm = {}  # the first argument's shape -> calls seen with it, up to half of HOT_CALLS

    @functools.wraps(fn)
    def run(*args, **kwargs):
        # A shape seen a few times only is counted and nothing else: the key costs more than a call that never repeats can save.
        probe = args[0] if args else kwargs.get(first)
        hot = HOT_CALLS
        if type(probe) in (_TENSORS or _tensor_types()):
            shape = probe.shape
            seen = warm.get(shape, 0)
            if seen < hot // 2:
                if seen == 0 and len(warm) >= 4 * MAX_PLANS:
                    warm.clear()
                warm[shape] = seen + 1
                return fn(*args, **kwargs)
            hot -= hot // 2
        try:
            values = bind(args, kwargs)
            key = _key(values, env, stream)
            state = plans.get(key, 0)
        except (TypeError, KeyError):  # a call fn itself rejects, or an argument the key cannot hold
            return fn(*args, **kwargs)
        if type(state) is not int:
            try:
                plans.move_to_end(key)
            except KeyError:  # another thread dropped it
                pass
            result = state(values)
            return fn(*args, **kwargs) if result is _UNPLANNED else result
        if state < 0 or state + 1 < hot or getattr(_state, "capture", None) is not None:  # not hot, gave up, or inside a recording
            if state >= 0:
                plans[key] = state + 1
                if state == 0:
                    while len(plans) > MAX_PLANS:
                        plans.popitem(last=False)
            return fn(*args, **kwargs)
        result, plan = _record(fn, args, kwargs, values, stream)
        plans[key] = _DECLINED if plan is None else plan
        while len(plans) > MAX_PLANS:
            plans.popitem(last=False)
        return result

    run.plans = plans
    return run


def _binder(parameters):
    """(args, kwargs) -> one value per parameter, however it was passed (a parameter left out has its default): calls that
    mean the same key the same. Raises TypeError or KeyError for a call the function rejects."""
    defaults = [_MISSING if p.default is p.empty else p.default for p in parameters]
    positional = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
    index = {p.name: i for i, p in enumerate(parameters) if p.kind is not p.POSITIONAL_ONLY}

    def bind(args, kwargs):
        given = len(args)
        if given > positional:
            raise TypeError("too many positional arguments")
        values = defaults.copy()
        values[:given] = args
        for name, value in kwargs.items():
            i = index[name]
            if i < given:
                raise TypeError("an argument passed twice")
            values[i] = value
        return values

    return bind


def _load():
    global _torch, _raw_current_stream
    import torch

    from cudnn._torch_stream import _raw_current_stream

    _torch = torch
    _TENSORS.update((torch.Tensor, torch.nn.Parameter))
    return torch


def _tensor_types():
    _load()
    return _TENSORS


def _key(values, env, stream):
    torch = _torch or _load()
    items = [v if type(v) in _VALUES else _item(v) for v in values]
    if stream is not None and values[stream] is not None and values[stream] is not _MISSING:
        items[stream] = _launch_stream(values[stream])
    # equal values of different types (0, False, 0.0) are different arguments
    key = (tuple(items), tuple(map(type, values)), torch.cuda.current_device())
    return key + tuple(map(os.environ.get, env)) if env else key


def _item(v):
    if isinstance(v, _torch.Tensor):
        return (v.dtype, v.shape, v.stride(), v.device)
    if _stream_like(v):
        return _launch_stream(v)
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, _torch.Tensor) for x in _flatten(v)):
            raise TypeError("a tensor inside a sequence")
        return tuple((type(x), _item(x)) for x in v)
    hash(v)
    return v


def _launch_stream(v):
    # whether it is torch's current stream decides which allocations a recording can place on it
    return _Stream, _stream_handle(v) == _current_stream(_torch.cuda.current_device())


def _current_stream(device):
    handle = _raw_current_stream(_torch, device)
    return _torch.cuda.current_stream(device).cuda_stream if handle is None else handle


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


def _record(fn, args, kwargs, values, stream):
    """Run ``fn(*args, **kwargs)`` recording it: (its result, the plan or None when there is none)."""
    name = getattr(fn, "__qualname__", fn)
    capture = None
    with _capturing:
        try:
            _hold()
            capture = _Capture(values, stream)
            recorder, strays = capture.recorder(), capture.strays()
        except Exception as e:
            _declined(name, e)
            capture = None
        else:
            _state.capture = capture
            try:
                with recorder, strays:
                    result = fn(*args, **kwargs)
            finally:
                _state.capture = None
    if capture is None:
        return fn(*args, **kwargs), None
    try:
        return result, capture.plan(result)
    except Exception as e:
        _declined(name, e)
        return result, None


def _declined(name, error):
    """A call that gets no plan keeps running the function; anything but a _Decline is a fault of this module's."""
    if isinstance(error, _Decline):
        log.info("launch plan: %s keeps running in Python: %s", name, error)
    else:
        log.warning("launch plan: %s keeps running in Python: %s: %s", name, type(error).__name__, error)
    if VERIFY:
        VERIFIED[f"{name} not planned: {error if isinstance(error, _Decline) else type(error).__name__}"] += 1


def _hold():
    """The tvm-ffi function that hands a plan's kernels to C++ (``cudnn.launch_plan.hold``)."""
    global _holder
    if _holder is None:
        if "tvm_ffi" not in sys.modules:
            raise _Decline("tvm-ffi is not loaded")
        try:
            import tvm_ffi

            from cudnn import _compiled_module

            _compiled_module._launch_plan_runtime()
            _holder = tvm_ffi.get_global_func("cudnn.launch_plan.hold")
        except Exception as e:
            _holder = False
            log.warning("launch plans are off: %s", e)
    if not _holder:
        raise _Decline("launch plans cannot run here")
    return _holder


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

    def __init__(self, values, stream):
        torch = self.torch = _torch or _load()
        self.inputs = [(i, v) for i, v in enumerate(values) if isinstance(v, torch.Tensor)]
        if not self.inputs:
            raise _Decline("no tensor argument")
        if not all(v.is_cuda for _, v in self.inputs):
            raise _Decline("a host tensor argument, whose values the function could read unseen")
        self.device = self.inputs[0][1].device
        if any(v.device != self.device for _, v in self.inputs):
            raise _Decline("tensor arguments on more than one device")
        streams = [(i, v) for i, v in enumerate(values) if v is not None and v is not _MISSING and (i == stream or _stream_like(v))]
        self.stream_arg = streams[0][0] if streams else None
        self.ambient = _current_stream(self.device)
        self.stream = _stream_handle(streams[0][1]) if streams else self.ambient
        self.allocations = []
        self.events = []  # ("fill", tensor, width, value, count) / ("launch", function, entry, positional)
        self.declined = None
        self.launching = False

    def decline(self, reason):
        if self.declined is None:
            self.declined = reason

    def launch(self, fn, args):
        try:
            entry, positional = _positional(fn, args)
        except Exception as e:  # a call path of the DSL's this module does not know
            entry, positional = None, e
        if entry is None:
            self.decline(f"{type(fn).__name__} is no tvm-ffi function this module can call")
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
                    try:
                        self.capture.observe(func, args, kwargs or {}, out)
                    except Exception as e:
                        self.capture.decline(f"torch operation {func} could not be recorded ({type(e).__name__})")
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
                    self.allocations.append(_Allocation(t, _current_stream(self.device), len(self.allocations)))
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
        classes = _dsl_classes()
        capture = self

        @contextlib.contextmanager
        def patched():
            saved = []
            guarded = []
            for wrapper in list(_watched):
                scope = wrapper.__globals__
                guarded.append((scope, scope["__i_target_func"]))
                scope["__i_target_func"] = _Guarded(scope["__i_target_func"], capture)
            for cls in classes:
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
        """The plan for the recorded call; raises _Decline with the reason there is none."""
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

        held = {}  # id -> compiled function kept alive for the plan
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
        if any(a.storage not in returned and not ref.expired() for a, ref in weak):
            raise _Decline("it keeps a tensor it allocated")
        return _Plan(self, inputs, returned, steps, held, result)

    def step(self, event, place, held):
        if event[0] == "fill":
            _, t, width, pattern, count = event
            where, offset = place(t.data_ptr(), t.numel() * t.element_size())
            if where is None:
                raise _Decline("a fill of a tensor the call was neither passed nor allocated")
            return ("fill", where, offset, width, pattern, count)
        _, fn, entry, positional = event
        held[id(fn)] = fn
        return ("launch", entry, [self.argument(v, place) for v in positional])

    def argument(self, v, place):
        """How one kernel argument follows from the call (see ``_LaunchPlan``)."""
        import tvm_ffi

        torch = self.torch
        cls = type(v)
        if v is None:
            return None
        if isinstance(v, bool):
            return {"bool": v}
        if _stream_like(v):
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
            return self.tensor(obj, place)
        if isinstance(v, torch.Tensor) or hasattr(cls, "__dlpack_c_exchange_api__"):
            return self.tensor(tvm_ffi.from_dlpack(v), place)
        if hasattr(cls, "__tvm_ffi_opaque_ptr__"):
            address = int(v.__tvm_ffi_opaque_ptr__())
            where, offset = place(address) if address else (-1, 0)
            if where is None:
                raise _Decline("a kernel argument pointer the call was neither passed nor allocated")
            return {"where": where, "offset": offset, "as": "pointer"}
        if hasattr(cls, "__tvm_ffi_int__"):
            return {"int": int(v.__tvm_ffi_int__())}
        if hasattr(cls, "__tvm_ffi_float__"):
            return {"float": float(v.__tvm_ffi_float__())}
        raise _Decline(f"a kernel argument {cls.__name__}")

    def tensor(self, t, place):
        if t.device.dlpack_device_type() != 2:
            raise _Decline("a kernel argument on the host")
        d = t.dtype
        layout = {"as": "tensor", "dim": list(t.shape), "stride": list(t.strides), "dtype": [int(d.type_code), int(d.bits), int(d.lanes)]}
        if 0 in tuple(t.shape) or t.data_ptr() == 0:
            return {"where": -1, "offset": 0, **layout}
        size = (1 + sum((n - 1) * s for n, s in zip(t.shape, t.strides))) * ((d.bits * d.lanes + 7) // 8)
        where, offset = place(t.data_ptr(), size)
        if where is None:
            raise _Decline("a kernel argument tensor the call was neither passed nor allocated")
        return {"where": where, "offset": offset, **layout}


def _dsl_classes():
    """The CuTeDSL classes whose ``__call__`` launches a compiled function; raises _Decline when they are not the ones this module knows."""
    if "cutlass" not in sys.modules:  # no CuTeDSL function exists to be launched
        return []
    try:
        from cutlass.base_dsl.jit_executor import JitCompiledFunction
        from cutlass.cutlass_dsl import tvm_ffi_provider
    except ImportError as e:
        raise _Decline(f"this CuTeDSL's call path is not one launch plans can watch ({e})")
    classes = [JitCompiledFunction] + [getattr(tvm_ffi_provider, n) for n in dir(tvm_ffi_provider) if "JitCompiledFunction" in n]
    classes = [c for c in classes if isinstance(c, type) and "__call__" in c.__dict__]
    if not classes:
        raise _Decline("this CuTeDSL's call path is not one launch plans can watch")
    return classes


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

    def __init__(self, capture, inputs, returned, steps, held, result):
        from cudnn import _compiled_module

        torch = self.torch = capture.torch
        self.device = capture.device
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
        slots = {-1: -1} | {("input", i): i for i in range(len(inputs))}
        on_launch_stream = lambda a: a.stream != capture.ambient and a.stream == capture.stream  # noqa: E731
        if any(a.stream not in (capture.ambient, capture.stream) for a in capture.allocations):
            raise _Decline("an allocation on a stream that is neither the current one nor the call's")
        self.outputs = []  # (shape, stride or None when contiguous, dtype, on the launch stream)
        for a in returned.values():
            slots[("allocation", a.index)] = len(align)
            align.append(1)
            shape, stride, dtype, offset = a.layout
            if offset:
                raise _Decline("a returned tensor is not at the start of its allocation")
            self.outputs.append((shape, None if stride == _dense_stride(shape) else stride, dtype, on_launch_stream(a)))
        self.arenas = []  # (bytes, on the launch stream)
        placed = {}  # allocation index -> (slot, offset) in an arena
        for launch_stream in (False, True):
            size = 0
            for a in capture.allocations:
                if a.storage in returned or on_launch_stream(a) != launch_stream:
                    continue
                placed[a.index] = (len(align), size)
                size += -(-a.bytes // _ARENA_ALIGN) * _ARENA_ALIGN
            if size:
                self.arenas.append((size, launch_stream))
                align.append(1)
        self.side = any(on for *_, on in self.outputs) or any(on for _, on in self.arenas)

        def locate(where, offset):
            if where in slots:
                return {"slot": slots[where], "offset": offset}
            slot, base = placed[where[1]]
            return {"slot": slot, "offset": base + offset}

        functions = []  # the tvm-ffi functions, in the order steps first call them
        recorded = []
        for step in steps:
            if step[0] == "fill":
                _, where, offset, width, pattern, count = step
                recorded.append({"memset": {**locate(where, offset), "width": width, "value": pattern, "count": count}})
                continue
            _, entry, args = step
            index = next((i for i, f in enumerate(functions) if f is entry), len(functions))
            if index == len(functions):
                functions.append(entry)
            out = []
            for a in args:
                if isinstance(a, dict) and "where" in a:
                    a = dict(a)
                    a.update(locate(a.pop("where"), a.pop("offset")))
                out.append(a)
            recorded.append({"function": index, "args": out})
        self.held = held
        text = json.dumps({"device": self.device.index or 0, "align": align, "extent": extent, "overlaps": overlaps, "steps": recorded})
        self.op = _compiled_module._LaunchPlan(text, _hold()(*functions))
        storages = {s: (i, a.layout) for i, (s, a) in enumerate(returned.items())}
        self.build = _builder(_template(result, capture.inputs, storages, torch))

    def __call__(self, values, zero=False):
        device = self.device
        stream = _current_stream(device) if self.stream_arg is None else _stream_handle(values[self.stream_arg])
        buffers = self.allocate(stream)
        if zero:
            for b in buffers:
                b.zero_()
        addresses = [values[i].data_ptr() for i in self.inputs] + [b.data_ptr() for b in buffers]
        if not self.op.execute(addresses, stream):
            return _UNPLANNED
        return self.build(values, buffers)

    def allocate(self, stream):
        """The returned tensors and the arenas, each on the stream the recorded call allocated it on."""
        torch, device = self.torch, self.device
        empty, strided, byte = torch.empty, torch.empty_strided, torch.uint8
        if not self.side or _current_stream(device) == stream:
            buffers = [
                empty(shape, dtype=dtype, device=device) if stride is None else strided(shape, stride, dtype=dtype, device=device)
                for shape, stride, dtype, _ in self.outputs
            ]
            buffers.extend([empty(size, dtype=byte, device=device) for size, _ in self.arenas])
            return buffers
        side = torch.cuda.ExternalStream(stream, device=device)
        buffers = []
        for shape, stride, dtype, on in self.outputs:
            if on:
                with torch.cuda.stream(side):
                    buffers.append(strided(shape, stride or _dense_stride(shape), dtype=dtype, device=device))
            else:
                buffers.append(strided(shape, stride or _dense_stride(shape), dtype=dtype, device=device))
        for size, on in self.arenas:
            if on:
                with torch.cuda.stream(side):
                    buffers.append(empty(size, dtype=byte, device=device))
            else:
                buffers.append(empty(size, dtype=byte, device=device))
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
    """How to rebuild the recorded call's result from a later call's values and buffers."""
    if isinstance(value, torch.Tensor):
        for accessor, t in inputs:
            if t is value:
                return ("input", accessor)
        storage = value.untyped_storage()._cdata
        if storage not in storages:
            raise _Decline("it returns a tensor it did not allocate and was not passed")
        index, (shape, stride, dtype, _) = storages[storage]
        if value.dtype != dtype:
            raise _Decline("it returns a tensor viewed as another dtype")
        if value.storage_offset() == 0 and tuple(value.shape) == shape and value.stride() == stride:
            return ("buffer", index)
        return ("view", index, tuple(value.shape), value.stride(), value.storage_offset())
    if type(value) in (list, tuple):
        return (type(value), [_template(v, inputs, storages, torch) for v in value])
    if type(value) is dict:
        return (dict, {k: _template(v, inputs, storages, torch) for k, v in value.items()})
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    raise _Decline(f"it returns a {type(value).__name__}")


def _builder(template):
    """``template`` as a function of a later call's (values, buffers)."""
    kind = template[0]
    if kind == "buffer":
        return lambda values, buffers, i=template[1]: buffers[i]
    if kind == "view":
        _, i, shape, stride, offset = template
        return lambda values, buffers: buffers[i].as_strided(shape, stride, offset)
    if kind == "input":
        return lambda values, buffers, i=template[1]: values[i]
    if kind == "value":
        return lambda values, buffers, v=template[1]: v
    if kind is dict:
        named = {k: _builder(v) for k, v in template[1].items()}
        return lambda values, buffers: {k: build(values, buffers) for k, build in named.items()}
    if kind is tuple and len(template[1]) > 1 and all(v[0] == "buffer" for v in template[1]):
        pick = itemgetter(*[v[1] for v in template[1]])
        return lambda values, buffers: pick(buffers)
    parts = [_builder(v) for v in template[1]]
    return lambda values, buffers: kind([build(values, buffers) for build in parts])


def _dense_stride(shape):
    """The strides torch gives a contiguous tensor of ``shape``."""
    stride, step = [], 1
    for size in reversed(shape):
        stride.append(step)
        step *= max(size, 1)
    return tuple(reversed(stride))


def _extent(t):
    """Bytes from ``t``'s first element to just past its last."""
    if t.numel() == 0:
        return 0
    return (1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))) * t.element_size()


def _verified(fn, bind, env, stream):
    """``fn`` with every call checked against its plan: the call runs, then is recorded and replayed on copies of its
    arguments; the replay has to return and leave in its arguments, bit for bit, what the call did."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        if getattr(_state, "capture", None) is not None or getattr(_state, "verifying", False):
            return fn(*args, **kwargs)
        if (_torch or _load()).cuda.is_current_stream_capturing():  # a capture takes no readback, and would keep all three runs
            VERIFIED[f"{fn.__qualname__} not checked: inside a CUDA graph capture"] += 1
            return fn(*args, **kwargs)
        try:
            _key(bind(args, kwargs), env, stream)
            recorded, replayed = _twin(args, kwargs), _twin(args, kwargs)
        except (TypeError, KeyError):
            return fn(*args, **kwargs)
        _state.verifying = True
        try:
            with _zeroing():
                result = fn(*args, **kwargs)
                _, plan = _record(fn, *recorded, bind(*recorded), stream)
            if plan is None:
                return result
            again = plan(bind(*replayed), zero=True)
            if again is _UNPLANNED:
                VERIFIED[f"{fn.__qualname__} not planned: the copies are placed differently"] += 1
                return result
            where = _difference(result, again, "result") or _difference((args, kwargs), replayed, "arguments")
            if where is not None:
                raise AssertionError(f"launch plan: the plan of {fn.__qualname__} does not reproduce the call ({where} differs)")
            VERIFIED["matched"] += 1
            return result
        finally:
            _state.verifying = False

    run.plans = collections.OrderedDict()
    return run


def _zeroing():
    """A mode that zero-fills the tensors a call allocates, so what a kernel leaves unwritten compares equal."""
    global _Zeroing
    if _Zeroing is None:
        from torch.utils._python_dispatch import TorchDispatchMode

        class Zeroing(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if func.overloadpacket.__name__ in _EMPTY:
                    for t in _flatten(out):
                        if isinstance(t, _torch.Tensor) and t.is_cuda:
                            t.zero_()
                return out

        _Zeroing = Zeroing
    return _Zeroing()


def _twin(args, kwargs):
    """The arguments with every CUDA tensor on a copy of its storage: same layout, same overlaps, other addresses."""
    torch = _torch
    copies = {}

    def twin(v):
        if not (isinstance(v, torch.Tensor) and v.is_cuda):
            return v
        storage = v.untyped_storage()
        copy = copies.get(storage._cdata)
        if copy is None:
            copy = copies[storage._cdata] = storage.clone()
        return torch.empty(0, dtype=v.dtype, device=v.device).set_(copy, v.storage_offset(), v.shape, v.stride())

    return tuple(twin(v) for v in args), {k: twin(v) for k, v in kwargs.items()}


def _difference(a, b, where):
    """Where ``a`` and ``b`` differ (tensors bit for bit), or None."""
    torch = _torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
            return where
        if a.numel() and not torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)):
            return where
        return None
    if type(a) is not type(b):
        return where
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return where
        return next((d for d in (_difference(x, y, f"{where}[{i}]") for i, (x, y) in enumerate(zip(a, b))) if d is not None), None)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return where
        return next((d for d in (_difference(a[k], b[k], f"{where}[{k!r}]") for k in a) if d is not None), None)
    return None if a == b or _stream_like(a) else where


@atexit.register
def _release():
    for plans in _all_plans:
        plans.clear()
    if VERIFIED:
        print("cudnn launch plans, verify mode:", ", ".join(f"{count} {what}" for what, count in sorted(VERIFIED.items())), file=sys.stderr)
