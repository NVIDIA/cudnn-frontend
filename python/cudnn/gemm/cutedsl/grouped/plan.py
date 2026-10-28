# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drop-in C++ execution for the grouped GEMM wrappers: planned calls run through ``cudnn_frontend::ops::GroupedGemm``.

A wrapper's first call for an operand signature derives, checks and compiles, and
its memo keeps the compiled API. The next call with that signature runs the memo
path once more, checked, but also asks ``execute(..., launch=False)`` for the
launch: the compiled kernel and its positional arguments, built by the same code
that launches. :class:`GroupedGemmPlan` sorts those arguments into the call's
operands (the caller's tensors and the outputs the wrapper allocates), views into
them, the stream, workspace addresses and constants, and refuses anything else.
It registers the kernel with that binding for C++ (``KernelLibrary::registry()``)
and creates the C++ op for the operands. Every later call with the signature
allocates the outputs and hands the operands' addresses and the stream to the C++
op, which launches the kernel; no wrapper or ``execute()`` code runs. The C++ op
also checks pointer alignment, the one per-call fact a signature does not fix: a
call whose tensors are less aligned than the planned call's (up to 16 bytes) runs
the wrapper instead.

``CUDNN_FRONTEND_GROUPED_GEMM_PLANS=0`` (read at import) keeps every call on the wrapper.
"""

import functools
import json
import logging
import os
import threading
from numbers import Integral, Real

from cudnn.api_base import TupleDict

from .backend_utils import _torch_stream_context, allocate_wrapper_workspace

ENABLED = os.environ.get("CUDNN_FRONTEND_GROUPED_GEMM_PLANS", "1") != "0"

_planning = threading.local()
_kernels = {}  # configuration -> (execute, compiled kernel, arguments), alive while C++ may call it


class GroupedGemmPlan:
    """One grouped GEMM call signature, planned from the launch its API states for a checked call.

    :param execute: the API's ``execute``; called once with ``launch=False``.
    :param arguments: what the wrapper passes to ``execute`` for this signature, except outputs and ``current_stream``.
    :param inputs: names in ``arguments`` that are the caller's keyword arguments of the same name.
    :param outputs: the outputs the wrapper allocated, by ``execute`` argument name.
    :param init: initial value of each output the kernel accumulates into; others are uninitialized.
    :param result: the wrapper's result: name -> output or input name, or None.
    :param stream: the call's ``current_stream``.
    :param stream_outputs: outputs the wrapper allocates on the launch stream.
    :param workspace_bytes: scratch passed as ``workspace=``, allocated per call on the launch stream.

    ``op`` is None when the launch has an argument the plan cannot bind.
    """

    def __init__(self, execute, arguments, inputs, outputs, init, result, stream, stream_outputs=(), workspace_bytes=0):
        import torch
        from cuda.bindings import driver as cuda
        from cudnn._torch_stream import _raw_current_stream

        self.device = arguments["a_tensor"].device
        self.inputs = tuple(name for name in inputs if isinstance(arguments.get(name), torch.Tensor))
        outputs = {name: t for name, t in outputs.items() if t is not None}
        self.outputs = tuple((name, tuple(t.shape), t.stride(), t.dtype, init.get(name), name in stream_outputs) for name, t in outputs.items())
        self.result = tuple(result.items())
        self.workspace_bytes = workspace_bytes
        self.raw_stream, self.torch = _raw_current_stream, torch
        self.op = None

        stream = cuda.CUstream(int(stream) if stream is not None else self.current_stream())
        workspace = allocate_wrapper_workspace("torch", workspace_bytes, self.device, stream) if workspace_bytes else None
        extra = {"workspace": workspace} if workspace is not None else {}
        launch = execute(**arguments, **outputs, **extra, current_stream=stream, launch=False)
        if launch is None:
            return
        operands = [(name, arguments[name]) for name in self.inputs] + list(outputs.items())
        base = workspace.data_ptr() if workspace is not None else 0
        compiled, args = launch
        execution = getattr(compiled, "execution_args", None)
        if execution is not None and execution.has_pointer_address_arg_specs:
            args = execution.convert_python_pointer_args_for_tvm_ffi(args, {})[0]
        binding = _bind(args, operands, stream, base, workspace_bytes)
        if binding is not None:
            self.launch = (execute, compiled, args)
            self.op = _create(self.launch, binding, operands, workspace_bytes, self.device)

    def current_stream(self):
        handle = self.raw_stream(self.torch, self.device)
        return self.torch.cuda.current_stream(self.device).cuda_stream if handle is None else handle

    def __call__(self, kwargs):
        """The call's result, or None when it must run the wrapper."""
        stream = kwargs.get("current_stream")
        current = self.current_stream()
        handle = int(stream) if stream is not None else current
        if handle == current:
            outs, workspace = self.allocate(lambda on_stream: True)
        else:  # what the wrapper allocates on the launch stream, there
            with _torch_stream_context(stream, self.device):
                outs, workspace = self.allocate(lambda on_stream: on_stream)
            outs.update(self.allocate(lambda on_stream: not on_stream, workspace=False)[0])
        addresses = [kwargs[name].data_ptr() for name in self.inputs]
        addresses += [outs[name].data_ptr() for name, *_ in self.outputs]
        if not self.op.execute(addresses, workspace.data_ptr() if workspace is not None else 0, handle):
            return None
        return TupleDict(**{name: None if source is None else outs[source] if source in outs else kwargs.get(source) for name, source in self.result})

    def allocate(self, which, workspace=True):
        """The outputs ``which(on_stream)`` selects, and the workspace (None for none)."""
        torch, device = self.torch, self.device
        outs = {name: _allocate(torch, shape, stride, dtype, device, init) for name, shape, stride, dtype, init, on_stream in self.outputs if which(on_stream)}
        return outs, torch.empty(self.workspace_bytes, dtype=torch.uint8, device=device) if workspace and self.workspace_bytes else None


def planned(wrapper):
    """``wrapper`` with its keyword calls run by a plan once its memo path has planned one."""
    if not ENABLED:
        return wrapper
    plans = {}  # key -> plan, or the number of calls that planned nothing
    kind = wrapper.__name__.removeprefix("grouped_gemm_").removesuffix("_wrapper_sm100") or "gemm"

    @functools.wraps(wrapper)
    def run(*args, **kwargs):
        if args or kwargs.get("b_ptrs") is not None or kwargs.get("wgrad_ptrs") is not None:  # not planned
            return wrapper(*args, **kwargs)
        try:
            key = _key(kwargs)
            plan = plans.get(key, 0)
        except TypeError:  # an unhashable argument
            return wrapper(**kwargs)
        if type(plan) is not int:
            result = plan(kwargs)
            return wrapper(**kwargs) if result is None else result
        if plan >= 2:  # derived, then memo path: it states no plan for this call
            return wrapper(**kwargs)
        _planning.plan = None
        _planning.call = (kind, kwargs)
        try:
            result = wrapper(**kwargs)
        finally:
            _planning.call = None
        offered, _planning.plan = _planning.plan, None
        plans[key] = offered if offered is not None and offered.op is not None else plan + 1
        return result

    run.plans = plans
    return run


def planning():
    """Whether the running wrapper call should plan (see :func:`offer`)."""
    return getattr(_planning, "call", None) is not None


def offer(plan):
    """Keep ``plan`` for the running call's signature."""
    _planning.plan = plan


def _key(kwargs):
    import torch

    tensor = torch.Tensor
    a = kwargs.get("a_tensor")
    return (
        a.device if isinstance(a, tensor) else None,
        os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"),
        os.getenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1"),
        *[(k, v.dtype, v.shape, v.stride()) if isinstance(v, tensor) else (k, v) for k, v in kwargs.items() if k != "current_stream"],
    )


def _bind(args, operands, stream, workspace, workspace_bytes):
    """How each kernel argument follows from the call (``GroupedGemm`` metadata ``args``), or None."""
    import torch
    import tvm_ffi
    from cuda.bindings import driver as cuda

    names = {id(t): name for name, t in operands}
    regions = [(t.data_ptr(), t.data_ptr() + _extent(t), {"operand": name}) for name, t in operands if t.numel()]
    if workspace_bytes:
        regions.append((workspace, workspace + workspace_bytes, {"workspace": True}))

    def place(address, kind, **tensor):
        for start, end, source in regions:
            if start <= address < end:
                return {**source, "offset": address - start, "as": kind, **tensor}
        return None

    def tensor_view(t, owned=False):
        """A tensor argument: an operand, a view into an operand or the workspace, or (``owned``) the API's own."""
        if isinstance(t, torch.Tensor):
            if id(t) in names:
                return {"operand": names[id(t)], "as": "tensor"}
            t = tvm_ffi.from_dlpack(t)
        if 0 in tuple(t.shape):
            return None
        d = t.dtype
        layout = {"dim": list(t.shape), "stride": list(t.strides), "dtype": [int(d.type_code), int(d.bits), int(d.lanes)]}
        placed = place(t.data_ptr(), "tensor", **layout)
        return placed or ({"address": t.data_ptr(), "as": "tensor", **layout} if owned else None)

    recipe = []
    for v in args:
        cls = type(v)
        if v is None:
            entry = None
        elif isinstance(v, tvm_ffi.Tensor) or hasattr(cls, "__tvm_ffi_object__"):
            obj = v if isinstance(v, tvm_ffi.Tensor) else v.__tvm_ffi_object__()
            entry = tensor_view(obj, owned=True) if isinstance(obj, tvm_ffi.Tensor) else None
        elif hasattr(cls, "__dlpack_c_exchange_api__") or isinstance(v, torch.Tensor):
            entry = tensor_view(v)
        elif hasattr(cls, "__cuda_stream__") or isinstance(v, cuda.CUstream):
            handle = v.__cuda_stream__()[1] if hasattr(cls, "__cuda_stream__") else int(v)
            entry = {"stream": True} if handle == int(stream) else None
        elif isinstance(v, bool):
            entry = {"bool": v}
        elif isinstance(v, Integral):
            entry = place(int(v), "int") or {"int": int(v)}
        elif isinstance(v, Real):
            entry = {"float": float(v)}
        elif hasattr(cls, "__tvm_ffi_opaque_ptr__"):
            address = int(v.__tvm_ffi_opaque_ptr__())
            entry = place(address, "pointer") or {"pointer": address}
        elif hasattr(cls, "__tvm_ffi_int__"):
            entry = {"int": int(v.__tvm_ffi_int__())}
        elif hasattr(cls, "__tvm_ffi_float__"):
            entry = {"float": float(v.__tvm_ffi_float__())}
        else:
            entry = None
        if entry is None and v is not None:
            return _decline(f"argument {len(recipe)} ({cls.__name__})")
        recipe.append(entry)
    return recipe


def _create(launch, binding, operands, workspace_bytes, device):
    """Register the kernel of ``launch`` with ``binding`` and create the C++ op for ``operands``; None when C++ refuses.

    ``launch``: (the API's execute, the compiled kernel, its arguments), kept alive with the registration: the
    API's own buffers among the arguments are bound by address."""
    import torch
    import tvm_ffi

    from cudnn import _compiled_module
    from cudnn.frost.compiled_cache import positional_entry

    kind, kwargs = _planning.call
    compiled = launch[1]
    try:
        options = json.dumps({k: _option(v) for k, v in kwargs.items() if k != "current_stream" and not isinstance(v, torch.Tensor)}, sort_keys=True)
        described = [(name, (t.data_ptr(), _data_type(t.dtype), tuple(t.shape), t.stride())) for name, t in operands]
    except TypeError as e:
        return _decline(str(e))
    entry = compiled if isinstance(compiled, tvm_ffi.Function) else positional_entry(compiled)
    if entry is None:
        return _decline("the kernel has no tvm-ffi entry")
    meta = {
        "args": binding,
        "align": {name: min(16, address & -address) for name, (address, *_) in described if address},
        "workspace_bytes": workspace_bytes,
    }
    with torch.cuda.device(device):
        config = _compiled_module._grouped_gemm_config(kind, described, options)
        text = json.dumps(meta)
        tvm_ffi.register_global_func(f"cudnn.ops.kernel.{config}", entry, override=True)
        tvm_ffi.register_global_func(f"cudnn.ops.meta.{config}", lambda: text, override=True)
        _kernels[config] = launch
        try:
            return _compiled_module._GroupedGemm(kind, described, options)
        except Exception as e:  # noqa: BLE001 -- C++ refuses the binding; the wrapper keeps the call
            return _decline(str(e))


def _decline(reason):
    logging.getLogger(__name__).info("grouped GEMM plan: the wrapper keeps the call: %s", reason)
    return None


def _option(v):
    import torch

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [_option(x) for x in v]
    raise TypeError(f"option {v!r} has no JSON form")


def _data_type(dtype):
    import torch

    import cudnn

    table = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int8: cudnn.data_type.INT8,
        torch.uint8: cudnn.data_type.UINT8,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
        torch.float8_e8m0fnu: cudnn.data_type.FP8_E8M0,
    }
    if dtype not in table:
        raise TypeError(f"operand dtype {dtype} has no DLPack equivalent here")
    return table[dtype]


def _extent(t):
    """Bytes from ``t``'s first element to just past its last."""
    return (1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))) * t.element_size()


def _allocate(torch, shape, stride, dtype, device, init):
    tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
    if init is not None:
        tensor.fill_(init)
    return tensor
