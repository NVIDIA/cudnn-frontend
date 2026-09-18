# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AOT export shared by every CuTeDSL-family engine.

An engine describes its kernel as a list of ``Step``s and calls ``link_steps``
(flow 2, writes one linked shared object) or ``register_steps`` (flow 3,
publishes into the tvm-ffi global table). A one-launch kernel is a one-element
list; the FROST GDN forward is four. Both end up as the payload's ``steps``,
which is what ``CuteDslStep`` in ``cutedsl_engine_interface.h`` reads.

Several exported kernels link into ONE shared object and stay individually
resolvable by name, so a sequence costs the same single dlopen as one launch.

The argument lists are the engine's business, not the graph's: only the engine
knows the kernel's signature, and CuTeDSL's rules for what survives into the
runtime signature are not guessable from the graph. A plain Python
``int``/``bool``/dataclass argument at ``cute.compile`` time is frozen as a
constant and disappears, while a ``cutlass.Int32`` / ``cutlass.Float32`` /
annotated parameter stays; and an optional port passed as ``None`` at compile
time stays as a positional slot that must still be filled with null (hence
``NONE`` among the argument kinds, rather than simply omitting it).

Small kernels state their steps outright, which is worth reading once as the
plain form of the thing. A plan of several launches over a carved workspace can
instead hand ``record_launch_sequence`` a probe run and let it watch: same
result, but the signatures stay in the one place that already knows them.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple


def _cudnn_dtype_name(dt: Any) -> str:
    return getattr(dt, "name", str(dt)).upper().replace("DATA_TYPE.", "")


def _contiguous_stride(shape) -> List[int]:
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * int(shape[i + 1])
    return stride


def tensor_arg(uid: int, data_type: Any, shape) -> Dict[str, Any]:
    """A variant-pack tensor, addressed by uid."""
    name = _cudnn_dtype_name(data_type)
    if data_type is None or name == "NOT_SET":
        raise ValueError(
            f"the tensor with uid {uid} has no data type, so the artifact cannot describe it. "
            "Call set_data_type() on it before exporting: an inferred dtype is fine at JIT time, "
            "but an AOT artifact has to state what the kernel was compiled for."
        )
    shape = [int(s) for s in shape]
    return {
        "kind": "TENSOR",
        "uid": int(uid),
        "data_type": _cudnn_dtype_name(data_type),
        "shape": shape,
        "stride": _contiguous_stride(shape),
    }


def workspace_arg(offset: int, data_type: str, shape) -> Dict[str, Any]:
    """A slice of the engine workspace. ``data_type`` is a cuDNN type name."""
    shape = [int(s) for s in shape]
    return {
        "kind": "WORKSPACE",
        "workspace_offset": int(offset),
        "data_type": data_type,
        "shape": shape,
        "stride": _contiguous_stride(shape),
    }


def i64_arg(value: int) -> Dict[str, Any]:
    return {"kind": "SCALAR_I64", "value": int(value)}


def f64_arg(value: float) -> Dict[str, Any]:
    return {"kind": "SCALAR_F64", "value": float(value)}


class Step:
    """One entry of the launch sequence, before it is lowered to a payload.

    ``fn`` is the object ``cute.compile`` handed back. It is kept here rather
    than a symbol name because export and register_global need different things
    out of it: export writes it to an object file, register_global publishes the
    live tvm-ffi function it already holds.
    """

    def __init__(self, name: str, fn: Any, args: List[Dict[str, Any]]):
        self.name = name
        self.fn = fn
        self.args = args


class MemsetZero:
    """Zero a workspace region, stream-ordered, before the next step reads it."""

    def __init__(self, offset: int, nbytes: int):
        self.offset = int(offset)
        self.nbytes = int(nbytes)


def gpu_arch() -> str:
    from cuda.bindings import runtime as rt

    err, dev = rt.cudaGetDevice()
    if int(err) != 0:
        raise RuntimeError(f"cudaGetDevice failed: {err}")
    err, major = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev)
    if int(err) != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute failed: {err}")
    err, minor = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev)
    if int(err) != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute failed: {err}")
    # 'a' selects the architecture-specific feature set, which is what the
    # FROST kernels are compiled against.
    return f"sm_{major}{minor}a"


def link_steps(steps, graph_name: str) -> Tuple[List[Dict[str, Any]], bytes, List[str]]:
    """Export every CALL step to an object file and link the set into one .so.

    Returns the payload's step list, the module bytes, and the runtime
    dependencies the deploy box needs on its library path. Loading the result
    needs neither a compiler nor cutlass.
    """
    import cutlass.cute as cute
    import tvm_ffi  # importable-at-export check: the loader will need it too

    _ = tvm_ffi

    payload_steps: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cudnn_fe_aot_") as tmp:
        objects = []
        for index, step in enumerate(steps):
            if isinstance(step, MemsetZero):
                payload_steps.append({"kind": "MEMSET_ZERO", "workspace_offset": step.offset, "nbytes": step.nbytes})
                continue
            # Distinct per step: several kernels share one object file, and
            # CuTeDSL derives its internal symbol prefix from this name.
            symbol = f"cudnn_aot_{graph_name}_{index}_{step.name}"
            obj = os.path.join(tmp, f"step{index}.o")
            step.fn.export_to_c(obj, symbol)
            objects.append(obj)
            payload_steps.append({"kind": "CALL", "function_name": symbol, "args": step.args})

        so = os.path.join(tmp, "kernels.so")
        runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
        link = [os.environ.get("CC", "gcc"), "-shared", "-o", so, *objects, *runtime_libs]
        proc = subprocess.run(link, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Linking the AOT module failed:\n{' '.join(link)}\n{proc.stderr}")
        with open(so, "rb") as f:
            module_bytes = f.read()

    # The SONAMEs, not the build-box paths: the deploy box resolves them
    # however it resolves libraries.
    runtime_deps = [os.path.basename(p) for p in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)]
    return payload_steps, module_bytes, runtime_deps


def register_steps(steps, symbol_prefix: str) -> List[Dict[str, Any]]:
    """Publish every CALL step in the tvm-ffi global table; nothing is written.

    ``cute.compile`` with ``--enable-tvm-ffi`` hands back an object that already
    holds a live tvm-ffi function whose cubin is in this CUDA context, so flow 3
    is a handful of table insertions and an argument description.
    """
    import tvm_ffi

    payload_steps: List[Dict[str, Any]] = []
    for index, step in enumerate(steps):
        if isinstance(step, MemsetZero):
            payload_steps.append({"kind": "MEMSET_ZERO", "workspace_offset": step.offset, "nbytes": step.nbytes})
            continue
        fn = getattr(step.fn, "_tvm_ffi_function", None)
        if fn is None:
            fn = step.fn if isinstance(step.fn, tvm_ffi.Function) else None
        if fn is None:
            raise RuntimeError(f"the compiled kernel for step '{step.name}' exposes no tvm-ffi function; was --enable-tvm-ffi set?")
        symbol = f"{symbol_prefix}.{index}"
        tvm_ffi.register_global_func(symbol, fn, override=True)
        payload_steps.append({"kind": "CALL", "global_symbol": symbol, "args": step.args})
    return payload_steps


def base_payload(workspace_size: int) -> Dict[str, Any]:
    return {
        "abi": "tvm-ffi",
        "abi_version": 1,
        "sm_arch": gpu_arch(),
        "runtime_deps": [],
        "engine_workspace_size": int(workspace_size),
    }


# cuDNN type name <- the name torch and DeviceView both spell their dtypes with
_DTYPE_NAMES = {
    "bfloat16": "BFLOAT16",
    "float16": "HALF",
    "float32": "FLOAT",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
}


def _torch_dtype(data_type, port: str):
    import torch

    import cudnn

    dt = {
        cudnn.data_type.HALF: torch.float16,
        cudnn.data_type.BFLOAT16: torch.bfloat16,
        cudnn.data_type.FLOAT: torch.float32,
        cudnn.data_type.INT32: torch.int32,
    }.get(data_type)
    if dt is None:
        raise ValueError(
            f"the graph tensor for '{port}' has data type {data_type}, which the artifact cannot describe. "
            "Call set_data_type() on it before exporting: an inferred dtype is fine at JIT time, but an AOT "
            "artifact has to state what the kernel was compiled for."
        )
    return dt


def _positional(fn, args, kwargs):
    """The call's arguments as the exported symbol will receive them.

    A CuTeDSL entry point with keyword-only parameters or defaults is called
    through a generated wrapper, but the symbol underneath is positional-only.
    The wrapper carries the real signature, so binding against it turns any
    call shape into the one order the payload can describe.
    """
    if not kwargs:
        return args
    import inspect

    wrapper = getattr(fn, "_kwargs_wrapper", None)
    if wrapper is None:
        raise RuntimeError(f"a kernel was called with keyword arguments {list(kwargs)} but exposes no signature to resolve them against")
    signature = inspect.signature(wrapper)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments[name] for name in signature.parameters)


def _recording():
    """Patch in a recorder over every compiled-kernel call and workspace memset.

    Returns (calls, restore). ``calls`` fills with ``(fn, args)`` for a launch
    and ``(None, (ptr, nbytes))`` for a memset, in issue order.
    """
    from cutlass.cutlass_dsl import tvm_ffi_provider as provider

    from cudnn.frost import buffers

    classes = [provider.TVMFFIJitCompiledFunctionWithKwargs, provider.TVMFFIJitCompiledFunctionBase]
    saved = [(c, c.__call__) for c in classes] + [(buffers, buffers.memset_zero_async)]
    calls, depth = [], [0]

    def wrap_call(original):
        def recording(self, *args, **kwargs):
            # The subclass __call__ delegates to the base's, so only the
            # outermost frame is a launch.
            if depth[0] == 0:
                calls.append((self, _positional(self, args, kwargs)))
            depth[0] += 1
            try:
                return original(self, *args, **kwargs)
            finally:
                depth[0] -= 1

        return recording

    def recording_memset(ptr, nbytes, stream):
        calls.append((None, (int(ptr), int(nbytes))))
        return saved[-1][1](ptr, nbytes, stream)

    for c in classes:
        c.__call__ = wrap_call(c.__call__)
    buffers.memset_zero_async = recording_memset

    def restore():
        for owner, original in saved:
            if owner is buffers:
                buffers.memset_zero_async = original
            else:
                owner.__call__ = original

    return calls, restore


def record_launch_sequence(run, tensors, workspace_bytes: int):
    """``(steps, workspace_bytes)`` for a plan, by running it and watching.

    A real kernel family's launch sequence lives in host Python: which kernels
    run, in what order, over which buffers. Transcribing that into an artifact
    by hand costs a second copy of every kernel's runtime signature and its
    compile-cache key -- free to drift from the first copy, and growing with
    every kernel in the family. Running the plan and watching keeps one copy:
    the kernel's own call site.

    ``tensors`` are the graph tensors the plan touches; ``run(buffers,
    workspace, stream)`` issues its launches over throwaway copies of them,
    keyed by uid. A plan retrofits by supplying those two, which is why the
    same recorder serves plans whose buffers are keyed by node port and plans
    whose buffers are keyed by tensor object.

    Two runs over two distinct buffer sets. The first compiles, since these
    kernels compile on first execute when the real buffers are known. The
    second is recorded, and its fresh addresses defeat the pointer-keyed
    descriptor memo the kernels hold, so the descriptor builds appear in the
    sequence. They have to: execute() may not mutate the graph, so an artifact
    rebuilds descriptors on every call rather than remembering last call's
    pointers.

    Export already requires the target GPU -- the cubin is architecture
    specific -- so running the kernel to describe it asks for nothing new.
    """
    import torch

    stream = torch.cuda.current_stream().cuda_stream

    def fresh():
        out = {}
        for t in tensors:
            name = t.get_name() or f"uid {t.get_uid()}"
            dtype = _torch_dtype(t.get_data_type(), name)
            # as_strided, not zeros: a plan's tensors carry the graph's own
            # strides, which need not be row-major contiguous (an SDPA q/k/v is
            # BHSD over BSHD storage). The kernel is compiled for that layout.
            dim, stride = [int(d) for d in t.get_dim()], [int(v) for v in t.get_stride()]
            span = 1 + sum((d - 1) * v for d, v in zip(dim, stride))
            out[t.get_uid()] = torch.zeros(span, dtype=dtype, device="cuda").as_strided(dim, stride)
        return out

    def workspace():
        return torch.zeros(max(workspace_bytes, 1), dtype=torch.uint8, device="cuda")

    # Held past the second run so the caching allocator cannot hand the same
    # addresses back and quietly re-arm the descriptor memo.
    warmup, warmup_ws = fresh(), workspace()
    run(warmup, warmup_ws, stream)
    torch.cuda.synchronize()

    bufs, ws = fresh(), workspace()
    calls, restore = _recording()
    try:
        run(bufs, ws, stream)
        torch.cuda.synchronize()
    finally:
        restore()
    del warmup, warmup_ws
    if not calls:
        raise RuntimeError("the AOT probe recorded no launches, so there is nothing to export")

    uid_of_ptr = {b.data_ptr(): uid for uid, b in bufs.items()}
    ws_ptr = ws.data_ptr()

    # Buffers the plan allocated for itself, which are neither variant-pack
    # tensors nor workspace slices -- the zero-filled stand-ins a kernel takes
    # for optional ports the graph did not ask for. An artifact can only
    # address the pack and the workspace, so they are appended to the workspace
    # and zeroed by a step. That is sound only because they ARE zero, so that
    # is checked rather than assumed: a plan-allocated buffer carrying real
    # content is a genuinely un-exportable argument and says so.
    carved, extra = {}, workspace_bytes
    for fn, payload in calls:
        if fn is None:
            continue
        for a in payload:
            ptr = getattr(a, "data_ptr", None)
            if ptr is None or not callable(ptr):
                continue
            ptr = ptr()
            if ptr in uid_of_ptr or 0 <= ptr - ws_ptr < max(workspace_bytes, 1) or ptr in carved:
                continue
            if a.count_nonzero().item():
                raise RuntimeError(
                    f"the plan reached a kernel with a {list(a.shape)} {a.dtype} buffer of its own that is not "
                    "all zeros, so the artifact cannot reproduce it: it is neither a variant-pack tensor nor a "
                    "workspace slice, and its contents are not recoverable from the graph."
                )
            nbytes = a.numel() * a.element_size()
            extra = (extra + 127) & ~127
            carved[ptr] = (extra, nbytes)
            extra += nbytes
    workspace_bytes = extra

    def spec(a, index=None, step=None):
        if a is None:
            return {"kind": "NONE"}
        if isinstance(a, (tuple, list)):
            # ONE parameter carrying a tuple of ints (a problem-size descriptor),
            # not one slot per element -- the kernel's arity says so. The loader
            # builds the container once, so this costs nothing per call.
            return {"kind": "ARRAY_I64", "values": [int(getattr(x, "value", x)) for x in a]}
        if type(a).__name__ == "CUstream":
            return {"kind": "STREAM"}
        # cutlass.Float32 / cutlass.Int32 and friends survive into the runtime
        # signature as scalars, unlike a plain Python int frozen at compile time
        kind = type(a).__name__
        # a cutlass scalar wraps its python value; plain int/float go as-is
        value = a.value if hasattr(a, "value") and not hasattr(a, "data_ptr") else a
        if isinstance(value, float) or kind.startswith(("Float", "Double")):
            return f64_arg(value)
        if isinstance(value, int) or kind.startswith(("Int", "Uint", "Bool")):  # bool included, deliberately
            return i64_arg(value)
        name = _DTYPE_NAMES[str(a.dtype).split(".")[-1]]
        shape = [int(s) for s in a.shape]
        # the recorded object's own strides; only fall back to row-major for a
        # view that does not carry them (a workspace slice)
        get = getattr(a, "stride", None)
        stride = [int(v) for v in get()] if callable(get) else _contiguous_stride(shape)
        ptr = a.data_ptr()
        if ptr in uid_of_ptr:
            return {"kind": "TENSOR", "uid": uid_of_ptr[ptr], "data_type": name, "shape": shape, "stride": stride}
        offset = ptr - ws_ptr
        if 0 <= offset < max(workspace_bytes, 1):
            return {"kind": "WORKSPACE", "workspace_offset": offset, "data_type": name, "shape": shape, "stride": stride}
        if ptr in carved:
            return {"kind": "WORKSPACE", "workspace_offset": carved[ptr][0], "data_type": name, "shape": shape, "stride": stride}
        raise RuntimeError(
            f"step {step} argument {index} (shape {shape}, {name}) points at neither a graph tensor nor the "
            "engine workspace. Every buffer an exported plan touches must come from the variant pack or the "
            "workspace, since those are the only two the artifact can address."
        )

    # zeroed once, ahead of every launch that reads them
    steps = [MemsetZero(off, nbytes) for off, nbytes in sorted(carved.values())]
    for index, (fn, payload) in enumerate(calls):
        if fn is None:
            ptr, nbytes = payload
            steps.append(MemsetZero(ptr - ws_ptr, nbytes))
        else:
            steps.append(Step(f"step{index}", fn, [spec(a, i, index) for i, a in enumerate(payload)]))
    return steps, workspace_bytes
