# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time export of a Python-engine plan: what ``graph.serialize()`` writes
when the selected plan is a CuTeDSL engine.

The artifact is the plan itself, frozen: the compiled kernels (each linked into a
small shared object from the object the DSL exported) and the launch sequence the
plan's ``execute()`` issues, with every argument bound except buffer addresses and
the stream. ``Graph::deserialize()`` in C++ (``experimental/aot_engine.h``) reads it
back and executes it with no Python, no JIT and no tvm-ffi headers; the Python
``graph.deserialize()`` goes through the same C++ path.

How the launch sequence is taken. A plan that supports export implements
``CompiledPlan.launches()``: exactly what ``execute()`` would issue, without issuing
it, and ``execute()`` is those launches run in order, so the two cannot disagree.
Export calls it twice, over a variant pack that describes every operand exactly as
the graph declares it but places each at a distinct, recognizable address -- a
different set of addresses each time. An argument equal in both is a constant of
the artifact; an argument that moved with exactly one operand's (or the
workspace's) address is that address plus a fixed offset; anything else depends on
addresses in a way the artifact cannot express, and export refuses it by name.
Nothing is launched, nothing is allocated, and no device memory is read.

What the artifact therefore is: the plan for the DECLARED shapes, like a cuDNN
backend plan. The kernels themselves are shape-generic, so graphs of many shapes
share one kernel (and one load); execute-time shape overrides are refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._aot_runtime import _dist_version
from .base import ExecutionContext, Launch, VariantPack

FORMAT_VERSION = 1  # experimental/aot_engine.h AOT_FORMAT_VERSION

# Sentinel addresses. Operand k sits at (k + 1) << 40 in the first binding and
# moves by (k + 1) << 30 in the second, so an argument derived from one operand
# moves by that operand's own step, and one derived from two (a difference, a
# sum) matches none. Offsets must stay below the step (1 GiB) -- far above any
# carve a plan takes from a real buffer.
_BASE_SHIFT = 40
_STEP_SHIFT = 30
_MAX_OFFSET = 1 << _STEP_SHIFT
# Stream sentinels: odd, huge, never a plausible shape, stride or address.
_STREAMS = ((1 << 62) + 0x5EED1, (1 << 62) + 0x5EED3)


def _base(k: int, run: int) -> int:
    return ((k + 1) << _BASE_SHIFT) + run * ((k + 1) << _STEP_SHIFT)


def _declared_pack(graph, order: Sequence[int], run: int, workspace_bytes: int, device: int) -> VariantPack:
    """A VariantPack whose every operand is exactly its declaration, at a sentinel address.

    Built the way ``pygraph._normalize`` describes a caller buffer that reports the
    declared geometry and covers exactly the declared span, on the plan's device --
    which is the contract a caller of the artifact honours.
    """
    from cudnn import _pybind_module
    from cudnn.datatypes import _dlpack_code_bits, _dlpack_lanes
    from cudnn.graph_types import storage_geometry, storage_slot_bytes

    native = _pybind_module.VariantPackNative(len(order))
    for i, uid in enumerate(order):
        t = graph._tensor_by_uid.get(uid)
        if t is None or not t.dim:
            raise NotImplementedError(f"cannot export: variant-pack tensor uid {uid} has no declared geometry")
        dims, strides = storage_geometry(t.dim, t.stride, t.data_type)
        slot = storage_slot_bytes(t.data_type)
        if not slot:
            raise NotImplementedError(f"cannot export: tensor {t.name or uid!r} has data type {t.data_type} of unknown width")
        span = 0 if any(int(d) == 0 for d in dims) else 1 + sum((int(d) - 1) * int(s) for d, s in zip(dims, strides))
        native.set_operand(i, _base(i, run), tuple(dims), tuple(strides), *_dlpack_code_bits(t.data_type), _dlpack_lanes(t.data_type), span * slot, 2, device)
    from_graph = native.describe_from(graph._declared_layout(list(order)), [])
    ws = _base(len(order), run) if workspace_bytes else 0
    return VariantPack(tuple(order), native, ws, workspace_bytes, tuple(from_graph))


def _is_stream(value: Any, run: int) -> bool:
    if type(value).__name__ == "CUstream":
        value = int(value)
    return type(value) is int and value == _STREAMS[run]


def _lower_arg(a: Any, b: Any, order: Sequence[int], where: str) -> Dict[str, Any]:
    """One argument of the two bindings, as the artifact states it."""
    if _is_stream(a, 0) and _is_stream(b, 1):
        return {"stream": None}
    if a is None and b is None:
        return {"none": None}
    if type(a) is bool and type(b) is bool and a == b:
        return {"bool": a}
    if type(a) is int and type(b) is int:
        if a == b:
            return {"int": a}
        for k in range(len(order) + 1):
            offset = a - _base(k, 0)
            if 0 <= offset < _MAX_OFFSET and b - _base(k, 1) == offset:
                return {"tensor": order[k], "offset": offset} if k < len(order) else {"workspace": offset}
        raise NotImplementedError(f"cannot export: {where} depends on buffer addresses in a way the artifact cannot state")
    if type(a) is float and type(b) is float and (a == b or (a != a and b != b)):
        # JSON cannot spell the non-finite values; the C++ reader maps these three strings.
        return {"float": a if math.isfinite(a) else ("nan" if a != a else "inf" if a > 0 else "-inf")}
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) == len(b) and all(type(x) is int and x == y for x, y in zip(a, b)):
            return {"array": list(a)}
        raise NotImplementedError(f"cannot export: {where} is a tuple that is not constant integers")
    raise NotImplementedError(f"cannot export: {where} is a {type(a).__name__}, which the artifact cannot state")


# ---------------------------------------------------------------------------
# Kernel modules
# ---------------------------------------------------------------------------

_LINKED: Dict[str, bytes] = {}  # sha256 of the exported object -> the linked shared object
_LINK_LOCK = threading.Lock()


def _kernel_object(owner: Any) -> Tuple[bytes, str]:
    """``(object file bytes, tvm-ffi symbol)`` of a compiled kernel."""
    entry = getattr(owner, "_compiled_cache_entry", None)
    if entry is not None:  # reloaded from (or just written to) the compiled-plan cache
        with open(os.path.join(entry, "entry.json")) as f:
            symbol = json.load(f)["symbol"]
        with open(os.path.join(entry, "kernel.o"), "rb") as f:
            return f.read(), symbol
    if hasattr(owner, "export_to_c"):  # an in-process --enable-tvm-ffi object
        symbol = "cudnn_aot_kernel"
        with tempfile.TemporaryDirectory(prefix="cudnn_aot_") as tmp:
            path = os.path.join(tmp, "kernel.o")
            owner.export_to_c(path, function_name=symbol)
            with open(path, "rb") as f:
                return f.read(), symbol
    raise NotImplementedError(f"cannot export: the kernel {type(owner).__name__} was not compiled with --enable-tvm-ffi")


def runtime_libraries() -> List[str]:
    """The CuTeDSL runtime libraries an exported kernel links against (absolute paths)."""
    import cutlass.cute as cute

    return list(cute.runtime.find_runtime_libraries(enable_tvm_ffi=True))


def _link(obj: bytes) -> bytes:
    """The exported object as a shared object the C++ loader can dlopen.

    Links against the runtime libraries by SONAME (they carry one), so the
    artifact resolves them however the deploying process resolves libraries.
    """
    key = hashlib.sha256(obj).hexdigest()
    with _LINK_LOCK:
        if key in _LINKED:
            return _LINKED[key]
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("AOT export links each kernel into a shared object and needs a C compiler driver; none of $CC, cc, gcc, clang is available")
    with tempfile.TemporaryDirectory(prefix="cudnn_aot_") as tmp:
        src, out = os.path.join(tmp, "kernel.o"), os.path.join(tmp, "kernel.so")
        with open(src, "wb") as f:
            f.write(obj)
        # -Bsymbolic: the kernel's internal references bind inside its own
        # object, whatever else the process has loaded under the same names.
        cmd = [cc, "-shared", "-Wl,-Bsymbolic", "-o", out, src, *runtime_libraries()]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"linking an AOT kernel failed:\n{' '.join(cmd)}\n{proc.stderr}")
        with open(out, "rb") as f:
            linked = f.read()
    with _LINK_LOCK:
        _LINKED[key] = linked
    return linked


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _launches(graph, plan, order: Sequence[int], run: int, device: int) -> List[Launch]:
    pack = _declared_pack(graph, order, run, int(plan.get_workspace_size()), device)
    return list(plan.launches(graph, pack, ExecutionContext(handle=None, stream=_STREAMS[run], workspace=None)))


def _fill_step(fn: Any, a: Tuple, b: Tuple, order: Sequence[int], where: str) -> Dict[str, Any]:
    from cudnn.frost import buffers

    if fn is buffers.fill_word_async:  # (ptr, count, word, stream)
        step = {"op": "fill32", "count": int(a[1]), "word": int(a[2])}
        consts = ((a[1], b[1]), (a[2], b[2]))
    elif fn is buffers._fill_word_2d_async:  # (ptr, pitch_words, width, height, word, stream)
        step = {"op": "fill32_2d", "pitch": int(a[1]) * 4, "width": int(a[2]), "height": int(a[3]), "word": int(a[4])}
        consts = ((a[1], b[1]), (a[2], b[2]), (a[3], b[3]), (a[4], b[4]))
    else:
        raise NotImplementedError(f"cannot export: {where} calls {getattr(fn, '__name__', fn)!r}, which is neither a compiled kernel nor a known fill")
    if any(x != y for x, y in consts) or not (_is_stream(a[-1], 0) and _is_stream(b[-1], 1)):
        raise NotImplementedError(f"cannot export: {where} is a fill whose extent or stream depends on the call")
    dst = _lower_arg(a[0], b[0], order, f"{where} destination")
    if "tensor" not in dst and "workspace" not in dst:
        raise NotImplementedError(f"cannot export: {where} fills an address that is neither an operand nor the workspace")
    step["dst"] = dst
    return step


def export(graph, plan) -> List[int]:
    """``graph.serialize()`` of a graph whose selected plan is ``plan`` (a Python engine)."""
    from cudnn.frost import device as _device

    import cudnn

    lowered = graph._lowered_graph
    if lowered is None:
        raise NotImplementedError("cannot export: this graph has no C++ lowering to carry the plan")
    order = graph._variant_pack_uids()
    if order is None:
        raise NotImplementedError("cannot export: this graph's variant pack is not known")
    # The device the plan was compiled for, which need not be the current one.
    device = _device.resolve_device(getattr(plan, "device", None))
    first, second = _launches(graph, plan, order, 0, device), _launches(graph, plan, order, 1, device)
    if len(first) != len(second) or any(x.fn is not y.fn for x, y in zip(first, second)):
        raise NotImplementedError("cannot export: the plan's launch sequence depends on buffer addresses")
    if not first:
        raise NotImplementedError("cannot export: the plan launches nothing for the declared shapes")

    modules: List[bytes] = []
    module_of: Dict[int, int] = {}
    symbols: Dict[int, str] = {}
    steps: List[Dict[str, Any]] = []
    for i, (x, y) in enumerate(zip(first, second)):
        where = f"launch {i}"
        if x.owner is None:
            steps.append(_fill_step(x.fn, tuple(x.args), tuple(y.args), order, where))
            continue
        key = id(x.owner)
        if key not in module_of:
            obj, symbol = _kernel_object(x.owner)
            module_of[key], symbols[key] = len(modules), symbol
            modules.append(_link(obj))
        if len(x.args) != len(y.args):
            raise NotImplementedError(f"cannot export: {where} takes a different number of arguments per call")
        args = [_lower_arg(p, q, order, f"{where} argument {j}") for j, (p, q) in enumerate(zip(x.args, y.args))]
        steps.append({"op": "call", "module": module_of[key], "symbol": symbols[key], "args": args})

    major, minor = _device.compute_capability(device)
    runtime = [os.path.basename(p) for p in runtime_libraries()]
    payload = {
        "format": FORMAT_VERSION,
        "abi": "tvm-ffi",
        "target": {"compute_capability": [int(major), int(minor)], "sm_count": int(_device.multiprocessor_count(device))},
        "producer": {
            "cudnn_frontend": str(cudnn.__version__),
            "nvidia-cutlass-dsl": _dist_version("nvidia-cutlass-dsl"),
            "apache-tvm-ffi": _dist_version("apache-tvm-ffi"),
            "runtime_libraries": runtime,
        },
        "workspace_size": int(plan.get_workspace_size()),
        "steps": steps,
    }
    return lowered._serialize_aot(json.dumps(payload), modules, list(order))
