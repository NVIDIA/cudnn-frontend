# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind prepared Cake launches to XLA-owned buffers and the XLA CUDA stream."""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import subprocess
import os
import uuid

import jax
import numpy as np


@dataclass(frozen=True)
class TensorRef:
    index: int
    shape: tuple
    dtype: str
    offset: int = 0


_TARGET = "cudnn_cake_launch_" + uuid.uuid4().hex


class _Argument(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in ("kind", "index", "offset", "ndim")] + [
        ("shape", ctypes.c_int64 * 8),
        ("code", ctypes.c_int64),
        ("bits", ctypes.c_int64),
        ("integer", ctypes.c_int64),
        ("floating", ctypes.c_double),
    ]


class _Stage(ctypes.Structure):
    _fields_ = [("function", ctypes.c_void_p), ("count", ctypes.c_int64), ("arguments", ctypes.POINTER(_Argument))]


@lru_cache(maxsize=1)
def _library():
    import tvm_ffi
    from tvm_ffi import libinfo
    from . import loader

    source = Path(__file__).with_name("jax_ffi.cc")
    nvcc, cuda = loader._toolkit()
    signature = source.read_bytes() + f"{jax.__version__}:{tvm_ffi.__version__}:{cuda.full}".encode()
    directory = loader.cache_dir() / ("jax_ffi_" + hashlib.sha256(signature).hexdigest()[:24])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bridge.so"
    with (directory / "build.lock").open("a") as lock:
        loader._lock_exclusive(lock, directory / "build.lock", "Cake JAX FFI")
        if not path.exists():
            if loader._NO_BUILD.get():
                raise RuntimeError("Cake JAX bridge was not prepared before no_build")
            cuda_root = Path(nvcc).parent.parent
            lib = Path(libinfo.find_libtvm_ffi())
            cuda_lib = cuda_root / ("lib64" if (cuda_root / "lib64").is_dir() else "lib")
            temporary = path.with_name(f"bridge.{os.getpid()}.tmp.so")
            command = [
                "c++",
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-O2",
                str(source),
                "-o",
                str(temporary),
                "-I" + jax.ffi.include_dir(),
                "-I" + libinfo.find_include_path(),
                "-I" + libinfo.find_dlpack_include_path(),
                "-I" + str(cuda_root / "include"),
                str(lib),
                "-L" + str(cuda_lib),
                "-lcudart",
                "-Wl,-rpath," + str(lib.parent),
                "-Wl,-rpath," + str(cuda_lib),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Cake JAX bridge build failed:\n" + result.stderr)
            os.replace(temporary, path)
    library = ctypes.CDLL(str(path))
    library.cake_jax_plan.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Stage), ctypes.c_int, ctypes.POINTER(_Argument)]
    library.cake_jax_plan.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(library.cake_jax_launch), platform="CUDA")
    return library


def _argument(value):
    if isinstance(value, TensorRef):
        dtype = np.dtype(value.dtype)
        code = 4 if dtype.name == "bfloat16" else {"f": 2, "i": 0, "u": 1}[dtype.kind]
        if value.index < 0 or len(value.shape) > 8 or value.offset < 0 or any(d < 0 for d in value.shape) or math.prod(value.shape) * dtype.itemsize >= 2**63:
            raise ValueError("invalid Cake tensor reference")
        return _Argument(7, value.index, value.offset, len(value.shape), (ctypes.c_int64 * 8)(*value.shape), code, dtype.itemsize * 8, 0, 0)
    arg = _Argument()
    if value is None:
        arg.kind = 0
    elif isinstance(value, (bool, int, np.integer)):
        arg.kind, arg.integer = (2 if isinstance(value, bool) else 1), int(value)
    elif isinstance(value, (float, np.floating)):
        arg.kind, arg.floating = 3, float(value)
    else:
        raise TypeError(f"unsupported Cake argument: {type(value)}")
    return arg


def make_call(stages, input_count, output_shape_dtype, device, zero_buffers=()):
    """Prepare a native launch sequence; input buffers precede output buffers."""
    total = input_count + len(output_shape_dtype)
    if input_count < 0 or any(ref.index < input_count or ref.index >= total for ref in zero_buffers):
        raise ValueError("zero buffers must refer to outputs")
    for _, args in stages:
        if any(isinstance(arg, TensorRef) and not 0 <= arg.index < total for arg in args):
            raise ValueError("Cake tensor reference is outside the call buffers")
    library = _library()
    arguments = [(_Argument * len(args))(*map(_argument, args)) for _, args in stages]
    records = (_Stage * len(stages))(*(_Stage(fn.__chandle__(), len(args), args) for (fn, _), args in zip(stages, arguments)))
    zeros = (_Argument * len(zero_buffers))(*map(_argument, zero_buffers))
    plan = library.cake_jax_plan(device, input_count, len(stages), records, len(zeros), zeros)
    invoke = jax.ffi.ffi_call(_TARGET, output_shape_dtype)
    return lambda *args: invoke(*args, plan=np.int64(plan))
