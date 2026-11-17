# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native execution of serialized cuDNN plans on XLA's stream."""

import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import uuid

import jax
import numpy as np

_TARGET = "cudnn_sdpa_" + uuid.uuid4().hex


@lru_cache(maxsize=1)
def _library():
    from cudnn._cake import loader
    import cudnn

    source = Path(__file__).with_name("jax_ffi.cc")
    checkout_headers = source.parents[3] / "include"
    if checkout_headers.is_dir():
        headers = checkout_headers
    else:
        spec = importlib.util.find_spec("include")
        if spec is None:
            raise ImportError("cuDNN Frontend headers are required for the JAX bridge")
        headers = Path(next(iter(spec.submodule_search_locations)))
    nvcc, cuda = loader._toolkit()
    toolkit = Path(nvcc).parent.parent
    try:
        spec = importlib.util.find_spec("nvidia.cudnn")
    except ModuleNotFoundError:
        spec = None
    cudnn_root = (
        Path(os.environ["CUDNN_PATH"]) if "CUDNN_PATH" in os.environ else (Path("/usr") if spec is None else Path(next(iter(spec.submodule_search_locations))))
    )
    cudnn_lib = next((p for p in (cudnn_root / "lib", cudnn_root / "lib64", cudnn_root / "lib/x86_64-linux-gnu") if (p / "libcudnn.so.9").exists()), None)
    if cudnn_lib is None:
        raise ImportError("Set CUDNN_PATH to the cuDNN headers and libraries")
    signature = source.read_bytes() + f"{jax.__version__}:{cudnn.__version__}:{cuda.full}:{headers}".encode()
    directory = loader.cache_dir() / ("sdpa_jax_" + hashlib.sha256(signature).hexdigest()[:24])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bridge.so"
    with (directory / "build.lock").open("a") as lock:
        loader._lock_exclusive(lock, directory / "build.lock", "SDPA JAX FFI")
        if not path.exists():
            temporary = path.with_name(f"bridge.{os.getpid()}.tmp.so")
            command = ["c++", "-shared", "-fPIC", "-std=c++17", "-O2", str(source), "-o", str(temporary)]
            command += ["-I" + str(p) for p in (headers, toolkit / "include", cudnn_root / "include", jax.ffi.include_dir())]
            command += [
                str(cudnn_lib / "libcudnn.so.9"),
                "-L" + str(toolkit / "lib64"),
                "-lcudart",
                "-Wl,-rpath," + str(cudnn_lib),
                "-Wl,-rpath," + str(toolkit / "lib64"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("cuDNN JAX bridge build failed:\n" + result.stderr)
            os.replace(temporary, path)
    library = ctypes.CDLL(str(path))
    library.cudnn_jax_prepare.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.cudnn_jax_prepare.restype = ctypes.c_int64
    library.cudnn_jax_error.restype = ctypes.c_char_p
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(library.cudnn_jax_execute), platform="CUDA")
    return library


def make_call(graph, inputs, outputs):
    """Bind fixed descriptors. The last XLA output owns the workspace."""
    if graph.selected_engine is not None:
        raise NotImplementedError("This JAX bridge does not yet export frontend kernel plans")
    library = _library()
    descriptors = inputs + outputs
    uids = np.asarray([t.get_uid() for t, _ in descriptors], np.int64)
    sizes = np.asarray([np.prod(s.shape) * np.dtype(s.dtype).itemsize for _, s in descriptors], np.int64)
    blob = graph.serialize(as_bytes=True)
    plan = library.cudnn_jax_prepare(blob, len(blob), uids.ctypes.data, sizes.ctypes.data, len(uids))
    if not plan:
        raise RuntimeError(library.cudnn_jax_error().decode())
    results = [s for _, s in outputs] + [jax.ShapeDtypeStruct((max(1, graph.get_workspace_size()),), np.uint8)]
    invoke = jax.ffi.ffi_call(_TARGET, results, vmap_method="sequential")
    return lambda *args: invoke(*args, plan=np.int64(plan))[:-1]
