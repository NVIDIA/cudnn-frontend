# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build the host shim against the installed cuBLAS."""

import ctypes as C
import fcntl
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile


@lru_cache(maxsize=1)
def load_helper():
    source = Path(__file__).parent / "csrc" / "compact_gqa_blas.cpp"
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_root / "include/cublas_v2.h").is_file():
        raise RuntimeError("The compact GQA host shim requires CUDA development headers.")
    key = hashlib.sha256(source.read_bytes() + str(cuda_root.resolve()).encode() + platform.machine().encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cudnn_frontend" / "compact_gqa" / key
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "compact_gqa_blas.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                output = Path(tmp) / library.name
                subprocess.run(
                    [
                        os.environ.get("CXX", "g++"),
                        "-shared",
                        "-fPIC",
                        "-O3",
                        str(source),
                        "-I" + str(cuda_root / "include"),
                        "-L" + str(cuda_root / "lib64"),
                        "-lcublas",
                        "-o",
                        str(output),
                    ],
                    check=True,
                    timeout=120,
                )
                output.replace(library)
    lib = C.CDLL(str(library))
    vp, i = C.c_void_p, C.c_int
    lib.gqa_create.argtypes = [vp, vp, C.c_size_t, C.POINTER(i)]
    lib.gqa_create.restype = vp
    lib.gqa_set_workspace.argtypes = [vp, vp, C.c_size_t]
    lib.gqa_set_workspace.restype = i
    lib.gqa_gemm_series.argtypes = [vp, vp, i, i, i, i]
    lib.gqa_gemm_series.restype = i
    lib.gqa_packed_gemm.argtypes = [vp, vp, i, C.POINTER(i), i]
    lib.gqa_packed_gemm.restype = i
    lib.gqa_destroy.argtypes = [vp]
    lib.gqa_destroy.restype = i
    return lib
