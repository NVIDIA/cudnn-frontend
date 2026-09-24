# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the CuTeDSL runtime libraries resolvable before C++ loads an AOT plan.

Imported by ``pygraph.deserialize()``; deliberately free of every engine and of
cutlass, so a process that only runs AOT plans loads neither (see
``cudnn.engines.aot`` for the export side).
"""

import importlib.metadata
import os
from typing import Any, List


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_PRELOADED: List[Any] = []


def preload_runtime() -> None:
    """Load the CuTeDSL runtime libraries globally, from the installed wheels, without importing cutlass.

    The C++ loader resolves them by SONAME, which finds an already-loaded copy;
    this is what lets a Python process that never compiles anything run an
    artifact without LD_LIBRARY_PATH. Best effort: whatever cannot be found here
    is reported by the C++ loader, with its name.
    """
    if _PRELOADED:
        return
    import ctypes
    import importlib.util

    paths: List[str] = []
    spec = importlib.util.find_spec("tvm_ffi")
    if spec is not None and spec.submodule_search_locations:
        paths.append(os.path.join(list(spec.submodule_search_locations)[0], "lib", "libtvm_ffi.so"))
    env = os.environ.get("CUTE_DSL_LIBS", "")
    runtime = next((p for p in env.split(":") if p.endswith("libcute_dsl_runtime.so")), None)
    if runtime is None:
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        if spec is not None and spec.submodule_search_locations:
            root = list(spec.submodule_search_locations)[0]
            major = _dist_version("cuda-bindings").split(".")[0]
            variants = ["cu13", "cu12"] if major != "12" else ["cu12", "cu13"]
            runtime = next((p for p in (os.path.join(root, v, "lib", "libcute_dsl_runtime.so") for v in variants) if os.path.exists(p)), None)
    if runtime is not None:
        paths.append(runtime)
    for path in paths:
        if os.path.exists(path):
            _PRELOADED.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
