# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated GEMM source cache, independent of graph analysis and arch compilers."""

import functools
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile

_LOG = logging.getLogger("cudnn.gemm.frost.compiler")


def _fallback_cache_dir() -> Path:
    """A writable per-user cache under the system temp dir."""
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return Path(tempfile.gettempdir()) / f"cudnn_gemm_kernel_cache{suffix}"


@functools.lru_cache(maxsize=None)
def _usable_cache_dir(base: str) -> Path:
    """Resolve a writable kernel cache directory."""
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise PermissionError(f"{p} is not writable")
        return p
    except OSError as exc:
        fallback = _fallback_cache_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        _LOG.warning(
            "cudnn.gemm.frost: kernel cache %s is unusable (%s); falling back to %s. Set CUDNN_FRONTEND_GEMM_KERNEL_CACHE to choose another location.",
            base,
            exc,
            fallback,
        )
        return fallback


def _cache_dir() -> Path:
    """Select the generated-kernel source cache directory."""
    base = os.environ.get("CUDNN_FRONTEND_GEMM_KERNEL_CACHE")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_gemm", "kernel_cache")
    return _usable_cache_dir(base)


def _write_atomic(path: Path, src: str) -> None:
    """Publish ``src`` at ``path`` in one rename."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            # Preserve the original write failure.
            pass
        raise


def _import_kernel(src: str) -> object:
    """Import a generated kernel from the source cache."""
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    out_dir = _cache_dir() / f"gen_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_kernel.py"
    _write_atomic(out_file, src)

    mod_name = f"_cudnn_gemm_generated_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, out_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    mod.FROST_SOURCE_DIGEST = digest
    return mod
