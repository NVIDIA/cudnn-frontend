# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent cache of compiled FROST kernels, across processes.

``cute.compile`` runs the whole backend (NVVM, ptxas, host stub) once per
process for every distinct kernel -- 0.7 s for a GEMM, 2.6 s for an SDPA
prefill on SM100 -- and the DSL's own file cache stores only MLIR bytecode, so
the heavy part is paid again at every start-up. A serving process that warms
tens of plans pays minutes. This module keeps the exported object of every
kernel compiled with ``--enable-tvm-ffi`` and reloads it (a few milliseconds)
instead.

Layout, and the rules that keep a stale artifact from ever being reused:

    <root>/v1/<env_hash>/manifest.json           the environment identity, human-readable
    <root>/v1/<env_hash>/<entry_hash>/kernel.o   the exported tvm-ffi object
    <root>/v1/<env_hash>/<entry_hash>/entry.json the key it was stored under (commit marker)

- ``env_hash`` is the content hash of :func:`environment_manifest`: frontend,
  cutlass-dsl and tvm-ffi versions, the CUDA driver, and the device's name,
  compute capability, SM count and L2 size (FROST bakes the last two into
  kernels). Any change lands in a different directory; nothing is ever
  "matched approximately".
- ``entry_hash`` is the content hash of the caller's key -- for the FROST GEMM
  engine the digest of the generated source, which already carries the tile
  config, dtypes, fusion chain and ``--gpu-arch`` -- plus the exported symbol
  and the compile options. ``entry.json`` embeds the full key and is compared
  on load, so a hash collision or a foreign file is a miss.
- Missing, unreadable, mismatched: a miss, never an error. The object is
  written to a temporary name and renamed into place; ``entry.json`` is written
  after it, so a crash never leaves a loadable entry without its key.
- Bump ``_SCHEMA`` on any incompatible change to this layout.

``compile_cached`` is a drop-in for ``cute.compile`` at a kernel's compile
site. A reloaded tvm-ffi function is positional-only, so it is wrapped with a
kwargs wrapper built from the kernel function's Python signature -- the same
thing the in-process object does -- and the caller cannot tell a hit from a
miss. Kernels whose in-process object converts raw pointer arguments are not
cached (the reload path has no converter); everything else is.

Location: ``CUDNN_FRONTEND_COMPILED_CACHE`` (a directory), else
``$XDG_CACHE_HOME/cudnn_frontend/compiled_plans`` (``~/.cache`` when unset).
``CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`` turns the cache off. A caller
that manages its own workspace (FlashInfer) points :func:`set_cache_dir` at
it once per process.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_SCHEMA = "v1"
_ENV_DIR = "CUDNN_FRONTEND_COMPILED_CACHE"
_ENV_DISABLE = "CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"
_OBJECT = "kernel.o"
_ENTRY = "entry.json"
_MANIFEST = "manifest.json"

_LOG = logging.getLogger("cudnn.frost.compiled_cache")
_LOCK = threading.Lock()
_dir_override: Optional[Path] = None
_stats = {"hits": 0, "misses": 0, "bypassed": 0, "invalid": 0, "export_failed": 0}


# ---------------------------------------------------------------------------
# Location and switches
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """Whether compiled kernels are persisted and reloaded at all."""
    return os.environ.get(_ENV_DISABLE, "0").strip().lower() not in ("1", "true", "yes", "on")


def set_cache_dir(path) -> None:
    """Point the cache at ``path`` for this process (``None`` restores the default)."""
    global _dir_override
    with _LOCK:
        _dir_override = None if path is None else Path(path)


def get_cache_dir() -> Path:
    """The cache root: :func:`set_cache_dir`, else ``$CUDNN_FRONTEND_COMPILED_CACHE``,
    else ``$XDG_CACHE_HOME/cudnn_frontend/compiled_plans``. Falls back to a
    per-user directory under the system temp dir when that is not writable."""
    with _LOCK:
        if _dir_override is not None:
            return _dir_override
    base = os.environ.get(_ENV_DIR)
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_frontend", "compiled_plans")
    root = Path(base)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError(f"{root} is not writable")
        return root
    except OSError as exc:
        suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
        fallback = Path(tempfile.gettempdir()) / f"cudnn_frontend_compiled_plans{suffix}"
        fallback.mkdir(parents=True, exist_ok=True)
        _LOG.warning("compiled-plan cache %s is unusable (%s); using %s. Set %s to choose another location.", root, exc, fallback, _ENV_DIR)
        return fallback


def stats() -> Dict[str, int]:
    """Counters for this process: hits, misses, bypassed (not cacheable), invalid (entry rejected), export_failed."""
    with _LOCK:
        return dict(_stats)


def reset_stats() -> None:
    with _LOCK:
        for k in _stats:
            _stats[k] = 0


def _count(name: str) -> None:
    with _LOCK:
        _stats[name] += 1


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 -- the package may be vendored or absent
        return "unknown"


def environment_manifest(device: Optional[int] = None) -> Dict[str, str]:
    """Everything a compiled object depends on, as deterministic strings.

    A field that cannot be read becomes ``"unknown"``, and :func:`compile_cached`
    then persists nothing: an environment that cannot identify itself must not
    share an entry with one that happens to carry the same unknowns. Never
    loosen a field to widen hits: a wrong hit is a wrong kernel.
    """
    import cudnn

    manifest: Dict[str, str] = {"schema": _SCHEMA, "cudnn_frontend": str(getattr(cudnn, "__version__", "unknown"))}
    try:
        import cutlass

        manifest["cutlass_dsl"] = str(getattr(cutlass, "__version__", _dist_version("nvidia-cutlass-dsl")))
    except Exception:  # noqa: BLE001
        manifest["cutlass_dsl"] = "unknown"
    manifest["tvm_ffi"] = _dist_version("apache-tvm-ffi")
    try:
        from cuda.bindings import driver as _cu

        err, ver = _cu.cuDriverGetVersion()
        manifest["cuda_driver"] = str(ver) if int(err) == 0 else "unknown"
    except Exception:  # noqa: BLE001
        manifest["cuda_driver"] = "unknown"
    try:
        from cudnn.frost import device as _dev

        d = _dev.resolve_device(device)
        cc = _dev.compute_capability(d)
        manifest["device_name"] = str(_dev.device_name(d))
        manifest["compute_capability"] = f"{cc[0]}.{cc[1]}"
        manifest["sm_count"] = str(_dev.multiprocessor_count(d))
        manifest["l2_bytes"] = str(_dev.l2_cache_bytes(d))
    except Exception:  # noqa: BLE001 -- no device: the manifest still names that fact
        for k in ("device_name", "compute_capability", "sm_count", "l2_bytes"):
            manifest[k] = "unknown"
    return manifest


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _entry_dir(root: Path, manifest: Dict[str, str], key: str) -> Path:
    return root / _SCHEMA / _digest(_canonical(manifest)) / _digest(key)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=f".{os.getpid()}.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 -- unreadable is a miss
        return None


def _signature_of(fn: Callable) -> Optional[inspect.Signature]:
    """The Python signature the in-process kwargs wrapper was built from."""
    for candidate in (fn, getattr(fn, "__wrapped__", None), getattr(fn, "func", None), getattr(fn, "_func", None)):
        if candidate is None:
            continue
        try:
            sig = inspect.signature(candidate)
        except (TypeError, ValueError):
            continue
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values()):
            return None
        return sig
    return None


# ---------------------------------------------------------------------------
# The drop-in
# ---------------------------------------------------------------------------


def _try_load(entry: Path, key: str, symbol: str, fn: Callable):
    """The reloaded, kwargs-wrapped kernel, or None (missing, mismatched, unloadable)."""
    record = _read_json(entry / _ENTRY)
    if not isinstance(record, dict):
        return None
    if record.get("schema") != _SCHEMA or record.get("key") != key or record.get("symbol") != symbol:
        _count("invalid")
        _LOG.warning("compiled-plan cache: ignoring %s (embedded key does not match); treating as a miss", entry)
        return None
    obj = entry / _OBJECT
    if not obj.is_file():
        return None
    sig = _signature_of(fn)
    if sig is None:
        return None
    try:
        import cutlass
        from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper_from_signature

        module = cutlass.runtime.load_module(str(obj), enable_tvm_ffi=True)
        raw = getattr(module, symbol)
        wrapper = make_kwargs_wrapper_from_signature(raw, sig)
        wrapper._compiled_cache_module = module  # keep the engine alive as long as the callable
        wrapper._compiled_cache_entry = str(entry)
        return wrapper
    except Exception as exc:  # noqa: BLE001 -- a stale or foreign artifact is a miss
        _count("invalid")
        _LOG.warning("compiled-plan cache: could not reload %s (%s); recompiling", obj, exc)
        return None


def _export(entry: Path, compiled: Any, key: str, symbol: str, manifest: Dict[str, str], fn: Callable) -> None:
    env_dir = entry.parent
    if not (env_dir / _MANIFEST).exists():
        _write_atomic(env_dir / _MANIFEST, (_canonical(manifest) + "\n").encode("utf-8"))
    entry.mkdir(parents=True, exist_ok=True)
    # Unique per export, not per process: two threads may export the same
    # entry concurrently (nothing serializes compiles), and a shared temp name
    # would let one publish the other's half-written object.
    tmp = entry / f".{_OBJECT}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        compiled.export_to_c(str(tmp), function_name=symbol)
        os.replace(tmp, entry / _OBJECT)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    sig = _signature_of(fn)
    record = {"schema": _SCHEMA, "key": key, "symbol": symbol, "signature": str(sig) if sig is not None else None}
    _write_atomic(entry / _ENTRY, (json.dumps(record, indent=1) + "\n").encode("utf-8"))


def compile_cached(fn: Callable, *args: Any, cache_key: Optional[str], symbol: str = "kernel", **kwargs: Any) -> Any:
    """``cute.compile(fn, *args, **kwargs)`` with a persistent object behind it.

    ``cache_key`` names the kernel exactly (for a generated kernel: the digest
    of its source); ``None`` means "not cacheable" and compiles as before. The
    compile options must include ``--enable-tvm-ffi`` -- that is what makes the
    object exportable and reloadable; other kernels compile as before.
    """
    import cutlass.cute as cute

    options = str(kwargs.get("options") or "")
    if not enabled() or cache_key is None or "--enable-tvm-ffi" not in options:
        _count("bypassed")
        return cute.compile(fn, *args, **kwargs)
    manifest = environment_manifest()
    if any(v == "unknown" for v in manifest.values()):
        # An environment that cannot identify itself shares nothing: two
        # incompatible stacks with the same unknowns would otherwise hash alike.
        _count("bypassed")
        return cute.compile(fn, *args, **kwargs)
    entry = _entry_dir(get_cache_dir(), manifest, f"{cache_key}|{symbol}|{options}")
    loaded = _try_load(entry, cache_key, symbol, fn)
    if loaded is not None:
        _count("hits")
        return loaded
    compiled = cute.compile(fn, *args, **kwargs)
    _count("misses")
    execution_args = getattr(compiled, "execution_args", None)
    if getattr(execution_args, "has_pointer_address_arg_specs", False) or not hasattr(compiled, "export_to_c"):
        return compiled  # the in-process object converts raw pointers; a reloaded one could not
    if _signature_of(fn) is None:
        return compiled
    try:
        _export(entry, compiled, cache_key, symbol, manifest, fn)
    except Exception as exc:  # noqa: BLE001 -- persistence is best-effort; the kernel is compiled either way
        _count("export_failed")
        _LOG.warning("compiled-plan cache: could not persist %s (%s); the kernel will be recompiled next process", entry, exc)
        return compiled
    # Hand back the artifact rather than the in-process object, so a hit and a
    # miss run the same thing and a bad artifact fails here, not next start-up.
    loaded = _try_load(entry, cache_key, symbol, fn)
    return loaded if loaded is not None else compiled
