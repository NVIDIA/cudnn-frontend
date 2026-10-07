# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An OPT-IN, node-local disk cache of reference (oracle) outputs for the SDPA test suites.

The reference of a cell is a deterministic function of (the reference code, the device class, the cell's inputs).  The
first process that needs it computes and stores it; every later process -- the next sweep, a sibling xdist worker, the same
module re-run after an unrelated edit -- loads it instead.  The stored tensors ARE a previous run's reference values
(bitwise, not an approximation), so the test's tolerances are untouched.  A hit saves the oracle's GPU time only (a few
seconds per 200-cell module); the point is the VERIFY mode below, which turns every later run into a bitwise check of the
reference against its own past.

Key = sha256 over
  * the CONTENT of every reference source file (``_REF_SOURCES``) AND the source of the caller's ``compute`` callable (a
    test-module helper is covered too);
  * the device's SM count (torch's CUDA Philox lays draws out by grid size, which follows the SM count: one seed is a
    different dataset on two parts with different SM counts), the device name and torch's version (cuBLAS choices);
  * the caller's ``key`` -- shape, seed, dtype / recipe, mask flags -- everything the inputs or the reference depend on.

The SOURCE of ``compute`` is hashed, not its closure: the tensors a callable captured are covered only through ``key``, so the
caller names what produced them -- shape, seed, dtype AND a digest of its own input-generation code (how it draws and
quantizes the operands; ``test_sdpa_bwd_fp8_sm107.py::_inputs_recipe_digest`` is the pattern).  A change there without a key
change HITS a stale entry unless VERIFY is on.

Off unless ``CUDNN_TEST_REF_CACHE=<dir>`` names a directory.  Make it node-LOCAL (a scratch disk or tmpfs of the node), never
a cross-site network file system -- the same rule as the compiled-plan cache.  ``CUDNN_TEST_REF_CACHE_VERIFY=1`` recomputes
on every hit and asserts bitwise equality (the self-test of the cache: a reference that drifts fails the test that uses it).
Entries are written atomically (``torch.save`` to a temp file + ``os.replace``), so concurrent workers never read a torn file.
"""

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

_ENV_DIR = "CUDNN_TEST_REF_CACHE"
_ENV_VERIFY = "CUDNN_TEST_REF_CACHE_VERIFY"
_REF_SOURCES = (
    "sdpa/fp8_ref.py",
    "sdpa/fp16_ref.py",
    "sdpa/mxfp8_ref.py",
    "sdpa/helpers.py",
    "gated_attention_block/cutedsl/gated_block_reference.py",
)
_stats = {"hit": 0, "miss": 0, "verified": 0, "bytes_loaded": 0, "bytes_stored": 0}
_src_digest: Optional[str] = None


def cache_dir() -> Optional[Path]:
    d = os.environ.get(_ENV_DIR)
    return Path(d) if d else None


def verify() -> bool:
    return bool(os.environ.get(_ENV_VERIFY))


def _sources_digest() -> str:
    """One digest over the reference sources' CONTENT (not mtime: the mtime of a checkout copy says nothing)."""
    global _src_digest
    if _src_digest is None:
        root = Path(__file__).resolve().parent.parent  # test/python
        h = hashlib.sha256()
        for rel in _REF_SOURCES:
            p = root / rel
            h.update(rel.encode())
            h.update(p.read_bytes() if p.exists() else b"<absent>")
        _src_digest = h.hexdigest()[:16]
    return _src_digest


def _device_tag() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"{p.name.replace(' ', '_')}-cc{p.major}{p.minor}-sm{p.multi_processor_count}-torch{torch.__version__}"


def _key_digest(compute: Callable, key: Dict[str, Any]) -> str:
    try:
        src = inspect.getsource(compute)
    except (OSError, TypeError):
        src = repr(compute)
    blob = json.dumps(
        dict(key=key, device=_device_tag(), sources=_sources_digest(), compute=hashlib.sha256(src.encode()).hexdigest()[:16]), sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def cached_reference(name: str, key: Dict[str, Any], compute: Callable[[], Any], *, device="cuda"):
    """``compute()`` -- or its stored result when the cache is on and holds this key.

    ``name`` is a human label for the entry's file name; ``key`` must name EVERY input the result depends on (shape, seed,
    dtype / recipe, mask flags, the scalars) -- including a digest of the code that produced the captured operands: only
    ``compute``'s own source is hashed here, never its closure.  The result may be a tensor or a (nested) dict / list / tuple of tensors and
    plain Python values; tensors come back on ``device``.  Under ``CUDNN_TEST_REF_CACHE_VERIFY=1`` a hit is recomputed and
    must be bitwise the stored value, else the caller's test fails here."""
    root = cache_dir()
    if root is None:
        return compute()
    digest = _key_digest(compute, key)
    path = root / f"{name}-{digest}.pt"
    if path.exists():
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 -- a torn or foreign file: recompute and overwrite
            stored = None
        if stored is not None and stored.get("digest") == digest:
            _stats["hit"] += 1
            _stats["bytes_loaded"] += path.stat().st_size
            value = _to_device(stored["value"], device)
            if verify():
                fresh = compute()
                assert _same(_to_device(fresh, "cpu"), _to_device(value, "cpu")), f"ref_cache: the stored reference {path.name} differs from a fresh compute"
                _stats["verified"] += 1
            return value
    value = compute()
    _stats["miss"] += 1
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(root), prefix=path.stem + ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(dict(digest=digest, key=json.dumps(key, sort_keys=True, default=str), value=_to_device(value, "cpu")), tmp)
        os.replace(tmp, path)
        _stats["bytes_stored"] += path.stat().st_size
    except Exception:  # noqa: BLE001 -- the cache must never fail a test
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return value


def stats() -> Dict[str, int]:
    return dict(_stats)
