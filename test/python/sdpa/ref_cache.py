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
  * the CONTENT of every reference source file (``_REF_SOURCES`` -- this module included, so a change of the cache's own format
    or key re-keys every entry);
  * the CONTENT of the RECIPE's files, hashed WHOLE (``_recipe_files``): the file ``compute`` is defined in and every file on the
    CALL STACK at the call that is not the interpreter's own -- the recipe module, a shared wrapper that built the oracle closure,
    the test module; frames of the standard library and of site-packages (pytest, pluggy, torch: the harness) are skipped, not
    stopped at, so a decorator's library frame between two recipe files does not end the walk.  The recipe is everything between
    the seed and the oracle call -- the draw order, the quantization call sites, the oracle wrapper, a module constant -- and none
    of it is visible from the callable or from ``key`` (the tensors a callable captured are not hashed), so the whole files stand
    in for it: an edit anywhere in them is a MISS, never a stale hit.  The probes that found the gaps: Q's and dO's draws swapped
    in the run helper, every hashed helper and the oracle's own source unchanged -- a key over the leaf helpers answered the
    changed recipe from the old entry; then the same swap one module away, a shared wrapper building the closure and making the
    call -- a key over the callable's file and the direct caller's answered it too.  What the stack cannot see is a helper the
    recipe IMPORTS and has already returned from when the oracle is called (a quantizer in another module): such a module is
    covered through ``_REF_SOURCES`` only -- list it there (the fp8 suite's ``quant`` scales through ``sdpa/helpers.py``, listed);
  * the device's SM count (torch's CUDA Philox lays draws out by grid size, which follows the SM count: one seed is a
    different dataset on two parts with different SM counts), the device name and torch's version (cuBLAS choices);
  * the caller's ``key`` -- what the files' content does not pin because it arrives as a parameter: shape, seed, dtype /
    recipe, mask flags, the scalars.

The cost of the coarse recipe digest is a miss on every edit of a recipe file (the first run after it repopulates; a hit saves
seconds) and one key per call path (a driver that runs pytest in-process is a frame too); the gain is a key that names the
inputs' recipe literally.  Content is hashed, never a path or an mtime, so identical copies of one tree -- another checkout, a
sibling worker, the next sweep -- share every entry.  Files are read as they are on disk at the call (the reference sources once
per process, at the first call), NOT as they were imported: an edit made WHILE a cache-enabled run is in progress keys that run's
later entries by the new content although the old, already-imported code computed them, and the next run -- now running the new
code -- hits them.  After editing during a run, run once with VERIFY on, or point the cache at a fresh directory.

Off unless ``CUDNN_TEST_REF_CACHE=<dir>`` names a directory.  Make it node-LOCAL (a scratch disk or tmpfs of the node), never
a cross-site network file system -- the same rule as the compiled-plan cache.  ``CUDNN_TEST_REF_CACHE_VERIFY=1`` recomputes
on every hit and asserts bitwise equality (the self-test of the cache: a reference that drifts fails the test that uses it);
``=0`` / ``false`` / ``no`` / ``off`` / empty leave it OFF with the variable set (``_env_switch``).  Entries are written
atomically (``torch.save`` to a temp file + ``os.replace``), so concurrent workers never read a torn file.

The directory is TRUSTED like the test tree: writable by the account or CI job that runs the tests, never a world-writable path --
entry names are predictable, and whoever can write it decides which reference values a run compares against.  What the loader
can be made to EXECUTE is bounded regardless: an entry is read with torch's weights-only unpickler, which admits tensors and
plain Python containers / scalars only -- all ``cached_reference`` ever stores; anything else is refused at store time -- so a
pickle payload planted at an entry path is refused as a miss (recomputed and overwritten), never run.
"""

import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

_ENV_DIR = "CUDNN_TEST_REF_CACHE"
_ENV_VERIFY = "CUDNN_TEST_REF_CACHE_VERIFY"
_OFF_WORDS = frozenset({"", "0", "false", "no", "off"})
_REF_SOURCES = (  # CLOSED under its own relative / `sdpa.` imports (pinned by test_ref_cache.py): a listed module's import joins the list
    "sdpa/ref_cache.py",
    "sdpa/fp8_ref.py",
    "sdpa/fp16_ref.py",
    "sdpa/mxfp8_ref.py",
    "sdpa/helpers.py",
    "sdpa/fp16.py",
    "sdpa/fp8.py",
    "sdpa/mxfp8.py",
    "sdpa/mxfp8_quant.py",
    "sdpa/block_scale_o_ref.py",
    "sdpa/random_config.py",
    "sdpa/softmax_knobs.py",
    "gated_attention_block/cutedsl/gated_block_reference.py",
)
_stats = {"hit": 0, "miss": 0, "verified": 0, "refused": 0, "bytes_loaded": 0, "bytes_stored": 0}
_src_digest: Optional[str] = None


def cache_dir() -> Optional[Path]:
    d = os.environ.get(_ENV_DIR)
    return Path(d) if d else None


def _env_switch(name: str) -> bool:
    """An on / off environment variable: unset, empty, ``0``, ``false``, ``no`` or ``off`` (any case, surrounding blanks ignored)
    is OFF; every other value (``1``, ``true``, ``yes``, ...) is ON."""
    return os.environ.get(name, "").strip().lower() not in _OFF_WORDS


def verify() -> bool:
    return _env_switch(_ENV_VERIFY)


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


def _file_digest(path: str) -> str:
    """A digest of one file's CONTENT as it is on disk NOW, read on every call (a test module is ~100 KB: hashing it per reference is
    nothing next to the oracle).  No memo: an (mtime, size) stamp misses a same-size rewrite within the mtime granularity.  What the
    per-call read leaves open -- an edit DURING a cache-enabled run re-keys that run's later entries -- is in the module docstring."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "<absent>"


def _source_file_of(obj) -> Optional[str]:
    """The file ``obj`` was defined in (a ``functools.partial`` by its function), or None for a builtin / a callable without one."""
    for candidate in (obj, getattr(obj, "func", None)):
        if candidate is None:
            continue
        try:
            path = inspect.getsourcefile(candidate)
        except (OSError, TypeError):
            continue
        if path and os.path.isfile(path):
            return path
    return None


_lib_roots: Optional[tuple] = None


def _library_roots() -> tuple:
    """Where the interpreter's own code lives, resolved once: the standard library, the site-packages directories (pytest, pluggy,
    torch, ...) and the scripts directory the ``pytest`` entry point runs from, each with a trailing separator.  A call-stack frame
    under one of them is the harness, not a recipe.  Deliberately NOT ``sys.prefix``: a system interpreter's prefix is ``/usr``,
    under which a checkout may well live, and a root that swallowed the test tree would silently empty every recipe."""
    global _lib_roots
    if _lib_roots is None:
        import site
        import sysconfig

        paths = sysconfig.get_paths()
        roots = {paths.get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib", "scripts")}
        roots.add(os.path.dirname(sys.executable))
        for getter in (site.getsitepackages, site.getusersitepackages):
            try:
                found = getter()
            except Exception:  # noqa: BLE001 -- a stripped-down interpreter without these helpers
                continue
            roots.update(found if isinstance(found, (list, tuple)) else [found])
        _lib_roots = tuple(os.path.join(os.path.realpath(r), "") for r in roots if r)
    return _lib_roots


def _recipe_files(compute: Callable, frame) -> list:
    """The files a reference's RECIPE lives in, realpaths: the file ``compute`` is defined in, then every file on the call stack from
    ``frame`` (the caller of ``cached_reference``) outward that is not the interpreter's own -- the recipe module, a shared wrapper
    that built the closure, the test module.  Library frames (``_library_roots``) are SKIPPED, not stopped at: a decorator's frame
    between two recipe files must not end the walk.  Frames without a file (``<string>``, frozen importlib) contribute nothing."""
    roots, files = _library_roots(), []

    def add(path):
        if path and os.path.isfile(path):
            path = os.path.realpath(path)
            if path not in files and not path.startswith(roots):
                files.append(path)

    add(_source_file_of(compute))
    while frame is not None:
        add(frame.f_code.co_filename)
        frame = frame.f_back
    return files


def _recipe_digest(compute: Callable, frame=None) -> str:
    """One digest over the CONTENT of the recipe's files (``_recipe_files``), order-free: the same files reached in another call order
    are one recipe.  ``<no-source>`` when nothing resolves (a builtin ``compute`` and no caller frame)."""
    digests = sorted(_file_digest(path) for path in _recipe_files(compute, frame))
    return "+".join(digests) if digests else "<no-source>"


def _device_tag() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"{p.name.replace(' ', '_')}-cc{p.major}{p.minor}-sm{p.multi_processor_count}-torch{torch.__version__}"


def _key_digest(compute: Callable, key: Dict[str, Any], frame=None) -> str:
    try:
        src = inspect.getsource(compute)
    except (OSError, TypeError):
        src = repr(compute)
    blob = json.dumps(
        dict(
            key=key,
            device=_device_tag(),
            sources=_sources_digest(),
            recipe=_recipe_digest(compute, frame),
            compute=hashlib.sha256(src.encode()).hexdigest()[:16],
        ),
        sort_keys=True,
        default=str,
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


_PLAIN_LEAVES = (torch.Tensor, bool, int, float, str, type(None))
_MISS = object()


def _unstorable(obj, where="value") -> Optional[str]:
    """Where ``obj`` first leaves what the cache stores -- tensors and plain Python scalars / strings / None in dicts (string
    keys), lists and tuples, exactly what the weights-only loader admits on the way back -- or None when it is storable."""
    if isinstance(obj, _PLAIN_LEAVES):
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                return f"{where} has a {type(k).__name__} key {k!r}"
            bad = _unstorable(v, f"{where}[{k!r}]")
            if bad:
                return bad
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad = _unstorable(v, f"{where}[{i}]")
            if bad:
                return bad
        return None
    return f"{where} is a {type(obj).__name__}"


def _load_entry(path: Path, digest: str):
    """The value stored for ``digest`` at ``path``, or ``_MISS``.  The directory is shared and an entry's name is predictable, so
    the file is read with torch's weights-only unpickler -- tensors and plain Python containers / scalars, what ``_unstorable``
    admitted on the way in -- and anything else (a torn file, a foreign file, a pickle payload planted at the entry path, another
    key's entry) is REFUSED: a miss that recomputes and overwrites it, never a crash and never an executed payload."""
    try:
        stored = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 -- torn, foreign, or refused by the weights-only unpickler
        return _MISS
    if not isinstance(stored, dict) or stored.get("digest") != digest or "value" not in stored:
        return _MISS
    return stored["value"]


def cached_reference(name: str, key: Dict[str, Any], compute: Callable[[], Any], *, device="cuda"):
    """``compute()`` -- or its stored result when the cache is on and holds this key.

    ``name`` is a human label for the entry's file name; ``key`` must name EVERY input the result depends on that arrives as a
    parameter (shape, seed, dtype / recipe, mask flags, the scalars).  The RECIPE -- the file ``compute`` is defined in and every
    test-tree file on the call stack at this call: the recipe module, a shared wrapper that built the closure, the test module -- is
    hashed WHOLE (``_recipe_files``), so the code that drew and quantized the captured operands needs no digest of its own: an edit
    anywhere in it is a miss (a helper it imported and has already returned from is covered through ``_REF_SOURCES`` only).  The
    result may be a tensor or a (nested) dict / list / tuple of tensors and
    plain Python scalars / strings / None -- anything else raises ``TypeError`` at store time (with the cache off nothing is
    checked); tensors come back on ``device``.  An entry the weights-only loader refuses is a miss (``_load_entry``).  Under
    ``CUDNN_TEST_REF_CACHE_VERIFY=1`` a hit is recomputed and must be bitwise the stored value, else the caller's test fails here."""
    root = cache_dir()
    if root is None:
        return compute()
    digest = _key_digest(compute, key, sys._getframe(1))  # the caller's frame: the recipe's files are on the stack behind it
    path = root / f"{name}-{digest}.pt"
    if path.exists():
        value = _load_entry(path, digest)
        if value is _MISS:
            _stats["refused"] += 1
        else:
            _stats["hit"] += 1
            _stats["bytes_loaded"] += path.stat().st_size
            value = _to_device(value, device)
            if verify():
                fresh = compute()
                assert _same(_to_device(fresh, "cpu"), _to_device(value, "cpu")), f"ref_cache: the stored reference {path.name} differs from a fresh compute"
                _stats["verified"] += 1
            return value
    value = compute()
    _stats["miss"] += 1
    bad = _unstorable(value)
    if bad:
        raise TypeError(f"ref_cache: {name}: a reference must be tensors and plain Python scalars / strings / None in dicts, lists and tuples -- {bad}")
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
