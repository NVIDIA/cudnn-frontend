# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa.ref_cache``: off without the variable, a miss computes and stores, a hit is bitwise the stored value without a compute, VERIFY
recomputes and fails on a drift (``=0`` is off), the key covers what the reference depends on, and an entry is read with the weights-only
unpickler: a pickle payload or a foreign file planted at an entry path is refused as a miss, never executed.  The key covers the
RECIPE -- every test-tree file on the call stack at the call plus the file the oracle is defined in, hashed whole -- so an edit there
that leaves every hashed helper and the oracle's own source unchanged (two draws swapped) is a miss, also when a shared wrapper in
another module builds the oracle closure and makes the call; identical copies of one recipe share the key; the harness's own frames
(pytest, torch, the standard library) are skipped, not stopped at.  ``_REF_SOURCES`` is closed under its own imports."""

import importlib.util
import os
import re
import sys

import pytest
import torch

from sdpa import ref_cache

pytestmark = pytest.mark.L0


def _oracle(calls, value=1.0):
    def compute():
        calls.append(1)
        return dict(o=torch.full((4, 8), value, device="cuda"), stats=torch.arange(4, dtype=torch.float32, device="cuda"), amax=value)

    return compute


def test_off_without_the_variable_computes_every_time(monkeypatch):
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE", raising=False)
    calls = []
    for _ in range(2):
        ref_cache.cached_reference("unit", dict(seed=0), _oracle(calls))
    assert len(calls) == 2 and ref_cache.cache_dir() is None


def test_a_miss_stores_and_a_hit_returns_the_stored_value_without_a_compute(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    first = ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls))
    assert len(calls) == 1 and len(list(tmp_path.glob("unit-*.pt"))) == 1
    second = ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls, value=7.0))  # a hit never runs compute
    assert len(calls) == 1
    assert second["o"].device.type == "cuda" and torch.equal(second["o"], first["o"]) and torch.equal(second["stats"], first["stats"])
    assert second["amax"] == first["amax"] == 1.0


def test_the_key_covers_the_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    calls = []
    ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls))
    ref_cache.cached_reference("unit", dict(seed=2, s=512), _oracle(calls))  # another seed: another entry
    ref_cache.cached_reference("unit", dict(seed=1, s=1024), _oracle(calls))  # another shape: another entry
    assert len(calls) == 3 and len(list(tmp_path.glob("unit-*.pt"))) == 3


def test_verify_mode_recomputes_and_fails_on_a_drift(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "1")
    calls = []
    ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls))
    before = ref_cache.stats()["verified"]
    ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls))  # a hit: recomputed and compared bitwise
    assert len(calls) == 2 and ref_cache.stats()["verified"] == before + 1
    with pytest.raises(AssertionError, match="differs from a fresh compute"):
        ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls, value=1.0 + 2**-20))


def test_a_torn_file_is_recomputed_and_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    ref_cache.cached_reference("unit", dict(seed=4), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    path.write_bytes(b"not a torch file")
    value = ref_cache.cached_reference("unit", dict(seed=4), _oracle(calls))
    assert len(calls) == 2 and value["amax"] == 1.0 and path.stat().st_size > 16


def test_verify_is_an_on_off_switch(monkeypatch):
    for off in ("", "0", "false", "No", "OFF", " 0 "):
        monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", off)
        assert not ref_cache.verify(), repr(off)
    for on in ("1", "true", "YES", "on", "2"):
        monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", on)
        assert ref_cache.verify(), repr(on)
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY")
    assert not ref_cache.verify()


def test_verify_0_leaves_a_hit_unrecomputed(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "0")
    calls = []
    ref_cache.cached_reference("unit", dict(seed=5), _oracle(calls))
    before = ref_cache.stats()["verified"]
    ref_cache.cached_reference("unit", dict(seed=5), _oracle(calls))
    assert len(calls) == 1 and ref_cache.stats()["verified"] == before, "=0 is OFF: the hit is served, not recomputed"


def _entry_digest(path):
    return path.stem.rsplit("-", 1)[1]


def test_a_pickle_payload_planted_at_the_entry_path_is_refused_not_executed(monkeypatch, tmp_path):
    """The directory is shared and an entry's name is predictable, so an entry is read with torch's weights-only unpickler -- tensors
    and plain Python containers / scalars only.  A checkpoint carrying a pickle payload under this key's own digest is refused: a miss
    that recomputes and overwrites it, never an executed payload."""
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    marker = tmp_path / "payload-executed.marker"

    class _Payload:
        def __reduce__(self):  # unpickling this object calls open(marker, "w"): the marker exists iff the payload ran
            return (open, (str(marker), "w"))

    calls = []
    ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    torch.save(dict(digest=_entry_digest(path), key="{}", value=_Payload()), path)
    before = ref_cache.stats().get("refused", 0)
    value = ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))
    assert not marker.exists(), "the planted payload was executed"
    assert len(calls) == 2 and value["amax"] == 1.0 and ref_cache.stats()["refused"] == before + 1
    ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))  # the refused file was overwritten by a valid entry: a hit again
    assert len(calls) == 2


def test_a_foreign_file_at_the_entry_path_is_a_miss_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    ref_cache.cached_reference("unit", dict(seed=7), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    # not a dict at all / another key's entry / this key's digest without a value
    for foreign in (torch.zeros(3), dict(digest="another-key", value=1.0), dict(digest=_entry_digest(path))):
        torch.save(foreign, path)
        assert ref_cache.cached_reference("unit", dict(seed=7), _oracle(calls))["amax"] == 1.0
    assert len(calls) == 4


def test_a_nested_entry_round_trips_through_the_safe_loader(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    nested = (
        torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3),
        dict(n=1, x=2.5, none=None, s="s", b=True, l=[torch.zeros(2, device="cuda"), (3, 4.0)]),
        7,
    )
    calls = []

    def compute():
        calls.append(1)
        return nested

    first = ref_cache.cached_reference("nested", dict(seed=8), compute)
    second = ref_cache.cached_reference("nested", dict(seed=8), compute)
    assert len(calls) == 1 and ref_cache._same(first, second)
    assert isinstance(second, tuple) and isinstance(second[1]["l"], list) and isinstance(second[1]["l"][1], tuple) and second[1]["none"] is None
    assert second[0].device.type == "cuda" and second[1]["l"][0].device.type == "cuda"


def test_a_result_the_cache_cannot_represent_is_refused_at_store_time(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    with pytest.raises(TypeError, match=r"tensors and plain Python .* value\['obj'\] is a object"):
        ref_cache.cached_reference("unit", dict(seed=9), lambda: dict(o=torch.zeros(2), obj=object()))
    assert not list(tmp_path.glob("unit-*.pt")), "nothing the loader would refuse is ever written"


# ---- the key covers the RECIPE: the whole module that draws / quantizes the operands and calls the oracle

_RECIPE_MODULE = """
import torch

from sdpa import ref_cache

CALLS = []


def run(name):
    # The shape of a suite's run helper: draw the operands from one seed, quantize them, run the oracle through the cache.
    gen = torch.Generator().manual_seed(0)

    def draw():
        return torch.randn(8, generator=gen)

    def quant(x):
        return (x * 4).round() / 4

    {assign}
    q, do = quant(q32), quant(do32)

    def oracle():
        CALLS.append(1)
        return dict(o=q * 3 + do, lse=q.sum())

    fresh = dict(o=q * 3 + do, lse=q.sum())
    return ref_cache.cached_reference(name, dict(seed=0, n=8), oracle, device="cpu"), oracle, fresh
"""
_DRAW_Q_FIRST = "q32, do32 = draw(), draw()"
_DRAW_DO_FIRST = "do32, q32 = draw(), draw()"  # Q now takes the SECOND draw; ``draw`` / ``quant`` and the oracle's own source are unchanged


def _import_recipe(path, template, **fields):
    """The module ``template`` (formatted with ``fields``) written to ``path`` -- its own copy: a run of its own -- imported from
    there and registered under a name of its own, so another written module can import it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.format(**fields))
    spec = importlib.util.spec_from_file_location(f"ref_cache_recipe_{path.parent.name}_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_key_covers_the_recipe_that_produced_the_operands_not_only_its_hashed_helpers(monkeypatch, tmp_path):
    """Swap Q's and dO's draws in the run helper: ``draw`` / ``quant`` and the oracle's own source are unchanged, so a key that hashed
    only those answered the changed recipe from the OLD entry -- two hits, zero misses, the old O / LSE for a different Q.  The key
    covers the recipe's whole module: the changed recipe is a miss that computes and stores its own reference."""
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    original = _import_recipe(tmp_path / "a" / "recipe.py", _RECIPE_MODULE, assign=_DRAW_Q_FIRST)
    swapped = _import_recipe(tmp_path / "b" / "recipe.py", _RECIPE_MODULE, assign=_DRAW_DO_FIRST)
    value_a, oracle_a, fresh_a = original.run("recipe")
    value_b, oracle_b, fresh_b = swapped.run("recipe")
    assert not torch.equal(fresh_a["o"], fresh_b["o"]), "the swap changes Q: the two recipes have different references"
    assert ref_cache._key_digest(oracle_a, dict(seed=0, n=8)) != ref_cache._key_digest(oracle_b, dict(seed=0, n=8)), "one key for two recipes"
    assert len(original.CALLS) == 1 and len(swapped.CALLS) == 1, "the changed recipe is a MISS, never a hit on the old entry"
    assert ref_cache._same(value_a, fresh_a) and ref_cache._same(value_b, fresh_b) and not ref_cache._same(value_b, value_a)
    assert len(list((tmp_path / "cache").glob("recipe-*.pt"))) == 2


def test_the_key_is_stable_across_identical_copies_of_the_recipe(monkeypatch, tmp_path):
    """Two copies of one recipe at different paths -- a re-run, another checkout of the same tree, a sibling worker -- share the key:
    content is hashed, never a path or an mtime.  The second copy hits without a compute and gets the first one's values bitwise."""
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    first = _import_recipe(tmp_path / "run1" / "recipe.py", _RECIPE_MODULE, assign=_DRAW_Q_FIRST)
    second = _import_recipe(tmp_path / "run2" / "recipe.py", _RECIPE_MODULE, assign=_DRAW_Q_FIRST)
    value_1, oracle_1, fresh = first.run("stable")
    value_2, oracle_2, _ = second.run("stable")
    assert ref_cache._key_digest(oracle_1, dict(seed=0, n=8)) == ref_cache._key_digest(oracle_2, dict(seed=0, n=8))
    assert len(first.CALLS) == 1 and len(second.CALLS) == 0, "the second copy is a hit"
    assert ref_cache._same(value_1, fresh) and ref_cache._same(value_2, fresh)
    assert len(list((tmp_path / "cache").glob("stable-*.pt"))) == 1


# ---- the recipe may be one module away from both the oracle and the call: a shared wrapper builds the closure and calls the cache

_WRAPPER_MODULE = """
import torch

from sdpa import ref_cache


@torch.no_grad()  # a LIBRARY frame between the recipe and this wrapper: the stack walk skips it, it does not stop there
def oracle_through_cache(name, key, calls, q, do):
    # The shape of a shared oracle wrapper: handed the operands, it BUILDS the oracle closure and makes the call -- so the file the
    # oracle is defined in and the direct caller of ``cached_reference`` are both this module, and neither is the recipe.
    def oracle():
        calls.append(1)
        return dict(o=q * 3 + do, lse=q.sum())

    return ref_cache.cached_reference(name, key, oracle, device="cpu"), oracle
"""

_RECIPE_THROUGH_WRAPPER = """
import torch

from {wrapper} import oracle_through_cache

CALLS = []


def run(name):
    gen = torch.Generator().manual_seed(0)

    def draw():
        return torch.randn(8, generator=gen)

    def quant(x):
        return (x * 4).round() / 4

    {assign}
    q, do = quant(q32), quant(do32)
    fresh = dict(o=q * 3 + do, lse=q.sum())
    value, oracle = oracle_through_cache(name, dict(seed=0, n=8), CALLS, q, do)
    return value, oracle, fresh
"""


def test_the_key_covers_a_recipe_whose_oracle_a_shared_wrapper_builds_and_calls(monkeypatch, tmp_path):
    """One shared wrapper builds the oracle closure from the operands it is handed and calls the cache; two recipes call it, differing
    only in the order of Q's and dO's draws.  Seen from the callable and from the direct caller alone the two are identical (one
    wrapper file, one source), so a key over those answered the swapped recipe from the old entry.  The key walks the whole call
    stack -- through the wrapper's torch.no_grad frame -- so the recipe module is in it and the swapped recipe is a miss."""
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    wrapper = _import_recipe(tmp_path / "shared" / "oracle_wrapper.py", _WRAPPER_MODULE)
    original = _import_recipe(tmp_path / "a" / "recipe.py", _RECIPE_THROUGH_WRAPPER, wrapper=wrapper.__name__, assign=_DRAW_Q_FIRST)
    swapped = _import_recipe(tmp_path / "b" / "recipe.py", _RECIPE_THROUGH_WRAPPER, wrapper=wrapper.__name__, assign=_DRAW_DO_FIRST)
    value_a, oracle_a, fresh_a = original.run("wrapped")
    value_b, oracle_b, fresh_b = swapped.run("wrapped")
    assert not torch.equal(fresh_a["o"], fresh_b["o"]), "the swap changes Q: the two recipes have different references"
    assert ref_cache._key_digest(oracle_a, dict(seed=0, n=8)) == ref_cache._key_digest(oracle_b, dict(seed=0, n=8)), "without the stack the two oracles are one"
    assert len(original.CALLS) == 1 and len(swapped.CALLS) == 1, "the swapped recipe is a MISS, never a hit on the old entry"
    assert ref_cache._same(value_a, fresh_a) and ref_cache._same(value_b, fresh_b) and not ref_cache._same(value_b, value_a)
    assert len(list((tmp_path / "cache").glob("wrapped-*.pt"))) == 2


def test_the_recipe_files_are_the_test_trees_frames_never_the_harness():
    """What the recipe digest hashes for a call made from here: the file the oracle is defined in and this module (the caller), never a
    frame of pytest, pluggy, torch or the standard library -- the harness is skipped, not stopped at (a library frame between two
    test-tree frames must not end the walk, see the wrapper pin above)."""
    import sysconfig

    import _pytest
    import pluggy

    # Where the harness actually lives -- site-packages, Debian's dist-packages, a venv: located by module, not by a path spelling.
    harness = tuple(os.path.join(os.path.dirname(os.path.realpath(m.__file__)), "") for m in (_pytest, pluggy, torch))
    harness += (os.path.join(os.path.realpath(sysconfig.get_paths()["stdlib"]), ""),)
    frame, raw = sys._getframe(), []
    while frame is not None:
        raw.append(os.path.realpath(frame.f_code.co_filename))
        frame = frame.f_back
    assert any(f.startswith(harness) for f in raw), "pytest's own frames are on this stack"
    files = ref_cache._recipe_files(lambda: None, sys._getframe())
    assert os.path.realpath(__file__) in files, files
    assert all(os.path.isfile(f) and not f.startswith(harness) for f in files), files


# ---- _REF_SOURCES is closed under its own imports: a listed reference module imports no unlisted test-tree module

_FROM_IMPORT = re.compile(r"^\s*from\s+(\.+[\w.]*|sdpa(?:\.[\w.]+)?)\s+import\s+\(?([^\n]*)", re.M)
_PLAIN_IMPORT = re.compile(r"^\s*import\s+(sdpa(?:\.[\w.]+)?)\b", re.M)


def _imported_test_tree_files(rel):
    """The test-tree files the module ``rel`` (relative to test/python) imports through its relative and ``sdpa.``-prefixed import
    statements, in-function (lazy) imports included: ``from .x import``, ``from sdpa.x import``, ``from sdpa import x, y as z``,
    ``import sdpa.x``.  Imports of ``cudnn`` / ``torch`` / the standard library are not test-tree files."""
    root = ref_cache.Path(ref_cache.__file__).resolve().parent.parent
    found = set()

    def add(candidate):
        if (root / f"{candidate}.py").is_file():
            found.add(f"{candidate}.py")
            return True
        return False

    for module, names in _FROM_IMPORT.findall((root / rel).read_text()):
        if module.startswith("."):
            base = ref_cache.Path(rel).parent
            for _ in range(len(module) - len(module.lstrip(".")) - 1):
                base = base.parent
            target = base.joinpath(*module.strip(".").split(".")) if module.strip(".") else base
        else:
            target = ref_cache.Path(*module.split("."))
        if not add(target.as_posix()):  # `from <package> import a, b as c`: each name may be a module of that package
            for name in names.split("#")[0].strip("() ").split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    add((target / name).as_posix())
    for module in _PLAIN_IMPORT.findall((root / rel).read_text()):
        add(ref_cache.Path(*module.split(".")).as_posix())
    return found


def test_the_reference_sources_are_closed_under_their_own_imports():
    """Every listed reference source exists, and whatever test-tree module a listed source imports -- a quantizer, a helper, a
    constant -- is listed too, else an edit there changes the reference while the sources digest stands still: a hashed set that
    reads as complete and is not (``mxfp8_ref.py`` reached ``mxfp8_quant.py`` and ``mxfp8.py`` unlisted)."""
    root = ref_cache.Path(ref_cache.__file__).resolve().parent.parent
    listed = set(ref_cache._REF_SOURCES)
    assert "sdpa/ref_cache.py" in listed, "a change of the cache's own key or format re-keys every entry"
    assert all((root / rel).is_file() for rel in listed), sorted(rel for rel in listed if not (root / rel).is_file())
    unlisted = sorted(f"{rel} -> {dep}" for rel in listed for dep in _imported_test_tree_files(rel) if dep not in listed)
    assert not unlisted, unlisted
    assert _imported_test_tree_files("sdpa/fp8_ref.py") >= {"sdpa/helpers.py", "sdpa/fp16_ref.py"}, "the parser sees a relative import"


def test_the_fp8_suites_recipe_imports_listed_reference_sources_only():
    """The one live caller: what the stack cannot see is a helper the recipe imports and has already returned from when the oracle is
    called (its ``quant`` scales through ``sdpa/helpers.py``), so every ``sdpa.`` module the suite imports must be a listed reference
    source, or an edit there is invisible to the key."""
    imported = _imported_test_tree_files("sdpa/frost/test_sdpa_bwd_fp8_sm107.py")
    assert {"sdpa/helpers.py", "sdpa/ref_cache.py"} <= imported, imported
    assert imported <= set(ref_cache._REF_SOURCES), sorted(imported - set(ref_cache._REF_SOURCES))
