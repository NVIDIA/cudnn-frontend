# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tier lists in tiers/ (README.md there) stay in step with the tree -- host-only, no kernel runs.

Tripwire: every node id a list names still COLLECTS.  The modules the lists name are collected once in a child pytest (the ids are
what ``pytest --collect-only -q`` prints from test/python), so a renamed or re-parametrized test cannot silently drop out of a tier.
Selection pins: ``-m nightly_only`` / ``-m smoke`` select exactly the listed cells of a module -- the conftest hook applies the
markers from the right list and runs before pytest's ``-m`` filter -- and the hook deselects nothing on its own."""

import csv
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0

_HERE = os.path.dirname(os.path.abspath(__file__))
_TIERS = os.path.join(_HERE, "tiers")
_LISTS = sorted(name for name in os.listdir(_TIERS) if name.endswith(".txt"))
_SMOKE_ARCHES = sorted(name[len("smoke_") : -len(".txt")] for name in _LISTS if name.startswith("smoke_"))


def _ids(name):
    with open(os.path.join(_TIERS, name)) as fh:
        return {line.strip() for line in fh if line.strip() and not line.lstrip().startswith("#")}


def _collect_run(args, env=None, ok=(0,)):
    """A child pytest collect-only run for ``args`` (from test/python, addopts cleared, no cache; a fresh test run of its own)."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST_") and k != "CUDNN_TEST_RUN_ID"}
    child_env.update(env or {})
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *args]
    done = subprocess.run(cmd, cwd=_HERE, env=child_env, capture_output=True, text=True, timeout=1800)
    assert done.returncode in ok, f"child collection exited {done.returncode}:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return done


def _collected(done):
    return {line.strip() for line in done.stdout.splitlines() if "::" in line and not line.startswith(" ")}


def _collect(args, env=None, ok=(0,)):
    """The node ids a child pytest collects for ``args``."""
    return _collected(_collect_run(args, env, ok))


def _module_with_most_ids(name):
    ids = _ids(name)
    per_module = {}
    for nodeid in ids:
        per_module[nodeid.split("::")[0]] = per_module.get(nodeid.split("::")[0], 0) + 1
    module = max(per_module, key=per_module.get)
    return module, {nodeid for nodeid in ids if nodeid.startswith(module + "::")}


@pytest.fixture(scope="module")
def collected_ids():
    """One collection of every module any list names -- the expensive step, shared by the tripwires."""
    modules = sorted({nodeid.split("::")[0] for name in _LISTS for nodeid in _ids(name)})
    missing = [m for m in modules if not os.path.isfile(os.path.join(_HERE, m))]
    assert not missing, f"the tier lists name modules that no longer exist: {missing}"
    return _collect(modules)


@pytest.mark.parametrize("name", _LISTS)
def test_every_listed_cell_collects(name, collected_ids):
    ids = _ids(name)
    assert ids, f"tiers/{name} lists nothing"
    stale = sorted(ids - collected_ids)
    arch = name[len("smoke_") : -len(".txt")] if name.startswith("smoke_") else None
    if stale and arch:
        # Cells a module collects only on that arch (test_mhas_v2's cc 10.7 functions) collect under the tiers' own
        # override (CUDNN_TEST_TIER_ARCH): re-collect just their modules under it before calling them stale.
        stale = sorted(set(stale) - _collect(sorted({s.split("::")[0] for s in stale}), env={"CUDNN_TEST_TIER_ARCH": arch}))
    assert not stale, f"{len(stale)} id(s) in tiers/{name} no longer collect (renamed / re-parametrized? fix the list in the same commit):\n" + "\n".join(
        stale[:25]
    )


def test_nightly_only_rows_name_a_reason_and_a_kept_twin():
    with open(os.path.join(_TIERS, "nightly_only_reasons.tsv"), newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header, rows = rows[0], rows[1:]
    assert header[0] == "nodeid" and len(header) == 3
    assert {row[0] for row in rows} == _ids("nightly_only.txt"), "nightly_only_reasons.tsv and nightly_only.txt list different cells"
    assert all(len(row) == 3 and row[1].strip() and row[2].strip() for row in rows), "every demoted cell needs a reason and a kept twin"


@pytest.mark.parametrize("arch", _SMOKE_ARCHES)
def test_smoke_code_paths_cover_exactly_the_listed_cells(arch):
    with open(os.path.join(_TIERS, f"smoke_{arch}_code_paths.tsv"), newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))[1:]
    assert {row[0] for row in rows} == _ids(f"smoke_{arch}.txt")
    assert all(len(row) == 2 and row[1].strip() for row in rows), "every SMOKE cell names the code path it stands for"


def test_nightly_only_marker_selects_exactly_the_listed_cells_and_deselects_nothing_else():
    module, listed = _module_with_most_ids("nightly_only.txt")
    assert _collect(["-m", "nightly_only", module]) == listed
    everything = _collect([module])
    assert listed < everything, "the tier hook only adds markers; the module's other cells must still collect"


@pytest.mark.parametrize("arch", _SMOKE_ARCHES)
def test_smoke_marker_selects_exactly_the_listed_cells_of_its_arch(arch):
    module, listed = _module_with_most_ids(f"smoke_{arch}.txt")
    assert _collect(["-m", "smoke", module], env={"CUDNN_TEST_TIER_ARCH": arch}) == listed


def test_an_arch_without_a_smoke_list_has_an_empty_smoke_tier_and_says_so():
    module, _ = _module_with_most_ids(f"smoke_{_SMOKE_ARCHES[0]}.txt")
    done = _collect_run(["-m", "smoke", module], env={"CUDNN_TEST_TIER_ARCH": "cc999"}, ok=(5,))  # 5 = nothing selected
    assert _collected(done) == set()
    # ... and one stderr line names the missing list, so an empty `-m smoke` run is not a silent exit 5
    assert "[tiers] no SMOKE list for this GPU (smoke_cc999.txt is not in" in done.stderr, done.stderr[-2000:]
    # the plain collection of the same module says nothing (the warning is tied to a `-m` expression naming smoke)
    assert "[tiers]" not in _collect_run([module], env={"CUDNN_TEST_TIER_ARCH": "cc999"}).stderr


def test_the_tier_hook_is_pinned_ahead_of_the_mark_plugins_deselection(request):
    """`-m smoke` / `-m "not nightly_only"` see the tier markers only if the conftest's pytest_collection_modifyitems runs before
    the built-in mark plugin's (deselect_by_mark).  Plain impls run last-registered-first, which puts the conftest first today;
    the explicit tryfirst keeps it so whatever a later pytest does to its own impl."""
    path = os.path.join(_HERE, "conftest.py")
    plugin = next(p for p in request.config.pluginmanager.get_plugins() if os.path.abspath(getattr(p, "__file__", "") or "") == path)
    impl = getattr(plugin.pytest_collection_modifyitems, "pytest_impl", None)
    assert impl and impl.get("tryfirst"), "test/python/conftest.py::pytest_collection_modifyitems must be @pytest.hookimpl(tryfirst=True)"
