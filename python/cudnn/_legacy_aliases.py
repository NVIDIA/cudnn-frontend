# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-system aliases for the pre-1.27 CuTeDSL GEMM module paths.

The 1.27 reorganization ("Reorganize Gemm fusion") moved the CuTeDSL GEMM
fusion packages under ``cudnn.gemm.cutedsl.*``. The supported entry point —
``from cudnn import <symbol>`` — never changed, but code importing modules by
their old dotted paths (``import cudnn.grouped_gemm``,
``from cudnn.grouped_gemm.grouped_gemm_wgrad.api import ...``) broke: the
attribute-level aliases in ``cudnn.__getattr__`` cannot satisfy an import
statement, because ``import a.b`` resolves ``a.b`` through the import system,
not through ``getattr(a, "b")``.

A table of eager ``sys.modules[legacy] = canonical`` entries can't fix this
either without importing every CuTeDSL package (and therefore torch and
nvidia-cutlass-dsl) at ``import cudnn`` time, which
``test_import_boundaries.py`` forbids. So the aliases are installed as a
:class:`importlib.abc.MetaPathFinder` that redirects a legacy path to its
canonical module lazily, at the moment the legacy path is first imported:

* ``sys.modules[legacy_name] is sys.modules[canonical_name]`` after the
  import, so ``is`` checks and monkeypatching see one module, not a copy;
* a :class:`DeprecationWarning` names the canonical path on first use;
* ``import cudnn`` itself stays framework-free.

The loader hands the already-imported canonical module back from
``create_module``; the import machinery only fills module attributes that are
unset, so the canonical module keeps its own ``__name__`` / ``__spec__`` /
``__path__`` and merely gains a second ``sys.modules`` key.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

# Legacy package roots mapped to their canonical packages. Descendants follow
# the root: child modules keep their names, except the per-fusion subpackages,
# which dropped the redundant family prefix in the move
# (``grouped_gemm.grouped_gemm_wgrad`` -> ``grouped.wgrad``);
# _STRIP_CHILD_PREFIX records that rename.
_LEGACY_ROOTS = {
    "cudnn.gemm_amax": "cudnn.gemm.cutedsl.dense.amax",
    "cudnn.gemm_swiglu": "cudnn.gemm.cutedsl.dense.swiglu",
    "cudnn.gemm_srelu": "cudnn.gemm.cutedsl.dense.srelu",
    "cudnn.gemm_dsrelu": "cudnn.gemm.cutedsl.dense.dsrelu",
    "cudnn.gemm_proj_rope_mxfp8": "cudnn.gemm.cutedsl.dense.proj_rope_mxfp8",
    "cudnn.grouped_gemm": "cudnn.gemm.cutedsl.grouped",
    "cudnn.discrete_grouped_gemm": "cudnn.gemm.cutedsl.discrete_grouped",
}

_STRIP_CHILD_PREFIX = {
    "cudnn.grouped_gemm": "grouped_gemm_",
    "cudnn.discrete_grouped_gemm": "discrete_grouped_gemm_",
    # ``from cudnn.grouped_gemm import grouped_gemm_wgrad`` resolves the child
    # against the parent's __name__, which is already canonical — so the
    # prefixed child names must also be aliased under the canonical roots.
    "cudnn.gemm.cutedsl.grouped": "grouped_gemm_",
    "cudnn.gemm.cutedsl.discrete_grouped": "discrete_grouped_gemm_",
}


def _canonical_name(fullname: str) -> "str | None":
    """Map a legacy dotted path to its canonical one; None if not legacy."""
    # The finder sits at sys.meta_path[0] and sees every import in the
    # process; get non-cudnn names out of the way on one comparison.
    if not fullname.startswith("cudnn."):
        return None
    for root, canonical_root in _LEGACY_ROOTS.items():
        if fullname == root:
            return canonical_root
        if fullname.startswith(root + "."):
            rest = fullname[len(root) + 1 :].split(".")
            prefix = _STRIP_CHILD_PREFIX.get(root)
            if prefix and rest[0].startswith(prefix):
                rest[0] = rest[0][len(prefix) :]
            return ".".join([canonical_root, *rest])
    # Prefixed child names spelled directly under a canonical root.
    for root in ("cudnn.gemm.cutedsl.grouped", "cudnn.gemm.cutedsl.discrete_grouped"):
        if fullname.startswith(root + "."):
            rest = fullname[len(root) + 1 :].split(".")
            prefix = _STRIP_CHILD_PREFIX[root]
            if rest[0].startswith(prefix):
                rest[0] = rest[0][len(prefix) :]
                return ".".join([root, *rest])
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical: str):
        self._canonical = canonical
        self._saved = {}

    def create_module(self, spec):
        # Returning an existing module makes the import machinery bind the
        # legacy name in sys.modules to the canonical module itself. The
        # machinery then stamps the legacy spec onto that module
        # (_init_module_attrs overwrites __spec__/__loader__ and, for a
        # returned module, __name__/__package__), so snapshot the canonical
        # metadata here and restore it in exec_module, which runs after.
        module = importlib.import_module(self._canonical)
        self._saved = {attr: getattr(module, attr) for attr in ("__name__", "__spec__", "__loader__", "__package__") if hasattr(module, attr)}
        return module

    def exec_module(self, module):
        # Already executed under its canonical name; just undo the legacy
        # metadata stamped between create_module and here.
        for attr, value in self._saved.items():
            setattr(module, attr, value)


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = _canonical_name(fullname)
        if canonical is None or canonical == fullname:
            return None
        warnings.warn(
            f"'{fullname}' moved to '{canonical}' in cudnn-frontend 1.27; the old"
            f" path is kept as a deprecated alias. Prefer the top-level"
            f" 'from cudnn import <symbol>' entry points, which never moved.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical))


def install() -> None:
    """Idempotently install the legacy-alias finder on ``sys.meta_path``.

    It must run BEFORE the default PathFinder: an aliased parent package has
    the canonical ``__path__``, so PathFinder would resolve a legacy child
    like ``cudnn.grouped_gemm.grouped_gemm_wgrad.api`` to the real ``api.py``
    and execute it a second time as a distinct module under the legacy name.
    """
    if not any(isinstance(finder, _LegacyAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyAliasFinder())
