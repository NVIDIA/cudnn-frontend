# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 detectors for the frontend-only APIs (recipes R9 / R11), registered by ``conftest.py``.

For tests under ``COVERED_DIRS`` -- the per-operation homes of the former ``fe_api`` tree --
``execute()`` on any ``APIBase`` subclass runs under torch's sync debug mode, so a torch-level
host sync inside it (``.item()``, ``.cpu()``, ``torch.cuda.synchronize``, ``Event.synchronize``,
``torch.tensor(list, device="cuda")``) raises instead of silently serialising the caller. An engine
that must block declares it (recipe R6) and its tests carry ``@pytest.mark.allow_host_sync``.

The guard wraps each class's own ``execute`` once and stays for the process; the autouse fixture
only arms it for the duration of a covered test. Subclasses defined later -- a test body's lazy
``from cudnn import X`` -- are guarded at class creation through ``APIBase.__init_subclass__``.
torch is imported lazily, so a torch-free (JAX-only) session collects and never arms.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from cudnn.api_base import APIBase

TEST_ROOT = Path(__file__).resolve().parent

# (operation, backend) directories whose tests the detector arms for; a renamed or new frontend-only
# API directory must be added here (test_rule8_detector.py fails when a listed one disappears). An
# operation joins once its APIs are migrated, in the same change.
COVERED_DIRS = frozenset({("deepseek_sparse_attention", "cutedsl")})


_WRAPPED = "_rule8_sync_guarded"
_armed = False  # set per covered test by the autouse fixture


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def covered(path) -> bool:
    try:
        parts = Path(path).resolve().relative_to(TEST_ROOT).parts
    except ValueError:
        return False
    return len(parts) > 2 and (parts[0], parts[1]) in COVERED_DIRS


def _guard_class(cls) -> None:
    own = cls.__dict__.get("execute")
    if own is None or getattr(own, _WRAPPED, False):
        return

    @functools.wraps(own)  # signature-stability tests inspect execute()
    def guarded(self, *args, **kwargs):
        if not _armed:
            return own(self, *args, **kwargs)
        import torch

        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return own(self, *args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    setattr(guarded, _WRAPPED, True)
    cls.execute = guarded


def _guard_all_subclasses() -> None:
    seen, todo = set(), [APIBase]
    while todo:
        c = todo.pop()
        for s in c.__subclasses__():
            if s not in seen:
                seen.add(s)
                todo.append(s)
    for cls in seen:
        _guard_class(cls)


def _guard_new_subclass(cls, **kwargs):
    super(APIBase, cls).__init_subclass__(**kwargs)
    _guard_class(cls)


APIBase.__init_subclass__ = classmethod(_guard_new_subclass)
_guard_all_subclasses()  # subclasses imported before this module


@pytest.fixture(autouse=True)
def _execute_never_blocks_the_host(request):
    """Arm the guard on every ``APIBase.execute()`` of a covered test. Opt out per test with
    ``@pytest.mark.allow_host_sync`` and say why."""
    global _armed
    torch = _torch()
    if torch is None or not covered(request.node.path) or request.node.get_closest_marker("allow_host_sync"):
        yield
        return
    _guard_all_subclasses()
    _armed = torch.cuda.is_available()
    try:
        yield
    finally:
        _armed = False


@pytest.fixture
def compile_allocates_nothing():
    """Recipe R11 detector: ``compile_allocates_nothing(api)`` runs ``api.compile()`` and
    asserts the torch caching allocator saw no new allocation (compile-time stand-ins
    must be fake cute tensors, never ``torch.empty``)."""
    import torch

    def run(api):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        api.compile()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        assert after == before, f"{type(api).__name__}.compile() made {after - before} torch allocation(s) (Rule 8, recipe R11)"

    return run
