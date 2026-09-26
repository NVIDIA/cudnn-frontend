# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 detectors for every frontend-only API in this tree (recipe R9).

``execute()`` on any ``APIBase`` subclass runs under torch's sync debug mode, so a
torch-level host sync inside it (``.item()``, ``.cpu()``, ``torch.cuda.synchronize``,
``Event.synchronize``, ``torch.tensor(list, device="cuda")``) raises instead of silently
serialising the caller. An engine that must block declares it (recipe R6) and its tests
carry ``@pytest.mark.allow_host_sync``; nothing else may.

The guard wraps each class's own ``execute`` once and stays for the process; the autouse
fixture only arms it for the duration of each test. Subclasses defined after this file
is imported -- a test body's lazy ``from cudnn import X`` -- are guarded at class
creation through ``APIBase.__init_subclass__``, so coverage does not depend on which
test imported the API first.
"""

from __future__ import annotations

import functools

import pytest
import torch

from cudnn.api_base import APIBase

_WRAPPED = "_rule8_sync_guarded"
_armed = False  # set per test by the autouse fixture


def _guard_class(cls) -> None:
    own = cls.__dict__.get("execute")
    if own is None or getattr(own, _WRAPPED, False):
        return

    @functools.wraps(own)  # signature-stability tests inspect execute()
    def guarded(self, *args, **kwargs):
        if not _armed:
            return own(self, *args, **kwargs)
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
_guard_all_subclasses()  # subclasses imported before this conftest


@pytest.fixture(autouse=True)
def _execute_never_blocks_the_host(request):
    """Arm the guard on every ``APIBase.execute()`` for this test. Opt out per test with
    ``@pytest.mark.allow_host_sync`` and say why."""
    global _armed
    _guard_all_subclasses()
    _armed = not request.node.get_closest_marker("allow_host_sync") and torch.cuda.is_available()
    try:
        yield
    finally:
        _armed = False


@pytest.fixture
def compile_allocates_nothing():
    """Recipe R11 detector: ``compile_allocates_nothing(api)`` runs ``api.compile()`` and
    asserts the torch caching allocator saw no new allocation (compile-time stand-ins
    must be fake cute tensors, never ``torch.empty``)."""

    def run(api):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        api.compile()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        assert after == before, f"{type(api).__name__}.compile() made {after - before} torch allocation(s) (Rule 8, recipe R11)"

    return run
