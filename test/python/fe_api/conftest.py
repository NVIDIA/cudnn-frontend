# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 detectors for every frontend-only API in this tree (recipe R9).

``execute()`` on any ``APIBase`` subclass runs under torch's sync debug mode, so a
torch-level host sync inside it (``.item()``, ``.cpu()``, ``torch.cuda.synchronize``,
``Event.synchronize``) raises instead of silently serialising the caller. An engine
that must block declares it (recipe R6) and its tests carry
``@pytest.mark.allow_host_sync``; nothing else may.
"""

from __future__ import annotations

import functools

import pytest
import torch

from cudnn.api_base import APIBase

_WRAPPED = "_rule8_sync_guarded"


def _all_subclasses(cls):
    seen, todo = set(), [cls]
    while todo:
        c = todo.pop()
        for s in c.__subclasses__():
            if s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


@pytest.fixture(autouse=True)
def _execute_never_blocks_the_host(request, monkeypatch):
    """Arm ``torch.cuda.set_sync_debug_mode("error")`` for the duration of every
    ``execute()`` call (own or inherited) on every ``APIBase`` subclass imported so far.
    Opt out per test with ``@pytest.mark.allow_host_sync`` and say why."""
    if request.node.get_closest_marker("allow_host_sync") or not torch.cuda.is_available():
        yield
        return

    def guard(real):
        @functools.wraps(real)  # signature-stability tests inspect execute()
        def guarded(self, *args, **kwargs):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                return real(self, *args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(previous)

        setattr(guarded, _WRAPPED, True)
        return guarded

    for cls in _all_subclasses(APIBase) | {APIBase}:
        own = cls.__dict__.get("execute")
        if own is not None and not getattr(own, _WRAPPED, False):
            monkeypatch.setattr(cls, "execute", guard(own))
    yield


@pytest.fixture
def compile_allocates_nothing():
    """Recipe R10 detector: ``compile_allocates_nothing(api)`` runs ``api.compile()`` and
    asserts the torch caching allocator saw no new allocation (compile-time stand-ins
    must be fake cute tensors, never ``torch.empty``)."""

    def run(api):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        api.compile()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        assert after == before, f"{type(api).__name__}.compile() made {after - before} torch allocation(s) (Rule 8, recipe R10)"

    return run
