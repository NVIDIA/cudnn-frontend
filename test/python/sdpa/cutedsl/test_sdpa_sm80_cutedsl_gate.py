# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM80 SDPA wrappers decline a too-old CuTe DSL by version (AGENTS.md Rule 7).

Host-only: the wrapper entry is an independent caller-facing path to the kernel
modules (no engine check_support in front of it), so it must raise an error
that names the installed version *before* importing a kernel module, where a
4.5.x wheel would otherwise fail with the DSL's own TypeError.

The ``*_before_the_dsl_import`` cells run the same decline with the DSL's
``cutlass.experimental`` package made unimportable, as the 4.6.2 wheel ships
it: a module-level import of a DSL-bound module anywhere in a wrapper module's
import chain then surfaces as ``ModuleNotFoundError`` ahead of the decline
(``sdpa/bwd/api_dsl.py`` once imported ``tile_dsl.thd`` at module scope).
"""

import importlib
import sys

import pytest
import torch

from cudnn.frost import buffers

pytestmark = pytest.mark.L0

# The two wrapper modules and the DSL-bound module whose module-level import once reached one of them: all three
# re-execute under ``no_experimental_primitives``, so an import cached by an earlier test cannot hide the chain.
_WRAPPER_CHAIN = ("cudnn.sdpa.fwd.api_dsl", "cudnn.sdpa.bwd.api_dsl", "cudnn.frost.tile_dsl.thd")
_ABSENT = object()


@pytest.fixture
def old_dsl(monkeypatch):
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))


@pytest.fixture
def no_experimental_primitives(monkeypatch):
    """``cutlass.experimental`` unimportable (a 4.6.2 wheel) on whatever DSL the host runs.

    The modules of ``_WRAPPER_CHAIN`` leave ``sys.modules`` so the test's import executes their module scope
    again; a module-level import of the DSL-bound ``tile_dsl.thd`` on that path then fails with
    ``ModuleNotFoundError`` instead of reaching the version decline.  Teardown restores the cached module
    objects (monkeypatch) AND the attributes a re-import rebinds on the parent packages (below), so later
    tests keep the one module object everything else imported.
    """
    for name in ("cutlass.experimental", "cutlass.experimental.primitives"):
        monkeypatch.setitem(sys.modules, name, None)
    rebound = []
    for name in _WRAPPER_CHAIN:
        parent, _, child = name.rpartition(".")
        pkg = importlib.import_module(parent)
        rebound.append((pkg, child, vars(pkg).get(child, _ABSENT)))  # vars(): a lazy package __getattr__ must not import it here
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for pkg, child, before in rebound:
        if before is _ABSENT:
            vars(pkg).pop(child, None)
        else:
            setattr(pkg, child, before)


def test_bwd_wrapper_declines_by_version(old_dsl):
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_bwd_wrapper_sm80(q_tensor=t, k_tensor=t, v_tensor=t, o_tensor=t, do_tensor=t, lse_tensor=torch.zeros(1, 1, 8, 1))


def test_fwd_wrapper_declines_by_version(old_dsl):
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_fwd_wrapper_sm80(t, t, t)


def test_bwd_wrapper_declines_before_the_dsl_import(old_dsl, no_experimental_primitives):
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_bwd_wrapper_sm80(q_tensor=t, k_tensor=t, v_tensor=t, o_tensor=t, do_tensor=t, lse_tensor=torch.zeros(1, 1, 8, 1))


def test_fwd_wrapper_declines_before_the_dsl_import(old_dsl, no_experimental_primitives):
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_fwd_wrapper_sm80(t, t, t)
