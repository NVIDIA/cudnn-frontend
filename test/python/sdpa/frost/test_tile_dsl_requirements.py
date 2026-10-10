# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.requirements`` -- the host-callable CuTe DSL gates of the ``tile_dsl`` primitives, importable BELOW the DSL floor.

The primitive modules import ``cutlass.experimental`` at module level; on the package-floor wheel (``nvidia-cutlass-dsl==4.6.2``)
that import raises ``ModuleNotFoundError`` before any function in them can run, so a host-side ``check_support`` that imported its
gate from ``tile_dsl.tma`` received the raw import failure instead of the version message (reproduced with the real wheel).
This module deliberately does NOT carry the suite's ``requires_dsl`` marker: it must run where the DSL is old or absent -- the
one place the gate matters.  The below-floor condition is reproduced in a FRESH process by blocking ``cutlass.experimental``
through ``sys.modules`` and substituting the version state; no wheel is downgraded.  A probe that imported ``tma`` first (the
in-process decline tests of ``test_tile_dsl_gather4_sm107.py``) cannot see this failure, which is why this one forks.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.L0]

# Runs in a fresh interpreter: the blocker and the substituted version state must not leak into the test process.
_BELOW_FLOOR_PROBE = r"""
import json, sys
sys.modules["cutlass.experimental"] = None  # `from cutlass.experimental import ...` -> ModuleNotFoundError, as on the 4.6.2 wheel
import cudnn.frost.buffers as buffers
buffers._DSL_STATE = (True, ("nvidia-cutlass-dsl", "4.6.2"))  # the package-floor wheel's version, substituted (no downgrade)
out = {}
try:
    from cudnn.frost.tile_dsl.tma import tma_gather4_requirement_error as _via_tma  # the primitive module: must NOT be the host entry
    out["tma_import"] = "ok"
except ModuleNotFoundError as e:
    out["tma_import"] = "ModuleNotFoundError: " + str(e)
from cudnn.frost.tile_dsl.requirements import requirement_error, tma_gather4_requirement_error
out["version_msg"] = tma_gather4_requirement_error()
out["generic_msg"] = requirement_error("tile_dsl.mask.apply_membership_words")
buffers._DSL_STATE = (True, ("nvidia-cutlass-dsl", "4.7.0"))  # at the floor: the arch half decides
buffers._cutedsl_has_sm107 = lambda: False
out["arch_msg"] = tma_gather4_requirement_error((10, 7))
out["served_msg"] = tma_gather4_requirement_error((10, 0))
out["cutlass_imported"] = "cutlass" in sys.modules
print("PROBE " + json.dumps(out))
"""


def _run_below_floor_probe(timeout_s=300):
    """Run the probe in a fresh interpreter and return its JSON record."""
    proc = subprocess.run([sys.executable, "-c", _BELOW_FLOOR_PROBE], capture_output=True, text=True, timeout=timeout_s, env=dict(os.environ))
    assert proc.returncode == 0, f"probe process failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE ")]
    assert len(lines) == 1, proc.stdout[-2000:]
    return json.loads(lines[0][len("PROBE ") :])


def test_host_gate_is_importable_and_names_the_floor_below_the_dsl_floor():
    """Fresh process, ``cutlass.experimental`` blocked, version state 4.6.2: importing the gate through ``tile_dsl.tma`` dies
    with ``ModuleNotFoundError`` (the condition reproduced), while ``tile_dsl.requirements`` imports and returns the message
    naming 4.6.2 and the 4.7.0 floor, the arch half names ``sm_107a`` on a cc 10.7 part and is silent on a served part, and
    nothing of the DSL was imported along the way."""
    out = _run_below_floor_probe()
    assert out["tma_import"].startswith("ModuleNotFoundError"), f"the probe must reproduce the below-floor import failure: {out['tma_import']}"
    msg = out["version_msg"]
    assert msg is not None and "4.6.2" in msg and "4.7.0" in msg and "tma_gather4" in msg, msg
    assert out["generic_msg"] is not None and "4.6.2" in out["generic_msg"] and "apply_membership_words" in out["generic_msg"], out["generic_msg"]
    assert out["arch_msg"] is not None and "sm_107a" in out["arch_msg"], out["arch_msg"]
    assert out["served_msg"] is None, out["served_msg"]
    assert out["cutlass_imported"] is False, "the host gate must not import the DSL"


def test_host_gate_module_imports_nothing_from_the_dsl():
    """Static twin of the fresh-process pin: no import line of ``tile_dsl/requirements.py`` names ``cutlass``."""
    import cudnn.frost.tile_dsl as tile_dsl

    with open(os.path.join(os.path.dirname(tile_dsl.__file__), "requirements.py")) as f:
        imports = [ln for ln in f.read().splitlines() if ln.startswith(("import ", "from "))]
    assert imports, "the module must import its building blocks (buffers) explicitly"
    assert all("cutlass" not in ln for ln in imports), imports


def test_host_gate_answers_in_this_process_for_the_installed_dsl():
    """Whatever this box installs -- the full DSL, an old wheel, or none -- the gate imports and answers consistently with
    ``buffers.cutedsl_state()``: a version message exactly when the installed public wheel is below the floor."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    from cudnn.frost.tile_dsl.requirements import tma_gather4_requirement_error

    installed, version = cutedsl_state()
    msg = tma_gather4_requirement_error()
    if installed and cutedsl_too_old(version):
        assert msg is not None and version[1] in msg and "tma_gather4" in msg, msg
    else:
        assert msg is None, msg
