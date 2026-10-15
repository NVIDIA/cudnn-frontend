# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing the engine must not import the kernels, or the CuTe DSL.

AGENTS.md Rule 7 wants an engine on an unsupported install to DECLINE, not
explode. That only works if nothing imports a kernel module before the decline
runs: every kernel imports ``cutlass.primitives`` at its top, so on a DSL
without it the failure is a bare ``ModuleNotFoundError`` raised from inside the
import machinery -- out of the engine's reach, with its plan already ranked.

Keeping ``cudnn.norm`` lazy was not enough. The engine lives in
``cudnn.norm.fprop.engine``, and importing that imports the ``cudnn.norm.fprop``
PACKAGE first, whose ``__init__`` pulled in ``api`` and with it all twelve
kernels -- 287 cutlass modules loaded before any gate could speak.

Each check runs in a FRESH interpreter. In-process it would prove nothing:
pytest has almost certainly imported the DSL already for some other test, and
``sys.modules`` is global.
"""

import subprocess
import sys
import textwrap

import pytest

# Importing cudnn at all is what the rest of the suite needs; if it is not built,
# these subprocess probes cannot say anything either.
pytest.importorskip("cudnn", reason="needs a built cudnn frontend")

# Everything up to and including CONSTRUCTING the engines must stay DSL-free.
_COLD = [
    ("the package itself", "import cudnn.norm"),
    ("the engine module", "import cudnn.norm.fprop.engine"),
    ("the backward engine module", "import cudnn.norm.bprop.engine"),
    ("the lowering module", "import cudnn.norm.fprop.engines"),
    ("the kernels package", "import cudnn.norm.fprop.kernels"),
    ("the subpackage", "import cudnn.norm.fprop"),
    (
        "constructing the engine objects",
        "import cudnn.norm.fprop.engine as e; e.FrostNormFwdEngines({'norm_fprop_sm100': 21000})",
    ),
]

_PROBE = textwrap.dedent("""
    import sys
    {stmt}
    loaded = sorted(m for m in sys.modules if m == "cutlass" or m.startswith("cutlass."))
    print(len(loaded))
    print(",".join(loaded[:8]))
    """)


def _cutlass_modules_after(stmt):
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(stmt=stmt)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, f"probe failed:\n{out.stderr[-2000:]}"
    lines = out.stdout.strip().splitlines()
    return int(lines[0]), (lines[1] if len(lines) > 1 else "")


@pytest.mark.L0
@pytest.mark.parametrize("label,stmt", _COLD, ids=[c[0].replace(" ", "-") for c in _COLD])
def test_import_does_not_load_the_cute_dsl(label, stmt):
    n, sample = _cutlass_modules_after(stmt)
    assert n == 0, f"{label} loaded {n} cutlass modules before any gate could run (e.g. {sample})"


@pytest.mark.L0
def test_resolving_the_api_does_load_it():
    """The mirror image: the laziness must be real, not an accident of a renamed import."""
    n, _ = _cutlass_modules_after("from cudnn.norm import norm_fprop")
    assert n > 0, "norm_fprop resolved without loading the DSL -- the probe is not measuring what it claims"


@pytest.mark.L0
def test_missing_primitive_layer_declines_instead_of_raising():
    """With cutlass.primitives unreachable, the API must raise OUR message.

    Simulated by making that one import fail; everything else about the DSL
    stays as it is, which is the shape of a public wheel that exports the layer
    under ``cutlass.experimental.primitives`` instead.
    """
    probe = textwrap.dedent("""
        import importlib
        import cudnn.frost.buffers as b

        real = importlib.import_module
        def fake(name, *a, **k):
            if name == "cutlass.primitives":
                raise ModuleNotFoundError("No module named 'cutlass.primitives'")
            return real(name, *a, **k)
        importlib.import_module = fake
        b._DSL_PRIMITIVES = None

        import cudnn.norm
        try:
            cudnn.norm.norm_fprop
        except ImportError as exc:
            print("DECLINED:", exc)
        else:
            print("NO ERROR")
        """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    line = out.stdout.strip().splitlines()[-1]
    assert line.startswith("DECLINED:"), f"expected a gated ImportError, got: {line}"
    assert "cutlass.primitives" in line and "does not export" in line, f"message is not actionable: {line}"
