# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each dispatch stage is allowed to drag in.

These boundaries are load-bearing, not tidiness:

* The graph API must not require a framework. A caller describing a graph has
  not chosen an engine yet, so making description depend on torch would make
  the whole front end depend on it.
* Deciding whether an engine COULD serve a graph must not cost the machinery
  that would serve it. Importing the CuTe DSL measured ~1.0 s and 357 modules;
  paying that only to decline is why `closed_under` existed, and deleting that
  workaround is only safe while this holds.

Each check runs in a FRESH interpreter: once a module is imported in the test
process it stays, so an in-process assertion would pass for the wrong reason.
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0

_HEAVY = ("torch", "cutlass", "nvidia_cutlass_dsl")


def _modules(code: str) -> set:
    probe = code + "\nimport json, sys; print('@@' + json.dumps(sorted(sys.modules)))"
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if run.returncode != 0:
        # Not skip: a probe that cannot `import cudnn` is the regression this
        # file exists to catch, and skipping would report it as "not checked".
        pytest.fail(f"probe code failed: {run.stderr.strip()[-200:]}")
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("@@"))
    return set(json.loads(line[2:]))


def _imported_by(code: str) -> set:
    """What ``code`` ADDED to a fresh interpreter.

    A delta, not an absolute set: some distributions inject packages at startup
    through a .pth file (nvidia_cutlass_dsl is one), so asking "is it in
    sys.modules" would blame us for something the interpreter did before our
    first line ran.
    """
    return _modules(code) - _modules("")


def _assert_absent(mods: set, stage: str) -> None:
    present = sorted(m for m in _HEAVY if m in mods)
    assert not present, f"{stage} imported {present}; it must not"


@pytest.mark.L0
def test_importing_cudnn_pulls_no_framework():
    _assert_absent(_imported_by("import cudnn"), "import cudnn")


@pytest.mark.L0
def test_gnn_import_failure_has_optional_dependency_guidance():
    code = """
import sys
sys.modules["torch"] = None
import cudnn
try:
    cudnn.gnn
except ImportError as error:
    # GNN does not use CuTeDSL, but this is currently the existing optional
    # dependency extra that installs Torch.
    assert "nvidia-cudnn-frontend[cutedsl]" in str(error)
else:
    raise AssertionError("cudnn.gnn unexpectedly imported without torch")
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.L0
def test_importing_cudnn_ops_pulls_no_framework():
    _assert_absent(_imported_by("import cudnn\nimport cudnn.ops"), "import cudnn.ops")


@pytest.mark.L0
def test_ops_symbol_reports_install_hint_without_torch():
    probe = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("blocked torch for import-boundary test")
        return None

sys.meta_path.insert(0, BlockTorch())
import cudnn.ops
try:
    cudnn.ops.fft_causal_conv1d
except ImportError as error:
    assert "pip install nvidia-cudnn-frontend[cutedsl]" in str(error)
else:
    raise AssertionError("lazy symbol access unexpectedly succeeded without torch")
"""
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.L0
def test_describing_a_graph_pulls_no_framework():
    """Build and validate an SDPA graph through the graph API alone."""
    _assert_absent(
        _imported_by("""
import cudnn
g = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)
dim, stride = (2, 8, 256, 128), (256 * 8 * 128, 128, 8 * 128, 1)
q, k, v = [g.tensor(dim=dim, stride=stride, data_type=cudnn.data_type.HALF, name=n) for n in "qkv"]
o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
o.set_output(True).set_dim(dim).set_stride(stride).set_data_type(cudnn.data_type.HALF)
g.validate()
"""),
        "describing a graph",
    )


@pytest.mark.L0
def test_classification_and_facts_pull_no_framework():
    """The manifest classifies and the analyzer describes; neither lowers."""
    _assert_absent(
        _imported_by("import cudnn\nfrom cudnn.engines import manifest\nfrom cudnn.sdpa import graph_analyzer"),
        "classification + facts",
    )


@pytest.mark.L0
@pytest.mark.parametrize("module", ["cudnn.sdpa.fwd.engines", "cudnn.sdpa.bwd.engines"])
def test_support_check_pulls_no_framework(module):
    """Capabilities and mismatch() are pure data and comparisons.

    They live in the same file as the lowering helpers, so the DSL adapter and
    torch are resolved inside the functions that need them rather than at
    module level. Regressing that would restore a ~1.0 s cost paid by every
    process that merely asks whether an engine applies.
    """
    _assert_absent(_imported_by(f"import cudnn\nimport {module}"), module)
