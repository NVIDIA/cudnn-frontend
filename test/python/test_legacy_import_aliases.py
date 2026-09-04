# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-1.27 module paths must keep importing after the gemm reorg.

The 1.27 reorganization moved the CuTeDSL GEMM packages under
``cudnn.gemm.cutedsl.*`` while keeping ``from cudnn import <symbol>`` stable.
``cudnn._legacy_aliases`` is the compatibility layer for the module paths
themselves: ``import cudnn.grouped_gemm.grouped_gemm_wgrad.api`` must bind the
canonical module (same object, not a copy) and warn, without making
``import cudnn`` pull a framework.

Probes that import real fusion modules need the optional deps and run in a
fresh interpreter (an in-process assertion would pass for the wrong reason
once modules are cached); mapping tests are pure and run in-process.
"""

import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0

# The eager grouped-gemm import chain needs nvidia-cutlass-dsl and
# cuda-python at module scope (torch is deferred to call time).
_HAVE_CUTEDSL = bool(importlib.util.find_spec("cutlass")) and bool(importlib.util.find_spec("cuda"))


def _run(probe: str) -> None:
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_legacy_name_mapping():
    from cudnn._legacy_aliases import _canonical_name

    assert _canonical_name("cudnn.grouped_gemm") == "cudnn.gemm.cutedsl.grouped"
    assert _canonical_name("cudnn.grouped_gemm.grouped_gemm_wgrad.api") == "cudnn.gemm.cutedsl.grouped.wgrad.api"
    # Helper modules kept their names in the move.
    assert _canonical_name("cudnn.grouped_gemm.moe_utils") == "cudnn.gemm.cutedsl.grouped.moe_utils"
    assert _canonical_name("cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu") == "cudnn.gemm.cutedsl.discrete_grouped.swiglu"
    assert _canonical_name("cudnn.gemm_amax.api") == "cudnn.gemm.cutedsl.dense.amax.api"
    # Old-style child names spelled under the canonical root (the fromlist
    # path of ``from cudnn.grouped_gemm import grouped_gemm_wgrad``).
    assert _canonical_name("cudnn.gemm.cutedsl.grouped.grouped_gemm_wgrad") == "cudnn.gemm.cutedsl.grouped.wgrad"
    # Non-legacy paths are left to the normal import machinery.
    assert _canonical_name("cudnn.sdpa") is None
    assert _canonical_name("cudnn.gemm.cutedsl.grouped.wgrad") is None
    assert _canonical_name("numpy") is None


def test_finder_installed_by_import_cudnn():
    _run("""
import sys
import cudnn
from cudnn._legacy_aliases import _LegacyAliasFinder
assert any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path)
import cudnn  # reimport must not stack a second finder
cudnn._legacy_aliases.install()
assert sum(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path) == 1
""")


@pytest.mark.skipif(not _HAVE_CUTEDSL, reason="requires nvidia-cutlass-dsl and torch")
def test_legacy_import_binds_canonical_module_and_warns():
    _run("""
import sys
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import cudnn.grouped_gemm
    import cudnn.grouped_gemm.grouped_gemm_wgrad.api

import cudnn.gemm.cutedsl.grouped

assert sys.modules["cudnn.grouped_gemm"] is sys.modules["cudnn.gemm.cutedsl.grouped"]
assert (
    sys.modules["cudnn.grouped_gemm.grouped_gemm_wgrad.api"]
    is sys.modules["cudnn.gemm.cutedsl.grouped.wgrad.api"]
)
# The canonical module keeps its identity; only sys.modules gains a key.
assert sys.modules["cudnn.grouped_gemm"].__name__ == "cudnn.gemm.cutedsl.grouped"

from cudnn.grouped_gemm import grouped_gemm_wgrad  # fromlist spelling
assert grouped_gemm_wgrad is sys.modules["cudnn.gemm.cutedsl.grouped.wgrad"]

from cudnn import grouped_gemm_wgrad_wrapper_sm100  # supported entry point
assert grouped_gemm_wgrad_wrapper_sm100 is grouped_gemm_wgrad.api.grouped_gemm_wgrad_wrapper_sm100

deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
assert any("cudnn.gemm.cutedsl.grouped" in str(w.message) for w in deprecations), deprecations
""")


def test_optional_dep_failure_names_the_real_cause():
    _run("""
import importlib.abc
import sys

class BlockOptionalDeps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "cutlass", "nvidia_cutlass_dsl", "cuda"):
            raise ImportError(f"blocked {fullname} for legacy-alias test")
        return None

sys.meta_path.insert(0, BlockOptionalDeps())
import cudnn
try:
    cudnn.grouped_gemm_wgrad_wrapper_sm100
except ImportError as error:
    assert "pip install nvidia-cudnn-frontend[cutedsl]" in str(error), error
    assert "blocked" in str(error), error
    assert error.__cause__ is not None
else:
    raise AssertionError("lazy symbol access unexpectedly succeeded without optional deps")
""")


def test_internal_missing_module_is_not_blamed_on_optional_deps():
    _run("""
import cudnn
cudnn._LAZY_OPTIONAL_IMPORTS["bogus_symbol"] = (".does_not_exist", None)
try:
    cudnn.bogus_symbol
except ImportError as error:
    assert "packaging bug" in str(error), error
    assert "pip install" not in str(error), error
else:
    raise AssertionError("bogus lazy symbol unexpectedly resolved")
""")
