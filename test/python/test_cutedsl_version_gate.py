# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A too-old CuTe DSL is reported as a version problem, not a missing dependency.

pyproject's floor on nvidia-cutlass-dsl is the downstream floor (4.6.2), below
what the FROST-derived kernels need (CUTEDSL_MIN_VERSION). python/cudnn/AGENTS.md
Rule 7 requires every route to decline below the kernel floor with an error that
names the installed version; these host-only checks pin that contract at the two
shared seams (the lazy-import wrapper and the decode-update route).
"""

import pytest
import torch

import cudnn
from cudnn.frost import buffers

pytestmark = pytest.mark.L0

_FLOOR = ".".join(str(x) for x in buffers.CUTEDSL_MIN_VERSION)


@pytest.fixture
def dsl_state(monkeypatch):
    def set_state(installed, version):
        monkeypatch.setattr(buffers, "_DSL_STATE", (installed, version))

    return set_state


def test_too_old_dsl_names_the_versions(dsl_state):
    dsl_state(True, ("nvidia-cutlass-dsl", "4.6.2"))
    message = buffers.cutedsl_requirement_error("GemmSreluSm100")
    assert f"requires nvidia-cutlass-dsl >= {_FLOOR}" in message
    assert "found 4.6.2" in message
    assert "nvidia-cudnn-frontend[cutedsl]" not in message


@pytest.mark.parametrize(
    "state",
    [
        (True, ("nvidia-cutlass-dsl", "4.7.0")),
        (True, ("nvidia-cutlass-dsl", "4.8.0a0+20260904")),
        (True, ("nvidia-cutlass-dsl-internal", "0.3.0+2026")),
        (False, None),
    ],
    ids=["at-floor", "prerelease-above", "internal-rc", "not-installed"],
)
def test_healthy_or_absent_dsl_yields_no_version_error(dsl_state, state):
    dsl_state(*state)
    assert buffers.cutedsl_requirement_error("x") is None


def test_lazy_import_wrapper_distinguishes_too_old_from_missing(dsl_state):
    inner = TypeError("set_name_prefix() got an unexpected keyword argument 'remove_cutlass_symbol'")

    dsl_state(True, ("nvidia-cutlass-dsl", "4.6.2"))
    too_old = cudnn._optional_dependency_message("GemmSreluSm100", inner)
    assert "found 4.6.2" in too_old and "remove_cutlass_symbol" in too_old
    assert "nvidia-cudnn-frontend[cutedsl]" not in too_old

    dsl_state(False, None)
    missing = cudnn._optional_dependency_message("GemmSreluSm100", ImportError("No module named 'cutlass'"))
    assert "pip install nvidia-cudnn-frontend[cutedsl]" in missing


def test_decode_update_route_declines_by_version_before_importing_the_kernel(dsl_state, monkeypatch):
    from cudnn.ops import _causal_conv1d_update as update

    dsl_state(True, ("nvidia-cutlass-dsl", "4.6.2"))
    monkeypatch.setattr(update, "_validate_semantic_contract", lambda *args: None)
    monkeypatch.setattr(update, "_require_native_subset", lambda *args: None)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    conv_state = torch.zeros(2, 8, 3, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        update._validated_native_update(x, conv_state, weight, None, "silu", None, None)


@pytest.mark.parametrize(
    "version,too_old",
    [
        ("4.6.2a0", True),
        ("4.6.2rc1", True),
        ("4.6.9b1", True),
        ("4.7.0a0", False),
        ("4.7.0rc1", False),
        ("4.7.0a0+20260727", False),
        ("4.5.1", True),
        ("4.7.0", False),
        ("4.6", True),
        ("4", True),
        ("4.7", False),
        ("5", False),
    ],
)
def test_prerelease_versions_compare_by_release_number(version, too_old):
    """A public prerelease of X compares as X against the floor (Rule 7): 4.6.2rc1 is
    too old, 4.7.0a0 is not. Before, any letter in a component made the check
    return False and an unsupported DSL walked through every gate."""
    assert buffers.cutedsl_too_old(("nvidia-cutlass-dsl", version)) is too_old


def test_missing_third_party_module_is_not_blamed_on_the_dsl(dsl_state):
    """With an old DSL installed, a lazy import that fails because torch is missing
    must report the optional-dependency hint, not a DSL upgrade."""
    dsl_state(True, ("nvidia-cutlass-dsl", "4.6.2"))
    missing_torch = ModuleNotFoundError("No module named 'torch'", name="torch")
    message = cudnn._optional_dependency_message("sdpa_torch", missing_torch)
    assert "requires nvidia-cutlass-dsl" not in message
    assert "pip install nvidia-cudnn-frontend[cutedsl]" in message
    assert "'torch' module" in message  # the hint alone would not fetch torch, so name it
    # cuda-python is a separate dependency (the [cutedsl] extra), not the DSL stack:
    missing_cuda = ModuleNotFoundError("No module named 'cuda.bindings'", name="cuda.bindings")
    cuda_message = cudnn._optional_dependency_message("sdpa_torch", missing_cuda)
    assert "requires nvidia-cutlass-dsl" not in cuda_message and "cuda.bindings" in cuda_message
    # ...while a DSL-stack failure on the same old DSL still names the version
    dsl_failure = TypeError("set_name_prefix() got an unexpected keyword argument 'remove_cutlass_symbol'")
    assert "found 4.6.2" in cudnn._optional_dependency_message("GemmSreluSm100", dsl_failure)
    missing_cutlass = ModuleNotFoundError("No module named 'cutlass.experimental'", name="cutlass.experimental")
    assert "found 4.6.2" in cudnn._optional_dependency_message("GemmSreluSm100", missing_cutlass)
