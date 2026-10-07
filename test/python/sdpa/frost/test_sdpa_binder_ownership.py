# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prevent production forward fallback and implicit backward binder ownership."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from cudnn.sdpa.fwd import prepared
from test_sdpa_native_bwd_binding import _fixture

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("name", ["bind_dense", "bind_dense_split", "bind_thd", "_bind_thd_python", "execute_thd", "execute_quantized"])
def test_forward_python_framing_is_test_only(name):
    assert not hasattr(prepared, name), f"{name} must remain a test-only oracle"


def test_production_does_not_import_binding_oracles():
    for path in Path(prepared.__file__).parents[1].rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "binding_reference" not in (node.module or ""), path
            elif isinstance(node, ast.Import):
                assert all("binding_reference" not in alias.name for alias in node.names), path


def test_backward_host_must_explicitly_choose_its_binder():
    spec, _, _, _, _ = _fixture()
    from cudnn.sdpa.bwd.prepared import BwdLaunchSpec

    with pytest.raises(TypeError, match="native_binding"):
        BwdLaunchSpec(spec.artifact, spec.fn, spec.operands, spec.workspace_bytes, spec.device_index, spec.scale)
    assert spec.native_binding is False and spec.native is None
    assert replace(spec, native_binding=True).native is not None
