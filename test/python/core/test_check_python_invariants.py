# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detectors of scripts/check_python_invariants.py (the python-source-invariants pre-commit hook), on synthetic sources."""

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_python_invariants.py"
_spec = importlib.util.spec_from_file_location("check_python_invariants", _SCRIPT)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)

_KERNEL_FILE = "python/cudnn/gemm/frost/example.py"


def test_called_decorator_and_nested_kernel():
    tree = ast.parse("def outer():\n    @cute.kernel()\n    def missing():\n        pass\n")
    assert lint.missing_name_prefixes(tree) == ["3:outer.missing"]


def test_experimental_layer_decorator_on_a_method():
    """A kernel on the ``cute.experimental`` layer is a method of its kernel class; the prefix call is the next statement of the class body."""
    missing = "class K:\n    @cute.experimental.kernel\n    def body(self):\n        pass\n"
    assert lint.missing_name_prefixes(ast.parse(missing)) == ["3:K.body"]
    prefixed = missing + '    body.set_name_prefix("cudnn", remove_cutlass_symbol=True)\n'
    assert lint.missing_name_prefixes(ast.parse(prefixed)) == []


def test_prefix_must_be_cudnn_and_name_the_kernel():
    src = '@cute.kernel\ndef k():\n    pass\nother.set_name_prefix("cudnn")\n'
    assert lint.missing_name_prefixes(ast.parse(src)) == ["2:k"]
    src = '@cute.kernel\ndef k():\n    pass\nk.set_name_prefix("cutlass")\n'
    assert lint.missing_name_prefixes(ast.parse(src)) == ["2:k"]


def test_kernel_rule_applies_only_under_its_roots():
    src = "@cute.kernel\ndef k():\n    pass\n"
    assert len(lint.check_file(_KERNEL_FILE, src)) == 1
    assert lint.check_file("python/cudnn/hstu/kernel.py", src) == []


@pytest.mark.parametrize(
    "call",
    [
        "torch.cuda.get_device_properties(0)",
        "torch.cuda.get_device_capability(0)",
        "torch.cuda.get_device_properties(device=0)",
        "get_device_capability(0)",
    ],
)
def test_device0_query_is_reported(call):
    errors = lint.check_file("python/cudnn/x.py", f"p = {call}\n")
    assert len(errors) == 1 and "device 0" in errors[0]


@pytest.mark.parametrize(
    "src",
    [
        "p = torch.cuda.get_device_properties(device)\n",
        "p = torch.cuda.get_device_capability()\n",
        "p = torch.cuda.get_device_properties(0)  # allow-device-0\n",
    ],
)
def test_device0_query_allowed(src):
    assert lint.check_file("python/cudnn/x.py", src) == []


def test_main_exit_status(tmp_path, monkeypatch):
    target = tmp_path / "python" / "cudnn" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("p = torch.cuda.get_device_properties(0)\n")
    monkeypatch.chdir(tmp_path)
    assert lint.main(["python/cudnn/x.py"]) == 1
    target.write_text("p = 1\n")
    assert lint.main(["python/cudnn/x.py"]) == 0
