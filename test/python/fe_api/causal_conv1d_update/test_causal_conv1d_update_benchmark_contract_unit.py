# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-contract tests for the causal-convolution benchmark environment gate."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUPPORTED_COMPUTE_CAPABILITIES = (
    (8, 0),
    (8, 6),
    (8, 7),
    (8, 9),
    (9, 0),
    (10, 0),
    (10, 3),
    (11, 0),
    (12, 0),
    (12, 1),
)


def _fake_fla_ops(monkeypatch):
    packages = {}
    for name in ("fla", "fla.modules", "fla.modules.conv", "fla.modules.conv.triton"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
        monkeypatch.setitem(sys.modules, name, package)

    ops = types.ModuleType("fla.modules.conv.triton.ops")

    def causal_conv1d_update(x, cache, residual=None, weight=None, bias=None, activation=None):
        del residual, weight, bias, activation
        return x, cache

    causal_conv1d_update.__module__ = ops.__name__
    ops.causal_conv1d_update = causal_conv1d_update
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    packages["fla"].modules = packages["fla.modules"]
    packages["fla.modules"].conv = packages["fla.modules.conv"]
    packages["fla.modules.conv"].triton = packages["fla.modules.conv.triton"]
    packages["fla.modules.conv.triton"].ops = ops


def _load_benchmark(monkeypatch):
    _fake_fla_ops(monkeypatch)
    benchmark_path = _REPO_ROOT / "benchmark" / "causal_conv1d_update_sm100.py"
    module_name = "_test_causal_conv1d_update_sm100_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, benchmark_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"CuTe DSL benchmark dependencies unavailable: {exc}")
    return module


def test_benchmark_accepts_supported_architectures_without_slurm(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_package_version", lambda unused: "0.5.2")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)

    for capability in _SUPPORTED_COMPUTE_CAPABILITIES:
        monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda device=None, value=capability: value)
        benchmark._validate_environment()

    assert benchmark._slurm_provenance() == {
        "slurm_job_id": None,
        "slurmd_node_name": None,
    }


def test_benchmark_records_available_slurm_provenance(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURMD_NODENAME", "customer-node-7")

    assert benchmark._slurm_provenance() == {
        "slurm_job_id": "12345",
        "slurmd_node_name": "customer-node-7",
    }


@pytest.mark.parametrize("filename", ["causal_conv1d_update_sm100.py", "fla_short_conv_shim_sm100.py"])
def test_benchmark_scripts_never_require_slurm_environment(filename):
    tree = ast.parse((_REPO_ROOT / "benchmark" / filename).read_text())
    required_lookups = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    ]

    assert required_lookups == []


def test_benchmark_rejects_unavailable_or_unsupported_cuda(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(benchmark, "_package_version", lambda unused: "0.5.2")
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        benchmark._validate_environment()

    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda device=None: (11, 1))
    with pytest.raises(RuntimeError, match="functionally supported causal-conv compute capability"):
        benchmark._validate_environment()
