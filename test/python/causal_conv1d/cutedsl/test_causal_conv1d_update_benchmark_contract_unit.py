# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-contract tests for the causal-convolution benchmark environment gate."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _fake_benchmark_dependencies(monkeypatch):
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

    native = types.ModuleType("cudnn.causal_conv1d_update_sm100")

    class _CausalConv1dUpdatePlan:
        pass

    _CausalConv1dUpdatePlan.__module__ = "cudnn.causal_conv1d_update_sm100.api"
    native._CausalConv1dUpdatePlan = _CausalConv1dUpdatePlan
    monkeypatch.setitem(sys.modules, native.__name__, native)


def _load_benchmark(monkeypatch):
    _fake_benchmark_dependencies(monkeypatch)
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


def test_benchmark_accepts_supported_cuda_without_slurm(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    monkeypatch.setattr(benchmark, "_package_version", lambda unused: "0.5.2")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)

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
