# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-visible contract checks that do not compile a CuTe kernel."""

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_public_api():
    try:
        import cudnn.causal_conv1d_bulk_sm100 as bulk
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    return bulk, bulk.CausalConv1dBulkFwdSm100, bulk.causal_conv1d_bulk_fwd_wrapper_sm100


def _load_benchmark_module():
    path = Path(__file__).with_name("benchmark_causal_conv1d_bulk_sm100.py")
    spec = importlib.util.spec_from_file_location("_causal_conv1d_bulk_benchmark_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operation_package_exports_class_and_wrapper():
    bulk, api_class, wrapper = _load_public_api()
    assert bulk.__all__ == [
        "CausalConv1dBulkFwdSm100",
        "causal_conv1d_bulk_fwd_wrapper_sm100",
    ]
    assert bulk.CausalConv1dBulkFwdSm100 is api_class
    assert bulk.causal_conv1d_bulk_fwd_wrapper_sm100 is wrapper


def test_public_signatures_keep_optional_state_and_packed_metadata_explicit():
    _, api_class, wrapper = _load_public_api()

    assert tuple(inspect.signature(api_class).parameters) == (
        "sample_x",
        "sample_weight",
        "sample_output",
        "sample_cu_seqlens",
        "sample_initial_state",
        "sample_final_state",
    )
    assert tuple(inspect.signature(api_class.execute).parameters) == (
        "self",
        "x_tensor",
        "weight_tensor",
        "output_tensor",
        "cu_seqlens_tensor",
        "initial_state_tensor",
        "final_state_tensor",
        "current_stream",
    )
    assert tuple(inspect.signature(wrapper).parameters) == (
        "x_tensor",
        "weight_tensor",
        "cu_seqlens_tensor",
        "initial_state_tensor",
        "output_final_state",
        "current_stream",
    )
    assert inspect.signature(wrapper).parameters["output_final_state"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "capability",
    [
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
    ],
)
def test_benchmark_accepts_functional_gpu_without_slurm(monkeypatch, capability):
    benchmark = _load_benchmark_module()
    properties = SimpleNamespace(
        major=capability[0],
        minor=capability[1],
        name="Customer GPU",
    )
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda device=None: properties)

    assert benchmark._validate_environment() == (properties, capability)
    assert benchmark._slurm_metadata({}) == {}


def test_benchmark_rejects_unsupported_gpu(monkeypatch):
    benchmark = _load_benchmark_module()
    properties = SimpleNamespace(major=10, minor=1, name="Unsupported GPU")
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda device=None: properties)

    with pytest.raises(RuntimeError, match="does not support compute capability 10.1"):
        benchmark._validate_environment()


def test_benchmark_requires_cuda(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        benchmark._validate_environment()


def test_benchmark_slurm_metadata_is_optional():
    benchmark = _load_benchmark_module()

    assert benchmark._slurm_metadata({}) == {}
    assert benchmark._slurm_metadata({"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "gpu-node"}) == {
        "job_id": "123",
        "node_name": "gpu-node",
    }


def test_support_rejects_the_package_wide_4_5_dsl_floor(monkeypatch):
    _, api_class, _ = _load_public_api()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.5.0")))

    with pytest.raises(RuntimeError, match=r"nvidia-cutlass-dsl>=4\.7\.0; found 4\.5\.0"):
        api.check_support()


def test_support_accepts_a_source_dsl_without_distribution_metadata(monkeypatch):
    _, api_class, _ = _load_public_api()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    assert api.check_support()


@pytest.mark.parametrize(
    "capability,n_channels,expected_vec8",
    [
        ((8, 0), 8, False),
        ((8, 6), 8, False),
        ((8, 7), 8, False),
        ((8, 9), 8, False),
        ((9, 0), 8, False),
        ((10, 0), 8, True),
        ((10, 0), 7, False),
        ((10, 3), 8, True),
        ((11, 0), 8, True),
        ((12, 0), 8, True),
        ((12, 1), 8, True),
    ],
)
def test_support_selects_schedule_from_exact_arch(monkeypatch, capability, n_channels, expected_vec8):
    _, api_class, _ = _load_public_api()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, n_channels, dtype=torch.bfloat16)
    weight = torch.zeros(n_channels, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    assert api.check_support()
    assert api.compute_capability == capability
    assert api.use_vec8_schedule is expected_vec8


@pytest.mark.parametrize("capability", [(7, 5), (10, 1), (11, 1)])
def test_support_rejects_unlisted_arch(monkeypatch, capability):
    _, api_class, _ = _load_public_api()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    with pytest.raises(RuntimeError, match="does not support compute capability"):
        api.check_support()


def test_constructor_rejects_metadata_only_tensor_descriptors():
    _, api_class, _ = _load_public_api()
    from cudnn.api_base import TensorDesc

    sample_x = TensorDesc(
        dtype=torch.bfloat16,
        shape=(1, 2, 8),
        stride=(16, 8, 1),
        stride_order=(2, 1, 0),
        device=torch.device("cuda"),
    )
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.zeros(1, 2, 8, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match=r"sample_x must be a torch.Tensor, got TensorDesc"):
        api_class(sample_x, weight, output)


@pytest.mark.parametrize("missing_name", ["sample_x", "sample_weight", "sample_output"])
def test_constructor_rejects_none_for_required_samples(missing_name):
    _, api_class, _ = _load_public_api()
    samples = {
        "sample_x": torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        "sample_weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "sample_output": torch.zeros(1, 2, 8, dtype=torch.bfloat16),
    }
    samples[missing_name] = None

    with pytest.raises(TypeError, match=rf"{missing_name} must be a torch.Tensor, got NoneType"):
        api_class(**samples)


def test_top_level_lazy_exports_resolve_from_a_clean_source_package(tmp_path):
    # The suite normally overlays this operation onto a prebuilt frontend. Use
    # a fresh interpreter and this checkout's __init__.py to exercise the
    # top-level lazy-export route that a built wheel installs.
    _load_public_api()
    import cudnn

    source = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    probe = tmp_path / "cudnn"
    shutil.copytree(source, probe)
    compiled_modules = list(Path(cudnn.__file__).resolve().parent.glob("_compiled_module*.so"))
    assert len(compiled_modules) == 1
    (probe / compiled_modules[0].name).symlink_to(compiled_modules[0])

    script = """
import cudnn
from cudnn import CausalConv1dBulkFwdSm100, causal_conv1d_bulk_fwd_wrapper_sm100
from cudnn.causal_conv1d_bulk_sm100 import (
    CausalConv1dBulkFwdSm100 as package_class,
    causal_conv1d_bulk_fwd_wrapper_sm100 as package_wrapper,
)

assert CausalConv1dBulkFwdSm100 is package_class
assert causal_conv1d_bulk_fwd_wrapper_sm100 is package_wrapper
assert cudnn.CausalConv1dBulkFwdSm100 is package_class
assert cudnn.causal_conv1d_bulk_fwd_wrapper_sm100 is package_wrapper
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), environment.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
