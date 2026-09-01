# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-visible contract checks that do not compile a CuTe kernel."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_private_backend():
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


def test_operation_package_does_not_publish_backend_lifecycle():
    bulk, api_class, wrapper = _load_private_backend()
    assert bulk.__all__ == []
    # Backend contract tests can reach private implementation seams by an
    # explicit module path; wildcard users and generated API docs cannot.
    assert bulk.CausalConv1dBulkFwdSm100 is api_class
    assert bulk.causal_conv1d_bulk_fwd_wrapper_sm100 is wrapper


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
    requested_devices = []

    def get_device_properties(device):
        requested_devices.append(device)
        return properties

    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", get_device_properties)

    assert benchmark._validate_environment() == (properties, capability)
    assert requested_devices == [3]
    assert benchmark._slurm_metadata({}) == {}


def test_benchmark_rejects_unsupported_gpu(monkeypatch):
    benchmark = _load_benchmark_module()
    properties = SimpleNamespace(major=10, minor=1, name="Unsupported GPU")
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda device=None: properties)

    with pytest.raises(RuntimeError, match=r"does not support compute capability 10\.1"):
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
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.5.0")))

    with pytest.raises(RuntimeError, match=r"nvidia-cutlass-dsl>=4\.7\.0; found 4\.5\.0"):
        api.check_support()


def test_support_accepts_a_source_dsl_without_distribution_metadata(monkeypatch):
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    api = api_class(x, weight, output, sample_bias=bias)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    assert api.check_support()
    assert api.bias_desc.shape == (8,)


def test_support_accepts_fp32_weight_only_without_bias(monkeypatch):
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.float32)
    output = torch.empty_like(x)
    api = api_class(x, weight, output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    assert api.check_support()
    assert api.weight_desc.dtype == torch.float32

    with_bias = api_class(
        x,
        weight,
        output,
        sample_bias=torch.zeros(8, dtype=torch.float32),
    )
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    with pytest.raises(ValueError, match="only when Bias is omitted"):
        with_bias.check_support()


@pytest.mark.parametrize("case", ["rank", "shape", "dtype", "stride", "alignment"])
def test_support_rejects_invalid_bias_contract(monkeypatch, case):
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    if case == "rank":
        bias = torch.zeros(1, 8, dtype=torch.bfloat16)
    elif case == "shape":
        bias = torch.zeros(7, dtype=torch.bfloat16)
    elif case == "dtype":
        bias = torch.zeros(8, dtype=torch.float32)
    elif case == "stride":
        bias = torch.zeros(8, 2, dtype=torch.bfloat16)[:, 0]
    elif case == "alignment":
        bias = torch.zeros(9, dtype=torch.bfloat16)[1:]
    else:
        raise AssertionError(f"unhandled case {case}")

    api = api_class(x, weight, output, sample_bias=bias)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))

    with pytest.raises(ValueError, match="Bias"):
        api.check_support()


def test_bias_presence_must_match_the_compiled_signature(monkeypatch):
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module

    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    with_bias = api_class(x, weight, output, sample_bias=bias)
    monkeypatch.setattr(with_bias, "_require_cuda", lambda desc, name: None)
    assert with_bias.check_support()
    with_bias._compiled_kernel = object()
    with pytest.raises(ValueError, match="Bias presence must match"):
        with_bias.execute(x, weight, output)

    without_bias = api_class(x, weight, output)
    monkeypatch.setattr(without_bias, "_require_cuda", lambda desc, name: None)
    assert without_bias.check_support()
    without_bias._compiled_kernel = object()
    with pytest.raises(ValueError, match="Bias presence must match"):
        without_bias.execute(x, weight, output, bias_tensor=bias)


def test_constructor_and_wrapper_reject_non_tensor_bias():
    _, api_class, wrapper = _load_private_backend()
    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)

    with pytest.raises(TypeError, match=r"sample_bias must be a torch\.Tensor"):
        api_class(x, weight, output, sample_bias=object())
    with pytest.raises(TypeError, match=r"Bias must be a torch\.Tensor or None"):
        wrapper(x, weight, bias_tensor=object())


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
    _, api_class, _ = _load_private_backend()
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
    _, api_class, _ = _load_private_backend()
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


def test_support_accepts_metadata_only_tensor_descriptors(monkeypatch):
    _, api_class, _ = _load_private_backend()
    import cudnn.causal_conv1d_bulk_sm100.api as api_module
    from cudnn.api_base import TensorDesc

    sample_x = TensorDesc(
        dtype=torch.bfloat16,
        shape=(1, 2, 8),
        stride=(16, 8, 1),
        stride_order=(2, 1, 0),
        device=torch.device("cuda"),
    )
    sample_weight = TensorDesc(
        dtype=torch.bfloat16,
        shape=(8, 4),
        stride=(4, 1),
        stride_order=(1, 0),
        device=torch.device("cuda"),
    )
    sample_output = TensorDesc(
        dtype=torch.bfloat16,
        shape=(1, 2, 8),
        stride=(16, 8, 1),
        stride_order=(2, 1, 0),
        device=torch.device("cuda"),
    )
    api = api_class(sample_x, sample_weight, sample_output)
    monkeypatch.setattr(api_module, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    assert api.check_support()
    assert api._sample_alignment_remainders == {}


@pytest.mark.parametrize("missing_name", ["sample_x", "sample_weight", "sample_output"])
def test_constructor_rejects_none_for_required_samples(missing_name):
    _, api_class, _ = _load_private_backend()
    samples = {
        "sample_x": torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        "sample_weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "sample_output": torch.zeros(1, 2, 8, dtype=torch.bfloat16),
    }
    samples[missing_name] = None

    with pytest.raises(TypeError, match=rf"{missing_name} must be a torch.Tensor or TensorDesc, got NoneType"):
        api_class(**samples)


@pytest.mark.L1
def test_top_level_exports_only_the_semantic_api_from_a_clean_source_package(tmp_path):
    # The suite normally overlays this operation onto a prebuilt frontend. Use
    # a fresh interpreter and this checkout's __init__.py to exercise the
    # top-level lazy-export route that a built wheel installs.
    _load_private_backend()
    import cudnn

    source = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    probe = tmp_path / "cudnn"
    shutil.copytree(source, probe)
    compiled_modules = list(Path(cudnn.__file__).resolve().parent.glob("_compiled_module*.so"))
    if len(compiled_modules) != 1:
        pytest.skip(f"expected one compiled module next to {cudnn.__file__}, found {len(compiled_modules)}")
    try:
        (probe / compiled_modules[0].name).symlink_to(compiled_modules[0])
    except OSError as error:
        pytest.skip(f"cannot link the compiled module into the probe package: {error}")

    script = """
import cudnn
from cudnn.ops import causal_conv1d

assert callable(causal_conv1d)
for name in (
    "CausalConv1dBulkAutogradPrototype",
    "CausalConv1dBulkBwdPrototype",
    "CausalConv1dBulkFwdSm100",
    "causal_conv1d_bulk_fwd_wrapper_sm100",
    "compile_causal_conv1d_bulk_bwd_prototype",
):
    assert name not in dir(cudnn)
    try:
        getattr(cudnn, name)
    except AttributeError:
        pass
    else:
        raise AssertionError(f"{name} leaked through the public cudnn namespace")
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
