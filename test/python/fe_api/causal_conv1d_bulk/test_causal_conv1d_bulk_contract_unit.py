# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-visible contract checks that do not compile a CuTe kernel."""

import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_public_api():
    try:
        import cudnn.causal_conv1d_bulk_sm100 as bulk
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    return bulk, bulk.CausalConv1dBulkFwdSm100, bulk.causal_conv1d_bulk_fwd_wrapper_sm100


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
